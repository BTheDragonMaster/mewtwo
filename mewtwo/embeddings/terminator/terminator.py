from typing import Union, Optional

from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin
from mewtwo.embeddings.terminator.a_tract import ATract
from mewtwo.embeddings.terminator.u_tract import UTract
from mewtwo.embeddings.sequence import RNASequence
from mewtwo.embeddings.terminator.loop import get_max_loop_size


class Terminator:
    def __init__(self,
                 hairpin: Union[RNAFoldHairpin, TransTermHPHairpin], a_tract: ATract, u_tract: UTract,
                 sequence: RNASequence,
                 termination_efficiency: Optional[float] = None,
                 start: Optional[int] = None, end: Optional[int] = None, pot: Optional[int] = None,
                 species: Optional[str] = None, chromosome: Optional[str] = None, strand: Optional[str] = None,
                 is_synthetic: bool = False):
        self.start = start
        self.end = end
        self.pot = pot
        self.species = species
        self.chromosome = chromosome
        self.te = termination_efficiency
        self.sequence = sequence
        self.strand = strand

        self.hairpin = hairpin
        self.a_tract = a_tract
        self.u_tract = u_tract
        self.is_synthetic = is_synthetic

    def to_vector(self, max_loop_size: int, max_stem_size: int = 10, a_tract_size: int = 10, u_tract_size: int = 10,
                  one_hot: bool = False) -> list[int]:
        assert not self.hairpin.contains_multiple_hairpins()
        vector = []
        vector.extend(self.a_tract.to_vector(a_tract_size=a_tract_size, one_hot=one_hot))
        vector.extend(self.hairpin.stem.to_vector(max_stem_size=max_stem_size, one_hot=one_hot))
        vector.extend(self.hairpin.loop.to_vector(max_loop_size=max_loop_size, one_hot=one_hot))
        vector.extend(self.u_tract.to_vector(u_tract_size=u_tract_size, one_hot=one_hot))
        return vector

    def get_base_coords(self, max_loop_size, max_stem_size, a_tract_size, u_tract_size):
        pass


def get_terminator_part_sizes(terminators: list[Terminator]) -> tuple[int, int, int, int]:
    loops = []
    a_tracts = []
    u_tracts = []
    stems = []
    for terminator in terminators:
        loops.append(terminator.hairpin.loop)
        a_tracts.append(terminator.a_tract)
        u_tracts.append(terminator.u_tract)
        stems.append(terminator.hairpin.stem)

    max_loop_size = get_max_loop_size(loops)
    max_stem_size = max([len(stem.basepairs) for stem in stems])
    max_a_tract_size = max([len(a_tract.sequence) for a_tract in a_tracts])
    max_u_tract_size = max([len(u_tract.sequence) for u_tract in u_tracts])

    return max_loop_size, max_stem_size, max_a_tract_size, max_u_tract_size
