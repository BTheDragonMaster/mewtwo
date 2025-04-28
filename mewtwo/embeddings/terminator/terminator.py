from typing import Union, Optional

from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin
from mewtwo.embeddings.terminator.a_tract import ATract
from mewtwo.embeddings.terminator.u_tract import UTract
from mewtwo.embeddings.sequence import RNASequence


class Terminator:
    def __init__(self, start: int, end: int, pot: int, species: str, chromosome: str, strand: str,
                 sequence: RNASequence, termination_efficiency: Optional[float],
                 hairpin: Union[RNAFoldHairpin, TransTermHPHairpin], a_tract: ATract, u_tract: UTract):
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

    def to_vector(self, one_hot: bool = False) -> list[int]:
        assert not self.hairpin.contains_multiple_hairpins()
        vector = []
        vector.extend(self.a_tract.sequence.to_vector(one_hot))
        vector.extend(self.hairpin.stem.to_vector())
        vector.extend(self.hairpin.loop.to_vector())
        vector.extend(self.u_tract.to_vector())
        return vector
