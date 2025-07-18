from mewtwo.embeddings.bases import BasePair, Base, PairingType
from mewtwo.embeddings.sequence import RNASequence


class Stem:
    def __init__(self, upstream_sequence: RNASequence, upstream_structure: str,
                 downstream_sequence: RNASequence, downstream_structure: str):
        assert len(upstream_sequence) == len(upstream_structure)
        assert len(downstream_sequence) == len(downstream_structure)

        self.upstream_sequence = upstream_sequence
        self.upstream_structure = upstream_structure
        self.downstream_sequence = downstream_sequence
        self.downstream_structure = downstream_structure
        self.basepairs = self.get_basepairs()

    def get_basepairs(self):
        basepairs = []

        reverse_downstream_structure = list(reversed(self.downstream_structure))
        reverse_downstream_sequence = list(reversed(self.downstream_sequence))
        downstream_index = 0

        for i, character in enumerate(self.upstream_structure):
            if character == '(':
                while reverse_downstream_structure[downstream_index] != ')':
                    basepairs.append(BasePair(Base.ZERO_PADDING, reverse_downstream_sequence[downstream_index], False))
                    downstream_index += 1
                basepairs.append(
                    BasePair(self.upstream_sequence[i], reverse_downstream_sequence[downstream_index], True))
                downstream_index += 1

            else:
                if reverse_downstream_structure[downstream_index] == '.':
                    basepairs.append(BasePair(self.upstream_sequence[i],
                                              reverse_downstream_sequence[downstream_index], False))
                    downstream_index += 1
                else:
                    basepairs.append(BasePair(self.upstream_sequence[i], Base.ZERO_PADDING, False))

        return basepairs

    def to_vector(self, max_stem_size: int, one_hot: bool = False,
                  pairing_type: PairingType = PairingType.STRUCTURE_BASED) -> list[int]:
        vector = []

        for i in range(max_stem_size):
            try:
                basepair = self.basepairs[i]
            except IndexError:
                basepair = BasePair(Base.ZERO_PADDING, Base.ZERO_PADDING, False)
            vector.extend(basepair.to_vector(one_hot=one_hot, pairing_type=pairing_type))

        return vector


def get_max_stem_size(stems: list[Stem]) -> int:
    max_stem_size = 0
    for stem in stems:
        stem_size = len(stem.basepairs)
        if stem_size > max_stem_size:
            max_stem_size += 1

    return max_stem_size
