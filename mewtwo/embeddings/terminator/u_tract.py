
from typing import Optional

from mewtwo.embeddings.sequence import RNASequence, SeqType, convert_to_rna, get_sequence_type, DNASequence
from mewtwo.embeddings.bases import Base, base_to_vector


class UTract:
    def __init__(self, sequence: str, pot: Optional[int] = None) -> None:
        seq_type = get_sequence_type(sequence)
        if SeqType.RNA in seq_type:
            sequence = RNASequence(sequence)
        else:
            sequence = convert_to_rna(DNASequence(sequence))
        self.sequence = sequence
        self.pot = pot
        if self.pot is not None:
            assert self.pot < len(self.sequence)

    def to_vector(self, u_tract_size: int = 10, one_hot: bool = False) -> list[int]:
        vector = []
        for i in range(u_tract_size):
            try:
                base = self.sequence[i]
            except IndexError:
                base = Base.ZERO_PADDING
            vector.extend(base_to_vector(base, one_hot))

            if self.pot is not None:

                if i == self.pot:
                    vector.append(1)
                else:
                    vector.append(0)

        return vector
