from mewtwo.embeddings.sequence import RNASequence, DNASequence, get_sequence_type, SeqType, convert_to_rna
from mewtwo.embeddings.bases import Base, base_to_vector


class ATract:
    def __init__(self, sequence: str) -> None:

        seq_type = get_sequence_type(sequence)
        if SeqType.RNA in seq_type:
            sequence = RNASequence(sequence)
        else:
            sequence = convert_to_rna(DNASequence(sequence))
        self.sequence = sequence

    def to_vector(self, a_tract_size: int = 10, one_hot: bool = False) -> list[int]:
        vector: list[int] = []
        if len(self.sequence) < a_tract_size:
            for i in range(a_tract_size - len(self.sequence)):
                vector.extend(base_to_vector(Base.ZERO_PADDING, one_hot))
        for base in self.sequence[-a_tract_size:]:
            vector.extend(base_to_vector(base, one_hot))

        return vector
