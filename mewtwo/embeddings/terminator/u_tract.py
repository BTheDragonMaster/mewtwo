from mewtwo.embeddings.sequence import RNASequence
from mewtwo.embeddings.bases import Base, base_to_vector


class UTract:
    def __init__(self, sequence: RNASequence, pot: int) -> None:
        self.sequence = sequence
        self.pot = pot
        assert self.pot < len(self.sequence)

    def to_vector(self, u_tract_size: int = 10, one_hot: bool = False) -> list[int]:
        vector = []
        for i in range(u_tract_size):
            try:
                base = self.sequence[i]
            except IndexError:
                base = Base.ZERO_PADDING
            vector.extend(base_to_vector(base, one_hot))

            if i == self.pot:
                vector.append(1)
            else:
                vector.append(0)

        return vector
