from mewtwo.embeddings.sequence import RNASequence
from mewtwo.embeddings.bases import Base, base_to_vector


class Loop:
    def __init__(self, loop_sequence: RNASequence, loop_structure: str) -> None:
        self.sequence = loop_sequence
        self.structure = loop_structure

    def to_vector(self, max_loop_size, one_hot=False):
        assert len(self.sequence) <= max_loop_size
        assert max_loop_size % 2 == 1

        if len(self.sequence) % 2 == 0:
            center_base = Base.ZERO_PADDING
        else:
            center_base = self.sequence[len(self.sequence) // 2]
        center_vector = base_to_vector(center_base, one_hot=one_hot)
        padding = (max_loop_size - len(self.sequence)) // 2
        left_padding = []
        right_padding = []
        for i in range(padding):
            left_padding.extend(base_to_vector(Base.ZERO_PADDING, one_hot=one_hot))
            right_padding.extend(base_to_vector(Base.ZERO_PADDING, one_hot=one_hot))

        left_vector = []
        right_vector = []
        for i, base in enumerate(self.sequence):
            if i < len(self.sequence) // 2:

                left_vector.extend(base_to_vector(base, one_hot=one_hot))
            elif i >= len(self.sequence) / 2:
                right_vector.extend(base_to_vector(base, one_hot=one_hot))

        vector = left_padding + left_vector + center_vector + right_vector + right_padding
        return vector


def get_max_loop_size(loops: list[Loop]) -> int:
    max_loop_size = 0
    for loop in loops:
        loop_size = len(loop.sequence)
        if loop_size > max_loop_size:

            max_loop_size = loop_size


    if max_loop_size % 2 == 0:

        max_loop_size += 1

    return max_loop_size
