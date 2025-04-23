from mewtwo.embeddings.sequence import RNASequence


class Loop:
    def __init__(self, loop_sequence: RNASequence, loop_structure: str) -> None:
        self.sequence = loop_sequence
        self.structure = loop_structure
