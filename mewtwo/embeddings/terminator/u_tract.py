from mewtwo.embeddings.sequence import RNASequence


class UTract:
    def __init__(self, sequence: RNASequence, pot: int) -> None:
        self.sequence = sequence
        self.pot = pot
