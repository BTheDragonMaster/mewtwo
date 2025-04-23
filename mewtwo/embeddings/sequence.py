from enum import Enum
from typing import Union
from mewtwo.embeddings.bases import Base


class SeqType(Enum):
    DNA = 1
    RNA = 2


class Sequence:
    def __init__(self, sequence: str, seq_type: SeqType = SeqType['DNA']):
        self.sequence = sequence.upper()
        self.seq_type = seq_type
        self._check_sequence()

    def __eq__(self, other):
        if self.sequence == other.sequence and type(self) == type(other):
            return True

        return False

    def __getitem__(self, index: Union[slice, int]) -> Union['Sequence', 'RNASequence', 'DNASequence', Base]:
        if isinstance(index, int):
            return Base[self.sequence[index]]
        elif isinstance(index, slice):
            return type(self)(self.sequence[index])

    def _check_sequence(self):
        for character in self.sequence:
            if self.seq_type == SeqType["DNA"]:
                try:
                    base = Base[character]
                    if base.name not in ['A', 'C', 'G', 'T']:
                        raise ValueError(f"DNA sequence must be comprised of bases A, T, C, and G. Found {character} in {self.sequence}")
                except KeyError:
                    raise ValueError(f"DNA sequence must be comprised of bases A, T, C, and G. Found {character} in {self.sequence}")
            elif self.seq_type == SeqType["RNA"]:
                try:
                    base = Base[character]
                    if base.name not in ['A', 'C', 'G', 'U']:
                        raise ValueError(f"RNA sequence must be comprised of bases A, C, G and U. Found {character} in {self.sequence}")
                except KeyError:
                    raise ValueError(f"RNA sequence must be comprised of bases A, C, G, and U. Found {character} in {self.sequence}")


class DNASequence(Sequence):
    def __init__(self, sequence):
        super().__init__(sequence, SeqType["DNA"])


class RNASequence(Sequence):
    def __init__(self, sequence):
        super().__init__(sequence, SeqType["RNA"])


def convert_to_rna(dna_sequence: DNASequence) -> RNASequence:
    rna_sequence = []
    for character in dna_sequence.sequence:
        if character == 'T':
            rna_sequence.append('U')
        else:
            rna_sequence.append(character)

    return RNASequence(''.join(rna_sequence))


def convert_to_dna(rna_sequence: RNASequence) -> DNASequence:
    dna_sequence = []
    for character in rna_sequence.sequence:
        if character == 'U':
            dna_sequence.append('T')
        else:
            dna_sequence.append(character)

    return DNASequence(''.join(dna_sequence))

