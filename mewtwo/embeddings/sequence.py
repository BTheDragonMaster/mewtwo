from enum import Flag
from typing import Union
from mewtwo.embeddings.bases import Base, base_to_vector


class SeqType(Flag):
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

    def __hash__(self):
        return hash(self.sequence)

    def __repr__(self):
        return self.sequence

    def __getitem__(self, index: Union[slice, int]) -> Union['Sequence', 'RNASequence', 'DNASequence', Base]:
        if isinstance(index, int):
            return Base[self.sequence[index]]
        elif isinstance(index, slice):
            return type(self)(self.sequence[index])

    def __len__(self):
        return len(self.sequence)

    def _check_sequence(self):
        for character in self.sequence:
            if self.seq_type == SeqType.DNA:
                try:
                    base = Base[character]
                    if base not in Base.DNA:
                        raise ValueError(f"DNA sequence must be comprised of bases A, T, C, and G. Found {character} in {self.sequence}")
                except KeyError:
                    raise ValueError(f"DNA sequence must be comprised of bases A, T, C, and G. Found {character} in {self.sequence}")
            elif self.seq_type == SeqType.RNA:
                try:
                    base = Base[character]
                    if base not in Base.RNA:
                        raise ValueError(f"RNA sequence must be comprised of bases A, C, G and U. Found {character} in {self.sequence}")
                except KeyError:
                    raise ValueError(f"RNA sequence must be comprised of bases A, C, G, and U. Found {character} in {self.sequence}")

    def to_vector(self, one_hot: bool = False) -> list[int]:
        vector = []
        for character in self.sequence:
            base = Base[character]
            vector.extend(base_to_vector(base, one_hot=one_hot))

        return vector


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


def get_sequence_type(sequence: str) -> SeqType:

    is_dna = False
    is_rna = False

    try:
        DNASequence(sequence)
        is_dna = True
    except ValueError:
        pass

    try:
        RNASequence(sequence)
        is_rna = True
    except ValueError:
        pass

    seq_types = []

    if is_dna and is_rna:
        return SeqType.DNA | SeqType.RNA
    elif is_dna:
        return SeqType.DNA
    elif is_rna:
        return SeqType.RNA
    else:
        raise ValueError("Sequence is not DNA or RNA")
