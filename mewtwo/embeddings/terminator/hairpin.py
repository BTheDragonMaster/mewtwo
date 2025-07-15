from enum import Enum
from typing import Optional

from mewtwo.embeddings.terminator.loop import Loop
from mewtwo.embeddings.terminator.stem import Stem
from mewtwo.embeddings.bases import BasePair, Base
from mewtwo.embeddings.sequence import DNASequence, RNASequence, get_sequence_type, convert_to_rna, SeqType


class HairpinType(Enum):
    RNAFOLD = 1
    TRANSTERMHP = 2


class Hairpin:
    def __init__(self, hairpin_id: str,
                 prediction_software: HairpinType, distance_to_pot: Optional[int] = None):

        self.hairpin_id = hairpin_id
        self.distance_to_pot = distance_to_pot

        self.prediction_software = prediction_software
        self.hairpin_sequence = None
        self.hairpin_structure = None
        self.loop = None
        self.stem = None

    def __eq__(self, other):
        if type(self) != other(type):
            return False
        if self.prediction_software != other.prediction_software:
            return False
        if self.hairpin_id != other.hairpin_id:
            return False
        return True

    def contains_multiple_hairpins(self):
        seen_right_shoulder = False

        for character in self.hairpin_structure:
            if character == ')':
                seen_right_shoulder = True
            if character == '(' and seen_right_shoulder:
                return True

        return False

    def set_hairpin_parts(self):
        assert not self.contains_multiple_hairpins()

        last_left_shoulder = 0
        first_right_shoulder = 0
        for i, character in enumerate(self.hairpin_structure):
            if character == '(':
                last_left_shoulder = i
            elif character == ')':
                first_right_shoulder = i
                break

        left_shoulder_structure = self.hairpin_structure[:last_left_shoulder + 1]
        left_shoulder_sequence = self.hairpin_sequence[:last_left_shoulder + 1]
        loop_structure = self.hairpin_structure[last_left_shoulder + 1:first_right_shoulder]
        loop_sequence = self.hairpin_sequence[last_left_shoulder + 1:first_right_shoulder]
        right_shoulder_structure = self.hairpin_structure[first_right_shoulder:]
        right_shoulder_sequence = self.hairpin_sequence[first_right_shoulder:]

        self.loop = Loop(loop_sequence, loop_structure)
        self.stem = Stem(left_shoulder_sequence, left_shoulder_structure, right_shoulder_sequence, right_shoulder_structure)

    def to_vector(self, max_stem_size, max_loop_size):
        assert self.hairpin_sequence is not None and self.hairpin_structure is not None


class RNAFoldHairpin(Hairpin):

    def __init__(self, hairpin_id: str, free_energy: float, hairpin_sequence: str, hairpin_structure: str,
                 distance_to_pot: Optional[int] = None):
        super().__init__(hairpin_id, HairpinType.RNAFOLD, distance_to_pot)
        seq_type = get_sequence_type(hairpin_sequence)
        if SeqType.RNA in seq_type:
            sequence = RNASequence(hairpin_sequence)
        else:
            sequence = convert_to_rna(DNASequence(hairpin_sequence))
        self.hairpin_sequence = sequence
        self.hairpin_structure = hairpin_structure
        self.free_energy = free_energy
        if not self.contains_multiple_hairpins():
            self.set_hairpin_parts()


class TransTermHPHairpin(Hairpin):

    def __init__(self, hairpin_id, hairpin_score, hairpin, distance_to_pot: Optional[int] = None):
        super().__init__(hairpin_id, HairpinType.TRANSTERMHP, distance_to_pot)
        self.set_hairpin_sequence(hairpin)
        self.set_hairpin_structure(hairpin)
        self.hairpin_score = hairpin_score
        if not self.contains_multiple_hairpins():
            self.set_hairpin_parts()

    def set_hairpin_sequence(self, hairpin):

        hairpin_sequence = ''.join(hairpin.split())
        hairpin_sequence = hairpin_sequence.replace('-', '')

        seq_type = get_sequence_type(hairpin_sequence)
        if SeqType.RNA in seq_type:
            sequence = RNASequence(hairpin_sequence)
        else:
            sequence = convert_to_rna(DNASequence(hairpin_sequence))
        self.hairpin_sequence = sequence

    def set_hairpin_structure(self, hairpin):
        left_shoulder, loop, right_shoulder = hairpin.upper().split()

        upstream_structure = []
        downstream_structure = []

        loop_structure = []
        for _ in loop:
            loop_structure.append('.')

        for i, base in enumerate(left_shoulder):
            pairing_base = right_shoulder[-i - 1]
            assert not (base == '-' and pairing_base == '-')

            if base == '-':
                downstream_structure.append('.')
            elif pairing_base == '-':
                upstream_structure.append('.')
            else:
                if BasePair(Base[base], Base[pairing_base], True).is_watson_crick():
                    upstream_structure.append('(')
                    downstream_structure.append(')')
                else:
                    upstream_structure.append('.')
                    downstream_structure.append('.')
        downstream_structure.reverse()

        hairpin_structure = upstream_structure + loop_structure + downstream_structure

        self.hairpin_structure = ''.join(hairpin_structure)

