from mewtwo.embeddings.terminator.loop import Loop
from mewtwo.embeddings.terminator.stem import Stem
from mewtwo.embeddings.bases import BasePair, Base


class Hairpin:
    def __init__(self, hairpin_id: str, distance_to_pot: int,
                 prediction_software: str):
        assert prediction_software in ["RNAFold", "TransTermHP"]

        self.hairpin_id = hairpin_id
        self.distance_to_pot = distance_to_pot

        self.prediction_software = prediction_software
        self.hairpin_sequence = None
        self.hairpin_structure = None

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

    def get_hairpin_parts(self):
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

        loop = Loop(loop_sequence, loop_structure)
        stem = Stem(left_shoulder_sequence, left_shoulder_structure, right_shoulder_sequence, right_shoulder_structure)

        return loop, stem

    def to_vector(self, max_stem_size, max_loop_size):
        assert self.hairpin_sequence is not None and self.hairpin_structure is not None


class RNAFoldHairpin(Hairpin):

    def __init__(self, hairpin_id, distance_to_pot, free_energy, hairpin_sequence, hairpin_structure):
        super().__init__(hairpin_id, distance_to_pot, "RNAFold")
        self.hairpin_sequence = hairpin_sequence.upper()
        self.hairpin_structure = hairpin_structure.upper()
        self.free_energy = free_energy


class TransTermHPHairpin(Hairpin):

    def __init__(self, hairpin_id, distance_to_pot, hairpin_score, hairpin):
        super().__init__(hairpin_id, distance_to_pot, "TransTermHP")
        self.set_hairpin_sequence(hairpin)
        self.set_hairpin_structure(hairpin)
        self.hairpin_score = hairpin_score

    def set_hairpin_sequence(self, hairpin):
        hairpin_sequence = ''.join(hairpin.split())
        hairpin_sequence = hairpin_sequence.replace('-', '')
        self.hairpin_sequence = hairpin_sequence.upper()

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

