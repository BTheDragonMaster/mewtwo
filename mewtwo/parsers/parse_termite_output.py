from mewtwo.parsers.tabular import Tabular
from sys import argv


class Loop:
    def __init__(self, loop_sequence, loop_structure):
        self.sequence = loop_sequence
        self.structure = loop_structure


class Stem:
    def __init__(self, upstream_sequence, upstream_structure, downstream_sequence, downstream_structure):
        self.upstream_sequence = upstream_sequence
        self.upstream_structure = upstream_structure
        self.downstream_sequence = downstream_sequence
        self.downstream_structure = downstream_structure

    def get_basepairs(self):
        basepairs = []
        pairing_indices_upstream = []
        pairing_indices_downstream = []
        for i, character in enumerate(self.upstream_structure):
            if character == '(':
                pairing_indices_upstream.append(i)

        for j, character in enumerate(self.downstream_structure):
            if character == ')':
                pairing_indices_downstream.append(j)

        assert len(pairing_indices_upstream) == len(pairing_indices_downstream)

        pairing_indices_downstream.reverse()

        for i, bp_index_1 in enumerate(pairing_indices_upstream):
            bp_index_2 = pairing_indices_downstream[i]
            basepairs.append((self.upstream_sequence[bp_index_1], self.downstream_sequence[bp_index_2]))

        return basepairs

    def to_vector(self, max_stem_size):
        pass


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
        self.hairpin_sequence = hairpin_sequence
        self.hairpin_structure = hairpin_structure
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
        self.hairpin_sequence = hairpin_sequence

    def set_hairpin_structure(self, hairpin):
        left_shoulder, loop, right_shoulder = hairpin.split()

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
                upstream_structure.append('(')
                downstream_structure.append(')')
        downstream_structure.reverse()

        hairpin_structure = upstream_structure + loop_structure + downstream_structure

        self.hairpin_structure = ''.join(hairpin_structure)


def rnafold_hairpins_from_termite(input_file, get_rnafold: bool = True, get_transtermhp: bool = False,
                                  get_mutually_exclusive: bool = False, species_column: bool = False):

    if get_rnafold and get_transtermhp:
        assert not get_mutually_exclusive

    assert get_rnafold or get_transtermhp

    rnafold_hairpins = {}
    transtermhp_hairpins = {}

    if not species_column:
        termite_data = Tabular(input_file, [0, 6])
    else:
        termite_data = Tabular(input_file, [0, 1, 7])
    for datapoint in termite_data.data:
        hairpin_id = '|'.join(datapoint)
        if termite_data.get_value(datapoint, 'rnafold') == '+':
            hairpin = RNAFoldHairpin(hairpin_id,
                                     termite_data.get_value(datapoint, "rnafold POT distance to hairpin"),
                                     termite_data.get_value(datapoint, "rnafold energy"),
                                     termite_data.get_value(datapoint, "rnafold hairpin"),
                                     termite_data.get_value(datapoint, "rnafold hairpin structure"))
            rnafold_hairpins[hairpin_id] = hairpin
        if termite_data.get_value(datapoint, 'transtermhp') == '+':
            hairpin = TransTermHPHairpin(hairpin_id,
                                         termite_data.get_value(datapoint, "transtermhp POT distance to hairpin"),
                                         termite_data.get_value(datapoint, "transtermhp hairpin score"),
                                         termite_data.get_value(datapoint, "transtermhp hairpin"))
            transtermhp_hairpins[hairpin_id] = hairpin
    hairpins = []
    if get_rnafold:
        for hairpin_id, hairpin in rnafold_hairpins.items():
            hairpins.append(hairpin)
        if get_mutually_exclusive:
            for hairpin_id, hairpin in transtermhp_hairpins.items():
                if hairpin_id not in rnafold_hairpins:
                    hairpins.append(hairpin)

    if get_transtermhp:
        for hairpin_id, hairpin in transtermhp_hairpins.items():
            hairpins.append(hairpin)
        if get_mutually_exclusive:
            for hairpin_id, hairpin in rnafold_hairpins.items():
                if hairpin_id not in transtermhp_hairpins:
                    hairpins.append(hairpin)

    return hairpins


if __name__ == "__main__":
    hairpins = rnafold_hairpins_from_termite(argv[1], get_rnafold=True, get_transtermhp=True, species_column=True)
    counter = 0
    for hairpin in hairpins:
        if hairpin.contains_multiple_hairpins():
            counter += 1
            print(hairpin.distance_to_pot, hairpin.free_energy, hairpin.hairpin_sequence, hairpin.hairpin_structure)

        else:
            loop, stem = hairpin.get_hairpin_parts()
            # stem.get_basepairs()



    print(f"{counter} structures contain multiple hairpins")





