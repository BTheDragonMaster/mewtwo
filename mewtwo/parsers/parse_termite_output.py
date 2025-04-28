from mewtwo.parsers.tabular import Tabular
from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin
from mewtwo.embeddings.bases import BasePair, Base

from sys import argv
from pprint import pprint


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
    basepairs_to_counts_rnafold = {}
    basepairs_to_counts_transtermhp = {}
    for hairpin in hairpins:
        if hairpin.contains_multiple_hairpins():
            counter += 1
            print(hairpin.distance_to_pot, hairpin.free_energy, hairpin.hairpin_sequence, hairpin.hairpin_structure)

        else:
            loop, stem = hairpin.get_hairpin_parts()
            for basepair in stem.get_basepairs():
                if hairpin.prediction_software == 'RNAFold':

                    if basepair not in basepairs_to_counts_rnafold:

                        basepairs_to_counts_rnafold[basepair] = 0
                    basepairs_to_counts_rnafold[basepair] += 1
                else:
                    if basepair not in basepairs_to_counts_transtermhp:
                        basepairs_to_counts_transtermhp[basepair] = 0
                    basepairs_to_counts_transtermhp[basepair] += 1
                    if basepair == BasePair(Base.A, Base.A, True):
                        print(hairpin.hairpin_sequence)
                        print(hairpin.hairpin_structure)

    print(f"{counter} structures contain multiple hairpins")
    pprint(basepairs_to_counts_rnafold)
    pprint(basepairs_to_counts_transtermhp)





