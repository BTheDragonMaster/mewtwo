from mewtwo.parsers.tabular import Tabular
from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin
from mewtwo.embeddings.bases import BasePair, Base
from mewtwo.embeddings.terminator.terminator import Terminator
from mewtwo.embeddings.sequence import DNASequence, convert_to_rna
from mewtwo.embeddings.terminator.a_tract import ATract
from mewtwo.embeddings.terminator.u_tract import UTract

from sys import argv
from pprint import pprint


def parse_termite_data(input_file: str, species_column: bool) -> Tabular:
    if not species_column:
        termite_data = Tabular(input_file, [0, 6])
    else:
        termite_data = Tabular(input_file, [0, 1, 7])

    return termite_data


def get_termite_terminators(input_file: str, prioritise_rnafold: bool = True, species_column: bool = False,
                            te_only: bool = True) -> list[Terminator]:
    """

    Parameters
    ----------
    input_file: str, tabular termite output file
    prioritise_rnafold: bool, if True, prioritise hairpins predicted with RNAFold. Otherwise, prioritise hairpins
        predicted with TransTermHP
    species_column: bool, if True, termite input contains an additional species column
    te_only: bool, if True, only return terminators for which the termination efficiency is known

    Returns
    -------
    list of terminator instances

    """
    termite_data = parse_termite_data(input_file, species_column)

    rnafold_terminators = {}
    transtermhp_terminators = {}

    for datapoint in termite_data.data:
        terminator_id = '|'.join(datapoint)
        if not species_column:
            species = "unknown"
        else:
            species = termite_data.get_value(datapoint, "Species")
        chromosome = termite_data.get_value(datapoint, 'chromosome')
        pot = int(termite_data.get_value(datapoint, 'POT'))
        start = int(termite_data.get_value(datapoint, 'start'))
        end = int(termite_data.get_value(datapoint, 'end'))
        strand = termite_data.get_value(datapoint, 'strand')
        sequence = convert_to_rna(DNASequence(termite_data.get_value(datapoint, 'sequence')))

        te = termite_data.get_value(datapoint, "termination efficiency")
        if te == '.':
            te = None
        else:
            te = float(te)

        if te_only and te is None:
            continue

        if termite_data.get_value(datapoint, 'rnafold') == '+':

            hairpin = RNAFoldHairpin(terminator_id,
                                     termite_data.get_value(datapoint, "rnafold POT distance to hairpin"),
                                     termite_data.get_value(datapoint, "rnafold energy"),
                                     termite_data.get_value(datapoint, "rnafold hairpin"),
                                     termite_data.get_value(datapoint, "rnafold hairpin structure"))
            if not hairpin.contains_multiple_hairpins():
                a_tract_sequence = convert_to_rna(DNASequence(termite_data.get_value(datapoint, "rnafold a tract")))
                a_tract = ATract(a_tract_sequence)
                u_tract_sequence = convert_to_rna(DNASequence(termite_data.get_value(datapoint, "rnafold u tract")))
                relative_pot = int(termite_data.get_value(datapoint, "rnafold POT distance to hairpin"))
                u_tract = UTract(u_tract_sequence, relative_pot)

                terminator = Terminator(start, end, pot, species, chromosome, strand, sequence, te, hairpin, a_tract,
                                        u_tract)
                rnafold_terminators[terminator_id] = terminator

        if termite_data.get_value(datapoint, 'transtermhp') == '+':
            hairpin = TransTermHPHairpin(terminator_id,
                                         termite_data.get_value(datapoint, "transtermhp POT distance to hairpin"),
                                         termite_data.get_value(datapoint, "transtermhp hairpin score"),
                                         termite_data.get_value(datapoint, "transtermhp hairpin"))
            if not hairpin.contains_multiple_hairpins():
                a_tract_sequence = convert_to_rna(DNASequence(termite_data.get_value(datapoint, "transtermhp a tract")))
                a_tract = ATract(a_tract_sequence)
                u_tract_sequence = convert_to_rna(DNASequence(termite_data.get_value(datapoint, "transtermhp u tract")))
                relative_pot = int(termite_data.get_value(datapoint, "transtermhp POT distance to hairpin"))
                u_tract = UTract(u_tract_sequence, relative_pot)

                terminator = Terminator(start, end, pot, species, chromosome, strand, sequence, te, hairpin, a_tract,
                                        u_tract)
                transtermhp_terminators[terminator_id] = terminator

    terminators = []

    if prioritise_rnafold:
        prioritised_terminators = rnafold_terminators
        other_terminators = transtermhp_terminators
    else:
        prioritised_terminators = transtermhp_terminators
        other_terminators = rnafold_terminators

    for terminator_id, terminator in prioritised_terminators.items():
        terminators.append(terminator)

    for terminator_id, terminator in other_terminators.items():
        if terminator_id not in prioritised_terminators:
            terminators.append(terminator)

    return terminators


def rnafold_hairpins_from_termite(input_file: str, get_rnafold: bool = True, get_transtermhp: bool = False,
                                  get_mutually_exclusive: bool = False, species_column: bool = False):

    if get_rnafold and get_transtermhp:
        assert not get_mutually_exclusive

    assert get_rnafold or get_transtermhp

    rnafold_hairpins = {}
    transtermhp_hairpins = {}

    termite_data = parse_termite_data(input_file, species_column)

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
            for basepair in hairpin.stem.get_basepairs():
                if hairpin.prediction_software == 'RNAFold':

                    if basepair not in basepairs_to_counts_rnafold:

                        basepairs_to_counts_rnafold[basepair] = 0
                    basepairs_to_counts_rnafold[basepair] += 1
                else:
                    if basepair not in basepairs_to_counts_transtermhp:
                        basepairs_to_counts_transtermhp[basepair] = 0
                    basepairs_to_counts_transtermhp[basepair] += 1
                    if basepair == BasePair(Base.A, Base.A, False):
                        print(hairpin.hairpin_sequence)
                        print(hairpin.hairpin_structure)
                        print(hairpin.distance_to_pot)

    print(f"{counter} structures contain multiple hairpins")
    pprint(basepairs_to_counts_rnafold)
    pprint(basepairs_to_counts_transtermhp)

    quantified_terminators = get_termite_terminators(argv[1], species_column=True, te_only=True)
    print(len(quantified_terminators))

    for terminator in quantified_terminators:
        print(terminator.te)
        print(terminator.hairpin.stem.to_vector(10))
    # all_terminators = get_termite_terminators(argv[1], species_column=True, te_only=False)
    # print(len(all_terminators))





