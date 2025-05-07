from mewtwo.parsers.tabular import Tabular
from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin
from mewtwo.embeddings.bases import BasePair, Base
from mewtwo.embeddings.terminator.terminator import Terminator, get_terminator_part_sizes
from mewtwo.embeddings.sequence import DNASequence, convert_to_rna, convert_to_dna
from mewtwo.embeddings.terminator.a_tract import ATract
from mewtwo.embeddings.terminator.u_tract import UTract
from mewtwo.machine_learning.random_forest.train_random_forest import train_random_forest
from mewtwo.machine_learning.train_test_split import split_data
from mewtwo.machine_learning.prepare_data import terminators_to_ml_input
from mewtwo.machine_learning.neural_network import train_nn

from sys import argv
import os
from pprint import pprint
from statistics import median

def termite_to_dnabert_input(input_file: str, output_dir: str, species_column: bool = True) -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    terminators = get_termite_terminators(input_file, species_column=True, te_only=True)
    spec_to_term = sort_by_species(terminators)

    bacillus_terminators = []
    ecoli_terminators = []

    for spec, term in spec_to_term.items():
        if 'Bacillus' in spec and '(d)' in spec:
            bacillus_terminators.extend(term)
        elif 'Escherichia' in spec and '(a)' in spec:
            ecoli_terminators.extend(term)

    all_out_file = os.path.join(output_dir, 'all.txt')
    ecoli_out_file = os.path.join(output_dir, 'ecoli.txt')
    bacillus_out_file = os.path.join(output_dir, 'bacillus.txt')

    with open(all_out_file, 'w') as all_out:
        with open(ecoli_out_file, 'w') as ecoli_out:
            with open(bacillus_out_file, 'w') as bacillus_out:
                for terminator in ecoli_terminators:
                    terminator_data = f"{convert_to_dna(terminator.sequence).sequence}\t{terminator.te}\n"
                    all_out.write(terminator_data)
                    ecoli_out.write(terminator_data)
                for terminator in bacillus_terminators:
                    terminator_data = f"{convert_to_dna(terminator.sequence).sequence}\t{terminator.te}\n"
                    all_out.write(terminator_data)
                    bacillus_out.write(terminator_data)



def parse_termite_data(input_file: str, species_column: bool) -> Tabular:
    if not species_column:
        termite_data = Tabular(input_file, [0, 6])
    else:
        termite_data = Tabular(input_file, [0, 1, 7])

    return termite_data


def sort_by_species(terminators: list[Terminator]) -> dict[str, list[Terminator]]:

    species_to_terminators: dict[str, list[Terminator]] = {}
    for terminator in terminators:
        if terminator.species not in species_to_terminators:
            species_to_terminators[terminator.species] = []
        species_to_terminators[terminator.species].append(terminator)

    return species_to_terminators

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
        elif 0 <= float(te) <= 100:
            te = float(te)
        else:
            print(datapoint, te)
            te = None

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

    quantified_terminators = get_termite_terminators(argv[1], species_column=True, te_only=True)
    species_to_terminators = sort_by_species(quantified_terminators)
    bacillus_terminators = []
    ecoli_terminators = []


    for species, species_terminators in species_to_terminators.items():
        if 'Bacillus' in species and '(d)' in species:
            bacillus_terminators.extend(species_terminators)
        elif 'Escherichia' in species and '(a)' in species:
            ecoli_terminators.extend(species_terminators)

    all_terminators = bacillus_terminators + ecoli_terminators
    train_terminators, test_terminators, crossvalidation_sets = split_data(all_terminators, test_size=0.1)

    for crossval_nr, crossvalidation_set in crossvalidation_sets.items():
        train_random_forest(crossvalidation_set.train, crossvalidation_set.test, one_hot=True)

    # train_terminators, test_terminators, _ = split_data(ecoli_terminators, test_size=0.1)
    #
    # train_random_forest(train_terminators, test_terminators, one_hot=True)
    #
    # train_terminators, test_terminators, _ = split_data(all_terminators, test_size=0.1)
    #
    # train_random_forest(train_terminators, test_terminators, one_hot=True)

    print(min([t.te for t in all_terminators]), max([t.te for t in all_terminators]))

    termite_to_dnabert_input(argv[1], argv[2], species_column=True)

    # train_nn(train_terminators, test_terminators)





    # for terminator in quantified_terminators:
    #     print(terminator.te)
    #     print(terminator.to_vector(max_loop_size=max_loop, max_stem_size=max_stem, a_tract_size=max_a, u_tract_size=max_u))
    #     print(terminator.to_vector(max_loop_size=max_loop, max_stem_size=max_stem, a_tract_size=max_a,
    #                                u_tract_size=max_u, one_hot=True))

    # print(max_loop, max_stem, max_a, max_u)

    # all_terminators = get_termite_terminators(argv[1], species_column=True, te_only=False)
    # print(len(all_terminators))





