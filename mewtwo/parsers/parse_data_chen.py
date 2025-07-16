from sys import argv

from mewtwo.parsers.tabular import Tabular
from mewtwo.data_processing.compute_te import ts_to_te
from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin
from mewtwo.embeddings.terminator.a_tract import ATract
from mewtwo.embeddings.terminator.u_tract import UTract
from mewtwo.embeddings.terminator.terminator import Terminator


def get_chen_data(input_file, type_column):
    if not type_column:
        chen_data = Tabular(input_file, [0])
    else:
        chen_data = Tabular(input_file, [1])

    return chen_data


def chen_to_dnabert_input(input_file, output_file, type_column: bool = True):
    with open(output_file, 'w') as out:
        chen_data = get_chen_data(input_file, type_column)
        for datapoint in chen_data.data:
            sequence = chen_data.get_value(datapoint, "Sequence")
            ts = float(chen_data.get_value(datapoint, "Average Strength"))
            te = max(0.0, ts_to_te(ts))
            if 1.0 >= te >= -0.0000001:
                out.write(f"{sequence}\t{te}\n")


def get_chen_terminators(input_file, type_column: bool = True):
    chen_data = get_chen_data(input_file, type_column)
    terminators = []
    for datapoint in chen_data.data:
        ts = float(chen_data.get_value(datapoint, "Average Strength"))
        te = max(0.0, ts_to_te(ts))
        if 1.0 >= te >= -0.0000001:
            terminator_type = chen_data.get_value(datapoint, 'Type')
            if terminator_type == 'Synthetic':
                is_synthetic = True
            else:
                is_synthetic = False

            free_energy = chen_data.get_value(datapoint, 'dGH')
            hairpin_structure = chen_data.get_value(datapoint, 'Structure')
            hairpin_sequence = chen_data.get_value(datapoint, 'Hairpin')
            sequence = chen_data.get_value(datapoint, 'Sequence')
            a_tract_seq = chen_data.get_value(datapoint, 'A-tract')
            u_tract_seq = chen_data.get_value(datapoint, 'U-tract')
            terminator_id = chen_data.get_value(datapoint, 'Name')

            hairpin = RNAFoldHairpin(terminator_id, free_energy, hairpin_sequence, hairpin_structure)
            if not hairpin.contains_multiple_hairpins():

                a_tract = ATract(a_tract_seq)
                u_tract = UTract(u_tract_seq)
                terminator = Terminator(hairpin, a_tract, u_tract, sequence, termination_efficiency=te,
                                        is_synthetic=is_synthetic)
                terminators.append(terminator)

    return terminators


if __name__ == "__main__":
    chen_to_dnabert_input(argv[1], argv[2])
