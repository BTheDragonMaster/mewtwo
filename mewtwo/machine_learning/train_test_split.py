from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from mewtwo.embeddings.terminator.terminator import get_terminator_part_sizes, Terminator
from mewtwo.parsers.parse_dnabert_data import parse_dnabert_data

import os
from sys import argv
from math import isclose


class CrossvalidationFold:
    def __init__(self, train: list[Terminator], test: list[Terminator]):
        self.train = train
        self.test = test


def split_data(terminators, test_size: float = 0.5, n_crossval_sets: int = 5):
    max_loop, max_stem, max_a, max_u = get_terminator_part_sizes(terminators)
    x = []
    y = []
    labels = []
    for terminator in terminators:
        x.append(terminator.to_vector(max_loop, max_stem, max_a, max_u))
        y.append(terminator.te)
        labels.append(terminator.species)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=250589)
    sss.get_n_splits(x, labels)
    train_indices, test_indices = next(sss.split(x, labels))

    train_x = []
    train_terminators = []
    train_labels = []

    for index in train_indices:
        train_x.append(x[index])
        train_terminators.append(terminators[index])
        train_labels.append(labels[index])

    test_terminators = []

    for index in test_indices:
        test_terminators.append(terminators[index])

    skf = StratifiedKFold(n_splits=n_crossval_sets, shuffle=True, random_state=100125)
    crossvalidation_sets = {}

    for i, (train_i, test_i) in enumerate(skf.split(train_x, train_labels)):
        train_terminators_c = [train_terminators[j] for j in train_i]
        test_terminators_c = [train_terminators[j] for j in test_i]

        crossvalidation_sets[i] = CrossvalidationFold(train_terminators_c, test_terminators_c)

    return train_terminators, test_terminators, crossvalidation_sets

def bin_data(y, n_bins: int = 10):
    bin_ranges = []
    range_start = 0.0
    step = 1.0 / n_bins
    bins = []

    for i in range(n_bins):
        range_end = range_start + step
        bin_ranges.append((range_start, range_end))
        range_start = range_end

    for i, y_data in enumerate(y):
        bin_index = 0
        bin_range = bin_ranges[bin_index]

        if y_data < 0.0 and not isclose(y_data, 0.0):
            raise ValueError(f"Expected value between 0.0 and 1.0. Got {y_data}")

        while bin_range[1] <= y_data and not isclose(y_data, bin_range[1]):
            bin_index += 1
            try:
                bin_range = bin_ranges[bin_index]
            except IndexError:
                raise ValueError(f"Expected value between 0.0 and 1.0. Got {y_data}")


        bins.append(bin_index)

    return bins

def split_data_from_file(input_file: str, output_dir: str, test_size: float = 0.25, validation_size: float = 0.33333,
                         nr_stratification_bins: int = 10):
    """

    Parameters
    ----------
    input_file: input tabular file with sequences in column one and float in column 2
    output_dir: output directory
    test_size: proportion of ALL data
    validation_size: proportion of TRAINING data
    nr_stratification_bins: number of bins between 0 and 1 to perform stratification on

    Returns
    -------

    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_out = os.path.join(output_dir, 'train.txt')
    validation_out = os.path.join(output_dir, 'validation.txt')
    test_out = os.path.join(output_dir, 'test.txt')

    x, y = parse_dnabert_data(input_file)
    bins = bin_data(y, nr_stratification_bins)

    # Split train and test data
    tts = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=250589)
    tts.get_n_splits(x, bins)

    seen_indices, test_indices = next(tts.split(x, bins))

    seen_x = []
    seen_y = []
    seen_bins = []

    test_x = []
    test_y = []

    for index in seen_indices:
        seen_x.append(x[index])
        seen_y.append(y[index])
        seen_bins.append(bins[index])

    for index in test_indices:
        test_x.append(x[index])
        test_y.append(y[index])

    tvs = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=100125)
    tvs.get_n_splits(seen_x, seen_bins)

    train_indices, validation_indices = next(tvs.split(seen_x, seen_bins))

    train_x = []
    train_y = []

    for index in train_indices:
        train_x.append(seen_x[index])
        train_y.append(seen_y[index])

    validation_x = []
    validation_y = []

    for index in validation_indices:
        validation_x.append(seen_x[index])
        validation_y.append(seen_y[index])

    out_files = test_out, train_out, validation_out
    out_x = test_x, train_x, validation_x
    out_y = test_y, train_y, validation_y

    for i, out_file in enumerate(out_files):

        with open(out_file, 'w') as out:
            for j, seq in enumerate(out_x[i]):
                te = out_y[i][j]
                out.write(f"{seq}\t{te}\n")


if __name__ == "__main__":
    split_data_from_file(argv[1], argv[2])
