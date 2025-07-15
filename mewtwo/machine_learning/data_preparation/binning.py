from math import isclose


def bin_data(y, n_bins: int = 5):
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