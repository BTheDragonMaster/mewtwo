from statistics import mean

from mewtwo.machine_learning.data_preparation.binning import bin_data


def get_sample_weights(targets: list[float], n_bins=5) -> list[float]:
    bins = bin_data(targets, n_bins=n_bins)
    weights = []
    for i in range(len(targets)):
        count = bins.count(bins[i])
        bin_weight = 1.0 / (count + 1e-6)

        weights.append(bin_weight)

    normalized_weights = []

    for weight in weights:
        normalized_weights.append(weight / mean(weights))

    return normalized_weights


