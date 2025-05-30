import os
from sys import argv
from statistics import mean, stdev

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir


def write_average_performance(crossvalidation_dir: str, out_file: str) -> None:
    test_scores: list[float] = []
    pearsons: list[float] = []
    spearmans: list[float] = []

    for folder_name, folder_path in iterate_over_dir(crossvalidation_dir, get_dirs=True):
        if 'crossvalidation_results' in folder_name:
            performance_file = os.path.join(folder_path, "performance.txt")
            with open(performance_file, 'r') as performances:
                performances.readline()
                test_score, pearson, spearman = performances.readline().strip().split('\t')
                test_scores.append(float(test_score))
                pearsons.append(float(pearson))
                spearmans.append(float(spearman))

    av_test = mean(test_scores)
    av_pearson = mean(pearsons)
    av_spearman = mean(spearmans)

    stdev_test = stdev(test_scores)
    stdev_pearson = stdev(pearsons)
    stdev_spearman = stdev(spearmans)

    with open(out_file, 'w') as out:
        out.write(f"\ttest_score\tpearson\tspearman\n")
        out.write(f"Mean\t{av_test}\t{av_pearson}\t{av_spearman}\n")
        out.write(f"Stdev\t{stdev_test}\t{stdev_pearson}\t{stdev_spearman}\n")

if __name__ == "__main__":
    write_average_performance(argv[1], argv[2])
