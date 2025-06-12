import os
from sys import argv

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir
from mewtwo.parsers.tabular import Tabular


def sort_models_by_performance(config_dir):
    models_and_performances = []
    for model_name, folder_path in iterate_over_dir(config_dir, get_dirs=True):
        summary_file = os.path.join(folder_path, "summary.txt")
        summary_data = Tabular(summary_file, [0])
        best_performance = max(summary_data.get_column("pearsonr"))
        models_and_performances.append((model_name, float(best_performance)))

    models_and_performances.sort(key=lambda x: x[1], reverse=True)
    for model, performance in models_and_performances:
        print(f"{model}\t{performance:.4f}\n")


if __name__ == "__main__":
    sort_models_by_performance(argv[1])
