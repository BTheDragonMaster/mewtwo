import os
from shutil import move
from sys import argv

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir


def sort_bigwig_files(input_folder: str) -> None:

    for bw_file_name, bw_file_path in iterate_over_dir(input_folder, '.bw'):
        experiment_type, experiment_number, sample_number, strand = bw_file_name.split('_')
        sub_folder = os.path.join(input_folder, f"{experiment_type}_{experiment_number}_{strand}")
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
        new_file_path = os.path.join(sub_folder, f"{bw_file_name}.bw")
        move(bw_file_path, new_file_path)
        print(f"Moved {bw_file_path} to {new_file_path}")


if __name__ == "__main__":
    sort_bigwig_files(argv[1])
