# NOTE: this script is only relevant for data generated in-house
from shutil import move
import os
from sys import argv

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir


def rename_file(directory: str, bigwig_file_location: str, bigwig_file_name: str) -> None:
    strand = bigwig_file_name.split('_')[-1]
    sample_name = f"{bigwig_file_name.split('_')[1]}_{bigwig_file_name.split('_')[2]}_{bigwig_file_name.split('_')[3]}_{strand}"
    new_location = os.path.join(directory, f"{sample_name}.bw")

    move(bigwig_file_location, new_location)


if __name__ == "__main__":
    directory = argv[1]

    for bw_file_name, bw_file_path in iterate_over_dir(directory, '.bw'):

        rename_file(directory, bw_file_path, bw_file_name)
