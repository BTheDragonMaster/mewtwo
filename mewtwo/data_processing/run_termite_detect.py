"""
Required folder architecture:

bigwig/
    experiment_1_forward/
        replicate_1_forward.bw
        replicate_2_forward.bw
        replicate_3_forward.bw
        ...
    experiment_2_forward/
        replicate_1_forward.bw
        ...
    ...
genome/
    genome.fasta
"""

import argparse
from argparse import ArgumentParser
import os

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir


def parse_arguments() -> argparse.Namespace:
    """
    Return arguments from ArgumentParser object
    """
    parser = ArgumentParser(description="Python script to run TERMITe. Note: only works on in-house folder \
    architecture specified at the top of this file.")
    parser.add_argument('--idr_threshold', type=float, default=0.05, help="IDR threshold")
    parser.add_argument('--window_size', type=int, default=10, help="Number of nucleotides to report on either side of \
    the peak summit")
    args = parser.parse_args()
    return args


def get_termite_command(input_folder_name: str, input_folder: str, output_folder: str, prefix: str, strand: str,
                        genome_location: str, idr_threshold: float = 0.05, window_size: int = 10) -> str:
    assert strand in ['forward', 'reverse']
    file_string = "--rna-3prime-ends"
    sample_string = "--sample-names"
    for sample_name, _ in iterate_over_dir(input_folder, extension='.bw'):
        file_name = sample_name + '.bw'
        if (file_name.endswith('forward.bw') and strand == 'forward') or \
                (file_name.endswith('reverse.bw') and strand == 'reverse'):
            file_string += f" /data/{input_folder_name}/{file_name}"
            sample_string += f" {sample_name}"
    docker_command = f"docker run --rm -v $(pwd):/data --platform=linux/amd64 termite find_stable_rna_ends {file_string} {sample_string} --idr-threshold {idr_threshold} --out-dir /data/{output_folder} --name-prefix {prefix} --strand {strand} --genome {genome_location} --window-size {window_size}"

    return docker_command


def run_termite_multiple(input_folder: str, idr_threshold: float = 0.05, window_size: int = 10) -> None:
    """

    Parameters
    ----------
    input_folder: folder with the subfolders 'bigwig' and 'genome'

    """
    bigwig_folder = os.path.join(input_folder, 'bigwig')
    genome_folder = os.path.join(input_folder, 'genome')
    output_folder = 'termite'
    genome = None
    for file_name, file_path in iterate_over_dir(genome_folder, '.fasta'):
        genome = f"/data/genome/{file_name}.fasta"
    assert genome is not None

    for sub_folder_name, sub_folder_path in iterate_over_dir(bigwig_folder, get_dirs=True):

        if sub_folder_name.endswith('forward'):
            strand = 'forward'

        elif sub_folder_name.endswith('reverse'):
            strand = 'reverse'
        else:
            continue

        sample_name = '_'.join(sub_folder_name.split('_')[:-1])
        command = get_termite_command(f"bigwig/{sub_folder_name}", sub_folder_path, output_folder,
                                      prefix=sample_name, strand=strand, genome_location=genome,
                                      idr_threshold=idr_threshold, window_size=window_size)
        print(f"Running {command}...")
        os.system(command)


def main() -> None:
    args = parse_arguments()
    run_termite_multiple(os.getcwd(), args.idr_threshold, args.window_size)

if __name__ == "__main__":
    main()





