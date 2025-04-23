#!usr/bin/env python
import argparse
import os
import subprocess
from argparse import ArgumentParser

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir


def parse_arguments() -> argparse.Namespace:
    """
    Return arguments from ArgumentParser object
    """
    parser = ArgumentParser(description="Convert all BAM files in a folder to BIGWIG format using deepTools")
    parser.add_argument('-i', required=True, type=str, help="Input directory containing BAM files")
    parser.add_argument('-o', required=True, type=str, help="Output directory for BIGWIG files")
    args = parser.parse_args()
    return args


def convert_to_bigwig(input_bam_file: str, output_bigwig_file: str, strand: str) -> None:
    """
    Call deepTools bamCoverage file with CPM normalization on BAM file to obtain BIGWIG file
    :param input_bam_file: str, input BAM file
    :param output_bigwig_file: str, output BIGWIG file
    """
    assert strand in ["forward", "reverse"]
    command = ['bamCoverage', '-b', input_bam_file, '-o', output_bigwig_file,
               '--normalizeUsing', 'CPM', "--outFileFormat", "bigwig", "--binSize", "1",
               "--Offset", "1",  "--samFlagExclude", "256", "--filterRNAstrand", strand]
    subprocess.call(command)


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.o):
        os.mkdir(args.o)
    for bam_file_name, bam_file in iterate_over_dir(args.i, extension='.bam'):
        bigwig_file_forward = os.path.join(args.o, f"{bam_file_name}_forward.bw")
        bigwig_file_reverse = os.path.join(args.o, f"{bam_file_name}_reverse.bw")
        convert_to_bigwig(bam_file, bigwig_file_forward, "forward")
        convert_to_bigwig(bam_file, bigwig_file_reverse, "reverse")


