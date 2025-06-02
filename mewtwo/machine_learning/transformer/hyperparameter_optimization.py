import os
from argparse import ArgumentParser

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir
from mewtwo.machine_learning.transformer.model import load_model
from mewtwo.machine_learning.transformer.config.config_types import SchedulerType
from mewtwo.machine_learning.transformer.finetune_bert import finetune


def parse_arguments():
    parser = ArgumentParser(description="Perform hyperparameter optimisation from directory of configuration files")
    parser.add_argument('-c', type=str, required=True, help="Directory of model configuration files")
    parser.add_argument('-i', type=str, required=True, help="Path to training data")
    parser.add_argument('-v', type=str, required=True, help="Path to validation data")
    parser.add_argument('-o', type=str, required=True, help="Output directory")
    parser.add_argument('-e', type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    for file_name, file_path in iterate_over_dir(args.c, extension='.config'):
        model = load_model(args.i, args.v, config_file=file_path)
        summary_file = os.path.join(args.o, "summary.txt")

        with open(summary_file, 'w') as summary:
            finetune(model, summary, args.e, args.o)
