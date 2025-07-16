import os
from argparse import ArgumentParser
from shutil import copy

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir
from mewtwo.machine_learning.transformer.model import load_model
from mewtwo.machine_learning.transformer.finetune_bert import finetune


def parse_arguments():
    parser = ArgumentParser(description="Perform hyperparameter optimisation from directory of configuration files")
    parser.add_argument('-c', type=str, required=True, help="Directory of model configuration files")
    parser.add_argument('-i', type=str, required=True, help="Path to training data")
    parser.add_argument('-v', type=str, required=True, help="Path to validation data")
    parser.add_argument('-o', type=str, required=True, help="Output directory")
    parser.add_argument('-e', type=int, default=20, help="Training epochs")
    parser.add_argument('-f', action="store_true", help="If given, save failed contigs to folder named 'failed'")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    if not os.path.exists(args.o):
        os.mkdir(args.o)

    failed_configs = []

    for file_name, file_path in iterate_over_dir(args.c, extension='.config'):

        out_dir = os.path.join(args.o, file_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        try:
            model = load_model(args.i, args.v, config_file=file_path)
            summary_file = os.path.join(out_dir, "summary.txt")

            with open(summary_file, 'w') as summary:
                finetune(model, summary, args.e, out_dir, header=True)

        except Exception as e:
            f"Could not train model with config {file_name}: {e}"
            failed_configs.append((file_path, file_name))

    if args.f and failed_configs:
        failed_dir = os.path.join(args.o, "failed")
        if not os.path.exists(failed_dir):
            os.mkdir(failed_dir)

        for file_path, file_name in failed_configs:

            failed_path = os.path.join(failed_dir, f"{file_name}.config")

            if os.path.exists(failed_path):
                os.remove(failed_path)

            copy(file_path, failed_path)


if __name__ == "__main__":
    main()
