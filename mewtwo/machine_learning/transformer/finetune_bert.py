import argparse
import os

from mewtwo.machine_learning.transformer.model import load_model
from mewtwo.machine_learning.transformer.config.config_types import SchedulerType


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True,
                        help="Tabular input data, with sequence in one column and efficiency in the second")
    parser.add_argument("-v", type=str, required=True,
                        help="Tabular input data, with sequence in one column and efficiency in the second")
    parser.add_argument("-c", type=str, default=None, help="Path to configuration file.")
    parser.add_argument("-o", type=str, required=True, help="Output directory")
    parser.add_argument("-e", type=int, default=15, help="Nr of epochs")

    parser.add_argument("-s", action="store_true", help="If given, save model to output_folder/checkpoint.pt")
    parser.add_argument("-m", type=str, default=None, help="If given, train from this checkpoint")

    args = parser.parse_args()

    if args.c is None:
        assert args.m

    if args.c is not None and args.m is not None:
        print("Warning: config file given alongside existing model. Config file will be ignored.")

    if args.c is None and args.m is None:
        raise ValueError("Model config or previous model checkpoint must be given.")

    return args


def finetune(model, summary, epochs, out_dir):
    config_file = os.path.join(out_dir, "model.config")

    current_epoch = model.config.epochs

    for i in range(epochs):
        current_epoch += i + 1
        print(f"LR at epoch {current_epoch}: {model.optimizer.param_groups[0]['lr']}")

        out_file = os.path.join(out_dir, f"epoch_{current_epoch:03d}.txt")

        avg_train_loss = model.train_model()
        avg_loss, all_preds, all_labels = model.evaluate_model()

        print(f"Epoch {current_epoch}\t- Train loss:\t{avg_train_loss:.4f}")
        print(f" \t- Eval loss:\t{avg_loss:.4f}")

        summary.write(f"{current_epoch}\t{avg_train_loss:.5f}\t{avg_loss:.5f}\n")

        with open(out_file, 'w') as out:
            out.write("actual\tpredicted\n")
            for j, prediction in enumerate(all_preds):
                label = all_labels[j]
                out.write(f"{label}\t{prediction}\n")

        model.update_epoch(current_epoch)

        if model.scheduler is not None and model.config.scheduler_config.type == SchedulerType.REDUCE_ON_PLATEAU:
            model.scheduler.step(avg_loss)

    model.config.write_model_config(config_file)


def main():

    args = parse_arguments()

    if not os.path.exists(args.o):
        os.mkdir(args.o)

    summary_file = os.path.join(args.o, "summary.txt")

    if args.m is not None:
        summary = open(summary_file, 'a')
        model = load_model(args.i, args.v, model_checkpoint=args.m)
    elif args.c is not None:
        summary = open(summary_file, 'w')
        summary.write("epoch\taverage_train_loss\taverage_eval_loss\n")
        model = load_model(args.i, args.v, config_file=args.c)
    else:
        raise ValueError("Model or config file must be given")

    current_epoch = model.config.epochs

    if args.m is None:
        avg_loss, all_preds, all_labels = model.evaluate_model()
        print(f"Epoch {current_epoch}\t- Eval loss:\t{avg_loss:.4f}")

    finetune(model, summary, args.e, args.o)

    summary.close()

    if args.s is not None:
        model.save_model_checkpoint(os.path.join(args.o, "checkpoint.pt"))


if __name__ == "__main__":
    main()
