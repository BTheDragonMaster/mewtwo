import argparse
import os
from typing import TextIO

from mewtwo.machine_learning.transformer.model import load_model, Model
from mewtwo.machine_learning.transformer.config.config_types import SchedulerType, EarlyStoppingMetricType

from scipy.stats import pearsonr, spearmanr


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


def metric_has_improved(old_metric: float, new_metric: float, metric_type: EarlyStoppingMetricType) -> bool:
    if metric_type in EarlyStoppingMetricType.MAX_METRICS:
        if new_metric > old_metric:
            return True
        else:
            return False
    elif metric_type in EarlyStoppingMetricType.MIN_METRICS:
        if new_metric < old_metric:
            return True
        else:
            return False
    else:
        raise ValueError(f"Unrecognised early stopping metric: {metric_type.name}")


def get_metric(eval_loss: float, pearson: float, spearman: float, metric_type: EarlyStoppingMetricType) -> float:
    if metric_type == EarlyStoppingMetricType.PEARSON_R:
        return pearson
    elif metric_type == EarlyStoppingMetricType.SPEARMAN_R:
        return spearman
    elif metric_type == EarlyStoppingMetricType.EVAL_LOSS:
        return eval_loss
    else:
        raise ValueError(f"Unrecognised early stopping metric: {metric_type.name}")


def finetune(model: Model, summary: TextIO, epochs: int, out_dir: str, header=False) -> None:
    if header:
        summary.write("epoch\taverage_train_loss\taverage_eval_loss\tpearsonr\tspearmanr\n")
    config_file = os.path.join(out_dir, "model.config")

    starting_epoch = model.config.epochs
    epochs_without_improvement = 0

    best_model_path = os.path.join(out_dir, "best_checkpoint.pt")
    if model.config.early_stopping_config and \
            model.config.early_stopping_config.metric in EarlyStoppingMetricType.MAX_METRICS:
        best_metric = -1.1
    else:
        best_metric = 1.1

    for i in range(epochs):
        current_epoch = starting_epoch + i + 1
        print(f"LR at epoch {current_epoch}: {model.optimizer.param_groups[0]['lr']}")

        out_file = os.path.join(out_dir, f"epoch_{current_epoch:03d}.txt")

        avg_train_loss = model.train_model()
        avg_loss, all_preds, all_labels = model.evaluate_model()

        pearson = pearsonr(all_labels, all_preds).statistic
        spearman = spearmanr(all_labels, all_preds).statistic

        print(f"Epoch {current_epoch}\t- Train loss:\t{avg_train_loss:.4f}")
        print(f" \t- Eval loss:\t{avg_loss:.4f}")

        print(f" \t- PearsonR:\t{pearson:.4f}")
        print(f" \t- SpearmanR:\t{spearman:.4f}")

        with open(out_file, 'w') as out:
            out.write("actual\tpredicted\n")
            for j, prediction in enumerate(all_preds):
                label = all_labels[j]
                out.write(f"{label}\t{prediction}\n")

        summary.write(f"{current_epoch}\t{avg_train_loss:.5f}\t{avg_loss:.5f}\t{pearson:.5f}\t{spearman:.5f}\n")

        model.update_epoch(current_epoch)

        if model.scheduler is not None and \
                model.config.scheduler_config.type in SchedulerType.REDUCE_ON_PLATEAU_SCHEDULERS:
            model.scheduler.step(avg_loss)

        if model.config.early_stopping_config is not None:
            metric = get_metric(avg_loss, pearson, spearman, model.config.early_stopping_config.metric)
            if metric_has_improved(best_metric, metric, model.config.early_stopping_config.metric):
                best_metric = metric
                epochs_without_improvement = 0
                model.save_model_checkpoint(best_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= model.config.early_stopping_config.patience:
                    print(f"Early stopping at epoch {current_epoch}")
                    break

    model.config.write_model_config(config_file)


def main():

    args = parse_arguments()

    if not os.path.exists(args.o):
        os.mkdir(args.o)

    summary_file = os.path.join(args.o, "summary.txt")

    if args.m is not None:
        summary = open(summary_file, 'a')
        write_header = False
        model = load_model(args.i, args.v, model_checkpoint=args.m)
    elif args.c is not None:
        summary = open(summary_file, 'w')
        write_header = True
        model = load_model(args.i, args.v, config_file=args.c)
    else:
        raise ValueError("Model or config file must be given")

    current_epoch = model.config.epochs

    if args.m is None:
        avg_loss, all_preds, all_labels = model.evaluate_model()
        print(f"Epoch {current_epoch}\t- Eval loss:\t{avg_loss:.4f}")

    finetune(model, summary, args.e, args.o, header=write_header)

    summary.close()

    if args.s is not None:
        model.save_model_checkpoint(os.path.join(args.o, "checkpoint.pt"))


if __name__ == "__main__":
    main()
