
from argparse import ArgumentParser
import os

from mewtwo.parsers.parse_model_config import ModelConfig, AdapterConfig, SchedulerConfig, LossFunctionConfig, \
    EarlyStoppingConfig, HiddenLayerConfig

from mewtwo.machine_learning.transformer.config.config_types import FinetuningType, LossFunctionType, SchedulerType, \
    EarlyStoppingMetricType


def parse_arguments():
    parser = ArgumentParser(description="Write config files for hyperparameter optimization")
    parser.add_argument('--batch_sizes', type=int, nargs='*', default=[4, 8])
    parser.add_argument('--learning_rates', type=float, nargs='*', default=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    parser.add_argument('--finetuning_types', type=str, nargs='*', default=["LINEAR_HEAD", "ADAPTER"])
    parser.add_argument('--dropout', type=float, nargs='*', default=[0.1, 0.2, 0.3])
    parser.add_argument('--lora_rank', type=int, nargs='*', default=[4, 8])
    parser.add_argument('--lora_dropout', type=float, nargs='*', default=[0.0, 0.05, 0.1])
    parser.add_argument('--training_epochs', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, nargs='*', default=[3, 6, 10])
    parser.add_argument('--loss_functions', type=str, nargs='*', default=['MSE', 'MSE_PEARSON', 'MSE_SPEARMAN',
                                                                          'WEIGHTED_MSE', 'WEIGHTED_MSE_PEARSON',
                                                                          "WEIGHTED_MSE_SPEARMAN", 'PEARSON', "SPEARMAN"])
    parser.add_argument('--loss_function_alpha', type=float, nargs='*', default=[0.25, 0.5, 0.75])
    parser.add_argument('--early_stopping_metrics', type=str, nargs='*', default=["PEARSON_R", "SPEARMAN_R", "EVAL_LOSS"])
    parser.add_argument('--early_stopping_patiences', type=int, nargs='*', default=[2, 3])
    parser.add_argument('--use_early_stopping', action="store_true")
    parser.add_argument('--scheduler_types', type=str, nargs='*', default=["COS_ANNEAL_WARMUP", "REDUCE_ON_PLATEAU",
                                                                           "REDUCE_ON_PLATEAU_WARMUP", "WARMUP_ONLY"])
    parser.add_argument('--plateau_patiences', type=int, nargs='*', default=[2, 3])
    parser.add_argument('--plateau_factors', type=float, nargs='*', default=[0.5])
    parser.add_argument('--second_layer_dim', type=int, nargs='*', default=[])
    parser.add_argument('-o', type=str, required=True, help="Output directory")
    args = parser.parse_args()
    return args


def get_hyperoptimization_configs(batch_sizes, learning_rates,
                                  finetuning_types,
                                  hidden_layer_dropouts, lora_rank,
                                  lora_dropout, training_epochs,
                                  warmup_epochs_options, loss_functions, loss_function_alpha,
                                  scheduler_types, use_early_stopping,
                                  early_stopping_metrics, early_stopping_patiences,
                                  plateau_patiences, plateau_factors, second_layer_dims):
    model_configs = []

    for batch_size in batch_sizes:
        for finetuning_type_str in finetuning_types:
            finetuning_type = FinetuningType.from_string_description(finetuning_type_str)
            if finetuning_type.name != "ADAPTER":
                lora_rank_options = [None]
                lora_dropout_options = [None]
            else:
                lora_rank_options = lora_rank[:]
                lora_dropout_options = lora_dropout[:]

            for lora_r in lora_rank_options:
                if lora_r is not None:
                    lora_alpha = 2 * lora_r
                else:
                    lora_alpha = None

                for lora_dropout_option in lora_dropout_options:
                    if lora_r:
                        adapter_config = AdapterConfig(lora_r, lora_alpha, lora_dropout_option)
                    else:
                        adapter_config = None

                    for scheduler_config_string in scheduler_types:
                        scheduler_type = SchedulerType[scheduler_config_string]
                        if scheduler_type not in SchedulerType.WARMUP_SCHEDULERS:
                            warmup_epochs = [None]
                        else:
                            warmup_epochs = warmup_epochs_options[:]

                        if scheduler_type not in SchedulerType.REDUCE_ON_PLATEAU_SCHEDULERS:
                            plateau_patience_options = [None]
                            plateau_factor_options = [None]
                        else:
                            plateau_patience_options = plateau_patiences[:]
                            plateau_factor_options = plateau_factors[:]

                        for warmup_epochs_option in warmup_epochs:
                            for plateau_patience in plateau_patience_options:
                                for plateau_factor in plateau_factor_options:
                                    scheduler_config = SchedulerConfig(scheduler_type, training_epochs,
                                                                       warmup_epochs_option, plateau_patience,
                                                                       plateau_factor)

                                    for loss_function_str in loss_functions:
                                        loss_function = LossFunctionType.from_string_description(loss_function_str)
                                        if loss_function not in LossFunctionType.NEEDS_ALPHA:
                                            loss_function_alpha_options = [None]
                                        else:
                                            loss_function_alpha_options = loss_function_alpha[:]

                                        for alpha_option in loss_function_alpha_options:
                                            loss_function_config = LossFunctionConfig(loss_function, alpha_option)

                                            for learning_rate in learning_rates:
                                                for hidden_layer_dropout in hidden_layer_dropouts:
                                                    if not second_layer_dims:
                                                        second_layer_dims = [None]

                                                    for second_layer_dim in second_layer_dims:
                                                        hidden_layer_config = HiddenLayerConfig(hidden_layer_dropout,
                                                                                                second_layer_dim)
                                                        early_stopping_config = None

                                                        if not use_early_stopping:
                                                            early_stopping_metric_options = [None]
                                                            early_stopping_patience_options = [None]

                                                        else:
                                                            early_stopping_metric_options = early_stopping_metrics[:]
                                                            early_stopping_patience_options = early_stopping_patiences[:]

                                                        for early_stopping_metric in early_stopping_metric_options:
                                                            for early_stopping_patience in early_stopping_patience_options:
                                                                if early_stopping_metric is not None and \
                                                                        early_stopping_patience is not None:
                                                                    early_stopping_config = EarlyStoppingConfig(
                                                                        EarlyStoppingMetricType[early_stopping_metric],
                                                                        early_stopping_patience)
                                                                model_config = ModelConfig(finetuning_type, learning_rate,
                                                                                           hidden_layer_config,
                                                                                           loss_function_config, 0,
                                                                                           batch_size,
                                                                                           early_stopping_config,
                                                                                           adapter_config,
                                                                                           scheduler_config)

                                                                model_configs.append(model_config)

    return model_configs


def main():
    args = parse_arguments()
    if not os.path.exists(args.o):
        os.mkdir(args.o)

    if args.use_early_stopping:
        early_stopping_metric_options = args.early_stopping_metrics
        early_stopping_patience_options = args.early_stopping_patiences
    else:
        early_stopping_metric_options = [None]
        early_stopping_patience_options = [None]

    model_configs = get_hyperoptimization_configs(args.batch_sizes, args.learning_rates, args.finetuning_types,
                                                  args.dropout, args.lora_rank,
                                                  args.lora_dropout, args.training_epochs,
                                                  args.warmup_epochs, args.loss_functions,
                                                  args.loss_function_alpha, args.scheduler_types,
                                                  args.use_early_stopping,
                                                  early_stopping_metric_options, early_stopping_patience_options,
                                                  args.plateau_patiences, args.plateau_factors, args.second_layer_dim)

    for i, model_config in enumerate(model_configs):
        out_file = os.path.join(args.o, f"model_{i + 1:03}.config")
        model_config.write_model_config(out_file)


if __name__ == "__main__":

    main()
