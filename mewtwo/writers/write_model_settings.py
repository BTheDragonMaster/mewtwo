
from argparse import ArgumentParser
import os

from mewtwo.parsers.parse_model_config import ModelConfig, AdapterConfig, SchedulerConfig, LossFunctionConfig

from mewtwo.machine_learning.transformer.config.config_types import FinetuningType, LossFunctionType, SchedulerType


def parse_arguments():
    parser = ArgumentParser(description="Write config files for hyperparameter optimization")
    parser.add_argument('-o', type=str, required=True, help="Output directory")
    args = parser.parse_args()
    return args


def get_hyperoptimization_configs():
    model_configs = []

    batch_sizes = [4, 8]
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    hidden_layer_dropouts = [0.1, 0.2]

    for batch_size in batch_sizes:
        for finetuning_type in FinetuningType:
            if finetuning_type.name == "ADAPTER":
                lora_rank_options = [4]
                lora_dropout_options = [0.05]
            else:
                lora_rank_options = [None]
                lora_dropout_options = [None]

            for lora_r in lora_rank_options:
                if lora_r is not None:
                    lora_alpha = 2 * lora_r
                else:
                    lora_alpha = None

                for lora_dropout in lora_dropout_options:
                    if lora_r:
                        adapter_config = AdapterConfig(lora_r, lora_alpha, lora_dropout)
                    else:
                        adapter_config = None

                    for scheduler_option in [s for s in SchedulerType] + [None]:
                        if scheduler_option == SchedulerType.COS_ANNEAL_WARMUP:

                            training_epochs = 10
                            warmup_epochs_options = [2]
                        else:
                            training_epochs = None
                            warmup_epochs_options = [None]

                        if scheduler_option == SchedulerType.COS_ANNEAL_WARMUP:

                            for warmup_epochs_option in warmup_epochs_options:
                                if scheduler_option is None:
                                    scheduler_config = None
                                else:
                                    scheduler_config = SchedulerConfig(scheduler_option, training_epochs,
                                                                       warmup_epochs_option)

                                for loss_function in LossFunctionType:
                                    if loss_function not in LossFunctionType.USES_SPEARMAN and loss_function not in LossFunctionType.WEIGHTED:
                                        if loss_function not in LossFunctionType.NEEDS_ALPHA:
                                            loss_function_alpha_options = [None]
                                        else:
                                            loss_function_alpha_options = [0.25, 0.5, 0.75]

                                        for alpha_option in loss_function_alpha_options:
                                            loss_function_config = LossFunctionConfig(loss_function, alpha_option)

                                            for learning_rate in learning_rates:
                                                for hidden_layer_dropout in hidden_layer_dropouts:
                                                    model_config = ModelConfig(finetuning_type, learning_rate,
                                                                               hidden_layer_dropout,
                                                                               loss_function_config, 0, batch_size,
                                                                               adapter_config,
                                                                               scheduler_config)
                                                    model_configs.append(model_config)

    return model_configs


def main():
    args = parse_arguments()
    if not os.path.exists(args.o):
        os.mkdir(args.o)

    model_configs = get_hyperoptimization_configs()
    for i, model_config in enumerate(model_configs):
        out_file = os.path.join(args.o, f"model_{i + 1:03}.config")
        model_config.write_model_config(out_file)


if __name__ == "__main__":

    main()


