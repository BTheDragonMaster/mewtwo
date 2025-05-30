from dataclasses import dataclass
from typing import Optional, Union
from math import isclose


from mewtwo.machine_learning.transformer.config.config_types import LossFunctionType, FinetuningType, SchedulerType


@dataclass
class AdapterConfig:
    rank: int
    alpha: int
    dropout: float

    def __eq__(self, other):
        if type(self) == type(other) and \
                self.rank == other.rank and \
                self.alpha == other.alpha and \
                isclose(self.dropout, other.dropout, rel_tol=0.01):
            return True

        else:
            return False

    @classmethod
    def from_file(cls, input_file) -> Union["AdapterConfig", None]:
        rank = None
        alpha = None
        dropout = None

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                if field == "adapter_r":
                    rank = int(value)
                if field == "adapter_alpha":
                    alpha = int(value)
                if field == "adapter_dropout":
                    dropout = float(value)

        if rank is None or alpha is None or dropout is None:
            return None
        else:
            return AdapterConfig(rank, alpha, dropout)


@dataclass
class LossFunctionConfig:
    type: LossFunctionType
    alpha: Optional[float]

    def __post_init__(self):
        if self.alpha is not None:
            assert -0.00000001 < self.alpha < 1.000000001

    def __eq__(self, other):
        if type(self) == type(other) and self.type == other.type:
            if self.alpha is not None and other.alpha is not None and isclose(self.alpha, other.alpha, rel_tol=0.01):
                return True
            elif self.alpha is None and other.alpha is None:
                return True

        return False

    @classmethod
    def from_file(cls, input_file) -> "LossFunctionConfig":
        function_type = None
        alpha = None

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                if field == "loss_function":
                    function_type = LossFunctionType[value]
                elif field == "loss_function_alpha":
                    alpha = float(value)

        assert function_type is not None

        return LossFunctionConfig(function_type, alpha)


@dataclass
class SchedulerConfig:
    type: SchedulerType
    training_epochs: Optional[int] = None
    warmup_epochs: Optional[int] = None

    def __post_init__(self):
        if self.type == SchedulerType.COS_ANNEAL_WARMUP:
            assert self.training_epochs is not None
            assert self.warmup_epochs is not None
            if self.warmup_epochs > self.training_epochs:
                raise ValueError("Number of warmup steps is greater than the number of training steps.")

    def __eq__(self, other):
        if type(self) == type(other) and \
                self.training_epochs == other.training_epochs and \
                self.type == other.type and \
                self.warmup_epochs == other.warmup_epochs:
            return True

        return False

    @classmethod
    def from_file(cls, input_file) -> Union["SchedulerConfig", None]:
        scheduler_type = None
        training_steps = None
        warmup_steps = None

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                if field == "scheduler":
                    scheduler_type = SchedulerType[value]
                if field == "scheduler_training_steps":
                    training_steps = int(value)
                if field == "scheduler_warmup_steps":
                    warmup_steps = int(value)

        if scheduler_type is None:
            return None
        else:
            return SchedulerConfig(scheduler_type, training_steps, warmup_steps)


@dataclass
class ModelConfig:
    finetuning_mode: FinetuningType
    learning_rate: float
    hidden_layer_dropout: float
    loss_function_config: LossFunctionConfig
    epochs: int
    batch_size: int
    adapter_config: Optional[AdapterConfig] = None
    scheduler_config: Optional[SchedulerConfig] = None

    def __eq__(self, other):
        if self.finetuning_mode == other.finetuning_mode and \
            isclose(self.learning_rate, other.learning_rate, rel_tol=0.01) and \
            isclose(self.hidden_layer_dropout, other.hidden_layer_dropout, rel_tol=0.01) and \
            self.loss_function_config == other.loss_function_config and \
            self.adapter_config == other.adapter_config and \
                self.scheduler_config == other.scheduler_config:

            return True
        else:
            return False

    def write_model_config(self, out_file):

        with open(out_file, 'w') as out:
            out.write(f"finetuning_mode\t{self.finetuning_mode.name}\n")
            out.write(f"learning_rate\t{self.learning_rate:.10f}\n")
            out.write(f"hidden_layer_dropout\t{self.hidden_layer_dropout:.2f}\n")
            out.write(f"loss_function\t{self.loss_function_config.type.name}\n")
            out.write(f"training_epochs\t{self.epochs}\n")
            out.write(f"batch_size\t{self.batch_size}\n")

            if self.loss_function_config.alpha is not None:
                out.write(f"loss_function_alpha\t{self.loss_function_config.alpha}\n")

            if self.adapter_config is not None:
                out.write(f"adapter_r\t{self.adapter_config.rank}\n")
                out.write(f"adapter_alpha\t{self.adapter_config.alpha}\n")
                out.write(f"adapter_dropout\t{self.adapter_config.dropout}\n")

            if self.scheduler_config is not None:
                out.write(f"scheduler\t{self.scheduler_config.type.name}\n")
                if self.scheduler_config.training_epochs is not None:
                    out.write(f"scheduler_training_steps\t{self.scheduler_config.training_epochs}\n")
                if self.scheduler_config.warmup_epochs is not None:
                    out.write(f"scheduler_warmup_steps\t{self.scheduler_config.warmup_epochs}\n")

    @classmethod
    def from_file(cls, input_file):

        adapter_config = AdapterConfig.from_file(input_file)
        scheduler_config = SchedulerConfig.from_file(input_file)
        loss_function_config = LossFunctionConfig.from_file(input_file)

        field_to_value = {}

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                field_to_value[field] = value

        return cls(FinetuningType.from_string_description(field_to_value["finetuning_mode"]),
                   float(field_to_value["learning_rate"]),
                   float(field_to_value["hidden_layer_dropout"]),
                   loss_function_config,
                   int(field_to_value["training_epochs"]),
                   int(field_to_value["batch_size"]),
                   adapter_config,
                   scheduler_config)
