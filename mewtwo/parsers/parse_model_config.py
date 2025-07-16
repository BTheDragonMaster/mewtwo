from dataclasses import dataclass
from typing import Optional, Union
from math import isclose


from mewtwo.machine_learning.transformer.config.config_types import EarlyStoppingMetricType, LossFunctionType, \
    FinetuningType, SchedulerType


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
    plateau_patience: Optional[int] = None
    factor: Optional[float] = None

    def __post_init__(self):
        if self.type in SchedulerType.WARMUP_SCHEDULERS:
            assert self.warmup_epochs is not None

            if self.type == SchedulerType.COS_ANNEAL_WARMUP:
                assert self.training_epochs is not None
                if self.warmup_epochs > self.training_epochs:
                    raise ValueError("Number of warmup steps is greater than the number of training steps.")
        if self.type in SchedulerType.REDUCE_ON_PLATEAU_SCHEDULERS:
            assert self.plateau_patience is not None
            assert self.factor is not None

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
        training_epochs = None
        warmup_epochs = None
        plateau_patience = None
        factor = None

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                if field == "scheduler":
                    scheduler_type = SchedulerType[value]
                if field == "scheduler_training_epochs":
                    training_epochs = int(value)
                if field == "scheduler_warmup_epochs":
                    warmup_epochs = int(value)
                if field == "plateau_patience":
                    plateau_patience = int(value)
                if field == "factor":
                    factor = float(value)

        if scheduler_type is None:
            return None
        else:
            return SchedulerConfig(scheduler_type, training_epochs, warmup_epochs, plateau_patience, factor)


@dataclass
class EarlyStoppingConfig:
    metric: EarlyStoppingMetricType
    patience: int

    def __post_init__(self):
        assert self.metric is not None
        assert self.patience is not None

    def __eq__(self, other):
        if type(self) == type(other) and self.metric == other.metric and self.patience == other.patience:
            return True
        return False

    @classmethod
    def from_file(cls, input_file) -> Union["EarlyStoppingConfig", None]:
        metric = None
        patience = None

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                if field == "early_stopping_patience":
                    metric = EarlyStoppingMetricType[value]
                if field == "early_stopping_patience":
                    patience = int(value)

        if metric is None or patience is None:
            return None
        else:
            return EarlyStoppingConfig(metric, patience)


@dataclass
class HiddenLayerConfig:
    dropout: float
    second_layer_dim: Optional[int]

    def __eq__(self, other):
        if type(self) == type(other) and isclose(self.dropout, other.dropout, rel_tol=0.01) and \
                self.second_layer_dim == other.second_layer_dim:
            return True
        return False

    @classmethod
    def from_file(cls, input_file) -> Union["HiddenLayerConfig", None]:
        dropout = None
        second_layer_dim = None

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                if field == "hidden_layer_dropout":
                    dropout = float(value)
                if field == "second_layer_dim":
                    second_layer_dim = int(value)

        if dropout is None:
            raise ValueError("Hidden layer config must specify dropout")
        else:
            return HiddenLayerConfig(dropout, second_layer_dim)

@dataclass
class ModelConfig:
    finetuning_mode: FinetuningType
    learning_rate: float
    hidden_layer_config: HiddenLayerConfig
    loss_function_config: LossFunctionConfig
    epochs: int
    batch_size: int
    early_stopping_config: Optional[EarlyStoppingConfig] = None
    adapter_config: Optional[AdapterConfig] = None
    scheduler_config: Optional[SchedulerConfig] = None

    def __eq__(self, other):
        if self.finetuning_mode == other.finetuning_mode and \
            isclose(self.learning_rate, other.learning_rate, rel_tol=0.01) and \
            self.hidden_layer_config == other.hidden_layer_config and \
            self.loss_function_config == other.loss_function_config and \
            self.adapter_config == other.adapter_config and \
                self.scheduler_config == other.scheduler_config and \
                self.early_stopping_config == other.early_stopping_config:

            return True
        else:
            return False

    def write_model_config(self, out_file):
        assert out_file.endswith('.config')

        with open(out_file, 'w') as out:
            out.write(f"finetuning_mode\t{self.finetuning_mode.name}\n")
            out.write(f"learning_rate\t{self.learning_rate:.10f}\n")
            out.write(f"hidden_layer_dropout\t{self.hidden_layer_config.dropout:.2f}\n")
            if self.hidden_layer_config.second_layer_dim:
                out.write(f"second_layer_dim\t{self.hidden_layer_config.second_layer_dim}\n")
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
                    out.write(f"scheduler_training_epochs\t{self.scheduler_config.training_epochs}\n")
                if self.scheduler_config.warmup_epochs is not None:
                    out.write(f"scheduler_warmup_epochs\t{self.scheduler_config.warmup_epochs}\n")
                if self.scheduler_config.plateau_patience is not None:
                    out.write(f"plateau_patience\t{self.scheduler_config.plateau_patience}\n")
                if self.scheduler_config.factor is not None:
                    out.write(f"factor\t{self.scheduler_config.factor}\n")

            if self.early_stopping_config is not None:
                out.write(f"early_stopping_patience\t{self.early_stopping_config.patience}\n")
                out.write(f"early_stopping_metric\t{self.early_stopping_config.metric.name}\n")

    @classmethod
    def from_file(cls, input_file):

        adapter_config = AdapterConfig.from_file(input_file)
        scheduler_config = SchedulerConfig.from_file(input_file)
        loss_function_config = LossFunctionConfig.from_file(input_file)
        early_stopping_config = EarlyStoppingConfig.from_file(input_file)
        hidden_layer_config = HiddenLayerConfig.from_file(input_file)

        field_to_value = {}

        with open(input_file, 'r') as model_config:
            for line in model_config:
                line = line.strip()
                field, value = line.split('\t')
                field_to_value[field] = value

        return cls(FinetuningType.from_string_description(field_to_value["finetuning_mode"]),
                   float(field_to_value["learning_rate"]),
                   hidden_layer_config,
                   loss_function_config,
                   int(field_to_value["training_epochs"]),
                   int(field_to_value["batch_size"]),
                   early_stopping_config,
                   adapter_config,
                   scheduler_config)
