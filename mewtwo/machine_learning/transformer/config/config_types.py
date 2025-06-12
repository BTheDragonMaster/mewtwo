from enum import Enum, Flag


class EarlyStoppingMetricType(Flag):
    EVAL_LOSS = 1
    SPEARMAN_R = 2
    PEARSON_R = 4

    MAX_METRICS = PEARSON_R | SPEARMAN_R
    MIN_METRICS = EVAL_LOSS


class LossFunctionType(Flag):
    MSE = 1
    PEARSON = 2
    MSE_PEARSON = 4
    WEIGHTED_MSE = 8
    WEIGHTED_MSE_PEARSON = 16
    SPEARMAN = 32
    MSE_SPEARMAN = 64
    WEIGHTED_MSE_SPEARMAN = 128

    WEIGHTED = WEIGHTED_MSE | WEIGHTED_MSE_PEARSON | WEIGHTED_MSE_SPEARMAN
    NEEDS_ALPHA = MSE_PEARSON | WEIGHTED_MSE_PEARSON | MSE_SPEARMAN | WEIGHTED_MSE_SPEARMAN
    CORRELATION_ONLY = PEARSON | SPEARMAN
    USES_SPEARMAN = WEIGHTED_MSE_SPEARMAN | SPEARMAN | MSE_SPEARMAN

    @staticmethod
    def from_string_description(string_description) -> "LossFunctionType":
        return LossFunctionType[string_description.upper()]


class FinetuningType(Enum):
    LINEAR_HEAD = 1
    ADAPTER = 2

    @staticmethod
    def from_string_description(string_description) -> "FinetuningType":
        return FinetuningType[string_description.upper()]


class SchedulerType(Flag):
    REDUCE_ON_PLATEAU = 1
    COS_ANNEAL_WARMUP = 2  # Cosine annealing with warmup
    REDUCE_ON_PLATEAU_WARMUP = 4
    WARMUP_ONLY = 8

    WARMUP_SCHEDULERS = COS_ANNEAL_WARMUP | REDUCE_ON_PLATEAU_WARMUP | WARMUP_ONLY
    REDUCE_ON_PLATEAU_SCHEDULERS = REDUCE_ON_PLATEAU | REDUCE_ON_PLATEAU_WARMUP

    @staticmethod
    def from_string_description(string_description) -> "SchedulerType":
        return SchedulerType[string_description.upper()]
