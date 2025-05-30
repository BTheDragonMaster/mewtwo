from enum import Enum


class SchedulerType(Enum):
    REDUCE_ON_PLATEAU = 1
    COSINE_ANNEALING_WARM_START = 2