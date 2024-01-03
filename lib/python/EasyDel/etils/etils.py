from dataclasses import dataclass
from typing import Literal


@dataclass
class EasyDelOptimizers:
    """
    The code snippet is defining a data class called `EasyDelOptimizers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """
    ADAFACTOR: str = "adafactor"
    LION: str = "lion"
    ADAMW: str = 'adamw'


@dataclass
class EasyDelSchedulers:
    """
    The code snippet is defining a data class called `EasyDelSchedulers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """
    LINEAR: str = "linear"
    COSINE: str = "cosine"
    NONE: str = "none"
    WARM_UP_COSINE: str = "warm_up_cosine"
    WARM_UP_LINEAR: str = "warm_up_linear"


@dataclass
class EasyDelGradientCheckPointers:
    """
    The code snippet is defining a data class called `EasyDelGradientCheckPointers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """
    EVERYTHING_SAVEABLE: str = "everything_saveable"
    NOTHING_SAVEABLE: str = "nothing_saveable"
    CHECKPOINT_DOTS: str = "checkpoint_dots"
    CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS: str = "checkpoint_dots_with_no_batch_dims"


AVAILABLE_GRADIENT_CHECKPOINTS = Literal[
    "everything_saveable",
    "nothing_saveable",
    "checkpoint_dots",
    "checkpoint_dots_with_no_batch_dims"
]

AVAILABLE_SCHEDULERS = Literal[
    "linear",
    "cosine",
    "none",
    "warm_up_cosine",
    "warm_up_linear"
]

AVAILABLE_OPTIMIZERS = Literal[
    "adafactor",
    "lion",
    'adamw'
]
