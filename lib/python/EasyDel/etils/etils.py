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
    ADAFACTOR: Literal["adafactor"] = "adafactor"  # Fix Pycharm Debugging Issue
    LION: Literal["lion"] = "lion"  # Fix Pycharm Debugging Issue
    ADAMW: Literal["adamw"] = 'adamw'  # Fix Pycharm Debugging Issue


@dataclass
class EasyDelSchedulers:
    """
    The code snippet is defining a data class called `EasyDelSchedulers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """
    LINEAR: Literal["linear"] = "linear"  # Fix Pycharm Debugging Issue
    COSINE: Literal["cosine"] = "cosine"  # Fix Pycharm Debugging Issue
    NONE: Literal["none"] = "none"  # Fix Pycharm Debugging Issue
    WARM_UP_COSINE: Literal["warm_up_cosine"] = "warm_up_cosine"  # Fix Pycharm Debugging Issue
    WARM_UP_LINEAR: Literal["warm_up_linear"] = "warm_up_linear"  # Fix Pycharm Debugging Issue


@dataclass
class EasyDelGradientCheckPointers:
    """
    The code snippet is defining a data class called `EasyDelGradientCheckPointers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """
    EVERYTHING_SAVEABLE: Literal["everything_saveable"] = "everything_saveable"  # Fix Pycharm Debugging Issue
    NOTHING_SAVEABLE: Literal["nothing_saveable"] = "nothing_saveable"  # Fix Pycharm Debugging Issue
    CHECKPOINT_DOTS: Literal["checkpoint_dots"] = "checkpoint_dots"  # Fix Pycharm Debugging Issue
    CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS: Literal["checkpoint_dots_with_no_batch_dims"] = \
        "checkpoint_dots_with_no_batch_dims"  # Fix Pycharm Debugging Issue


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
