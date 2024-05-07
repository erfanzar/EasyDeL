import logging
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


def get_logger(name, level: int = logging.INFO) -> logging.Logger:
    """
    Function to create and configure a logger.
    :param name: str: The name of the logger.
    :param level: int: The logging level. Defaults to logging.INFO.
    :return logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    # Set the logging level
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def set_loggers_level(level: int = logging.WARNING):
    """
    Function to set the logging level of all loggers to the specified level.
    :param level: int: The logging level to set. Defaults to logging.WARNING.
    """
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)
