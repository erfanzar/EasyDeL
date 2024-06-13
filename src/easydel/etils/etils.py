import logging
from dataclasses import dataclass
from typing import Literal, List, Tuple, Dict, Any

import argparse


@dataclass
class EasyDeLOptimizers:
    """The code snippet is defining a data class called `EasyDeLOptimizers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """

    ADAFACTOR: Literal["adafactor"] = "adafactor"
    LION: Literal["lion"] = "lion"
    ADAMW: Literal["adamw"] = "adamw"


@dataclass
class EasyDeLSchedulers:
    """The code snippet is defining a data class called `EasyDeLSchedulers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """

    LINEAR: Literal["linear"] = "linear"
    COSINE: Literal["cosine"] = "cosine"
    NONE: Literal["none"] = "none"
    WARM_UP_COSINE: Literal["warm_up_cosine"] = "warm_up_cosine"
    WARM_UP_LINEAR: Literal["warm_up_linear"] = "warm_up_linear"


@dataclass
class EasyDeLGradientCheckPointers:
    """The code snippet is defining a data class called `EasyDeLGradientCheckPointers` using the `@dataclass`
    decorator. A data class is a class that is primarily used to store data, and it automatically
    generates special methods such as `__init__`, `__repr__`, and `__eq__` based on the class
    attributes.
    """

    EVERYTHING_SAVEABLE: Literal["everything_saveable"] = "everything_saveable"
    NOTHING_SAVEABLE: Literal["nothing_saveable"] = "nothing_saveable"
    CHECKPOINT_DOTS: Literal["checkpoint_dots"] = "checkpoint_dots"
    CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS: Literal[
        "checkpoint_dots_with_no_batch_dims"
    ] = "checkpoint_dots_with_no_batch_dims"


AVAILABLE_GRADIENT_CHECKPOINTS = Literal[
    "everything_saveable",
    "nothing_saveable",
    "checkpoint_dots",
    "checkpoint_dots_with_no_batch_dims",
]

AVAILABLE_SCHEDULERS = Literal[
    "linear",
    "cosine",
    "none",
    "warm_up_cosine",
    "warm_up_linear"
]

AVAILABLE_OPTIMIZERS = Literal["adafactor", "lion", "adamw"]


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
    """Function to set the logging level of all loggers to the specified level.

    Args:
        level: int: The logging level to set. Defaults to
            logging.WARNING.
    """
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)


def define_flags_with_default(
        _required_fields: List = None, **kwargs
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Defines flags with default values using argparse.

    Args:
        _required_fields: A dictionary with required flag names
        **kwargs: Keyword arguments representing flag names and default values.

    Returns:
        A tuple containing:
            - An argparse.Namespace object containing parsed arguments.
            - A dictionary mapping flag names to default values.
    """
    _required_fields = _required_fields if _required_fields is not None else []
    parser = argparse.ArgumentParser()

    default_values = {}

    for name, value in kwargs.items():
        default_values[name] = value

        # Custom type handling:
        if isinstance(value, tuple):
            # For tuples, use a custom action to convert the string to a tuple of ints
            parser.add_argument(
                f"--{name}",
                type=str,  # Read as string
                default=str(value),  # Store default as string
                help=f"Value for {name} (comma-separated integers)",
                action=StoreTupleAction,
            )
        else:
            # For other types, infer type from default value
            parser.add_argument(
                f"--{name}", type=type(value), default=value, help=f"Value for {name}"
            )

    args = parser.parse_args()
    for key in _required_fields:
        if getattr(args, key) == "":
            raise ValueError(f"Required field {key} for argument parser.")
    return args, default_values


class StoreTupleAction(argparse.Action):
    """Custom action to store a comma-separated string as a tuple of ints."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, tuple(int(v) for v in values.split(",")))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value for {option_string}: {values} "
                f"(should be comma-separated integers)"
            )
