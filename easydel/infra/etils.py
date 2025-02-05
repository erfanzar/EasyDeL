# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import typing as tp
from enum import Enum


class EasyDeLOptimizers(str, Enum):
	"""
	Enum defining available optimizers for EasyDeL.
	Each enum member represents a different optimization algorithm.
	"""

	ADAFACTOR = "adafactor"
	LION = "lion"
	ADAMW = "adamw"
	RMSPROP = "rmsprop"


class EasyDeLSchedulers(str, Enum):
	"""
	Enum defining available schedulers for EasyDeL.
	Each enum member represents a different learning rate schedule.
	"""

	NONE = None
	LINEAR = "linear"
	COSINE = "cosine"


class EasyDeLGradientCheckPointers(str, Enum):
	"""
	Enum defining available gradient checkpointing strategies for EasyDeL.
	Each enum member represents a different checkpointing approach.
	"""

	EVERYTHING_SAVEABLE = "everything_saveable"
	NOTHING_SAVEABLE = "nothing_saveable"
	CHECKPOINT_DOTS = "checkpoint_dots"
	CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS = "checkpoint_dots_with_no_batch_dims"
	NONE = ""


class EasyDeLQuantizationMethods(str, Enum):
	"""
	Enum defining available quantization strategies for EasyDeL.
	Each enum member represents a different quantization approach.
	"""

	NONE = None
	NF4 = "nf4"
	A8BIT = "8bit"


class EasyDeLPlatforms(str, Enum):
	"""
	Enum defining available platforms for EasyDeL.
	Each enum member represents a different kernel usage approach.
	"""

	JAX = "jax"
	TRITON = "triton"
	PALLAS = "pallas"


class EasyDeLBackends(str, Enum):
	"""
	Enum defining available backends for EasyDeL.
	Each enum member represents a different kernel usage approach.
	"""

	CPU = "cpu"
	GPU = "gpu"
	TPU = "tpu"


AVAILABLE_GRADIENT_CHECKPOINTS = tp.Literal[
	"everything_saveable",
	"nothing_saveable",
	"checkpoint_dots",
	"checkpoint_dots_with_no_batch_dims",
	"",
]

AVAILABLE_SCHEDULERS = tp.Literal[
	"linear",
	"cosine",
	"none",
]

AVAILABLE_OPTIMIZERS = tp.Literal[
	"adafactor",
	"lion",
	"adamw",
	"rmsprop",
]


AVAILABLE_ATTENTION_MECHANISMS = tp.Literal[
	"vanilla",
	"flash_attn2",
	"splash",
	"ring",
	"cudnn",
	"blockwise",
	"sdpa",
]


DEFAULT_ATTENTION_MECHANISM = "vanilla"
AVAILABLE_SPARSE_MODULE_TYPES = tp.Literal["bcoo", "bcsr", "coo", "csr"]


def define_flags_with_default(
	_required_fields: tp.List = None, **kwargs
) -> tp.Tuple[argparse.Namespace, tp.Dict[str, tp.Any]]:
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
			) from None
