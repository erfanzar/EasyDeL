# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""EasyDeL utilities for enumerations and type definitions.

This module provides enumerations and type definitions used throughout the EasyDeL
framework for configuration and runtime options. It includes enums for optimizers,
schedulers, quantization methods, platforms, backends, and various configuration
options.

Enumerations:
    EasyDeLOptimizers: Available optimization algorithms
    EasyDeLSchedulers: Learning rate scheduling strategies
    EasyDeLGradientCheckPointers: Gradient checkpointing methods
    EasyDeLPlatforms: Kernel execution platforms
    EasyDeLBackends: JAX backend hardware targets

Type Aliases:
    AVAILABLE_GRADIENT_CHECKPOINTS: Literal type for gradient checkpoint options
    AVAILABLE_SCHEDULERS: Literal type for scheduler options
    AVAILABLE_OPTIMIZERS: Literal type for optimizer options
    AVAILABLE_ATTENTION_MECHANISMS: Literal type for attention implementations
    AVAILABLE_SPARSE_MODULE_TYPES: Literal type for sparse matrix formats
    AVAILABLE_GRADIENT_CHECKPOINT_TARGETS: Literal type for checkpoint names used in models

Constants:
    DEFAULT_ATTENTION_MECHANISM: Default attention mechanism to use

Functions:
    define_flags_with_default: Create argparse flags from default values

Example:
    >>> from easydel.infra.etils import EasyDeLOptimizers, EasyDeLBackends
    >>>
    >>> # Use enums for configuration
    >>> optimizer = EasyDeLOptimizers.ADAMW
    >>> backend = EasyDeLBackends.TPU
    >>>
    >>> # Parse command-line arguments with defaults
    >>> args, defaults = define_flags_with_default(
    ...     learning_rate=1e-3,
    ...     batch_size=32,
    ...     _required_fields=["model_name"]
    ... )
"""

import argparse
import typing as tp
from enum import Enum


class EasyDeLOptimizers(str, Enum):
    """
    Enumeration of available optimizers in the EasyDeL library.

    Attributes:
        ADAFACTOR: Represents the Adafactor optimizer.
        LION: Represents the Lion optimizer.
        ADAMW: Represents the AdamW optimizer.
        RMSPROP: Represents the RMSprop optimizer.
    """

    ADAFACTOR = "adafactor"
    ADAMW = "adamw"
    MARS = "mars"
    MUON = "muon"
    RMSPROP = "rmsprop"
    LION = "lion"
    SKEW = "skew"
    QUAD = "quad"


class EasyDeLSchedulers(str, Enum):
    """
    Enumeration of available learning rate schedulers in EasyDeL.

    Attributes:
        NONE: Indicates no scheduler should be used.
        LINEAR: Represents a linear learning rate decay scheduler.
        COSINE: Represents a cosine annealing learning rate scheduler.
    """

    NONE = None
    LINEAR = "linear"
    COSINE = "cosine"


class EasyDeLGradientCheckPointers(str, Enum):
    """
    Enumeration of gradient checkpointing strategies available in EasyDeL.

    Gradient checkpointing is a technique to reduce memory usage during training
    by recomputing activations during the backward pass instead of storing them.

    Attributes:
        EVERYTHING_SAVEABLE: Checkpoints residuals, attentions, and hidden states.
            This is the most memory-intensive checkpointing strategy.
        NOTHING_SAVEABLE: Checkpoints only the residuals.
            This strategy saves the most memory but requires more recomputation.
        CHECKPOINT_DOTS: Checkpoints matrix multiplications and intermediate activations.
        CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS: Similar to CHECKPOINT_DOTS but avoids checkpointing
            operations involving batch dimensions.
        NONE: No gradient checkpointing is applied.
    """

    EVERYTHING_SAVEABLE = "everything_saveable"
    NOTHING_SAVEABLE = "nothing_saveable"
    CHECKPOINT_DOTS = "checkpoint_dots"
    CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS = "checkpoint_dots_with_no_batch_dims"
    NONE = ""
    DOTS_SAVEABLE = "dots_saveable"
    DOTS_WITH_NO_BATCH_DIMS_AVAILABLE = "dots_with_no_batch_dims_saveable"
    SAVE_ANYTHING_EXCEPT_THESE_NAMES = "save_anything_except_these_names"
    SAVE_ANY_NAMES_BUT_THESE = "save_any_names_but_these"
    SAVE_ONLY_THESE_NAMES = "save_only_these_names"
    SAVE_FROM_BOTH_POLICIES = "save_from_both_policies"


class EasyDeLPlatforms(str, Enum):
    """
    Enumeration of platforms or kernel execution backends supported by EasyDeL.

    This allows selecting optimized kernel implementations for different hardware
    or software environments.

    Attributes:
        JAX: Use standard JAX kernel implementations.
        TRITON: Use Triton-based kernel implementations (often for GPUs).
        PALLAS: Use Pallas-based kernel implementations (often for TPUs).
    """

    JAX = "jax"
    TRITON = "triton"
    PALLAS = "pallas"


class EasyDeLBackends(str, Enum):
    """
    Enumeration of JAX backend types supported by EasyDeL.

    Specifies the target hardware device type for JAX computations.

    Attributes:
        CPU: Use the CPU backend.
        GPU: Use the GPU backend.
        TPU: Use the TPU backend.
        TT: Use the Tenstorrent backend.
    """

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    TT = "tt"


AVAILABLE_GRADIENT_CHECKPOINTS = tp.Literal[
    "everything_saveable",
    "nothing_saveable",
    "checkpoint_dots",
    "checkpoint_dots_with_no_batch_dims",
    "",
    "dots_saveable",
    "dots_with_no_batch_dims_saveable",
    "save_anything_except_these_names",
    "save_any_names_but_these",
    "save_only_these_names",
    "save_from_both_policies",
]

AVAILABLE_SCHEDULERS = tp.Literal["linear", "cosine", "none"]

AVAILABLE_OPTIMIZERS = tp.Literal["adafactor", "adamw", "mars", "muon", "rmsprop", "lion", "skew", "quad"]

AVAILABLE_MOE_METHODS = tp.Literal["fused_moe", "standard_moe", "dense_moe"]
AVAILABLE_ATTENTION_MECHANISMS = tp.Literal[
    "auto",
    "vanilla",
    "flash_attn2",
    "blocksparse",
    "ring",
    "cudnn",
    "blockwise",
    "sdpa",
    "autoregressive_decodeattn",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "page_attention",
]

DEFAULT_ATTENTION_MECHANISM = "vanilla"
AVAILABLE_SPARSE_MODULE_TYPES = tp.Literal["bcoo", "bcsr", "coo", "csr"]

AVAILABLE_GRADIENT_CHECKPOINT_TARGETS = tp.Literal[
    "attn_dense",
    "attn_key",
    "attn_key_value",
    "attn_output",
    "attn_qkv",
    "attn_query",
    "attn_receptance",
    "attn_value",
    "attn_weights",
    "embeddings",
    "layer_output",
    "lm_head_output",
    "mlp_down",
    "mlp_gate",
    "mlp_output",
    "mlp_up",
    "model_output",
    "moe_expert_output",
    "moe_gate_logits",
    "moe_output",
    "moe_router_logits",
    "normed_input",
    "residual",
]


def define_flags_with_default(
    _required_fields: list | None = None, **kwargs
) -> tuple[argparse.Namespace, dict[str, tp.Any]]:
    """Defines command-line flags using argparse based on provided keyword arguments.

    This function dynamically creates argparse arguments for each key-value pair in `kwargs`.
    It infers the argument type from the default value and handles tuple types specifically.
    It also supports marking certain fields as required.

    Args:
        _required_fields (tp.List, optional): A list of flag names that are mandatory.
            An error will be raised if these flags are not provided or are empty strings.
            Defaults to None.
        **kwargs: Keyword arguments where keys are flag names (without `--`) and values
            are their default values.

    Returns:
        tp.Tuple[argparse.Namespace, tp.Dict[str, tp.Any]]: A tuple containing:
            - An `argparse.Namespace` object holding the parsed command-line arguments.
            - A dictionary mapping the original flag names to their default values.

    Raises:
        ValueError: If a required field (from `_required_fields`) is not provided
            or is an empty string on the command line.
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
            parser.add_argument(f"--{name}", type=type(value), default=value, help=f"Value for {name}")

    args = parser.parse_args()
    for key in _required_fields:
        if getattr(args, key) == "":
            raise ValueError(f"Required field {key} for argument parser.")
    return args, default_values


class StoreTupleAction(argparse.Action):
    """
    Custom argparse action to parse a comma-separated string into a tuple of integers.

    This action is used by `define_flags_with_default` when a default value is a tuple.
    It takes the comma-separated string provided on the command line and attempts to
    convert each part into an integer, storing the result as a tuple in the namespace.

    Raises:
        argparse.ArgumentTypeError: If the provided value cannot be parsed as a comma-separated
            list of integers.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, tuple(int(v) for v in values.split(",")))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value for {option_string}: {values} (should be comma-separated integers)"
            ) from None
