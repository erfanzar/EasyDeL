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

The module serves as a central location for all enumeration-based configurations,
ensuring type safety and consistency across the EasyDeL codebase. By using these
enums and type aliases, the framework provides clear documentation of valid options
and enables IDE autocompletion support.

Enumerations:
    EasyDeLOptimizers: Available optimization algorithms for training.
    EasyDeLSchedulers: Learning rate scheduling strategies.
    EasyDeLGradientCheckPointers: Gradient checkpointing methods for memory optimization.
    EasyDeLPlatforms: Kernel execution platforms (JAX, Triton, Pallas).
    EasyDeLBackends: JAX backend hardware targets (CPU, GPU, TPU, TT).

Type Aliases:
    AVAILABLE_GRADIENT_CHECKPOINTS: Literal type for gradient checkpoint options.
    AVAILABLE_SCHEDULERS: Literal type for scheduler options.
    AVAILABLE_OPTIMIZERS: Literal type for optimizer options.
    AVAILABLE_MOE_METHODS: Literal type for mixture-of-experts methods.
    AVAILABLE_ATTENTION_MECHANISMS: Literal type for attention implementations.
    AVAILABLE_SPARSE_MODULE_TYPES: Literal type for sparse matrix formats.
    AVAILABLE_GRADIENT_CHECKPOINT_TARGETS: Literal type for checkpoint target names.

Constants:
    DEFAULT_ATTENTION_MECHANISM: Default attention mechanism to use ("vanilla").

Classes:
    StoreTupleAction: Custom argparse action for parsing tuple arguments.

Functions:
    define_flags_with_default: Create argparse flags from default values.

Example:
    Basic usage with enums for configuration:

    >>> from easydel.infra.etils import EasyDeLOptimizers, EasyDeLBackends
    >>>
    >>> # Use enums for configuration
    >>> optimizer = EasyDeLOptimizers.ADAMW
    >>> backend = EasyDeLBackends.TPU
    >>>
    >>> # Check enum values
    >>> print(optimizer.value)
    'adamw'

    Using define_flags_with_default for command-line argument parsing:

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
    """Enumeration of available optimizers in the EasyDeL library.

    This enum provides a type-safe way to specify which optimizer to use during
    model training. Each optimizer has different characteristics in terms of
    memory usage, convergence speed, and suitability for different model sizes.

    Inherits from both `str` and `Enum` to allow direct string comparisons and
    serialization while maintaining enum benefits.

    Attributes:
        ADAFACTOR: Adafactor optimizer, memory-efficient for large models.
            Automatically scales learning rate and uses factored second moments.
        ADAMW: AdamW optimizer with decoupled weight decay regularization.
            The most commonly used optimizer for transformer training.
        MARS: MARS (Memory-efficient Adaptive Rate Scaling) optimizer.
            Designed for efficient training with adaptive learning rates.
        MUON: Muon optimizer, an experimental optimizer variant.
            Provides alternative optimization dynamics.
        RMSPROP: RMSprop optimizer with adaptive learning rates.
            Divides gradient by running average of recent gradient magnitudes.
        LION: Lion (EvoLved Sign Momentum) optimizer.
            Memory-efficient optimizer using sign operations.
        SKEW: Skew optimizer for specialized training scenarios.
            Applies skewed gradient updates.
        QUAD: Quadratic optimizer variant.
            Uses quadratic approximations for parameter updates.

    Example:
        >>> from easydel.infra.etils import EasyDeLOptimizers
        >>> optimizer = EasyDeLOptimizers.ADAMW
        >>> print(optimizer.value)
        'adamw'
        >>> # String comparison works directly
        >>> assert optimizer == "adamw"
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
    """Enumeration of available learning rate schedulers in EasyDeL.

    Learning rate schedulers adjust the learning rate during training to improve
    convergence and final model performance. This enum provides the available
    scheduling strategies.

    Inherits from both `str` and `Enum` for string compatibility.

    Attributes:
        NONE: No scheduler is applied; learning rate remains constant.
            Use when you want full control over learning rate or for debugging.
        LINEAR: Linear decay from initial to final learning rate.
            Decreases learning rate linearly over the training steps.
        COSINE: Cosine annealing scheduler.
            Smoothly decreases learning rate following a cosine curve,
            often with optional warmup period.

    Example:
        >>> from easydel.infra.etils import EasyDeLSchedulers
        >>> scheduler = EasyDeLSchedulers.COSINE
        >>> print(scheduler.value)
        'cosine'
        >>> # Check if scheduling is enabled
        >>> if scheduler != EasyDeLSchedulers.NONE:
        ...     print("Learning rate scheduling is enabled")
    """

    NONE = None
    LINEAR = "linear"
    COSINE = "cosine"


class EasyDeLGradientCheckPointers(str, Enum):
    """Enumeration of gradient checkpointing strategies available in EasyDeL.

    Gradient checkpointing (also known as activation checkpointing) is a technique
    to reduce memory usage during training by recomputing activations during the
    backward pass instead of storing them. This allows training larger models or
    using larger batch sizes at the cost of additional computation.

    Different strategies provide different trade-offs between memory savings and
    computational overhead.

    Attributes:
        EVERYTHING_SAVEABLE: Checkpoints residuals, attentions, and hidden states.
            This is the most memory-intensive checkpointing strategy, saving
            almost everything. Minimal recomputation but highest memory usage.
        NOTHING_SAVEABLE: Checkpoints only the residuals.
            This strategy saves the most memory but requires the most recomputation
            during the backward pass.
        CHECKPOINT_DOTS: Checkpoints matrix multiplications and intermediate activations.
            Good balance between memory savings and recomputation overhead.
        CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS: Similar to CHECKPOINT_DOTS but avoids
            checkpointing operations involving batch dimensions.
            Useful for specific memory optimization patterns.
        NONE: No gradient checkpointing is applied.
            All activations are stored in memory. Fastest but uses most memory.
        DOTS_SAVEABLE: Saves dot product operations specifically.
            Targets the most memory-intensive operations.
        DOTS_WITH_NO_BATCH_DIMS_AVAILABLE: Saves dots without batch dimensions.
            Variant that excludes batch dimension operations.
        SAVE_ANYTHING_EXCEPT_THESE_NAMES: Policy to save all except specified names.
            Allows fine-grained control by exclusion list.
        SAVE_ANY_NAMES_BUT_THESE: Alias for exclusion-based saving policy.
            Similar to SAVE_ANYTHING_EXCEPT_THESE_NAMES.
        SAVE_ONLY_THESE_NAMES: Policy to save only specified names.
            Allows fine-grained control by inclusion list.
        SAVE_FROM_BOTH_POLICIES: Combines inclusion and exclusion policies.
            Uses both include and exclude lists for maximum flexibility.

    Example:
        >>> from easydel.infra.etils import EasyDeLGradientCheckPointers
        >>> # For memory-constrained training
        >>> checkpoint_policy = EasyDeLGradientCheckPointers.NOTHING_SAVEABLE
        >>> # For faster training with more memory
        >>> checkpoint_policy = EasyDeLGradientCheckPointers.NONE
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
    """Enumeration of platforms or kernel execution backends supported by EasyDeL.

    This enum allows selecting optimized kernel implementations for different
    hardware or software environments. Each platform provides different levels
    of optimization and hardware support.

    Attributes:
        JAX: Use standard JAX kernel implementations.
            Most portable option, works across all JAX-supported hardware.
            Good default choice for compatibility.
        TRITON: Use Triton-based kernel implementations.
            Optimized for NVIDIA GPUs with custom CUDA kernels.
            Provides significant speedups for GPU workloads.
        PALLAS: Use Pallas-based kernel implementations.
            Optimized for Google TPUs and other accelerators.
            Recommended for TPU deployments.

    Example:
        >>> from easydel.infra.etils import EasyDeLPlatforms
        >>> # For GPU with Triton support
        >>> platform = EasyDeLPlatforms.TRITON
        >>> # For TPU deployment
        >>> platform = EasyDeLPlatforms.PALLAS
        >>> # For maximum compatibility
        >>> platform = EasyDeLPlatforms.JAX
    """

    JAX = "jax"
    TRITON = "triton"
    PALLAS = "pallas"


class EasyDeLBackends(str, Enum):
    """Enumeration of JAX backend types supported by EasyDeL.

    Specifies the target hardware device type for JAX computations. This enum
    is used to configure which hardware backend JAX should use for executing
    operations.

    Attributes:
        CPU: Use the CPU backend.
            Available on all systems. Useful for debugging and small-scale
            testing. Slowest option for large models.
        GPU: Use the GPU backend.
            Requires CUDA-compatible NVIDIA GPU. Provides significant
            acceleration for matrix operations.
        TPU: Use the TPU backend.
            Requires Google Cloud TPU access. Optimized for large-scale
            distributed training.
        TT: Use the Tenstorrent backend.
            For Tenstorrent accelerator hardware. Experimental support.

    Example:
        >>> from easydel.infra.etils import EasyDeLBackends
        >>> # Select backend based on available hardware
        >>> backend = EasyDeLBackends.GPU
        >>> print(backend.value)
        'gpu'
    """

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    TT = "tt"


# Type alias for valid gradient checkpoint options.
# Used for type hints to ensure only valid checkpoint strategies are used.
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

# Type alias for valid scheduler options.
# Includes "none" as a string option for command-line compatibility.
AVAILABLE_SCHEDULERS = tp.Literal["linear", "cosine", "none"]

# Type alias for valid optimizer options.
# Maps to the values in EasyDeLOptimizers enum.
AVAILABLE_OPTIMIZERS = tp.Literal["adafactor", "adamw", "mars", "muon", "rmsprop", "lion", "skew", "quad"]

# Type alias for mixture-of-experts implementation methods.
# fused_moe: Uses fused kernels for efficiency
# standard_moe: Standard implementation with separate operations
# dense_moe: Dense computation across all experts
AVAILABLE_MOE_METHODS = tp.Literal["fused_moe", "standard_moe", "dense_moe"]

# Type alias for valid attention mechanism implementations.
# Provides various optimized attention implementations for different use cases.
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
    "unified_attention",
    "paged_flash_attention",
]

# Default attention mechanism used when none is explicitly specified.
# "vanilla" provides the most compatible standard attention implementation.
DEFAULT_ATTENTION_MECHANISM = "vanilla"

# Type alias for sparse matrix format types.
# bcoo: Batched Coordinate format
# bcsr: Batched Compressed Sparse Row format
# coo: Coordinate format
# csr: Compressed Sparse Row format
AVAILABLE_SPARSE_MODULE_TYPES = tp.Literal["bcoo", "bcsr", "coo", "csr"]

# Type alias for gradient checkpoint target names.
# These names identify specific computation points in the model where
# checkpointing can be applied. Used with name-based checkpoint policies.
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
    """Define command-line flags using argparse based on provided keyword arguments.

    This function dynamically creates argparse arguments for each key-value pair
    in `kwargs`. It infers the argument type from the default value and provides
    special handling for tuple types using comma-separated strings.

    The function is particularly useful for creating configuration-driven scripts
    where default values are defined in code but can be overridden via command line.

    Args:
        _required_fields: A list of flag names that are mandatory. An error will
            be raised if these flags are not provided or are empty strings on
            the command line. Defaults to None (no required fields).
        **kwargs: Keyword arguments where keys are flag names (without `--`)
            and values are their default values. The type of each argument is
            inferred from the default value type.

    Returns:
        A tuple containing:
            - An `argparse.Namespace` object holding the parsed command-line
              arguments with their values.
            - A dictionary mapping the original flag names to their default
              values, useful for tracking which values were overridden.

    Raises:
        ValueError: If a required field (from `_required_fields`) is not provided
            or is an empty string on the command line.
        SystemExit: If invalid command-line arguments are provided (standard
            argparse behavior).

    Note:
        - For tuple default values, provide comma-separated integers on the
          command line (e.g., `--shape 1,2,3`).
        - Boolean arguments should use the flag (e.g., `--verbose True`).
        - The function calls `parser.parse_args()` which reads from `sys.argv`.

    Example:
        Basic usage with various argument types:

        >>> args, defaults = define_flags_with_default(
        ...     learning_rate=1e-3,
        ...     batch_size=32,
        ...     model_name="",
        ...     use_cache=True,
        ...     shape=(1, 2, 3),
        ...     _required_fields=["model_name"]
        ... )
        >>> print(defaults)
        {'learning_rate': 0.001, 'batch_size': 32, 'model_name': '', ...}

        Command line usage:

        .. code-block:: bash

            python script.py --learning_rate 0.0001 --batch_size 64 \\
                --model_name bert-base --shape 2,4,8
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
    """Custom argparse action to parse a comma-separated string into a tuple of integers.

    This action class is used by `define_flags_with_default` when a default value
    is a tuple. It enables command-line specification of tuple values using
    comma-separated integers (e.g., `--shape 1,2,3` becomes `(1, 2, 3)`).

    The action is automatically applied to any argument whose default value is
    a tuple when using `define_flags_with_default`.

    Attributes:
        Inherits all attributes from argparse.Action including:
            - dest: The name of the attribute to be added to the namespace.
            - option_strings: The command-line option strings.
            - default: The default value if the argument is not provided.

    Raises:
        argparse.ArgumentTypeError: If the provided value cannot be parsed as
            a comma-separated list of integers. This occurs when non-numeric
            values are provided or the format is incorrect.

    Example:
        Direct usage (typically not needed as define_flags_with_default handles this):

        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument(
        ...     "--shape",
        ...     type=str,
        ...     default="(1, 2, 3)",
        ...     action=StoreTupleAction
        ... )
        >>> # With command line: --shape 4,5,6
        >>> # Result: args.shape = (4, 5, 6)

        Error handling:

        >>> # With command line: --shape a,b,c
        >>> # Raises: ArgumentTypeError: Invalid value for --shape: a,b,c
        ...          (should be comma-separated integers)
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        """Parse and store the comma-separated string as a tuple of integers.

        This method is called by argparse when the associated argument is
        encountered on the command line.

        Args:
            parser: The ArgumentParser object that contains this action.
                Used for error handling and help generation.
            namespace: The Namespace object that will be returned by
                parse_args(). The parsed tuple will be stored as an
                attribute on this object.
            values: The string value provided on the command line.
                Expected to be comma-separated integers (e.g., "1,2,3").
            option_string: The option string that was used to invoke this
                action (e.g., "--shape"). Used in error messages.
                Defaults to None.

        Returns:
            None. The result is stored in the namespace object.

        Raises:
            argparse.ArgumentTypeError: If `values` cannot be split and
                converted to integers. The error message includes the
                option string and invalid value for debugging.
        """
        try:
            setattr(namespace, self.dest, tuple(int(v) for v in values.split(",")))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value for {option_string}: {values} (should be comma-separated integers)"
            ) from None
