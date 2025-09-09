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

"""Custom exception classes for EasyDeL framework.

This module defines specialized exception classes used throughout EasyDeL
to provide clear and specific error messages for different failure scenarios.

Exception Hierarchy:
    All exceptions inherit from Python's base Exception class and are used
    to signal specific types of errors that can occur during model training,
    inference, or configuration.

Example:
    >>> try:
    ...     # Some EasyDeL operation
    ...     model.train()
    ... except EasyDeLRuntimeError as e:
    ...     print(f"Runtime error occurred: {e}")
    ... except EasyDeLNotImplementedFeatureError as e:
    ...     print(f"Feature not yet implemented: {e}")
"""


class EasyDeLRuntimeError(Exception):
    """General runtime error in EasyDeL operations.

    Raised when an unexpected runtime condition occurs during model
    execution, training, or inference that prevents normal operation.

    This is the most general EasyDeL exception and should be used when
    a more specific exception type doesn't apply.

    Example:
        >>> if model_state is None:
        ...     raise EasyDeLRuntimeError("Model state must be initialized before training")
    """


class EasyDeLSyntaxRuntimeError(Exception):
    """Syntax or configuration error in EasyDeL code.

    Raised when there are syntax errors in configuration files, model
    definitions, or when invalid parameter combinations are detected
    at runtime.

    This differs from Python's SyntaxError as it relates to EasyDeL-specific
    syntax and configuration requirements.

    Example:
        >>> if config.num_layers < 1:
        ...     raise EasyDeLSyntaxRuntimeError("Number of layers must be at least 1")
    """


class EasyDeLTimerError(Exception):
    """Error related to timing or profiling operations.

    Raised when timer operations fail, such as when trying to stop
    a timer that was never started, or when timing measurements
    produce invalid results.

    Used primarily by EasyDeL's profiling and benchmarking utilities.

    Example:
        >>> if timer_id not in active_timers:
        ...     raise EasyDeLTimerError(f"Timer '{timer_id}' was not started")
    """


class EasyDeLBreakRequest(Exception):
    """Signal to break out of a training or generation loop.

    This exception is used as a control flow mechanism to cleanly
    exit from training loops, generation loops, or other iterative
    processes when certain conditions are met (e.g., NaN detected,
    user interrupt, convergence criteria).

    Not typically an error condition, but rather a signal for
    controlled termination.

    Example:
        >>> if jnp.isnan(loss):
        ...     raise EasyDeLBreakRequest("NaN detected in loss, stopping training")
    """


class EasyDeLBlockWiseFFNError(Exception):
    """Error in block-wise feed-forward network operations.

    Raised when block-wise FFN operations fail, typically due to
    incorrect input shapes, incompatible chunk sizes, or when the
    sequence length is not divisible by the specified chunk size.

    This error is specific to the memory-efficient block-wise
    computation used in models with very long sequences.

    Example:
        >>> if seq_len % chunk_size != 0:
        ...     raise EasyDeLBlockWiseFFNError(
        ...         f"Sequence length {seq_len} must be divisible by chunk size {chunk_size}"
        ...     )
    """


class EasyDeLProcessError(Exception):
    """Error in multi-process or distributed operations.

    Raised when errors occur in distributed training, multi-device
    operations, or inter-process communication. This includes failures
    in device mesh creation, collective operations, or process
    synchronization.

    Example:
        >>> if jax.process_count() < required_processes:
        ...     raise EasyDeLProcessError(
        ...         f"Need at least {required_processes} processes, got {jax.process_count()}"
        ...     )
    """


class EasyDeLComputeError(Exception):
    """Error during numerical computation or model forward pass.

    Raised when mathematical operations fail, tensors have invalid
    values, or when the model's forward pass encounters an error.
    This includes overflow, underflow, or dimension mismatches during
    computation.

    Example:
        >>> if jnp.isinf(logits).any():
        ...     raise EasyDeLComputeError("Infinite values detected in model logits")
    """


class EasyDeLNotImplementedFeatureError(Exception):
    """Requested feature is not yet implemented.

    Raised when attempting to use a feature that is planned but not
    yet implemented in the current version of EasyDeL. This helps
    distinguish between bugs and intentionally unimplemented features.

    Example:
        >>> def new_attention_mechanism():
        ...     raise EasyDeLNotImplementedFeatureError(
        ...         "Custom attention mechanism will be available in v2.1"
        ...     )
    """
