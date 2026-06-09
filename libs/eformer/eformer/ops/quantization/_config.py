# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Quantization configuration and unified interface."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .implicit_array_1bit import Array1B, straight_through_1bit
from .implicit_array_8bit import Array8B, straight_through_8bit
from .implicit_array_nf4 import ArrayNF4, straight_through_nf4


class QuantizationType(enum.StrEnum):
    """Supported quantization types."""

    NF4 = "nf4"  # 4-bit NormalFloat (optimized for Gaussian distributions)
    INT8 = "int8"  # 8-bit integer quantization
    TERNARY = "ternary"  # 1-bit ternary {-1, 0, 1}
    BINARY = "binary"  # 1-bit binary {-1, 1}


@dataclass
class QuantizationConfig:
    """
    Configuration for quantization behavior.

    This config controls how weights are quantized during training and inference.

    Attributes:
        dtype: The quantization type to use (NF4, INT4, INT8, etc.)
        block_size: Block size for block-wise quantization (default: 64)
                   Only applicable for NF4, Q4_0, and block-quantized formats.
        simulate: If True, uses straight-through estimation without actual bit packing.
                 Useful for QAT (quantization-aware training) simulation.
        use_kernel: If True and available, use optimized TPU/GPU kernels.
                   Auto-detected based on device type.

    Example:
        >>> # NF4 quantization with 64-element blocks
        >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
        >>>
        >>> # INT8 quantization
        >>> config = QuantizationConfig(dtype=QuantizationType.INT8, block_size=64)
        >>>
        >>> # Binary quantization
        >>> config = QuantizationConfig(dtype=QuantizationType.BINARY)
        >>>
        >>> # Simulation mode (no actual bit packing)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     simulate=True  # QAT mode
        ... )
    """

    dtype: QuantizationType | str = QuantizationType.NF4
    block_size: int = 64
    simulate: bool = False
    use_kernel: bool = True

    def __post_init__(self):
        """Post-initialization to convert string dtype to QuantizationType enum."""
        if isinstance(self.dtype, str):
            self.dtype = QuantizationType(self.dtype)


def quantize(
    array: jax.Array,
    config: QuantizationConfig | None = None,
    dtype: QuantizationType | str | None = None,
    block_size: int = 64,
    simulate: bool = False,
) -> Array1B | Array8B | ArrayNF4 | jax.Array:
    """
    Quantize an array using the specified configuration.

    This is the unified quantization interface that dispatches to the appropriate
    quantization implementation based on the dtype.

    Args:
        array: Input array to quantize (typically float32/bfloat16)
        config: QuantizationConfig object (if provided, overrides other args)
        dtype: Quantization type (NF4, INT8, BINARY, TERNARY)
        block_size: Block size for blockwise quantization
        simulate: If True, use simulation mode (STE without bit packing)

    Returns:
        Quantized array as ImplicitArray (or regular array if simulate=True)

    Example:
        >>> # Using config
        >>> config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
        >>> quantized = quantize(weights, config=config)
        >>>
        >>> # Direct parameters
        >>> quantized = quantize(weights, dtype=QuantizationType.NF4, block_size=64)
        >>>
        >>> # Simulation mode (for QAT)
        >>> quantized = quantize(weights, dtype=QuantizationType.NF4, simulate=True)

    See Also:
        - straight_through: Unified STE wrapper for all quantization types
        - QuantizationConfig: Configuration dataclass
        - QuantizationType: Enum of supported types
    """
    # Import here to avoid circular dependencies

    # Resolve config
    if config is not None:
        dtype = config.dtype
        block_size = config.block_size
        simulate = config.simulate
    elif dtype is None:
        dtype = QuantizationType.NF4

    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)

    # Dispatch to appropriate quantization
    if dtype == QuantizationType.NF4:
        quantized = ArrayNF4.quantize(array, block_size=block_size)
        return quantized.materialize() if simulate else quantized

    elif dtype == QuantizationType.INT8:
        quantized = Array8B.quantize(array)
        return quantized.materialize() if simulate else quantized

    elif dtype == QuantizationType.BINARY:
        # Binary: quantize to {-1, 1}
        binary = jnp.sign(array)
        binary = jnp.where(binary == 0, 1, binary)  # Handle zeros
        quantized = Array1B.quantize(binary.astype(jnp.int8))
        return quantized.materialize() if simulate else quantized

    elif dtype == QuantizationType.TERNARY:
        # Ternary: quantize to {-1, 0, 1}
        threshold = 0.5 * jnp.std(array)
        ternary = jnp.where(array > threshold, 1, jnp.where(array < -threshold, -1, 0))
        quantized = Array1B.quantize(ternary.astype(jnp.int8))
        return quantized.materialize() if simulate else quantized

    else:
        supported = ", ".join([t.value for t in QuantizationType])
        raise ValueError(f"Unsupported quantization type: {dtype}. Supported types: {supported}")


def straight_through(
    array: jax.Array,
    config: QuantizationConfig | None = None,
    dtype: QuantizationType | str | None = None,
    block_size: int = 64,
) -> jax.Array:
    """
    Unified straight-through estimator for all quantization types.

    This function quantizes in the forward pass but passes gradients straight through
    to the original float32 weights in the backward pass. Use this for training with
    quantization awareness.

    Args:
        array: Input array to quantize (typically trainable weights)
        config: QuantizationConfig object (if provided, overrides other args)
        dtype: Quantization type (NF4, INT8, BINARY, TERNARY)
        block_size: Block size for blockwise quantization

    Returns:
        Materialized quantized array with straight-through gradients

    Example:
        >>> # In training loop
        >>> @jax.jit
        ... def train_step(params, inputs, targets):
        ...     def loss_fn(params):
        ...         # Quantize weights with STE
        ...         quant_w = straight_through(params['weight'], dtype=QuantizationType.NF4)
        ...         preds = inputs @ quant_w
        ...         return jnp.mean((preds - targets) ** 2)
        ...     loss, grads = jax.value_and_grad(loss_fn)(params)
        ...     # grads flow to float32 params, not quantized weights
        ...     return loss, grads

    Technical Details:
        - Forward: Uses quantized representation (memory efficient)
        - Backward: Gradients bypass quantization (grad_input = grad_output)
        - Always materializes to ensure compatibility with standard ops
        - Underlying float32 params are updated during optimization

    See Also:
        - quantize: Unified quantization interface
        - ste: Low-level STE decorator in jaximus
    """
    # Import type-specific STE functions

    # Resolve config
    if config is not None:
        dtype = config.dtype
        block_size = config.block_size
    elif dtype is None:
        dtype = QuantizationType.NF4

    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)

    # Dispatch to appropriate STE
    if dtype == QuantizationType.NF4:
        return straight_through_nf4(array, block_size=block_size)

    elif dtype == QuantizationType.INT8:
        return straight_through_8bit(array)

    elif dtype in {QuantizationType.BINARY, QuantizationType.TERNARY}:
        return straight_through_1bit(array)

    else:
        supported = ", ".join([t.value for t in QuantizationType])
        raise ValueError(f"Unsupported quantization type: {dtype}. Supported types: {supported}")
