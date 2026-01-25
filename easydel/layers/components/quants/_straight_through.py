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

"""Straight-through estimator (STE) functions for quantization-aware training.

This module provides straight-through estimator implementations for various
quantization formats. STEs enable gradient-based training of quantized neural
networks by allowing gradients to flow through non-differentiable quantization
operations.

The key insight of STEs is that during the forward pass, the quantized values
are used (providing accurate representation of inference behavior), while
during the backward pass, gradients are passed through unchanged to the
original floating-point weights (enabling optimization).

Available STE Functions:
    - straight_through: Unified interface for all quantization types
    - straight_through_mxfp8: MXFP8 (E5M2) floating-point quantization
    - straight_through_nvfp8: NVIDIA FP8 (E4M3) quantization
    - straight_through_mxfp4: MXFP4 (E2M1) floating-point quantization
    - straight_through_nf4: 4-bit NormalFloat quantization (from eformer)
    - straight_through_8bit: 8-bit integer quantization (from eformer)
    - straight_through_1bit: Binary/ternary quantization (from eformer)

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from easydel.layers.components.quants import straight_through, QuantizationType
    >>>
    >>> # Use STE in a training loop
    >>> @jax.jit
    ... def train_step(params, x, y):
    ...     def loss_fn(params):
    ...         # Quantize weights with STE
    ...         w_quant = straight_through(params['w'], dtype=QuantizationType.NF4)
    ...         pred = x @ w_quant
    ...         return jnp.mean((pred - y) ** 2)
    ...     loss, grads = jax.value_and_grad(loss_fn)(params)
    ...     return loss, grads  # grads flow to float32 params

Technical Details:
    The STE decorator from eformer wraps functions so that:
    - Forward: f(x) is computed normally (quantized output)
    - Backward: df/dx = identity (gradients pass through unchanged)

    This allows training with quantization effects visible in the forward pass
    while maintaining full-precision gradients for stable optimization.

See Also:
    - quantize: Full quantization (not STE, for inference)
    - QuantizationConfig: Configuration for quantization parameters
    - eformer.jaximus.ste: Low-level STE decorator
"""

from __future__ import annotations

import jax
from eformer.jaximus import ste
from eformer.ops import straight_through_1bit as straight_through_1bit
from eformer.ops import straight_through_8bit as straight_through_8bit
from eformer.ops import straight_through_nf4 as straight_through_nf4
from jax import numpy as jnp

from ._configs import QuantizationConfig, QuantizationType


@ste
def straight_through_mxfp8(weights: jax.Array) -> jax.Array:
    """Apply straight-through estimation with MXFP8 (E5M2) quantization.

    Quantizes weights to 8-bit floating point (E5M2 format) in the forward pass
    while passing gradients through unchanged in the backward pass. E5M2 format
    has 5 exponent bits and 2 mantissa bits, providing wider dynamic range but
    lower precision than E4M3.

    Args:
        weights: Input array to quantize. Typically model weights in float32
            or bfloat16.

    Returns:
        Array with the same shape and original dtype as input, but with values
        that have been quantized through E5M2 representation. The output dtype
        matches the input dtype (values are cast back after quantization).

    Note:
        The @ste decorator ensures gradients bypass this operation during
        backpropagation, enabling gradient-based optimization of the original
        full-precision weights.

    Example:
        >>> weights = jnp.array([0.1, 0.5, 1.0], dtype=jnp.float32)
        >>> quantized = straight_through_mxfp8(weights)
        >>> # quantized has same dtype as weights, but values are discretized
    """
    return weights.astype(jnp.float8_e5m2).astype(weights.dtype)


@ste
def straight_through_nvfp8(weights: jax.Array) -> jax.Array:
    """Apply straight-through estimation with NVIDIA FP8 (E4M3) quantization.

    Quantizes weights to 8-bit floating point (E4M3 format) in the forward pass
    while passing gradients through unchanged in the backward pass. E4M3 format
    has 4 exponent bits and 3 mantissa bits, providing higher precision but
    narrower dynamic range than E5M2. This format is optimized for NVIDIA
    hardware accelerators.

    Args:
        weights: Input array to quantize. Typically model weights in float32
            or bfloat16.

    Returns:
        Array with the same shape and original dtype as input, but with values
        that have been quantized through E4M3 representation.

    Note:
        E4M3 is particularly well-suited for inference on NVIDIA GPUs with
        FP8 tensor core support. The @ste decorator enables training with
        this quantization scheme.

    Example:
        >>> weights = jnp.array([0.1, 0.5, 1.0], dtype=jnp.float32)
        >>> quantized = straight_through_nvfp8(weights)
    """
    return weights.astype(jnp.float8_e4m3).astype(weights.dtype)


@ste
def straight_through_mxfp4(weights: jax.Array) -> jax.Array:
    """Apply straight-through estimation with MXFP4 (E2M1) quantization.

    Quantizes weights to 4-bit floating point (E2M1FN format) in the forward pass
    while passing gradients through unchanged in the backward pass. E2M1 format
    has 2 exponent bits and 1 mantissa bit, providing aggressive compression
    with significant precision loss.

    Args:
        weights: Input array to quantize. Typically model weights in float32
            or bfloat16.

    Returns:
        Array with the same shape and original dtype as input, but with values
        that have been quantized through E2M1 representation. This represents
        a 4x memory reduction compared to FP16/BF16.

    Warning:
        MXFP4 quantization is very aggressive and may cause significant
        accuracy degradation. It is best used with microscaling techniques
        or for specific layers that are tolerant to low precision.

    Example:
        >>> weights = jnp.array([0.1, 0.5, 1.0], dtype=jnp.float32)
        >>> quantized = straight_through_mxfp4(weights)
    """
    return weights.astype(jnp.float4_e2m1fn).astype(weights.dtype)


def straight_through(
    array: jax.Array,
    config: QuantizationConfig | None = None,
    dtype: QuantizationType | str | None = None,
    block_size: int = 64,
) -> jax.Array:
    """Unified straight-through estimator for all supported quantization types.

    This is the main entry point for quantization-aware training (QAT). It applies
    quantization in the forward pass to simulate inference behavior while allowing
    gradients to flow through unchanged in the backward pass to enable optimization
    of the original full-precision weights.

    The function dispatches to the appropriate type-specific STE implementation
    based on the specified quantization format.

    Args:
        array: Input array to quantize. Typically trainable model weights in
            float32 or bfloat16 format.
        config: QuantizationConfig object specifying dtype, block_size, and other
            settings. If provided, overrides the dtype and block_size arguments.
            Defaults to None.
        dtype: Quantization type to apply. Can be a QuantizationType enum value
            or its string representation (e.g., "nf4", "int8"). Ignored if config
            is provided. Defaults to QuantizationType.NF4 if neither config nor
            dtype is specified.
        block_size: Block size for block-wise quantization schemes (NF4).
            Larger blocks improve throughput but may reduce accuracy. Only used
            for NF4 quantization. Defaults to 64.

    Returns:
        Quantized array with the same shape as input. The array is materialized
        (not an implicit array) to ensure compatibility with standard JAX
        operations. Gradients with respect to this output will pass through
        unchanged to the input array during backpropagation.

    Raises:
        ValueError: If an unsupported quantization type is specified. The error
            message lists all supported types.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from easydel.layers.components.quants import straight_through, QuantizationType
        >>>
        >>> # Basic usage with NF4 quantization
        >>> weights = jnp.ones((128, 256))
        >>> quantized = straight_through(weights, dtype=QuantizationType.NF4)
        >>>
        >>> # Using configuration object
        >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
        >>> quantized = straight_through(weights, config=config)
        >>>
        >>> # In a training loop for QAT
        >>> @jax.jit
        ... def train_step(params, inputs, targets):
        ...     def loss_fn(params):
        ...         # Quantize weights with STE during forward pass
        ...         quant_w = straight_through(params['weight'], dtype=QuantizationType.NF4)
        ...         preds = inputs @ quant_w
        ...         return jnp.mean((preds - targets) ** 2)
        ...     loss, grads = jax.value_and_grad(loss_fn)(params)
        ...     # Gradients flow to original float32 params
        ...     return loss, grads

    Technical Details:
        Forward Pass:
            - Weights are quantized using the specified format
            - Quantization error affects the output (simulates inference)
            - Memory usage reflects quantized representation

        Backward Pass:
            - Gradients bypass quantization operation entirely
            - grad(output) is passed directly to grad(input)
            - Enables optimization of full-precision master weights

        This approach allows the model to learn to be robust to quantization
        effects while maintaining stable gradient-based optimization.

    See Also:
        - quantize: Full quantization without STE (for inference)
        - QuantizationConfig: Configuration dataclass for quantization settings
        - straight_through_nf4: NF4-specific STE implementation
        - straight_through_8bit: INT8-specific STE implementation
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

    if dtype == QuantizationType.NF4:
        return straight_through_nf4(array, block_size=block_size)

    elif dtype == QuantizationType.INT8:
        return straight_through_8bit(array)

    elif dtype in {QuantizationType.BINARY, QuantizationType.TERNARY}:
        return straight_through_1bit(array)

    elif dtype == QuantizationType.MXFP4:
        return straight_through_mxfp4(array)

    elif dtype == QuantizationType.MXFP8:
        return straight_through_mxfp8(array)

    elif dtype == QuantizationType.NVFP8:
        return straight_through_nvfp8(array)

    else:
        supported = ", ".join([t.value for t in QuantizationType])
        raise ValueError(f"Unsupported quantization type: {dtype}. Supported types: {supported}")
