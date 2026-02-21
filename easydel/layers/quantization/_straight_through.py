# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
    - straight_through_mxfp8: MXFP8 microscaling floating-point quantization
    - straight_through_nvfp8: NVIDIA FP8 (E4M3) quantization
    - straight_through_mxfp4: MXFP4 (E2M1) microscaling quantization
    - straight_through_nf4: 4-bit NormalFloat quantization (via ejkernel)
    - straight_through_8bit: 8-bit affine quantization (via ejkernel)
    - straight_through_1bit: Binary/ternary quantization

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from easydel.layers.quantization import straight_through, QuantizationType
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
from ejkernel.quantization import dequantize as ej_dequantize
from ejkernel.quantization import quantize as ej_quantize
from jax import numpy as jnp

from ._configs import (
    QuantizationConfig,
    QuantizationType,
    resolve_ejkernel_quant_params,
    resolve_jax_native_dtype,
)


def _ejkernel_dequantized(
    weights: jax.Array,
    *,
    mode: str,
    group_size: int,
    bits: int,
) -> jax.Array:
    if mode == "affine":
        wq, scales, biases = ej_quantize(weights, group_size=group_size, bits=bits, mode=mode)
    else:
        wq, scales = ej_quantize(weights, group_size=group_size, bits=bits, mode=mode)
        biases = None
    dequantized = ej_dequantize(
        wq,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    return dequantized.astype(weights.dtype)


@ste
def _straight_through_ejkernel(
    weights: jax.Array,
    *,
    mode: str,
    group_size: int,
    bits: int,
) -> jax.Array:
    return _ejkernel_dequantized(weights, mode=mode, group_size=group_size, bits=bits)


@ste
def _straight_through_cast(weights: jax.Array, *, dtype: jnp.dtype) -> jax.Array:
    return weights.astype(dtype)


def straight_through_mxfp8(weights: jax.Array) -> jax.Array:
    """Apply straight-through estimation with MXFP8 microscaling quantization.

    Quantizes weights using ejkernel's microscaling FP8 format in the forward
    pass while passing gradients through unchanged in the backward pass. This
    mode uses E4M3 codes with a shared E8M0 exponent per group.

    Args:
        weights: Input array to quantize. Typically model weights in float32
            or bfloat16.

    Returns:
        Array with the same shape and original dtype as input, but with values
        that have been quantized through microscaling FP8 representation.

    Note:
        The @ste decorator ensures gradients bypass this operation during
        backpropagation, enabling gradient-based optimization of the original
        full-precision weights.

    Example:
        >>> weights = jnp.array([0.1, 0.5, 1.0], dtype=jnp.float32)
        >>> quantized = straight_through_mxfp8(weights)
        >>> # quantized has same dtype as weights, but values are discretized
    """
    return _straight_through_ejkernel(weights, mode="mxfp8", group_size=32, bits=8)


def straight_through_nvfp8(weights: jax.Array) -> jax.Array:
    """Apply straight-through estimation with NVIDIA FP8 (E4M3) quantization.

    Quantizes weights using ejkernel's NVIDIA FP8 (E4M3) mode in the forward
    pass while passing gradients through unchanged in the backward pass. This
    mode uses per-group E4M3 scales.

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
    return _straight_through_ejkernel(weights, mode="nvfp8", group_size=16, bits=8)


def straight_through_mxfp4(weights: jax.Array) -> jax.Array:
    """Apply straight-through estimation with MXFP4 (E2M1) quantization.

    Quantizes weights using ejkernel's microscaling FP4 format in the forward
    pass while passing gradients through unchanged in the backward pass. This
    mode uses E2M1 codes with a shared E8M0 exponent per group.

    Args:
        weights: Input array to quantize. Typically model weights in float32
            or bfloat16.

    Returns:
        Array with the same shape and original dtype as input, but with values
        that have been quantized through microscaling FP4 representation.

    Warning:
        MXFP4 quantization is very aggressive and may cause significant
        accuracy degradation. It is best used with microscaling techniques
        or for specific layers that are tolerant to low precision.

    Example:
        >>> weights = jnp.array([0.1, 0.5, 1.0], dtype=jnp.float32)
        >>> quantized = straight_through_mxfp4(weights)
    """
    return _straight_through_ejkernel(weights, mode="mxfp4", group_size=32, bits=4)


def straight_through_nf4(weights: jax.Array, block_size: int = 64) -> jax.Array:
    """Apply straight-through estimation with NF4 quantization via ejkernel."""
    return _straight_through_ejkernel(weights, mode="nf4", group_size=int(block_size), bits=4)


def straight_through_8bit(
    weights: jax.Array,
    axis: int | None = None,
    *,
    group_size: int = 64,
) -> jax.Array:
    """Apply straight-through estimation with 8-bit affine quantization.

    Note:
        The `axis` argument is accepted for API compatibility but is not used
        by ejkernel's group-wise quantization.
    """
    _ = axis
    return _straight_through_ejkernel(weights, mode="affine", group_size=int(group_size), bits=8)


@ste
def straight_through_1bit(weights: jax.Array, axis: int | None = None) -> jax.Array:
    """Apply straight-through estimation with binary quantization.

    Note:
        The `axis` argument is accepted for API compatibility but is not used.
    """
    _ = axis
    quantized = jnp.sign(weights)
    quantized = jnp.where(quantized == 0, 1, quantized)
    return quantized.astype(weights.dtype)


def straight_through(
    array: jax.Array,
    config: QuantizationConfig | None = None,
    dtype: QuantizationType | str | None = None,
    group_size: int | None = None,
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
        config: QuantizationConfig object specifying dtype, group_size, and other
            settings. If provided, overrides the dtype and group_size arguments.
            Defaults to None.
            When config.jax_native is True and the dtype has a native JAX
            representation (MXFP4/MXFP8/NVFP8), the forward pass uses
            `astype` and the backward pass remains straight-through.
        dtype: Quantization type to apply. Can be a QuantizationType enum value
            or its string representation (e.g., "nf4", "int8"). Ignored if config
            is provided. Defaults to QuantizationType.NF4 if neither config nor
            dtype is specified.
        group_size: Group size for group-wise quantization schemes (NF4, affine,
            MXFP, NVFP). Larger groups improve throughput but may reduce accuracy.
            Defaults depend on dtype when not provided.

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
        >>> from easydel.layers.quantization import straight_through, QuantizationType
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
        group_size = config.group_size
    elif dtype is None:
        dtype = QuantizationType.NF4

    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)

    if dtype in {QuantizationType.BINARY, QuantizationType.TERNARY}:
        return straight_through_1bit(array)

    if config is not None and config.jax_native:
        jax_dtype = resolve_jax_native_dtype(dtype)
        if jax_dtype is not None:
            return _straight_through_cast(array, dtype=jax_dtype)

    if dtype in {
        QuantizationType.AFFINE,
        QuantizationType.INT8,
        QuantizationType.NF4,
        QuantizationType.MXFP4,
        QuantizationType.MXFP8,
        QuantizationType.NVFP8,
    }:
        config_for_ejkernel = config if config is not None else QuantizationConfig(dtype=dtype, group_size=group_size)
        mode, group_size, bits, _ = resolve_ejkernel_quant_params(config_for_ejkernel)
        return _straight_through_ejkernel(array, mode=mode, group_size=group_size, bits=bits)

    supported = ", ".join([t.value for t in QuantizationType])
    raise ValueError(f"Unsupported quantization type: {dtype}. Supported types: {supported}")
