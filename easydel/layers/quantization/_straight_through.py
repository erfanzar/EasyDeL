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
from ejkernel.quantization import dequantize as ej_dequantize  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.quantization import quantize as ej_quantize  # pyright: ignore[reportMissingTypeStubs]
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
    """Round-trip ``weights`` through ejkernel quantize/dequantize.

    Performs a fake-quantization pass: weights are packed via
    :func:`ejkernel.quantization.quantize` and immediately unpacked, returning
    a tensor of the same shape and dtype as the input but with values
    discretized to the chosen quantization grid.

    Args:
        weights: Floating-point weights to round-trip.
        mode: ejkernel quantization mode (``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``).
        group_size: Number of elements that share a single scale/bias.
        bits: Bit-width of the quantized representation.

    Returns:
        Array with the same shape and dtype as ``weights`` but with values
        rounded to the quantization grid.
    """
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
    """Straight-through wrapper around :func:`_ejkernel_dequantized`.

    Forward pass returns the round-tripped (quantized-and-dequantized)
    weights; backward pass returns the upstream gradient unchanged.

    Args:
        weights: Floating-point weights.
        mode: ejkernel quantization mode.
        group_size: Group size for the per-block scale/bias.
        bits: Bit-width of the quantized representation.

    Returns:
        Discretized weights with the input dtype preserved; gradients pass
        through to ``weights`` unchanged.
    """
    return _ejkernel_dequantized(weights, mode=mode, group_size=group_size, bits=bits)


@ste
def _straight_through_cast(weights: jax.Array, *, dtype: jnp.dtype) -> jax.Array:
    """Straight-through cast to a low-precision dtype.

    Used by :func:`straight_through` when the target quantization type has a
    JAX-native dtype (e.g. ``float4_e2m1fn`` for MXFP4). Forward emits the
    cast tensor; backward passes the gradient through to the original full
    precision weights.

    Args:
        weights: Floating-point weights to cast.
        dtype: JAX-native low-precision dtype.

    Returns:
        ``weights.astype(dtype)`` in the forward pass; identity gradient.
    """
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
    """STE-quantize ``weights`` to 4-bit NormalFloat (QLoRA's NF4).

    NF4 is the QLoRA quantization scheme: each contiguous block of
    ``block_size`` weights is divided by its block-wise max abs, mapped onto
    a fixed 16-entry codebook of points sampled from the standard-normal
    inverse CDF (so codes are denser near zero than uniform int4), and the
    16-level codes are stored alongside the per-block scale. The forward pass
    here round-trips through that pack/unpack so the model sees the
    discretization error at training time; the ``@ste`` decorator on the
    inner helper makes the backward pass an identity through to ``weights``.

    Args:
        weights: Floating-point weights to quantize. Any shape; ejkernel
            applies the grouping along the last axis.
        block_size: Number of contiguous weights that share a single fp32
            scale. Must be one of ``{16, 32, 64, 128, 256, 512, 1024}`` —
            64 is the QLoRA default and is what most published NF4 weights
            use. Smaller blocks improve quality at the cost of more scales.

    Returns:
        Tensor with the same shape and dtype as ``weights`` containing
        weights snapped to the NF4 grid. The straight-through wrapper
        guarantees ``∂out/∂weights = I`` for the autodiff path.
    """
    return _straight_through_ejkernel(weights, mode="nf4", group_size=int(block_size), bits=4)


def straight_through_8bit(
    weights: jax.Array,
    axis: int | None = None,
    *,
    group_size: int = 64,
) -> jax.Array:
    """STE-quantize ``weights`` to 8-bit affine (per-group scale + bias).

    Implements the standard symmetric/asymmetric int8 scheme: each contiguous
    group of ``group_size`` weights gets a fp32 ``(scale, bias)`` pair, the
    weights are mapped onto the 256-level int8 grid, and the forward pass
    here returns the dequantized round-trip. Backward pass is identity via
    the inner ``@ste``.

    Args:
        weights: Floating-point weights to quantize. Any shape.
        axis: Accepted for API parity with axis-wise STE variants but
            **ignored**. ejkernel groups along the last axis using
            ``group_size``; pass ``group_size=weights.shape[-1]`` to recover
            per-channel quantization on the last axis.
        group_size: Number of contiguous weights per ``(scale, bias)``.
            Defaults to 64. Same power-of-two restriction as NF4.

    Returns:
        Tensor with the same shape and dtype as ``weights`` snapped to the
        per-group int8 grid; gradients flow through to ``weights`` unchanged.
    """
    _ = axis
    return _straight_through_ejkernel(weights, mode="affine", group_size=int(group_size), bits=8)


@ste
def straight_through_1bit(weights: jax.Array, axis: int | None = None) -> jax.Array:
    """STE-quantize ``weights`` to 1-bit ``{-1, +1}`` via the sign function.

    Implements BinaryConnect-style binarization: the forward pass maps each
    weight to its sign (with the convention that exactly-zero weights become
    ``+1`` to avoid dead units), and the backward pass — courtesy of the
    ``@ste`` decorator on this function — passes the upstream gradient
    through to ``weights`` unchanged. This is the most aggressive of the
    quantization options here and is dispatched to from
    :func:`straight_through` for both ``BINARY`` and ``TERNARY``
    quantization types.

    Args:
        weights: Floating-point weights to binarize. Any shape.
        axis: Accepted for API parity with the other STE variants but
            **ignored** — sign is applied element-wise.

    Returns:
        Tensor with the same shape and dtype as ``weights`` whose values are
        all ``+1`` or ``-1``; gradients flow through unchanged.
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
