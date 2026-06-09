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

"""
Low-level Quantization Functions.

This module provides core quantization and dequantization functions for various
bit-widths including NF4 (4-bit NormalFloat), INT8, and 1-bit formats. It also
includes TPU-optimized Pallas kernels for efficient matrix multiplication with
quantized weights.

Key Functions:
    Quantization/Dequantization:
        - quantize_int8 / dequantize_int8: 8-bit integer quantization with scaling
        - quantize_and_pack_nf4 / dequantize_nf4: NF4 block-wise quantization
        - pack_weights_1bit / unpack_weights_1bit: 1-bit ternary weight packing

    TPU Kernels:
        - bmm_nf4: Pallas kernel for NF4 batch matrix multiplication
        - bmm_nf4_transpose: Transposed variant for backward passes
        - nf4_matmul: High-level wrapper for TPU-optimized matmul

    Configuration:
        - is_kernel_available(): Check if TPU kernels are available
        - nf4_use_kernel(): Context manager for kernel mode control

    Utilities:
        - get_nf4(): Get NF4 lookup table
        - nf4xf32_to_f32(): Polynomial approximation for NF4 dequantization
        - i8tou8, u4toi4, i4tou4: Bit conversion utilities

Environment Variables:
    USE_NF4_KERNEL_TPU: Set to "0", "false", or "off" to disable TPU kernels.
        Default is enabled ("1").

Example:
    >>> import jax.numpy as jnp
    >>> from eformer.ops.quantization.quantization_functions import (
    ...     quantize_int8, dequantize_int8,
    ...     quantize_and_pack_nf4, dequantize_nf4
    ... )
    >>>
    >>> # INT8 quantization
    >>> weights = jnp.ones((128, 256))
    >>> quants, scales = quantize_int8(weights, axis=-1)
    >>> reconstructed = dequantize_int8(quants, scales)
    >>>
    >>> # NF4 quantization
    >>> packed, absmax = quantize_and_pack_nf4(weights, block_size=64)
    >>> reconstructed = dequantize_nf4(packed, absmax, block_size=64)
"""

import functools
import os
from contextlib import contextmanager
from functools import partial
from math import ceil

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Global flag for kernel mode
# Automatically set to True on TPU, False elsewhere
_USE_KERNEL_ON_TPU = os.getenv("USE_NF4_KERNEL_TPU", "1")  # Lazy initialization
if _USE_KERNEL_ON_TPU.lower() in ["1", "true", "on"]:
    _USE_KERNEL_ON_TPU = None
else:
    _USE_KERNEL_ON_TPU = False


def _is_tpu():
    """Check if the current device is a TPU."""
    try:
        return jax.local_devices()[0].platform == "tpu"
    except Exception:
        return False


def _get_kernel_state():
    """
    Get the current kernel state with lazy initialization.

    On first call, automatically enables kernels on TPU, disables on other devices.
    This lazy behavior avoids triggering JAX initialization at import time.
    """
    global _USE_KERNEL_ON_TPU
    if _USE_KERNEL_ON_TPU is None:
        _USE_KERNEL_ON_TPU = _is_tpu()
    return _USE_KERNEL_ON_TPU


def is_kernel_available():
    """
    Check if NF4 kernels are available on the current device.

    Returns:
        bool: True if running on TPU (where kernels are supported), False otherwise
    """
    return _is_tpu()


@contextmanager
def nf4_use_kernel(value: bool):
    """
    Context manager to enable/disable NF4 kernel mode.

    Note: Kernels are only enabled on TPU devices. On other devices,
    this setting has no effect and the code will fall back to materialization.

    Args:
        value: Whether to enable kernel mode (only effective on TPU)

    Example:
        >>> with nf4_use_kernel(True):
        ...     result = input @ quantized_weight  # Uses kernel on TPU
    """
    global _USE_KERNEL_ON_TPU
    old = _get_kernel_state()  # Initialize if needed
    # Only enable kernel mode if on TPU
    _USE_KERNEL_ON_TPU = value and _is_tpu()
    yield
    _USE_KERNEL_ON_TPU = old


def get_nf4():
    """
    Get the NF4 (4-bit NormalFloat) lookup table.

    Creates the lookup table lazily to avoid triggering JAX initialization
    at import time. The NF4 format uses 16 values optimized for Gaussian
    distributions, providing better accuracy than uniform quantization for
    neural network weights.

    Returns:
        jax.Array: A 16-element float32 array containing the NF4 codebook
            values ranging from -1.0 to 1.0. The values are symmetric around
            zero and optimized for normally distributed data.

    Note:
        The codebook values are derived from the quantiles of a standard
        normal distribution, making NF4 particularly effective for neural
        network weights which tend to follow Gaussian distributions.
    """

    return jnp.asarray(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
    )


def nf4xf32_to_f32(x):
    """
    Fast polynomial approximation for NF4 dequantization.

    This is significantly faster than table lookups and provides
    accurate approximation of the NF4 codebook values.

    Args:
            x: Integer array (0-15) representing NF4 quantized values

    Returns:
            Float32 array with approximated NF4 values
    """
    x = x.astype(jnp.float32)
    return (
        x
        * (
            x * (x * (x * (1.82943132356953e-5 * x - 0.00068587779130373) + 0.0100420261313669) - 0.0722703570217226)
            + 0.346075459755188
        )
        - 0.994166218659335
    )


# Bit operation utilities
sr = jax.lax.shift_right_logical  # Logical right shift (zero-fill)
sl = jax.lax.shift_left  # Left shift
ba = jax.lax.bitwise_and  # Bitwise AND


def i8tou8(x):
    """
    Convert signed int8 to unsigned uint8.

    Handles the two's complement representation by adding 256 to negative values.

    Args:
        x (jax.Array): Input array with int8 values (-128 to 127).

    Returns:
        jax.Array: Output array with equivalent uint8 values (0 to 255).

    Example:
        >>> i8tou8(jnp.array([-1, 0, 127], dtype=jnp.int8))
        Array([255, 0, 127], dtype=int32)
    """
    return jnp.where(x < 0, 256 + x, x)


def u4toi4(x):
    """
    Convert unsigned 4-bit integer to signed 4-bit integer.

    Maps unsigned values 0-15 to signed range -8 to 7. Values 8-15 become
    negative values -8 to -1.

    Args:
        x (jax.Array): Input array with uint4 values (0 to 15).

    Returns:
        jax.Array: Output array with signed int4 values (-8 to 7).

    Example:
        >>> u4toi4(jnp.array([0, 7, 8, 15]))
        Array([0, 7, -8, -1], dtype=int32)
    """
    return jnp.where(x >= 8, x - 16, x)


def i4tou4(x):
    """
    Convert signed 4-bit integer to unsigned 4-bit integer.

    Maps signed values -8 to 7 to unsigned range 0-15. Negative values -8 to -1
    become 8 to 15.

    Args:
        x (jax.Array): Input array with signed int4 values (-8 to 7).

    Returns:
        jax.Array: Output array with uint4 values (0 to 15).

    Example:
        >>> i4tou4(jnp.array([-8, -1, 0, 7]))
        Array([8, 15, 0, 7], dtype=int32)
    """
    return jnp.where(x < 0, 16 + x, x)


def quantize_int8(x: jax.Array, axis: int | tuple = -1):
    """
    Quantize floating-point values to 8-bit integers with per-axis scaling.

    Computes a scale factor based on the maximum absolute value along the
    specified axis, then quantizes values to the int8 range [-127, 127].

    Args:
        x (jax.Array): Input array to quantize. Can be any floating-point dtype.
        axis (int | tuple): Axis or axes along which to compute the scale factor.
            Defaults to -1 (last axis). Can be a single int or tuple of ints.

    Returns:
        tuple[jax.Array, jax.Array]: A tuple containing:
            - quant (jax.Array): int8 array with quantized values in range [-127, 127].
            - scale (jax.Array): Float array of scale factors with shape broadcastable
              to the input (dimensions along `axis` are kept as size 1).

    Example:
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> quant, scale = quantize_int8(x, axis=-1)
        >>> # quant is int8, scale has shape (2, 1)
        >>> reconstructed = dequantize_int8(quant, scale)

    Note:
        A tiny epsilon is added to the scale to avoid division by zero for
        arrays with all zeros. The scale preserves the original dtype of
        the input array.
    """
    if not isinstance(axis, tuple):
        axis = (axis,)
    axis = tuple(z % x.ndim for z in axis)
    amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    scale = (amax / 127.0 + jnp.finfo(x.dtype).tiny).astype(x.dtype)
    quant = jnp.round(x / scale).astype(jnp.int8)
    return quant, scale


def dequantize_int8(quants, scales):
    """
    Dequantize 8-bit integers back to floating-point values.

    Multiplies the quantized int8 values by their corresponding scale factors
    to reconstruct approximate original values.

    Args:
        quants (jax.Array): int8 array containing quantized values in range [-127, 127].
        scales (jax.Array): Float array of scale factors, must be broadcastable
            with quants. Typically has shape with size-1 dimensions where
            quantization was performed.

    Returns:
        jax.Array: Dequantized array with the same shape as quants, dtype
            determined by the scales array.

    Example:
        >>> quants = jnp.array([[42, 85, 127], [32, 64, 96]], dtype=jnp.int8)
        >>> scales = jnp.array([[0.024], [0.047]])  # Shape (2, 1)
        >>> result = dequantize_int8(quants, scales)
        >>> # result has shape (2, 3) with float values

    Note:
        This is the inverse operation of quantize_int8. Due to rounding during
        quantization, the reconstructed values are approximate.
    """

    return quants * scales


@functools.partial(jax.jit, static_argnames=["block_size"])
def single_quantize_and_pack_nf4(blocks, block_size=64):
    """
    Quantize and pack an array to NF4 format in a single pass.

    This function performs block-wise NF4 quantization by:
    1. Reshaping the input into blocks along the last dimension
    2. Computing per-block absolute maximum values for scaling
    3. Normalizing values by absmax
    4. Finding nearest NF4 codebook values
    5. Packing two 4-bit values into each uint8 byte

    Args:
        blocks (jax.Array): Input array with shape (..., features) where
            features must be divisible by block_size.
        block_size (int): Number of elements per quantization block.
            Defaults to 64. Must be even for packing.

    Returns:
        tuple[jax.Array, jax.Array]: A tuple containing:
            - packed (jax.Array): uint8 array with packed 4-bit values.
              Shape (..., num_blocks, block_size // 2).
            - absmax (jax.Array): Absolute maximum per block for dequantization.
              Shape (..., num_blocks).

    Example:
        >>> x = jnp.ones((128, 256))  # 256 features, 4 blocks of 64
        >>> packed, absmax = single_quantize_and_pack_nf4(x, block_size=64)
        >>> packed.shape  # (128, 4, 32) - 4 blocks, 32 bytes each
        >>> absmax.shape  # (128, 4) - one scale per block

    Note:
        This is optimized for JAX JIT compilation. For the high-level API,
        use quantize_and_pack_nf4 instead.
    """
    orig_shape = blocks.shape

    # Reshape only last dimension: (..., features) -> (..., num_blocks, block_size)
    *batch_dims, features = orig_shape
    num_blocks = features // block_size
    blocks = blocks.reshape(*batch_dims, num_blocks, block_size)

    # Compute absmax per block along last dimension
    absmax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)  # (..., num_blocks, 1)
    normalized = blocks / (absmax + jnp.finfo(blocks.dtype).tiny)

    # Quantize using NF4 codebook
    errors = normalized[..., None] - get_nf4()  # (..., num_blocks, block_size, 16)
    quantized = jnp.argmin(jnp.abs(errors), axis=-1)  # (..., num_blocks, block_size)

    # Pack two 4-bit values into one 8-bit value
    quantized = quantized.reshape(*batch_dims, num_blocks, block_size // 2, 2)
    packed = (quantized[..., 0] << 4) | quantized[..., 1]  # (..., num_blocks, block_size // 2)

    return packed.astype(jnp.uint8), absmax.squeeze(-1)  # (..., num_blocks, block_size//2), (..., num_blocks)


@functools.partial(jax.jit, static_argnames=["block_size"])
def single_dequantize_nf4(packed_values, absmax, block_size):
    """
    Dequantize packed NF4 values back to floating-point.

    This function reverses the NF4 quantization by:
    1. Unpacking two 4-bit values from each uint8 byte
    2. Converting 4-bit indices to NF4 codebook values using polynomial approximation
    3. Scaling by per-block absmax values
    4. Flattening back to original feature dimension

    Args:
        packed_values (jax.Array): uint8 array with packed 4-bit values.
            Shape (..., num_blocks, block_size // 2).
        absmax (jax.Array): Absolute maximum per block used during quantization.
            Shape (..., num_blocks).
        block_size (int): Number of elements per quantization block.
            Must match the value used during quantization.

    Returns:
        jax.Array: Dequantized float32 array with shape (..., num_blocks * block_size),
            which equals the original feature dimension.

    Example:
        >>> # Given packed data from single_quantize_and_pack_nf4
        >>> reconstructed = single_dequantize_nf4(packed, absmax, block_size=64)
        >>> reconstructed.shape  # Original shape restored

    Note:
        Uses a polynomial approximation (nf4xf32_to_f32) instead of table lookup
        for faster dequantization on accelerators.
    """
    # Unpack 4-bit values from 8-bit packed format
    high = (packed_values >> 4) & 0xF  # (..., num_blocks, block_size // 2)
    low = packed_values & 0xF  # (..., num_blocks, block_size // 2)
    unpacked = jnp.stack([high, low], axis=-1)  # (..., num_blocks, block_size // 2, 2)

    # Get shape info
    *batch_dims, num_blocks, _ = packed_values.shape
    unpacked = unpacked.reshape(*batch_dims, num_blocks, block_size)  # (..., num_blocks, block_size)

    dequantized = nf4xf32_to_f32(unpacked)

    # Scale by absmax
    scaled = dequantized * absmax[..., None]  # (..., num_blocks, block_size)

    # Flatten last two dimensions back to original feature dimension
    scaled = scaled.reshape(*batch_dims, num_blocks * block_size)  # (..., features)
    return scaled


@functools.partial(jax.jit, static_argnames=["block_size"])
def quantize_and_pack_nf4(blocks, block_size=64):
    """
    Quantize and pack an array using NF4 (4-bit NormalFloat) quantization.

    High-level API for NF4 quantization. Quantizes the input array into 4-bit
    NormalFloat format with block-wise scaling and packs two values per byte.

    Args:
        blocks (jax.Array): Input array to quantize. The last dimension must be
            divisible by block_size. Shape: (..., features).
        block_size (int): Number of elements per quantization block.
            Defaults to 64. Larger blocks use less memory for scales but may
            have higher quantization error.

    Returns:
        tuple[jax.Array, jax.Array]: A tuple containing:
            - packed (jax.Array): uint8 array with packed 4-bit values.
              Shape: (..., num_blocks, block_size // 2).
            - absmax (jax.Array): Per-block scale factors for dequantization.
              Shape: (..., num_blocks).

    Example:
        >>> weights = jnp.ones((128, 256), dtype=jnp.float32)
        >>> packed, absmax = quantize_and_pack_nf4(weights, block_size=64)
        >>> # Reconstruct with dequantize_nf4
        >>> reconstructed = dequantize_nf4(packed, absmax, block_size=64)

    See Also:
        - dequantize_nf4: Reverse operation to reconstruct values.
        - single_quantize_and_pack_nf4: Internal implementation.
    """
    # Single function now handles all batch dimensions
    return single_quantize_and_pack_nf4(blocks, block_size)


@functools.partial(jax.jit, static_argnames=["block_size"])
def dequantize_nf4(packed_values, absmax, block_size):
    """
    Dequantize an array from NF4 (4-bit NormalFloat) format.

    High-level API for NF4 dequantization. Unpacks 4-bit values and scales
    them by per-block absmax values to reconstruct approximate original values.

    Args:
        packed_values (jax.Array): uint8 array with packed 4-bit values as
            produced by quantize_and_pack_nf4. Shape: (..., num_blocks, block_size // 2).
        absmax (jax.Array): Per-block scale factors from quantization.
            Shape: (..., num_blocks).
        block_size (int): Number of elements per quantization block.
            Must match the value used during quantization.

    Returns:
        jax.Array: Dequantized float32 array with shape (..., features),
            where features = num_blocks * block_size.

    Example:
        >>> # Given packed data from quantize_and_pack_nf4
        >>> reconstructed = dequantize_nf4(packed, absmax, block_size=64)
        >>> # reconstructed approximates original values

    See Also:
        - quantize_and_pack_nf4: Forward quantization operation.
        - single_dequantize_nf4: Internal implementation.

    Note:
        Due to quantization, the reconstructed values are approximate.
        NF4 provides better accuracy than uniform 4-bit quantization for
        normally distributed data (typical of neural network weights).
    """
    # Single function now handles all batch dimensions
    return single_dequantize_nf4(packed_values, absmax, block_size)


@jax.jit
def pack_weights_1bit(quantized_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Packs a JAX array of quantized weights into a compact format using 2 bits per value.

    Parameters:
    -----------
    quantized_weights : jnp.ndarray
        An array containing ternary quantized weights {-1, 0, 1}. The first dimension must be
        a multiple of 4.

    Returns:
    --------
    jnp.ndarray
        A packed jnp.uint8 array.
    """
    original_shape = quantized_weights.shape
    if original_shape[0] % 4 != 0:
        raise ValueError(f"The first dimension must be a multiple of {4}. Got shape {original_shape}.")

    unpacked = (quantized_weights + 1).astype(jnp.uint8)
    reshaped = unpacked.reshape((4, original_shape[0] // 4, *original_shape[1:]))
    shifter = jnp.arange(0, 2 * 4, 2, dtype=jnp.uint8)
    shifter = shifter.reshape((4,) + (1,) * (reshaped.ndim - 1))
    shifted_values = reshaped << shifter
    packed = jnp.sum(shifted_values, axis=0, dtype=jnp.uint8)

    return packed


@functools.partial(jax.jit, static_argnames="dtype")
def unpack_weights_1bit(packed: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Unpacks a JAX array of quantized weights, matching the logic of the PyTorch original.
    This function concatenates the unpacked bit groups.

    Parameters:
    -----------
    packed : jnp.ndarray
        A packed jnp.uint8 array.
    dtype : jnp.dtype
        The dtype of the returned array (e.g., jnp.int8). This is a static argument for JIT.

    Returns:
    --------
    jnp.ndarray
        An unpacked array with ternary values {-1, 0, 1}.
    """
    shifter = jnp.arange(0, 2 * 4, 2, dtype=jnp.uint8)
    shifter = shifter.reshape((4,) + (1,) * packed.ndim)
    unpacked_groups = (packed >> shifter) & 3
    original_row_dim = packed.shape[0] * 4
    unpacked_shape = (original_row_dim, *packed.shape[1:])
    unpacked = unpacked_groups.reshape(unpacked_shape)

    return unpacked.astype(dtype) - 1


# TPU Pallas Kernels for optimized matmul
# Set BLOCK_OVERRIDE to (block_x, block_y, block_k) to manually specify block sizes
BLOCK_OVERRIDE = None


def bmm_nf4(
    inputs_ref,
    quants_ref,
    scale_ref,
    outputs_ref,
    accum_ref,
    *,
    block_k,
):
    """
    Pallas kernel for NF4 matrix multiplication with on-the-fly dequantization.

    This kernel performs efficient matrix multiplication by dequantizing weights
    on-the-fly during computation, avoiding materialization of full-precision weights.

    Args:
            inputs_ref: Reference to input activation tensor
            quants_ref: Reference to quantized weight tensor (int4)
            scale_ref: Reference to scale factors
            outputs_ref: Reference to output tensor
            accum_ref: Reference to accumulator tensor
            block_k: Block size for K dimension (static)
    """

    @pl.when(pl.program_id(axis=2) == 0)
    def _():
        accum_ref[...] = jnp.zeros_like(accum_ref)

    quants = quants_ref[...]
    scale = scale_ref[...]

    if quants.dtype == jnp.int8:
        w1 = (quants.astype(jnp.float32) / 127.5) * scale.astype(jnp.float32)
    else:
        # Convert int4 to unsigned, then dequantize
        quants = i4tou4(quants.astype(jnp.int32))
        quants = nf4xf32_to_f32(quants)
        w1 = quants * scale

    inputs = inputs_ref[...]
    accum_ref[...] += jnp.dot(inputs, w1.reshape(block_k, -1), preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(axis=2) == (pl.num_programs(axis=2) - 1))
    def _():
        outputs_ref[...] = accum_ref[...].astype(outputs_ref.dtype)


def bmm_nf4_transpose(
    inputs_ref,
    quants_ref,
    scale_ref,
    outputs_ref,
    accum_ref,
    *,
    block_k,
):
    """
    Pallas kernel for transposed NF4 matrix multiplication.

    This kernel handles the transpose case where weights need to be accessed
    in a different order. Used for backward passes in training.

    Args:
            inputs_ref: Reference to input activation tensor
            quants_ref: Reference to quantized weight tensor (int4)
            scale_ref: Reference to scale factors
            outputs_ref: Reference to output tensor
            accum_ref: Reference to accumulator tensor
            block_k: Block size for K dimension (static)
    """
    accum_ref[...] = jnp.zeros_like(accum_ref)

    loop_iterations = max(1, inputs_ref.shape[-1] // block_k)

    def matmul_loop(i, _):
        inputs = pl.load(inputs_ref, (slice(None), pl.dslice(i * block_k, block_k)))
        quants = pl.load(quants_ref, (slice(None), slice(None), pl.dslice(i * block_k, block_k)))
        scale = pl.load(scale_ref, (slice(None), slice(None), pl.dslice(i * block_k, block_k)))

        quants = quants.astype(jnp.int32)
        quants = i8tou8(quants)

        w1 = nf4xf32_to_f32(sr(quants, 4)) * scale
        w1 = w1.reshape(-1, inputs.shape[-1])
        w2 = nf4xf32_to_f32(quants & 0b1111) * scale
        w2 = w2.reshape(-1, inputs.shape[-1])

        output1 = inputs @ w1.T
        output2 = inputs @ w2.T

        output = jnp.concatenate((output1, output2), -1)
        accum_ref[...] += output

    jax.lax.fori_loop(0, loop_iterations, matmul_loop, init_val=None)
    accum = accum_ref[...]
    outputs_ref[...] = accum.astype(outputs_ref.dtype)


@partial(jax.jit, static_argnames=("kernel", "backward", "blocks"))
def nf4_matmul(inputs, *tensors, kernel, backward=False, blocks=None):
    """
    Fast matrix multiplication using Pallas TPU kernels.

    This function provides optimized matrix multiplication with automatic
    block size selection and padding for optimal TPU performance.

    Args:
            inputs: Input activation tensor
            *tensors: Quantized weight tensors (quants, scales)
            kernel: Pallas kernel function to use
            backward: Whether this is a backward pass
            blocks: Optional manual block size specification (block_x, block_y, block_k)

    Returns:
            Result of matrix multiplication
    """
    weight_transpose = backward

    inputs = inputs.astype(jnp.bfloat16)

    if blocks is None:
        if not backward:
            if BLOCK_OVERRIDE is not None:
                block_x, block_y, block_k = BLOCK_OVERRIDE
            else:
                block_x, block_y, block_k = 2048, 512, 256
        else:
            block_x, block_y, block_k = 256, 256, 512
    else:
        block_x, block_y, block_k = blocks

    if not weight_transpose:
        y = tensors[0].shape[2]
        quant_group_size = tensors[0].shape[1]
    else:
        quant_group_size = tensors[0].shape[1]
        y = quant_group_size * tensors[0].shape[0]

    x = inputs.shape[0]
    k = inputs.shape[1]

    # Adjust block sizes to fit dimensions
    if x < block_x:
        block_x = max(16, int(2 ** np.floor(np.log2(x))))
    if y < block_y:
        block_y = max(16, int(2 ** np.floor(np.log2(y))))
    if k < block_k:
        block_k = max(128, int(2 ** np.floor(np.log2(k))))

    # Pad inputs and tensors to match block sizes
    x_pad = (block_x - x) % block_x
    k_pad = (block_k - k) % block_k

    if x_pad or k_pad:
        inputs = jnp.pad(inputs.reshape(x, k), ((0, x_pad), (0, k_pad)))

    y_pad = (block_y - y) % block_y
    if y_pad:
        if not weight_transpose:
            tensors = [jnp.pad(t, ((0, 0), (0, 0), (0, y_pad))) for t in tensors]
        else:
            tensors = [jnp.pad(t, ((0, y_pad // quant_group_size), (0, 0), (0, 0))) for t in tensors]
    if k_pad:
        if not weight_transpose:
            tensors = [jnp.pad(t, ((0, k_pad // quant_group_size), (0, 0), (0, 0))) for t in tensors]
        else:
            tensors = [jnp.pad(t, ((0, 0), (0, 0), (0, k_pad))) for t in tensors]

    if not weight_transpose:
        expected = tensors[0].shape[0] * quant_group_size
        if inputs.shape[1] != expected:
            raise ValueError(f"Input shape mismatch: got {inputs.shape[1]}, expected {expected}.")
    else:
        expected = tensors[0].shape[2]
        if inputs.shape[1] != expected:
            raise ValueError(f"Input shape mismatch: got {inputs.shape[1]}, expected {expected}.")

    def kernel_call(inputs, *tensors):
        inputs_dtype = inputs.dtype
        grid = ceil(inputs.shape[0] / block_x), ceil(y / block_y), ceil(k / block_k)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(index_map=lambda i, j, k: (i, k), block_shape=(block_x, block_k)),
            ]
            + [
                (
                    pl.BlockSpec(
                        index_map=lambda i, j, k: (k, 0, j),
                        block_shape=(block_k // quant_group_size, t.shape[1], block_y),
                    )
                    if not weight_transpose
                    else pl.BlockSpec(
                        index_map=lambda i, j, k: (j, 0, k),
                        block_shape=(block_y // quant_group_size, t.shape[1], block_k),
                    )
                )
                for t in tensors
            ],
            out_specs=pl.BlockSpec(index_map=lambda i, j, k: (i, j), block_shape=(block_x, block_y)),
            scratch_shapes=[pltpu.VMEM((block_x, block_y), jnp.float32)],
        )
        outputs = pl.pallas_call(
            partial(kernel, block_k=block_k),
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct((grid[0] * block_x, grid[1] * block_y), inputs_dtype),
            compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))),
            interpret=jax.local_devices()[0].platform == "cpu",
        )(inputs, *tensors)
        if backward:
            outputs = outputs.reshape(outputs.shape[0], -1, 2, block_y // 2).mT.reshape(outputs.shape)
        return outputs

    result = kernel_call(inputs, *tensors)
    if x_pad or y_pad:
        result = result[:x, :y]
    return result
