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

"""Decode primitives shared by every checkpoint-quantization adapter.

Adapters do exactly one format-specific thing: turn the checkpoint's packed
tensors back into a dense float array. Everything after that — grouping,
packing, scale computation — is handled by ejkernel's
``prepack_quantized_weights``, which already serves the self-quantization
path.

Only primitives that ejkernel does not already expose live here. Bit
unpacking uses JAX's native sub-byte dtypes (``float4_e2m1fn``,
``float8_e8m0fnu``, ``uint4``) through ``bitcast_convert_type`` rather than
lookup tables, so no decode table is reimplemented.
"""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp

if tp.TYPE_CHECKING:
    from jax import Array

#: AWQ packs eight uint4 values into a uint32 in this order:
#: ``(0, 2, 4, 6, 1, 3, 5, 7)``. Indexing the unpacked lanes with the
#: inverse permutation restores ascending order.
AWQ_REVERSE_ORDER: tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)

#: Group sizes the ejkernel affine mode accepts.
SUPPORTED_AFFINE_GROUP_SIZES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)


def unpack_uint32_to_uint4(packed: Array, *, axis: int = -1, awq_order: bool = False) -> Array:
    """Expand uint32 lanes into their eight packed uint4 values.

    Args:
        packed: ``uint32`` array whose lanes each hold eight 4-bit values.
        axis: Axis along which the values were packed. The expanded values
            replace this axis, multiplying its length by eight.
        awq_order: When ``True``, undo AWQ's interleaved lane order. GPTQ and
            compressed-tensors pack sequentially and must leave this ``False``
            — reading one with the other's order silently scrambles weights.

    Returns:
        A ``uint4`` array with ``axis`` expanded eightfold.

    Raises:
        ValueError: If ``packed`` is not ``uint32``.
    """
    if packed.dtype != jnp.uint32:
        raise ValueError(f"Expected uint32 packed codes, got {packed.dtype}.")

    axis = axis % packed.ndim
    # bitcast appends a lane axis of width 8 (32 bits / 4 bits).
    lanes = jax.lax.bitcast_convert_type(packed, jnp.uint4)
    if awq_order:
        lanes = lanes[..., AWQ_REVERSE_ORDER]

    # Move the new trailing lane axis next to the packed axis, then flatten.
    lanes = jnp.moveaxis(lanes, -1, axis + 1)
    shape = list(packed.shape)
    shape[axis] = packed.shape[axis] * 8
    return lanes.reshape(shape)


def unpack_uint8_to_e2m1(packed: Array, *, axis: int = -1) -> Array:
    """Expand uint8 lanes into their two packed FP4 (E2M1) values.

    Args:
        packed: ``uint8`` array whose lanes each hold two E2M1 values.
        axis: Axis along which the values were packed; its length doubles.

    Returns:
        A ``float4_e2m1fn`` array with ``axis`` doubled.

    Raises:
        ValueError: If ``packed`` is not ``uint8``.
    """
    if packed.dtype != jnp.uint8:
        raise ValueError(f"Expected uint8 packed E2M1 codes, got {packed.dtype}.")

    axis = axis % packed.ndim
    lanes = jax.lax.bitcast_convert_type(packed, jnp.float4_e2m1fn)
    lanes = jnp.moveaxis(lanes, -1, axis + 1)
    shape = list(packed.shape)
    shape[axis] = packed.shape[axis] * 2
    return lanes.reshape(shape)


def decode_e8m0_scale(codes: Array, *, dtype: jnp.dtype = jnp.float32) -> Array:
    """Decode MX shared-exponent (E8M0) scale codes into floats.

    E8M0 stores a raw exponent with no mantissa, so decoding is exact.

    Args:
        codes: ``uint8`` E8M0 codes, or an already-decoded float array (passed
            through unchanged so callers need not branch).
        dtype: Output floating dtype.

    Returns:
        The decoded scales.
    """
    if codes.dtype != jnp.uint8:
        return codes.astype(dtype)
    return jax.lax.bitcast_convert_type(codes, jnp.float8_e8m0fnu).astype(dtype)


def decode_e4m3_scale(codes: Array, *, dtype: jnp.dtype = jnp.float32) -> Array:
    """Decode NVFP4 per-block (E4M3) scale codes into floats.

    Args:
        codes: ``uint8`` E4M3 codes, an ``float8_e4m3fn`` array, or an
            already-decoded float array.
        dtype: Output floating dtype.

    Returns:
        The decoded scales.
    """
    if codes.dtype == jnp.uint8:
        return jax.lax.bitcast_convert_type(codes, jnp.float8_e4m3fn).astype(dtype)
    return codes.astype(dtype)


def _resolve_channel_axis(
    length: int,
    shape: tuple[int, ...],
    channel_axis: int | None,
) -> int:
    """Decide which weight axis a 1-D per-channel scale indexes.

    Length matching alone cannot answer this: the codecs hand
    :func:`expand_scale` two different orientations — fp8/MXFP4/NVFP4 and
    compressed-tensors decode in HF ``[out, in]``, while AWQ and GPTQ decode
    in ``[in, out]`` — so "the output axis" is axis 0 for some callers and
    axis 1 for others. Whenever the weight is square both axes match the
    scale's length and any fixed tie-break silently transposes the scales for
    half of the callers, which loads cleanly and produces garbage logits.

    Args:
        length: Number of scale entries.
        shape: Target weight shape, rank 2.
        channel_axis: Axis the caller knows its per-channel scales index, or
            ``None`` to infer it from ``length``.

    Returns:
        The resolved axis index.

    Raises:
        ValueError: If ``channel_axis`` disagrees with ``length``, if no axis
            matches, or if the weight is square and no axis was declared.
    """
    if channel_axis is not None:
        axis = channel_axis % len(shape)
        if shape[axis] != length:
            raise ValueError(
                f"channel_axis={channel_axis} selects a weight axis of {shape[axis]}, but the scale has "
                f"{length} entries (weight shape {shape})."
            )
        return axis

    matches = [axis for axis, dim in enumerate(shape) if dim == length]
    if not matches:
        raise ValueError(f"1-D scale of length {length} matches neither axis of weight shape {shape}.")
    if len(matches) > 1:
        raise ValueError(
            f"1-D scale of length {length} is ambiguous for square weight shape {shape}: it could index "
            f"either axis. Pass channel_axis to say which one the checkpoint's layout uses."
        )
    return matches[0]


def expand_scale(
    scale: Array,
    shape: tp.Sequence[int],
    *,
    block_shape: tp.Sequence[int] | None = None,
    channel_axis: int | None = None,
) -> Array:
    """Broadcast per-tensor, per-channel, per-group or per-block scales to full shape.

    One primitive covers every scale granularity any supported format uses:
    a scalar (per-tensor), a vector (per-channel), a 2-D grid whose axes are
    coarser than the weight's (per-group along one axis, per-block along
    both). Each axis is repeated by its own factor and then trimmed, so
    partial trailing blocks — legal in block-scaled fp8 when a dimension is
    not a multiple of the block size — are handled without special-casing.

    When the checkpoint declares a block size (``weight_block_size`` /
    ``block_structure``), it MUST be used: for a non-divisible axis the
    repetition factor is the block size itself, not ``ceil(dim / entries)``.
    A 576-row weight with 128-row blocks carries 5 scale rows; repeating each
    by ``ceil(576 / 5) = 116`` puts every boundary in the wrong place and
    silently mis-scales most of the tensor. Without a declared block shape
    the ceil-derived factor is kept — exact whenever the axis divides evenly.

    Args:
        scale: Scale array with rank 0, or rank equal to ``len(shape)``.
        shape: Target weight shape.
        block_shape: The checkpoint's per-axis block sizes, aligned to the
            trailing axes of ``shape`` (a leading expert axis carries no
            block entry). Entries that do not agree with the scale grid
            (``scale_dim != ceil(dim / block)``) fall back to the derived
            factor rather than under-covering the axis.
        channel_axis: Axis a 1-D per-channel scale indexes. Codecs pass the
            axis their own layout puts output channels on; ``None`` infers it
            from the scale's length and raises when the weight is square,
            because guessing there transposes the scales silently.

    Returns:
        A float32 array of exactly ``shape``.

    Raises:
        ValueError: If ``scale`` has a rank that is neither 0 nor
            ``len(shape)``, an axis coarser than the target, or a 1-D length
            that cannot be attributed to one axis.
    """
    scale = scale.astype(jnp.float32)
    shape = tuple(int(dim) for dim in shape)

    if scale.ndim == 0:
        return jnp.broadcast_to(scale, shape)

    if scale.ndim == 1 and len(shape) == 2:
        # A per-channel vector is a grid with one of its axes degenerate.
        entries = scale.shape[0]
        axis = _resolve_channel_axis(entries, shape, channel_axis)
        scale = scale.reshape((entries, 1) if axis == 0 else (1, entries))

    if scale.ndim != len(shape):
        raise ValueError(f"Scale rank {scale.ndim} cannot be expanded to weight shape {shape}.")

    blocks: tuple[int | None, ...] = (None,) * len(shape)
    if block_shape:
        declared = tuple(int(dim) for dim in block_shape)
        if len(declared) <= len(shape):
            # Right-align: a leading expert axis has no block entry.
            blocks = (None,) * (len(shape) - len(declared)) + declared

    for axis, (scale_dim, target_dim) in enumerate(zip(scale.shape, shape, strict=True)):
        if scale_dim == target_dim:
            continue
        if scale_dim > target_dim:
            raise ValueError(
                f"Scale axis {axis} has {scale_dim} entries for a weight axis of {target_dim}; scales must be "
                f"coarser than or equal to the weight."
            )
        block = blocks[axis]
        if block is not None and block > 0 and scale_dim == -(-target_dim // block):
            # The checkpoint's own block size places partial-trailing-block
            # boundaries correctly; the ceil-derived factor does not.
            repeats = block
        else:
            repeats = -(-target_dim // scale_dim)  # ceil, exact for divisible axes
        scale = jnp.repeat(scale, repeats, axis=axis)

    return jax.lax.slice(scale, [0] * len(shape), list(shape))


def dequantize_with_scale(
    codes: Array,
    scale: Array,
    *,
    zero_point: Array | None = None,
    dtype: jnp.dtype = jnp.float32,
    block_shape: tp.Sequence[int] | None = None,
    channel_axis: int | None = None,
) -> Array:
    """Reconstruct dense values from quantized codes and their scales.

    Args:
        codes: Quantized values in any dtype convertible to float.
        scale: Scales at any supported granularity; broadcast via
            :func:`expand_scale`.
        zero_point: Optional zero-points, subtracted before scaling and
            broadcast the same way. Asymmetric schemes (AWQ/GPTQ) supply it.
        dtype: Output dtype.
        block_shape: The checkpoint's declared per-axis block sizes, forwarded
            to :func:`expand_scale` so non-divisible axes expand at the true
            block boundaries.
        channel_axis: Axis a 1-D per-channel scale indexes in ``codes``'s own
            layout, forwarded to :func:`expand_scale`.

    Returns:
        The dequantized array, shaped like ``codes``.
    """
    values = codes.astype(jnp.float32)
    if zero_point is not None:
        values = values - expand_scale(
            zero_point,
            codes.shape,
            block_shape=block_shape,
            channel_axis=channel_axis,
        )
    expanded = expand_scale(scale, codes.shape, block_shape=block_shape, channel_axis=channel_axis)
    return (values * expanded).astype(dtype)


def largest_supported_group_size(contracting_dim: int) -> int:
    """Pick the largest affine group size that evenly divides an axis.

    Per-channel schemes have one scale for a whole column, so any grouping
    reproduces them exactly once the scale is broadcast — the group size is
    purely a packing choice. Preferring the largest legal divisor minimises
    the number of stored scale entries.

    Args:
        contracting_dim: Length of the axis groups run along.

    Returns:
        A supported group size dividing ``contracting_dim``, falling back to
        the smallest supported size when nothing divides evenly.
    """
    for group_size in sorted(SUPPORTED_AFFINE_GROUP_SIZES, reverse=True):
        if contracting_dim % group_size == 0:
            return group_size
    return min(SUPPORTED_AFFINE_GROUP_SIZES)
