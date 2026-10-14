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

"""Per-encoding decoders: packed checkpoint tensors to a dense float weight.

Each function here undoes exactly one on-disk encoding. They are pure, take
and return arrays, and know nothing about configs, layers or registries — so
they are shared freely between adapters (compressed-tensors reuses the fp8 and
int decoders rather than owning copies) and are directly testable against a
reference.

**Orientation.** Every decoder returns EasyDeL's kernel layout,
``[in_features, out_features]``, optionally with a leading expert axis. Most
HuggingFace checkpoints store ``[out, in]``, so most decoders transpose; AWQ
already stores ``[in, out]`` and does not. Getting this wrong produces a
correctly-shaped, entirely wrong weight on square matrices, so each decoder
states its expected input layout explicitly.
"""

from __future__ import annotations

import typing as tp

from jax import numpy as jnp

from ._decode import (
    decode_e4m3_scale,
    decode_e8m0_scale,
    dequantize_with_scale,
    expand_scale,
    unpack_uint8_to_e2m1,
    unpack_uint32_to_uint4,
)

if tp.TYPE_CHECKING:
    from jax import Array


def _swap_last_two(array: Array) -> Array:
    """Transpose the trailing two axes, preserving any leading expert axis.

    Args:
        array: Array of rank >= 2.

    Returns:
        The array with its last two axes swapped.
    """
    return jnp.swapaxes(array, -1, -2)


def decode_scaled_elements(
    weight: Array,
    scale: Array,
    *,
    transpose: bool = True,
    dtype: jnp.dtype = jnp.float32,
    block_shape: tp.Sequence[int] | None = None,
) -> Array:
    """Decode any element-wise-scaled weight (fp8, int8, and their block forms).

    Covers per-tensor, per-channel and 2-D block scales uniformly, because
    :func:`~._decode.expand_scale` treats all three as a scale grid coarser
    than the weight. This is the decoder for ``fp8`` checkpoints
    (``float8_e4m3fn`` elements with ``weight_scale`` or ``weight_scale_inv``)
    and for compressed-tensors ``int8`` schemes alike — the element dtype
    differs, the arithmetic does not.

    Args:
        weight: Quantized elements, ``[out, in]`` in HF layout (or
            ``[experts, out, in]``).
        scale: Scales at any granularity.
        transpose: Whether the input is HF ``[out, in]`` and must be flipped
            to EasyDeL's ``[in, out]``.
        dtype: Output dtype.
        block_shape: The checkpoint's declared block sizes (e.g. DeepSeek's
            ``weight_block_size=[128, 128]``), aligned to the ``[out, in]``
            axes as stored. Required for correct scale expansion whenever a
            dimension is not a multiple of its block size.

    Returns:
        Dense weight in ``[in, out]`` layout.
    """
    # HF layout puts output channels on axis -2, so that is where a 1-D
    # per-channel `weight_scale` indexes — including for a square weight,
    # where length alone cannot tell the two axes apart.
    dense = dequantize_with_scale(weight, scale, dtype=dtype, block_shape=block_shape, channel_axis=-2)
    return _swap_last_two(dense) if transpose else dense


def decode_awq_int4(
    qweight: Array,
    scales: Array,
    qzeros: Array | None,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Decode an AWQ ``uint4`` weight.

    AWQ stores ``qweight`` as ``[in, out // 8]`` ``uint32`` — packed along the
    **output** axis, in the interleaved lane order ``(0, 2, 4, 6, 1, 3, 5, 7)``
    — with ``scales`` ``[in // group, out]`` and ``qzeros``
    ``[in // group, out // 8]`` packed the same way. Because the layout is
    already ``[in, out]``, no transpose is applied.

    Args:
        qweight: Packed ``uint32`` weight codes.
        scales: Per-group scales.
        qzeros: Packed ``uint32`` zero-points, or ``None`` for symmetric.
        dtype: Output dtype.

    Returns:
        Dense weight in ``[in, out]`` layout.
    """
    codes = unpack_uint32_to_uint4(qweight.astype(jnp.uint32), axis=-1, awq_order=True)
    zeros = None
    if qzeros is not None:
        zeros = unpack_uint32_to_uint4(qzeros.astype(jnp.uint32), axis=-1, awq_order=True).astype(jnp.float32)
    # AWQ decodes straight into [in, out], so output channels are axis -1 —
    # the opposite of the HF-layout codecs.
    return dequantize_with_scale(codes, scales, zero_point=zeros, dtype=dtype, channel_axis=-1)


def decode_gptq_int4(
    qweight: Array,
    scales: Array,
    qzeros: Array | None,
    *,
    g_idx: Array | None = None,
    zero_point_offset: int = 1,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Decode a GPTQ ``uint4`` weight.

    GPTQ differs from AWQ in two ways that silently corrupt weights if
    confused: it packs ``qweight`` along the **input** axis
    (``[in // 8, out]``) rather than the output axis, and it packs
    sequentially rather than in AWQ's interleaved lane order. ``qzeros``
    (``[in // group, out // 8]``) is packed along the output axis, and the
    stored zero-point is conventionally one less than the true value.

    Args:
        qweight: Packed ``uint32`` weight codes, ``[in // 8, out]``.
        scales: Per-group scales, ``[in // group, out]``.
        qzeros: Packed ``uint32`` zero-points, or ``None`` for symmetric.
        g_idx: Optional act-order group index, ``[in]``, mapping each input
            row to its scale group. Present when the checkpoint was produced
            with ``desc_act=True``, where groups are not contiguous and a
            gather is required instead of a uniform expansion.
        zero_point_offset: Added to the stored zero-points. GPTQ stores
            ``zero - 1``.
        dtype: Output dtype.

    Returns:
        Dense weight in ``[in, out]`` layout.
    """
    codes = unpack_uint32_to_uint4(qweight.astype(jnp.uint32), axis=0, awq_order=False)

    zeros = None
    if qzeros is not None:
        zeros = unpack_uint32_to_uint4(qzeros.astype(jnp.uint32), axis=-1, awq_order=False)
        zeros = zeros.astype(jnp.float32) + float(zero_point_offset)

    if g_idx is not None:
        # Act-order: each row picks its own group, so scales cannot be
        # expanded by repetition and must be gathered per row.
        row_index = g_idx.astype(jnp.int32)
        row_scales = jnp.take(scales.astype(jnp.float32), row_index, axis=0)
        values = codes.astype(jnp.float32)
        if zeros is not None:
            values = values - jnp.take(zeros, row_index, axis=0)
        return (values * row_scales).astype(dtype)

    # GPTQ unpacks along the input axis, leaving [in, out]: output channels
    # are axis -1.
    return dequantize_with_scale(codes, scales, zero_point=zeros, dtype=dtype, channel_axis=-1)


def decode_compressed_int4(
    weight_packed: Array,
    weight_scale: Array,
    *,
    transpose: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Decode a compressed-tensors ``pack-quantized`` int4 weight.

    compressed-tensors stores ``weight_packed`` as ``[out, in // 8]`` int32,
    packed sequentially (LSB-first, GPTQ lane order — not AWQ's interleave)
    along the **input** axis, i.e. the *last* axis of the HF ``[out, in]``
    layout. That is the transpose of GPTQ's ``[in // 8, out]`` packing, so
    reusing :func:`decode_gptq_int4` here silently produces a wrong-shaped,
    wrong-valued weight. Codes are stored unsigned with a ``+8`` offset
    (``pack_to_int32`` adds ``2**(bits-1)`` before packing); the symmetric
    scheme ships no zero-point tensor, so decoding subtracts the offset back.
    ``weight_scale`` is ``[out, in // group]`` (or per-channel ``[out, 1]``).

    Args:
        weight_packed: Packed ``int32``/``uint32`` codes, ``[out, in // 8]``.
        weight_scale: Per-group scales, ``[out, in // group]`` or ``[out, 1]``.
        transpose: Whether to flip the HF ``[out, in]`` result to EasyDeL's
            ``[in, out]``.
        dtype: Output dtype.

    Returns:
        Dense weight in ``[in, out]`` layout.
    """
    codes = unpack_uint32_to_uint4(weight_packed.astype(jnp.uint32), axis=-1, awq_order=False)
    values = codes.astype(jnp.float32) - 8.0
    dense = (values * expand_scale(weight_scale, codes.shape, channel_axis=-2)).astype(dtype)
    return _swap_last_two(dense) if transpose else dense


def decode_mxfp4(
    blocks: Array,
    scales: Array,
    *,
    transpose: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Decode an MXFP4 weight (E2M1 elements with E8M0 shared exponents).

    Both halves are exact: E2M1 codes bitcast directly to
    ``float4_e2m1fn``, and E8M0 is a raw exponent. Since EasyDeL's own
    ``mxfp4`` mode uses the same group size (32) and the same ``uint8`` E8M0
    scales, decoding and re-packing round-trips without loss — unlike
    requantizing to a wider block, which is a real accuracy cost.

    Args:
        blocks: ``uint8`` array holding two E2M1 codes per lane, packed along
            the last axis; ``[out, in // 2]`` in HF layout.
        scales: ``uint8`` E8M0 codes, ``[out, in // 32]``.
        transpose: Whether the input is HF ``[out, in]`` layout.
        dtype: Output dtype.

    Returns:
        Dense weight in ``[in, out]`` layout.
    """
    codes = unpack_uint8_to_e2m1(blocks, axis=-1).astype(jnp.float32)
    dense = codes * expand_scale(decode_e8m0_scale(scales), codes.shape, channel_axis=-2)
    dense = dense.astype(dtype)
    return _swap_last_two(dense) if transpose else dense


def decode_nvfp4(
    weight_packed: Array,
    weight_scale: Array,
    *,
    transpose: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Decode an NVFP4 weight's block level, leaving its global scale out.

    NVFP4 carries two scale levels: an E4M3 scale per group of 16 and one
    FP32 scale for the whole tensor. Only the block level is folded in here.
    The global level is a per-tensor scalar that commutes with the matmul, so
    it is carried on the canonical weight as ``output_scale`` and applied
    afterwards — exact, and it avoids trying to express a large global factor
    inside an E4M3 code.

    Args:
        weight_packed: ``uint8`` array holding two E2M1 codes per lane;
            ``[out, in // 2]`` in HF layout.
        weight_scale: E4M3 block scales, ``[out, in // 16]``, as ``uint8``
            codes or a float8 array.
        transpose: Whether the input is HF ``[out, in]`` layout.
        dtype: Output dtype.

    Returns:
        Dense weight in ``[in, out]`` layout, excluding the global scale.
    """
    codes = unpack_uint8_to_e2m1(weight_packed, axis=-1).astype(jnp.float32)
    dense = codes * expand_scale(decode_e4m3_scale(weight_scale), codes.shape, channel_axis=-2)
    dense = dense.astype(dtype)
    return _swap_last_two(dense) if transpose else dense
