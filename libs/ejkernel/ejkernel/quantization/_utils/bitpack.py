# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Bit-packing utilities for quantized weight storage.

This module provides functions to pack and unpack quantized values into
uint32 words in an MLX-compatible format. Values are stored LSB-first
(least significant bit first) within each 32-bit word.

The packing format stores a contiguous bitstream in uint32 words, using
``ceil(n_values * bits / 32)`` storage words and handling values that span word
boundaries for non-power-of-2 bit widths.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp


def _pack_bits_generic(values: jax.Array, bits: int) -> jax.Array:
    """Generic bit-packer supporting arbitrary bit-widths."""
    bits = int(bits)
    values = values.astype(jnp.uint32)
    n = values.shape[-1]
    bit_offsets = jnp.arange(n, dtype=jnp.uint32) * jnp.uint32(bits)
    word_idx = (bit_offsets // 32).astype(jnp.int32)
    shift = bit_offsets % 32
    split = shift + bits > 32
    low_bits = jnp.where(split, 32 - shift, bits)

    low_mask = (jnp.uint32(1) << low_bits) - 1
    low = jnp.left_shift(values & low_mask, shift)
    high = jnp.where(split, values >> low_bits, jnp.uint32(0))

    words = int((n * bits + 31) // 32)
    out = jnp.zeros((*values.shape[:-1], words), dtype=jnp.uint32)
    out = out.at[..., word_idx].add(low)

    high_idx = jnp.where(split, word_idx + 1, word_idx)
    out = out.at[..., high_idx].add(high)
    return out


def _unpack_bits_generic(packed: jax.Array, n: int, bits: int) -> jax.Array:
    """Generic bit-unpacker supporting arbitrary bit-widths."""
    bits = int(bits)
    bit_offsets = jnp.arange(n, dtype=jnp.uint32) * jnp.uint32(bits)
    word_idx = (bit_offsets // 32).astype(jnp.int32)
    shift = bit_offsets % 32
    split = shift + bits > 32
    low_bits = jnp.where(split, 32 - shift, bits)
    high_bits = bits - low_bits

    low_mask = (jnp.uint32(1) << low_bits) - 1
    low_word = jnp.take(packed, word_idx, axis=-1)
    low = (low_word >> shift) & low_mask

    max_idx = packed.shape[-1] - 1
    high_idx = jnp.minimum(word_idx + 1, max_idx)
    high_mask = (jnp.uint32(1) << high_bits) - 1
    high_word = jnp.take(packed, high_idx, axis=-1)
    high = jnp.where(split, high_word & high_mask, jnp.uint32(0))

    return low | (high << low_bits)


def _pack_bits_fast_grouped(
    values: jax.Array,
    *,
    bits: int,
    values_per_word: int,
    mask: int,
    strict_shape_alignment: bool,
) -> jax.Array:
    """Fast grouped packer for bit-widths that tile evenly into uint32."""
    values = values.astype(jnp.uint32)
    n = values.shape[-1]
    rem = n % values_per_word
    if strict_shape_alignment and rem != 0:
        raise ValueError(f"u{bits} fast path requires the last dimension to be a multiple of {values_per_word}.")

    if rem != 0:
        pad = values_per_word - rem
        pad_spec = [(0, 0)] * (values.ndim - 1) + [(0, pad)]
        values = jnp.pad(values, pad_spec)

    words = values.reshape(*values.shape[:-1], -1, values_per_word) & jnp.uint32(mask)
    shifts = jnp.arange(values_per_word, dtype=jnp.uint32) * jnp.uint32(bits)
    packed = jnp.left_shift(words, shifts)
    return jnp.sum(packed, axis=-1, dtype=jnp.uint32)


def _unpack_bits_fast_grouped(
    packed: jax.Array,
    n: int,
    *,
    bits: int,
    values_per_word: int,
    mask: int,
) -> jax.Array:
    """Fast grouped unpacker for bit-widths that tile evenly into uint32."""
    packed = packed.astype(jnp.uint32)
    shifts = jnp.arange(values_per_word, dtype=jnp.uint32) * jnp.uint32(bits)
    vals = (packed[..., None] >> shifts) & jnp.uint32(mask)
    vals = vals.reshape(*packed.shape[:-1], -1)
    return vals[..., :n]


def _pack_bits_u1_fast(values: jax.Array, *, strict_shape_alignment: bool) -> jax.Array:
    """Fast path for packing uint1-like codes into uint32."""
    return _pack_bits_fast_grouped(
        values,
        bits=1,
        values_per_word=32,
        mask=0x1,
        strict_shape_alignment=strict_shape_alignment,
    )


def _pack_bits_u2_fast(values: jax.Array, *, strict_shape_alignment: bool) -> jax.Array:
    """Fast path for packing uint2-like codes into uint32."""
    return _pack_bits_fast_grouped(
        values,
        bits=2,
        values_per_word=16,
        mask=0x3,
        strict_shape_alignment=strict_shape_alignment,
    )


def _pack_bits_u4_fast(values: jax.Array, *, strict_shape_alignment: bool) -> jax.Array:
    """Fast path for packing uint4-like codes into uint32."""
    return _pack_bits_fast_grouped(
        values,
        bits=4,
        values_per_word=8,
        mask=0xF,
        strict_shape_alignment=strict_shape_alignment,
    )


def _pack_bits_u8_fast(values: jax.Array, *, strict_shape_alignment: bool) -> jax.Array:
    """Fast path for packing uint8-like codes into uint32."""
    return _pack_bits_fast_grouped(
        values,
        bits=8,
        values_per_word=4,
        mask=0xFF,
        strict_shape_alignment=strict_shape_alignment,
    )


def _unpack_bits_u1_fast(packed: jax.Array, n: int) -> jax.Array:
    """Fast path for unpacking uint1-like codes from uint32."""
    return _unpack_bits_fast_grouped(packed, n, bits=1, values_per_word=32, mask=0x1)


def _unpack_bits_u2_fast(packed: jax.Array, n: int) -> jax.Array:
    """Fast path for unpacking uint2-like codes from uint32."""
    return _unpack_bits_fast_grouped(packed, n, bits=2, values_per_word=16, mask=0x3)


def _unpack_bits_u4_fast(packed: jax.Array, n: int) -> jax.Array:
    """Fast path for unpacking uint4-like codes from uint32."""
    return _unpack_bits_fast_grouped(packed, n, bits=4, values_per_word=8, mask=0xF)


def _unpack_bits_u8_fast(packed: jax.Array, n: int) -> jax.Array:
    """Fast path for unpacking uint8-like codes from uint32."""
    return _unpack_bits_fast_grouped(packed, n, bits=8, values_per_word=4, mask=0xFF)


def _pack_bits(
    values: jax.Array,
    bits: int,
    *,
    prefer_fast_u4_u8: bool = True,
    strict_shape_alignment: bool = False,
) -> jax.Array:
    """Pack quantized codes into uint32 words (LSB-first).

    Args:
        values: Array of quantized codes to pack. The last dimension contains
            the values to pack. Each value must fit in `bits` bits.
        bits: Number of bits per value.
        prefer_fast_u4_u8: Enables dedicated u1/u2/u4/u8 kernels for bits in
            {1, 2, 4, 8}.
        strict_shape_alignment: When True, requires aligned lengths for fast
            u4/u8 packing instead of auto-padding.

    Returns:
        Packed uint32 array with shape (*values.shape[:-1], n_words).
    """
    bits = int(bits)
    force_fast_on_mps = jax.default_backend() == "mps" and bits in {1, 2, 4, 8}

    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 1:
        return _pack_bits_u1_fast(values, strict_shape_alignment=strict_shape_alignment)
    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 2:
        return _pack_bits_u2_fast(values, strict_shape_alignment=strict_shape_alignment)
    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 4:
        return _pack_bits_u4_fast(values, strict_shape_alignment=strict_shape_alignment)
    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 8:
        return _pack_bits_u8_fast(values, strict_shape_alignment=strict_shape_alignment)
    return _pack_bits_generic(values, bits)


def _unpack_bits(
    packed: jax.Array,
    n: int,
    bits: int,
    *,
    prefer_fast_u4_u8: bool = True,
) -> jax.Array:
    """Unpack quantized codes from uint32 words (LSB-first).

    Args:
        packed: Packed uint32 array with shape (*batch_dims, n_words).
        n: Number of values to extract from the last dimension.
        bits: Number of bits per value.
        prefer_fast_u4_u8: Enables dedicated u1/u2/u4/u8 kernels for bits in
            {1, 2, 4, 8}.

    Returns:
        Unpacked uint32 array with shape (*packed.shape[:-1], n).
    """
    bits = int(bits)
    force_fast_on_mps = jax.default_backend() == "mps" and bits in {1, 2, 4, 8}

    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 1:
        return _unpack_bits_u1_fast(packed, n)
    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 2:
        return _unpack_bits_u2_fast(packed, n)
    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 4:
        return _unpack_bits_u4_fast(packed, n)
    if (prefer_fast_u4_u8 or force_fast_on_mps) and bits == 8:
        return _unpack_bits_u8_fast(packed, n)
    return _unpack_bits_generic(packed, n, bits)
