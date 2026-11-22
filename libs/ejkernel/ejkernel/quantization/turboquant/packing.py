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

"""Bit packing utilities for TurboQuant compressed representations.

Provides efficient packing/unpacking for:
- 4-bit codebook indices (2 values per uint8)
- 1-bit QJL signs (8 values per uint8)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pack_4bit(indices: jax.Array) -> jax.Array:
    """Pack pairs of 4-bit values into uint8.

    Takes an array where each element is in [0, 15] and packs
    consecutive pairs along the last axis into single uint8 values.
    The first value occupies the low nibble, the second the high nibble.

    Args:
        indices: Array with last dimension divisible by 2.
            Values must be in range [0, 15].

    Returns:
        uint8 array with last dimension halved.
    """
    shape = indices.shape
    assert shape[-1] % 2 == 0, f"Last dimension must be even, got {shape[-1]}"

    indices = indices.astype(jnp.uint8)
    reshaped = indices.reshape(*shape[:-1], shape[-1] // 2, 2)
    low = reshaped[..., 0]
    high = reshaped[..., 1]
    packed = low | (high << 4)
    return packed


def unpack_4bit(packed: jax.Array) -> jax.Array:
    """Unpack uint8 array into pairs of 4-bit values.

    Reverses pack_4bit: extracts two 4-bit values from each uint8.

    Args:
        packed: uint8 array of packed 4-bit pairs.

    Returns:
        uint8 array with last dimension doubled, values in [0, 15].
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    shape = packed.shape
    unpacked = jnp.stack([low, high], axis=-1)
    return unpacked.reshape(*shape[:-1], shape[-1] * 2)


def pack_signs(signs: jax.Array) -> jax.Array:
    """Pack boolean/binary sign values into bit-packed uint8.

    Each group of 8 sign values along the last axis is packed into
    one uint8 byte. sign[i] = True/1 sets bit i in the byte.

    Args:
        signs: Boolean or 0/1 array with last dimension divisible by 8.

    Returns:
        uint8 array with last dimension divided by 8.
    """
    shape = signs.shape
    assert shape[-1] % 8 == 0, f"Last dimension must be divisible by 8, got {shape[-1]}"

    signs = signs.astype(jnp.uint8)
    reshaped = signs.reshape(*shape[:-1], shape[-1] // 8, 8)

    bits = jnp.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=jnp.uint8)
    packed = jnp.sum(reshaped * bits[None, :], axis=-1).astype(jnp.uint8)
    return packed


def unpack_signs(packed: jax.Array) -> jax.Array:
    """Unpack bit-packed uint8 into float sign values (+1/-1).

    Reverses pack_signs and converts to float: each bit becomes
    +1.0 (if set) or -1.0 (if unset).

    Args:
        packed: uint8 array of bit-packed signs.

    Returns:
        float32 array with last dimension multiplied by 8.
        Values are +1.0 or -1.0.
    """
    shape = packed.shape
    bits = jnp.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=jnp.uint8)

    expanded = packed[..., None].astype(jnp.uint8)
    unpacked = (expanded & bits) > 0

    unpacked = unpacked.reshape(*shape[:-1], shape[-1] * 8)
    return jnp.where(unpacked, 1.0, -1.0).astype(jnp.float32)
