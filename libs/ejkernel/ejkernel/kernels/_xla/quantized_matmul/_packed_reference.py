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

"""XLA reference implementations for the packed-int4 gemv kernels.

The mandatory ``Platform.XLA`` fallbacks for ``packed_int4_gemv`` (W4A16,
split-K packing) and ``w4a4_gemv`` (adjacent-rows packing, integer-exact
math). Pure ``jnp`` — runnable on CPU/GPU/TPU, no Pallas imports — and
signature-identical to the Pallas implementations so
``kernel_registry.validate_signatures`` holds. The tiling keyword arguments
are accepted for signature parity and ignored: XLA schedules its own tiling.

The ``w4a4`` reference computes the integer dot in float32, which is exact
for int4 operands at these contraction sizes (partial sums stay far below
the 2^24 float32 integer boundary times the code magnitudes involved).
"""

from __future__ import annotations

import jax
from jax import numpy as jnp

from ..._registry import Backend, Platform, kernel_registry

__all__ = ["packed_int4_gemv_xla", "w4a4_gemv_xla"]


def _unpack_split_k(w_packed: jax.Array) -> jax.Array:
    """Decode the split-K-halves layout back to signed codes ``[K, N]``.

    Inverse of ``pack_int4_split_k``: the low nibble of packed row ``i`` is
    ``codes[i] + 8`` and the high nibble is ``codes[i + K/2] + 8``.

    Args:
        w_packed: ``uint8`` ``[K // 2, N]``.

    Returns:
        Signed codes ``[K, N]`` as int32.
    """
    packed = w_packed.astype(jnp.int32)
    low = (packed & 0xF) - 8
    high = (packed >> 4) - 8
    return jnp.concatenate([low, high], axis=0)


def _unpack_adjacent(w_packed: jax.Array) -> jax.Array:
    """Decode the adjacent-rows layout back to signed codes ``[K, N]``.

    Inverse of ``pack_int4_adjacent``: raw two's-complement nibbles, byte of
    packed row ``i`` holding rows ``2i`` (low) and ``2i + 1`` (high).

    Args:
        w_packed: ``uint8`` ``[K // 2, N]``.

    Returns:
        Signed codes ``[K, N]`` as int32.
    """
    packed = w_packed.astype(jnp.int32)
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    low = jnp.where(low > 7, low - 16, low)
    high = jnp.where(high > 7, high - 16, high)
    stacked = jnp.stack([low, high], axis=1)  # [K//2, 2, N]
    return stacked.reshape(low.shape[0] * 2, low.shape[1])


@kernel_registry.register("packed_int4_gemv", Platform.XLA, Backend.ANY)
def packed_int4_gemv_xla(
    x: jax.Array,
    w_packed: jax.Array,
    channel_scale: jax.Array,
    *,
    tile_n: int = 512,
    chunk: int = 256,
    interpret: bool = False,
) -> jax.Array:
    """Reference ``x @ dequant(w)`` for split-K packed int4 weights.

    Args:
        x: Activations ``[m, k]``, floating dtype.
        w_packed: ``uint8`` ``[k // 2, n]`` in split-K-halves layout.
        channel_scale: Per-output-channel scale, reshapeable to ``[1, n]``.
        tile_n: Ignored (signature parity with the Pallas kernel).
        chunk: Ignored (signature parity).
        interpret: Ignored (signature parity).

    Returns:
        ``[m, n]`` bf16 — the Pallas kernel casts activations to bf16 and
        emits bf16 regardless of input dtype, and the reference must match
        the op's output contract exactly.
    """
    del tile_n, chunk, interpret
    n_dim = w_packed.shape[1]
    codes = _unpack_split_k(w_packed)
    dense = codes.astype(jnp.float32) * channel_scale.reshape(1, n_dim).astype(jnp.float32)
    out = x.astype(jnp.float32) @ dense
    return out.astype(jnp.bfloat16)


@kernel_registry.register("w4a4_gemv", Platform.XLA, Backend.ANY)
def w4a4_gemv_xla(
    x: jax.Array,
    w_packed: jax.Array,
    channel_scale: jax.Array,
    *,
    tile_n: int = 1024,
    interpret: bool = False,
) -> jax.Array:
    """Reference W4A4 gemv: integer-exact math over adjacent-packed weights.

    Args:
        x: Activations ``[m, k]`` already quantized, dtype ``int4``.
        w_packed: ``uint8`` ``[k // 2, n]`` in adjacent-rows layout.
        channel_scale: Per-output-channel weight scale, reshapeable to
            ``[1, n]``.
        tile_n: Ignored (signature parity with the Pallas kernel).
        interpret: Ignored (signature parity).

    Returns:
        ``[m, n]`` bf16 — the caller applies its activation scale, matching
        the Pallas kernel's contract.

    Raises:
        ValueError: If ``x`` is not int4.
    """
    del tile_n, interpret
    if x.dtype != jnp.int4:
        raise ValueError(f"w4a4_gemv expects int4 activations, got {x.dtype}.")
    n_dim = w_packed.shape[1]
    codes = _unpack_adjacent(w_packed)
    out = x.astype(jnp.float32) @ codes.astype(jnp.float32)
    return (out * channel_scale.reshape(1, n_dim).astype(jnp.float32)).astype(jnp.bfloat16)
