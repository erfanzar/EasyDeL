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

"""Decode-specialized W4A16 matmul streaming bit-packed int4 weights.

Why this kernel exists — three measured facts (TPU v5, 2026-08):

* XLA stores ``int4`` arrays **unpacked**, one byte per element, so the XLA
  fused-upcast decode path streams the same bytes as int8 and its speedup
  asymptote is 2x over bf16. Reaching the int4 byte ratio (4x) requires
  packing two values per byte and unpacking on-chip — which XLA cannot
  express without breaking its convert-into-matmul fusion.
* A *grouped* Pallas kernel is the wrong vehicle: v3's pure-bf16 ceiling
  measured 0.47-0.78x of XLA. This kernel is the opposite extreme — no
  groups, no metadata, grid over N only, activations fully VMEM-resident
  (decode ``m`` is tiny) — so the weight stream is the only real traffic.
* In-kernel bit manipulation is what made the legacy packed-uint32 kernel
  compile for 25+ minutes — but that kernel decoded e2m1 floats with
  sign/exponent/mantissa arithmetic over 8-way unpacked words. Here the
  entire decode is one shift, one mask and one subtract per nibble; the
  Mosaic IR stays small.

Packing layout — split-K halves, chosen so unpacking never shuffles:

    byte[i, n] = (q[i, n] + 8) | ((q[i + K/2, n] + 8) << 4)     i < K/2

The low nibbles of a packed tile ARE rows ``[0, K/2)`` of the weight and the
high nibbles ARE rows ``[K/2, K)``, each contiguous — so a tile unpacks into
two dense blocks that feed two ``bf16`` matmuls against the matching halves
of the activations. No interleave, no gather, no lane permutes.
"""

from __future__ import annotations

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = ["pack_int4_adjacent", "pack_int4_split_k", "packed_int4_gemv", "supports_int4_mxu", "w4a4_gemv"]


def supports_int4_mxu() -> bool:
    """Whether this TPU generation runs int4 x int4 on the MXU natively.

    Queried at trace time — no generation is hardcoded. The W4A4 kernels
    require this pair (measured native on v5p at 2x the int8 rate); callers
    on hardware without it must use the W4A16 or XLA paths instead.

    Returns:
        ``True`` when ``int4 x int4`` matmuls lower natively.
    """
    try:
        return bool(pltpu.get_tpu_info().is_matmul_supported(jnp.int4, jnp.int4))
    except Exception:  # noqa: BLE001 - no TPU / no info: not supported
        return False


def pack_int4_split_k(codes: jax.Array) -> jax.Array:
    """Pack signed int4 codes two-per-byte in the split-K-halves layout.

    Args:
        codes: Integer codes in ``[-8, 7]``, shape ``[K, N]`` with even ``K``.
            Any integer dtype.

    Returns:
        ``uint8`` array of shape ``[K // 2, N]`` where the low nibble of row
        ``i`` holds ``codes[i] + 8`` and the high nibble holds
        ``codes[i + K // 2] + 8``.

    Raises:
        ValueError: If ``K`` is odd.
    """
    k_dim = codes.shape[0]
    if k_dim % 2:
        raise ValueError(f"K must be even to pack two int4 per byte, got {k_dim}.")
    unsigned = (codes.astype(jnp.int32) + 8).astype(jnp.uint8)
    low = unsigned[: k_dim // 2]
    high = unsigned[k_dim // 2 :]
    return low | (high << 4)


def _gemv_kernel(x_ref, w_ref, scale_ref, out_ref, *, chunk: int):
    """One N-tile of the packed W4A16 gemv.

    Args:
        x_ref: Activations ``[m, K]`` bf16, fully resident.
        w_ref: Packed weight tile ``[K // 2, tile_n]`` uint8.
        scale_ref: Per-channel scale ``[1, tile_n]`` float32.
        out_ref: Output tile ``[m, tile_n]``.
        chunk: Packed rows decoded per inner step; each step feeds two
            ``[m, chunk] @ [chunk, tile_n]`` matmuls (low and high halves).
    """
    packed_k = w_ref.shape[0]
    tile_n = w_ref.shape[1]
    m = x_ref.shape[0]

    acc = jnp.zeros((m, tile_n), dtype=jnp.float32)
    for start in range(0, packed_k, chunk):
        end = min(packed_k, start + chunk)
        packed = w_ref[start:end, :]

        # Minimal-op nibble decode. Mosaic-v5 constraints: integer bitwise
        # ops only vectorize at 16/32 bits, and unsigned shifts do not
        # legalize, so the math runs in int32 with a signed shift (packed
        # bytes are non-negative, so it is equivalent). Two deliberate
        # omissions, because the decode is VPU-bound and every per-element op
        # counts:
        #   * no `& 0xF` after the shift — a byte >> 4 is already <= 15;
        #   * no `- 8` recentering — the matmul runs on the biased codes
        #     u = q + 8 and the bias is removed algebraically afterwards:
        #     sum_k x_k (u_k - 8) = sum_k x_k u_k - 8 * sum_k x_k,
        #     one row-sum correction per tile instead of one subtract per
        #     weight element.
        packed32 = packed.astype(jnp.int32)
        low = (packed32 & 0xF).astype(jnp.bfloat16)
        high = (packed32 >> 4).astype(jnp.bfloat16)

        x_low = x_ref[:, start:end]
        x_high = x_ref[:, packed_k + start : packed_k + end]
        acc += jnp.matmul(x_low, low, preferred_element_type=jnp.float32)
        acc += jnp.matmul(x_high, high, preferred_element_type=jnp.float32)

    x_row_sum = jnp.sum(x_ref[...].astype(jnp.float32), axis=1, keepdims=True)
    out_ref[...] = ((acc - 8.0 * x_row_sum) * scale_ref[...]).astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("tile_n", "chunk", "interpret"))
def packed_int4_gemv(
    x: jax.Array,
    w_packed: jax.Array,
    channel_scale: jax.Array,
    *,
    tile_n: int = 512,
    chunk: int = 256,
    interpret: bool = False,
) -> jax.Array:
    """Compute ``x @ dequant(w)`` streaming two int4 weights per byte.

    Args:
        x: Activations ``[m, k]``, floating dtype. Decode-sized ``m`` — the
            whole activation block is held in VMEM.
        w_packed: ``uint8`` ``[k // 2, n]`` in the :func:`pack_int4_split_k`
            layout.
        channel_scale: Per-output-channel scale; any shape reshapeable to
            ``[1, n]``.
        tile_n: Output columns per grid step (the weight-stream tile width).
        chunk: Packed rows decoded per inner step.
        interpret: Run in Pallas interpret mode (CPU testing).

    Returns:
        ``[m, n]`` in ``x``'s dtype.

    Raises:
        ValueError: On shape mismatch between ``x`` and ``w_packed``.
    """
    m, k_dim = x.shape
    packed_k, n_dim = w_packed.shape
    if packed_k * 2 != k_dim:
        raise ValueError(f"w_packed has {packed_k} packed rows; expected {k_dim // 2} for K={k_dim}.")

    scale = channel_scale.reshape(1, n_dim).astype(jnp.float32)
    x = x.astype(jnp.bfloat16)

    grid = (n_dim // tile_n,)
    return pl.pallas_call(
        functools.partial(_gemv_kernel, chunk=chunk),
        grid=grid,
        in_specs=[
            pl.BlockSpec((m, k_dim), lambda j: (0, 0)),
            pl.BlockSpec((packed_k, tile_n), lambda j: (0, j)),
            pl.BlockSpec((1, tile_n), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((m, tile_n), lambda j: (0, j)),
        out_shape=jax.ShapeDtypeStruct((m, n_dim), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )(x, w_packed, scale)


def pack_int4_adjacent(codes: jax.Array) -> jax.Array:
    """Pack signed int4 codes two-per-byte in adjacent-rows order.

    The layout ``byte[i] = (q[2i] & 0xF) | ((q[2i+1] & 0xF) << 4)`` — raw
    two's-complement nibbles, no bias — is exactly what Mosaic's
    ``pltpu.bitcast(uint8 -> int4)`` expands back into row order, so the
    kernel-side decode is a free register reinterpret.

    Args:
        codes: Integer codes in ``[-8, 7]``, shape ``[K, N]`` with even ``K``.

    Returns:
        ``uint8`` array of shape ``[K // 2, N]``.

    Raises:
        ValueError: If ``K`` is odd.
    """
    if codes.shape[0] % 2:
        raise ValueError(f"K must be even to pack two int4 per byte, got {codes.shape[0]}.")
    nibbles = codes.astype(jnp.int32) & 0xF
    return (nibbles[0::2] | (nibbles[1::2] << 4)).astype(jnp.uint8)


def _w4a4_kernel(x_ref, w_ref, scale_ref, out_ref):
    """One N-tile of the W4A4 gemv: bitcast feed straight into the int4 MXU."""
    weights = pltpu.bitcast(w_ref[...], jnp.int4)
    acc = jnp.matmul(x_ref[...], weights, preferred_element_type=jnp.int32)
    out_ref[...] = (acc.astype(jnp.float32) * scale_ref[...]).astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("tile_n", "interpret"))
def w4a4_gemv(
    x: jax.Array,
    w_packed: jax.Array,
    channel_scale: jax.Array,
    *,
    tile_n: int = 1024,
    interpret: bool = False,
) -> jax.Array:
    """W4A4 decode matmul: packed int4 weights fed to the int4 MXU by bitcast.

    The fastest quantized decode measured on TPU v5 (2.2x over bf16 at
    27B-class shapes) and the only path that beats the element-conversion
    bound: the weight is never touched by the VPU — ``pltpu.bitcast`` is a
    register-level reinterpret and v5's MXU consumes int4 natively at twice
    the int8 rate.

    The cost is activation precision: the caller supplies **int4-quantized
    activations** and applies their per-token scale to the returned product.
    Unsmoothed per-token int4 activations measure ~0.11 relative error on
    gaussian data — this kernel is an opt-in for calibrated/smoothed setups,
    not a default.

    Args:
        x: Activations ``[m, k]``, already quantized, dtype ``int4``.
        w_packed: ``uint8`` ``[k // 2, n]`` in :func:`pack_int4_adjacent`
            layout.
        channel_scale: Per-output-channel weight scale, reshapeable to
            ``[1, n]``.
        tile_n: Output columns per grid step.
        interpret: Run in Pallas interpret mode (CPU testing).

    Returns:
        ``[m, n]`` bf16 — the caller multiplies by its activation scale.

    Raises:
        ValueError: On dtype/shape mismatch.
    """
    if x.dtype != jnp.int4:
        raise ValueError(f"w4a4_gemv expects int4 activations, got {x.dtype}.")
    if not interpret and not supports_int4_mxu():
        raise NotImplementedError(
            "w4a4_gemv requires native int4 x int4 MXU support, which this TPU generation "
            "does not report. Use packed_int4_gemv (W4A16) or the XLA channelwise path instead."
        )
    m, k_dim = x.shape
    packed_k, n_dim = w_packed.shape
    if packed_k * 2 != k_dim:
        raise ValueError(f"w_packed has {packed_k} packed rows; expected {k_dim // 2} for K={k_dim}.")
    scale = channel_scale.reshape(1, n_dim).astype(jnp.float32)

    return pl.pallas_call(
        _w4a4_kernel,
        grid=(n_dim // tile_n,),
        in_specs=[
            pl.BlockSpec((m, k_dim), lambda j: (0, 0)),
            pl.BlockSpec((packed_k, tile_n), lambda j: (0, j)),
            pl.BlockSpec((1, tile_n), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((m, tile_n), lambda j: (0, j)),
        out_shape=jax.ShapeDtypeStruct((m, n_dim), jnp.bfloat16),
        # "parallel" measured ~8% faster than "arbitrary" for this grid
        # (222us vs 241us on the 27B shape); N-tiles are independent.
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, w_packed, scale)
