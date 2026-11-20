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

"""XLA reference for the MoE row gathers (dispatch gather and expert-output gather-sum)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

F32 = jnp.float32


def moe_row_gather_xla(
    x: Array,
    indices: Array,
    scale: Array | None = None,
    dot_with: Array | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[Array, Array | None]:
    """``out[i] = scale[i] * x[indices[i]]`` (in ``x.dtype``) and ``dot[i] = <x[indices[i]], dot_with[i]>`` (fp32).

    ``out_dtype`` (default ``x.dtype``) casts the gathered rows at the end, e.g. int8 for integer-valued rows.

    Indices are clamped to ``[0, src_rows)``.

    Args:
        x: ``[src_rows, ...]`` source rows.
        indices: ``[rows]`` integer row ids.
        scale: Optional ``[rows]`` per-row factor, applied in fp32 before the cast back to ``x.dtype``.
        dot_with: Optional ``[rows, ...]``; when given, the fp32 dot of each gathered (unscaled) row with it.

    Returns:
        ``(out [rows, ...], dot [rows] or None)``.
    """
    rows = jnp.take(x, indices, axis=0, mode="clip")
    dot = None
    if dot_with is not None:
        dot = jnp.sum((rows.astype(F32) * dot_with.astype(F32)).reshape(rows.shape[0], -1), axis=-1)
    if scale is not None:
        factor = scale.astype(F32).reshape((*scale.shape, *((1,) * (rows.ndim - 1))))
        rows = (rows.astype(F32) * factor).astype(x.dtype)
    if out_dtype is not None and jnp.dtype(out_dtype) != rows.dtype:
        rows = rows.astype(out_dtype)
    return rows, dot


def pack_rows_paired(y: Array, pair_width: int) -> Array:
    """bfloat16 ``[M, H]`` -> int32 ``[M * H / 256, 128]``, columns paired within each ``pair_width`` block.

    Row ``i`` occupies packed rows ``[i * H / 256, (i + 1) * H / 256)``; packed row ``i * H / 256 + n * W / 256 + a``
    (``W = pair_width``), lane ``l`` holds the bits of column ``n * W + a * 128 + l`` (low 16) and of column
    ``n * W + W / 2 + a * 128 + l`` (high 16). With ``W = H`` this is the Pallas ``pack_rows`` layout. Pure bit
    movement.
    """
    m, h = y.shape
    if y.dtype != jnp.bfloat16 or h % pair_width or pair_width % 256:
        raise ValueError(f"pack_rows_paired needs bfloat16 rows and a 256-aligned pair width dividing {h}")
    bits = jax.lax.bitcast_convert_type(y, jnp.uint16).astype(jnp.uint32).reshape(m, h // pair_width, 2, pair_width // 2)
    words = bits[:, :, 0] | (bits[:, :, 1] << 16)
    return jax.lax.bitcast_convert_type(words, jnp.int32).reshape(m * h // 256, 128)


def unpack_rows_paired(packed: Array, hidden: int, pair_width: int) -> Array:
    """Inverse of :func:`pack_rows_paired`: int32 ``[M * H / 256, 128]`` -> bfloat16 ``[M, H]`` (bit for bit)."""
    m = packed.shape[0] * 256 // hidden
    words = jax.lax.bitcast_convert_type(packed, jnp.uint32).reshape(m, hidden // pair_width, pair_width // 2)
    lo = jax.lax.bitcast_convert_type((words & 0xFFFF).astype(jnp.uint16), jnp.bfloat16)
    hi = jax.lax.bitcast_convert_type((words >> 16).astype(jnp.uint16), jnp.bfloat16)
    return jnp.concatenate([lo, hi], axis=-1).reshape(m, hidden)


def moe_row_gather_quantized_xla(codes: Array, scale: Array, indices: Array) -> tuple[Array, Array]:
    """``(codes[indices], scale[indices])``: int8 code rows and their fp32 row scales, gathered together.

    Indices are clamped to ``[0, src_rows)``.

    Args:
        codes: ``[src_rows, hidden]`` int8 codes.
        scale: ``[src_rows]`` fp32 row scales.
        indices: ``[rows]`` integer row ids.

    Returns:
        ``(codes [rows, hidden], scale [rows])``.
    """
    return jnp.take(codes, indices, axis=0, mode="clip"), jnp.take(scale, indices, axis=0, mode="clip")


def moe_row_gather_sum_xla(
    y: Array,
    indices: Array,
    weights: Array,
    out_dtype: jax.typing.DTypeLike | None = None,
    skip_negative: bool = False,
) -> Array:
    """``out[t] = sum_j weights[t, j] * y[indices[t, j]]``, accumulated in fp32.

    Indices are clamped to ``[0, src_rows)``. With ``skip_negative`` a negative index marks an inactive slot: its row
    is selected away before the multiply, so it contributes exactly zero whatever its weight and whatever the clamped
    source row holds (a non-finite row that only inactive slots point at never reaches the sum).

    Args:
        y: ``[src_rows, ...]`` source rows.
        indices: ``[tokens, k]`` integer row ids.
        weights: ``[tokens, k]`` combine weights.
        out_dtype: Output dtype (default ``y.dtype``).
        skip_negative: Treat negative indices as inactive slots instead of clamping them to row 0.

    Returns:
        ``[tokens, ...]`` in ``out_dtype``.
    """
    tokens, k = indices.shape
    rows = jnp.take(y, indices.reshape(-1), axis=0, mode="clip").astype(F32)
    rows = rows.reshape((tokens, k, *y.shape[1:]))
    if skip_negative:
        live = (indices >= 0).reshape((tokens, k, *((1,) * (y.ndim - 1))))
        rows = jnp.where(live, rows, 0.0)
    factor = weights.astype(F32).reshape((tokens, k, *((1,) * (y.ndim - 1))))
    return jnp.sum(rows * factor, axis=1).astype(out_dtype or y.dtype)


def moe_row_gather_sum_packed_xla(
    packed: Array,
    indices: Array,
    weights: Array,
    hidden: int,
    pair_width: int,
    out_dtype: jax.typing.DTypeLike | None = None,
    skip_negative: bool = False,
) -> Array:
    """:func:`moe_row_gather_sum_xla` of the bfloat16 rows held in ``packed`` (:func:`pack_rows_paired` layout)."""
    y = unpack_rows_paired(packed, hidden, pair_width)
    return moe_row_gather_sum_xla(y, indices, weights, out_dtype, skip_negative)
