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
) -> tuple[Array, Array | None]:
    """``out[i] = scale[i] * x[indices[i]]`` (in ``x.dtype``) and ``dot[i] = <x[indices[i]], dot_with[i]>`` (fp32).

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
    return rows, dot


def moe_row_gather_sum_xla(
    y: Array,
    indices: Array,
    weights: Array,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> Array:
    """``out[t] = sum_j weights[t, j] * y[indices[t, j]]``, accumulated in fp32.

    Indices are clamped to ``[0, src_rows)``.

    Args:
        y: ``[src_rows, ...]`` source rows.
        indices: ``[tokens, k]`` integer row ids.
        weights: ``[tokens, k]`` combine weights.
        out_dtype: Output dtype (default ``y.dtype``).

    Returns:
        ``[tokens, ...]`` in ``out_dtype``.
    """
    tokens, k = indices.shape
    rows = jnp.take(y, indices.reshape(-1), axis=0, mode="clip").astype(F32)
    rows = rows.reshape((tokens, k, *y.shape[1:]))
    factor = weights.astype(F32).reshape((tokens, k, *((1,) * (y.ndim - 1))))
    return jnp.sum(rows * factor, axis=1).astype(out_dtype or y.dtype)
