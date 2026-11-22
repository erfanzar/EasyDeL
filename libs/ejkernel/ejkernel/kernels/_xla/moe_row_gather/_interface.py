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

"""XLA registry entries (priority 0 fallbacks) for the MoE row gathers."""

from __future__ import annotations

import jax
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import (
    moe_row_gather_quantized_xla,
    moe_row_gather_sum_packed_xla,
    moe_row_gather_sum_xla,
    moe_row_gather_xla,
)


@kernel_registry.register("moe_row_gather", Platform.XLA, Backend.ANY)
def moe_row_gather(
    x: Float[Array, "src_rows hidden"],
    indices: Int[Array, "rows"],
    scale: Float[Array, "rows"] | None = None,
    dot_with: Float[Array, "rows hidden"] | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[Float[Array, "rows hidden"], Float[Array, "rows"] | None]:
    """``out[i] = scale[i] * x[indices[i]]`` and optionally ``dot[i] = <x[indices[i]], dot_with[i]>``, as a gather.

    ``out_dtype`` (default ``x.dtype``) casts the gathered rows, e.g. int8 for int8 codes carried as bfloat16.

    Args:
        x: ``[src_rows, hidden]`` source rows.
        indices: ``[rows]`` row ids (clamped to range).
        scale: Optional ``[rows]`` fp32 factor applied before the cast back to ``x.dtype``.
        dot_with: Optional ``[rows, hidden]``; returns the fp32 dot of each gathered (unscaled) row with it.

    Returns:
        ``(out [rows, hidden] in x.dtype, dot [rows] fp32 or None)``.
    """
    return moe_row_gather_xla(x, indices, scale, dot_with, out_dtype)


@kernel_registry.register("moe_row_gather_quantized", Platform.XLA, Backend.ANY)
def moe_row_gather_quantized(
    codes: Int[Array, "src_rows hidden"],
    scale: Float[Array, "src_rows"],
    indices: Int[Array, "rows"],
) -> tuple[Int[Array, "rows hidden"], Float[Array, "rows"]]:
    """``(codes[indices], scale[indices])``: int8 code rows with their fp32 row scales, as two gathers.

    Args:
        codes: ``[src_rows, hidden]`` int8 codes.
        scale: ``[src_rows]`` fp32 row scales.
        indices: ``[rows]`` row ids (clamped to range).

    Returns:
        ``(codes [rows, hidden] int8, scale [rows] fp32)``.
    """
    return moe_row_gather_quantized_xla(codes, scale, indices)


@kernel_registry.register("moe_row_gather_sum", Platform.XLA, Backend.ANY)
def moe_row_gather_sum(
    y: Float[Array, "src_rows hidden"],
    indices: Int[Array, "tokens k"],
    weights: Float[Array, "tokens k"],
    out_dtype: jax.typing.DTypeLike | None = None,
    skip_negative: bool = False,
) -> Float[Array, "tokens hidden"]:
    """``out[t] = sum_j weights[t, j] * y[indices[t, j]]`` (fp32 accumulation), as a gather plus a reduction.

    Args:
        y: ``[src_rows, hidden]`` source rows.
        indices: ``[tokens, k]`` row ids (clamped to range; negative = inactive slot under ``skip_negative``).
        weights: ``[tokens, k]`` combine weights.
        out_dtype: Output dtype (default ``y.dtype``).
        skip_negative: A negative index contributes exactly zero (selected away before the multiply, never
            multiplied, so non-finite rows that only inactive slots point at cannot leak into the sum).

    Returns:
        ``[tokens, hidden]`` in ``out_dtype``.
    """
    return moe_row_gather_sum_xla(y, indices, weights, out_dtype, skip_negative)


@kernel_registry.register("moe_row_gather_sum_packed", Platform.XLA, Backend.ANY)
def moe_row_gather_sum_packed(
    packed: Int[Array, "packed_rows lanes"],
    indices: Int[Array, "tokens k"],
    weights: Float[Array, "tokens k"],
    hidden: int,
    pair_width: int,
    out_dtype: jax.typing.DTypeLike | None = None,
    skip_negative: bool = False,
) -> Float[Array, "tokens hidden"]:
    """``moe_row_gather_sum`` of bfloat16 rows given in the int32 ``pack_rows_paired`` layout.

    Args:
        packed: int32 ``[src_rows * hidden / 256, 128]`` (``pack_rows_paired(y, pair_width)``).
        indices: ``[tokens, k]`` row ids (clamped; negative = inactive slot under ``skip_negative``).
        weights: ``[tokens, k]`` combine weights.
        hidden: Row width of the packed bfloat16 rows.
        pair_width: Column-pairing block width of the layout.
        out_dtype: Output dtype (default bfloat16).
        skip_negative: As in ``moe_row_gather_sum``.

    Returns:
        ``[tokens, hidden]`` in ``out_dtype``.
    """
    return moe_row_gather_sum_packed_xla(packed, indices, weights, hidden, pair_width, out_dtype, skip_negative)


__all__ = ("moe_row_gather", "moe_row_gather_quantized", "moe_row_gather_sum", "moe_row_gather_sum_packed")
