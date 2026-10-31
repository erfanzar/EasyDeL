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

"""Registry entries (Pallas, TPU) for the MoE row gathers.

Off TPU, or for shapes/dtypes the DMA kernels do not support (see the ``*_unsupported_reason`` helpers), the calls
route to the XLA gather. Callers that must not fall back silently check the reason helpers first.
"""

from __future__ import annotations

import jax
from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.moe_row_gather._xla_impl import moe_row_gather_sum_xla, moe_row_gather_xla
from ._pallas_impl import gather_rows, gather_sum_rows, gather_sum_unsupported_reason, gather_unsupported_reason


def row_gather_unsupported_reason(x, indices, scale=None, dot_with=None) -> str | None:
    """Why ``moe_row_gather`` would use the XLA fallback for these arguments (``None``: the Pallas kernel runs).

    Only ``.shape`` / ``.dtype`` are read, so ``jax.ShapeDtypeStruct`` arguments work.
    """
    if jax.default_backend() != "tpu":
        return f"backend is {jax.default_backend()}, not tpu"
    if len(indices.shape) != 1:
        return f"indices must be 1-D, got shape {tuple(indices.shape)}"
    rows = indices.shape[0]
    if scale is not None and tuple(scale.shape) != (rows,):
        return f"scale shape {tuple(scale.shape)} != ({rows},)"
    if dot_with is not None and (
        tuple(dot_with.shape) != (rows, *tuple(x.shape)[1:]) or jax.numpy.dtype(dot_with.dtype) != x.dtype
    ):
        return f"dot_with must be {(rows, *tuple(x.shape)[1:])} in {x.dtype}"
    return gather_unsupported_reason(tuple(x.shape), x.dtype, rows, has_dot=dot_with is not None)


def row_gather_sum_unsupported_reason(y, indices, weights, out_dtype=None) -> str | None:
    """Why ``moe_row_gather_sum`` would use the XLA fallback for these arguments (``None``: the Pallas kernel runs)."""
    if jax.default_backend() != "tpu":
        return f"backend is {jax.default_backend()}, not tpu"
    if len(indices.shape) != 2 or tuple(weights.shape) != tuple(indices.shape):
        return f"indices {tuple(indices.shape)} and weights {tuple(weights.shape)} must both be [tokens, k]"
    tokens, k = indices.shape
    return gather_sum_unsupported_reason(tuple(y.shape), y.dtype, tokens, k, out_dtype)


@kernel_registry.register("moe_row_gather", Platform.PALLAS, Backend.TPU, priority=10)
def moe_row_gather(
    x: Float[Array, "src_rows hidden"],
    indices: Int[Array, "rows"],
    scale: Float[Array, "rows"] | None = None,
    dot_with: Float[Array, "rows hidden"] | None = None,
) -> tuple[Float[Array, "rows hidden"], Float[Array, "rows"] | None]:
    """``out[i] = scale[i] * x[indices[i]]`` and optionally ``dot[i] = <x[indices[i]], dot_with[i]>`` by row DMA.

    Args:
        x: ``[src_rows, hidden]`` source rows.
        indices: ``[rows]`` row ids (clamped to range).
        scale: Optional ``[rows]`` fp32 factor applied before the cast back to ``x.dtype``.
        dot_with: Optional ``[rows, hidden]``; returns the fp32 dot of each gathered (unscaled) row with it.

    Returns:
        ``(out [rows, hidden] in x.dtype, dot [rows] fp32 or None)``; bit-identical to the XLA entry for ``out``.
    """
    if row_gather_unsupported_reason(x, indices, scale, dot_with) is not None:
        return moe_row_gather_xla(x, indices, scale, dot_with)
    return gather_rows(x, indices, scale, dot_with)


@kernel_registry.register("moe_row_gather_sum", Platform.PALLAS, Backend.TPU, priority=10)
def moe_row_gather_sum(
    y: Float[Array, "src_rows hidden"],
    indices: Int[Array, "tokens k"],
    weights: Float[Array, "tokens k"],
    out_dtype: jax.typing.DTypeLike | None = None,
) -> Float[Array, "tokens hidden"]:
    """``out[t] = sum_j weights[t, j] * y[indices[t, j]]`` (fp32 accumulation) by row DMA.

    Args:
        y: ``[src_rows, hidden]`` source rows.
        indices: ``[tokens, k]`` row ids (clamped to range).
        weights: ``[tokens, k]`` combine weights.
        out_dtype: Output dtype (default ``y.dtype``).

    Returns:
        ``[tokens, hidden]`` in ``out_dtype``.
    """
    if row_gather_sum_unsupported_reason(y, indices, weights, out_dtype) is not None:
        return moe_row_gather_sum_xla(y, indices, weights, out_dtype)
    return gather_sum_rows(y, indices, weights, out_dtype)


__all__ = (
    "moe_row_gather",
    "moe_row_gather_sum",
    "row_gather_sum_unsupported_reason",
    "row_gather_unsupported_reason",
)
