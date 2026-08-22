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

"""Pallas TPU top-k: blockwise candidate superset, then an exact small top-k.

A full sort over a vocab-scale axis is wasted work when only ``k`` elements
survive. This splits the reduction axis into ``num_blocks`` lanes and keeps the
top ``m`` of each lane, giving a candidate superset of ``num_blocks * m``
elements. When ``m >= k`` the superset provably contains the global top-k: any
element of the global top-k that were missing would need ``m`` strictly greater
elements inside its own block, and ``m >= k`` of them cannot all be outside the
top-k.

The kernel emits the superset (values and their original indices); the caller
runs an exact ``jax.lax.top_k`` over ``num_blocks * m`` instead of the full
axis. With ``m = k`` this is exact by construction -- no fallback and no
data-dependent re-run, which keeps the whole thing one static jaxpr.

The reduction is a repeated max-and-mask ("sinking") rather than a sort: ``m``
passes of max + one-hot suppression. That is O(m) passes over VMEM-resident
data and avoids Mosaic's poor lowering of a full sort on the lane axis.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ejkernel.callib import ejit

#: Mosaic streams operands double-buffered, so every resident array counts
#: twice. The kernel holds the upcast input, the running masked copy, and a
#: lane iota -- roughly 5 f32-sized arrays of the row block at once.
_VMEM_BUDGET_BYTES = 24 * 1024 * 1024
_BYTES_PER_ELEM_RESIDENT = 5 * 4


def _pick_block_rows(rows: int, width: int) -> int:
    """Largest row block that keeps the working set inside VMEM.

    Defaulting to "all rows" put a 32 x 129280 f32 row set plus its temporaries
    in VMEM and the compiler refused at 79 MB. Rows are independent here, so
    blocking them is free apart from grid steps.

    Args:
        rows: Total rows.
        width: Reduction width.

    Returns:
        Rows per grid step, dividing ``rows``.
    """
    per_row = width * _BYTES_PER_ELEM_RESIDENT * 2
    block = max(1, min(rows, _VMEM_BUDGET_BYTES // max(per_row, 1)))
    while block > 1 and rows % block:
        block -= 1
    return block


def _topk_superset_kernel(x_ref, val_ref, idx_ref, *, m: int, num_blocks: int, block_width: int):
    """Keep the top ``m`` of each block.

    Args:
        x_ref: Input rows ``[block_rows, num_blocks * block_width]``.
        val_ref: Output values ``[block_rows, num_blocks, m]``.
        idx_ref: Output indices ``[block_rows, num_blocks, m]`` (int32).
        m: Candidates kept per block.
        num_blocks: Blocks along the reduction axis.
        block_width: Elements per block.
    """
    rows = x_ref.shape[0]
    x = x_ref[...].astype(jnp.float32).reshape(rows, num_blocks, block_width)
    neg_inf = jnp.finfo(jnp.float32).min

    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, num_blocks, block_width), 2)
    running = x
    for slot in range(m):
        best = jnp.max(running, axis=-1, keepdims=True)
        # Lowest lane index wins a tie, matching jax.lax.top_k.
        is_best = running == best
        first = jnp.min(jnp.where(is_best, lane, block_width), axis=-1, keepdims=True)
        val_ref[:, :, slot] = best[:, :, 0].astype(val_ref.dtype)
        idx_ref[:, :, slot] = first[:, :, 0].astype(jnp.int32)
        running = jnp.where(lane == first, neg_inf, running)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _topk_superset(
    operand: Array,
    k: int,
    m: int | None = None,
    num_blocks: int | None = None,
    block_rows: int | None = None,
) -> tuple[Array, Array]:
    """Differentiable wrapper around the Pallas forward.

    A ``pallas_call`` has no transpose rule, so without this the kernel would
    die under ``grad`` -- which matters because MoE router scores ARE trained
    through their top-k values.
    """
    return _topk_superset_fwd_impl(operand, k, m, num_blocks, block_rows)


def _topk_superset_fwd(operand, k, m, num_blocks, block_rows):
    values, indices = _topk_superset_fwd_impl(operand, k, m, num_blocks, block_rows)
    # Residuals must be JAX types: a numpy dtype object is not one. The value
    # cotangent already carries the operand's dtype, so shape alone suffices.
    return (values, indices), (indices, operand.shape)


def _topk_superset_bwd(k, m, num_blocks, block_rows, res, cotangents):
    """Scatter value-cotangents back to the selected positions.

    Same rule ``jax.lax.top_k`` transposes to: gradient flows only to the
    elements that were selected, and the integer index output carries none.
    Indices within a row are distinct, so scatter-add and scatter agree.
    """
    del k, m, num_blocks, block_rows
    indices, shape = res
    dvalues = cotangents[0]
    grad = jnp.zeros(shape, dvalues.dtype)
    row_ids = jnp.arange(shape[0], dtype=indices.dtype)[:, None]
    grad = grad.at[row_ids, indices].add(dvalues)
    return (grad,)


_topk_superset.defvjp(_topk_superset_fwd, _topk_superset_bwd)


@ejit(static_argnames=["k", "m", "num_blocks", "block_rows"])
def topk_superset_tpu(
    operand: Array,
    k: int,
    m: int | None = None,
    num_blocks: int | None = None,
    block_rows: int | None = None,
) -> tuple[Array, Array]:
    """Exact top-k over the last axis via a blockwise superset.

    Args:
        operand: 2-D ``[rows, width]``; the reduction is over ``width``.
        k: Number of results (static).
        m: Candidates kept per block; defaults to ``k`` (exact by construction).
        num_blocks: Blocks along ``width``; defaults to a lane-aligned split.
        block_rows: Rows per grid step; defaults to all rows.

    Returns:
        ``(values, indices)`` each ``[rows, k]``, matching ``jax.lax.top_k``.
    """
    return _topk_superset(operand, k, m, num_blocks, block_rows)


def _topk_superset_fwd_impl(
    operand: Array,
    k: int,
    m: int | None = None,
    num_blocks: int | None = None,
    block_rows: int | None = None,
) -> tuple[Array, Array]:
    """Forward body: blockwise superset then an exact top-k over it."""
    rows, width = operand.shape
    m = int(k if m is None else m)
    if num_blocks is None:
        # Lane-aligned blocks, and never so many that the superset exceeds the
        # axis we are trying to avoid scanning.
        num_blocks = max(1, min(width // 128, max(1, width // (8 * max(m, 1)))))
        while num_blocks > 1 and width % num_blocks:
            num_blocks -= 1
    block_width = width // num_blocks
    block_rows = int(_pick_block_rows(rows, width) if block_rows is None else block_rows)

    out_shape = (
        jax.ShapeDtypeStruct((rows, num_blocks, m), jnp.float32),
        jax.ShapeDtypeStruct((rows, num_blocks, m), jnp.int32),
    )
    vals, idxs = pl.pallas_call(
        functools.partial(_topk_superset_kernel, m=m, num_blocks=num_blocks, block_width=block_width),
        grid=(rows // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, width), lambda i: (i, 0))],
        out_specs=(
            pl.BlockSpec((block_rows, num_blocks, m), lambda i: (i, 0, 0)),
            pl.BlockSpec((block_rows, num_blocks, m), lambda i: (i, 0, 0)),
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=100 * 1024 * 1024,
        ),
    )(operand)

    # Block-local lane -> global index, then one exact top-k over the superset.
    offsets = (jnp.arange(num_blocks, dtype=jnp.int32) * block_width)[None, :, None]
    flat_vals = vals.reshape(rows, num_blocks * m)
    flat_idxs = (idxs + offsets).reshape(rows, num_blocks * m)

    best_vals, best_pos = jax.lax.top_k(flat_vals, k)
    best_idxs = jnp.take_along_axis(flat_idxs, best_pos, axis=-1)
    return best_vals.astype(operand.dtype), best_idxs
