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

"""Pallas TPU kernels for the Sinkhorn-Knopp doubly-stochastic projection.

The matrices are tiny -- DeepSeek-V4's hyper-connections normalise a 4x4 -- so
the arithmetic is free and the entire cost is dispatch. Written as a Python
loop, the iterations unroll into ~2 reductions plus 2 divides each, and every
reduction is a fusion boundary XLA cannot cross, so 20 iterations become ~117
device ops for a few hundred elements. V4 pays that twice per layer across 43
layers: ~3,400 dispatches per decode step.

Measured on v5p-8 (DeepSeek-V4-Flash A16W4, cc32): the projection cost 6.45 ms
of a 35.35 ms decode step (18%), and moving it into this kernel returned +19.1%
end to end, capturing 86% of the ablation ceiling with byte-identical output.

Two kernels share one autodiff rule (Pallas forward, XLA-reference backward):

* Lane-dense, for fp32 matrices of at most 16 entries (every 4x4 mixer). The
  input is transposed to ``[rows * cols, R, 128]``, so entry ``(i, j)`` of up
  to ``128 * R`` matrices is one slab with the matrices on sublanes and lanes.
  Every row or column sum is then an elementwise add across slabs, with no
  cross-lane or cross-sublane reduction, and one vector op serves up to 1024
  matrices instead of one padded tile per matrix. The adds follow the tiled
  kernel's ``jnp.sum`` order (a halving tree for the column sums over
  sublanes, adjacent pairs for the row sums over lanes; on TPU v5p these are
  the only two of the 15 four-term orders that reproduce it bit for bit), with
  the same ``eps`` and true division, so the two forwards are bit-identical.
  Measured on v5p (one TensorCore, chained calls): f32[128, 4, 4] 18.7 -> 2.0
  us and f32[2048, 4, 4] 140.2 -> 2.2 us per call.
* Tiled, for everything else: one padded tile per matrix. Two of its details
  were forced by measurement. The iterations run under ``fori_loop`` rather
  than unrolled, so exactly one buffer is live instead of one per iteration.
  And the leading dims are flattened and walked by a grid: Mosaic pads the two
  trailing dims out to a full (8, 128) tile, so a 4x4 matrix occupies 4 KB and
  a prefill-length input becomes megabytes per buffer -- enough to fail with
  ``CompileTimeScopedVmemOom`` no matter how few buffers are live.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ejkernel.callib import ejit

_VMEM_BUDGET_BYTES = 4 * 1024 * 1024
_LANES = 128
# Slab rows (128 matrices each) per lane-dense grid step: one (8, 128) fp32 tile per matrix entry.
_SLAB_BLOCK_ROWS = 8
# fp32 matrices with at most this many entries take the lane-dense kernel (a 4x4 has 16).
_SLAB_MAX_ENTRIES = 16


def _round_up(x: int, multiple: int) -> int:
    """Round ``x`` up to the nearest multiple of ``multiple``."""
    return ((x + multiple - 1) // multiple) * multiple


def _pick_block_rows(n_rows: int, rows: int, cols: int, itemsize: int) -> int:
    """Choose how many matrices one grid step handles.

    Mosaic materialises each matrix at its padded tile size -- sublanes rounded
    to 8, lanes to 128 -- so a 4x4 occupies 4 KB, not 64 bytes. Blocks are sized
    against that padded footprint, doubled for the input/output pair that Mosaic
    keeps resident while it prefetches the next step.

    Args:
        n_rows: Total matrices after flattening the leading dims.
        rows: Matrix rows.
        cols: Matrix columns.
        itemsize: Bytes per element.

    Returns:
        Matrices per grid step, at least 1 and at most ``n_rows``.
    """
    padded = _round_up(rows, 8) * _round_up(cols, 128) * itemsize
    per_row = 2 * padded
    block = max(1, min(n_rows, _VMEM_BUDGET_BYTES // max(per_row, 1)))
    # Prefer a block that divides the work evenly; otherwise the tail is padded.
    while block > 1 and n_rows % block:
        block -= 1
    return block


def _sinkhorn_kernel(x_ref, o_ref, *, n_iters: int, eps: float):
    """Alternate column/row normalisation in place, one live buffer.

    Args:
        x_ref: Input block ``[..., rows, cols]``.
        o_ref: Output block, written once at the end.
        n_iters: Iteration count.
        eps: Denominator floor.
    """

    def body(_, m):
        m = m / (jnp.sum(m, axis=-1, keepdims=True) + eps)
        return m / (jnp.sum(m, axis=-2, keepdims=True) + eps)

    m = x_ref[...]
    m = m / (jnp.sum(m, axis=-2, keepdims=True) + eps)
    m = jax.lax.fori_loop(0, n_iters - 1, body, m)
    o_ref[...] = m


def _padded_leaves(n: int) -> list:
    """Leaves ``0..n-1`` followed by ``None`` padding up to the next power of two."""
    size = 1
    while size < n:
        size *= 2
    return list(range(n)) + [None] * (size - n)


def _join(a, b):
    """Tree node ``(a, b)``; a ``None`` side (an exact-zero padding slot) drops out of the sum."""
    return a if b is None else (b if a is None else (a, b))


def _halving_tree(n: int):
    """Summation tree of a halving reduction (strides ``n/2, ..., 2, 1``) over ``n`` leaves.

    Leaves are ints and internal nodes ``(left, right)`` pairs; four leaves give
    ``((0, 2), (1, 3))``. This is the order of the tiled kernel's ``jnp.sum``
    over the sublane axis (``axis=-2``, the column sums) on TPU v5p, found by
    bit-comparing all 15 four-leaf trees against it. Padding slots are zeros in
    the tiled kernel and absent here; adding an exact zero never changes a sum.

    Args:
        n: Number of leaves.

    Returns:
        The tree, or the single leaf index when ``n == 1``.
    """
    nodes = _padded_leaves(n)
    while len(nodes) > 1:
        half = len(nodes) // 2
        nodes = [_join(nodes[i], nodes[i + half]) for i in range(half)]
    return nodes[0]


def _pairwise_tree(n: int):
    """Summation tree of an adjacent-pairs reduction (strides ``1, 2, ..., n/2``) over ``n`` leaves.

    Four leaves give ``((0, 1), (2, 3))``: the order of the tiled kernel's
    ``jnp.sum`` over the lane axis (``axis=-1``, the row sums) on TPU v5p, found
    the same way as :func:`_halving_tree`.

    Args:
        n: Number of leaves.

    Returns:
        The tree, or the single leaf index when ``n == 1``.
    """
    nodes = _padded_leaves(n)
    while len(nodes) > 1:
        nodes = [_join(nodes[i], nodes[i + 1]) for i in range(0, len(nodes), 2)]
    return nodes[0]


def _tree_sum(tree, values):
    """Sum ``values`` in the exact order of ``tree`` (from :func:`_halving_tree` or :func:`_pairwise_tree`)."""
    if isinstance(tree, int):
        return values[tree]
    left, right = tree
    return _tree_sum(left, values) + _tree_sum(right, values)


def _slab_kernel(x_ref, o_ref, *, rows: int, cols: int, n_iters: int, eps: float, row_tree, col_tree):
    """Lane-dense Sinkhorn over one ``[rows * cols, block_rows, 128]`` block.

    Entry ``(i, j)`` of every matrix in the block is the slab ``x_ref[i * cols + j]``, so a row or column sum is an
    elementwise add of whole slabs, in the order of ``row_tree`` / ``col_tree``. The op sequence is the tiled
    kernel's: column normalisation, then ``n_iters - 1`` (row, column) pairs, each a sum plus ``eps`` and a true
    divide.

    Args:
        x_ref: Input block ``[rows * cols, block_rows, 128]``.
        o_ref: Output block, same shape, written once at the end.
        rows: Matrix rows.
        cols: Matrix columns.
        n_iters: Iteration count.
        eps: Denominator floor.
        row_tree: Summation tree of the row sums (over ``cols`` leaves).
        col_tree: Summation tree of the column sums (over ``rows`` leaves).
    """

    def col_norm(m):
        out = list(m)
        for j in range(cols):
            s = _tree_sum(col_tree, [m[i * cols + j] for i in range(rows)]) + eps
            for i in range(rows):
                out[i * cols + j] = m[i * cols + j] / s
        return tuple(out)

    def row_norm(m):
        out = list(m)
        for i in range(rows):
            s = _tree_sum(row_tree, [m[i * cols + j] for j in range(cols)]) + eps
            for j in range(cols):
                out[i * cols + j] = m[i * cols + j] / s
        return tuple(out)

    m = col_norm(tuple(x_ref[k] for k in range(rows * cols)))
    m = jax.lax.fori_loop(0, n_iters - 1, lambda _, c: col_norm(row_norm(c)), m)
    for k in range(rows * cols):
        o_ref[k] = m[k]


def _slab_layout(n: int) -> tuple[int, int]:
    """Slab rows and rows per grid step for ``n`` matrices.

    Up to 1024 matrices take one block covering every row (a block equal to the
    full dim needs no 8-row alignment). Beyond that, the rows are padded to a
    multiple of 8 and walked 8 at a time on a parallel grid.

    Args:
        n: Number of matrices.

    Returns:
        ``(total_rows, block_rows)`` of the ``[rows * cols, total_rows, 128]`` slab.
    """
    r = -(-n // _LANES)
    if r <= _SLAB_BLOCK_ROWS:
        return r, r
    return _round_up(r, _SLAB_BLOCK_ROWS), _SLAB_BLOCK_ROWS


def _slab_pallas(slab, rows: int, cols: int, n_iters: int, eps: float, row_tree=None, col_tree=None):
    """Launch :func:`_slab_kernel` over a ``[rows * cols, total_rows, 128]`` slab.

    Args:
        slab: Lane-dense matrices, ``total_rows`` as produced by :func:`_slab_layout`.
        rows: Matrix rows.
        cols: Matrix columns.
        n_iters: Iteration count (static).
        eps: Denominator floor.
        row_tree: Row-sum order override (default :func:`_pairwise_tree`).
        col_tree: Column-sum order override (default :func:`_halving_tree`).

    Returns:
        The normalised slab, same shape and dtype.

    Raises:
        ValueError: ``total_rows`` is not a whole number of blocks.
    """
    entries, total_rows, lanes = slab.shape
    _, block_rows = _slab_layout(total_rows * lanes)
    block_rows = min(block_rows, total_rows)
    if total_rows % block_rows:
        raise ValueError(f"slab rows {total_rows} must be a multiple of the block rows {block_rows}")
    kernel = functools.partial(
        _slab_kernel,
        rows=rows,
        cols=cols,
        n_iters=n_iters,
        eps=float(eps),
        row_tree=_pairwise_tree(cols) if row_tree is None else row_tree,
        col_tree=_halving_tree(rows) if col_tree is None else col_tree,
    )
    spec = pl.BlockSpec((entries, block_rows, lanes), lambda i: (0, i, 0))
    return pl.pallas_call(
        kernel,
        grid=(total_rows // block_rows,),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(slab.shape, slab.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(slab)


def _sinkhorn_slab_call(matrix, n_iters: int, eps: float, row_tree=None, col_tree=None):
    """Lane-dense launch: ``[batch, seq, rows, cols]`` -> slab -> :func:`_slab_kernel` -> back.

    Padding matrices are all ones (finite throughout) and are sliced off.

    Args:
        matrix: Strictly positive matrices ``[batch, seq, rows, cols]``.
        n_iters: Iteration count (static).
        eps: Denominator floor.
        row_tree: Row-sum order override (default :func:`_pairwise_tree`).
        col_tree: Column-sum order override (default :func:`_halving_tree`).

    Returns:
        Normalised matrices, same shape and dtype.
    """
    batch, seq, rows, cols = matrix.shape
    n = batch * seq
    if n == 0:
        return matrix
    entries = rows * cols
    total_rows, _ = _slab_layout(n)
    slab = matrix.reshape(n, entries).T
    pad = total_rows * _LANES - n
    if pad:
        slab = jnp.pad(slab, ((0, 0), (0, pad)), constant_values=1.0)
    out = _slab_pallas(slab.reshape(entries, total_rows, _LANES), rows, cols, n_iters, eps, row_tree, col_tree)
    out = out.reshape(entries, total_rows * _LANES)
    if pad:
        out = out[:, :n]
    return out.T.reshape(batch, seq, rows, cols)


def _sinkhorn_reference(matrix, n_iters: int, eps: float):
    """The XLA loop, kept here to supply the backward pass.

    The fused kernel is forward-only -- a Mosaic call carries no VJP -- so
    reverse-mode differentiates this instead. Both compute the same function, so
    the gradient is exact rather than an approximation of the fused path.
    """
    matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        matrix = matrix / (jnp.sum(matrix, axis=-1, keepdims=True) + eps)
        matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    return matrix


def _use_slab(matrix) -> bool:
    """Whether ``matrix`` takes the lane-dense kernel: fp32 with at most ``_SLAB_MAX_ENTRIES`` entries per matrix."""
    rows, cols = matrix.shape[-2:]
    return matrix.dtype == jnp.float32 and rows * cols <= _SLAB_MAX_ENTRIES


def _sinkhorn_forward(matrix, n_iters: int, eps: float):
    """Forward kernel choice: lane-dense for small fp32 matrices, tiled otherwise (bit-identical on TPU v5p)."""
    if _use_slab(matrix):
        return _sinkhorn_slab_call(matrix, n_iters, eps)
    return _sinkhorn_pallas_call(matrix, n_iters, eps)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _sinkhorn_fused(matrix, n_iters: int, eps: float):
    """Forward through Pallas, backward through the reference."""
    return _sinkhorn_forward(matrix, n_iters, eps)


def _sinkhorn_fused_fwd(matrix, n_iters: int, eps: float):
    return _sinkhorn_forward(matrix, n_iters, eps), (matrix,)


def _sinkhorn_fused_bwd(n_iters: int, eps: float, res, cotangent):
    (matrix,) = res
    _, vjp = jax.vjp(lambda m: _sinkhorn_reference(m, n_iters, eps), matrix)
    return vjp(cotangent)


_sinkhorn_fused.defvjp(_sinkhorn_fused_fwd, _sinkhorn_fused_bwd)


@ejit(static_argnames=["n_iters", "eps"])
def sinkhorn_knopp_tpu(
    matrix: Float[Array, "batch seq rows cols"],
    n_iters: int = 20,
    eps: float = 1e-6,
) -> Float[Array, "batch seq rows cols"]:
    """Sinkhorn-Knopp projection in a single Pallas program.

    fp32 matrices of at most 16 entries run the lane-dense kernel, everything
    else the tiled one; both forwards are bit-identical, and the backward is the
    VJP of the XLA reference loop.

    Args:
        matrix: Strictly positive matrices ``[batch, seq, rows, cols]``.
        n_iters: Iterations (static).
        eps: Denominator floor.

    Returns:
        Normalised matrices, same shape and dtype.
    """
    return _sinkhorn_fused(matrix, n_iters, float(eps))


def _sinkhorn_pallas_call(matrix, n_iters: int, eps: float):
    """Grid-blocked Pallas launch (no autodiff rule; see ``_sinkhorn_fused``)."""
    batch, seq, rows, cols = matrix.shape
    flat = matrix.reshape(batch * seq, rows, cols)
    n_rows = flat.shape[0]

    block = _pick_block_rows(n_rows, rows, cols, matrix.dtype.itemsize)
    pad = (-n_rows) % block
    if pad:
        flat = jnp.pad(flat, ((0, pad), (0, 0), (0, 0)), constant_values=1.0)

    out = pl.pallas_call(
        functools.partial(_sinkhorn_kernel, n_iters=n_iters, eps=float(eps)),
        grid=(flat.shape[0] // block,),
        in_specs=[pl.BlockSpec((block, rows, cols), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((block, rows, cols), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(flat)
    if pad:
        out = out[:n_rows]
    return out.reshape(batch, seq, rows, cols)
