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

"""Pallas TPU kernel for the Sinkhorn-Knopp doubly-stochastic projection.

The matrices are tiny -- DeepSeek-V4's hyper-connections normalise a 4x4 -- so
the arithmetic is free and the entire cost is dispatch. Written as a Python
loop, the iterations unroll into ~2 reductions plus 2 divides each, and every
reduction is a fusion boundary XLA cannot cross, so 20 iterations become ~117
device ops for a few hundred elements. V4 pays that twice per layer across 43
layers: ~3,400 dispatches per decode step.

Measured on v5p-8 (DeepSeek-V4-Flash A16W4, cc32): the projection cost 6.45 ms
of a 35.35 ms decode step (18%), and moving it into this kernel returned +19.1%
end to end, capturing 86% of the ablation ceiling with byte-identical output.

Two structural details, both forced by measurement rather than chosen:

* The iterations run under ``fori_loop`` rather than unrolled, so exactly one
  buffer is live instead of one per iteration.
* The leading dims are flattened and walked by a grid. Mosaic pads the two
  trailing dims out to a full (8, 128) tile, so a 4x4 matrix occupies 4 KB and a
  prefill-length input becomes megabytes per buffer -- enough to fail with
  ``CompileTimeScopedVmemOom`` no matter how few buffers are live. Blocking the
  rows is what lets one kernel serve both decode and prefill shapes.
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


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _sinkhorn_fused(matrix, n_iters: int, eps: float):
    """Forward through Pallas, backward through the reference."""
    return _sinkhorn_pallas_call(matrix, n_iters, eps)


def _sinkhorn_fused_fwd(matrix, n_iters: int, eps: float):
    return _sinkhorn_pallas_call(matrix, n_iters, eps), (matrix,)


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
