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

"""TPU Pallas fused KL-divergence kernels using DMA semaphores.

The replicated-vocab path streams teacher and student logits through VMEM to
compute row-wise KL without materializing either softmax in HBM. The TP-vocab
path splits the work into two Pallas phases: local softmax statistics, followed
by local KL mass using globally merged LSEs. The custom VJP returns gradients
only for student logits; teacher logits are treated as detached distillation
targets.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.kernels._xla.fused_kl_divergence._xla_impl_fwd import fused_kl_divergence as _xla_fused_kl

from ._pallas_impl_bwd import _kl_bwd_pallas


def _default_block_v(vocab_size: int) -> int:
    """Choose the default TPU vocab tile width for KL forward kernels."""
    if vocab_size <= 256:
        return 256
    if vocab_size <= 1024:
        return 1024
    if vocab_size <= 4096:
        return 4096
    if vocab_size <= 16384:
        return 4096
    if vocab_size <= 65536:
        return 4096
    return 4096


def _default_block_m() -> int:
    """Return the row tile size used by the sparse-row TPU KL kernels."""
    return 256


def _pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a rank-2 tensor along rows so Pallas grid blocks are rectangular."""
    if pad_rows == 0:
        return x
    return jnp.pad(x, ((0, pad_rows), (0, 0)), constant_values=pad_value)


def _pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a rank-1 row tensor with the value expected by the caller."""
    if pad_rows == 0:
        return x
    return jnp.pad(x, (0, pad_rows), constant_values=pad_value)


def _flatten_logits(logits: jax.Array) -> tuple[jax.Array, tuple[int, ...]]:
    """Flatten leading dimensions into rows while preserving output shape."""
    if logits.ndim < 2:
        raise ValueError(f"fused_kl_divergence expects rank>=2 logits; got shape {logits.shape}")
    return logits.reshape(-1, logits.shape[-1]), logits.shape[:-1]


def _copy_two_tiles(
    student_ref,
    teacher_ref,
    s_tile_ref,
    t_tile_ref,
    sem_ref,
    row_start,
    col_start: int,
    block_m: int,
    size: int,
):
    """DMA-copy matching student and teacher vocab tiles into VMEM scratch."""
    t_copy = pltpu.make_async_copy(
        src_ref=teacher_ref.at[pl.ds(row_start, block_m), pl.ds(col_start, size)],
        dst_ref=t_tile_ref.at[pl.ds(0, block_m), pl.ds(0, size)],
        sem=sem_ref.at[0],
    )
    s_copy = pltpu.make_async_copy(
        src_ref=student_ref.at[pl.ds(row_start, block_m), pl.ds(col_start, size)],
        dst_ref=s_tile_ref.at[pl.ds(0, block_m), pl.ds(0, size)],
        sem=sem_ref.at[1],
    )
    t_copy.start()
    s_copy.start()
    t_copy.wait()
    s_copy.wait()


def _copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int):
    """DMA-copy one row-vector block from HBM into VMEM scratch."""
    copy = pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(row_start, block_m)],
        dst_ref=dst_ref.at[pl.ds(0, block_m)],
        sem=sem_ref.at[0],
    )
    copy.start()
    copy.wait()


def _kl_fwd_kernel(
    student_ref,
    teacher_ref,
    weights_ref,
    loss_ref,
    lse_t_ref,
    lse_s_ref,
    acc_ref,
    student_tile_ref,
    teacher_tile_ref,
    weight_ref,
    dma_sem_ref,
    *,
    direction: str,
    temperature: float,
    block_v: int,
    block_m: int,
):
    """Replicated-vocab KL forward kernel for one row block.

    The kernel scans teacher and student logits to compute local/global LSEs,
    then streams the vocab again to accumulate KL contribution. Rows with zero
    weight are skipped before any vocab DMA.
    """
    row_start = pl.program_id(0) * block_m
    _, vocab_size = student_ref.shape
    offsets = jnp.arange(block_v)
    row_offsets = jnp.arange(block_m)
    row_active = (row_start + row_offsets) < student_ref.shape[0]
    inv_t = 1.0 / float(temperature)
    is_reverse = direction == "reverse"
    _copy_rows_hbm_to_vmem(weights_ref, weight_ref, dma_sem_ref, row_start, block_m)
    weight = weight_ref[...].astype(jnp.float32)
    weight_abs = jnp.abs(weight)
    active = row_active & (weight != 0.0)

    loss_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)
    lse_t_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)
    lse_s_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)
    acc_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)

    @pl.when(jnp.any(active))
    def _compute_active_block():
        """Stream KL tiles only for blocks with nonzero row weights."""
        num_blocks = pl.cdiv(vocab_size, block_v)

        max_t = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        max_s = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, vocab_size - start)
            _copy_two_tiles(
                student_ref,
                teacher_ref,
                student_tile_ref,
                teacher_tile_ref,
                dma_sem_ref,
                row_start,
                start,
                block_m,
                size,
            )
            in_vocab = offsets < size
            t_tile = teacher_tile_ref[...].astype(jnp.float32) * inv_t
            s_tile = student_tile_ref[...].astype(jnp.float32) * inv_t
            max_t = jnp.maximum(max_t, jnp.max(jnp.where(in_vocab[None, :], t_tile, -jnp.inf), axis=1))
            max_s = jnp.maximum(max_s, jnp.max(jnp.where(in_vocab[None, :], s_tile, -jnp.inf), axis=1))

        sum_t = jnp.zeros((block_m,), dtype=jnp.float32)
        sum_s = jnp.zeros((block_m,), dtype=jnp.float32)
        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, vocab_size - start)
            _copy_two_tiles(
                student_ref,
                teacher_ref,
                student_tile_ref,
                teacher_tile_ref,
                dma_sem_ref,
                row_start,
                start,
                block_m,
                size,
            )
            in_vocab = offsets < size
            t_tile = teacher_tile_ref[...].astype(jnp.float32) * inv_t
            s_tile = student_tile_ref[...].astype(jnp.float32) * inv_t
            sum_t = sum_t + jnp.sum(
                jnp.where(in_vocab[None, :], jnp.exp(t_tile - max_t[:, None]), 0.0),
                axis=1,
            )
            sum_s = sum_s + jnp.sum(
                jnp.where(in_vocab[None, :], jnp.exp(s_tile - max_s[:, None]), 0.0),
                axis=1,
            )

        lse_t = jnp.log(sum_t) + max_t
        lse_s = jnp.log(sum_s) + max_s
        acc = jnp.zeros((block_m,), dtype=jnp.float32)
        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, vocab_size - start)
            _copy_two_tiles(
                student_ref,
                teacher_ref,
                student_tile_ref,
                teacher_tile_ref,
                dma_sem_ref,
                row_start,
                start,
                block_m,
                size,
            )
            in_vocab = offsets < size
            t_tile = teacher_tile_ref[...].astype(jnp.float32) * inv_t
            s_tile = student_tile_ref[...].astype(jnp.float32) * inv_t
            p_t = jnp.exp(t_tile - lse_t[:, None])
            p_s = jnp.exp(s_tile - lse_s[:, None])
            contrib = jnp.where(is_reverse, p_s * (s_tile - t_tile), p_t * (t_tile - s_tile))
            acc = acc + jnp.sum(jnp.where(in_vocab[None, :], contrib, 0.0), axis=1)

        per_row = jnp.where(is_reverse, acc + lse_t - lse_s, acc + lse_s - lse_t)
        loss_ref[...] = jnp.where(active, weight_abs * per_row, 0.0).astype(jnp.float32)
        lse_t_ref[...] = jnp.where(row_active, lse_t, 0.0).astype(jnp.float32)
        lse_s_ref[...] = jnp.where(row_active, lse_s, 0.0).astype(jnp.float32)
        acc_ref[...] = jnp.where(row_active, acc, 0.0).astype(jnp.float32)


def _kl_fwd_pallas(student_2d, teacher_2d, weights_1d, *, direction, temperature, block_v, block_m):
    """Launch replicated-vocab TPU Pallas KL forward and trim padded rows."""
    n_rows = student_2d.shape[0]
    n_rows_pad = pl.cdiv(n_rows, int(block_m)) * int(block_m)
    pad_rows = n_rows_pad - n_rows
    student_pad = _pad_rows_2d(student_2d, pad_rows)
    teacher_pad = _pad_rows_2d(teacher_2d, pad_rows)
    weights_pad = _pad_rows_1d(weights_1d, pad_rows)
    loss, lse_t, lse_s, acc = pl.pallas_call(
        functools.partial(
            _kl_fwd_kernel,
            direction=direction,
            temperature=float(temperature),
            block_v=int(block_v),
            block_m=int(block_m),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
            ],
            scratch_shapes=[
                pltpu.VMEM((int(block_m), int(block_v)), student_2d.dtype),
                pltpu.VMEM((int(block_m), int(block_v)), teacher_2d.dtype),
                pltpu.VMEM((int(block_m),), weights_1d.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            grid=(n_rows_pad // int(block_m),),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        out_shape=[
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
        ],
    )(student_pad, teacher_pad, weights_pad.astype(jnp.float32))
    return loss[:n_rows], lse_t[:n_rows], lse_s[:n_rows], acc[:n_rows]


def _kl_tp_lse_stats_kernel(
    student_ref,
    teacher_ref,
    weights_ref,
    max_t_ref,
    max_s_ref,
    sum_t_ref,
    sum_s_ref,
    student_tile_ref,
    teacher_tile_ref,
    weight_ref,
    dma_sem_ref,
    *,
    temperature: float,
    block_v: int,
    block_m: int,
):
    """Compute local teacher/student softmax stats for a TP vocab shard.

    This kernel only sees the local vocabulary slice. It returns per-row local
    max and sum-exp values for teacher and student logits; the caller rescales
    and merges them across ``tp`` to form global ``lse_t`` and ``lse_s``.
    """
    row_start = pl.program_id(0) * block_m
    _, vocab_size = student_ref.shape
    offsets = jnp.arange(block_v)
    row_offsets = jnp.arange(block_m)
    row_active = (row_start + row_offsets) < student_ref.shape[0]
    inv_t = 1.0 / float(temperature)
    _copy_rows_hbm_to_vmem(weights_ref, weight_ref, dma_sem_ref, row_start, block_m)
    weight = weight_ref[...].astype(jnp.float32)
    active = row_active & (weight != 0.0)

    max_t_ref[...] = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
    max_s_ref[...] = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
    sum_t_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)
    sum_s_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)

    @pl.when(jnp.any(active))
    def _compute_active_block():
        """Compute local teacher/student LSE pieces for active TP rows."""
        num_blocks = pl.cdiv(vocab_size, block_v)
        max_t = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        max_s = jnp.full((block_m,), -jnp.inf, dtype=jnp.float32)
        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, vocab_size - start)
            _copy_two_tiles(
                student_ref,
                teacher_ref,
                student_tile_ref,
                teacher_tile_ref,
                dma_sem_ref,
                row_start,
                start,
                block_m,
                size,
            )
            in_vocab = offsets < size
            t_tile = teacher_tile_ref[...].astype(jnp.float32) * inv_t
            s_tile = student_tile_ref[...].astype(jnp.float32) * inv_t
            max_t = jnp.maximum(max_t, jnp.max(jnp.where(in_vocab[None, :], t_tile, -jnp.inf), axis=1))
            max_s = jnp.maximum(max_s, jnp.max(jnp.where(in_vocab[None, :], s_tile, -jnp.inf), axis=1))

        sum_t = jnp.zeros((block_m,), dtype=jnp.float32)
        sum_s = jnp.zeros((block_m,), dtype=jnp.float32)
        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, vocab_size - start)
            _copy_two_tiles(
                student_ref,
                teacher_ref,
                student_tile_ref,
                teacher_tile_ref,
                dma_sem_ref,
                row_start,
                start,
                block_m,
                size,
            )
            in_vocab = offsets < size
            t_tile = teacher_tile_ref[...].astype(jnp.float32) * inv_t
            s_tile = student_tile_ref[...].astype(jnp.float32) * inv_t
            sum_t = sum_t + jnp.sum(jnp.where(in_vocab[None, :], jnp.exp(t_tile - max_t[:, None]), 0.0), axis=1)
            sum_s = sum_s + jnp.sum(jnp.where(in_vocab[None, :], jnp.exp(s_tile - max_s[:, None]), 0.0), axis=1)

        max_t_ref[...] = jnp.where(active, max_t, -jnp.inf).astype(jnp.float32)
        max_s_ref[...] = jnp.where(active, max_s, -jnp.inf).astype(jnp.float32)
        sum_t_ref[...] = jnp.where(active, sum_t, 0.0).astype(jnp.float32)
        sum_s_ref[...] = jnp.where(active, sum_s, 0.0).astype(jnp.float32)


def _kl_tp_acc_kernel(
    student_ref,
    teacher_ref,
    weights_ref,
    lse_t_ref,
    lse_s_ref,
    acc_ref,
    student_tile_ref,
    teacher_tile_ref,
    weight_ref,
    lse_t_scalar_ref,
    lse_s_scalar_ref,
    dma_sem_ref,
    *,
    direction: str,
    temperature: float,
    block_v: int,
    block_m: int,
):
    """Compute the local KL contribution using global teacher/student LSEs.

    The ``lse_t_ref`` and ``lse_s_ref`` inputs are already merged across the
    vocab-parallel axis. This kernel streams the local vocab shard once more and
    accumulates the shard-local KL term; the caller ``psum``s it over ``tp``.
    """
    row_start = pl.program_id(0) * block_m
    _, vocab_size = student_ref.shape
    offsets = jnp.arange(block_v)
    row_offsets = jnp.arange(block_m)
    row_active = (row_start + row_offsets) < student_ref.shape[0]
    inv_t = 1.0 / float(temperature)
    is_reverse = direction == "reverse"
    _copy_rows_hbm_to_vmem(weights_ref, weight_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(lse_t_ref, lse_t_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(lse_s_ref, lse_s_scalar_ref, dma_sem_ref, row_start, block_m)
    weight = weight_ref[...].astype(jnp.float32)
    active = row_active & (weight != 0.0)
    acc_ref[...] = jnp.zeros((block_m,), dtype=jnp.float32)

    @pl.when(jnp.any(active))
    def _compute_active_block():
        """Accumulate this shard's KL contribution using global LSEs."""
        num_blocks = pl.cdiv(vocab_size, block_v)
        lse_t = lse_t_scalar_ref[...].astype(jnp.float32)
        lse_s = lse_s_scalar_ref[...].astype(jnp.float32)
        acc = jnp.zeros((block_m,), dtype=jnp.float32)
        for block_idx in range(num_blocks):
            start = block_idx * block_v
            size = min(block_v, vocab_size - start)
            _copy_two_tiles(
                student_ref,
                teacher_ref,
                student_tile_ref,
                teacher_tile_ref,
                dma_sem_ref,
                row_start,
                start,
                block_m,
                size,
            )
            in_vocab = offsets < size
            t_tile = teacher_tile_ref[...].astype(jnp.float32) * inv_t
            s_tile = student_tile_ref[...].astype(jnp.float32) * inv_t
            p_t = jnp.exp(t_tile - lse_t[:, None])
            p_s = jnp.exp(s_tile - lse_s[:, None])
            contrib = jnp.where(is_reverse, p_s * (s_tile - t_tile), p_t * (t_tile - s_tile))
            acc = acc + jnp.sum(jnp.where(in_vocab[None, :], contrib, 0.0), axis=1)
        acc_ref[...] = jnp.where(active, acc, 0.0).astype(jnp.float32)


def _kl_tp_lse_stats_pallas(student_2d, teacher_2d, weights_1d, *, temperature, block_v, block_m):
    """Launch the first TP-vocab KL phase that computes local LSE stats."""
    n_rows = student_2d.shape[0]
    n_rows_pad = pl.cdiv(n_rows, int(block_m)) * int(block_m)
    pad_rows = n_rows_pad - n_rows
    student_pad = _pad_rows_2d(student_2d, pad_rows)
    teacher_pad = _pad_rows_2d(teacher_2d, pad_rows)
    weights_pad = _pad_rows_1d(weights_1d, pad_rows)
    max_t, max_s, sum_t, sum_s = pl.pallas_call(
        functools.partial(
            _kl_tp_lse_stats_kernel,
            temperature=float(temperature),
            block_v=int(block_v),
            block_m=int(block_m),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
                pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
            ],
            scratch_shapes=[
                pltpu.VMEM((int(block_m), int(block_v)), student_2d.dtype),
                pltpu.VMEM((int(block_m), int(block_v)), teacher_2d.dtype),
                pltpu.VMEM((int(block_m),), weights_1d.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            grid=(n_rows_pad // int(block_m),),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        out_shape=[
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
            jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
        ],
    )(student_pad, teacher_pad, weights_pad.astype(jnp.float32))
    return max_t[:n_rows], max_s[:n_rows], sum_t[:n_rows], sum_s[:n_rows]


def _kl_tp_acc_pallas(student_2d, teacher_2d, weights_1d, lse_t, lse_s, *, direction, temperature, block_v, block_m):
    """Launch the second TP-vocab KL phase that computes local KL mass."""
    n_rows = student_2d.shape[0]
    n_rows_pad = pl.cdiv(n_rows, int(block_m)) * int(block_m)
    pad_rows = n_rows_pad - n_rows
    student_pad = _pad_rows_2d(student_2d, pad_rows)
    teacher_pad = _pad_rows_2d(teacher_2d, pad_rows)
    weights_pad = _pad_rows_1d(weights_1d, pad_rows)
    lse_t_pad = _pad_rows_1d(lse_t, pad_rows)
    lse_s_pad = _pad_rows_1d(lse_s, pad_rows)
    acc = pl.pallas_call(
        functools.partial(
            _kl_tp_acc_kernel,
            direction=direction,
            temperature=float(temperature),
            block_v=int(block_v),
            block_m=int(block_m),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((int(block_m),), lambda row_block: (row_block,)),
            scratch_shapes=[
                pltpu.VMEM((int(block_m), int(block_v)), student_2d.dtype),
                pltpu.VMEM((int(block_m), int(block_v)), teacher_2d.dtype),
                pltpu.VMEM((int(block_m),), weights_1d.dtype),
                pltpu.VMEM((int(block_m),), lse_t.dtype),
                pltpu.VMEM((int(block_m),), lse_s.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            grid=(n_rows_pad // int(block_m),),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        out_shape=jax.ShapeDtypeStruct((n_rows_pad,), jnp.float32),
    )(
        student_pad,
        teacher_pad,
        weights_pad.astype(jnp.float32),
        lse_t_pad.astype(jnp.float32),
        lse_s_pad.astype(jnp.float32),
    )
    return acc[:n_rows]


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _fused_kl_core_pallas(student_2d, teacher_2d, weights_1d, direction, temperature, block_v, block_m):
    """Differentiable replicated-vocab KL loss wrapper."""
    loss, _lse_t, _lse_s, _acc = _kl_fwd_pallas(
        student_2d,
        teacher_2d,
        weights_1d,
        direction=direction,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
    )
    return loss


def _kl_core_fwd(student_2d, teacher_2d, weights_1d, direction, temperature, block_v, block_m):
    """Forward rule for replicated-vocab KL, saving LSEs and KL aux."""
    loss, lse_t, lse_s, acc = _kl_fwd_pallas(
        student_2d,
        teacher_2d,
        weights_1d,
        direction=direction,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
    )
    return loss, (student_2d, teacher_2d, lse_t, lse_s, acc, weights_1d)


def _kl_core_bwd(direction, temperature, block_v, block_m, residual, dy):
    """Backward rule for replicated-vocab KL using saved softmax stats."""
    student_2d, teacher_2d, lse_t, lse_s, acc, weights_1d = residual
    dstudent = _kl_bwd_pallas(
        student_2d,
        teacher_2d,
        lse_t,
        lse_s,
        acc,
        weights_1d,
        dy,
        direction=direction,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
    )
    return dstudent, jnp.zeros_like(teacher_2d), None


_fused_kl_core_pallas.defvjp(_kl_core_fwd, _kl_core_bwd)


def _kl_tp_loss_and_aux(
    student_2d,
    teacher_2d,
    weights_1d,
    *,
    direction,
    temperature,
    block_v,
    block_m,
    vocab_parallel_axis,
):
    """Build TP-vocab KL loss from local Pallas phases and TP collectives.

    The first Pallas phase computes local teacher/student softmax stats, JAX
    collectives merge them into global LSEs, and the second Pallas phase
    computes the local KL contribution. A final ``psum`` gives the full per-row
    KL while preserving the global LSEs for backward.
    """
    local_max_t, local_max_s, local_sum_t, local_sum_s = _kl_tp_lse_stats_pallas(
        student_2d,
        teacher_2d,
        weights_1d,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
    )
    active = weights_1d != 0.0
    global_max_t = jax.lax.pmax(local_max_t, vocab_parallel_axis)
    global_max_s = jax.lax.pmax(local_max_s, vocab_parallel_axis)
    scaled_t = jnp.where(
        jnp.isfinite(local_max_t) & jnp.isfinite(global_max_t),
        local_sum_t * jnp.exp(local_max_t - global_max_t),
        0.0,
    )
    scaled_s = jnp.where(
        jnp.isfinite(local_max_s) & jnp.isfinite(global_max_s),
        local_sum_s * jnp.exp(local_max_s - global_max_s),
        0.0,
    )
    global_sum_t = jax.lax.psum(scaled_t, vocab_parallel_axis)
    global_sum_s = jax.lax.psum(scaled_s, vocab_parallel_axis)
    lse_t = jnp.where(active, jnp.log(global_sum_t) + global_max_t, 0.0)
    lse_s = jnp.where(active, jnp.log(global_sum_s) + global_max_s, 0.0)
    local_acc = _kl_tp_acc_pallas(
        student_2d,
        teacher_2d,
        weights_1d,
        lse_t,
        lse_s,
        direction=direction,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
    )
    acc = jax.lax.psum(local_acc, vocab_parallel_axis)
    per_row = jnp.where(direction == "reverse", acc + lse_t - lse_s, acc + lse_s - lse_t)
    loss = jnp.where(active, jnp.abs(weights_1d) * per_row, 0.0)
    return loss.astype(jnp.float32), lse_t.astype(jnp.float32), lse_s.astype(jnp.float32), acc.astype(jnp.float32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _fused_kl_core_pallas_tp(
    student_2d,
    teacher_2d,
    weights_1d,
    direction,
    temperature,
    block_v,
    block_m,
    vocab_parallel_axis,
):
    """Differentiable TP-vocab KL wrapper backed by Pallas kernels."""
    loss, _lse_t, _lse_s, _acc = _kl_tp_loss_and_aux(
        student_2d,
        teacher_2d,
        weights_1d,
        direction=direction,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
        vocab_parallel_axis=vocab_parallel_axis,
    )
    return loss


def _kl_core_tp_fwd(student_2d, teacher_2d, weights_1d, direction, temperature, block_v, block_m, vocab_parallel_axis):
    """Forward rule for TP-vocab KL, saving global LSEs and full-row KL aux."""
    loss, lse_t, lse_s, acc = _kl_tp_loss_and_aux(
        student_2d,
        teacher_2d,
        weights_1d,
        direction=direction,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
        vocab_parallel_axis=vocab_parallel_axis,
    )
    return loss, (student_2d, teacher_2d, lse_t, lse_s, acc, weights_1d)


def _kl_core_tp_bwd(direction, temperature, block_v, block_m, vocab_parallel_axis, residual, dy):
    """Backward rule for TP-vocab KL.

    The local Pallas backward uses global ``lse_t`` / ``lse_s`` but writes only
    this shard's student-gradient slice. The returned gradient is scaled by the
    TP axis size to undo the cotangent splitting from ``shard_map``.
    """
    student_2d, teacher_2d, lse_t, lse_s, acc, weights_1d = residual
    axis_size = jax.lax.psum(jnp.array(1, dtype=jnp.float32), vocab_parallel_axis)
    dstudent = _kl_bwd_pallas(
        student_2d,
        teacher_2d,
        lse_t,
        lse_s,
        acc,
        weights_1d,
        dy,
        direction=direction,
        temperature=temperature,
        block_v=block_v,
        block_m=block_m,
    )
    return dstudent * axis_size, jnp.zeros_like(teacher_2d), None


_fused_kl_core_pallas_tp.defvjp(_kl_core_tp_fwd, _kl_core_tp_bwd)


def fused_kl_divergence_pallas(
    student_logits,
    teacher_logits,
    weights=None,
    *,
    reduction: str = "mean",
    direction: str = "forward",
    temperature: float = 1.0,
    beta: float = 0.5,
    vocab_parallel_axis: str | None = None,
    block_v: int = 0,
    block_m: int = 0,
):
    """Run TPU Pallas fused KL divergence.

    ``direction="forward"`` and ``"reverse"`` are implemented by Pallas.
    ``direction="jsd"`` falls back to XLA. With ``vocab_parallel_axis`` set,
    inputs are local vocab shards inside ``shard_map``; TP collectives merge
    teacher/student softmax statistics and KL mass across the full vocabulary.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Invalid reduction '{reduction}'; expected one of none/sum/mean.")
    if direction not in ("forward", "reverse", "jsd"):
        raise ValueError(f"Invalid direction '{direction}'; expected one of forward/reverse/jsd.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive; got {temperature}")
    if direction == "jsd":
        return _xla_fused_kl(
            student_logits,
            teacher_logits,
            weights,
            reduction=reduction,
            direction=direction,
            temperature=temperature,
            beta=beta,
            vocab_parallel_axis=vocab_parallel_axis,
        )
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"fused_kl_divergence: shape mismatch student={student_logits.shape} vs teacher={teacher_logits.shape}"
        )
    if student_logits.dtype == jnp.float16 or teacher_logits.dtype == jnp.float16:
        raise ValueError(
            "TPU Pallas fused_kl_divergence does not support float16 inputs. "
            "Use bfloat16 or float32; Mosaic rejects f16 VMEM vector loads in this kernel."
        )

    teacher_logits = teacher_logits.astype(student_logits.dtype)
    flat_student, leading = _flatten_logits(student_logits)
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    if weights is None:
        flat_weights = jnp.ones(flat_student.shape[0], dtype=jnp.float32)
    else:
        if weights.shape != leading:
            raise ValueError(f"weights.shape={weights.shape} must equal logits.shape[:-1]={leading}")
        flat_weights = weights.reshape(-1).astype(jnp.float32)
    default_bv = _default_block_v(int(flat_student.shape[-1]))
    bv = max(default_bv, int(block_v)) if int(block_v) > 0 else default_bv
    bm = _default_block_m()

    if vocab_parallel_axis is None:
        per_row = _fused_kl_core_pallas(
            flat_student,
            flat_teacher,
            flat_weights,
            direction,
            float(temperature),
            bv,
            bm,
        )
    else:
        per_row = _fused_kl_core_pallas_tp(
            flat_student,
            flat_teacher,
            flat_weights,
            direction,
            float(temperature),
            bv,
            bm,
            vocab_parallel_axis,
        )
    if temperature != 1.0:
        per_row = per_row * (float(temperature) ** 2)
    if reduction == "none":
        return per_row.reshape(leading)
    total = jnp.sum(per_row)
    if reduction == "sum":
        return total
    denom = jnp.maximum(jnp.sum(flat_weights), 1e-8)
    return total / denom
