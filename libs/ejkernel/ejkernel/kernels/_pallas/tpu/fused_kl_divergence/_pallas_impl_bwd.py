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

"""Backward TPU Pallas kernels for fused KL divergence.

This module computes the gradient of the fused (forward or reverse) KL-divergence
distillation loss w.r.t. the *student* logits only; the teacher logits are treated
as detached targets and receive no gradient. It is the backward companion of the
forward KL kernels and is driven by their custom VJP. The gradient is streamed one
``(block_m, block_v)`` tile at a time: aligned student and teacher logit tiles are
DMA-copied into VMEM and combined with the saved per-row teacher/student
log-sum-exp, the saved per-row reverse-KL correction ``acc``, the loss weights and
the upstream cotangents. The same kernel serves the replicated and vocab-parallel
(TP) paths; in TP mode the saved LSEs and ``acc`` are already global across the
vocab-parallel axis while student/teacher logits are local shards.

Gradient form (temperature ``T``, ``inv_t = 1/T``, ``p_s``/``p_t`` softmaxed at
temperature ``T``):
    * forward KL: ``weight * dy * inv_t * (p_s - p_t)``
    * reverse KL: ``weight * dy * inv_t * p_s * ((s - t) - acc)`` where ``acc`` is the
      saved per-row ``sum_v p_s * (s - t)`` term and ``s``/``t`` are temperature-scaled
      logits.

Contents:
    * ``_pallas_out_shape`` - manual-sharding-aware output ``ShapeDtypeStruct`` builder.
    * ``_pad_rows_2d`` / ``_pad_rows_1d`` - row padding so grid blocks stay rectangular.
    * ``_copy_two_tiles`` - paired student/teacher tile DMA helper.
    * ``_copy_rows_hbm_to_vmem`` - per-row scalar DMA helper.
    * ``_kl_bwd_kernel`` - the per-tile Pallas gradient kernel.
    * ``_kl_bwd_pallas`` - host-side launcher returning the un-padded student gradient.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _pallas_out_shape(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.ShapeDtypeStruct:
    """Build the ``out_shape`` for a Pallas call that is manual-sharding aware.

    The returned ``ShapeDtypeStruct`` carries a ``ManualAxisType`` marking every
    manually sharded mesh axis as varying, so the kernel output participates
    correctly in ``shard_map``/manual-collective contexts (e.g. vocab-parallel TP).

    Args:
        shape: Logical shape of the kernel output array.
        dtype: Element dtype of the kernel output array.

    Returns:
        A ``jax.ShapeDtypeStruct`` with the given shape/dtype and a
        ``manual_axis_type`` whose ``varying`` set is the current abstract mesh's
        manual axes.
    """
    manual_axes = frozenset(jax.sharding.get_abstract_mesh().manual_axes)
    return jax.ShapeDtypeStruct(
        shape,
        dtype,
        manual_axis_type=jax.sharding.ManualAxisType(varying=manual_axes),
    )


def _pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a row-major logit/gradient matrix to a whole Pallas row block.

    Args:
        x: Rank-2 array shaped ``(n_rows, vocab)`` to pad along the leading axis.
        pad_rows: Number of zero-content rows to append; ``0`` is a no-op pass-through.
        pad_value: Constant value used for the appended rows. Defaults to ``0.0``.

    Returns:
        ``x`` unchanged when ``pad_rows == 0``, otherwise a new array shaped
        ``(n_rows + pad_rows, vocab)`` with the trailing rows set to ``pad_value``.
    """
    if pad_rows == 0:
        return x
    return jnp.pad(x, ((0, pad_rows), (0, 0)), constant_values=pad_value)


def _pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a per-row vector to match the padded logit row count.

    Args:
        x: Rank-1 array shaped ``(n_rows,)`` (e.g. an LSE, ``acc``, weights or ``dy``).
        pad_rows: Number of entries to append; ``0`` is a no-op pass-through.
        pad_value: Constant value used for the appended entries. Defaults to ``0.0``.

    Returns:
        ``x`` unchanged when ``pad_rows == 0``, otherwise a new array shaped
        ``(n_rows + pad_rows,)`` with the trailing entries set to ``pad_value``.
    """
    if pad_rows == 0:
        return x
    return jnp.pad(x, (0, pad_rows), constant_values=pad_value)


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
    """DMA-copy aligned student and teacher vocab tiles into VMEM scratch.

    Issues two concurrent async DMAs (teacher on semaphore slot ``0``, student on
    slot ``1``) that move the same ``(block_m, size)`` ``[row_start:, col_start:]``
    window from HBM into the front of the VMEM scratch tiles, then waits on both so
    the tiles are ready on return.

    Args:
        student_ref: HBM ref of student logits shaped ``(n_rows, vocab_size)``.
        teacher_ref: HBM ref of teacher logits shaped ``(n_rows, vocab_size)``.
        s_tile_ref: VMEM scratch of shape ``(block_m, block_v)`` for the student tile.
        t_tile_ref: VMEM scratch of shape ``(block_m, block_v)`` for the teacher tile.
        sem_ref: DMA semaphore ref with at least two slots (``0`` teacher, ``1`` student).
        row_start: Index of the first row of the tile.
        col_start: Index of the first vocab column of the tile.
        block_m: Number of rows in the tile.
        size: Number of active vocab columns to copy (``<= block_v``; the last tile
            may be partial).

    Returns:
        None. Both copies complete synchronously before returning.
    """
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
    """DMA-copy one contiguous row vector tile from HBM into VMEM scratch.

    Starts and immediately waits on a single async DMA that moves ``block_m``
    contiguous entries from the HBM source ref into the front of the VMEM scratch
    ref, using semaphore slot ``0`` of ``sem_ref``.

    Args:
        src_ref: HBM ref of a rank-1 per-row array (LSE, ``acc``, weights or ``dy``).
        dst_ref: VMEM scratch ref of shape ``(block_m,)`` to receive the tile.
        sem_ref: DMA semaphore ref; slot ``0`` is used for this copy.
        row_start: Index of the first row to copy from ``src_ref``.
        block_m: Number of contiguous rows in the tile.

    Returns:
        None. The copy completes synchronously before returning.
    """
    copy = pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(row_start, block_m)],
        dst_ref=dst_ref.at[pl.ds(0, block_m)],
        sem=sem_ref.at[0],
    )
    copy.start()
    copy.wait()


def _kl_bwd_kernel(
    student_ref,
    teacher_ref,
    lse_t_ref,
    lse_s_ref,
    acc_ref,
    weights_ref,
    dy_ref,
    dstudent_ref,
    student_tile_ref,
    teacher_tile_ref,
    lse_t_scalar_ref,
    lse_s_scalar_ref,
    acc_scalar_ref,
    weight_scalar_ref,
    dy_scalar_ref,
    dma_sem_ref,
    *,
    direction: str,
    temperature: float,
    block_v: int,
    block_m: int,
):
    """Write one student-gradient tile for fused KL divergence.

    Each grid step handles one ``(block_m, block_v)`` tile of the student gradient.
    Aligned student/teacher logit tiles and the per-row scalars (teacher LSE, student
    LSE, ``acc``, weight, ``dy``) are DMA-copied into VMEM scratch. Logits are scaled
    by ``inv_t = 1/temperature``, softmax probabilities ``p_s``/``p_t`` are formed from
    the saved LSEs, and the gradient is written in fp32 as ``weight * dy * inv_t``
    times ``(p_s - p_t)`` for forward KL or ``p_s * ((s - t) - acc)`` for reverse KL.
    Padding rows and zero-weight rows produce zero; vocab columns beyond ``size`` (the
    active part of the last tile) are masked to zero.

    The same kernel is used for replicated-vocab and TP-vocab paths. In TP mode
    the saved ``lse_t`` / ``lse_s`` / ``acc`` values are already global across
    the vocab-parallel axis, while ``student_ref`` and ``teacher_ref`` are local
    vocab shards.

    Args:
        student_ref: HBM ref of student logits shaped ``(n_rows, vocab_size)``.
        teacher_ref: HBM ref of teacher logits shaped ``(n_rows, vocab_size)``.
        lse_t_ref: HBM ref of per-row teacher log-sum-exp shaped ``(n_rows,)`` (fp32);
            global in TP mode.
        lse_s_ref: HBM ref of per-row student log-sum-exp shaped ``(n_rows,)`` (fp32);
            global in TP mode.
        acc_ref: HBM ref of the per-row reverse-KL correction shaped ``(n_rows,)``
            (fp32), the saved ``sum_v p_s * (s - t)``; unused for forward KL.
        weights_ref: HBM ref of per-row loss weights shaped ``(n_rows,)`` (fp32).
        dy_ref: HBM ref of per-row upstream cotangents shaped ``(n_rows,)`` (fp32).
        dstudent_ref: VMEM output ref of shape ``(block_m, block_v)`` receiving the
            student gradient tile in ``dstudent_ref.dtype``.
        student_tile_ref: VMEM scratch of shape ``(block_m, block_v)`` for student logits.
        teacher_tile_ref: VMEM scratch of shape ``(block_m, block_v)`` for teacher logits.
        lse_t_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the teacher LSE tile.
        lse_s_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the student LSE tile.
        acc_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the ``acc`` tile.
        weight_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the weight tile.
        dy_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the ``dy`` tile.
        dma_sem_ref: DMA semaphore ref with two slots (paired tile copies use both;
            per-row copies use slot ``0``).
        direction: Either ``"forward"`` or ``"reverse"``; selects the gradient formula.
        temperature: Distillation temperature ``T``; logits are divided by ``T``.
        block_v: Vocab tile width (columns) handled per grid step.
        block_m: Row tile height handled per grid step.

    Returns:
        None. Results are written in place to ``dstudent_ref``.
    """
    row_start = pl.program_id(0) * block_m
    block_idx = pl.program_id(1)
    _, vocab_size = student_ref.shape
    start = block_idx * block_v
    size = jnp.minimum(block_v, vocab_size - start)
    offsets = jnp.arange(block_v)
    row_offsets = jnp.arange(block_m)
    row_active = (row_start + row_offsets) < student_ref.shape[0]
    inv_t = 1.0 / float(temperature)
    is_reverse = direction == "reverse"

    _copy_two_tiles(
        student_ref, teacher_ref, student_tile_ref, teacher_tile_ref, dma_sem_ref, row_start, start, block_m, size
    )
    in_vocab = offsets < size
    t_tile = teacher_tile_ref[...].astype(jnp.float32) * inv_t
    s_tile = student_tile_ref[...].astype(jnp.float32) * inv_t
    _copy_rows_hbm_to_vmem(lse_t_ref, lse_t_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(lse_s_ref, lse_s_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(acc_ref, acc_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(weights_ref, weight_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(dy_ref, dy_scalar_ref, dma_sem_ref, row_start, block_m)

    weight = weight_scalar_ref[...].astype(jnp.float32)
    active = row_active & (weight != 0.0)
    factor = jnp.where(active, weight * dy_scalar_ref[...].astype(jnp.float32) * inv_t, 0.0)
    p_s = jnp.exp(s_tile - lse_s_scalar_ref[...].astype(jnp.float32)[:, None])
    p_t = jnp.exp(t_tile - lse_t_scalar_ref[...].astype(jnp.float32)[:, None])
    grad = jnp.where(
        is_reverse,
        factor[:, None] * p_s * ((s_tile - t_tile) - acc_scalar_ref[...].astype(jnp.float32)[:, None]),
        factor[:, None] * (p_s - p_t),
    )
    dstudent_ref[...] = jnp.where(in_vocab[None, :], grad, 0.0).astype(dstudent_ref.dtype)


def _kl_bwd_pallas(
    student_2d,
    teacher_2d,
    lse_t,
    lse_s,
    acc,
    weights_1d,
    dy,
    *,
    direction,
    temperature,
    block_v,
    block_m,
):
    """Launch the TPU Pallas KL backward kernel and return the student gradient.

    Caps ``block_v`` at 1024, pads the row dimension up to a whole multiple of
    ``block_m`` for both logit matrices and all per-row auxiliaries, casts the
    auxiliaries to fp32, runs :func:`_kl_bwd_kernel` over a ``(row_blocks,
    vocab_blocks)`` parallel grid, then slices the padding back off. In TP mode this
    is called once per local vocab shard; the saved LSEs/``acc`` passed in are
    already global across the vocab-parallel axis.

    Args:
        student_2d: Student logits shaped ``(n_rows, vocab_size)`` (local shard in TP mode).
        teacher_2d: Teacher logits shaped ``(n_rows, vocab_size)`` (local shard in TP mode).
        lse_t: Per-row teacher log-sum-exp shaped ``(n_rows,)``; cast to fp32.
        lse_s: Per-row student log-sum-exp shaped ``(n_rows,)``; cast to fp32.
        acc: Per-row reverse-KL correction shaped ``(n_rows,)``; cast to fp32 (unused
            for forward KL but always passed).
        weights_1d: Per-row loss weights shaped ``(n_rows,)``; cast to fp32.
        dy: Per-row upstream cotangents shaped ``(n_rows,)``; cast to fp32.
        direction: Either ``"forward"`` or ``"reverse"``; selects the gradient formula.
        temperature: Distillation temperature ``T``.
        block_v: Requested vocab tile width; clamped to at most 1024.
        block_m: Row tile height for the grid.

    Returns:
        Gradient w.r.t. ``student_2d`` shaped ``(n_rows, vocab_size)`` in
        ``student_2d.dtype`` (padding rows removed). Teacher logits receive no gradient.
    """
    block_v = min(int(block_v), 1024)
    n_rows, vocab_size = student_2d.shape
    n_rows_pad = pl.cdiv(n_rows, int(block_m)) * int(block_m)
    pad_rows = n_rows_pad - n_rows
    student_pad = _pad_rows_2d(student_2d, pad_rows)
    teacher_pad = _pad_rows_2d(teacher_2d, pad_rows)
    lse_t_pad = _pad_rows_1d(lse_t, pad_rows)
    lse_s_pad = _pad_rows_1d(lse_s, pad_rows)
    acc_pad = _pad_rows_1d(acc, pad_rows)
    weights_pad = _pad_rows_1d(weights_1d, pad_rows)
    dy_pad = _pad_rows_1d(dy, pad_rows)
    n_blocks = (vocab_size + int(block_v) - 1) // int(block_v)
    out = pl.pallas_call(
        functools.partial(
            _kl_bwd_kernel,
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
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (int(block_m), int(block_v)),
                lambda row_block, vocab_block: (row_block, vocab_block),
            ),
            scratch_shapes=[
                pltpu.VMEM((int(block_m), int(block_v)), student_2d.dtype),
                pltpu.VMEM((int(block_m), int(block_v)), teacher_2d.dtype),
                pltpu.VMEM((int(block_m),), lse_t.dtype),
                pltpu.VMEM((int(block_m),), lse_s.dtype),
                pltpu.VMEM((int(block_m),), acc.dtype),
                pltpu.VMEM((int(block_m),), weights_1d.dtype),
                pltpu.VMEM((int(block_m),), dy.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            grid=(n_rows_pad // int(block_m), n_blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        out_shape=_pallas_out_shape(student_pad.shape, student_2d.dtype),
    )(
        student_pad,
        teacher_pad,
        lse_t_pad.astype(jnp.float32),
        lse_s_pad.astype(jnp.float32),
        acc_pad.astype(jnp.float32),
        weights_pad.astype(jnp.float32),
        dy_pad.astype(jnp.float32),
    )
    return out[:n_rows, :]


__all__ = ["_kl_bwd_pallas"]
