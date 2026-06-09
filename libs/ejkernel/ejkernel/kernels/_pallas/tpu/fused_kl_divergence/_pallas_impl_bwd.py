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

"""Backward TPU Pallas kernels for fused KL divergence."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a row-major logit/gradient matrix to a whole Pallas row block."""
    if pad_rows == 0:
        return x
    return jnp.pad(x, ((0, pad_rows), (0, 0)), constant_values=pad_value)


def _pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a per-row vector to match the padded logit row count."""
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
    """DMA-copy aligned student and teacher vocab tiles into VMEM scratch."""
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
    """DMA-copy one contiguous row vector tile from HBM into VMEM scratch."""
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
    """Write one student-gradient tile for fused KL.

    The same kernel is used for replicated-vocab and TP-vocab paths. In TP mode
    the saved ``lse_t`` / ``lse_s`` / ``acc`` values are already global across
    the vocab-parallel axis, while ``student_ref`` and ``teacher_ref`` are local
    vocab shards.
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
    """Launch the TPU Pallas KL backward kernel for one local vocab shard."""
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
        out_shape=jax.ShapeDtypeStruct(student_pad.shape, student_2d.dtype),
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
