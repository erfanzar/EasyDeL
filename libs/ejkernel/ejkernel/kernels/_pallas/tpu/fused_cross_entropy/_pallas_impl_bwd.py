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

"""Backward TPU Pallas kernels for fused cross-entropy."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _pad_rows_2d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a row-major logits/gradient matrix to a whole Pallas row block."""
    if pad_rows == 0:
        return x
    return jnp.pad(x, ((0, pad_rows), (0, 0)), constant_values=pad_value)


def _pad_rows_1d(x: jax.Array, pad_rows: int, pad_value: float = 0.0) -> jax.Array:
    """Pad a per-row vector to match the padded logits row count."""
    if pad_rows == 0:
        return x
    return jnp.pad(x, (0, pad_rows), constant_values=pad_value)


def _copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int):
    """DMA-copy one contiguous row vector tile from HBM into VMEM scratch."""
    copy = pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(row_start, block_m)],
        dst_ref=dst_ref.at[pl.ds(0, block_m)],
        sem=sem_ref.at[0],
    )
    copy.start()
    copy.wait()


def _ce_bwd_kernel(
    logits_ref,
    lse_ref,
    targets_ref,
    weights_ref,
    dy_ref,
    dlogits_ref,
    logits_tile_ref,
    lse_scalar_ref,
    target_scalar_ref,
    weight_scalar_ref,
    dy_scalar_ref,
    dma_sem_ref,
    *,
    ignore_index: int,
    label_smoothing: float,
    z_loss: float,
    global_vocab_size: int,
    block_v: int,
    block_m: int,
):
    """Write one CE gradient tile using global LSE and local target ids.

    In TP-vocab mode ``targets_ref`` contains local target ids, with non-owned
    targets outside ``[0, local_vocab)``. In non-TP mode those ids are the
    original global ids and ``global_vocab_size`` is zero.
    """
    row_start = pl.program_id(0) * block_m
    block_idx = pl.program_id(1)
    _, vocab_size = logits_ref.shape
    start = block_idx * block_v
    size = jnp.minimum(block_v, vocab_size - start)
    offsets = jnp.arange(block_v)
    row_offsets = jnp.arange(block_m)
    row_active = (row_start + row_offsets) < logits_ref.shape[0]
    vocab_idx = start + offsets

    copy = pltpu.make_async_copy(
        src_ref=logits_ref.at[pl.ds(row_start, block_m), pl.ds(start, size)],
        dst_ref=logits_tile_ref.at[pl.ds(0, block_m), pl.ds(0, size)],
        sem=dma_sem_ref.at[0],
    )
    copy.start()
    copy.wait()

    _copy_rows_hbm_to_vmem(lse_ref, lse_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(targets_ref, target_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(weights_ref, weight_scalar_ref, dma_sem_ref, row_start, block_m)
    _copy_rows_hbm_to_vmem(dy_ref, dy_scalar_ref, dma_sem_ref, row_start, block_m)

    target = target_scalar_ref[...].astype(jnp.int32)
    weight = weight_scalar_ref[...].astype(jnp.float32)
    valid = row_active & (target != ignore_index) & (weight != 0.0)
    factor = jnp.where(valid, weight * dy_scalar_ref[...].astype(jnp.float32), 0.0)
    lse = lse_scalar_ref[...].astype(jnp.float32)
    tile = logits_tile_ref[...].astype(jnp.float32)
    in_vocab = offsets < size
    prob = jnp.exp(tile - lse[:, None])
    target_in_local = (target >= 0) & (target < vocab_size)
    target_for_compare = jnp.where(target_in_local, target, -1)
    onehot = (vocab_idx[None, :] == target_for_compare[:, None]).astype(jnp.float32)
    loss_vocab_size = global_vocab_size if global_vocab_size > 0 else vocab_size
    low_conf = (
        float(label_smoothing) / float(loss_vocab_size - 1) if loss_vocab_size > 1 and label_smoothing > 0.0 else 0.0
    )
    eff_target_w = (1.0 - float(label_smoothing)) - low_conf
    z_mult = 1.0 + 2.0 * float(z_loss) * lse
    grad = factor[:, None] * jnp.where(
        in_vocab[None, :],
        z_mult[:, None] * prob - low_conf - eff_target_w * onehot,
        0.0,
    )
    dlogits_ref[...] = grad.astype(dlogits_ref.dtype)


def _ce_bwd_pallas(
    logits_2d,
    lse,
    targets_1d,
    weights_1d,
    dy,
    *,
    ignore_index,
    label_smoothing,
    z_loss,
    block_v,
    block_m,
    global_vocab_size=0,
):
    """Launch the TPU Pallas CE backward kernel.

    ``global_vocab_size`` is only needed by TP vocab-parallel label smoothing;
    the current TP forward rejects label smoothing, so the default non-TP value
    is used for the supported paths.
    """
    block_v = min(int(block_v), 4096)
    n_rows, vocab_size = logits_2d.shape
    n_rows_pad = pl.cdiv(n_rows, int(block_m)) * int(block_m)
    pad_rows = n_rows_pad - n_rows
    logits_pad = _pad_rows_2d(logits_2d, pad_rows)
    lse_pad = _pad_rows_1d(lse, pad_rows)
    targets_pad = _pad_rows_1d(targets_1d, pad_rows, ignore_index)
    weights_pad = _pad_rows_1d(weights_1d, pad_rows)
    dy_pad = _pad_rows_1d(dy, pad_rows)
    n_blocks = (vocab_size + int(block_v) - 1) // int(block_v)
    out = pl.pallas_call(
        functools.partial(
            _ce_bwd_kernel,
            ignore_index=int(ignore_index),
            label_smoothing=float(label_smoothing),
            z_loss=float(z_loss),
            global_vocab_size=int(global_vocab_size),
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
            out_specs=pl.BlockSpec(
                (int(block_m), int(block_v)),
                lambda row_block, vocab_block: (row_block, vocab_block),
            ),
            scratch_shapes=[
                pltpu.VMEM((int(block_m), int(block_v)), logits_2d.dtype),
                pltpu.VMEM((int(block_m),), lse.dtype),
                pltpu.VMEM((int(block_m),), targets_1d.dtype),
                pltpu.VMEM((int(block_m),), weights_1d.dtype),
                pltpu.VMEM((int(block_m),), dy.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            grid=(n_rows_pad // int(block_m), n_blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        out_shape=jax.ShapeDtypeStruct(logits_pad.shape, logits_2d.dtype),
    )(
        logits_pad,
        lse_pad.astype(jnp.float32),
        targets_pad.astype(jnp.int32),
        weights_pad.astype(jnp.float32),
        dy_pad.astype(jnp.float32),
    )
    return out[:n_rows, :]


__all__ = ["_ce_bwd_pallas"]
