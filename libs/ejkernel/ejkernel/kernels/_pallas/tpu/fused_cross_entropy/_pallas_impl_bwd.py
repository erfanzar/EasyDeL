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

"""Backward TPU Pallas kernels for fused cross-entropy.

This module computes the gradient of the fused softmax-cross-entropy loss w.r.t.
the logits without materializing the softmax in HBM. It is the backward companion
of the forward kernels and is invoked from their custom VJP. The gradient is
streamed one ``(block_m, block_v)`` tile at a time: logits are DMA-copied into VMEM
scratch and combined with the saved per-row log-sum-exp, target ids, loss weights
and upstream cotangents, with label smoothing and the auxiliary z-loss folded into
the per-element gradient. The kernel is shared by the replicated and vocab-parallel
(TP) paths; in TP mode the saved LSE is already global while logits and target ids
are local vocab shards.

Contents:
    * ``_pallas_out_shape`` - manual-sharding-aware output ``ShapeDtypeStruct`` builder.
    * ``_pad_rows_2d`` / ``_pad_rows_1d`` - row padding so grid blocks stay rectangular.
    * ``_copy_rows_hbm_to_vmem`` - per-row scalar DMA helper.
    * ``_ce_bwd_kernel`` - the per-tile Pallas gradient kernel.
    * ``_ce_bwd_pallas`` - host-side launcher returning the un-padded ``dlogits``.
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
    """Pad a row-major logits/gradient matrix to a whole Pallas row block.

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
    """Pad a per-row vector to match the padded logits row count.

    Args:
        x: Rank-1 array shaped ``(n_rows,)`` (e.g. LSE, targets, weights, ``dy``).
        pad_rows: Number of entries to append; ``0`` is a no-op pass-through.
        pad_value: Constant value used for the appended entries. Defaults to ``0.0``;
            callers pass ``ignore_index`` for the targets vector so padded rows are
            treated as ignored.

    Returns:
        ``x`` unchanged when ``pad_rows == 0``, otherwise a new array shaped
        ``(n_rows + pad_rows,)`` with the trailing entries set to ``pad_value``.
    """
    if pad_rows == 0:
        return x
    return jnp.pad(x, (0, pad_rows), constant_values=pad_value)


def _copy_rows_hbm_to_vmem(src_ref, dst_ref, sem_ref, row_start, block_m: int):
    """DMA-copy one contiguous row vector tile from HBM into VMEM scratch.

    Starts and immediately waits on a single async DMA that moves ``block_m``
    contiguous entries from the HBM source ref into the front of the VMEM scratch
    ref, using semaphore slot ``0`` of ``sem_ref``.

    Args:
        src_ref: HBM ref of a rank-1 per-row array (LSE, targets, weights or ``dy``).
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
    """Write one cross-entropy gradient tile using global LSE and local target ids.

    Each grid step handles one ``(block_m, block_v)`` tile of the gradient. The
    logits tile and the per-row scalars (LSE, target, weight, ``dy``) are DMA-copied
    from HBM into VMEM scratch, then the softmax-cross-entropy gradient is computed
    in fp32 as ``weight * dy * (z_mult * softmax - low_conf - eff_target_w * onehot)``
    where ``z_mult = 1 + 2 * z_loss * lse`` folds in the z-loss penalty and
    ``low_conf`` / ``eff_target_w`` apply label smoothing. Rows that are padding,
    ``ignore_index`` targets, or zero-weight contribute zero gradient; vocab columns
    beyond ``size`` (the active part of the last tile) are masked to zero.

    In TP-vocab mode ``targets_ref`` contains local target ids, with non-owned
    targets outside ``[0, local_vocab)`` (the one-hot is then all zeros for this
    shard). In non-TP mode those ids are the original global ids and
    ``global_vocab_size`` is zero.

    Args:
        logits_ref: HBM ref of student logits shaped ``(n_rows, vocab_size)``.
        lse_ref: HBM ref of per-row log-sum-exp shaped ``(n_rows,)`` (fp32); global
            across the vocab-parallel axis in TP mode.
        targets_ref: HBM ref of per-row target ids shaped ``(n_rows,)`` (int32);
            local ids in TP mode.
        weights_ref: HBM ref of per-row loss weights shaped ``(n_rows,)`` (fp32).
        dy_ref: HBM ref of per-row upstream cotangents shaped ``(n_rows,)`` (fp32).
        dlogits_ref: VMEM output ref of shape ``(block_m, block_v)`` receiving the
            gradient tile in ``dlogits_ref.dtype``.
        logits_tile_ref: VMEM scratch of shape ``(block_m, block_v)`` for the logits tile.
        lse_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the LSE row tile.
        target_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the target row tile.
        weight_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the weight row tile.
        dy_scalar_ref: VMEM scratch of shape ``(block_m,)`` for the ``dy`` row tile.
        dma_sem_ref: DMA semaphore ref (slot ``0``) used for all copies.
        ignore_index: Target id whose rows produce zero gradient.
        label_smoothing: Smoothing coefficient in ``[0, 1)``; spreads ``label_smoothing``
            mass uniformly over the other classes.
        z_loss: Coefficient of the auxiliary z-loss on the partition function.
        global_vocab_size: Global vocabulary size used to size the label-smoothing
            denominator in TP mode; ``0`` falls back to the local ``vocab_size``.
        block_v: Vocab tile width (columns) handled per grid step.
        block_m: Row tile height handled per grid step.

    Returns:
        None. Results are written in place to ``dlogits_ref``.
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
    """Launch the TPU Pallas cross-entropy backward kernel and return ``dlogits``.

    Caps ``block_v`` at 4096, pads the row dimension up to a whole multiple of
    ``block_m`` (targets padded with ``ignore_index`` so padded rows are ignored),
    casts the per-row auxiliaries to the kernel's expected dtypes, runs
    :func:`_ce_bwd_kernel` over a ``(row_blocks, vocab_blocks)`` parallel grid, then
    slices the padding back off.

    ``global_vocab_size`` is only needed by TP vocab-parallel label smoothing;
    the current TP forward rejects label smoothing, so the default non-TP value
    is used for the supported paths.

    Args:
        logits_2d: Student logits shaped ``(n_rows, vocab_size)`` (local shard in TP mode).
        lse: Per-row log-sum-exp shaped ``(n_rows,)``; cast to fp32 before the call.
        targets_1d: Per-row target ids shaped ``(n_rows,)``; cast to int32.
        weights_1d: Per-row loss weights shaped ``(n_rows,)``; cast to fp32.
        dy: Per-row upstream cotangents shaped ``(n_rows,)``; cast to fp32.
        ignore_index: Target id marking rows that produce zero gradient; also used as
            the pad value for the padded target rows.
        label_smoothing: Label-smoothing coefficient in ``[0, 1)``.
        z_loss: Coefficient of the auxiliary z-loss penalty.
        block_v: Requested vocab tile width; clamped to at most 4096.
        block_m: Row tile height for the grid.
        global_vocab_size: Global vocab size for TP label smoothing; ``0`` (default)
            selects the local ``vocab_size`` denominator.

    Returns:
        Gradient w.r.t. ``logits_2d`` shaped ``(n_rows, vocab_size)`` in
        ``logits_2d.dtype`` (padding rows removed).
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
        out_shape=_pallas_out_shape(logits_pad.shape, logits_2d.dtype),
    )(
        logits_pad,
        lse_pad.astype(jnp.float32),
        targets_pad.astype(jnp.int32),
        weights_pad.astype(jnp.float32),
        dy_pad.astype(jnp.float32),
    )
    return out[:n_rows, :]


__all__ = ["_ce_bwd_pallas"]
