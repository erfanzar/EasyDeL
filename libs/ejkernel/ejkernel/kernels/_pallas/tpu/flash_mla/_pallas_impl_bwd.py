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

"""Flash MLA backward pass TPU kernel implementation using Pallas.

This module provides the gradient computation kernels for Flash MLA on
Google TPUs. The backward pass is split into two main Pallas kernels:
1. dKV kernel: Computes gradients w.r.t. projected keys and values
2. dQ kernel: Computes gradients w.r.t. queries (and optionally bias)

Attention weights are recomputed on-the-fly from saved softmax statistics
(l and m) without materialising the full attention matrix.

After the Pallas kernels, JAX post-processing computes:
- GQA reduction (sum over query groups)
- dw_kc, dw_vc (KV projection weight gradients)
- dkv_latent (compressed KV gradient)
- db_q, db_k (RoPE gradients, by summing per-head contributions)
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._utils import (
    DEFAULT_MASK_VALUE,
    MIN_BLOCK_SIZE,
    TRANS_B_DIM_NUMBERS,
    below_or_on_diag,
)

ROPE_NONE = 0
ROPE_FUSED = 1
ROPE_DECOUPLED = 2


def _flash_mla_dkv_kernel(
    q_tile_ref,
    k_nope_tile_ref,
    v_tile_ref,
    bq_tile_ref,
    bk_tile_ref,
    bias_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    db_k_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    db_k_scratch_ref,
    *,
    rope_mode,
    d_nope,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    mask_value,
    q_seq_len,
    block_q,
    block_k,
):
    """Pallas kernel for computing ``dk_nope``, ``dv``, and optionally ``db_k``.

    Grid: ``(batch, q_heads, kv_len // block_k, seq_q // block_q)``.
    Axis 3 (q_blocks) is "arbitrary" — iterates over query blocks in order,
    accumulating ``dk_nope`` and ``dv`` in VMEM scratch per KV block, then
    stores when ``q_seq_index == q_seq_len // block_q - 1``.

    VMEM scratch:
        dk_scratch_ref: [block_k, d_nope] float32.
        dv_scratch_ref: [block_k, v_head_dim] float32.
        db_k_scratch_ref: [block_k, rope_dim] float32 (used only for RoPE
            modes; allocated but unused for ROPE_NONE).

    Args:
        q_tile_ref: [1, 1, block_q, q_head_dim].
        k_nope_tile_ref: [1, 1, block_k, d_nope].
        v_tile_ref: [1, 1, block_k, v_head_dim].
        bq_tile_ref: Optional [1, block_q, rope_dim].
        bk_tile_ref: Optional [1, block_k, rope_dim].
        bias_tile_ref: Optional [1, 1, block_q, block_k].
        l_tile_ref, m_tile_ref: [1, 1, block_q, MIN_BLOCK_SIZE] — residuals.
        do_tile_ref: [1, 1, block_q, v_head_dim].
        di_tile_ref: [1, 1, block_q, MIN_BLOCK_SIZE].
        dk_tile_ref: [1, 1, block_k, d_nope] — output.
        dv_tile_ref: [1, 1, block_k, v_head_dim] — output.
        db_k_tile_ref: [1, 1, block_k, rope_dim] output (or None).
        dk_scratch_ref, dv_scratch_ref, db_k_scratch_ref: VMEM accumulators.
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        d_nope: Nope key dimension.
        causal: Whether causal masking was applied.
        softmax_scale: Attention scale factor.
        sliding_window: Optional (left, right) window.
        logits_soft_cap: Optional soft cap.
        mask_value: Large negative value for masked positions.
        q_seq_len: Total query sequence length.
        block_q: Query block size.
        block_k: KV block size.
    """
    kv_seq_index = pl.program_id(axis=2)
    q_seq_index = pl.program_id(axis=3)

    @pl.when(q_seq_index == 0)
    def _init():
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)
        if rope_mode != ROPE_NONE:
            db_k_scratch_ref[:, :] = jnp.zeros(db_k_scratch_ref.shape, db_k_scratch_ref.dtype)

    if causal:
        should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k)
    else:
        should_run = True

    @pl.when(should_run)
    def _run():
        q = q_tile_ref[0, 0, :, :].astype(jnp.float32)
        k_nope = k_nope_tile_ref[0, 0, :, :]
        v = v_tile_ref[0, 0, :, :]
        l = l_tile_ref[0, 0, :, :]
        m = m_tile_ref[0, 0, :, :]
        do = do_tile_ref[0, 0, :, :]
        di = di_tile_ref[0, 0, :, :]

        if rope_mode == ROPE_NONE:
            logits = lax.dot_general(q, k_nope, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
        elif rope_mode == ROPE_FUSED:
            q_nope = q[:, :d_nope]
            q_rope = q[:, d_nope:]
            bk = bk_tile_ref[0, :, :].astype(jnp.float32)
            logits = lax.dot_general(
                q_nope,
                k_nope,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            ) + lax.dot_general(
                q_rope,
                bk,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
        else:
            bq = bq_tile_ref[0, :, :].astype(jnp.float32)
            bk = bk_tile_ref[0, :, :].astype(jnp.float32)
            logits = lax.dot_general(
                q, k_nope, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            ) + lax.dot_general(bq, bk, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        if softmax_scale != 1.0:
            logits *= softmax_scale

        softcap_tanh = None
        if logits_soft_cap is not None:
            softcap_tanh = jnp.tanh(logits / logits_soft_cap)
            logits = logits_soft_cap * softcap_tanh

        if bias_tile_ref is not None:
            logits += bias_tile_ref[0, 0, :, :].astype(jnp.float32)

        mask = None
        if causal:
            mask_shape = (block_q, block_k)
            row_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_index * block_q
            col_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_index * block_k
            mask = col_ids <= row_ids

        if sliding_window is not None:
            window_left, window_right = sliding_window
            mask_shape = (block_q, block_k)
            row_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_index * block_q
            col_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_index * block_k
            window_mask = (col_ids >= (row_ids - window_left)) & (col_ids <= (row_ids + window_right))
            mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

        if mask is not None:
            logits = logits + jnp.where(mask, 0.0, mask_value)

        kv_repeats = block_k // MIN_BLOCK_SIZE
        p = jnp.exp(logits - pltpu.repeat(m, kv_repeats, axis=1))
        p = p * pltpu.repeat(1.0 / l, kv_repeats, axis=1)

        dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
        dv_scratch_ref[:, :] += dv.astype(dv_scratch_ref.dtype)

        dp = lax.dot_general(do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        ds = (dp - pltpu.repeat(di, kv_repeats, axis=1)) * p

        if logits_soft_cap is not None:
            ds = ds * (1.0 - softcap_tanh * softcap_tanh)

        if softmax_scale != 1.0:
            ds = ds * softmax_scale

        if rope_mode == ROPE_NONE:
            dk = lax.dot(ds.T.astype(q.dtype), q, preferred_element_type=jnp.float32)
        elif rope_mode == ROPE_FUSED:
            q_nope = q[:, :d_nope]
            q_rope = q[:, d_nope:]
            dk = lax.dot(
                ds.T.astype(q_nope.dtype),
                q_nope,
                preferred_element_type=jnp.float32,
            )
            db_k_inc = lax.dot(
                ds.T.astype(q_rope.dtype),
                q_rope,
                preferred_element_type=jnp.float32,
            )
            db_k_scratch_ref[:, :] += db_k_inc.astype(db_k_scratch_ref.dtype)
        else:
            dk = lax.dot(ds.T.astype(q.dtype), q, preferred_element_type=jnp.float32)
            bq = bq_tile_ref[0, :, :].astype(jnp.float32)
            db_k_inc = lax.dot(ds.T.astype(bq.dtype), bq, preferred_element_type=jnp.float32)
            db_k_scratch_ref[:, :] += db_k_inc.astype(db_k_scratch_ref.dtype)

        dk_scratch_ref[:, :] += dk.astype(dk_scratch_ref.dtype)

    @pl.when(q_seq_index == q_seq_len // block_q - 1)
    def _store():
        dk_tile_ref[0, 0, :, :] = dk_scratch_ref[:, :].astype(dk_tile_ref.dtype)
        dv_tile_ref[0, 0, :, :] = dv_scratch_ref[:, :].astype(dv_tile_ref.dtype)
        if rope_mode != ROPE_NONE and db_k_tile_ref is not None:
            db_k_tile_ref[0, 0, :, :] = db_k_scratch_ref[:, :].astype(db_k_tile_ref.dtype)


def _flash_mla_bwd_dkv(
    q,
    k_nope,
    v,
    b_q,
    b_k,
    bias,
    l,
    m,
    do,
    di,
    *,
    rope_mode,
    d_nope,
    gqa_ratio,
    block_q,
    block_k,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    causal,
):
    """Compute gradients w.r.t. the projected keys, values, and RoPE keys.

    Sets up and launches ``_flash_mla_dkv_kernel`` which iterates over query
    blocks (the "arbitrary" axis), accumulating ``dk_nope`` and ``dv`` in
    VMEM scratch per KV block.  Optionally also computes ``db_k`` for
    ROPE_FUSED or ROPE_DECOUPLED modes.

    Grid: ``(batch, q_heads, kv_len // block_k, seq_q // block_q)``.
    Dimension semantics: ``(parallel, parallel, parallel, arbitrary)``.

    VMEM scratch: three buffers of shape ``(block_k, d_nope)``,
    ``(block_k, v_head_dim)``, and ``(block_k, rope_dim)``.

    Args:
        q: Query [B, q_heads, seq_q, q_head_dim].
        k_nope: Expanded nope-key [B, kv_heads, kv_len, d_nope].
        v: Expanded value [B, kv_heads, kv_len, v_head_dim].
        b_q: Optional query RoPE [B, seq_q, rope_dim].
        b_k: Optional key RoPE [B, kv_len, rope_dim].
        bias: Optional additive bias [B, q_heads, seq_q, kv_len].
        l: Saved softmax denominator [B, q_heads, seq_q, MIN_BLOCK_SIZE].
        m: Saved softmax max [B, q_heads, seq_q, MIN_BLOCK_SIZE].
        do: Output gradient [B, q_heads, seq_q, v_head_dim].
        di: Precomputed ``sum(o * do)`` [B, q_heads, seq_q, MIN_BLOCK_SIZE].
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        d_nope: Size of the nope (non-RoPE) key dimension.
        gqa_ratio: ``q_heads // kv_heads``.
        block_q: Query block size for the backward kernel.
        block_k: KV block size for the backward kernel.
        softmax_scale: Attention scale used in forward.
        sliding_window: Optional (left, right) window tuple.
        logits_soft_cap: Optional soft cap for logits.
        causal: Whether causal masking was applied.

    Returns:
        Three-element tuple ``(dk_nope, dv_all, db_k_all)`` where shapes are
        ``[B, q_heads, kv_len, d_nope]``, ``[B, q_heads, kv_len, v_head_dim]``,
        and ``[B, q_heads, kv_len, rope_dim]`` respectively.
        ``db_k_all`` is ``None`` when ``rope_mode == ROPE_NONE``.
    """
    batch_size, q_heads, seq_q, q_head_dim = q.shape
    _, _, kv_len, _ = k_nope.shape
    v_head_dim = v.shape[-1]

    grid = (
        batch_size,
        q_heads,
        kv_len // block_k,
        seq_q // block_q,
    )

    def q_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx):
        if causal:
            next_q = lax.select(
                below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k),
                q_seq_idx,
                0,
            )
        else:
            next_q = q_seq_idx
        return (batch_idx, head_idx, next_q, 0)

    def kv_index_map(batch_idx, head_idx, kv_seq_idx, _):
        return (batch_idx, head_idx // gqa_ratio, kv_seq_idx, 0)

    def bq_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx):
        del head_idx
        if causal:
            next_q = lax.select(
                below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k),
                q_seq_idx,
                0,
            )
        else:
            next_q = q_seq_idx
        return (batch_idx, next_q, 0)

    def bk_index_map(batch_idx, head_idx, kv_seq_idx, _):
        del head_idx
        return (batch_idx, kv_seq_idx, 0)

    def bias_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx):
        if causal:
            should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k)
            next_q = lax.select(should_run, q_seq_idx, 0)
            next_kv = lax.select(should_run, kv_seq_idx, 0)
        else:
            next_q = q_seq_idx
            next_kv = kv_seq_idx
        return (batch_idx, head_idx, next_q, next_kv)

    def lm_index_map(batch_idx, head_idx, _, q_seq_idx):
        return (batch_idx, head_idx, q_seq_idx, 0)

    def do_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx):
        if causal:
            next_q = lax.select(
                below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k),
                q_seq_idx,
                0,
            )
        else:
            next_q = q_seq_idx
        return (batch_idx, head_idx, next_q, 0)

    def dkv_index_map(batch_idx, head_idx, kv_seq_idx, _):
        return (batch_idx, head_idx, kv_seq_idx, 0)

    rope_dim = b_k.shape[-1] if b_k is not None else MIN_BLOCK_SIZE

    in_specs = [
        pl.BlockSpec((1, 1, block_q, q_head_dim), q_index_map),
        pl.BlockSpec((1, 1, block_k, d_nope), kv_index_map),
        pl.BlockSpec((1, 1, block_k, v_head_dim), kv_index_map),
        (pl.BlockSpec((1, block_q, b_q.shape[-1]), bq_index_map) if b_q is not None else None),
        (pl.BlockSpec((1, block_k, b_k.shape[-1]), bk_index_map) if b_k is not None else None),
        (pl.BlockSpec((1, 1, block_q, block_k), bias_index_map) if bias is not None else None),
        pl.BlockSpec((1, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((1, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((1, 1, block_q, v_head_dim), do_index_map),
        pl.BlockSpec((1, 1, block_q, MIN_BLOCK_SIZE), do_index_map),
    ]

    dk_shape = jax.ShapeDtypeStruct((batch_size, q_heads, kv_len, d_nope), q.dtype)
    dv_shape = jax.ShapeDtypeStruct((batch_size, q_heads, kv_len, v_head_dim), q.dtype)

    dk_spec = pl.BlockSpec((1, 1, block_k, d_nope), dkv_index_map)
    dv_spec = pl.BlockSpec((1, 1, block_k, v_head_dim), dkv_index_map)

    out_shapes = [dk_shape, dv_shape]
    out_specs = [dk_spec, dv_spec]

    if rope_mode != ROPE_NONE:
        db_k_shape = jax.ShapeDtypeStruct((batch_size, q_heads, kv_len, rope_dim), jnp.float32)
        db_k_spec = pl.BlockSpec((1, 1, block_k, rope_dim), dkv_index_map)
        out_shapes.append(db_k_shape)
        out_specs.append(db_k_spec)
    else:
        out_shapes.append(None)
        out_specs.append(None)

    scratch_shapes = [
        pltpu.VMEM((block_k, d_nope), jnp.float32),
        pltpu.VMEM((block_k, v_head_dim), jnp.float32),
        pltpu.VMEM((block_k, rope_dim), jnp.float32),
    ]

    kernel = functools.partial(
        _flash_mla_dkv_kernel,
        rope_mode=rope_mode,
        d_nope=d_nope,
        causal=causal,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=DEFAULT_MASK_VALUE,
        q_seq_len=seq_q,
        block_q=block_q,
        block_k=block_k,
    )

    with jax.named_scope("flash_mla_bwd_dkv"):
        results = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            ),
        )(q, k_nope, v, b_q, b_k, bias, l, m, do, di)

    if rope_mode != ROPE_NONE:
        dk_nope, dv_all, db_k_all = results
    else:
        dk_nope, dv_all = results[0], results[1]
        db_k_all = None

    return dk_nope, dv_all, db_k_all


def _flash_mla_dq_kernel(
    q_tile_ref,
    k_nope_tile_ref,
    v_tile_ref,
    bq_tile_ref,
    bk_tile_ref,
    bias_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    dbias_tile_ref,
    db_q_tile_ref,
    dq_scratch_ref,
    db_q_scratch_ref,
    *,
    rope_mode,
    d_nope,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    mask_value,
    kv_seq_len,
    block_q,
    block_k,
):
    """Pallas kernel for computing ``dq`` and optionally ``dbias`` / ``db_q``.

    Grid: ``(batch, q_heads, seq_q // block_q, kv_len // block_k)``.
    Axis 3 (kv_blocks) is "arbitrary" — iterates over KV blocks accumulating
    ``dq`` in VMEM scratch, then stores when
    ``kv_seq_index == kv_seq_len // block_k - 1``.

    VMEM scratch:
        dq_scratch_ref: [block_q, q_head_dim] float32.
        db_q_scratch_ref: [block_q, rope_dim] float32 (ROPE_DECOUPLED only).

    Args:
        q_tile_ref: [1, 1, block_q, q_head_dim].
        k_nope_tile_ref: [1, 1, block_k, d_nope].
        v_tile_ref: [1, 1, block_k, v_head_dim].
        bq_tile_ref: Optional [1, block_q, rope_dim].
        bk_tile_ref: Optional [1, block_k, rope_dim].
        bias_tile_ref: Optional [1, 1, block_q, block_k].
        l_tile_ref, m_tile_ref: [1, 1, block_q, MIN_BLOCK_SIZE] residuals.
        do_tile_ref: [1, 1, block_q, v_head_dim].
        di_tile_ref: [1, 1, block_q, MIN_BLOCK_SIZE].
        dq_tile_ref: [1, 1, block_q, q_head_dim] — output.
        dbias_tile_ref: Optional [1, 1, block_q, block_k] bias grad output.
        db_q_tile_ref: Optional [1, 1, block_q, rope_dim] grad output.
        dq_scratch_ref, db_q_scratch_ref: VMEM accumulators.
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        d_nope: Nope key dimension.
        causal: Whether causal masking was applied.
        softmax_scale: Attention scale factor.
        sliding_window: Optional (left, right) window.
        logits_soft_cap: Optional soft cap.
        mask_value: Large negative value for masked positions.
        kv_seq_len: Total KV length.
        block_q: Query block size.
        block_k: KV block size.
    """
    q_seq_index = pl.program_id(axis=2)
    kv_seq_index = pl.program_id(axis=3)

    @pl.when(kv_seq_index == 0)
    def _init():
        dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)
        if rope_mode == ROPE_DECOUPLED:
            db_q_scratch_ref[:, :] = jnp.zeros(db_q_scratch_ref.shape, db_q_scratch_ref.dtype)

    if causal:
        should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k)
        should_not_run = lax.select(should_run, False, True)
    else:
        should_run = True
        should_not_run = False

    @pl.when(should_run)
    def _run():
        q = q_tile_ref[0, 0, :, :].astype(jnp.float32)
        k_nope = k_nope_tile_ref[0, 0, :, :]
        v = v_tile_ref[0, 0, :, :]
        l = l_tile_ref[0, 0, :, :]
        m = m_tile_ref[0, 0, :, :]
        do = do_tile_ref[0, 0, :, :]
        di = di_tile_ref[0, 0, :, :]

        if rope_mode == ROPE_NONE:
            logits = lax.dot_general(q, k_nope, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
        elif rope_mode == ROPE_FUSED:
            q_nope = q[:, :d_nope]
            q_rope = q[:, d_nope:]
            bk = bk_tile_ref[0, :, :].astype(jnp.float32)
            logits = lax.dot_general(
                q_nope,
                k_nope,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            ) + lax.dot_general(
                q_rope,
                bk,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
        else:
            bq = bq_tile_ref[0, :, :].astype(jnp.float32)
            bk = bk_tile_ref[0, :, :].astype(jnp.float32)
            logits = lax.dot_general(
                q, k_nope, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            ) + lax.dot_general(bq, bk, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        if softmax_scale != 1.0:
            logits *= softmax_scale

        softcap_tanh = None
        if logits_soft_cap is not None:
            softcap_tanh = jnp.tanh(logits / logits_soft_cap)
            logits = logits_soft_cap * softcap_tanh

        if bias_tile_ref is not None:
            logits += bias_tile_ref[0, 0, :, :].astype(jnp.float32)

        mask = None
        if causal:
            mask_shape = (block_q, block_k)
            row_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_index * block_q
            col_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_index * block_k
            mask = col_ids <= row_ids

        if sliding_window is not None:
            window_left, window_right = sliding_window
            mask_shape = (block_q, block_k)
            row_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_index * block_q
            col_ids = lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_index * block_k
            window_mask = (col_ids >= (row_ids - window_left)) & (col_ids <= (row_ids + window_right))
            mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

        if mask is not None:
            logits = logits + jnp.where(mask, 0.0, mask_value)

        kv_repeats = block_k // MIN_BLOCK_SIZE
        p = jnp.exp(logits - pltpu.repeat(m, kv_repeats, axis=1))
        p = p * pltpu.repeat(1.0 / l, kv_repeats, axis=1)

        dp = lax.dot_general(do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        ds = (dp - pltpu.repeat(di, kv_repeats, axis=1)) * p

        if dbias_tile_ref is not None:
            dbias_tile_ref[0, 0, :, :] = ds.astype(dbias_tile_ref.dtype)

        if logits_soft_cap is not None:
            ds = ds * (1.0 - softcap_tanh * softcap_tanh)

        if softmax_scale != 1.0:
            ds = ds * softmax_scale

        if rope_mode == ROPE_NONE:
            dq_inc = lax.dot(
                ds.astype(k_nope.dtype),
                k_nope,
                preferred_element_type=jnp.float32,
            )
        elif rope_mode == ROPE_FUSED:
            bk = bk_tile_ref[0, :, :].astype(jnp.float32)
            dq_nope = lax.dot(
                ds.astype(k_nope.dtype),
                k_nope,
                preferred_element_type=jnp.float32,
            )
            dq_rope = lax.dot(ds.astype(bk.dtype), bk, preferred_element_type=jnp.float32)
            dq_inc = jnp.concatenate([dq_nope, dq_rope], axis=-1)
        else:
            dq_inc = lax.dot(
                ds.astype(k_nope.dtype),
                k_nope,
                preferred_element_type=jnp.float32,
            )
            bk = bk_tile_ref[0, :, :].astype(jnp.float32)
            db_q_inc = lax.dot(ds.astype(bk.dtype), bk, preferred_element_type=jnp.float32)
            db_q_scratch_ref[:, :] += db_q_inc.astype(db_q_scratch_ref.dtype)

        dq_scratch_ref[:, :] += dq_inc.astype(dq_scratch_ref.dtype)

    @pl.when(should_not_run)
    def _zero_dbias():
        if dbias_tile_ref is not None:
            dbias_tile_ref[0, 0, :, :] = jnp.zeros((block_q, block_k), dbias_tile_ref.dtype)

    @pl.when(kv_seq_index == kv_seq_len // block_k - 1)
    def _store():
        dq_tile_ref[0, 0, :, :] = dq_scratch_ref[:, :].astype(dq_tile_ref.dtype)
        if rope_mode == ROPE_DECOUPLED and db_q_tile_ref is not None:
            db_q_tile_ref[0, 0, :, :] = db_q_scratch_ref[:, :].astype(db_q_tile_ref.dtype)


def _flash_mla_bwd_dq(
    q,
    k_nope,
    v,
    b_q,
    b_k,
    bias,
    l,
    m,
    do,
    di,
    *,
    rope_mode,
    d_nope,
    gqa_ratio,
    block_q,
    block_k,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    causal,
):
    """Compute gradient w.r.t. queries, optional bias, and optional b_q.

    Sets up and launches ``_flash_mla_dq_kernel`` which iterates over KV
    blocks (the "arbitrary" axis), accumulating ``dq`` in VMEM scratch.
    Also optionally outputs ``dbias`` and ``db_q`` (for ROPE_DECOUPLED).

    Grid: ``(batch, q_heads, seq_q // block_q, kv_len // block_k)``.
    Dimension semantics: ``(parallel, parallel, parallel, arbitrary)``.

    VMEM scratch: two buffers of shape ``(block_q, q_head_dim)`` and
    ``(block_q, rope_dim)``.

    Args:
        q: Query [B, q_heads, seq_q, q_head_dim].
        k_nope: Expanded nope-key [B, kv_heads, kv_len, d_nope].
        v: Expanded value [B, kv_heads, kv_len, v_head_dim].
        b_q: Optional query RoPE [B, seq_q, rope_dim].
        b_k: Optional key RoPE [B, kv_len, rope_dim].
        bias: Optional additive bias [B, q_heads, seq_q, kv_len].
        l: Saved softmax denominator [B, q_heads, seq_q, MIN_BLOCK_SIZE].
        m: Saved softmax max [B, q_heads, seq_q, MIN_BLOCK_SIZE].
        do: Output gradient [B, q_heads, seq_q, v_head_dim].
        di: Precomputed ``sum(o * do)`` [B, q_heads, seq_q, MIN_BLOCK_SIZE].
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        d_nope: Size of the nope key dimension.
        gqa_ratio: ``q_heads // kv_heads``.
        block_q: Query block size.
        block_k: KV block size.
        softmax_scale: Attention scale used in forward.
        sliding_window: Optional (left, right) window tuple.
        logits_soft_cap: Optional soft cap.
        causal: Whether causal masking was applied.

    Returns:
        Three-element tuple ``(dq, dbias_out, db_q_all)`` where ``dq``
        has shape ``[B, q_heads, seq_q, q_head_dim]``, ``dbias_out`` matches
        the bias shape (or ``None``), and ``db_q_all`` has shape
        ``[B, q_heads, seq_q, rope_dim]`` (or ``None`` unless
        ``rope_mode == ROPE_DECOUPLED``).
    """
    batch_size, q_heads, seq_q, q_head_dim = q.shape
    _, _, kv_len, _ = k_nope.shape
    v_head_dim = v.shape[-1]

    grid = (
        batch_size,
        q_heads,
        seq_q // block_q,
        kv_len // block_k,
    )

    def q_index_map(batch_idx, head_idx, q_seq_idx, _):
        return (batch_idx, head_idx, q_seq_idx, 0)

    def kv_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx):
        if causal:
            next_kv = lax.select(
                below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k),
                kv_seq_idx,
                0,
            )
        else:
            next_kv = kv_seq_idx
        return (batch_idx, head_idx // gqa_ratio, next_kv, 0)

    def bq_index_map(batch_idx, head_idx, q_seq_idx, _):
        del head_idx
        return (batch_idx, q_seq_idx, 0)

    def bk_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx):
        del head_idx
        if causal:
            next_kv = lax.select(
                below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k),
                kv_seq_idx,
                0,
            )
        else:
            next_kv = kv_seq_idx
        return (batch_idx, next_kv, 0)

    def bias_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx):
        if causal:
            should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k)
            next_q = lax.select(should_run, q_seq_idx, 0)
            next_kv = lax.select(should_run, kv_seq_idx, 0)
        else:
            next_q = q_seq_idx
            next_kv = kv_seq_idx
        return (batch_idx, head_idx, next_q, next_kv)

    def lm_index_map(batch_idx, head_idx, q_seq_idx, _):
        return (batch_idx, head_idx, q_seq_idx, 0)

    in_specs = [
        pl.BlockSpec((1, 1, block_q, q_head_dim), q_index_map),
        pl.BlockSpec((1, 1, block_k, d_nope), kv_index_map),
        pl.BlockSpec((1, 1, block_k, v_head_dim), kv_index_map),
        (pl.BlockSpec((1, block_q, b_q.shape[-1]), bq_index_map) if b_q is not None else None),
        (pl.BlockSpec((1, block_k, b_k.shape[-1]), bk_index_map) if b_k is not None else None),
        (pl.BlockSpec((1, 1, block_q, block_k), bias_index_map) if bias is not None else None),
        pl.BlockSpec((1, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((1, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((1, 1, block_q, v_head_dim), q_index_map),
        pl.BlockSpec((1, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
    ]

    dq_shape = jax.ShapeDtypeStruct((batch_size, q_heads, seq_q, q_head_dim), q.dtype)
    dq_spec = pl.BlockSpec((1, 1, block_q, q_head_dim), q_index_map)

    out_shapes = [dq_shape]
    out_specs = [dq_spec]

    def dbias_out_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx):
        return (batch_idx, head_idx, q_seq_idx, kv_seq_idx)

    if bias is not None:
        dbias_shape = jax.ShapeDtypeStruct(bias.shape, bias.dtype)
        dbias_spec = pl.BlockSpec((1, 1, block_q, block_k), dbias_out_index_map)
        out_shapes.append(dbias_shape)
        out_specs.append(dbias_spec)
    else:
        out_shapes.append(None)
        out_specs.append(None)

    if rope_mode == ROPE_DECOUPLED:
        db_q_shape = jax.ShapeDtypeStruct((batch_size, q_heads, seq_q, b_q.shape[-1]), jnp.float32)
        db_q_spec = pl.BlockSpec((1, 1, block_q, b_q.shape[-1]), q_index_map)
        out_shapes.append(db_q_shape)
        out_specs.append(db_q_spec)
    else:
        out_shapes.append(None)
        out_specs.append(None)

    db_q_dim = b_q.shape[-1] if b_q is not None else MIN_BLOCK_SIZE
    scratch_shapes = [
        pltpu.VMEM((block_q, q_head_dim), jnp.float32),
        pltpu.VMEM((block_q, db_q_dim), jnp.float32),
    ]

    kernel = functools.partial(
        _flash_mla_dq_kernel,
        rope_mode=rope_mode,
        d_nope=d_nope,
        causal=causal,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=DEFAULT_MASK_VALUE,
        kv_seq_len=kv_len,
        block_q=block_q,
        block_k=block_k,
    )

    with jax.named_scope("flash_mla_bwd_dq"):
        results = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            ),
        )(q, k_nope, v, b_q, b_k, bias, l, m, do, di)

    dq = results[0]
    dbias_out = results[1] if bias is not None else None
    db_q_all = results[2] if rope_mode == ROPE_DECOUPLED else None

    return dq, dbias_out, db_q_all


def _flash_mla_bwd_impl(
    rope_mode,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_q,
    block_k,
    residuals,
    do,
):
    """Compute gradients for Flash MLA using Pallas kernels + JAX post-processing.

    Args:
        rope_mode: ROPE_NONE / ROPE_FUSED / ROPE_DECOUPLED.
        causal: Whether causal masking was applied.
        softmax_scale: Attention scale factor used in forward.
        sliding_window: Optional (left, right) window tuple.
        logits_soft_cap: Optional soft cap for logits.
        block_q: Query block size for backward kernels.
        block_k: KV block size for backward kernels.
        residuals: Tuple of (q, kv_latent, w_kc, w_vc, b_q, b_k, bias, o, l, m).
        do: Output gradient [batch, q_heads, seq_q, v_head_dim].

    Returns:
        Tuple of gradients (dq, dkv_latent, dw_kc, dw_vc, db_q, db_k, dbias).
    """
    q, kv_latent, w_kc, w_vc, b_q, b_k, bias, o, l, m = residuals

    batch_size, q_heads, _seq_q, _q_head_dim = q.shape
    _, kv_len, _kv_lora_rank = kv_latent.shape
    kv_heads = w_kc.shape[0]
    d_nope = w_kc.shape[2]
    v_head_dim = w_vc.shape[2]
    gqa_ratio = q_heads // kv_heads

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    kv_f32 = kv_latent.astype(jnp.float32)
    w_kc_f32 = w_kc.astype(jnp.float32)
    w_vc_f32 = w_vc.astype(jnp.float32)

    k_nope = jnp.einsum("btl,hld->bthd", kv_f32, w_kc_f32)
    k_nope = jnp.transpose(k_nope, (0, 2, 1, 3))

    v = jnp.einsum("btl,hld->bthd", kv_f32, w_vc_f32)
    v = jnp.transpose(v, (0, 2, 1, 3))

    dk_nope_all, dv_all, db_k_all = _flash_mla_bwd_dkv(
        q,
        k_nope,
        v,
        b_q,
        b_k,
        bias,
        l,
        m,
        do,
        di,
        rope_mode=rope_mode,
        d_nope=d_nope,
        gqa_ratio=gqa_ratio,
        block_q=block_q,
        block_k=block_k,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        causal=causal,
    )

    dq, dbias_out, db_q_all = _flash_mla_bwd_dq(
        q,
        k_nope,
        v,
        b_q,
        b_k,
        bias,
        l,
        m,
        do,
        di,
        rope_mode=rope_mode,
        d_nope=d_nope,
        gqa_ratio=gqa_ratio,
        block_q=block_q,
        block_k=block_k,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        causal=causal,
    )

    if gqa_ratio > 1:
        dk_nope_kv = dk_nope_all.reshape(batch_size, kv_heads, gqa_ratio, kv_len, d_nope).sum(axis=2)
        dv_kv = dv_all.reshape(batch_size, kv_heads, gqa_ratio, kv_len, v_head_dim).sum(axis=2)
    else:
        dk_nope_kv = dk_nope_all
        dv_kv = dv_all

    dk_nope_for_proj = jnp.transpose(dk_nope_kv, (0, 2, 1, 3))
    dv_for_proj = jnp.transpose(dv_kv, (0, 2, 1, 3))

    dw_kc = jnp.einsum("btl,bthd->hld", kv_f32, dk_nope_for_proj)
    dw_vc = jnp.einsum("btl,bthd->hld", kv_f32, dv_for_proj)
    dkv_from_k = jnp.einsum("bthd,hld->btl", dk_nope_for_proj, w_kc_f32)
    dkv_from_v = jnp.einsum("bthd,hld->btl", dv_for_proj, w_vc_f32)
    dkv_latent = (dkv_from_k + dkv_from_v).astype(kv_latent.dtype)

    dq = dq.astype(q.dtype)
    dw_kc = dw_kc.astype(w_kc.dtype)
    dw_vc = dw_vc.astype(w_vc.dtype)

    db_q = None
    db_k = None
    if rope_mode != ROPE_NONE and db_k_all is not None:
        db_k = db_k_all.sum(axis=1).astype(b_k.dtype)
    if rope_mode == ROPE_DECOUPLED and db_q_all is not None:
        db_q = db_q_all.sum(axis=1).astype(b_q.dtype)

    dbias = None
    if bias is not None and dbias_out is not None:
        dbias = dbias_out.astype(bias.dtype)

    return (dq, dkv_latent, dw_kc, dw_vc, db_q, db_k, dbias)
