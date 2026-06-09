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


"""Flash Attention backward pass TPU kernel implementation using Pallas.

This module provides the gradient computation kernels for Flash Attention on
Google TPUs. The backward pass is split into two main computations:
1. dKV kernel: Computes gradients with respect to keys and values
2. dQ kernel: Computes gradients with respect to queries

The backward pass follows the memory-efficient approach of the forward pass,
never materializing the full attention matrix. Instead, it recomputes attention
weights on-the-fly using the saved softmax statistics (l and m) from the
forward pass.

Algorithm overview:
1. Compute di = sum(O * dO) for each query position
2. For dK/dV: Iterate over query blocks, accumulating gradients
   - Recompute attention weights P from saved statistics
   - dV += P^T @ dO
   - dK += (dP * P)^T @ Q where dP = dO @ V^T - di
3. For dQ: Iterate over KV blocks, accumulating gradients
   - Recompute attention weights P
   - dQ += (dP * P) @ K

TPU-specific optimizations:
- Separate kernels for dKV and dQ to maximize parallelism
- Uses scratch memory to accumulate gradients across iterations
- Efficient memory prefetching with Pallas BlockSpecs
- Supports parallel execution across batch and head dimensions

Supported features:
- Causal and non-causal attention gradients
- Attention bias gradients (dAB)
- Segment ID masking
- Sliding window attention
- Logits soft capping gradient correction
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
    NUM_LANES,
    NUM_SUBLANES,
    TRANS_B_DIM_NUMBERS,
    BlockSizes,
    _verify_block,
    below_or_on_diag,
)


def _flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    softmax_scale: float,
    sliding_window,
    logits_soft_cap,
    causal: bool,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
):
    """Pallas kernel for computing dK and dV gradients.

    This kernel computes gradients with respect to keys (dK) and values (dV)
    by iterating over query blocks and accumulating contributions. For each
    query block, it recomputes attention weights and applies the chain rule.

    The gradient formulas are:
    - dV = P^T @ dO (attention weights transposed times output gradient)
    - dK = (dS * P)^T @ Q where dS = softmax_scale * (dP - di)

    Args:
        q_tile_ref: Query tile reference for gradient computation.
        k_tile_ref: Key tile reference (input, not modified).
        v_tile_ref: Value tile reference (input, not modified).
        ab_tile_ref: Optional attention bias tile.
        q_segment_ids_tile_ref: Optional query segment IDs.
        kv_segment_ids_tile_ref: Optional KV segment IDs.
        l_tile_ref: Saved softmax denominator from forward pass.
        m_tile_ref: Saved softmax max from forward pass.
        do_tile_ref: Output gradient tile reference.
        di_tile_ref: Precomputed sum(O * dO) for each position.
        dk_tile_ref: Output reference for key gradients.
        dv_tile_ref: Output reference for value gradients.
        dk_scratch_ref: VMEM scratch for accumulating dK.
        dv_scratch_ref: VMEM scratch for accumulating dV.
        softmax_scale: Scaling factor used in forward pass.
        sliding_window: Optional sliding window configuration.
        logits_soft_cap: Optional soft cap value for gradient correction.
        causal: Whether causal masking was applied.
        mask_value: Value used for masked positions.
        q_seq_len: Total query sequence length.
        block_q: Query block size for inner loop.
        block_k: Key block size for inner loop.
    """
    _, _, block_q_major, _ = q_tile_ref.shape
    _, _, block_k_major, _ = k_tile_ref.shape

    q_seq_index = pl.program_id(axis=3)
    kv_seq_index = pl.program_id(axis=2)

    @pl.when(q_seq_index == 0)
    def start_new_sequence():
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

    def q_body(j, _):
        start_q = j * block_q

        def k_body(i, _):
            start_k = i * block_k
            k = k_tile_ref[0, 0, pl.ds(start_k, block_k), :]
            v = v_tile_ref[0, 0, pl.ds(start_k, block_k), :]
            q = q_tile_ref[0, 0, pl.ds(start_q, block_q), :]
            l = l_tile_ref[0, 0, pl.ds(start_q, block_q), :]
            m = m_tile_ref[0, 0, pl.ds(start_q, block_q), :]
            do = do_tile_ref[0, 0, pl.ds(start_q, block_q), :]
            di = di_tile_ref[0, 0, pl.ds(start_q, block_q), :].astype(jnp.float32)

            logits = lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

            if ab_tile_ref is not None:
                ab = ab_tile_ref[
                    0,
                    0,
                    pl.dslice(j * block_q, block_q),
                    pl.dslice(i * block_k, block_k),
                ].astype(jnp.float32)
                logits += ab

            if softmax_scale != 1.0:
                logits *= softmax_scale

            softcap_tanh = None
            if logits_soft_cap is not None:
                softcap_tanh = jnp.tanh(logits / logits_soft_cap)
                logits = logits_soft_cap * softcap_tanh

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError()
                q_segment_ids = q_segment_ids_tile_ref[0, pl.ds(start_q, block_q), :]
                q_segment_ids = pltpu.repeat(q_segment_ids, repeats, axis=1)
                kv_segment_ids = kv_segment_ids_tile_ref[:, 0, pl.ds(start_k, block_k)]
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_index * block_q_major + start_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_index * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            if sliding_window is not None:
                window_left, window_right = sliding_window
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_index * block_q_major + start_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_index * block_k_major + start_k
                window_mask = (col_ids >= (row_ids - window_left)) & (col_ids <= (row_ids + window_right))
                mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

            logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

            p = jnp.exp(logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))
            p = p * pltpu.repeat(1 / l, block_k // MIN_BLOCK_SIZE, axis=1)
            dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
            dv_scratch_ref[pl.ds(start_k, block_k), :] += dv.astype(dv_scratch_ref.dtype)

            dp = lax.dot_general(do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
            ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

            if logits_soft_cap is not None:
                ds = ds * (1.0 - softcap_tanh * softcap_tanh)

            if softmax_scale != 1.0:
                ds = ds * softmax_scale

            dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
            dk_scratch_ref[pl.ds(start_k, block_k), :] += dk.astype(dk_scratch_ref.dtype)

        lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

    if causal:
        should_run = below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major)
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

    @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
    def end_of_q_sequence():
        dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
        dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)


def _flash_attention_bwd_dkv(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    softmax_scale: float,
    sliding_window,
    logits_soft_cap,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
):
    """Compute gradients with respect to keys and values.

    Sets up and executes the Pallas kernel for computing dK and dV gradients.
    The computation iterates over query blocks while processing each KV block,
    accumulating gradients in scratch memory.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias tensor.
        segment_ids: Optional SegmentIds for packed sequences.
        l: Saved softmax denominator from forward pass [batch, num_heads, q_seq_len].
        m: Saved softmax max from forward pass [batch, num_heads, q_seq_len].
        do: Output gradient [batch, num_heads, q_seq_len, head_dim].
        di: Precomputed sum(O * dO) [batch, num_heads, q_seq_len].
        block_q_major: Major query block size for memory prefetching.
        block_q: Inner query block size for computation.
        block_k_major: Major KV block size for memory prefetching.
        block_k: Inner KV block size for computation.
        softmax_scale: Scaling factor from forward pass.
        sliding_window: Optional sliding window configuration.
        logits_soft_cap: Optional soft cap value.
        causal: Whether causal masking was used.
        mask_value: Value for masked positions.

    Returns:
        Tuple of (dK, dV) gradient tensors with same shapes as k and v.
    """
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
    _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))

    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    grid = (
        batch_size,
        num_heads,
        kv_seq_len // block_k_major,
        q_seq_len // block_q_major,
    )

    def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        if causal:
            next_q_index = lax.select(
                below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                q_seq_index,
                0,
            )
        else:
            next_q_index = q_seq_index

        return (batch_index, head_index, next_q_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, head_index, kv_seq_index, _):
        return (batch_index, head_index, kv_seq_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, _, q_seq_index):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
            del head_index
            if causal:
                next_q_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                    q_seq_index,
                    0,
                )
            else:
                next_q_index = q_seq_index
            return (batch_index, next_q_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
            del head_index
            return (batch_index, 0, kv_seq_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), k.dtype),
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), v.dtype),
    ]

    def dkv_index_map(batch_index, head_index, kv_seq_index, _):
        return (batch_index, head_index, kv_seq_index, 0)

    dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
    out_specs = [dkv_spec, dkv_spec]
    scratch_shapes = [
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
    ]

    kernel = functools.partial(
        _flash_attention_dkv_kernel,
        block_q=block_q,  # type: ignore
        block_k=block_k,  # type: ignore
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        causal=causal,
        mask_value=mask_value,
        q_seq_len=q_seq_len,
    )
    name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dk, dv = pl.pallas_call(
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
        )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
        assert dk.shape == k.shape
        assert dv.shape == v.shape
    return dk, dv


def _flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    softmax_scale: float,
    sliding_window,
    logits_soft_cap,
    causal: bool,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
    """Pallas kernel for computing dQ gradients.

    This kernel computes gradients with respect to queries (dQ) by iterating
    over KV blocks and accumulating contributions. It also optionally computes
    gradients for the attention bias (dS/dAB).

    The gradient formula is:
    - dQ = (dS * P) @ K where dS = softmax_scale * (dP - di)

    Args:
        q_tile_ref: Query tile reference.
        k_tile_ref: Key tile reference.
        v_tile_ref: Value tile reference.
        ab_tile_ref: Optional attention bias tile.
        q_segment_ids_tile_ref: Optional query segment IDs.
        kv_segment_ids_tile_ref: Optional KV segment IDs.
        l_tile_ref: Saved softmax denominator.
        m_tile_ref: Saved softmax max.
        do_tile_ref: Output gradient tile.
        di_tile_ref: Precomputed sum(O * dO).
        dq_tile_ref: Output reference for query gradients.
        ds_tile_ref: Optional output for attention score gradients (dAB).
        dq_scratch_ref: VMEM scratch for accumulating dQ.
        softmax_scale: Scaling factor from forward pass.
        sliding_window: Optional sliding window configuration.
        logits_soft_cap: Optional soft cap for gradient correction.
        causal: Whether causal masking was used.
        mask_value: Value for masked positions.
        kv_seq_len: Total KV sequence length.
        block_k: KV block size for inner loop.
    """
    _, _, block_k_major, _ = k_tile_ref.shape
    _, _, block_q_major, _ = q_tile_ref.shape

    kv_seq_index = pl.program_id(axis=3)
    q_seq_index = pl.program_id(axis=2)

    @pl.when(kv_seq_index == 0)
    def start_new_sequence():
        dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

    def body(i, _):
        k_slice = pl.ds(i * block_k, block_k)
        q = q_tile_ref[0, 0, :, :]
        k = k_tile_ref[0, 0, k_slice, :]
        v = v_tile_ref[0, 0, k_slice, :]
        l = l_tile_ref[0, 0, :, :]
        m = m_tile_ref[0, 0, :, :]
        do = do_tile_ref[0, 0, :, :]
        di = di_tile_ref[0, 0, :].astype(jnp.float32)

        logits = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        if ab_tile_ref is not None:
            ab = ab_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)].astype(jnp.float32)
            logits += ab

        if softmax_scale != 1.0:
            logits *= softmax_scale

        softcap_tanh = None
        if logits_soft_cap is not None:
            softcap_tanh = jnp.tanh(logits / logits_soft_cap)
            logits = logits_soft_cap * softcap_tanh

        mask = None
        if q_segment_ids_tile_ref is not None:
            repeats, rem = divmod(block_k, NUM_LANES)
            if rem:
                raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
            q_segment_ids = pltpu.repeat(q_segment_ids_tile_ref[0], repeats, axis=1)
            kv_segment_ids = kv_segment_ids_tile_ref[:, 0, k_slice]
            mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

        if causal:
            mask_shape = (block_q_major, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_index * block_q_major
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_index * block_k_major + i * block_k
            causal_mask = col_ids <= row_ids
            mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

        if sliding_window is not None:
            window_left, window_right = sliding_window
            mask_shape = (block_q_major, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_index * block_q_major
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_index * block_k_major + i * block_k
            window_mask = (col_ids >= (row_ids - window_left)) & (col_ids <= (row_ids + window_right))
            mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

        logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

        p = jnp.exp(logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))
        p = p * pltpu.repeat(1 / l, block_k // MIN_BLOCK_SIZE, axis=1)

        dp = jax.lax.dot_general(
            do,
            v,
            TRANS_B_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

        if logits_soft_cap is not None:
            ds = ds * (1.0 - softcap_tanh * softcap_tanh)

        if softmax_scale != 1.0:
            ds = ds * softmax_scale

        if ds_tile_ref is not None:
            ds_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)] = ds.astype(ds_tile_ref.dtype)

        dq_scratch_ref[:, :] += lax.dot(
            ds.astype(k.dtype),
            k,
            preferred_element_type=jnp.float32,
        ).astype(dq_scratch_ref.dtype)

    if causal:
        should_run = below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major)
        should_not_run = lax.select(should_run, False, True)
    else:
        should_run = True
        should_not_run = False  # type: ignore

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

    @pl.when(should_not_run)
    def zero_out_ds():
        if ds_tile_ref is not None:
            ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

    @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
    def end_of_kv_sequence():
        dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    softmax_scale: float,
    sliding_window,
    logits_soft_cap,
    causal: bool,
    mask_value: float,
):
    """Compute gradients with respect to queries and attention bias.

    Sets up and executes the Pallas kernel for computing dQ and optionally
    dAB (attention bias gradient). The computation iterates over KV blocks
    while processing each query block.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias tensor.
        segment_ids: Optional SegmentIds for packed sequences.
        l: Saved softmax denominator from forward pass.
        m: Saved softmax max from forward pass.
        do: Output gradient tensor.
        di: Precomputed sum(O * dO).
        block_q_major: Query block size for memory prefetching.
        block_k_major: Major KV block size for memory prefetching.
        block_k: Inner KV block size for computation.
        softmax_scale: Scaling factor from forward pass.
        sliding_window: Optional sliding window configuration.
        logits_soft_cap: Optional soft cap value.
        causal: Whether causal masking was used.
        mask_value: Value for masked positions.

    Returns:
        Tuple of (dQ, dAB) where dAB is None if ab was None.
    """
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))

    di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

    grid = (
        batch_size,
        num_heads,
        q_seq_len // block_q_major,
        kv_seq_len // block_k_major,
    )

    def qo_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    do_spec = qo_spec

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            if causal:
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    out_specs = [
        dq_spec,
        dab_spec,
    ]
    scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

    kernel = functools.partial(
        _flash_attention_dq_kernel,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        causal=causal,
        mask_value=mask_value,
        block_k=block_k,  # type: ignore
        kv_seq_len=kv_seq_len,
    )
    name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dq, ds = pl.pallas_call(
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
        )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)

    return dq, ds


def _flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    softmax_scale: float,
    block_sizes: BlockSizes,
    sliding_window,
    logits_soft_cap,
    residuals,
    do,
):
    """VJP backward rule for Flash Attention.

    Computes ``di = sum(O * dO)``, then delegates to
    ``_flash_attention_bwd_dkv`` (for dK/dV) and ``_flash_attention_bwd_dq``
    (for dQ/dAB).  Returns gradients w.r.t. (q, k, v, ab, segment_ids).

    Args:
        save_residuals: Must be False — higher-order AD is not supported.
        causal: Whether causal masking was applied in the forward pass.
        softmax_scale: Scaling factor used in the forward pass.
        block_sizes: BlockSizes with backward-pass tile fields populated.
        sliding_window: Sliding-window config passed to backward kernels.
        logits_soft_cap: Soft-cap value used in the forward pass.
        residuals: Tuple ``(q, k, v, ab, segment_ids, o, l, m)`` saved by
            the forward rule.
        do: Upstream gradient of the attention output.

    Returns:
        Five-element tuple ``(dq, dk, dv, ds, None)`` where ``ds`` is the
        attention-bias gradient (or ``None`` if no bias was given) and the
        last element is the ``None`` gradient for ``segment_ids``.

    Raises:
        NotImplementedError: If ``save_residuals`` is True.
        ValueError: If ``block_sizes.has_backward_blocks`` is False.
    """
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    q, k, v, ab, segment_ids, o, l, m = residuals
    if not block_sizes.has_backward_blocks:
        raise ValueError("Program is being differentiated, but not all backward blocks are specified")

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

    dk, dv = _flash_attention_bwd_dkv(
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_major_dkv,
        block_k_major=block_sizes.block_k_major_dkv,
        block_k=block_sizes.block_k_dkv,
        block_q=block_sizes.block_q_dkv,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
    )

    dq, ds = _flash_attention_bwd_dq(
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_dq,
        block_k_major=block_sizes.block_k_major_dq,
        block_k=block_sizes.block_k_dq,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
    )
    return dq, dk, dv, ds, None
