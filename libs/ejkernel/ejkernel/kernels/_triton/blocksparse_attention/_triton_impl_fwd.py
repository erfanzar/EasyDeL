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


"""Block-sparse attention forward pass Triton kernel implementation.

This module contains the Triton GPU kernels for the forward pass of block-sparse
attention. The main kernel ``blocksparse_attn_fwd`` processes queries in blocks
and iterates over sparse key-value blocks determined by pre-computed mask boundaries,
skipping computation for entirely masked-out blocks.

The entry point ``_fwd_blocksparse_attn_call`` handles input validation, padding,
sparse mask extraction, and dispatches the Triton kernel with appropriate parameters.

Features:
    - Sparse block patterns (causal, sliding window, custom)
    - Grouped-query attention (GQA) / multi-query attention (MQA)
    - Segment-based masking for packed variable-length sequences
    - Context parallelism with load balancing
    - Logit soft capping (Gemma-2 style)
    - Attention sinks for streaming inference
"""

from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import ArrayLike

from ejkernel.callib import cdiv, next_power_of_2, strides_from_shape, triton_call
from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask
from ejkernel.ops import BwdParams, FwdParams

from ._utilities import make_causal_mask, make_segment_mask, make_sliding_window_mask, pad_to_block_size


@triton.jit
def _blocksparse_attn_fwd_inner(
    acc,
    l,
    m,
    query_block,
    q_segment_ids,
    query_positions_block,
    key_transpose_block_ptr,
    value_block_ptr,
    kv_segment_ids,
    kv_positions,
    softmax_aux_ptrs,
    lower,
    upper,
    softmax_scale,
    logits_soft_cap,
    query_span: tl.constexpr,
    kv_span: tl.constexpr,
    EVEN_QK_HEAD_DIMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
    USE_SEGMENT_MASK: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    USE_SINKS: tl.constexpr,
    NUM_SINKS: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Inner loop for block-sparse attention forward pass.

    Computes attention scores and accumulates weighted values for a range of KV blocks.
    Supports optional logit soft capping, causal masking, segment masking, sliding windows,
    and attention sinks.

    Args:
        acc: Accumulated attention output [BLOCK_M, V_HEAD_DIM]
        l: Normalization denominators for online softmax [BLOCK_M]
        m: Maximum logits for numerical stability [BLOCK_M]
        query_block: Query block [BLOCK_M, QK_HEAD_DIM]
        q_segment_ids: Query segment IDs [BLOCK_M]
        query_positions_block: Query positions [BLOCK_M]
        key_transpose_block_ptr: Block pointer to transposed keys
        value_block_ptr: Block pointer to values
        kv_segment_ids: Pointer to KV segment IDs
        kv_positions: Pointer to KV positions
        softmax_aux_ptrs: Pointer to auxiliary softmax values (attention sinks)
        lower: Lower bound KV block index
        upper: Upper bound KV block index
        softmax_scale: Scale factor for attention scores
        logits_soft_cap: Soft cap value for logits. When SOFTCAP is True, applies
            tanh-based capping: logits_soft_cap * tanh(qk / logits_soft_cap)
        query_span: Query indices in block [BLOCK_M] (constexpr)
        kv_span: KV indices in block [BLOCK_N] (constexpr)
        EVEN_QK_HEAD_DIMS: Whether head dims are power of 2 (constexpr)
        BLOCK_M: Query block size (constexpr)
        BLOCK_N: KV block size (constexpr)
        USE_CAUSAL: Enable causal masking (constexpr)
        USE_SEGMENT_MASK: Enable segment masking (constexpr)
        USE_SLIDING_WINDOW: Enable sliding window attention (constexpr)
        WINDOW_LEFT: Left window size (constexpr)
        WINDOW_RIGHT: Right window size (constexpr)
        USE_SINKS: Enable attention sinks (constexpr)
        NUM_SINKS: Number of sink tokens (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)

    Returns:
        Updated (acc, l, m) tuple
    """
    BIG_NEG: tl.constexpr = float("-inf")
    LOG2_CONST: tl.constexpr = 1.4426950408889634
    key_transpose_block_ptr = tl.advance(key_transpose_block_ptr, (0, lower))
    value_block_ptr = tl.advance(value_block_ptr, (lower, 0))
    kv_arange = tl.arange(0, BLOCK_N)

    for kv_block_offset in range(lower, upper, BLOCK_N):
        if EVEN_QK_HEAD_DIMS:
            key_transpose_block = tl.load(key_transpose_block_ptr)
        else:
            key_transpose_block = tl.load(key_transpose_block_ptr, boundary_check=(0,), padding_option="zero")
        attn_weights = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        attn_weights += tl.dot(query_block, key_transpose_block)

        kv_arange_offsetted = kv_block_offset + kv_arange

        if USE_SEGMENT_MASK:
            kv_segment_ids_block = tl.load(kv_segment_ids + kv_arange_offsetted)
            mask = make_segment_mask(q_segment_ids, kv_segment_ids_block, False)
        if USE_CAUSAL:
            kv_positions_block = tl.load(kv_positions + kv_arange_offsetted)
            causal_mask = make_causal_mask(query_positions_block, kv_positions_block, False)
            if USE_SEGMENT_MASK:
                mask = causal_mask & mask
            else:
                mask = causal_mask
        if USE_SLIDING_WINDOW:
            if not USE_CAUSAL:
                kv_positions_block = tl.load(kv_positions + kv_arange_offsetted)
            window_mask = make_sliding_window_mask(
                query_positions_block, kv_positions_block, WINDOW_LEFT, WINDOW_RIGHT, False
            )
            if USE_SEGMENT_MASK or USE_CAUSAL:
                mask = window_mask & mask
            else:
                mask = window_mask

        if SOFTCAP:
            softmax_scale_e = softmax_scale / LOG2_CONST
            x = (attn_weights * softmax_scale_e) / logits_soft_cap
            exp_2x = tl.exp(2.0 * x)
            tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
            attn_weights = (logits_soft_cap * tanh_x) * LOG2_CONST
        else:
            if USE_SEGMENT_MASK or USE_CAUSAL or USE_SLIDING_WINDOW:
                attn_weights = tl.where(mask, attn_weights * softmax_scale, BIG_NEG)
            else:
                attn_weights = attn_weights * softmax_scale

        if SOFTCAP and (USE_SEGMENT_MASK or USE_CAUSAL or USE_SLIDING_WINDOW):
            attn_weights = tl.where(mask, attn_weights, BIG_NEG)

        if USE_SINKS:
            sink_offs = tl.arange(0, 16)
            sink_mask = sink_offs < NUM_SINKS
            aux_logits = tl.load(softmax_aux_ptrs + sink_offs, mask=sink_mask, other=float("-inf"))
            aux_log2 = aux_logits * LOG2_CONST

            qk_max = tl.max(attn_weights, 1)
            aux_max = tl.max(tl.where(sink_mask, aux_log2, float("-inf")))
            m_new = tl.maximum(tl.maximum(qk_max, aux_max), m)
            m_new_safe = tl.where(m_new == BIG_NEG, 0.0, m_new)

            attn_weights -= m_new_safe[:, None]
            p = tl.math.exp2(attn_weights)

            aux_logits_row = tl.where(sink_mask[None, :], aux_log2[None, :], float("-inf"))
            l_aux_row = tl.sum(tl.exp2(aux_logits_row - m_new_safe[:, None]), axis=1)

            l_new = tl.sum(p, axis=1) + l_aux_row
        else:
            m_new = tl.maximum(m, tl.max(attn_weights, axis=1))
            m_new_safe = tl.where(m_new == BIG_NEG, 0.0, m_new)
            attn_weights -= m_new_safe[:, None]
            p = tl.math.exp2(attn_weights)
            l_new = tl.sum(p, axis=1)

        m_safe = tl.where(m_new == BIG_NEG, 0.0, m)
        alpha = tl.math.exp2(m_safe - m_new_safe)
        l = l * alpha + l_new
        acc *= alpha[:, None]
        value = tl.load(value_block_ptr)
        acc += tl.dot(p.to(value_block_ptr.type.element_ty), value)
        m = m_new
        value_block_ptr = tl.advance(value_block_ptr, (BLOCK_N, 0))
        key_transpose_block_ptr = tl.advance(key_transpose_block_ptr, (0, BLOCK_N))
    return acc, l, m


@triton.jit
def blocksparse_attn_fwd(
    q,
    k,
    v,
    q_positions,
    q_segment_ids,
    kv_positions,
    kv_segment_ids,
    lower_blocks,
    upper_blocks,
    lower_full_blocks,
    upper_full_blocks,
    query_global_offset,
    softmax_aux,
    softmax_scale,
    logits_soft_cap,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_aux_heads,
    stride_aux_sinks,
    Po,
    M,
    NUM_HEADS: tl.constexpr,
    Q_SEQ_LEN: tl.constexpr,
    KV_SEQ_LEN: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_PAD: tl.constexpr,
    EVEN_QK_HEAD_DIMS: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    IS_CONTEXT_PARALLELISM: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_SINKS: tl.constexpr,
    NUM_SINKS: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Triton kernel for block-sparse attention forward pass.

    Computes multi-head attention with sparse block patterns for efficient long-context processing.
    Supports grouped-query attention, causal masking, sliding windows, attention sinks, and
    logit soft capping for numerical stability.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        q_positions: Query position indices
        q_segment_ids: Query segment IDs
        kv_positions: KV position indices
        kv_segment_ids: KV segment IDs
        lower_blocks: Lower bounds for sparse block pattern
        upper_blocks: Upper bounds for sparse block pattern
        lower_full_blocks: Lower bounds for fully-masked blocks
        upper_full_blocks: Upper bounds for fully-masked blocks
        query_global_offset: Global query offsets for context parallelism
        softmax_aux: Auxiliary softmax values (attention sinks)
        softmax_scale: Scale factor for attention scores
        logits_soft_cap: Soft cap value for attention logits. When SOFTCAP is True,
            applies tanh-based capping: logits_soft_cap * tanh(qk / logits_soft_cap)
            to prevent scores from becoming too large (Gemma-2 style)
        stride_qz, stride_qh, stride_qm, stride_qk: Query tensor strides
        stride_kz, stride_kh, stride_kn, stride_kk: Key tensor strides
        stride_vz, stride_vh, stride_vn, stride_vk: Value tensor strides
        stride_oz, stride_oh, stride_om, stride_ok: Output tensor strides
        stride_aux_heads, stride_aux_sinks: Auxiliary tensor strides
        Po: Output tensor
        M: Log-sum-exp tensor for backward pass
        NUM_HEADS: Number of attention heads (constexpr)
        Q_SEQ_LEN: Query sequence length (constexpr)
        KV_SEQ_LEN: Key/value sequence length (constexpr)
        QK_HEAD_DIM: Query/key head dimension (constexpr)
        QK_HEAD_DIM_PAD: Padded head dimension (constexpr)
        EVEN_QK_HEAD_DIMS: Whether head dims are power of 2 (constexpr)
        V_HEAD_DIM: Value head dimension (constexpr)
        BLOCK_M: Query block size (constexpr)
        BLOCK_N: KV block size (constexpr)
        NUM_GROUPS: Number of query groups per KV head (constexpr)
        IS_CONTEXT_PARALLELISM: Enable context parallelism (constexpr)
        WINDOW_LEFT: Left window size for sliding window (constexpr)
        WINDOW_RIGHT: Right window size for sliding window (constexpr)
        CAUSAL: Enable causal masking (constexpr)
        USE_SINKS: Enable attention sinks (constexpr)
        NUM_SINKS: Number of sink tokens (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)
    """
    LOG2_CONST: tl.constexpr = 1.4426950408889634
    query_block_id = tl.program_id(0)
    batch_heads_size_id = tl.program_id(1).to(tl.int64)
    batch_size_id = (batch_heads_size_id // NUM_HEADS).to(tl.int64)
    num_query_block_programs = tl.num_programs(0)

    mask_offset = batch_size_id * num_query_block_programs + query_block_id
    lower_bound = tl.load(lower_blocks + mask_offset)
    lower_full_bound = tl.load(lower_full_blocks + mask_offset)
    upper_full_bound = tl.load(upper_full_blocks + mask_offset)
    upper_bound = tl.load(upper_blocks + mask_offset)

    query_offset = batch_heads_size_id * stride_qh

    k_offset = (batch_heads_size_id // NUM_GROUPS).to(tl.int64) * stride_kh
    v_offset = (batch_heads_size_id // NUM_GROUPS).to(tl.int64) * stride_vh
    out_offset = batch_heads_size_id * stride_oh

    if USE_SINKS:
        head_id = batch_heads_size_id - batch_size_id * NUM_HEADS
        aux_offset = head_id * stride_aux_heads
        softmax_aux_ptrs = softmax_aux + aux_offset
    else:
        softmax_aux_ptrs = softmax_aux

    query_block_ptr = tl.make_block_ptr(
        base=q + query_offset,
        shape=(Q_SEQ_LEN, QK_HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(query_block_id * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_HEAD_DIM_PAD),
        order=(1, 0),
    )

    key_transpose_block_ptr = tl.make_block_ptr(
        base=k + k_offset,
        shape=(QK_HEAD_DIM, KV_SEQ_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(QK_HEAD_DIM_PAD, BLOCK_N),
        order=(0, 1),
    )
    value_block_ptr = tl.make_block_ptr(
        base=v + v_offset,
        shape=(KV_SEQ_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=Po + out_offset,
        shape=(Q_SEQ_LEN, V_HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(query_block_id * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_HEAD_DIM),
        order=(1, 0),
    )

    if IS_CONTEXT_PARALLELISM:
        query_global_offs = tl.load(query_global_offset + query_block_id)
    else:
        query_global_offs = 0

    query_span = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_span = tl.arange(0, BLOCK_N)

    logsumexp_offset = batch_heads_size_id * Q_SEQ_LEN
    logsumexp_offset += query_span

    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

    num_blocks_to_attend = upper_bound - lower_bound
    if num_blocks_to_attend == 0:
        tl.store(out_block_ptr, acc.to(Po.type.element_ty))
        tl.store(M + logsumexp_offset, l)
        return

    q_scale = (softmax_scale * LOG2_CONST).to(tl.float32)

    if EVEN_QK_HEAD_DIMS:
        query_block = tl.load(query_block_ptr)
    else:
        query_block = tl.load(query_block_ptr, boundary_check=(1,), padding_option="zero")

    q_segment_ids += batch_size_id * Q_SEQ_LEN
    kv_segment_ids += batch_size_id * KV_SEQ_LEN

    q_positions += batch_size_id * Q_SEQ_LEN
    kv_positions += batch_size_id * KV_SEQ_LEN

    query_segment_ids_block = tl.load(q_segment_ids + query_span)
    query_positions_block = tl.load(q_positions + query_span)

    query_global_span = query_global_offs + query_span

    USE_SLIDING_WINDOW_LOCAL: tl.constexpr = WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0

    acc, l, m = _blocksparse_attn_fwd_inner(
        acc,
        l,
        m,
        query_block,
        query_segment_ids_block,
        query_positions_block,
        key_transpose_block_ptr,
        value_block_ptr,
        kv_segment_ids,
        kv_positions,
        softmax_aux_ptrs,
        lower_bound * BLOCK_N,
        lower_full_bound * BLOCK_N,
        q_scale,
        logits_soft_cap,
        query_global_span,
        kv_span,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_CAUSAL=False,
        USE_SEGMENT_MASK=True,
        USE_SLIDING_WINDOW=USE_SLIDING_WINDOW_LOCAL,
        WINDOW_LEFT=WINDOW_LEFT,
        WINDOW_RIGHT=WINDOW_RIGHT,
        USE_SINKS=USE_SINKS,
        NUM_SINKS=NUM_SINKS,
        SOFTCAP=SOFTCAP,
    )

    acc, l, m = _blocksparse_attn_fwd_inner(
        acc,
        l,
        m,
        query_block,
        query_segment_ids_block,
        query_positions_block,
        key_transpose_block_ptr,
        value_block_ptr,
        kv_segment_ids,
        kv_positions,
        softmax_aux_ptrs,
        lower_full_bound * BLOCK_N,
        upper_full_bound * BLOCK_N,
        q_scale,
        logits_soft_cap,
        query_global_span,
        kv_span,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_CAUSAL=False,
        USE_SEGMENT_MASK=False,
        USE_SLIDING_WINDOW=USE_SLIDING_WINDOW_LOCAL,
        WINDOW_LEFT=WINDOW_LEFT,
        WINDOW_RIGHT=WINDOW_RIGHT,
        USE_SINKS=USE_SINKS,
        NUM_SINKS=NUM_SINKS,
        SOFTCAP=SOFTCAP,
    )

    acc, l, m = _blocksparse_attn_fwd_inner(
        acc,
        l,
        m,
        query_block,
        query_segment_ids_block,
        query_positions_block,
        key_transpose_block_ptr,
        value_block_ptr,
        kv_segment_ids,
        kv_positions,
        softmax_aux_ptrs,
        upper_full_bound * BLOCK_N,
        upper_bound * BLOCK_N,
        q_scale,
        logits_soft_cap,
        query_global_span,
        kv_span,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_CAUSAL=CAUSAL,
        USE_SEGMENT_MASK=True,
        USE_SLIDING_WINDOW=USE_SLIDING_WINDOW_LOCAL,
        WINDOW_LEFT=WINDOW_LEFT,
        WINDOW_RIGHT=WINDOW_RIGHT,
        USE_SINKS=USE_SINKS,
        NUM_SINKS=NUM_SINKS,
        SOFTCAP=SOFTCAP,
    )

    invalid = l == 0.0
    l_safe = tl.where(invalid, 1.0, l)
    m = m + tl.math.log2(l_safe)
    acc = acc / l_safe[:, None]
    m = tl.where(invalid, 0.0, m)
    acc = tl.where(invalid[:, None], 0.0, acc)

    tl.store(M + logsumexp_offset, m)
    tl.store(out_block_ptr, acc.to(Po.type.element_ty))


def _fwd_blocksparse_attn_call(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    qkv_layouts: tuple[SparseMask] | None,
    softmax_scale: float,
    softmax_aux: ArrayLike | None,
    bias: ArrayLike | None,
    apply_load_balance: bool = True,
    sequence_parallelism_mesh_axis_name: str | None = None,
    window_left: int = -1,
    window_right: int = -1,
    causal: bool = True,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
) -> tuple[ArrayLike, Sequence[ArrayLike]]:
    """Execute block-sparse attention forward pass using Triton kernel.

    This function handles setup, validation, and dispatching for the block-sparse
    attention forward kernel. It uses sparse masks to skip computation for
    masked-out blocks, providing significant speedup for sparse attention patterns.

    Args:
        query: Query tensor [batch, num_heads, seq_len_q, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_len_k, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_len_k, head_dim]
        q_positions: Query position indices [batch, seq_len_q]
        q_segment_ids: Query segment IDs for packed sequences [batch, seq_len_q]
        kv_positions: Key/value position indices [batch, seq_len_k]
        kv_segment_ids: Key/value segment IDs [batch, seq_len_k]
        qkv_layouts: Pre-computed sparse masks or None to compute on-the-fly
        softmax_scale: Scaling factor for attention scores
        softmax_aux: Optional attention sink values
        bias: Attention bias (not currently supported)
        apply_load_balance: Whether to apply load balancing
        sequence_parallelism_mesh_axis_name: Mesh axis for sequence parallelism
        window_left: Left window size for sliding window (-1 = unlimited)
        window_right: Right window size for sliding window (-1 = unlimited)
        causal: Enable causal masking
        fwd_params: Forward pass configuration parameters
        bwd_params: Backward pass configuration parameters
        logits_soft_cap: Optional soft cap for logits

    Returns:
        tuple: (output, residuals) where output is attention result and
        residuals contain values needed for backward pass
    """
    if bias is not None:
        raise NotImplementedError("Bias is not supported in block-sparse attention")

    possible_hid_dim_vals = [16, 32, 64, 128, 192, 256]
    chex.assert_axis_dimension_comparator(
        query,
        axis=-1,
        pass_fn=lambda x: x in possible_hid_dim_vals,
        error_string=f"Attention hid_dim can take values {possible_hid_dim_vals}",
    )

    chex.assert_rank([query, key, value], 4)

    batch_size, num_heads, query_seq_len, qk_head_dim = query.shape
    _, num_kv_heads, kv_seq_len, value_head_dim = value.shape

    chex.assert_shape([key, value], [batch_size, num_kv_heads, kv_seq_len, None])
    chex.assert_shape([q_positions, q_segment_ids], [batch_size, query_seq_len])
    chex.assert_shape([kv_positions, kv_segment_ids], [batch_size, kv_seq_len])

    chex.assert_is_divisible(num_heads, num_kv_heads)

    if softmax_aux is None:
        use_sinks = False
        num_sinks_val = 0

        softmax_aux_tensor = jnp.zeros((num_heads, 1), dtype=query.dtype)
    else:
        use_sinks = True

        if softmax_aux.ndim == 1:
            num_sinks_val = softmax_aux.shape[0]
            softmax_aux_tensor = jnp.broadcast_to(softmax_aux[None, :], (num_heads, num_sinks_val))
        elif softmax_aux.ndim == 2:
            num_sinks_val = softmax_aux.shape[1]
            if softmax_aux.shape[0] == num_kv_heads:
                softmax_aux_tensor = jnp.repeat(softmax_aux, repeats=(num_heads // num_kv_heads), axis=0)
            elif softmax_aux.shape[0] == num_heads:
                softmax_aux_tensor = softmax_aux
            else:
                raise ValueError(
                    f"softmax_aux first dim must be num_kv_heads ({num_kv_heads}) or num_heads"
                    f" ({num_heads}), got {softmax_aux.shape[0]}"
                )
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {softmax_aux.shape}")

    using_sequence_parallelism = sequence_parallelism_mesh_axis_name is not None
    if using_sequence_parallelism:
        query_chunk_idx = jax.lax.axis_index(sequence_parallelism_mesh_axis_name)
        chex.assert_is_divisible(query_seq_len, fwd_params.q_blocksize)

        if apply_load_balance:
            axis_size = jax.lax.psum(1, sequence_parallelism_mesh_axis_name)
            query_global_offset_arange = jnp.zeros((query_seq_len // 2 // fwd_params.q_blocksize), dtype=jnp.int32)
            query_global_offset_first_part = query_chunk_idx * query_seq_len // 2 + query_global_offset_arange
            query_global_offset_second_part = (
                2 * axis_size - query_chunk_idx - 1
            ) * query_seq_len // 2 + query_global_offset_arange
            query_global_offset_second_part -= query_seq_len // 2
            query_global_offset = jnp.concatenate(
                [query_global_offset_first_part, query_global_offset_second_part],
                axis=0,
            )
        else:
            query_global_offset = (
                jnp.ones((query_seq_len // fwd_params.q_blocksize), dtype=jnp.int32) * query_chunk_idx * query_seq_len
            )
    else:
        query_global_offset = jnp.zeros((query_seq_len // fwd_params.q_blocksize), dtype=jnp.int32)

    (query,), q_positions, q_segment_ids = pad_to_block_size(
        inputs=(query,),
        indexs=q_positions,
        segment_ids=q_segment_ids,
        block_size=fwd_params.q_blocksize,
        pos_fill_value=-1,
        transposed_inputs=True,
    )
    (key, value), kv_positions, kv_segment_ids = pad_to_block_size(
        inputs=(key, value),
        indexs=kv_positions,
        segment_ids=kv_segment_ids,
        block_size=fwd_params.kv_blocksize,
        pos_fill_value=jnp.iinfo(jnp.int32).max,
        transposed_inputs=True,
    )
    orig_query_seq_len = query_seq_len
    query_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]

    num_query_blocks = cdiv(orig_query_seq_len, fwd_params.q_blocksize)

    grid = (num_query_blocks, batch_size * num_heads)

    if qkv_layouts is None or len(qkv_layouts) == 0:
        raise ValueError("Length of qkv_layouts should be at least equal to one")

    mask_fwd = qkv_layouts[0]

    num_groups = num_heads // num_kv_heads

    qk_head_dim_pad = next_power_of_2(qk_head_dim)

    if logits_soft_cap is None:
        logits_soft_cap_val = 0.0
        softcap_flag = False
    else:
        logits_soft_cap_val = float(logits_soft_cap)
        softcap_flag = True

    metaparams = dict(
        NUM_HEADS=num_heads,
        Q_SEQ_LEN=query_seq_len,
        KV_SEQ_LEN=kv_seq_len,
        QK_HEAD_DIM=qk_head_dim,
        QK_HEAD_DIM_PAD=qk_head_dim_pad,
        EVEN_QK_HEAD_DIMS=qk_head_dim_pad == qk_head_dim,
        V_HEAD_DIM=value_head_dim,
        BLOCK_M=fwd_params.q_blocksize,
        BLOCK_N=fwd_params.kv_blocksize,
        IS_CONTEXT_PARALLELISM=using_sequence_parallelism,
        NUM_GROUPS=num_groups,
        WINDOW_LEFT=window_left,
        WINDOW_RIGHT=window_right,
        CAUSAL=causal,
        USE_SINKS=use_sinks,
        NUM_SINKS=num_sinks_val,
        SOFTCAP=softcap_flag,
        num_stages=fwd_params.num_stages,
        num_warps=fwd_params.num_warps,
    )

    out_shape = (batch_size, num_heads, query_seq_len, value_head_dim)
    out, lse = triton_call(
        query,
        key,
        value,
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        mask_fwd.lower_bounds,
        mask_fwd.upper_bounds,
        mask_fwd.lower_full_bounds,
        mask_fwd.upper_full_bounds,
        query_global_offset,
        softmax_aux_tensor,
        softmax_scale,
        logits_soft_cap_val,
        *strides_from_shape(query.shape),
        *strides_from_shape(key.shape),
        *strides_from_shape(value.shape),
        *strides_from_shape(out_shape),
        *strides_from_shape(softmax_aux_tensor.shape),
        kernel=blocksparse_attn_fwd,
        out_shape=[
            jax.ShapeDtypeStruct(shape=out_shape, dtype=query.dtype),
            jax.ShapeDtypeStruct(shape=(batch_size, num_heads, query_seq_len), dtype=jnp.float32),
        ],
        grid=grid,
        name="ejkernel::triton::blocksparse_attn_fwd",
        **metaparams,
    )

    out = out[:, :, :orig_query_seq_len, :]
    lse = lse[:, :, :orig_query_seq_len]
    outputs_for_bwd_pass = [
        query,
        key,
        value,
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        qkv_layouts,
        out,
        lse,
        softmax_aux,
        bias,
    ]

    return out, outputs_for_bwd_pass
