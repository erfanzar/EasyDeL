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


"""Block-sparse attention backward pass Triton kernel implementation.

This module contains the Triton GPU kernels for the backward pass of block-sparse
attention. It computes gradients for queries (dQ), keys (dK), and values (dV) using
pre-computed sparse mask boundaries to skip irrelevant blocks.

The backward pass is organized into three phases:
    1. **Preprocessing** (``blocksparse_attn_bwd_preprocess``): Computes per-token delta
       values (element-wise product sum of output and output gradient).
    2. **dQ computation** (``blocksparse_attn_bwd_dq``): Iterates over sparse KV blocks
       to accumulate query gradients.
    3. **dK/dV computation** (``blocksparse_attn_bwd_dkdv``): Iterates over sparse query
       blocks to accumulate key and value gradients using atomic additions.

The entry point ``_bwd_blocksparse_attn_call`` orchestrates the full backward pass.
"""

import chex
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import ArrayLike

from ejkernel.callib import cdiv, next_power_of_2, strides_from_shape, triton_call
from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask
from ejkernel.ops import BwdParams, FwdParams

from ._utilities import make_causal_mask, make_segment_mask


@triton.jit
def blocksparse_attn_bwd_preprocess(
    Po,
    Do,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    delta,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Preprocessing kernel for block-sparse attention backward pass.

    Computes delta values (element-wise product sum of output and output gradients)
    needed for the backward pass gradient computation.

    Args:
        Po: Forward pass output tensor
        Do: Output gradient tensor
        stride_oz, stride_oh, stride_om, stride_ok: Output strides
        delta: Output buffer for delta values
        NUM_HEADS: Number of attention heads (constexpr)
        SEQ_LEN: Sequence length (constexpr)
        BLOCK_SIZE: Block size for computation (constexpr)
        HEAD_DIM: Head dimension (constexpr)
    """
    block_id = tl.program_id(0).to(tl.int64)
    batch_size_id = tl.program_id(1).to(tl.int64)
    num_heads_id = tl.program_id(2).to(tl.int64)

    init_offset = batch_size_id * stride_oz + num_heads_id * stride_oh

    block_arange = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    head_dim_arange = tl.arange(0, HEAD_DIM)

    offsets = init_offset + block_arange[:, None] * stride_om + head_dim_arange[None, :] * stride_ok

    valid = block_arange < SEQ_LEN
    out = tl.load(Po + offsets, mask=valid[:, None], other=0.0)
    dout = tl.load(Do + offsets, mask=valid[:, None], other=0.0).to(tl.float32)

    delta_val = tl.sum(out * dout, axis=1)

    delta_offsets = batch_size_id * (NUM_HEADS * SEQ_LEN) + num_heads_id * SEQ_LEN
    delta_offsets += block_arange

    tl.store(delta + delta_offsets, delta_val, mask=valid)


@triton.jit
def _blocksparse_attn_bwd_dkdv_inner(
    dk,
    dv,
    kv_segment_ids,
    kv_positions_block,
    query,
    q_segment_ids,
    q_positions,
    dout,
    lse,
    delta,
    key,
    value,
    qk_scale,
    logits_soft_cap,
    softmax_scale,
    stride_qm,
    stride_qk,
    stride_om,
    stride_ok,
    stride_lm,
    lower_bound,
    upper_bound,
    BLOCK_M: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_PAD: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    EVEN_QK_HEAD_DIMS: tl.constexpr,
    query_global_offset: tl.constexpr,
    kv_span: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
    USE_SEGMENT_MASK: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Inner loop for computing dK and dV gradients in backward pass.

    Accumulates gradients for key and value tensors across query blocks.
    When SOFTCAP is enabled, applies the Jacobian of the tanh soft cap function
    to ensure correct gradients.

    Args:
        dk: Accumulated key gradients [BLOCK_N, QK_HEAD_DIM]
        dv: Accumulated value gradients [BLOCK_N, V_HEAD_DIM]
        kv_segment_ids: KV segment IDs [BLOCK_N]
        kv_positions_block: KV positions [BLOCK_N]
        query: Query tensor pointer
        q_segment_ids: Query segment IDs pointer
        q_positions: Query positions pointer
        dout: Output gradients pointer
        lse: Log-sum-exp from forward pass pointer
        delta: Delta values for softmax gradients pointer
        key: Key tensor pointer
        value: Value tensor pointer
        qk_scale: QK scaling factor
        logits_soft_cap: Soft cap value. When SOFTCAP is True, Jacobian
            (1 - tanh²(x)) is applied to gradients for proper backpropagation
        softmax_scale: Softmax scaling factor
        stride_qm, stride_qk: Query strides
        stride_om, stride_ok: Output gradient strides
        stride_lm: LSE stride
        lower_bound: Lower query block index
        upper_bound: Upper query block index
        BLOCK_M: Query block size (constexpr)
        QK_HEAD_DIM: Query/key head dimension (constexpr)
        QK_HEAD_DIM_PAD: Padded head dimension (constexpr)
        V_HEAD_DIM: Value head dimension (constexpr)
        EVEN_QK_HEAD_DIMS: Whether head dims are power of 2 (constexpr)
        query_global_offset: Global query offset (constexpr)
        kv_span: KV indices in block (constexpr)
        USE_CAUSAL: Enable causal masking (constexpr)
        USE_SEGMENT_MASK: Enable segment masking (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)

    Returns:
        Updated (dk, dv) tuple
    """
    query_arange = lower_bound * BLOCK_M + tl.arange(0, BLOCK_M)
    value_head_dim_arange = tl.arange(0, V_HEAD_DIM)
    qk_head_dim_pad_arange = tl.arange(0, QK_HEAD_DIM_PAD)

    query_transpose_ptr = query + query_arange[None, :] * stride_qm + qk_head_dim_pad_arange[:, None] * stride_qk
    dout += query_arange[:, None] * stride_om + value_head_dim_arange[None, :] * stride_ok

    lse_offset = query_arange * stride_lm
    query_offset = 0
    output_offset = 0

    for query_block_id in range(lower_bound, upper_bound):
        if EVEN_QK_HEAD_DIMS:
            query_transpose = tl.load(query_transpose_ptr + query_offset)
        else:
            qk_head_dim_mask = qk_head_dim_pad_arange[:, None] < QK_HEAD_DIM
            query_transpose = tl.load(query_transpose_ptr + query_offset, mask=qk_head_dim_mask, other=0.0)

        attn_weights_transpose = tl.dot(key, query_transpose)

        query_seq_len_offset = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)

        if USE_SEGMENT_MASK:
            query_segment_ids_block = tl.load(q_segment_ids + query_seq_len_offset)
            mask = make_segment_mask(query_segment_ids_block, kv_segment_ids, True)
        if USE_CAUSAL:
            query_positions_block = tl.load(q_positions + query_seq_len_offset)
            causal_mask = make_causal_mask(query_positions_block, kv_positions_block, True)
            if USE_SEGMENT_MASK:
                mask = causal_mask & mask
            else:
                mask = causal_mask

        if SOFTCAP:
            LOG2_CONST: tl.constexpr = 1.4426950408889634
            s = attn_weights_transpose * softmax_scale
            x = s / logits_soft_cap
            exp_2x = tl.exp(2.0 * x)
            tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
            qk_after = (logits_soft_cap * tanh_x) * LOG2_CONST
            jac = (softmax_scale * LOG2_CONST) * (1.0 - tanh_x * tanh_x)
        else:
            qk_after = (attn_weights_transpose * qk_scale).to(tl.float32)
            jac = qk_scale

        lse_val = tl.load(lse + lse_offset)
        pT = tl.math.exp2(qk_after - lse_val[None, :])

        if USE_SEGMENT_MASK or USE_CAUSAL:
            pT = tl.where(mask, pT, 0.0)

        dout_val = tl.load(dout + output_offset)

        dv += tl.dot(pT.to(dout_val.type.element_ty), dout_val)

        delta_val = tl.load(delta + lse_offset)

        dpT = tl.dot(value, tl.trans(dout_val))
        dsT = pT * (dpT - delta_val[None, :])
        dsT = (dsT * jac).to(query.type.element_ty)

        dk += tl.dot(dsT, tl.trans(query_transpose))

        query_offset += BLOCK_M * stride_qm
        output_offset += BLOCK_M * stride_om
        lse_offset += BLOCK_M * stride_lm

    return dk, dv


@triton.jit
def _blocksparse_attn_bwd_dq_inner(
    dq,
    q_segment_ids,
    q_positions,
    key,
    value,
    kv_segment_ids,
    kv_positions,
    query,
    dout,
    lse,
    delta,
    logits_soft_cap,
    softmax_scale,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    lower_bound,
    upper_bound,
    BLOCK_N: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_PAD: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    EVEN_QK_HEAD_DIMS: tl.constexpr,
    kv_global_offset: tl.constexpr,
    query_span: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
    USE_SEGMENT_MASK: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Inner loop for computing dQ gradients in backward pass.

    Accumulates gradients for query tensor across KV blocks.
    When SOFTCAP is enabled, applies the Jacobian of the tanh soft cap function
    to ensure correct gradients.

    Args:
        dq: Accumulated query gradients [BLOCK_M, QK_HEAD_DIM]
        q_segment_ids: Query segment IDs [BLOCK_M]
        q_positions: Query positions [BLOCK_M]
        key: Key tensor pointer
        value: Value tensor pointer
        kv_segment_ids: KV segment IDs pointer
        kv_positions: KV positions pointer
        query: Query tensor pointer
        dout: Output gradients pointer
        lse: Log-sum-exp from forward pass pointer
        delta: Delta values for softmax gradients pointer
        logits_soft_cap: Soft cap value. When SOFTCAP is True, Jacobian
            (1 - tanh²(x)) is applied to gradients for proper backpropagation
        softmax_scale: Softmax scaling factor
        stride_kn, stride_kk: Key strides
        stride_vn, stride_vk: Value strides
        lower_bound: Lower KV block index
        upper_bound: Upper KV block index
        BLOCK_N: KV block size (constexpr)
        QK_HEAD_DIM: Query/key head dimension (constexpr)
        QK_HEAD_DIM_PAD: Padded head dimension (constexpr)
        V_HEAD_DIM: Value head dimension (constexpr)
        EVEN_QK_HEAD_DIMS: Whether head dims are power of 2 (constexpr)
        kv_global_offset: Global KV offset (constexpr)
        query_span: Query indices in block (constexpr)
        USE_CAUSAL: Enable causal masking (constexpr)
        USE_SEGMENT_MASK: Enable segment masking (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)

    Returns:
        Updated dq
    """
    kv_arange = lower_bound * BLOCK_N + tl.arange(0, BLOCK_N)
    qk_head_dim_pad_arange = tl.arange(0, QK_HEAD_DIM_PAD)
    value_head_dim_arange = tl.arange(0, V_HEAD_DIM)

    k_offsets = (kv_arange[None, :] * stride_kn + qk_head_dim_pad_arange[:, None] * stride_kk).to(tl.int64)
    v_offsets = (kv_arange[None, :] * stride_vn + value_head_dim_arange[:, None] * stride_vk).to(tl.int64)

    key_transpose_ptr = key + k_offsets
    value_transpose_ptr = value + v_offsets

    for kv_block_id in range(lower_bound, upper_bound):
        if EVEN_QK_HEAD_DIMS:
            key_transpose = tl.load(key_transpose_ptr)
        else:
            key_transpose = tl.load(
                key_transpose_ptr,
                mask=qk_head_dim_pad_arange[:, None] < QK_HEAD_DIM,
                other=0.0,
            )
        value_transpose = tl.load(value_transpose_ptr)
        attention_weights = tl.dot(query, key_transpose)

        kv_leq_len_offset = kv_block_id * BLOCK_N + tl.arange(0, BLOCK_N)

        if USE_SEGMENT_MASK:
            kv_segment_ids_block = tl.load(kv_segment_ids + kv_leq_len_offset)
            mask = make_segment_mask(q_segment_ids, kv_segment_ids_block, False)
        if USE_CAUSAL:
            kv_positions_block = tl.load(kv_positions + kv_leq_len_offset)
            causal_mask = make_causal_mask(q_positions, kv_positions_block, False)
            if USE_SEGMENT_MASK:
                mask = causal_mask & mask
            else:
                mask = causal_mask

        if SOFTCAP:
            LOG2_CONST: tl.constexpr = 1.4426950408889634
            s = attention_weights / LOG2_CONST
            x = s / logits_soft_cap
            exp_2x = tl.exp(2.0 * x)
            tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
            qk_after = (logits_soft_cap * tanh_x) * LOG2_CONST
            jac = (softmax_scale * LOG2_CONST) * (1.0 - tanh_x * tanh_x)
        else:
            qk_after = attention_weights
            jac = 1.0

        p = tl.math.exp2(qk_after - lse)

        if USE_SEGMENT_MASK or USE_CAUSAL:
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(dout, value_transpose).to(tl.float32)
        ds = p * (dp - delta[:, None])
        ds = (ds * jac).to(key_transpose.type.element_ty)

        dq += tl.dot(ds, tl.trans(key_transpose))

        key_transpose_ptr += BLOCK_N * stride_kn
        value_transpose_ptr += BLOCK_N * stride_vn
    return dq


@triton.jit
def blocksparse_attn_bwd_dq(
    q,
    k,
    v,
    q_positions,
    q_segment_ids,
    kv_positions,
    kv_segment_ids,
    dout,
    lse,
    delta,
    lower_blocks,
    upper_blocks,
    lower_full_blocks,
    upper_full_blocks,
    query_global_offset,
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
    stride_lz,
    stride_lh,
    stride_lm,
    dq,
    Q_SEQ_LEN: tl.constexpr,
    KV_SEQ_LEN: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_PAD: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    IS_CONTEXT_PARALLELISM: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Triton kernel for computing dQ gradients in block-sparse backward pass.

    Computes query gradients by iterating over sparse KV blocks as determined
    by the pre-computed mask boundaries. Uses the same block-sparse pattern
    as the forward pass for efficiency.

    Args:
        q, k, v: Input tensor pointers from forward pass
        q_positions, q_segment_ids: Query positions and segment IDs
        kv_positions, kv_segment_ids: Key/value positions and segment IDs
        dout: Output gradient pointer
        lse: Log-sum-exp values from forward pass
        delta: Delta values for efficient gradient computation
        lower_blocks, upper_blocks: Sparse mask boundaries
        lower_full_blocks, upper_full_blocks: Fully-attended block boundaries
        query_global_offset: Global query offset for context parallelism
        softmax_scale: Attention scaling factor
        logits_soft_cap: Soft cap value for logits
        stride_*: Tensor strides
        dq: Output gradient pointer for queries
        Q_SEQ_LEN, KV_SEQ_LEN: Sequence lengths (constexpr)
        QK_HEAD_DIM, V_HEAD_DIM: Head dimensions (constexpr)
        BLOCK_M, BLOCK_N: Block sizes (constexpr)
        NUM_GROUPS: Number of head groups (constexpr)
        IS_CONTEXT_PARALLELISM: Enable context parallelism (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)
    """
    LOG2_CONST: tl.constexpr = 1.4426950408889634
    query_block_id = tl.program_id(0)
    batch_size_id = tl.program_id(1).to(tl.int64)
    num_heads_id = tl.program_id(2).to(tl.int64)
    num_query_block_programs = tl.num_programs(0)

    mask_offset = batch_size_id * num_query_block_programs + query_block_id
    lower_bound = tl.load(lower_blocks + mask_offset)
    lower_full_bound = tl.load(lower_full_blocks + mask_offset)
    upper_full_bound = tl.load(upper_full_blocks + mask_offset)
    upper_bound = tl.load(upper_blocks + mask_offset)

    num_blocks_to_attend = upper_bound - lower_bound

    EVEN_QK_HEAD_DIMS_LOCAL: tl.constexpr = QK_HEAD_DIM_PAD == QK_HEAD_DIM

    query_init_offset = (batch_size_id * stride_qz + num_heads_id * stride_qh).to(tl.int64)

    k_init_offset = (batch_size_id * stride_kz + (num_heads_id // NUM_GROUPS) * stride_kh).to(tl.int64)

    v_init_offset = (batch_size_id * stride_vz + (num_heads_id // NUM_GROUPS) * stride_vh).to(tl.int64)

    out_init_offset = (batch_size_id * stride_oz + num_heads_id * stride_oh).to(tl.int64)

    lse_init_offset = (batch_size_id * stride_lz + num_heads_id * stride_lh).to(tl.int64)

    qk_head_dim_pad_arange = tl.arange(0, QK_HEAD_DIM_PAD)
    value_head_dim_arange = tl.arange(0, V_HEAD_DIM)

    if not EVEN_QK_HEAD_DIMS_LOCAL:
        qk_head_dim_mask = qk_head_dim_pad_arange[None, :] < QK_HEAD_DIM

    q += query_init_offset
    k += k_init_offset
    v += v_init_offset
    dout += out_init_offset
    dq += query_init_offset
    lse += lse_init_offset
    delta += lse_init_offset

    q_segment_ids += batch_size_id * Q_SEQ_LEN
    kv_segment_ids += batch_size_id * KV_SEQ_LEN

    q_positions += batch_size_id * Q_SEQ_LEN
    kv_positions += batch_size_id * KV_SEQ_LEN

    if IS_CONTEXT_PARALLELISM:
        query_global_offs = tl.load(query_global_offset + query_block_id)
    else:
        query_global_offs = 0

    query_block_offset = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)

    query_offsets = query_block_offset[:, None] * stride_qm + qk_head_dim_pad_arange[None, :] * stride_qk

    out_offsets = query_block_offset[:, None] * stride_om + value_head_dim_arange[None, :] * stride_ok

    if num_blocks_to_attend == 0:
        dq_zero = tl.zeros([BLOCK_M, QK_HEAD_DIM_PAD], dtype=tl.float32)
        if EVEN_QK_HEAD_DIMS_LOCAL:
            tl.store(dq + query_offsets, dq_zero.to(dq.type.element_ty))
        else:
            tl.store(dq + query_offsets, dq_zero.to(dq.type.element_ty), mask=qk_head_dim_mask)
        return

    dout_val = tl.load(dout + out_offsets)
    dq_val = tl.zeros([BLOCK_M, QK_HEAD_DIM_PAD], dtype=tl.float32)
    qk_scale = (softmax_scale * LOG2_CONST).to(tl.float32)

    if EVEN_QK_HEAD_DIMS_LOCAL:
        query_val = tl.load(q + query_offsets)
    else:
        query_val = tl.load(q + query_offsets, mask=qk_head_dim_mask, other=0.0)

    lse_offset = query_block_offset * stride_lm
    lse_val = tl.load(lse + lse_offset)
    delta_val = tl.load(delta + lse_offset)

    lse_val = lse_val[:, None]
    query_val = (query_val * qk_scale).to(k.type.element_ty)

    query_segment_ids_block = tl.load(q_segment_ids + query_block_offset)
    query_positions_block = tl.load(q_positions + query_block_offset)

    dq_val = _blocksparse_attn_bwd_dq_inner(
        dq_val,
        query_segment_ids_block,
        query_positions_block,
        k,
        v,
        kv_segment_ids,
        kv_positions,
        query_val,
        dout_val,
        lse_val,
        delta_val,
        logits_soft_cap,
        softmax_scale,
        stride_kn,
        stride_kk,
        stride_vn,
        stride_vk,
        lower_bound,
        lower_full_bound,
        BLOCK_N=BLOCK_N,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_PAD=QK_HEAD_DIM_PAD,
        V_HEAD_DIM=V_HEAD_DIM,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS_LOCAL,
        kv_global_offset=0,
        query_span=query_global_offs + query_block_offset,
        USE_CAUSAL=False,
        USE_SEGMENT_MASK=True,
        SOFTCAP=SOFTCAP,
    )

    dq_val = _blocksparse_attn_bwd_dq_inner(
        dq_val,
        query_segment_ids_block,
        query_positions_block,
        k,
        v,
        kv_segment_ids,
        kv_positions,
        query_val,
        dout_val,
        lse_val,
        delta_val,
        logits_soft_cap,
        softmax_scale,
        stride_kn,
        stride_kk,
        stride_vn,
        stride_vk,
        lower_full_bound,
        upper_full_bound,
        BLOCK_N=BLOCK_N,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_PAD=QK_HEAD_DIM_PAD,
        V_HEAD_DIM=V_HEAD_DIM,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS_LOCAL,
        kv_global_offset=0,
        query_span=query_global_offs + query_block_offset,
        USE_CAUSAL=False,
        USE_SEGMENT_MASK=False,
        SOFTCAP=SOFTCAP,
    )

    dq_val = _blocksparse_attn_bwd_dq_inner(
        dq_val,
        query_segment_ids_block,
        query_positions_block,
        k,
        v,
        kv_segment_ids,
        kv_positions,
        query_val,
        dout_val,
        lse_val,
        delta_val,
        logits_soft_cap,
        softmax_scale,
        stride_kn,
        stride_kk,
        stride_vn,
        stride_vk,
        upper_full_bound,
        upper_bound,
        BLOCK_N=BLOCK_N,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_PAD=QK_HEAD_DIM_PAD,
        V_HEAD_DIM=V_HEAD_DIM,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS_LOCAL,
        kv_global_offset=0,
        query_span=query_global_offs + query_block_offset,
        USE_CAUSAL=True,
        USE_SEGMENT_MASK=True,
        SOFTCAP=SOFTCAP,
    )

    dq_val *= softmax_scale.to(tl.float32)

    if EVEN_QK_HEAD_DIMS_LOCAL:
        tl.store(dq + query_offsets, dq_val.to(dq.type.element_ty))
    else:
        tl.store(
            dq + query_offsets,
            dq_val.to(dq.type.element_ty),
            mask=qk_head_dim_mask,
        )


@triton.jit
def blocksparse_attn_bwd_dkdv(
    q,
    k,
    v,
    q_positions,
    q_segment_ids,
    kv_positions,
    kv_segment_ids,
    dout,
    lse,
    delta,
    lower_blocks,
    upper_blocks,
    lower_full_blocks,
    upper_full_blocks,
    query_global_offset,
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
    stride_lz,
    stride_lh,
    stride_lm,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvk,
    dk,
    dv,
    Q_SEQ_LEN: tl.constexpr,
    KV_SEQ_LEN: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_PAD: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOAD_SECOND_OFFSET: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    IS_CONTEXT_PARALLELISM: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Triton kernel for computing dK and dV gradients in block-sparse backward pass.

    Computes key and value gradients by iterating over sparse query blocks as
    determined by the pre-computed mask boundaries. Uses the transposed block-sparse
    pattern for efficient gradient accumulation.

    Args:
        q, k, v: Input tensor pointers from forward pass
        q_positions, q_segment_ids: Query positions and segment IDs
        kv_positions, kv_segment_ids: Key/value positions and segment IDs
        dout: Output gradient pointer
        lse: Log-sum-exp values from forward pass
        delta: Delta values for efficient gradient computation
        lower_blocks, upper_blocks: Sparse mask boundaries (transposed view)
        lower_full_blocks, upper_full_blocks: Fully-attended block boundaries
        query_global_offset: Global query offset for context parallelism
        softmax_scale: Attention scaling factor
        logits_soft_cap: Soft cap value for logits
        stride_*: Tensor strides for all inputs and outputs
        dk, dv: Output gradient pointers for keys and values
        Q_SEQ_LEN, KV_SEQ_LEN: Sequence lengths (constexpr)
        QK_HEAD_DIM, V_HEAD_DIM: Head dimensions (constexpr)
        BLOCK_M, BLOCK_N: Block sizes (constexpr)
        LOAD_SECOND_OFFSET: Secondary loading offset (constexpr)
        NUM_GROUPS: Number of head groups (constexpr)
        IS_CONTEXT_PARALLELISM: Enable context parallelism (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)
    """
    LOG2_CONST: tl.constexpr = 1.4426950408889634
    kv_block_id = tl.program_id(0)
    batch_size_id = tl.program_id(1).to(tl.int64)
    num_heads_id = tl.program_id(2).to(tl.int64)
    num_kv_block_programs = tl.num_programs(0)

    if IS_CONTEXT_PARALLELISM:
        query_global_offs = tl.load(query_global_offset)
        if LOAD_SECOND_OFFSET and kv_block_id * BLOCK_N - query_global_offs >= Q_SEQ_LEN // 2:
            query_global_offs = tl.load(query_global_offset + 1)
    else:
        query_global_offs = 0

    mask_offset = batch_size_id * num_kv_block_programs + kv_block_id
    lower_bound = tl.load(lower_blocks + mask_offset)
    lower_full_bound = tl.load(lower_full_blocks + mask_offset)
    upper_full_bound = tl.load(upper_full_blocks + mask_offset)
    upper_bound = tl.load(upper_blocks + mask_offset)

    EVEN_QK_HEAD_DIMS_LOCAL: tl.constexpr = QK_HEAD_DIM_PAD == QK_HEAD_DIM

    query_init_offset = (batch_size_id * stride_qz + num_heads_id * stride_qh).to(tl.int64)

    k_init_offset = (batch_size_id * stride_kz + (num_heads_id // NUM_GROUPS) * stride_kh).to(tl.int64)

    v_init_offset = (batch_size_id * stride_vz + (num_heads_id // NUM_GROUPS) * stride_vh).to(tl.int64)

    out_init_offset = (batch_size_id * stride_oz + num_heads_id * stride_oh).to(tl.int64)

    lse_init_offset = (batch_size_id * stride_lz + num_heads_id * stride_lh).to(tl.int64)

    dk_init_offset = (batch_size_id * stride_dkz + (num_heads_id // NUM_GROUPS) * stride_dkh).to(tl.int64)

    dv_init_offset = (batch_size_id * stride_dvz + (num_heads_id // NUM_GROUPS) * stride_dvh).to(tl.int64)

    q += query_init_offset
    k += k_init_offset
    v += v_init_offset
    dout += out_init_offset

    dk += dk_init_offset
    dv += dv_init_offset
    lse += lse_init_offset
    delta += lse_init_offset

    q_segment_ids += batch_size_id * Q_SEQ_LEN
    kv_segment_ids += batch_size_id * KV_SEQ_LEN

    q_positions += batch_size_id * Q_SEQ_LEN
    kv_positions += batch_size_id * KV_SEQ_LEN

    qk_head_dim_pad_arange = tl.arange(0, QK_HEAD_DIM_PAD)
    value_head_dim_arange = tl.arange(0, V_HEAD_DIM)
    qk_scale = (softmax_scale * LOG2_CONST).to(tl.float32)

    kv_block_offset = kv_block_id * BLOCK_N + tl.arange(0, BLOCK_N)

    dv_val = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)
    dk_val = tl.zeros([BLOCK_N, QK_HEAD_DIM_PAD], dtype=tl.float32)

    k_offsets = kv_block_offset[:, None] * stride_kn + qk_head_dim_pad_arange[None, :] * stride_kk

    v_offsets = kv_block_offset[:, None] * stride_vn + value_head_dim_arange[None, :] * stride_vk

    dk_offsets = kv_block_offset[:, None] * stride_dkn + qk_head_dim_pad_arange[None, :] * stride_dkk

    dv_offsets = kv_block_offset[:, None] * stride_dvn + value_head_dim_arange[None, :] * stride_dvk

    if EVEN_QK_HEAD_DIMS_LOCAL:
        key_val = tl.load(k + k_offsets)
    else:
        qk_head_dim_mask = qk_head_dim_pad_arange[None, :] < QK_HEAD_DIM
        key_val = tl.load(k + k_offsets, mask=qk_head_dim_mask, other=0.0)

    value_val = tl.load(v + v_offsets)

    kv_segment_ids_block = tl.load(kv_segment_ids + kv_block_offset)
    kv_positions_block = tl.load(kv_positions + kv_block_offset)

    dk_val, dv_val = _blocksparse_attn_bwd_dkdv_inner(
        dk_val,
        dv_val,
        kv_segment_ids_block,
        kv_positions_block,
        q,
        q_segment_ids,
        q_positions,
        dout,
        lse,
        delta,
        key_val,
        value_val,
        qk_scale,
        logits_soft_cap,
        softmax_scale,
        stride_qm,
        stride_qk,
        stride_om,
        stride_ok,
        stride_lm,
        lower_bound,
        lower_full_bound,
        BLOCK_M=BLOCK_M,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_PAD=QK_HEAD_DIM_PAD,
        V_HEAD_DIM=V_HEAD_DIM,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS_LOCAL,
        query_global_offset=query_global_offs,
        kv_span=kv_block_offset,
        USE_CAUSAL=True,
        USE_SEGMENT_MASK=True,
        SOFTCAP=SOFTCAP,
    )

    dk_val, dv_val = _blocksparse_attn_bwd_dkdv_inner(
        dk_val,
        dv_val,
        kv_segment_ids_block,
        kv_positions_block,
        q,
        q_segment_ids,
        q_positions,
        dout,
        lse,
        delta,
        key_val,
        value_val,
        qk_scale,
        logits_soft_cap,
        softmax_scale,
        stride_qm,
        stride_qk,
        stride_om,
        stride_ok,
        stride_lm,
        lower_full_bound,
        upper_full_bound,
        BLOCK_M=BLOCK_M,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_PAD=QK_HEAD_DIM_PAD,
        V_HEAD_DIM=V_HEAD_DIM,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS_LOCAL,
        query_global_offset=query_global_offs,
        kv_span=kv_block_offset,
        USE_CAUSAL=False,
        USE_SEGMENT_MASK=False,
        SOFTCAP=SOFTCAP,
    )

    dk_val, dv_val = _blocksparse_attn_bwd_dkdv_inner(
        dk_val,
        dv_val,
        kv_segment_ids_block,
        kv_positions_block,
        q,
        q_segment_ids,
        q_positions,
        dout,
        lse,
        delta,
        key_val,
        value_val,
        qk_scale,
        logits_soft_cap,
        softmax_scale,
        stride_qm,
        stride_qk,
        stride_om,
        stride_ok,
        stride_lm,
        upper_full_bound,
        upper_bound,
        BLOCK_M=BLOCK_M,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_PAD=QK_HEAD_DIM_PAD,
        V_HEAD_DIM=V_HEAD_DIM,
        EVEN_QK_HEAD_DIMS=EVEN_QK_HEAD_DIMS_LOCAL,
        query_global_offset=query_global_offs,
        kv_span=kv_block_offset,
        USE_CAUSAL=False,
        USE_SEGMENT_MASK=True,
        SOFTCAP=SOFTCAP,
    )

    dk_val *= softmax_scale.to(tl.float32)

    tl.atomic_add(dv + dv_offsets, dv_val.to(dv.type.element_ty))
    if EVEN_QK_HEAD_DIMS_LOCAL:
        tl.atomic_add(dk + dk_offsets, dk_val.to(dk.type.element_ty))
    else:
        tl.atomic_add(
            dk + dk_offsets,
            dk_val.to(dk.type.element_ty),
            mask=qk_head_dim_mask,
        )


@triton.jit
def blocksparse_attn_bwd(
    q,
    k,
    v,
    q_positions,
    q_segment_ids,
    kv_positions,
    kv_segment_ids,
    dout,
    lse,
    delta,
    lower_blocks_query,
    upper_blocks_query,
    lower_full_blocks_query,
    upper_full_blocks_query,
    lower_blocks_kv,
    upper_blocks_kv,
    lower_full_blocks_kv,
    upper_full_blocks_kv,
    query_global_offset,
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
    stride_lz,
    stride_lh,
    stride_lm,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvk,
    dq,
    dk,
    dv,
    Q_SEQ_LEN: tl.constexpr,
    KV_SEQ_LEN: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    QK_HEAD_DIM_PAD: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M_DKDV: tl.constexpr,
    BLOCK_N_DKDV: tl.constexpr,
    BLOCK_M_DQ: tl.constexpr,
    BLOCK_N_DQ: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Main Triton kernel wrapper for block-sparse attention backward pass.

    Computes gradients for query, key, and value tensors by calling specialized
    dQ and dKdV kernels. When SOFTCAP is enabled, properly handles the Jacobian
    of the tanh soft cap function for correct gradient computation.

    Args:
        q, k, v: Query, key, value tensors
        q_positions, q_segment_ids: Query position and segment IDs
        kv_positions, kv_segment_ids: KV position and segment IDs
        dout: Output gradient tensor
        lse: Log-sum-exp from forward pass
        delta: Precomputed delta values (out * dout sum)
        lower_blocks_query, upper_blocks_query: Query block bounds
        lower_full_blocks_query, upper_full_blocks_query: Query full block bounds
        lower_blocks_kv, upper_blocks_kv: KV block bounds
        lower_full_blocks_kv, upper_full_blocks_kv: KV full block bounds
        query_global_offset: Global query offsets for context parallelism
        softmax_scale: Softmax scaling factor
        logits_soft_cap: Soft cap value. When SOFTCAP is True, Jacobian
            (1 - tanh²(x)) is applied during gradient computation
        stride_qz, stride_qh, stride_qm, stride_qk: Query strides
        stride_kz, stride_kh, stride_kn, stride_kk: Key strides
        stride_vz, stride_vh, stride_vn, stride_vk: Value strides
        stride_oz, stride_oh, stride_om, stride_ok: Output gradient strides
        stride_lz, stride_lh, stride_lm: LSE strides
        stride_dkz, stride_dkh, stride_dkn, stride_dkk: dK strides
        stride_dvz, stride_dvh, stride_dvn, stride_dvk: dV strides
        dq, dk, dv: Output gradient buffers
        Q_SEQ_LEN: Query sequence length (constexpr)
        KV_SEQ_LEN: Key/value sequence length (constexpr)
        QK_HEAD_DIM: Query/key head dimension (constexpr)
        QK_HEAD_DIM_PAD: Padded head dimension (constexpr)
        V_HEAD_DIM: Value head dimension (constexpr)
        BLOCK_M_DKDV: Query block size for dKdV kernel (constexpr)
        BLOCK_N_DKDV: KV block size for dKdV kernel (constexpr)
        BLOCK_M_DQ: Query block size for dQ kernel (constexpr)
        BLOCK_N_DQ: KV block size for dQ kernel (constexpr)
        NUM_GROUPS: Number of query groups per KV head (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)
    """
    blocksparse_attn_bwd_dkdv(
        q,
        k,
        v,
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        dout,
        lse,
        delta,
        lower_blocks_kv,
        upper_blocks_kv,
        lower_full_blocks_kv,
        upper_full_blocks_kv,
        query_global_offset,
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
        stride_lz,
        stride_lh,
        stride_lm,
        stride_dkz,
        stride_dkh,
        stride_dkn,
        stride_dkk,
        stride_dvz,
        stride_dvh,
        stride_dvn,
        stride_dvk,
        dk,
        dv,
        Q_SEQ_LEN,
        KV_SEQ_LEN,
        QK_HEAD_DIM,
        QK_HEAD_DIM_PAD,
        V_HEAD_DIM,
        BLOCK_M_DKDV,
        BLOCK_N_DKDV,
        False,
        NUM_GROUPS,
        False,
        SOFTCAP,
    )

    blocksparse_attn_bwd_dq(
        q,
        k,
        v,
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        dout,
        lse,
        delta,
        lower_blocks_query,
        upper_blocks_query,
        lower_full_blocks_query,
        upper_full_blocks_query,
        query_global_offset,
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
        stride_lz,
        stride_lh,
        stride_lm,
        dq,
        Q_SEQ_LEN,
        KV_SEQ_LEN,
        QK_HEAD_DIM,
        QK_HEAD_DIM_PAD,
        V_HEAD_DIM,
        BLOCK_M_DQ,
        BLOCK_N_DQ,
        NUM_GROUPS,
        False,
        SOFTCAP,
    )


def _bwd_blocksparse_attn_call(
    softmax_scale: float,
    apply_load_balance: bool,
    sequence_parallelism_mesh_axis_name: str | None,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_params: FwdParams,
    bwd_params: BwdParams,
    logits_soft_cap: float | None,
    res: ArrayLike,
    dout: ArrayLike,
):
    """Execute block-sparse attention backward pass using Triton kernels.

    This function orchestrates the backward pass by dispatching to the
    preprocessing kernel, dQ kernel, and dK/dV kernel. Uses sparse masks
    to skip computation for masked-out blocks.

    Args:
        softmax_scale: Attention scaling factor
        apply_load_balance: Whether to apply load balancing
        sequence_parallelism_mesh_axis_name: Mesh axis for sequence parallelism
        window_left: Left window size for sliding window (-1 = unlimited)
        window_right: Right window size for sliding window (-1 = unlimited)
        causal: Enable causal masking
        fwd_params: Forward pass configuration parameters
        bwd_params: Backward pass configuration parameters
        logits_soft_cap: Optional soft cap for logits
        res: Residuals from forward pass (q, k, v, positions, segments, output, lse, mask)
        dout: Gradient of output tensor

    Returns:
        tuple: (dq, dk, dv, None...) gradients with respect to query, key, value
        and None for position/segment inputs
    """
    (
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
        _softmax_aux,
        _bias,
    ) = res
    qkv_layouts: tuple[SparseMask]

    BLOCK_M_DKDV = bwd_params.q_blocksize
    BLOCK_N_DKDV = bwd_params.kv_blocksize
    BLOCK_M_DQ = bwd_params.q_blocksize
    BLOCK_N_DQ = bwd_params.kv_blocksize

    chex.assert_rank([query, key, value, out, dout], 4)

    batch_size, num_heads, query_seq_len, qk_head_dim = query.shape
    _, num_kv_heads, kv_seq_len, value_head_dim = value.shape
    qk_head_dim_pad = next_power_of_2(qk_head_dim)

    num_groups = num_heads // num_kv_heads

    chex.assert_is_divisible(query_seq_len, bwd_params.q_blocksize)
    chex.assert_is_divisible(kv_seq_len, bwd_params.kv_blocksize)
    chex.assert_is_divisible(kv_seq_len, bwd_params.kv_blocksize)

    num_query_blocks = query_seq_len // BLOCK_M_DQ
    num_kv_blocks = kv_seq_len // BLOCK_N_DKDV
    grid = (num_kv_blocks, batch_size, num_heads)

    using_sequence_parallelism = sequence_parallelism_mesh_axis_name is not None

    use_separate_kernel_impl = True

    if using_sequence_parallelism:
        query_chunk_idx = jax.lax.axis_index(sequence_parallelism_mesh_axis_name)
        chex.assert_is_divisible(query_seq_len, BLOCK_M_DKDV)
        chex.assert_is_divisible(query_seq_len, BLOCK_M_DQ)

        if apply_load_balance:
            axis_size = jax.lax.psum(1, sequence_parallelism_mesh_axis_name)
            query_global_offset_dkdv = jnp.zeros((2), dtype=jnp.int32)
            query_global_offset_dq = jnp.zeros((query_seq_len // BLOCK_M_DQ), dtype=jnp.int32)

            query_global_offset_first_part = query_chunk_idx * query_seq_len // 2
            query_global_offset_second_part = (
                2 * axis_size - query_chunk_idx - 1
            ) * query_seq_len // 2 - query_seq_len // 2

            half_seq_len = 1
            query_global_offset_dkdv = query_global_offset_dkdv.at[:half_seq_len].set(query_global_offset_first_part)
            query_global_offset_dkdv = query_global_offset_dkdv.at[half_seq_len:].set(query_global_offset_second_part)

            half_seq_len = query_seq_len // BLOCK_M_DQ // 2
            query_global_offset_dq = query_global_offset_dq.at[:half_seq_len].set(query_global_offset_first_part)
            query_global_offset_dq = query_global_offset_dq.at[half_seq_len:].set(query_global_offset_second_part)

        else:
            query_global_offset_dkdv = query_chunk_idx * query_seq_len

            query_global_offset_dq = (
                jnp.ones((query_seq_len // BLOCK_M_DQ), dtype=jnp.int32) * query_chunk_idx * query_seq_len
            )
    else:
        query_global_offset_dkdv = jnp.zeros((1,), dtype=jnp.int32)
        query_global_offset_dq = jnp.zeros((query_seq_len // BLOCK_M_DQ), dtype=jnp.int32)

    if len(qkv_layouts) != 3:
        raise ValueError("Length of qkv_layouts should be equal to three")

    mask_dq = qkv_layouts[1]
    mask_dkdv = qkv_layouts[2]

    preprocess_block_size = 128

    seq_len_eff = out.shape[2]
    pre_grid = (cdiv(seq_len_eff, preprocess_block_size), batch_size, num_heads)

    out_shape = jax.ShapeDtypeStruct(shape=lse.shape, dtype=jnp.float32)

    delta = triton_call(
        out,
        dout,
        *strides_from_shape(out.shape),
        NUM_HEADS=num_heads,
        SEQ_LEN=seq_len_eff,
        kernel=blocksparse_attn_bwd_preprocess,
        out_shape=out_shape,
        grid=pre_grid,
        name="ejkernel::triton::blocksparse_attn_bwd_preprocess",
        BLOCK_SIZE=preprocess_block_size,
        HEAD_DIM=value_head_dim,
        num_warps=bwd_params.num_warps,
        num_stages=bwd_params.num_stages,
        zeroed_outputs=(0,),
    )

    if seq_len_eff != query_seq_len:
        pad_len = query_seq_len - seq_len_eff

        lse_padded = jnp.pad(lse, ((0, 0), (0, 0), (0, pad_len)), constant_values=0.0)
        delta_padded = jnp.pad(delta, ((0, 0), (0, 0), (0, pad_len)), constant_values=0.0)
        dout_padded = jnp.pad(dout, ((0, 0), (0, 0), (0, pad_len), (0, 0)), constant_values=0.0)
    else:
        lse_padded = lse
        delta_padded = delta
        dout_padded = dout

    if logits_soft_cap is None:
        logits_soft_cap_val = 0.0
        softcap_flag = False
    else:
        logits_soft_cap_val = float(logits_soft_cap)
        softcap_flag = True

    metaparams = dict(
        Q_SEQ_LEN=query_seq_len,
        KV_SEQ_LEN=kv_seq_len,
        QK_HEAD_DIM=qk_head_dim,
        QK_HEAD_DIM_PAD=qk_head_dim_pad,
        V_HEAD_DIM=value_head_dim,
        NUM_GROUPS=num_groups,
        SOFTCAP=softcap_flag,
        num_stages=bwd_params.num_stages,
        num_warps=bwd_params.num_warps,
    )

    dk_shape = (batch_size, num_kv_heads, kv_seq_len, qk_head_dim)
    dv_shape = (batch_size, num_kv_heads, kv_seq_len, value_head_dim)
    zeroed_outputs = (0, 1) if use_separate_kernel_impl else (1, 2)

    dq_shape_dtype = jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype)
    dk_shape_dtype = jax.ShapeDtypeStruct(shape=dk_shape, dtype=jnp.float32)
    dv_shape_dtype = jax.ShapeDtypeStruct(shape=dv_shape, dtype=jnp.float32)

    if use_separate_kernel_impl:
        out_shape = [dq_shape_dtype]
        dq_metaparams = dict(
            **metaparams,
            BLOCK_M=BLOCK_M_DQ,
            BLOCK_N=BLOCK_N_DQ,
            IS_CONTEXT_PARALLELISM=using_sequence_parallelism,
        )
        dq = triton_call(
            query,
            key,
            value,
            q_positions,
            q_segment_ids,
            kv_positions,
            kv_segment_ids,
            dout_padded,
            lse_padded,
            delta_padded,
            mask_dq.lower_bounds,
            mask_dq.upper_bounds,
            mask_dq.lower_full_bounds,
            mask_dq.upper_full_bounds,
            query_global_offset_dq,
            softmax_scale,
            logits_soft_cap_val,
            *strides_from_shape(query.shape),
            *strides_from_shape(key.shape),
            *strides_from_shape(value.shape),
            *strides_from_shape(dout_padded.shape),
            *strides_from_shape(lse_padded.shape),
            kernel=blocksparse_attn_bwd_dq,
            out_shape=out_shape,
            grid=(num_query_blocks, batch_size, num_heads),
            name="ejkernel::triton::blocksparse_attn_bwd_dq",
            **dq_metaparams,
        )[0]

        out_shape = [dk_shape_dtype, dv_shape_dtype]
        dkdv_metaparams = dict(
            **metaparams,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            IS_CONTEXT_PARALLELISM=using_sequence_parallelism,
            LOAD_SECOND_OFFSET=using_sequence_parallelism and apply_load_balance,
        )
        dk, dv = triton_call(
            query,
            key,
            value,
            q_positions,
            q_segment_ids,
            kv_positions,
            kv_segment_ids,
            dout_padded,
            lse_padded,
            delta_padded,
            mask_dkdv.lower_bounds,
            mask_dkdv.upper_bounds,
            mask_dkdv.lower_full_bounds,
            mask_dkdv.upper_full_bounds,
            query_global_offset_dkdv,
            softmax_scale,
            logits_soft_cap_val,
            *strides_from_shape(query.shape),
            *strides_from_shape(key.shape),
            *strides_from_shape(value.shape),
            *strides_from_shape(dout_padded.shape),
            *strides_from_shape(lse_padded.shape),
            *strides_from_shape(dk_shape),
            *strides_from_shape(dv_shape),
            kernel=blocksparse_attn_bwd_dkdv,
            out_shape=out_shape,
            grid=grid,
            name="ejkernel::triton::blocksparse_attn_bwd_dkdv",
            **dkdv_metaparams,
            zeroed_outputs=zeroed_outputs,
        )
    else:
        out_shape = [dq_shape_dtype, dk_shape_dtype, dv_shape_dtype]
        metaparams = dict(
            **metaparams,
            BLOCK_M_DKDV=BLOCK_M_DKDV,
            BLOCK_N_DKDV=BLOCK_N_DKDV,
            BLOCK_M_DQ=BLOCK_M_DQ,
            BLOCK_N_DQ=BLOCK_N_DQ,
        )
        query_global_offset = 0
        dq, dk, dv = triton_call(
            query,
            key,
            value,
            q_positions,
            q_segment_ids,
            kv_positions,
            kv_segment_ids,
            dout_padded,
            lse_padded,
            delta_padded,
            mask_dq.lower_bounds,
            mask_dq.upper_bounds,
            mask_dq.lower_full_bounds,
            mask_dq.upper_full_bounds,
            mask_dkdv.lower_bounds,
            mask_dkdv.upper_bounds,
            mask_dkdv.lower_full_bounds,
            mask_dkdv.upper_full_bounds,
            query_global_offset,
            softmax_scale,
            logits_soft_cap_val,
            *strides_from_shape(query.shape),
            *strides_from_shape(key.shape),
            *strides_from_shape(value.shape),
            *strides_from_shape(dout_padded.shape),
            *strides_from_shape(lse_padded.shape),
            *strides_from_shape(dk_shape),
            *strides_from_shape(dv_shape),
            kernel=blocksparse_attn_bwd,
            out_shape=out_shape,
            grid=grid,
            name="ejkernel::triton::blocksparse_attn_bwd",
            **metaparams,
            zeroed_outputs=zeroed_outputs,
        )

    return (
        dq.astype(query.dtype),
        dk.astype(key.dtype),
        dv.astype(value.dtype),
        None,
        None,
        None,
        None,
        (
            SparseMask(lower_bounds=None, upper_bounds=None, lower_full_bounds=None, upper_full_bounds=None),
            SparseMask(lower_bounds=None, upper_bounds=None, lower_full_bounds=None, upper_full_bounds=None),
            SparseMask(lower_bounds=None, upper_bounds=None, lower_full_bounds=None, upper_full_bounds=None),
        ),
        None,
        None,
    )
