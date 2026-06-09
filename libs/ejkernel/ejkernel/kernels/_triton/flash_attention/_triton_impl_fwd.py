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


"""Flash Attention forward pass Triton kernel implementation.

This module contains the Triton GPU kernels for the forward pass of Flash Attention.
It implements tiled, memory-efficient exact attention computation using the online
softmax algorithm to avoid materializing the full N x N attention matrix.

The main entry point is ``_fwd_attention_kernel_call``, which handles input
validation, stride computation, and dispatches the ``_attn_fwd`` Triton kernel.
The inner loop ``_attn_fwd_inner`` processes individual key-value blocks against
a query block, accumulating weighted values with numerically stable softmax.

Features handled by these kernels:
    - Causal and non-causal attention patterns
    - Sliding window (local) attention
    - Attention bias (additive or boolean mask)
    - Dropout with reproducible seeded masks
    - Variable-length sequences via cumulative sequence lengths
    - Grouped-query attention (GQA) / multi-query attention (MQA)
    - Logit soft capping (e.g. Gemma-2 style)
    - Attention sinks for streaming inference
    - Segment-based masking for packed sequences
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Bool, Float, Int
from triton import Config

from ejkernel.callib import triton_call
from ejkernel.ops import BwdParams, FwdParams
from ejkernel.utils import dtype_index, get_strides

from ._utilities import (
    attention_pack_from_cu_static,
    attention_pack_with_static_shape,
    attention_unpack_with_static_shape,
    calc_bias_strides,
    padded_load,
)


def config_prune_kernel(
    configs: list[Config],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Config]:
    """Prune autotuning configurations for forward pass kernel.

    Filters out configurations where block dimensions exceed sequence lengths.
    Falls back to small default configs if all configs are pruned.

    Args:
        configs: List of triton autotuning configurations
        named_args: Dictionary with kernel arguments including QSeq and KSeq
        **kwargs: Additional unused arguments

    Returns:
        list[Config]: Valid configurations for the given problem size
    """
    kept_configs = []
    for config in configs:
        largerst_m = config.kwargs["BLOCK_M"] > named_args["QSeq"]
        largerst_n = config.kwargs["BLOCK_N"] > named_args["KSeq"]
        if largerst_m or largerst_n:
            pass
        else:
            kept_configs.append(config)
    if kept_configs:
        return kept_configs
    return [
        Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=2, num_stages=4),
        Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=4),
        Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=2, num_stages=3),
        Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=3),
    ]


@triton.jit
def _attn_fwd_inner(
    q,
    m_i,
    me_i,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    acc_o,
    offs_m,
    offs_n,
    offs_d,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    dropout_offs,
    window_left,
    window_right,
    logits_soft_cap,
    softmax_aux_ptrs,
    num_sinks,
    stride_kn,
    stride_vn,
    index_start_n,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    q_segment_ids_ptr,
    kv_segment_ids_ptr,
    stride_qsm,
    stride_ksn,
    USE_SEGMENTS: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    MASKED: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Inner loop for flash attention forward pass.

    Computes attention scores and accumulates weighted values for a range of KV blocks.
    Uses online softmax algorithm for numerical stability without materializing full
    attention matrix. Supports optional bias, dropout, causal/sliding window masking,
    logit soft capping, and attention sinks.

    Args:
        q: Query block [BLOCK_M, head_dim]
        m_i: Maximum logits for numerical stability [BLOCK_M]
        me_i: Exponentiated max for rescaling [BLOCK_M]
        k_ptrs: Pointer to key tensor
        v_ptrs: Pointer to value tensor
        bias_ptrs: Pointer to optional bias tensor
        acc_o: Accumulated attention output [BLOCK_M, head_dim]
        offs_m: Query position offsets [BLOCK_M]
        offs_n: KV position offsets [BLOCK_N]
        offs_d: Head dimension offsets
        softmax_scale: Scale factor for attention scores (1/sqrt(d))
        dropout_prob: Dropout probability for attention weights
        dropout_seed: Random seed for dropout
        dropout_offs: Offset for dropout random generation
        window_left: Left window size for sliding window attention
        window_right: Right window size for sliding window attention
        logits_soft_cap: Soft cap value for logits (Gemma-2 style capping)
        softmax_aux_ptrs: Pointer to auxiliary softmax values (attention sinks)
        num_sinks: Number of sink tokens for attention sinks
        stride_kn, stride_vn: Key/value sequence strides
        index_start_n: Starting KV block index
        actual_seqlen_q, actual_seqlen_k: Actual sequence lengths
        headdim: Head dimension size
        q_segment_ids_ptr: Pointer to query segment IDs
        kv_segment_ids_ptr: Pointer to KV segment IDs
        stride_qsm, stride_ksn: Segment ID strides
        USE_SEGMENTS: Enable segment masking (constexpr)
        USE_DROPOUT: Enable dropout (constexpr)
        IS_CAUSAL: Enable causal masking (constexpr)
        BIAS_ON: Enable attention bias (constexpr)
        BOOL_BIAS: Treat bias as boolean mask (constexpr)
        MASKED: Apply masking logic (constexpr)
        SLIDING: Enable sliding window (constexpr)
        SOFTCAP: Enable logit soft capping (constexpr)
        USE_SINKS: Enable attention sinks (constexpr)
        PADDED_COLS: KV dimension needs padding check (constexpr)
        PADDED_HEADS: Head dimension needs padding check (constexpr)
        BLOCK_M: Query block size (constexpr)
        BLOCK_N: KV block size (constexpr)

    Returns:
        Updated (m_i, me_i, acc_o) tuple with updated softmax statistics and
        accumulated attention output
    """
    LN2: tl.constexpr = 1.44269504089
    index_start_n = tl.multiple_of(index_start_n, BLOCK_N)
    offset_k_ptrs = k_ptrs + index_start_n * stride_kn
    k = padded_load(
        offset_k_ptrs,
        index_start_n + offs_n,
        offs_d,
        PA0=PADDED_COLS,
        PA1=PADDED_HEADS,
        LA0=actual_seqlen_k,
        LA1=headdim,
    )
    if BIAS_ON:
        if PADDED_COLS:
            bias = tl.load(
                bias_ptrs + index_start_n,
                mask=(offs_m[:, None] < actual_seqlen_q) & ((index_start_n + offs_n) < actual_seqlen_k)[None, :],
                other=0.0,
            )
        else:
            bias = tl.load(
                bias_ptrs + index_start_n,
                mask=offs_m[:, None] < actual_seqlen_q,
                other=0.0,
            )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16)

    if USE_SINKS:
        qk += tl.dot(q.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16)))
    else:
        qk += tl.dot(q, tl.trans(k))

    if PADDED_COLS:
        qk += tl.where(
            (index_start_n + offs_n)[None, :] < actual_seqlen_k,
            0,
            float("-inf"),
        )

    if MASKED and IS_CAUSAL:
        causal_mask = offs_m[:, None] >= (index_start_n + offs_n - actual_seqlen_k + actual_seqlen_q)[None, :]
        qk += tl.where(causal_mask, 0, float("-inf"))

    if SLIDING:
        shift = actual_seqlen_k - actual_seqlen_q
        j_aligned = (index_start_n + offs_n)[None, :] - shift
        i_idx = offs_m[:, None]
        in_window = (j_aligned >= (i_idx - window_left)) & (j_aligned <= (i_idx + window_right))
        qk = tl.where(in_window, qk, float("-inf"))

    if BIAS_ON:
        if BOOL_BIAS:
            BIG_NEG: tl.constexpr = -2147483648
            qk = tl.where(bias, qk, BIG_NEG)
        else:
            qk += bias * (LN2 / softmax_scale)

    attn_mask = (offs_m[:, None] < actual_seqlen_q) & (offs_n[None, :] >= 0)
    if PADDED_COLS:
        attn_mask = attn_mask & ((index_start_n + offs_n)[None, :] < actual_seqlen_k)
    if MASKED and IS_CAUSAL:
        attn_mask = attn_mask & causal_mask
    if SLIDING:
        attn_mask = attn_mask & in_window
    if USE_SEGMENTS:
        q_ids = tl.load(q_segment_ids_ptr + offs_m * stride_qsm, mask=offs_m < actual_seqlen_q, other=-1)
        kv_ids = tl.load(
            kv_segment_ids_ptr + (index_start_n + offs_n) * stride_ksn,
            mask=(index_start_n + offs_n) < actual_seqlen_k,
            other=-1,
        )
        seg_mask = (q_ids[:, None] == kv_ids[None, :]) & (q_ids[:, None] >= 0)
        attn_mask = attn_mask & seg_mask
    if BIAS_ON and BOOL_BIAS:
        attn_mask = attn_mask & bias

    if SOFTCAP:
        qk_natural = qk * (softmax_scale / LN2)

        x = qk_natural / logits_soft_cap
        exp_2x = tl.exp(2.0 * x)
        tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
        qk = (logits_soft_cap * tanh_x) * LN2
    else:
        qk = qk * softmax_scale

    qk = tl.where(attn_mask, qk, float("-inf"))

    if USE_SINKS:
        sink_offs = tl.arange(0, 16)
        sink_mask = sink_offs < num_sinks
        aux_logits = tl.load(softmax_aux_ptrs + sink_offs, mask=sink_mask, other=float("-inf")).to(tl.bfloat16)

        if SOFTCAP:
            x_aux = aux_logits / logits_soft_cap
            x_aux = tl.maximum(tl.minimum(x_aux, 8.0), -8.0)
            exp_2x = tl.exp(2.0 * x_aux)
            tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
            aux_natural = logits_soft_cap * tanh_x
            aux_log2 = aux_natural * LN2
        else:
            aux_log2 = aux_logits * LN2

        qk_max = tl.max(qk, 1)
        aux_max = tl.max(tl.where(sink_mask, aux_log2, float("-inf")))
        m_ij = tl.maximum(tl.maximum(qk_max, aux_max), me_i)
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        P_ij = tl.exp2(qk - m_ij_safe[:, None])

        aux_log2_row = tl.where(sink_mask[None, :], aux_log2[None, :], float("-inf"))
        l_aux_row = tl.sum(tl.exp2(aux_log2_row - m_ij_safe[:, None]), axis=1)

        l_ij = tl.sum(P_ij, 1) + l_aux_row
    else:
        m_ij = tl.maximum(tl.max(qk, 1), me_i)
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        P_ij = tl.exp2(qk - m_ij_safe[:, None])
        l_ij = tl.sum(P_ij, 1)

    if USE_DROPOUT:
        dropout_offs = dropout_offs + index_start_n
        dropout_mask = tl.rand(dropout_seed, dropout_offs) > dropout_prob
        P_ij = tl.where(dropout_mask, P_ij, 0.0)

    acc_o_scale = tl.where(m_ij == float("-inf"), 0.0, tl.exp2(m_i - m_ij_safe))
    acc_o = acc_o * acc_o_scale[:, None]

    offset_v_ptrs = v_ptrs + index_start_n * stride_vn
    v = padded_load(
        offset_v_ptrs,
        index_start_n + offs_n,
        offs_d,
        PA0=PADDED_COLS,
        PA1=PADDED_HEADS,
        LA0=actual_seqlen_k,
        LA1=headdim,
    )

    acc_o += tl.dot(P_ij.to(tl.bfloat16), v.to(tl.bfloat16))
    m_i = m_ij
    l_i_new = tl.exp2(me_i - m_ij_safe) + l_ij
    me_i = m_ij_safe + tl.log2(l_i_new)
    return m_i, me_i, acc_o.to(tl.bfloat16)


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["QSeq"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["KSeq"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _attn_fwd(
    q,
    k,
    v,
    QSeg,
    KSeg,
    B,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    logits_soft_cap,
    softmax_aux,
    num_sinks,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_qsz,
    stride_qsm,
    stride_ksz,
    stride_ksn,
    stride_oz,
    stride_om,
    stride_oh,
    stride_bz,
    stride_bm,
    stride_bh,
    nheads_q,
    num_repeats,
    window_left,
    window_right,
    QSeq,
    cum_seqlens_q,
    KSeq,
    cum_seqlens_k,
    max_seqlen_q_rounded,
    headdim,
    CQSeq,
    CKSeq,
    DRuntime,
    Po,
    M,
    VARLEN: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    USE_SEGMENTS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Triton kernel for flash attention forward pass.

    Main kernel that orchestrates tiled computation of attention across blocks.
    Processes queries in blocks (axis 0 of the grid) and iterates through all
    key/value blocks, maintaining running softmax statistics (m, l) for
    numerical stability (online softmax).

    Grid shape: ``(ceil(QSeq / BLOCK_M), batch * nheads_q)``.

    Args:
        q: Query tensor pointer [batch, seq_q, nheads_q, head_dim]
        k: Key tensor pointer [batch, seq_k, nheads_kv, head_dim]
        v: Value tensor pointer [batch, seq_k, nheads_kv, head_dim]
        QSeg: Query segment IDs pointer [batch, seq_q] (or None)
        KSeg: Key segment IDs pointer [batch, seq_k] (or None)
        B: Bias tensor pointer [batch, nheads_q, seq_q, seq_k] (or None)
        softmax_scale: QK^T scaling factor (multiplied by log2(e) internally)
        dropout_prob: Dropout probability for attention weights
        dropout_seed: Integer seed for reproducible dropout
        logits_soft_cap: Tanh soft-cap value. 0.0 disables capping.
        softmax_aux: Attention sink logit pointer [nheads_q, num_sinks]
        num_sinks: Number of attention sink tokens
        stride_qz, stride_qm, stride_qh: Query strides (batch, seq, head)
        stride_kz, stride_kn, stride_kh: Key strides
        stride_vz, stride_vn, stride_vh: Value strides
        stride_qsz, stride_qsm: Query segment-ID strides
        stride_ksz, stride_ksn: Key segment-ID strides
        stride_oz, stride_om, stride_oh: Output strides
        stride_bz, stride_bm, stride_bh: Bias strides
        nheads_q: Number of query attention heads
        num_repeats: GQA/MQA ratio: nheads_q // nheads_kv
        window_left: Left sliding-window bound (-1 = disabled)
        window_right: Right sliding-window bound (-1 = disabled)
        QSeq: Runtime query sequence length
        cum_seqlens_q: Pointer to cumulative query lengths (variable-length mode)
        KSeq: Runtime key/value sequence length
        cum_seqlens_k: Pointer to cumulative key lengths (variable-length mode)
        max_seqlen_q_rounded: Query seq length rounded to BLOCK_M multiple
        headdim: Actual (unpadded) head dimension
        CQSeq: Compile-time query sequence length (constexpr)
        CKSeq: Compile-time key sequence length (constexpr)
        DRuntime: Runtime head dimension (same as headdim, passed as scalar)
        Po: Output tensor pointer [batch, seq_q, nheads_q, head_dim]
        M: Log-sum-exp output pointer [batch, nheads_q, seq_q] (float32)
        VARLEN: Enable variable-length mode (constexpr)
        USE_DROPOUT: Enable attention dropout (constexpr)
        IS_CAUSAL: Enable causal (lower-triangular) masking (constexpr)
        BIAS_ON: Enable bias addition (constexpr)
        SLIDING: Enable sliding-window attention (constexpr)
        SOFTCAP: Enable logit tanh soft-capping (constexpr)
        USE_SINKS: Enable attention sinks (constexpr)
        BOOL_BIAS: Treat bias as a boolean mask rather than float addend (constexpr)
        USE_SEGMENTS: Enable segment-based cross-sequence masking (constexpr)
        BLOCK_HEADDIM: Compile-time head dimension rounded to power of 2 (constexpr)
        PADDED_HEADS: True when head_dim is not a power of 2 (constexpr)
        EVEN_M: True when QSeq % BLOCK_M == 0 (heuristic, constexpr)
        EVEN_N: True when KSeq % BLOCK_N == 0 (heuristic, constexpr)
        BLOCK_M: Query tile size (constexpr)
        BLOCK_N: Key/value tile size (constexpr)
    """
    i_start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_head_q = off_zh % nheads_q
    off_head_kv = off_head_q // num_repeats
    off_z = off_zh // nheads_q

    if VARLEN:
        cu_q0 = tl.load(cum_seqlens_q + off_z)
        cu_q1 = tl.load(cum_seqlens_q + off_z + 1)
        cu_k0 = tl.load(cum_seqlens_k + off_z)
        cu_k1 = tl.load(cum_seqlens_k + off_z + 1)
        actual_seqlen_q = cu_q1 - cu_q0
        actual_seqlen_k = cu_k1 - cu_k0
        if i_start_m * BLOCK_M >= actual_seqlen_q:
            return
        cu_seq_start_q = cu_q0
        cu_seq_start_k = cu_k0
        off_z = 0
    else:
        actual_seqlen_q = QSeq
        actual_seqlen_k = KSeq
        cu_seq_start_q = 0
        cu_seq_start_k = 0

    LN2: tl.constexpr = 1.44269504089
    softmax_scale = softmax_scale * LN2

    offs_m = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    fully_masked_lines = (actual_seqlen_q - actual_seqlen_k) if IS_CAUSAL else 0
    if IS_CAUSAL and fully_masked_lines >= (i_start_m + 1) * BLOCK_M:
        return

    q_ptrs = (
        q
        + off_z * stride_qz
        + off_head_q * stride_qh
        + cu_seq_start_q * stride_qm
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        k
        + off_z * stride_kz
        + off_head_kv * stride_kh
        + cu_seq_start_k * stride_kn
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        v
        + off_z * stride_vz
        + off_head_kv * stride_vh
        + cu_seq_start_k * stride_vn
        + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    q_seg_ptrs = QSeg + off_z * stride_qsz + cu_seq_start_q * stride_qsm
    kv_seg_ptrs = KSeg + off_z * stride_ksz + cu_seq_start_k * stride_ksn

    if BIAS_ON:
        bias_ptrs = (
            B
            + off_z * stride_bz
            + off_head_kv * stride_bh
            + cu_seq_start_q * stride_bm
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    else:
        bias_ptrs = None

    if USE_DROPOUT:
        dropout_off = actual_seqlen_k * (cu_seq_start_q + actual_seqlen_q * (off_head_q + nheads_q * off_z))
        dropout_offs = dropout_off + offs_m[:, None] * actual_seqlen_k + offs_n[None, :]
    else:
        dropout_offs = None

    if USE_SINKS:
        softmax_aux_ptrs = softmax_aux + off_head_q * num_sinks
    else:
        softmax_aux_ptrs = softmax_aux

    me_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.bfloat16)

    pad_rows = (not EVEN_M) or (VARLEN and (i_start_m * BLOCK_M > actual_seqlen_q))
    q = padded_load(q_ptrs, offs_m, offs_d, PA0=pad_rows, PA1=PADDED_HEADS, LA0=actual_seqlen_q, LA1=headdim)

    if IS_CAUSAL:
        end_n = tl.minimum(actual_seqlen_k - actual_seqlen_q + (i_start_m + 1) * BLOCK_M, actual_seqlen_k)
        if end_n < 0:
            return
    else:
        end_n = actual_seqlen_k

    uneven_n = actual_seqlen_k % BLOCK_N != 0
    attention_padding = VARLEN & uneven_n

    if IS_CAUSAL:
        first_masked_col = i_start_m * BLOCK_M + 1 + actual_seqlen_k - actual_seqlen_q
    elif attention_padding:
        first_masked_col = actual_seqlen_k
    else:
        first_masked_col = end_n
    nb_full_blocks = first_masked_col // BLOCK_N

    next_start_n = 0
    if nb_full_blocks > 0:
        for _ in range(0, nb_full_blocks):
            m_i, me_i, acc_o = _attn_fwd_inner(
                q,
                m_i,
                me_i,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                acc_o,
                offs_m,
                offs_n,
                offs_d,
                softmax_scale,
                dropout_prob,
                dropout_seed,
                dropout_offs,
                window_left,
                window_right,
                logits_soft_cap,
                softmax_aux_ptrs,
                num_sinks,
                stride_kn,
                stride_vn,
                next_start_n,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                q_seg_ptrs,
                kv_seg_ptrs,
                stride_qsm,
                stride_ksn,
                USE_DROPOUT=USE_DROPOUT,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                MASKED=False,
                SLIDING=SLIDING,
                SOFTCAP=SOFTCAP,
                USE_SINKS=USE_SINKS,
                USE_SEGMENTS=USE_SEGMENTS,
                PADDED_COLS=False,
                PADDED_HEADS=PADDED_HEADS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            next_start_n += BLOCK_N

    if next_start_n < end_n:
        for index_start_n in range(next_start_n, end_n, BLOCK_N):
            pad_cols = (not EVEN_N) or VARLEN
            m_i, me_i, acc_o = _attn_fwd_inner(
                q,
                m_i,
                me_i,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                acc_o,
                offs_m,
                offs_n,
                offs_d,
                softmax_scale,
                dropout_prob,
                dropout_seed,
                dropout_offs,
                window_left,
                window_right,
                logits_soft_cap,
                softmax_aux_ptrs,
                num_sinks,
                stride_kn,
                stride_vn,
                index_start_n,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                q_seg_ptrs,
                kv_seg_ptrs,
                stride_qsm,
                stride_ksn,
                USE_DROPOUT=USE_DROPOUT,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                MASKED=True,
                SLIDING=SLIDING,
                SOFTCAP=SOFTCAP,
                USE_SINKS=USE_SINKS,
                USE_SEGMENTS=USE_SEGMENTS,
                PADDED_COLS=pad_cols,
                PADDED_HEADS=PADDED_HEADS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

    invalid = me_i == float("-inf")
    me_i = tl.where(invalid, 0.0, me_i)
    m_i = tl.where(invalid, 0.0, m_i)
    acc_o = tl.where(invalid[:, None], 0.0, acc_o)

    if USE_DROPOUT:
        o_scale = tl.exp2((m_i - me_i) - tl.log2(1 - dropout_prob))
    else:
        o_scale = tl.exp2(m_i - me_i)
    acc_o = acc_o * o_scale[:, None]
    if IS_CAUSAL and fully_masked_lines > i_start_m * BLOCK_M:
        acc_o = tl.where(offs_m[:, None] < fully_masked_lines, 0, acc_o)

    offs_m = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = M + off_zh * max_seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, me_i)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Po
        + off_z * stride_oz
        + off_head_q * stride_oh
        + cu_seq_start_q * stride_om
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim))


def _fwd_attention_kernel_call(
    q: Float[Array, "batch seq_len_q num_heads head_dim"] | None,
    k: Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    v: Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
) -> tuple[Float[Array, "batch seq_len_q num_heads head_dim"], Float[Array, "batch num_heads max_seqlen_q_rounded"]]:
    """Execute flash attention forward pass using Triton kernel.

    This function handles all setup, validation, and dispatching for the flash
    attention forward kernel. It supports multiple modes including variable-length
    sequences, attention masks, sliding windows, and more.

    Args:
        q: Query tensor of shape [batch, seq_len_q, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len_k, num_kv_heads, head_dim]
        v: Value tensor of shape [batch, seq_len_k, num_kv_heads, head_dim]
        attention_mask: Optional boolean or integer attention mask
        bias: Optional attention bias tensor
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability applied to attention weights
        causal: Whether to apply causal (autoregressive) masking
        dropout_seed: Random seed for dropout
        fwd_params: Forward pass configuration parameters
        bwd_params: Backward pass configuration parameters (for compatibility)
        cum_seqlens_q: Cumulative sequence lengths for variable-length queries
        cum_seqlens_k: Cumulative sequence lengths for variable-length keys
        sliding_window: Sliding window size (int or tuple of left/right)
        logits_soft_cap: Soft cap value for attention logits
        softmax_aux: Auxiliary softmax values for attention sinks
        q_segment_ids: Query segment IDs for packed sequences
        kv_segment_ids: Key/value segment IDs for packed sequences

    Returns:
        tuple: (output, lse) where output is attention result and lse is
            log-sum-exp values for backward pass
    """
    if sliding_window is None:
        window_left = 0
        window_right = 0
        sliding_flag = False
    else:
        if isinstance(sliding_window, int):
            window_left = int(sliding_window)
            window_right = 0 if causal else int(sliding_window)
        else:
            wl, wr = sliding_window
            window_left = int(wl)
            window_right = int(wr)
        assert window_left >= 0 and window_right >= 0
        sliding_flag = (window_left > 0) or (window_right > 0)

    if logits_soft_cap is None:
        logits_soft_cap_val = 0.0
        softcap_flag = False
    else:
        logits_soft_cap_val = float(logits_soft_cap)
        softcap_flag = True

    if softmax_aux is None:
        use_sinks = False
        num_sinks_val = 0
        softmax_aux_tensor = jnp.zeros((1,), dtype=q.dtype)
    else:
        use_sinks = True

        if softmax_aux.ndim == 1:
            num_sinks_val = softmax_aux.shape[0]
            num_heads = q.shape[2]
            softmax_aux_tensor = jnp.broadcast_to(softmax_aux[None, :], (num_heads, num_sinks_val))
        elif softmax_aux.ndim == 2:
            num_sinks_val = softmax_aux.shape[1]
            softmax_aux_tensor = softmax_aux
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {softmax_aux.shape}")

        pass

    use_segments = (q_segment_ids is not None) or (kv_segment_ids is not None)
    if use_segments:
        if q_segment_ids is None:
            q_segment_ids = kv_segment_ids
        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids
        q_segment_ids = jnp.asarray(q_segment_ids, dtype=jnp.int32)
        kv_segment_ids = jnp.asarray(kv_segment_ids, dtype=jnp.int32)
        if q_segment_ids.ndim != 2 or kv_segment_ids.ndim != 2:
            raise ValueError("q_segment_ids/kv_segment_ids must be 2D int32 arrays.")
        if q_segment_ids.shape[0] != q.shape[0] or q_segment_ids.shape[1] != q.shape[1]:
            raise ValueError("q_segment_ids must have shape [batch, seq_len_q].")
        if kv_segment_ids.shape[0] != k.shape[0] or kv_segment_ids.shape[1] != k.shape[1]:
            raise ValueError("kv_segment_ids must have shape [batch, seq_len_k].")
        qsz, qsm = get_strides(q_segment_ids.shape)
        ksz, ksn = get_strides(kv_segment_ids.shape)
    else:
        q_segment_ids = jnp.zeros((1,), dtype=jnp.int32)
        kv_segment_ids = jnp.zeros((1,), dtype=jnp.int32)
        qsz = qsm = ksz = ksn = 0

    if fwd_params is None:
        fwd_params = FwdParams()

    default_block_m = max(16, min(128, int(q.shape[1])))
    default_block_n = max(16, min(128, int(k.shape[1])))
    block_m = default_block_m if fwd_params.q_blocksize is None else max(16, int(fwd_params.q_blocksize))
    block_n = default_block_n if fwd_params.kv_blocksize is None else max(16, int(fwd_params.kv_blocksize))
    num_warps = 4 if fwd_params.num_warps is None else int(fwd_params.num_warps)
    num_stages = 2 if fwd_params.num_stages is None else int(fwd_params.num_stages)

    varlen_from_cu = (cum_seqlens_q is not None) and (cum_seqlens_k is not None)
    if varlen_from_cu:
        if use_segments:
            raise NotImplementedError("segment_ids are not supported with cum_seqlens in triton flash-attention.")
        assert cum_seqlens_q.dtype == jnp.int32 and cum_seqlens_k.dtype == jnp.int32
        batch = q.shape[0]
        QSeq_max = int(q.shape[1])
        KSeq_max = int(k.shape[1])
        nheads_q = q.shape[2]
        nheads_kv = k.shape[2]
        head_dim = q.shape[3]
        assert nheads_q % nheads_kv == 0
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype in [jnp.float16, jnp.bfloat16]

        max_seqlen_q = QSeq_max
        max_seqlen_k = KSeq_max

        q_packed = attention_pack_from_cu_static(q, cum_seqlens_q, max_tokens=batch * QSeq_max)
        k_packed = attention_pack_from_cu_static(k, cum_seqlens_k, max_tokens=batch * KSeq_max)
        v_packed = attention_pack_from_cu_static(v, cum_seqlens_k, max_tokens=batch * KSeq_max)

        qz, qm, qh, _ = get_strides(q_packed.shape)
        kz, kn, kh, _ = get_strides(k_packed.shape)
        vz, vn, vh, _ = get_strides(v_packed.shape)
        oz, om, oh, _ = get_strides(q_packed.shape)

        if bias is not None:
            raise ValueError("Bias with VARLEN requires a packed bias; pass None or pre-pack bias.")

        softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
        BOOL_BIAS = False

        max_seqlen_q_rounded = math.ceil(max_seqlen_q / 128) * 128
        BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
        PADDED_HEADS = BLOCK_HEADDIM > head_dim
        num_repeats = nheads_q // nheads_kv

        metaparams = dict(
            VARLEN=True,
            USE_DROPOUT=(dropout_prob > 0),
            IS_CAUSAL=causal,
            BIAS_ON=False,
            SLIDING=sliding_flag,
            SOFTCAP=softcap_flag,
            USE_SINKS=use_sinks,
            BOOL_BIAS=BOOL_BIAS,
            USE_SEGMENTS=False,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            PADDED_HEADS=PADDED_HEADS,
            BLOCK_N=block_n,
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        out_shape = [
            jax.ShapeDtypeStruct(q_packed.shape, q_packed.dtype),
            jax.ShapeDtypeStruct((batch, nheads_q, max_seqlen_q_rounded), jnp.float32),
        ]

        out, lse = triton_call(
            q_packed,
            k_packed,
            v_packed,
            q_segment_ids,
            kv_segment_ids,
            jnp.zeros((1,), q.dtype),
            softmax_scale,
            dropout_prob,
            dropout_seed if dropout_seed is not None else jnp.zeros((1,), q.dtype),
            logits_soft_cap_val,
            softmax_aux_tensor,
            num_sinks_val,
            qz,
            qm,
            qh,
            kz,
            kn,
            kh,
            vz,
            vn,
            vh,
            qsz,
            qsm,
            ksz,
            ksn,
            oz,
            om,
            oh,
            0,
            0,
            0,
            nheads_q,
            num_repeats,
            window_left,
            window_right,
            max_seqlen_q,
            cum_seqlens_q,
            max_seqlen_k,
            cum_seqlens_k,
            max_seqlen_q_rounded,
            head_dim,
            max_seqlen_q // 128,
            max_seqlen_k // 128,
            dtype_index(q_packed),
            kernel=_attn_fwd,
            out_shape=out_shape,
            grid=lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch * nheads_q),
            name="ejkernel::triton::flash_attn_fwd_varlen",
            **metaparams,
        )

        out_unpacked = attention_unpack_with_static_shape(out, cum_seqlens_q, batch, QSeq_max)
        return out_unpacked, lse

    if attention_mask is not None and varlen_from_cu:
        varlen_mode = attention_mask.shape[0] > 1
        assert bias is None, "Attention mask is not supported along with attention bias. Just use bias instead."
        assert q.shape[1] == k.shape[1], "Attention mask is not supported with QSeq != KSeq"
    else:
        varlen_mode = False

    batch, QSeq, nheads_q, head_dim = q.shape
    _, KSeq, nheads_kv, _ = k.shape
    expected_kv_shape = (batch, KSeq, nheads_kv, head_dim)
    assert k.shape == expected_kv_shape
    assert v.shape == expected_kv_shape
    assert nheads_q % nheads_kv == 0
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [jnp.float16, jnp.bfloat16]

    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    BOOL_BIAS = False
    varlen_mode = varlen_mode and (batch > 1)
    if not varlen_mode and attention_mask is not None:
        assert bias is None, "when using attention mask (bool) you can't use bias"
        BOOL_BIAS = True
        bias = attention_mask.astype(jnp.bool_)

    if varlen_mode:
        cum_seqlens_q = jnp.zeros(shape=(attention_mask.shape[0] + 1,), dtype=jnp.int32)
        cum_seqlens_q = cum_seqlens_q.at[1:].set(jnp.cumsum(attention_mask.sum(axis=1, dtype="i4"), axis=0, dtype="i4"))
        max_seqlen_q = attention_mask.shape[1]
        max_seqlen_k = attention_mask.shape[1]
        q = attention_pack_with_static_shape(q, attention_mask)
        k = attention_pack_with_static_shape(k, attention_mask)
        v = attention_pack_with_static_shape(v, attention_mask)
        QSeq = q.shape[1]
    else:
        cum_seqlens_q = None
        max_seqlen_q = QSeq
        max_seqlen_k = KSeq

    bz, bh, bm = calc_bias_strides(bias, batch, nheads_q, QSeq, KSeq)

    max_seqlen_q_rounded = math.ceil(max_seqlen_q / 128) * 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    PADDED_HEADS = BLOCK_HEADDIM > head_dim
    num_repeats = nheads_q // nheads_kv

    qz, qm, qh, _ = get_strides(q.shape)
    oz, om, oh, _ = get_strides(q.shape)
    kz, kn, kh, _ = get_strides(k.shape)
    vz, vn, vh, _ = get_strides(v.shape)

    metaparams = dict(
        VARLEN=varlen_mode,
        USE_DROPOUT=(dropout_prob > 0),
        IS_CAUSAL=causal,
        BIAS_ON=(bias is not None),
        SLIDING=sliding_flag,
        SOFTCAP=softcap_flag,
        USE_SINKS=use_sinks,
        BOOL_BIAS=BOOL_BIAS,
        USE_SEGMENTS=use_segments,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        PADDED_HEADS=PADDED_HEADS,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out_shape = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((batch, nheads_q, max_seqlen_q_rounded), jnp.float32),
    ]

    out, lse = triton_call(
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        bias if bias is not None else jnp.zeros((1,), q.dtype),
        softmax_scale,
        dropout_prob,
        dropout_seed if dropout_seed is not None else jnp.zeros((1,), q.dtype),
        logits_soft_cap_val,
        softmax_aux_tensor,
        num_sinks_val,
        qz,
        qm,
        qh,
        kz,
        kn,
        kh,
        vz,
        vn,
        vh,
        qsz,
        qsm,
        ksz,
        ksn,
        oz,
        om,
        oh,
        bz,
        bm,
        bh,
        nheads_q,
        num_repeats,
        window_left,
        window_right,
        QSeq,
        cum_seqlens_q if cum_seqlens_q is not None else jnp.zeros((1,), jnp.int32),
        KSeq,
        jnp.zeros((1,), jnp.int32),
        max_seqlen_q_rounded,
        head_dim,
        max_seqlen_q // 128,
        max_seqlen_k // 128,
        dtype_index(q),
        kernel=_attn_fwd,
        out_shape=out_shape,
        grid=lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch * nheads_q),
        name="ejkernel::triton::flash_attn_fwd",
        **metaparams,
    )

    if varlen_mode:
        out = attention_unpack_with_static_shape(out, cum_seqlens_q, *attention_mask.shape)
    return out, lse
