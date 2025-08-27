# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import math
import typing as tp

import chex
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from eformer.callib import triton_call
from triton import Config

from .._utils import dtype_index, get_sharding, get_strides, safe_autotune
from ._utils import attention_pack_with_static_shape, attention_unpack_with_static_shape, calc_bias_strides, padded_load

BIG_NEG: tl.constexpr = jnp.iinfo(jnp.int32).min
LN2: tl.constexpr = 1.44269504089


def config_prune_kernel(
    configs: list[Config],
    named_args: dict[str, tp.Any],
    **kwargs,
) -> list[Config]:
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
    stride_kn,
    stride_vn,
    index_start_n,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    USE_DROPOUT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    MASKED: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
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

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
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

    if BIAS_ON:
        if BOOL_BIAS:
            qk = tl.where(bias, qk, BIG_NEG)
        else:
            qk += bias * (LN2 / softmax_scale)
    m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, me_i)
    P_ij = tl.exp2(qk * softmax_scale - m_ij[:, None])
    l_ij = tl.sum(P_ij, 1)

    if USE_DROPOUT:
        dropout_offs = dropout_offs + index_start_n
        dropout_mask = tl.rand(dropout_seed, dropout_offs) > dropout_prob
        P_ij = tl.where(dropout_mask, P_ij, 0.0)

    acc_o_scale = tl.exp2(m_i - m_ij)
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

    P_ij = P_ij.to(v.dtype)
    acc_o += tl.dot(P_ij, v)
    m_i = m_ij
    l_i_new = tl.exp2(me_i - m_ij) + l_ij
    me_i = m_ij + tl.log2(l_i_new)
    return m_i, me_i, acc_o


@safe_autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        ###
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=1),
    ],
    key=[
        "CKSeq",
        "CQSeq",
        "DRuntime",
        "VARLEN",
        "USE_DROPOUT",
        "IS_CAUSAL",
        "BIAS_ON",
        "BLOCK_HEADDIM",
    ],
    prune_configs_by={"early_config_prune": config_prune_kernel},
)
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
    B,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_oz,
    stride_om,
    stride_oh,
    stride_bz,
    stride_bm,
    stride_bh,
    nheads_q,
    num_repeats,
    QSeq,
    cum_seqlens_q,
    KSeq,
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
    BOOL_BIAS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i_start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_head_q = off_zh % nheads_q
    off_head_kv = off_head_q // num_repeats
    off_z = off_zh // nheads_q

    if VARLEN:
        cu_seq_start_q = tl.load(cum_seqlens_q + off_z)
        actual_seqlen_q = tl.load(cum_seqlens_q + off_z + 1) - cu_seq_start_q
        if i_start_m * BLOCK_M >= actual_seqlen_q:
            return
        actual_seqlen_k = actual_seqlen_q
        cu_seq_start_k = cu_seq_start_q
        off_z = 0
    else:
        actual_seqlen_q = QSeq
        actual_seqlen_k = KSeq
        cu_seq_start_q = 0
        cu_seq_start_k = 0

    softmax_scale = softmax_scale * LN2

    offs_m = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    fully_masked_lines = actual_seqlen_q - actual_seqlen_k if IS_CAUSAL else 0
    if fully_masked_lines >= (i_start_m + 1) * BLOCK_M:
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

    me_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    pad_rows = (not EVEN_M) or (VARLEN and (i_start_m * BLOCK_M > actual_seqlen_q))
    q = padded_load(
        q_ptrs,
        offs_m,
        offs_d,
        PA0=pad_rows,
        PA1=PADDED_HEADS,
        LA0=actual_seqlen_q,
        LA1=headdim,
    )
    if IS_CAUSAL:
        end_n = min(
            actual_seqlen_k - actual_seqlen_q + (i_start_m + 1) * BLOCK_M,
            actual_seqlen_k,
        )
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
                stride_kn,
                stride_vn,
                next_start_n,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                USE_DROPOUT=USE_DROPOUT,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                MASKED=False,
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
                stride_kn,
                stride_vn,
                index_start_n,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                USE_DROPOUT=USE_DROPOUT,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                MASKED=True,
                PADDED_COLS=pad_cols,
                PADDED_HEADS=PADDED_HEADS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

    if USE_DROPOUT:
        o_scale = tl.exp2((m_i - me_i) - tl.log2(1 - dropout_prob))
    else:
        o_scale = tl.exp2(m_i - me_i)
    acc_o = acc_o * o_scale[:, None]
    if fully_masked_lines > i_start_m * BLOCK_M:
        acc_o = tl.where(offs_m[:, None] < fully_masked_lines, 0, acc_o)
    i_start_m = tl.program_id(0)
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

    tl.store(
        out_ptrs,
        acc_o,
        mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim),
    )


def _fwd_attention_kernel_call(
    q: chex.Array | None,
    k: chex.Array | None,
    v: chex.Array | None,
    attention_mask: chex.Array | None = None,
    bias: chex.Array | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    varlen_mode: bool = True,
):
    if attention_mask is not None and varlen_mode:
        varlen_mode = attention_mask.shape[0] > 1
        assert bias is None, "Attention mask is not supported along with attention bias. Just use bias instead."
        assert q.shape[1] == k.shape[1], "Attention mask is not supported with QSeq != KSeq"
    else:
        varlen_mode = False
    batch, QSeq, nheads_q, head_dim = q.shape
    _, KSeq, nheads_kv, _ = k.shape
    expected_kv_shape = (batch, KSeq, nheads_kv, head_dim)

    assert (
        k.shape == expected_kv_shape
    ), f"key shape is {k.shape = } and we excepted it to be like {expected_kv_shape = }"
    assert (
        v.shape == expected_kv_shape
    ), f"value shape is {v.shape = } and we excepted it to be like {expected_kv_shape = }"

    assert nheads_q % nheads_kv == 0, f"{nheads_q = } is not divisible by {nheads_kv =}"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [jnp.float16, jnp.bfloat16], f"Only support fp16 and bf16 got {q.dtype}"

    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    BOOL_BIAS = False
    varlen_mode = varlen_mode and (batch > 1)
    if not varlen_mode and attention_mask is not None:
        assert bias is None, "when using attention mask (bool) you can't use bias"
        BOOL_BIAS = True
        bias = jnp.astype(attention_mask, jnp.bool)

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

    bz, bh, bm = calc_bias_strides(
        bias,
        batch,
        nheads_q,
        QSeq,
        KSeq,
    )

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
        BOOL_BIAS=BOOL_BIAS,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        PADDED_HEADS=PADDED_HEADS,
    )
    out_shape = [
        jax.ShapeDtypeStruct(q.shape, q.dtype, sharding=get_sharding(q)),
        jax.ShapeDtypeStruct((batch, nheads_q, max_seqlen_q_rounded), jnp.float32),
    ]

    out, lse = triton_call(
        q,
        k,
        v,
        bias if bias is not None else jnp.zeros((1,), jnp.float16),
        softmax_scale,
        dropout_prob,
        dropout_seed if dropout_seed is not None else jnp.zeros((1,), jnp.float16),
        qz,
        qm,
        qh,
        kz,
        kn,
        kh,
        vz,
        vn,
        vh,
        oz,
        om,
        oh,
        bz,
        bm,
        bh,
        nheads_q,
        num_repeats,
        QSeq,
        cum_seqlens_q if cum_seqlens_q is not None else jnp.zeros((1,), jnp.float16),
        KSeq,
        max_seqlen_q_rounded,
        head_dim,
        max_seqlen_q // 128,
        max_seqlen_k // 128,
        dtype_index(q),
        kernel=_attn_fwd,
        out_shape=out_shape,
        disable_verbose_logging=False,
        grid=lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch * nheads_q),
        name="triton::ops::_attn_fwd",
        **metaparams,
    )

    if varlen_mode:
        out = attention_unpack_with_static_shape(out, cum_seqlens_q, *attention_mask.shape)
    return out, lse
