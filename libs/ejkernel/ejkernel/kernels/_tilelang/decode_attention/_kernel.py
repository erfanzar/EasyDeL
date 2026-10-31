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

"""Dedicated single-query decode attention kernel.

Single-Q FlashDecoding-style kernel: one CTA per ``(batch, head)``. Each
CTA loads its query vector, walks the KV cache in chunks of ``BLOCK_K``,
keeps an online-softmax running ``(m, l, o)``, and writes the final
``(B, H, D)`` output plus the natural-log LSE.

This avoids the full-FA dispatch overhead (no LSE intermediate buffer
sized for seq_q>1, no bwd-prep). On a B=4 H=8 D=128 L=128 H100 workload
the dedicated kernel is ~2x faster than reusing the generic FA kernel
with seq_q=1.

Layout: KV is laid out as ``(B, L, H, D)`` (after the contiguous-paging
reshape done in the JAX glue).
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for decode kernel: {dtype}")
    return mapping[canonical]


def _index_dtype_str(dtype) -> str:
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.int32): "int32",
        jnp.dtype(jnp.int64): "int64",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported index dtype for decode kernel: {dtype}")
    return mapping[canonical]


def make_decode_prim_func(
    *,
    batch: int,
    num_heads: int,
    seq_len_kv: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build the dedicated single-Q decode ``@T.prim_func``.

    Grid: ``(num_heads, batch)``. Inside each CTA:

    1. Load Q ``(D,)`` into a fragment.
    2. Loop over K/V blocks of size ``BLOCK_K`` along the sequence axis.
    3. For each block compute ``s = Q @ K^T * scale`` (``D``-wide reduction),
       update the online ``(m, l)`` and accumulate ``o = exp(s - m) @ V``.
    4. After the loop, normalise ``o /= l`` and emit ``lse = m + log(l)``.

    Returns:
        ``@T.prim_func`` with buffers ``(Q, K, V, O, LSE)`` where:

        * ``Q``: ``(batch, num_heads, head_dim)``
        * ``K``, ``V``: ``(batch, seq_len_kv, num_heads, head_dim)``
        * ``O``: ``(batch, num_heads, head_dim)``
        * ``LSE``: ``(batch, num_heads)`` in fp32 (natural log)
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, H, L, D = batch, num_heads, seq_len_kv, head_dim
    BK = block_k
    log2e = 1.4426950408889634
    scale_log2e = float(softmax_scale) * log2e

    @T.prim_func
    def decode_fwd(
        Q: T.Tensor((B, H, D), ts),
        K: T.Tensor((B, L, H, D), ts),
        V: T.Tensor((B, L, H, D), ts),
        O: T.Tensor((B, H, D), ts),
        LSE: T.Tensor((B, H), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            Q_loc = T.alloc_fragment((D,), accum)
            K_shared = T.alloc_shared((BK, D), ts)
            V_shared = T.alloc_shared((BK, D), ts)
            S_local = T.alloc_fragment((BK,), accum)
            P_local = T.alloc_fragment((BK,), accum)
            P_cast = T.alloc_fragment((BK,), accum)
            QK_prod = T.alloc_fragment((BK, D), accum)
            PV_prod = T.alloc_fragment((BK, D), accum)
            row_max = T.alloc_fragment((1,), accum)
            row_sum = T.alloc_fragment((1,), accum)
            m_run = T.alloc_fragment((1,), accum)
            l_run = T.alloc_fragment((1,), accum)
            O_local = T.alloc_fragment((D,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[bx, hx, d])

            T.fill(O_local, 0)
            m_run[0] = -float("inf")
            l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(L, BK), num_stages=num_stages):
                for i, d in T.Parallel(BK, D):
                    k_idx = k_iter * BK + i
                    in_range = k_idx < L
                    K_shared[i, d] = T.if_then_else(in_range, K[bx, k_idx, hx, d], T.Cast(ts, 0.0))
                    V_shared[i, d] = T.if_then_else(in_range, V[bx, k_idx, hx, d], T.Cast(ts, 0.0))

                for i, d in T.Parallel(BK, D):
                    QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                for i in T.Parallel(BK):
                    k_idx = k_iter * BK + i
                    S_local[i] = T.if_then_else(
                        k_idx < L,
                        S_local[i] * scale_log2e,
                        -float("inf"),
                    )

                T.reduce_max(S_local, row_max, dim=0, clear=True)
                m_new = T.alloc_fragment((1,), accum)
                m_new[0] = T.max(m_run[0], row_max[0])

                alpha = T.alloc_fragment((1,), accum)
                alpha[0] = T.exp2(m_run[0] - m_new[0])

                for i in T.Parallel(BK):
                    P_local[i] = T.exp2(S_local[i] - m_new[0])
                T.reduce_sum(P_local, row_sum, dim=0, clear=True)
                l_run[0] = l_run[0] * alpha[0] + row_sum[0]

                for d in T.Parallel(D):
                    O_local[d] = O_local[d] * alpha[0]

                for i in T.Parallel(BK):
                    P_cast[i] = P_local[i]
                for i, d in T.Parallel(BK, D):
                    PV_prod[i, d] = P_cast[i] * T.Cast(accum, V_shared[i, d])
                pv_sum = T.alloc_fragment((D,), accum)
                T.reduce_sum(PV_prod, pv_sum, dim=0, clear=True)
                for d in T.Parallel(D):
                    O_local[d] = O_local[d] + pv_sum[d]

                m_run[0] = m_new[0]

            inv_l = T.alloc_fragment((1,), accum)
            inv_l[0] = 1.0 / T.max(l_run[0], 1e-30)
            for d in T.Parallel(D):
                O[bx, hx, d] = T.Cast(ts, O_local[d] * inv_l[0])
            ln2 = 0.6931471805599453
            LSE[bx, hx] = (m_run[0] + T.log2(T.max(l_run[0], 1e-30))) * ln2

    return decode_fwd


def make_paged_decode_prim_func(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    total_tokens: int,
    max_pages: int,
    page_size: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    logits_soft_cap: float,
    dtype,
    index_dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build paged single-Q decode attention.

    ``ReqToTokens`` contains physical page IDs. The kernel maps logical KV
    token positions through that page table, applies per-sequence ``SeqLens``,
    supports GQA, and emits both output and natural-log LSE.
    """
    ts = _dtype_str(dtype)
    index_ts = _index_dtype_str(index_dtype)
    accum = "float32"
    B, HQ, HKV, TKV, MP, PS, D, BK = (
        batch,
        num_q_heads,
        num_kv_heads,
        total_tokens,
        max_pages,
        page_size,
        head_dim,
        block_k,
    )
    max_tokens = MP * PS
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    ln2 = 0.6931471805599453
    scale = float(softmax_scale)
    scale_log2e = scale * log2e
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0

    @T.prim_func
    def paged_decode_fwd(
        Q: T.Tensor((B, HQ, D), ts),
        K: T.Tensor((TKV, HKV, D), ts),
        V: T.Tensor((TKV, HKV, D), ts),
        ReqToTokens: T.Tensor((B, MP), index_ts),
        SeqLens: T.Tensor((B,), index_ts),
        O: T.Tensor((B, HQ, D), ts),
        LSE: T.Tensor((B, HQ), accum),
    ):
        with T.Kernel(HQ, B, threads=threads) as (hx, bx):
            Q_loc = T.alloc_fragment((D,), accum)
            K_shared = T.alloc_shared((BK, D), ts)
            V_shared = T.alloc_shared((BK, D), ts)
            S_local = T.alloc_fragment((BK,), accum)
            P_local = T.alloc_fragment((BK,), accum)
            QK_prod = T.alloc_fragment((BK, D), accum)
            PV_prod = T.alloc_fragment((BK, D), accum)
            row_max = T.alloc_fragment((1,), accum)
            row_sum = T.alloc_fragment((1,), accum)
            m_run = T.alloc_fragment((1,), accum)
            l_run = T.alloc_fragment((1,), accum)
            m_new = T.alloc_fragment((1,), accum)
            alpha = T.alloc_fragment((1,), accum)
            O_local = T.alloc_fragment((D,), accum)
            pv_sum = T.alloc_fragment((D,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _index_ref = T.alloc_fragment((1,), index_ts)
            _hkv_ref = T.alloc_fragment((HKV,), accum)

            kv_head = hx // q_heads_per_kv
            seq_len = T.Cast("int32", SeqLens[bx])

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[bx, hx, d])

            T.fill(O_local, 0)
            m_run[0] = -1e30
            l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                tile_start = k_iter * BK
                if tile_start < seq_len:
                    for i, d in T.Parallel(BK, D):
                        logical_idx = tile_start + i
                        logical_page = logical_idx // PS
                        safe_page = T.min(logical_page, MP - 1)
                        page_offset = logical_idx - logical_page * PS
                        token_valid = logical_idx < seq_len
                        page_ok = logical_page < MP
                        page_idx = T.if_then_else(page_ok, T.Cast("int32", ReqToTokens[bx, safe_page]), 0)
                        phys_idx = page_idx * PS + page_offset
                        in_range = token_valid & (page_idx >= 0) & (phys_idx < TKV)
                        K_shared[i, d] = T.if_then_else(
                            in_range,
                            K[phys_idx, kv_head, d],
                            T.Cast(ts, 0.0),
                        )
                        V_shared[i, d] = T.if_then_else(
                            in_range,
                            V[phys_idx, kv_head, d],
                            T.Cast(ts, 0.0),
                        )

                    for i, d in T.Parallel(BK, D):
                        QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                    T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                    for i in T.Parallel(BK):
                        logical_idx = tile_start + i
                        logical_page = logical_idx // PS
                        safe_page = T.min(logical_page, MP - 1)
                        page_offset = logical_idx - logical_page * PS
                        token_valid = logical_idx < seq_len
                        page_ok = logical_page < MP
                        page_idx = T.if_then_else(page_ok, T.Cast("int32", ReqToTokens[bx, safe_page]), 0)
                        phys_idx = page_idx * PS + page_offset
                        in_range = token_valid & (page_idx >= 0) & (phys_idx < TKV)
                        score = S_local[i]
                        if use_softcap:
                            score_natural = score * scale
                            tanh_arg = score_natural / softcap
                            soft_score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                            S_local[i] = T.if_then_else(in_range, soft_score * log2e, -1e30)
                        else:
                            S_local[i] = T.if_then_else(in_range, score * scale_log2e, -1e30)

                    T.reduce_max(S_local, row_max, dim=0, clear=True)
                    m_new[0] = T.max(m_run[0], row_max[0])
                    alpha[0] = T.exp2(m_run[0] - m_new[0])

                    for i in T.Parallel(BK):
                        P_local[i] = T.exp2(S_local[i] - m_new[0])
                    T.reduce_sum(P_local, row_sum, dim=0, clear=True)
                    l_run[0] = l_run[0] * alpha[0] + row_sum[0]

                    for d in T.Parallel(D):
                        O_local[d] = O_local[d] * alpha[0]

                    for i, d in T.Parallel(BK, D):
                        PV_prod[i, d] = P_local[i] * T.Cast(accum, V_shared[i, d])
                    T.reduce_sum(PV_prod, pv_sum, dim=0, clear=True)
                    for d in T.Parallel(D):
                        O_local[d] = O_local[d] + pv_sum[d]

                    m_run[0] = m_new[0]

            inv_l = T.alloc_fragment((1,), accum)
            inv_l[0] = 1.0 / T.max(l_run[0], 1e-30)
            for d in T.Parallel(D):
                O[bx, hx, d] = T.Cast(ts, O_local[d] * inv_l[0])
            LSE[bx, hx] = (m_run[0] + T.log2(T.max(l_run[0], 1e-30))) * ln2

    return paged_decode_fwd
