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

"""TileLang prim_func builder for ragged paged attention v2.

KV-cache layout
---------------
Pages are stored as ``[num_pages, page_size, num_combined_kv_heads, head_dim]``
where ``num_combined_kv_heads = num_kv_heads * 2``.  K heads occupy even
indices (``kv_head * 2``) and V heads occupy odd indices (``kv_head * 2 + 1``).

Grid
----
``T.Kernel(num_q_heads, total_query_tokens)`` — one CTA per (query head, query
token) pair.

Sequence discovery
------------------
Each CTA scans ``QueryStartLoc`` to find which sequence it belongs to, then
reads ``ContextLens[seq_idx]`` to determine the causal position of its query
token.

Shared memory
-------------
Two ``(block_k, head_dim)`` tiles (``K_shared``, ``V_shared``) in the compute
dtype.

Online softmax
--------------
Flash-Attention-2 recurrence in log₂ space using ``T.exp2``.  Optional
tanh soft-capping and sliding-window masking are applied per position.
Optional ``SoftmaxAux`` primes ``m_run`` with per-head sink statistics.
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
        raise TypeError(f"Unsupported dtype for ragged_page_attention_v2: {dtype}")
    return mapping[canonical]


def _index_dtype_str(dtype) -> str:
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.int32): "int32",
        jnp.dtype(jnp.int64): "int64",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported index dtype for ragged_page_attention_v2: {dtype}")
    return mapping[canonical]


def make_ragged_page_attention_v2_prim_func(
    *,
    total_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    num_seqs: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    mask_value: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_softmax_aux: bool,
    dtype,
    index_dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build a native ragged paged-attention v2 kernel ``@T.prim_func``.

    All parameters are baked in as Python-level constants for compile-time
    specialisation.

    Grid: ``T.Kernel(num_q_heads, total_query_tokens)``.

    Shared memory per CTA: ``K_shared`` and ``V_shared``, each
    ``(block_k, head_dim)`` in the compute dtype.

    Pipeline: ``T.Pipelined(ceil(max_tokens / block_k), num_stages)`` over
    the KV tile dimension, where ``max_tokens = pages_per_seq * page_size``.

    Args:
        total_tokens: Total query tokens ``TQ`` across all sequences.
        num_q_heads: Number of query heads ``HQ``.
        num_kv_heads: Number of KV heads ``HKV``.
        num_pages: Physical page pool size ``P``.
        page_size: Tokens per page ``PS``.
        pages_per_seq: Maximum pages per sequence ``MB``.
        num_seqs: Number of active sequences ``NS``.
        head_dim: Head dimension ``D``.
        block_k: KV tile size ``BK`` (typically 64 or 128).
        softmax_scale: Pre-computed attention scale.
        mask_value: Logit fill for masked positions; stored in log₂ space.
        sliding_window: One-sided sliding-window radius; negative value disables.
        logits_soft_cap: Logit soft-cap; non-positive value disables.
        has_softmax_aux: Whether ``SoftmaxAux`` contains meaningful sink values.
            When ``True`` ``m_run`` is initialised to ``SoftmaxAux[hx] * log2e``
            and ``l_run`` to 1.0 instead of the cold-start defaults.
        dtype: Compute dtype for Q/K/V/O.
        index_dtype: Integer dtype for context lengths and block tables.
        num_stages: Software pipeline stages (default 3).
        threads: Threads per CUDA CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` with signature::

            (Q, KV, ContextLens, BlockTables, QueryStartLoc, SoftmaxAux, O)

        where all shapes are fully static.
    """
    ts = _dtype_str(dtype)
    index_ts = _index_dtype_str(index_dtype)
    accum = "float32"
    TQ, HQ, HKV, P, PS, MB, NS, D, BK = (
        total_tokens,
        num_q_heads,
        num_kv_heads,
        num_pages,
        page_size,
        pages_per_seq,
        num_seqs,
        head_dim,
        block_k,
    )
    COMBINED = HKV * 2
    max_tokens = MB * PS
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale)
    scale_log2e = scale * log2e
    mask_value_log2e = max(float(mask_value) * log2e, -1e30)
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    window = sliding_window
    use_sinks = has_softmax_aux

    @T.prim_func
    def ragged_page_attention_v2_fwd(
        Q: T.Tensor((TQ, HQ, D), ts),
        KV: T.Tensor((P, PS, COMBINED, D), ts),
        ContextLens: T.Tensor((NS,), index_ts),
        BlockTables: T.Tensor((NS, MB), index_ts),
        QueryStartLoc: T.Tensor((NS + 1,), index_ts),
        SoftmaxAux: T.Tensor((HQ,), ts),
        O: T.Tensor((TQ, HQ, D), ts),
    ):
        with T.Kernel(HQ, TQ, threads=threads) as (hx, qx):
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
            seq_idx_buf = T.alloc_fragment((1,), "int32")
            q_start_buf = T.alloc_fragment((1,), "int32")
            q_end_buf = T.alloc_fragment((1,), "int32")
            _ts_ref = T.alloc_fragment((1,), ts)
            _index_ref = T.alloc_fragment((1,), index_ts)
            _hkv_ref = T.alloc_fragment((HKV,), accum)
            _combined_ref = T.alloc_fragment((COMBINED,), accum)

            kv_head = hx // q_heads_per_kv
            kv_head_k = kv_head * 2
            kv_head_v = kv_head_k + 1

            seq_idx_buf[0] = 0
            q_start_buf[0] = T.Cast("int32", QueryStartLoc[0])
            q_end_buf[0] = T.Cast("int32", QueryStartLoc[1])
            for s in T.serial(NS):
                s0 = T.Cast("int32", QueryStartLoc[s])
                s1 = T.Cast("int32", QueryStartLoc[s + 1])
                hit = (qx >= s0) & (qx < s1)
                seq_idx_buf[0] = T.if_then_else(hit, s, seq_idx_buf[0])
                q_start_buf[0] = T.if_then_else(hit, s0, q_start_buf[0])
                q_end_buf[0] = T.if_then_else(hit, s1, q_end_buf[0])

            seq_idx = seq_idx_buf[0]
            q_start = q_start_buf[0]
            q_end = q_end_buf[0]
            q_count = q_end - q_start
            q_offset = qx - q_start
            context_len = T.Cast("int32", ContextLens[seq_idx])
            query_pos = context_len - q_count + q_offset
            if window > 0:
                valid_start = T.max(0, query_pos - window + 1)
            else:
                valid_start = 0

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[qx, hx, d])

            T.fill(O_local, 0)
            if use_sinks:
                m_run[0] = T.Cast(accum, SoftmaxAux[hx]) * log2e
                l_run[0] = 1.0
            else:
                m_run[0] = -1e30
                l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                tile_start = k_iter * BK
                tile_end = tile_start + BK
                if (tile_start < context_len) & (tile_start <= query_pos) & (tile_end > valid_start):
                    for i, d in T.Parallel(BK, D):
                        logical_idx = tile_start + i
                        logical_block = logical_idx // PS
                        safe_block = T.min(logical_block, MB - 1)
                        page_offset = logical_idx - logical_block * PS
                        token_valid = (
                            (logical_idx < context_len) & (logical_idx <= query_pos) & (logical_idx >= valid_start)
                        )
                        page_ok = logical_block < MB
                        page_idx = T.if_then_else(page_ok, T.Cast("int32", BlockTables[seq_idx, safe_block]), 0)
                        page_valid = (page_idx >= 0) & (page_idx < P)
                        in_range = token_valid & page_valid
                        K_shared[i, d] = T.if_then_else(
                            in_range,
                            KV[page_idx, page_offset, kv_head_k, d],
                            T.Cast(ts, 0.0),
                        )
                        V_shared[i, d] = T.if_then_else(
                            in_range,
                            KV[page_idx, page_offset, kv_head_v, d],
                            T.Cast(ts, 0.0),
                        )

                    for i, d in T.Parallel(BK, D):
                        QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                    T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                    for i in T.Parallel(BK):
                        logical_idx = tile_start + i
                        logical_block = logical_idx // PS
                        safe_block = T.min(logical_block, MB - 1)
                        token_valid = (
                            (logical_idx < context_len) & (logical_idx <= query_pos) & (logical_idx >= valid_start)
                        )
                        page_ok = logical_block < MB
                        page_idx = T.if_then_else(page_ok, T.Cast("int32", BlockTables[seq_idx, safe_block]), 0)
                        page_valid = (page_idx >= 0) & (page_idx < P)
                        in_range = token_valid & page_valid
                        score = S_local[i]
                        if use_softcap:
                            score_natural = score * scale
                            tanh_arg = score_natural / softcap
                            soft_score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                            S_local[i] = T.if_then_else(in_range, soft_score * log2e, mask_value_log2e)
                        else:
                            S_local[i] = T.if_then_else(in_range, score * scale_log2e, mask_value_log2e)

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
                O[qx, hx, d] = T.Cast(ts, O_local[d] * inv_l[0])

    return ragged_page_attention_v2_fwd
