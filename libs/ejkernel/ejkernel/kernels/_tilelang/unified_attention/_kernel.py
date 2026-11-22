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

"""TileLang kernel for unified paged causal attention (decode + prefill).

The kernel supports paged KV-cache storage (physical blocks referenced via a
block-table), causal masking, optional sliding-window attention, optional ALiBi
positional biases, optional query-query (QQ) bias, optional softmax auxiliary
input, and optional logit soft-capping (tanh).

Grid: ``(num_q_heads, total_tokens)`` — one CTA per ``(query_token, head)``.
Each CTA scans through all KV blocks for its sequence.

Shared memory: two tiles ``K_shared (BK, D, dtype)`` and
``V_shared (BK, D, dtype)`` are pipelined via ``T.Pipelined`` with
``num_stages`` stages.  ``BK == block_size`` (tile K equals the physical
block size).

Online softmax is computed with base-2 arithmetic (``log2e`` scaling)
for efficiency.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Return the TileLang dtype string for a supported floating-point dtype.

    Args:
        dtype: any dtype specifier accepted by ``jnp.dtype``.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, or ``"float32"``.

    Raises:
        TypeError: if ``dtype`` is not one of the three supported types.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for unified_attention: {dtype}")
    return mapping[canonical]


def make_unified_attention_prim_func(
    *,
    total_tokens: int,
    num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_blocks: int,
    block_size: int,
    max_blocks_per_seq: int,
    head_dim: int,
    block_k: int,
    qq_dim: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_alibi: bool,
    has_qq_bias: bool,
    has_softmax_aux: bool,
    dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build the unified paged causal attention forward ``@T.prim_func``.

    This is a decode + prefill kernel supporting both single-token and
    multi-token queries per sequence.

    Grid: ``(num_q_heads, total_tokens)`` — each CTA handles one
    ``(query_token, q_head)`` pair and scans all KV blocks for its sequence
    via the block table.

    Shared memory: ``K_shared (BK, D, dtype)`` and ``V_shared (BK, D, dtype)``
    are software-pipelined with ``num_stages`` stages via ``T.Pipelined``.

    Online softmax uses base-2 scaling (multiplied by ``log2e``).  If
    ``has_softmax_aux`` is ``True``, ``SoftmaxAux[hx]`` is used as the
    initial running maximum ``m_run``.

    Causal mask: ``kv_pos <= q_pos`` where ``q_pos = context_len + q_offset``
    and ``context_len = kv_len - q_len``.  The mask includes the sliding window
    predicate ``kv_pos > q_pos - sliding_window`` when
    ``sliding_window > 0``.

    Logit soft-capping: when ``logits_soft_cap > 0`` the raw logit ``score``
    is replaced by ``softcap * tanh(score / softcap)`` via a base-2 exp
    approximation.

    Args:
        total_tokens: total query tokens across all sequences ``TQ``.
        num_seqs: number of sequences ``NS``.
        num_q_heads: number of query heads ``HQ``.
        num_kv_heads: number of KV heads ``HKV`` (GQA supported).
        num_blocks: number of physical KV cache blocks ``NB``.
        block_size: tokens per physical block ``BS``.
        max_blocks_per_seq: maximum blocks per sequence ``MB``.
        head_dim: head dimension ``D``.
        block_k: tile size along the KV dimension (must equal ``block_size``).
        qq_dim: side length of the square QQ-bias matrix (1 if unused).
        softmax_scale: scale applied to raw dot-product scores.
        sliding_window: attend only to keys within this window; 0 = disabled.
        logits_soft_cap: tanh soft-cap value; 0.0 = disabled.
        has_alibi: if ``True`` read ALiBi slopes from ``AlibiSlopes``.
        has_qq_bias: if ``True`` read query-query bias from ``QQBias``.
        has_softmax_aux: if ``True`` initialise ``m_run`` from ``SoftmaxAux``.
        dtype: tensor dtype for Q, K, V, and output.
        num_stages: number of software-pipeline stages (default 3).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Q, KCache, VCache, KVLens, BlockTables, QueryStartLoc,
        AlibiSlopes, QQBias, SoftmaxAux, O)``
        where ``O`` is ``(TQ, HQ, D, dtype)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    TQ, NS, HQ, HKV, NB, BS, MB, D, BK, QQ = (
        total_tokens,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        num_blocks,
        block_size,
        max_blocks_per_seq,
        head_dim,
        block_k,
        qq_dim,
    )
    max_tokens = MB * BS
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale)
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    window = int(sliding_window)
    use_window = window > 0

    @T.prim_func
    def unified_attention_fwd(
        Q: T.Tensor((TQ, HQ, D), ts),
        KCache: T.Tensor((NB, BS, HKV, D), ts),
        VCache: T.Tensor((NB, BS, HKV, D), ts),
        KVLens: T.Tensor((NS,), "int32"),
        BlockTables: T.Tensor((NS, MB), "int32"),
        QueryStartLoc: T.Tensor((NS + 1,), "int32"),
        AlibiSlopes: T.Tensor((HQ,), ts),
        QQBias: T.Tensor((QQ, QQ), ts),
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
            _qq_ref = T.alloc_fragment((QQ,), accum)
            _hkv_ref = T.alloc_fragment((HKV,), accum)

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
            q_len = q_end - q_start
            q_offset = qx - q_start
            kv_len = T.Cast("int32", KVLens[seq_idx])
            context_len = kv_len - q_len
            q_pos = context_len + q_offset
            kv_head = hx // q_heads_per_kv

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[qx, hx, d])

            T.fill(O_local, 0)
            if has_softmax_aux:
                m_run[0] = T.Cast(accum, SoftmaxAux[hx]) * log2e
                l_run[0] = 1.0
            else:
                m_run[0] = -1e30
                l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                for i, d in T.Parallel(BK, D):
                    kv_pos = k_iter * BK + i
                    logical_block = kv_pos // BS
                    safe_block = T.min(logical_block, MB - 1)
                    block_offset = kv_pos - logical_block * BS
                    phys_block = T.Cast("int32", BlockTables[seq_idx, safe_block])
                    block_valid = (logical_block < MB) & (phys_block >= 0) & (phys_block < NB)
                    K_shared[i, d] = T.if_then_else(
                        (kv_pos < kv_len) & block_valid,
                        KCache[phys_block, block_offset, kv_head, d],
                        T.Cast(ts, 0.0),
                    )
                    V_shared[i, d] = T.if_then_else(
                        (kv_pos < kv_len) & block_valid,
                        VCache[phys_block, block_offset, kv_head, d],
                        T.Cast(ts, 0.0),
                    )

                for i, d in T.Parallel(BK, D):
                    QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                for i in T.Parallel(BK):
                    kv_pos = k_iter * BK + i
                    logical_block = kv_pos // BS
                    safe_block = T.min(logical_block, MB - 1)
                    phys_block = T.Cast("int32", BlockTables[seq_idx, safe_block])
                    block_valid = (logical_block < MB) & (phys_block >= 0) & (phys_block < NB)
                    token_valid = (
                        (q_offset >= 0) & (q_offset < q_len) & (kv_pos < kv_len) & (kv_pos <= q_pos) & block_valid
                    )
                    if use_window:
                        token_valid = token_valid & (kv_pos > q_pos - window)
                    score = S_local[i] * scale
                    if has_alibi:
                        key_rel = kv_pos - context_len
                        score = score + T.Cast(accum, AlibiSlopes[hx]) * T.Cast(accum, key_rel)
                    if has_qq_bias:
                        key_rel_q = kv_pos - context_len
                        row_safe = T.min(T.max(q_offset, 0), QQ - 1)
                        key_safe = T.min(T.max(key_rel_q, 0), QQ - 1)
                        qq_valid = (q_offset >= 0) & (q_offset < QQ) & (key_rel_q >= 0) & (key_rel_q < QQ)
                        score = score + T.if_then_else(qq_valid, T.Cast(accum, QQBias[row_safe, key_safe]), 0.0)
                    if use_softcap:
                        tanh_arg = score / softcap
                        score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                    S_local[i] = T.if_then_else(token_valid, score * log2e, -1e30)

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
            inv_l[0] = 1.0 / T.max(l_run[0], 1e-6)
            for d in T.Parallel(D):
                O[qx, hx, d] = T.Cast(ts, O_local[d] * inv_l[0])

    return unified_attention_fwd
