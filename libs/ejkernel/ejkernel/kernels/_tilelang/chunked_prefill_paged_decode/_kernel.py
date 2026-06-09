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

"""TileLang ``@T.prim_func`` factory for chunked-prefill + paged-decode.

The single kernel ``chunked_prefill_paged_decode_fwd`` handles:

1. **KV-cache write** (done by thread ``hx == 0`` only): maps the new token
   at position ``context_len + q_offset`` into the physical block determined
   by ``BlockTables[seq_idx, dst_block]`` and writes ``KNew`` / ``VNew``
   into ``KOut`` / ``VOut`` in-place.

2. **Paged causal attention** (all threads): for each query token the kernel
   iterates over K/V tiles that cover the full context ``[0, kv_len)``,
   reading new tokens from ``KNew`` / ``VNew`` and cached tokens from
   ``KCache`` / ``VCache`` via the page table.  Online softmax in float32
   with log2-space arithmetic accumulates ``O_local``.

Grid: ``(num_q_heads, total_tokens)``. ``KV_BLOCK_SIZE = block_size``
(block_k matches the paging granularity).  K/V tiles are software-pipelined
with ``num_stages`` stages.
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
        raise TypeError(f"Unsupported dtype for chunked_prefill_paged_decode: {dtype}")
    return mapping[canonical]


def make_chunked_prefill_paged_decode_prim_func(
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
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_alibi: bool,
    has_softmax_aux: bool,
    dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build the fused cache-write + paged-decode-attention ``@T.prim_func``.

    Grid: ``(num_q_heads, total_tokens)`` — one CTA per ``(head, query token)``.

    The kernel first locates which sequence a query token belongs to by
    scanning ``QueryStartLoc``, then:

    * If ``hx == 0``: writes the new KV token at position
      ``context_len + q_offset`` into the physical page returned by
      ``BlockTables[seq_idx, dst_block]``.
    * All threads: run paged causal attention over the full context via
      ``max_blocks_per_seq * block_size`` K/V tile iterations pipelined
      with ``num_stages`` stages.

    ``sliding_window`` is encoded as a negative integer to mean "no window".
    ``logits_soft_cap`` is encoded as a negative float to mean "no cap".

    All arithmetic is log2-space (multiply by ``log2e = 1.4426...`` before
    exp2) to avoid redundant ``log2`` / ``exp`` conversions.

    Args:
        total_tokens: total number of tokens across all sequences (``TQ``).
        num_seqs: number of sequences (``NS``).
        num_q_heads: number of query heads (``HQ``).
        num_kv_heads: number of KV heads (``HKV``); must divide ``HQ``.
        num_blocks: total physical page blocks in the cache (``NB``).
        block_size: tokens per physical page block (``BS``).
        max_blocks_per_seq: max logical blocks per sequence (``MB``).
        head_dim: head feature dimension (``D``).
        block_k: KV tile size; should equal ``block_size``.
        softmax_scale: ``QK^T`` multiplier (pre-computed).
        sliding_window: left-context window size, or ``-1`` for no window.
        logits_soft_cap: soft-cap value, or ``-1.0`` for no cap.
        has_alibi: whether the ALiBi-slopes buffer is active.
        has_softmax_aux: whether the attention-sink buffer is active.
        dtype: token tensor dtype (float16 / bfloat16 / float32).
        num_stages: KV-load pipeline stages (default 3).
        threads: threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Q, KNew, VNew, KCache, VCache, KVLens, BlockTables, QueryStartLoc,
        AlibiSlopes, SoftmaxAux, O, KOut, VOut)``
        where ``KOut`` / ``VOut`` alias ``KCache`` / ``VCache`` (in-place update).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    TQ, NS, HQ, HKV, NB, BS, MB, D, BK = (
        total_tokens,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        num_blocks,
        block_size,
        max_blocks_per_seq,
        head_dim,
        block_k,
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
    def chunked_prefill_paged_decode_fwd(
        Q: T.Tensor((TQ, HQ, D), ts),
        KNew: T.Tensor((TQ, HKV, D), ts),
        VNew: T.Tensor((TQ, HKV, D), ts),
        KCache: T.Tensor((NB, BS, HKV, D), ts),
        VCache: T.Tensor((NB, BS, HKV, D), ts),
        KVLens: T.Tensor((NS,), "int32"),
        BlockTables: T.Tensor((NS, MB), "int32"),
        QueryStartLoc: T.Tensor((NS + 1,), "int32"),
        AlibiSlopes: T.Tensor((HQ,), ts),
        SoftmaxAux: T.Tensor((HQ,), ts),
        O: T.Tensor((TQ, HQ, D), ts),
        KOut: T.Tensor((NB, BS, HKV, D), ts),
        VOut: T.Tensor((NB, BS, HKV, D), ts),
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

            if hx == 0:
                dst_pos = context_len + q_offset
                dst_block = dst_pos // BS
                dst_block_safe = T.min(T.max(dst_block, 0), MB - 1)
                dst_offset = dst_pos - dst_block * BS
                dst_offset_safe = T.min(T.max(dst_offset, 0), BS - 1)
                phys_block = T.Cast("int32", BlockTables[seq_idx, dst_block_safe])
                phys_block_safe = T.min(T.max(phys_block, 0), NB - 1)
                block_valid = (dst_block >= 0) & (dst_block < MB) & (phys_block >= 0) & (phys_block < NB)
                token_valid = (q_offset >= 0) & (q_offset < q_len) & block_valid
                for kh, d in T.Parallel(HKV, D):
                    KOut[phys_block_safe, dst_offset_safe, kh, d] = T.if_then_else(
                        token_valid,
                        KNew[qx, kh, d],
                        KOut[phys_block_safe, dst_offset_safe, kh, d],
                    )
                    VOut[phys_block_safe, dst_offset_safe, kh, d] = T.if_then_else(
                        token_valid,
                        VNew[qx, kh, d],
                        VOut[phys_block_safe, dst_offset_safe, kh, d],
                    )

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
                    live_idx = q_start + (kv_pos - context_len)
                    live_valid = (kv_pos >= context_len) & (kv_pos < kv_len) & (live_idx >= q_start) & (live_idx < q_end)
                    logical_block = kv_pos // BS
                    safe_block = T.min(logical_block, MB - 1)
                    block_offset = kv_pos - logical_block * BS
                    phys_block = T.Cast("int32", BlockTables[seq_idx, safe_block])
                    block_valid = (logical_block < MB) & (phys_block >= 0) & (phys_block < NB)
                    cache_valid = (kv_pos < kv_len) & block_valid
                    K_shared[i, d] = T.if_then_else(
                        live_valid,
                        KNew[live_idx, kv_head, d],
                        T.if_then_else(
                            cache_valid,
                            KCache[phys_block, block_offset, kv_head, d],
                            T.Cast(ts, 0.0),
                        ),
                    )
                    V_shared[i, d] = T.if_then_else(
                        live_valid,
                        VNew[live_idx, kv_head, d],
                        T.if_then_else(
                            cache_valid,
                            VCache[phys_block, block_offset, kv_head, d],
                            T.Cast(ts, 0.0),
                        ),
                    )

                for i, d in T.Parallel(BK, D):
                    QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                for i in T.Parallel(BK):
                    kv_pos = k_iter * BK + i
                    token_valid = (kv_pos < kv_len) & (kv_pos <= q_pos)
                    if use_window:
                        token_valid = token_valid & (kv_pos > q_pos - window)
                    score = S_local[i] * scale
                    if has_alibi:
                        key_rel = kv_pos - context_len
                        score = score + T.Cast(accum, AlibiSlopes[hx]) * T.Cast(accum, key_rel)
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

    return chunked_prefill_paged_decode_fwd
