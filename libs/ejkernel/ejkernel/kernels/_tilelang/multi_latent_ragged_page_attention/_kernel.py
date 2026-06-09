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

"""TileLang kernel for Multi-head Latent Attention (MLA) over a ragged paged KV cache.

This module builds a single ``@T.prim_func`` that implements the forward pass of
native MLA decode attention (DeepSeek-V2 style).  Key design decisions:

**Paged KV-cache layout**:
    ``KVCache[num_pages, page_size_per_pack, kv_packing, cache_dim]``
    where ``kv_packing = 32 // (dtype.itemsize * 8)`` (i.e. the number of
    logical tokens packed per physical element to reach a 32-bit alignment).
    ``cache_dim = nope_dim_padded + pe_dim_padded`` — the cache stores NoPE
    and RoPE components interleaved along the last axis.

**Grid**: ``(num_q_heads, total_tokens)`` — one CTA per ``(head, token)`` pair.

**Live-token handling**: New KV data that has not yet been written to the cache
    (tokens in the current request's query chunk) is read directly from the
    ``KValues`` / ``KPe`` buffers and merged with cached pages during the same
    ``T.Pipelined`` loop.  The write-back to ``KVOut`` (alias of ``KVCache``)
    is performed in the ``hx == 0`` lane to avoid redundant writes.

**Online softmax**: The kernel maintains a 2-tuple ``(m_run, l_run)`` and uses
    ``log2`` arithmetic (``T.exp2``) throughout for efficiency.

**Optional features** baked in at compile time:
    - Sliding-window causal mask (``sliding_window > 0``).
    - Logits soft-cap via a ``tanh``-like approximation (``logits_soft_cap > 0``).
    - Per-tensor quantisation scales ``q_scale``, ``k_scale``, ``v_scale``
      applied to the pre-softmax scores and output respectively.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Map a JAX/NumPy dtype to the TileLang dtype string for MLA kernels.

    Args:
        dtype: Any dtype accepted by ``jnp.dtype()`` — float16, bfloat16,
            float32.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, ``"float32"``.

    Raises:
        TypeError: If *dtype* is not one of the three supported types.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for multi_latent_ragged_page_attention: {dtype}")
    return mapping[canonical]


def make_multi_latent_ragged_page_attention_prim_func(
    *,
    total_tokens: int,
    num_q_heads: int,
    num_pages: int,
    page_size_per_pack: int,
    kv_packing: int,
    cache_dim: int,
    max_num_seqs: int,
    pages_per_seq: int,
    nope_dim: int,
    pe_dim: int,
    nope_dim_padded: int,
    block_k: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    mask_value: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build the native MLA ragged paged-attention forward ``@T.prim_func``.

    All compile-time constants (shapes, scales, mask thresholds, optional
    features) are captured at build time; the returned function takes only
    runtime tensors.

    Args:
        total_tokens: Total query tokens across all active sequences (``TQ``).
        num_q_heads: Number of query heads (``HQ``).
        num_pages: Physical page count in the KV cache (``NP``).
        page_size_per_pack: Tokens per page in the packed dimension (``PSP``).
        kv_packing: Packing factor along the fourth cache axis; equals
            ``32 // (dtype.itemsize * 8)`` (``PACK``).
        cache_dim: Last dimension of the KV cache; must equal
            ``nope_dim_padded + pe_dim_padded`` (``CD``).
        max_num_seqs: Upper bound on the number of active sequences (``NS``).
        pages_per_seq: Physical pages allocated per sequence (``PPS``).
        nope_dim: NoPE (non-positional) key/value latent dimension (``DN``).
        pe_dim: RoPE (positional-encoding) key dimension (``DPE``).
        nope_dim_padded: ``nope_dim`` rounded up to the next multiple of 128
            (``DNP``); used as the split point inside ``cache_dim``.
        block_k: KV tile size for the inner pipeline loop (``BK``).
        softmax_scale: Softmax temperature scale applied to dot-products before
            the optional soft-cap.
        sliding_window: Window size for sliding-window causal mask.  Pass
            ``<= 0`` to disable.
        logits_soft_cap: Soft-cap value for logit stabilisation.  Pass
            ``<= 0.0`` to disable.
        mask_value: Value written for masked (out-of-window / future) positions
            before the online softmax.
        q_scale: Per-tensor query quantisation scale; multiplied into the
            attention score alongside *softmax_scale* and *k_scale*.
        k_scale: Per-tensor key quantisation scale.
        v_scale: Per-tensor value quantisation scale; applied to the output
            accumulator before the final cast.
        dtype: Activation dtype (float16, bfloat16, float32).
        num_stages: Number of software-pipeline stages for the ``T.Pipelined``
            KV loop (default 3).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature::

            (QNope:  [TQ, HQ, DN],
             QPe:    [TQ, HQ, DPE],
             KValues:[TQ, DN],
             KPe:    [TQ, DPE],
             KVCache:[NP, PSP, PACK, CD],
             KVLens: [NS],
             BlockTables: [NS * PPS],
             QueryStartLoc: [NS + 1],
             Distribution: [3],
             O:      [TQ, HQ, DN],          # output
             KVOut:  [NP, PSP, PACK, CD])   # in-place updated cache alias

        ``O`` and ``KVOut`` are both written; ``KVOut`` is an in-place alias
        of ``KVCache`` (input_output_alias ``4 -> 1`` in the FFI wrapper).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    TQ, HQ, NP, PSP, PACK, CD, NS, PPS, DN, DPE, DNP, BK = (
        total_tokens,
        num_q_heads,
        num_pages,
        page_size_per_pack,
        kv_packing,
        cache_dim,
        max_num_seqs,
        pages_per_seq,
        nope_dim,
        pe_dim,
        nope_dim_padded,
        block_k,
    )
    page_size = PSP * PACK
    max_tokens = PPS * page_size
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale) * float(q_scale) * float(k_scale)
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    window = int(sliding_window)
    use_window = window > 0
    mask_log2 = max(float(mask_value) * log2e, -1e30)
    out_scale = float(v_scale)

    @T.prim_func
    def multi_latent_ragged_page_attention_fwd(
        QNope: T.Tensor((TQ, HQ, DN), ts),
        QPe: T.Tensor((TQ, HQ, DPE), ts),
        KValues: T.Tensor((TQ, DN), ts),
        KPe: T.Tensor((TQ, DPE), ts),
        KVCache: T.Tensor((NP, PSP, PACK, CD), ts),
        KVLens: T.Tensor((NS,), "int32"),
        BlockTables: T.Tensor((NS * PPS,), "int32"),
        QueryStartLoc: T.Tensor((NS + 1,), "int32"),
        Distribution: T.Tensor((3,), "int32"),
        O: T.Tensor((TQ, HQ, DN), ts),
        KVOut: T.Tensor((NP, PSP, PACK, CD), ts),
    ):
        with T.Kernel(HQ, TQ, threads=threads) as (hx, qx):
            Q_nope_loc = T.alloc_fragment((DN,), accum)
            Q_pe_loc = T.alloc_fragment((DPE,), accum)
            K_nope_shared = T.alloc_shared((BK, DN), ts)
            K_pe_shared = T.alloc_shared((BK, DPE), ts)
            V_shared = T.alloc_shared((BK, DN), ts)
            S_local = T.alloc_fragment((BK,), accum)
            S_pe = T.alloc_fragment((BK,), accum)
            P_local = T.alloc_fragment((BK,), accum)
            Valid_local = T.alloc_fragment((BK,), "int32")
            QK_nope_prod = T.alloc_fragment((BK, DN), accum)
            QK_pe_prod = T.alloc_fragment((BK, DPE), accum)
            PV_prod = T.alloc_fragment((BK, DN), accum)
            row_max = T.alloc_fragment((1,), accum)
            row_sum = T.alloc_fragment((1,), accum)
            m_run = T.alloc_fragment((1,), accum)
            l_run = T.alloc_fragment((1,), accum)
            m_new = T.alloc_fragment((1,), accum)
            alpha = T.alloc_fragment((1,), accum)
            O_local = T.alloc_fragment((DN,), accum)
            pv_sum = T.alloc_fragment((DN,), accum)
            seq_idx_buf = T.alloc_fragment((1,), "int32")
            q_start_buf = T.alloc_fragment((1,), "int32")
            q_end_buf = T.alloc_fragment((1,), "int32")
            _psp_ref = T.alloc_fragment((PSP,), accum)
            _pack_ref = T.alloc_fragment((PACK,), accum)
            _cache_ref = T.alloc_fragment((CD,), accum)

            num_active = T.Cast("int32", Distribution[2])
            seq_idx_buf[0] = 0
            q_start_buf[0] = T.Cast("int32", QueryStartLoc[0])
            q_end_buf[0] = T.Cast("int32", QueryStartLoc[1])
            for s in T.serial(NS):
                s0 = T.Cast("int32", QueryStartLoc[s])
                s1 = T.Cast("int32", QueryStartLoc[s + 1])
                hit = (s < num_active) & (qx >= s0) & (qx < s1)
                seq_idx_buf[0] = T.if_then_else(hit, s, seq_idx_buf[0])
                q_start_buf[0] = T.if_then_else(hit, s0, q_start_buf[0])
                q_end_buf[0] = T.if_then_else(hit, s1, q_end_buf[0])

            seq_idx = seq_idx_buf[0]
            q_start = q_start_buf[0]
            q_end = q_end_buf[0]
            q_len = q_end - q_start
            q_offset = qx - q_start
            kv_len = T.Cast("int32", KVLens[seq_idx])
            write_start = kv_len - q_len
            q_pos = write_start + q_offset

            if hx == 0:
                dst_pos = write_start + q_offset
                dst_page = dst_pos // page_size
                dst_page_safe = T.min(T.max(dst_page, 0), PPS - 1)
                dst_inner = dst_pos - dst_page * page_size
                dst_pack_idx = dst_inner // PACK
                dst_pack_off = dst_inner - dst_pack_idx * PACK
                phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + dst_page_safe])
                phys_page_safe = T.min(T.max(phys_page, 0), NP - 1)
                token_valid = (
                    (q_offset >= 0) & (q_offset < q_len) & (dst_page >= 0) & (phys_page >= 0) & (phys_page < NP)
                )
                for d in T.Parallel(CD):
                    nope_safe = T.min(d, DN - 1)
                    pe_idx = d - DNP
                    pe_safe = T.min(T.max(pe_idx, 0), DPE - 1)
                    is_nope = d < DN
                    is_pe = (d >= DNP) & (d < DNP + DPE)
                    update_value = T.if_then_else(
                        is_nope,
                        KValues[qx, nope_safe],
                        T.if_then_else(is_pe, KPe[qx, pe_safe], T.Cast(ts, 0.0)),
                    )
                    KVOut[phys_page_safe, dst_pack_idx, dst_pack_off, d] = T.if_then_else(
                        token_valid,
                        update_value,
                        KVOut[phys_page_safe, dst_pack_idx, dst_pack_off, d],
                    )

            for d in T.Parallel(DN):
                Q_nope_loc[d] = T.Cast(accum, QNope[qx, hx, d])
            for d in T.Parallel(DPE):
                Q_pe_loc[d] = T.Cast(accum, QPe[qx, hx, d])

            T.fill(O_local, 0)
            m_run[0] = -1e30
            l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                for i, d in T.Parallel(BK, DN):
                    kv_pos = k_iter * BK + i
                    live_idx = q_start + (kv_pos - write_start)
                    live_valid = (kv_pos >= write_start) & (kv_pos < kv_len) & (live_idx >= q_start) & (live_idx < q_end)
                    logical_page = kv_pos // page_size
                    safe_page = T.min(logical_page, PPS - 1)
                    inner = kv_pos - logical_page * page_size
                    pack_idx = inner // PACK
                    pack_off = inner - pack_idx * PACK
                    phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + safe_page])
                    cache_valid = (kv_pos < kv_len) & (logical_page < PPS) & (phys_page >= 0) & (phys_page < NP)
                    K_nope_shared[i, d] = T.if_then_else(
                        live_valid,
                        KValues[live_idx, d],
                        T.if_then_else(cache_valid, KVCache[phys_page, pack_idx, pack_off, d], T.Cast(ts, 0.0)),
                    )
                    V_shared[i, d] = T.if_then_else(
                        live_valid,
                        KValues[live_idx, d],
                        T.if_then_else(cache_valid, KVCache[phys_page, pack_idx, pack_off, d], T.Cast(ts, 0.0)),
                    )
                for i, d in T.Parallel(BK, DPE):
                    kv_pos = k_iter * BK + i
                    live_idx = q_start + (kv_pos - write_start)
                    live_valid = (kv_pos >= write_start) & (kv_pos < kv_len) & (live_idx >= q_start) & (live_idx < q_end)
                    logical_page = kv_pos // page_size
                    safe_page = T.min(logical_page, PPS - 1)
                    inner = kv_pos - logical_page * page_size
                    pack_idx = inner // PACK
                    pack_off = inner - pack_idx * PACK
                    phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + safe_page])
                    cache_valid = (kv_pos < kv_len) & (logical_page < PPS) & (phys_page >= 0) & (phys_page < NP)
                    K_pe_shared[i, d] = T.if_then_else(
                        live_valid,
                        KPe[live_idx, d],
                        T.if_then_else(
                            cache_valid,
                            KVCache[phys_page, pack_idx, pack_off, DNP + d],
                            T.Cast(ts, 0.0),
                        ),
                    )

                for i, d in T.Parallel(BK, DN):
                    QK_nope_prod[i, d] = Q_nope_loc[d] * T.Cast(accum, K_nope_shared[i, d])
                T.reduce_sum(QK_nope_prod, S_local, dim=1, clear=True)
                for i, d in T.Parallel(BK, DPE):
                    QK_pe_prod[i, d] = Q_pe_loc[d] * T.Cast(accum, K_pe_shared[i, d])
                T.reduce_sum(QK_pe_prod, S_pe, dim=1, clear=True)

                for i in T.Parallel(BK):
                    kv_pos = k_iter * BK + i
                    logical_page = kv_pos // page_size
                    safe_page = T.min(logical_page, PPS - 1)
                    phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + safe_page])
                    token_valid = (
                        (q_offset >= 0)
                        & (q_offset < q_len)
                        & (kv_pos < kv_len)
                        & (kv_pos <= q_pos)
                        & (logical_page < PPS)
                        & (phys_page >= 0)
                        & (phys_page < NP)
                    )
                    if use_window:
                        token_valid = token_valid & (kv_pos > q_pos - window)
                    score = (S_local[i] + S_pe[i]) * scale
                    if use_softcap:
                        tanh_arg = score / softcap
                        score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                    Valid_local[i] = T.if_then_else(token_valid, 1, 0)
                    S_local[i] = T.if_then_else(token_valid, score * log2e, mask_log2)

                T.reduce_max(S_local, row_max, dim=0, clear=True)
                m_new[0] = T.max(m_run[0], row_max[0])
                alpha[0] = T.exp2(m_run[0] - m_new[0])

                for i in T.Parallel(BK):
                    P_local[i] = T.if_then_else(Valid_local[i] != 0, T.exp2(S_local[i] - m_new[0]), 0.0)
                T.reduce_sum(P_local, row_sum, dim=0, clear=True)
                l_run[0] = l_run[0] * alpha[0] + row_sum[0]

                for d in T.Parallel(DN):
                    O_local[d] = O_local[d] * alpha[0]

                for i, d in T.Parallel(BK, DN):
                    PV_prod[i, d] = P_local[i] * T.Cast(accum, V_shared[i, d])
                T.reduce_sum(PV_prod, pv_sum, dim=0, clear=True)
                for d in T.Parallel(DN):
                    O_local[d] = O_local[d] + pv_sum[d]

                m_run[0] = m_new[0]

            inv_l = T.alloc_fragment((1,), accum)
            inv_l[0] = 1.0 / T.max(l_run[0], 1e-6)
            for d in T.Parallel(DN):
                O[qx, hx, d] = T.Cast(ts, O_local[d] * inv_l[0] * out_scale)

    return multi_latent_ragged_page_attention_fwd
