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

"""TileLang prim_func builder for TurboQuant ragged paged attention v2.

Score approximation
-------------------
For each KV token the kernel computes two terms and sums them:

1. **Codebook term** (``S_mse``): dot-product of the rotated query ``Q_rot``
   against the per-dimension codebook centroids scaled by the original L2-norm.
2. **Residual term** (``S_corr``): dot-product of the projected query ``Q_proj``
   against the binary QJL sign bits scaled by the residual L2-norm and a
   normalisation factor ``qjl_factor = sqrt(2π) / qjl_dim``.

The combined score approximates the true dot-product between the un-quantised
query and the compressed key.

Value dequantisation
--------------------
Values are decompressed by looking up codebook centroids per dimension and
applying the inverse rotation: ``V_deq[i, d] = sum_j centroid[idx[j]] * R[j, d]``
scaled by the value L2-norm.

Grid
----
``T.Kernel(num_q_heads, total_query_tokens)`` — one CTA per (query head, query
token) pair.

Shared-memory-like fragment usage
----------------------------------
``S_mse_prod[BK, D]``, ``S_corr_prod[BK, QJL]``, ``V_deq[BK, D]`` are
register-file fragments (no explicit shared-memory tiles for K/V since the
compressed representation does not fit the strided tile model).
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
        jnp.dtype(jnp.uint8): "uint8",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for ragged_page_attention_v2_turboquant: {dtype}")
    return mapping[canonical]


def make_rpa_v2_turboquant_prim_func(
    *,
    total_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    num_seqs: int,
    head_dim: int,
    packed_idx_dim: int,
    packed_sign_dim: int,
    qjl_dim: int,
    key_levels: int,
    value_levels: int,
    block_k: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_softmax_aux: bool,
    q_dtype,
    norm_dtype,
    codebook_dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build the TurboQuant RPA v2 inference kernel ``@T.prim_func``.

    All parameters are baked in at Python level so TileLang can optimise the
    inner loops at compile time.

    Grid: ``T.Kernel(num_q_heads, total_query_tokens)``.

    Per-CTA computation:

    1. Rotate the query: ``Q_rot = Q @ Rotation.T`` (full ``D×D`` matmul in
       register fragments).
    2. Project the query: ``Q_proj = Q @ QJLProjection.T`` (``QJL×D`` matmul).
    3. For each KV tile iterate ``T.Pipelined(ceil(max_tokens/BK), num_stages)``:
       a. Compute ``S_mse`` via codebook lookup + norm scaling.
       b. Compute ``S_corr`` via sign extraction + residual norm scaling.
       c. Dequantise values: codebook lookup + inverse rotation.
       d. Run Flash-Attention-2 online softmax update.

    Args:
        total_tokens: Total query token count ``TQ``.
        num_q_heads: ``HQ``.
        num_kv_heads: ``HKV``; ``HQ`` must be divisible by ``HKV``.
        num_pages: Physical page pool size ``P``.
        page_size: Tokens per page ``PS``.
        pages_per_seq: Maximum pages per sequence ``PPS``.
        num_seqs: Number of active sequences ``NS``.
        head_dim: Head dimension ``D``.
        packed_idx_dim: ``ceil(D/2)`` — bytes per token/head in index tensors.
        packed_sign_dim: ``ceil(qjl_dim/8)`` — bytes per token/head for signs.
        qjl_dim: QJL projection dimension ``QJL``.
        key_levels: Codebook size for keys ``KL``.
        value_levels: Codebook size for values ``VL``.
        block_k: KV tile size ``BK``.
        softmax_scale: Attention scale (applied after score approximation).
        sliding_window: One-sided window radius; negative disables.
        logits_soft_cap: Soft-cap; non-positive disables.
        has_softmax_aux: Whether to prime ``m_run`` from ``SoftmaxAux``.
        q_dtype: Query/output floating-point dtype.
        norm_dtype: Norm tensor dtype (same float family as ``q_dtype``).
        codebook_dtype: Codebook and rotation/projection matrix dtype.
        num_stages: Pipeline stages (default 3).
        threads: Threads per CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` with signature::

            (Q, KeyIndices, KeySigns, KeyNorms, ValueIndices, ValueNorms,
             ContextLens, BlockTables, QueryStartLoc, Rotation, QJLProjection,
             KeyCodebook, ValueCodebook, SoftmaxAux, O)
    """
    q_ts = _dtype_str(q_dtype)
    norm_ts = _dtype_str(norm_dtype)
    cb_ts = _dtype_str(codebook_dtype)
    accum = "float32"
    TQ, HQ, HKV, P, PS, PPS, NS, D, PID, PSD, QJL, KL, VL, BK = (
        total_tokens,
        num_q_heads,
        num_kv_heads,
        num_pages,
        page_size,
        pages_per_seq,
        num_seqs,
        head_dim,
        packed_idx_dim,
        packed_sign_dim,
        qjl_dim,
        key_levels,
        value_levels,
        block_k,
    )
    max_tokens = PPS * PS
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale)
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    window = int(sliding_window)
    use_window = window > 0
    qjl_factor = 1.2533141373155001 / float(qjl_dim)

    @T.prim_func
    def rpa_v2_tq_fwd(
        Q: T.Tensor((TQ, HQ, D), q_ts),
        KeyIndices: T.Tensor((P, PS, HKV, PID), "uint8"),
        KeySigns: T.Tensor((P, PS, HKV, PSD), "uint8"),
        KeyNorms: T.Tensor((P, PS, HKV, 2), norm_ts),
        ValueIndices: T.Tensor((P, PS, HKV, PID), "uint8"),
        ValueNorms: T.Tensor((P, PS, HKV), norm_ts),
        ContextLens: T.Tensor((NS,), "int32"),
        BlockTables: T.Tensor((NS, PPS), "int32"),
        QueryStartLoc: T.Tensor((NS + 1,), "int32"),
        Rotation: T.Tensor((D, D), cb_ts),
        QJLProjection: T.Tensor((QJL, D), cb_ts),
        KeyCodebook: T.Tensor((KL,), cb_ts),
        ValueCodebook: T.Tensor((VL,), cb_ts),
        SoftmaxAux: T.Tensor((HQ,), q_ts),
        O: T.Tensor((TQ, HQ, D), q_ts),
    ):
        with T.Kernel(HQ, TQ, threads=threads) as (hx, qx):
            Q_rot = T.alloc_fragment((D,), accum)
            Q_proj = T.alloc_fragment((QJL,), accum)
            S_mse_prod = T.alloc_fragment((BK, D), accum)
            S_corr_prod = T.alloc_fragment((BK, QJL), accum)
            S_mse = T.alloc_fragment((BK,), accum)
            S_corr = T.alloc_fragment((BK,), accum)
            S_local = T.alloc_fragment((BK,), accum)
            P_local = T.alloc_fragment((BK,), accum)
            V_deq = T.alloc_fragment((BK, D), accum)
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
            _q_ref = T.alloc_fragment((1,), q_ts)
            _norm_ref = T.alloc_fragment((1,), norm_ts)
            _cb_ref = T.alloc_fragment((1,), cb_ts)
            _hkv_ref = T.alloc_fragment((HKV,), accum)
            _pid_ref = T.alloc_fragment((PID,), accum)
            _psd_ref = T.alloc_fragment((PSD,), accum)
            _kl_ref = T.alloc_fragment((KL,), accum)
            _vl_ref = T.alloc_fragment((VL,), accum)

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
            context_len = T.Cast("int32", ContextLens[seq_idx])
            write_start = context_len - q_len
            q_pos = write_start + q_offset
            kv_head = hx // q_heads_per_kv

            for rd in T.Parallel(D):
                Q_rot[rd] = 0.0
                for d in T.serial(D):
                    Q_rot[rd] = Q_rot[rd] + T.Cast(accum, Q[qx, hx, d]) * T.Cast(accum, Rotation[rd, d])

            for m in T.Parallel(QJL):
                Q_proj[m] = 0.0
                for d in T.serial(D):
                    Q_proj[m] = Q_proj[m] + T.Cast(accum, Q[qx, hx, d]) * T.Cast(accum, QJLProjection[m, d])

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
                    logical_page = kv_pos // PS
                    safe_page = T.min(logical_page, PPS - 1)
                    page_offset = kv_pos - logical_page * PS
                    phys_page = T.Cast("int32", BlockTables[seq_idx, safe_page])
                    page_valid = (logical_page < PPS) & (phys_page >= 0) & (phys_page < P)
                    idx_byte = T.Cast("int32", KeyIndices[phys_page, page_offset, kv_head, d // 2])
                    idx = T.if_then_else((d % 2) == 0, idx_byte & 15, (idx_byte >> 4) & 15)
                    idx = T.min(idx, KL - 1)
                    centroid = T.if_then_else(page_valid, T.Cast(accum, KeyCodebook[idx]), 0.0)
                    orig_norm = T.if_then_else(
                        page_valid,
                        T.Cast(accum, KeyNorms[phys_page, page_offset, kv_head, 0]),
                        0.0,
                    )
                    S_mse_prod[i, d] = Q_rot[d] * centroid * orig_norm

                T.reduce_sum(S_mse_prod, S_mse, dim=1, clear=True)

                for i, m in T.Parallel(BK, QJL):
                    kv_pos = k_iter * BK + i
                    logical_page = kv_pos // PS
                    safe_page = T.min(logical_page, PPS - 1)
                    page_offset = kv_pos - logical_page * PS
                    phys_page = T.Cast("int32", BlockTables[seq_idx, safe_page])
                    page_valid = (logical_page < PPS) & (phys_page >= 0) & (phys_page < P)
                    sign_byte = T.Cast("int32", KeySigns[phys_page, page_offset, kv_head, m // 8])
                    sign_bit = (sign_byte >> (m % 8)) & 1
                    sign = T.if_then_else(sign_bit == 1, 1.0, -1.0)
                    res_norm = T.if_then_else(
                        page_valid,
                        T.Cast(accum, KeyNorms[phys_page, page_offset, kv_head, 1]),
                        0.0,
                    )
                    S_corr_prod[i, m] = Q_proj[m] * sign * res_norm * qjl_factor

                T.reduce_sum(S_corr_prod, S_corr, dim=1, clear=True)

                for i, d in T.Parallel(BK, D):
                    kv_pos = k_iter * BK + i
                    logical_page = kv_pos // PS
                    safe_page = T.min(logical_page, PPS - 1)
                    page_offset = kv_pos - logical_page * PS
                    phys_page = T.Cast("int32", BlockTables[seq_idx, safe_page])
                    page_valid = (logical_page < PPS) & (phys_page >= 0) & (phys_page < P)
                    v_norm = T.if_then_else(page_valid, T.Cast(accum, ValueNorms[phys_page, page_offset, kv_head]), 0.0)
                    V_deq[i, d] = 0.0
                    for jd in T.serial(D):
                        idx_byte = T.Cast("int32", ValueIndices[phys_page, page_offset, kv_head, jd // 2])
                        idx = T.if_then_else((jd % 2) == 0, idx_byte & 15, (idx_byte >> 4) & 15)
                        idx = T.min(idx, VL - 1)
                        centroid = T.if_then_else(page_valid, T.Cast(accum, ValueCodebook[idx]), 0.0)
                        V_deq[i, d] = V_deq[i, d] + centroid * T.Cast(accum, Rotation[jd, d])
                    V_deq[i, d] = V_deq[i, d] * v_norm

                for i in T.Parallel(BK):
                    kv_pos = k_iter * BK + i
                    token_valid = (kv_pos < context_len) & (kv_pos <= q_pos)
                    if use_window:
                        token_valid = token_valid & (kv_pos > q_pos - window)
                    score = (S_mse[i] + S_corr[i]) * scale
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
                    PV_prod[i, d] = P_local[i] * V_deq[i, d]
                T.reduce_sum(PV_prod, pv_sum, dim=0, clear=True)
                for d in T.Parallel(D):
                    O_local[d] = O_local[d] + pv_sum[d]

                m_run[0] = m_new[0]

            inv_l = T.alloc_fragment((1,), accum)
            inv_l[0] = 1.0 / T.max(l_run[0], 1e-6)
            for d in T.Parallel(D):
                O[qx, hx, d] = T.Cast(q_ts, O_local[d] * inv_l[0])

    return rpa_v2_tq_fwd
