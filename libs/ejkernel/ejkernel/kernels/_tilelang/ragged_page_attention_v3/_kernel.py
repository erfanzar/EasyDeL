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

"""TileLang prim_func builder for ragged paged attention v3 (fused cache update).

KV-cache layout (packed)
------------------------
``[num_pages, page_size, kv_groups, kv_packing, head_dim_padded]``

K and V heads are interleaved in the ``kv_groups * kv_packing`` axis:
  * K head ``kh`` → combined index ``kh * 2``  → group ``(kh*2) // kv_packing``,
    lane ``(kh*2) % kv_packing``.
  * V head ``kh`` → combined index ``kh*2 + 1`` → analogous indices.

``kv_packing = 32 // (dtype.itemsize * 8)`` elements fit per 32-bit word.
For float16: ``kv_packing = 2``; for float32: ``kv_packing = 1``.

Cache-write strategy
--------------------
Only the CTA with ``hx == 0`` writes new tokens; all CTAs read both live new
tokens and cached history via a priority mux:

    if live_valid (new token in current batch):
        use KNew / VNew directly
    elif cache_valid (existing cached token):
        use KVCache
    else:
        use zero

This avoids a separate cache-write kernel while keeping the attention logic
clean.

Grid
----
``T.Kernel(num_q_heads, total_query_tokens)`` — one CTA per (query head, query
token) pair.

Two prim_func variants are generated: one with ``SoftmaxAux`` (when
``use_aux=True``) and one without.
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
        raise TypeError(f"Unsupported dtype for ragged_page_attention_v3: {dtype}")
    return mapping[canonical]


def make_rpa_v3_prim_func(
    *,
    total_tokens: int,
    max_num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    kv_groups: int,
    kv_packing: int,
    head_dim_padded: int,
    block_k: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    use_aux: bool,
    q_dtype,
    kv_dtype,
    aux_dtype=None,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build the fused RPA v3 cache-update and attention kernel ``@T.prim_func``.

    Two variants are produced:

    * ``use_aux=True``: includes ``SoftmaxAux`` buffer — ``m_run`` is primed from
      it and ``l_run`` starts at 1.0.
    * ``use_aux=False``: cold-start online softmax (``m_run = -inf``,
      ``l_run = 0``).

    The prim_func for both variants has ``KVCache`` aliased to ``KVOut`` so the
    cache update is in-place from JAX's perspective.

    Args:
        total_tokens: Total query tokens ``TQ``.
        max_num_seqs: Static upper bound on sequences ``NS`` (from ``kv_lens``
            shape).
        num_q_heads: ``HQ``.
        num_kv_heads: ``HKV``; ``HQ`` must be divisible by ``HKV``.
        head_dim: Logical head dimension ``D``.
        num_pages: Physical page pool size ``P``.
        page_size: Tokens per page ``PS``.
        pages_per_seq: Pages per sequence ``PPS``.
        kv_groups: ``ceil(num_kv_heads * 2 / kv_packing)`` — outer cache axis.
        kv_packing: Elements per word = ``32 // (dtype.itemsize * 8)``.
        head_dim_padded: Padded head dim in the cache (``DP >= D``).
        block_k: KV tile size ``BK``.
        softmax_scale: Attention scale.
        sliding_window: One-sided window; negative disables.
        logits_soft_cap: Logit soft-cap; non-positive disables.
        q_scale: Query affine scale; ``-1.0`` (or any non-positive value) disables.
        k_scale: Key score affine scale; ``-1.0`` disables.
        v_scale: Output affine scale; ``-1.0`` disables.
        use_aux: Whether the ``SoftmaxAux`` path is used.
        q_dtype: Query/output dtype.
        kv_dtype: KV cache dtype.
        aux_dtype: Dtype of ``SoftmaxAux``; defaults to ``kv_dtype`` when ``None``.
        num_stages: Software pipeline stages (default 3).
        threads: Threads per CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` with signature (when ``use_aux=True``)::

            (Q, KNew, VNew, KVCache, KVLens, BlockTables, QueryStartLoc,
             Distribution, SoftmaxAux, O, KVOut)

        or (when ``use_aux=False``)::

            (Q, KNew, VNew, KVCache, KVLens, BlockTables, QueryStartLoc,
             Distribution, O, KVOut)

        ``KVCache`` and ``KVOut`` share the same buffer (in-place alias).
    """
    q_ts = _dtype_str(q_dtype)
    kv_ts = _dtype_str(kv_dtype)
    aux_ts = _dtype_str(kv_dtype if aux_dtype is None else aux_dtype)
    accum = "float32"
    TQ, NS, HQ, HKV, D = total_tokens, max_num_seqs, num_q_heads, num_kv_heads, head_dim
    P, PS, PPS, KG, KP, DP, BK = num_pages, page_size, pages_per_seq, kv_groups, kv_packing, head_dim_padded, block_k
    max_tokens = PPS * PS
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale)
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    window = int(sliding_window)
    use_window = window > 0
    q_scale_value = float(q_scale)
    k_scale_value = float(k_scale)
    v_scale_value = float(v_scale)
    use_q_scale = q_scale_value > 0.0
    use_k_scale = k_scale_value > 0.0
    use_v_scale = v_scale_value > 0.0

    if use_aux:

        @T.prim_func
        def rpa_v3_fwd(
            Q: T.Tensor((TQ, HQ, D), q_ts),
            KNew: T.Tensor((TQ, HKV, D), kv_ts),
            VNew: T.Tensor((TQ, HKV, D), kv_ts),
            KVCache: T.Tensor((P, PS, KG, KP, DP), kv_ts),
            KVLens: T.Tensor((NS,), "int32"),
            BlockTables: T.Tensor((NS * PPS,), "int32"),
            QueryStartLoc: T.Tensor((NS + 1,), "int32"),
            Distribution: T.Tensor((3,), "int32"),
            SoftmaxAux: T.Tensor((HQ,), aux_ts),
            O: T.Tensor((TQ, HQ, D), q_ts),
            KVOut: T.Tensor((P, PS, KG, KP, DP), kv_ts),
        ):
            with T.Kernel(HQ, TQ, threads=threads) as (hx, qx):
                Q_loc = T.alloc_fragment((D,), accum)
                K_shared = T.alloc_shared((BK, D), kv_ts)
                V_shared = T.alloc_shared((BK, D), kv_ts)
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
                _q_ref = T.alloc_fragment((1,), q_ts)
                _kv_ref = T.alloc_fragment((1,), kv_ts)
                _aux_ref = T.alloc_fragment((1,), aux_ts)
                _hkv_ref = T.alloc_fragment((HKV,), accum)
                _kg_ref = T.alloc_fragment((KG,), accum)
                _kp_ref = T.alloc_fragment((KP,), accum)
                _dp_ref = T.alloc_fragment((DP,), accum)
                _dist_ref = T.alloc_fragment((3,), accum)

                seq_idx_buf[0] = 0
                q_start_buf[0] = T.Cast("int32", QueryStartLoc[0])
                q_end_buf[0] = T.Cast("int32", QueryStartLoc[1])
                num_seqs = T.Cast("int32", Distribution[2])
                for s in T.serial(NS):
                    s0 = T.Cast("int32", QueryStartLoc[s])
                    s1 = T.Cast("int32", QueryStartLoc[s + 1])
                    hit = (s < num_seqs) & (qx >= s0) & (qx < s1)
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
                kv_head = hx // q_heads_per_kv
                if use_window:
                    valid_start = T.max(0, q_pos - window + 1)
                else:
                    valid_start = 0

                if hx == 0:
                    dst_pos = write_start + q_offset
                    dst_page = dst_pos // PS
                    dst_page_safe = T.min(T.max(dst_page, 0), PPS - 1)
                    dst_offset = dst_pos - dst_page * PS
                    phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + dst_page_safe])
                    phys_page_safe = T.min(T.max(phys_page, 0), P - 1)
                    dst_offset_safe = T.min(T.max(dst_offset, 0), PS - 1)
                    page_valid = (dst_page >= 0) & (dst_page < PPS) & (phys_page >= 0) & (phys_page < P)
                    token_valid = (q_offset >= 0) & (q_offset < q_len) & page_valid
                    for kh, d in T.Parallel(HKV, DP):
                        d_safe = T.min(d, D - 1)
                        k_combined = kh * 2
                        v_combined = k_combined + 1
                        k_group = k_combined // KP
                        k_lane = k_combined - k_group * KP
                        v_group = v_combined // KP
                        v_lane = v_combined - v_group * KP
                        k_update = T.if_then_else(d < D, T.Cast(kv_ts, KNew[qx, kh, d_safe]), T.Cast(kv_ts, 0.0))
                        v_update = T.if_then_else(d < D, T.Cast(kv_ts, VNew[qx, kh, d_safe]), T.Cast(kv_ts, 0.0))
                        KVOut[phys_page_safe, dst_offset_safe, k_group, k_lane, d] = T.if_then_else(
                            token_valid,
                            k_update,
                            KVOut[phys_page_safe, dst_offset_safe, k_group, k_lane, d],
                        )
                        KVOut[phys_page_safe, dst_offset_safe, v_group, v_lane, d] = T.if_then_else(
                            token_valid,
                            v_update,
                            KVOut[phys_page_safe, dst_offset_safe, v_group, v_lane, d],
                        )

                for d in T.Parallel(D):
                    q_val = T.Cast(accum, Q[qx, hx, d])
                    if use_q_scale:
                        q_val = q_val / q_scale_value
                    Q_loc[d] = q_val

                T.fill(O_local, 0)
                m_run[0] = T.Cast(accum, SoftmaxAux[hx]) * log2e
                l_run[0] = 1.0

                for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                    tile_start = k_iter * BK
                    tile_end = tile_start + BK
                    if (tile_start < kv_len) & (tile_start <= q_pos) & (tile_end > valid_start):
                        for i, d in T.Parallel(BK, D):
                            kv_pos = tile_start + i
                            live_idx = q_start + (kv_pos - write_start)
                            live_valid = (
                                (kv_pos >= write_start) & (kv_pos < kv_len) & (live_idx >= q_start) & (live_idx < q_end)
                            )
                            logical_page = kv_pos // PS
                            safe_page = T.min(logical_page, PPS - 1)
                            page_offset = kv_pos - logical_page * PS
                            phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + safe_page])
                            page_valid = (logical_page < PPS) & (phys_page >= 0) & (phys_page < P)
                            cache_valid = (kv_pos < kv_len) & page_valid
                            combined_k = kv_head * 2
                            combined_v = combined_k + 1
                            group_k = combined_k // KP
                            lane_k = combined_k - group_k * KP
                            group_v = combined_v // KP
                            lane_v = combined_v - group_v * KP
                            K_shared[i, d] = T.if_then_else(
                                live_valid,
                                KNew[live_idx, kv_head, d],
                                T.if_then_else(
                                    cache_valid,
                                    KVCache[phys_page, page_offset, group_k, lane_k, d],
                                    T.Cast(kv_ts, 0.0),
                                ),
                            )
                            V_shared[i, d] = T.if_then_else(
                                live_valid,
                                VNew[live_idx, kv_head, d],
                                T.if_then_else(
                                    cache_valid,
                                    KVCache[phys_page, page_offset, group_v, lane_v, d],
                                    T.Cast(kv_ts, 0.0),
                                ),
                            )

                        for i, d in T.Parallel(BK, D):
                            QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                        T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                        for i in T.Parallel(BK):
                            kv_pos = tile_start + i
                            token_valid = (kv_pos < kv_len) & (kv_pos <= q_pos) & (kv_pos >= valid_start)
                            score = S_local[i] * scale
                            if use_k_scale:
                                score = score * k_scale_value
                            if use_q_scale:
                                score = score * q_scale_value
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
                    out_val = O_local[d] * inv_l[0]
                    if use_v_scale:
                        out_val = out_val * v_scale_value
                    O[qx, hx, d] = T.Cast(q_ts, out_val)

        return rpa_v3_fwd

    @T.prim_func
    def rpa_v3_fwd(
        Q: T.Tensor((TQ, HQ, D), q_ts),
        KNew: T.Tensor((TQ, HKV, D), kv_ts),
        VNew: T.Tensor((TQ, HKV, D), kv_ts),
        KVCache: T.Tensor((P, PS, KG, KP, DP), kv_ts),
        KVLens: T.Tensor((NS,), "int32"),
        BlockTables: T.Tensor((NS * PPS,), "int32"),
        QueryStartLoc: T.Tensor((NS + 1,), "int32"),
        Distribution: T.Tensor((3,), "int32"),
        O: T.Tensor((TQ, HQ, D), q_ts),
        KVOut: T.Tensor((P, PS, KG, KP, DP), kv_ts),
    ):
        with T.Kernel(HQ, TQ, threads=threads) as (hx, qx):
            Q_loc = T.alloc_fragment((D,), accum)
            K_shared = T.alloc_shared((BK, D), kv_ts)
            V_shared = T.alloc_shared((BK, D), kv_ts)
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
            _q_ref = T.alloc_fragment((1,), q_ts)
            _kv_ref = T.alloc_fragment((1,), kv_ts)
            _hkv_ref = T.alloc_fragment((HKV,), accum)
            _kg_ref = T.alloc_fragment((KG,), accum)
            _kp_ref = T.alloc_fragment((KP,), accum)
            _dp_ref = T.alloc_fragment((DP,), accum)
            _dist_ref = T.alloc_fragment((3,), accum)

            seq_idx_buf[0] = 0
            q_start_buf[0] = T.Cast("int32", QueryStartLoc[0])
            q_end_buf[0] = T.Cast("int32", QueryStartLoc[1])
            num_seqs = T.Cast("int32", Distribution[2])
            for s in T.serial(NS):
                s0 = T.Cast("int32", QueryStartLoc[s])
                s1 = T.Cast("int32", QueryStartLoc[s + 1])
                hit = (s < num_seqs) & (qx >= s0) & (qx < s1)
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
            kv_head = hx // q_heads_per_kv
            if use_window:
                valid_start = T.max(0, q_pos - window + 1)
            else:
                valid_start = 0

            if hx == 0:
                dst_pos = write_start + q_offset
                dst_page = dst_pos // PS
                dst_page_safe = T.min(T.max(dst_page, 0), PPS - 1)
                dst_offset = dst_pos - dst_page * PS
                phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + dst_page_safe])
                phys_page_safe = T.min(T.max(phys_page, 0), P - 1)
                dst_offset_safe = T.min(T.max(dst_offset, 0), PS - 1)
                page_valid = (dst_page >= 0) & (dst_page < PPS) & (phys_page >= 0) & (phys_page < P)
                token_valid = (q_offset >= 0) & (q_offset < q_len) & page_valid
                for kh, d in T.Parallel(HKV, DP):
                    d_safe = T.min(d, D - 1)
                    k_combined = kh * 2
                    v_combined = k_combined + 1
                    k_group = k_combined // KP
                    k_lane = k_combined - k_group * KP
                    v_group = v_combined // KP
                    v_lane = v_combined - v_group * KP
                    k_update = T.if_then_else(d < D, T.Cast(kv_ts, KNew[qx, kh, d_safe]), T.Cast(kv_ts, 0.0))
                    v_update = T.if_then_else(d < D, T.Cast(kv_ts, VNew[qx, kh, d_safe]), T.Cast(kv_ts, 0.0))
                    KVOut[phys_page_safe, dst_offset_safe, k_group, k_lane, d] = T.if_then_else(
                        token_valid,
                        k_update,
                        KVOut[phys_page_safe, dst_offset_safe, k_group, k_lane, d],
                    )
                    KVOut[phys_page_safe, dst_offset_safe, v_group, v_lane, d] = T.if_then_else(
                        token_valid,
                        v_update,
                        KVOut[phys_page_safe, dst_offset_safe, v_group, v_lane, d],
                    )

            for d in T.Parallel(D):
                q_val = T.Cast(accum, Q[qx, hx, d])
                if use_q_scale:
                    q_val = q_val / q_scale_value
                Q_loc[d] = q_val

            T.fill(O_local, 0)
            m_run[0] = -1e30
            l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                tile_start = k_iter * BK
                tile_end = tile_start + BK
                if (tile_start < kv_len) & (tile_start <= q_pos) & (tile_end > valid_start):
                    for i, d in T.Parallel(BK, D):
                        kv_pos = tile_start + i
                        live_idx = q_start + (kv_pos - write_start)
                        live_valid = (
                            (kv_pos >= write_start) & (kv_pos < kv_len) & (live_idx >= q_start) & (live_idx < q_end)
                        )
                        logical_page = kv_pos // PS
                        safe_page = T.min(logical_page, PPS - 1)
                        page_offset = kv_pos - logical_page * PS
                        phys_page = T.Cast("int32", BlockTables[seq_idx * PPS + safe_page])
                        page_valid = (logical_page < PPS) & (phys_page >= 0) & (phys_page < P)
                        cache_valid = (kv_pos < kv_len) & page_valid
                        combined_k = kv_head * 2
                        combined_v = combined_k + 1
                        group_k = combined_k // KP
                        lane_k = combined_k - group_k * KP
                        group_v = combined_v // KP
                        lane_v = combined_v - group_v * KP
                        K_shared[i, d] = T.if_then_else(
                            live_valid,
                            KNew[live_idx, kv_head, d],
                            T.if_then_else(
                                cache_valid,
                                KVCache[phys_page, page_offset, group_k, lane_k, d],
                                T.Cast(kv_ts, 0.0),
                            ),
                        )
                        V_shared[i, d] = T.if_then_else(
                            live_valid,
                            VNew[live_idx, kv_head, d],
                            T.if_then_else(
                                cache_valid,
                                KVCache[phys_page, page_offset, group_v, lane_v, d],
                                T.Cast(kv_ts, 0.0),
                            ),
                        )

                    for i, d in T.Parallel(BK, D):
                        QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                    T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                    for i in T.Parallel(BK):
                        kv_pos = tile_start + i
                        token_valid = (kv_pos < kv_len) & (kv_pos <= q_pos) & (kv_pos >= valid_start)
                        score = S_local[i] * scale
                        if use_k_scale:
                            score = score * k_scale_value
                        if use_q_scale:
                            score = score * q_scale_value
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
                out_val = O_local[d] * inv_l[0]
                if use_v_scale:
                    out_val = out_val * v_scale_value
                O[qx, hx, d] = T.Cast(q_ts, out_val)

    return rpa_v3_fwd
