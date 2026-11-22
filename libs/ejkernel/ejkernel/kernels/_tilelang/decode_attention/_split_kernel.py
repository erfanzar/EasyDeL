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

"""FlashDecoding-style split-K decode for tile-lang.

For decode workloads with long KV (L >= a few hundred), the regular
single-CTA-per-(batch, head) scan is bottlenecked by HBM bandwidth and
under-utilises the SMs. Split-K partitions the K/V dimension across
``num_splits`` CTAs that each produce a partial ``(m_i, l_i, o_i)``;
a tiny combine kernel then merges them with the canonical log-sum-exp
reduction:

    m = max_i(m_i)
    s_i = exp(m_i - m) * l_i
    o = sum_i(s_i * o_i) / sum_i(s_i)

KV layout into the per-split kernel is ``(B, L, H, D)``. Each split
covers ``L // num_splits`` contiguous tokens.
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
        raise TypeError(f"Unsupported dtype for split-K decode: {dtype}")
    return mapping[canonical]


def make_split_decode_prim_func(
    *,
    batch: int,
    num_heads: int,
    seq_len_kv: int,
    head_dim: int,
    num_splits: int,
    block_k: int,
    softmax_scale: float,
    dtype,
    threads: int = 128,
    num_stages: int = 3,
):
    """Per-split decode forward.

    Grid: ``(num_splits, num_heads, batch)``. Each CTA owns one slice of
    the K/V axis ``[split * (L // num_splits), (split + 1) * (L // num_splits))``
    and writes ``(O_partial, M_partial, L_partial)``.

    Returns:
        ``@T.prim_func`` with buffers
        ``(Q, K, V, O_partial, M_partial, L_partial)`` where the partials
        have shape ``(num_splits, batch, num_heads, ...)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, H, L, D = batch, num_heads, seq_len_kv, head_dim
    BK = block_k
    SPLIT = num_splits
    SPLIT_LEN = L // SPLIT
    log2e = 1.4426950408889634
    scale_log2e = float(softmax_scale) * log2e

    @T.prim_func
    def split_decode(
        Q: T.Tensor((B, H, D), ts),
        K: T.Tensor((B, L, H, D), ts),
        V: T.Tensor((B, L, H, D), ts),
        O_partial: T.Tensor((SPLIT, B, H, D), accum),
        M_partial: T.Tensor((SPLIT, B, H), accum),
        L_partial: T.Tensor((SPLIT, B, H), accum),
    ):
        with T.Kernel(SPLIT, H, B, threads=threads) as (sx, hx, bx):
            Q_loc = T.alloc_fragment((D,), accum)
            K_shared = T.alloc_shared((BK, D), ts)
            V_shared = T.alloc_shared((BK, D), ts)
            QK_prod = T.alloc_fragment((BK, D), accum)
            S_local = T.alloc_fragment((BK,), accum)
            P_local = T.alloc_fragment((BK,), accum)
            PV_prod = T.alloc_fragment((BK, D), accum)
            pv_sum = T.alloc_fragment((D,), accum)
            row_max = T.alloc_fragment((1,), accum)
            row_sum = T.alloc_fragment((1,), accum)
            m_run = T.alloc_fragment((1,), accum)
            l_run = T.alloc_fragment((1,), accum)
            m_new = T.alloc_fragment((1,), accum)
            alpha = T.alloc_fragment((1,), accum)
            O_local = T.alloc_fragment((D,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[bx, hx, d])

            T.fill(O_local, 0)
            m_run[0] = -float("inf")
            l_run[0] = 0.0

            kv_start = sx * SPLIT_LEN
            for k_iter in T.Pipelined(T.ceildiv(SPLIT_LEN, BK), num_stages=num_stages):
                for i, d in T.Parallel(BK, D):
                    k_idx = kv_start + k_iter * BK + i
                    in_range = (k_idx < kv_start + SPLIT_LEN) & (k_idx < L)
                    K_shared[i, d] = T.if_then_else(in_range, K[bx, k_idx, hx, d], T.Cast(ts, 0.0))
                    V_shared[i, d] = T.if_then_else(in_range, V[bx, k_idx, hx, d], T.Cast(ts, 0.0))

                for i, d in T.Parallel(BK, D):
                    QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                for i in T.Parallel(BK):
                    k_idx = kv_start + k_iter * BK + i
                    valid = (k_idx < kv_start + SPLIT_LEN) & (k_idx < L)
                    S_local[i] = T.if_then_else(
                        valid,
                        S_local[i] * scale_log2e,
                        -float("inf"),
                    )

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

            for d in T.Parallel(D):
                O_partial[sx, bx, hx, d] = O_local[d]
            M_partial[sx, bx, hx] = m_run[0]
            L_partial[sx, bx, hx] = l_run[0]

    return split_decode


def make_combine_prim_func(
    *,
    batch: int,
    num_heads: int,
    head_dim: int,
    num_splits: int,
    dtype,
    threads: int = 128,
):
    """Combine kernel.

    Grid: ``(num_heads, batch)``. Each CTA reads the ``num_splits`` partial
    ``(o_i, m_i, l_i)`` rows for its ``(batch, head)`` and emits the
    canonical log-sum-exp merge into the final ``(B, H, D)`` output and
    the natural-log LSE ``(B, H)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, H, D = batch, num_heads, head_dim
    SPLIT = num_splits

    @T.prim_func
    def combine(
        O_partial: T.Tensor((SPLIT, B, H, D), accum),
        M_partial: T.Tensor((SPLIT, B, H), accum),
        L_partial: T.Tensor((SPLIT, B, H), accum),
        O: T.Tensor((B, H, D), ts),
        LSE: T.Tensor((B, H), accum),
    ):
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            m_global = T.alloc_fragment((1,), accum)
            s_total = T.alloc_fragment((1,), accum)
            O_out = T.alloc_fragment((D,), accum)
            scale_per_split = T.alloc_fragment((SPLIT,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)

            m_global[0] = -float("inf")
            for s in T.serial(SPLIT):
                m_global[0] = T.max(m_global[0], M_partial[s, bx, hx])

            s_total[0] = 0.0
            for s in T.serial(SPLIT):
                e_i = T.exp2(M_partial[s, bx, hx] - m_global[0])
                scale_per_split[s] = e_i
                s_total[0] = s_total[0] + e_i * L_partial[s, bx, hx]

            T.fill(O_out, 0)
            inv_denom = T.alloc_fragment((1,), accum)
            inv_denom[0] = 1.0 / T.max(s_total[0], 1e-30)
            for s in T.serial(SPLIT):
                w = scale_per_split[s] * inv_denom[0]
                for d in T.Parallel(D):
                    O_out[d] = O_out[d] + w * O_partial[s, bx, hx, d]

            for d in T.Parallel(D):
                O[bx, hx, d] = T.Cast(ts, O_out[d])
            ln2 = 0.6931471805599453
            LSE[bx, hx] = (m_global[0] + T.log2(T.max(s_total[0], 1e-30))) * ln2

    return combine
