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

"""TileLang prim_func builders for dense ragged single-query decode attention.

Two kernel variants are produced by :func:`make_ragged_decode_prim_func`:

* **No-sink variant** (``aux_kind == 0`` or ``num_sinks == 0``): grid is
  ``T.Kernel(num_q_heads, batch)``.  Online softmax is initialised cold
  (``m_run = -inf``, ``l_run = 0``).

* **Sink variant** (``aux_kind != 0`` and ``num_sinks > 0``): same grid but
  ``m_run``/``l_run`` are primed from ``softmax_aux`` before the KV loop so
  that sink tokens dominate the normalisation.

Both variants share the same FlashAttention-2 recurrence in log₂ space
(``T.exp2``) with optional tanh soft-capping and optional sliding-window masking.

Shared-memory layout per CTA: one ``(block_k, head_dim)`` tile each for
``K_shared`` and ``V_shared`` in the compute dtype.

Accumulation: ``float32`` fragments throughout; final output is cast back to the
input dtype on store.

Pipeline: ``T.Pipelined`` with ``num_stages`` (default 3) software pipeline stages.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Convert a JAX/NumPy floating-point dtype to its TileLang string name.

    Args:
        dtype: Any dtype accepted by ``jnp.dtype()``.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, or ``"float32"``.

    Raises:
        TypeError: if ``dtype`` is not a supported floating-point type.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for ragged decode attention: {dtype}")
    return mapping[canonical]


def _index_dtype_str(dtype) -> str:
    """Convert a JAX/NumPy integer dtype to its TileLang string name.

    Args:
        dtype: Any dtype accepted by ``jnp.dtype()``.

    Returns:
        One of ``"int32"`` or ``"int64"``.

    Raises:
        TypeError: if ``dtype`` is not int32 or int64.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.int32): "int32",
        jnp.dtype(jnp.int64): "int64",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported index dtype for ragged decode attention: {dtype}")
    return mapping[canonical]


def make_ragged_decode_prim_func(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    window_left: int,
    window_right: int,
    logits_soft_cap: float,
    dtype,
    index_dtype,
    aux_dtype=None,
    aux_kind: int = 0,
    aux_heads: int = 0,
    num_sinks: int = 0,
    num_stages: int = 3,
    threads: int = 128,
):
    """Return a TileLang ``@T.prim_func`` for dense ragged single-token decode.

    All parameters are baked in as Python-level constants so TileLang can fold
    them at compile time.

    Grid: ``T.Kernel(num_q_heads, batch)`` — one CTA per (head, batch) pair.

    Shared memory per CTA:
        * ``K_shared``: ``(block_k, head_dim)`` in compute dtype.
        * ``V_shared``: ``(block_k, head_dim)`` in compute dtype.

    Pipeline: ``T.Pipelined(ceil(seq_len/block_k), num_stages=num_stages)`` over
    the KV tile dimension.

    Online softmax uses log₂ space (``T.exp2``) to avoid transcendental overhead.
    When ``logits_soft_cap > 0`` each raw dot-product score is first passed
    through ``tanh(score / cap) * cap`` before scaling.

    Args:
        batch: Batch size ``B``.
        num_q_heads: Number of query heads ``HQ``.
        num_kv_heads: Number of KV heads ``HKV``; must divide ``num_q_heads``.
            GQA grouping ``q_heads_per_kv = HQ // HKV`` is computed internally.
        seq_len: Full KV sequence length ``L``.
        head_dim: Attention head dimension ``D``.
        block_k: KV tile size ``BK`` (typically 64 or 128).
        softmax_scale: Attention scale; usually ``1/sqrt(head_dim)``.
        window_left: Left sliding-window half-size; ``-1`` disables.
        window_right: Right sliding-window half-size; ``-1`` disables.
        logits_soft_cap: Logit soft-cap; ``-1.0`` or any non-positive value disables.
        dtype: Compute dtype for Q/K/V/O (float16, bfloat16, or float32).
        index_dtype: Integer dtype for sequence boundary tensors (int32 or int64).
        aux_dtype: Dtype of the ``Aux`` sink tensor; falls back to ``dtype``
            when ``None``.
        aux_kind: Sink layout selector (0 = none, 1 = shared, 2 = per-KV-head,
            3 = per-Q-head).  A non-zero value combined with ``num_sinks > 0``
            selects the sink-primed code path.
        aux_heads: Number of rows in the ``Aux`` tensor; 0 when unused.
        num_sinks: Number of sink positions; 0 when unused.
        num_stages: Software pipeline stages (TileLang ``T.Pipelined``).
        threads: Threads per CUDA CTA; must be a multiple of 32.

    Returns:
        A TileLang ``@T.prim_func``.  Signature depends on whether sinks are
        active:

        * **With sinks**: ``(Q, K, V, SequenceStart, SequenceEnd, Aux, O)``
        * **Without sinks**: ``(Q, K, V, SequenceStart, SequenceEnd, O)``

        All shapes are fully static (baked into the prim_func at build time).
    """
    ts = _dtype_str(dtype)
    index_ts = _index_dtype_str(index_dtype)
    aux_ts = _dtype_str(dtype if aux_dtype is None else aux_dtype)
    accum = "float32"
    B, HQ, HKV, L, D, BK = batch, num_q_heads, num_kv_heads, seq_len, head_dim, block_k
    AH, NS = max(aux_heads, 1), max(num_sinks, 1)
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale)
    scale_log2e = scale * log2e
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    use_window = window_left >= 0 and window_right >= 0
    use_aux = aux_kind != 0 and num_sinks > 0

    if use_aux:

        @T.prim_func
        def ragged_decode_fwd(
            Q: T.Tensor((B, HQ, D), ts),
            K: T.Tensor((B, L, HKV, D), ts),
            V: T.Tensor((B, L, HKV, D), ts),
            SequenceStart: T.Tensor((B,), index_ts),
            SequenceEnd: T.Tensor((B,), index_ts),
            Aux: T.Tensor((AH, NS), aux_ts),
            O: T.Tensor((B, HQ, D), ts),
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
                _aux_ref = T.alloc_fragment((1,), aux_ts)
                _index_ref = T.alloc_fragment((1,), index_ts)
                _hkv_ref = T.alloc_fragment((HKV,), accum)
                _aux_heads_ref = T.alloc_fragment((AH,), accum)

                kv_head = hx // q_heads_per_kv
                aux_row = 0
                if aux_kind == 2:
                    aux_row = kv_head
                if aux_kind == 3:
                    aux_row = hx
                start_pos = T.Cast("int32", SequenceStart[bx])
                end_pos = T.Cast("int32", SequenceEnd[bx])
                query_pos = end_pos - 1

                for d in T.Parallel(D):
                    Q_loc[d] = T.Cast(accum, Q[bx, hx, d])

                m_run[0] = -1e30
                for s in T.serial(NS):
                    m_run[0] = T.max(m_run[0], T.Cast(accum, Aux[aux_row, s]) * log2e)
                l_run[0] = 0.0
                for s in T.serial(NS):
                    l_run[0] = l_run[0] + T.exp2(T.Cast(accum, Aux[aux_row, s]) * log2e - m_run[0])
                T.fill(O_local, 0)

                for k_iter in T.Pipelined(T.ceildiv(L, BK), num_stages=num_stages):
                    for i, d in T.Parallel(BK, D):
                        k_idx = k_iter * BK + i
                        if use_window:
                            range_valid = (
                                (k_idx < L)
                                & (k_idx >= start_pos)
                                & (k_idx < end_pos)
                                & (k_idx >= query_pos - window_left)
                                & (k_idx <= query_pos + window_right)
                            )
                        else:
                            range_valid = (k_idx < L) & (k_idx >= start_pos) & (k_idx < end_pos)
                        K_shared[i, d] = T.if_then_else(
                            range_valid,
                            K[bx, k_idx, kv_head, d],
                            T.Cast(ts, 0.0),
                        )
                        V_shared[i, d] = T.if_then_else(
                            range_valid,
                            V[bx, k_idx, kv_head, d],
                            T.Cast(ts, 0.0),
                        )

                    for i, d in T.Parallel(BK, D):
                        QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                    T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                    for i in T.Parallel(BK):
                        k_idx = k_iter * BK + i
                        if use_window:
                            range_valid = (
                                (k_idx < L)
                                & (k_idx >= start_pos)
                                & (k_idx < end_pos)
                                & (k_idx >= query_pos - window_left)
                                & (k_idx <= query_pos + window_right)
                            )
                        else:
                            range_valid = (k_idx < L) & (k_idx >= start_pos) & (k_idx < end_pos)
                        score = S_local[i]
                        if use_softcap:
                            score_natural = score * scale
                            tanh_arg = score_natural / softcap
                            soft_score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                            S_local[i] = T.if_then_else(range_valid, soft_score * log2e, -1e30)
                        else:
                            S_local[i] = T.if_then_else(range_valid, score * scale_log2e, -1e30)

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

        return ragged_decode_fwd

    @T.prim_func
    def ragged_decode_fwd(
        Q: T.Tensor((B, HQ, D), ts),
        K: T.Tensor((B, L, HKV, D), ts),
        V: T.Tensor((B, L, HKV, D), ts),
        SequenceStart: T.Tensor((B,), index_ts),
        SequenceEnd: T.Tensor((B,), index_ts),
        O: T.Tensor((B, HQ, D), ts),
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
            start_pos = T.Cast("int32", SequenceStart[bx])
            end_pos = T.Cast("int32", SequenceEnd[bx])
            query_pos = end_pos - 1

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[bx, hx, d])

            T.fill(O_local, 0)
            m_run[0] = -1e30
            l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(L, BK), num_stages=num_stages):
                for i, d in T.Parallel(BK, D):
                    k_idx = k_iter * BK + i
                    range_valid = (k_idx < L) & (k_idx >= start_pos) & (k_idx < end_pos)
                    if use_window:
                        range_valid = (
                            range_valid & (k_idx >= query_pos - window_left) & (k_idx <= query_pos + window_right)
                        )
                    K_shared[i, d] = T.if_then_else(
                        range_valid,
                        K[bx, k_idx, kv_head, d],
                        T.Cast(ts, 0.0),
                    )
                    V_shared[i, d] = T.if_then_else(
                        range_valid,
                        V[bx, k_idx, kv_head, d],
                        T.Cast(ts, 0.0),
                    )

                for i, d in T.Parallel(BK, D):
                    QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                for i in T.Parallel(BK):
                    k_idx = k_iter * BK + i
                    range_valid = (k_idx < L) & (k_idx >= start_pos) & (k_idx < end_pos)
                    if use_window:
                        range_valid = (
                            range_valid & (k_idx >= query_pos - window_left) & (k_idx <= query_pos + window_right)
                        )
                    score = S_local[i]
                    if use_softcap:
                        score_natural = score * scale
                        tanh_arg = score_natural / softcap
                        soft_score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                        S_local[i] = T.if_then_else(range_valid, soft_score * log2e, -1e30)
                    else:
                        S_local[i] = T.if_then_else(range_valid, score * scale_log2e, -1e30)

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

    return ragged_decode_fwd
