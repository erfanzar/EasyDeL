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

"""TileLang prim_func factories for native selected-block sparse attention.

This module provides three ``@T.prim_func`` factory functions:

``make_sparse_fwd_prim_func``
    Forward-pass causal sparse attention.  Grid: ``(HQ, T, B)``.  Each CTA
    iterates over a fixed set of KV blocks identified by ``BlockIndices`` and
    ``BlockCounts``, computing online softmax with either ``T.exp2`` (small
    blocks) or ``T.exp`` (larger blocks).  GQA head mapping is handled via
    ``G = HQ // HKV``.

``make_sparse_bwd_partials_prim_func``
    Backward pass — computes ``dQ`` and partial gradients ``dKPart``,
    ``dVPart`` (shape ``[B, T, HQ, NS, BS, D]``).  Each CTA recomputes the
    forward attention weights (recompute-style backward) and accumulates
    per-block partials using the Bahdanau delta rule.

``make_sparse_reduce_kv_prim_func``
    Scatter-reduce step that sums ``dKPart`` and ``dVPart`` into the final
    ``dK`` and ``dV`` tensors indexed over ``HKV`` KV heads.

**Index layout** (``index_layout`` / ``count_layout``):
    - ``0`` (``_TOKEN_LAYOUT``): ``BlockIndices[B, T, HKV, NS]``,
      ``BlockCounts[B, T, HKV]``.
    - ``1`` (``_BLOCK_LAYOUT``): ``BlockIndices[B, HKV, NB, NS]``,
      ``BlockCounts[B, HKV, NB]``.
    The layout code is baked in at compile time as a Python-level ``if``; the
    TileLang compiler does not see both branches.

**Shared memory**: ``K_shared`` and ``V_shared`` are ``(BS, D)`` tiles in the
    activation dtype; only used when ``BS >= 64`` (block-level path).
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Map a JAX/NumPy dtype to the TileLang type string for sparse attention.

    Args:
        dtype: Any dtype accepted by ``jnp.dtype()`` — float16, bfloat16,
            float32.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, ``"float32"``.

    Raises:
        TypeError: If *dtype* is not one of the three supported floating-point
            types.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for native_sparse_attention: {dtype}")
    return mapping[canonical]


def make_sparse_fwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_selected: int,
    block_size: int,
    index_dim1: int,
    index_dim2: int,
    count_dim1: int,
    count_dim2: int,
    index_layout: int,
    count_layout: int,
    count_is_scalar: bool,
    count_value: int,
    softmax_scale: float,
    dtype,
    threads: int = 128,
):
    """Build the sparse-attention forward ``@T.prim_func``.

    Grid: ``(num_q_heads, seq_len, batch)``.  One CTA per ``(head, token,
    batch)`` triple.

    The CTA iterates over ``num_selected`` KV block indices (from
    ``BlockIndices``) and, for each valid block, loads a ``(block_size,
    head_dim)`` tile of K and V into shared memory (block-level path when
    ``block_size >= 64``) or processes tokens sequentially (token-level path
    when ``block_size < 64``).  Causal masking ensures only tokens
    ``<= query_token`` contribute.

    Two softmax implementations are selected at build time:
        - ``block_size <= 32``: ``T.exp2`` and pre-multiplied scale
          (``softmax_scale * log2e``) for speed.
        - ``block_size > 32``: standard ``T.exp`` for numerical safety.

    Args:
        batch: Batch size ``B``.
        seq_len: Sequence length ``T``.
        num_q_heads: Number of query heads ``HQ``.
        num_kv_heads: Number of KV heads ``HKV``; ``HQ`` must be divisible by
            ``HKV`` (GQA).
        head_dim: Per-head dimension ``D``.
        num_selected: Maximum number of selected KV blocks per token/block
            (``NS``); determines the last dim of ``BlockIndices``.
        block_size: Tokens per KV block (``BS``).
        index_dim1: Second dimension of ``BlockIndices`` (``T`` for
            ``_TOKEN_LAYOUT``, ``HKV`` for ``_BLOCK_LAYOUT``).
        index_dim2: Third dimension of ``BlockIndices`` (``HKV`` for
            ``_TOKEN_LAYOUT``, ``NB`` for ``_BLOCK_LAYOUT``).
        count_dim1: Second dimension of ``BlockCounts``.
        count_dim2: Third dimension of ``BlockCounts``.
        index_layout: ``0`` for token-level layout, ``1`` for block-level.
        count_layout: ``0`` for token-level layout, ``1`` for block-level.
        count_is_scalar: If ``True``, *count_value* is used as a static count
            for all positions; ``BlockCounts`` is ignored.
        count_value: Scalar count used when *count_is_scalar* is ``True``.
        softmax_scale: Attention temperature multiplier.
        dtype: Activation dtype (float16, bfloat16, float32).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Q, K, V, BlockIndices, BlockCounts, O)`` where all tensors are in
        *dtype* except ``BlockIndices`` and ``BlockCounts`` which are int32.
        Output ``O`` has shape ``[B, T, HQ, D]``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, T_, HQ, HKV, D = batch, seq_len, num_q_heads, num_kv_heads, head_dim
    NS, BS = num_selected, block_size
    I1, I2, C1, C2 = index_dim1, index_dim2, count_dim1, count_dim2
    NB = (T_ + BS - 1) // BS
    G = HQ // HKV
    scale = float(softmax_scale)
    use_exp2 = BS <= 32
    use_block_level = BS >= 64
    score_scale = scale * 1.4426950408889634 if use_exp2 else scale
    idx_layout = int(index_layout)
    cnt_layout = int(count_layout)
    cnt_scalar = bool(count_is_scalar)
    cnt_value = int(count_value)

    @T.prim_func
    def sparse_fwd(
        Q: T.Tensor((B, T_, HQ, D), ts),
        K: T.Tensor((B, T_, HKV, D), ts),
        V: T.Tensor((B, T_, HKV, D), ts),
        BlockIndices: T.Tensor((B, I1, I2, NS), "int32"),
        BlockCounts: T.Tensor((B, C1, C2), "int32"),
        O: T.Tensor((B, T_, HQ, D), ts),
    ):
        with T.Kernel(HQ, T_, B, threads=threads) as (hx, tx, bx):
            acc = T.alloc_fragment((D,), accum)
            q_loc = T.alloc_fragment((D,), accum)
            K_shared = T.alloc_shared((BS, D), ts)
            V_shared = T.alloc_shared((BS, D), ts)
            S_local = T.alloc_fragment((BS,), accum)
            P_local = T.alloc_fragment((BS,), accum)
            QK_prod = T.alloc_fragment((BS, D), accum)
            PV_prod = T.alloc_fragment((BS, D), accum)
            pv_sum = T.alloc_fragment((D,), accum)
            prod = T.alloc_fragment((D,), accum)
            score = T.alloc_fragment((1,), accum)
            row_max = T.alloc_fragment((1,), accum)
            row_sum = T.alloc_fragment((1,), accum)
            m_run = T.alloc_fragment((1,), accum)
            m_new = T.alloc_fragment((1,), accum)
            l_run = T.alloc_fragment((1,), accum)
            alpha = T.alloc_fragment((1,), accum)
            p = T.alloc_fragment((1,), accum)
            block_idx = T.alloc_fragment((1,), "int32")
            count = T.alloc_fragment((1,), "int32")
            _b_ref = T.alloc_fragment((1,), accum)
            _t_ref = T.alloc_fragment((1,), accum)
            _hkv_ref = T.alloc_fragment((1,), accum)
            _nb_ref = T.alloc_fragment((1,), accum)
            _i1_ref = T.alloc_fragment((1,), accum)
            _i2_ref = T.alloc_fragment((1,), accum)
            _c1_ref = T.alloc_fragment((1,), accum)
            _c2_ref = T.alloc_fragment((1,), accum)
            kvh = hx // G
            q_block = tx // BS
            _b_ref[0] = B
            _t_ref[0] = T_
            _hkv_ref[0] = HKV
            _nb_ref[0] = NB
            _i1_ref[0] = I1
            _i2_ref[0] = I2
            _c1_ref[0] = C1
            _c2_ref[0] = C2

            for d in T.Parallel(D):
                q_loc[d] = T.Cast(accum, Q[bx, tx, hx, d])
                acc[d] = 0.0
            m_run[0] = -1.0e30
            l_run[0] = 0.0

            if cnt_scalar:
                count[0] = cnt_value
            else:
                if cnt_layout == 0:
                    count[0] = BlockCounts[bx, tx, kvh]
                else:
                    count[0] = BlockCounts[bx, kvh, q_block]

            for si in T.serial(NS):
                if idx_layout == 0:
                    block_idx[0] = BlockIndices[bx, tx, kvh, si]
                else:
                    block_idx[0] = BlockIndices[bx, kvh, q_block, si]

                block_start = block_idx[0] * BS
                if (si < count[0]) & (block_idx[0] >= 0) & (block_idx[0] < NB) & (block_start <= tx):
                    if use_block_level:
                        for pi, d in T.Parallel(BS, D):
                            token = block_start + pi
                            safe_token = T.min(token, T_ - 1)
                            token_valid = (token < T_) & (token <= tx)
                            K_shared[pi, d] = T.if_then_else(
                                token_valid,
                                K[bx, safe_token, kvh, d],
                                T.Cast(ts, 0.0),
                            )
                            V_shared[pi, d] = T.if_then_else(
                                token_valid,
                                V[bx, safe_token, kvh, d],
                                T.Cast(ts, 0.0),
                            )

                        for pi, d in T.Parallel(BS, D):
                            QK_prod[pi, d] = q_loc[d] * T.Cast(accum, K_shared[pi, d])
                        T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                        for pi in T.Parallel(BS):
                            token = block_start + pi
                            token_valid = (token < T_) & (token <= tx)
                            S_local[pi] = T.if_then_else(token_valid, S_local[pi] * score_scale, -1.0e30)

                        T.reduce_max(S_local, row_max, dim=0, clear=True)
                        m_new[0] = T.max(m_run[0], row_max[0])
                        if use_exp2:
                            alpha[0] = T.exp2(m_run[0] - m_new[0])
                            for pi in T.Parallel(BS):
                                P_local[pi] = T.exp2(S_local[pi] - m_new[0])
                        else:
                            alpha[0] = T.exp(m_run[0] - m_new[0])
                            for pi in T.Parallel(BS):
                                P_local[pi] = T.exp(S_local[pi] - m_new[0])
                        T.reduce_sum(P_local, row_sum, dim=0, clear=True)
                        l_run[0] = l_run[0] * alpha[0] + row_sum[0]

                        for d in T.Parallel(D):
                            acc[d] = acc[d] * alpha[0]
                        for pi, d in T.Parallel(BS, D):
                            PV_prod[pi, d] = P_local[pi] * T.Cast(accum, V_shared[pi, d])
                        T.reduce_sum(PV_prod, pv_sum, dim=0, clear=True)
                        for d in T.Parallel(D):
                            acc[d] = acc[d] + pv_sum[d]
                        m_run[0] = m_new[0]
                    else:
                        for pi in T.serial(BS):
                            token = block_start + pi
                            safe_token = T.min(token, T_ - 1)
                            token_valid = (token < T_) & (token <= tx)
                            for d in T.Parallel(D):
                                prod[d] = T.if_then_else(
                                    token_valid,
                                    q_loc[d] * T.Cast(accum, K[bx, safe_token, kvh, d]),
                                    0.0,
                                )
                            T.reduce_sum(prod, score, dim=0, clear=True)
                            score[0] = T.if_then_else(token_valid, score[0] * score_scale, -1.0e30)
                            m_new[0] = T.max(m_run[0], score[0])
                            if use_exp2:
                                alpha[0] = T.exp2(m_run[0] - m_new[0])
                                p[0] = T.if_then_else(token_valid, T.exp2(score[0] - m_new[0]), 0.0)
                            else:
                                alpha[0] = T.exp(m_run[0] - m_new[0])
                                p[0] = T.if_then_else(token_valid, T.exp(score[0] - m_new[0]), 0.0)
                            for d in T.Parallel(D):
                                acc[d] = acc[d] * alpha[0] + p[0] * T.Cast(accum, V[bx, safe_token, kvh, d])
                            l_run[0] = l_run[0] * alpha[0] + p[0]
                            m_run[0] = m_new[0]

            for d in T.Parallel(D):
                if l_run[0] > 0.0:
                    O[bx, tx, hx, d] = T.Cast(ts, acc[d] / l_run[0])
                else:
                    O[bx, tx, hx, d] = T.Cast(ts, 0.0)

    return sparse_fwd


def make_sparse_bwd_partials_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_selected: int,
    block_size: int,
    index_dim1: int,
    index_dim2: int,
    count_dim1: int,
    count_dim2: int,
    index_layout: int,
    count_layout: int,
    count_is_scalar: bool,
    count_value: int,
    softmax_scale: float,
    dtype,
    threads: int = 128,
):
    """Build the sparse-attention backward ``@T.prim_func`` for partial K/V grads.

    Grid: ``(num_q_heads, seq_len, batch)``.  One CTA per ``(head, token,
    batch)`` triple.

    This kernel implements a *recompute-style* backward: it re-runs the sparse
    forward attention to reconstruct the attention weights, then computes:

    - ``dQ[b, t, h, :]`` via the standard softmax-attention VJP rule.
    - ``dKPart[b, t, h, si, pi, :]`` — unscattered partial gradient wrt the
      *si*-th selected block's *pi*-th token, for each query position.
    - ``dVPart[b, t, h, si, pi, :]`` — corresponding partial gradient wrt V.

    The partial K/V grads are later reduced by ``make_sparse_reduce_kv_prim_func``.

    Note:
        Uses ``T.exp`` (not ``T.exp2``) for all softmax operations regardless
        of ``block_size``, unlike the forward pass.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimension.
        num_selected: Maximum number of selected KV blocks per position.
        block_size: Tokens per KV block.
        index_dim1: Second dim of ``BlockIndices``.
        index_dim2: Third dim of ``BlockIndices``.
        count_dim1: Second dim of ``BlockCounts``.
        count_dim2: Third dim of ``BlockCounts``.
        index_layout: ``0`` = token layout, ``1`` = block layout.
        count_layout: ``0`` = token layout, ``1`` = block layout.
        count_is_scalar: If ``True``, *count_value* is the static block count.
        count_value: Static count used when *count_is_scalar* is ``True``.
        softmax_scale: Attention temperature.
        dtype: Activation dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Q, K, V, BlockIndices, BlockCounts, dO, dQ, dKPart, dVPart)``.
        ``dQ`` is in *dtype*; ``dKPart`` and ``dVPart`` are float32 with shape
        ``[B, T, HQ, NS, BS, D]``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, T_, HQ, HKV, D = batch, seq_len, num_q_heads, num_kv_heads, head_dim
    NS, BS = num_selected, block_size
    I1, I2, C1, C2 = index_dim1, index_dim2, count_dim1, count_dim2
    NB = (T_ + BS - 1) // BS
    G = HQ // HKV
    scale = float(softmax_scale)
    idx_layout = int(index_layout)
    cnt_layout = int(count_layout)
    cnt_scalar = bool(count_is_scalar)
    cnt_value = int(count_value)

    @T.prim_func
    def sparse_bwd_partials(
        Q: T.Tensor((B, T_, HQ, D), ts),
        K: T.Tensor((B, T_, HKV, D), ts),
        V: T.Tensor((B, T_, HKV, D), ts),
        BlockIndices: T.Tensor((B, I1, I2, NS), "int32"),
        BlockCounts: T.Tensor((B, C1, C2), "int32"),
        dO: T.Tensor((B, T_, HQ, D), ts),
        dQ: T.Tensor((B, T_, HQ, D), ts),
        dKPart: T.Tensor((B, T_, HQ, NS, BS, D), accum),
        dVPart: T.Tensor((B, T_, HQ, NS, BS, D), accum),
    ):
        with T.Kernel(HQ, T_, B, threads=threads) as (hx, tx, bx):
            q_loc = T.alloc_fragment((D,), accum)
            do_loc = T.alloc_fragment((D,), accum)
            scores = T.alloc_fragment((NS, BS), accum)
            probs = T.alloc_fragment((NS, BS), accum)
            z = T.alloc_fragment((NS, BS), accum)
            ds = T.alloc_fragment((NS, BS), accum)
            valid = T.alloc_fragment((NS, BS), "int32")
            block_idx = T.alloc_fragment((NS,), "int32")
            token_idx = T.alloc_fragment((NS, BS), "int32")
            count = T.alloc_fragment((1,), "int32")
            row_max = T.alloc_fragment((1,), accum)
            row_sum = T.alloc_fragment((1,), accum)
            mu = T.alloc_fragment((1,), accum)
            dq_loc = T.alloc_fragment((D,), accum)
            _b_ref = T.alloc_fragment((1,), accum)
            _t_ref = T.alloc_fragment((1,), accum)
            _hkv_ref = T.alloc_fragment((1,), accum)
            _nb_ref = T.alloc_fragment((1,), accum)
            _i1_ref = T.alloc_fragment((1,), accum)
            _i2_ref = T.alloc_fragment((1,), accum)
            _c1_ref = T.alloc_fragment((1,), accum)
            _c2_ref = T.alloc_fragment((1,), accum)
            kvh = hx // G
            q_block = tx // BS
            _b_ref[0] = B
            _t_ref[0] = T_
            _hkv_ref[0] = HKV
            _nb_ref[0] = NB
            _i1_ref[0] = I1
            _i2_ref[0] = I2
            _c1_ref[0] = C1
            _c2_ref[0] = C2

            for d in T.Parallel(D):
                q_loc[d] = T.Cast(accum, Q[bx, tx, hx, d])
                do_loc[d] = T.Cast(accum, dO[bx, tx, hx, d])
                dq_loc[d] = 0.0

            if cnt_scalar:
                count[0] = cnt_value
            else:
                if cnt_layout == 0:
                    count[0] = BlockCounts[bx, tx, kvh]
                else:
                    count[0] = BlockCounts[bx, kvh, q_block]

            row_max[0] = -1.0e30
            for si in T.serial(NS):
                if idx_layout == 0:
                    block_idx[si] = BlockIndices[bx, tx, kvh, si]
                else:
                    block_idx[si] = BlockIndices[bx, kvh, q_block, si]
                for pi in T.serial(BS):
                    token_idx[si, pi] = block_idx[si] * BS + pi
                    valid[si, pi] = T.if_then_else(
                        (si < count[0])
                        & (block_idx[si] >= 0)
                        & (block_idx[si] < NB)
                        & (token_idx[si, pi] < T_)
                        & (token_idx[si, pi] <= tx),
                        1,
                        0,
                    )
                    scores[si, pi] = 0.0
                    for d in T.serial(D):
                        if valid[si, pi] != 0:
                            scores[si, pi] = scores[si, pi] + q_loc[d] * T.Cast(accum, K[bx, token_idx[si, pi], kvh, d])
                    scores[si, pi] = T.if_then_else(valid[si, pi] != 0, scores[si, pi] * scale, -1.0e30)
                    row_max[0] = T.max(row_max[0], scores[si, pi])

            row_sum[0] = 0.0
            for si in T.serial(NS):
                for pi in T.serial(BS):
                    probs[si, pi] = T.if_then_else(valid[si, pi] != 0, T.exp(scores[si, pi] - row_max[0]), 0.0)
                    row_sum[0] = row_sum[0] + probs[si, pi]
            for si in T.serial(NS):
                for pi in T.serial(BS):
                    probs[si, pi] = T.if_then_else(row_sum[0] > 0.0, probs[si, pi] / row_sum[0], 0.0)

            mu[0] = 0.0
            for si in T.serial(NS):
                for pi in T.serial(BS):
                    z[si, pi] = 0.0
                    for d in T.serial(D):
                        if valid[si, pi] != 0:
                            z[si, pi] = z[si, pi] + T.Cast(accum, V[bx, token_idx[si, pi], kvh, d]) * do_loc[d]
                    mu[0] = mu[0] + probs[si, pi] * z[si, pi]

            for si in T.serial(NS):
                for pi in T.serial(BS):
                    ds[si, pi] = T.if_then_else(valid[si, pi] != 0, probs[si, pi] * (z[si, pi] - mu[0]), 0.0)

            for d in T.Parallel(D):
                for si in T.serial(NS):
                    for pi in T.serial(BS):
                        if valid[si, pi] != 0:
                            dq_loc[d] = dq_loc[d] + scale * ds[si, pi] * T.Cast(accum, K[bx, token_idx[si, pi], kvh, d])
                            dKPart[bx, tx, hx, si, pi, d] = scale * ds[si, pi] * q_loc[d]
                            dVPart[bx, tx, hx, si, pi, d] = probs[si, pi] * do_loc[d]
                        else:
                            dKPart[bx, tx, hx, si, pi, d] = 0.0
                            dVPart[bx, tx, hx, si, pi, d] = 0.0
                dQ[bx, tx, hx, d] = T.Cast(ts, dq_loc[d])

    return sparse_bwd_partials


def make_sparse_reduce_kv_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_selected: int,
    block_size: int,
    index_dim1: int,
    index_dim2: int,
    count_dim1: int,
    count_dim2: int,
    index_layout: int,
    count_layout: int,
    count_is_scalar: bool,
    count_value: int,
    dtype,
    threads: int = 128,
):
    """Build the partial K/V gradient scatter-reduce ``@T.prim_func``.

    Grid: ``(num_kv_heads, seq_len, batch)``.  One CTA per ``(kv_head, token,
    batch)`` triple.

    For each query token ``qt`` in ``[0, seq_len)`` and for each selected block
    slot ``si`` that maps to physical token ``kt`` (the CTA's ``kx``), the CTA
    adds ``dKPart[b, qt, hx, si, pi, :]`` (summed over the GQA group) into
    ``acc_k`` and similarly for ``dVPart``.  The accumulated float32 values are
    then cast to *dtype* and written to ``dK`` / ``dV``.

    This kernel has ``O(T^2)`` work per batch item in the worst case because it
    scans all query tokens for every KV token; it is intended to be run once
    per backward pass, not in the inner loop.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimension.
        num_selected: Maximum number of selected KV blocks per position.
        block_size: Tokens per KV block.
        index_dim1: Second dim of ``BlockIndices``.
        index_dim2: Third dim of ``BlockIndices``.
        count_dim1: Second dim of ``BlockCounts``.
        count_dim2: Third dim of ``BlockCounts``.
        index_layout: ``0`` = token layout, ``1`` = block layout.
        count_layout: ``0`` = token layout, ``1`` = block layout.
        count_is_scalar: If ``True``, *count_value* is the static block count.
        count_value: Static count used when *count_is_scalar* is ``True``.
        dtype: Activation dtype; ``dK`` and ``dV`` are written in this dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(BlockIndices, BlockCounts, dKPart, dVPart, dK, dV)`` where
        ``dKPart`` / ``dVPart`` are float32 and ``dK`` / ``dV`` are in *dtype*,
        both with shape ``[B, T, HKV, D]``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, T_, HQ, HKV, D = batch, seq_len, num_q_heads, num_kv_heads, head_dim
    NS, BS = num_selected, block_size
    I1, I2, C1, C2 = index_dim1, index_dim2, count_dim1, count_dim2
    NB = (T_ + BS - 1) // BS
    G = HQ // HKV
    idx_layout = int(index_layout)
    cnt_layout = int(count_layout)
    cnt_scalar = bool(count_is_scalar)
    cnt_value = int(count_value)

    @T.prim_func
    def sparse_reduce_kv(
        BlockIndices: T.Tensor((B, I1, I2, NS), "int32"),
        BlockCounts: T.Tensor((B, C1, C2), "int32"),
        dKPart: T.Tensor((B, T_, HQ, NS, BS, D), accum),
        dVPart: T.Tensor((B, T_, HQ, NS, BS, D), accum),
        dK: T.Tensor((B, T_, HKV, D), ts),
        dV: T.Tensor((B, T_, HKV, D), ts),
    ):
        with T.Kernel(HKV, T_, B, threads=threads) as (kvh, kt, bx):
            acc_k = T.alloc_fragment((D,), accum)
            acc_v = T.alloc_fragment((D,), accum)
            block_idx = T.alloc_fragment((1,), "int32")
            count = T.alloc_fragment((1,), "int32")
            token_idx = T.alloc_fragment((1,), "int32")
            q_block = T.alloc_fragment((1,), "int32")
            valid = T.alloc_fragment((1,), "int32")
            _b_ref = T.alloc_fragment((1,), accum)
            _t_ref = T.alloc_fragment((1,), accum)
            _hq_ref = T.alloc_fragment((1,), accum)
            _nb_ref = T.alloc_fragment((1,), accum)
            _i1_ref = T.alloc_fragment((1,), accum)
            _i2_ref = T.alloc_fragment((1,), accum)
            _c1_ref = T.alloc_fragment((1,), accum)
            _c2_ref = T.alloc_fragment((1,), accum)
            _b_ref[0] = B
            _t_ref[0] = T_
            _hq_ref[0] = HQ
            _nb_ref[0] = NB
            _i1_ref[0] = I1
            _i2_ref[0] = I2
            _c1_ref[0] = C1
            _c2_ref[0] = C2

            for d in T.Parallel(D):
                acc_k[d] = 0.0
                acc_v[d] = 0.0

            for qt in T.serial(T_):
                q_block[0] = qt // BS
                if cnt_scalar:
                    count[0] = cnt_value
                else:
                    if cnt_layout == 0:
                        count[0] = BlockCounts[bx, qt, kvh]
                    else:
                        count[0] = BlockCounts[bx, kvh, q_block[0]]
                for si in T.serial(NS):
                    if idx_layout == 0:
                        block_idx[0] = BlockIndices[bx, qt, kvh, si]
                    else:
                        block_idx[0] = BlockIndices[bx, kvh, q_block[0], si]
                    for pi in T.serial(BS):
                        token_idx[0] = block_idx[0] * BS + pi
                        valid[0] = T.if_then_else(
                            (si < count[0])
                            & (block_idx[0] >= 0)
                            & (block_idx[0] < NB)
                            & (token_idx[0] == kt)
                            & (kt <= qt),
                            1,
                            0,
                        )
                        if valid[0] != 0:
                            for g in T.serial(G):
                                hx = kvh * G + g
                                for d in T.Parallel(D):
                                    acc_k[d] = acc_k[d] + dKPart[bx, qt, hx, si, pi, d]
                                    acc_v[d] = acc_v[d] + dVPart[bx, qt, hx, si, pi, d]

            for d in T.Parallel(D):
                dK[bx, kt, kvh, d] = T.Cast(ts, acc_k[d])
                dV[bx, kt, kvh, d] = T.Cast(ts, acc_v[d])

    return sparse_reduce_kv
