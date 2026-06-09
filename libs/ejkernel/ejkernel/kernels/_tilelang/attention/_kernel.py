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

"""Native tile-lang kernels for generic attention auxiliary weights."""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Return the tile-lang dtype string for an activation dtype."""
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang attention: {dtype}")
    return mapping[canonical]


def _scalar_dtype_str(dtype) -> str:
    """Return a tile-lang dtype string for compact feature buffers."""
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.bool_): "bool",
        jnp.dtype(jnp.int8): "int8",
        jnp.dtype(jnp.int16): "int16",
        jnp.dtype(jnp.int32): "int32",
        jnp.dtype(jnp.int64): "int64",
        jnp.dtype(jnp.uint8): "uint8",
        jnp.dtype(jnp.uint16): "uint16",
        jnp.dtype(jnp.uint32): "uint32",
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang attention feature buffer: {dtype}")
    return mapping[canonical]


def make_attention_weights_prim_func(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_q: int,
    block_k: int,
    softmax_scale: float,
    causal: bool,
    dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    softmax_aux_shape: tuple[int, int],
    softmax_aux_dtype,
    use_softmax_aux: bool,
    window: tuple[int, int] | None,
    dropout_prob: float,
    logits_soft_cap: float | None,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build a dense attention-weights ``@T.prim_func``.

    The kernel materialises ``(B, Hq, Sq, Sk)`` probabilities for the
    generic ``attention`` auxiliary output. It recomputes the logits from
    ``Q`` and ``K`` in ``(B, H, N, D)`` layout, applies the XLA reference
    ordering for soft-cap, causal/window masks, bias-vs-mask precedence,
    attention sinks, softmax, and optional native dropout.
    """
    ts = _dtype_str(dtype)
    bias_ts = _scalar_dtype_str(bias_dtype)
    mask_ts = _scalar_dtype_str(mask_dtype)
    aux_ts = _scalar_dtype_str(softmax_aux_dtype)
    accum = "float32"
    B, H, HK = batch, num_heads, num_kv_heads
    NQ, NK, D = seq_len_q, seq_len_k, head_dim
    BQ, BK_TILE = block_q, block_k
    BB, BH, BQS, BKS = bias_shape
    MB, MH, MQS, MKS = mask_shape
    AH, NS = softmax_aux_shape
    G = H // HK
    scale = float(softmax_scale)
    use_cap = logits_soft_cap is not None
    cap = float(logits_soft_cap) if use_cap else 1.0
    inv_cap = 1.0 / cap
    use_window = window is not None
    window_left = int(window[0]) if use_window else 0
    window_right = int(window[1]) if use_window else 0
    use_dropout = float(dropout_prob) > 0.0
    keep_prob = 1.0 - float(dropout_prob)
    inv_keep_prob = 1.0 / keep_prob if use_dropout else 1.0
    neg_big = -1.0e30
    pad_neg = -3.0e38

    @T.prim_func
    def attention_weights(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, HK, NK, D), ts),
        Bias: T.Tensor((BB, BH, BQS, BKS), bias_ts),
        AttentionMask: T.Tensor((MB, MH, MQS, MKS), mask_ts),
        SoftmaxAux: T.Tensor((AH, NS), aux_ts),
        DropoutSeed: T.Tensor((2,), "uint32"),
        Weights: T.Tensor((B, H, NQ, NK), ts),
    ):
        with T.Kernel(T.ceildiv(NQ, BQ), H, B, threads=threads) as (qx, qh, bz):
            kv_head = qh // G
            Q_shared = T.alloc_shared((BQ, D), ts)
            K_shared = T.alloc_shared((BK_TILE, D), ts)
            logits = T.alloc_fragment((BQ, BK_TILE), accum)
            scores = T.alloc_fragment((BQ, BK_TILE), accum)
            probs = T.alloc_fragment((BQ, BK_TILE), accum)
            row_max = T.alloc_fragment((BQ,), accum)
            row_sum = T.alloc_fragment((BQ,), accum)
            tile_sum = T.alloc_fragment((BQ,), accum)
            _bias_dtype_ref = T.alloc_fragment((1,), bias_ts)
            _mask_dtype_ref = T.alloc_fragment((1,), mask_ts)
            _aux_dtype_ref = T.alloc_fragment((1,), aux_ts)
            _shape_ref = T.alloc_fragment((1,), accum)
            _shape_ref[0] = T.Cast(accum, BB + BH + BQS + BKS + MB + MH + MQS + MKS + AH + NS)

            for qi, di in T.Parallel(BQ, D):
                q_idx = qx * BQ + qi
                if q_idx < NQ:
                    Q_shared[qi, di] = Q[bz, qh, q_idx, di]
                else:
                    Q_shared[qi, di] = T.Cast(ts, 0.0)
            T.fill(row_max, neg_big)

            for kt in T.Pipelined(T.ceildiv(NK, BK_TILE), num_stages=num_stages):
                for ki, di in T.Parallel(BK_TILE, D):
                    k_idx = kt * BK_TILE + ki
                    if k_idx < NK:
                        if G == 1:
                            K_shared[ki, di] = K[bz, qh, k_idx, di]
                        else:
                            K_shared[ki, di] = K[bz, kv_head, k_idx, di]
                    else:
                        K_shared[ki, di] = T.Cast(ts, 0.0)
                T.clear(logits)
                T.gemm(Q_shared, K_shared, logits, transpose_B=True)
                for qi, ki in T.Parallel(BQ, BK_TILE):
                    q_idx = qx * BQ + qi
                    k_idx = kt * BK_TILE + ki
                    q_c = T.min(q_idx, NQ - 1)
                    k_c = T.min(k_idx, NK - 1)
                    in_bounds = (q_idx < NQ) & (k_idx < NK)
                    valid = in_bounds
                    if causal:
                        valid = valid & (k_idx <= q_idx)
                    if use_window:
                        diff = k_idx - q_idx
                        valid = valid & (diff >= -window_left) & (diff <= window_right)
                    s = logits[qi, ki] * scale
                    if use_cap:
                        s = cap * (1.0 - 2.0 / (T.exp(2.0 * s * inv_cap) + 1.0))
                    s = T.if_then_else(valid, s, neg_big)
                    if use_bias:
                        bb = 0 if BB == 1 else bz
                        bh = 0
                        if BH == H:
                            bh = qh
                        elif BH == HK:
                            bh = kv_head
                        bq = 0 if BQS == 1 else q_c
                        bk = 0 if BKS == 1 else k_c
                        s = s + T.Cast(accum, Bias[bb, bh, bq, bk])
                    if use_mask and not use_bias:
                        mb = 0 if MB == 1 else bz
                        mh = 0
                        if MH == H:
                            mh = qh
                        elif MH == HK:
                            mh = kv_head
                        mq = 0 if MQS == 1 else q_c
                        mk = 0 if MKS == 1 else k_c
                        keep = T.Cast("int32", AttentionMask[mb, mh, mq, mk]) != 0
                        s = T.if_then_else(keep, s, neg_big)
                    scores[qi, ki] = T.if_then_else(in_bounds, s, pad_neg)
                T.reduce_max(scores, row_max, dim=1, clear=False)

            if use_softmax_aux:
                for sink in T.serial(NS):
                    ah = 0
                    if AH == H:
                        ah = qh
                    elif AH == HK:
                        ah = kv_head
                    aux = T.Cast(accum, SoftmaxAux[ah, sink])
                    for qi in T.Parallel(BQ):
                        q_idx = qx * BQ + qi
                        if q_idx < NQ:
                            row_max[qi] = T.max(row_max[qi], aux)

            T.fill(row_sum, 0)
            for kt in T.Pipelined(T.ceildiv(NK, BK_TILE), num_stages=num_stages):
                for ki, di in T.Parallel(BK_TILE, D):
                    k_idx = kt * BK_TILE + ki
                    if k_idx < NK:
                        if G == 1:
                            K_shared[ki, di] = K[bz, qh, k_idx, di]
                        else:
                            K_shared[ki, di] = K[bz, kv_head, k_idx, di]
                    else:
                        K_shared[ki, di] = T.Cast(ts, 0.0)
                T.clear(logits)
                T.gemm(Q_shared, K_shared, logits, transpose_B=True)
                for qi, ki in T.Parallel(BQ, BK_TILE):
                    q_idx = qx * BQ + qi
                    k_idx = kt * BK_TILE + ki
                    q_c = T.min(q_idx, NQ - 1)
                    k_c = T.min(k_idx, NK - 1)
                    in_bounds = (q_idx < NQ) & (k_idx < NK)
                    valid = in_bounds
                    if causal:
                        valid = valid & (k_idx <= q_idx)
                    if use_window:
                        diff = k_idx - q_idx
                        valid = valid & (diff >= -window_left) & (diff <= window_right)
                    s = logits[qi, ki] * scale
                    if use_cap:
                        s = cap * (1.0 - 2.0 / (T.exp(2.0 * s * inv_cap) + 1.0))
                    s = T.if_then_else(valid, s, neg_big)
                    if use_bias:
                        bb = 0 if BB == 1 else bz
                        bh = 0
                        if BH == H:
                            bh = qh
                        elif BH == HK:
                            bh = kv_head
                        bq = 0 if BQS == 1 else q_c
                        bk = 0 if BKS == 1 else k_c
                        s = s + T.Cast(accum, Bias[bb, bh, bq, bk])
                    if use_mask and not use_bias:
                        mb = 0 if MB == 1 else bz
                        mh = 0
                        if MH == H:
                            mh = qh
                        elif MH == HK:
                            mh = kv_head
                        mq = 0 if MQS == 1 else q_c
                        mk = 0 if MKS == 1 else k_c
                        keep = T.Cast("int32", AttentionMask[mb, mh, mq, mk]) != 0
                        s = T.if_then_else(keep, s, neg_big)
                    scores[qi, ki] = T.if_then_else(in_bounds, s, pad_neg)
                    probs[qi, ki] = T.if_then_else(in_bounds, T.exp(scores[qi, ki] - row_max[qi]), 0.0)
                T.reduce_sum(probs, tile_sum, dim=1, clear=True)
                for qi in T.Parallel(BQ):
                    row_sum[qi] = row_sum[qi] + tile_sum[qi]

            if use_softmax_aux:
                for sink in T.serial(NS):
                    ah = 0
                    if AH == H:
                        ah = qh
                    elif AH == HK:
                        ah = kv_head
                    aux = T.Cast(accum, SoftmaxAux[ah, sink])
                    for qi in T.Parallel(BQ):
                        q_idx = qx * BQ + qi
                        if q_idx < NQ:
                            row_sum[qi] = row_sum[qi] + T.exp(aux - row_max[qi])

            for kt in T.Pipelined(T.ceildiv(NK, BK_TILE), num_stages=num_stages):
                for ki, di in T.Parallel(BK_TILE, D):
                    k_idx = kt * BK_TILE + ki
                    if k_idx < NK:
                        if G == 1:
                            K_shared[ki, di] = K[bz, qh, k_idx, di]
                        else:
                            K_shared[ki, di] = K[bz, kv_head, k_idx, di]
                    else:
                        K_shared[ki, di] = T.Cast(ts, 0.0)
                T.clear(logits)
                T.gemm(Q_shared, K_shared, logits, transpose_B=True)
                for qi, ki in T.Parallel(BQ, BK_TILE):
                    q_idx = qx * BQ + qi
                    k_idx = kt * BK_TILE + ki
                    q_c = T.min(q_idx, NQ - 1)
                    k_c = T.min(k_idx, NK - 1)
                    in_bounds = (q_idx < NQ) & (k_idx < NK)
                    valid = in_bounds
                    if causal:
                        valid = valid & (k_idx <= q_idx)
                    if use_window:
                        diff = k_idx - q_idx
                        valid = valid & (diff >= -window_left) & (diff <= window_right)
                    s = logits[qi, ki] * scale
                    if use_cap:
                        s = cap * (1.0 - 2.0 / (T.exp(2.0 * s * inv_cap) + 1.0))
                    s = T.if_then_else(valid, s, neg_big)
                    if use_bias:
                        bb = 0 if BB == 1 else bz
                        bh = 0
                        if BH == H:
                            bh = qh
                        elif BH == HK:
                            bh = kv_head
                        bq = 0 if BQS == 1 else q_c
                        bk = 0 if BKS == 1 else k_c
                        s = s + T.Cast(accum, Bias[bb, bh, bq, bk])
                    if use_mask and not use_bias:
                        mb = 0 if MB == 1 else bz
                        mh = 0
                        if MH == H:
                            mh = qh
                        elif MH == HK:
                            mh = kv_head
                        mq = 0 if MQS == 1 else q_c
                        mk = 0 if MKS == 1 else k_c
                        keep = T.Cast("int32", AttentionMask[mb, mh, mq, mk]) != 0
                        s = T.if_then_else(keep, s, neg_big)
                    scores[qi, ki] = T.if_then_else(in_bounds, s, pad_neg)
                    p = T.exp(scores[qi, ki] - row_max[qi]) / T.max(row_sum[qi], 1.0e-30)
                    if use_dropout:
                        linear = T.Cast("int64", (q_idx * NK + k_c))
                        seed_mix = T.Cast("int64", DropoutSeed[0]) + T.Cast("int64", DropoutSeed[1]) * 65537
                        rnd_i = (linear * 1103515245 + seed_mix * 12345 + 12345) % 2147483647
                        rnd = T.Cast(accum, rnd_i) * 4.656612875245797e-10
                        drop = T.if_then_else(rnd >= float(dropout_prob), inv_keep_prob, 0.0)
                        p = p * drop
                    if in_bounds:
                        Weights[bz, qh, q_idx, k_idx] = T.Cast(ts, p)

    return attention_weights


def make_attention_weights_bwd_dq_prim_func(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_q: int,
    block_d: int,
    softmax_scale: float,
    logits_soft_cap: float | None,
    dtype,
    threads: int = 128,
):
    """Build native dQ for the dense attention-weights output."""
    ts = _dtype_str(dtype)
    accum = "float32"
    B, H, HK = batch, num_heads, num_kv_heads
    NQ, NK, D = seq_len_q, seq_len_k, head_dim
    BQ, BD = block_q, block_d
    G = H // HK
    scale = float(softmax_scale)
    use_cap = logits_soft_cap is not None
    cap = float(logits_soft_cap) if use_cap else 1.0
    inv_cap = 1.0 / cap

    @T.prim_func
    def attention_weights_bwd_dq(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, HK, NK, D), ts),
        W: T.Tensor((B, H, NQ, NK), ts),
        dW: T.Tensor((B, H, NQ, NK), ts),
        dQ: T.Tensor((B, H, NQ, D), ts),
    ):
        with T.Kernel(T.ceildiv(D, BD), T.ceildiv(NQ, BQ), B * H, threads=threads) as (dx, qx, bh):
            bz = bh // H
            qh = bh - bz * H
            kv_head = qh // G
            row_dot = T.alloc_fragment((BQ,), accum)
            acc = T.alloc_fragment((BQ, BD), accum)
            _dtype_ref = T.alloc_fragment((1,), ts)
            _hk_ref = T.alloc_fragment((HK,), accum)

            T.clear(row_dot)
            T.clear(acc)
            for k_idx in T.serial(NK):
                for qi in T.Parallel(BQ):
                    q_idx = qx * BQ + qi
                    if q_idx < NQ:
                        w = T.Cast(accum, W[bz, qh, q_idx, k_idx])
                        dw = T.Cast(accum, dW[bz, qh, q_idx, k_idx])
                        row_dot[qi] = row_dot[qi] + w * dw

            for k_idx in T.serial(NK):
                for qi, di in T.Parallel(BQ, BD):
                    q_idx = qx * BQ + qi
                    d_idx = dx * BD + di
                    if (q_idx < NQ) and (d_idx < D):
                        w = T.Cast(accum, W[bz, qh, q_idx, k_idx])
                        dw = T.Cast(accum, dW[bz, qh, q_idx, k_idx])
                        coeff = w * (dw - row_dot[qi])
                        if use_cap:
                            qk = 0.0
                            for rd in T.serial(D):
                                qk = qk + T.Cast(accum, Q[bz, qh, q_idx, rd]) * T.Cast(accum, K[bz, kv_head, k_idx, rd])
                            capped = cap * (1.0 - 2.0 / (T.exp(2.0 * qk * scale * inv_cap) + 1.0))
                            coeff = coeff * (1.0 - (capped * inv_cap) * (capped * inv_cap))
                        acc[qi, di] = acc[qi, di] + coeff * scale * T.Cast(accum, K[bz, kv_head, k_idx, d_idx])

            for qi, di in T.Parallel(BQ, BD):
                q_idx = qx * BQ + qi
                d_idx = dx * BD + di
                if (q_idx < NQ) and (d_idx < D):
                    dQ[bz, qh, q_idx, d_idx] = T.Cast(ts, acc[qi, di])

    return attention_weights_bwd_dq


def make_attention_weights_bwd_dk_prim_func(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_k: int,
    block_d: int,
    softmax_scale: float,
    logits_soft_cap: float | None,
    dtype,
    threads: int = 128,
):
    """Build native dK for the dense attention-weights output."""
    ts = _dtype_str(dtype)
    accum = "float32"
    B, H, HK = batch, num_heads, num_kv_heads
    NQ, NK, D = seq_len_q, seq_len_k, head_dim
    BK, BD = block_k, block_d
    G = H // HK
    scale = float(softmax_scale)
    use_cap = logits_soft_cap is not None
    cap = float(logits_soft_cap) if use_cap else 1.0
    inv_cap = 1.0 / cap

    @T.prim_func
    def attention_weights_bwd_dk(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, HK, NK, D), ts),
        W: T.Tensor((B, H, NQ, NK), ts),
        dW: T.Tensor((B, H, NQ, NK), ts),
        dK: T.Tensor((B, HK, NK, D), ts),
    ):
        with T.Kernel(T.ceildiv(D, BD), T.ceildiv(NK, BK), B * HK, threads=threads) as (dx, kx, bh):
            bz = bh // HK
            kv_head = bh - bz * HK
            row_dot = T.alloc_fragment((BK,), accum)
            acc = T.alloc_fragment((BK, BD), accum)
            _dtype_ref = T.alloc_fragment((1,), ts)
            _h_ref = T.alloc_fragment((H,), accum)

            T.clear(acc)
            for g in T.serial(G):
                qh = kv_head * G + g
                for q_idx in T.serial(NQ):
                    T.clear(row_dot)
                    for kk in T.serial(NK):
                        w_all = T.Cast(accum, W[bz, qh, q_idx, kk])
                        dw_all = T.Cast(accum, dW[bz, qh, q_idx, kk])
                        for ki in T.Parallel(BK):
                            row_dot[ki] = row_dot[ki] + w_all * dw_all
                    for ki, di in T.Parallel(BK, BD):
                        k_idx = kx * BK + ki
                        d_idx = dx * BD + di
                        if (k_idx < NK) and (d_idx < D):
                            w = T.Cast(accum, W[bz, qh, q_idx, k_idx])
                            dw = T.Cast(accum, dW[bz, qh, q_idx, k_idx])
                            coeff = w * (dw - row_dot[ki])
                            if use_cap:
                                qk = 0.0
                                for rd in T.serial(D):
                                    qk = qk + T.Cast(accum, Q[bz, qh, q_idx, rd]) * T.Cast(
                                        accum, K[bz, kv_head, k_idx, rd]
                                    )
                                capped = cap * (1.0 - 2.0 / (T.exp(2.0 * qk * scale * inv_cap) + 1.0))
                                coeff = coeff * (1.0 - (capped * inv_cap) * (capped * inv_cap))
                            acc[ki, di] = acc[ki, di] + coeff * scale * T.Cast(accum, Q[bz, qh, q_idx, d_idx])

            for ki, di in T.Parallel(BK, BD):
                k_idx = kx * BK + ki
                d_idx = dx * BD + di
                if (k_idx < NK) and (d_idx < D):
                    dK[bz, kv_head, k_idx, d_idx] = T.Cast(ts, acc[ki, di])

    return attention_weights_bwd_dk
