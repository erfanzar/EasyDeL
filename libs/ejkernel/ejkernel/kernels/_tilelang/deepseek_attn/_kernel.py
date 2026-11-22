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

"""Native tile-lang kernels for DeepSeek Sparse Attention (DSA).

Native ``@T.prim_func`` kernels carry every DSA matmul; nothing is done with
``jnp.einsum`` / ``jax.vmap``:

* :func:`make_dsa_indexer_prim_func` — the Lightning Indexer. Per query/key
  tile it runs one ``GEMM`` per indexer head, applies the ReLU and the
  per-head learned weight, accumulates the head scores, and writes the
  ``(B, S, S)`` relevance matrix with the causal frontier baked in.
* :func:`make_matmul_prim_func` — a tiled ``(M, K) @ (K, N)`` GEMM. The KV
  up-projection (``KV @ W_kc`` / ``KV @ W_vc``) and its gradient
  contractions (``dK_r @ W_kc^T`` and ``KV^T @ dK_r``) all run through it,
  so the DSA forward *and* backward are fully differentiable through native
  kernels — the attention proper is handed to the verified FlashAttention
  kernels with the indexer mask folded into the logits as a bias.
* :func:`make_topk_bias_prim_func` — a native top-k row-rank kernel that
  constructs the additive attention bias on GPU.
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
        raise TypeError(f"Unsupported dtype for tile-lang deepseek_attn: {dtype}")
    return mapping[canonical]


def make_dsa_indexer_prim_func(
    *,
    batch: int,
    seq_len: int,
    index_heads: int,
    index_head_dim: int,
    block_t: int,
    block_s: int,
    index_scale: float,
    causal: bool,
    dtype,
    threads: int = 128,
):
    """Build the Lightning-Indexer ``@T.prim_func``.

    Grid: ``(ceildiv(S, BT), B)``. Each CTA owns ``BT`` query rows and walks
    the key axis in ``BS`` tiles, running one ``GEMM`` per indexer head.

    Returns:
        ``@T.prim_func`` with buffers ``(QI, KI, IW, Score)`` where ``Score``
        is the ``(B, S, S)`` indexer relevance matrix (future positions set
        to ``-1e30`` when ``causal``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, HI, DI = batch, seq_len, index_heads, index_head_dim
    BT, BS = block_t, block_s
    iscale = float(index_scale)

    @T.prim_func
    def dsa_indexer(
        QI: T.Tensor((B, S, HI, DI), ts),
        KI: T.Tensor((B, S, DI), ts),
        IW: T.Tensor((B, S, HI), ts),
        Score: T.Tensor((B, S, S), accum),
    ):
        with T.Kernel(T.ceildiv(S, BT), B, threads=threads) as (tx, bz):
            qi_sh = T.alloc_shared((BT, DI), ts)
            ki_sh = T.alloc_shared((BS, DI), ts)
            raw = T.alloc_fragment((BT, BS), accum)
            acc = T.alloc_fragment((BT, BS), accum)
            iw_loc = T.alloc_fragment((BT,), accum)

            for sb in T.serial(T.ceildiv(S, BS)):
                T.clear(acc)
                T.copy(KI[bz, sb * BS : (sb + 1) * BS, :], ki_sh)
                for h in T.serial(HI):
                    T.copy(QI[bz, tx * BT : (tx + 1) * BT, h, :], qi_sh)
                    T.clear(raw)
                    T.gemm(qi_sh, ki_sh, raw, transpose_B=True)
                    for t in T.Parallel(BT):
                        iw_loc[t] = T.Cast(accum, IW[bz, T.min(tx * BT + t, S - 1), h])
                    for t, s in T.Parallel(BT, BS):
                        acc[t, s] = acc[t, s] + iw_loc[t] * T.max(raw[t, s] * iscale, 0.0)

                for t, s in T.Parallel(BT, BS):
                    t_idx = tx * BT + t
                    s_idx = sb * BS + s
                    if (t_idx < S) and (s_idx < S):
                        keep = True
                        if causal:
                            keep = s_idx <= t_idx
                        Score[bz, t_idx, s_idx] = T.if_then_else(keep, acc[t, s], -1.0e30)

    return dsa_indexer


def make_matmul_prim_func(
    *,
    m: int,
    k: int,
    n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    dtype,
    out_dtype,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build a tiled ``(M, K) @ (K, N) -> (M, N)`` GEMM ``@T.prim_func``.

    Used for the KV up-projection and both of its gradient contractions, so
    the DSA forward and backward stay native (the JAX glue only reshapes /
    transposes the operands around this kernel). The output is float32 — the
    glue casts as needed.

    Returns:
        ``@T.prim_func`` with buffers ``(A, B, C)``.
    """
    ts = _dtype_str(dtype)
    out_ts = _dtype_str(out_dtype)
    accum = "float32"
    M, K, N = m, k, n
    BM, BN, BK = block_m, block_n, block_k
    k_ragged = (K % BK) != 0

    if k_ragged:

        @T.prim_func
        def matmul_ragged(
            A: T.Tensor((M, K), ts),
            B: T.Tensor((K, N), ts),
            C: T.Tensor((M, N), out_ts),
        ):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (nx, mx):
                C_loc = T.alloc_fragment((BM, BN), accum)
                _dtype_ref = T.alloc_fragment((1,), ts)
                _out_dtype_ref = T.alloc_fragment((1,), out_ts)
                T.clear(C_loc)
                for kk in T.serial(K):
                    for i, j in T.Parallel(BM, BN):
                        gm = mx * BM + i
                        gn = nx * BN + j
                        if (gm < M) and (gn < N):
                            C_loc[i, j] = C_loc[i, j] + T.Cast(accum, A[gm, kk]) * T.Cast(accum, B[kk, gn])
                for i, j in T.Parallel(BM, BN):
                    gm = mx * BM + i
                    gn = nx * BN + j
                    if (gm < M) and (gn < N):
                        C[gm, gn] = T.Cast(out_ts, C_loc[i, j])

        return matmul_ragged

    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), ts),
        B: T.Tensor((K, N), ts),
        C: T.Tensor((M, N), out_ts),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (nx, mx):
            A_sh = T.alloc_shared((BM, BK), ts)
            B_sh = T.alloc_shared((BK, BN), ts)
            C_loc = T.alloc_fragment((BM, BN), accum)
            T.clear(C_loc)
            for kk in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
                T.copy(A[mx * BM : (mx + 1) * BM, kk * BK : (kk + 1) * BK], A_sh)
                T.copy(B[kk * BK : (kk + 1) * BK, nx * BN : (nx + 1) * BN], B_sh)
                T.gemm(A_sh, B_sh, C_loc)
            for i, j in T.Parallel(BM, BN):
                gm = mx * BM + i
                gn = nx * BN + j
                if (gm < M) and (gn < N):
                    C[gm, gn] = T.Cast(out_ts, C_loc[i, j])

    return matmul


def make_add_prim_func(
    *,
    m: int,
    n: int,
    dtype,
    block_m: int,
    block_n: int,
    threads: int = 128,
):
    """Build a native elementwise add kernel for DeepSeek glue reductions."""
    ts = _dtype_str(dtype)
    M, N = m, n
    BM, BN = block_m, block_n

    @T.prim_func
    def add(
        A: T.Tensor((M, N), ts),
        B: T.Tensor((M, N), ts),
        C: T.Tensor((M, N), ts),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (nx, mx):
            _dtype_ref = T.alloc_fragment((1,), ts)
            for i, j in T.Parallel(BM, BN):
                gm = mx * BM + i
                gn = nx * BN + j
                if (gm < M) and (gn < N):
                    C[gm, gn] = A[gm, gn] + B[gm, gn]

    return add


def make_cast_prim_func(
    *,
    m: int,
    n: int,
    in_dtype,
    out_dtype,
    block_m: int,
    block_n: int,
    threads: int = 128,
):
    """Build a native 2D dtype-cast kernel for DeepSeek glue tensors."""
    in_ts = _dtype_str(in_dtype)
    out_ts = _dtype_str(out_dtype)
    M, N = m, n
    BM, BN = block_m, block_n

    @T.prim_func
    def cast(
        A: T.Tensor((M, N), in_ts),
        B: T.Tensor((M, N), out_ts),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (nx, mx):
            _in_ref = T.alloc_fragment((1,), in_ts)
            _out_ref = T.alloc_fragment((1,), out_ts)
            for i, j in T.Parallel(BM, BN):
                gm = mx * BM + i
                gn = nx * BN + j
                if (gm < M) and (gn < N):
                    B[gm, gn] = T.Cast(out_ts, A[gm, gn])

    return cast


def make_pad_lastdim_prim_func(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    in_dim: int,
    out_dim: int,
    dtype,
    block_d: int = 32,
    threads: int = 128,
):
    """Build a native 4D last-dimension zero-pad kernel."""
    ts = _dtype_str(dtype)
    B, S, H, DI, DO = batch, seq_len, heads, in_dim, out_dim
    BD = block_d

    @T.prim_func
    def pad_lastdim(
        A: T.Tensor((B, S, H, DI), ts),
        O: T.Tensor((B, S, H, DO), ts),
    ):
        with T.Kernel(T.ceildiv(DO, BD), H, B * S, threads=threads) as (dx, h, bs):
            _dtype_ref = T.alloc_fragment((1,), ts)
            _shape_ref = T.alloc_fragment((1,), "int32")
            _shape_ref[0] = DI
            b = bs // S
            s = bs - b * S
            for d in T.Parallel(BD):
                od = dx * BD + d
                if od < DO:
                    if od < DI:
                        O[b, s, h, od] = A[b, s, h, od]
                    else:
                        O[b, s, h, od] = T.Cast(ts, 0.0)

    return pad_lastdim


def make_crop_lastdim_prim_func(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    in_dim: int,
    out_dim: int,
    dtype,
    block_d: int = 32,
    threads: int = 128,
):
    """Build a native 4D last-dimension crop kernel."""
    ts = _dtype_str(dtype)
    B, S, H, DI, DO = batch, seq_len, heads, in_dim, out_dim
    BD = block_d

    @T.prim_func
    def crop_lastdim(
        A: T.Tensor((B, S, H, DI), ts),
        O: T.Tensor((B, S, H, DO), ts),
    ):
        with T.Kernel(T.ceildiv(DO, BD), H, B * S, threads=threads) as (dx, h, bs):
            _dtype_ref = T.alloc_fragment((1,), ts)
            _shape_ref = T.alloc_fragment((1,), "int32")
            _shape_ref[0] = DI
            b = bs // S
            s = bs - b * S
            for d in T.Parallel(BD):
                od = dx * BD + d
                if od < DO:
                    O[b, s, h, od] = A[b, s, h, od]

    return crop_lastdim


def make_pack_shared_tail_prim_func(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    main_dim: int,
    tail_dim: int,
    out_dim: int,
    dtype,
    block_d: int = 32,
    threads: int = 128,
):
    """Build a native pack kernel for per-head main plus head-shared tail."""
    ts = _dtype_str(dtype)
    B, S, H, DM, DT, DO = batch, seq_len, heads, main_dim, tail_dim, out_dim
    BD = block_d

    @T.prim_func
    def pack_shared_tail(
        Main: T.Tensor((B, S, H, DM), ts),
        Tail: T.Tensor((B, S, DT), ts),
        O: T.Tensor((B, S, H, DO), ts),
    ):
        with T.Kernel(T.ceildiv(DO, BD), H, B * S, threads=threads) as (dx, h, bs):
            _dtype_ref = T.alloc_fragment((1,), ts)
            b = bs // S
            s = bs - b * S
            for d in T.Parallel(BD):
                od = dx * BD + d
                if od < DO:
                    if od < DM:
                        O[b, s, h, od] = Main[b, s, h, od]
                    elif od < DM + DT:
                        O[b, s, h, od] = Tail[b, s, od - DM]
                    else:
                        O[b, s, h, od] = T.Cast(ts, 0.0)

    return pack_shared_tail


def make_reduce_shared_tail_prim_func(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    main_dim: int,
    tail_dim: int,
    in_dim: int,
    dtype,
    block_d: int = 32,
    threads: int = 128,
):
    """Build a native head-reduction kernel for shared RoPE tails."""
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, DM, DT, DI = batch, seq_len, heads, main_dim, tail_dim, in_dim
    BD = block_d

    @T.prim_func
    def reduce_shared_tail(
        G: T.Tensor((B, S, H, DI), ts),
        TailGrad: T.Tensor((B, S, DT), ts),
    ):
        with T.Kernel(T.ceildiv(DT, BD), S, B, threads=threads) as (dx, s, b):
            acc = T.alloc_fragment((BD,), accum)
            _dtype_ref = T.alloc_fragment((1,), ts)
            _shape_ref = T.alloc_fragment((1,), "int32")
            _shape_ref[0] = DI
            for d in T.Parallel(BD):
                td = dx * BD + d
                acc[d] = 0.0
                if td < DT:
                    for h in T.serial(H):
                        acc[d] = acc[d] + T.Cast(accum, G[b, s, h, DM + td])
                    TailGrad[b, s, td] = T.Cast(ts, acc[d])

    return reduce_shared_tail


def make_topk_bias_prim_func(
    *,
    batch: int,
    seq_len: int,
    index_topk: int,
    block_s: int,
    causal: bool,
    threads: int = 128,
):
    """Build a native top-k additive-bias kernel for DSA.

    For each ``(batch, query, key)`` score, the kernel counts how many scores
    in the query row are larger. The key is kept when its rank is within
    ``index_topk``; the diagonal is always kept, matching the XLA reference.

    Returns:
        ``@T.prim_func`` with buffers ``(Score, Bias)`` where ``Bias`` has
        shape ``(B, 1, S, S)`` and values ``0`` or ``-1e30``.
    """
    accum = "float32"
    B, S, BS = batch, seq_len, block_s
    KTOP = min(int(index_topk), S)

    @T.prim_func
    def topk_bias(
        Score: T.Tensor((B, S, S), accum),
        Bias: T.Tensor((B, 1, S, S), accum),
    ):
        with T.Kernel(T.ceildiv(S, BS), S, B, threads=threads) as (sx, tq, bz):
            rank = T.alloc_fragment((BS,), "int32")
            cur = T.alloc_fragment((BS,), accum)
            keep = T.alloc_fragment((BS,), "int32")
            _ref = T.alloc_fragment((KTOP,), accum)

            for s in T.Parallel(BS):
                sk = sx * BS + s
                cur[s] = T.if_then_else(sk < S, Score[bz, tq, sk], -1.0e30)
                rank[s] = 0

            for r in T.serial(S):
                rv = Score[bz, tq, r]
                for s in T.Parallel(BS):
                    sk = sx * BS + s
                    before = (rv > cur[s]) | ((rv == cur[s]) & (r < sk))
                    rank[s] = rank[s] + T.if_then_else((sk < S) & before, 1, 0)

            for s in T.Parallel(BS):
                sk = sx * BS + s
                keep[s] = T.if_then_else((sk < S) & (rank[s] < KTOP), 1, 0)
                if causal:
                    keep[s] = T.if_then_else(sk <= tq, keep[s], 0)
                keep[s] = T.if_then_else(sk == tq, 1, keep[s])
                if sk < S:
                    Bias[bz, 0, tq, sk] = T.if_then_else(keep[s] != 0, 0.0, -1.0e30)

    return topk_bias
