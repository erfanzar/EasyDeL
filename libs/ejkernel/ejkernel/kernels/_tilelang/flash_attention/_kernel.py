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

"""tile-lang prim_func factories for FlashAttention-2 forward and backward.

The kernels accept Q / K / V laid out as ``(B, H, N, D)`` with row-major
strides — this matches the canonical FA2 layout from Dao et al. and avoids
strided gather/scatter inside the tile. The JAX glue in :mod:`._impl`
transposes the public ``(B, N, H, D)`` layout into ``(B, H, N, D)`` before
launch.

Forward:
    ``flash_attention_fwd(Q, K, V, O, L)``

    Produces:
        * ``O`` -- the attention output ``(B, H, N, D)``
        * ``L`` -- log-sum-exp ``(B, H, N)`` in fp32, kept for the backward pass

Backward:
    Two-kernel design (Tri Dao FA2 style):
        1. ``flash_attention_bwd_preprocess(O, dO, D)``
           computes ``D[b,h,n] = sum_d(O * dO)`` once, in fp32
        2. ``flash_attention_bwd(Q, K, V, dO, L, D_row, dQ, dK, dV)``
           parallelizes over K-blocks and re-derives ``P`` per Q-block
           from ``L``. ``dQ`` is accumulated via atomic-adds to avoid a
           separate dQ kernel.

All kernels are pure tile-lang (no Triton, no Pallas, no XLA) and lower
through TVM-FFI to a CUDA cubin that the JAX bridge calls via
``jax.ffi.ffi_call``. The factories return ``@T.prim_func`` ASTs so that
all shape / dtype / tile params are baked into the cache key.
"""

from __future__ import annotations

import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Return the tile-lang dtype string for a NumPy/JAX dtype object."""
    import jax.numpy as jnp

    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang flash attention: {dtype}")
    return mapping[canonical]


def _scalar_dtype_str(dtype) -> str:
    """Return a tile-lang dtype string for scalar feature buffers."""
    import jax.numpy as jnp

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
        raise TypeError(f"Unsupported dtype for tile-lang flash attention feature buffer: {dtype}")
    return mapping[canonical]


def make_fwd_prim_func(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    causal: bool,
    dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build the FA2 forward ``@T.prim_func``.

    Args:
        batch: Static batch size baked into the kernel.
        num_heads: Number of (Q == KV) heads — MHA only in v0.
        seq_len_q: Query sequence length.
        seq_len_k: Key/Value sequence length.
        head_dim: Per-head feature dimension.
        block_m: Tile size along the query sequence axis.
        block_n: Tile size along the key sequence axis.
        softmax_scale: Multiplier applied to ``QK^T`` before the softmax.
        causal: When ``True``, applies an upper-triangular mask aligned to
            ``seq_len_q - seq_len_k`` so that decode-style shorter Q with
            longer K still produces a valid causal frontier.
        dtype: Activation dtype (``float16`` / ``bfloat16`` / ``float32``).
        num_stages: Software-pipeline depth for the K/V load loop.
        threads: CTA size.

    Returns:
        A ``@T.prim_func`` whose buffers are ``(Q, K, V, O, L)``.
    """

    ts = _dtype_str(dtype)
    accum_dtype = "float32"
    B, H, NQ, NK, D = batch, num_heads, seq_len_q, seq_len_k, head_dim
    BM, BN = block_m, block_n
    log2e = 1.4426950408889634
    scale_log2e = float(softmax_scale) * log2e

    causal_offset = NK - NQ

    @T.prim_func
    def fa_fwd(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, H, NK, D), ts),
        V: T.Tensor((B, H, NK, D), ts),
        O: T.Tensor((B, H, NQ, D), ts),
        L: T.Tensor((B, H, NQ), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(NQ, BM), H, B, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((BM, D), ts)
            K_shared = T.alloc_shared((BN, D), ts)
            V_shared = T.alloc_shared((BN, D), ts)
            S_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_shared = T.alloc_shared((BM, BN), ts)
            O_local = T.alloc_fragment((BM, D), accum_dtype)
            m_local = T.alloc_fragment((BM,), accum_dtype)
            m_prev = T.alloc_fragment((BM,), accum_dtype)
            l_local = T.alloc_fragment((BM,), accum_dtype)
            row_sum = T.alloc_fragment((BM,), accum_dtype)
            alpha = T.alloc_fragment((BM,), accum_dtype)

            T.copy(Q[bz, by, bx * BM : (bx + 1) * BM, :], Q_shared)
            T.fill(O_local, 0)
            T.fill(l_local, 0)
            T.fill(m_local, -float("inf"))

            if causal:
                kv_end = T.min(NK, (bx + 1) * BM + causal_offset)
                loop_count = T.ceildiv(T.max(0, kv_end), BN)
            else:
                loop_count = T.ceildiv(NK, BN)

            for k_iter in T.Pipelined(loop_count, num_stages=num_stages):
                T.copy(K[bz, by, k_iter * BN : (k_iter + 1) * BN, :], K_shared)
                T.copy(V[bz, by, k_iter * BN : (k_iter + 1) * BN, :], V_shared)

                T.clear(S_local)
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                if causal:
                    for i, j in T.Parallel(BM, BN):
                        q_idx = bx * BM + i
                        k_idx = k_iter * BN + j
                        S_local[i, j] = T.if_then_else(
                            (k_idx <= q_idx + causal_offset) & (k_idx < NK),
                            S_local[i, j] * scale_log2e,
                            -float("inf"),
                        )
                else:
                    for i, j in T.Parallel(BM, BN):
                        k_idx = k_iter * BN + j
                        S_local[i, j] = T.if_then_else(
                            k_idx < NK,
                            S_local[i, j] * scale_log2e,
                            -float("inf"),
                        )

                T.copy(m_local, m_prev)
                T.reduce_max(S_local, m_local, dim=1, clear=False)

                for i in T.Parallel(BM):
                    alpha[i] = T.exp2(m_prev[i] - m_local[i])

                for i, j in T.Parallel(BM, BN):
                    P_local[i, j] = T.exp2(S_local[i, j] - m_local[i])

                T.fill(row_sum, 0)
                T.reduce_sum(P_local, row_sum, dim=1, clear=True)
                for i in T.Parallel(BM):
                    l_local[i] = l_local[i] * alpha[i] + row_sum[i]

                for i, j in T.Parallel(BM, D):
                    O_local[i, j] = O_local[i, j] * alpha[i]

                for i, j in T.Parallel(BM, BN):
                    P_shared[i, j] = T.Cast(ts, P_local[i, j])
                T.gemm(P_shared, V_shared, O_local)

            for i, j in T.Parallel(BM, D):
                q_idx = bx * BM + i
                O_local[i, j] = T.if_then_else(
                    q_idx < NQ,
                    O_local[i, j] / T.max(l_local[i], 1e-30),
                    0.0,
                )

            for i in T.Parallel(BM):
                q_idx = bx * BM + i
                if q_idx < NQ:
                    L[bz, by, q_idx] = m_local[i] + T.log2(T.max(l_local[i], 1e-30))

            for i, j in T.Parallel(BM, D):
                q_idx = bx * BM + i
                if q_idx < NQ:
                    O[bz, by, q_idx, j] = T.Cast(ts, O_local[i, j])

    return fa_fwd


def make_bwd_preprocess_prim_func(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    head_dim: int,
    block_m: int,
    dtype,
    threads: int = 128,
):
    """Build the FA backward pre-processing ``@T.prim_func``.

    Computes ``D[b, h, q] = sum_d(O[b,h,q,d] * dO[b,h,q,d])`` in fp32. The
    main backward kernel then re-uses ``D`` to evaluate the softmax-gradient
    correction ``dS = P * (dP - D)`` without recomputing the running stats.

    Args:
        batch: Static batch size.
        num_heads: Static head count.
        seq_len_q: Query sequence length.
        head_dim: Per-head feature dimension.
        block_m: Tile size along the query axis.
        dtype: Activation dtype (must match the forward pass).
        threads: CTA size.

    Returns:
        ``@T.prim_func`` with buffers ``(O, dO, D)``.
    """
    ts = _dtype_str(dtype)
    accum_dtype = "float32"
    B, H, NQ, D = batch, num_heads, seq_len_q, head_dim
    BM = block_m

    @T.prim_func
    def fa_bwd_pre(
        O: T.Tensor((B, H, NQ, D), ts),
        dO: T.Tensor((B, H, NQ, D), ts),
        Delta: T.Tensor((B, H, NQ), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(NQ, BM), H, B, threads=threads) as (bx, by, bz):
            o_tile = T.alloc_fragment((BM, D), accum_dtype)
            do_tile = T.alloc_fragment((BM, D), accum_dtype)
            prod = T.alloc_fragment((BM, D), accum_dtype)
            row_sum = T.alloc_fragment((BM,), accum_dtype)
            _ts_ref = T.alloc_fragment((1,), ts)

            for i, j in T.Parallel(BM, D):
                q_idx = bx * BM + i
                o_tile[i, j] = T.if_then_else(
                    q_idx < NQ,
                    T.Cast(accum_dtype, O[bz, by, q_idx, j]),
                    0.0,
                )
                do_tile[i, j] = T.if_then_else(
                    q_idx < NQ,
                    T.Cast(accum_dtype, dO[bz, by, q_idx, j]),
                    0.0,
                )
            for i, j in T.Parallel(BM, D):
                prod[i, j] = o_tile[i, j] * do_tile[i, j]
            T.fill(row_sum, 0)
            T.reduce_sum(prod, row_sum, dim=1, clear=True)

            for i in T.Parallel(BM):
                q_idx = bx * BM + i
                if q_idx < NQ:
                    Delta[bz, by, q_idx] = row_sum[i]

    return fa_bwd_pre


def make_bwd_dkdv_prim_func(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    causal: bool,
    dtype,
    num_stages: int = 2,
    threads: int = 128,
):
    """Build the FA2 dK/dV backward ``@T.prim_func``.

    Parallelizes over ``(k_block, head, batch)``. Inside each CTA we loop
    over Q-blocks and accumulate ``dK`` and ``dV`` into block-local fragments.
    Because each CTA owns its slice of ``dK`` and ``dV`` exclusively, no
    atomics are required.

    Returns:
        ``@T.prim_func`` with buffers ``(Q, K, V, dO, L, Delta, dK, dV)``.
    """
    ts = _dtype_str(dtype)
    accum_dtype = "float32"
    B, H, NQ, NK, D = batch, num_heads, seq_len_q, seq_len_k, head_dim
    BM, BN = block_m, block_n
    log2e = 1.4426950408889634
    scale_log2e = float(softmax_scale) * log2e
    causal_offset = NK - NQ

    @T.prim_func
    def fa_bwd_dkdv(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, H, NK, D), ts),
        V: T.Tensor((B, H, NK, D), ts),
        dO: T.Tensor((B, H, NQ, D), ts),
        L: T.Tensor((B, H, NQ), accum_dtype),
        Delta: T.Tensor((B, H, NQ), accum_dtype),
        dK: T.Tensor((B, H, NK, D), ts),
        dV: T.Tensor((B, H, NK, D), ts),
    ):
        with T.Kernel(T.ceildiv(NK, BN), H, B, threads=threads) as (kx, by, bz):
            K_shared = T.alloc_shared((BN, D), ts)
            V_shared = T.alloc_shared((BN, D), ts)
            Q_shared = T.alloc_shared((BM, D), ts)
            dO_shared = T.alloc_shared((BM, D), ts)

            S_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_local = T.alloc_fragment((BM, BN), accum_dtype)
            dP_local = T.alloc_fragment((BM, BN), accum_dtype)
            dS_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_shared = T.alloc_shared((BM, BN), ts)
            dS_shared = T.alloc_shared((BM, BN), ts)
            dK_local = T.alloc_fragment((BN, D), accum_dtype)
            dV_local = T.alloc_fragment((BN, D), accum_dtype)
            L_tile = T.alloc_fragment((BM,), accum_dtype)
            D_tile = T.alloc_fragment((BM,), accum_dtype)

            T.copy(K[bz, by, kx * BN : (kx + 1) * BN, :], K_shared)
            T.copy(V[bz, by, kx * BN : (kx + 1) * BN, :], V_shared)
            T.fill(dK_local, 0)
            T.fill(dV_local, 0)

            if causal:
                q_start = T.max(0, kx * BN - causal_offset) // BM
                q_end_excl = T.ceildiv(NQ, BM)
            else:
                q_start = 0
                q_end_excl = T.ceildiv(NQ, BM)

            for q_iter in T.Pipelined(q_end_excl - q_start, num_stages=num_stages):
                m_iter = q_start + q_iter
                T.copy(Q[bz, by, m_iter * BM : (m_iter + 1) * BM, :], Q_shared)
                T.copy(dO[bz, by, m_iter * BM : (m_iter + 1) * BM, :], dO_shared)
                for i in T.Parallel(BM):
                    q_idx = m_iter * BM + i
                    L_tile[i] = T.if_then_else(q_idx < NQ, L[bz, by, q_idx], 0.0)
                    D_tile[i] = T.if_then_else(q_idx < NQ, Delta[bz, by, q_idx], 0.0)

                T.clear(S_local)
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    q_idx = m_iter * BM + i
                    k_idx = kx * BN + j
                    valid = (q_idx < NQ) & (k_idx < NK)
                    if causal:
                        valid = valid & (k_idx <= q_idx + causal_offset)
                    S_local[i, j] = T.if_then_else(
                        valid,
                        S_local[i, j] * scale_log2e - L_tile[i],
                        -float("inf"),
                    )
                for i, j in T.Parallel(BM, BN):
                    P_local[i, j] = T.exp2(S_local[i, j])

                for i, j in T.Parallel(BM, BN):
                    P_shared[i, j] = T.Cast(ts, P_local[i, j])
                T.gemm(P_shared, dO_shared, dV_local, transpose_A=True)

                T.clear(dP_local)
                T.gemm(dO_shared, V_shared, dP_local, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    dS_local[i, j] = P_local[i, j] * (dP_local[i, j] - D_tile[i]) * float(softmax_scale)
                for i, j in T.Parallel(BM, BN):
                    dS_shared[i, j] = T.Cast(ts, dS_local[i, j])

                T.gemm(dS_shared, Q_shared, dK_local, transpose_A=True)

            for i, j in T.Parallel(BN, D):
                k_idx = kx * BN + i
                if k_idx < NK:
                    dK[bz, by, k_idx, j] = T.Cast(ts, dK_local[i, j])
                    dV[bz, by, k_idx, j] = T.Cast(ts, dV_local[i, j])

    return fa_bwd_dkdv


def make_bwd_dq_prim_func(
    *,
    batch: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    causal: bool,
    dtype,
    num_stages: int = 2,
    threads: int = 128,
):
    """Build the FA2 dQ backward ``@T.prim_func``.

    Parallelizes over ``(q_block, head, batch)`` and loops over K-blocks.
    Each CTA writes the dQ slice for its Q-block exclusively, so no atomics.

    Returns:
        ``@T.prim_func`` with buffers ``(Q, K, V, dO, L, Delta, dQ)``. ``dQ``
        is in the input dtype to match the public gradient layout.
    """
    ts = _dtype_str(dtype)
    accum_dtype = "float32"
    B, H, NQ, NK, D = batch, num_heads, seq_len_q, seq_len_k, head_dim
    BM, BN = block_m, block_n
    log2e = 1.4426950408889634
    scale_log2e = float(softmax_scale) * log2e
    causal_offset = NK - NQ

    @T.prim_func
    def fa_bwd_dq(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, H, NK, D), ts),
        V: T.Tensor((B, H, NK, D), ts),
        dO: T.Tensor((B, H, NQ, D), ts),
        L: T.Tensor((B, H, NQ), accum_dtype),
        Delta: T.Tensor((B, H, NQ), accum_dtype),
        dQ: T.Tensor((B, H, NQ, D), ts),
    ):
        with T.Kernel(T.ceildiv(NQ, BM), H, B, threads=threads) as (mx, by, bz):
            Q_shared = T.alloc_shared((BM, D), ts)
            dO_shared = T.alloc_shared((BM, D), ts)
            K_shared = T.alloc_shared((BN, D), ts)
            V_shared = T.alloc_shared((BN, D), ts)

            S_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_local = T.alloc_fragment((BM, BN), accum_dtype)
            dP_local = T.alloc_fragment((BM, BN), accum_dtype)
            dS_local = T.alloc_fragment((BM, BN), accum_dtype)
            dS_shared = T.alloc_shared((BM, BN), ts)
            dQ_local = T.alloc_fragment((BM, D), accum_dtype)
            L_tile = T.alloc_fragment((BM,), accum_dtype)
            D_tile = T.alloc_fragment((BM,), accum_dtype)

            T.copy(Q[bz, by, mx * BM : (mx + 1) * BM, :], Q_shared)
            T.copy(dO[bz, by, mx * BM : (mx + 1) * BM, :], dO_shared)
            T.fill(dQ_local, 0)
            for i in T.Parallel(BM):
                q_idx = mx * BM + i
                L_tile[i] = T.if_then_else(q_idx < NQ, L[bz, by, q_idx], 0.0)
                D_tile[i] = T.if_then_else(q_idx < NQ, Delta[bz, by, q_idx], 0.0)

            if causal:
                k_end_excl = T.ceildiv(T.min(NK, (mx + 1) * BM + causal_offset), BN)
                k_end_excl = T.max(0, k_end_excl)
            else:
                k_end_excl = T.ceildiv(NK, BN)

            for k_iter in T.Pipelined(k_end_excl, num_stages=num_stages):
                T.copy(K[bz, by, k_iter * BN : (k_iter + 1) * BN, :], K_shared)
                T.copy(V[bz, by, k_iter * BN : (k_iter + 1) * BN, :], V_shared)

                T.clear(S_local)
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    q_idx = mx * BM + i
                    k_idx = k_iter * BN + j
                    valid = (q_idx < NQ) & (k_idx < NK)
                    if causal:
                        valid = valid & (k_idx <= q_idx + causal_offset)
                    S_local[i, j] = T.if_then_else(
                        valid,
                        S_local[i, j] * scale_log2e - L_tile[i],
                        -float("inf"),
                    )
                for i, j in T.Parallel(BM, BN):
                    P_local[i, j] = T.exp2(S_local[i, j])

                T.clear(dP_local)
                T.gemm(dO_shared, V_shared, dP_local, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    dS_local[i, j] = P_local[i, j] * (dP_local[i, j] - D_tile[i]) * float(softmax_scale)
                for i, j in T.Parallel(BM, BN):
                    dS_shared[i, j] = T.Cast(ts, dS_local[i, j])

                T.gemm(dS_shared, K_shared, dQ_local)

            for i, j in T.Parallel(BM, D):
                q_idx = mx * BM + i
                if q_idx < NQ:
                    dQ[bz, by, q_idx, j] = T.Cast(ts, dQ_local[i, j])

    return fa_bwd_dq


def make_fwd_prim_func_full(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    causal: bool,
    dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    q_segment_shape: tuple[int, int],
    q_segment_dtype,
    kv_segment_shape: tuple[int, int],
    kv_segment_dtype,
    use_segments: bool,
    softmax_aux_shape: tuple[int, int],
    softmax_aux_dtype,
    use_softmax_aux: bool,
    window: tuple[int, int] | None = None,
    dropout_prob: float = 0.0,
    logits_soft_cap: float | None = None,
    normalize_output: bool = True,
    num_stages: int = 2,
    threads: int = 128,
):
    """Build the feature-complete FA2 forward ``@T.prim_func``.

    This variant is used whenever a caller requests any of the score-space
    features. Bias, user masks, segment ids, attention sinks, sliding-window
    masks and dropout are evaluated inside the kernel from compact inputs. The
    leaner :func:`make_fwd_prim_func` stays the path for the common no-feature
    case, so the autotuned hot kernel is never perturbed.

    GQA/MQA is handled by indexing ``K`` / ``V`` at
    ``kv_head = q_head // (num_heads // num_kv_heads)``.

    Returns:
        ``@T.prim_func`` with buffers
        ``(Q, K, V, Bias, AttentionMask, QSegmentIds, KVSegmentIds,
        SoftmaxAux, DropoutSeed, O, L)``.
    """
    ts = _dtype_str(dtype)
    bias_ts = _scalar_dtype_str(bias_dtype)
    mask_ts = _scalar_dtype_str(mask_dtype)
    qseg_ts = _scalar_dtype_str(q_segment_dtype)
    kvseg_ts = _scalar_dtype_str(kv_segment_dtype)
    aux_ts = _scalar_dtype_str(softmax_aux_dtype)
    accum_dtype = "float32"
    B, H, HK = batch, num_heads, num_kv_heads
    NQ, NK, D = seq_len_q, seq_len_k, head_dim
    BM, BN = block_m, block_n
    BB, BH, BQ, BK = bias_shape
    MB, MH, MQ, MK = mask_shape
    SB, SQ = q_segment_shape
    SKB, SK = kv_segment_shape
    AH, NS = softmax_aux_shape
    G = H // HK
    log2e = 1.4426950408889634
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
    causal_offset = NK - NQ
    neg_inf = -float("inf")
    m_init = -1.0e30
    neg_big = -1.0e30

    @T.prim_func
    def fa_fwd_full(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, HK, NK, D), ts),
        V: T.Tensor((B, HK, NK, D), ts),
        Bias: T.Tensor((BB, BH, BQ, BK), bias_ts),
        AttentionMask: T.Tensor((MB, MH, MQ, MK), mask_ts),
        QSegmentIds: T.Tensor((SB, SQ), qseg_ts),
        KVSegmentIds: T.Tensor((SKB, SK), kvseg_ts),
        SoftmaxAux: T.Tensor((AH, NS), aux_ts),
        DropoutSeed: T.Tensor((2,), "uint32"),
        O: T.Tensor((B, H, NQ, D), ts),
        L: T.Tensor((B, H, NQ), accum_dtype),
        M: T.Tensor((B, H, NQ), accum_dtype),
        AM: T.Tensor((B, H, NQ), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(NQ, BM), H, B, threads=threads) as (bx, by, bz):
            kv_head = by // G
            Q_shared = T.alloc_shared((BM, D), ts)
            K_shared = T.alloc_shared((BN, D), ts)
            V_shared = T.alloc_shared((BN, D), ts)
            S_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_shared = T.alloc_shared((BM, BN), ts)
            O_local = T.alloc_fragment((BM, D), accum_dtype)
            m_local = T.alloc_fragment((BM,), accum_dtype)
            m_prev = T.alloc_fragment((BM,), accum_dtype)
            l_local = T.alloc_fragment((BM,), accum_dtype)
            row_sum = T.alloc_fragment((BM,), accum_dtype)
            alpha = T.alloc_fragment((BM,), accum_dtype)
            sink_max = T.alloc_fragment((1,), accum_dtype)
            is_max = T.alloc_fragment((BM, BN), accum_dtype)
            am_block = T.alloc_fragment((BM,), accum_dtype)
            am_local = T.alloc_fragment((BM,), accum_dtype)
            _hk_ref = T.alloc_fragment((HK,), accum_dtype)
            _bias_dtype_ref = T.alloc_fragment((1,), bias_ts)
            _mask_dtype_ref = T.alloc_fragment((1,), mask_ts)
            _qseg_dtype_ref = T.alloc_fragment((1,), qseg_ts)
            _kvseg_dtype_ref = T.alloc_fragment((1,), kvseg_ts)
            _aux_dtype_ref = T.alloc_fragment((1,), aux_ts)
            _shape_ref = T.alloc_fragment((1,), accum_dtype)
            _shape_ref[0] = T.Cast(
                accum_dtype,
                BB + BH + BQ + BK + MB + MH + MQ + MK + SB + SQ + SKB + SK + AH + NS,
            )

            T.copy(Q[bz, by, bx * BM : (bx + 1) * BM, :], Q_shared)
            T.fill(O_local, 0)
            T.fill(l_local, 0)
            T.fill(m_local, m_init)
            for i in T.Parallel(BM):
                am_local[i] = -1.0

            if causal:
                kv_end = T.min(NK, (bx + 1) * BM + causal_offset)
                loop_count = T.ceildiv(T.max(0, kv_end), BN)
            else:
                loop_count = T.ceildiv(NK, BN)

            for k_iter in T.Pipelined(loop_count, num_stages=num_stages):
                T.copy(K[bz, kv_head, k_iter * BN : (k_iter + 1) * BN, :], K_shared)
                T.copy(V[bz, kv_head, k_iter * BN : (k_iter + 1) * BN, :], V_shared)

                T.clear(S_local)
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    q_idx = bx * BM + i
                    k_idx = k_iter * BN + j
                    q_c = T.min(q_idx, NQ - 1)
                    k_c = T.min(k_idx, NK - 1)
                    valid = k_idx < NK
                    if causal:
                        valid = valid & (k_idx <= q_idx + causal_offset)
                    if use_mask:
                        mb = 0 if MB == 1 else bz
                        mh = 0 if MH == 1 else by
                        mq = 0 if MQ == 1 else q_c
                        mk = 0 if MK == 1 else k_c
                        valid = valid & (T.Cast("int32", AttentionMask[mb, mh, mq, mk]) != 0)
                    if use_window:
                        diff = k_idx - q_idx
                        valid = valid & (diff >= -window_left) & (diff <= window_right)
                    if use_segments:
                        sb = 0 if SB == 1 else bz
                        skb = 0 if SKB == 1 else bz
                        qseg = T.Cast("int64", QSegmentIds[sb, q_c])
                        kseg = T.Cast("int64", KVSegmentIds[skb, k_c])
                        valid = valid & (qseg == kseg) & (qseg >= 0)
                    bias_val = 0.0
                    if use_bias:
                        bb = 0 if BB == 1 else bz
                        bh = 0 if BH == 1 else by
                        bq = 0 if BQ == 1 else q_c
                        bk = 0 if BK == 1 else k_c
                        bias_val = T.Cast(accum_dtype, Bias[bb, bh, bq, bk])
                    s = S_local[i, j] * scale + bias_val
                    if use_cap:
                        s = cap * (1.0 - 2.0 / (T.exp(2.0 * s * inv_cap) + 1.0))
                    s = s + T.if_then_else(valid, 0.0, neg_big)
                    S_local[i, j] = T.if_then_else(valid, s * log2e, neg_inf)

                T.copy(m_local, m_prev)
                T.reduce_max(S_local, m_local, dim=1, clear=False)

                for i, j in T.Parallel(BM, BN):
                    k_as_f = T.Cast(accum_dtype, k_iter * BN + j)
                    is_max[i, j] = T.if_then_else(S_local[i, j] >= m_local[i], k_as_f, -1.0)
                T.reduce_max(is_max, am_block, dim=1, clear=True)
                for i in T.Parallel(BM):
                    am_local[i] = T.if_then_else(am_block[i] >= 0.0, am_block[i], am_local[i])

                for i in T.Parallel(BM):
                    alpha[i] = T.exp2(m_prev[i] - m_local[i])

                for i, j in T.Parallel(BM, BN):
                    P_local[i, j] = T.exp2(S_local[i, j] - m_local[i])

                T.fill(row_sum, 0)
                T.reduce_sum(P_local, row_sum, dim=1, clear=True)
                for i in T.Parallel(BM):
                    l_local[i] = l_local[i] * alpha[i] + row_sum[i]

                for i, j in T.Parallel(BM, BN):
                    q_idx = bx * BM + i
                    k_idx = k_iter * BN + j
                    q_c = T.min(q_idx, NQ - 1)
                    k_c = T.min(k_idx, NK - 1)
                    drop = 1.0
                    if use_dropout:
                        linear = T.Cast("int64", (((bz * H + by) * NQ + q_c) * NK + k_c))
                        seed_mix = T.Cast("int64", DropoutSeed[0]) + T.Cast("int64", DropoutSeed[1]) * 65537
                        rnd_i = (linear * 1103515245 + seed_mix * 12345 + 12345) % 2147483647
                        rnd = T.Cast(accum_dtype, rnd_i) * 4.656612875245797e-10
                        drop = T.if_then_else(rnd >= float(dropout_prob), inv_keep_prob, 0.0)
                    P_local[i, j] = P_local[i, j] * drop

                for i, j in T.Parallel(BM, D):
                    O_local[i, j] = O_local[i, j] * alpha[i]

                for i, j in T.Parallel(BM, BN):
                    P_shared[i, j] = T.Cast(ts, P_local[i, j])
                T.gemm(P_shared, V_shared, O_local)

            if use_softmax_aux:
                sink_max[0] = neg_inf
                for s in T.serial(NS):
                    ah = 0 if AH == 1 else by
                    sink_max[0] = T.max(sink_max[0], T.Cast(accum_dtype, SoftmaxAux[ah, s]) * log2e)
                for i in T.Parallel(BM):
                    m_prev[i] = m_local[i]
                    m_local[i] = T.max(m_local[i], sink_max[0])
                for i in T.Parallel(BM):
                    alpha[i] = T.exp2(m_prev[i] - m_local[i])
                    l_local[i] = l_local[i] * alpha[i]
                    am_local[i] = T.if_then_else(m_local[i] > m_prev[i], -1.0, am_local[i])
                for s in T.serial(NS):
                    ah = 0 if AH == 1 else by
                    for i in T.Parallel(BM):
                        l_local[i] = l_local[i] + T.exp2(T.Cast(accum_dtype, SoftmaxAux[ah, s]) * log2e - m_local[i])
                for i, j in T.Parallel(BM, D):
                    O_local[i, j] = O_local[i, j] * alpha[i]

            for i, j in T.Parallel(BM, D):
                q_idx = bx * BM + i
                if normalize_output:
                    O_local[i, j] = T.if_then_else(
                        q_idx < NQ,
                        O_local[i, j] / T.max(l_local[i], 1e-30),
                        0.0,
                    )
                else:
                    O_local[i, j] = T.if_then_else(q_idx < NQ, O_local[i, j], 0.0)

            for i in T.Parallel(BM):
                q_idx = bx * BM + i
                if q_idx < NQ:
                    L[bz, by, q_idx] = m_local[i] + T.log2(T.max(l_local[i], 1e-30))
                    M[bz, by, q_idx] = m_local[i]
                    AM[bz, by, q_idx] = am_local[i]

            for i, j in T.Parallel(BM, D):
                q_idx = bx * BM + i
                if q_idx < NQ:
                    O[bz, by, q_idx, j] = T.Cast(ts, O_local[i, j])

    return fa_fwd_full


def make_bwd_dkdv_prim_func_full(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    causal: bool,
    dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    q_segment_shape: tuple[int, int],
    q_segment_dtype,
    kv_segment_shape: tuple[int, int],
    kv_segment_dtype,
    use_segments: bool,
    window: tuple[int, int] | None = None,
    dropout_prob: float = 0.0,
    logits_soft_cap: float | None = None,
    normalize_output: bool = True,
    num_stages: int = 2,
    threads: int = 128,
):
    """Feature-complete FA2 dK/dV backward.

    Parallelizes over ``(k_block, kv_head, batch)``. Each CTA loads its KV
    slab once and loops over the ``G = num_heads // num_kv_heads`` query
    heads that share it, accumulating ``dK`` / ``dV`` exclusively — so GQA
    needs no atomics. The recompute re-applies ``bias`` (pre-cap),
    ``logits_soft_cap``, ``mask`` (post-cap), and the post-softmax dropout
    mask generated from the seed so the gradient matches the modified forward
    exactly.

    With ``normalize_output=False`` the forward returns the un-normalised
    ``sum_j exp(s_j - m) V_j`` — whose gradient carries the extra
    ``-[j == argmax]`` term from differentiating the stabilising max, so the
    recompute subtracts the running max ``M`` and the ``argmax`` row is
    detected via ``u >= 1``.

    Returns:
        ``@T.prim_func`` with buffers
        ``(Q, K, V, dO, L, M, Delta, Bias, AttentionMask, QSegmentIds,
        KVSegmentIds, DropoutSeed, dK, dV)``.
    """
    ts = _dtype_str(dtype)
    bias_ts = _scalar_dtype_str(bias_dtype)
    mask_ts = _scalar_dtype_str(mask_dtype)
    qseg_ts = _scalar_dtype_str(q_segment_dtype)
    kvseg_ts = _scalar_dtype_str(kv_segment_dtype)
    accum_dtype = "float32"
    B, H, HK = batch, num_heads, num_kv_heads
    NQ, NK, D = seq_len_q, seq_len_k, head_dim
    BM, BN = block_m, block_n
    BB, BH, BQ, BK = bias_shape
    MB, MH, MQ, MK = mask_shape
    SB, SQ = q_segment_shape
    SKB, SK = kv_segment_shape
    G = H // HK
    log2e = 1.4426950408889634
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
    causal_offset = NK - NQ
    neg_big = -1.0e30

    @T.prim_func
    def fa_bwd_dkdv_full(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, HK, NK, D), ts),
        V: T.Tensor((B, HK, NK, D), ts),
        dO: T.Tensor((B, H, NQ, D), ts),
        L: T.Tensor((B, H, NQ), accum_dtype),
        M: T.Tensor((B, H, NQ), accum_dtype),
        AM: T.Tensor((B, H, NQ), accum_dtype),
        Delta: T.Tensor((B, H, NQ), accum_dtype),
        Bias: T.Tensor((BB, BH, BQ, BK), bias_ts),
        AttentionMask: T.Tensor((MB, MH, MQ, MK), mask_ts),
        QSegmentIds: T.Tensor((SB, SQ), qseg_ts),
        KVSegmentIds: T.Tensor((SKB, SK), kvseg_ts),
        DropoutSeed: T.Tensor((2,), "uint32"),
        dK: T.Tensor((B, HK, NK, D), accum_dtype),
        dV: T.Tensor((B, HK, NK, D), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(NK, BN), HK, B, threads=threads) as (kx, hk, bz):
            K_shared = T.alloc_shared((BN, D), ts)
            V_shared = T.alloc_shared((BN, D), ts)
            Q_shared = T.alloc_shared((BM, D), ts)
            dO_shared = T.alloc_shared((BM, D), ts)

            S_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_local = T.alloc_fragment((BM, BN), accum_dtype)
            s2_local = T.alloc_fragment((BM, BN), accum_dtype)
            dP_local = T.alloc_fragment((BM, BN), accum_dtype)
            dS_local = T.alloc_fragment((BM, BN), accum_dtype)
            Pw_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_shared = T.alloc_shared((BM, BN), ts)
            dS_shared = T.alloc_shared((BM, BN), ts)
            dK_local = T.alloc_fragment((BN, D), accum_dtype)
            dV_local = T.alloc_fragment((BN, D), accum_dtype)
            L_tile = T.alloc_fragment((BM,), accum_dtype)
            M_tile = T.alloc_fragment((BM,), accum_dtype)
            D_tile = T.alloc_fragment((BM,), accum_dtype)
            am_tile = T.alloc_fragment((BM,), accum_dtype)
            _h_ref = T.alloc_fragment((H,), accum_dtype)
            _bias_dtype_ref = T.alloc_fragment((1,), bias_ts)
            _mask_dtype_ref = T.alloc_fragment((1,), mask_ts)
            _qseg_dtype_ref = T.alloc_fragment((1,), qseg_ts)
            _kvseg_dtype_ref = T.alloc_fragment((1,), kvseg_ts)
            _shape_ref = T.alloc_fragment((1,), accum_dtype)
            _shape_ref[0] = T.Cast(
                accum_dtype,
                BB + BH + BQ + BK + MB + MH + MQ + MK + SB + SQ + SKB + SK,
            )

            T.copy(K[bz, hk, kx * BN : (kx + 1) * BN, :], K_shared)
            T.copy(V[bz, hk, kx * BN : (kx + 1) * BN, :], V_shared)
            T.fill(dK_local, 0)
            T.fill(dV_local, 0)

            if causal:
                q_start = T.max(0, kx * BN - causal_offset) // BM
            else:
                q_start = 0
            q_end_excl = T.ceildiv(NQ, BM)

            for g in T.serial(G):
                by = hk * G + g
                for q_iter in T.Pipelined(q_end_excl - q_start, num_stages=num_stages):
                    m_iter = q_start + q_iter
                    T.copy(Q[bz, by, m_iter * BM : (m_iter + 1) * BM, :], Q_shared)
                    T.copy(dO[bz, by, m_iter * BM : (m_iter + 1) * BM, :], dO_shared)
                    for i in T.Parallel(BM):
                        q_idx = m_iter * BM + i
                        L_tile[i] = T.if_then_else(q_idx < NQ, L[bz, by, q_idx], 0.0)
                        M_tile[i] = T.if_then_else(q_idx < NQ, M[bz, by, q_idx], 0.0)
                        D_tile[i] = T.if_then_else(q_idx < NQ, Delta[bz, by, q_idx], 0.0)
                        am_tile[i] = T.if_then_else(q_idx < NQ, AM[bz, by, q_idx], -1.0)

                    T.clear(S_local)
                    T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                    for i, j in T.Parallel(BM, BN):
                        q_idx = m_iter * BM + i
                        k_idx = kx * BN + j
                        q_c = T.min(q_idx, NQ - 1)
                        k_c = T.min(k_idx, NK - 1)
                        valid = (q_idx < NQ) & (k_idx < NK)
                        if causal:
                            valid = valid & (k_idx <= q_idx + causal_offset)
                        if use_mask:
                            mb = 0 if MB == 1 else bz
                            mh = 0 if MH == 1 else by
                            mq = 0 if MQ == 1 else q_c
                            mk = 0 if MK == 1 else k_c
                            valid = valid & (T.Cast("int32", AttentionMask[mb, mh, mq, mk]) != 0)
                        if use_window:
                            diff = k_idx - q_idx
                            valid = valid & (diff >= -window_left) & (diff <= window_right)
                        if use_segments:
                            sb = 0 if SB == 1 else bz
                            skb = 0 if SKB == 1 else bz
                            qseg = T.Cast("int64", QSegmentIds[sb, q_c])
                            kseg = T.Cast("int64", KVSegmentIds[skb, k_c])
                            valid = valid & (qseg == kseg) & (qseg >= 0)
                        bias_val = 0.0
                        if use_bias:
                            bb = 0 if BB == 1 else bz
                            bh = 0 if BH == 1 else by
                            bq = 0 if BQ == 1 else q_c
                            bk = 0 if BK == 1 else k_c
                            bias_val = T.Cast(accum_dtype, Bias[bb, bh, bq, bk])
                        s1 = S_local[i, j] * scale + bias_val
                        if use_cap:
                            s2 = cap * (1.0 - 2.0 / (T.exp(2.0 * s1 * inv_cap) + 1.0))
                        else:
                            s2 = s1
                        s2_local[i, j] = s2
                        s3 = s2 + T.if_then_else(valid, 0.0, neg_big)
                        if normalize_output:
                            s_norm = s3 * log2e - L_tile[i]
                        else:
                            s_norm = s3 - M_tile[i] / log2e
                        S_local[i, j] = T.if_then_else(valid, s_norm, -float("inf"))
                    for i, j in T.Parallel(BM, BN):
                        if normalize_output:
                            P_local[i, j] = T.exp2(S_local[i, j])
                        else:
                            P_local[i, j] = T.exp(S_local[i, j])

                    for i, j in T.Parallel(BM, BN):
                        q_idx = m_iter * BM + i
                        k_idx = kx * BN + j
                        q_c = T.min(q_idx, NQ - 1)
                        k_c = T.min(k_idx, NK - 1)
                        drop = 1.0
                        if use_dropout:
                            linear = T.Cast("int64", (((bz * H + by) * NQ + q_c) * NK + k_c))
                            seed_mix = T.Cast("int64", DropoutSeed[0]) + T.Cast("int64", DropoutSeed[1]) * 65537
                            rnd_i = (linear * 1103515245 + seed_mix * 12345 + 12345) % 2147483647
                            rnd = T.Cast(accum_dtype, rnd_i) * 4.656612875245797e-10
                            drop = T.if_then_else(rnd >= float(dropout_prob), inv_keep_prob, 0.0)
                        Pw_local[i, j] = P_local[i, j] * drop
                    for i, j in T.Parallel(BM, BN):
                        P_shared[i, j] = T.Cast(ts, Pw_local[i, j])
                    T.gemm(P_shared, dO_shared, dV_local, transpose_A=True)

                    T.clear(dP_local)
                    T.gemm(dO_shared, V_shared, dP_local, transpose_B=True)

                    for i, j in T.Parallel(BM, BN):
                        q_idx = m_iter * BM + i
                        k_idx = kx * BN + j
                        q_c = T.min(q_idx, NQ - 1)
                        k_c = T.min(k_idx, NK - 1)
                        drop = 1.0
                        if use_dropout:
                            linear = T.Cast("int64", (((bz * H + by) * NQ + q_c) * NK + k_c))
                            seed_mix = T.Cast("int64", DropoutSeed[0]) + T.Cast("int64", DropoutSeed[1]) * 65537
                            rnd_i = (linear * 1103515245 + seed_mix * 12345 + 12345) % 2147483647
                            rnd = T.Cast(accum_dtype, rnd_i) * 4.656612875245797e-10
                            drop = T.if_then_else(rnd >= float(dropout_prob), inv_keep_prob, 0.0)
                        dpw = dP_local[i, j] * drop
                        if normalize_output:
                            ds3 = P_local[i, j] * (dpw - D_tile[i])
                        else:
                            am_hit = T.Cast(accum_dtype, k_idx) == am_tile[i]
                            ds3 = P_local[i, j] * dpw - T.if_then_else(am_hit, D_tile[i], 0.0)
                        if use_cap:
                            cap_g = 1.0 - (s2_local[i, j] * inv_cap) * (s2_local[i, j] * inv_cap)
                        else:
                            cap_g = 1.0
                        dS_local[i, j] = ds3 * cap_g * scale
                    for i, j in T.Parallel(BM, BN):
                        dS_shared[i, j] = T.Cast(ts, dS_local[i, j])

                    T.gemm(dS_shared, Q_shared, dK_local, transpose_A=True)

            for i, j in T.Parallel(BN, D):
                k_idx = kx * BN + i
                if k_idx < NK:
                    dK[bz, hk, k_idx, j] = dK_local[i, j]
                    dV[bz, hk, k_idx, j] = dV_local[i, j]

    return fa_bwd_dkdv_full


def make_bwd_dq_prim_func_full(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    causal: bool,
    dtype,
    bias_shape: tuple[int, int, int, int],
    bias_dtype,
    use_bias: bool,
    mask_shape: tuple[int, int, int, int],
    mask_dtype,
    use_mask: bool,
    q_segment_shape: tuple[int, int],
    q_segment_dtype,
    kv_segment_shape: tuple[int, int],
    kv_segment_dtype,
    use_segments: bool,
    window: tuple[int, int] | None = None,
    dropout_prob: float = 0.0,
    logits_soft_cap: float | None = None,
    normalize_output: bool = True,
    num_stages: int = 2,
    threads: int = 128,
):
    """Feature-complete FA2 dQ backward.

    Parallelizes over ``(q_block, q_head, batch)`` and loops over K-blocks,
    reading ``K`` / ``V`` at ``kv_head = q_head // G`` for GQA. Re-applies
    bias / soft-cap / mask / dropout in the recompute so dQ matches the
    modified forward; ``normalize_output=False`` subtracts the running max
    ``M`` and carries the ``argmax`` gradient term.

    Returns:
        ``@T.prim_func`` with buffers
        ``(Q, K, V, dO, L, M, Delta, Bias, AttentionMask, QSegmentIds,
        KVSegmentIds, DropoutSeed, dQ)``.
    """
    ts = _dtype_str(dtype)
    bias_ts = _scalar_dtype_str(bias_dtype)
    mask_ts = _scalar_dtype_str(mask_dtype)
    qseg_ts = _scalar_dtype_str(q_segment_dtype)
    kvseg_ts = _scalar_dtype_str(kv_segment_dtype)
    accum_dtype = "float32"
    B, H, HK = batch, num_heads, num_kv_heads
    NQ, NK, D = seq_len_q, seq_len_k, head_dim
    BM, BN = block_m, block_n
    BB, BH, BQ, BK = bias_shape
    MB, MH, MQ, MK = mask_shape
    SB, SQ = q_segment_shape
    SKB, SK = kv_segment_shape
    G = H // HK
    log2e = 1.4426950408889634
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
    causal_offset = NK - NQ
    neg_big = -1.0e30

    @T.prim_func
    def fa_bwd_dq_full(
        Q: T.Tensor((B, H, NQ, D), ts),
        K: T.Tensor((B, HK, NK, D), ts),
        V: T.Tensor((B, HK, NK, D), ts),
        dO: T.Tensor((B, H, NQ, D), ts),
        L: T.Tensor((B, H, NQ), accum_dtype),
        M: T.Tensor((B, H, NQ), accum_dtype),
        AM: T.Tensor((B, H, NQ), accum_dtype),
        Delta: T.Tensor((B, H, NQ), accum_dtype),
        Bias: T.Tensor((BB, BH, BQ, BK), bias_ts),
        AttentionMask: T.Tensor((MB, MH, MQ, MK), mask_ts),
        QSegmentIds: T.Tensor((SB, SQ), qseg_ts),
        KVSegmentIds: T.Tensor((SKB, SK), kvseg_ts),
        DropoutSeed: T.Tensor((2,), "uint32"),
        dQ: T.Tensor((B, H, NQ, D), ts),
    ):
        with T.Kernel(T.ceildiv(NQ, BM), H, B, threads=threads) as (mx, by, bz):
            kv_head = by // G
            Q_shared = T.alloc_shared((BM, D), ts)
            dO_shared = T.alloc_shared((BM, D), ts)
            K_shared = T.alloc_shared((BN, D), ts)
            V_shared = T.alloc_shared((BN, D), ts)

            S_local = T.alloc_fragment((BM, BN), accum_dtype)
            P_local = T.alloc_fragment((BM, BN), accum_dtype)
            s2_local = T.alloc_fragment((BM, BN), accum_dtype)
            dP_local = T.alloc_fragment((BM, BN), accum_dtype)
            dS_local = T.alloc_fragment((BM, BN), accum_dtype)
            dS_shared = T.alloc_shared((BM, BN), ts)
            dQ_local = T.alloc_fragment((BM, D), accum_dtype)
            L_tile = T.alloc_fragment((BM,), accum_dtype)
            M_tile = T.alloc_fragment((BM,), accum_dtype)
            D_tile = T.alloc_fragment((BM,), accum_dtype)
            am_tile = T.alloc_fragment((BM,), accum_dtype)
            _hk_ref = T.alloc_fragment((HK,), accum_dtype)
            _bias_dtype_ref = T.alloc_fragment((1,), bias_ts)
            _mask_dtype_ref = T.alloc_fragment((1,), mask_ts)
            _qseg_dtype_ref = T.alloc_fragment((1,), qseg_ts)
            _kvseg_dtype_ref = T.alloc_fragment((1,), kvseg_ts)
            _shape_ref = T.alloc_fragment((1,), accum_dtype)
            _shape_ref[0] = T.Cast(
                accum_dtype,
                BB + BH + BQ + BK + MB + MH + MQ + MK + SB + SQ + SKB + SK,
            )

            T.copy(Q[bz, by, mx * BM : (mx + 1) * BM, :], Q_shared)
            T.copy(dO[bz, by, mx * BM : (mx + 1) * BM, :], dO_shared)
            T.fill(dQ_local, 0)
            for i in T.Parallel(BM):
                q_idx = mx * BM + i
                L_tile[i] = T.if_then_else(q_idx < NQ, L[bz, by, q_idx], 0.0)
                M_tile[i] = T.if_then_else(q_idx < NQ, M[bz, by, q_idx], 0.0)
                D_tile[i] = T.if_then_else(q_idx < NQ, Delta[bz, by, q_idx], 0.0)
                am_tile[i] = T.if_then_else(q_idx < NQ, AM[bz, by, q_idx], -1.0)

            if causal:
                k_end_excl = T.ceildiv(T.min(NK, (mx + 1) * BM + causal_offset), BN)
                k_end_excl = T.max(0, k_end_excl)
            else:
                k_end_excl = T.ceildiv(NK, BN)

            for k_iter in T.Pipelined(k_end_excl, num_stages=num_stages):
                T.copy(K[bz, kv_head, k_iter * BN : (k_iter + 1) * BN, :], K_shared)
                T.copy(V[bz, kv_head, k_iter * BN : (k_iter + 1) * BN, :], V_shared)

                T.clear(S_local)
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    q_idx = mx * BM + i
                    k_idx = k_iter * BN + j
                    q_c = T.min(q_idx, NQ - 1)
                    k_c = T.min(k_idx, NK - 1)
                    valid = (q_idx < NQ) & (k_idx < NK)
                    if causal:
                        valid = valid & (k_idx <= q_idx + causal_offset)
                    if use_mask:
                        mb = 0 if MB == 1 else bz
                        mh = 0 if MH == 1 else by
                        mq = 0 if MQ == 1 else q_c
                        mk = 0 if MK == 1 else k_c
                        valid = valid & (T.Cast("int32", AttentionMask[mb, mh, mq, mk]) != 0)
                    if use_window:
                        diff = k_idx - q_idx
                        valid = valid & (diff >= -window_left) & (diff <= window_right)
                    if use_segments:
                        sb = 0 if SB == 1 else bz
                        skb = 0 if SKB == 1 else bz
                        qseg = T.Cast("int64", QSegmentIds[sb, q_c])
                        kseg = T.Cast("int64", KVSegmentIds[skb, k_c])
                        valid = valid & (qseg == kseg) & (qseg >= 0)
                    bias_val = 0.0
                    if use_bias:
                        bb = 0 if BB == 1 else bz
                        bh = 0 if BH == 1 else by
                        bq = 0 if BQ == 1 else q_c
                        bk = 0 if BK == 1 else k_c
                        bias_val = T.Cast(accum_dtype, Bias[bb, bh, bq, bk])
                    s1 = S_local[i, j] * scale + bias_val
                    if use_cap:
                        s2 = cap * (1.0 - 2.0 / (T.exp(2.0 * s1 * inv_cap) + 1.0))
                    else:
                        s2 = s1
                    s2_local[i, j] = s2
                    s3 = s2 + T.if_then_else(valid, 0.0, neg_big)
                    if normalize_output:
                        s_norm = s3 * log2e - L_tile[i]
                    else:
                        s_norm = s3 - M_tile[i] / log2e
                    S_local[i, j] = T.if_then_else(valid, s_norm, -float("inf"))
                for i, j in T.Parallel(BM, BN):
                    if normalize_output:
                        P_local[i, j] = T.exp2(S_local[i, j])
                    else:
                        P_local[i, j] = T.exp(S_local[i, j])

                T.clear(dP_local)
                T.gemm(dO_shared, V_shared, dP_local, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    q_idx = mx * BM + i
                    k_idx = k_iter * BN + j
                    q_c = T.min(q_idx, NQ - 1)
                    k_c = T.min(k_idx, NK - 1)
                    drop = 1.0
                    if use_dropout:
                        linear = T.Cast("int64", (((bz * H + by) * NQ + q_c) * NK + k_c))
                        seed_mix = T.Cast("int64", DropoutSeed[0]) + T.Cast("int64", DropoutSeed[1]) * 65537
                        rnd_i = (linear * 1103515245 + seed_mix * 12345 + 12345) % 2147483647
                        rnd = T.Cast(accum_dtype, rnd_i) * 4.656612875245797e-10
                        drop = T.if_then_else(rnd >= float(dropout_prob), inv_keep_prob, 0.0)
                    dpw = dP_local[i, j] * drop
                    if normalize_output:
                        ds3 = P_local[i, j] * (dpw - D_tile[i])
                    else:
                        am_hit = T.Cast(accum_dtype, k_idx) == am_tile[i]
                        ds3 = P_local[i, j] * dpw - T.if_then_else(am_hit, D_tile[i], 0.0)
                    if use_cap:
                        cap_g = 1.0 - (s2_local[i, j] * inv_cap) * (s2_local[i, j] * inv_cap)
                    else:
                        cap_g = 1.0
                    dS_local[i, j] = ds3 * cap_g * scale
                for i, j in T.Parallel(BM, BN):
                    dS_shared[i, j] = T.Cast(ts, dS_local[i, j])

                T.gemm(dS_shared, K_shared, dQ_local)

            for i, j in T.Parallel(BM, D):
                q_idx = mx * BM + i
                if q_idx < NQ:
                    dQ[bz, by, q_idx, j] = T.Cast(ts, dQ_local[i, j])

    return fa_bwd_dq_full
