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

"""TileLang prim_func factories for paged single-query (decode) attention.

``make_page_attention_prim_func`` generates one of two ``@T.prim_func``
variants depending on ``heads_first_cache``:

**heads-first cache** (``heads_first_cache=True``):
    ``K: [HKV, num_pages, page_size, D]``,
    ``V: [HKV, num_pages, page_size, D]``.

**pages-first cache** (``heads_first_cache=False``):
    ``K: [num_pages, HKV, page_size, D]``,
    ``V: [num_pages, HKV, page_size, D]``.

The layout is inferred from the cache shape by ``_infer_cache_layout`` in
``_interface.py`` and baked in at compile time.

**Grid**: ``(num_q_heads, batch)`` — one CTA per ``(head, sequence)``.

Each CTA performs:
1. Load ``Q[b, h, :]`` into a register fragment.
2. Loop over ``max_tokens`` KV positions in tiles of ``BLOCK_K`` using
   a ``num_stages``-stage software pipeline.
3. For each tile: translate the logical position through ``BlockTables``
   to a physical page, apply context-length + optional sliding-window mask,
   and perform online softmax (``log2`` / ``T.exp2``).
4. Write ``O[b, h, d] = cast(acc * (1/l), dtype)``.

Optional features baked in at compile time:
- **Sliding-window causal mask** (``sliding_window > 0``).
- **Logits soft-cap** via tanh approximation (``logits_soft_cap > 0``).
- **Max-context-length clip** (``max_context_len``).

No backward pass is provided — this is a decode-only kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Map a JAX/NumPy activation dtype to the TileLang type string.

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
        raise TypeError(f"Unsupported dtype for page_attention: {dtype}")
    return mapping[canonical]


def _index_dtype_str(dtype) -> str:
    """Map a JAX/NumPy index dtype to the TileLang type string.

    Args:
        dtype: Any dtype accepted by ``jnp.dtype()`` — int32 or int64.

    Returns:
        ``"int32"`` or ``"int64"``.

    Raises:
        TypeError: If *dtype* is not int32 or int64.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.int32): "int32",
        jnp.dtype(jnp.int64): "int64",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported index dtype for page_attention: {dtype}")
    return mapping[canonical]


def make_page_attention_prim_func(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    max_blocks: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    mask_value: float,
    max_context_len: int,
    sliding_window: int,
    logits_soft_cap: float,
    heads_first_cache: bool,
    dtype,
    index_dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build a paged decode attention ``@T.prim_func``.

    Grid: ``(num_q_heads, batch)``.  One CTA per ``(query head, sequence)``.

    Each CTA iterates over KV token positions in tiles of ``BLOCK_K`` via
    a ``num_stages``-stage software pipeline.  For each tile it:
    1. Translates the logical position through ``BlockTables`` to a physical
       page, loading ``K_shared`` and ``V_shared`` with shape
       ``(BLOCK_K, head_dim)`` in the activation dtype.
    2. Applies context-length masking (and optionally sliding-window masking).
    3. Optionally applies a ``tanh``-style logit soft-cap.
    4. Accumulates the online softmax in ``log2`` space (``T.exp2``).

    Two ``@T.prim_func`` bodies are generated depending on *heads_first_cache*:
    - ``heads_first_cache=True``: cache indexed as ``K[kv_head, page, offset, d]``.
    - ``heads_first_cache=False``: cache indexed as ``K[page, kv_head, offset, d]``.

    Args:
        batch: Number of sequences (batch size) ``B``.
        num_q_heads: Query head count ``HQ``.
        num_kv_heads: KV head count ``HKV``; must divide ``HQ``.
        num_pages: Total physical pages in the KV cache ``P``.
        page_size: Tokens per physical page ``PS``.
        max_blocks: Maximum logical pages per sequence ``MB``; determines
            ``max_tokens = max_blocks * page_size``.
        head_dim: Per-head feature dimension ``D``.
        block_k: KV tile size for the pipelined loop ``BK``.
        softmax_scale: Attention temperature applied to raw dot-products.
        mask_value: Value filled for masked positions (converted to log2 space
            internally); typically a large negative float.
        max_context_len: Hard cap on the attended context length; KV positions
            beyond this are masked.
        sliding_window: Sliding-window size; positions older than
            ``context_len - sliding_window`` are masked.  Pass ``<= 0`` to
            disable.
        logits_soft_cap: Soft-cap threshold; pass ``<= 0.0`` to disable.
        heads_first_cache: Whether the KV cache is indexed
            ``[HKV, pages, page_size, D]`` (``True``) or
            ``[pages, HKV, page_size, D]`` (``False``).
        dtype: Activation dtype (float16, bfloat16, float32).
        index_dtype: Index dtype for ``ContextLens`` and ``BlockTables``
            (int32 or int64).
        num_stages: Software-pipeline stages for the KV loop (default 3).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(Q: [B, HQ, D], K: [...], V: [...], ContextLens: [B], BlockTables: [B, MB],
        O: [B, HQ, D])``.  ``O`` is written in *dtype*.
    """
    ts = _dtype_str(dtype)
    index_ts = _index_dtype_str(index_dtype)
    accum = "float32"
    B, HQ, HKV, P, PS, MB, D, BK = batch, num_q_heads, num_kv_heads, num_pages, page_size, max_blocks, head_dim, block_k
    max_tokens = MB * PS
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale)
    scale_log2e = scale * log2e
    mask_value_log2e = max(float(mask_value) * log2e, -1e30)
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    context_cap = max_context_len
    window = sliding_window

    if heads_first_cache:

        @T.prim_func
        def page_attention_fwd(
            Q: T.Tensor((B, HQ, D), ts),
            K: T.Tensor((HKV, P, PS, D), ts),
            V: T.Tensor((HKV, P, PS, D), ts),
            ContextLens: T.Tensor((B,), index_ts),
            BlockTables: T.Tensor((B, MB), index_ts),
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
                context_len_raw = T.Cast("int32", ContextLens[bx])
                context_len = T.min(context_len_raw, context_cap)
                if window > 0:
                    valid_start = T.max(0, context_len - window)
                else:
                    valid_start = 0

                for d in T.Parallel(D):
                    Q_loc[d] = T.Cast(accum, Q[bx, hx, d])

                T.fill(O_local, 0)
                m_run[0] = -1e30
                l_run[0] = 0.0

                for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                    for i, d in T.Parallel(BK, D):
                        logical_idx = k_iter * BK + i
                        logical_block = logical_idx // PS
                        safe_block = T.min(logical_block, MB - 1)
                        page_offset = logical_idx - logical_block * PS
                        token_valid = (logical_idx < context_len) & (logical_idx >= valid_start)
                        page_ok = logical_block < MB
                        page_idx = T.if_then_else(page_ok, T.Cast("int32", BlockTables[bx, safe_block]), 0)
                        page_valid = (page_idx >= 0) & (page_idx < P)
                        in_range = token_valid & page_valid
                        K_shared[i, d] = T.if_then_else(
                            in_range,
                            K[kv_head, page_idx, page_offset, d],
                            T.Cast(ts, 0.0),
                        )
                        V_shared[i, d] = T.if_then_else(
                            in_range,
                            V[kv_head, page_idx, page_offset, d],
                            T.Cast(ts, 0.0),
                        )

                    for i, d in T.Parallel(BK, D):
                        QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                    T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                    for i in T.Parallel(BK):
                        logical_idx = k_iter * BK + i
                        logical_block = logical_idx // PS
                        safe_block = T.min(logical_block, MB - 1)
                        token_valid = (logical_idx < context_len) & (logical_idx >= valid_start)
                        page_ok = logical_block < MB
                        page_idx = T.if_then_else(page_ok, T.Cast("int32", BlockTables[bx, safe_block]), 0)
                        page_valid = (page_idx >= 0) & (page_idx < P)
                        in_range = token_valid & page_valid
                        score = S_local[i]
                        if use_softcap:
                            score_natural = score * scale
                            tanh_arg = score_natural / softcap
                            soft_score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                            S_local[i] = T.if_then_else(in_range, soft_score * log2e, mask_value_log2e)
                        else:
                            S_local[i] = T.if_then_else(in_range, score * scale_log2e, mask_value_log2e)

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

        return page_attention_fwd

    @T.prim_func
    def page_attention_fwd(
        Q: T.Tensor((B, HQ, D), ts),
        K: T.Tensor((P, HKV, PS, D), ts),
        V: T.Tensor((P, HKV, PS, D), ts),
        ContextLens: T.Tensor((B,), index_ts),
        BlockTables: T.Tensor((B, MB), index_ts),
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
            context_len_raw = T.Cast("int32", ContextLens[bx])
            context_len = T.min(context_len_raw, context_cap)
            if window > 0:
                valid_start = T.max(0, context_len - window)
            else:
                valid_start = 0

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[bx, hx, d])

            T.fill(O_local, 0)
            m_run[0] = -1e30
            l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                for i, d in T.Parallel(BK, D):
                    logical_idx = k_iter * BK + i
                    logical_block = logical_idx // PS
                    safe_block = T.min(logical_block, MB - 1)
                    page_offset = logical_idx - logical_block * PS
                    token_valid = (logical_idx < context_len) & (logical_idx >= valid_start)
                    page_ok = logical_block < MB
                    page_idx = T.if_then_else(page_ok, T.Cast("int32", BlockTables[bx, safe_block]), 0)
                    page_valid = (page_idx >= 0) & (page_idx < P)
                    in_range = token_valid & page_valid
                    K_shared[i, d] = T.if_then_else(
                        in_range,
                        K[page_idx, kv_head, page_offset, d],
                        T.Cast(ts, 0.0),
                    )
                    V_shared[i, d] = T.if_then_else(
                        in_range,
                        V[page_idx, kv_head, page_offset, d],
                        T.Cast(ts, 0.0),
                    )

                for i, d in T.Parallel(BK, D):
                    QK_prod[i, d] = Q_loc[d] * T.Cast(accum, K_shared[i, d])
                T.reduce_sum(QK_prod, S_local, dim=1, clear=True)
                for i in T.Parallel(BK):
                    logical_idx = k_iter * BK + i
                    logical_block = logical_idx // PS
                    safe_block = T.min(logical_block, MB - 1)
                    token_valid = (logical_idx < context_len) & (logical_idx >= valid_start)
                    page_ok = logical_block < MB
                    page_idx = T.if_then_else(page_ok, T.Cast("int32", BlockTables[bx, safe_block]), 0)
                    page_valid = (page_idx >= 0) & (page_idx < P)
                    in_range = token_valid & page_valid
                    score = S_local[i]
                    if use_softcap:
                        score_natural = score * scale
                        tanh_arg = score_natural / softcap
                        soft_score = softcap * (2.0 / (1.0 + T.exp2(-2.0 * tanh_arg * inv_ln2)) - 1.0)
                        S_local[i] = T.if_then_else(in_range, soft_score * log2e, mask_value_log2e)
                    else:
                        S_local[i] = T.if_then_else(in_range, score * scale_log2e, mask_value_log2e)

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

    return page_attention_fwd
