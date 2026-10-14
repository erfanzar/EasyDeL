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

"""TileLang prim_func factory for chunked-prefill paged attention.

``make_prefill_page_attention_prim_func`` builds a single ``@T.prim_func``
that handles the prefill phase of paged attention, where a contiguous chunk
of query tokens (the current prefill chunk) attends to the *full* KV context
stored in the paged cache.

Key differences from the decode kernel in ``page_attention/_kernel.py``:
- Grid: ``(num_q_heads, chunk_size)`` instead of ``(num_q_heads, batch)``.
  Each CTA handles one query token from the current prefill chunk.
- The cache is always heads-first: ``K/V: [HKV, total_pages, page_size, D]``.
- Causal mask uses ``logical_idx <= query_pos`` (relative to the chunk)
  rather than a bare ``logical_idx < context_len``.
- ``page_indices`` is a flat ``[pages_per_seq]`` array (not a per-batch table)
  because this kernel processes a single sequence at a time.

Optional features baked in at compile time:
- **Sliding-window causal mask** (``sliding_window > 0``).
- **Logits soft-cap** via tanh approximation (``logits_soft_cap > 0``).

No backward pass is provided — this is an inference-only kernel.
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
        raise TypeError(f"Unsupported dtype for prefill_page_attention: {dtype}")
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
        raise TypeError(f"Unsupported index dtype for prefill_page_attention: {dtype}")
    return mapping[canonical]


def make_prefill_page_attention_prim_func(
    *,
    chunk_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    total_pages: int,
    pages_per_seq: int,
    page_size: int,
    head_dim: int,
    block_k: int,
    softmax_scale: float,
    mask_value: float,
    sliding_window: int,
    logits_soft_cap: float,
    dtype,
    index_dtype,
    num_stages: int = 3,
    threads: int = 128,
):
    """Build a chunked-prefill paged attention ``@T.prim_func``.

    Grid: ``(num_q_heads, chunk_size)``.  One CTA per ``(query head, query token)``
    in the current prefill chunk.

    Each CTA:
    1. Loads ``Q[qx, hx, :]`` into a register fragment.
    2. Loops over ``max_tokens = pages_per_seq * page_size`` KV positions in
       tiles of ``BLOCK_K`` via a ``num_stages``-stage software pipeline.
    3. For each tile: translates logical KV positions through ``PageIndices``
       to physical pages, applies the causal mask (``kv_pos <= query_pos``
       and ``kv_pos >= valid_start`` for sliding-window), and optionally
       applies a tanh-style logit soft-cap.
    4. Maintains online softmax state in ``log2`` space using ``T.exp2``.
    5. Writes ``O[qx, hx, d]`` in *dtype*.

    The query position within the *full* context is derived as::

        query_pos = context_len - chunk_size + qx

    where ``context_len`` is read from the scalar ``ContextLen`` tensor.

    Args:
        chunk_size: Number of query tokens in the current prefill chunk
            (``QN``).
        num_q_heads: Number of query heads ``HQ``.
        num_kv_heads: Number of KV heads ``HKV``; must divide ``HQ``.
        total_pages: Total physical pages in the KV cache ``P``.
        pages_per_seq: Number of physical pages allocated for this sequence
            ``MB``; determines ``max_tokens = pages_per_seq * page_size``.
        page_size: Tokens per physical page ``PS``.
        head_dim: Per-head feature dimension ``D``.
        block_k: KV tile size for the pipelined loop ``BK``.
        softmax_scale: Attention temperature.
        mask_value: Fill value for masked positions (converted to log2 space).
        sliding_window: Sliding-window size; pass ``<= 0`` to disable.
        logits_soft_cap: Soft-cap threshold; pass ``<= 0.0`` to disable.
        dtype: Activation dtype (float16, bfloat16, float32).
        index_dtype: Index dtype for ``ContextLen`` and ``PageIndices``
            (int32 or int64).
        num_stages: Software-pipeline stages (default 3).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature::

            (Q:          [chunk_size, HQ, D],
             K:          [HKV, total_pages, page_size, D],
             V:          [HKV, total_pages, page_size, D],
             ContextLen: [1],
             PageIndices:[pages_per_seq],
             O:          [chunk_size, HQ, D])

        ``O`` is written in *dtype*.
    """
    ts = _dtype_str(dtype)
    index_ts = _index_dtype_str(index_dtype)
    accum = "float32"
    QN, HQ, HKV, P, MB, PS, D, BK = (
        chunk_size,
        num_q_heads,
        num_kv_heads,
        total_pages,
        pages_per_seq,
        page_size,
        head_dim,
        block_k,
    )
    max_tokens = MB * PS
    q_heads_per_kv = HQ // HKV
    log2e = 1.4426950408889634
    inv_ln2 = 1.4426950408889634
    scale = float(softmax_scale)
    scale_log2e = scale * log2e
    mask_value_log2e = max(float(mask_value) * log2e, -1e30)
    softcap = float(logits_soft_cap)
    use_softcap = softcap > 0.0
    window = sliding_window

    @T.prim_func
    def prefill_page_attention_fwd(
        Q: T.Tensor((QN, HQ, D), ts),
        K: T.Tensor((HKV, P, PS, D), ts),
        V: T.Tensor((HKV, P, PS, D), ts),
        ContextLen: T.Tensor((1,), index_ts),
        PageIndices: T.Tensor((MB,), index_ts),
        O: T.Tensor((QN, HQ, D), ts),
    ):
        with T.Kernel(HQ, QN, threads=threads) as (hx, qx):
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
            context_len = T.Cast("int32", ContextLen[0])
            query_pos = context_len - QN + qx
            if window > 0:
                valid_start = T.max(0, query_pos - window + 1)
            else:
                valid_start = 0

            for d in T.Parallel(D):
                Q_loc[d] = T.Cast(accum, Q[qx, hx, d])

            T.fill(O_local, 0)
            m_run[0] = -1e30
            l_run[0] = 0.0

            for k_iter in T.Pipelined(T.ceildiv(max_tokens, BK), num_stages=num_stages):
                for i, d in T.Parallel(BK, D):
                    logical_idx = k_iter * BK + i
                    logical_block = logical_idx // PS
                    safe_block = T.min(logical_block, MB - 1)
                    page_offset = logical_idx - logical_block * PS
                    token_valid = (logical_idx < context_len) & (logical_idx <= query_pos) & (logical_idx >= valid_start)
                    page_ok = logical_block < MB
                    page_idx = T.if_then_else(page_ok, T.Cast("int32", PageIndices[safe_block]), 0)
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
                    token_valid = (logical_idx < context_len) & (logical_idx <= query_pos) & (logical_idx >= valid_start)
                    page_ok = logical_block < MB
                    page_idx = T.if_then_else(page_ok, T.Cast("int32", PageIndices[safe_block]), 0)
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
                O[qx, hx, d] = T.Cast(ts, O_local[d] * inv_l[0])

    return prefill_page_attention_fwd
