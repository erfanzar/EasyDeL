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

"""XLA implementation of Dual-Stream Ragged Paged Attention.

Computes exact joint-softmax attention for ragged continuous-batching queries
over two paged KV caches simultaneously:

1. **Global stream (read-only):** Full-history paged KV cache (e.g., Pass-1 KV
   shared into Pass 2+ of a Loop Transformer), attended causally (``overlap=True``)
   or strictly outside the local window (``overlap=False``).
2. **Local stream (fused write + sliding-window read):** Sliding-window paged KV
   cache updated in-place with ``local_keys`` / ``local_values`` and attended
   over the local causal window of width ``sliding_window``.

Supports independent per-stream page pools, page counts, and rebased sequence
lengths (e.g., after leading-page eviction in a sliding-window cache group).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.callib import ejit
from ejkernel.kernels._xla.ragged_page_attention_v3._xla_impl_fwd import (
    DEFAULT_MASK_VALUE,
    align_to,
    get_dtype_packing,
    kv_pair_in_head_dim,
    merge_kv,
)


def _unpack_kv_page_block(
    kv_page_block: jax.Array,
    *,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    kv_tokens_per_block: int,
    num_kv_heads_x2: int,
    head_dim_padded: int,
) -> tuple[jax.Array, jax.Array]:
    """Unpack a sliced block of merged KV pages into separate K and V arrays."""
    kv_tok = kv_page_block.reshape(kv_tokens_per_block, num_kv_heads_x2, head_dim_padded)
    if kv_pair_in_head_dim(actual_head_dim):
        kv_tok = kv_tok[:, :actual_num_kv_heads, :]
        k_block = kv_tok[:, :, :actual_head_dim]
        v_block = kv_tok[:, :, actual_head_dim : actual_head_dim * 2]
        return k_block, v_block
    kv_tok = kv_tok[:, : actual_num_kv_heads * 2, :].reshape(
        kv_tokens_per_block,
        actual_num_kv_heads,
        2,
        head_dim_padded,
    )
    return kv_tok[:, :, 0, :actual_head_dim], kv_tok[:, :, 1, :actual_head_dim]


def _online_softmax_step(
    state: tuple[jax.Array, jax.Array, jax.Array],
    q_block: jax.Array,
    k_block: jax.Array,
    v_block: jax.Array,
    mask: jax.Array,
    *,
    softmax_scale: float,
    logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    mask_value: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run one blocked online-softmax accumulation step in float32."""
    acc, l, m = state
    logits = jnp.einsum("bihd,kid->bihk", q_block, k_block, preferred_element_type=jnp.float32)
    logits = logits * softmax_scale
    if k_scale is not None:
        logits = logits * k_scale
    if q_scale is not None:
        logits = logits * q_scale
    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)
    logits = jnp.where(mask, -jnp.inf, logits) if mask_value == -jnp.inf else logits + jnp.where(mask, mask_value, 0.0)

    cur_max = jnp.max(jnp.where(mask, -jnp.inf, logits), axis=-1)
    new_m = jnp.maximum(m, cur_max)
    shift_m = jnp.where(jnp.isfinite(new_m), new_m, 0.0)
    exp_logits = jnp.where(mask, 0.0, jnp.exp(logits - shift_m[..., None]))
    rescale = jnp.where(jnp.isfinite(m), jnp.exp(m - shift_m), 0.0)
    l = rescale * l + jnp.sum(exp_logits, axis=-1)
    acc = rescale[..., None] * acc + jnp.einsum(
        "bihk,kid->bihd",
        exp_logits,
        v_block,
        preferred_element_type=jnp.float32,
    )
    return acc, l, new_m


def static_validate_dual_stream_rpa_inputs(
    queries: jax.Array,
    local_keys: jax.Array,
    local_values: jax.Array,
    global_kv_cache: jax.Array,
    local_kv_cache: jax.Array,
    global_kv_lens: jax.Array,
    global_block_tables: jax.Array,
    local_kv_lens: jax.Array,
    local_block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    softmax_aux: jax.Array | None = None,
    *,
    sliding_window: int = 128,
    logits_soft_cap: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> None:
    """Validate static shapes, dtypes, and scalar options for dual-stream paged attention."""
    if not (queries.ndim == local_keys.ndim == local_values.ndim == 3):
        raise ValueError(
            f"Expected 3D tensors for queries/local_keys/local_values, got "
            f"{queries.shape=}, {local_keys.shape=}, {local_values.shape=}."
        )
    if local_keys.shape != local_values.shape:
        raise ValueError(f"Expected {local_keys.shape=} == {local_values.shape=}.")
    if not (queries.shape[0] == local_keys.shape[0] and queries.shape[2] == local_keys.shape[2]):
        raise ValueError(
            f"Expected matching token count and head_dim between queries {queries.shape} "
            f"and local_keys {local_keys.shape}."
        )
    if queries.shape[1] % local_keys.shape[1] != 0:
        raise ValueError(
            f"num_q_heads ({queries.shape[1]}) must be divisible by num_kv_heads ({local_keys.shape[1]})."
        )
    if global_kv_cache.ndim != 5 or local_kv_cache.ndim != 5:
        raise ValueError(
            f"Expected 5D paged KV caches, got {global_kv_cache.shape=} and {local_kv_cache.shape=}."
        )
    if global_kv_cache.shape[1:] != local_kv_cache.shape[1:]:
        raise ValueError(
            f"Expected global_kv_cache.shape[1:] ({global_kv_cache.shape[1:]}) == "
            f"local_kv_cache.shape[1:] ({local_kv_cache.shape[1:]})."
        )
    if global_kv_cache.dtype != local_kv_cache.dtype:
        raise ValueError(
            f"Expected matching dtypes for global_kv_cache ({global_kv_cache.dtype}) "
            f"and local_kv_cache ({local_kv_cache.dtype})."
        )
    if global_kv_lens.ndim != 1 or local_kv_lens.ndim != 1 or global_kv_lens.shape != local_kv_lens.shape:
        raise ValueError(
            f"Expected matching 1D global_kv_lens ({global_kv_lens.shape}) and local_kv_lens ({local_kv_lens.shape})."
        )
    max_num_seqs = int(global_kv_lens.shape[0])
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected query_start_loc shape ({max_num_seqs + 1},), got {query_start_loc.shape}.")
    if distribution.shape != (3,):
        raise ValueError(f"Expected distribution shape (3,), got {distribution.shape}.")
    if global_block_tables.size % max_num_seqs != 0:
        raise ValueError(
            f"global_block_tables size ({global_block_tables.size}) must be divisible by max_num_seqs ({max_num_seqs})."
        )
    if local_block_tables.size % max_num_seqs != 0:
        raise ValueError(
            f"local_block_tables size ({local_block_tables.size}) must be divisible by max_num_seqs ({max_num_seqs})."
        )
    if int(sliding_window) <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}.")
    if logits_soft_cap is not None and logits_soft_cap == 0.0:
        raise ValueError("logits_soft_cap must be non-zero when provided.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"chunk_prefill_size must be positive, got {chunk_prefill_size}.")
    if num_kv_pages_per_block is not None and num_kv_pages_per_block <= 0:
        raise ValueError(f"num_kv_pages_per_block must be positive, got {num_kv_pages_per_block}.")
    if num_queries_per_block is not None and num_queries_per_block <= 0:
        raise ValueError(f"num_queries_per_block must be positive, got {num_queries_per_block}.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"vmem_limit_bytes must be positive, got {vmem_limit_bytes}.")
    if softmax_aux is not None and softmax_aux.shape != (queries.shape[1],):
        raise ValueError(f"Expected softmax_aux shape ({queries.shape[1]},), got {softmax_aux.shape}.")


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
        "overlap",
        "logits_soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
    donate_argnames=("local_kv_cache",),
)
def dual_stream_ragged_paged_attention_xla(
    queries: jax.Array,
    local_keys: jax.Array,
    local_values: jax.Array,
    global_kv_cache: jax.Array,
    local_kv_cache: jax.Array,
    global_kv_lens: jax.Array,
    global_block_tables: jax.Array,
    local_kv_lens: jax.Array,
    local_block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    softmax_aux: jax.Array | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int = 128,
    overlap: bool = True,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """XLA blocked online-softmax dual-stream ragged paged attention."""
    del chunk_prefill_size, vmem_limit_bytes
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    static_validate_dual_stream_rpa_inputs(
        queries,
        local_keys,
        local_values,
        global_kv_cache,
        local_kv_cache,
        global_kv_lens,
        global_block_tables,
        local_kv_lens,
        local_block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )

    global_block_tables = global_block_tables.reshape(-1)
    local_block_tables = local_block_tables.reshape(-1)

    total_q, actual_num_q_heads, actual_head_dim = queries.shape
    actual_num_kv_heads = local_keys.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded = local_kv_cache.shape
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    max_num_seqs = int(global_kv_lens.shape[0])
    g_pages_per_seq = int(global_block_tables.shape[0]) // max_num_seqs
    l_pages_per_seq = int(local_block_tables.shape[0]) // max_num_seqs
    l_tokens_per_seq = l_pages_per_seq * page_size

    assert get_dtype_packing(local_kv_cache.dtype) == kv_packing
    if kv_pair_in_head_dim(actual_head_dim):
        assert num_kv_heads_x2 == align_to(actual_num_kv_heads, kv_packing)
        assert head_dim_padded == actual_head_dim * 2
    else:
        assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
        assert head_dim_padded == align_to(actual_head_dim, 128)

    qblocks = 8 if num_queries_per_block is None else max(1, int(num_queries_per_block))
    qblocks = min(qblocks, max(1, total_q))

    def _choose_kv_blocks(pages_per_seq: int) -> int:
        if num_kv_pages_per_block is not None:
            return max(1, min(pages_per_seq, int(num_kv_pages_per_block)))
        for cap in (256, 128, 64):
            if pages_per_seq >= cap:
                return cap
        return max(1, pages_per_seq)

    g_kvblocks = _choose_kv_blocks(g_pages_per_seq)
    l_kvblocks = _choose_kv_blocks(l_pages_per_seq)
    g_kv_tokens_per_block = g_kvblocks * page_size
    l_kv_tokens_per_block = l_kvblocks * page_size

    pad_q = qblocks - 1
    if pad_q:
        queries = jnp.concatenate(
            [queries, jnp.zeros((pad_q, actual_num_q_heads, actual_head_dim), queries.dtype)],
            axis=0,
        )
        local_keys = jnp.concatenate(
            [local_keys, jnp.zeros((pad_q, actual_num_kv_heads, actual_head_dim), local_keys.dtype)],
            axis=0,
        )
        local_values = jnp.concatenate(
            [local_values, jnp.zeros((pad_q, actual_num_kv_heads, actual_head_dim), local_values.dtype)],
            axis=0,
        )

    padded_total_q = queries.shape[0]
    q_grouped = queries.reshape(
        padded_total_q,
        actual_num_kv_heads,
        actual_num_q_heads_per_kv_head,
        actual_head_dim,
    )
    merged_local_kv = merge_kv(local_keys, local_values)
    o_grouped = jnp.zeros_like(q_grouped)

    arange_q = jnp.arange(qblocks, dtype=jnp.int32)
    arange_g_kv = jnp.arange(g_kv_tokens_per_block, dtype=jnp.int32)
    arange_l_kv = jnp.arange(l_kv_tokens_per_block, dtype=jnp.int32)
    sw_i32 = jnp.int32(sliding_window)

    sinks_h = (
        softmax_aux.reshape(actual_num_kv_heads, actual_num_q_heads_per_kv_head).astype(jnp.float32)
        if softmax_aux is not None
        else None
    )

    def _seq_body(seq_idx, carry):
        o_acc, local_cache_acc = carry

        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        q_len = q_end - q_start
        g_kv_len = global_kv_lens[seq_idx]
        l_kv_len = local_kv_lens[seq_idx]

        g_write_start = g_kv_len - q_len
        l_write_start = l_kv_len - q_len
        num_q_blocks = (q_len + qblocks - 1) // qblocks

        # Gather local pages and write new Pass-2 K/V tokens.
        l_indices = lax.dynamic_slice(local_block_tables, (seq_idx * l_pages_per_seq,), (l_pages_per_seq,))
        l_pages = local_cache_acc[l_indices]
        l_pages_flat = l_pages.reshape(
            l_tokens_per_seq,
            num_kv_heads_x2_per_kv_packing,
            kv_packing,
            head_dim_padded,
        )

        def _update_local_kv_block(qb, kv_flat_pad):
            q_off = qb * qblocks
            dst = l_write_start + q_off
            src = q_start + q_off
            updates = lax.dynamic_slice(
                merged_local_kv,
                (src, 0, 0, 0),
                (qblocks, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
            )
            existing = lax.dynamic_slice(
                kv_flat_pad,
                (dst, 0, 0, 0),
                (qblocks, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
            )
            q_valid = (q_off + arange_q) < q_len
            updates = jnp.where(q_valid[:, None, None, None], updates, existing)
            return lax.dynamic_update_slice(kv_flat_pad, updates, (dst, 0, 0, 0))

        l_pages_flat_padded = jnp.concatenate(
            [
                l_pages_flat,
                jnp.zeros(
                    (qblocks - 1, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                    dtype=l_pages_flat.dtype,
                ),
            ],
            axis=0,
        )
        l_pages_flat_padded = lax.fori_loop(0, num_q_blocks, _update_local_kv_block, l_pages_flat_padded)
        l_pages = l_pages_flat_padded[:l_tokens_per_seq].reshape(l_pages.shape)
        local_cache_acc = local_cache_acc.at[l_indices].set(l_pages)

        # Gather global pages (read-only).
        g_indices = lax.dynamic_slice(global_block_tables, (seq_idx * g_pages_per_seq,), (g_pages_per_seq,))
        g_pages = global_kv_cache[g_indices]

        g_pages_padded = jnp.concatenate(
            [g_pages, jnp.zeros((g_kvblocks - 1, *g_pages.shape[1:]), dtype=g_pages.dtype)],
            axis=0,
        )
        l_pages_padded = jnp.concatenate(
            [l_pages, jnp.zeros((l_kvblocks - 1, *l_pages.shape[1:]), dtype=l_pages.dtype)],
            axis=0,
        )

        def _process_query_block(qb, o_inner):
            q_off = qb * qblocks
            q_global_start = q_start + q_off
            q_block = lax.dynamic_slice(
                q_grouped,
                (q_global_start, 0, 0, 0),
                (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
            )
            if q_scale is not None:
                q_block = q_block / q_scale
                if jnp.issubdtype(local_kv_cache.dtype, jnp.floating):
                    finfo = jnp.finfo(local_kv_cache.dtype)
                    q_block = jnp.clip(q_block, finfo.min, finfo.max)
                q_block = q_block.astype(local_kv_cache.dtype)

            q_tok = q_off + arange_q
            q_valid = q_tok < q_len
            g_q_pos = g_write_start + q_tok
            l_q_pos = l_write_start + q_tok

            init_acc = jnp.zeros(
                (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
                dtype=jnp.float32,
            )
            if sinks_h is not None:
                init_m = jnp.broadcast_to(
                    sinks_h[None, :, :],
                    (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head),
                )
                init_l = jnp.ones_like(init_m)
            else:
                init_m = jnp.full(
                    (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head),
                    -jnp.inf,
                    dtype=jnp.float32,
                )
                init_l = jnp.zeros_like(init_m)

            # Stream 1: global paged KV (read-only).
            g_q_hi = jnp.minimum(g_kv_len, g_write_start + q_off + qblocks)
            g_visible_hi = g_q_hi if overlap else jnp.maximum(g_q_hi - sw_i32, 0)
            g_end_blocks = (g_visible_hi + g_kv_tokens_per_block - 1) // g_kv_tokens_per_block

            def _global_kv_block(kb, state):
                page_block = lax.dynamic_slice(
                    g_pages_padded,
                    (kb * g_kvblocks, 0, 0, 0, 0),
                    (g_kvblocks, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                )
                k_block, v_block = _unpack_kv_page_block(
                    page_block,
                    actual_num_kv_heads=actual_num_kv_heads,
                    actual_head_dim=actual_head_dim,
                    kv_tokens_per_block=g_kv_tokens_per_block,
                    num_kv_heads_x2=num_kv_heads_x2,
                    head_dim_padded=head_dim_padded,
                )
                kv_pos = kb * jnp.int32(g_kv_tokens_per_block) + arange_g_kv
                kv_valid = kv_pos < g_kv_len
                upper = g_q_pos[:, None] if overlap else (g_q_pos[:, None] - sw_i32)
                mask = (kv_pos[None, :] > upper) | (~kv_valid[None, :]) | (~q_valid[:, None])
                return _online_softmax_step(
                    state,
                    q_block,
                    k_block,
                    v_block,
                    mask[:, None, None, :],
                    softmax_scale=softmax_scale,
                    logits_soft_cap=logits_soft_cap,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    mask_value=mask_value,
                )

            state = lax.fori_loop(0, g_end_blocks, _global_kv_block, (init_acc, init_l, init_m))

            # Stream 2: local sliding-window paged KV (continuing the same online-softmax state).
            l_lo = jnp.maximum(l_write_start + q_off - sw_i32 + 1, 0)
            l_start_block = l_lo // jnp.int32(l_kv_tokens_per_block)
            l_hi = jnp.minimum(l_kv_len, l_write_start + q_off + qblocks)
            l_end_blocks = (l_hi + l_kv_tokens_per_block - 1) // l_kv_tokens_per_block

            def _local_kv_block(kb, inner_state):
                page_block = lax.dynamic_slice(
                    l_pages_padded,
                    (kb * l_kvblocks, 0, 0, 0, 0),
                    (l_kvblocks, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                )
                k_block, v_block = _unpack_kv_page_block(
                    page_block,
                    actual_num_kv_heads=actual_num_kv_heads,
                    actual_head_dim=actual_head_dim,
                    kv_tokens_per_block=l_kv_tokens_per_block,
                    num_kv_heads_x2=num_kv_heads_x2,
                    head_dim_padded=head_dim_padded,
                )
                kv_pos = kb * jnp.int32(l_kv_tokens_per_block) + arange_l_kv
                kv_valid = kv_pos < l_kv_len
                mask = (
                    (kv_pos[None, :] > l_q_pos[:, None])
                    | (kv_pos[None, :] <= (l_q_pos[:, None] - sw_i32))
                    | (~kv_valid[None, :])
                    | (~q_valid[:, None])
                )
                return _online_softmax_step(
                    inner_state,
                    q_block,
                    k_block,
                    v_block,
                    mask[:, None, None, :],
                    softmax_scale=softmax_scale,
                    logits_soft_cap=logits_soft_cap,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    mask_value=mask_value,
                )

            acc, l, _ = lax.fori_loop(l_start_block, l_end_blocks, _local_kv_block, state)
            out_block = (acc / jnp.maximum(l, 1e-6)[..., None]).astype(queries.dtype)
            if v_scale is not None:
                out_block = out_block * v_scale

            existing = lax.dynamic_slice(
                o_inner,
                (q_global_start, 0, 0, 0),
                (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
            )
            out_block = jnp.where(q_valid[:, None, None, None], out_block, existing)
            return lax.dynamic_update_slice(o_inner, out_block, (q_global_start, 0, 0, 0))

        o_acc = lax.fori_loop(0, num_q_blocks, _process_query_block, o_acc)
        return o_acc, local_cache_acc

    num_seqs = distribution[2]
    o_grouped, local_kv_cache = lax.fori_loop(0, num_seqs, _seq_body, (o_grouped, local_kv_cache))
    out = o_grouped.reshape(padded_total_q, actual_num_q_heads, actual_head_dim)[:total_q]
    return out, local_kv_cache
