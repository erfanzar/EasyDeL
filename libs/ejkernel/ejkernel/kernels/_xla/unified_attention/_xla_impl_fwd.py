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

"""Unified Attention forward pass using XLA/JAX.

This module provides the forward pass for unified attention, a flexible
attention implementation for inference with paged KV caches and ragged
(variable-length) batches.

Key Components:
    - _unified_attention_fwd: Main forward function

Algorithm:
    1. Iterate over sequences in the ragged batch.
    2. For each sequence, iterate over query blocks of size ``qblocks`` (16).
    3. For each query block, iterate over KV cache blocks and accumulate
       attention using the online softmax algorithm.
    4. Normalise each query block's accumulator and write to the output buffer.

Features:
    - Paged KV cache with block-table indirection
    - Ragged batches: variable-length sequences packed in a single tensor
    - GQA/MQA support with automatic head grouping
    - ALiBi positional encoding
    - Query-query attention bias (prefill self-attention)
    - Attention sinks via softmax_aux
    - Sliding window local attention
    - Logit soft capping for numerical stability

Memory Layout:
    - queries: [total_tokens, num_q_heads, head_dim]  (all sequences packed)
    - key_cache / value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    - block_tables: [num_seqs, max_blocks_per_seq]  (logical → physical mapping)
    - query_start_loc: [num_seqs + 1]  (cumulative query token counts)
    - kv_lens: [num_seqs]  (total KV context length per sequence)

Note:
    This is a pure-JAX/XLA reference implementation intended for correctness
    verification.  For GPU production workloads, prefer the Triton backend.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.callib import ejit

DEFAULT_MASK_VALUE = -2.381976426469702e38


@ejit(static_argnames=("softmax_scale", "sliding_window", "logits_soft_cap"))
def _unified_attention_fwd(
    *,
    queries: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    softmax_scale: float,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    alibi_slopes: jax.Array | None,
    qq_bias: jax.Array | None,
    softmax_aux: jax.Array | None,
) -> jax.Array:
    """Compute unified attention forward pass for ragged batches with paged KV cache.

    Handles variable-length sequences packed together using block-tabled
    KV caches. Processes sequences one at a time, dividing each sequence's
    queries into blocks and iterating over KV cache blocks with online
    softmax for memory-efficient computation.

    Supports ALiBi position encoding, query-query attention bias,
    attention sinks (softmax_aux), sliding window, and logit soft capping.

    Args:
        queries: Packed query tokens [total_tokens, num_q_heads, head_dim].
        key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
        value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
        kv_lens: Context length per sequence [num_seqs].
        block_tables: Block table mapping [num_seqs, max_blocks_per_seq].
        query_start_loc: Cumulative query counts [num_seqs + 1].
        softmax_scale: Scaling factor for QK^T.
        sliding_window: Optional sliding window size (None or 0 disables).
        logits_soft_cap: Optional soft cap value (None or 0 disables).
        alibi_slopes: Optional ALiBi slopes [num_q_heads] (None disables).
        qq_bias: Optional query-query bias [dim, dim] (None disables).
        softmax_aux: Optional attention sink logits [num_q_heads] (None disables).

    Returns:
        Attention output [total_tokens, num_q_heads, head_dim].

    Raises:
        ValueError: If input shapes, dtypes, or configurations are invalid.
    """
    if queries.ndim != 3:
        raise ValueError("queries must be rank-3: [total_tokens, num_q_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key_cache/value_cache must be rank-4: [num_blocks, block_size, num_kv_heads, head_dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache/value_cache shape mismatch")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise ValueError("kv_lens/block_tables/query_start_loc must be int32")

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_blocks, block_size, num_kv_heads, head_dim_kv = map(int, key_cache.shape)
    if head_dim_kv != head_dim:
        raise ValueError("head_dim mismatch between queries and KV cache")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")

    num_seqs, _max_blocks_per_seq = map(int, block_tables.shape)
    if kv_lens.shape != (num_seqs,) or query_start_loc.shape != (num_seqs + 1,):
        raise ValueError("kv_lens/query_start_loc shapes must match block_tables[0]")

    use_alibi = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    use_sinks = softmax_aux is not None

    if use_alibi:
        if alibi_slopes.shape != (num_q_heads,):
            raise ValueError("alibi_slopes must have shape (num_q_heads,)")
        alibi_h = alibi_slopes.reshape(num_kv_heads, num_q_heads // num_kv_heads).astype(jnp.float32)
    else:
        alibi_h = jnp.zeros((1, 1), dtype=jnp.float32)

    if use_qq_bias:
        if qq_bias.ndim != 2 or qq_bias.shape[0] != qq_bias.shape[1]:
            raise ValueError("qq_bias must be square [num_query_tokens, num_query_tokens]")
        qq_dim = int(qq_bias.shape[0])
        qq_bias_f32 = qq_bias.astype(jnp.float32)
    else:
        qq_dim = 0
        qq_bias_f32 = jnp.zeros((1, 1), dtype=jnp.float32)

    if use_sinks:
        if softmax_aux.shape != (num_q_heads,):
            raise ValueError("softmax_aux must have shape (num_q_heads,)")
        sink_h = softmax_aux.reshape(num_kv_heads, num_q_heads // num_kv_heads).astype(jnp.float32)
    else:
        sink_h = jnp.zeros((1, 1), dtype=jnp.float32)

    if sliding_window is None:
        sliding_window_val = 0
    else:
        sliding_window_val = int(sliding_window)
        if sliding_window_val <= 0:
            sliding_window_val = 0

    if logits_soft_cap is None:
        logits_soft_cap_val = 0.0
    else:
        logits_soft_cap_val = float(logits_soft_cap)

    qblocks = 16
    pad_q = qblocks - 1
    q_grouped = queries.reshape(total_tokens, num_kv_heads, num_q_heads // num_kv_heads, head_dim)
    if pad_q:
        q_grouped = jnp.concatenate(
            [q_grouped, jnp.zeros((pad_q, num_kv_heads, num_q_heads // num_kv_heads, head_dim), dtype=q_grouped.dtype)],
            axis=0,
        )
    padded_total_tokens = int(q_grouped.shape[0])
    o_grouped = jnp.zeros_like(q_grouped)

    arange_q = jnp.arange(qblocks, dtype=jnp.int32)
    arange_k = jnp.arange(block_size, dtype=jnp.int32)
    mask_value = jnp.array(DEFAULT_MASK_VALUE, dtype=jnp.float32)

    def _seq_body(seq_idx, o_acc):
        """Process attention for a single sequence in the ragged batch.

        Determines the query range and KV context for this sequence,
        then dispatches block-level query processing with online softmax.

        Args:
            seq_idx: Index of the current sequence.
            o_acc: Output accumulator being updated.

        Returns:
            Updated output accumulator with this sequence's results.
        """
        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[seq_idx]
        context_len = kv_len - q_len

        num_q_blocks = (q_len + jnp.int32(qblocks) - 1) // jnp.int32(qblocks)
        num_kv_blocks = (kv_len + jnp.int32(block_size) - 1) // jnp.int32(block_size)

        def _process_query_block(qb, o_inner):
            """Compute attention for one query block against all KV blocks.

            Loads a block of queries, initializes online softmax state,
            iterates over all KV cache blocks, normalizes, and writes
            the result to the output buffer.

            Args:
                qb: Query block index within this sequence.
                o_inner: Output buffer being updated.

            Returns:
                Updated output buffer with this query block's results.
            """
            q_off = qb * jnp.int32(qblocks)
            q_global_start = q_start + q_off
            q_block = jax.lax.dynamic_slice(
                q_grouped,
                (q_global_start, 0, 0, 0),
                (qblocks, num_kv_heads, num_q_heads // num_kv_heads, head_dim),
            ).astype(jnp.float32)

            row_pos = q_off + arange_q
            q_valid = row_pos < q_len
            q_abs = context_len + row_pos

            init_acc = jnp.zeros((qblocks, num_kv_heads, num_q_heads // num_kv_heads, head_dim), dtype=jnp.float32)
            if use_sinks:
                init_l = jnp.ones((qblocks, num_kv_heads, num_q_heads // num_kv_heads), dtype=jnp.float32)
                init_m = jnp.broadcast_to(
                    sink_h[None, :, :], (qblocks, num_kv_heads, num_q_heads // num_kv_heads)
                ).astype(jnp.float32)
            else:
                init_l = jnp.zeros((qblocks, num_kv_heads, num_q_heads // num_kv_heads), dtype=jnp.float32)
                init_m = jnp.full((qblocks, num_kv_heads, num_q_heads // num_kv_heads), -jnp.inf, dtype=jnp.float32)

            def _process_kv_block(kb, carry):
                """Process one KV cache block and update the online softmax state.

                Loads a physical KV block, computes attention scores with
                causal masking, optional ALiBi bias, qq_bias, sliding window,
                and soft capping. Updates the running online softmax accumulator.

                Args:
                    kb: KV block index within this sequence's block table.
                    carry: Tuple of (acc, l, m) online softmax state.

                Returns:
                    Updated (acc, l, m) state.
                """
                acc, l, m = carry
                physical_block = block_tables[seq_idx, kb]
                k_block = key_cache[physical_block].astype(jnp.float32)
                v_block = value_cache[physical_block].astype(jnp.float32)

                kv_pos = kb * jnp.int32(block_size) + arange_k
                kv_valid = kv_pos < kv_len

                allowed = kv_pos[None, :] <= q_abs[:, None]
                if sliding_window_val > 0:
                    left = q_abs - jnp.int32(sliding_window_val) + jnp.int32(1)
                    allowed = allowed & (kv_pos[None, :] >= left[:, None])
                valid = q_valid[:, None] & kv_valid[None, :] & allowed
                valid = valid[:, None, None, :]

                logits = (
                    jnp.einsum(
                        "bihd,kid->bihk",
                        q_block,
                        k_block,
                        preferred_element_type=jnp.float32,
                    )
                    * softmax_scale
                )

                if use_alibi:
                    key_rel = (kv_pos - context_len).astype(jnp.float32)
                    logits = logits + alibi_h[None, :, :, None] * key_rel[None, None, None, :]

                if use_qq_bias:
                    row_idx = row_pos
                    row_valid = row_idx < jnp.int32(qq_dim)
                    row_idx_clip = jnp.clip(row_idx, 0, qq_dim - 1)

                    key_rel_pos = kv_pos - context_len
                    key_is_q = (key_rel_pos >= 0) & (key_rel_pos < jnp.int32(qq_dim))
                    key_rel_clip = jnp.clip(key_rel_pos, 0, qq_dim - 1)

                    qq_rows = qq_bias_f32[row_idx_clip]
                    qq = qq_rows[:, key_rel_clip]
                    qq = jnp.where(row_valid[:, None] & key_is_q[None, :], qq, 0.0)
                    logits = logits + qq[:, None, None, :]

                if logits_soft_cap_val > 0:
                    logits = logits_soft_cap_val * jnp.tanh(logits / logits_soft_cap_val)

                logits = logits + jnp.where(valid, 0.0, mask_value)

                cur_max = jnp.max(logits, axis=-1)
                new_m = jnp.maximum(m, cur_max)
                exp_logits = jnp.exp(logits - new_m[..., None])
                exp_logits = jnp.where(valid, exp_logits, 0.0)

                rescale = jnp.exp(m - new_m)
                l = rescale * l + jnp.sum(exp_logits, axis=-1)
                acc = rescale[..., None] * acc + jnp.einsum(
                    "bihk,kid->bihd",
                    exp_logits,
                    v_block,
                    preferred_element_type=jnp.float32,
                )
                return acc, l, new_m

            acc, l, _m = jax.lax.fori_loop(0, num_kv_blocks, _process_kv_block, (init_acc, init_l, init_m))
            l = jnp.maximum(l, 1e-6)
            out_block = (acc / l[..., None]).astype(queries.dtype)

            existing = jax.lax.dynamic_slice(
                o_inner,
                (q_global_start, 0, 0, 0),
                (qblocks, num_kv_heads, num_q_heads // num_kv_heads, head_dim),
            )
            out_block = jnp.where(q_valid[:, None, None, None], out_block, existing)
            return jax.lax.dynamic_update_slice(o_inner, out_block, (q_global_start, 0, 0, 0))

        return jax.lax.fori_loop(0, num_q_blocks, _process_query_block, o_acc)

    o_grouped = jax.lax.fori_loop(0, num_seqs, _seq_body, o_grouped)
    out = o_grouped.reshape(padded_total_tokens, num_q_heads, head_dim)[:total_tokens]
    return out
