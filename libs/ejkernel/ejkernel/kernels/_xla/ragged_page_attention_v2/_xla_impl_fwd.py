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

"""Ragged Paged Attention V2 forward pass using XLA/JAX.

This module implements the XLA reference forward pass for ragged paged
attention V2.  It is a read-only cache kernel: KV pages are pre-populated
and this kernel only reads from them.

KV cache layout:
    ``kv_pages[page_id, token_in_page, combined_kv_head, head_dim]``
    where the combined-kv-head axis interleaves keys at even indices
    (``0, 2, 4, ...``) and values at odd indices (``1, 3, 5, ...``).

Algorithm:
    Three nested ``fori_loop`` / ``cond`` calls:

    1. **Sequence loop** (``num_seqs`` iterations): skip sequences with no
       queries via ``lax.cond``.
    2. **Query-block loop** (``num_query_blocks`` per sequence): tile the
       query tokens into blocks of ``qblocks`` (default 8).
    3. **KV-block loop** (``num_kv_blocks`` per query block): tile pages into
       blocks of ``kvblocks * page_size`` tokens; accumulate online-softmax.

    KV position ``kv_pos`` is attended to by query at position ``q_pos`` iff:

    * ``kv_pos < context_len`` (bounds), AND
    * ``kv_pos <= q_pos`` (causal), AND
    * (if ``sliding_window`` set) ``kv_pos > q_pos - sliding_window``.

    Query position is
    ``q_pos = context_len - num_queries + q_block_offset + token_in_block``.

    If ``softmax_aux`` is provided, the online-softmax running max is
    seeded with ``softmax_aux[h]`` and the normaliser with ``1.0`` per head,
    providing virtual sink-token probability mass.

Note:
    This is a correctness-focused XLA fallback.  For TPU production
    workloads prefer the Pallas implementation with async DMA.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ejkernel.callib import ejit

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@ejit(static_argnums=(7, 8, 9))
def _ragged_paged_attention(
    queries: jnp.ndarray,
    kv_pages: jnp.ndarray,
    context_lens: jnp.ndarray,
    block_tables: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    num_seqs: jnp.ndarray,
    softmax_scale: float,
    logits_soft_cap: float | None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sliding_window: int | None = None,
    softmax_aux: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute ragged paged attention V2 forward pass.

    Processes variable-length sequences packed into a flat query tensor.
    KV data is read from a paged cache addressed via ``block_tables``; no
    cache writes are performed (read-only variant).

    Query block size ``qblocks`` is 8 if ``total_query_tokens >= 8`` else
    ``max(1, total_query_tokens)``.  KV block size ``kvblocks`` is 64 if
    ``max_pages_per_sequence >= 64`` else ``max(1, max_pages_per_sequence)``.
    Both query and KV arrays are padded to multiples of these block sizes so
    that ``lax.dynamic_slice`` calls remain in-bounds.

    Args:
        queries: Packed query tokens, shape
            ``[total_query_tokens, num_q_heads, head_size]``.
        kv_pages: Interleaved paged KV cache, shape
            ``[num_pages, page_size, num_kv_heads * 2, head_dim]``.
            Axis-2 even indices are keys; odd indices are values.
        context_lens: Total KV context length per sequence, shape ``[num_seqs]``.
        block_tables: Physical-page lookup table, shape
            ``[num_seqs, max_pages_per_seq]``.
        query_start_loc: Cumulative query token offsets, shape ``[num_seqs+1]``.
        num_seqs: Number of active sequences (scalar or shape ``[1]``).
            Static at trace time via ``static_argnums``.
        softmax_scale: Multiplicative scale for QK^T logits.
        logits_soft_cap: Optional tanh capping radius.  Static at trace time.
        compute_dtype: Dtype for intermediate matmul and accumulation.
            Static at trace time.
        sliding_window: Optional left-only sliding window size.  Static.
        softmax_aux: Optional per-head sink logits, shape ``[num_q_heads]``.
            Initialises the online-softmax running max/normaliser.

    Returns:
        Attention output, shape ``[total_query_tokens, num_q_heads, head_size]``,
        same dtype as ``queries``.
    """
    total_query_tokens, num_q_heads, head_size = queries.shape
    page_size = kv_pages.shape[1]
    num_kv_heads = kv_pages.shape[2] // 2
    max_pages_per_sequence = block_tables.shape[-1]
    out_shape = (total_query_tokens, num_q_heads, head_size)
    q_heads_per_group = num_q_heads // num_kv_heads

    queries = queries.reshape(total_query_tokens, num_kv_heads, q_heads_per_group, head_size)
    qblocks = 8 if total_query_tokens >= 8 else max(1, total_query_tokens)
    kvblocks = 64 if max_pages_per_sequence >= 64 else max(1, max_pages_per_sequence)

    padd = (-total_query_tokens) % qblocks
    if padd > 0:
        padding_shape = (padd, num_kv_heads, q_heads_per_group, head_size)
        padded_queries = jnp.concatenate([queries, jnp.zeros(padding_shape, dtype=queries.dtype)], axis=0)
    else:
        padded_queries = queries

    padded_queries = jnp.concatenate(
        [padded_queries, jnp.zeros((qblocks, num_kv_heads, q_heads_per_group, head_size), dtype=padded_queries.dtype)],
        axis=0,
    )

    attention_output = jnp.zeros_like(padded_queries)

    have_sinks = softmax_aux is not None
    if have_sinks:
        if softmax_aux.ndim != 1 or softmax_aux.shape[0] != num_q_heads:
            raise ValueError(f"softmax_aux must have shape [num_q_heads] = [{num_q_heads}], got {softmax_aux.shape}")
        sinks_h = softmax_aux.reshape(num_kv_heads, q_heads_per_group)

    def _compute_attention_for_sequence(seq_idx, output_accumulator):
        """Process attention for a single sequence in the ragged batch.

        Determines the query range for this sequence and dispatches to
        the block-level attention computation if there are queries.

        Args:
            seq_idx: Index of the current sequence.
            output_accumulator: Running output buffer being updated.

        Returns:
            Updated output_accumulator with this sequence's results.
        """
        num_queries_for_seq = query_start_loc[seq_idx + 1] - query_start_loc[seq_idx]

        def _process_sequence_with_queries():
            """Execute attention computation for a sequence with one or more queries.

            Divides the sequence's queries into blocks and processes each
            block against the full KV cache using the online softmax algorithm.

            Returns:
                Updated output_accumulator with this sequence's attention results.
            """
            num_query_blocks = (num_queries_for_seq + qblocks - 1) // qblocks

            def _process_query_block(query_block_idx, block_output_accumulator):
                """Process one query block against the entire KV cache.

                Loads a block of queries, iterates over all KV blocks using
                online softmax to accumulate the attention output, then
                writes the normalized result back to the output buffer.

                Args:
                    query_block_idx: Index of the current query block.
                    block_output_accumulator: Output buffer being updated.

                Returns:
                    Updated block_output_accumulator.
                """
                query_block_offset = query_block_idx * qblocks
                q_global_start = query_start_loc[seq_idx] + query_block_offset
                query_block = jax.lax.dynamic_slice(
                    padded_queries,
                    (q_global_start, 0, 0, 0),
                    (qblocks, num_kv_heads, q_heads_per_group, head_size),
                )
                valid_queries = (jax.lax.iota(jnp.int32, qblocks) + query_block_offset) < num_queries_for_seq
                query_block = jnp.where(valid_queries[:, None, None, None], query_block, 0)

                kv_cache_len_for_seq = context_lens[seq_idx]

                q_start_tok = kv_cache_len_for_seq - num_queries_for_seq + query_block_offset
                query_token_indices = jnp.arange(qblocks, dtype=jnp.int32) + q_start_tok

                kv_tokens_per_block = page_size * kvblocks
                base_k_ids = jnp.arange(kv_tokens_per_block, dtype=jnp.int32)
                num_kv_blocks = (kv_cache_len_for_seq + kv_tokens_per_block - 1) // kv_tokens_per_block

                def _process_kv_block(kv_block_idx, online_softmax_carry):
                    """Process one KV block and update the online softmax state.

                    Gathers KV pages for this block, computes attention scores
                    with causal and validity masking, and updates the running
                    max, sum-of-exponentials, and weighted value accumulator.

                    Args:
                        kv_block_idx: Index of the current KV block.
                        online_softmax_carry: Tuple of (output_block, sum_exp_block,
                            max_score_block) representing the online softmax state.

                    Returns:
                        Updated (output_block, sum_exp_block, max_score_block).
                    """
                    output_block, sum_exp_block, max_score_block = online_softmax_carry

                    page_map_start = kv_block_idx * kvblocks
                    page_indices_for_block = jax.lax.dynamic_slice(
                        block_tables, (seq_idx, page_map_start), (1, kvblocks)
                    )
                    page_indices_for_kv_block = jnp.squeeze(page_indices_for_block, axis=0)

                    key_block_shape = (kvblocks * page_size, num_kv_heads, head_size)
                    key_block = kv_pages[page_indices_for_kv_block, :, 0::2, :].reshape(key_block_shape)
                    value_block = kv_pages[page_indices_for_kv_block, :, 1::2, :].reshape(key_block_shape)

                    kv_token_start_index = kv_block_idx * kv_tokens_per_block
                    kv_token_indices = base_k_ids + kv_token_start_index

                    attention_scores_block = (
                        jnp.einsum(
                            "bihd,kid->bihk",
                            query_block.astype(compute_dtype),
                            key_block.astype(compute_dtype),
                            optimize=True,
                        )
                        * softmax_scale
                    )
                    if logits_soft_cap is not None:
                        attention_scores_block = jnp.tanh(attention_scores_block / logits_soft_cap) * logits_soft_cap

                    causal_mask = jnp.expand_dims(query_token_indices, 1) >= jnp.expand_dims(kv_token_indices, 0)
                    if sliding_window is not None:
                        left_window = int(sliding_window) if isinstance(sliding_window, int) else int(sliding_window[0])
                        left_keep = jnp.expand_dims(kv_token_indices, 0) > jnp.expand_dims(
                            query_token_indices - left_window, 1
                        )
                        causal_mask = jnp.logical_and(causal_mask, left_keep)
                    kv_bound = jnp.expand_dims(kv_token_indices, 0) < kv_cache_len_for_seq
                    attention_mask = (causal_mask & kv_bound)[:, None, None, :]

                    attention_scores_block = jnp.where(attention_mask, attention_scores_block, -jnp.inf)

                    current_max = jnp.max(attention_scores_block, axis=3)
                    new_max = jnp.maximum(max_score_block, current_max)

                    probs = jnp.exp(attention_scores_block - jnp.expand_dims(new_max, axis=3))
                    probs = jnp.where(attention_mask, probs, 0.0)

                    rescale = jnp.exp(max_score_block - new_max)
                    sum_exp_block = (rescale * sum_exp_block) + jnp.sum(probs, axis=3)
                    value_update = jnp.einsum("bihk,kid->bihd", probs, value_block.astype(compute_dtype), optimize=True)
                    output_block = jnp.expand_dims(rescale, 3) * output_block + value_update

                    return output_block, sum_exp_block, new_max

                init_output_block = jnp.zeros((qblocks, num_kv_heads, q_heads_per_group, head_size), dtype=compute_dtype)
                if have_sinks:
                    init_sum_exp = jnp.ones((qblocks, num_kv_heads, q_heads_per_group), dtype=compute_dtype)
                    init_max = jnp.broadcast_to(
                        sinks_h[None, :, :].astype(compute_dtype),
                        (qblocks, num_kv_heads, q_heads_per_group),
                    )
                else:
                    init_sum_exp = jnp.zeros((qblocks, num_kv_heads, q_heads_per_group), dtype=compute_dtype)
                    init_max = jnp.full((qblocks, num_kv_heads, q_heads_per_group), -jnp.inf, dtype=compute_dtype)

                output_block, sum_exp_block, _max_block = jax.lax.fori_loop(
                    0,
                    num_kv_blocks,
                    _process_kv_block,
                    (init_output_block, init_sum_exp, init_max),
                )

                sum_exp_block = jnp.maximum(sum_exp_block, 1e-6)
                normalized_output_block = (output_block / jnp.expand_dims(sum_exp_block, axis=3)).astype(
                    padded_queries.dtype
                )

                normalized_output_block = normalized_output_block.astype(padded_queries.dtype)

                existing = jax.lax.dynamic_slice(
                    block_output_accumulator,
                    (q_global_start, 0, 0, 0),
                    (qblocks, num_kv_heads, q_heads_per_group, head_size),
                )
                merged = jnp.where(valid_queries[:, None, None, None], normalized_output_block, existing)

                return jax.lax.dynamic_update_slice(block_output_accumulator, merged, (q_global_start, 0, 0, 0))

            return jax.lax.fori_loop(0, num_query_blocks, _process_query_block, output_accumulator)

        return jax.lax.cond(
            num_queries_for_seq > 0,
            _process_sequence_with_queries,
            lambda: output_accumulator,
        )

    num_S = (num_seqs[0] if num_seqs.shape != () else num_seqs).astype(jnp.int32)

    return jax.lax.slice(
        jax.lax.fori_loop(0, num_S, _compute_attention_for_sequence, attention_output),
        (0, 0, 0, 0),
        (total_query_tokens, num_kv_heads, q_heads_per_group, head_size),
    ).reshape(out_shape)
