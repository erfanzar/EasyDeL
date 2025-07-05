from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(7, 8))
def _ragged_paged_attention(
    queries: jnp.ndarray,
    key_pages: jnp.ndarray,
    value_pages: jnp.ndarray,
    sequence_lengths: jnp.ndarray,
    sequence_page_indices: jnp.ndarray,
    cumulative_query_lengths: jnp.ndarray,
    num_sequences: jnp.ndarray,
    softmax_scale: float,
    soft_cap: float | None = None,
) -> jnp.ndarray:
    total_query_tokens, num_q_heads, head_size = queries.shape
    __, page_size, num_kv_heads, _ = value_pages.shape
    _, max_pages_per_sequence = sequence_page_indices.shape
    out_shape = (total_query_tokens, num_q_heads, head_size)
    q_heads_per_group = num_q_heads // num_kv_heads
    queries = queries.reshape(total_query_tokens, num_kv_heads, q_heads_per_group, head_size)
    qblocks = min(4, total_query_tokens if total_query_tokens > 0 else 4)
    kvblocks = min(32, max_pages_per_sequence if max_pages_per_sequence > 0 else 32)

    queries = queries * softmax_scale

    padd = (qblocks - total_query_tokens % qblocks) % qblocks + qblocks
    if padd > 0:
        padding_shape = (padd, num_kv_heads, q_heads_per_group, head_size)
        query_padding = jnp.zeros(padding_shape, dtype=queries.dtype)
        padded_queries = jnp.concatenate([queries, query_padding], axis=0)
    else:
        padded_queries = queries

    attention_output = jnp.zeros_like(padded_queries)

    def _compute_attention_for_sequence(seq_idx, output_accumulator):
        num_queries_for_seq = cumulative_query_lengths[seq_idx + 1] - cumulative_query_lengths[seq_idx]

        def _process_sequence_with_queries():
            num_query_blocks = (num_queries_for_seq + qblocks - 1) // qblocks

            def _process_query_block(query_block_idx, block_output_accumulator):
                query_block_offset = query_block_idx * qblocks
                query_block_global_start = cumulative_query_lengths[seq_idx] + query_block_offset

                query_block_slice_starts = (query_block_global_start, 0, 0, 0)
                query_block_shape = (qblocks, num_kv_heads, q_heads_per_group, head_size)
                query_block = jax.lax.dynamic_slice(padded_queries, query_block_slice_starts, query_block_shape)

                kv_cache_len_for_seq = sequence_lengths[seq_idx]
                query_block_start_token_idx = kv_cache_len_for_seq - num_queries_for_seq + query_block_offset
                query_token_indices = jnp.arange(qblocks, dtype=jnp.int32) + query_block_start_token_idx

                kv_tokens_per_block = page_size * kvblocks
                num_kv_blocks = (kv_cache_len_for_seq + kv_tokens_per_block - 1) // kv_tokens_per_block

                def _process_kv_block(kv_block_idx, online_softmax_carry):
                    output_block, sum_exponentials_block, max_score_block = online_softmax_carry

                    page_map_start_index = kv_block_idx * kvblocks
                    page_indices_for_block = jax.lax.dynamic_slice(
                        sequence_page_indices, (seq_idx, page_map_start_index), (1, kvblocks)
                    )
                    page_indices_for_kv_block = jnp.squeeze(page_indices_for_block, axis=0)

                    key_block_shape = (kvblocks * page_size, num_kv_heads, head_size)
                    key_block = key_pages[page_indices_for_kv_block, :, :, :].reshape(key_block_shape)
                    value_block = value_pages[page_indices_for_kv_block, :, :, :].reshape(key_block_shape)

                    kv_token_start_index = kv_block_idx * kv_tokens_per_block
                    kv_token_indices = jnp.arange(kvblocks * page_size, dtype=jnp.int32) + kv_token_start_index

                    attention_scores_block = jnp.einsum("bihd,kid->bihk", query_block, key_block)

                    if soft_cap is not None:
                        attention_scores_block = jnp.tanh(attention_scores_block / soft_cap) * soft_cap

                    causal_mask = jnp.expand_dims(query_token_indices, 1) >= jnp.expand_dims(kv_token_indices, 0)
                    kv_boundary_mask = jnp.expand_dims(kv_token_indices, 0) < kv_cache_len_for_seq
                    attention_mask = (causal_mask & kv_boundary_mask)[:, None, None, :]

                    attention_scores_block = jnp.where(attention_mask, attention_scores_block, -jnp.inf)

                    current_max_score = jnp.max(attention_scores_block, axis=3)
                    new_max_score_block = jnp.maximum(max_score_block, current_max_score)

                    probabilities_block = jnp.exp(attention_scores_block - jnp.expand_dims(new_max_score_block, axis=3))
                    probabilities_block = jnp.where(attention_mask, probabilities_block, 0.0)

                    rescale_factor = jnp.exp(max_score_block - new_max_score_block)

                    sum_exponentials_block = (rescale_factor * sum_exponentials_block) + jnp.sum(
                        probabilities_block, axis=3
                    )
                    value_update = jnp.einsum("bihk,kid->bihd", probabilities_block, value_block)
                    output_block = jnp.expand_dims(rescale_factor, 3) * output_block + value_update

                    return output_block, sum_exponentials_block, new_max_score_block

                initial_output_block = jnp.zeros(query_block_shape, dtype=padded_queries.dtype)
                initial_sum_exponentials = jnp.zeros((qblocks, num_kv_heads, q_heads_per_group), dtype=jnp.float32)
                initial_max_score = jnp.full((qblocks, num_kv_heads, q_heads_per_group), -jnp.inf, dtype=jnp.float32)

                output_block, sum_exponentials_block, _ = jax.lax.fori_loop(
                    0,
                    num_kv_blocks,
                    _process_kv_block,
                    (
                        initial_output_block,
                        initial_sum_exponentials,
                        initial_max_score,
                    ),
                )

                sum_exponentials_block = jnp.maximum(sum_exponentials_block, 1e-6)
                normalized_output_block = output_block / jnp.expand_dims(sum_exponentials_block, axis=3)

                return jax.lax.dynamic_update_slice(
                    block_output_accumulator,
                    normalized_output_block,
                    query_block_slice_starts,
                )

            return jax.lax.fori_loop(
                0,
                num_query_blocks,
                _process_query_block,
                output_accumulator,
            )

        return jax.lax.cond(
            num_queries_for_seq > 0,
            _process_sequence_with_queries,
            lambda: output_accumulator,
        )

    return jax.lax.slice(
        jax.lax.fori_loop(0, num_sequences[0], _compute_attention_for_sequence, attention_output),
        (0, 0, 0, 0),
        (total_query_tokens, num_kv_heads, q_heads_per_group, head_size),
    ).reshape(out_shape)
