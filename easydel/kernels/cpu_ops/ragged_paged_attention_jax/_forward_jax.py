# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(7, 8))
def _ragged_paged_attention(
    queries: jnp.ndarray,
    key_pages: jnp.ndarray,
    value_pages: jnp.ndarray,
    context_lens: jnp.ndarray,
    block_tables: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    num_seqs: jnp.ndarray,
    softmax_scale: float,
    soft_cap: float | None,
) -> jnp.ndarray:
    total_query_tokens, num_q_heads, head_size = queries.shape
    page_size, num_kv_heads = value_pages.shape[1], value_pages.shape[2]
    max_pages_per_sequence = block_tables.shape[-1]
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
        num_queries_for_seq = query_start_loc[seq_idx + 1] - query_start_loc[seq_idx]

        def _process_sequence_with_queries():
            num_query_blocks = (num_queries_for_seq + qblocks - 1) // qblocks

            def _process_query_block(query_block_idx, block_output_accumulator):
                query_block_offset = query_block_idx * qblocks
                query_block_global_start = query_start_loc[seq_idx] + query_block_offset
                query_block = jax.lax.dynamic_slice(
                    padded_queries,
                    (query_block_global_start, 0, 0, 0),
                    (qblocks, num_kv_heads, q_heads_per_group, head_size),
                )
                kv_cache_len_for_seq = context_lens[seq_idx]
                query_block_start_token_idx = kv_cache_len_for_seq - num_queries_for_seq + query_block_offset
                query_token_indices = jnp.arange(qblocks, dtype=jnp.int32) + query_block_start_token_idx
                kv_tokens_per_block = page_size * kvblocks
                num_kv_blocks = (kv_cache_len_for_seq + kv_tokens_per_block - 1) // kv_tokens_per_block

                def _process_kv_block(kv_block_idx, online_softmax_carry):
                    output_block, sum_exponentials_block, max_score_block = online_softmax_carry
                    page_map_start_index = kv_block_idx * kvblocks
                    page_indices_for_block = jax.lax.dynamic_slice(
                        block_tables,
                        (seq_idx, page_map_start_index),
                        (1, kvblocks),
                    )
                    page_indices_for_kv_block = jnp.squeeze(page_indices_for_block, axis=0)
                    key_block_shape = (kvblocks * page_size, num_kv_heads, head_size)
                    key_block = key_pages[page_indices_for_kv_block, :, :, :].reshape(key_block_shape)
                    value_block = value_pages[page_indices_for_kv_block, :, :, :].reshape(key_block_shape)
                    kv_token_start_index = kv_block_idx * kv_tokens_per_block
                    kv_token_indices = jnp.arange(kvblocks * page_size, dtype=jnp.int32) + kv_token_start_index
                    attention_scores_block = jnp.einsum("bihd,kid->bihk", query_block, key_block, optimize=True)
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

                initial_output_block = jnp.zeros(
                    (qblocks, num_kv_heads, q_heads_per_group, head_size),
                    dtype=padded_queries.dtype,
                )
                initial_sum_exponentials = jnp.zeros(
                    (qblocks, num_kv_heads, q_heads_per_group),
                    dtype=padded_queries.dtype,
                )
                initial_max_score = jnp.full(
                    (qblocks, num_kv_heads, q_heads_per_group),
                    -jnp.inf,
                    dtype=padded_queries.dtype,
                )

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
                    (query_block_global_start, 0, 0, 0),
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
        jax.lax.fori_loop(0, num_seqs[0], _compute_attention_for_sequence, attention_output),
        (0, 0, 0, 0),
        (total_query_tokens, num_kv_heads, q_heads_per_group, head_size),
    ).reshape(out_shape)
