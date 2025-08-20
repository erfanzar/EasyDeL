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


import jax
import jax.numpy as jnp

from easydel.utils.compiling_utils import ejit


@ejit(static_argnums=(7, 8))
def _ragged_paged_attention_optimized(
    queries: jnp.ndarray,
    kv_pages: jnp.ndarray,  # [P, PS, 2*KVH, D] (K at 0::2, V at 1::2)
    context_lens: jnp.ndarray,
    block_tables: jnp.ndarray,  # [S, max_pages_per_sequence]
    query_start_loc: jnp.ndarray,  # [S+1]
    num_seqs: jnp.ndarray,  # [1] or scalar
    softmax_scale: float,
    soft_cap: float | None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    total_query_tokens, num_q_heads, head_size = queries.shape
    page_size = kv_pages.shape[1]
    num_kv_heads = kv_pages.shape[2] // 2
    max_pages_per_sequence = block_tables.shape[-1]
    out_shape = (total_query_tokens, num_q_heads, head_size)
    q_heads_per_group = num_q_heads // num_kv_heads

    queries = queries.reshape(total_query_tokens, num_kv_heads, q_heads_per_group, head_size) * softmax_scale
    queries = queries.astype(compute_dtype)

    qblocks = 8 if total_query_tokens > 256 else 4
    qblocks = min(qblocks, total_query_tokens if total_query_tokens > 0 else 4)
    kvblocks = 128 if max_pages_per_sequence > 256 else 64
    kvblocks = min(kvblocks, max_pages_per_sequence if max_pages_per_sequence > 0 else 64)

    padd = (qblocks - total_query_tokens % qblocks) % qblocks + qblocks
    padded_queries = jnp.pad(queries, ((0, padd), (0, 0), (0, 0), (0, 0)), mode="constant")

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
                        block_tables, (seq_idx, page_map_start_index), (1, kvblocks)
                    )
                    page_indices_for_kv_block = jnp.squeeze(page_indices_for_block, axis=0)

                    # Extract KV blocks correctly
                    kv_block_shape = (kvblocks * page_size, num_kv_heads, head_size)
                    kv_data = kv_pages[page_indices_for_kv_block]  # [kvblocks, page_size, 2*KVH, D]
                    key_block = kv_data[:, :, 0::2, :].reshape(kv_block_shape)
                    value_block = kv_data[:, :, 1::2, :].reshape(kv_block_shape)

                    kv_token_start_index = kv_block_idx * kv_tokens_per_block
                    kv_token_indices = jnp.arange(kvblocks * page_size, dtype=jnp.int32) + kv_token_start_index

                    # More efficient einsum with precision control
                    attention_scores_block = jnp.einsum("bihd,kid->bihk", query_block, key_block, optimize=True)
                    if soft_cap is not None:
                        attention_scores_block = soft_cap * jnp.tanh(attention_scores_block / soft_cap)

                    # Fused mask creation and application
                    attention_mask = (
                        (query_token_indices[:, None] >= kv_token_indices[None, :])
                        & (kv_token_indices[None, :] < kv_cache_len_for_seq)
                    )[:, None, None, :]

                    # Use -1e9 instead of -inf for better numerical stability
                    attention_scores_block = jnp.where(attention_mask, attention_scores_block, -1e9)

                    current_max_score = jnp.max(attention_scores_block, axis=3)
                    new_max_score_block = jnp.maximum(max_score_block, current_max_score)

                    # Fused exp and mask application
                    probabilities_block = jnp.exp(attention_scores_block - new_max_score_block[:, :, :, None])
                    probabilities_block = probabilities_block * attention_mask

                    rescale_factor = jnp.exp(max_score_block - new_max_score_block)
                    sum_exponentials_block = (rescale_factor * sum_exponentials_block) + jnp.sum(
                        probabilities_block, axis=3
                    )

                    # More efficient value aggregation
                    value_update = jnp.einsum("bihk,kid->bihd", probabilities_block, value_block, optimize=True)
                    output_block = rescale_factor[:, :, :, None] * output_block + value_update

                    return (
                        output_block.astype(compute_dtype),
                        sum_exponentials_block.astype(compute_dtype),
                        new_max_score_block.astype(compute_dtype),
                    )

                # Initialize with correct dtypes
                initial_carry = (
                    jnp.zeros((qblocks, num_kv_heads, q_heads_per_group, head_size), dtype=compute_dtype),
                    jnp.zeros((qblocks, num_kv_heads, q_heads_per_group), dtype=compute_dtype),
                    jnp.full((qblocks, num_kv_heads, q_heads_per_group), -1e9, dtype=compute_dtype),
                )

                output_block, sum_exponentials_block, _ = jax.lax.fori_loop(
                    0, num_kv_blocks, _process_kv_block, initial_carry
                )

                # More stable normalization
                normalized_output_block = (
                    output_block / jnp.maximum(sum_exponentials_block[:, :, :, None], 1e-10)
                ).astype(padded_queries.dtype)

                return jax.lax.dynamic_update_slice(
                    block_output_accumulator,
                    normalized_output_block,
                    (query_block_global_start, 0, 0, 0),
                )

            return jax.lax.fori_loop(0, num_query_blocks, _process_query_block, output_accumulator)

        return jax.lax.cond(
            num_queries_for_seq > 0,
            _process_sequence_with_queries,
            lambda: output_accumulator,
        )

    num_S = (num_seqs[0] if num_seqs.shape != () else num_seqs).astype(jnp.int32)

    return jax.lax.slice(
        jax.lax.fori_loop(
            0,
            num_S,
            _compute_attention_for_sequence,
            attention_output,
        ),
        (0, 0, 0, 0),
        (total_query_tokens, num_kv_heads, q_heads_per_group, head_size),
    ).reshape(out_shape)
