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

import jax.numpy as jnp

from ._forward_jax import _ragged_paged_attention
from ._forward_jax_optimized import _ragged_paged_attention_optimized


def ragged_paged_attention(
    queries: jnp.ndarray,
    kv_pages: jnp.ndarray,
    context_lens: jnp.ndarray,
    block_tables: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    num_seqs: jnp.ndarray,
    softmax_scale: float | None = None,
    soft_cap: float | None = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    optimized: bool = False,
) -> jnp.ndarray:
    """Performs paged attention for batched, ragged sequences.

    This function implements a FlashAttention-style algorithm to compute attention
    for multiple sequences of varying lengths. The Key-Value (KV) cache for these
    sequences is stored in non-contiguous memory blocks called "pages". This is
    a common technique in LLM inference servers to manage memory efficiently.

    The attention is computed by iterating through blocks of queries and, for each
    query block, iterating through the relevant blocks of key-value pages. An
    online softmax algorithm is used to compute the attention output in a single
    pass over the KV data, which is memory-efficient.

    Args:
        queries: The query tensor for all sequences, concatenated together.
            Shape: `[total_query_tokens, num_q_heads, head_size]`.
        kv_pages: The paged Key/value cache.
            Shape: `[num_pages, page_size, num_kv_heads_combined, head_size]`.
        context_lens: The total length of each sequence in the KV cache.
            Shape: `[num_seqs]`.
        block_tables: A map from each sequence to its list of
            physical page indices in the KV cache.
            Shape: `[num_seqs, max_pages_per_sequence]`.
        query_start_loc: The cumulative sum of query lengths for each
            sequence, used to index into the `queries` tensor.
            Shape: `[num_seqs + 1]`.
        num_seqs: The total number of sequences in the batch which should be a shape[1] int32.
        softmax_scale: The scaling factor to apply to the attention scores
            before the softmax operation (typically `1 / sqrt(head_size)`).
        soft_cap: An optional value to cap the attention scores with `tanh`.

    Returns:
        The attention output tensor, with the same shape and dtype as `queries`.
        Shape: `[total_query_tokens, num_q_heads, head_size]`.
    """
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5
    fn = _ragged_paged_attention_optimized if optimized else _ragged_paged_attention
    return fn(
        queries=queries,
        kv_pages=kv_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        softmax_scale=softmax_scale,
        soft_cap=soft_cap,
        compute_dtype=compute_dtype,
    )
