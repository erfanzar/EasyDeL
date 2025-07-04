import jax.numpy as jnp

from ._forward_jax import _ragged_paged_attention


def ragged_paged_attention(
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
        key_pages: The paged Key cache.
            Shape: `[num_pages, page_size, num_kv_heads, head_size]`.
        value_pages: The paged Value cache.
            Shape: `[num_pages, page_size, num_kv_heads, head_size]`.
        sequence_lengths: The total length of each sequence in the KV cache.
            Shape: `[num_sequences]`.
        sequence_page_indices: A map from each sequence to its list of
            physical page indices in the KV cache.
            Shape: `[num_sequences, max_pages_per_sequence]`.
        cumulative_query_lengths: The cumulative sum of query lengths for each
            sequence, used to index into the `queries` tensor.
            Shape: `[num_sequences + 1]`.
        num_sequences: The total number of sequences in the batch which should be a scalar int32.
        softmax_scale: The scaling factor to apply to the attention scores
            before the softmax operation (typically `1 / sqrt(head_size)`).
        soft_cap: An optional value to cap the attention scores with `tanh`.

    Returns:
        The attention output tensor, with the same shape and dtype as `queries`.
        Shape: `[total_query_tokens, num_q_heads, head_size]`.
    """
    return _ragged_paged_attention(
        queries=queries,
        key_pages=key_pages,
        value_pages=value_pages,
        sequence_lengths=sequence_lengths,
        sequence_page_indices=sequence_page_indices,
        cumulative_query_lengths=cumulative_query_lengths,
        num_sequences=num_sequences,
        softmax_scale=softmax_scale,
        soft_cap=soft_cap,
    )
