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

"""Ragged Paged Attention V3 implementation using Triton kernels.

This module implements an advanced version of ragged paged attention with
integrated KV cache update capabilities. Unlike v2 which operates on
pre-populated KV caches, v3 simultaneously updates the cache with new
key-value pairs while computing attention, making it ideal for chunked
prefill and continuous batching scenarios.

Key differences from ragged_page_attention_v2:
1. **Integrated cache update**: Accepts new keys/values to insert into the cache
2. **Distribution control**: Explicit control over decode vs prefill distribution
3. **Quantization support**: Optional q_scale, k_scale, v_scale for int8 inference
4. **Chunked prefill**: Native support for processing long prompts in chunks

The implementation handles three types of sequences in a single batch:
- Decode sequences: Single new token, attending to cached context
- Short prefill: Multiple tokens processed together
- Long prefill: Tokens processed in chunks for memory efficiency

Core inputs:
- `queries/keys/values`: New tokens to process and cache
- `kv_cache`: Paged cache with special packing format
- `distribution`: [num_decode, num_prefill, num_chunked] distribution

The cache format uses a packed layout:
[num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded]

This enables efficient memory access patterns for mixed precision and
vectorized operations on GPU.

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.ragged_page_attention_v3 import ragged_page_attention_v3
    >>>
    >>> # New tokens to process
    >>> total_tokens = 20
    >>> num_q_heads, num_kv_heads, head_dim = 8, 8, 64
    >>> queries = jnp.ones((total_tokens, num_q_heads, head_dim))
    >>> keys = jnp.ones((total_tokens, num_kv_heads, head_dim))
    >>> values = jnp.ones((total_tokens, num_kv_heads, head_dim))
    >>>
    >>> # Paged KV cache
    >>> num_pages, page_size = 100, 16
    >>> kv_cache = jnp.zeros((num_pages, page_size, 16, 1, 64))
    >>>
    >>> # Sequence metadata
    >>> num_seqs = 3
    >>> kv_lens = jnp.array([50, 80, 30])
    >>> block_tables = jnp.zeros((num_seqs * 10,), dtype=jnp.int32)
    >>> query_start_loc = jnp.array([0, 5, 12, 20])
    >>> distribution = jnp.array([1, 1, 1])  # 1 decode, 1 prefill, 1 chunked
    >>>
    >>> output, updated_cache = ragged_page_attention_v3(
    ...     queries, keys, values, kv_cache,
    ...     kv_lens, block_tables, query_start_loc, distribution
    ... )

Reference:
    vLLM: Efficient Memory Management for Large Language Model Serving
    https://arxiv.org/abs/2309.06180
"""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import ragged_paged_attention_triton


@kernel_registry.register("ragged_page_attention_v3", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v3(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Compute ragged paged attention with integrated KV cache update.

    Performs paged attention on packed variable-length sequences while simultaneously
    updating the KV cache with new key-value pairs. This is designed for continuous
    batching scenarios where decode and prefill sequences are processed together.

    The function handles three types of sequences based on the distribution parameter:
    decode (single token), short prefill (full prompt), and chunked prefill (partial).
    Each type is processed with optimized kernel paths.

    Args:
        queries: Packed query tensor of shape (total_tokens, num_q_heads, head_dim).
            Contains new query tokens to process.
        keys: Packed key tensor of shape (total_tokens, num_kv_heads, head_dim).
            New keys to insert into the cache.
        values: Packed value tensor of shape (total_tokens, num_kv_heads, head_dim).
            New values to insert into the cache.
        kv_cache: Paged KV cache with packed format
            (num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded).
            Keys and values are interleaved in the head dimension.
        kv_lens: KV lengths per sequence (max_num_seqs,). Indicates total context
            length including both cached and new tokens.
        block_tables: Flattened page table mapping (max_num_seqs * pages_per_seq,).
            Maps logical pages to physical page indices in kv_cache.
        query_start_loc: Cumulative query offsets (max_num_seqs + 1,). Indicates
            where each sequence's queries start in the packed queries tensor.
        distribution: Sequence type distribution (3,). Format: [num_decode, num_prefill,
            num_chunked] indicating how sequences are distributed across types.
        softmax_aux: Optional attention sink values (num_q_heads,). Contributes
            to softmax normalizer for streaming attention patterns.
        softmax_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        sliding_window: Optional sliding window size for local attention. Limits
            context to the last N tokens.
        logits_soft_cap: Optional soft capping value for attention logits. Applies
            tanh-based soft capping for numerical stability.
        q_scale: Optional quantization scale for queries (int8 inference).
        k_scale: Optional quantization scale for keys (int8 inference).
        v_scale: Optional quantization scale for values (int8 inference).
        chunk_prefill_size: Size of chunks for long prefill sequences. Controls
            memory-compute tradeoff for very long prompts.
        num_kv_pages_per_block: Number of KV pages per compute block. Higher values
            may improve throughput at cost of memory.
        num_queries_per_block: Number of query tokens per compute block (``BLOCK_M``).
            Defaults to 128 when None.
        vmem_limit_bytes: Ignored; accepted for API compatibility (TPU-specific).

    Returns:
        Tuple of (attention_output, updated_kv_cache):
            - attention_output: Shape (total_tokens, num_q_heads, head_dim)
            - updated_kv_cache: Updated cache with new keys/values inserted

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.ragged_page_attention_v3 import ragged_page_attention_v3
        >>>
        >>> # Process mixed decode + prefill batch
        >>> queries = jnp.ones((10, 8, 64))  # 10 total tokens
        >>> keys = jnp.ones((10, 8, 64))
        >>> values = jnp.ones((10, 8, 64))
        >>> kv_cache = jnp.zeros((50, 16, 16, 1, 64))  # 50 pages
        >>>
        >>> kv_lens = jnp.array([100, 50])  # 2 sequences
        >>> block_tables = jnp.arange(20, dtype=jnp.int32)  # Page mapping
        >>> query_start_loc = jnp.array([0, 1, 10])  # 1 decode token, 9 prefill tokens
        >>> distribution = jnp.array([1, 1, 0])  # 1 decode, 1 prefill, 0 chunked
        >>>
        >>> output, cache = ragged_page_attention_v3(
        ...     queries, keys, values, kv_cache,
        ...     kv_lens, block_tables, query_start_loc, distribution
        ... )
    """
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5
    return ragged_paged_attention_triton(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
