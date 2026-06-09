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

"""Ragged paged attention v3 interface for mixed prefill and decode.

This module provides the third-generation ragged paged attention implementation
that combines KV cache updates with attention computation in a single fused
operation. It is designed for high-performance inference serving where prefill
and decode requests are batched together.

Key advantages over v2:
1. Fused KV cache update: Writes new K/V tokens while computing attention
2. Optimized for mixed workloads: Handles both prefill (many queries) and
   decode (single query) efficiently
3. Quantization support: Optional Q/K/V scaling for INT8/FP8 inference
4. Chunked prefill: Breaks long prefills into chunks for memory efficiency

This implementation is particularly suited for:
- vLLM-style inference with continuous batching
- Mixed prefill + decode batches
- Long-context inference with chunked prefill
- Quantized inference (INT8/FP8)

Memory layout:
- Queries/Keys/Values: Packed tokens [total_tokens, num_heads, head_dim]
- KV cache: Paged blocks with packed K/V layout
- Distribution: Tensor describing prefill/decode/chunked-prefill distribution

The KV cache uses a special packed layout where keys and values are
interleaved within each page for cache-friendly access patterns.

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.ragged_page_attention_v3 import ragged_page_attention_v3
    >>>
    >>> # Mixed batch: 1 prefill (100 tokens), 2 decodes (1 token each)
    >>> total_tokens = 102
    >>> queries = jnp.ones((total_tokens, 8, 64))
    >>> keys = jnp.ones((total_tokens, 8, 64))
    >>> values = jnp.ones((total_tokens, 8, 64))
    >>>
    >>> # KV cache and metadata
    >>> kv_cache = jnp.zeros((1000, 16, 16, 1, 64))  # Packed layout
    >>> kv_lens = jnp.array([100, 50, 75], dtype=jnp.int32)
    >>> block_tables = jnp.zeros((30,), dtype=jnp.int32)
    >>> query_start_loc = jnp.array([0, 100, 101, 102], dtype=jnp.int32)
    >>> distribution = jnp.array([1, 2, 0], dtype=jnp.int32)  # 1 prefill, 2 decodes
    >>>
    >>> output, updated_cache = ragged_page_attention_v3(
    ...     queries, keys, values, kv_cache,
    ...     kv_lens, block_tables, query_start_loc, distribution
    ... )

Note:
    This XLA implementation is a reference fallback. For TPU production
    workloads, prefer the Pallas implementation for better performance.
"""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_paged_attention as _ragged_paged_attention


@kernel_registry.register("ragged_page_attention_v3", Platform.XLA, Backend.ANY)
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
    """Compute ragged paged attention with fused KV cache update.

    This function handles mixed prefill and decode workloads, updating the
    KV cache with new keys/values while simultaneously computing attention.

    Args:
        queries: Packed query tokens [total_tokens, num_q_heads, head_dim].
            All queries from all sequences concatenated together.
        keys: Packed key tokens [total_tokens, num_kv_heads, head_dim].
            New keys to be inserted into the KV cache.
        values: Packed value tokens [total_tokens, num_kv_heads, head_dim].
            New values to be inserted into the KV cache.
        kv_cache: Paged KV cache with packed layout
            [num_pages, page_size, num_kv_heads*2/kv_packing, kv_packing, head_dim_padded].
            Keys and values are interleaved within pages.
        kv_lens: KV context length for each sequence [max_num_seqs].
            Only the first num_seqs values are used.
        block_tables: Flattened block table [max_num_seqs * pages_per_seq].
            Maps logical block indices to physical page indices.
        query_start_loc: Cumulative query counts [max_num_seqs + 1].
            query_start_loc[i] gives starting index in queries for sequence i.
        distribution: Batch distribution descriptor [3].
            [num_prefill_seqs, num_decode_seqs, num_chunked_prefill_seqs].
            Describes how sequences are partitioned by type.
        softmax_aux: Optional attention sink logits [num_q_heads].
            Per-head auxiliary logits for softmax normalization.
        softmax_scale: Scaling factor for QK^T. Default 1.0.
        sliding_window: Optional sliding window size for local attention.
        logits_soft_cap: Optional soft cap for attention logits via tanh.
        q_scale: Optional scale factor for quantized queries.
        k_scale: Optional scale factor for quantized keys.
        v_scale: Optional scale factor for quantized values.
        chunk_prefill_size: Optional chunk size for chunked prefill.
            If None, processes entire prefill at once.
        num_kv_pages_per_block: Unused in XLA backend (Pallas-specific).
        num_queries_per_block: Unused in XLA backend (Pallas-specific).
        vmem_limit_bytes: Unused in XLA backend (Pallas-specific).

    Returns:
        Tuple of (attention_output, updated_kv_cache):
            - attention_output: [total_tokens, num_q_heads, head_dim]
            - updated_kv_cache: Same shape as input kv_cache, with new K/V written

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> # Single decode request
        >>> queries = jnp.ones((1, 8, 64))
        >>> keys = jnp.ones((1, 8, 64))
        >>> values = jnp.ones((1, 8, 64))
        >>> kv_cache = jnp.zeros((100, 16, 16, 1, 64))
        >>> kv_lens = jnp.array([50], dtype=jnp.int32)
        >>> block_tables = jnp.arange(10, dtype=jnp.int32)
        >>> query_start_loc = jnp.array([0, 1], dtype=jnp.int32)
        >>> distribution = jnp.array([0, 1, 0], dtype=jnp.int32)
        >>>
        >>> output, new_cache = ragged_page_attention_v3(
        ...     queries, keys, values, kv_cache,
        ...     kv_lens, block_tables, query_start_loc, distribution
        ... )

    Note:
        The distribution tensor is critical for proper operation:
        - distribution[0]: Number of full prefill sequences
        - distribution[1]: Number of decode sequences (single token)
        - distribution[2]: Number of chunked prefill sequences
    """
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5

    return _ragged_paged_attention(
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
