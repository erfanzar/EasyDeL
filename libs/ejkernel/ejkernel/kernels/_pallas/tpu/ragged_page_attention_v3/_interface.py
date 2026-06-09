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


"""Ragged Paged Attention V3 for TPU with KV cache write and quantization support.

This module provides the third generation of ragged paged attention for TPU,
featuring simultaneous KV cache updates during attention computation. Unlike
v2 which only reads from the KV cache, v3 can write new keys and values while
computing attention, enabling true continuous batching.

Ragged paged attention v3 is essential for:
1. Single-pass prefill with KV cache population
2. Continuous batching with online KV cache updates
3. Quantized inference with scaling factors
4. Optimized kernels for different head dimensions (64 and 128)

Key Features:
    - KV cache write: Updates cache during attention computation
    - Dual head dimension support: Optimized kernels for head_dim=64 and 128
    - Quantization support: q_scale, k_scale, v_scale for quantized inference
    - Distribution tensor: Dynamic batch composition information
    - Chunked prefill: Configurable prefill chunk sizes
    - Attention sinks: Streaming inference support
    - Sliding window: Local attention for long sequences

Improvements over V2:
    - Simultaneous attention compute and KV cache write
    - Separate key/value inputs for new tokens
    - Distribution tensor for dynamic workload balancing
    - Better support for quantized models
    - Specialized kernels for head_dim=64 vs 128

Data Layout:
    - queries: New query tokens [total_tokens, num_q_heads, head_dim]
    - keys: New key tokens [total_tokens, num_kv_heads, head_dim]
    - values: New value tokens [total_tokens, num_kv_heads, head_dim]
    - kv_cache: Existing cache [num_pages, page_size, num_kv_heads*2/packing, packing, head_dim]
    - distribution: Batch composition info [3] (prefill/decode/total counts)

Returns:
    Tuple of:
        - Attention output [total_tokens, num_q_heads, head_dim]
        - Updated KV cache with new tokens written

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3 import ragged_page_attention_v3
    >>>
    >>> # Process new tokens and update cache
    >>> total_tokens = 22
    >>> num_heads, head_dim = 32, 128
    >>>
    >>> queries = jnp.ones((total_tokens, num_heads, head_dim))
    >>> keys = jnp.ones((total_tokens, num_heads, head_dim))
    >>> values = jnp.ones((total_tokens, num_heads, head_dim))
    >>> kv_cache = jnp.ones((256, 16, num_heads, 2, head_dim))
    >>> kv_lens = jnp.array([100, 200, 50, 150])
    >>> block_tables = jnp.zeros((64,), dtype=jnp.int32)
    >>> query_start_loc = jnp.array([0, 10, 20, 21, 22])
    >>> distribution = jnp.array([2, 2, 4])  # 2 prefill, 2 decode, 4 total
    >>>
    >>> output, updated_cache = ragged_page_attention_v3(
    ...     queries, keys, values, kv_cache, kv_lens, block_tables, query_start_loc, distribution
    ... )
"""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import ragged_paged_attention as _128_ragged_paged_attention
from ._pallas_impl_fwd_h64 import ragged_paged_attention as _64_ragged_paged_attention


@kernel_registry.register("ragged_page_attention_v3", Platform.PALLAS, Backend.TPU)
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
    """Ragged paged attention with KV cache write for mixed prefill and decode.

    This kernel processes queries, keys, and values together, simultaneously
    computing attention and updating the KV cache. This enables single-pass
    prefill operations and efficient continuous batching.

    Args:
        queries: Concatenated query tokens [total_tokens, num_q_heads, head_dim].
        keys: Concatenated key tokens [total_tokens, num_kv_heads, head_dim]
            to be written to cache and used for attention.
        values: Concatenated value tokens [total_tokens, num_kv_heads, head_dim]
            to be written to cache and used for attention.
        kv_cache: Existing KV cache [num_pages, page_size, num_kv_heads*2/packing,
            packing, head_dim]. Keys and values are stored with packing factor.
        kv_lens: KV context lengths per sequence [max_num_seqs]. Specifies the
            existing KV length for each sequence before this operation.
        block_tables: Flattened page table [max_num_seqs * pages_per_seq]. Maps
            logical pages to physical page indices.
        query_start_loc: Cumulative query positions [max_num_seqs+1]. Position i
            gives the starting index in queries for sequence i.
        distribution: Batch composition info [3] containing:
            - [0]: Number of prefill sequences
            - [1]: Number of decode sequences
            - [2]: Total number of sequences
        softmax_aux: Optional attention sink logits [num_q_heads] for streaming.
        softmax_scale: Scaling factor for QK^T (default: 1/sqrt(head_dim)).
        sliding_window: Size of sliding window for local attention.
        logits_soft_cap: Optional soft cap value for attention logits.
        q_scale: Quantization scale for queries (for quantized models).
        k_scale: Quantization scale for keys (for quantized models).
        v_scale: Quantization scale for values (for quantized models).
        chunk_prefill_size: Chunk size for prefill processing.
        num_kv_pages_per_block: Number of KV pages per compute block in kernel.
        num_queries_per_block: Number of queries per compute block in kernel.
        vmem_limit_bytes: VMEM budget limit for the Pallas kernel.

    Returns:
        Tuple of:
            - Attention output [total_tokens, num_q_heads, head_dim]
            - Updated KV cache with new keys/values written

    Note:
        - Automatically selects optimized kernel based on head_dim (64 or 128)
        - New keys and values are written to cache at positions [kv_lens[i]:kv_lens[i]+num_new_tokens]
        - Supports quantized inference via q_scale, k_scale, v_scale parameters
    """
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5
    if queries.shape[-1] == 64:
        return _64_ragged_paged_attention(
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
    return _128_ragged_paged_attention(
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
