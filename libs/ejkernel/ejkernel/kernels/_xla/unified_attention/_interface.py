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

"""Unified attention interface for paged KV cache with mixed workloads.

This module provides an XLA implementation of unified attention that handles
ragged (variable-length) batches with paged key-value caches. It is designed
for inference scenarios where multiple sequences of different lengths are
processed together with a shared paged KV cache.

Unified attention combines features needed for modern LLM serving:
1. Paged KV cache: Memory-efficient storage with block-level allocation
2. Ragged batches: Variable-length sequences in a single batch
3. Continuous batching: Mix of prefill and decode requests
4. ALiBi position encoding: Linear bias for position-aware attention

This implementation is particularly suited for:
- vLLM-style inference servers
- Continuous batching systems
- Memory-efficient long-context inference

Key features:
1. Block-tabled KV cache: O(1) memory allocation/deallocation per block
2. Per-sequence context tracking: Independent sequence lengths
3. Query-query attention: Optional self-attention within query tokens
4. ALiBi slopes: Built-in support for ALiBi positional encoding
5. Attention sinks: Auxiliary logits for probability mass absorption
6. Sliding window: Local attention for long-context efficiency

Memory layout:
- Queries: Packed tokens [total_tokens, num_q_heads, head_dim]
- KV cache: Paged blocks [num_blocks, block_size, num_kv_heads, head_dim]
- Block tables: Logical-to-physical mapping [num_seqs, max_blocks_per_seq]

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.unified_attention import unified_attention
    >>>
    >>> # Two sequences with different lengths
    >>> total_tokens = 10  # 4 tokens from seq 0, 6 from seq 1
    >>> queries = jnp.ones((total_tokens, 8, 64))  # [total, heads, dim]
    >>>
    >>> # Paged KV cache
    >>> key_cache = jnp.ones((100, 16, 8, 64))   # [blocks, block_size, heads, dim]
    >>> value_cache = jnp.ones((100, 16, 8, 64))
    >>>
    >>> # Per-sequence metadata
    >>> kv_lens = jnp.array([128, 256], dtype=jnp.int32)
    >>> block_tables = jnp.array([[0, 1, 2, 3, -1], [4, 5, 6, 7, 8]], dtype=jnp.int32)
    >>> query_start_loc = jnp.array([0, 4, 10], dtype=jnp.int32)
    >>>
    >>> output = unified_attention(
    ...     queries, key_cache, value_cache,
    ...     kv_lens, block_tables, query_start_loc
    ... )
    >>> output.shape
    (10, 8, 64)

Note:
    This XLA implementation is a reference fallback. For production GPU
    workloads, prefer the Triton implementation for better performance.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _unified_attention_fwd


@kernel_registry.register("unified_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def unified_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    block_dim: int = 128,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Compute unified attention for ragged batches with paged KV cache.

    This function handles variable-length sequences packed together, where
    each sequence has its KV cache stored in non-contiguous pages. It supports
    mixed prefill and decode operations in a single batch.

    Args:
        queries: Packed query tokens [total_tokens, num_q_heads, head_dim].
            All queries from all sequences concatenated together.
        key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim].
            Physical storage for all KV cache blocks.
        value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim].
            Must have same layout as key_cache.
        kv_lens: Context length for each sequence [num_seqs].
            Number of valid tokens in KV cache per sequence.
        block_tables: Block table mapping [num_seqs, max_blocks_per_seq].
            Maps logical block indices to physical block indices for each sequence.
            Use -1 for invalid/padding entries.
        query_start_loc: Cumulative query token counts [num_seqs + 1].
            query_start_loc[i] gives the starting index in queries for sequence i.
            query_start_loc[num_seqs] equals total_tokens.
        alibi_slopes: Optional ALiBi slopes [num_q_heads].
            Per-head linear bias for position encoding.
        qq_bias: Optional query-query attention bias [num_query_tokens, num_query_tokens].
            Additive bias for attention between query tokens (prefill self-attention).
        softmax_aux: Optional attention sink logits [num_q_heads].
            Per-head auxiliary logits that participate in softmax normalization.
        softmax_scale: Scaling factor for QK^T. Defaults to 1/sqrt(head_dim).
        causal: Whether to apply causal masking. Default True.
            Currently only causal attention is supported.
        sliding_window: Optional sliding window size for local attention.
            If set, queries can only attend to the last `sliding_window` KV tokens.
        logits_soft_cap: Optional soft cap for attention logits via tanh.
        seq_threshold_3d: Unused in XLA backend (Triton-specific optimization).
        num_par_softmax_segments: Unused in XLA backend (Triton-specific).
        block_dim: Unused in XLA backend (CUDA-specific).
        num_warps: Unused in XLA backend (Triton-specific).
        num_stages: Unused in XLA backend (Triton-specific).

    Returns:
        Attention output [total_tokens, num_q_heads, head_dim].
        Same packed layout as input queries.

    Raises:
        NotImplementedError: If causal=False (only causal attention supported).

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> # Setup for 2 sequences
        >>> queries = jnp.ones((10, 8, 64))  # 10 total tokens
        >>> key_cache = jnp.ones((50, 16, 8, 64))
        >>> value_cache = jnp.ones((50, 16, 8, 64))
        >>> kv_lens = jnp.array([100, 200], dtype=jnp.int32)
        >>> block_tables = jnp.zeros((2, 10), dtype=jnp.int32)  # Simplified
        >>> query_start_loc = jnp.array([0, 4, 10], dtype=jnp.int32)
        >>>
        >>> output = unified_attention(
        ...     queries, key_cache, value_cache,
        ...     kv_lens, block_tables, query_start_loc
        ... )
        >>> output.shape
        (10, 8, 64)

    Note:
        - GQA/MQA is supported: num_q_heads can be a multiple of num_kv_heads
        - Block size must match between key_cache and value_cache
        - For GPU production use, prefer the Triton implementation
    """
    del seq_threshold_3d, num_par_softmax_segments, num_warps, num_stages
    if not causal:
        raise NotImplementedError("unified_attention (XLA) only supports causal attention.")

    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5

    return _unified_attention_fwd(
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        softmax_scale=float(softmax_scale),
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        softmax_aux=softmax_aux,
    )
