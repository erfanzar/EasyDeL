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

"""vLLM-style unified (paged) attention implemented in Triton.

This is a JAX/Triton port of vLLM's `triton_unified_attention.py`, adapted to
ejkernel's `triton_call` interface.

Core inputs:
- `queries`: packed ragged queries, shape `[total_tokens, num_q_heads, head_dim]`
- `key_cache`/`value_cache`: paged KV cache, shape `[num_blocks, block_size, num_kv_heads, head_dim]`
- `query_start_loc`: cumulative query offsets, shape `[num_seqs + 1]` (int32)
- `kv_lens`: KV lengths per sequence, shape `[num_seqs]` (int32)
- `block_tables`: mapping `[num_seqs, max_blocks_per_seq]` (int32)

Supported features (inference-only):
- causal masking (required)
- optional sliding window via `sliding_window` (window length)
- optional logit softcap (`logits_soft_cap`)
- optional attention sink (`softmax_aux`): contributes to softmax normalizer only
- optional ALiBi slopes (`alibi_slopes`)
- optional query-query bias (`qq_bias`) for TreeAttention-like decode
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import unified_attention_triton


@kernel_registry.register("unified_attention", Platform.TRITON, Backend.GPU)
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
    """Compute vLLM-style unified paged attention using Triton kernels.

    Performs efficient paged attention on packed ragged queries, where multiple
    variable-length sequences are concatenated into a single tensor. This is
    optimized for inference workloads with paged KV caches.

    The function processes packed queries against block-organized KV caches,
    with each sequence's KV pages mapped via block_tables. This enables efficient
    memory utilization for serving large language models.

    Args:
        queries: Packed query tensor of shape (total_tokens, num_q_heads, head_dim).
            Queries from all sequences are concatenated together.
        key_cache: Paged key cache of shape (num_blocks, block_size, num_kv_heads, head_dim).
            Keys are stored in fixed-size blocks for efficient memory management.
        value_cache: Paged value cache of shape (num_blocks, block_size, num_kv_heads, head_dim).
            Values follow the same organization as keys.
        kv_lens: KV lengths for each sequence, shape (num_seqs,). Indicates how many
            KV tokens are valid for each sequence.
        block_tables: Page table mapping logical blocks to physical blocks,
            shape (num_seqs, max_blocks_per_seq).
        query_start_loc: Cumulative query offsets, shape (num_seqs + 1,). Indicates
            where each sequence's queries start in the packed queries tensor.
        alibi_slopes: Optional ALiBi positional bias slopes, shape (num_q_heads,).
            When provided, applies linear position biases for attention.
        qq_bias: Optional query-query bias for TreeAttention-like decode,
            shape (num_query_tokens, num_query_tokens). Enables tree-based
            speculative decoding patterns.
        softmax_aux: Optional attention sink values, shape (num_q_heads,).
            Contributes to softmax normalizer only, useful for streaming attention.
        softmax_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        causal: Whether to apply causal masking (default: True). Must be True
            for this implementation.
        sliding_window: Optional sliding window size for local attention. If specified,
            each query only attends to the last `sliding_window` tokens.
        logits_soft_cap: Optional soft capping value for attention logits. When specified,
            applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
        seq_threshold_3d: Optional sequence threshold for 3D kernel variant. Controls
            when to use 3D grid parallelization.
        num_par_softmax_segments: Optional number of parallel softmax segments for
            improved parallelization on long sequences.
        block_dim: Accepted for API compatibility with CUDA; ignored by Triton.
        num_warps: Number of Triton warps for kernel execution. If None, uses default.
        num_stages: Number of Triton pipeline stages. If None, uses default.

    Returns:
        Attention output of shape (total_tokens, num_q_heads, head_dim), with
        results packed in the same order as the input queries.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.unified_attention import unified_attention
        >>>
        >>> # Packed queries from 3 sequences
        >>> total_tokens = 15
        >>> num_q_heads, head_dim = 8, 64
        >>> queries = jnp.ones((total_tokens, num_q_heads, head_dim))
        >>>
        >>> # Paged KV cache
        >>> num_blocks, block_size, num_kv_heads = 50, 16, 8
        >>> key_cache = jnp.ones((num_blocks, block_size, num_kv_heads, head_dim))
        >>> value_cache = jnp.ones((num_blocks, block_size, num_kv_heads, head_dim))
        >>>
        >>> # Sequence metadata
        >>> kv_lens = jnp.array([20, 35, 15])  # KV lengths per sequence
        >>> query_start_loc = jnp.array([0, 5, 10, 15])  # Query boundaries
        >>> block_tables = jnp.zeros((3, 5), dtype=jnp.int32)  # Page tables
        >>>
        >>> output = unified_attention(
        ...     queries, key_cache, value_cache,
        ...     kv_lens, block_tables, query_start_loc
        ... )
        >>> print(output.shape)  # (15, 8, 64)
    """
    return unified_attention_triton(
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        kv_lens=kv_lens,
        query_start_loc=query_start_loc,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        softmax_aux=softmax_aux,
        num_warps=num_warps,
        num_stages=num_stages,
    )
