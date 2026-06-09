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


"""Ragged Paged Attention implementation using Triton kernels.

This module implements ragged paged attention, an extension of paged attention
that efficiently handles batches of sequences with highly variable lengths.
Unlike standard paged attention which processes one query per sequence, ragged
paged attention processes multiple queries per sequence in a single batch,
making it ideal for prefill operations during LLM inference.

Key differences from standard page_attention:
1. **Ragged queries**: Multiple queries per sequence packed into a single tensor
2. **Query-level granularity**: Each query token can attend to the appropriate
   portion of the KV cache based on its position
3. **Prefill-optimized**: Designed for processing prompt tokens efficiently
4. **Combined KV format**: Keys and values are interleaved in memory

The "ragged" nature refers to handling variable-length sequences in a packed
format, where query_start_loc indicates the boundaries between sequences.

Architecture:
- Queries from multiple sequences are concatenated: [seq0_queries, seq1_queries, ...]
- Each query knows its position within its sequence via metadata
- KV cache is organized in pages, with each page containing both K and V
- Block tables map logical pages to physical pages for each sequence

Use cases:
- Prefill phase: Processing entire prompts before generation
- Chunked prefill: Processing long prompts in multiple passes
- Variable-length batching: Efficiently batching requests of different lengths

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.ragged_page_attention_v2 import ragged_page_attention_v2
    >>>
    >>>
    >>> total_tokens = 16
    >>> num_q_heads, head_dim = 12, 64
    >>> queries = jnp.ones((total_tokens, num_q_heads, head_dim))
    >>>
    >>>
    >>> num_pages, page_size, num_kv_heads = 50, 16, 12
    >>> kv_pages = jnp.ones((num_pages, page_size, 2 * num_kv_heads, head_dim))
    >>>
    >>>
    >>> num_seqs = 3
    >>> context_lens = jnp.array([5, 8, 3])
    >>> query_start_loc = jnp.array([0, 5, 13, 16])
    >>> block_tables = jnp.zeros((num_seqs, 10), dtype=jnp.int32)
    >>>
    >>> output = ragged_page_attention_v2(
    ...     queries, kv_pages, context_lens, block_tables,
    ...     query_start_loc, num_seqs
    ... )
    >>> print(output.shape)

Reference:
    vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
    https://arxiv.org/abs/2309.06180
"""

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import ragged_paged_attention_triton_call, ragged_paged_attention_triton_call_qblock

DEFAULT_MASK_VALUE = -2.381976426469702e38


@kernel_registry.register("ragged_page_attention_v2", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v2(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "num_seqs_plus_one"],
    num_seqs: Array | int,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    optimized: bool = False,
    sliding_window: int | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Compute ragged paged attention for variable-length sequences.

    This function efficiently processes multiple variable-length sequences in a
    single batch, where queries from all sequences are packed into a flat tensor.
    It's particularly useful for the prefill phase of LLM inference where entire
    prompts of varying lengths need to be processed.

    The KV cache is organized in pages, with keys and values interleaved in the
    same tensor (combined format). Each sequence can span multiple pages, and
    the block_tables parameter maps logical page indices to physical page locations.

    Args:
        queries: Packed query tensor of shape (total_tokens, num_q_heads, head_dim),
            where total_tokens is the sum of all sequence lengths. Queries from
            different sequences are concatenated.
        kv_pages: Paged KV cache of shape (num_pages, page_size, num_combined_kv_heads, head_dim),
            where num_combined_kv_heads = 2 * num_kv_heads (keys and values interleaved).
            The first half of the head dimension contains keys, the second half values.
        context_lens: Context length for each sequence, shape (num_seqs,). Specifies
            how many KV tokens are valid for each sequence.
        block_tables: Page table mapping logical to physical pages, shape
            (num_seqs, pages_per_seq). For each sequence, maps logical page indices
            to physical page indices in kv_pages. Use -1 or any invalid index for
            unused page slots.
        query_start_loc: Cumulative query offsets, shape (num_seqs + 1,). Indicates
            where each sequence's queries start in the packed queries tensor.
            Example: [0, 5, 13, 16] means sequence 0 has queries 0:5, sequence 1
            has queries 5:13, sequence 2 has queries 13:16.
        num_seqs: Number of sequences in the batch. Can be an integer or a
            scalar JAX array.
        softmax_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        logits_soft_cap: Optional soft capping value for attention logits. When specified,
            applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
            Helps with numerical stability (e.g., Gemma-2 uses 20.0).
        compute_dtype: Computation dtype (ignored in Triton implementation).
        optimized: Optimization flag (ignored in Triton implementation).
        sliding_window: Optional sliding window size for local attention. If specified,
            each query only attends to the last `sliding_window` tokens.
        softmax_aux: Not supported in Triton implementation (raises error if provided).
        mask_value: Value to use for masked positions. Defaults to -2.38e38.
        num_kv_pages_per_block: Number of KV pages to process per block. Higher
            values may improve performance but increase memory usage. Defaults to 8.
        num_queries_per_block: Not supported in Triton (TPU-specific parameter).
        vmem_limit_bytes: Not supported in Triton (TPU-specific parameter).

    Returns:
        Attention output of shape (total_tokens, num_q_heads, head_dim), with
        results packed in the same order as the input queries.

    Raises:
        NotImplementedError: If softmax_aux, num_queries_per_block, or vmem_limit_bytes
            are provided (these are TPU-specific features).
        AssertionError: If combined KV heads is not even, or if dimensions mismatch.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.ragged_page_attention_v2 import ragged_page_attention_v2
        >>>
        >>>
        >>> num_seqs = 3
        >>> query_lens = [4, 6, 3]
        >>> total_tokens = sum(query_lens)
        >>>
        >>>
        >>> num_q_heads, head_dim = 8, 64
        >>> queries = jnp.ones((total_tokens, num_q_heads, head_dim))
        >>>
        >>>
        >>> num_pages, page_size, num_kv_heads = 20, 16, 8
        >>> kv_pages = jnp.ones((num_pages, page_size, 2 * num_kv_heads, head_dim))
        >>>
        >>>
        >>> context_lens = jnp.array([10, 20, 8])
        >>> query_start_loc = jnp.array([0, 4, 10, 13])
        >>> block_tables = jnp.array([
        ...     [0, 1, -1, -1],
        ...     [2, 3, 4, -1],
        ...     [5, -1, -1, -1],
        ... ])
        >>>
        >>> output = ragged_page_attention_v2(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs
        ... )
        >>> print(output.shape)
    """

    if optimized:
        return ragged_paged_attention_triton_call_qblock(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            causal=True,
            block_q=num_queries_per_block or 64,
            tile_size=32,
            num_warps=num_warps,
            num_stages=num_stages,
        ).astype(queries.dtype)
    else:
        block_m = 128 if num_queries_per_block is None else int(num_queries_per_block)
        block_npages = 2 if num_kv_pages_per_block is None else int(num_kv_pages_per_block)
        return ragged_paged_attention_triton_call(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            block_m=block_m,
            block_npages=block_npages,
            causal=True,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            num_warps=num_warps,
            num_stages=num_stages,
        ).astype(queries.dtype)
