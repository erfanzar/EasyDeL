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


"""Chunked Prefill Attention with Paged KV Cache for TPU.

This module provides a TPU-optimized implementation of chunked prefill attention
that reads from a paged KV cache. Unlike decode-phase page attention which
processes single tokens, this kernel handles multiple query tokens during the
prefill phase of LLM inference.

Chunked prefill is essential for:
1. Efficient prompt processing with paged KV cache
2. Continuous batching where prefill and decode coexist
3. Memory-efficient handling of long input sequences
4. Low-latency first-token generation

Key Features:
    - Chunked processing: Handles multiple query tokens at once
    - Paged KV cache integration: Reads from non-contiguous pages
    - Causal masking: Each query only attends to itself and past tokens
    - Grouped Query Attention (GQA): Efficient multi-query variants
    - Sliding window: Local attention for long sequences
    - Logits soft capping: Numerical stability for certain models

Difference from Decode Attention:
    - Prefill: Multiple query tokens, building up the KV cache
    - Decode: Single query token per sequence, reading from full cache
    - This kernel bridges both phases in continuous batching

Algorithm overview:
1. Query chunk is processed against existing KV cache pages
2. Causal masking ensures queries only see valid context
3. Pages are loaded via page_indices mapping
4. Attention is computed with flash attention tiling
5. Results are accumulated and returned

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.prefill_page_attention import prefill_page_attention
    >>>
    >>> # Prefill a chunk of tokens
    >>> chunk_size, num_heads, head_dim = 64, 32, 128
    >>> num_pages, page_size = 16, 16
    >>>
    >>> query = jnp.ones((chunk_size, num_heads, head_dim))
    >>> key_cache = jnp.ones((num_heads, 256, page_size, head_dim))
    >>> value_cache = jnp.ones((num_heads, 256, page_size, head_dim))
    >>> context_len = jnp.array([chunk_size])
    >>> page_indices = jnp.arange(num_pages)
    >>>
    >>> output = prefill_page_attention(
    ...     query, key_cache, value_cache, context_len, page_indices
    ... )

Note:
    chunk_size must be divisible by page_size for correct operation.
"""

import functools

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import (
    DEFAULT_MASK_VALUE,
    chunked_prefill_attention_kernel,
)

if hasattr(pl, "ANY"):
    _HBM_ANY = pl.ANY
else:
    _HBM_ANY = pltpu.ANY


@kernel_registry.register("prefill_page_attention", Platform.PALLAS, Backend.TPU)
@ejit(
    static_argnames=[
        "softmax_scale",
        "mask_value",
        "attn_logits_soft_cap",
        "sliding_window",
    ],
)
@jaxtyping.jaxtyped(typechecker=beartype)
def prefill_page_attention(
    query: Float[Array, "chunk_size num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_len: Int[Array, "1"],
    page_indices: Int[Array, "num_pages"],
    *,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "chunk_size num_heads head_dim"]:
    """Chunked prefill attention with paged KV cache for TPU.

    This kernel processes a chunk of query tokens during prefill phase,
    reading from a paged KV cache. It supports causal masking and sliding window.

    Args:
        query: Query tensor [chunk_size, num_q_heads, head_dim] for prefill tokens.
        key_cache: Paged key cache [num_kv_heads, total_num_pages, page_size, head_dim].
        value_cache: Paged value cache [num_kv_heads, total_num_pages, page_size, head_dim].
        context_len: Total context length including this chunk [1].
        page_indices: Page indices for this sequence [num_pages].
        softmax_scale: Attention scaling factor (default: 1/sqrt(head_dim)).
        mask_value: Value used for masked positions in attention.
        attn_logits_soft_cap: Optional soft cap for attention logits.
        sliding_window: If set, only attend to the last `sliding_window` tokens.
        block_k: Backend tuning hint accepted for operation-level autotune;
            ignored by this Pallas kernel.
        num_warps: Backend tuning hint accepted for operation-level autotune;
            ignored by this Pallas kernel.
        num_stages: Backend tuning hint accepted for operation-level autotune;
            ignored by this Pallas kernel.

    Returns:
        Attention output [chunk_size, num_q_heads, head_dim].

    Note:
        - This is designed for prefill phase where we process multiple query tokens
        - Uses causal masking so each query can only attend to itself and past tokens
        - The KV cache should already contain the keys/values for this sequence
        - chunk_size must be divisible by page_size
    """
    _ = block_k, num_warps, num_stages
    chunk_size, num_q_heads, head_dim = query.shape
    num_kv_heads, _total_num_pages, page_size, head_dim_k = key_cache.shape

    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key_cache and value_cache must have the same shape. Got {key_cache.shape} and {value_cache.shape}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"Number of Q heads must be divisible by number of KV heads. Got {num_q_heads} and {num_kv_heads}."
        )
    if head_dim_k != head_dim:
        raise ValueError(f"head_dim of Q must be the same as that of K/V. Got {head_dim} and {head_dim_k}.")
    if chunk_size % page_size != 0:
        raise ValueError(f"chunk_size must be divisible by page_size. Got {chunk_size} and {page_size}.")

    attn_group_size = num_q_heads // num_kv_heads
    pages_per_chunk = chunk_size // page_size
    num_pages = page_indices.shape[0]
    num_kv_chunks = num_pages // pages_per_chunk

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim).astype(query.dtype)

    q = query.transpose((1, 0, 2))
    q = q.reshape(num_kv_heads, attn_group_size, chunk_size, head_dim)
    q = q * softmax_scale

    q_block_spec = pl.BlockSpec((1, attn_group_size, chunk_size, head_dim), lambda i, *_: (i, 0, 0, 0))
    lm_block_spec = pl.BlockSpec((1, attn_group_size, chunk_size, 1), lambda i, *_: (i, 0, 0, 0))
    lm_shape = jax.ShapeDtypeStruct(shape=(num_kv_heads, attn_group_size, chunk_size, 1), dtype=jnp.float32)

    out, _, _ = pl.pallas_call(
        functools.partial(
            chunked_prefill_attention_kernel,
            chunk_size=chunk_size,
            page_size=page_size,
            num_kv_chunks=num_kv_chunks,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            sliding_window=sliding_window,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                q_block_spec,
                pl.BlockSpec(memory_space=_HBM_ANY),
                pl.BlockSpec(memory_space=_HBM_ANY),
            ],
            out_specs=[
                q_block_spec,
                lm_block_spec,
                lm_block_spec,
            ],
            scratch_shapes=[
                pltpu.VMEM((2, pages_per_chunk, page_size, head_dim), key_cache.dtype),
                pltpu.VMEM((2, pages_per_chunk, page_size, head_dim), value_cache.dtype),
                pltpu.SemaphoreType.DMA,
            ],
            grid=(num_kv_heads,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_kv_heads, attn_group_size, chunk_size, head_dim), q.dtype),
            lm_shape,
            lm_shape,
        ],
    )(
        context_len.reshape((1,)),
        page_indices,
        jnp.asarray([0], jnp.int32),
        q,
        key_cache,
        value_cache,
    )

    out = out.reshape(num_q_heads, chunk_size, head_dim)
    out = out.transpose((1, 0, 2))

    return out.astype(query.dtype)
