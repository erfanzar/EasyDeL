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

"""Ragged paged attention v2 interface for variable-length batches.

This module provides the public API for paged attention with ragged (variable-length)
sequences. Supports multiple query tokens per sequence and FlashAttention-style
online softmax for memory-efficient computation.
"""

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _ragged_paged_attention


@kernel_registry.register("ragged_page_attention_v2", Platform.XLA, Backend.ANY)
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
    """Paged attention for ragged (variable-length) sequences with optional attention sinks.

    Implements a FlashAttention-style online-softmax algorithm that reads
    KV data from a paged cache (non-contiguous memory) and handles batches
    where different sequences have different numbers of query tokens and
    different KV context lengths.  This is the XLA reference implementation;
    no custom Pallas/CUDA calls are used.

    KV cache layout:
        ``kv_pages[page_id, token_in_page, kv_head * 2 + {0=K,1=V}, head_dim]``
        — keys and values are interleaved along the combined-head axis.

    Registered under ``"ragged_page_attention_v2"`` for ``Platform.XLA``,
    ``Backend.ANY``.

    Note:
        The ``optimized``, ``mask_value``, ``num_kv_pages_per_block``,
        ``num_queries_per_block``, and ``vmem_limit_bytes`` arguments are
        accepted for API compatibility with the Pallas backend but are ignored
        by this implementation.

    Args:
        queries: Packed query tokens for all sequences.
            Shape: ``[total_query_tokens, num_q_heads, head_dim]``.
        kv_pages: Interleaved paged KV cache.
            Shape: ``[num_pages, page_size, num_kv_heads * 2, head_dim]``.
            K pages are at even indices along axis 2; V pages at odd indices.
        context_lens: Total KV context length per sequence (includes cached
            tokens from prior steps).
            Shape: ``[num_seqs]``, dtype ``int32``.
        block_tables: Per-sequence page mapping.
            Shape: ``[num_seqs, max_pages_per_seq]``.  Each row maps a
            logical page index to a physical page index in ``kv_pages``.
        query_start_loc: Cumulative query token offsets.
            Shape: ``[num_seqs + 1]``.  Sequence ``i``'s queries are at
            ``queries[query_start_loc[i]:query_start_loc[i+1]]``.
        num_seqs: Number of active sequences in the batch.  Scalar or
            shape ``[1]`` int32.  Only the first ``num_seqs`` rows of
            ``context_lens`` and ``block_tables`` are processed.
        softmax_scale: Multiplicative scale for QK^T logits.
            Defaults to ``head_dim ** -0.5`` when ``None``.
        logits_soft_cap: Optional tanh capping radius for attention logits.
        compute_dtype: Dtype for intermediate matmul computations.
            Default: ``bfloat16``.
        optimized: Accepted for API compatibility; ignored by this backend.
        sliding_window: Optional sliding window size (left window only).
            Limits attention to the most recent ``sliding_window`` tokens.
        softmax_aux: Optional per-head attention-sink logits.
            Shape: ``[num_q_heads]``.  Seeds the online-softmax running max
            and normaliser so the model can absorb probability mass.
        mask_value: Accepted for API compatibility; ignored by this backend.
        num_kv_pages_per_block: Accepted for API compatibility; ignored.
        num_queries_per_block: Accepted for API compatibility; ignored.
        vmem_limit_bytes: Accepted for API compatibility; ignored.
        num_warps: Accepted for API compatibility; ignored.
        num_stages: Accepted for API compatibility; ignored.

    Returns:
        Attention output, same shape and dtype as ``queries``:
        ``[total_query_tokens, num_q_heads, head_dim]``.
    """
    del mask_value, num_kv_pages_per_block, num_queries_per_block, vmem_limit_bytes
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5
    fn = _ragged_paged_attention
    return fn(
        queries=queries,
        kv_pages=kv_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        compute_dtype=compute_dtype,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
    )
