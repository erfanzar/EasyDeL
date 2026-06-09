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

"""Chunked prefill + paged decode attention public interface for the CuTe backend.

This module exposes :func:`chunked_prefill_paged_decode`, registered under
``("chunked_prefill_paged_decode", Platform.CUTE, Backend.GPU)`` in the
kernel registry.

The function implements a two-phase pipeline on GPU:
1. **KV cache update** (CuTe DSL kernel): scatter-writes packed key/value
   tokens into the correct physical slots of the block-tabled paged KV cache.
2. **Attention** (Triton ``unified_attention`` kernel): computes paged causal
   attention on the freshly updated cache.

Only causal attention is supported; non-causal calls raise
:class:`NotImplementedError`.  All three index arrays (``kv_lens``,
``block_tables``, ``query_start_loc``) must have ``int32`` dtype.
"""

from __future__ import annotations

import math

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._cute_impl_fwd import chunked_prefill_paged_decode_cute


@kernel_registry.register("chunked_prefill_paged_decode", Platform.CUTE, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def chunked_prefill_paged_decode(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
]:
    """CuTe GPU implementation of chunked prefill + paged decode attention.

    Updates the block-tabled KV cache with the supplied packed keys/values
    (via a CuTe DSL scatter kernel) and then computes causal paged attention
    on the updated cache (via the Triton ``unified_attention`` kernel).

    Args:
        queries: Packed query tokens, shape
            ``(total_tokens, num_q_heads, head_dim)``.
        keys: Packed key tokens to insert into the KV cache, shape
            ``(total_tokens, num_kv_heads, head_dim)``.
        values: Packed value tokens to insert into the KV cache, same shape
            as *keys*.
        key_cache: Existing block-tabled key cache, shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: Existing block-tabled value cache, same shape as
            *key_cache*.
        kv_lens: Total KV length (including context) per sequence,
            shape ``(num_seqs,)``, dtype ``int32``.
        block_tables: Logical-to-physical block mapping, shape
            ``(num_seqs, max_blocks_per_seq)``, dtype ``int32``.
        query_start_loc: Cumulative packed-query start offsets, shape
            ``(num_seqs + 1,)``, dtype ``int32``.
        alibi_slopes: Optional ALiBi position bias slopes, shape
            ``(num_q_heads,)``.
        softmax_aux: Optional attention-sink auxiliary logits, shape
            ``(num_q_heads,)``.
        softmax_scale: Attention score scale. Defaults to
            ``1 / sqrt(head_dim)`` when ``None``.
        causal: Whether to apply causal masking. Must be ``True``.
        sliding_window: Optional local-attention window size.
        logits_soft_cap: Optional tanh soft-capping value for logits.
        seq_threshold_3d: Optional Triton decode-kernel threshold hint.
        num_par_softmax_segments: Optional Triton segmented-softmax hint.
        num_warps: Optional Triton warp-count override.
        num_stages: Optional Triton pipeline-stage override.

    Returns:
        A 3-tuple ``(attention_output, updated_key_cache, updated_value_cache)``
        where ``attention_output`` has shape
        ``(total_tokens, num_q_heads, head_dim)`` and the updated caches
        have the same shape as the input caches.

    Raises:
        NotImplementedError: If *causal* is ``False``.
        ValueError: If tensor dtypes or shapes are invalid.
        EjkernelRuntimeError: If Triton unified attention is not available.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(queries.shape[-1])
    return chunked_prefill_paged_decode_cute(
        queries=queries,
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        alibi_slopes=alibi_slopes,
        softmax_aux=softmax_aux,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        num_warps=num_warps,
        num_stages=num_stages,
    )
