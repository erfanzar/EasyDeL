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

"""Registry interface for XLA Dual-Stream Ragged Paged Attention."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ejkernel.kernels._xla.ragged_page_attention_v3._sinks import reduce_sinks_to_heads

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import dual_stream_ragged_paged_attention_xla


@kernel_registry.register("dual_stream_ragged_page_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def dual_stream_ragged_page_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    local_keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    local_values: Float[Array, "total_tokens num_kv_heads head_dim"],
    global_kv_cache: Float[
        Array, "num_global_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"
    ],
    local_kv_cache: Float[
        Array, "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"
    ],
    global_kv_lens: Int32[Array, "max_num_seqs"],
    global_block_tables: Int32[Array, "max_num_seqs_times_global_pages"],
    local_kv_lens: Int32[Array, "max_num_seqs"],
    local_block_tables: Int32[Array, "max_num_seqs_times_local_pages"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    softmax_aux: Float[Array, "..."] | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int = 128,
    overlap: bool = True,
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
    Float[Array, "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Dual-stream ragged paged attention (XLA fallback).

    Attends jointly over a read-only global paged KV cache (e.g., Pass-1 shared
    KV) and an in-place-updated local sliding-window paged KV cache (e.g.,
    Pass-2 local KV) under one exact float32 softmax normalizer.

    Args:
        queries: Ragged queries ``[total_tokens, num_q_heads, head_dim]``.
        local_keys: New local-stream keys ``[total_tokens, num_kv_heads, head_dim]``
            to append into ``local_kv_cache``.
        local_values: New local-stream values ``[total_tokens, num_kv_heads, head_dim]``
            to append into ``local_kv_cache``.
        global_kv_cache: Read-only global paged KV cache.
        local_kv_cache: Local sliding-window paged KV cache updated in-place.
        global_kv_lens: Per-sequence lengths in ``global_kv_cache`` (including current tokens).
        global_block_tables: Flattened page table for ``global_kv_cache``.
        local_kv_lens: Per-sequence (possibly eviction-rebased) lengths in ``local_kv_cache``.
        local_block_tables: Flattened page table for ``local_kv_cache``.
        query_start_loc: Cumulative token offsets ``[max_num_seqs + 1]``.
        distribution: Batch split ``[decode_end, prefill_end, total_seqs]``.
        softmax_aux: Optional attention sink logits (reduced per query head, applied once).
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        sliding_window: Width ``W`` of the local sliding window.
        overlap: If ``True``, global stream attends causally ``k <= q``; if
            ``False``, global stream attends disjointly ``k <= q - W``.
        logits_soft_cap: Optional tanh logit soft cap.
        q_scale: Optional query quantization scale.
        k_scale: Optional key quantization scale.
        v_scale: Optional value quantization scale.
        chunk_prefill_size: Optional prefill chunk size hint.
        num_kv_pages_per_block: Optional KV pages per tile block.
        num_queries_per_block: Optional queries per tile block.
        vmem_limit_bytes: Optional TPU VMEM budget hint (ignored on XLA).

    Returns:
        Tuple ``(attention_output, updated_local_kv_cache)``.
    """
    if softmax_scale is None:
        softmax_scale = float(queries.shape[-1]) ** -0.5
    softmax_aux = reduce_sinks_to_heads(softmax_aux, queries.shape[1], local_keys.shape[1])
    return dual_stream_ragged_paged_attention_xla(
        queries,
        local_keys,
        local_values,
        global_kv_cache,
        local_kv_cache,
        global_kv_lens,
        global_block_tables,
        local_kv_lens,
        local_block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale=float(softmax_scale),
        sliding_window=int(sliding_window),
        overlap=bool(overlap),
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
