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

"""Public interface for TPU multi-latent ragged paged attention v2."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import DEFAULT_MASK_VALUE, mla_ragged_paged_attention_v2


def _default_num_kv_pages_per_block(
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"], kv_lens: Int32[Array, "max_num_seqs"]
) -> int:
    """Heuristic KV-pages-per-block based on average pages per sequence.

    Longer contexts have more pages and benefit from larger tiles; short
    contexts need smaller tiles to avoid wasting compute on padding.

    Args:
        block_tables: Flat page-index table [max_num_seqs * pages_per_seq].
        kv_lens: Context lengths per sequence [max_num_seqs].

    Returns:
        Heuristic KV-pages-per-block (2, 4, 8, or 16).
    """
    pages_per_seq = max(1, int(block_tables.shape[0]) // max(1, int(kv_lens.shape[0])))
    if pages_per_seq >= 256:
        return 16
    if pages_per_seq >= 128:
        return 8
    if pages_per_seq >= 64:
        return 4
    return 2


def _default_num_queries_per_block(queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"]) -> int:
    """Heuristic queries-per-block based on total token count.

    More tokens in the batch allow larger query blocks without excessive
    padding at the tail.

    Args:
        queries_nope: Non-positional queries [total_tokens, num_q_heads, kv_latent_dim].

    Returns:
        Heuristic queries-per-block (16, 32, or 64).
    """
    total_tokens = int(queries_nope.shape[0])
    if total_tokens >= 2048:
        return 64
    if total_tokens >= 256:
        return 32
    return 16


@kernel_registry.register("multi_latent_ragged_page_attention_v2", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def multi_latent_ragged_page_attention_v2(
    queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
    keys_values: Float[Array, "total_tokens kv_latent_dim"],
    keys_pe: Float[Array, "total_tokens qk_pe_dim"],
    kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> tuple[
    Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
]:
    """Compute TPU MLA ragged paged attention v2 and update paged KV cache.

    V2 extends the v1 kernel with per-case ``(decode, prefill, mixed)``
    block-size tuples so that the Pallas grid can be specialised for each
    request type within a single batch.

    Args:
        queries_nope: Query non-positional component.
            Shape ``[total_tokens, num_q_heads, kv_latent_dim]``.
        queries_pe: Query positional component.
            Shape ``[total_tokens, num_q_heads, qk_pe_dim]``.
        keys_values: New cache KV-compressed tokens.
            Shape ``[total_tokens, kv_latent_dim]``.
        keys_pe: New cache key positional component.
            Shape ``[total_tokens, qk_pe_dim]``.
        kv_cache: Paged KV cache.
            Shape ``[num_pages, page_size/kv_packing, kv_packing, kv_dim_padded]``.
        kv_lens: Context length per sequence ``[max_num_seqs]``.
        block_tables: Flat page-index table ``[max_num_seqs * pages_per_seq]``.
        query_start_loc: Cumulative query token counts ``[max_num_seqs + 1]``.
        distribution: Three-element array ``[n_decode, n_prefill, n_mixed]``
            describing how the batch is partitioned.
        softmax_scale: Attention scaling factor (default: ``(nope_dim + pe_dim)^-0.5``).
        sliding_window: Optional local-attention window size.
        logits_soft_cap: Optional Gemma-2-style soft cap for logits.
        mask_value: Fill value for masked positions.
        q_scale: Optional FP8 query de-quantisation scale.
        k_scale: Optional FP8 key de-quantisation scale.
        v_scale: Optional FP8 value de-quantisation scale.
        chunk_prefill_size: Optional chunk size for long prefill sequences.
        num_kv_pages_per_block: Pages per kernel KV block.  May be a scalar
            or a ``(decode, prefill, mixed)`` tuple.
        num_queries_per_block: Queries per kernel Q block.  May be a scalar
            or a ``(decode, prefill, mixed)`` tuple.
        vmem_limit_bytes: Optional TPU VMEM limit hint for the Pallas compiler.
        debug_mode: If True, run debug path in the underlying kernel.

    Returns:
        Tuple ``(outputs, updated_kv_cache)`` where:
        - ``outputs``: ``[total_tokens, num_q_heads, kv_latent_dim]``
        - ``updated_kv_cache``: same shape as ``kv_cache``
    """
    if softmax_scale is None:
        softmax_scale = (queries_nope.shape[-1] + queries_pe.shape[-1]) ** -0.5

    if num_kv_pages_per_block is None:
        num_kv_pages_per_block = _default_num_kv_pages_per_block(block_tables, kv_lens)

    if num_queries_per_block is None:
        num_queries_per_block = _default_num_queries_per_block(queries_nope)

    return mla_ragged_paged_attention_v2(
        queries_nope=queries_nope,
        queries_pe=queries_pe,
        keys_values=keys_values,
        keys_pe=keys_pe,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )


__all__ = ("multi_latent_ragged_page_attention_v2",)
