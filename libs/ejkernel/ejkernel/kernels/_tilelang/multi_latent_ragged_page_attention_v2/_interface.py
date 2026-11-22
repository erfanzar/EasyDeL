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

"""TileLang multi-latent ragged paged attention v2 public interface.

This module registers ``multi_latent_ragged_page_attention_v2`` for
``Platform.TILELANG / Backend.GPU``.  The v2 registration is functionally
identical to v1 (both call
``_run_multi_latent_ragged_page_attention_native``); it exists as a
separately named entry so callers that explicitly request ``_v2`` routing
via the kernel registry receive the TileLang implementation rather than
falling through to another backend.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ..multi_latent_ragged_page_attention._interface import (
    DEFAULT_MASK_VALUE,
    _run_multi_latent_ragged_page_attention_native,
)


@kernel_registry.register("multi_latent_ragged_page_attention_v2", Platform.TILELANG, Backend.GPU)
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
    """Run native MLA ragged paged attention and in-place KV-cache update (v2).

    Functionally identical to
    :func:`multi_latent_ragged_page_attention._interface.multi_latent_ragged_page_attention`;
    registered under a distinct name for callers that request the v2 variant.
    All validation and dispatch are performed by
    ``_run_multi_latent_ragged_page_attention_native``; refer to that
    function for complete parameter documentation.

    Args:
        queries_nope: ``[total_tokens, num_q_heads, kv_latent_dim]`` NoPE queries.
        queries_pe: ``[total_tokens, num_q_heads, qk_pe_dim]`` RoPE queries.
        keys_values: ``[total_tokens, kv_latent_dim]`` NoPE key/value latents.
        keys_pe: ``[total_tokens, qk_pe_dim]`` RoPE key values.
        kv_cache: ``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``
            paged KV cache.
        kv_lens: ``[max_num_seqs]`` int32 KV context lengths.
        block_tables: ``[max_num_seqs_times_pages_per_seq]`` int32 physical page map.
        query_start_loc: ``[max_num_seqs_plus_1]`` int32 query token offsets.
        distribution: ``[3]`` int32 runtime metadata.
        softmax_scale: Attention temperature; defaults to
            ``1 / sqrt(kv_latent_dim + qk_pe_dim)``.
        sliding_window: Sliding-window size; ``None`` disables.
        logits_soft_cap: Logit soft-cap; ``None`` disables.
        mask_value: Masked-position fill; defaults to ``DEFAULT_MASK_VALUE``.
        q_scale: Query quantisation scale (default 1.0).
        k_scale: Key quantisation scale (default 1.0).
        v_scale: Value output scale (default 1.0).
        chunk_prefill_size: Ignored by this backend.
        num_kv_pages_per_block: KV pages per CTA tile hint (int, 3-tuple, or None).
        num_queries_per_block: Query per CTA hint; ignored by this backend.
        vmem_limit_bytes: VMEM limit; ignored by this backend.
        debug_mode: Debug flag; ignored by this backend.

    Returns:
        ``(O, KVOut)`` — attention output and updated KV cache, both with the
        same shapes as the inputs.
    """
    return _run_multi_latent_ragged_page_attention_native(
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
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


__all__ = ["multi_latent_ragged_page_attention_v2"]
