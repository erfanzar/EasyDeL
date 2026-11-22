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

"""Public interface for TPU multi-latent ragged paged attention.

This module exposes the TPU/Pallas MLA ragged paged-attention kernel using
the same argument naming style as `ragged_page_attention_v3` where possible:
`kv_cache`, `block_tables`, and `query_start_loc`.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import DEFAULT_MASK_VALUE, mla_ragged_paged_attention


@kernel_registry.register("multi_latent_ragged_page_attention", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def multi_latent_ragged_page_attention(
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
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> tuple[
    Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
]:
    """Compute TPU MLA ragged paged attention and update paged KV cache.

    Args:
        queries_nope: Query non-positional component.
            Shape `[total_tokens, num_q_heads, qk_nope_dim]`.
        queries_pe: Query positional component.
            Shape `[total_tokens, num_q_heads, qk_pe_dim]`.
        keys_values: New cache K/V-compressed component for incoming tokens.
            Shape `[total_tokens, qk_nope_dim]`.
        keys_pe: New cache key positional component for incoming tokens.
            Shape `[total_tokens, qk_pe_dim]`.
        kv_cache: Paged cache tensor.
            Shape `[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]`.
        kv_lens: Per-sequence KV lengths before writing new tokens.
        block_tables: Flattened sequence-to-page mapping.
        query_start_loc: Ragged query start offsets (`max_num_seqs + 1`).
        distribution: Workload partition tensor `[decode_end, prefill_end, total]`.
        softmax_scale: Attention scale factor for logits.
        sliding_window: Optional causal sliding-window radius.
        logits_soft_cap: Optional logits soft cap (`cap * tanh(x / cap)`).
        mask_value: Value used for masked logits.
        q_scale: Optional query dequantization/requantization scale.
        k_scale: Optional key scale.
        v_scale: Optional value scale.
        chunk_prefill_size: Optional chunk size for prefill sequences.
        num_kv_pages_per_block: Tuned pages per kernel KV block.
        num_queries_per_block: Tuned queries per kernel Q block.
        vmem_limit_bytes: Optional TPU VMEM limit hint.
        debug_mode: If True, run debug path in the underlying kernel.

    Returns:
        Tuple `(outputs, updated_kv_cache)` where:
        - `outputs`: `[total_tokens, num_q_heads, qk_nope_dim]`
        - `updated_kv_cache`: same shape as `kv_cache`
    """
    if softmax_scale is None:
        softmax_scale = (queries_nope.shape[-1] + queries_pe.shape[-1]) ** -0.5

    if num_kv_pages_per_block is None:
        pages_per_seq = max(1, int(block_tables.shape[0]) // max(1, int(kv_lens.shape[0])))
        if pages_per_seq >= 256:
            num_kv_pages_per_block = 16
        elif pages_per_seq >= 128:
            num_kv_pages_per_block = 8
        elif pages_per_seq >= 64:
            num_kv_pages_per_block = 4
        else:
            num_kv_pages_per_block = 2

    if num_queries_per_block is None:
        total_tokens = int(queries_nope.shape[0])
        if total_tokens >= 2048:
            num_queries_per_block = 64
        elif total_tokens >= 256:
            num_queries_per_block = 32
        else:
            num_queries_per_block = 16

    return mla_ragged_paged_attention(
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
    )


__all__ = ("multi_latent_ragged_page_attention",)
