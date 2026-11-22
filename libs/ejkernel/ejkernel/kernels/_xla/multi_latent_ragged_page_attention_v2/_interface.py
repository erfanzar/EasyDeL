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

"""Registry entry point for the XLA MLA ragged paged-attention v2 kernel.

Registers ``multi_latent_ragged_page_attention_v2`` under
``(Platform.XLA, Backend.ANY)`` and normalises per-case block-size tuples
before forwarding to the v1 XLA implementation.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import DEFAULT_MASK_VALUE, multi_latent_ragged_page_attention_v2_impl


@kernel_registry.register("multi_latent_ragged_page_attention_v2", Platform.XLA, Backend.ANY)
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
    """Compute MLA ragged paged attention v2 using XLA.

    XLA fallback implementation.  Normalises per-case tuple block sizes
    to scalars and delegates to the v1 XLA kernel.

    Args:
        queries_nope: Query non-positional component
            ``[total_tokens, num_q_heads, kv_latent_dim]``.
        queries_pe: Query positional component
            ``[total_tokens, num_q_heads, qk_pe_dim]``.
        keys_values: New cache KV-compressed tokens ``[total_tokens, kv_latent_dim]``.
        keys_pe: New cache key positional component ``[total_tokens, qk_pe_dim]``.
        kv_cache: Paged KV cache
            ``[num_pages, page_size/kv_packing, kv_packing, kv_dim_padded]``.
        kv_lens: Context length per sequence ``[max_num_seqs]``.
        block_tables: Flat page-index table ``[max_num_seqs * pages_per_seq]``.
        query_start_loc: Cumulative query token counts ``[max_num_seqs + 1]``.
        distribution: Three-element array ``[n_decode, n_prefill, n_mixed]``.
        softmax_scale: Attention scaling factor (default: ``(nope_dim + pe_dim)^-0.5``).
        sliding_window: Optional local-attention window size.
        logits_soft_cap: Optional soft cap for logits.
        mask_value: Fill value for masked positions.
        q_scale: Optional FP8 query de-quantisation scale.
        k_scale: Optional FP8 key de-quantisation scale.
        v_scale: Optional FP8 value de-quantisation scale.
        chunk_prefill_size: Optional chunk size for long prefill sequences.
        num_kv_pages_per_block: Pages per kernel KV block (scalar or triple).
        num_queries_per_block: Queries per kernel Q block (scalar or triple).
        vmem_limit_bytes: Optional TPU VMEM limit hint (unused on XLA path).
        debug_mode: If True, enable debug path (unused on XLA path).

    Returns:
        Tuple ``(outputs, updated_kv_cache)`` where:
        - ``outputs``: ``[total_tokens, num_q_heads, kv_latent_dim]``
        - ``updated_kv_cache``: same shape as ``kv_cache``
    """
    if softmax_scale is None:
        softmax_scale = (queries_nope.shape[-1] + queries_pe.shape[-1]) ** -0.5

    return multi_latent_ragged_page_attention_v2_impl(
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
