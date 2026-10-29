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

"""Registry entry point for the TPU Pallas sparse MLA ragged paged-attention kernel."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.sparse_multi_latent_ragged_page_attention._interface import (
    sparse_multi_latent_ragged_page_attention as _xla_sparse_mla,
)
from ...._xla.sparse_multi_latent_ragged_page_attention._xla_impl_fwd import DEFAULT_MASK_VALUE, validate_inputs
from ._pallas_impl_fwd import sparse_mla_ragged_paged_attention_pallas, supports


@kernel_registry.register("sparse_multi_latent_ragged_page_attention", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def sparse_multi_latent_ragged_page_attention(
    queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
    keys_values: Float[Array, "total_tokens kv_latent_dim"],
    keys_pe: Float[Array, "total_tokens qk_pe_dim"],
    kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    selected: Int32[Array, "total_tokens num_selected"],
    *,
    chunk_size: int = 1,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_queries_per_block: int | None = None,
    num_chunks_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
]:
    """Sparse MLA ragged paged attention on TPU (Pallas).

    Same contract as the XLA reference: the step's latent tokens are written into ``kv_cache``
    and query ``t`` attends only to tokens ``j`` with ``j // chunk_size`` in ``selected[t]``,
    ``j <= pos(t)`` and ``j < kv_lens[seq(t)]``. Layouts the kernel cannot address with one DMA
    per chunk (``chunk_size`` not a multiple of the cache packing, or not dividing the page size)
    are routed to the XLA reference.

    Args:
        queries_nope: ``[T, H, L]`` weight-absorbed latent queries.
        queries_pe: ``[T, H, R]`` rope queries.
        keys_values: ``[T, L]`` new latent tokens.
        keys_pe: ``[T, R]`` new rope keys.
        kv_cache: Paged cache ``[P, page_size/pack, pack, L_pad + R_pad]`` (donated).
        kv_lens: ``[S]`` context lengths including the current step.
        block_tables: ``[S * pages_per_seq]`` flat page table.
        query_start_loc: ``[S + 1]`` cumulative query counts.
        distribution: ``[n_decode, n_prefill, n_total]``.
        selected: ``[T, W]`` chunk ids, ``-1``-padded, unique per row.
        chunk_size: Tokens per chunk.
        softmax_scale: Logit scale (default ``(L + R) ** -0.5``).
        sliding_window: Optional local window.
        logits_soft_cap: Optional tanh soft cap.
        mask_value: Logit fill for masked keys.
        q_scale: Optional query dequantisation scale.
        k_scale: Optional key dequantisation scale.
        v_scale: Optional value dequantisation scale.
        num_queries_per_block: Used only by the XLA fallback.
        num_chunks_per_block: Chunks fetched per kernel step (default ``128 // chunk_size``).
        vmem_limit_bytes: Optional VMEM limit for the Pallas compiler.

    Returns:
        ``(outputs [T, H, L], updated_kv_cache)``.
    """
    common = dict(
        chunk_size=chunk_size,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    args = (
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        selected,
    )
    if not supports(tuple(kv_cache.shape), int(chunk_size)):
        return _xla_sparse_mla(*args, **common, num_queries_per_block=num_queries_per_block)
    validate_inputs(
        *args,
        chunk_size=chunk_size,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
    )
    if softmax_scale is None:
        softmax_scale = (queries_nope.shape[-1] + queries_pe.shape[-1]) ** -0.5
    return sparse_mla_ragged_paged_attention_pallas(
        *args,
        chunk_size=int(chunk_size),
        sm_scale=float(softmax_scale),
        sliding_window=sliding_window,
        soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_chunks_per_block=num_chunks_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )


__all__ = ("sparse_multi_latent_ragged_page_attention",)
