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

"""Registry entry point for the XLA sparse MLA ragged paged-attention kernel."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import DEFAULT_MASK_VALUE, sparse_multi_latent_ragged_page_attention_impl, validate_inputs


@kernel_registry.register("sparse_multi_latent_ragged_page_attention", Platform.XLA, Backend.ANY)
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
    """Sparse MLA ragged paged attention (XLA fallback and numerical reference).

    Writes the current step's latent tokens into ``kv_cache`` exactly like
    ``multi_latent_ragged_page_attention_v2``, then lets query ``t`` attend only to tokens
    ``j`` with ``j // chunk_size`` in ``selected[t]`` (``-1`` = padding), ``j <= pos(t)``
    and ``j < kv_lens[seq(t)]``. Selected chunk ids must be unique per row.

    Args:
        queries_nope: ``[T, H, L]`` weight-absorbed latent queries.
        queries_pe: ``[T, H, R]`` rope queries.
        keys_values: ``[T, L]`` new latent tokens.
        keys_pe: ``[T, R]`` new rope keys.
        kv_cache: Paged cache ``[P, page_size/pack, pack, L_pad + R_pad]``.
        kv_lens: ``[S]`` context lengths including the current step.
        block_tables: ``[S * pages_per_seq]`` flat page table.
        query_start_loc: ``[S + 1]`` cumulative query counts.
        distribution: ``[n_decode, n_prefill, n_total]``.
        selected: ``[T, W]`` chunk ids, ``-1``-padded.
        chunk_size: Tokens per chunk (``index_kpool`` for GLM-5, 1 for per-token top-k).
        softmax_scale: Logit scale (default ``(L + R) ** -0.5``).
        sliding_window: Optional local window.
        logits_soft_cap: Optional tanh soft cap.
        mask_value: Accepted for API parity with v2.
        q_scale: Optional query dequantisation scale.
        k_scale: Optional key dequantisation scale.
        v_scale: Optional value dequantisation scale.
        num_queries_per_block: Query rows per gather step.
        num_chunks_per_block: Accepted for API parity with the Pallas backend (unused).
        vmem_limit_bytes: Accepted for API parity with the Pallas backend (unused).

    Returns:
        ``(outputs [T, H, L], updated_kv_cache)``.
    """
    del num_chunks_per_block, vmem_limit_bytes
    validate_inputs(
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
        chunk_size=chunk_size,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
    )
    if softmax_scale is None:
        softmax_scale = (queries_nope.shape[-1] + queries_pe.shape[-1]) ** -0.5
    return sparse_multi_latent_ragged_page_attention_impl(
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
        chunk_size=int(chunk_size),
        softmax_scale=float(softmax_scale),
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_queries_per_block=num_queries_per_block,
    )


__all__ = ("sparse_multi_latent_ragged_page_attention",)
