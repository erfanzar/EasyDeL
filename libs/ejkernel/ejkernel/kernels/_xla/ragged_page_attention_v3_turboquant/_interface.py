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

"""Ragged paged attention v3 interface with TurboQuant KV cache compression.

Provides the XLA kernel for RPA v3 with TurboQuant-compressed KV caches.
This is the fused update + attention variant: new K/V tokens are compressed
into the paged cache and attention is computed against the updated cache in
one call.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32, UInt8

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_paged_attention_turboquant as _impl


@kernel_registry.register("ragged_page_attention_v3_turboquant", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v3_turboquant(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
    value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    rotation_matrix: Float[Array, "head_dim head_dim"],
    qjl_projection: Float[Array, "qjl_dim head_dim"],
    key_codebook: Float[Array, "key_levels"],
    value_codebook: Float[Array, "value_levels"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    bits: int = 4,
    qjl_dim: int = 128,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    Float[Array, "num_pages page_size num_kv_heads two"],
    UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    Float[Array, "num_pages page_size num_kv_heads"],
]:
    """Compute ragged paged attention v3 with TurboQuant-compressed KV cache on XLA.

    This XLA kernel handles mixed prefill and decode workloads,
    compressing new K/V tokens with TurboQuant and computing attention
    asymmetrically from the compressed cache using pre-rotated queries,
    codebook lookup, and QJL sign-bit correction.

    Args:
        queries: Packed query tokens [total_tokens, num_q_heads, head_dim].
        keys: New key tokens [total_tokens, num_kv_heads, head_dim].
        values: New value tokens [total_tokens, num_kv_heads, head_dim].
        key_indices_pages: Packed 4-bit codebook indices for keys.
        key_signs_pages: Packed QJL sign bits for keys.
        key_norms_pages: Key norms [original_norm, residual_norm].
        value_indices_pages: Packed 4-bit codebook indices for values.
        value_norms_pages: Value original norms.
        kv_lens: Context length per sequence [max_num_seqs].
        block_tables: Flattened page table [max_num_seqs * pages_per_seq].
        query_start_loc: Cumulative query counts [max_num_seqs + 1].
        distribution: [num_prefill, num_decode, num_total].
        rotation_matrix: Orthogonal rotation Pi [head_dim, head_dim].
        qjl_projection: QJL projection S [qjl_dim, head_dim].
        key_codebook: Lloyd-Max centroids for keys [2^(bits-1)].
        value_codebook: Lloyd-Max centroids for values [2^bits].
        softmax_aux: Optional attention sink logits [num_q_heads].
        softmax_scale: QK^T scaling factor.
        sliding_window: Optional sliding window size.
        logits_soft_cap: Optional logit soft capping.
        bits: Total bits per coordinate (default 4).
        qjl_dim: QJL projection dimension (default 128).
        chunk_prefill_size: Accepted for API parity with standard RPA v3.
            Currently unused in the XLA TurboQuant backend.
        num_kv_pages_per_block: Pages per KV processing block.
        num_queries_per_block: Queries per processing block.
        vmem_limit_bytes: Accepted for API parity with standard RPA v3.
            Currently unused in the XLA TurboQuant backend.

    Returns:
        Tuple of (output, key_indices_pages, key_signs_pages,
                  key_norms_pages, value_indices_pages, value_norms_pages).
    """
    _ = (chunk_prefill_size, vmem_limit_bytes)

    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5

    return _impl(
        queries,
        keys,
        values,
        key_indices_pages,
        key_signs_pages,
        key_norms_pages,
        value_indices_pages,
        value_norms_pages,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        rotation_matrix,
        qjl_projection,
        key_codebook,
        value_codebook,
        softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        bits=bits,
        qjl_dim=qjl_dim,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )
