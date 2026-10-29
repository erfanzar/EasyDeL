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

"""Ragged paged attention v2 interface with TurboQuant KV cache compression.

Provides the XLA kernel for RPA v2 with TurboQuant-compressed KV caches.
This is a read-only cache variant: pages are pre-populated and no cache
writes are performed. The kernel accepts separate page tensors for codebook
indices, QJL signs, and norms, plus precomputed rotation/projection matrices
and codebooks.
"""

import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int32, UInt8

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_paged_attention_v2_turboquant as _impl


@kernel_registry.register("ragged_page_attention_v2_turboquant", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v2_turboquant(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    key_signs_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_sign_dim"],
    key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"],
    value_indices_pages: UInt8[Array, "num_pages page_size num_kv_heads packed_idx_dim"],
    value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"],
    context_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_one"],
    num_seqs: Int32[Array, "1"] | int,
    rotation_matrix: Float[Array, "head_dim head_dim"],
    qjl_projection: Float[Array, "qjl_dim head_dim"],
    key_codebook: Float[Array, "key_levels"],
    value_codebook: Float[Array, "value_levels"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    sliding_window: int | None = None,
    mask_value: float | None = None,
    bits: int = 4,
    qjl_dim: int = 128,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Ragged paged attention v2 with TurboQuant-compressed KV cache (read-only).

    Read-only cache variant: pages must be pre-populated with compressed KV
    data before this call.  No cache writes are performed.

    Attention is computed asymmetrically via the TurboQuant unbiased
    inner-product estimator:

    .. math::
        \\langle q, k \\rangle \\approx
        \\langle q_{\\text{rot}},\\, c[\\text{idx}] \\rangle \\cdot \\|k\\|
        + \\frac{\\sqrt{\\pi/2}}{m} \\,
        \\langle q_{\\text{proj}},\\, \\text{signs} \\rangle \\cdot \\|r_k\\|

    Registered under ``"ragged_page_attention_v2_turboquant"`` for
    ``Platform.XLA``, ``Backend.ANY``.

    Note:
        ``mask_value``, ``vmem_limit_bytes``, ``num_warps``, and ``num_stages``
        are accepted for API compatibility but are ignored by this backend.

    Args:
        queries: Packed query tokens, shape
            ``[total_tokens, num_q_heads, head_dim]``.
        key_indices_pages: Packed 4-bit Lloyd-Max codebook indices for cached
            keys, shape ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        key_signs_pages: Packed 1-bit QJL residual signs for cached keys,
            shape ``[num_pages, page_size, num_kv_heads, qjl_dim // 8]``,
            dtype ``uint8``.
        key_norms_pages: Per-token key norms.  Column 0 is ``||k||``; column 1
            is the quantisation residual norm ``||r_k||``.
            Shape ``[num_pages, page_size, num_kv_heads, 2]``.
        value_indices_pages: Packed 4-bit codebook indices for cached values,
            shape ``[num_pages, page_size, num_kv_heads, head_dim // 2]``,
            dtype ``uint8``.
        value_norms_pages: Per-token original value norms,
            shape ``[num_pages, page_size, num_kv_heads]``.
        context_lens: KV context length per sequence, shape ``[num_seqs]``.
        block_tables: Page table, shape ``[num_seqs, pages_per_seq]``.
        query_start_loc: Cumulative query counts, shape ``[num_seqs + 1]``.
        num_seqs: Number of active sequences (scalar or shape ``[1]`` int32).
        rotation_matrix: Haar-distributed orthogonal matrix *Pi*,
            shape ``[head_dim, head_dim]``.
        qjl_projection: Random Gaussian QJL projection *S*,
            shape ``[qjl_dim, head_dim]``.
        key_codebook: Lloyd-Max centroids for keys, shape ``[2^bits]``.
        value_codebook: Lloyd-Max centroids for values, shape ``[2^bits]``.
        softmax_aux: Optional per-head attention-sink logits,
            shape ``[num_q_heads]``.
        softmax_scale: QK^T scaling factor.  Defaults to ``head_dim ** -0.5``.
        logits_soft_cap: Optional tanh capping radius.
        compute_dtype: Dtype for intermediate computation (accepted but
            forwarded to implementation).
        sliding_window: Optional sliding window size.
        mask_value: Accepted for API compatibility; ignored.
        bits: Quantisation bits per coordinate (default 4).
        qjl_dim: QJL projection dimension (default 128).
        num_kv_pages_per_block: Pages per KV processing block.
        num_queries_per_block: Queries per processing block.
        vmem_limit_bytes: Accepted for API compatibility; ignored.
        num_warps: Accepted for API compatibility; ignored.
        num_stages: Accepted for API compatibility; ignored.

    Returns:
        Attention output, shape ``[total_tokens, num_q_heads, head_dim]``.
    """
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5

    return _impl(
        queries,
        key_indices_pages,
        key_signs_pages,
        key_norms_pages,
        value_indices_pages,
        value_norms_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
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
