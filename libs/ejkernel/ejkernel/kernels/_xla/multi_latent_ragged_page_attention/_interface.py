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

"""Registry entry point for the XLA MLA ragged paged-attention kernel.

Registers ``multi_latent_ragged_page_attention`` under
``(Platform.XLA, Backend.ANY)`` and computes the default softmax scale before
forwarding to the jit-compiled implementation in ``_xla_impl_fwd``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import DEFAULT_MASK_VALUE, multi_latent_ragged_page_attention_impl


@kernel_registry.register("multi_latent_ragged_page_attention", Platform.XLA, Backend.ANY)
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
    """Compute MLA ragged paged attention using XLA (registry entry point).

    Fused operation that:

    1. Writes incoming ``keys_values`` / ``keys_pe`` tokens into the paged KV
       cache at positions derived from ``kv_lens`` and ``block_tables``.
    2. Computes causal ragged paged attention over the updated cache with
       online softmax (Flash-Attention-style numerically stable accumulation).

    This is the XLA reference implementation.  It is numerically equivalent to
    the Pallas TPU/GPU kernels and is used as the correctness baseline.

    Args:
        queries_nope: Query non-positional (compressed latent) component
            ``[total_tokens, num_q_heads, kv_latent_dim]``.
        queries_pe: Query positional component
            ``[total_tokens, num_q_heads, qk_pe_dim]``.
        keys_values: Incoming latent KV component (merged K and V in the MLA
            sense) ``[total_tokens, kv_latent_dim]``.
        keys_pe: Incoming key positional component ``[total_tokens, qk_pe_dim]``.
        kv_cache: Paged KV cache to be updated in-place (donated).
            Shape: ``[num_pages, page_size/kv_packing, kv_packing, kv_dim_padded]``
            where ``kv_dim_padded = align128(kv_latent_dim) + align128(qk_pe_dim)``.
        kv_lens: Total context length for each sequence before this step
            ``[max_num_seqs]``.  int32.
        block_tables: Flat page table ``[max_num_seqs * pages_per_seq]``.  int32.
        query_start_loc: Ragged token start offsets ``[max_num_seqs + 1]``.  int32.
        distribution: Workload descriptor ``[decode_end, prefill_end, total_seqs]``.
            Only ``distribution[2]`` (total number of active sequences) is used by
            this XLA implementation.
        softmax_scale: QK scaling factor.  Defaults to
            ``(kv_latent_dim + qk_pe_dim) ** -0.5`` when None.
        sliding_window: If set, restricts each query to attend only to the most
            recent ``sliding_window`` cached tokens.
        logits_soft_cap: If set, applies ``cap * tanh(logits / cap)`` before
            softmax.  Must be non-zero when provided.
        mask_value: Additive fill value for masked (invalid / future) positions.
            Defaults to ``-0.7 * finfo(float32).max``.
        q_scale: Optional FP8 query dequantization scale (multiplied into logits).
        k_scale: Optional FP8 key dequantization scale (multiplied into logits).
        v_scale: Optional FP8 value dequantization scale (multiplied into output).
        chunk_prefill_size: Accepted for API compatibility; ignored on this path.
        num_kv_pages_per_block: Number of KV cache pages per inner attention
            loop block.  Heuristically chosen when None.
        num_queries_per_block: Number of query tokens per outer loop block.
            Heuristically chosen when None.
        vmem_limit_bytes: Accepted for API compatibility; ignored on this path.
        debug_mode: Accepted for API compatibility; ignored on this path.

    Returns:
        Tuple ``(outputs, updated_kv_cache)`` where:
            - ``outputs``: ``[total_tokens, num_q_heads, kv_latent_dim]``,
              same dtype as ``queries_nope``.
            - ``updated_kv_cache``: same shape and dtype as the input
              ``kv_cache``.
    """
    if softmax_scale is None:
        softmax_scale = (queries_nope.shape[-1] + queries_pe.shape[-1]) ** -0.5

    return multi_latent_ragged_page_attention_impl(
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


__all__ = ("multi_latent_ragged_page_attention",)
