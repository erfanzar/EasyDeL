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
"""Registry entry point for the XLA prefill paged attention kernel.

Registers ``prefill_page_attention`` under ``(Platform.XLA, Backend.ANY)``
and delegates to the core implementation in ``_impl``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._impl import DEFAULT_MASK_VALUE, Array, Float, Int
from ._impl import prefill_page_attention as _prefill_page_attention_impl


@kernel_registry.register("prefill_page_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def prefill_page_attention(
    query: Float[Array, "chunk_size num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_len: Int[Array, "1"],
    page_indices: Int[Array, "num_pages"],
    *,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "chunk_size num_heads head_dim"]:
    """Compute chunked prefill paged attention using XLA operations (registry entry point).

    Registry wrapper; see ``_impl.prefill_page_attention`` for the full
    algorithm description.

    Args:
        query: Query chunk ``[chunk_size, num_q_heads, head_dim]``.
        key_cache: Full paged key cache ``[num_kv_heads, total_pages, page_size, head_dim]``.
        value_cache: Full paged value cache, same shape as ``key_cache``.
        context_len: Total context length (including this chunk) as a
            length-1 integer array ``[1]``.  Query positions are inferred as
            ``context_len[0] - chunk_size + arange(chunk_size)``.
        page_indices: Physical page indices for this sequence ``[num_pages]``.
        softmax_scale: QK scaling factor.  Defaults to ``1 / sqrt(head_dim)``.
        mask_value: Additive fill value for masked positions.  Defaults to
            ``-0.7 * finfo(float32).max``.
        attn_logits_soft_cap: If set, applies ``cap * tanh(logits / cap)``
            before softmax.
        sliding_window: If set, restricts attention to the last
            ``sliding_window`` cached tokens per query position.
        block_k: Backend tuning hint accepted for operation-level autotune;
            ignored by XLA.
        num_warps: Backend tuning hint accepted for operation-level autotune;
            ignored by XLA.
        num_stages: Backend tuning hint accepted for operation-level autotune;
            ignored by XLA.

    Returns:
        Attention output ``[chunk_size, num_q_heads, head_dim]``,
        same dtype as ``query``.

    Note:
        - GQA is supported: ``num_q_heads`` must be a multiple of ``num_kv_heads``.
        - Causal masking is always active.
        - Page size is inferred from ``key_cache.shape[2]``.
    """
    _ = block_k, num_warps, num_stages
    return _prefill_page_attention_impl(
        query,
        key_cache,
        value_cache,
        context_len,
        page_indices,
        softmax_scale=softmax_scale,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window=sliding_window,
    )


__all__ = ("prefill_page_attention",)
