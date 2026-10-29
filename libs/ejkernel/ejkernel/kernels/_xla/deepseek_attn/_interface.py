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

"""DeepSeek Attention (DSA) public interface — XLA backend.

This module exposes the XLA reference implementation of DeepSeek Sparse
Attention, combining MLA (compressed KV latent + on-the-fly projection)
with a Lightning Indexer for dynamic top-k token selection.

Reference:
    DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
    https://arxiv.org/abs/2512.02556
"""

from __future__ import annotations

from functools import partial

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import deepseek_attn_backward
from ._xla_impl_fwd import _deepseek_attention_fwd


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 12))
def _deepseek_attn_core(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    index_topk: int,
    softmax_scale: float,
    index_softmax_scale: float | None,
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None,
    causal: bool,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Core DSA forward wrapped with ``custom_vjp``.

    Indexer inputs (``query_index``, ``key_index``, ``index_weights``) are
    differentiable arguments here but receive zero gradients in the backward
    rule because the top-k selection is non-differentiable.

    Non-diff arguments (``index_topk``, ``softmax_scale``,
    ``index_softmax_scale``, ``causal``) are forwarded unchanged to both
    the forward and backward rules.
    """
    return _deepseek_attention_fwd(
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        query_index=query_index,
        key_index=key_index,
        index_weights=index_weights,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        index_softmax_scale=index_softmax_scale,
        b_q=b_q,
        b_k=b_k,
        causal=causal,
    )


def _deepseek_attn_core_fwd(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    index_topk: int,
    softmax_scale: float,
    index_softmax_scale: float | None,
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None,
    causal: bool,
):
    """Forward rule for ``custom_vjp``: run the full DSA forward and save residuals.

    Saves ``(query, key_value, w_kc, w_vc, query_index, key_index,
    index_weights, b_q, b_k)`` for the backward pass.
    """
    out = _deepseek_attention_fwd(
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        query_index=query_index,
        key_index=key_index,
        index_weights=index_weights,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        index_softmax_scale=index_softmax_scale,
        b_q=b_q,
        b_k=b_k,
        causal=causal,
    )
    residuals = (query, key_value, w_kc, w_vc, query_index, key_index, index_weights, b_q, b_k)
    return out, residuals


def _deepseek_attn_core_bwd(
    index_topk: int,
    softmax_scale: float,
    index_softmax_scale: float | None,
    causal: bool,
    residuals,
    do: Float[Array, "batch seq_len q_heads v_head_dim"],
):
    """Backward rule for ``custom_vjp``.

    Delegates to ``deepseek_attn_backward`` which re-runs the forward
    inside ``jax.vjp`` with ``stop_gradient`` on the indexer inputs,
    returning zero tangents for ``query_index``, ``key_index``, and
    ``index_weights``.
    """
    query, key_value, w_kc, w_vc, query_index, key_index, index_weights, b_q, b_k = residuals
    return deepseek_attn_backward(
        query,
        key_value,
        w_kc,
        w_vc,
        query_index,
        key_index,
        index_weights,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        index_softmax_scale=index_softmax_scale,
        b_q=b_q,
        b_k=b_k,
        causal=causal,
        do=do,
    )


_deepseek_attn_core.defvjp(_deepseek_attn_core_fwd, _deepseek_attn_core_bwd)


@kernel_registry.register("deepseek_attn", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def deepseek_attn(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    index_topk: int = 2048,
    softmax_scale: float | None = None,
    index_softmax_scale: float | None = None,
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    causal: bool = True,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """DeepSeek Sparse Attention (DSA) with MLA + Lightning Indexer — XLA backend.

    Combines MLA attention (compressed KV latent with on-the-fly projection)
    with a lightweight learned indexer that dynamically selects the top-k most
    relevant KV tokens per query position.

    MLA Inputs (same convention as flash_mla):
        - key_value: Compressed KV latent [batch, seq_len, kv_lora_rank]
        - w_kc: Key projection weights [kv_lora_rank, kv_heads, qk_nope_head_dim]
        - w_vc: Value projection weights [kv_lora_rank, kv_heads, v_head_dim]
        - b_q, b_k: Optional RoPE bias tensors

    DSA Inputs (Lightning Indexer):
        - query_index: Indexer query projections (derived from MLA's q_lora)
        - key_index: Indexer key projections (from hidden states, shared across heads)
        - index_weights: Learned per-head aggregation weights
        - index_topk: Number of tokens to select (default: 2048)

    Args:
        query: Query tensor [batch, seq_len, q_heads, q_head_dim].
        key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
        w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim].
        w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim].
        query_index: Indexer queries [batch, seq_len, index_heads, index_head_dim].
        key_index: Indexer keys [batch, seq_len, index_head_dim].
        index_weights: Per-head weights [batch, seq_len, index_heads].
        index_topk: Number of KV tokens to select per query (default: 2048).
        softmax_scale: Attention scale (default: effective_head_dim^-0.5).
        index_softmax_scale: Indexer scale (default: index_head_dim^-0.5).
        b_q: Optional query RoPE [batch, seq_len, qk_rope_head_dim].
        b_k: Optional key RoPE [batch, seq_len, qk_rope_head_dim].
        causal: Whether to apply causal masking (default: True).

    Returns:
        Attention output [batch, seq_len, q_heads, v_head_dim].

    References:
        - DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
        - DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
    """
    if softmax_scale is None:
        d_nope = w_kc.shape[2]
        if b_k is not None:
            rope_dim = b_k.shape[2]
            effective_dim = d_nope + rope_dim
        else:
            effective_dim = d_nope
        softmax_scale = float(effective_dim**-0.5)

    return _deepseek_attn_core(
        query,
        key_value,
        w_kc,
        w_vc,
        query_index,
        key_index,
        index_weights,
        index_topk,
        softmax_scale,
        index_softmax_scale,
        b_q,
        b_k,
        causal,
    )
