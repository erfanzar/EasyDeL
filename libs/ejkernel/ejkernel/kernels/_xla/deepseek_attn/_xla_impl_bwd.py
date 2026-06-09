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

"""Backward pass for the XLA DeepSeek Sparse Attention kernel.

The gradient computation re-runs the forward pass inside ``jax.vjp`` with
``lax.stop_gradient`` applied to the three Lightning Indexer inputs
(``query_index``, ``key_index``, ``index_weights``).  This means:

- ``query``, ``key_value``, ``w_kc``, ``w_vc``, ``b_q``, ``b_k`` receive
  meaningful gradients via JAX's autodiff.
- ``query_index``, ``key_index``, ``index_weights`` receive **zero** gradients
  because the top-k selection in the indexer is non-differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._xla_impl_fwd import _deepseek_attention_fwd


def deepseek_attn_backward(
    query,
    key_value,
    w_kc,
    w_vc,
    query_index,
    key_index,
    index_weights,
    *,
    index_topk: int,
    softmax_scale: float,
    index_softmax_scale: float | None,
    b_q,
    b_k,
    causal: bool,
    do,
):
    """Differentiate the XLA DeepSeek attention reference implementation.

    Re-runs the forward inside ``jax.vjp`` with ``stop_gradient`` applied
    to the three indexer inputs so that only ``(query, key_value, w_kc,
    w_vc, b_q, b_k)`` receive non-zero gradients.  The indexer tangents
    are set to ``zeros_like``.

    Args:
        query: Query tensor [batch, seq_len, q_heads, q_head_dim].
        key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
        w_kc: Key projection weights.
        w_vc: Value projection weights.
        query_index: Indexer queries (stopped — receives zero gradients).
        key_index: Indexer keys (stopped — receives zero gradients).
        index_weights: Indexer weights (stopped — receives zero gradients).
        index_topk: Number of tokens selected.
        softmax_scale: Attention scaling factor.
        index_softmax_scale: Indexer scaling factor.
        b_q: Optional query RoPE bias.
        b_k: Optional key RoPE bias.
        causal: Whether causal masking was applied.
        do: Upstream gradient of the attention output.

    Returns:
        Nine-element tuple of gradients matching the differentiable args
        of ``_deepseek_attn_core``.
    """

    def _forward(
        query,
        key_value,
        w_kc,
        w_vc,
        b_q,
        b_k,
    ):
        return _deepseek_attention_fwd(
            query=query,
            key_value=key_value,
            w_kc=w_kc,
            w_vc=w_vc,
            query_index=jax.lax.stop_gradient(query_index),
            key_index=jax.lax.stop_gradient(key_index),
            index_weights=jax.lax.stop_gradient(index_weights),
            index_topk=index_topk,
            softmax_scale=softmax_scale,
            index_softmax_scale=index_softmax_scale,
            b_q=b_q,
            b_k=b_k,
            causal=causal,
        )

    _, pullback = jax.vjp(
        _forward,
        query,
        key_value,
        w_kc,
        w_vc,
        b_q,
        b_k,
    )
    dq, dkey_value, dw_kc, dw_vc, db_q, db_k = pullback(do)
    return (
        dq,
        dkey_value,
        dw_kc,
        dw_vc,
        jnp.zeros_like(query_index),
        jnp.zeros_like(key_index),
        jnp.zeros_like(index_weights),
        db_q,
        db_k,
    )


__all__ = ("deepseek_attn_backward",)
