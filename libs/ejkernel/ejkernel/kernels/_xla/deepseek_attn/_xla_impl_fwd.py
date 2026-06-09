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

"""DeepSeek Attention (DSA) forward pass — XLA reference implementation.

Implements DeepSeek Sparse Attention from DeepSeek-V3.2 on top of MLA.
Follows the same tensor conventions as flash_mla:
  - key_value is the compressed KV latent [batch, seq_len, kv_lora_rank]
  - w_kc / w_vc project to keys/values on-the-fly
  - b_q / b_k provide optional RoPE components

The DSA algorithm:
  1. Lightning Indexer scores all KV tokens cheaply (ReLU, low-rank).
  2. Top-k token indices are selected per query position.
  3. Full MLA attention is computed with non-selected tokens masked out.

Reference:
    DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
    https://arxiv.org/abs/2512.02556
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int


def _lightning_indexer(
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    index_topk: int,
    softmax_scale: float,
    causal: bool = True,
) -> Int[Array, "batch seq_len index_topk"]:
    """Compute Lightning Indexer scores and select top-k tokens.

    The indexer computes per-head relevance scores using ReLU activation:
        I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s) * scale

    Then aggregates across heads and selects top-k tokens.

    Args:
        query_index: Indexer query projections [batch, seq_len, index_heads, index_head_dim].
        key_index: Indexer key projections [batch, seq_len, index_head_dim].
        index_weights: Per-head learned weights [batch, seq_len, index_heads].
        index_topk: Number of tokens to select per query position.
        softmax_scale: Scaling factor for indexer scores.
        causal: Whether to apply causal masking (default: True).

    Returns:
        Top-k token indices [batch, seq_len, index_topk].
    """
    _B, T, _H_I, _D_I = query_index.shape

    raw_scores = jnp.einsum("bthd,bsd->bths", query_index, key_index) * softmax_scale
    raw_scores = jax.nn.relu(raw_scores)
    index_score = jnp.einsum("bths,bth->bts", raw_scores, index_weights)

    if causal:
        q_ids = jnp.arange(T, dtype=jnp.int32)[None, :, None]
        k_ids = jnp.arange(T, dtype=jnp.int32)[None, None, :]
        future_mask = k_ids > q_ids
        index_score = jnp.where(future_mask, -jnp.inf, index_score)

    k = jnp.minimum(index_topk, T)
    _, topk_indices = lax.top_k(index_score, k)

    return topk_indices.astype(jnp.int32)


def _deepseek_attention_fwd(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    index_topk: int,
    softmax_scale: float,
    index_softmax_scale: float | None = None,
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    causal: bool = True,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """DeepSeek Sparse Attention forward pass with MLA-style inputs.

    Computes MLA attention (on-the-fly KV reconstruction from a compressed
    latent vector) with a Lightning Indexer sparse mask applied before softmax.

    Scoring modes (selected by presence of ``b_q`` / ``b_k``):

    1. **Both present** (``b_q`` and ``b_k``): The full ``query`` is treated
       as the nope component; RoPE contributions are computed separately::

           scores = (q_nope @ k_nope.T + b_q @ b_k.T) * softmax_scale

    2. **``b_k`` only**: ``query`` is split into nope ``[..., :d_nope]`` and
       rope ``[..., d_nope:]`` portions::

           scores = (q_nope @ k_nope.T + q_rope @ b_k.T) * softmax_scale

    3. **Neither** (pure nope): ``query`` is used directly::

           scores = query @ k_nope.T * softmax_scale

    After scoring, a ``[seq_len, seq_len]`` sparse mask is built from the
    Lightning Indexer top-k indices.  Non-selected positions are set to
    ``finfo.min`` (effectively −∞), while self-attention positions are
    always forced to 0 so every token can attend to itself.

    Args:
        query: Query tensor ``[batch, seq_len, q_heads, q_head_dim]``.
        key_value: Compressed KV latent ``[batch, seq_len, kv_lora_rank]``.
        w_kc: Key projection ``[kv_lora_rank, kv_heads, qk_nope_head_dim]``.
        w_vc: Value projection ``[kv_lora_rank, kv_heads, v_head_dim]``.
        query_index: Indexer query projections
            ``[batch, seq_len, index_heads, index_head_dim]``.
        key_index: Indexer key projections
            ``[batch, seq_len, index_head_dim]``.
        index_weights: Learned per-head indexer weights
            ``[batch, seq_len, index_heads]``.
        index_topk: Number of KV tokens to select per query position.
        softmax_scale: Attention softmax scaling factor.
        index_softmax_scale: Indexer scoring scale.  Defaults to
            ``index_head_dim ** -0.5`` when ``None``.
        b_q: Optional query RoPE bias ``[batch, seq_len, qk_rope_head_dim]``.
        b_k: Optional key RoPE bias ``[batch, seq_len, qk_rope_head_dim]``.
        causal: Whether to apply causal masking. Default True.

    Returns:
        Attention output ``[batch, seq_len, q_heads, v_head_dim]``.
    """
    B, T, H_Q, _D_Q = query.shape
    H_KV = w_kc.shape[1]
    d_nope = w_kc.shape[2]
    G = H_Q // H_KV

    if index_softmax_scale is None:
        index_softmax_scale = float(query_index.shape[-1] ** -0.5)

    topk_indices = _lightning_indexer(
        query_index=query_index,
        key_index=key_index,
        index_weights=index_weights,
        index_topk=index_topk,
        softmax_scale=index_softmax_scale,
        causal=causal,
    )

    k_nope = jnp.einsum("btl,lhd->bthd", key_value, w_kc)
    v = jnp.einsum("btl,lhd->bthd", key_value, w_vc)

    if b_q is not None and b_k is not None:
        q_nope = query
        k_nope_exp = jnp.repeat(k_nope, G, axis=2)
        s_nope = jnp.einsum("bshd,bthd->bhst", q_nope, k_nope_exp)
        s_rope = jnp.einsum("bsr,btr->bst", b_q, b_k)
        scores = (s_nope + s_rope[:, None, :, :]) * softmax_scale
    elif b_k is not None:
        q_nope = query[..., :d_nope]
        q_rope = query[..., d_nope:]
        k_nope_exp = jnp.repeat(k_nope, G, axis=2)
        s_nope = jnp.einsum("bshd,bthd->bhst", q_nope, k_nope_exp)
        s_rope = jnp.einsum("bshr,btr->bhst", q_rope, b_k)
        scores = (s_nope + s_rope) * softmax_scale
    else:
        k_nope_exp = jnp.repeat(k_nope, G, axis=2)
        scores = jnp.einsum("bshd,bthd->bhst", query, k_nope_exp) * softmax_scale

    large_neg = jnp.finfo(scores.dtype).min
    index_mask = jnp.full((B, T, T), large_neg, dtype=scores.dtype)
    index_mask = index_mask.at[
        jnp.arange(B)[:, None, None],
        jnp.arange(T)[None, :, None],
        topk_indices,
    ].set(0.0)

    if causal:
        q_ids = jnp.arange(T, dtype=jnp.int32)[:, None]
        k_ids = jnp.arange(T, dtype=jnp.int32)[None, :]
        future_mask = k_ids > q_ids
        causal_fill = jnp.where(future_mask, large_neg, 0.0)
        index_mask = index_mask + causal_fill[None, :, :]

    self_attend = jnp.eye(T, dtype=scores.dtype)[None, :, :]
    index_mask = jnp.where(self_attend > 0, 0.0, index_mask)

    scores = scores + index_mask[:, None, :, :]

    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_weights = jnp.where(jnp.isnan(attn_weights), 0.0, attn_weights)

    v_exp = jnp.repeat(v, G, axis=2)
    output = jnp.einsum("bhst,bthd->bshd", attn_weights, v_exp)

    return output
