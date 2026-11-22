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

"""DeepSeek Sparse Attention (DSA) — Pallas TPU interface.

Two-phase implementation:
  Phase 1 (XLA): Lightning Indexer selects top-k tokens.
  Phase 2 (Pallas): Tiled MLA attention with sparse mask.

Reference:
    DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.deepseek_attn._xla_impl_fwd import _lightning_indexer
from ._pallas_impl_fwd import (
    ROPE_DECOUPLED,
    ROPE_FUSED,
    ROPE_NONE,
    _build_sparse_mask,
    deepseek_attn_pallas_impl,
)


@kernel_registry.register("deepseek_attn", Platform.PALLAS, Backend.TPU)
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
    """DeepSeek Sparse Attention on TPU using Pallas.

    Phase 1 runs the Lightning Indexer (XLA) to select top-k tokens.
    Phase 2 runs a tiled Pallas MLA kernel with the sparse mask applied.

    Args:
        query: Query tensor [batch, seq_len, q_heads, q_head_dim].
        key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
        w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim].
        w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim].
        query_index: Indexer queries [batch, seq_len, index_heads, index_head_dim].
        key_index: Indexer keys [batch, seq_len, index_head_dim].
        index_weights: Per-head weights [batch, seq_len, index_heads].
        index_topk: Number of tokens to select (default: 2048).
        softmax_scale: Attention scale (default: effective_head_dim^-0.5).
        index_softmax_scale: Indexer scale (default: index_head_dim^-0.5).
        b_q: Optional query RoPE [batch, seq_len, qk_rope_head_dim].
        b_k: Optional key RoPE [batch, seq_len, qk_rope_head_dim].
        causal: Whether to apply causal masking (default: True).

    Returns:
        Attention output [batch, seq_len, q_heads, v_head_dim].
    """
    _batch, seq_q, q_heads, q_head_dim = query.shape
    kv_len = key_value.shape[1]
    kv_heads = int(w_kc.shape[1])
    d_nope = int(w_kc.shape[2])

    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads}).")

    if b_k is None:
        rope_mode = ROPE_NONE
        if q_head_dim != d_nope:
            raise ValueError(f"No RoPE: query head_dim ({q_head_dim}) must equal d_nope ({d_nope}).")
        effective_dim = d_nope
    elif b_q is None:
        rope_mode = ROPE_FUSED
        rope_dim = int(b_k.shape[2])
        if q_head_dim != d_nope + rope_dim:
            raise ValueError(
                f"Fused RoPE: query head_dim ({q_head_dim}) must equal d_nope+rope_dim ({d_nope + rope_dim})."
            )
        effective_dim = d_nope + rope_dim
    else:
        rope_mode = ROPE_DECOUPLED
        rope_dim = int(b_k.shape[2])
        if b_q.shape[2] != rope_dim:
            raise ValueError(f"b_q/b_k rope_dim mismatch: {b_q.shape[2]} vs {rope_dim}.")
        effective_dim = d_nope + rope_dim

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(effective_dim)

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

    sparse_mask = _build_sparse_mask(topk_indices, kv_len)

    block_q = min(128, seq_q)
    block_k = min(128, kv_len)
    block_b = 1

    while seq_q % block_q != 0 and block_q > 64:
        block_q //= 2
    while kv_len % block_k != 0 and block_k > 64:
        block_k //= 2

    q_t = query.transpose(0, 2, 1, 3)
    w_kc_t = jnp.transpose(w_kc, (1, 0, 2))
    w_vc_t = jnp.transpose(w_vc, (1, 0, 2))

    o = deepseek_attn_pallas_impl(
        q_t,
        key_value,
        w_kc_t,
        w_vc_t,
        b_q,
        b_k,
        sparse_mask,
        rope_mode,
        causal,
        softmax_scale,
        block_b,
        block_q,
        block_k,
    )

    return o.transpose(0, 2, 1, 3)
