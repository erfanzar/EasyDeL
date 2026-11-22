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

"""DeepSeek Sparse Attention forward pass and custom VJP wiring for TPU Pallas."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..flash_mla._pallas_impl_fwd import _flash_mla_pallas_call
from ..flash_mla._utils import DEFAULT_MASK_VALUE
from ._pallas_impl_bwd import _deepseek_attn_bwd_impl

ROPE_NONE = 0
ROPE_FUSED = 1
ROPE_DECOUPLED = 2


def _build_sparse_mask(
    topk_indices: jnp.ndarray,
    seq_len: int,
) -> jnp.ndarray:
    """Build a dense float32 keep-mask from top-k indices.

    Sets 1.0 at each selected position and guarantees the diagonal
    (self-attend) is always 1.0 regardless of the indexer output.

    Args:
        topk_indices: Token indices selected by the Lightning Indexer
            [batch, seq_q, topk].
        seq_len: Full KV sequence length (width of the output mask).

    Returns:
        Float32 mask [batch, seq_q, seq_len] with 1.0 at selected positions.
    """
    batch, seq_q, _topk = topk_indices.shape
    mask = jnp.zeros((batch, seq_q, seq_len), dtype=jnp.float32)
    mask = mask.at[
        jnp.arange(batch)[:, None, None],
        jnp.arange(seq_q)[None, :, None],
        topk_indices,
    ].set(1.0)
    self_mask = jnp.eye(seq_len, dtype=jnp.float32)[None, :, :]
    return jnp.maximum(mask, self_mask)


def _sparse_mask_to_bias(sparse_mask: jnp.ndarray) -> jnp.ndarray:
    """Convert a float keep-mask into the additive bias Flash MLA expects.

    Positions with mask > 0.5 get bias 0.0 (attend); positions with
    mask <= 0.5 get ``DEFAULT_MASK_VALUE`` (masked out before softmax).

    Args:
        sparse_mask: Float32 keep-mask [batch, seq_q, kv_len].

    Returns:
        Additive bias [batch, 1, seq_q, kv_len] broadcastable over heads.
    """
    mask_value = jnp.asarray(DEFAULT_MASK_VALUE, dtype=jnp.float32)
    return jnp.where(sparse_mask[:, None, :, :] > 0.5, 0.0, mask_value)


def _deepseek_attn_pallas_call(
    q,
    kv_latent,
    w_kc,
    w_vc,
    b_q,
    b_k,
    sparse_mask,
    *,
    rope_mode,
    causal,
    softmax_scale,
    block_b,
    block_q,
    block_k,
    save_residuals: bool = False,
):
    """Run the DeepSeek TPU forward pass via the raw Flash-MLA Pallas call.

    Converts the float keep-mask into an additive bias (with
    ``stop_gradient``) and delegates to ``_flash_mla_pallas_call``.

    Args:
        q: Query tensor [batch, q_heads, seq_q, q_head_dim].
        kv_latent: Compressed KV latent [batch, kv_len, kv_lora_rank].
        w_kc: Key projection [kv_heads, kv_lora_rank, d_nope].
        w_vc: Value projection [kv_heads, kv_lora_rank, v_head_dim].
        b_q: Optional query RoPE [batch, seq_q, rope_dim].
        b_k: Optional key RoPE [batch, kv_len, rope_dim].
        sparse_mask: Float32 keep-mask [batch, seq_q, kv_len].
        rope_mode: RoPE mode constant (ROPE_NONE/FUSED/DECOUPLED).
        causal: Whether to apply causal masking.
        softmax_scale: Attention scaling factor.
        block_b: Batch block size for Pallas grid.
        block_q: Query block size for Pallas grid.
        block_k: Key block size for Pallas grid.
        save_residuals: If True, also return (logsumexp, max) for backward.

    Returns:
        Attention output, or (output, logsumexp, max) when ``save_residuals=True``.
    """
    bias = _sparse_mask_to_bias(jax.lax.stop_gradient(sparse_mask))
    return _flash_mla_pallas_call(
        q,
        kv_latent,
        w_kc,
        w_vc,
        b_q,
        b_k,
        bias,
        rope_mode=rope_mode,
        causal=causal,
        softmax_scale=softmax_scale,
        sliding_window=None,
        logits_soft_cap=None,
        block_b=block_b,
        block_q=block_q,
        block_k=block_k,
        save_residuals=save_residuals,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(7, 13))
def deepseek_attn_pallas_impl(
    q,
    kv_latent,
    w_kc,
    w_vc,
    b_q,
    b_k,
    sparse_mask,
    rope_mode,
    causal,
    softmax_scale,
    block_b,
    block_q,
    block_k,
):
    """DeepSeek Sparse Attention forward with explicit TPU ``custom_vjp``.

    Wraps ``_deepseek_attn_pallas_call`` so that the backward pass uses
    the Flash-MLA Pallas backward kernels with the sparse bias.

    Non-diff args (indices 7-12): ``rope_mode``, ``causal``,
    ``softmax_scale``, ``block_b``, ``block_q``, ``block_k``.
    """
    return _deepseek_attn_pallas_call(
        q,
        kv_latent,
        w_kc,
        w_vc,
        b_q,
        b_k,
        sparse_mask,
        rope_mode=rope_mode,
        causal=causal,
        softmax_scale=softmax_scale,
        block_b=block_b,
        block_q=block_q,
        block_k=block_k,
    )


def _deepseek_attn_fwd(
    q,
    kv_latent,
    w_kc,
    w_vc,
    b_q,
    b_k,
    sparse_mask,
    rope_mode,
    causal,
    softmax_scale,
    block_b,
    block_q,
    block_k,
):
    """Forward rule for ``custom_vjp``: run raw forward and save residuals.

    Saves ``(q, kv_latent, w_kc, w_vc, b_q, b_k, sparse_mask, o, l, m)``
    where ``l`` and ``m`` are the per-row logsumexp and max from Flash-MLA.
    """
    o, l, m = _deepseek_attn_pallas_call(
        q,
        kv_latent,
        w_kc,
        w_vc,
        b_q,
        b_k,
        sparse_mask,
        rope_mode=rope_mode,
        causal=causal,
        softmax_scale=softmax_scale,
        block_b=block_b,
        block_q=block_q,
        block_k=block_k,
        save_residuals=True,
    )
    residuals = (q, kv_latent, w_kc, w_vc, b_q, b_k, sparse_mask, o, l, m)
    return o, residuals


def _deepseek_attn_bwd(
    rope_mode,
    causal,
    softmax_scale,
    block_b,
    block_q,
    block_k,
    residuals,
    do,
):
    """Backward rule for ``custom_vjp``.

    Delegates to ``_deepseek_attn_bwd_impl`` which converts the sparse mask
    back to an additive bias and calls the Flash-MLA backward kernels.
    Returns gradients for all seven differentiable inputs; the sparse-mask
    gradient is always zeros.
    """
    del block_b
    return _deepseek_attn_bwd_impl(
        rope_mode,
        causal,
        softmax_scale,
        block_q,
        block_k,
        residuals,
        do,
    )


deepseek_attn_pallas_impl.defvjp(_deepseek_attn_fwd, _deepseek_attn_bwd)


__all__ = (
    "ROPE_DECOUPLED",
    "ROPE_FUSED",
    "ROPE_NONE",
    "_build_sparse_mask",
    "_sparse_mask_to_bias",
    "deepseek_attn_pallas_impl",
)
