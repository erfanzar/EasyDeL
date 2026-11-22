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

"""Backward utilities for DeepSeek Sparse Attention on TPU Pallas."""

from __future__ import annotations

import jax.numpy as jnp

from ..flash_mla._pallas_impl_bwd import _flash_mla_bwd_impl
from ..flash_mla._utils import DEFAULT_MASK_VALUE


def _sparse_mask_to_bias(sparse_mask):
    """Convert a float keep-mask into an additive bias for Flash MLA backward.

    Same logic as the forward ``_sparse_mask_to_bias`` — kept as a
    separate copy to avoid a circular import between fwd and bwd modules.

    Args:
        sparse_mask: Float32 keep-mask [batch, seq_q, kv_len].

    Returns:
        Additive bias [batch, 1, seq_q, kv_len].
    """
    mask_value = jnp.asarray(DEFAULT_MASK_VALUE, dtype=jnp.float32)
    return jnp.where(sparse_mask[:, None, :, :] > 0.5, 0.0, mask_value)


def _deepseek_attn_bwd_impl(
    rope_mode,
    causal,
    softmax_scale,
    block_q,
    block_k,
    residuals,
    do,
):
    """Compute DeepSeek TPU gradients using the Flash-MLA backward kernels.

    Converts the keep-mask to an additive bias and delegates to
    ``_flash_mla_bwd_impl``.  The bias gradient from Flash-MLA is
    discarded and replaced with ``zeros_like(sparse_mask)`` because
    the top-k selection is non-differentiable.

    Args:
        rope_mode: RoPE mode constant (ROPE_NONE/FUSED/DECOUPLED).
        causal: Whether causal masking was applied.
        softmax_scale: Attention scaling factor.
        block_q: Query block size for the backward kernel.
        block_k: Key block size for the backward kernel.
        residuals: Tuple ``(q, kv_latent, w_kc, w_vc, b_q, b_k,
            sparse_mask, o, l, m)`` saved during forward.
        do: Upstream gradient of the attention output.

    Returns:
        Seven-element tuple of gradients: ``(dq, dkv_latent, dw_kc, dw_vc,
        db_q, db_k, d_sparse_mask)`` where ``d_sparse_mask`` is always zeros.
    """
    q, kv_latent, w_kc, w_vc, b_q, b_k, sparse_mask, o, l, m = residuals
    bias = _sparse_mask_to_bias(sparse_mask)

    dq, dkv_latent, dw_kc, dw_vc, db_q, db_k, _dbias = _flash_mla_bwd_impl(
        rope_mode,
        causal,
        softmax_scale,
        None,
        None,
        block_q,
        block_k,
        (q, kv_latent, w_kc, w_vc, b_q, b_k, bias, o, l, m),
        do,
    )

    return (
        dq,
        dkv_latent,
        dw_kc,
        dw_vc,
        db_q,
        db_k,
        jnp.zeros_like(sparse_mask),
    )


__all__ = ("_deepseek_attn_bwd_impl",)
