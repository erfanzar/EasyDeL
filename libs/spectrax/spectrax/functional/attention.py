# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scaled dot-product attention primitive.

Implements the standard ``softmax(QK^T / sqrt(d)) V`` flavor used by
multi-head attention. Inputs are expected in the
``(..., seq, head_dim)`` layout that SpectraX uses everywhere — the
trailing two axes are the per-token feature axis and the per-head
features; all leading axes (batch, head, etc.) broadcast and are
preserved in the output. Optional masks, causality, custom scale, and
attention-weight dropout are all supported in a single fused path.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..common_types import DEFAULT_MASK_VALUE
from ..core._typing import Array, ArrayLike, PRNGKey


def scaled_dot_product_attention(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    dropout_rate: float = 0.0,
    key: PRNGKey | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> Array:
    """Compute multi-head scaled dot-product attention.

    Treats the trailing two axes of ``q`` / ``k`` / ``v`` as
    ``(seq, head_dim)``; all leading axes are batch axes that are
    broadcast and preserved in the output.

    Args:
        q: Query tensor, shape ``(..., seq_q, head_dim)``.
        k: Key tensor, shape ``(..., seq_k, head_dim)``.
        v: Value tensor, shape ``(..., seq_k, head_dim)``.
        mask: Optional mask broadcastable to ``(..., seq_q, seq_k)``.
            ``bool`` masks zero out disallowed positions; floating
            masks are added to the logits.
        dropout_rate: Attention-weight dropout probability.
        key: PRNG key for dropout, required when ``dropout_rate > 0``.
        is_causal: Apply a lower-triangular causal mask on top of
            ``mask``.
        scale: Logit scale (defaults to ``1 / sqrt(head_dim)``).

    Returns:
        Attention output of shape ``(..., seq_q, head_dim)``.
    """
    qa = jnp.asarray(q)
    ka = jnp.asarray(k)
    va = jnp.asarray(v)
    d = qa.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(d)
    logits = jnp.einsum("...qd,...kd->...qk", qa, ka) * s
    mask_fill = jnp.asarray(DEFAULT_MASK_VALUE, dtype=logits.dtype)
    if is_causal:
        q_len = qa.shape[-2]
        k_len = ka.shape[-2]
        causal = jnp.tril(jnp.ones((q_len, k_len), dtype=jnp.bool_), k=k_len - q_len)
        logits = jnp.where(causal, logits, mask_fill)
    if mask is not None:
        m = jnp.asarray(mask)
        if m.dtype == jnp.bool_:
            logits = jnp.where(m, logits, mask_fill)
        else:
            logits = logits + m
    attn = jax.nn.softmax(logits, axis=-1)
    if dropout_rate > 0.0 and key is not None:
        keep_rate = 1.0 - dropout_rate
        drop_mask = jax.random.bernoulli(key, keep_rate, shape=attn.shape)
        attn = jnp.where(drop_mask, attn / keep_rate, 0.0)
    return jnp.einsum("...qk,...kd->...qd", attn, va)
