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

    Raises:
        ValueError: If dropout is enabled without a PRNG key, or the
            dropout probability is outside ``[0, 1)``.
    """
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    if dropout_rate > 0.0 and key is None:
        raise ValueError("dropout_rate > 0 requires a PRNG key")
    qa = jnp.asarray(q)
    ka = jnp.asarray(k)
    va = jnp.asarray(v)
    d = qa.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(d)
    out_dtype = jnp.result_type(qa.dtype, ka.dtype, va.dtype)
    # Logits, softmax and the value reduction accumulate in at least float32, never in a
    # bf16/fp16 activation dtype; the result is cast back to the input dtype.
    acc_dtype = jnp.promote_types(out_dtype, jnp.float32)
    if not jnp.issubdtype(out_dtype, jnp.inexact):
        # Integral / bool inputs: attention outputs are fractional averages, so return the
        # floating accumulation dtype instead of truncating to an integer (r11 S14-1).
        out_dtype = acc_dtype
    logits = jnp.einsum("...qd,...kd->...qk", qa, ka, preferred_element_type=acc_dtype).astype(acc_dtype) * s
    mask_fill = jnp.asarray(DEFAULT_MASK_VALUE, dtype=acc_dtype)
    allowed = None
    if is_causal:
        q_len = qa.shape[-2]
        k_len = ka.shape[-2]
        allowed = jnp.tril(jnp.ones((q_len, k_len), dtype=jnp.bool_), k=k_len - q_len)
    if mask is not None:
        m = jnp.asarray(mask)
        if m.dtype == jnp.bool_:
            allowed = m if allowed is None else jnp.logical_and(allowed, m)
        else:
            logits = logits + m.astype(acc_dtype)
    if allowed is not None:
        logits = jnp.where(allowed, logits, mask_fill)
    attn = jax.nn.softmax(logits, axis=-1)
    if allowed is not None:
        # A row with no allowed key would otherwise softmax the uniform fill value and
        # average disallowed values; it must produce zero context instead.
        attn = jnp.where(jnp.any(allowed, axis=-1, keepdims=True), attn, 0.0)
    if dropout_rate > 0.0 and key is not None:
        keep_rate = 1.0 - dropout_rate
        drop_mask = jax.random.bernoulli(key, keep_rate, shape=attn.shape)
        attn = jnp.where(drop_mask, attn / keep_rate, 0.0)
    out = jnp.einsum("...qk,...kd->...qd", attn, va.astype(acc_dtype))
    return out.astype(out_dtype)
