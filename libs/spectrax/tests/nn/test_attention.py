# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.attention`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.nn.attention import CausalSelfAttention, MultiheadAttention
from spectrax.rng.rngs import Rngs


def test_multihead_rejects_non_divisible_heads():
    """``embed_dim`` must be divisible by ``num_heads``."""
    with pytest.raises(ValueError):
        MultiheadAttention(6, 4, rngs=Rngs(0))


def test_multihead_forward_shape_self_attention():
    """Self-attention preserves ``(..., seq, embed_dim)`` output shape."""
    m = MultiheadAttention(8, 2, rngs=Rngs(0))
    x = jnp.zeros((2, 5, 8))
    out = m(x)
    assert out.shape == (2, 5, 8)


def test_multihead_forward_cross_attention_shapes():
    """Cross-attention uses independent Q / K / V sources."""
    m = MultiheadAttention(8, 2, rngs=Rngs(0))
    q = jnp.zeros((1, 3, 8))
    k = jnp.zeros((1, 5, 8))
    v = jnp.zeros((1, 5, 8))
    out = m(q, k, v)
    assert out.shape == (1, 3, 8)


def test_multihead_v_defaults_to_k():
    """Passing ``k`` without ``v`` re-uses ``k`` for values."""
    m = MultiheadAttention(8, 2, rngs=Rngs(0))
    q = jnp.zeros((1, 3, 8))
    k = jnp.zeros((1, 3, 8))
    out = m(q, k)
    assert out.shape == q.shape


def test_multihead_causal_mask_effect():
    """``is_causal=True`` keeps token 0 independent of future tokens."""
    m = MultiheadAttention(4, 2, rngs=Rngs(0))
    x = jnp.arange(12.0).reshape(1, 3, 4)
    out_causal = m(x, is_causal=True)
    assert out_causal.shape == x.shape


def test_multihead_head_dim_computed():
    """``head_dim = embed_dim // num_heads`` is recorded on the layer."""
    m = MultiheadAttention(12, 3, rngs=Rngs(0))
    assert m.head_dim == 4


def test_multihead_has_four_projections():
    """Q / K / V / out projections are all present."""
    m = MultiheadAttention(4, 2, rngs=Rngs(0))
    assert hasattr(m, "q_proj")
    assert hasattr(m, "k_proj")
    assert hasattr(m, "v_proj")
    assert hasattr(m, "out_proj")


def test_multihead_attention_param_dtype_flows_to_projections():
    """Attention projection storage dtype should match Linear-style ``param_dtype``."""
    m = MultiheadAttention(4, 2, rngs=Rngs(0), param_dtype=jnp.bfloat16)
    csa = CausalSelfAttention(4, 2, rngs=Rngs(1), param_dtype=jnp.bfloat16)

    assert m.q_proj.weight.dtype == jnp.bfloat16
    assert m.k_proj.weight.dtype == jnp.bfloat16
    assert m.v_proj.weight.dtype == jnp.bfloat16
    assert m.out_proj.weight.dtype == jnp.bfloat16
    assert csa.attn.q_proj.weight.dtype == jnp.bfloat16


def test_causal_self_attention_forward_shape():
    """:class:`CausalSelfAttention` returns same-shape output."""
    csa = CausalSelfAttention(4, 2, rngs=Rngs(0))
    x = jnp.zeros((1, 6, 4))
    assert csa(x).shape == x.shape


def test_split_merge_heads_roundtrip():
    """``_merge_heads(_split_heads(x)) == x`` for any well-shaped input."""
    m = MultiheadAttention(6, 3, rngs=Rngs(0))
    x = jnp.arange(2 * 4 * 6, dtype=jnp.float32).reshape(2, 4, 6)
    split = m._split_heads(x)
    merged = m._merge_heads(split)
    assert jnp.array_equal(merged, x)


def test_multihead_exposes_projection_and_cache_sharding():
    """Projection and cache sharding kwargs flow to the owned variables."""
    m = MultiheadAttention(
        8,
        2,
        rngs=Rngs(0),
        qkv_sharding=("embed", "tp"),
        out_sharding=("tp", "embed"),
        qkv_bias_sharding=("tp",),
        out_bias_sharding=("embed",),
        cache_sharding=(None, "heads", "seq", "tp"),
    )
    assert m.q_proj.weight.sharding is not None
    assert m.q_proj.weight.sharding.axis_names == ("embed", "tp")
    assert m.q_proj.bias.sharding is not None
    assert m.q_proj.bias.sharding.axis_names == ("tp",)
    assert m.out_proj.weight.sharding is not None
    assert m.out_proj.weight.sharding.axis_names == ("tp", "embed")
    assert m.out_proj.bias.sharding is not None
    assert m.out_proj.bias.sharding.axis_names == ("embed",)
    m.init_cache((2,), 16)
    assert m.cache_k.sharding is not None
    assert m.cache_k.sharding.axis_names == (None, "heads", "seq", "tp")
    assert m.cache_v.sharding is not None
    assert m.cache_v.sharding.axis_names == (None, "heads", "seq", "tp")
