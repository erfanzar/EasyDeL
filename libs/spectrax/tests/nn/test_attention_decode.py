# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for KV-cache decode path on :class:`MultiheadAttention`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.core.variable import Buffer
from spectrax.nn.attention import MultiheadAttention
from spectrax.rng.rngs import Rngs
from spectrax.transforms.jit import jit


def test_decode_false_preserves_existing_behavior():
    """With ``decode=False`` the layer is unchanged."""
    attn = MultiheadAttention(8, 2, decode=False, rngs=Rngs(0))
    x = jnp.ones((1, 4, 8))
    y = attn(x)
    assert y.shape == x.shape


def test_init_cache_allocates_buffers_with_right_shapes():
    """``init_cache`` creates ``cache_k``/``cache_v``/``cache_index`` of expected shape."""
    attn = MultiheadAttention(8, 2, decode=True, rngs=Rngs(0))
    attn.init_cache(batch_shape=(1,), max_length=4)
    assert isinstance(attn.cache_k, Buffer)
    assert attn.cache_k.value.shape == (1, 2, 4, 4)
    assert attn.cache_v.value.shape == (1, 2, 4, 4)
    assert attn.cache_index.value.shape == ()
    assert attn.cache_k.kind == "cache"


def test_decode_without_init_cache_raises():
    """Decode mode without allocated buffers raises at call time."""
    attn = MultiheadAttention(8, 2, decode=True, rngs=Rngs(0))
    x = jnp.ones((1, 1, 8))
    with pytest.raises(RuntimeError):
        attn(x)


def test_decode_single_step_advances_cache_index():
    """A single decode call advances ``cache_index`` by the number of time steps."""
    attn = MultiheadAttention(4, 2, decode=True, rngs=Rngs(0))
    attn.init_cache(batch_shape=(1,), max_length=4)
    x = jnp.ones((1, 1, 4))
    _ = attn(x)
    assert int(attn.cache_index.value) == 1


def test_decode_matches_full_sequence_output_at_position_t():
    """Decoding step-by-step matches full self-attention on the same input prefix."""
    full = MultiheadAttention(4, 2, decode=False, rngs=Rngs(0))
    dec = MultiheadAttention(4, 2, decode=True, rngs=Rngs(0))
    for k in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        getattr(dec, k).weight.value = getattr(full, k).weight.value
        getattr(dec, k).bias.value = getattr(full, k).bias.value
    dec.init_cache(batch_shape=(1,), max_length=3)
    seq = jnp.arange(1 * 3 * 4.0).reshape((1, 3, 4))
    with jax.default_matmul_precision("float32"):
        full_out = full(seq, is_causal=True)
        step_outs = []
        for t in range(3):
            step_outs.append(dec(seq[:, t : t + 1, :]))
    dec_out = jnp.concatenate(step_outs, axis=1)
    assert jnp.allclose(full_out, dec_out, atol=1e-2)


def test_jit_decode_advances_cache_under_mutable_cache():
    """Under ``jit(mutable='cache')`` each call advances the cache index."""
    attn = MultiheadAttention(4, 2, decode=True, rngs=Rngs(0))
    attn.init_cache(batch_shape=(1,), max_length=3)

    @jit(mutable="cache")
    def step(layer, tok):
        """Execute one training step and return the result."""
        return layer(tok)

    _ = step(attn, jnp.ones((1, 1, 4)))
    assert int(attn.cache_index.value) == 1
    _ = step(attn, jnp.ones((1, 1, 4)))
    assert int(attn.cache_index.value) == 2
