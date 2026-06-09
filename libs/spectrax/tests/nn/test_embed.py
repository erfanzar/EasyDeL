# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.embed`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.init import constant, normal
from spectrax.nn.embed import Embed
from spectrax.rng.rngs import Rngs


def test_embed_weight_shape_and_axis_names():
    """Embedding table shape is ``(vocab, features)`` with axis names set."""
    m = Embed(10, 4, rngs=Rngs(0))
    assert m.weight.shape == (10, 4)
    assert m.weight.axis_names == ("vocab", "embed")


def test_embed_lookup_single_id():
    """``lookup([id])`` returns the corresponding row."""
    m = Embed(8, 3, rngs=Rngs(0))
    ids = jnp.asarray([2])
    out = m.lookup(ids)
    assert out.shape == (1, 3)
    assert jnp.allclose(out, m.weight.value[2:3])


def test_embed_forward_is_lookup():
    """Calling the layer is equivalent to :meth:`Embed.lookup`."""
    m = Embed(8, 3, rngs=Rngs(0))
    ids = jnp.asarray([0, 1, 2])
    assert jnp.allclose(m(ids), m.lookup(ids))


def test_embed_lookup_batched():
    """Lookup preserves leading batch dimensions."""
    m = Embed(16, 4, rngs=Rngs(0))
    ids = jnp.zeros((3, 5), dtype=jnp.int32)
    out = m.lookup(ids)
    assert out.shape == (3, 5, 4)


def test_embed_attend_produces_logits():
    """``attend(q)`` computes ``q @ W.T`` for classification heads."""
    m = Embed(5, 3, rngs=Rngs(0), w_init=constant(1.0))
    q = jnp.ones((2, 3))
    logits = m.attend(q)
    assert logits.shape == (2, 5)
    assert jnp.allclose(logits, jnp.full((2, 5), 3.0))


def test_embed_custom_initializer():
    """Custom ``w_init`` controls the table initialization."""
    m = Embed(4, 2, rngs=Rngs(0), w_init=constant(7.0))
    assert jnp.all(m.weight.value == 7.0)


def test_embed_default_init_not_zero():
    """The default normal init yields non-degenerate weights."""
    m = Embed(4, 2, rngs=Rngs(0))
    assert not jnp.all(m.weight.value == 0)


def test_embed_dtype_controls_storage():
    """``dtype`` sets the table's storage dtype."""
    m = Embed(4, 2, rngs=Rngs(0), dtype=jnp.float16)
    assert m.weight.dtype == jnp.float16


def test_embed_attend_matches_manual_matmul():
    """``attend`` is equivalent to ``q @ W.T``."""
    m = Embed(4, 3, rngs=Rngs(0), w_init=normal(stddev=0.1))
    q = jnp.ones((1, 3))
    assert jnp.allclose(m.attend(q), q @ m.weight.value.T)
