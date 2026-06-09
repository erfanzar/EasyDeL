# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :func:`spectrax.functional.scaled_dot_product_attention`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectrax.functional.attention import scaled_dot_product_attention as sdpa


def test_sdpa_output_shape():
    """Output preserves ``(..., seq_q, head_dim)``."""
    q = jnp.zeros((1, 2, 5, 4))
    k = jnp.zeros((1, 2, 7, 4))
    v = jnp.zeros((1, 2, 7, 4))
    assert sdpa(q, k, v).shape == (1, 2, 5, 4)


def test_sdpa_attends_to_single_nonzero_key():
    """With one non-zero key, softmax concentrates on that key's value."""
    q = jnp.asarray([[1.0, 0.0]])
    k = jnp.asarray([[100.0, 0.0], [-100.0, 0.0]])
    v = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    out = sdpa(q, k, v)
    assert jnp.allclose(out, jnp.asarray([[1.0, 2.0]]), atol=1e-3)


def test_sdpa_bool_mask_zeros_disallowed():
    """A boolean mask ``False`` blocks attention to that key."""
    q = jnp.asarray([[1.0]])
    k = jnp.asarray([[1.0], [1.0]])
    v = jnp.asarray([[10.0], [100.0]])
    mask = jnp.asarray([[True, False]])
    out = sdpa(q, k, v, mask=mask)
    assert jnp.allclose(out, jnp.asarray([[10.0]]))


def test_sdpa_float_mask_adds_to_logits():
    """A float mask is added to the logits."""
    q = jnp.asarray([[1.0]])
    k = jnp.asarray([[1.0], [1.0]])
    v = jnp.asarray([[10.0], [20.0]])
    mask = jnp.asarray([[0.0, -1e9]])
    out = sdpa(q, k, v, mask=mask)
    assert jnp.allclose(out, jnp.asarray([[10.0]]), atol=1e-3)


def test_sdpa_is_causal_blocks_future():
    """``is_causal`` makes position 0 ignore key 1."""
    q = jnp.asarray([[1.0], [1.0]])
    k = jnp.asarray([[1.0], [1.0]])
    v = jnp.asarray([[10.0], [20.0]])
    out = sdpa(q, k, v, is_causal=True)
    assert jnp.allclose(out[0], 10.0)


def test_sdpa_causal_mask_aligns_decode_prefix_when_q_len_differs_from_k_len():
    """Causal cross-attention should align queries to the right edge of keys."""
    q = jnp.zeros((2, 1))
    k = jnp.zeros((5, 1))
    v = jnp.asarray([[10.0], [20.0], [30.0], [40.0], [50.0]])

    out = sdpa(q, k, v, is_causal=True, scale=0.0)

    assert jnp.allclose(out[:, 0], jnp.asarray([25.0, 30.0]))


def test_sdpa_custom_scale():
    """A custom scale is applied in place of ``1/sqrt(d)``."""
    q = jnp.ones((1, 1, 4))
    k = jnp.ones((1, 1, 4))
    v = jnp.ones((1, 1, 4))
    out_default = sdpa(q, k, v)
    out_scaled = sdpa(q, k, v, scale=0.0)
    assert jnp.allclose(out_default, out_scaled)


def test_sdpa_dropout_is_reproducible_with_key():
    """With dropout, the same key produces the same output."""
    q = jnp.ones((1, 4, 2))
    k = jnp.ones((1, 4, 2))
    v = jnp.arange(8.0).reshape(1, 4, 2)
    out_a = sdpa(q, k, v, dropout_rate=0.5, key=jax.random.PRNGKey(7))
    out_b = sdpa(q, k, v, dropout_rate=0.5, key=jax.random.PRNGKey(7))
    assert jnp.array_equal(out_a, out_b)
