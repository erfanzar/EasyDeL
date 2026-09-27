# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :func:`spectrax.functional.scaled_dot_product_attention`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
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


def test_sdpa_fully_masked_row_returns_zero_context():
    """A row with no allowed key must not average disallowed values (r10 S14-01)."""
    q = jnp.asarray([[1.0], [1.0]])
    k = jnp.asarray([[1.0], [1.0]])
    v = jnp.asarray([[10.0], [20.0]])
    mask = jnp.asarray([[True, True], [False, False]])
    out = sdpa(q, k, v, mask=mask)
    assert jnp.allclose(out[0], 15.0)
    assert jnp.array_equal(out[1], jnp.zeros((1,)))


def test_sdpa_integral_inputs_return_floating_context():
    """Integer/bool inputs must not truncate the fractional attention average (r11 S14-1)."""
    q = jnp.asarray([[0]], dtype=jnp.int32)
    k = jnp.asarray([[0], [0]], dtype=jnp.int32)
    v = jnp.asarray([[1], [2]], dtype=jnp.int32)
    out = sdpa(q, k, v)
    assert jnp.issubdtype(out.dtype, jnp.floating), out.dtype
    assert jnp.allclose(out, jnp.asarray([[1.5]]))  # equal logits -> mean of 1 and 2, not int 1

    qb = jnp.asarray([[False]])
    kb = jnp.asarray([[False], [False]])
    vb = jnp.asarray([[True], [False]])
    outb = sdpa(qb, kb, vb)
    assert jnp.issubdtype(outb.dtype, jnp.floating), outb.dtype
    assert jnp.allclose(outb, jnp.asarray([[0.5]]))


def test_sdpa_floating_inputs_keep_their_dtype():
    """The integral-output fix must not change the dtype of floating inputs."""
    q = jnp.ones((1, 3, 4), jnp.bfloat16)
    assert sdpa(q, q, q).dtype == jnp.bfloat16
    q32 = jnp.ones((1, 3, 4), jnp.float32)
    assert sdpa(q32, q32, q32).dtype == jnp.float32


def test_sdpa_low_precision_softmax_accumulates_in_float32():
    """fp16 inputs over 70k equal keys: a float16 softmax sum overflows (>65504) and zeroes the
    output; float32 accumulation keeps it at the mean value 1 (r10 S14-02)."""
    n = 70000
    q = jnp.zeros((1, 4), dtype=jnp.float16)
    k = jnp.zeros((n, 4), dtype=jnp.float16)
    v = jnp.ones((n, 4), dtype=jnp.float16)
    out = sdpa(q, k, v)
    assert out.dtype == jnp.float16
    assert jnp.allclose(out.astype(jnp.float32), 1.0, atol=1e-2), out


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


def test_sdpa_dropout_without_key_is_rejected():
    """Enabling dropout without a key must never silently disable regularization."""
    q = jnp.ones((1, 2, 2))
    with pytest.raises(ValueError, match="requires a PRNG key"):
        sdpa(q, q, q, dropout_rate=0.5)


def test_sdpa_dropout_probability_one_is_rejected():
    """Dropout of 1 would divide by zero in the attention-weight scaling."""
    q = jnp.ones((1, 2, 2))
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        sdpa(q, q, q, dropout_rate=1.0, key=jax.random.PRNGKey(0))


def test_sdpa_dropout_is_reproducible_with_key():
    """With dropout, the same key produces the same output."""
    q = jnp.ones((1, 4, 2))
    k = jnp.ones((1, 4, 2))
    v = jnp.arange(8.0).reshape(1, 4, 2)
    out_a = sdpa(q, k, v, dropout_rate=0.5, key=jax.random.PRNGKey(7))
    out_b = sdpa(q, k, v, dropout_rate=0.5, key=jax.random.PRNGKey(7))
    assert jnp.array_equal(out_a, out_b)
