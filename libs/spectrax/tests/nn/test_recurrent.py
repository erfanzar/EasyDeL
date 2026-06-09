# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.nn.recurrent`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.nn.recurrent import (
    RNN,
    Bidirectional,
    ConvLSTMCell,
    GRUCell,
    LSTMCell,
    OptimizedLSTMCell,
    SimpleRNNCell,
)
from spectrax.rng.rngs import Rngs


def test_simple_rnn_cell_shapes():
    """Simple RNN carry and output have hidden shape."""
    cell = SimpleRNNCell(4, 8, rngs=Rngs(0))
    carry = cell.initial_carry((2,))
    assert carry.shape == (2, 8)
    new_carry, y = cell(carry, jnp.ones((2, 4)))
    assert new_carry.shape == (2, 8)
    assert y.shape == (2, 8)


def test_simple_rnn_cell_deterministic():
    """Same seed -> same output."""
    x = jnp.ones((1, 3))
    c = jnp.zeros((1, 5))
    a = SimpleRNNCell(3, 5, rngs=Rngs(42))
    b = SimpleRNNCell(3, 5, rngs=Rngs(42))
    _, ya = a(c, x)
    _, yb = b(c, x)
    assert jnp.allclose(ya, yb)


def test_lstm_cell_shapes():
    """LSTM produces two-tuple carry and hidden-shape output."""
    cell = LSTMCell(4, 8, rngs=Rngs(0))
    h, c = cell.initial_carry((2,))
    assert h.shape == (2, 8)
    assert c.shape == (2, 8)
    (h2, c2), y = cell((h, c), jnp.ones((2, 4)))
    assert h2.shape == y.shape == (2, 8)
    assert c2.shape == (2, 8)


def test_lstm_cell_matches_manual_reference():
    """LSTM forward matches a hand-rolled reference."""
    cell = LSTMCell(3, 4, rngs=Rngs(1))
    h = jnp.zeros((1, 4))
    c = jnp.zeros((1, 4))
    x = jnp.asarray([[0.5, -0.2, 1.0]])
    z = x @ cell.W_x.value + h @ cell.W_h.value + cell.b.value
    i, f, g, o = jnp.split(z, 4, axis=-1)
    i_, f_, g_, o_ = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jnp.tanh(g), jax.nn.sigmoid(o)
    c_ref = f_ * c + i_ * g_
    h_ref = o_ * jnp.tanh(c_ref)
    (h_out, c_out), _ = cell((h, c), x)
    assert jnp.allclose(h_out, h_ref, atol=1e-6)
    assert jnp.allclose(c_out, c_ref, atol=1e-6)


def test_optimized_lstm_cell_matches_lstm_when_weights_aligned():
    """Fused LSTM with concatenated weights reproduces standard LSTM numerics."""
    std = LSTMCell(3, 4, rngs=Rngs(0))
    fused = OptimizedLSTMCell(3, 4, rngs=Rngs(1))
    W = jnp.concatenate([std.W_x.value, std.W_h.value], axis=0)
    fused.W.value = W
    fused.b.value = std.b.value
    x = jnp.asarray([[1.0, 0.0, -0.5]])
    carry = std.initial_carry((1,))
    (h_a, c_a), _ = std(carry, x)
    (h_b, c_b), _ = fused(carry, x)
    assert jnp.allclose(h_a, h_b, atol=1e-6)
    assert jnp.allclose(c_a, c_b, atol=1e-6)


def test_gru_cell_shapes():
    """GRU carry/output shape."""
    cell = GRUCell(4, 6, rngs=Rngs(0))
    carry = cell.initial_carry((3,))
    assert carry.shape == (3, 6)
    new_carry, y = cell(carry, jnp.ones((3, 4)))
    assert new_carry.shape == (3, 6)
    assert y.shape == (3, 6)


def test_gru_zero_input_preserves_hidden_with_z_one():
    """If ``z`` saturates at 1 the output equals the previous hidden state."""
    cell = GRUCell(2, 3, rngs=Rngs(0))
    Wx = jnp.zeros((2, 9))
    Wh = jnp.zeros((3, 9))
    b = jnp.zeros((9,)).at[3:6].set(100.0)
    cell.W_x.value = Wx
    cell.W_h.value = Wh
    cell.b.value = b
    h = jnp.ones((1, 3)) * 0.3
    x = jnp.ones((1, 2))
    new_h, _ = cell(h, x)
    assert jnp.allclose(new_h, h, atol=1e-5)


def test_conv_lstm_cell_shapes():
    """ConvLSTM maps ``(N,H,W,Cin)`` to ``(N,H,W,Cout)``."""
    cell = ConvLSTMCell(2, 4, kernel_size=3, rngs=Rngs(0))
    h, c = cell.initial_carry((1, 5, 5))
    assert h.shape == (1, 5, 5, 4)
    x = jnp.ones((1, 5, 5, 2))
    (h2, c2), y = cell((h, c), x)
    assert h2.shape == y.shape == (1, 5, 5, 4)
    assert c2.shape == (1, 5, 5, 4)


def test_conv_lstm_cell_rejects_bad_batch_shape():
    """ConvLSTM requires a 3-element batch shape."""
    cell = ConvLSTMCell(2, 4, rngs=Rngs(0))
    with pytest.raises(ValueError):
        cell.initial_carry((2,))


def test_conv_lstm_cell_rejects_bad_kernel():
    """ConvLSTM rejects non-2D kernel specs."""
    with pytest.raises(ValueError):
        ConvLSTMCell(2, 4, kernel_size=(3, 3, 3), rngs=Rngs(0))


def test_rnn_wrapper_output_shape_batch_major():
    """Default RNN treats axis 1 as time."""
    cell = SimpleRNNCell(4, 8, rngs=Rngs(0))
    rnn = RNN(cell)
    xs = jnp.ones((3, 7, 4))
    ys = rnn(xs)
    assert ys.shape == (3, 7, 8)


def test_rnn_wrapper_time_major_output_shape():
    """``time_major=True`` uses axis 0 as time."""
    cell = SimpleRNNCell(4, 8, rngs=Rngs(0))
    rnn = RNN(cell, time_major=True)
    xs = jnp.ones((7, 3, 4))
    ys = rnn(xs)
    assert ys.shape == (7, 3, 8)


def test_rnn_wrapper_return_carry():
    """``return_carry=True`` returns ``(ys, final_carry)``."""
    cell = LSTMCell(4, 8, rngs=Rngs(0))
    rnn = RNN(cell, return_carry=True)
    xs = jnp.ones((2, 5, 4))
    ys, (h, c) = rnn(xs)
    assert ys.shape == (2, 5, 8)
    assert h.shape == (2, 8)
    assert c.shape == (2, 8)


def test_rnn_reverse_flips_time_axis():
    """Reverse mode is equivalent to flipping the input time axis beforehand."""
    cell = SimpleRNNCell(3, 4, rngs=Rngs(2))
    fwd = RNN(cell, reverse=False)
    rev = RNN(cell, reverse=True)
    xs = jnp.arange(2 * 5 * 3.0).reshape((2, 5, 3))
    ys_rev = rev(xs)
    ys_fwd = fwd(xs[:, ::-1])[:, ::-1]
    assert jnp.allclose(ys_rev, ys_fwd)


def test_rnn_initial_carry_override():
    """User-provided initial carry threads through the scan."""
    cell = SimpleRNNCell(2, 3, rngs=Rngs(0))
    rnn = RNN(cell, return_carry=True)
    xs = jnp.zeros((1, 4, 2))
    h0 = jnp.ones((1, 3)) * 0.5
    _, final = rnn(xs, initial_carry=h0)
    assert final.shape == h0.shape


def test_rnn_gradient_flows_through_time():
    """Gradients propagate through the whole scan."""
    cell = SimpleRNNCell(2, 3, rngs=Rngs(0))
    rnn = RNN(cell)

    def loss(W_xh):
        """Compute the loss."""
        cell.W_xh.value = W_xh
        ys = rnn(jnp.ones((1, 4, 2)))
        return ys.sum()

    g = jax.grad(loss)(cell.W_xh.value)
    assert g.shape == cell.W_xh.value.shape
    assert jnp.any(g != 0.0)


def test_bidirectional_concat_doubles_feature_dim():
    """Concat merge doubles the output feature size."""
    fwd = RNN(SimpleRNNCell(3, 4, rngs=Rngs(0)))
    bwd = RNN(SimpleRNNCell(3, 4, rngs=Rngs(1)))
    bi = Bidirectional(fwd, bwd, merge_mode="concat")
    xs = jnp.ones((1, 6, 3))
    ys = bi(xs)
    assert ys.shape == (1, 6, 8)


def test_bidirectional_sum_preserves_feature_dim():
    """Sum merge keeps the output feature size."""
    fwd = RNN(SimpleRNNCell(3, 4, rngs=Rngs(0)))
    bwd = RNN(SimpleRNNCell(3, 4, rngs=Rngs(1)))
    bi = Bidirectional(fwd, bwd, merge_mode="sum")
    xs = jnp.ones((1, 6, 3))
    ys = bi(xs)
    assert ys.shape == (1, 6, 4)


def test_bidirectional_rejects_unknown_merge():
    """Unknown merge modes raise."""
    fwd = RNN(SimpleRNNCell(3, 4, rngs=Rngs(0)))
    bwd = RNN(SimpleRNNCell(3, 4, rngs=Rngs(1)))
    with pytest.raises(ValueError):
        Bidirectional(fwd, bwd, merge_mode="bogus")


def test_rnn_rejects_non_cell():
    """RNN requires an RNNCellBase instance."""
    with pytest.raises(TypeError):
        RNN(object())


def test_bidirectional_rejects_non_rnn():
    """Bidirectional requires two RNN instances."""
    with pytest.raises(TypeError):
        Bidirectional(object(), object())
