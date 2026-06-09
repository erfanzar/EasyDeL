# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal MSE training loop on synthetic regression data.

This example trains a small MLP to fit a random-linear target using
hand-rolled SGD. It demonstrates:

* :func:`spx.value_and_grad` — a module-aware ``jax.value_and_grad``
  that differentiates w.r.t. the ``parameters`` subset of the module's
  :class:`spx.State` without any manual param/state bookkeeping.
* :func:`spx.jit` — tracing through a :class:`spx.Module` while still
  surviving in-place parameter mutation across the transform boundary.
* ``jax.tree.map`` over the returned grad :class:`spx.State` to write
  a vanilla SGD step back into the live module via :func:`spx.update`.

Run::

    python -m examples.01_basics.02_training_loop
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx


class Regressor(spx.Module):
    """Two-layer MLP mapping ``R^d_in -> R^d_out``."""

    def __init__(self, d_in: int, hidden: int, d_out: int, *, rngs: spx.Rngs):
        """Initialize two dense layers joined by a ReLU."""
        super().__init__()
        self.fc1 = spx.nn.Linear(d_in, hidden, rngs=rngs)
        self.fc2 = spx.nn.Linear(hidden, d_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with a single ReLU nonlinearity."""
        return self.fc2(jax.nn.relu(self.fc1(x)))


def mse_loss(model: Regressor, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean-squared error between ``model(x)`` and target ``y``."""
    return jnp.mean((model(x) - y) ** 2)


def make_data(key, n: int, d_in: int, d_out: int):
    """Synthesize a linear-plus-noise regression dataset."""
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (n, d_in))
    w_true = jax.random.normal(k2, (d_in, d_out)) / jnp.sqrt(d_in)
    y = x @ w_true + 0.05 * jax.random.normal(k3, (n, d_out))
    return x, y


def main():
    """Run ~100 SGD steps and report initial vs. final loss."""
    d_in, hidden, d_out, bs, steps, lr = 32, 64, 8, 8, 100, 0.05
    model = Regressor(d_in, hidden, d_out, rngs=spx.Rngs(0))
    x, y = make_data(jax.random.PRNGKey(1), 256, d_in, d_out)

    @spx.jit(mutable="parameters")
    def train_step(model, xb, yb):
        """One jitted value-and-grad step returning the new loss."""
        loss, grads = spx.value_and_grad(mse_loss)(model, xb, yb)
        parameters = spx.tree_state(model).filter("parameters")
        new_parameters = jax.tree.map(lambda p, g: p - lr * g, parameters, grads)
        spx.update(model, new_parameters)
        return loss

    initial = float(mse_loss(model, x, y))
    final = initial
    for i in range(steps):
        idx = jax.random.randint(jax.random.PRNGKey(100 + i), (bs,), 0, x.shape[0])
        final = float(train_step(model, x[idx], y[idx]))

    print(f"initial loss: {initial:.4f}")
    print(f"final   loss: {final:.4f}")
    print(f"reduction   : {(initial - final) / initial * 100:.1f}%")


if __name__ == "__main__":
    main()
