# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train with :class:`spectrax.contrib.optimizer.Optimizer` and :mod:`optax`.

Hand-rolled SGD gets old fast. :class:`spectrax.contrib.Optimizer` is a
pytree-registered, module-aware wrapper around an
:class:`optax.GradientTransformation`. The two flavors shown here are:

* :meth:`Optimizer.update` — functional: takes ``(parameters, grads)``,
  returns ``(new_parameters, new_optimizer)``. Safe inside :func:`spx.jit`.
* :meth:`Optimizer.apply_eager` — sugar: writes the new parameters
  back into a live module via :func:`spx.update`. Convenient in
  hand-written Python loops.

This example uses the functional path in a jitted step to fit the
same synthetic regression task as ``02_training_loop.py`` with Adam,
then prints a five-point loss curve.

Run::

    python -m examples.01_basics.05_optimizer
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import spectrax as spx
from spectrax.contrib.optimizer import Optimizer


class Regressor(spx.Module):
    """Two-layer MLP used as the optimization target."""

    def __init__(self, d_in: int, hidden: int, d_out: int, *, rngs: spx.Rngs):
        """Stack Linear -> ReLU -> Linear."""
        super().__init__()
        self.fc1 = spx.nn.Linear(d_in, hidden, rngs=rngs)
        self.fc2 = spx.nn.Linear(hidden, d_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with a single ReLU nonlinearity."""
        return self.fc2(jax.nn.relu(self.fc1(x)))


def mse(model: Regressor, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean-squared error objective."""
    return jnp.mean((model(x) - y) ** 2)


def make_data(key, n: int, d_in: int, d_out: int):
    """Build a random-linear-plus-noise synthetic regression dataset."""
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (n, d_in))
    w = jax.random.normal(k2, (d_in, d_out)) / jnp.sqrt(d_in)
    y = x @ w + 0.05 * jax.random.normal(k3, (n, d_out))
    return x, y


def main():
    """Fit a regressor with Adam and print a loss curve."""
    d_in, hidden, d_out, bs, steps = 32, 64, 8, 8, 100
    model = Regressor(d_in, hidden, d_out, rngs=spx.Rngs(0))
    opt = Optimizer.create(model, optax.adam(3e-3))
    x, y = make_data(jax.random.PRNGKey(1), 256, d_in, d_out)

    @spx.jit(mutable="parameters")
    def step(model, opt, xb, yb):
        """One Adam step: value_and_grad, opt.update, write-back."""
        loss, grads = spx.value_and_grad(mse)(model, xb, yb)
        parameters = spx.tree_state(model).filter("parameters")
        new_parameters, new_opt = opt.update(parameters, grads)
        spx.update(model, new_parameters)
        return loss, new_opt

    curve = []
    for i in range(steps):
        idx = jax.random.randint(jax.random.PRNGKey(100 + i), (bs,), 0, x.shape[0])
        loss, opt = step(model, opt, x[idx], y[idx])
        if i % (steps // 5) == 0 or i == steps - 1:
            curve.append((i, float(loss)))

    print(f"optimizer step count: {int(opt.step)}")
    print("loss curve:")
    for i, lv in curve:
        print(f"  step {i:4d}  loss={lv:.4f}")


if __name__ == "__main__":
    main()
