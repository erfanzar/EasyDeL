# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware ``spx.grad`` / ``spx.value_and_grad``.

Both functions select the differentiation target via ``wrt=`` — the
default is ``"parameters"`` on the first :class:`~spectrax.Module` argument.
The returned gradient is a :class:`~spectrax.State` whose leaves mirror
the selected slice of the module's variables, so it round-trips
naturally through optimizers.

Key concepts demonstrated:

* A scalar-loss case: ``spx.value_and_grad`` returns
  ``(loss, grads_state)``.
* A vector-output case reduced to a scalar (sum) before differentiating
  — JAX's autodiff requires a scalar primal.
* Exploring the grad pytree via :meth:`~spectrax.State.paths`.

Run::

    python -m examples.03_transformations.02_grad
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

import spectrax as spx


def mse_loss(model, x, y):
    """Mean-squared-error between ``model(x)`` and ``y`` — a scalar."""
    return ((model(x) - y) ** 2).mean()


def sum_output(model, x):
    """Reduce the vector output ``model(x)`` to a scalar for autodiff."""
    return model(x).sum()


def main():
    """Train-step shell: compute value + grads and dump the gradient pytree."""
    model = spx.nn.MLPBlock(features=4, hidden_features=16, rngs=spx.Rngs(0))
    x = jnp.ones((3, 4))
    y = jnp.zeros((3, 4))

    loss_val, grads = spx.value_and_grad(mse_loss)(model, x, y)
    print(f"scalar loss: {float(loss_val):.4f}")
    print(f"grads type: {type(grads).__name__}")
    print("grad leaves (collection, path, shape):")
    for c, p, arr in list(grads.items())[:6]:
        print(f"  ({c!r:>8s}, {p:<20s}) shape={arr.shape}")

    grads_only = spx.grad(sum_output)(model, x)
    print(f"\nvector-output case: sum-reduced, {len(list(grads_only.paths()))} grad leaves")
    total = sum(float(jnp.abs(a).sum()) for _c, _p, a in grads_only.items())
    print(f"sum |grad| over vector-output case: {total:.4f}")


if __name__ == "__main__":
    main()
