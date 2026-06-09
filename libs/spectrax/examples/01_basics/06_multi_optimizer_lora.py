# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Use ``MultiOptimizer`` to train base weights and LoRA adapters separately.

``spectrax.contrib.MultiOptimizer`` is useful when different variable
collections need different optimizer transforms. This example wraps the
first linear layer with LoRA, then trains:

* regular ``"parameters"`` with a small SGD learning rate;
* adapter ``"lora"`` variables with a larger Adam learning rate.

Run::

    python -m examples.01_basics.06_multi_optimizer_lora
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax

import spectrax as spx
from spectrax import nn
from spectrax.contrib.optimizer import MultiOptimizer


class AdapterRegressor(spx.Module):
    """Tiny regression model with one LoRA-wrapped projection."""

    def __init__(self, d_in: int, hidden: int, d_out: int, *, rngs: spx.Rngs):
        """Create ``LoRA(Linear) -> GELU -> Linear``."""
        super().__init__()
        self.fc1 = nn.wrap_lora(nn.Linear(d_in, hidden, rngs=rngs), rank=4, rngs=rngs)
        self.fc2 = nn.Linear(hidden, d_out, rngs=rngs)

    def forward(self, x: jax.Array) -> jax.Array:
        """Forward pass."""
        return self.fc2(jax.nn.gelu(self.fc1(x)))


def mse(model: AdapterRegressor, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean-squared error objective."""
    return jnp.mean((model(x) - y) ** 2)


def make_batch(key: jax.Array, batch: int, d_in: int, d_out: int) -> tuple[jax.Array, jax.Array]:
    """Synthetic linear regression data."""
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (batch, d_in))
    w = jax.random.normal(kw, (d_in, d_out)) / jnp.sqrt(d_in)
    return x, x @ w


def count_collection(model: spx.Module, collection: str) -> int:
    """Count scalar leaves in one variable collection."""
    return sum(v.value.size for _path, v in spx.iter_variables(model, select=collection))


def main() -> None:
    """Train with independent optimizer policies for base and adapter leaves."""
    model = AdapterRegressor(16, 32, 4, rngs=spx.Rngs(0))
    optimizer = MultiOptimizer.create(
        model,
        {
            "parameters": optax.sgd(1e-3),
            "lora": optax.adam(2e-2),
        },
    )

    print(f"base parameter leaves : {count_collection(model, 'parameters')}")
    print(f"LoRA adapter leaves   : {count_collection(model, 'lora')}")
    print(f"optimizer slices      : {len(optimizer.subs)}")

    @spx.jit(mutable=("parameters", "lora"))
    def step(model: AdapterRegressor, optimizer: MultiOptimizer, x: jax.Array, y: jax.Array):
        """One jitted step that mutates both parameter collections."""
        loss, grads = spx.value_and_grad(mse, wrt=("parameters", "lora"))(model, x, y)
        parameters = spx.tree_state(model).filter("parameters", "lora")
        new_parameters, optimizer = optimizer.update(parameters, grads)
        spx.update(model, new_parameters)
        return loss, optimizer

    curve = []
    for i in range(20):
        x, y = make_batch(jax.random.PRNGKey(i), 8, 16, 4)
        loss, optimizer = step(model, optimizer, x, y)
        if i in {0, 1, 5, 10, 19}:
            curve.append((i, float(loss)))

    print("loss curve:")
    for i, value in curve:
        print(f"  step {i:2d}: {value:.4f}")
    print("sub-optimizer steps:", [int(sub.step) for sub in optimizer.subs])


if __name__ == "__main__":
    main()
