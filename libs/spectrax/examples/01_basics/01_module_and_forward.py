# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Define a tiny :class:`spx.Module` and run a forward pass.

This is the hello-world for spectrax. The key ideas:

* A :class:`spx.Module` subclass declares submodules and parameters
  in ``__init__`` (always remember ``super().__init__()`` first) and
  the computation in ``__call__``.
* Parameters live inside :class:`spx.nn.Linear` / :class:`spx.nn.LayerNorm`
  leaves as :class:`spx.Parameter` cells, initialized lazily from a
  :class:`spx.Rngs` stream.
* Modules are JAX pytrees — you can pass them through ``jit``, ``vmap``,
  and ``grad`` without hand-threaded ``(parameters, apply)`` tuples.

Run::

    python -m examples.01_basics.01_module_and_forward
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx


class TinyMLP(spx.Module):
    """A two-layer MLP with LayerNorm and ReLU nonlinearity."""

    def __init__(self, d_in: int, hidden: int, d_out: int, *, rngs: spx.Rngs):
        """Build the layer stack.

        Args:
            d_in: Input feature count.
            hidden: Hidden feature count.
            d_out: Output feature count.
            rngs: PRNG source for parameter initialization.
        """
        super().__init__()
        self.fc1 = spx.nn.Linear(d_in, hidden, rngs=rngs)
        self.norm = spx.nn.LayerNorm(hidden)
        self.fc2 = spx.nn.Linear(hidden, d_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: linear -> layernorm -> relu -> linear."""
        h = self.fc1(x)
        h = self.norm(h)
        h = jax.nn.relu(h)
        return self.fc2(h)


def main():
    """Instantiate ``TinyMLP``, push a random batch through, print shapes."""
    model = TinyMLP(d_in=32, hidden=64, d_out=16, rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 32))
    y = model(x)

    print(f"input shape : {x.shape}")
    print(f"output shape: {y.shape}")
    print(f"output mean : {float(jnp.mean(y)):+.4f}")

    print("\nparameters:")
    for path, var in spx.iter_variables(model, "parameters"):
        print(f"  {path:20s} {var.shape!s:16s} {var.dtype}")


if __name__ == "__main__":
    main()
