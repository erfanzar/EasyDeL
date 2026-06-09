# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Surgery on a live module's :class:`spx.State` pytree.

This example walks through the four core state-manipulation primitives:

* :func:`spx.tree_state` — read the current ``State`` without also
  returning a ``GraphDef``.
* :func:`spx.clone` — deep-copy the module so we can mutate the copy
  without affecting the original.
* :func:`spx.update` — write new leaf values back into a live module
  in place (only matching ``(collection, path)`` pairs are applied).
* :func:`spx.pop` — detach-and-return variables matching a selector;
  used here to evict a :class:`spx.Buffer` counter after a forward pass.

The pattern ``iter_variables`` + ``tree_state`` + ``update`` is how
every ad-hoc parameter edit (freezing, re-initialization, LoRA
injection) starts life before being wrapped in a real helper.

Run::

    python -m examples.01_basics.04_state_pop_update
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx


class Counted(spx.Module):
    """Linear layer with a :class:`spx.Buffer` forward-call counter."""

    def __init__(self, d: int, *, rngs: spx.Rngs):
        """Create the projection and a scalar buffer starting at zero."""
        super().__init__()
        self.lin = spx.nn.Linear(d, d, rngs=rngs)
        self.calls = spx.Buffer(jnp.zeros((), jnp.int32))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Increment the counter and return ``lin(x)``."""
        self.calls.value = self.calls.value + 1
        return self.lin(x)


def main():
    """Demonstrate ``tree_state``, ``clone``, ``update``, and ``pop``."""
    model = Counted(d=32, rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 32))
    _ = model(x)
    _ = model(x)

    state = spx.tree_state(model)
    print("paths in state:")
    for coll, path in state.paths():
        print(f"  [{coll}] {path}")

    twin = spx.clone(model)
    zeroed = state.overlay(spx.State({})).map(lambda path, value: jnp.zeros_like(value))
    spx.update(twin, zeroed.filter("parameters"))
    print(f"\noriginal weight[0,0]: {float(model.lin.weight.value[0, 0]):+.4f}")
    print(f"twin     weight[0,0]: {float(twin.lin.weight.value[0, 0]):+.4f}")

    buffers = spx.pop(model, spx.of_type(spx.Buffer))
    print(f"\npopped buffers    : {buffers.raw()}")
    print(f"model variables after pop: {len(list(spx.iter_variables(model)))}")
    print(f"twin variables (untouched): {len(list(spx.iter_variables(twin)))}")


if __name__ == "__main__":
    main()
