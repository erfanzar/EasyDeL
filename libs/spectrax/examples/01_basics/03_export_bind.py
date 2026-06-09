# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-trip a :class:`spx.Module` through :func:`spx.export` and :func:`spx.bind`.

Every spectrax transform ultimately calls :func:`spx.export` to split
a live module into:

* a :class:`spx.GraphDef` — the static structure (class identities,
  attribute names, shared-weight aliases);
* a :class:`spx.State` — a collection-partitioned pytree of arrays
  (``parameters``, ``buffers``, ``intermediates``, ...).

:func:`spx.bind` reverses the split: given the same ``GraphDef`` and
a (possibly modified) ``State``, it reconstructs a live, callable
module **without re-running ``__init__``**. This is the foundation for
checkpointing, functional transforms, and parameter surgery — and the
first thing every user should get hands-on with.

Run::

    python -m examples.01_basics.03_export_bind
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx


class Small(spx.Module):
    """Three-layer sequential MLP used purely for round-trip demo."""

    def __init__(self, rngs: spx.Rngs):
        """Stack three linear layers separated by ReLU activations."""
        super().__init__()
        self.net = spx.nn.Sequential(
            spx.nn.Linear(32, 64, rngs=rngs),
            spx.nn.ReLU(),
            spx.nn.Linear(64, 64, rngs=rngs),
            spx.nn.ReLU(),
            spx.nn.Linear(64, 16, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Pure ``Sequential`` forward."""
        return self.net(x)


def summarize_state(state: spx.State) -> None:
    """Pretty-print every ``(collection, path, shape, dtype)`` leaf."""
    for coll, entries in state.raw().items():
        print(f"  [{coll}]")
        for path, arr in entries.items():
            print(f"    {path:30s} {arr.shape!s:16s} {arr.dtype}")


def main():
    """Export a model, inspect state, bind a new module, check parity."""
    model = Small(rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 32))
    y_ref = model(x)

    gdef, state = spx.export(model)
    print(f"graphdef type  : {type(gdef).__name__}")
    print(f"state collections: {list(state.raw().keys())}")
    print("state contents:")
    summarize_state(state)

    rebuilt = spx.bind(gdef, state)
    y_new = rebuilt(x)

    max_err = float(jnp.max(jnp.abs(y_ref - y_new)))
    print(f"\nroundtrip max|y - y'| = {max_err:.2e}")
    assert max_err == 0.0, "bind(export(m)) must reproduce the exact forward"


if __name__ == "__main__":
    main()
