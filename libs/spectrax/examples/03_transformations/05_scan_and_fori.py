# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware ``spx.scan`` and ``spx.fori_loop``.

``spx.scan(fn, init_module, xs)`` is a drop-in for
:func:`jax.lax.scan` that threads ``init_module``'s state as the scan
carry and maps ``fn(module, x) -> y`` across the leading axis of
``xs``. Combined with :func:`spectrax.export` / :func:`spectrax.bind`,
the same trick also powers the "stack of N identical layers"
pattern: stack the per-layer state along a new leading axis, then use
a plain :func:`jax.lax.scan` whose body rebinds a fresh layer from its
slice at every step. That compiles **one** layer and runs it N times.

``spx.fori_loop`` is the simple fixed-trip counterpart — a
Python-style ``for i in range(lo, hi)`` compiled into one XLA loop.

Key concepts demonstrated:

* ``spx.scan`` iterating a single Linear over a sequence of inputs —
  the RNN-shaped primitive.
* A stacked-layer variant using ``jax.lax.scan`` plus ``spx.bind``:
  N layers, one compile.
* ``spx.fori_loop`` driving a residual refinement over a single
  carrier module and data vector.

Run::

    python -m examples.03_transformations.05_scan_and_fori
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.lax as lax
import jax.numpy as jnp

import spectrax as spx


def sequence_step(model, x):
    """One step of the RNN-style scan: apply ``model`` to ``x``."""
    return model(x)


def stack_states(layer, n_layers):
    """Broadcast ``layer``'s exported State along a new axis of size ``n_layers``."""
    _, state = spx.export(layer)
    return jax.tree.map(lambda v: jnp.broadcast_to(v, (n_layers, *v.shape)), state)


def main():
    """Demonstrate three loop primitives: sequence scan, layer-stack scan, fori_loop."""
    layer = spx.nn.Linear(4, 4, rngs=spx.Rngs(0))
    xs = jnp.arange(12.0).reshape((3, 4))

    ys = spx.scan(sequence_step, layer, xs)
    print(f"spx.scan over sequence: input {xs.shape} -> output {ys.shape}")

    n_layers = 4
    gdef, _ = spx.export(layer)
    stacked = stack_states(layer, n_layers)

    def stack_body(x, layer_state):
        """Rebind a fresh Linear from one slice of the stacked state."""
        live = spx.bind(gdef, layer_state)
        return live(x), None

    x0 = jnp.ones((4,))
    x_final, _ = lax.scan(stack_body, x0, stacked)
    print(f"stacked-layer scan: {n_layers} layers, |output|_2 = {float(jnp.linalg.norm(x_final)):.4f}")

    refiner = spx.nn.Linear(4, 4, rngs=spx.Rngs(1))

    def refine(i, model, x):
        """One residual refinement step: ``x <- x + 0.1 * model(x)``."""
        del i
        return x + 0.1 * model(x)

    x_refined = spx.fori_loop(0, 5, refine, refiner, x0)
    print(f"spx.fori_loop 5 steps: drift = {float(jnp.linalg.norm(x_refined - x0)):.4f}")


if __name__ == "__main__":
    main()
