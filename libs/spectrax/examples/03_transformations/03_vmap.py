# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware ``spx.vmap`` — vectorize a Module over a batch axis.

spectrax's ``vmap`` always broadcasts :class:`~spectrax.Module`
arguments with ``in_axes=None`` (parameters are shared across the
batch). User-provided ``in_axes`` / ``out_axes`` therefore apply only
to the *non-module* positional / keyword arguments — typically the
data tensors themselves.

Key concepts demonstrated:

* Mapping over batch-axis 0 of the input (``in_axes=0``) while the
  model's parameters stay replicated.
* Different ``in_axes`` values (e.g. axis ``1``) to pull the batch out
  of a non-leading dim.
* Controlling the output layout with ``out_axes``.

Run::

    python -m examples.03_transformations.03_vmap
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

import spectrax as spx


def single_example_forward(model, x):
    """Apply ``model`` to a single (non-batched) vector ``x``."""
    return model(x)


def main():
    """Vmap a per-sample forward across several batching conventions."""
    model = spx.nn.Linear(6, 4, rngs=spx.Rngs(0))
    batch = jnp.arange(30.0).reshape((5, 6))

    batched = spx.vmap(single_example_forward, in_axes=0, out_axes=0)
    ys = batched(model, batch)
    print(f"vmap over axis 0 -> input {batch.shape}, output {ys.shape}")

    batch_t = batch.T
    mapped_axis1 = spx.vmap(single_example_forward, in_axes=1, out_axes=0)
    ys_t = mapped_axis1(model, batch_t)
    print(f"vmap over axis 1 -> input {batch_t.shape}, output {ys_t.shape}")

    tailed = spx.vmap(single_example_forward, in_axes=0, out_axes=1)
    ys_tailed = tailed(model, batch)
    print(f"out_axes=1 -> output {ys_tailed.shape} (features-first)")

    print(f"numerical check (axis-0 vs axis-1): allclose={bool(jnp.allclose(ys, ys_t))}")
    print(f"numerical check (axis-0 vs out_axes=1.T): allclose={bool(jnp.allclose(ys, ys_tailed.T))}")


if __name__ == "__main__":
    main()
