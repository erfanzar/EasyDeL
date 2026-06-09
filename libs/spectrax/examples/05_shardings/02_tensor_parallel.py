# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tensor-parallel column/row linear pattern on a 1-D ``(tp,)`` mesh.

The Megatron-LM-style two-linear MLP has an up-projection with the
output feature dim sharded (``sharding=(None, "tp")``, aka
column-parallel) followed by a down-projection with the input feature
dim sharded (``sharding=("tp", None)``, row-parallel). The matmul
outputs an activation that is partial-sum along ``tp``; XLA's SPMD
partitioner inserts the reduce-scatter / all-reduce at the boundary.

A forward pass is executed inside the mesh context to exercise the
whole pipeline. Works on a single CPU device (replicated) or any
multi-device backend.

Run from the repo root::

    python -m examples.05_shardings.02_tensor_parallel
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.sharding import get_partition_spec, logical_axis_rules

RULES = [("tp", "tp")]


class TpMLP(spx.Module):
    """Column-parallel up-proj + row-parallel down-proj two-layer MLP."""

    def __init__(self, d: int, hidden: int, *, rngs: spx.Rngs) -> None:
        """Create the column- and row-parallel Linears."""
        super().__init__()
        self.up = spx.nn.Linear(d, hidden, sharding=(None, "tp"), rngs=rngs)
        self.down = spx.nn.Linear(hidden, d, sharding=("tp", None), rngs=rngs)

    def forward(self, x):
        """Two-layer MLP with a GELU between up- and down-projections."""
        return self.down(jax.nn.gelu(self.up(x)))


def main():
    """Build a ``(tp,)`` mesh, run a forward pass, and print weight specs."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("tp",))
    print(f"mesh: {dict(mesh.shape)}  devices={len(jax.devices())}")

    with mesh, logical_axis_rules(RULES):
        model = TpMLP(d=16, hidden=64, rngs=spx.Rngs(0))
        specs = get_partition_spec(model)
        x = jnp.ones((2, 16))
        y = model(x)
        jax.block_until_ready(y)

    print("\nup  (column-parallel) spec:", specs["parameters"]["up.weight"])
    print("down (row-parallel)    spec:", specs["parameters"]["down.weight"])
    print(f"\nforward ok: x{tuple(x.shape)} -> y{tuple(y.shape)}")
    print(f"y mean={float(y.mean()):+.4f}")


if __name__ == "__main__":
    main()
