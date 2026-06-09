# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FSDP x TP hybrid on a 2-D ``(fsdp, tp)`` mesh.

Each Linear's weight is sharded along *both* mesh axes: the axis not
used for tensor parallelism is also used for FSDP weight sharding.
The column-parallel up-projection uses ``sharding=("fsdp", "tp")`` so
the input dim rides ``fsdp`` and the output dim rides ``tp``; the
row-parallel down-projection uses ``sharding=("tp", "fsdp")`` — the
*same* axes, swapped so TP stays on the contraction dim.

Works on any device count: a single CPU collapses both axes to
size-1 and replicates, a 4-chip TPU gives a 2x2 grid, etc.

Run from the repo root::

    python -m examples.05_shardings.03_fsdp_tp_hybrid
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.sharding import get_named_sharding, get_partition_spec, logical_axis_rules

RULES = [("fsdp", "fsdp"), ("tp", "tp")]


class HybridMLP(spx.Module):
    """MLP sharded by both FSDP (non-contraction dim) and TP (contraction dim)."""

    def __init__(self, d: int, hidden: int, *, rngs: spx.Rngs) -> None:
        """Create up- and down-projections with two-axis sharding."""
        super().__init__()
        self.up = spx.nn.Linear(d, hidden, sharding=("fsdp", "tp"), rngs=rngs)
        self.down = spx.nn.Linear(hidden, d, sharding=("tp", "fsdp"), rngs=rngs)

    def forward(self, x):
        """Column-parallel up + row-parallel down, SiLU between."""
        return self.down(jax.nn.silu(self.up(x)))


def main():
    """Build a 2-D mesh, instantiate the hybrid model, and print per-weight specs."""
    ndev = len(jax.devices())
    axis_dims = (2, -1) if (ndev >= 4 and ndev % 2 == 0) else (1, -1)
    mesh = spx.create_mesh(axis_dims=axis_dims, axis_names=("fsdp", "tp"))
    print(f"mesh: {dict(mesh.shape)}  axis_names={mesh.axis_names}")

    with mesh, logical_axis_rules(RULES):
        model = HybridMLP(d=32, hidden=128, rngs=spx.Rngs(0))
        specs = get_partition_spec(model)
        named = get_named_sharding(model, mesh.jax_mesh)
        x = jnp.ones((4, 32))
        y = model(x)
        jax.block_until_ready(y)

    print("\nweight PartitionSpecs (logical ``('fsdp', 'tp')`` resolved 1:1):")
    for col, entries in specs.items():
        for path, spec in entries.items():
            if "weight" in path:
                print(f"  [{col}] {path:20s} sharding.spec={spec}")

    print("\nNamedShardings on mesh:")
    for col, entries in named.items():
        for path, ns in entries.items():
            if "weight" in path:
                print(f"  [{col}] {path:20s} {ns.spec}")

    print(f"\nforward ok: x{tuple(x.shape)} -> y{tuple(y.shape)}")


if __name__ == "__main__":
    main()
