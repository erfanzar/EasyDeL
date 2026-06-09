# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hint intermediate activation shardings with ``with_sharding_constraint_by_name``.

XLA's SPMD partitioner usually figures out activation shardings on
its own, but a hint is sometimes needed — e.g. to force a
reduce-scatter at a specific point, or to declare the sharding of
an activation that doesn't flow trivially from its inputs.
:func:`spectrax.sharding.with_sharding_constraint_by_name` lets you
express that in *logical* axis names resolved through the active
:func:`logical_axis_rules` context. This example lowers the model
to HLO and grep's for the resulting ``Sharding`` annotations.

Run from the repo root::

    python -m examples.05_shardings.05_with_sharding_constraint
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.sharding import logical_axis_rules, with_sharding_constraint_by_name


class ConstrainedMLP(spx.Module):
    """Two-layer MLP that constrains its hidden and output activations by name."""

    def __init__(self, d: int, hidden: int, *, rngs: spx.Rngs) -> None:
        """Create column- then row-parallel Linears."""
        super().__init__()
        self.up = spx.nn.Linear(d, hidden, sharding=("embed", "ffn"), rngs=rngs)
        self.down = spx.nn.Linear(hidden, d, sharding=("ffn", "embed"), rngs=rngs)

    def forward(self, x):
        """Apply up-proj, constrain mid activation, apply down-proj, constrain output."""
        x = with_sharding_constraint_by_name(x, ("batch", "embed"))
        h = jax.nn.gelu(self.up(x))
        h = with_sharding_constraint_by_name(h, ("batch", "ffn"))
        y = self.down(h)
        return with_sharding_constraint_by_name(y, ("batch", "embed"))


RULES = [("batch", "fsdp"), ("embed", None), ("ffn", "tp")]


def main():
    """Build mesh, trace the model to HLO, and show the sharding annotations."""
    ndev = len(jax.devices())
    axis_dims = (2, -1) if (ndev >= 4 and ndev % 2 == 0) else (1, -1)
    mesh = spx.create_mesh(axis_dims=axis_dims, axis_names=("fsdp", "tp"))
    print(f"mesh: {dict(mesh.shape)}")

    with mesh, logical_axis_rules(RULES):
        model = ConstrainedMLP(d=16, hidden=64, rngs=spx.Rngs(0))

        @jax.jit
        def run(x):
            """Call the constrained MLP inside jit so XLA sees the constraints."""
            return model(x)

        x = jnp.ones((8, 16))
        y = run(x)
        jax.block_until_ready(y)
        hlo = run.lower(x).compile().as_text()

    lines = [ln for ln in hlo.splitlines() if "sharding" in ln.lower()]
    print(f"\nHLO sharding-annotated lines: {len(lines)} (first 3 shown)")
    for ln in lines[:3]:
        print(" ", ln.strip()[:160])
    print(f"forward ok: x{tuple(x.shape)} -> y{tuple(y.shape)}")


if __name__ == "__main__":
    main()
