# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FSDP on a 1-D mesh: shard weights along a single ``fsdp`` axis.

Builds a mesh with one axis named ``fsdp`` and constructs two
:class:`spectrax.nn.Linear` layers whose first (input) dimension is
sharded along that axis. :func:`spectrax.sharding.get_partition_spec`
and :func:`spectrax.sharding.get_named_sharding` are then used to
inspect how the logical ``sharding=`` annotations resolve against the
live mesh. On a single CPU device the resulting shards are trivially
replicated, but the API path (including ``with mesh:`` context and
``NamedSharding`` construction) runs end-to-end.

Run from the repo root::

    python -m examples.05_shardings.01_fsdp
"""

from __future__ import annotations

import jax

import spectrax as spx
from spectrax.sharding import get_named_sharding, get_partition_spec, logical_axis_rules

RULES = [("fsdp", "fsdp")]


class TwoLinear(spx.Module):
    """Two stacked Linears whose weights are FSDP-sharded on their input dim."""

    def __init__(self, d_in: int, d_mid: int, d_out: int, *, rngs: spx.Rngs) -> None:
        """Create two Linears with ``sharding=("fsdp", None)``."""
        super().__init__()
        self.fc1 = spx.nn.Linear(d_in, d_mid, sharding=("fsdp", None), rngs=rngs)
        self.fc2 = spx.nn.Linear(d_mid, d_out, sharding=("fsdp", None), rngs=rngs)

    def forward(self, x):
        """Run fc2(silu(fc1(x)))."""
        return self.fc2(jax.nn.silu(self.fc1(x)))


def main():
    """Build a 1-D FSDP mesh, instantiate the model, and print resolved shardings."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("fsdp",))
    print(f"mesh: {dict(mesh.shape)}  devices={len(jax.devices())}")

    with mesh, logical_axis_rules(RULES):
        model = TwoLinear(16, 32, 16, rngs=spx.Rngs(0))
        specs = get_partition_spec(model)
        named = get_named_sharding(model, mesh.jax_mesh)

    print("\nper-parameter PartitionSpec (logical -> physical):")
    for col, entries in specs.items():
        for path, spec in entries.items():
            print(f"  [{col}] {path:20s} spec={spec}")

    print("\nper-parameter NamedSharding mesh axes:")
    for col, entries in named.items():
        for path, ns in entries.items():
            print(f"  [{col}] {path:20s} spec={ns.spec}")

    with mesh:
        x = jax.numpy.ones((4, 16))
        y = model(x)
    print(f"\nforward ok: x{tuple(x.shape)} -> y{tuple(y.shape)}")


if __name__ == "__main__":
    main()
