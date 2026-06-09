# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logical axis rules: decouple model annotations from the physical mesh.

Annotate weights with *model-level* logical axis names (``heads``,
``embed``, ``ffn``, ``vocab``) instead of physical axis names. The
mapping from those logical names to the physical mesh axes is then
supplied at runtime via
:func:`spectrax.sharding.logical_axis_rules`. The same model, on the
same mesh, can then be resharded by simply swapping the rules.

This file exercises that by printing the resolved PartitionSpecs of
the same model under two rule sets that swap which logical axis
lands on ``fsdp`` vs ``tp``.

Run from the repo root::

    python -m examples.05_shardings.04_logical_rules
"""

from __future__ import annotations

import jax

import spectrax as spx
from spectrax.sharding import current_axis_rules, get_partition_spec, logical_axis_rules


class TinyTransformerBlock(spx.Module):
    """Attention-style block with weights annotated by *logical* axis names."""

    def __init__(self, d: int, heads: int, ffn: int, *, rngs: spx.Rngs) -> None:
        """Create qkv/o/up/down Linears with logical axis annotations."""
        super().__init__()
        self.qkv = spx.nn.Linear(d, 3 * heads, sharding=("embed", "heads"), rngs=rngs)
        self.o = spx.nn.Linear(heads, d, sharding=("heads", "embed"), rngs=rngs)
        self.up = spx.nn.Linear(d, ffn, sharding=("embed", "ffn"), rngs=rngs)
        self.down = spx.nn.Linear(ffn, d, sharding=("ffn", "embed"), rngs=rngs)

    def forward(self, x):
        """Trivial forward that touches every Linear for shape validation."""
        return self.down(jax.nn.gelu(self.up(x))) + self.o(self.qkv(x)[..., : self.o.in_features])


RULES_A = [("heads", "tp"), ("ffn", "tp"), ("embed", "fsdp"), ("vocab", "tp")]
RULES_B = [("heads", "fsdp"), ("ffn", "fsdp"), ("embed", "tp"), ("vocab", "fsdp")]


def dump_specs(label: str, model: spx.Module) -> None:
    """Print PartitionSpecs for every weight under the currently-active rules."""
    print(f"\n== {label} == (active rules: {dict(current_axis_rules())})")
    specs = get_partition_spec(model)
    for col, entries in specs.items():
        for path, spec in entries.items():
            if "weight" in path:
                print(f"  [{col}] {path:20s} spec={spec}")


def main():
    """Build a 2-D mesh, instantiate the model, print specs under both rule sets."""
    ndev = len(jax.devices())
    axis_dims = (2, -1) if (ndev >= 4 and ndev % 2 == 0) else (1, -1)
    mesh = spx.create_mesh(axis_dims=axis_dims, axis_names=("fsdp", "tp"))
    print(f"mesh: {dict(mesh.shape)}")

    with mesh:
        model = TinyTransformerBlock(d=32, heads=32, ffn=64, rngs=spx.Rngs(0))
        with logical_axis_rules(RULES_A):
            dump_specs("RULES_A (heads/ffn -> tp, embed -> fsdp)", model)
        with logical_axis_rules(RULES_B):
            dump_specs("RULES_B (heads/ffn -> fsdp, embed -> tp)", model)


if __name__ == "__main__":
    main()
