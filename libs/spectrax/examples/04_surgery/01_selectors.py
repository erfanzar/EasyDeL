# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Selector cookbook for spectrax model surgery.

Walks a small transformer with the graph-iteration helpers
(:func:`spx.iter_variables`, :func:`spx.iter_modules`, :func:`spx.find`)
and composes predicates with :func:`spx.of_type`,
:func:`spx.path_contains`, :func:`spx.path_startswith`,
:func:`spx.path_endswith`, :func:`spx.any_of`, :func:`spx.all_of`,
:func:`spx.not_`, :data:`spx.Everything`, and :data:`spx.Nothing`.

Run::

    python -m examples.04_surgery.01_selectors
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import spectrax as spx
from spectrax import nn


class TinyBlock(spx.Module):
    """A minimal attention + MLP block used as a selector target."""

    def __init__(self, d: int, rngs: spx.Rngs):
        """Construct attention and MLP linears of width ``d``."""
        super().__init__()
        self.attn_q = nn.Linear(d, d, rngs=rngs)
        self.attn_k = nn.Linear(d, d, rngs=rngs)
        self.attn_v = nn.Linear(d, d, rngs=rngs)
        self.attn_out = nn.Linear(d, d, rngs=rngs)
        self.mlp_in = nn.Linear(d, d * 2, rngs=rngs)
        self.mlp_out = nn.Linear(d * 2, d, rngs=rngs)

    def __call__(self, x):
        """Run attention-shaped linears then an MLP for illustration."""
        q = self.attn_q(x) + self.attn_k(x) + self.attn_v(x)
        return self.mlp_out(self.mlp_in(self.attn_out(q)))


class TinyTransformer(spx.Module):
    """Embedding + stacked blocks + head, all small and CPU-friendly."""

    def __init__(self, vocab: int, d: int, n_layers: int, rngs: spx.Rngs):
        """Build embed, ``n_layers`` blocks, and an output head."""
        super().__init__()
        self.emb = nn.Embed(vocab, d, rngs=rngs)
        self.blocks = nn.ModuleList([TinyBlock(d, rngs) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab, rngs=rngs)


def main():
    """Exercise every selector primitive on a tiny transformer."""
    model = TinyTransformer(vocab=32, d=16, n_layers=2, rngs=spx.Rngs(0))

    total = sum(v.value.size for _, v in spx.iter_variables(model))
    print(f"total parameters: {total}")

    attn_weights = [p for p, _ in spx.iter_variables(model, select=spx.path_contains("attn"))]
    print(f"attn variable paths: {len(attn_weights)}")

    hit = spx.find(model, spx.all_of(spx.path_contains("attn_q"), spx.path_endswith("weight")))
    print(f"first attn_q weight: {hit[0] if hit else None}")

    linears = [p for p, _ in spx.iter_modules(model, select=nn.Linear)]
    print(f"Linear modules: {len(linears)}")

    head_or_emb = spx.any_of(spx.path_startswith("head"), spx.path_startswith("emb"))
    for p, v in spx.iter_variables(model, select=head_or_emb):
        print(f"  boundary var: {p} {v.value.shape}")

    interior = spx.all_of(
        spx.of_type(spx.Parameter),
        spx.not_(spx.any_of(spx.path_startswith("head"), spx.path_startswith("emb"))),
    )
    interior_paths = [p for p, _ in spx.iter_variables(model, select=interior)]
    print(f"interior parameters: {len(interior_paths)}")

    every = sum(1 for _ in spx.iter_variables(model, select=spx.Everything))
    none = sum(1 for _ in spx.iter_variables(model, select=spx.Nothing))
    print(f"Everything={every} Nothing={none}")

    block0_params = sum(v.value.size for _, v in spx.iter_variables(model, select=spx.path_startswith("blocks.0")))
    print(f"blocks.0 param count: {block0_params}")


if __name__ == "__main__":
    main()
