# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Freeze a subset of parameters by masking gradients via selectors.

The embedding table stays frozen while the rest of the network trains.
We derive a frozen-path set from a selector, compute full gradients,
then zero out any gradient whose canonical path lies in that set.

Run::

    python -m examples.04_surgery.04_freeze_params
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax import nn


class Tiny(spx.Module):
    """Embedding + mean-pool + linear head, tiny enough to train on CPU."""

    def __init__(self, vocab: int, d: int, rngs: spx.Rngs):
        """Build the embedding table and classification head."""
        super().__init__()
        self.emb = nn.Embed(vocab, d, rngs=rngs)
        self.head = nn.Linear(d, vocab, rngs=rngs)

    def __call__(self, ids):
        """Embed tokens, mean-pool, project to vocab logits."""
        h = self.emb(ids).mean(axis=1)
        return self.head(h)


def frozen_path_set(module: spx.Module, selector) -> set[tuple[str, str]]:
    """Collect ``(collection, path)`` tuples for every variable the selector picks."""
    paths: set[tuple[str, str]] = set()
    for path, var in spx.iter_variables(module, select=selector):
        paths.add((var.kind, path))
    return paths


def mask_grads(grads, frozen: set[tuple[str, str]]):
    """Zero out any gradient whose ``(collection, path)`` is in ``frozen``."""
    spx.tree_state(grads) if not hasattr(grads, "items") else grads
    masked = type(grads)()
    for c, p, leaf in grads.items():
        if (c, p) in frozen:
            masked = masked.set(c, p, jnp.zeros_like(leaf))
        else:
            masked = masked.set(c, p, leaf)
    return masked


def main():
    """Freeze embedding grads while still training the head."""
    model = Tiny(vocab=32, d=16, rngs=spx.Rngs(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (4, 8), 0, 32)
    labels = jax.random.randint(jax.random.PRNGKey(2), (4,), 0, 32)

    frozen = frozen_path_set(model, spx.path_startswith("emb"))
    print(f"frozen variables: {sorted(frozen)}")

    def loss_fn(m):
        """Cross-entropy over the pooled logits."""
        logits = m(ids)
        return -(jax.nn.log_softmax(logits) * jax.nn.one_hot(labels, 32)).sum(-1).mean()

    loss, grads = spx.value_and_grad(loss_fn)(model)
    jax.block_until_ready(loss)
    print(f"loss: {float(loss):.4f}")

    print("--- raw grads ---")
    for c, p, g in grads.items():
        print(f"  {c}/{p:20s} norm={float(jnp.linalg.norm(g)):.4f}")

    masked = mask_grads(grads, frozen)
    print("--- masked grads (embedding zeroed) ---")
    for c, p, g in masked.items():
        print(f"  {c}/{p:20s} norm={float(jnp.linalg.norm(g)):.4f}")


if __name__ == "__main__":
    main()
