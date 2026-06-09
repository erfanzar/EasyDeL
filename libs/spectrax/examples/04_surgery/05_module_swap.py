# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replace submodules in place via attribute rebinding and :func:`spx.update`.

Two swaps are demonstrated:

* **Activation swap** — replace ``nn.GELU`` with ``nn.ReLU`` by
  assigning a new module to the parent attribute; :func:`spx.iter_modules`
  picks up the new child immediately.
* **State-level patch** — build a fresh :class:`nn.Linear` with the
  desired shape, export its state, and push it through
  :func:`spx.update` to overwrite the original layer's weights.

Run::

    python -m examples.04_surgery.05_module_swap
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax import nn


class SmallNet(spx.Module):
    """Linear -> activation -> linear, all tiny for CPU demos."""

    def __init__(self, d: int, rngs: spx.Rngs):
        """Construct two linears around an activation placeholder."""
        super().__init__()
        self.fc1 = nn.Linear(d, d, rngs=rngs)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d, d, rngs=rngs)

    def __call__(self, x):
        """Forward ``fc2(act(fc1(x)))``."""
        return self.fc2(self.act(self.fc1(x)))


def list_children(tag: str, model: spx.Module) -> None:
    """Print every child module under ``model`` with its path."""
    print(f"--- {tag} ---")
    for path, mod in spx.iter_modules(model, skip_root=True):
        print(f"  {path:10s} {type(mod).__name__}")


def main():
    """Swap an activation and patch a Linear's weights via state-level update."""
    model = SmallNet(d=8, rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 8))
    y_before = model(x)

    list_children("before swap", model)
    model.act = nn.ReLU()
    list_children("after activation swap", model)

    y_after = model(x)
    diff = float(jnp.max(jnp.abs(y_before - y_after)))
    print(f"output changed after GELU -> ReLU: max_abs_delta={diff:.4f}")

    donor = nn.Linear(8, 8, rngs=spx.Rngs(99))
    _, donor_state = spx.export(donor)
    live_path_prefix = "fc1."
    patch = spx.tree_state(model)
    patch = type(patch)()
    patch = patch.set("parameters", live_path_prefix + "weight", donor_state.get("parameters", "weight"))
    patch = patch.set("parameters", live_path_prefix + "bias", donor_state.get("parameters", "bias"))
    spx.update(model, patch)

    _, new_state = spx.export(model)
    matches_donor = bool(
        jnp.allclose(new_state.get("parameters", "fc1.weight"), donor_state.get("parameters", "weight"))
    )
    print(f"fc1 weights replaced from donor: {matches_donor}")

    clone = spx.clone(model)
    print(f"clone is independent: {clone is not model}")

    popped = spx.pop(clone, spx.path_startswith("fc2"))
    print(f"popped fc2 variables: {list(popped.paths())}")


if __name__ == "__main__":
    main()
