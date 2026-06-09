# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SPMD pipeline training under the GPipe schedule.

Each stage is a small transformer-ish block (RMSNorm + MLP) stacked
a couple of times per stage. The GPipe schedule runs all forwards
first, then all backwards — simple and memory-heavy. The bubble
ratio is ``(n-1)/(m+n-1)``.

Run::

    python -m examples.06_spmd_scheduled.02_gpipe
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.nn import Linear, ModuleList, PipelineSequential, RMSNorm
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import GPipe
from spectrax.runtime.types import MpMdMesh


class MiniBlock(spx.Module):
    """Residual ``RMSNorm -> Linear -> GELU -> Linear`` block."""

    def __init__(self, d: int, *, rngs: spx.Rngs):
        """Initialise the norm and two linear projections."""
        super().__init__()
        self.norm = RMSNorm(d)
        self.up = Linear(d, 2 * d, rngs=rngs)
        self.down = Linear(2 * d, d, rngs=rngs)

    def __call__(self, x):
        """Run the residual feed-forward sub-layer."""
        h = self.norm(x)
        return x + self.down(jax.nn.gelu(self.up(h)))


class Stage(spx.Module):
    """A pipeline stage containing several :class:`MiniBlock` s."""

    def __init__(self, d: int, depth: int, *, rngs: spx.Rngs):
        """Build ``depth`` blocks sharing dimension ``d``."""
        super().__init__()
        self.blocks = ModuleList([MiniBlock(d, rngs=rngs) for _ in range(depth)])

    def __call__(self, x):
        """Apply every block in sequence."""
        for b in self.blocks:
            x = b(x)
        return x


def mse(out, target):
    """Squared-error loss averaged over all elements."""
    return jnp.mean((out - target) ** 2)


def main():
    """Train one GPipe step on a 2-stage stack and report loss + bubble ratio."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",))
    n_stages = int(mesh.shape["pp"])
    d, depth, m = 32, 2, 4

    stages = [Stage(d, depth, rngs=spx.Rngs(i)) for i in range(max(n_stages, 2))]
    model = PipelineSequential(*stages)

    x = jax.random.normal(jax.random.PRNGKey(0), (m, 8, d))
    y = jax.random.normal(jax.random.PRNGKey(1), (m, 8, d))

    schedule = GPipe(microbatches=m)
    print(f"GPipe(m={m})  pp={n_stages}  bubble={schedule.bubble_ratio(model.num_stages):.3f}")
    print(
        f"total_steps={schedule.total_steps(model.num_stages)}  peak_acts={schedule.peak_activations(model.num_stages)}"
    )

    if n_stages == model.num_stages and n_stages >= 2:
        mpmd_mesh = MpMdMesh(mesh, "pp")
        loss, grads = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=mse)
        print(f"train loss: {float(loss):.4f}  stages of grads: {len(grads)}")
    else:
        print(f"[note] need pp=={model.num_stages} devices for SPMD; eager fallback.")
        loss = mse(model(x), y)
        print(f"eager loss: {float(loss):.4f}")


if __name__ == "__main__":
    main()
