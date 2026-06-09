# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interleaved H1 schedule with virtual stages.

``InterleavedH1(virtual_stages=V)`` gives each physical rank ``V``
logical chunks of the model, cycling through them. With more,
smaller chunks per rank the bubble shrinks at the cost of extra
peer-to-peer transfers. We print the schedule grid so the
interleaving pattern is visible.

Run::

    python -m examples.06_spmd_scheduled.05_virtual_stages
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.nn import Linear, PipelineSequential, RMSNorm
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import InterleavedH1
from spectrax.runtime.types import MpMdMesh


class MiniBlock(spx.Module):
    """Residual FFN block used as one logical micro-stage."""

    def __init__(self, d: int, *, rngs: spx.Rngs):
        """Initialise the norm and two linear projections."""
        super().__init__()
        self.norm = RMSNorm(d)
        self.up = Linear(d, 2 * d, rngs=rngs)
        self.down = Linear(2 * d, d, rngs=rngs)

    def __call__(self, x):
        """Apply the residual FFN."""
        return x + self.down(jax.nn.gelu(self.up(self.norm(x))))


def mse(out, target):
    """Plain MSE loss."""
    return jnp.mean((out - target) ** 2)


def _print_grid(schedule, n_ranks: int, max_rows: int = 14) -> None:
    """Print the action grid produced by ``schedule.build(n_stages=...)``."""
    grid = schedule.build(n_stages=n_ranks)
    print(f"  grid: {len(grid)} steps x {n_ranks} ranks")
    for t, row in enumerate(grid[:max_rows]):
        cells = [" .  " if a is None else f"{a.phase.name[0]}{a.microbatch}v{a.virtual_stage}" for a in row]
        print(f"  t={t:2d}  " + " ".join(f"{c:>5s}" for c in cells))
    if len(grid) > max_rows:
        print(f"  ... (+{len(grid) - max_rows} more)")


def main():
    """Run an InterleavedH1 SPMD step and print the V=2 schedule grid."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",))
    pp = int(mesh.shape["pp"])
    V = 2

    d, m = 16, 4
    n_stages = max(pp, 2)
    stages = [MiniBlock(d, rngs=spx.Rngs(i)) for i in range(n_stages)]
    model = PipelineSequential(*stages)

    x = jax.random.normal(jax.random.PRNGKey(0), (m, 8, d))
    y = jax.random.normal(jax.random.PRNGKey(1), (m, 8, d))

    schedule = InterleavedH1(microbatches=m, virtual_stages=V)
    print(f"pp={pp}  num_stages={model.num_stages}  V={V}  microbatches={m}")
    print(
        f"bubble={schedule.bubble_ratio(model.num_stages):.3f}  steps={schedule.total_steps(model.num_stages)}  vpr={schedule.virtual_stages_per_rank()}"
    )
    _print_grid(schedule, model.num_stages)

    if pp == model.num_stages and pp >= 2:
        mpmd_mesh = MpMdMesh(mesh, "pp")
        loss, grads = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=mse)
        print(f"InterleavedH1 train loss: {float(loss):.4f}  per-stage grads: {len(grads)}")
    else:
        print(f"[note] need pp=={model.num_stages} devices for SPMD; eager fallback.")
        print(f"eager loss: {float(mse(model(x), y)):.4f}")


if __name__ == "__main__":
    main()
