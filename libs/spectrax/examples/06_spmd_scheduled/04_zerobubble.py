# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-Bubble H1 schedule demo.

ZB-H1 splits the backward pass into ``BWD_I`` (input grads — on the
critical path, needed for the previous stage) and ``BWD_W``
(weight grads — deferrable, used to fill bubbles). Compared to
1F1B, ZB-H1 slots BWD_W into what would otherwise be idle time,
shrinking the bubble without raising peak activations.

We print the bubble ratio alongside GPipe / Std1F1B for context.

Run::

    python -m examples.06_spmd_scheduled.04_zerobubble
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.nn import Linear, PipelineSequential, RMSNorm
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import GPipe, Std1F1B, ZeroBubbleH1
from spectrax.runtime.types import MpMdMesh


class MiniBlock(spx.Module):
    """Residual FFN block."""

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
    """Mean-squared-error loss."""
    return jnp.mean((out - target) ** 2)


def main():
    """Run one ZeroBubbleH1 step and compare bubble ratios to GPipe / 1F1B."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",))
    n_stages = int(mesh.shape["pp"])
    d, m = 32, 4

    model = PipelineSequential(*[MiniBlock(d, rngs=spx.Rngs(i)) for i in range(max(n_stages, 2))])
    x = jax.random.normal(jax.random.PRNGKey(0), (m, 8, d))
    y = jax.random.normal(jax.random.PRNGKey(1), (m, 8, d))

    n = model.num_stages
    print(f"pp={n_stages}  num_stages={n}  microbatches={m}")
    for name, sc in [("GPipe", GPipe(m)), ("Std1F1B", Std1F1B(m)), ("ZeroBubbleH1", ZeroBubbleH1(m))]:
        print(
            f"  {name:13s}  bubble={sc.bubble_ratio(n):.3f}  steps={sc.total_steps(n)}  peak_acts={sc.peak_activations(n)}"
        )

    schedule = ZeroBubbleH1(microbatches=m)
    if n_stages == n and n_stages >= 2:
        mpmd_mesh = MpMdMesh(mesh, "pp")
        loss, grads = sxcall(model, (x, y), mesh=mpmd_mesh, schedule=schedule, loss_fn=mse)
        print(f"ZB-H1 train loss: {float(loss):.4f}  per-stage grads: {len(grads)}")
    else:
        print(f"[note] need pp=={n} devices for SPMD; eager fallback.")
        print(f"eager loss: {float(mse(model(x), y)):.4f}")


if __name__ == "__main__":
    main()
