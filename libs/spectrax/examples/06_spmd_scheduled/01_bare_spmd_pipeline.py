# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smallest possible scheduled pipeline.

Builds a trivial stack of MLP blocks, wraps them in
:class:`spectrax.pipeline.PipelineSequential`, and drives one
forward + backward step through
:func:`spectrax.pipeline.sxcall` with a :class:`GPipe` schedule
on an :class:`MpMdMesh`.

The number of pipeline stages is sized to the pp mesh axis so the
file works whether there are 1, 2, 4, or 8 devices available.

Run::

    python -m examples.06_spmd_scheduled.01_bare_spmd_pipeline
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.nn import Linear, PipelineSequential
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import GPipe
from spectrax.runtime.types import MpMdMesh


class Block(spx.Module):
    """One tiny MLP stage: ``gelu(Linear(x))``."""

    def __init__(self, d: int, *, rngs: spx.Rngs):
        """Create a ``d x d`` linear layer for this stage."""
        super().__init__()
        self.fc = Linear(d, d, rngs=rngs)

    def __call__(self, x):
        """Apply the linear layer followed by a GELU nonlinearity."""
        return jax.nn.gelu(self.fc(x))


def mse(out, target):
    """Mean-squared-error loss applied at the final pipeline stage."""
    return jnp.mean((out - target) ** 2)


def main():
    """Build a 2..N stage MLP, run one SPMD pipeline step, print the loss."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",))
    n_stages = int(mesh.shape["pp"])
    print(f"mesh: pp={n_stages}  devices={jax.device_count()}")

    d = 16
    stages = [Block(d, rngs=spx.Rngs(i)) for i in range(max(n_stages, 2))]
    model = PipelineSequential(*stages)

    m = 4
    x = jax.random.normal(jax.random.PRNGKey(0), (m, d))
    y = jax.random.normal(jax.random.PRNGKey(1), (m, d))

    if n_stages >= 2 and n_stages == model.num_stages:
        schedule = GPipe(microbatches=m)
        mpmd_mesh = MpMdMesh(mesh, "pp")
        loss, grads = sxcall(
            model,
            (x, y),
            mesh=mpmd_mesh,
            schedule=schedule,
            loss_fn=mse,
        )
        print(f"schedule: GPipe(m={m})  total_steps={schedule.total_steps(n_stages)}")
        print(f"loss: {float(loss):.4f}  per-stage grads: {len(grads)}")
    else:
        print(f"[note] need pp=={model.num_stages} devices for SPMD; running eager fallback.")
        out = model(x)
        loss = mse(out, y)
        print(f"eager loss: {float(loss):.4f}")


if __name__ == "__main__":
    main()
