# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same model as 02 but scheduled with :class:`Std1F1B`.

1F1B interleaves one forward and one backward per step once the
pipeline is full. That caps activation memory at roughly
``n_stages`` live microbatches rather than GPipe's ``m``.

We time one step for both GPipe and 1F1B and print a comparison.

Run::

    python -m examples.06_spmd_scheduled.03_1f1b
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import spectrax as spx
from spectrax.nn import Linear, PipelineSequential, RMSNorm
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import GPipe, Std1F1B
from spectrax.runtime.types import MpMdMesh


class MiniBlock(spx.Module):
    """Residual ``RMSNorm -> up -> GELU -> down`` block; one per stage."""

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
    """Plain MSE loss applied at the last stage."""
    return jnp.mean((out - target) ** 2)


def _time(fn, n: int = 3):
    """Return (result, seconds-per-call) for a zero-arg callable."""
    out = fn()
    jax.block_until_ready(out[0])
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        jax.block_until_ready(out[0])
    return out, (time.perf_counter() - t0) / n


def main():
    """Run one GPipe step and one Std1F1B step, print loss and step-time."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",))
    n_stages = int(mesh.shape["pp"])
    d, m = 32, 4

    model = PipelineSequential(*[MiniBlock(d, rngs=spx.Rngs(i)) for i in range(max(n_stages, 2))])
    x = jax.random.normal(jax.random.PRNGKey(0), (m, 8, d))
    y = jax.random.normal(jax.random.PRNGKey(1), (m, 8, d))

    n = model.num_stages
    gp, f1 = GPipe(m), Std1F1B(m)
    print(f"GPipe   peak_acts={gp.peak_activations(n)}  steps={gp.total_steps(n)}  bubble={gp.bubble_ratio(n):.3f}")
    print(f"Std1F1B peak_acts={f1.peak_activations(n)}  steps={f1.total_steps(n)}  bubble={f1.bubble_ratio(n):.3f}")

    if n_stages == n and n_stages >= 2:
        mpmd_mesh = MpMdMesh(mesh, "pp")
        (loss_g, _), tg = _time(lambda: sxcall(model, (x, y), mesh=mpmd_mesh, schedule=gp, loss_fn=mse))
        (loss_f, _), tf = _time(lambda: sxcall(model, (x, y), mesh=mpmd_mesh, schedule=f1, loss_fn=mse))
        print(f"GPipe   loss={float(loss_g):.4f}  step={tg * 1e3:.2f} ms")
        print(f"Std1F1B loss={float(loss_f):.4f}  step={tf * 1e3:.2f} ms")
    else:
        print(f"[note] need pp=={n} devices for SPMD; eager fallback.")
        print(f"eager loss: {float(mse(model(x), y)):.4f}")


if __name__ == "__main__":
    main()
