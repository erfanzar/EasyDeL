# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end pipeline tests: loss + grads match a single-device baseline.

Kept small on purpose (d=4, batch=8, M=4) so compile dominates only once
per schedule. Module-scoped fixtures share model, inputs, reference loss
and grads, and ``loss_fn`` across every test in the file, which also lets
the runtime compile cache hit between tests.

The tiny 2-stage by 2-microbatch configuration still exercises every
runtime code path (cross-stage ppermute, per-rank phase dispatch, scan
body branching) while keeping XLA compile time low — XLA compile time
scales with the schedule grid (``T x n_stages``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh

import spectrax as spx
from spectrax import nn
from spectrax.nn import PipelineSequential
from spectrax.runtime.mpmd import sxcall
from spectrax.runtime.schedules import GPipe, InterleavedH1, Std1F1B, ZeroBubbleH1
from spectrax.runtime.spmd.runtime import spmd_run
from spectrax.runtime.types import MpMdMesh

_N_STAGES = 2
_D = 4
_M = 2
_BATCH = 4


class _Stage(spx.Module):
    """Identical-shape stage used by pipeline e2e tests."""

    def __init__(self, d, *, rngs):
        """Build a single-linear-layer stage of width ``d``."""
        super().__init__()
        self.fc = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        """Apply the linear layer followed by a ReLU."""
        return jax.nn.relu(self.fc(x))


def _loss_fn(out, y):
    """Mean-squared-error loss; module-level so ``id(loss_fn)`` stays stable across tests."""
    return ((out - y) ** 2).mean()


@pytest.fixture(scope="module")
def model():
    """Module-scoped ``PipelineSequential`` of ``_N_STAGES`` identical stages."""
    return PipelineSequential(*[_Stage(_D, rngs=spx.Rngs(i + 1)) for i in range(_N_STAGES)])


@pytest.fixture(scope="module")
def xy():
    """Module-scoped ``(x, y)`` inputs sampled once so compile cache can hit."""
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _D))
    return x, y


@pytest.fixture(scope="module")
def mesh():
    """Module-scoped 1D ``pp`` :class:`MpMdMesh` with ``_N_STAGES`` devices."""
    devices = jax.devices()[:_N_STAGES]
    if len(devices) < _N_STAGES:
        pytest.skip(f"need {_N_STAGES} devices; have {len(devices)}")
    return MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")


@pytest.fixture(scope="module")
def reference(model, xy):
    """Single-device loss + per-stage param grads, computed once per module."""
    x, y = xy

    def full_loss(m, x, y):
        """Baseline loss on the un-pipelined model."""
        return ((m(x) - y) ** 2).mean()

    loss_val, grads_full = spx.value_and_grad(full_loss)(model, x, y)
    per_stage = []
    for i in range(model.num_stages):
        prefix = f"{i}."
        stage_params = {}
        for c, p, v in grads_full.items():
            if c == "parameters" and p.startswith(prefix):
                stage_params[p[len(prefix) :]] = v
        per_stage.append(stage_params)
    return loss_val, per_stage


@pytest.mark.parametrize("schedule_cls", [GPipe, Std1F1B, ZeroBubbleH1])
def test_pipeline_loss_matches_reference(schedule_cls, model, xy, mesh, reference):
    """Pipeline loss equals single-device loss for every schedule."""
    x, y = xy
    ref_loss, _ = reference
    loss, _grads = sxcall(
        model,
        (x, y),
        mesh=mesh,
        schedule=schedule_cls(microbatches=_M),
        loss_fn=_loss_fn,
    )
    assert jnp.allclose(loss, ref_loss, atol=1e-4, rtol=1e-4), (
        f"Pipeline loss {float(loss):.6f} != reference {float(ref_loss):.6f} for schedule {schedule_cls.__name__}"
    )


def test_pipeline_grads_match_reference_gpipe(model, xy, mesh, reference):
    """GPipe grads equal single-device grads (runtime correctness)."""
    x, y = xy
    _, ref_grads = reference
    _loss, pipeline_grads = sxcall(
        model,
        (x, y),
        mesh=mesh,
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )
    for stage_i, (pg, rg) in enumerate(zip(pipeline_grads, ref_grads, strict=False)):
        for path, ref_leaf in rg.items():
            pipe_leaf = pg.get("parameters", path)
            assert jnp.allclose(pipe_leaf, ref_leaf, atol=1e-3, rtol=1e-3), (
                f"Grad mismatch at stage {stage_i}, path {path!r}"
            )


def test_pipeline_interleaved_smoke(model, xy, mesh):
    """Interleaved1F1B runs without crashing (loss finite)."""
    x, y = xy
    model_2x = PipelineSequential(*[_Stage(_D, rngs=spx.Rngs(i + 1)) for i in range(_N_STAGES * 2)])
    schedule = InterleavedH1(microbatches=_M, virtual_stages=2)
    loss, _grads = sxcall(
        model_2x,
        (x, y),
        mesh=mesh,
        schedule=schedule,
        loss_fn=_loss_fn,
    )
    assert jnp.isfinite(loss)


def test_pipeline_batch_not_divisible_by_microbatches_errors(model, mesh):
    """Microbatching requires ``batch % microbatches == 0``."""
    x = jax.random.normal(jax.random.PRNGKey(0), (7, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (7, _D))
    with pytest.raises(ValueError, match="not divisible"):
        sxcall(
            model,
            (x, y),
            mesh=mesh,
            schedule=GPipe(microbatches=_M),
            loss_fn=_loss_fn,
        )


def test_pipeline_stage_mismatch_errors(mesh):
    """``num_stages`` must match the pp axis size."""
    mismatch = PipelineSequential(_Stage(_D, rngs=spx.Rngs(1)))
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _D))
    with pytest.raises(ValueError, match="stages"):
        sxcall(
            mismatch,
            (x, y),
            mesh=mesh,
            schedule=GPipe(microbatches=_M),
            loss_fn=_loss_fn,
        )


def test_spmd_run_reloads_mutated_live_parameters():
    """SPMD extraction/placement caches must not reuse stale parameter values."""
    devices = jax.devices()[:_N_STAGES]
    if len(devices) < _N_STAGES:
        pytest.skip(f"need {_N_STAGES} devices; have {len(devices)}")
    raw_mesh = Mesh(devices, axis_names=("pp",))
    model = PipelineSequential(*[_Stage(_D, rngs=spx.Rngs(i + 10)) for i in range(_N_STAGES)])
    x = jax.random.normal(jax.random.PRNGKey(20), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(21), (_BATCH, _D))

    loss_before, _ = spmd_run(
        model,
        (x, y),
        mesh=raw_mesh,
        axis="pp",
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )
    model[0].fc.weight.value = jnp.zeros_like(model[0].fc.weight.value)
    loss_after, _ = spmd_run(
        model,
        (x, y),
        mesh=raw_mesh,
        axis="pp",
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )

    assert not jnp.allclose(loss_before, loss_after)
