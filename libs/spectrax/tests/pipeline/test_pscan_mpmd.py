# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end ``sxjit + treduce`` tests on TPU."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

import spectrax as spx
from spectrax import nn
from spectrax.core.stage_assignment import assign_stage
from spectrax.runtime.mpmd import sxjit, sxstage_iter, treduce
from spectrax.runtime.mpmd.pscan_compiler import build_pscan_plan, has_pscan
from spectrax.runtime.schedules import (
    DualPipeV,
    Eager1F1B,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Std1F1B,
    ZeroBubbleH1,
)
from spectrax.runtime.types import MpMdMesh
from spectrax.sharding import logical_axis_rules

_D = 4
_BATCH = 4
_M = 4


class TwoStage(spx.Module):
    """Two linear layers with one explicit pipeline boundary."""

    def __init__(self, d, *, rngs):
        """Initialize with l1, l2."""
        super().__init__()
        self.l1 = nn.Linear(d, d, rngs=rngs)
        self.l2 = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        """Run the forward pass."""
        h = self.l1(x)
        h = sxstage_iter(h)
        return self.l2(h)


class FourStage(spx.Module):
    """Four linear layers with three explicit pipeline boundaries."""

    def __init__(self, d, *, rngs):
        """Initialize with l1, l2, l3 and 1 other(s)."""
        super().__init__()
        self.l1 = nn.Linear(d, d, rngs=rngs)
        self.l2 = nn.Linear(d, d, rngs=rngs)
        self.l3 = nn.Linear(d, d, rngs=rngs)
        self.l4 = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        """Run the forward pass."""
        h = self.l1(x)
        h = sxstage_iter(h)
        h = self.l2(h)
        h = sxstage_iter(h)
        h = self.l3(h)
        h = sxstage_iter(h)
        return self.l4(h)


class TPAnnotatedTwoStage(spx.Module):
    """Two-stage model with TP-annotated stage-local weights."""

    def __init__(self, d, *, rngs):
        """Initialize with l1, l2."""
        super().__init__()
        self.l1 = nn.Linear(d, d, use_bias=False, sharding=(None, "model"), rngs=rngs)
        self.l2 = nn.Linear(d, d, use_bias=False, sharding=(None, "model"), rngs=rngs)

    def forward(self, x):
        """Run the forward pass."""
        h = self.l1(x)
        h = sxstage_iter(h)
        return self.l2(h)


class StageTaggedAffine(spx.Module):
    """Custom stage using raw Parameter/Buffer creation under ``assign_stage``."""

    def __init__(self, d, *, total, current):
        """Initialize the instance."""
        super().__init__()
        with assign_stage(total=total, current=current):
            self.weight = spx.Parameter(jnp.eye(d))
            self.bias = spx.Buffer(jnp.zeros((d,)))

    def forward(self, x):
        """Run the forward pass."""
        return x @ self.weight + self.bias


class MisassignedTwoStage(spx.Module):
    """Two-stage model whose explicit stage hints intentionally disagree with execution."""

    def __init__(self, d):
        """Initialize with left, right."""
        super().__init__()
        self.left = StageTaggedAffine(d, total=2, current=1)
        self.right = StageTaggedAffine(d, total=2, current=0)

    def forward(self, x):
        """Run the forward pass."""
        h = self.left(x)
        h = sxstage_iter(h)
        return self.right(h)


def _micro_loss(model, mb):
    """Per-microbatch squared-error loss."""
    i_x, i_y = mb
    pred = model(i_x[None])
    return jnp.sum((pred - i_y[None]) ** 2)


def _full_loss(model, x, y):
    """Full-batch reference loss."""
    pred = model(x)
    return jnp.sum((pred - y) ** 2)


def _microbatch_reference(model, x, y):
    """Sequential non-pipelined microbatch reference."""
    losses = []
    grads = None
    for mb in range(_M):
        loss_mb, grad_mb = spx.value_and_grad(_micro_loss)(model, (x[mb], y[mb]))
        losses.append(loss_mb)
        grads = grad_mb if grads is None else jax.tree.map(lambda a, b: a + b, grads, grad_mb)
    return jnp.stack(losses), grads


@pytest.fixture(scope="module")
def mesh():
    """Two-rank TPU mesh for MPMD tests."""
    devices = [d for d in jax.devices() if d.platform == "tpu"][:2]
    if len(devices) < 2:
        pytest.skip("need 2 TPU devices for pscan mpmd tests")
    return MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")


@pytest.fixture(scope="module")
def model():
    """Shared tiny two-stage model."""
    return TwoStage(_D, rngs=spx.Rngs(0))


@pytest.fixture(scope="module")
def virtual_model():
    """Shared four-stage model for virtual-stage schedules."""
    return FourStage(_D, rngs=spx.Rngs(1))


@pytest.fixture(scope="module")
def xy():
    """Shared batch for every schedule/body-mode combination."""
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _D))
    return x, y


@pytest.fixture(scope="module")
def reference(model, xy):
    """Sequential non-pipelined microbatch loss buffer and summed grads."""
    x, y = xy
    return _microbatch_reference(model, x, y)


@pytest.fixture(scope="module")
def virtual_reference(virtual_model, xy):
    """Sequential non-pipelined reference for the 4-logical-stage model."""
    x, y = xy
    return _microbatch_reference(virtual_model, x, y)


@pytest.fixture(scope="module")
def mesh_pp_tp():
    """Two pipeline stages, each with a 2-way TP submesh."""
    devices = [d for d in jax.devices() if d.platform == "tpu"][:4]
    if len(devices) < 4:
        pytest.skip("need 4 TPU devices for pp x tp pscan tests")
    return spx.create_mesh(
        axis_dims=(2, 2),
        axis_names=("pp", "tp"),
        mpmd_axis="pp",
    )


@pytest.mark.parametrize("schedule_cls", [GPipe, Std1F1B, Eager1F1B, ZeroBubbleH1])
@pytest.mark.parametrize("body_mode", ["scalar_loss", "prediff"])
def test_pscan_mpmd_matches_reference(schedule_cls, body_mode, mesh, model, xy, reference):
    """Compiled ``treduce`` matches a non-pipelined reference on TPU."""
    x, y = xy
    ref_losses, ref_grads = reference
    schedule = schedule_cls(microbatches=_M)

    @sxjit(mesh=mesh)
    def step(model, x, y):
        """Execute one training step and return the result."""

        def body(mb):
            """Loop body function."""
            if body_mode == "scalar_loss":
                return _micro_loss(model, mb)
            return spx.value_and_grad(_micro_loss)(model, mb)

        return treduce(body, (x, y), schedule=schedule)

    losses, grads = step(model, x, y)
    assert losses.shape == (_M,)
    assert jnp.allclose(losses, ref_losses, atol=1e-4, rtol=1e-4)

    ref_leaves = jax.tree.leaves(ref_grads)
    got_leaves = jax.tree.leaves(grads)
    assert len(got_leaves) == len(ref_leaves)
    for got, ref in zip(got_leaves, ref_leaves, strict=True):
        assert jnp.allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("schedule_name", "schedule_factory"),
    [
        ("InterleavedH1", lambda m: InterleavedH1(microbatches=m, virtual_stages=2)),
        ("Interleaved1F1BPlusOne", lambda m: Interleaved1F1BPlusOne(microbatches=m, virtual_stages=2)),
        ("InterleavedGPipe", lambda m: InterleavedGPipe(microbatches=m, virtual_stages=2)),
        ("KimiK2", lambda m: KimiK2(microbatches=m, virtual_stages=2)),
        ("DualPipeV", lambda m: DualPipeV(microbatches=m)),
    ],
)
@pytest.mark.parametrize("body_mode", ["scalar_loss", "prediff"])
def test_pscan_mpmd_virtual_schedules_match_reference(
    schedule_name,
    schedule_factory,
    body_mode,
    mesh,
    virtual_model,
    xy,
    virtual_reference,
):
    """Compiled ``treduce`` matches reference for virtual-stage schedules on TPU."""
    del schedule_name
    x, y = xy
    ref_losses, ref_grads = virtual_reference
    schedule = schedule_factory(_M)

    @sxjit(mesh=mesh)
    def step(model, x, y):
        """Execute one training step and return the result."""

        def body(mb):
            """Loop body function."""
            if body_mode == "scalar_loss":
                return _micro_loss(model, mb)
            return spx.value_and_grad(_micro_loss)(model, mb)

        return treduce(body, (x, y), schedule=schedule)

    losses, grads = step(virtual_model, x, y)
    assert losses.shape == (_M,)
    assert jnp.allclose(losses, ref_losses, atol=1e-4, rtol=1e-4)

    ref_leaves = jax.tree.leaves(ref_grads)
    got_leaves = jax.tree.leaves(grads)
    assert len(got_leaves) == len(ref_leaves)
    for got, ref in zip(got_leaves, ref_leaves, strict=True):
        assert jnp.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_pscan_plan_auto_places_tp_consts_on_stage_submesh(mesh_pp_tp):
    """TP-annotated stage weights are resolved against the owning stage submesh."""
    d = 8
    x = jax.random.normal(jax.random.PRNGKey(11), (_BATCH, d))
    y = jax.random.normal(jax.random.PRNGKey(12), (_BATCH, d))

    with mesh_pp_tp, logical_axis_rules([("model", "tp")]):
        model = TPAnnotatedTwoStage(d, rngs=spx.Rngs(7))

        def step(model, x, y):
            """Execute one training step and return the result."""

            def body(mb):
                """Loop body function."""
                i_x, i_y = mb
                pred = model(i_x[None])
                return jnp.sum((pred - i_y[None]) ** 2)

            return treduce(body, (x, y), schedule=GPipe(microbatches=_M))

        closed_jaxpr = jax.make_jaxpr(step)(model, x, y)
        [pscan_eqn] = has_pscan(closed_jaxpr.jaxpr)
        outer_args = (model, x, y)
        outer_flat_args = tuple(jax.tree.leaves(outer_args))
        mpmd_mesh = mesh_pp_tp.mpmd_mesh
        assert mpmd_mesh is not None
        stage_shardings = [mpmd_mesh.sub_sharding(i) for i in range(mpmd_mesh.mpmd_dim)]
        rank_submeshes = [mpmd_mesh.submesh(i) for i in range(mpmd_mesh.mpmd_dim)]
        plan = build_pscan_plan(
            closed_jaxpr,
            outer_args,
            outer_flat_args,
            pscan_eqn,
            mpmd_mesh,
            stage_shardings,
            rank_submeshes,
        )

    stage0_weights = [c for c in plan.per_loc_consts[(0, 0)] if getattr(c, "shape", None) == (d, d)]
    stage1_weights = [c for c in plan.per_loc_consts[(1, 0)] if getattr(c, "shape", None) == (d, d)]
    assert len(stage0_weights) == 1
    assert len(stage1_weights) == 1

    weight0 = stage0_weights[0]
    weight1 = stage1_weights[0]
    assert weight0.sharding.spec == PartitionSpec(None, "tp")
    assert weight1.sharding.spec == PartitionSpec(None, "tp")
    assert np.array_equal(weight0.sharding.mesh.devices, rank_submeshes[0].devices)
    assert np.array_equal(weight1.sharding.mesh.devices, rank_submeshes[1].devices)


def test_assign_stage_mismatch_raises_for_forward_mpmd(mesh):
    """Explicit stage hints must agree with the traced pipeline segmentation."""
    model = MisassignedTwoStage(_D)
    x = jax.random.normal(jax.random.PRNGKey(23), (_BATCH, _D))

    @sxjit(mesh=mesh)
    def step(model, x):
        """Execute one training step and return the result."""
        return model(x)

    with pytest.raises(ValueError, match="assign_stage"):
        step(model, x)
