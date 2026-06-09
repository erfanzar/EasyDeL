# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :func:`sxstage_region` markers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import spectrax as spx
from spectrax.runtime.mpmd import sxjit, sxvalue_and_grad
from spectrax.runtime.mpmd.markers import (
    cluster_jaxpr_by_markers,
    has_stage_regions,
    stage_region_cluster_boundaries,
    stage_region_specs,
    sxstage_region_enter_p,
    sxstage_region_exit_p,
)
from spectrax.runtime.types import MpMdMesh


def _two_rank_mesh() -> MpMdMesh:
    """Build a tiny two-rank MPMD mesh or skip when unavailable."""
    devices = jax.devices()[:2]
    if len(devices) < 2:
        pytest.skip("need at least 2 JAX devices for a two-rank MPMD schedule")
    return MpMdMesh(Mesh(np.asarray(devices, dtype=object).reshape(2), axis_names=("pp",)), "pp")


def test_stage_region_is_identity_for_eager_jit_and_grad():
    """Stage region is identity for eager jit and grad."""
    region = spx.sxstage_region("encoder", schedule=spx.GPipe(microbatches=2))

    def body(x):
        """Loop body function."""
        return region(lambda y: y * y + 1.0)(x).sum()

    x = jnp.arange(4, dtype=jnp.float32)
    assert np.allclose(np.asarray(body(x)), np.asarray((x * x + 1.0).sum()))
    assert np.allclose(np.asarray(jax.jit(body)(x)), np.asarray(body(x)))
    assert np.allclose(np.asarray(jax.grad(body)(x)), np.asarray(2.0 * x))


def test_stage_region_markers_and_metadata_are_in_jaxpr():
    """Stage region markers and metadata are in jaxpr."""
    region = spx.sxstage_region(
        "decoder",
        schedule=spx.DualPipeV(microbatches=4),
        batch_argnums=(1,),
        static_argnums=(2,),
        donate_argnums=(0,),
    )

    def body(x):
        """Loop body function."""
        return region(lambda y: y + 1)(x)

    jaxpr = jax.make_jaxpr(body)(jnp.ones((2,), dtype=jnp.float32)).jaxpr
    primitive_names = [eqn.primitive.name for eqn in jaxpr.eqns]
    assert "sxstage_region_enter" in primitive_names
    assert "sxstage_region_exit" in primitive_names

    specs = stage_region_specs(jaxpr)
    assert specs
    assert {spec.name for spec in specs} == {"decoder"}
    assert {spec.schedule_name for spec in specs} == {"DualPipeV"}
    assert {spec.microbatches for spec in specs} == {4}
    assert {spec.batch_argnums for spec in specs} == {(1,)}
    assert {spec.static_argnums for spec in specs} == {(2,)}
    assert {spec.donate_argnums for spec in specs} == {(0,)}


def test_stage_region_does_not_mark_integer_bool_metadata_operands():
    """Region markers should wrap floating activations, not static-shaped metadata."""
    region = spx.sxstage_region("vision")

    def body(x, grid_thw, mask):
        """Use int and bool metadata inside a marked region."""

        def inner(y, grid, keep):
            offset = grid.astype(y.dtype).sum()
            return jnp.where(keep, y + offset, y - offset)

        return region(inner)(x, grid_thw, mask)

    jaxpr = jax.make_jaxpr(body)(
        jnp.ones((2, 3), dtype=jnp.float32),
        jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
        jnp.asarray([[True, False, True], [False, True, False]]),
    ).jaxpr
    primitives = [eqn.primitive for eqn in jaxpr.eqns]

    assert primitives.count(sxstage_region_enter_p) == 1
    assert primitives.count(sxstage_region_exit_p) == 1


def test_exit_only_region_marker_does_not_enable_region_scheduling():
    """A dangling exit marker is not a valid region span."""
    region = spx.sxstage_region("bad_exit_only")

    def body(x):
        """Emit only an exit marker."""
        return region.exit(x + 1.0)

    jaxpr = jax.make_jaxpr(body)(jnp.ones((2,), dtype=jnp.float32)).jaxpr

    assert not has_stage_regions(jaxpr)


def test_serial_stage_regions_restart_local_stage_numbering():
    """Two serial leaf regions should form two independent two-stage paths."""
    vision = spx.sxstage_region("vision")
    text = spx.sxstage_region("text")

    def body(x):
        """Run two independently marked towers in sequence."""

        def vision_path(y):
            y = y * 2.0 + 1.0
            y = spx.sxstage_iter(y, stage=0)
            return jnp.sin(y)

        def text_path(y):
            y = y - 0.25
            y = spx.sxstage_iter(y, stage=0)
            return jnp.tanh(y)

        return text(text_path)(vision(vision_path)(x)).sum()

    jaxpr = jax.make_jaxpr(body)(jnp.ones((2, 3), dtype=jnp.float32)).jaxpr
    boundaries = stage_region_cluster_boundaries(jaxpr)
    clusters = cluster_jaxpr_by_markers(jaxpr, extra_boundary_positions=boundaries)

    assert len(boundaries) == 1
    assert len(clusters) == 4
    assert all("sxstage_iter" not in {eqn.primitive.name for eqn in cluster.eqns} for cluster in clusters)


def test_reused_stage_region_does_not_hide_parent_marker_between_calls():
    """Repeated calls with one region spec keep parent markers visible."""
    region = spx.sxstage_region("tower", schedule=spx.GPipe(microbatches=1))

    def body(x):
        """Body with two local markers and one parent marker."""
        x = region(lambda y: spx.sxstage_iter(y + 1.0, stage=0))(x)
        x = spx.sxstage_iter(x * 2.0, stage=0)
        return region(lambda y: spx.sxstage_iter(y + 3.0, stage=0))(x)

    jaxpr = jax.make_jaxpr(body)(jnp.ones((1,), dtype=jnp.float32)).jaxpr
    clusters = cluster_jaxpr_by_markers(jaxpr, ignore_region_local_markers=True)
    assert len(clusters) == 2


def test_stage_region_local_markers_do_not_split_parent_scheduled_sxjit():
    """Region-local sxstage_iter markers do not split the parent schedule."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1)
    mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")
    region = spx.sxstage_region("encoder", schedule=spx.GPipe(microbatches=1))

    @sxjit(mesh=mesh, schedule=spx.GPipe(microbatches=1))
    def body(x):
        """Loop body function."""
        return region(lambda y: spx.sxstage_iter(y + 1.0, stage=0))(x).sum()

    out = body(jnp.ones((1,), dtype=jnp.float32))
    assert np.allclose(np.asarray(out), np.asarray(2.0, dtype=np.float32))


def test_sxjit_schedule_executes_serial_stage_regions_on_two_ranks():
    """Scheduled sxjit should execute vision stages V0,V1 then text stages T0,T1."""
    mesh = _two_rank_mesh()
    vision = spx.sxstage_region("vision")
    text = spx.sxstage_region("text")

    def reference(x):
        """Unmarked reference computation."""
        h = jnp.sin(x * 1.5 + 0.5)
        h = jnp.tanh(h - 0.25) + 0.125 * h
        return h.mean()

    @sxjit(mesh=mesh, schedule=spx.GPipe(microbatches=2), batch_argnums=(0,))
    def scheduled(x):
        """Serial two-stage vision path followed by two-stage text path."""

        def vision_path(y):
            y = y * 1.5 + 0.5
            y = spx.sxstage_iter(y, stage=0)
            return jnp.sin(y)

        def text_path(y):
            carry = y
            y = y - 0.25
            y = spx.sxstage_iter(y, stage=0)
            return jnp.tanh(y) + 0.125 * carry

        return text(text_path)(vision(vision_path)(x)).mean()

    x = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32).reshape(4, 4)
    out = scheduled(x)
    plan = scheduled._mpmd_state["schedule_plan"]

    assert plan["serial_region_plan"] is True
    assert len(plan["cluster_jaxprs_per_loc"]) == 4
    assert tuple(plan["loc_for_logical"]) == ((0, 0), (1, 0), (0, 0), (1, 0))
    assert np.allclose(np.asarray(out), np.asarray(reference(x)), atol=1e-5)


@pytest.mark.parametrize(
    "schedule",
    [
        spx.GPipe(microbatches=2),
        spx.GPipe(microbatches=2, lazy_bwd_batching=True),
        spx.KimiK2(microbatches=4, virtual_stages=2),
        spx.DualPipeV(microbatches=4),
    ],
    ids=["gpipe", "gpipe_lazy_bwd", "kimi_k2", "dualpipe_v"],
)
def test_sxjit_schedule_serial_stage_region_value_and_grad_matches_jax(schedule):
    """Backward scheduling should preserve gradients across serial region paths."""
    mesh = _two_rank_mesh()
    left = spx.sxstage_region("left")
    right = spx.sxstage_region("right")
    stages_per_region = len(jax.devices()[:2]) * schedule.virtual_stages_per_rank()

    def reference(x, scale):
        """Reference computation without region markers."""
        h = x
        for stage in range(stages_per_region):
            factor = scale if stage == 0 else 1.0
            h = jnp.sin(h * factor + jnp.asarray(0.03 * (stage + 1), dtype=h.dtype))
        skip = h
        for stage in range(stages_per_region):
            h = jnp.tanh(h + jnp.asarray(0.05 * (stage + 1), dtype=h.dtype))
            if stage == 0:
                h = h * (skip - 0.1)
        return h.mean()

    @sxjit(mesh=mesh, schedule=schedule, batch_argnums=(0,))
    def scheduled(x, scale):
        """Same computation split into two independently staged regions."""

        def left_path(y, factor):
            for stage in range(stages_per_region):
                local_factor = factor if stage == 0 else 1.0
                y = jnp.sin(y * local_factor + jnp.asarray(0.03 * (stage + 1), dtype=y.dtype))
                if stage != stages_per_region - 1:
                    y = spx.sxstage_iter(y, stage=stage)
            return y

        def right_path(y):
            skip = y
            for stage in range(stages_per_region):
                y = jnp.tanh(y + jnp.asarray(0.05 * (stage + 1), dtype=y.dtype))
                if stage == 0:
                    y = y * (skip - 0.1)
                if stage != stages_per_region - 1:
                    y = spx.sxstage_iter(y, stage=stage)
            return y

        return right(right_path)(left(left_path)(x, scale)).mean()

    x = jnp.linspace(-0.5, 0.75, 16, dtype=jnp.float32).reshape(4, 4)
    scale = jnp.asarray(1.75, dtype=jnp.float32)
    value, (gx, gscale) = sxvalue_and_grad(scheduled, argnums=(0, 1))(x, scale)
    ref_value, ref_grads = jax.value_and_grad(reference, argnums=(0, 1))(x, scale)

    assert np.allclose(np.asarray(value), np.asarray(ref_value), atol=1e-5)
    assert np.allclose(np.asarray(gx), np.asarray(ref_grads[0]), atol=1e-5)
    assert np.allclose(np.asarray(gscale), np.asarray(ref_grads[1]), atol=1e-5)


def test_sxjit_schedule_shape_trace_uses_representative_microbatch():
    """The schedule build trace must see per-microbatch leading dimensions."""
    mesh = _two_rank_mesh()

    @sxjit(mesh=mesh, schedule=spx.GPipe(microbatches=2), batch_argnums=(0,))
    def scheduled(x, bias):
        """Choose a branch that is only correct for the microbatch shape."""
        if x.shape[0] == 2:
            h = x + bias
        else:
            h = x * 1000.0
        h = spx.sxstage_iter(h, stage=0)
        return h.mean()

    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    bias = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    out = scheduled(x, bias)

    assert np.allclose(np.asarray(out), np.asarray((x + bias).mean()), atol=1e-5)
