# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :func:`sxstage_iter` + jaxpr clustering."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Var
from jax.sharding import Mesh, PartitionSpec

from spectrax.runtime.mpmd import (
    cluster_jaxpr_by_markers,
    split_by_markers,
    sxenter_loop,
    sxexit_loop,
    sxloop,
    sxstage_iter,
)
from spectrax.runtime.mpmd.markers import marker_edge_shardings
from spectrax.runtime.mpmd.runtime import _edge_transfer_sharding
from spectrax.runtime.types.mesh import MpMdMesh


def test_marker_is_identity_eager():
    """Calling the marker on a concrete array returns it unchanged."""
    x = jnp.arange(3, dtype=jnp.float32)
    y = sxstage_iter(x, stage=0)
    assert np.array_equal(np.asarray(y), np.asarray(x))


def test_marker_is_identity_under_jit():
    """Marker passes through unchanged when invoked from within ``jax.jit``."""

    @jax.jit
    def f(x):
        """Jitted identity-plus-one marked at stage 0."""
        return sxstage_iter(x, stage=0) + 1

    out = f(jnp.ones((3,)))
    assert np.array_equal(np.asarray(out), 2 * np.ones((3,)))


def test_marker_shows_up_in_jaxpr():
    """The ``sxstage_iter`` primitive is present in the traced jaxpr."""

    def model(x):
        """Trivial two-op model with a single marker between ops."""
        h = x * 2
        h = sxstage_iter(h, stage=0)
        return h + 1

    jxpr = jax.make_jaxpr(model)(jnp.ones((3,)))
    prim_names = [str(e.primitive) for e in jxpr.jaxpr.eqns]
    assert "sxstage_iter" in prim_names


def test_marker_carries_edge_sharding_metadata():
    """``sxstage_iter(..., sharding=...)`` stores a boundary transfer spec."""

    def model(x):
        """Model factory helper."""
        h = x * 2
        h = sxstage_iter(h, stage=0, sharding=PartitionSpec(("fsdp", "sp"), "tp"))
        return h + 1

    jxpr = jax.make_jaxpr(model)(jnp.ones((2, 4)))
    assert marker_edge_shardings(jxpr.jaxpr) == [PartitionSpec(("fsdp", "sp"), "tp")]
    clusters = cluster_jaxpr_by_markers(jxpr.jaxpr)
    assert len(clusters) == 2


def test_edge_transfer_sharding_binds_to_destination_submesh_and_drops_pp():
    """Boundary specs are rebound to the target rank mesh, never the full PP axis."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    mpmd_mesh = MpMdMesh(Mesh(devices, ("pp", "tp")), "pp")
    rank_submeshes = [mpmd_mesh.submesh(0)]
    fallback = mpmd_mesh.sub_sharding(0)
    x = jnp.ones((2, 4), dtype=jnp.float32)

    target = _edge_transfer_sharding(
        x,
        edge_sharding=PartitionSpec("pp", "tp"),
        fallback_sharding=fallback,
        dst_rank=0,
        rank_submeshes=rank_submeshes,
        mpmd_mesh=mpmd_mesh,
    )

    assert target.mesh is rank_submeshes[0]
    assert target.spec == PartitionSpec(None, "tp")


def test_marker_transposes_as_identity():
    """Autograd through the marker produces identical grads to unmarked."""

    def marked(x):
        """Model with a marker between ops."""
        h = x * 2
        h = sxstage_iter(h)
        return (h * h).sum()

    def unmarked(x):
        """Equivalent model without the marker."""
        h = x * 2
        return (h * h).sum()

    x = jnp.arange(4, dtype=jnp.float32)
    g_marked = jax.grad(marked)(x)
    g_unmarked = jax.grad(unmarked)(x)
    assert jnp.allclose(g_marked, g_unmarked)


def test_cluster_jaxpr_three_segments():
    """Two markers partition the jaxpr into three marker-free segments."""

    def model(x):
        """Model with two markers, producing three clusters."""
        h = x * 2
        h = sxstage_iter(h, stage=0)
        h = h + 1
        h = sxstage_iter(h, stage=1)
        return h * h

    jxpr = jax.make_jaxpr(model)(jnp.ones((3,)))
    clusters = cluster_jaxpr_by_markers(jxpr.jaxpr)
    assert len(clusters) == 3
    for c in clusters:
        prims = {str(e.primitive) for e in c.eqns}
        assert "sxstage_iter" not in prims


def test_cluster_outvars_preserve_definition_order_for_donated_carry():
    """Stage carry outputs must stay in definition order for XLA aliasing.

    eSurge decode donates KV cache pages into a stage and returns the updated
    pages from attention calls. If the splitter permutes those output vars, XLA
    aliases each input buffer to the wrong result slot and inserts full-page
    copies. This regression test checks the splitter's structural contract
    without depending on a TPU or the attention kernel.
    """

    def model(a, b, c, x):
        a1 = a + 1.0
        b1 = b + 2.0
        c1 = c + 3.0
        h = sxstage_iter(x * 2.0, stage=0)
        return b1 + h, c1 + h, a1 + h

    jxpr = jax.make_jaxpr(model)(
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
    )
    clusters = cluster_jaxpr_by_markers(jxpr.jaxpr)

    defined_before_marker: list[Var] = []
    for eqn in jxpr.jaxpr.eqns:
        if str(eqn.primitive) == "sxstage_iter":
            break
        defined_before_marker.extend(outvar for outvar in eqn.outvars if isinstance(outvar, Var))

    cluster0_ids = {id(v) for v in clusters[0].outvars}
    expected = [v for v in defined_before_marker if id(v) in cluster0_ids]
    actual = clusters[0].outvars[: len(expected)]

    assert [id(v) for v in actual] == [id(v) for v in expected]


def test_split_by_markers_matches_original():
    """Composing per-stage functions reproduces the un-split model output."""
    W = jnp.ones((3, 3)) * 0.5

    def model(x):
        """Two-marker model over a small matmul + relu."""
        h = x @ W
        h = sxstage_iter(h, stage=0)
        h = jax.nn.relu(h)
        h = sxstage_iter(h, stage=1)
        return h.sum(axis=-1)

    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    expected = model(x)

    stage_fns = split_by_markers(model, x)
    assert len(stage_fns) == 3

    h = (x,)
    for f in stage_fns:
        h = f(*h)
    assert jnp.allclose(h[0], expected)


def test_cluster_rematerializes_input_only_preamble_in_consumers():
    """Input-only preamble values should not become pipeline activations."""

    def model(x, mask):
        """Build reusable mask/position data before the stage cut."""
        pos = jnp.arange(x.shape[0], dtype=x.dtype)
        bias = jnp.where(mask, pos, 0)
        h = x * 2
        h = sxstage_iter(h, stage=0)
        return h + bias

    x = jnp.arange(4, dtype=jnp.float32)
    mask = jnp.array([True, False, True, False])
    expected = model(x, mask)
    jxpr = jax.make_jaxpr(model)(x, mask)
    clusters = cluster_jaxpr_by_markers(jxpr.jaxpr)

    assert len(clusters) == 2
    assert len(clusters[0].outvars) == 1

    stage_fns = split_by_markers(model, x, mask)
    input_by_var = {
        id(jxpr.jaxpr.invars[0]): x,
        id(jxpr.jaxpr.invars[1]): mask,
    }
    stage0_args = tuple(input_by_var[id(v)] for v in clusters[0].invars)
    carry = stage_fns[0](*stage0_args)
    value_by_var = {
        **input_by_var,
        id(clusters[0].outvars[0]): carry[0],
    }
    for eqn in jxpr.jaxpr.eqns:
        if str(eqn.primitive) == "sxstage_iter":
            for marker_in, marker_out in zip(eqn.invars, eqn.outvars, strict=True):
                if id(marker_in) in value_by_var:
                    value_by_var[id(marker_out)] = value_by_var[id(marker_in)]
    stage1_args = tuple(value_by_var[id(v)] for v in clusters[1].invars)
    out = stage_fns[1](*stage1_args)
    assert jnp.allclose(out[0], expected)


def test_split_by_markers_no_markers_yields_single_stage():
    """A function with no markers clusters into exactly one stage."""

    def model(x):
        """Marker-free model."""
        return x * 3 + 1

    stage_fns = split_by_markers(model, jnp.ones((3,)))
    assert len(stage_fns) == 1
    (out,) = stage_fns[0](jnp.ones((3,)))
    assert jnp.allclose(out, 4 * jnp.ones((3,)))


def test_split_by_markers_with_closure_constants():
    """Constants closed-over by the traced fn are baked into each stage."""
    bias = jnp.array([10.0, 20.0, 30.0])

    def model(x):
        """Model closing over a constant bias vector."""
        h = x + bias
        h = sxstage_iter(h)
        return h * bias

    stage_fns = split_by_markers(model, jnp.zeros((3,)))
    assert len(stage_fns) == 2
    (h,) = stage_fns[0](jnp.zeros((3,)))
    (out,) = stage_fns[1](h)
    expected = model(jnp.zeros((3,)))
    assert jnp.allclose(out, expected)


def test_return_clusters_flag():
    """``return_clusters=True`` also returns the raw Jaxpr list + consts."""

    def model(x):
        """Single-marker model used for cluster-return inspection."""
        h = x * 2
        h = sxstage_iter(h)
        return h + 1

    stage_fns, clusters, _consts = split_by_markers(model, jnp.ones((3,)), return_clusters=True)
    assert len(stage_fns) == 2
    assert len(clusters) == 2


def test_mpmd_loop_scan_over_sequence():
    """sxloop iterates over a scanned sequence correctly."""
    weights = jnp.ones((3, 4, 4))

    def body(carry, w):
        """Loop body function."""
        return jnp.dot(carry, w), None

    x = jnp.ones((4, 4))
    out, _ = sxloop(body, x, weights)
    expected = jnp.dot(jnp.dot(jnp.dot(x, weights[0]), weights[1]), weights[2])
    assert jnp.allclose(out, expected)


def test_mpmd_loop_length_only():
    """sxloop with length and no xs repeats a closed-over body."""
    w = jnp.ones((4, 4)) * 2

    def body(carry, _):
        """Loop body function."""
        return jnp.dot(carry, w), None

    x = jnp.ones((4, 4))
    out, _ = sxloop(body, x, length=3)
    expected = jnp.full((4, 4), 8.0**3)
    assert jnp.allclose(out, expected)


def test_mpmd_loop_grad():
    """Grad flows correctly through sxloop."""
    weights = jnp.ones((2, 4, 4))

    def body(carry, w):
        """Loop body function."""
        return jnp.dot(carry, w), None

    def f(x):
        """Helper function."""
        out, _ = sxloop(body, x, weights)
        return out.sum()

    g = jax.grad(f)(jnp.ones((4, 4)))
    assert g.shape == (4, 4)
    assert jnp.all(g > 0)


def test_mpmd_enter_loop_is_identity():
    """sxenter_loop returns its input unchanged."""
    x = jnp.arange(3, dtype=jnp.float32)
    y = sxenter_loop(x, name="encoder")
    assert np.array_equal(np.asarray(y), np.asarray(x))


def test_mpmd_exit_loop_is_identity():
    """sxexit_loop returns its input unchanged."""
    x = jnp.arange(3, dtype=jnp.float32)
    y = sxexit_loop(x, name="decoder")
    assert np.array_equal(np.asarray(y), np.asarray(x))


def test_mpmd_enter_exit_loop_grad():
    """Grad flows through enter/exit markers as identities."""

    def f(x):
        """Helper function."""
        x = sxenter_loop(x, name="loop")
        x = x**2
        x = sxexit_loop(x, name="loop")
        return x.sum()

    g = jax.grad(f)(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(g, jnp.array([2.0, 4.0, 6.0]))


def test_mpmd_enter_exit_loop_under_jit():
    """Markers pass through jax.jit correctly."""

    @jax.jit
    def f(x):
        """Helper function."""
        x = sxenter_loop(x, name="loop")
        x = x * 3
        x = sxexit_loop(x, name="loop")
        return x

    out = f(jnp.ones((3,)))
    assert jnp.allclose(out, jnp.full((3,), 3.0))


def test_cluster_jaxpr_preserves_loop_markers():
    """Loop markers are kept inside clusters (not stripped like stage markers)."""

    def model(x):
        """Model factory helper."""
        x = sxenter_loop(x, name="layer_loop")
        x = x * 2
        x = sxexit_loop(x, name="layer_loop")
        x = sxstage_iter(x)
        return x + 1

    clusters = cluster_jaxpr_by_markers(jax.make_jaxpr(model)(jnp.ones((3,))).jaxpr)
    assert len(clusters) == 2
    enter_names = [e.primitive.name for e in clusters[0].eqns]
    assert "sxenter_loop" in enter_names
    assert "sxexit_loop" in enter_names
    assert "sxenter_loop" not in [e.primitive.name for e in clusters[1].eqns]
    assert "sxexit_loop" not in [e.primitive.name for e in clusters[1].eqns]
