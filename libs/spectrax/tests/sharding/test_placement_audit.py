# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``audit_placement`` and strict sharding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from jax.sharding import PartitionSpec as P
from spectrax import nn
from spectrax.core.sharding import Sharding
from spectrax.sharding import (
    audit_placement,
    explain_partition_spec_sanitization,
    is_strict_sharding,
    set_strict_sharding,
    strict_sharding,
    with_sharding_constraint,
)


class _StagedPair(spx.Module):
    def __init__(self, rngs):
        super().__init__()
        with spx.assign_stage(total=2, current=0):
            self.a = nn.Linear(8, 8, rngs=rngs)
        with spx.assign_stage(total=2, current=1):
            self.b = nn.Linear(8, 4, rngs=rngs)


class _Untagged(spx.Module):
    def __init__(self):
        super().__init__()
        self.w = spx.Parameter(jnp.ones((3,)), sharding=Sharding(mesh_axes=("dp",)))
        self.big = spx.Parameter(jnp.ones((16,)))


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs at least 4 devices")
    return spx.create_mesh((2, -1), ("pp", "dp"), mpmd_axis="pp")


def _placed_pair(mesh):
    with mesh:
        return _StagedPair(spx.Rngs(0))


def test_stage_tagged_model_is_clean(mesh):
    report = audit_placement(_placed_pair(mesh), mesh)
    assert report.ok, report.format()
    assert report.checked == 4
    assert report.stage_counts == {0: 2, 1: 2}


def test_value_on_wrong_stage_is_reported(mesh):
    model = _placed_pair(mesh)
    model.a.weight._raw_set(jax.device_put(model.a.weight._raw_get(), mesh.mpmd_mesh.sub_sharding(1)))
    report = audit_placement(model, mesh)
    (err,) = report.errors
    assert (err.code, err.path) == ("misplaced", "a.weight")
    with pytest.raises(spx.ShardingError, match="misplaced"):
        report.raise_for_errors()


def test_dropped_axis_and_untagged_replication_are_warnings(mesh):
    report = audit_placement(_Untagged(), mesh, replicated_warn_bytes=0, check_values=False)
    assert report.ok
    assert report.codes() == {"dropped_axis": 1, "replicated_across_stages": 2}


def test_explain_sanitization_reports_each_dropped_axis(mesh):
    spec, reasons = explain_partition_spec_sanitization(P("pp", "missing", "dp"), mesh=mesh, shape=(4, 4, 3))
    assert spec == P(None, None, None)
    assert len(reasons) == 2
    assert "absent from the mesh" in reasons[0] and "not divisible" in reasons[1]
    assert explain_partition_spec_sanitization(P("dp"), mesh=mesh, shape=(8,)) == (P("dp"), ())


def test_strict_mode_rejects_unresolved_stage(mesh):
    x = jnp.ones((8, 8))
    assert with_sharding_constraint(x, P("dp"), mesh=mesh) is x
    with strict_sharding():
        with pytest.raises(spx.ShardingError, match="could not resolve the MPMD stage"):
            with_sharding_constraint(x, P("dp"), mesh=mesh)
        assert with_sharding_constraint(x, P("dp"), mesh=mesh, ignore_mpmd=True) is x


def test_strict_mode_rejects_dropped_axis(mesh):
    x = jnp.ones((3,))
    with_sharding_constraint(x, P("dp"), mesh=mesh, stage=0)
    with strict_sharding():
        with pytest.raises(spx.ShardingError, match="not divisible"):
            with_sharding_constraint(x, P("dp"), mesh=mesh, stage=0)
        out = with_sharding_constraint(jnp.ones((8,)), P("dp"), mesh=mesh, stage=0)
        assert out.sharding.spec == P("dp")
        with strict_sharding(False):
            with_sharding_constraint(x, P("dp"), mesh=mesh, stage=0)


@pytest.mark.cpu_ok
def test_strict_mode_scoping():
    assert not is_strict_sharding()
    set_strict_sharding(True)
    try:
        assert is_strict_sharding()
        with strict_sharding(False):
            assert not is_strict_sharding()
    finally:
        set_strict_sharding(False)
    assert not is_strict_sharding()
