# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`spectrax.pipeline.MpMdMesh`."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from spectrax.runtime.types import MpMdMesh


def _get_devices(n: int):
    """Return the first ``n`` JAX devices, or skip the test if unavailable."""
    devs = jax.devices()[:n]
    if len(devs) < n:
        pytest.skip(f"need {n} devices; have {len(devs)}")
    return devs


def test_mpmd_mesh_1d():
    """Single-axis (pp,) mesh: every device is its own stage."""
    d = _get_devices(4)
    mm = MpMdMesh(Mesh(np.array(d), ("pp",)), "pp")
    assert mm.mpmd_dim == 4
    assert mm.mpmd_axis == 0
    assert mm.spmd_axis_names == ()
    assert len(mm.unstack()) == 4


def test_mpmd_mesh_2d_pp_first():
    """(pp, dp) mesh: pp is the MPMD axis, dp stays SPMD."""
    d = _get_devices(4)
    mm = MpMdMesh(Mesh(np.array(d).reshape(2, 2), ("pp", "dp")), "pp")
    assert mm.mpmd_dim == 2
    assert mm.mpmd_axis == 0
    assert mm.spmd_axis_names == ("dp",)
    subs = mm.unstack()
    assert len(subs) == 2
    for sub in subs:
        assert dict(sub.shape) == {"dp": 2}


def test_mpmd_mesh_2d_pp_second():
    """(dp, pp) mesh: MPMD axis index should be 1."""
    d = _get_devices(4)
    mm = MpMdMesh(Mesh(np.array(d).reshape(2, 2), ("dp", "pp")), "pp")
    assert mm.mpmd_dim == 2
    assert mm.mpmd_axis == 1
    assert mm.spmd_axis_names == ("dp",)


def test_mpmd_mesh_rejects_unknown_axis():
    """``mpmd_axis_name`` must appear in ``jax_mesh.axis_names``."""
    d = _get_devices(2)
    with pytest.raises(ValueError, match="mpmd_axis_name"):
        MpMdMesh(Mesh(np.array(d), ("pp",)), "bogus")


def test_mpmd_mesh_submesh_out_of_range():
    """``submesh(i)`` rejects out-of-range indices."""
    d = _get_devices(2)
    mm = MpMdMesh(Mesh(np.array(d), ("pp",)), "pp")
    with pytest.raises(IndexError):
        mm.submesh(2)
    with pytest.raises(IndexError):
        mm.submesh(-1)


def test_sub_sharding_replicated():
    """Default ``sub_sharding(i)`` returns a replicated NamedSharding."""
    d = _get_devices(4)
    mm = MpMdMesh(Mesh(np.array(d).reshape(2, 2), ("pp", "dp")), "pp")
    sh = mm.sub_sharding(0)
    assert isinstance(sh, NamedSharding)
    assert dict(sh.mesh.shape) == {"dp": 2}
    assert sh.spec == PartitionSpec()


def test_sub_sharding_with_spec():
    """``sub_sharding`` accepts a PartitionSpec over non-MPMD axes."""
    d = _get_devices(4)
    mm = MpMdMesh(Mesh(np.array(d).reshape(2, 2), ("pp", "dp")), "pp")
    sh = mm.sub_sharding(1, PartitionSpec("dp"))
    assert sh.spec == PartitionSpec("dp")


def test_sub_sharding_rejects_mpmd_axis():
    """``sub_sharding`` rejects a spec that mentions the MPMD axis."""
    d = _get_devices(4)
    mm = MpMdMesh(Mesh(np.array(d).reshape(2, 2), ("pp", "dp")), "pp")
    with pytest.raises(ValueError, match="MPMD axis"):
        mm.sub_sharding(0, PartitionSpec("pp"))


def test_device_mpmd_idx():
    """``device_mpmd_idx`` returns which stage a device belongs to."""
    d = _get_devices(4)
    mm = MpMdMesh(Mesh(np.array(d).reshape(2, 2), ("pp", "dp")), "pp")
    assert mm.device_mpmd_idx(d[0]) == 0
    assert mm.device_mpmd_idx(d[1]) == 0
    assert mm.device_mpmd_idx(d[2]) == 1
    assert mm.device_mpmd_idx(d[3]) == 1


def test_device_mpmd_idx_missing():
    """Unknown device raises ValueError."""
    d = _get_devices(2)
    mm = MpMdMesh(Mesh(np.array(d), ("pp",)), "pp")
    with pytest.raises(ValueError, match="not part of this mesh"):
        mm.device_mpmd_idx(object())


def test_my_mpmd_axis_index_single_process():
    """Single-process run should return ``None`` from ``my_mpmd_axis_index``."""
    d = _get_devices(2)
    mm = MpMdMesh(Mesh(np.array(d), ("pp",)), "pp")
    assert mm.my_mpmd_axis_index() is None
