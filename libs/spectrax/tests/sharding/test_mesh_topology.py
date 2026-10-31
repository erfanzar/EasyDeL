# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device order of :func:`spectrax.sharding.mesh._cached_mesh_impl` on a multi-process, single-slice TPU job.

Uses fake v5p-128 chips (4x4x4 torus, 16 hosts of 2x2x1 chips) and only inspects which chips land in which mesh
group, so no array computation runs.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

mesh_mod = importlib.import_module("spectrax.sharding.mesh")

pytestmark = pytest.mark.cpu_ok

NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


class FakeChip:
    """Just the attributes ``jax.experimental.mesh_utils`` reads from a TPU device."""

    def __init__(self, idx: int, coords: tuple[int, int, int], process_index: int, platform: str = "tpu"):
        self.id = idx
        self.coords = coords
        self.core_on_chip = 0
        self.process_index = process_index
        self.slice_index = 0
        self.platform = platform
        self.device_kind = "TPU v5"

    def __repr__(self) -> str:
        return f"FakeChip({self.coords}, p{self.process_index})"


def v5p_128(platform: str = "tpu") -> list[FakeChip]:
    """64 chips on a 4x4x4 torus, 16 hosts of 2x2x1 chips (host bounds 2,2,4), process-major like ``jax.devices()``."""
    chips: list[FakeChip] = []
    for z in range(4):
        for yb in range(2):
            for xb in range(2):
                proc = xb + 2 * yb + 4 * z
                for y in range(2):
                    for x in range(2):
                        chips.append(FakeChip(len(chips), (2 * xb + x, 2 * yb + y, z), proc, platform))
    return chips


def axis_groups(devs: np.ndarray, axis: str) -> np.ndarray:
    """``[n_groups, axis_size]`` devices of every group along ``axis``."""
    i = NAMES.index(axis)
    return np.moveaxis(devs, i, -1).reshape(-1, devs.shape[i])


def is_torus_ring(group) -> bool:
    """The chips differ on exactly one torus coordinate and cover all 4 positions of it."""
    coords = [d.coords for d in group]
    varying = [k for k in range(3) if len({c[k] for c in coords}) > 1]
    return len(varying) == 1 and len({c[varying[0]] for c in coords}) == 4


@pytest.fixture
def fake_pod(monkeypatch):
    chips = v5p_128()
    monkeypatch.setattr(mesh_mod.jax, "devices", lambda backend=None: chips)
    monkeypatch.setattr(mesh_mod.jax, "device_count", lambda backend=None: len(chips))
    monkeypatch.setattr(mesh_mod.jax, "local_device_count", lambda backend=None: 4)
    monkeypatch.setattr(mesh_mod.jax, "process_count", lambda: 16)
    monkeypatch.setattr(mesh_mod, "Mesh", lambda devices, names, axis_types=None: devices)
    monkeypatch.delenv("MEGASCALE_NUM_SLICES", raising=False)
    return chips


def build(dims, dcn=None) -> np.ndarray:
    return mesh_mod._cached_mesh_impl.__wrapped__(
        axis_dims=tuple(dims),
        axis_names=NAMES,
        axis_types=None,
        dcn_mesh_dims=dcn,
        should_sort_granules_by_key=True,
        allow_split_physical_axes=True,
        backend="tpu",
    )


def test_multi_process_single_slice_tpu_lays_axes_on_torus_rings(fake_pod):
    devs = build((1, 1, 16, 1, 4, 1))
    assert devs.shape == (1, 1, 16, 1, 4, 1)
    assert sorted(d.id for d in devs.flat) == list(range(64))
    for group in axis_groups(devs, "tp"):
        assert is_torus_ring(group), [d.coords for d in group]
        assert len({d.process_index for d in group}) == 2  # the ring crosses a host boundary
    for group in axis_groups(devs, "fsdp"):
        coords = [d.coords for d in group]
        assert len(set(coords)) == 16
        assert sum(len({c[k] for c in coords}) > 1 for k in range(3)) == 2  # one torus plane


def test_explicit_dcn_mesh_dims_keep_the_per_host_layout(fake_pod):
    devs = build((1, 1, 16, 1, 4, 1), dcn=(1, 1, 16, 1, 1, 1))
    for group in axis_groups(devs, "tp"):
        assert len({d.process_index for d in group}) == 1
    assert sorted(d.id for d in devs.flat) == list(range(64))


def test_pipeline_meshes_keep_whole_host_stages(fake_pod):
    devs = build((2, 1, 8, 1, 4, 1))
    for group in axis_groups(devs, "tp"):
        assert len({d.process_index for d in group}) == 1
    stage_procs = [{d.process_index for d in devs[s].flat} for s in range(2)]
    assert all(len(p) == 8 for p in stage_procs) and not (stage_procs[0] & stage_procs[1])


def test_helper_is_tpu_only():
    gpu_like = v5p_128(platform="gpu")
    assert (
        mesh_mod._single_slice_tpu_mesh_devices(
            (1, 1, 16, 1, 4, 1), NAMES, gpu_like, dcn_mesh_dims=None, allow_split_physical_axes=True
        )
        is None
    )


def test_unmappable_shape_warns_and_falls_back(monkeypatch):
    warned: list[str] = []

    class _Logger:
        def warn_once(self, message, *args, **kwargs):
            warned.append(message % args if args else message)

    monkeypatch.setattr(mesh_mod, "logger", _Logger())
    out = mesh_mod._single_slice_tpu_mesh_devices(
        (1, 1, 8, 1, 8, 1), NAMES, v5p_128(), dcn_mesh_dims=None, allow_split_physical_axes=False
    )
    assert out is None
    assert len(warned) == 1 and "falling back to the per-host hybrid mesh" in warned[0]
