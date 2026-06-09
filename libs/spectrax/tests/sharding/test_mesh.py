# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.sharding.mesh`.

Runs against the CPU backend with four simulated devices (set by
:mod:`conftest`). Covers :func:`create_mesh`, :func:`create_cpu_mesh`,
:func:`parse_mesh_from_string`, :func:`force_cpu`, :func:`cpu_context`,
and :func:`calculate_host_mesh_shape`.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, Mesh

from spectrax.sharding import (
    DEFAULT_MESH_AXIS_DIMS,
    DEFAULT_MESH_AXIS_NAMES,
    SpxMesh,
    calculate_host_mesh_shape,
    cpu_context,
    create_cpu_mesh,
    create_mesh,
    current_mesh,
    force_cpu,
    get_incontext_mesh,
    parse_mesh_from_string,
    use_mesh,
)


def test_create_mesh_default_axes():
    """Default (1, 1, -1, 1, 1, 1) expands to fill every device on the fsdp axis."""
    mesh = create_mesh()
    assert isinstance(mesh, SpxMesh)
    assert isinstance(mesh.jax_mesh, Mesh)
    assert not mesh.is_mpmd
    assert mesh.mpmd_mesh is None
    assert mesh.axis_names == DEFAULT_MESH_AXIS_NAMES
    assert mesh.shape["fsdp"] == 4
    for name in ("pp", "dp", "ep", "tp", "sp"):
        assert mesh.shape[name] == 1


def test_create_mesh_with_mpmd_axis():
    """``mpmd_axis=`` produces an SpxMesh with a populated mpmd_mesh view."""
    mesh = create_mesh((2, 1, 1, 1, 2, 1), mpmd_axis="pp")
    assert isinstance(mesh, SpxMesh)
    assert mesh.is_mpmd
    assert mesh.mpmd_mesh is not None
    assert mesh.mpmd_mesh.mpmd_dim == 2
    assert mesh.mpmd_mesh.mpmd_axis_name == "pp"
    assert mesh.mpmd_mesh is mesh.mpmd_mesh


def test_create_mesh_rejects_unknown_mpmd_axis():
    """``mpmd_axis`` not in ``axis_names`` is rejected."""
    with pytest.raises(ValueError, match="mpmd_axis"):
        create_mesh((2, 2), ("a", "b"), mpmd_axis="bogus")


def test_create_mesh_custom_axes():
    """Custom axis_dims + axis_names round-trip correctly."""
    mesh = create_mesh((2, 2), ("data", "model"))
    assert mesh.axis_names == ("data", "model")
    assert mesh.shape["data"] == 2
    assert mesh.shape["model"] == 2


def test_create_mesh_is_cached():
    """Calling create_mesh with identical args returns the same object."""
    a = create_mesh((2, 2), ("data", "model"))
    b = create_mesh((2, 2), ("data", "model"))
    assert a is b


def test_create_mesh_with_axis_types_string():
    """String axis_types resolves to AxisType enums."""
    mesh = create_mesh((2, 2), ("data", "model"), axis_types="explicit")
    for at in mesh.axis_types:
        assert at == AxisType.Explicit


def test_create_mesh_with_axis_types_sequence():
    """Per-axis axis_types sequence applies element-wise."""
    mesh = create_mesh((2, 2), ("data", "model"), axis_types=("auto", "explicit"))
    assert mesh.axis_types[0] == AxisType.Auto
    assert mesh.axis_types[1] == AxisType.Explicit


def test_create_mesh_rejects_unknown_axis_type():
    """Unknown string in axis_types raises ValueError."""
    with pytest.raises(ValueError, match="axis_types"):
        create_mesh((2, 2), ("data", "model"), axis_types="bogus")


def test_create_mesh_rejects_length_mismatch():
    """axis_types sequence wrong length raises ValueError."""
    with pytest.raises(ValueError, match="length"):
        create_mesh((2, 2), ("data", "model"), axis_types=("auto", "explicit", "manual"))


def test_create_cpu_mesh_is_cpu():
    """create_cpu_mesh returns a mesh whose devices are all CPU."""
    mesh = create_cpu_mesh((1, 1, 4, 1, 1, 1))
    assert isinstance(mesh, SpxMesh)
    for d in mesh.devices.flatten():
        assert d.platform == "cpu"


def test_parse_mesh_from_string_named():
    """Named form 'a:2,b:2' parses correctly."""
    mesh = parse_mesh_from_string("a:2,b:2", ("a", "b"))
    assert isinstance(mesh, SpxMesh)
    assert mesh.shape["a"] == 2
    assert mesh.shape["b"] == 2


def test_parse_mesh_from_string_positional():
    """Positional form '2,2' maps to the given names in order."""
    mesh = parse_mesh_from_string("2,2", ("a", "b"))
    assert isinstance(mesh, SpxMesh)
    assert mesh.shape["a"] == 2
    assert mesh.shape["b"] == 2


def test_parse_mesh_from_string_unknown_name():
    """Unknown name in named form raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        parse_mesh_from_string("x:2,y:2", ("a", "b"))


def test_parse_mesh_from_string_missing_name():
    """Named form that skips one of ``names`` raises ValueError."""
    with pytest.raises(ValueError, match="Not all axis names"):
        parse_mesh_from_string("a:4", ("a", "b"))


def test_parse_mesh_from_string_length_mismatch():
    """Positional length != len(names) raises ValueError."""
    with pytest.raises(ValueError, match="match"):
        parse_mesh_from_string("2,2,2", ("a", "b"))


def test_force_cpu_sets_default_device():
    """force_cpu places newly-created arrays on a CPU device."""
    with force_cpu() as cpu:
        x = jnp.ones((4,))
        device = x.device if hasattr(x, "device") else next(iter(x.devices()))
        if hasattr(device, "platform"):
            assert device.platform == "cpu"
        assert cpu.platform == "cpu"


def test_cpu_context_yields_cpu_mesh():
    """cpu_context yields a CPU-backed mesh."""
    with cpu_context() as mesh:
        assert isinstance(mesh, SpxMesh)
        assert current_mesh() is mesh
        assert get_incontext_mesh() is mesh
        for d in mesh.devices.flatten():
            assert d.platform == "cpu"


def test_get_incontext_mesh_returns_spxmesh():
    """SpectraX mesh lookup preserves SpxMesh metadata."""
    mesh = create_mesh((2, 2), ("data", "model"))
    with mesh as active:
        assert active is mesh
        assert current_mesh() is mesh
        assert get_incontext_mesh() is mesh


def test_use_mesh_yields_spxmesh():
    """use_mesh is a SpectraX-facing context helper, not a raw Mesh escape hatch."""
    mesh = create_mesh((2, 2), ("data", "model"))
    with use_mesh(mesh) as active:
        assert active is mesh
        assert current_mesh() is mesh
        assert get_incontext_mesh() is mesh


def test_calculate_host_mesh_shape_single_process():
    """With 1 process, host mesh == global mesh."""
    out = calculate_host_mesh_shape((2, 4), total_devices=8, num_processes=1)
    assert out == (2, 4)


def test_calculate_host_mesh_shape_multi_process_splits_leading():
    """Process count absorbs the leading axis first."""
    out = calculate_host_mesh_shape((2, 4), total_devices=4, num_processes=2)
    assert out == (1, 4)


def test_calculate_host_mesh_shape_multi_process_splits_multiple_axes():
    """When one axis can't absorb all processes, the next axis shares the split."""
    out = calculate_host_mesh_shape((2, 2, 2), total_devices=2, num_processes=4)
    assert out == (1, 1, 2)


def test_calculate_host_mesh_shape_size_mismatch_errors():
    """Global mesh size must equal ``total_devices * num_processes``."""
    with pytest.raises(ValueError, match="doesn't match"):
        calculate_host_mesh_shape((2, 4), total_devices=4, num_processes=3)


def test_calculate_host_mesh_shape_rejects_zero_counts():
    """Explicit zero is invalid, not a request to use defaults."""
    with pytest.raises(ValueError, match="total_devices"):
        calculate_host_mesh_shape((1,), total_devices=0, num_processes=1)
    with pytest.raises(ValueError, match="num_processes"):
        calculate_host_mesh_shape((1,), total_devices=1, num_processes=0)


def test_defaults_are_consistent():
    """DEFAULT constants have matching length."""
    assert len(DEFAULT_MESH_AXIS_DIMS) == len(DEFAULT_MESH_AXIS_NAMES)
