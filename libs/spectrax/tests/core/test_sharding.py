# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.sharding`."""

from __future__ import annotations

import pytest
from jax.sharding import PartitionSpec

from spectrax.core.sharding import Sharding, normalize_sharding


def test_sharding_defaults_are_none():
    """A bare :class:`Sharding` has no axis info."""
    s = Sharding()
    assert s.axis_names is None
    assert s.mesh_axes is None


def test_sharding_is_frozen_dataclass():
    """:class:`Sharding` is immutable once constructed."""
    s = Sharding(axis_names=("in", "out"))
    with pytest.raises(AttributeError):
        s.axis_names = ("a",)


def test_to_partition_spec_replicated_when_empty():
    """With no axis info, ``to_partition_spec`` returns ``PartitionSpec()``."""
    s = Sharding()
    assert s.to_partition_spec() == PartitionSpec()


def test_to_partition_spec_mesh_axes_direct():
    """``mesh_axes`` is passed straight into :class:`PartitionSpec`."""
    s = Sharding(mesh_axes=("data", "model"))
    assert s.to_partition_spec() == PartitionSpec("data", "model")


def test_to_partition_spec_axis_names_without_map_is_replicated():
    """Without a ``mesh_map`` the spec is fully replicated."""
    s = Sharding(axis_names=("in", "out"))
    assert s.to_partition_spec() == PartitionSpec(None, None)


def test_to_partition_spec_with_mesh_map_resolves_axes():
    """``mesh_map`` resolves logical names to mesh axes."""
    s = Sharding(axis_names=("in", "out"))
    spec = s.to_partition_spec({"in": "mp", "out": None})
    assert spec == PartitionSpec("mp", None)


def test_to_partition_spec_with_compound_axis_names():
    """Tuple entries shard one array dimension over multiple mesh axes."""
    s = Sharding(axis_names=(("data", "sequence"), "model"))
    spec = s.to_partition_spec({"data": "fsdp", "sequence": "sp", "model": "tp"})
    assert spec == PartitionSpec(("fsdp", "sp"), "tp")


def test_to_partition_spec_compound_axis_names_drop_replicated_members():
    """Missing compound members replicate while resolved members still shard."""
    s = Sharding(axis_names=(("data", "sequence"), "model"))
    spec = s.to_partition_spec({"data": "fsdp", "model": "tp"})
    assert spec == PartitionSpec("fsdp", "tp")


def test_to_partition_spec_missing_map_key_is_replicated():
    """Keys absent from ``mesh_map`` become ``None`` in the spec."""
    s = Sharding(axis_names=("foo",))
    assert s.to_partition_spec({"other": "dp"}) == PartitionSpec(None)


def test_to_partition_spec_preserves_none_axis_names():
    """``None`` entries in ``axis_names`` stay ``None`` regardless of the map."""
    s = Sharding(axis_names=("in", None, "out"))
    spec = s.to_partition_spec({"in": "dp", "out": "mp"})
    assert spec == PartitionSpec("dp", None, "mp")


def test_normalize_sharding_none():
    """``None`` passes through."""
    assert normalize_sharding(None) is None


def test_normalize_sharding_existing_instance():
    """An existing :class:`Sharding` is returned as-is."""
    s = Sharding(axis_names=("x",))
    assert normalize_sharding(s) is s


def test_normalize_sharding_tuple_wraps():
    """A tuple is wrapped as ``Sharding(axis_names=tuple)``."""
    out = normalize_sharding(("in", "out"))
    assert isinstance(out, Sharding)
    assert out.axis_names == ("in", "out")


def test_normalize_sharding_rejects_other_types():
    """Unrecognized inputs raise :class:`TypeError`."""
    with pytest.raises(TypeError):
        normalize_sharding("invalid")
    with pytest.raises(TypeError):
        normalize_sharding(123)
