# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sharding metadata for parameters.

Layers never depend on a live JAX mesh at construction time. Instead they
attach *logical* axis names (``("in", "out")``) to each
:class:`~spectrax.Parameter`. A downstream consumer provides a
``mesh_map`` that resolves logical axis names to physical mesh axes, and
:meth:`Sharding.to_partition_spec` yields a
:class:`jax.sharding.PartitionSpec` suitable for ``with_sharding_constraint``
or ``jit``'s ``in_shardings`` / ``out_shardings``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from jax.sharding import PartitionSpec as Ps

if TYPE_CHECKING:
    from jax.sharding import PartitionSpec

__all__ = ["AxisName", "AxisNameEntry", "AxisNames", "Sharding", "normalize_sharding"]


AxisName: TypeAlias = str | None
"""A single logical axis name. ``None`` denotes a replicated axis."""

AxisNameEntry: TypeAlias = AxisName | tuple[AxisName, ...]
"""One array dimension's axis annotation.

A tuple entry shards one array dimension over multiple mesh axes, matching
``jax.sharding.PartitionSpec(("fsdp", "sp"), "tp")``.
"""

AxisNames: TypeAlias = tuple[AxisNameEntry, ...]
"""Per-dimension logical axis annotations."""

MeshAxisEntry: TypeAlias = str | None | tuple[str, ...]
"""Pre-resolved per-dimension physical mesh axis specification.

The same shape :data:`AxisNameEntry` carries for logical axes, but the
strings are concrete mesh-axis names (e.g. ``"data"``, ``"model"``)
rather than logical names that still need to be resolved through a
``mesh_map``.
"""


@dataclass(frozen=True)
class Sharding:
    """Metadata describing how a parameter should be sharded.

    Attributes:
        axis_names: Per-dimension logical axis names
            (e.g. ``("in", "out")``). Resolved through a user-supplied
            ``mesh_map`` at consumption time.
        mesh_axes: Pre-resolved per-dimension mesh axes
            (e.g. ``("data", "model")``). When present these override
            ``axis_names`` and are passed straight into
            :class:`~jax.sharding.PartitionSpec`.
    """

    axis_names: AxisNames | None = None
    mesh_axes: tuple[MeshAxisEntry, ...] | None = None

    def to_partition_spec(
        self,
        mesh_map: Mapping[str, str | tuple[str | None, ...] | None] | None = None,
    ) -> PartitionSpec | None:
        """Materialize a :class:`~jax.sharding.PartitionSpec`.

        Resolution order:

        * If :attr:`mesh_axes` is set, build the ``PartitionSpec``
          directly from those pre-resolved mesh axes.
        * Else if :attr:`axis_names` is set and ``mesh_map`` is given,
          resolve each logical name via ``mesh_map`` (missing keys
          collapse to replicated).
        * Else if :attr:`axis_names` is set and ``mesh_map`` is
          ``None``, return a fully-replicated spec of the same rank.
        * Else (no axis info at all) return an empty
          ``PartitionSpec()``.

        Args:
            mesh_map: Optional mapping from logical axis name to
                physical mesh axis (or fused-tuple thereof, or
                ``None`` for replicated).

        Returns:
            The materialized :class:`~jax.sharding.PartitionSpec`.
            ``None`` is only returned in environments where
            ``jax.sharding`` cannot be imported.
        """

        if self.mesh_axes is not None:
            return Ps(*self.mesh_axes)
        if self.axis_names is None:
            return Ps()
        if mesh_map is None:
            return Ps(*[None for _ in self.axis_names])
        return Ps(*_resolve_axis_names(self.axis_names, mesh_map))


def normalize_sharding(s: Sharding | AxisNames | None) -> Sharding | None:
    """Coerce a sharding spec into a :class:`Sharding` instance.

    Accepted inputs:

    * ``None`` — returns ``None``.
    * a :class:`Sharding` — returned unchanged.
    * an :data:`AxisNames` tuple — wrapped as
      ``Sharding(axis_names=s)``.

    Args:
        s: The sharding spec to normalize.

    Returns:
        Either ``None`` or a :class:`Sharding` instance.

    Raises:
        TypeError: On any other input type.
    """
    if s is None:
        return None
    if isinstance(s, Sharding):
        return s
    if isinstance(s, tuple):
        return Sharding(axis_names=s)
    raise TypeError(f"Unsupported sharding spec: {s!r}")


def _expand_mesh_axis(axis: str | tuple[str | None, ...] | None) -> tuple[str, ...]:
    """Flatten one mesh-axis spec into a tuple of mesh-axis names.

    ``None`` becomes the empty tuple, a bare string becomes a single-
    element tuple, and a tuple is returned with ``None`` entries
    filtered out. Used to normalize the result of a logical->mesh
    rule lookup before re-packaging into a partition spec.

    Args:
        axis: A mesh-axis name, a tuple of names (with possible
            ``None`` entries), or ``None``.

    Returns:
        A tuple of concrete mesh-axis strings (no ``None`` entries).
    """
    if axis is None:
        return ()
    if isinstance(axis, tuple):
        return tuple(item for item in axis if item is not None)
    return (axis,)


def _resolve_axis_name(
    axis: AxisNameEntry,
    mesh_map: Mapping[str, str | tuple[str | None, ...] | None],
) -> MeshAxisEntry:
    """Resolve one axis name (or fused-tuple) through ``mesh_map``.

    Strings look up directly; tuples (representing a fused logical
    axis) recurse, concatenate, and trim back to a string when only
    one mesh axis survives. ``None`` and empty results collapse to
    ``None`` (replicated).

    Args:
        axis: A single logical axis name or a tuple of fused names.
        mesh_map: Mapping from logical axis name to physical mesh
            axis specification.

    Returns:
        The resolved mesh-axis entry, or ``None`` for replication.
    """
    if axis is None:
        return None
    if isinstance(axis, tuple):
        resolved = tuple(
            mesh_axis for item in axis if item is not None for mesh_axis in _expand_mesh_axis(mesh_map.get(item))
        )
        if not resolved:
            return None
        if len(resolved) == 1:
            return resolved[0]
        return resolved
    expanded = _expand_mesh_axis(mesh_map.get(axis))
    if not expanded:
        return None
    if len(expanded) == 1:
        return expanded[0]
    return expanded


def _resolve_axis_names(
    axis_names: AxisNames,
    mesh_map: Mapping[str, str | tuple[str | None, ...] | None],
) -> tuple[MeshAxisEntry, ...]:
    """Vector form of :func:`_resolve_axis_name` over a per-dim axis tuple.

    Args:
        axis_names: Per-dimension logical axis annotations.
        mesh_map: Mapping from logical axis name to physical mesh
            axis specification.

    Returns:
        A tuple of resolved mesh-axis entries, one per dimension.
    """
    return tuple(_resolve_axis_name(axis, mesh_map) for axis in axis_names)
