# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small sharding-spec helpers for the SpectraX MPMD runtime."""

from __future__ import annotations

import jax


def _stage_axis_size(mesh: object, axis: object) -> int:
    """Return the size of ``axis`` on ``mesh``; unknown axes behave replicated.

    Args:
        mesh: JAX mesh or SpectraX mesh descriptor used for placement.
        axis: Logical or positional axis used by the operation.

    Returns:
        Return the size of ``axis`` on ``mesh``; unknown axes behave replicated.
    """
    if axis is None:
        return 1
    try:
        return int(mesh.shape[axis])
    except Exception:
        return 1


def _stage_axis_product(mesh: object, axis: object) -> int:
    """Return the product of every mesh axis referenced by one spec entry.

    Args:
        mesh: JAX mesh or SpectraX mesh descriptor used for placement.
        axis: Logical or positional axis used by the operation.

    Returns:
        Return the product of every mesh axis referenced by one spec entry.
    """
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        product = 1
        for part in axis:
            product *= _stage_axis_size(mesh, part)
        return product
    return _stage_axis_size(mesh, axis)


def _trim_trailing_replicated_stage_axes(spec: object, mesh: object) -> object:
    """Drop trailing spec entries that are equivalent to replication.

    JAX commonly canonicalizes outputs by omitting trailing replicated
    dimensions.  For example, on a stage mesh where ``fsdp`` and ``sp`` both
    have size 1, ``P('tp', ('fsdp', 'sp'))`` and ``P('tp')`` are physically
    identical.  Scheduled MPMD stage jits key on the full sharding signature,
    so normalize to the canonical shorter form before the first launch.

    Args:
        spec: Partition specification or related sharding specification.
        mesh: JAX mesh or SpectraX mesh descriptor used for placement.

    Returns:
        Result described by this helper.
    """
    try:
        parts = list(tuple(spec))
    except Exception:
        return spec
    while parts and _stage_axis_product(mesh, parts[-1]) <= 1:
        parts.pop()
    return jax.sharding.PartitionSpec(*parts)


def _spec_axis_factors(spec: object, mesh: object) -> tuple[int, ...]:
    """Return the per-dimension mesh partition factor for ``spec``."""
    try:
        return tuple(_stage_axis_product(mesh, axis) for axis in tuple(spec))
    except Exception:
        return ()


def _spec_axis_shape_mismatches(spec: object, mesh: object, shape: tuple[int, ...]) -> tuple[str, ...]:
    """Return per-dimension shape/factor mismatches for an explicit stage spec."""
    messages: list[str] = []
    try:
        parts = tuple(spec)
    except Exception:
        return ()
    for dim, axis_entry in enumerate(parts):
        axes = _axis_entry_names(axis_entry)
        factor = _stage_axis_product(mesh, axis_entry)
        if factor <= 1:
            continue
        axis_expr = "*".join(axes) if axes else "<replicated>"
        if dim >= len(shape):
            messages.append(f"dim{dim}:missing_shape_for_axes_{axis_expr}_product_{factor}")
            continue
        size = int(shape[dim])
        if size % factor:
            messages.append(f"dim{dim}:size_{size}_not_divisible_by_axes_{axis_expr}_product_{factor}")
    return tuple(messages)


def _axis_entry_names(axis: object) -> tuple[str, ...]:
    """Return mesh-axis names referenced by one ``PartitionSpec`` entry."""
    if axis is None:
        return ()
    if isinstance(axis, tuple):
        return tuple(str(part) for part in axis if part is not None)
    return (str(axis),)
