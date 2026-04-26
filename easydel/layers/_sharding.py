# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared sharding helpers for layer parameter layouts.

This module is now a thin compatibility shim over the canonical
``RuntimeShardingResolver`` so layers do not carry a second sharding
resolution path.
"""

from __future__ import annotations

import typing as tp

from jax.sharding import PartitionSpec
from spectrax import common_types

from easydel.infra.sharding import coerce_runtime_sharding_resolver


def _is_valid_mesh(mesh: tp.Any) -> bool:
    if mesh is None:
        return False
    if getattr(mesh, "empty", False):
        return False
    return getattr(mesh, "shape", None) is not None


def _mesh_axis_size(mesh: tp.Any, axis_name: str) -> int:
    shape = getattr(mesh, "shape", None)
    if shape is None:
        return 1
    try:
        return int(shape[axis_name])
    except Exception:
        pass
    try:
        return int(shape.get(axis_name, 1))
    except Exception:
        return 1


def _mesh_partition_product(mesh: tp.Any, axis_spec: tp.Any) -> int:
    if axis_spec is None:
        return 1
    if isinstance(axis_spec, (list, tuple)):
        product = 1
        for axis_name in axis_spec:
            if axis_name is None:
                continue
            product *= _mesh_axis_size(mesh, str(axis_name))
        return int(product)
    return _mesh_axis_size(mesh, str(axis_spec))


def _coerce_partition_spec(spec: tp.Any) -> PartitionSpec | None:
    if isinstance(spec, PartitionSpec):
        return spec
    if isinstance(spec, (tuple, list)):
        try:
            return PartitionSpec(*tuple(spec))
        except Exception:
            return None
    return None


def _sanitize_spec_for_shape(*, spec: tp.Any, shape: tuple[int, ...], mesh: tp.Any) -> tp.Any:
    pspec = _coerce_partition_spec(spec)
    if pspec is None:
        return spec

    axes = list(tuple(pspec))
    changed = False
    for dim_index, axis_spec in enumerate(axes):
        if axis_spec is None or dim_index >= len(shape):
            continue
        shard_factor = _mesh_partition_product(mesh, axis_spec)
        if shard_factor > 1 and int(shape[dim_index]) % shard_factor != 0:
            axes[dim_index] = None
            changed = True
    if not changed:
        return pspec
    return PartitionSpec(*axes)


def pick_mesh(*, partition_manager: tp.Any | None = None, mesh: tp.Any | None = None) -> tp.Any | None:
    if _is_valid_mesh(mesh):
        return mesh

    if partition_manager is not None:
        for attr_name in ("mesh", "_mesh", "device_mesh"):
            candidate = getattr(partition_manager, attr_name, None)
            if _is_valid_mesh(candidate):
                return candidate

    try:
        from spectrax import get_incontext_mesh

        candidate = get_incontext_mesh(raise_error=False)
        if _is_valid_mesh(candidate):
            return candidate
    except Exception:
        pass

    try:
        from jax._src.interpreters import pxla

        candidate = pxla.thread_resources.env.physical_mesh
        if _is_valid_mesh(candidate):
            return candidate
    except Exception:
        pass

    return None


def resolve_safe_sharding(
    *,
    axes: tp.Any,
    shape: tuple[int, ...],
    runtime_sharding_resolver: tp.Any | None = None,
    axis_policy: tp.Any | None = None,
    partition_manager: tp.Any | None = None,
    mesh: tp.Any | None = None,
    mode: str = common_types.MODE_TRAIN,
) -> tp.Any:
    """Resolve sharding axes through the canonical runtime resolver."""
    mesh_obj = pick_mesh(partition_manager=runtime_sharding_resolver or partition_manager, mesh=mesh)
    resolver = coerce_runtime_sharding_resolver(
        runtime_sharding_resolver if runtime_sharding_resolver is not None else axis_policy,
        mesh=mesh_obj,
    )
    try:
        return resolver.resolve(axes=axes, mode=mode, shape=shape)
    except Exception:
        if partition_manager is None or not hasattr(partition_manager, "resolve"):
            return axes
        resolved = partition_manager.resolve(axes=axes, mode=mode, shape=shape)
        if mesh_obj is None:
            return resolved
        return _sanitize_spec_for_shape(spec=resolved, shape=shape, mesh=mesh_obj)
