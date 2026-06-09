# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup-time device placement helpers for JAX shardings.

These helpers are intentionally for initialization/loading paths. They avoid
multi-controller JAX's illegal direct reshard from one physical device set to a
different physical device set by first rewrapping exact target-subset shards and
then, only when the source value is host-fetchable, staging through host memory.
They never invent array values; callers that need optimizer-zero synthesis
should do that in the optimizer initialization layer where the zero invariant is
known.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jax import device_get
from jax.sharding import NamedSharding, PartitionSpec

from spectrax._internal.logging import get_logger

logger = get_logger(__name__)


def mesh_axis_product(mesh: object, axis_entry: object) -> int:
    """Return the partition factor for a PartitionSpec axis entry."""
    if axis_entry is None:
        return 1
    if isinstance(axis_entry, tuple):
        product = 1
        for axis in axis_entry:
            product *= mesh_axis_product(mesh, axis)
        return product
    shape = getattr(mesh, "shape", {})
    try:
        return int(shape.get(axis_entry, shape.get(str(axis_entry), 1)))
    except Exception:
        return 1


def spec_shape_mismatches(spec: PartitionSpec, mesh: object, shape: tuple[int, ...]) -> tuple[str, ...]:
    """Return human-readable spec/shape divisibility mismatches."""
    messages: list[str] = []
    for dim, axis_entry in enumerate(tuple(spec)):
        factor = mesh_axis_product(mesh, axis_entry)
        if factor <= 1:
            continue
        if dim >= len(shape):
            messages.append(f"dim{dim}:missing_shape_for_axis_product_{factor}")
            continue
        size = int(shape[dim])
        if size % factor:
            messages.append(f"dim{dim}:size_{size}_not_divisible_by_axis_product_{factor}")
    return tuple(messages)


def named_sharding_for_shape(
    source_sharding: NamedSharding,
    shape: tuple[int, ...],
    spec: PartitionSpec,
    *,
    context: str,
) -> NamedSharding:
    """Create a NamedSharding on the source mesh after validating shape divisibility."""
    mismatches = spec_shape_mismatches(spec, source_sharding.mesh, shape)
    if mismatches:
        raise ValueError(
            "SpectraX cannot preserve the incoming sharding for a layout transform. "
            f"context={context}, shape={shape}, source_spec={source_sharding.spec}, "
            f"target_spec={spec}, mesh_axes={getattr(source_sharding.mesh, 'axis_names', None)}, "
            f"invalid_dims={mismatches}. Change batch size, chunk size, or the sharding policy."
        )
    target_sharding = NamedSharding(source_sharding.mesh, spec)
    memory_kind = getattr(source_sharding, "memory_kind", None)
    if memory_kind is not None and hasattr(target_sharding, "with_memory_kind"):
        try:
            target_sharding = target_sharding.with_memory_kind(memory_kind)
        except Exception:
            pass
    return target_sharding


def named_sharding_for_value(value: object, spec: PartitionSpec, *, context: str) -> NamedSharding | None:
    """Return a target NamedSharding for ``value`` on its current mesh."""
    source_sharding = getattr(value, "sharding", None)
    if not isinstance(source_sharding, NamedSharding):
        return None
    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    return named_sharding_for_shape(source_sharding, shape, spec, context=context)


def with_named_sharding(value: tp.Any, spec: PartitionSpec, *, context: str) -> tp.Any:
    """Apply a sharding constraint on ``value`` while preserving its current mesh."""
    target_sharding = named_sharding_for_value(value, spec, context=context)
    if target_sharding is None:
        return value
    return jax.lax.with_sharding_constraint(value, target_sharding)


def reshape_with_named_shardings(
    value: tp.Any,
    shape: tuple[int, ...],
    *,
    in_sharding: NamedSharding,
    out_sharding: NamedSharding,
) -> tp.Any:
    """Reshape with explicit in/out shardings so layout metadata is not dropped."""

    def _reshape(x: tp.Any) -> tp.Any:
        return jnp.reshape(x, shape)

    return jax.jit(_reshape, in_shardings=in_sharding, out_shardings=out_sharding)(value)


def transpose_with_named_shardings(
    value: tp.Any,
    permutation: tuple[int, ...],
    *,
    in_sharding: NamedSharding,
    out_sharding: NamedSharding,
) -> tp.Any:
    """Transpose with explicit in/out shardings so layout metadata is not dropped."""

    def _transpose(x: tp.Any) -> tp.Any:
        return jnp.transpose(x, permutation)

    return jax.jit(_transpose, in_shardings=in_sharding, out_shardings=out_sharding)(value)


def _device_set_from_sharding(sharding: object) -> set[object] | None:
    devices = getattr(sharding, "device_set", None)
    if devices is not None:
        try:
            return set(devices() if callable(devices) else devices)
        except Exception:
            pass
    mesh = getattr(sharding, "mesh", None)
    if mesh is not None:
        try:
            return set(mesh.devices.flat)
        except Exception:
            return None
    return None


def _device_set_from_value(value: object) -> set[object] | None:
    sharding = getattr(value, "sharding", None)
    devices = _device_set_from_sharding(sharding)
    if devices is not None:
        return devices
    value_devices = getattr(value, "devices", None)
    if value_devices is None:
        return None
    try:
        return set(value_devices() if callable(value_devices) else value_devices)
    except Exception:
        return None


def _device_ids(devices: set[object] | None) -> tuple[int, ...] | None:
    if devices is None:
        return None
    return tuple(sorted(int(getattr(device, "id", idx)) for idx, device in enumerate(devices)))


def _device_id_preview(device_ids: tuple[int, ...] | None) -> str:
    if device_ids is None:
        return "unknown"
    if len(device_ids) <= 12:
        return repr(device_ids)
    head = ", ".join(str(device_id) for device_id in device_ids[:6])
    tail = ", ".join(str(device_id) for device_id in device_ids[-3:])
    return f"({head}, ..., {tail})"


def _mesh_axis_names(sharding: object) -> tuple[object, ...] | None:
    mesh = getattr(sharding, "mesh", None)
    if mesh is None:
        return None
    axis_names = getattr(mesh, "axis_names", None)
    return tuple(axis_names) if axis_names is not None else None


def _index_key(index: object) -> object:
    if index is None:
        return None
    if not isinstance(index, tuple):
        index = (index,)
    parts: list[object] = []
    for part in index:
        if isinstance(part, slice):
            parts.append(("slice", part.start, part.stop, part.step))
        elif isinstance(part, list | tuple):
            parts.append(tuple(part))
        else:
            parts.append(part)
    return tuple(parts)


def _index_shape(index: object, shape: tuple[int, ...]) -> tuple[int, ...]:
    if index is None:
        return shape
    if not isinstance(index, tuple):
        index = (index,)
    out: list[int] = []
    for dim, selector in zip(shape, index, strict=False):
        if isinstance(selector, slice):
            start, stop, step = selector.indices(dim)
            out.append(max(0, (stop - start + (step - 1)) // step))
        elif isinstance(selector, int):
            continue
        else:
            try:
                out.append(len(selector))
            except TypeError:
                out.append(dim)
    if len(index) < len(shape):
        out.extend(shape[len(index) :])
    return tuple(out)


def _same_dtype(a: object, b: object) -> bool:
    try:
        return jnp.dtype(a) == jnp.dtype(b)
    except Exception:
        return a == b


def _try_rewrap_from_target_subset(
    leaf: tp.Any,
    sharding: jax.sharding.Sharding,
    *,
    path: str,
    label: str,
    diagnostics: dict[str, int],
) -> tp.Any | None:
    """Rewrap setup leaves when target shards already exist on target devices."""
    if not isinstance(leaf, jax.Array):
        return None
    source_devices = _device_set_from_value(leaf)
    target_devices = _device_set_from_sharding(sharding)
    if source_devices is None or target_devices is None:
        return None
    if source_devices == target_devices or not target_devices <= source_devices:
        return None

    shape = tuple(leaf.shape)
    try:
        target_map = sharding.addressable_devices_indices_map(shape)
    except Exception:
        return None
    if not target_map:
        try:
            return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=leaf.dtype)
        except Exception:
            return None

    source_by_device_index: dict[tuple[object, object], object] = {}
    source_indices_by_device: dict[object, list[object]] = {}
    try:
        source_shards = tuple(leaf.addressable_shards)
    except Exception:
        return None
    for shard in source_shards:
        device = getattr(shard, "device", None)
        shard_index = getattr(shard, "index", None)
        shard_data = getattr(shard, "data", None)
        if device is None or shard_data is None:
            continue
        index_key = _index_key(shard_index)
        source_by_device_index[(device, index_key)] = shard_data
        source_indices_by_device.setdefault(device, []).append(index_key)

    arrays: list[object] = []
    for device, target_index in target_map.items():
        target_index_key = _index_key(target_index)
        shard_data = source_by_device_index.get((device, target_index_key))
        if shard_data is None:
            if jax.process_index() == 0 and diagnostics.get("rewrap_index_miss_logged", 0) < 5:
                logger.warning(
                    "%s refused setup subset rewrap for %s because no exact source shard matched "
                    "target_device=%s target_index=%s source_indices_on_device=%s.",
                    label,
                    path or "<leaf>",
                    getattr(device, "id", device),
                    target_index_key,
                    source_indices_by_device.get(device, ()),
                )
                diagnostics["rewrap_index_miss_logged"] = diagnostics.get("rewrap_index_miss_logged", 0) + 1
            return None
        expected_shape = _index_shape(target_index, shape)
        actual_shape = tuple(getattr(shard_data, "shape", ()))
        actual_dtype = getattr(shard_data, "dtype", None)
        if actual_shape != expected_shape or not _same_dtype(actual_dtype, leaf.dtype):
            raise ValueError(
                f"{label} refused setup subset rewrap for {path or '<leaf>'}: "
                f"target_index={target_index_key}, expected_shard_shape={expected_shape}, "
                f"actual_shard_shape={actual_shape}, actual_shard_dtype={actual_dtype}, "
                f"value_dtype={getattr(leaf, 'dtype', None)}."
            )
        arrays.append(shard_data)

    diagnostics["subset_rewrapped"] = diagnostics.get("subset_rewrapped", 0) + 1
    return jax.make_array_from_single_device_arrays(shape, sharding, arrays, dtype=leaf.dtype)


def _try_host_stage_addressable_shards_by_index(
    leaf: tp.Any,
    sharding: jax.sharding.Sharding,
    *,
    path: str,
    label: str,
    diagnostics: dict[str, int],
) -> tp.Any | None:
    """Setup-only fallback that moves exact matching local shards via host."""
    if not isinstance(leaf, jax.Array):
        return None
    source_devices = _device_set_from_value(leaf)
    target_devices = _device_set_from_sharding(sharding)
    if source_devices is None or target_devices is None:
        return None
    if source_devices == target_devices or not target_devices <= source_devices:
        return None

    shape = tuple(leaf.shape)
    try:
        target_map = sharding.addressable_devices_indices_map(shape)
    except Exception:
        return None
    if not target_map:
        try:
            return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=leaf.dtype)
        except Exception:
            return None

    source_by_index: dict[object, object] = {}
    try:
        source_shards = tuple(leaf.addressable_shards)
    except Exception:
        return None
    for shard in source_shards:
        shard_index = getattr(shard, "index", None)
        shard_data = getattr(shard, "data", None)
        if shard_data is None:
            continue
        source_by_index.setdefault(_index_key(shard_index), shard_data)

    arrays: list[object] = []
    for device, target_index in target_map.items():
        target_index_key = _index_key(target_index)
        shard_data = source_by_index.get(target_index_key)
        if shard_data is None:
            if jax.process_index() == 0 and diagnostics.get("host_shard_index_miss_logged", 0) < 5:
                logger.warning(
                    "%s refused setup per-shard host staging for %s because no addressable source shard "
                    "matched target_index=%s.",
                    label,
                    path or "<leaf>",
                    target_index_key,
                )
                diagnostics["host_shard_index_miss_logged"] = diagnostics.get("host_shard_index_miss_logged", 0) + 1
            return None
        expected_shape = _index_shape(target_index, shape)
        actual_shape = tuple(getattr(shard_data, "shape", ()))
        actual_dtype = getattr(shard_data, "dtype", None)
        if actual_shape != expected_shape or not _same_dtype(actual_dtype, leaf.dtype):
            raise ValueError(
                f"{label} refused setup per-shard host staging for {path or '<leaf>'}: "
                f"target_index={target_index_key}, expected_shard_shape={expected_shape}, "
                f"actual_shard_shape={actual_shape}, actual_shard_dtype={actual_dtype}, "
                f"value_dtype={getattr(leaf, 'dtype', None)}."
            )
        host_shard = device_get(shard_data)
        arrays.append(jax.device_put(host_shard, jax.sharding.SingleDeviceSharding(device)))

    diagnostics["host_staged_shards"] = diagnostics.get("host_staged_shards", 0) + len(arrays)
    return jax.make_array_from_single_device_arrays(shape, sharding, arrays, dtype=leaf.dtype)


def _same_setup_sharding(value: object, sharding: object) -> bool:
    current = getattr(value, "sharding", None)
    if current is None or sharding is None:
        return False
    if current is sharding or current == sharding:
        return True
    if type(current) is not type(sharding):
        return False
    if getattr(current, "spec", None) != getattr(sharding, "spec", None):
        return False
    if getattr(current, "memory_kind", None) != getattr(sharding, "memory_kind", None):
        return False
    return _device_set_from_sharding(current) == _device_set_from_sharding(sharding)


def _path_to_string(path: tuple[object, ...]) -> str:
    parts: list[str] = []
    for entry in path:
        key = getattr(entry, "key", None)
        if key is not None:
            parts.append(str(key))
            continue
        idx = getattr(entry, "idx", None)
        if idx is not None:
            parts.append(str(idx))
            continue
        name = getattr(entry, "name", None)
        if name is not None:
            parts.append(str(name))
            continue
        parts.append(str(entry))
    return "/".join(parts)


def place_setup_leaf_with_sharding(
    leaf: tp.Any,
    sharding: jax.sharding.Sharding,
    *,
    path: str = "",
    label: str = "SpectraX setup placement",
    diagnostics: dict[str, int] | None = None,
    donate: bool = False,
) -> tp.Any:
    """Place one setup leaf without illegal cross-device-set resharding.

    The helper is intentionally conservative: when device sets differ, it first
    tries an exact target-subset rewrap. If that is not possible, it stages the
    existing value through host memory and places that value on the target
    sharding. It never synthesizes replacement data.
    """
    if sharding is None or not hasattr(leaf, "shape"):
        return leaf
    if _same_setup_sharding(leaf, sharding):
        return leaf

    source_devices = _device_set_from_value(leaf)
    target_devices = _device_set_from_sharding(sharding)
    if source_devices is None or target_devices is None or source_devices == target_devices:
        return jax.device_put(leaf, sharding, donate=donate)

    if diagnostics is None:
        diagnostics = {}
    diagnostics["cross_device_set"] = diagnostics.get("cross_device_set", 0) + 1
    process_index = jax.process_index()
    if process_index == 0 and diagnostics.get("logged", 0) < 5:
        source_ids = _device_ids(source_devices)
        target_ids = _device_ids(target_devices)
        global_device_count = jax.device_count()
        source_is_full_global = len(source_devices) == global_device_count
        target_is_subset = len(target_devices) < len(source_devices) and target_devices <= source_devices
        source_sharding = getattr(leaf, "sharding", None)
        logger.warning(
            "%s detected cross-device-set setup placement at %s on process %d; "
            "shape=%s dtype=%s source_sharding=%s source_axes=%s source_device_count=%d "
            "source_device_ids=%s target_axes=%s target_spec=%s target_device_count=%d "
            "target_device_ids=%s source_is_full_global=%s target_is_subset=%s. "
            "Trying exact subset rewrap before any setup host staging.",
            label,
            path or "<leaf>",
            process_index,
            tuple(getattr(leaf, "shape", ())),
            getattr(leaf, "dtype", None),
            type(source_sharding).__name__ if source_sharding is not None else None,
            _mesh_axis_names(source_sharding),
            len(source_devices),
            _device_id_preview(source_ids),
            _mesh_axis_names(sharding),
            getattr(sharding, "spec", None),
            len(target_devices),
            _device_id_preview(target_ids),
            source_is_full_global,
            target_is_subset,
        )
        diagnostics["logged"] = diagnostics.get("logged", 0) + 1

    rewrapped = _try_rewrap_from_target_subset(
        leaf,
        sharding,
        path=path,
        label=label,
        diagnostics=diagnostics,
    )
    if rewrapped is not None:
        return rewrapped

    shard_staged = _try_host_stage_addressable_shards_by_index(
        leaf,
        sharding,
        path=path,
        label=label,
        diagnostics=diagnostics,
    )
    if shard_staged is not None:
        return shard_staged

    try:
        host_leaf = device_get(leaf)
    except RuntimeError as exc:
        raise ValueError(
            f"{label} cannot host-stage non-addressable setup leaf at {path or '<leaf>'}; "
            "exact subset rewrap was not possible, and synthesizing replacement values is forbidden. "
            f"shape={tuple(getattr(leaf, 'shape', ()))}, dtype={getattr(leaf, 'dtype', None)}, "
            f"source_axes={_mesh_axis_names(getattr(leaf, 'sharding', None))}, "
            f"source_spec={getattr(getattr(leaf, 'sharding', None), 'spec', None)}, "
            f"target_axes={_mesh_axis_names(sharding)}, target_spec={getattr(sharding, 'spec', None)}."
        ) from exc
    diagnostics["host_staged"] = diagnostics.get("host_staged", 0) + 1
    return jax.device_put(host_leaf, sharding, donate=donate)


def place_setup_tree_with_shardings(
    tree: tp.Any,
    shardings: tp.Any,
    *,
    label: str = "SpectraX setup placement",
    donate: bool = False,
) -> tp.Any:
    """Place a setup-time pytree with per-leaf sharding diagnostics."""
    diagnostics: dict[str, int] = {"cross_device_set": 0, "host_staged": 0, "logged": 0, "subset_rewrapped": 0}

    def _place(path: tuple[object, ...], leaf: tp.Any, sharding: tp.Any) -> tp.Any:
        if not isinstance(sharding, jax.sharding.Sharding) or not hasattr(leaf, "shape"):
            return leaf
        return place_setup_leaf_with_sharding(
            leaf,
            sharding,
            path=_path_to_string(path),
            label=label,
            diagnostics=diagnostics,
            donate=donate,
        )

    placed = jax.tree_util.tree_map_with_path(
        _place,
        tree,
        shardings,
        is_leaf=lambda x: isinstance(x, jax.sharding.Sharding) or x is None,
    )
    if diagnostics["cross_device_set"] and jax.process_index() == 0:
        logger.warning(
            "%s handled %d cross-device-set setup leaves "
            "(%d exact subset rewraps, %d host-staged real values). "
            "This helper is for setup/loading, not compiled training steps.",
            label,
            diagnostics["cross_device_set"],
            diagnostics["subset_rewrapped"],
            diagnostics["host_staged"],
        )
    return placed
