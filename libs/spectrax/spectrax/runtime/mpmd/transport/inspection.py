# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transport sharding inspection and ABI helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ....core.variable import Variable
from ..utils.sharding import _spec_axis_shape_mismatches
from ..utils.tree import _is_leaf


def _array_payload(value: object) -> object:
    """Return the concrete array carried by an array-like wrapper, if any."""
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, Variable):
        try:
            inner = value.value
            if isinstance(inner, jax.Array):
                return inner
        except Exception:
            return value
    jax_array = getattr(value, "__jax_array__", None)
    if callable(jax_array):
        try:
            inner = jax_array()
            if isinstance(inner, jax.Array):
                return inner
        except Exception:
            pass
    return value


def _value_sharding(value: object) -> object:
    """Return the physical sharding for ``value``'s array payload."""
    payload = _array_payload(value)
    sharding = getattr(payload, "sharding", None)
    return sharding if sharding is not None else getattr(value, "sharding", None)


def _device_set_from_sharding(sharding: object) -> set[object] | None:
    """Return the devices described by a sharding object, if inspectable."""
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


def _array_device_set(value: object) -> set[object] | None:
    """Return the device set holding ``value``'s shards, or ``None`` when unknown.

    Tries the ``.devices`` attribute on the array (callable on newer
    JAX, plain attribute on older). Returns ``None`` whenever the
    object is not a JAX array (Python scalars, ``None``, etc.) or
    when the attribute access raises.

    Args:
        value: object value (array or otherwise).

    Returns:
        A set of :class:`jax.Device` objects or ``None``.
    """
    payload = _array_payload(value)
    devices = getattr(payload, "devices", None)
    if devices is None:
        return _device_set_from_sharding(getattr(payload, "sharding", None))
    try:
        return set(devices() if callable(devices) else devices)
    except Exception:
        return _device_set_from_sharding(getattr(payload, "sharding", None))


def _device_id_tuple(devices: set[object] | None) -> tuple[int, ...] | None:
    """Return a sorted tuple of integer device IDs (or ``None``).

    Used as a stable, hashable key for sharding-decision caches —
    ``set`` itself is not hashable, and device objects don't always
    sort by their natural ``__lt__``.

    Args:
        devices: Device collection used to construct or inspect a mesh.

    Returns:
        Return a sorted tuple of integer device IDs (or ``None``).
    """
    if devices is None:
        return None
    return tuple(sorted(int(getattr(device, "id", idx)) for idx, device in enumerate(devices)))


def _sharding_device_set(sharding: object) -> set[object] | None:
    """Return the device set backing a sharding spec, or ``None``.

    Tries the spec's ``device_set`` attribute first (callable or
    direct), then falls back to flattening the spec's ``mesh.devices``.
    Used to compare two shardings for "same physical placement"
    independent of axis layout.

    Args:
        sharding: JAX sharding object describing how an array is placed.

    Returns:
        Return the device set backing a sharding spec, or ``None``.
    """
    return _device_set_from_sharding(sharding)


def _tree_nbytes(x: object) -> int:
    """Sum the byte sizes of every array leaf in ``x`` without touching devices.

    Computes ``size * dtype.itemsize`` per leaf, ignoring leaves that
    are not arrays. The result is reported to the schedule stats
    collector as a transfer-size estimate. No :func:`block_until_ready`
    is issued so the cost is purely metadata access.

    Args:
        x: A pytree whose array leaves should be measured.

    Returns:
        Total bytes across all array leaves.
    """
    total = 0
    for leaf in jax.tree.leaves(x, is_leaf=_is_leaf):
        size = getattr(leaf, "size", None)
        dtype = getattr(leaf, "dtype", None)
        if size is None or dtype is None:
            continue
        try:
            total += int(size) * int(jnp.dtype(dtype).itemsize)
        except Exception:
            continue
    return total


def _sharding_mesh_signature(sharding: object) -> tuple[object, ...] | None:
    """Return the mesh axis layout that is part of a sharding ABI."""
    mesh = getattr(sharding, "mesh", None)
    if mesh is None:
        return None
    axis_names = tuple(getattr(mesh, "axis_names", ()) or ())
    try:
        axis_sizes = tuple(int(mesh.shape[axis]) for axis in axis_names)
    except Exception:
        axis_sizes = ()
    try:
        device_grid_shape = tuple(np.asarray(mesh.devices).shape)
    except Exception:
        device_grid_shape = ()
    return (axis_names, axis_sizes, device_grid_shape)


def _same_sharding(a: object, b: object) -> bool:
    """Return ``True`` iff ``a`` and ``b`` represent the same physical sharding.

    Type, partition spec, and physical device set must all match. Two shardings
    can compare equal at the object level while carrying different stage-local
    meshes; those are not the same runtime ABI for MPMD transport.
    ``None`` operands always compare unequal so that "no sharding" is never
    confused with "any sharding".

    Args:
        a: Positional arguments forwarded to the wrapped callable.
        b: B value consumed by this operation.

    Returns:
        Return ``True`` iff ``a`` and ``b`` represent the same physical sharding.
    """
    if a is None or b is None:
        return False
    if not isinstance(a, jax.sharding.Sharding) or not isinstance(b, jax.sharding.Sharding):
        return False
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if getattr(a, "spec", None) != getattr(b, "spec", None):
        return False
    if _sharding_mesh_signature(a) != _sharding_mesh_signature(b):
        return False
    a_devices = _sharding_device_set(a)
    b_devices = _sharding_device_set(b)
    if a_devices is not None or b_devices is not None:
        return a_devices == b_devices
    return a == b


def _device_id_preview(devices: set[object] | None) -> str:
    """Format a large device set compactly for process-aware diagnostics."""
    device_ids = _device_id_tuple(devices)
    if device_ids is None:
        return "unknown"
    if len(device_ids) <= 12:
        return repr(device_ids)
    head = ", ".join(str(device_id) for device_id in device_ids[:6])
    tail = ", ".join(str(device_id) for device_id in device_ids[-3:])
    return f"({head}, ..., {tail})"


def _mesh_axis_names(sharding: object) -> tuple[object, ...] | None:
    """Return mesh axis names for logging, when the sharding exposes them."""
    mesh = getattr(sharding, "mesh", None)
    if mesh is None:
        return None
    axis_names = getattr(mesh, "axis_names", None)
    return tuple(axis_names) if axis_names is not None else None


def _index_key(index: object) -> object:
    """Convert a JAX shard index into a hashable equality key."""
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
    """Return the local shard shape selected by a global shard index."""
    if index is None:
        return shape
    if not isinstance(index, tuple):
        index = (index,)
    out: list[int] = []
    for axis, selector in enumerate(index):
        dim = shape[axis] if axis < len(shape) else 1
        if isinstance(selector, slice):
            start, stop, step = selector.indices(dim)
            out.append(max(0, stop - start) if step == 1 else len(range(start, stop, step)))
        elif selector is None:
            out.append(1)
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


def _drop_leading_index_axis(index: object) -> object:
    """Remove the synthetic leading pair-lane axis from a shard index."""
    if index is None:
        return None
    if not isinstance(index, tuple):
        index = (index,)
    return tuple(index[1:])


def _leading_index_axis_start(index: object) -> int | None:
    """Return the selected start offset for a shard's leading axis."""
    if index is None:
        return None
    if not isinstance(index, tuple):
        index = (index,)
    if not index:
        return None
    selector = index[0]
    if isinstance(selector, slice):
        start, _stop, step = selector.indices(2)
        if step != 1:
            return None
        return int(start)
    if isinstance(selector, int):
        return int(selector)
    return None


def _shape_nbytes(shape: tuple[int, ...], dtype: object) -> int | None:
    """Return the byte size of one local shard shape."""
    try:
        itemsize = int(jnp.dtype(dtype).itemsize)
    except Exception:
        return None
    total = 1
    for dim in shape:
        total *= int(dim)
    return total * itemsize


def _same_dtype(a: object, b: object) -> bool:
    """Return whether two dtype-like objects describe the same JAX dtype."""
    try:
        return jnp.dtype(a) == jnp.dtype(b)
    except Exception:
        return a == b


def _addressable_shard_nbytes(value: object) -> tuple[int, ...]:
    """Summarize the unique local shard byte sizes currently addressable."""
    payload = _array_payload(value)
    if not hasattr(payload, "addressable_shards"):
        return ()
    out: set[int] = set()
    try:
        shards = tuple(payload.addressable_shards)
    except Exception:
        return ()
    for shard in shards:
        data = getattr(shard, "data", None)
        nbytes = getattr(data, "nbytes", None)
        if nbytes is not None:
            try:
                out.add(int(nbytes))
                continue
            except Exception:
                pass
        shape = tuple(getattr(data, "shape", ()))
        if shape:
            size = _shape_nbytes(shape, getattr(payload, "dtype", None))
            if size is not None:
                out.add(size)
    return tuple(sorted(out))


def _target_shard_nbytes_for_shape_dtype(shape: tuple[int, ...], dtype: object, sharding: object) -> tuple[int, ...]:
    """Summarize target local shard byte sizes for an expected shape/dtype."""
    if not hasattr(sharding, "addressable_devices_indices_map"):
        return ()
    out: set[int] = set()
    try:
        mapping = sharding.addressable_devices_indices_map(shape)
    except Exception:
        return ()
    for index in mapping.values():
        size = _shape_nbytes(_index_shape(index, shape), dtype)
        if size is not None:
            out.add(size)
    return tuple(sorted(out))


def _target_shard_nbytes(value: object, sharding: object) -> tuple[int, ...]:
    """Summarize target local shard byte sizes for ``value`` on ``sharding``."""
    payload = _array_payload(value)
    if not hasattr(payload, "shape") or not hasattr(payload, "dtype"):
        return ()
    return _target_shard_nbytes_for_shape_dtype(
        tuple(getattr(payload, "shape", ())), getattr(payload, "dtype", None), sharding
    )


def _ordered_sharding_index_abi(
    sharding: object, shape: tuple[int, ...]
) -> tuple[tuple[object, tuple[int, ...]], ...] | None:
    """Return the shard-index ABI for ``sharding`` in mesh-device order.

    The physical devices may differ across pipeline ranks, so the device ids are
    intentionally not part of the ABI. What must match is the per-mesh-position
    global index and local shard shape.
    """
    if sharding is None or not hasattr(sharding, "devices_indices_map"):
        return None
    try:
        mapping = sharding.devices_indices_map(shape)
    except Exception:
        return None
    if not mapping:
        return ()

    mesh = getattr(sharding, "mesh", None)
    devices = None
    if mesh is not None:
        try:
            devices = tuple(mesh.devices.flat)
        except Exception:
            devices = None
    if devices is None:
        try:
            devices = tuple(sorted(mapping, key=lambda device: int(getattr(device, "id", id(device)))))
        except Exception:
            devices = tuple(mapping)

    out: list[tuple[object, tuple[int, ...]]] = []
    for device in devices:
        if device not in mapping:
            return None
        index = mapping[device]
        out.append((_index_key(index), _index_shape(index, shape)))
    return tuple(out)


def _same_index_sharding_abi(value: object, target_sharding: object) -> bool:
    """Prove a cross-device retarget keeps the exact shard-index ABI.

    This is deliberately stricter than shape-only matching: the source and
    target must have the same number of devices, the same mesh axis signature,
    and the same ordered shard indices/shard shapes for the value's global
    shape. It permits rank-to-rank activation handoff while continuing to reject
    full-mesh -> stage-mesh subset placement.
    """
    payload = _array_payload(value)
    if not isinstance(payload, jax.Array) or not hasattr(payload, "shape"):
        return False
    value = payload
    source_sharding = _value_sharding(value)
    if source_sharding is None or target_sharding is None:
        return False
    source_devices = _array_device_set(value)
    target_devices = _sharding_device_set(target_sharding)
    if source_devices is None or target_devices is None:
        return False
    if source_devices == target_devices or len(source_devices) != len(target_devices):
        return False
    if _mesh_axis_names(source_sharding) != _mesh_axis_names(target_sharding):
        return False
    shape = tuple(value.shape)
    if isinstance(source_sharding, jax.sharding.NamedSharding) and _spec_axis_shape_mismatches(
        getattr(source_sharding, "spec", None),
        getattr(source_sharding, "mesh", None),
        shape,
    ):
        return False
    if isinstance(target_sharding, jax.sharding.NamedSharding) and _spec_axis_shape_mismatches(
        getattr(target_sharding, "spec", None),
        getattr(target_sharding, "mesh", None),
        shape,
    ):
        return False
    source_abi = _ordered_sharding_index_abi(source_sharding, shape)
    target_abi = _ordered_sharding_index_abi(target_sharding, shape)
    return source_abi is not None and source_abi == target_abi


def _mesh_device_id_grid(mesh: object) -> tuple[int, ...] | None:
    """Return mesh device ids in concrete mesh order."""
    try:
        devices = tuple(np.asarray(mesh.devices).flat)
    except Exception:
        return None
    return tuple(int(getattr(device, "id", idx)) for idx, device in enumerate(devices))


def _mesh_shape_key(mesh: object) -> tuple[tuple[object, int], ...] | None:
    """Return mesh axis sizes in the ABI order used by JAX executables."""
    if mesh is None:
        return None
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    if not axis_names:
        return ()
    shape = getattr(mesh, "shape", None)
    if shape is not None:
        try:
            return tuple((axis, int(shape[axis])) for axis in axis_names)
        except Exception:
            pass
    try:
        device_shape = tuple(int(dim) for dim in np.asarray(mesh.devices, dtype=object).shape)
    except Exception:
        return None
    if len(device_shape) != len(axis_names):
        return None
    return tuple((axis, dim) for axis, dim in zip(axis_names, device_shape, strict=False))
