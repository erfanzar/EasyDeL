# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""JAX mesh creation helpers for SpectraX SPMD and MPMD training.

Wraps :func:`jax.experimental.mesh_utils.create_device_mesh` and
:func:`jax.experimental.mesh_utils.create_hybrid_device_mesh` with
sensible defaults for the common pipeline / data / FSDP / tensor /
sequence / expert parallelism axis set, plus three conveniences:

* :func:`create_mesh` — the main entry point. Automatically detects
  multi-slice TPU pod setups and multi-process distributed runs, and
  picks the right mesh-creation path.
* :func:`parse_mesh_from_string` — parse a human-readable config like
  ``"dp:2,tp:4"`` or the positional form ``"2,4"``.
* :func:`cpu_context` — one-shot context manager giving you a CPU
  mesh with JAX pinned to CPU; handy for local debug / unit tests.

The implementation is now owned by SpectraX so mesh construction can
carry MPMD metadata all the way into the runtime.
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

import jax
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh, create_hybrid_device_mesh
from jax.sharding import AxisType, Mesh

__all__ = [
    "DEFAULT_MESH_AXIS_DIMS",
    "DEFAULT_MESH_AXIS_NAMES",
    "SpxMesh",
    "calculate_host_mesh_shape",
    "cpu_context",
    "create_cpu_mesh",
    "create_mesh",
    "current_mesh",
    "force_cpu",
    "parse_mesh_from_string",
    "use_mesh",
]


_CURRENT_MESH: threading.local = threading.local()


def current_mesh() -> SpxMesh | None:
    """Return the innermost active :class:`SpxMesh`, if any.

    Returns:
        Return the innermost active :class:`SpxMesh`, if any.
    """
    stack = getattr(_CURRENT_MESH, "stack", ())
    return stack[-1] if stack else None


@dataclass(frozen=True)
class SpxMesh:
    """Unified spectrax mesh — a :class:`~jax.sharding.Mesh` plus optional
    MPMD axis tagging.

    Every spectrax API takes :class:`SpxMesh` as ``mesh=...``. Pure-SPMD
    setups create one with ``mpmd_axis=None`` and behave like a plain
    JAX ``Mesh`` (use ``mesh.jax_mesh`` to drop into JAX). When
    ``mpmd_axis`` is set, ``mesh.mpmd_mesh`` exposes the
    :class:`~spectrax.runtime.types.MpMdMesh` view that pipeline
    runtimes consume.

    Attributes:
        jax_mesh: The underlying :class:`~jax.sharding.Mesh`.
        mpmd_axis: Optional axis name reserved for MPMD pipeline
            stages. ``None`` for pure-SPMD setups.

    Properties:
        ``mpmd_mesh`` : :class:`~spectrax.runtime.types.MpMdMesh` | ``None`` —
            built lazily; ``None`` iff ``mpmd_axis is None``.
        ``is_mpmd`` : bool — convenience for ``mpmd_axis is not None``.
        ``axis_names`` / ``shape`` / ``devices`` — forward to the
            underlying ``jax_mesh`` for ergonomic access.
    """

    jax_mesh: Mesh
    mpmd_axis: str | None = None
    _mpmd_mesh_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that ``mpmd_axis`` (if set) names an axis of ``jax_mesh``."""
        if self.mpmd_axis is not None and self.mpmd_axis not in self.jax_mesh.axis_names:
            raise ValueError(f"mpmd_axis {self.mpmd_axis!r} not in jax_mesh.axis_names {self.jax_mesh.axis_names}.")

    @property
    def is_mpmd(self) -> bool:
        """``True`` iff there is a real MPMD pipeline (mpmd_axis set AND its dim > 1).

                @erfanzar NOTE:
        Naming an mpmd_axis with dim==1 is a degenerate single-stage
        pipeline — semantically equivalent to no pipeline at all.
        Treating it as MPMD routes every ``spx.jit`` through the
        heavy MPMD pipeline runtime (cluster builder, per-rank
        re-jit, layout reassignment) and forces XLA to insert
        layout-reformat copies at JIT boundaries because the MPMD
        path does not honor caller-allocated array layouts. We
        require the axis to actually have parallel size to flip
        this on.

        Returns:
            Result described by this helper.
        """
        if self.mpmd_axis is None:
            return False
        try:
            return int(self.jax_mesh.shape[self.mpmd_axis]) > 1
        except (KeyError, AttributeError, TypeError):
            return True

    @property
    def mpmd_mesh(self):
        """Lazy :class:`~spectrax.runtime.types.MpMdMesh` view.

        Returns ``None`` when ``mpmd_axis`` is unset. Construction is
        deferred (and cached) to avoid a circular import between
        ``spectrax.sharding`` and ``spectrax.runtime``.
        """
        if self.mpmd_axis is None:
            return None
        if "mm" not in self._mpmd_mesh_cache:
            from ..runtime.types.mesh import MpMdMesh

            self._mpmd_mesh_cache["mm"] = MpMdMesh(self.jax_mesh, self.mpmd_axis)
        return self._mpmd_mesh_cache["mm"]

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Tuple of mesh-axis names from the underlying :class:`jax.sharding.Mesh`.

        Names follow the order they were passed to ``Mesh(devices, axis_names)``.
        Includes the pipeline axis if one is configured (use
        :attr:`spmd_axis_names` on :class:`MpMdMesh` to get only the
        SPMD axes).

        Returns:
            Result described by this helper.
        """
        return self.jax_mesh.axis_names

    @property
    def shape(self):
        """Per-axis device count, as an ``OrderedDict`` keyed by axis name.

        Equivalent to ``self.jax_mesh.shape`` and identical to the
        plain JAX mesh in shape.
        """
        return self.jax_mesh.shape

    @property
    def devices(self):
        """Multi-dimensional NumPy array of devices indexed by mesh axes.

        Same shape as :attr:`shape`. Forwarded from the underlying
        :class:`jax.sharding.Mesh`; SpectraX does not re-shape the
        device grid.
        """
        return self.jax_mesh.devices

    def __enter__(self):
        """Enter the mesh context and yield this :class:`SpxMesh`."""
        self.jax_mesh.__enter__()
        stack = list(getattr(_CURRENT_MESH, "stack", ()))
        stack.append(self)
        _CURRENT_MESH.stack = tuple(stack)
        return self

    def __exit__(self, *args):
        """Exit the underlying ``jax_mesh`` context and pop ``self`` from the SpectraX stack.

        Mirrors :meth:`__enter__`: the SpectraX-level ``_CURRENT_MESH``
        stack is unwound to ``self`` (handling well-nested and
        out-of-order exits) before the JAX mesh's ``__exit__`` runs.
        object exception from the JAX side propagates unchanged.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
        """
        stack = list(getattr(_CURRENT_MESH, "stack", ()))
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx] is self:
                del stack[idx]
                break
        _CURRENT_MESH.stack = tuple(stack)
        return self.jax_mesh.__exit__(*args)

    def __getattr__(self, name: str):
        """Forward unknown attributes to the underlying :class:`jax.sharding.Mesh`.

        Makes :class:`SpxMesh` a near-drop-in for plain ``Mesh`` in
        user code. Dunder names skip this path (they're looked up on
        the type, not the instance, so ``__getattr__`` isn't invoked).

        Args:
            name: Name used for lookup, logging, or registration.
        """
        try:
            return getattr(self.jax_mesh, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} has no attribute {name!r} (forwarded lookup on jax_mesh also failed)."
            ) from None


DEFAULT_MESH_AXIS_DIMS: tuple[int, ...] = (1, 1, -1, 1, 1, 1)
"""Default per-axis dimension sizes for :func:`create_mesh`.

Matches the ``(pp, dp, fsdp, ep, tp, sp)`` convention: FSDP absorbs
all remaining devices by default (``-1`` means "fill with what's
left"), everything else replicated.
"""


DEFAULT_MESH_AXIS_NAMES: tuple[str, ...] = ("pp", "dp", "fsdp", "ep", "tp", "sp")
"""Default axis names for :func:`create_mesh`.

- ``pp``: pipeline parallelism (split model by stages — see
  :mod:`spectrax.runtime.mpmd`)
- ``dp``: data parallelism (replicate model, split batch)
- ``fsdp``: fully-sharded data parallelism (ZeRO-3-like)
- ``ep``: expert parallelism (mixture-of-experts)
- ``tp``: tensor parallelism (split model-width dims)
- ``sp``: sequence parallelism (split sequence dim)
"""


_AXIS_TYPE_BY_NAME: dict[str, AxisType] = {
    "auto": AxisType.Auto,
    "explicit": AxisType.Explicit,
    "manual": AxisType.Manual,
}


_SPX_MESH_CACHE: dict[tuple[object, ...], SpxMesh] = {}
_SPX_TOPOLOGY_MESH_CACHE: dict[tuple[int, str], Mesh] = {}


@contextlib.contextmanager
def use_mesh(mesh: SpxMesh | Mesh) -> Iterator[SpxMesh]:
    """Enter a mesh context and yield a :class:`SpxMesh`.

    Passing a raw :class:`jax.sharding.Mesh` is accepted at migration
    boundaries, but SpectraX code receives the wrapped ``SpxMesh`` so
    MPMD metadata can continue to flow through sharding helpers.

    Args:
        mesh: JAX mesh or SpectraX mesh descriptor used for placement.

    Returns:
        Result described by this helper.
    """
    spx_mesh = mesh if isinstance(mesh, SpxMesh) else _wrap_spx(mesh, None)
    with spx_mesh:
        yield spx_mesh


def _device_sort_key(device: object) -> tuple[object, ...]:
    """Build a stable sort key for a JAX device.

    The key is ``(process_index, slice_index, *coords, id)``, ordered
    to keep devices on the same host/slice contiguous. Used as a
    tiebreaker when ranking stage placements in
    :func:`_topology_mpmd_order`.

    Args:
        device: Device value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    coords = getattr(device, "coords", None)
    if coords is None:
        coords = ()
    try:
        coords_tuple = tuple(int(v) for v in coords)
    except Exception:
        coords_tuple = ()
    return (
        int(getattr(device, "process_index", 0)),
        int(getattr(device, "slice_index", 0)),
        *coords_tuple,
        int(getattr(device, "id", 0)),
    )


def _stage_center(devices: np.ndarray) -> tuple[float, ...] | None:
    """Compute the centroid of a set of devices in topology coordinate space.

    Each device exposes a ``coords`` tuple from the underlying TPU/GPU
    interconnect. The centroid is the per-component mean. Returns
    ``None`` if any device lacks coords or has non-numeric ones, which
    forces callers to fall back to the default mesh ordering.

    Args:
        devices: Device collection used to construct or inspect a mesh.

    Returns:
        Result described by this helper.
    """
    coords: list[tuple[float, ...]] = []
    for device in devices.reshape(-1):
        raw = getattr(device, "coords", None)
        if raw is None:
            return None
        try:
            coords.append(tuple(float(v) for v in raw))
        except Exception:
            return None
    if not coords:
        return None
    width = max(len(c) for c in coords)
    padded = [c + (0.0,) * (width - len(c)) for c in coords]
    return tuple(float(sum(c[i] for c in padded) / len(padded)) for i in range(width))


def _center_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Squared Euclidean distance between two centroid tuples.

    Pads the shorter tuple with zeros so different-rank centroids can
    still be compared. Squared (not square-rooted) since callers only
    need the relative ordering.

    Args:
        a: Positional arguments forwarded to the wrapped callable.
        b: B value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    width = max(len(a), len(b))
    aa = a + (0.0,) * (width - len(a))
    bb = b + (0.0,) * (width - len(b))
    return sum((x - y) ** 2 for x, y in zip(aa, bb, strict=True))


def _topology_mpmd_order(jax_mesh: Mesh, mpmd_axis: str) -> tuple[int, ...] | None:
    """Compute a topology-aware permutation of stages along ``mpmd_axis``.

    Given a JAX mesh and the name of its pipeline axis, return a
    tuple of stage indices ordered so that adjacent stages are
    physically close on the interconnect. The first stage is the one
    whose centroid sorts smallest; each subsequent stage is the
    closest unvisited one (greedy nearest-neighbor walk on
    centroids).

    Returns ``None`` if the heuristic isn't useful — fewer than three
    stages, missing topology coords, or the natural mesh order
    already matches the topology order. ``None`` signals "leave the
    mesh device order alone".

    Args:
        jax_mesh: Jax mesh value consumed by this operation.
        mpmd_axis: Mpmd axis value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    axis = jax_mesh.axis_names.index(mpmd_axis)
    n = int(jax_mesh.devices.shape[axis])
    if n <= 2:
        return None
    centers: list[tuple[float, ...]] = []
    tie_keys: list[tuple[object, ...]] = []
    for idx in range(n):
        sub = np.take(jax_mesh.devices, indices=[idx], axis=axis)
        center = _stage_center(sub)
        if center is None:
            return None
        centers.append(center)
        tie_keys.append(min((_device_sort_key(device) for device in sub.reshape(-1)), default=(idx,)))

    remaining = set(range(n))
    current = min(remaining, key=lambda i: (centers[i], tie_keys[i], i))
    order = [current]
    remaining.remove(current)
    while remaining:
        current = min(
            remaining, key=lambda i: (_center_distance(centers[current], centers[i]), centers[i], tie_keys[i], i)
        )
        order.append(current)
        remaining.remove(current)
    if order == list(range(n)):
        return None
    return tuple(order)


def _topology_order_mpmd_axis(jax_mesh: Mesh, mpmd_axis: str | None) -> Mesh:
    """Return a permuted mesh with stages ordered by physical proximity.

    Wraps :func:`_topology_mpmd_order`: if the helper returns a
    non-trivial permutation, the device array is re-sliced along
    ``mpmd_axis`` and a fresh :class:`jax.sharding.Mesh` is built
    (cached by ``id(jax_mesh) x mpmd_axis`` so subsequent calls
        return the same mesh object). Otherwise the original mesh is
    returned unchanged.

    Args:
        jax_mesh: Jax mesh value consumed by this operation.
        mpmd_axis: Mpmd axis value consumed by this operation.

    Returns:
        Return a permuted mesh with stages ordered by physical proximity.
    """
    if mpmd_axis is None:
        return jax_mesh
    key = (id(jax_mesh), mpmd_axis)
    cached = _SPX_TOPOLOGY_MESH_CACHE.get(key)
    if cached is not None:
        return cached
    order = _topology_mpmd_order(jax_mesh, mpmd_axis)
    if order is None:
        return jax_mesh
    axis = jax_mesh.axis_names.index(mpmd_axis)
    devices = np.take(jax_mesh.devices, indices=order, axis=axis)
    reordered = Mesh(devices, jax_mesh.axis_names, axis_types=getattr(jax_mesh, "axis_types", None))
    _SPX_TOPOLOGY_MESH_CACHE[key] = reordered
    return reordered


def _get_num_slices(devices: Sequence[object]) -> int:
    """Count the distinct TPU pod slices across ``devices``.

    Inspects each device for a ``slice_index`` attribute (exposed on
    multi-slice TPU pod runtimes). Falls back to the
    ``MEGASCALE_NUM_SLICES`` environment variable when device objects
    don't carry the attribute.

    Args:
        devices: Device collection used to construct or inspect a mesh.

    Returns:
        Result described by this helper.
    """
    num_slices = 1
    if devices and hasattr(devices[0], "slice_index"):
        try:
            num_slices = len({d.slice_index for d in devices})
        except Exception:
            pass
    if num_slices == 1:
        num_slices = int(os.environ.get("MEGASCALE_NUM_SLICES", num_slices))
    return num_slices


def _normalize_axis_types(
    axis_names: Sequence[str],
    axis_types: Sequence[AxisType | str] | AxisType | str | None,
) -> tuple[AxisType, ...] | None:
    """Coerce an ``axis_types`` argument into a ``(AxisType, ...)`` tuple.

    Accepts either a single value (applied to every axis), a sequence
    parallel to ``axis_names``, or ``None``.

    Args:
        axis_names: Axis names; fixes the output tuple length.
        axis_types: ``AxisType`` values, their string names
            (``"auto"`` / ``"explicit"`` / ``"manual"``), or ``None``.

    Returns:
        A tuple of ``AxisType`` of length ``len(axis_names)``, or
        ``None`` if ``axis_types`` was ``None``.

    Raises:
        ValueError: On unknown string names or length mismatches.
        TypeError: On unsupported entry types.
    """
    if axis_types is None:
        return None
    if isinstance(axis_types, (AxisType, str)):
        axis_types_seq: Sequence[AxisType | str] = (axis_types,) * len(axis_names)
    else:
        axis_types_seq = tuple(axis_types)
        if len(axis_types_seq) == 1 and len(axis_names) > 1:
            axis_types_seq = axis_types_seq * len(axis_names)
    normalized: list[AxisType] = []
    for at in axis_types_seq:
        if isinstance(at, str):
            key = at.strip().lower()
            if key not in _AXIS_TYPE_BY_NAME:
                raise ValueError(
                    f"axis_types entries must be one of {{'auto', 'explicit', 'manual'}} or AxisType; got {at!r}."
                )
            normalized.append(_AXIS_TYPE_BY_NAME[key])
        elif isinstance(at, AxisType):
            normalized.append(at)
        else:
            raise TypeError(f"axis_types entries must be strings or AxisType values, got {type(at).__name__}.")
    if len(normalized) != len(axis_names):
        raise ValueError(f"axis_types length ({len(normalized)}) must match axis_names length ({len(axis_names)}).")
    return tuple(normalized)


def calculate_host_mesh_shape(
    global_mesh_shape: Sequence[int],
    total_devices: int | None = None,
    num_processes: int | None = None,
) -> tuple[int, ...]:
    """Compute the per-host mesh shape for a multi-process global mesh.

    Given the desired global mesh, decide how each host slices off
    its share. Greedy packing: consume ``num_processes`` copies off
    the leading axis that can supply them; move to the next axis when
    the current one runs out.

    Args:
        global_mesh_shape: Desired global mesh dimensions across all
            processes.
        total_devices: Devices on this host. Default:
            ``jax.local_device_count()``.
        num_processes: Total processes. Default: ``jax.process_count()``.

    Returns:
        Per-host mesh shape (same rank as ``global_mesh_shape``).

    Raises:
        ValueError: If the global mesh size doesn't match
            ``total_devices * num_processes`` or the packing can't
            produce a shape with exactly ``total_devices`` entries.

    Example:
        >>> calculate_host_mesh_shape((2, 4), total_devices=4, num_processes=2)
        (1, 4)
    """
    if total_devices is None:
        total_devices = jax.local_device_count()
    if num_processes is None:
        num_processes = jax.process_count()
    if total_devices <= 0:
        raise ValueError(f"total_devices must be > 0, got {total_devices}.")
    if num_processes <= 0:
        raise ValueError(f"num_processes must be > 0, got {num_processes}.")
    total_mesh_size = int(np.prod(global_mesh_shape))
    if total_mesh_size != total_devices * num_processes:
        raise ValueError(
            f"Mesh size {total_mesh_size} doesn't match available devices "
            f"{total_devices * num_processes} (local x processes)."
        )

    def _walk(remaining: int, dims: tuple[int, ...]) -> tuple[int, ...] | None:
        """Find inter-host factors that multiply to ``remaining``, dim by dim.

        Prefers larger factors on the leading dims (matching the historical
        greedy intent of keeping per-host blocks contiguous on trailing axes).

        Args:
            remaining: Remaining value consumed by this operation.
            dims: Dims value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if not dims:
            return () if remaining == 1 else None
        d = dims[0]
        for f in range(d, 0, -1):
            if d % f == 0 and remaining % f == 0:
                rest = _walk(remaining // f, dims[1:])
                if rest is not None:
                    return (f, *rest)
        return None

    inter_host = _walk(num_processes, tuple(global_mesh_shape))
    if inter_host is None:
        raise ValueError(
            f"Cannot factor num_processes={num_processes} across global mesh "
            f"shape {tuple(global_mesh_shape)} so that each per-process slice "
            f"contains exactly {total_devices} devices."
        )
    host_mesh = tuple(d // f for d, f in zip(global_mesh_shape, inter_host, strict=False))
    return host_mesh


def _cached_mesh(
    axis_dims: Sequence[int],
    axis_names: Sequence[str],
    axis_types: Sequence[AxisType] | None,
    dcn_mesh_dims: Sequence[int] | None,
    should_sort_granules_by_key: bool,
    allow_split_physical_axes: bool,
    backend: str | None,
) -> Mesh:
    """Hashable-args wrapper around :func:`_cached_mesh_impl`.

    Coerces every sequence argument to a tuple (so :func:`functools.cache`
    can use them as dict keys) and falls back to
    :func:`jax.default_backend` when ``backend`` is ``None``. The actual
    mesh creation happens in :func:`_cached_mesh_impl`.

    Args:
        axis_dims: Axis dims value consumed by this operation.
        axis_names: Named mesh or collective axes used by the operation.
        axis_types: Axis types value consumed by this operation.
        dcn_mesh_dims: Dcn mesh dims value consumed by this operation.
        should_sort_granules_by_key: Should sort granules by key value consumed by this operation.
        allow_split_physical_axes: Allow split physical axes value consumed by this operation.
        backend: Backend value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return _cached_mesh_impl(
        axis_dims=tuple(axis_dims),
        axis_names=tuple(axis_names),
        axis_types=None if axis_types is None else tuple(axis_types),
        dcn_mesh_dims=None if dcn_mesh_dims is None else tuple(dcn_mesh_dims),
        should_sort_granules_by_key=should_sort_granules_by_key,
        allow_split_physical_axes=allow_split_physical_axes,
        backend=backend or jax.default_backend(),
    )


@functools.cache
def _cached_mesh_impl(
    axis_dims: tuple[int, ...],
    axis_names: tuple[str, ...],
    axis_types: tuple[AxisType, ...] | None,
    dcn_mesh_dims: tuple[int, ...] | None,
    should_sort_granules_by_key: bool,
    allow_split_physical_axes: bool,
    backend: str,
) -> Mesh:
    """Cached mesh-creation implementation.

    Three branches:

        1. **Multi-slice TPU pods** — one mesh axis is divided across the
           slices, the rest replicated via ``create_hybrid_device_mesh``
           with ``process_is_granule=False``.
        2. **Multi-process (non-TPU) setups** — per-host submesh via
           :func:`calculate_host_mesh_shape` + ``create_hybrid_device_mesh``
           with ``process_is_granule=True``.
        3. **Single-process** — ordinary :func:`create_device_mesh`.

    Args:
        axis_dims: Axis dims value consumed by this operation.
        axis_names: Named mesh or collective axes used by the operation.
        axis_types: Axis types value consumed by this operation.
        dcn_mesh_dims: Dcn mesh dims value consumed by this operation.
        should_sort_granules_by_key: Should sort granules by key value consumed by this operation.
        allow_split_physical_axes: Allow split physical axes value consumed by this operation.
        backend: Backend value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    devices = jax.devices(backend)
    total_devices = jax.device_count(backend)
    local_devices = jax.local_device_count(backend)
    process_count = jax.process_count()
    global_mesh_shape = np.arange(total_devices).reshape(axis_dims).shape

    num_slices = _get_num_slices(devices)

    def fill_minus_one(shape: tuple[int, ...], target: int) -> tuple[int, ...]:
        """Replace a single ``-1`` entry with the value needed to reach ``target`` product.

        Args:
            shape: Array shape requested by the initializer or helper.
            target: Target value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        shp = list(shape)
        minus = [i for i, v in enumerate(shp) if v == -1]
        if len(minus) > 1:
            raise ValueError("Only one -1 is supported in dcn_mesh_dims.")
        prod_known = 1
        for v in shp:
            if v != -1:
                if v <= 0:
                    raise ValueError(f"dcn_mesh_dims entries must be > 0 or -1, got {v}.")
                prod_known *= v
        if minus:
            if target % prod_known != 0:
                raise ValueError(f"dcn_mesh_dims product ({prod_known}) does not divide target ({target}).")
            shp[minus[0]] = target // prod_known
        if np.prod(shp) != target:
            raise ValueError(f"dcn_mesh_dims product {int(np.prod(shp))} must equal {target}; got {tuple(shp)}.")
        return tuple(int(v) for v in shp)

    if num_slices > 1:
        dynamic_axis = next(
            (i for i, dim in enumerate(global_mesh_shape) if dim % num_slices == 0),
            None,
        )

        if dynamic_axis is None:
            raise ValueError(
                f"Multi-slice detected (num_slices={num_slices}) but no mesh "
                f"axis in {global_mesh_shape} is divisible by num_slices."
            )
        per_slice_mesh_shape = list(global_mesh_shape)
        per_slice_mesh_shape[dynamic_axis] //= num_slices
        per_slice_tuple = tuple(per_slice_mesh_shape)
        if dcn_mesh_dims is None:
            dcn_list = [1] * len(axis_dims)
            dcn_list[dynamic_axis] = num_slices
            dcn: tuple[int, ...] = tuple(dcn_list)
        else:
            dcn = fill_minus_one(dcn_mesh_dims, num_slices)
        ndarray = create_hybrid_device_mesh(
            mesh_shape=per_slice_tuple,
            dcn_mesh_shape=dcn,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
            process_is_granule=False,
            should_sort_granules_by_key=should_sort_granules_by_key,
        )
    elif process_count > 1:
        local_mesh_shape = calculate_host_mesh_shape(
            global_mesh_shape=global_mesh_shape,
            total_devices=local_devices,
            num_processes=process_count,
        )
        if dcn_mesh_dims is None:
            ratios = [int(g // le) for g, le in zip(global_mesh_shape, local_mesh_shape, strict=False)]
            if np.prod(ratios) != process_count:
                ratios = [1] * len(axis_dims)
                for i in range(len(axis_dims)):
                    ratios[i] = process_count
                    break
            dcn = tuple(ratios)
        else:
            dcn = fill_minus_one(dcn_mesh_dims, process_count)
        ndarray = create_hybrid_device_mesh(
            mesh_shape=local_mesh_shape,
            dcn_mesh_shape=dcn,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
            process_is_granule=True,
            should_sort_granules_by_key=should_sort_granules_by_key,
        )
    else:
        ndarray = create_device_mesh(
            mesh_shape=global_mesh_shape,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )

    return Mesh(ndarray, axis_names, axis_types=axis_types)


def create_mesh(
    axis_dims: Sequence[int] = DEFAULT_MESH_AXIS_DIMS,
    axis_names: Sequence[str] = DEFAULT_MESH_AXIS_NAMES,
    *,
    mpmd_axis: str | None = None,
    dcn_mesh_dims: Sequence[int] | None = None,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = True,
    backend: str | None = None,
    use_jax: bool = False,
    axis_types: Sequence[AxisType | str] | AxisType | str | Literal["auto", "explicit", "manual"] | None = None,
) -> SpxMesh:
    """Create an :class:`SpxMesh` backed by a JAX mesh.

    Single-line drop-in for most spectrax SPMD / FSDP / TP / PP setups.
    The default axis names ``(pp, dp, fsdp, ep, tp, sp)`` match what
    :mod:`spectrax.sharding.partition` and :mod:`spectrax.runtime`
    expect; passing ``axis_dims=(1, 1, -1, 1, 1, 1)`` packs every
    device into the FSDP axis — the canonical "distribute everything"
    setup.

    The function auto-detects:

    * **Multi-slice TPU pods** — via device ``slice_index`` or
      ``MEGASCALE_NUM_SLICES`` env. Splits one mesh axis across
      slices and sets up the DCN (data-center-network) mapping.
    * **Multi-process (non-TPU)** — via ``jax.process_count()``.
      Derives a per-host submesh via
      :func:`calculate_host_mesh_shape`.
    * **Single-process** — plain :func:`create_device_mesh`.

    Results are cached by argument tuple — calling
    ``create_mesh(same_args)`` twice returns the same
    :class:`~jax.sharding.Mesh` object.

    Args:
        axis_dims: Dimensions for each mesh axis. One entry may be
            ``-1`` meaning "use all remaining devices". Must multiply
            to the total device count.
        axis_names: Names for each axis. Length must match
            ``axis_dims``.
        mpmd_axis: Optional axis name to tag as MPMD (pipeline-parallel).
            When set, the returned :class:`SpxMesh`'s ``mpmd_mesh``
            attribute is the :class:`~spectrax.runtime.types.MpMdMesh`
            view consumed by pipeline runtimes (``sxcall`` / ``sxjit``).
            ``None`` (default) -> pure-SPMD mesh.
        dcn_mesh_dims: Explicit DCN (inter-slice / inter-host) mesh
            dimensions. ``None`` means "auto-calculate"; one ``-1``
            is allowed to fill remaining.
        should_sort_granules_by_key: Whether to sort granules (hosts
            / slices) for consistent device ordering across processes.
        allow_split_physical_axes: Allow physical device axes to be
            split across logical mesh axes (needed for most exotic
            topologies).
        backend: JAX backend. ``None`` -> :func:`jax.default_backend`.
        use_jax: If ``True``, use :func:`jax.make_mesh` for
            single-process / single-slice setups (drops the
            explicit-mesh-utils path). Multi-slice and multi-process
            topologies always use mesh_utils.
        axis_types: Optional per-axis :class:`AxisType` — either
            ``"auto"`` / ``"explicit"`` / ``"manual"`` (string), an
            ``AxisType`` enum, or a sequence per-axis. Default
            ``None`` = JAX default.

    Returns:
        An :class:`SpxMesh` ready for SpectraX mesh contexts and
        sharding helpers. Use :func:`spectrax.to_jax_mesh` only at
        explicit JAX API boundaries that require a raw
        :class:`~jax.sharding.Mesh`.

    Example:
        >>> import spectrax as spx
        >>> mesh = spx.sharding.create_mesh(
        ...     axis_dims=(2, 4),
        ...     axis_names=("data", "model"),
        ... )
        >>> with mesh:
        ...     ...
    """
    if mpmd_axis is not None and mpmd_axis not in axis_names:
        raise ValueError(f"mpmd_axis {mpmd_axis!r} not in axis_names {tuple(axis_names)}.")

    axis_types_norm = _normalize_axis_types(axis_names, axis_types)
    if use_jax:
        devices = jax.devices(backend)
        num_slices = _get_num_slices(devices)
        process_count = jax.process_count()
        if num_slices == 1 and process_count == 1 and dcn_mesh_dims is None:
            total_devices = len(devices)
            axis_dims = np.arange(total_devices).reshape(axis_dims).shape
            jm = jax.make_mesh(
                axis_shapes=axis_dims,
                axis_names=axis_names,
                axis_types=axis_types_norm,
                devices=devices,
            )
            return _wrap_spx(jm, mpmd_axis)
    jm = _cached_mesh(
        axis_dims=axis_dims,
        axis_names=axis_names,
        axis_types=axis_types_norm,
        dcn_mesh_dims=dcn_mesh_dims,
        should_sort_granules_by_key=should_sort_granules_by_key,
        allow_split_physical_axes=allow_split_physical_axes,
        backend=backend,
    )
    return _wrap_spx(jm, mpmd_axis)


def _wrap_spx(jax_mesh: Mesh, mpmd_axis: str | None) -> SpxMesh:
    """Wrap ``jax_mesh`` in :class:`SpxMesh`, caching per ``(jax_mesh,
    mpmd_axis)`` so repeat calls to :func:`create_mesh` return the
    *same* :class:`SpxMesh` object (matches the underlying
    ``_cached_mesh`` semantics).

    Args:
        jax_mesh: Jax mesh value consumed by this operation.
        mpmd_axis: Mpmd axis value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    key = (id(jax_mesh), mpmd_axis)
    cached = _SPX_MESH_CACHE.get(key)
    if cached is not None:
        return cached
    jax_mesh = _topology_order_mpmd_axis(jax_mesh, mpmd_axis)
    sm = SpxMesh(jax_mesh=jax_mesh, mpmd_axis=mpmd_axis)
    _SPX_MESH_CACHE[key] = sm
    return sm


def parse_mesh_from_string(axis_dims: str, names: Sequence[str]) -> SpxMesh:
    """Parse a mesh configuration from a human-readable string.

    Two formats supported:

    * **Named** — ``"dp:2,tp:4"`` explicitly maps axis name to size.
      Every name in ``names`` must appear exactly once.
    * **Positional** — ``"2,4"`` maps by position to the ``names``
      list.

    Useful for CLI / env-var driven configuration.

    Args:
        axis_dims: The config string (one of the two formats above).
        names: Axis names the result is expected to carry.

    Returns:
        The resulting :class:`SpxMesh`.

    Raises:
        ValueError: Unknown name, length mismatch, or the named
            entries don't cover ``names`` exactly.

    Example:
        >>> spx.sharding.parse_mesh_from_string("dp:2,tp:4", ("dp", "tp"))
        >>> spx.sharding.parse_mesh_from_string("2,4", ("data", "model"))
    """
    if ":" in axis_dims:
        dims: list[int] = []
        dim_names: list[str] = []
        for axis in axis_dims.split(","):
            name, dim = axis.split(":")
            if name not in names:
                raise ValueError(f"Axis name {name!r} not found in provided names: {tuple(names)}.")
            dims.append(int(dim))
            dim_names.append(name)
        if set(dim_names) != set(names):
            raise ValueError(
                f"Not all axis names were used in axis_dims. Expected: {tuple(names)}; got: {tuple(dim_names)}."
            )
    else:
        dims = [int(x) for x in axis_dims.split(",")]
        dim_names = list(names)
    if len(dims) != len(names):
        raise ValueError(f"Number of dimensions ({len(dims)}) must match names ({len(names)}).")
    return create_mesh(tuple(dims), tuple(dim_names))


def create_cpu_mesh(
    axis_dims: Sequence[int] = DEFAULT_MESH_AXIS_DIMS,
    axis_names: Sequence[str] = DEFAULT_MESH_AXIS_NAMES,
) -> SpxMesh:
    """Create a CPU-backed mesh for local debugging / unit tests.

    Shortcut for ``create_mesh(..., backend="cpu")``. Pairs naturally
    with ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` to
    simulate a multi-device setup on a single host.

    Args:
        axis_dims: Per-axis dims. Default
            :data:`DEFAULT_MESH_AXIS_DIMS`.
        axis_names: Per-axis names. Default
            :data:`DEFAULT_MESH_AXIS_NAMES`.

    Returns:
        An :class:`SpxMesh` whose devices are all CPU.
    """
    return create_mesh(
        axis_dims=tuple(axis_dims),
        axis_names=tuple(axis_names),
        backend="cpu",
    )


@contextlib.contextmanager
def force_cpu() -> Iterator[object]:
    """Temporarily pin JAX's default device to CPU.

    Unrelated state (meshes, already-placed arrays) is not affected.
    On exit, the previous default device is restored.

    Yields:
        The CPU device that was installed as default.

    Example:
        >>> with spx.sharding.force_cpu() as cpu:
        ...     y = jnp.ones((4,)) + 1      # runs on `cpu`
    """
    cpu = jax.local_devices(backend="cpu")[0]
    with jax.default_device(cpu):
        yield cpu


@contextlib.contextmanager
def cpu_context() -> Iterator[SpxMesh]:
    """Combined CPU mesh + forced-CPU execution context.

    Equivalent to::

        with force_cpu(), create_cpu_mesh() as mesh:
            yield mesh

    The commonest "debug my sharded code locally" entry point. Note
    that ``create_cpu_mesh`` does not accept ``mpmd_axis``, so the
    yielded mesh is always pure-SPMD.

    Yields:
        The CPU :class:`SpxMesh` created for the scope.

    Example:
        >>> with spx.sharding.cpu_context() as mesh:
        ...     @jax.jit
        ...     def step(x):
        ...         return x * 2
        ...     result = step(jnp.ones((4, 4)))
    """
    mesh = create_cpu_mesh()
    with force_cpu(), mesh:
        yield mesh
