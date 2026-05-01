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
"""Canonical sharding configuration and runtime resolution for EasyDeL.

EasyDeL's persistent tensor placement now lives on parameter metadata. Runtime
code should therefore depend on:

- :class:`AxisPolicy`: immutable semantic axis configuration stored on configs.
- :class:`RuntimeShardingResolver`: the only lowering helper from semantic or
  metadata sharding declarations to concrete JAX shardings.

``NamedSharding`` is the primary runtime type whenever a mesh is available.
``PartitionSpec`` remains as a compatibility/reporting form and for the few JAX
APIs that still require it explicitly.
"""

from __future__ import annotations

import copy
import dataclasses
import typing as tp
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import spectrax as spx
from jax.interpreters import pxla
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis, common_types
from spectrax.common_types import EMPTY, MODE_DECODE, MODE_TRAIN, NOT_GIVEN
from spectrax.core.stage_assignment import (
    PIPELINE_STAGE_METADATA_KEY,
    current_stage_assignment,
    metadata_stage_assignment,
    resolve_stage_rank,
)
from spectrax.runtime.types.mesh import MpMdMesh
from spectrax.sharding import current_axis_rules as spx_current_axis_rules
from spectrax.sharding import logical_axis_rules as spx_logical_axis_rules

_HAS_SPECTRAX_MESH_TYPES = True

MeshLike: tp.TypeAlias = spx.SpxMesh | Mesh | MpMdMesh
OptionalMesh: tp.TypeAlias = MeshLike | None
StageMesh: tp.TypeAlias = Mesh | None
AxisEntry = str | tuple[str, ...] | None
AxisEntries = tuple[AxisEntry, ...]
LogicalAxisRules = Sequence[tuple[str, str | None]] | Mapping[str, str | None]
_UNSUPPORTED_SIMPLE_RULE = object()

CANONICAL_MESH_AXIS_NAMES: tuple[str, ...] = ("pp", "dp", "fsdp", "ep", "tp", "sp")
_SIMPLE_SEMANTIC_AXES: tuple[str, ...] = (
    common_types.DATA_PARALLEL,
    common_types.FULLY_SHARDED_DATA_PARALLEL,
    common_types.TENSOR_PARALLEL,
    common_types.SEQUENCE_PARALLEL,
    common_types.EXPERT_PARALLEL,
    common_types.BATCH,
    common_types.LENGTH,
    common_types.QUERY_LENGTH,
    common_types.KV_LENGTH,
    common_types.HEAD,
    common_types.KV_HEAD,
    common_types.EMBED,
    common_types.MLP_INTERMEDIATE,
    common_types.VOCAB,
    common_types.EXPERT,
    common_types.EXPERT_GATE,
    common_types.HEAD_DIM,
    common_types.KV_HEAD_DIM,
    common_types.BIAS_HEAD_SEQ,
    common_types.BIAS_KV_SEQ,
)


def _coerce_partition_axis(value: AxisPolicy | PartitionAxis | dict[str, tp.Any] | None) -> PartitionAxis:
    """Coerce *value* into a :class:`PartitionAxis` instance.

    Args:
        value: An :class:`AxisPolicy`, :class:`PartitionAxis`, mapping of
            field overrides, or ``None`` for defaults.

    Returns:
        PartitionAxis: A defensively copied or freshly constructed
        partition-axis object.

    Raises:
        TypeError: If *value* is none of the accepted input types.
    """
    if isinstance(value, AxisPolicy):
        return value.to_partition_axis()
    if value is None:
        return PartitionAxis()
    if isinstance(value, PartitionAxis):
        return copy.deepcopy(value)
    if isinstance(value, dict):
        return PartitionAxis(**value)
    raise TypeError(f"Unsupported partition-axis value: {type(value).__name__}")


def coerce_axis_policy(value: AxisPolicy | PartitionAxis | dict[str, tp.Any] | None) -> AxisPolicy:
    """Normalize any supported axis-policy input to :class:`AxisPolicy`."""
    if isinstance(value, AxisPolicy):
        return value
    return AxisPolicy.from_partition_axis(value)


def _normalize_axis_entry(value: tp.Any) -> AxisEntry:
    """Coerce a single axis entry into the canonical ``AxisEntry`` form.

    Accepts strings, sequences of strings, ``None``, the sentinel ``EMPTY``,
    or the empty marker ``"_"``. Sequences are flattened one level so nested
    tuples collapse into a single tuple of mesh axis names.

    Args:
        value: Raw axis specification.

    Returns:
        AxisEntry: ``None`` for empty inputs, a single string for one axis,
        or a tuple of strings for compound axes.

    Raises:
        TypeError: If *value* is neither ``None``, a string, nor a sequence.
    """
    if value in (None, EMPTY, "_"):
        return None
    if isinstance(value, list | tuple):
        entries = tuple(_normalize_axis_entry(item) for item in value)
        compact = tuple(item for item in entries if item is not None)
        if not compact:
            return None
        flattened: list[str] = []
        for item in compact:
            if isinstance(item, tuple):
                flattened.extend(item)
            else:
                flattened.append(item)
        return tuple(flattened)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized == EMPTY:
            return None
        return normalized
    raise TypeError(f"Unsupported axis entry: {value!r}")


def _normalize_axes(axes: tp.Iterable[tp.Any]) -> AxisEntries:
    """Normalize an iterable of axis entries into a tuple of canonical entries.

    Args:
        axes: Iterable of raw axis specifications.

    Returns:
        AxisEntries: Tuple of normalized :func:`_normalize_axis_entry` results.
    """
    return tuple(_normalize_axis_entry(axis) for axis in axes)


def _normalize_logical_axis_rules(overrides: LogicalAxisRules | None) -> tuple[tuple[str, str | None], ...]:
    """Coerce logical axis-rule overrides into a sorted tuple of pairs.

    Args:
        overrides: Either a mapping or iterable of
            ``(logical_name, mesh_axis)`` pairs (mesh_axis may be ``None``).

    Returns:
        tuple: Tuple of ``(str, str | None)`` pairs with both keys and values
        coerced to strings (or ``None``).
    """
    if overrides is None:
        return ()
    items = overrides.items() if isinstance(overrides, Mapping) else overrides
    normalized: list[tuple[str, str | None]] = []
    for logical_name, mesh_axis in items:
        normalized.append((str(logical_name), None if mesh_axis is None else str(mesh_axis)))
    return tuple(normalized)


def _simple_rule_value(value: tp.Any) -> str | None | object:
    """Reduce an axis entry to a simple-rule value, or mark it unsupported.

    Args:
        value: Raw axis entry.

    Returns:
        str | None | object: ``None`` for empty entries, a single string for
        one mesh axis, or :data:`_UNSUPPORTED_SIMPLE_RULE` for compound
        (multi-axis) entries which cannot be expressed as a simple rule.
    """
    normalized = _normalize_axis_entry(value)
    if normalized is None:
        return None
    if isinstance(normalized, tuple):
        return _UNSUPPORTED_SIMPLE_RULE
    return normalized


def is_valid_mesh(mesh: OptionalMesh | tp.Any) -> bool:
    """Return whether *mesh* looks like a usable JAX/SpectraX mesh."""
    if mesh is None:
        return False
    if getattr(mesh, "empty", False):
        return False
    return getattr(mesh, "shape", None) is not None


def resolve_stage_mesh(
    mesh: OptionalMesh = None,
    *,
    arr: tp.Any = None,
    stage: int | tuple[int, int] | None = None,
    fallback_to_context: bool = True,
) -> StageMesh:
    """Resolve the mesh that should own work or buffers in the current MPMD context."""
    try:
        stage_mesh = spx.get_current_stage_mesh(mesh, arr=arr, stage=stage, raise_error=False)
        if is_valid_mesh(stage_mesh):
            return stage_mesh
    except TypeError:
        stage_mesh = spx.get_current_stage_mesh(mesh, arr=arr, raise_error=False)
        if is_valid_mesh(stage_mesh):
            return stage_mesh

    if fallback_to_context:
        try:
            stage_mesh = spx.get_current_stage_mesh(arr=arr, stage=stage, raise_error=False)
            if is_valid_mesh(stage_mesh):
                return stage_mesh
        except TypeError:
            stage_mesh = spx.get_current_stage_mesh(arr=arr, raise_error=False)
            if is_valid_mesh(stage_mesh):
                return stage_mesh

    try:
        jax_mesh = spx.to_jax_mesh(mesh)
        if is_valid_mesh(jax_mesh):
            return jax_mesh
    except Exception:
        pass

    return mesh if is_valid_mesh(mesh) else None


def resolve_stage_cache_mesh(mesh: OptionalMesh = None, *, arr: tp.Any = None) -> StageMesh:
    """Resolve the mesh that should own a cache buffer."""
    return resolve_stage_mesh(mesh, arr=arr)


def resolve_array_mesh(arr: tp.Any) -> StageMesh:
    """Resolve a JAX mesh from an array's named sharding, if present."""
    sharding = getattr(arr, "sharding", None)
    if isinstance(sharding, NamedSharding) and is_valid_mesh(sharding.mesh):
        return sharding.mesh
    return None


def mesh_matches(lhs: OptionalMesh | tp.Any, rhs: OptionalMesh | tp.Any) -> bool:
    """Return whether two mesh objects describe the same concrete device mesh."""
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return False
    try:
        if getattr(lhs, "axis_names", None) != getattr(rhs, "axis_names", None):
            return False
        if getattr(lhs, "devices", None).shape != getattr(rhs, "devices", None).shape:
            return False

        lhs_fingerprint = tuple(
            (
                getattr(device, "process_index", None),
                getattr(device, "id", None),
                getattr(device, "platform", None),
                getattr(device, "device_kind", None),
            )
            for device in lhs.devices.flat
        )
        rhs_fingerprint = tuple(
            (
                getattr(device, "process_index", None),
                getattr(device, "id", None),
                getattr(device, "platform", None),
                getattr(device, "device_kind", None),
            )
            for device in rhs.devices.flat
        )
        return lhs_fingerprint == rhs_fingerprint
    except Exception:
        return False


def pick_array_mesh(*arrays: tp.Any) -> StageMesh | jax.sharding.AbstractMesh:
    """Pick the first named-sharding mesh from arrays, then fall back to context."""
    for array in arrays:
        if array is None:
            continue
        mesh = resolve_array_mesh(array)
        if mesh is not None:
            return mesh
    return resolve_stage_mesh(spx.get_incontext_mesh(raise_error=False), fallback_to_context=False)


def partition_spec_for_mesh(array: tp.Any, mesh: OptionalMesh | tp.Any) -> PartitionSpec:
    """Return an array's PartitionSpec when it is already on ``mesh``."""
    if array is None:
        return PartitionSpec()
    sharding = getattr(array, "sharding", None)
    if isinstance(sharding, NamedSharding) and mesh_matches(sharding.mesh, mesh):
        return sharding.spec
    return PartitionSpec()


def normalize_axis_names(axis: tp.Any) -> tuple[str, ...]:
    """Normalize one axis, many axes, or no axis into concrete mesh axis names."""
    if axis is None or axis is EMPTY:
        return ()
    if isinstance(axis, (tuple, list)):
        return tuple(str(item) for item in axis if item and item is not EMPTY)
    return (str(axis),)


def axis_index(axis: tp.Any) -> jax.Array:
    """Return a row-major linearized index for one or more mesh axes."""
    axis_names = normalize_axis_names(axis)
    if not axis_names:
        return jnp.int32(0)
    idx = jax.lax.axis_index(axis_names[0]).astype(jnp.int32)
    for axis_name in axis_names[1:]:
        axis_size = jax.lax.psum(jnp.int32(1), axis_name)
        idx = idx * axis_size + jax.lax.axis_index(axis_name).astype(jnp.int32)
    return idx


def mesh_axis_size(mesh: OptionalMesh | tp.Any, axis_name: tp.Any) -> int:
    """Return product of mesh sizes for one axis, many axes, or no axis."""
    if axis_name is None or axis_name is EMPTY:
        return 1
    if isinstance(axis_name, (list, tuple)):
        product = 1
        for name in axis_name:
            if name is not None and name is not EMPTY:
                product *= mesh_axis_size(mesh, str(name))
        return max(1, int(product))

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


def _mesh_axis_size(mesh: MeshLike, axis_name: str) -> int:
    """Internal alias for :func:`mesh_axis_size`.

    Args:
        mesh: A JAX/SpectraX mesh-like object.
        axis_name: Axis name to look up.

    Returns:
        int: The mesh size along ``axis_name``, or ``1`` if missing.
    """
    return mesh_axis_size(mesh, axis_name)


def _mesh_partition_product(mesh: MeshLike, axis_spec: AxisEntry) -> int:
    """Internal alias for :func:`mesh_partition_product`.

    Args:
        mesh: A JAX/SpectraX mesh-like object.
        axis_spec: Single axis name, tuple of axis names, or ``None``.

    Returns:
        int: Total shard multiplicity implied by ``axis_spec``.
    """
    return mesh_partition_product(mesh, axis_spec)


def mesh_partition_product(mesh: MeshLike, axis_spec: AxisEntry) -> int:
    """Return shard multiplicity implied by a PartitionSpec entry."""
    return mesh_axis_size(mesh, axis_spec)


def coerce_partition_spec(spec: tp.Any) -> PartitionSpec | None:
    """Coerce a tuple/list PartitionSpec-like value into ``PartitionSpec``."""
    if isinstance(spec, PartitionSpec):
        return spec
    if isinstance(spec, (tuple, list)):
        try:
            return PartitionSpec(*tuple(spec))
        except Exception:
            return None
    return None


def _resolve_named_sharding_mesh(mesh: MeshLike) -> tuple[Mesh, MpMdMesh | None]:
    """Split a SpectraX mesh wrapper into ``(jax_mesh, optional_mpmd_mesh)``.

    Args:
        mesh: A ``jax.sharding.Mesh``, :class:`spx.SpxMesh`, or
            :class:`MpMdMesh`.

    Returns:
        tuple: ``(base JAX mesh, mpmd mesh or None)``. For non-SpectraX
        meshes ``(mesh, None)`` is returned.
    """
    if _HAS_SPECTRAX_MESH_TYPES and isinstance(mesh, spx.SpxMesh):
        return mesh.jax_mesh, mesh.mpmd_mesh
    if _HAS_SPECTRAX_MESH_TYPES and isinstance(mesh, MpMdMesh):
        return mesh.jax_mesh, mesh
    return mesh, None


def _stage_local_mesh(
    mesh: MeshLike,
    metadata: dict[str, tp.Any] | None,
) -> Mesh:
    """Return the stage-local submesh implied by parameter metadata.

    For MPMD meshes, parameters that record a ``stage_assignment`` should
    only live on the submesh owning that stage. Non-MPMD meshes (or
    parameters without a stage assignment) get the full mesh.

    Args:
        mesh: The full mesh (possibly an MPMD wrapper).
        metadata: Parameter metadata that may carry a ``stage_assignment``.

    Returns:
        Mesh: The owning JAX mesh (full mesh or stage submesh).
    """
    assignment = metadata_stage_assignment(metadata)
    if assignment is None:
        if _HAS_SPECTRAX_MESH_TYPES and isinstance(mesh, spx.SpxMesh):
            return mesh.jax_mesh
        if _HAS_SPECTRAX_MESH_TYPES and isinstance(mesh, MpMdMesh):
            return mesh.jax_mesh
        return mesh

    base_mesh, mpmd_mesh = _resolve_named_sharding_mesh(mesh)
    if mpmd_mesh is None:
        return base_mesh
    owner = resolve_stage_rank(assignment, mpmd_mesh.mpmd_dim)
    if owner is None:
        return base_mesh
    return mpmd_mesh.submesh(owner)


def is_mpmd_mesh(mesh: OptionalMesh) -> bool:
    """Return whether a mesh is a SpectraX MPMD mesh wrapper."""
    if mesh is None or not _HAS_SPECTRAX_MESH_TYPES:
        return False
    if isinstance(mesh, spx.SpxMesh):
        return bool(mesh.is_mpmd)
    return isinstance(mesh, MpMdMesh)


def _is_mpmd_mesh(mesh: OptionalMesh) -> bool:
    """Internal alias for :func:`is_mpmd_mesh`.

    Args:
        mesh: A mesh-like object or ``None``.

    Returns:
        bool: ``True`` if *mesh* is a SpectraX MPMD wrapper.
    """
    return is_mpmd_mesh(mesh)


def sanitize_partition_spec_for_shape(
    spec: tp.Any,
    shape: tuple[int, ...],
    mesh: MeshLike,
) -> tp.Any:
    """Drop non-divisible sharding axes for a concrete tensor shape."""
    pspec = coerce_partition_spec(spec)
    if pspec is None:
        return spec

    axes = list(tuple(pspec))
    changed = False

    if len(axes) > len(shape):
        axes = axes[: len(shape)]
        changed = True

    for dim_index, axis_spec in enumerate(axes):
        if axis_spec is None:
            continue
        shard_factor = _mesh_partition_product(mesh, axis_spec)
        if shard_factor > 1 and int(shape[dim_index]) % shard_factor != 0:
            axes[dim_index] = None
            changed = True

    if not changed:
        return pspec
    return PartitionSpec(*axes)


def sanitize_sharding_axes_for_shape(
    *,
    mesh: MeshLike,
    runtime_sharding_resolver: tp.Any,
    axes: tp.Sequence[object | None],
    mode: tp.Any,
    shape: tuple[int, ...],
) -> tuple[object | None, ...]:
    """Drop logical axes whose resolved mesh sharding cannot divide ``shape``."""
    spec = runtime_sharding_resolver.resolve(axes=axes, mode=mode, shape=shape)
    safe: list[object | None] = []
    for logical_axis, axis_spec, dim in zip(axes, spec, shape, strict=False):
        if logical_axis is None or axis_spec is None:
            safe.append(logical_axis)
            continue
        shard_factor = mesh_partition_product(mesh, axis_spec)
        safe.append(None if int(dim) % shard_factor != 0 else logical_axis)
    return tuple(safe)


def _flatten_mapping(tree: tp.Any, prefix: tuple[tp.Any, ...] = ()) -> dict[tuple[tp.Any, ...], tp.Any]:
    """Flatten a nested-dict pytree into a path-keyed dict.

    Args:
        tree: A nested mapping or leaf value.
        prefix: Internal accumulator of the current path; callers should
            leave the default.

    Returns:
        dict: Mapping from path tuples (one element per nesting level) to
        leaf values.
    """
    if not isinstance(tree, dict):
        return {prefix: tree}
    flat: dict[tuple[tp.Any, ...], tp.Any] = {}
    for key, value in tree.items():
        flat.update(_flatten_mapping(value, (*prefix, key)))
    return flat


def _unflatten_mapping(flat: Mapping[tuple[tp.Any, ...], tp.Any]) -> dict[tp.Any, tp.Any]:
    """Inverse of :func:`_flatten_mapping`: rebuild a nested dict from path keys.

    Args:
        flat: Mapping from path tuples to leaf values.

    Returns:
        dict: A nested-dict reconstruction.
    """
    root: dict[tp.Any, tp.Any] = {}
    for path, value in flat.items():
        cursor = root
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        if path:
            cursor[path[-1]] = value
    return root


def sanitize_partition_specs_for_shape_tree(
    partition_specs: tp.Any,
    shape_tree: tp.Any,
    mesh: MeshLike,
) -> tuple[tp.Any, int]:
    """Sanitize a PartitionSpec tree against concrete tensor shapes."""
    adjusted = {"count": 0}

    def _sanitize(spec: tp.Any, shape_obj: tp.Any) -> tp.Any:
        """Sanitize a single ``(PartitionSpec, shape)`` pair.

        Args:
            spec: Candidate :class:`PartitionSpec` (other types pass through).
            shape_obj: Object exposing a ``.shape`` attribute.

        Returns:
            The safe partition spec, increasing ``adjusted["count"]`` from the
            enclosing scope when changes are made.
        """
        if not isinstance(spec, PartitionSpec) or not hasattr(shape_obj, "shape"):
            return spec
        safe_spec = sanitize_partition_spec_for_shape(
            spec=spec,
            shape=tuple(shape_obj.shape),
            mesh=mesh,
        )
        if safe_spec != spec:
            adjusted["count"] += 1
        return safe_spec

    try:
        sanitized = jax.tree_util.tree_map(_sanitize, partition_specs, shape_tree)
        return sanitized, adjusted["count"]
    except Exception:
        flat_specs = _flatten_mapping(partition_specs)
        flat_shapes = _flatten_mapping(shape_tree)
        adjusted_count = 0
        for key, spec in flat_specs.items():
            if not isinstance(spec, PartitionSpec):
                continue
            shape_obj = flat_shapes.get(key)
            shape = tuple(getattr(shape_obj, "shape", ()))
            if not shape:
                continue
            safe_spec = sanitize_partition_spec_for_shape(spec=spec, shape=shape, mesh=mesh)
            if safe_spec != spec:
                flat_specs[key] = safe_spec
                adjusted_count += 1
        if adjusted_count == 0:
            return partition_specs, adjusted_count
        return _unflatten_mapping(flat_specs), adjusted_count


def pick_mesh(*, partition_manager: tp.Any | None = None, mesh: OptionalMesh = None) -> StageMesh:
    """Pick the best mesh from an explicit mesh, resolver-like object, or context."""
    candidate = resolve_stage_mesh(mesh, fallback_to_context=False)
    if candidate is not None:
        return candidate

    if partition_manager is not None:
        for attr_name in ("mesh", "_mesh", "device_mesh"):
            candidate = resolve_stage_mesh(getattr(partition_manager, attr_name, None), fallback_to_context=False)
            if candidate is not None:
                return candidate

    candidate = resolve_stage_mesh(spx.get_incontext_mesh(raise_error=False), fallback_to_context=False)
    if candidate is not None:
        return candidate

    return None


def resolve_safe_sharding(
    *,
    axes: tp.Any,
    shape: tuple[int, ...],
    runtime_sharding_resolver: tp.Any | None = None,
    axis_policy: tp.Any | None = None,
    partition_manager: tp.Any | None = None,
    mesh: OptionalMesh = None,
    mode: str = MODE_TRAIN,
) -> tp.Any:
    """Resolve sharding axes through ``RuntimeShardingResolver`` with safe fallback."""
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
        return sanitize_partition_spec_for_shape(spec=resolved, shape=shape, mesh=mesh_obj)


@dataclasses.dataclass(frozen=True, slots=True)
class TensorLayout:
    """Sharding metadata that supports compound per-dimension axis entries.

    EasyDeL keeps this semantic adapter so older layout declarations such as
    ``RowWise`` and ``ColumnWise`` can be lowered to native ``spectrax.Sharding``
    metadata, including compound placements such as ``(fsdp, sp)``.
    """

    axes: AxisEntries
    mode: str | int = MODE_TRAIN

    def __post_init__(self) -> None:
        """Normalize ``axes`` after dataclass construction.

        Returns:
            None.
        """
        object.__setattr__(self, "axes", _normalize_axes(self.axes))

    @classmethod
    def from_any(cls, value: tp.Any) -> TensorLayout | None:
        """Best-effort coercion from a variety of layout-like inputs.

        Accepts ``TensorLayout``, mappings with ``axes``/``mode`` keys,
        named-tuple-like layouts, plain sequences (treated as ``axes``), or
        objects that simply expose ``axes`` and ``mode`` attributes.

        Args:
            value: Candidate layout description.

        Returns:
            TensorLayout | None: A :class:`TensorLayout` instance, or
            ``None`` if *value* could not be interpreted.
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                axes=value.get("axes", ()),
                mode=value.get("mode", MODE_TRAIN),
            )
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return cls(axes=value.axes, mode=value.mode)
        if isinstance(value, type) and issubclass(value, tuple) and hasattr(value, "_fields"):
            return cls(axes=value.axes, mode=value.mode)
        if isinstance(value, list | tuple):
            return cls(axes=value)
        if hasattr(value, "axes") and hasattr(value, "mode"):
            return cls(axes=value.axes, mode=value.mode)
        return None

    def to_dict(self) -> dict[str, tp.Any]:
        """Return a serializable dict view of this layout.

        Returns:
            dict: Keys ``axes`` and ``mode`` mirroring the dataclass fields.
        """
        return {"axes": self.axes, "mode": self.mode}

    def as_spectrax_sharding(self) -> spx.Sharding:
        """Convert this layout to a native :class:`spx.Sharding`.

        Returns:
            spx.Sharding: Sharding metadata with the same axis names.
        """
        return spx.Sharding(axis_names=self.axes)


def sharding_for_layout(layout: TensorLayout | tp.Any) -> spx.Sharding | None:
    """Return native SpecTrax sharding for train-time parameter layouts."""
    tensor_layout = TensorLayout.from_any(layout)
    if tensor_layout is None:
        return None
    if tensor_layout.mode != MODE_TRAIN:
        return None
    return tensor_layout.as_spectrax_sharding()


def metadata_for_layout(
    layout: TensorLayout | tp.Any,
    *,
    pipeline_stage: tuple[int, int] | None = None,
) -> dict[str, tp.Any]:
    """Build variable metadata for a persistent parameter layout."""
    stage_assignment = pipeline_stage if pipeline_stage is not None else current_stage_assignment()
    metadata: dict[str, tp.Any] = {}
    if stage_assignment is not None:
        metadata[PIPELINE_STAGE_METADATA_KEY] = stage_assignment
    tensor_layout = TensorLayout.from_any(layout)
    if tensor_layout is None:
        return metadata
    if tensor_layout.mode == MODE_TRAIN:
        metadata["sharding"] = tensor_layout.as_spectrax_sharding()
        return metadata
    metadata["tensor_layout"] = tensor_layout
    return metadata


def _sharding_axis_names_from_metadata(metadata: Mapping[str, tp.Any]) -> tp.Any | None:
    """Extract per-dimension axis names from a metadata mapping.

    Looks for a SpectraX ``sharding`` entry first (dict or
    :class:`spx.Sharding`) and falls back to a raw ``axis_names`` field.

    Args:
        metadata: Variable metadata dict.

    Returns:
        Any | None: Tuple of axis names per dimension, or ``None`` when none
        are recorded.
    """
    sharding = metadata.get("sharding")
    if isinstance(sharding, dict):
        sharding = spx.Sharding(
            axis_names=tuple(sharding["axis_names"]) if sharding.get("axis_names") is not None else None,
            mesh_axes=tuple(sharding["mesh_axes"]) if sharding.get("mesh_axes") is not None else None,
        )
    if isinstance(sharding, spx.Sharding) and sharding.axis_names is not None:
        return sharding.axis_names
    return metadata.get("axis_names")


def _metadata_has_compound_axis_names(metadata: Mapping[str, tp.Any]) -> bool:
    """Return whether *metadata* declares any compound (multi-axis) sharding.

    Args:
        metadata: Variable metadata dict.

    Returns:
        bool: ``True`` if any per-dimension entry is a list or tuple of mesh
        axis names (i.e. the dim is sharded across multiple mesh axes).
    """
    axis_names = _sharding_axis_names_from_metadata(metadata)
    if axis_names is None:
        return False
    try:
        return any(isinstance(axis, list | tuple) for axis in axis_names if axis is not None)
    except TypeError:
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class AxisPolicy:
    """Immutable wrapper around a :class:`PartitionAxis` that exposes EasyDeL's logical axis vocabulary.

    EasyDeL describes parameter and activation sharding using *logical* axis
    names (``batch``, ``hidden``, ``head``, ``query_sequence``, ...) instead of
    the concrete mesh axis names (``dp``, ``fsdp``, ``tp``, ``sp``, ``ep``,
    ``pp``). The mapping between logical names and mesh names lives on the
    underlying :class:`spectrax.PartitionAxis` and depends on the resolution
    *mode* (e.g. ``"train"`` vs ``"decode"``, where during single-token decode
    several sequence-bound axes collapse to replicated).

    ``AxisPolicy`` packages that ``PartitionAxis`` together with three
    facilities used throughout the rest of the codebase:

    1. **Coercion** — :meth:`from_partition_axis`, :meth:`from_dict`, and
       :meth:`from_any` accept the various inputs (existing
       :class:`AxisPolicy`, raw :class:`PartitionAxis`, dict of overrides,
       ``None``) and return a normalized policy; this is the single entry
       point for accepting user-provided axis configuration.
    2. **Resolution** — :meth:`resolve_axis` and :meth:`resolve_spec` dispatch
       a sequence of logical names plus a mode to the wrapped partition axis
       and return mesh-axis entries / a :class:`PartitionSpec`. Compound
       entries such as ``("fsdp", "sp")`` are preserved.
    3. **Logical axis rules** — :meth:`logical_axis_rule_pairs` produces the
       ``(logical_name, mesh_axis_or_None)`` rule list consumed by
       ``spectrax.logical_axis_rules`` for the simple (single-axis) cases.
       Compound placements are intentionally excluded since spectrax's rule
       system only supports one mesh axis per logical name; those flow
       through :class:`TensorLayout` + :class:`RuntimeShardingResolver`
       instead.

    Attributes:
        partition_axis (PartitionAxis): The wrapped logical-to-mesh-axis
            mapping. Coerced in ``__post_init__`` so dict-style or ``None``
            inputs work transparently.
    """

    partition_axis: PartitionAxis = dataclasses.field(default_factory=PartitionAxis, repr=False)

    def __post_init__(self) -> None:
        """Coerce the held ``partition_axis`` to a :class:`PartitionAxis`.

        Returns:
            None.
        """
        object.__setattr__(self, "partition_axis", _coerce_partition_axis(self.partition_axis))

    def __getattr__(self, name: str) -> tp.Any:
        """Forward attribute lookups to the underlying ``PartitionAxis``.

        Args:
            name: Attribute name.

        Returns:
            Any: The matching attribute on ``self.partition_axis``.

        Raises:
            AttributeError: If the underlying partition axis has no such
                attribute.
        """
        return getattr(self.partition_axis, name)

    @classmethod
    def from_partition_axis(cls, value: PartitionAxis | dict[str, tp.Any] | None) -> AxisPolicy:
        """Build an ``AxisPolicy`` from a partition-axis-like value.

        Args:
            value: A :class:`PartitionAxis`, dict of overrides, or ``None``.

        Returns:
            AxisPolicy: A new policy wrapping the coerced partition axis.
        """
        return cls(partition_axis=_coerce_partition_axis(value))

    @classmethod
    def from_dict(cls, value: dict[str, tp.Any]) -> AxisPolicy:
        """Build an ``AxisPolicy`` from a dict of partition-axis overrides.

        Args:
            value: Dict of :class:`PartitionAxis` field overrides.

        Returns:
            AxisPolicy: A new policy.
        """
        return cls.from_partition_axis(value)

    @classmethod
    def from_any(cls, value: AxisPolicy | PartitionAxis | dict[str, tp.Any] | None) -> AxisPolicy:
        """Coerce any supported input into an :class:`AxisPolicy`.

        Args:
            value: An ``AxisPolicy``, ``PartitionAxis``, dict, or ``None``.

        Returns:
            AxisPolicy: A normalized policy.
        """
        return coerce_axis_policy(value)

    def to_partition_axis(self) -> PartitionAxis:
        """Return a deep copy of the underlying ``PartitionAxis``.

        Returns:
            PartitionAxis: An independently mutable copy.
        """
        return copy.deepcopy(self.partition_axis)

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize the underlying partition axis to a dict.

        Returns:
            dict: Mapping from each ``PartitionAxis`` field name to a
            deep-copied value.
        """
        return {
            field.name: copy.deepcopy(getattr(self.partition_axis, field.name))
            for field in dataclasses.fields(self.partition_axis)
        }

    def resolve_axis(
        self,
        axes: tp.Sequence[str | None],
        mode: str,
    ) -> list[tp.Any]:
        """Delegate axis resolution to the held ``PartitionAxis``.

        Args:
            axes: Sequence of logical axis names (``None`` for replicated).
            mode: Resolution mode (e.g. ``"train"`` or ``"infer"``).

        Returns:
            list: Resolved axis entries from
            :meth:`PartitionAxis.resolve_axis`.
        """
        return self.partition_axis.resolve_axis(axes=axes, mode=mode)

    def resolve_spec(
        self,
        axes: tp.Sequence[str | None],
        mode: str,
    ) -> PartitionSpec:
        """Resolve a logical-axis sequence into a :class:`PartitionSpec`.

        Args:
            axes: Sequence of logical axis names.
            mode: Resolution mode.

        Returns:
            PartitionSpec: The resolved partition spec.
        """
        return self.partition_axis.resolve_spec(axes=axes, mode=mode)

    def logical_axis_rule_pairs(
        self,
        *,
        mode: str = MODE_TRAIN,
        overrides: LogicalAxisRules | None = None,
    ) -> tuple[tuple[str, str | None], ...]:
        """Return spectrax logical-axis rules for the resolvable simple aliases.

        Compound placements such as ``("fsdp", "sp")`` intentionally stay out of
        this rule set because ``spectrax.logical_axis_rules`` only supports
        ``logical -> single mesh axis`` mappings. Those richer cases continue to
        flow through ``TensorLayout`` + ``RuntimeShardingResolver``.
        """
        rules: dict[str, str | None] = {axis_name: axis_name for axis_name in CANONICAL_MESH_AXIS_NAMES}
        for semantic_axis in _SIMPLE_SEMANTIC_AXES:
            try:
                resolved = self.partition_axis.resolve_axis([semantic_axis], mode=mode)[0]
            except Exception:
                continue
            simple = _simple_rule_value(resolved)
            if simple is _UNSUPPORTED_SIMPLE_RULE:
                continue
            rules[str(semantic_axis)] = simple
        rules.update(dict(_normalize_logical_axis_rules(overrides)))
        return tuple(rules.items())


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeShardingResolver:
    """Stateless resolver that lowers semantic sharding declarations to concrete JAX shardings.

    A ``RuntimeShardingResolver`` pairs an :class:`AxisPolicy` (the logical
    axis vocabulary) with an optional active mesh, and exposes the helpers
    that EasyDeL modules and trainers use to translate between *layout*
    descriptions (``TensorLayout``, :class:`spx.Sharding`, plain tuples of
    logical names, raw :class:`PartitionSpec`, parameter metadata dicts) and
    the underlying JAX :class:`NamedSharding` / :class:`PartitionSpec` types
    that ``jax.jit`` and ``jax.lax.with_sharding_constraint`` consume.

    Notable behaviours:

    - Resolution is mode-aware. ``mode`` may be ``MODE_TRAIN``, ``MODE_DECODE``,
      ``NOT_GIVEN`` (defer to the layout's own mode), or an ``int`` shape
      index — in the integer case a dimension equal to ``1`` selects
      ``MODE_DECODE`` so single-token decode automatically collapses sequence
      axes.
    - Resolved specs are *sanitized* against a concrete shape: any axis whose
      mesh size does not divide the matching tensor dimension is dropped via
      :func:`sanitize_partition_spec_for_shape`, so callers can keep one
      logical layout per parameter and let runtime tensor shapes decide
      which axes are actually shardable.
    - Stage-local meshes are honored. When parameter metadata records a
      ``stage_assignment``, the named-sharding helpers route through the
      stage-local submesh instead of the global one, keeping pipeline
      parallel weights bound to their owning stage.

    The class is a frozen dataclass so multiple resolver instances bound to
    different meshes can co-exist (use :meth:`with_mesh` to swap meshes
    without replacing the policy).

    Attributes:
        axis_policy (AxisPolicy): The :class:`AxisPolicy` providing logical
            axis names and the underlying :class:`PartitionAxis`.
        mesh (OptionalMesh): Active mesh used as the default when individual
            method calls don't supply ``mesh=...``. ``None`` lets calls fall
            back to the ambient JAX context (or raise when a mesh is
            required).
    """

    axis_policy: AxisPolicy
    mesh: OptionalMesh = None

    def with_mesh(self, mesh: OptionalMesh) -> RuntimeShardingResolver:
        """Return a copy of this resolver bound to a different mesh.

        Args:
            mesh: New mesh to use, or ``None`` for context-driven resolution.

        Returns:
            RuntimeShardingResolver: A frozen copy with ``mesh`` replaced.
        """
        return RuntimeShardingResolver(axis_policy=self.axis_policy, mesh=mesh)

    @property
    def paxis(self) -> PartitionAxis:
        """Return a deep copy of the underlying :class:`PartitionAxis`.

        Returns:
            PartitionAxis: An independently mutable copy of the policy's
            partition axis.
        """
        return self.axis_policy.to_partition_axis()

    def _resolve_mode(
        self,
        *,
        mode: str | int | object,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        layout_mode: str | int = MODE_TRAIN,
    ) -> str:
        """Resolve a dynamic mode token into ``MODE_TRAIN`` or ``MODE_DECODE``.

        Integer modes are treated as a shape dimension index: when that
        dimension equals ``1`` the call is in decode mode.

        Args:
            mode: ``MODE_TRAIN``, ``MODE_DECODE``, an int, or ``NOT_GIVEN``.
            shape: Concrete tensor shape (required when ``mode`` is an int).
            layout_mode: Fallback mode used when ``mode`` is ``NOT_GIVEN``.

        Returns:
            str: ``MODE_TRAIN`` or ``MODE_DECODE``.

        Raises:
            ValueError: If ``mode`` is an int and ``shape`` was not provided.
        """
        selected_mode = layout_mode if mode is NOT_GIVEN else mode
        if isinstance(selected_mode, int):
            if shape is NOT_GIVEN:
                raise ValueError("shape is required when resolving a dynamic sharding mode.")
            return MODE_DECODE if shape[selected_mode] == 1 else MODE_TRAIN
        return selected_mode

    def logical_axis_rule_pairs(
        self,
        *,
        mode: str | int | object = MODE_TRAIN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        overrides: LogicalAxisRules | None = None,
    ) -> tuple[tuple[str, str | None], ...]:
        """Materialize logical-axis rule pairs for the resolved mode.

        Args:
            mode: Resolution mode (``MODE_TRAIN``, ``MODE_DECODE``, an int
                shape index, or ``NOT_GIVEN``).
            shape: Concrete shape required when ``mode`` is an int.
            overrides: Optional extra logical-to-mesh axis rules.

        Returns:
            tuple: Tuple of ``(logical_name, mesh_axis_or_None)`` pairs.
        """
        resolved_mode = self._resolve_mode(mode=mode, shape=shape, layout_mode=MODE_TRAIN)
        return self.axis_policy.logical_axis_rule_pairs(mode=resolved_mode, overrides=overrides)

    @contextmanager
    def logical_axis_rules(
        self,
        *,
        mode: str | int | object = MODE_TRAIN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        overrides: LogicalAxisRules | None = None,
    ) -> Iterator[Mapping[str, str | None]]:
        """Open a spectrax logical-axis-rules scope derived from ``axis_policy``."""
        rules = self.logical_axis_rule_pairs(mode=mode, shape=shape, overrides=overrides)
        with spx_logical_axis_rules(rules):
            yield spx_current_axis_rules()

    def _sanitize_spec(
        self,
        spec: PartitionSpec,
        *,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> PartitionSpec:
        """Drop sharding axes that don't divide *shape* on the active mesh.

        Args:
            spec: Partition spec to sanitize.
            shape: Concrete tensor shape (skipped when ``NOT_GIVEN``).
            mesh: Mesh override (defaults to ``self.mesh``).

        Returns:
            PartitionSpec: The (possibly relaxed) safe partition spec.
        """
        active_mesh = self.mesh if mesh is None else mesh
        if active_mesh is None or shape is NOT_GIVEN:
            return spec
        base_mesh, _ = _resolve_named_sharding_mesh(active_mesh)
        return sanitize_partition_spec_for_shape(spec=spec, shape=shape, mesh=base_mesh)

    def _resolve_axis_name_entry(self, axis: tp.Any, mode: str) -> AxisEntry:
        """Translate a logical axis (or compound axis tuple) to mesh axes.

        Args:
            axis: Logical axis name, tuple of names, or ``None``.
            mode: Resolution mode (``MODE_TRAIN`` / ``MODE_DECODE``).

        Returns:
            AxisEntry: ``None`` (replicated), a single mesh axis name, or a
            tuple of mesh axis names when the logical entry is compound.
        """
        normalized = _normalize_axis_entry(axis)
        if normalized is None:
            return None

        logical_rules = dict(self.logical_axis_rule_pairs(mode=mode))

        def _resolve_one(name: str) -> str | None:
            """Look up a logical axis name in the resolved rule dict.

            Args:
                name: Logical axis name.

            Returns:
                str | None: The mapped mesh axis name, or ``None`` to
                replicate (matching SpectraX's default for unknown rules).
            """
            resolved = logical_rules.get(name, None)
            if resolved is not None or name in logical_rules:
                return resolved
            # Match SpecTrax's built-in behavior for unknown logical names:
            # missing rules replicate instead of surfacing a raw invalid mesh axis.
            return None

        if isinstance(normalized, tuple):
            resolved_axes = tuple(axis for item in normalized if (axis := _resolve_one(item)) is not None)
            return resolved_axes or None
        return _resolve_one(normalized)

    def _partition_spec_for_axis_names(self, axes: tp.Iterable[tp.Any], mode: str) -> PartitionSpec:
        """Build a :class:`PartitionSpec` from logical axis names.

        Args:
            axes: Iterable of logical axis names (one per tensor dim).
            mode: Resolution mode.

        Returns:
            PartitionSpec: The composed partition spec.
        """
        return PartitionSpec(*(self._resolve_axis_name_entry(axis, mode) for axis in axes))

    def partition_spec_for_layout(
        self,
        layout: tp.Any,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> PartitionSpec:
        """Alias of :meth:`resolve_layout` returning a sanitized spec.

        Args:
            layout: Layout-like object accepted by :meth:`resolve_layout`.
            mode: Resolution mode.
            shape: Concrete tensor shape used for safety checks.
            mesh: Mesh override (defaults to ``self.mesh``).

        Returns:
            PartitionSpec: The resolved partition spec.
        """
        return self.resolve_layout(layout=layout, mode=mode, shape=shape, mesh=mesh)

    def resolve_layout(
        self,
        layout: tp.Any,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> PartitionSpec:
        """Resolve a layout-like object to a sanitized :class:`PartitionSpec`.

        Accepts existing :class:`PartitionSpec`s (returned as-is after
        sanitization), :class:`TensorLayout`, :class:`spx.Sharding`, or plain
        sequences of logical axis names.

        Args:
            layout: Layout description to resolve.
            mode: Resolution mode (``NOT_GIVEN`` falls back to the layout's
                own mode or ``MODE_TRAIN``).
            shape: Concrete tensor shape used for safety checks.
            mesh: Mesh override.

        Returns:
            PartitionSpec: The resolved partition spec.

        Raises:
            TypeError: If *layout* is not a supported type.
        """
        if isinstance(layout, PartitionSpec):
            return self._sanitize_spec(layout, shape=shape, mesh=mesh)

        tensor_layout = TensorLayout.from_any(layout)
        if tensor_layout is not None:
            resolved_mode = self._resolve_mode(mode=mode, shape=shape, layout_mode=tensor_layout.mode)
            spec = self._partition_spec_for_axis_names(tensor_layout.axes, resolved_mode)
            return self._sanitize_spec(spec, shape=shape, mesh=mesh)

        if isinstance(layout, spx.Sharding):
            if layout.mesh_axes is not None:
                spec = PartitionSpec(*layout.mesh_axes)
                return self._sanitize_spec(spec, shape=shape, mesh=mesh)
            if layout.axis_names is not None:
                resolved_mode = self._resolve_mode(mode=mode, shape=shape, layout_mode=MODE_TRAIN)
                spec = self._partition_spec_for_axis_names(layout.axis_names, resolved_mode)
                return self._sanitize_spec(spec, shape=shape, mesh=mesh)
            return PartitionSpec()

        if isinstance(layout, list | tuple):
            resolved_mode = self._resolve_mode(mode=mode, shape=shape, layout_mode=MODE_TRAIN)
            spec = self._partition_spec_for_axis_names(layout, resolved_mode)
            return self._sanitize_spec(spec, shape=shape, mesh=mesh)

        raise TypeError(f"Unsupported sharding layout: {layout!r}")

    def named_sharding_for_layout(
        self,
        layout: tp.Any,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> NamedSharding:
        """Build a :class:`NamedSharding` for a layout, honoring stage-local meshes.

        Args:
            layout: Layout description (see :meth:`resolve_layout`).
            mode: Resolution mode.
            shape: Concrete tensor shape.
            mesh: Mesh override.
            metadata: Optional variable metadata; when it carries a
                ``stage_assignment`` the resulting NamedSharding is attached
                to the stage-local submesh.

        Returns:
            NamedSharding: A JAX named sharding referencing the resolved mesh.

        Raises:
            ValueError: If no mesh is available.
        """
        active_mesh = self.mesh if mesh is None else mesh
        if active_mesh is None:
            raise ValueError("A mesh is required to build NamedSharding.")
        stage_mesh = _stage_local_mesh(active_mesh, metadata)
        spec = self.partition_spec_for_layout(layout=layout, mode=mode, shape=shape, mesh=stage_mesh)
        return self.named_sharding_for_spec(spec, shape=shape, mesh=stage_mesh)

    def resolve_metadata(
        self,
        metadata: dict[str, tp.Any] | None,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> PartitionSpec | None:
        """Resolve a parameter's metadata into a :class:`PartitionSpec`.

        Looks up keys in priority order: ``tensor_layout``, ``sharding``,
        ``axis_names``.

        Args:
            metadata: Parameter metadata dict, or ``None``.
            mode: Resolution mode.
            shape: Concrete tensor shape.
            mesh: Mesh override.

        Returns:
            PartitionSpec | None: The resolved spec, or ``None`` if the
            metadata is absent or contains no sharding info.
        """
        if not metadata:
            return None

        tensor_layout = metadata.get("tensor_layout")
        if tensor_layout is not None:
            return self.resolve_layout(tensor_layout, mode=mode, shape=shape, mesh=mesh)

        sharding = metadata.get("sharding")
        if isinstance(sharding, dict):
            sharding = spx.Sharding(
                axis_names=tuple(sharding["axis_names"]) if sharding.get("axis_names") is not None else None,
                mesh_axes=tuple(sharding["mesh_axes"]) if sharding.get("mesh_axes") is not None else None,
            )
        if sharding is not None:
            return self.resolve_layout(sharding, mode=mode, shape=shape, mesh=mesh)

        axis_names = metadata.get("axis_names")
        if axis_names is not None:
            return self.resolve_layout(tuple(axis_names), mode=mode, shape=shape, mesh=mesh)

        return None

    def partition_spec_for_metadata(
        self,
        metadata: dict[str, tp.Any] | None,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> PartitionSpec | None:
        """Alias of :meth:`resolve_metadata`.

        Args:
            metadata: Parameter metadata dict.
            mode: Resolution mode.
            shape: Concrete tensor shape.
            mesh: Mesh override.

        Returns:
            PartitionSpec | None: The resolved partition spec, or ``None``.
        """
        return self.resolve_metadata(metadata=metadata, mode=mode, shape=shape, mesh=mesh)

    def named_sharding_for_metadata(
        self,
        metadata: dict[str, tp.Any] | None,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> NamedSharding | None:
        """Build a :class:`NamedSharding` from parameter metadata.

        Args:
            metadata: Parameter metadata dict.
            mode: Resolution mode.
            shape: Concrete tensor shape.
            mesh: Mesh override (defaults to ``self.mesh``).

        Returns:
            NamedSharding | None: ``None`` when the metadata is missing or
            no mesh is available; otherwise a JAX named sharding routed
            through the appropriate stage-local mesh.
        """
        if not metadata:
            return None
        active_mesh = self.mesh if mesh is None else mesh
        if active_mesh is None:
            return None
        if metadata.get("tensor_layout") is not None:
            return self.named_sharding_for_layout(
                metadata["tensor_layout"],
                mode=mode,
                shape=shape,
                mesh=active_mesh,
                metadata=metadata,
            )

        spec = self.partition_spec_for_metadata(metadata, mode=mode, shape=shape, mesh=active_mesh)
        if spec is None:
            return None
        return self.named_sharding_for_spec(spec, shape=shape, mesh=_stage_local_mesh(active_mesh, metadata))

    def partition_spec_for_variable(
        self,
        var: spx.Variable,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> PartitionSpec | None:
        """Resolve a SpectraX variable to a :class:`PartitionSpec`.

        Tries native :meth:`spx.Variable.named_sharding` first when the
        variable doesn't carry compound axis metadata, then falls back to
        metadata-driven resolution and finally to the variable's existing
        :class:`NamedSharding` if present.

        Args:
            var: The SpectraX variable.
            mode: Resolution mode.
            shape: Concrete tensor shape (otherwise inferred from the
                variable's value).
            mesh: Mesh override.

        Returns:
            PartitionSpec | None: The resolved spec, or ``None`` when no
            sharding info is available.
        """
        active_mesh = self.mesh if mesh is None else mesh
        if (
            var.metadata.get("tensor_layout") is None
            and active_mesh is not None
            and not _metadata_has_compound_axis_names(var.metadata)
        ):
            value = getattr(var, "value", None)
            value_shape = tuple(value.shape) if hasattr(value, "shape") else shape
            resolved_mode = self._resolve_mode(mode=mode, shape=value_shape, layout_mode=MODE_TRAIN)
            try:
                with self.logical_axis_rules(mode=resolved_mode, shape=value_shape):
                    named = var.named_sharding(active_mesh)
                if value_shape is NOT_GIVEN:
                    return named.spec
                return self._sanitize_spec(named.spec, shape=value_shape, mesh=named.mesh)
            except Exception:
                pass

        resolved = self.resolve_metadata(var.metadata, mode=mode, shape=shape, mesh=mesh)
        if resolved is not None:
            return resolved

        value = getattr(var, "value", None)
        if value is not None:
            sharding = getattr(value, "sharding", None)
            if isinstance(sharding, NamedSharding):
                return self._sanitize_spec(sharding.spec, shape=tuple(value.shape), mesh=mesh)
        return None

    def named_sharding_for_spec(
        self,
        spec: PartitionSpec,
        *,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> NamedSharding:
        """Wrap a :class:`PartitionSpec` in a :class:`NamedSharding`.

        Args:
            spec: Partition spec to wrap.
            shape: Concrete tensor shape used for safety sanitization.
            mesh: Mesh override (required either here or on ``self``).

        Returns:
            NamedSharding: The named sharding bound to the resolved JAX mesh.

        Raises:
            ValueError: If no mesh is available.
        """
        active_mesh = self.mesh if mesh is None else mesh
        if active_mesh is None:
            raise ValueError("A mesh is required to build NamedSharding.")
        base_mesh, _ = _resolve_named_sharding_mesh(active_mesh)
        safe_spec = self._sanitize_spec(spec, shape=shape, mesh=base_mesh)
        return NamedSharding(base_mesh, safe_spec)

    def named_sharding_for_variable(
        self,
        var: spx.Variable,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: OptionalMesh = None,
    ) -> NamedSharding | None:
        """Build a :class:`NamedSharding` for a SpectraX variable.

        Args:
            var: The SpectraX variable.
            mode: Resolution mode.
            shape: Concrete tensor shape.
            mesh: Mesh override.

        Returns:
            NamedSharding | None: A named sharding for the variable, or
            ``None`` when no mesh is available or no spec can be resolved.
        """
        active_mesh = self.mesh if mesh is None else mesh
        if active_mesh is None:
            return None
        if var.metadata.get("tensor_layout") is None:
            resolved_mode = self._resolve_mode(mode=mode, shape=shape, layout_mode=MODE_TRAIN)
            try:
                with self.logical_axis_rules(mode=resolved_mode, shape=shape):
                    named = var.named_sharding(active_mesh)
                if shape is NOT_GIVEN:
                    return named
                return self.named_sharding_for_spec(named.spec, shape=shape, mesh=named.mesh)
            except Exception:
                pass
        spec = self.partition_spec_for_variable(var, mode=mode, shape=shape, mesh=active_mesh)
        if spec is None:
            return None
        return self.named_sharding_for_spec(spec, shape=shape, mesh=_stage_local_mesh(active_mesh, var.metadata))

    def resolve(
        self,
        axes: tp.Sequence[str | None] | tp.Any = NOT_GIVEN,
        mode: str | int | object = NOT_GIVEN,
        dynamic_axes: tp.Any = NOT_GIVEN,
        shape: tp.Sequence[int] | object = NOT_GIVEN,
    ) -> PartitionSpec:
        """High-level entry point for axis-based or layout-based resolution.

        Either provide ``axes`` (sequence of logical names) and ``mode``, or
        a layout-like ``dynamic_axes`` value (e.g. :class:`TensorLayout`).
        Layout-shaped ``axes`` are auto-promoted to ``dynamic_axes``.

        Args:
            axes: Sequence of logical axis names or a layout-like object.
            mode: Resolution mode.
            dynamic_axes: A :class:`TensorLayout`-compatible object that
                supplies its own mode.
            shape: Concrete tensor shape used for safety checks.

        Returns:
            PartitionSpec: The resolved partition spec.

        Raises:
            ValueError: If neither a layout nor ``axes``+``mode`` are
                provided, or if ``dynamic_axes`` is not layout-compatible.
        """
        if dynamic_axes is NOT_GIVEN and axes is not NOT_GIVEN:
            if (
                (isinstance(axes, tuple) and hasattr(axes, "_fields"))
                or (isinstance(axes, type) and issubclass(axes, tuple) and hasattr(axes, "_fields"))
                or (hasattr(axes, "axes") and hasattr(axes, "mode"))
            ):
                dynamic_axes = axes
                axes = NOT_GIVEN

        if axes is NOT_GIVEN or mode is NOT_GIVEN:
            if dynamic_axes is NOT_GIVEN:
                raise ValueError("if axes or mode is empty you should provide dynamic axes")
            tensor_layout = TensorLayout.from_any(dynamic_axes)
            if tensor_layout is None:
                raise ValueError("dynamic_axes must be a TensorLayout-compatible object")
            return self.resolve_layout(
                tensor_layout,
                mode=tensor_layout.mode,
                shape=tuple(shape) if shape is not NOT_GIVEN else NOT_GIVEN,
                mesh=self.mesh,
            )

        return self.resolve_layout(
            axes,
            mode=mode,
            shape=tuple(shape) if shape is not NOT_GIVEN else NOT_GIVEN,
            mesh=self.mesh,
        )

    def shard(
        self,
        x: jax.Array,
        axes: tp.Sequence[str | None] | tp.Any = NOT_GIVEN,
        mode: str | int | object = NOT_GIVEN,
        dynamic_axes: tp.Any = NOT_GIVEN,
        auto_correct: bool = True,
    ) -> jax.Array:
        """Apply a sharding constraint to *x* using the resolved partition spec.

        Args:
            x: Input array.
            axes: Sequence of logical axis names (or layout) to resolve.
            mode: Resolution mode.
            dynamic_axes: Optional layout-like override.
            auto_correct: Whether to drop non-divisible axes via shape-aware
                sanitization.

        Returns:
            jax.Array: ``x`` with a sharding constraint applied (using
            ``self.mesh``).
        """
        spec = self.resolve(
            axes=axes,
            mode=mode,
            dynamic_axes=dynamic_axes,
            shape=tuple(x.shape),
        )
        if auto_correct:
            spec = self._sanitize_spec(spec, shape=tuple(x.shape))
        return spx.with_sharding_constraint(x, spec, mesh=self.mesh)


def coerce_runtime_sharding_resolver(
    value: RuntimeShardingResolver | AxisPolicy | PartitionAxis | dict[str, tp.Any] | None = None,
    *,
    mesh: OptionalMesh = None,
) -> RuntimeShardingResolver:
    """Normalize supported sharding inputs to a runtime resolver."""
    if isinstance(value, RuntimeShardingResolver):
        return value.with_mesh(mesh if mesh is not None else value.mesh)

    inferred_mesh = mesh if mesh is not None else getattr(value, "mesh", None)
    axis_source = value
    for attr_name in ("axis_policy", "partition_axis", "paxis"):
        candidate = getattr(value, attr_name, None)
        if candidate is not None:
            axis_source = candidate
            break

    return RuntimeShardingResolver(axis_policy=coerce_axis_policy(axis_source), mesh=inferred_mesh)


@contextmanager
def logical_axis_rules(
    value: RuntimeShardingResolver | AxisPolicy | PartitionAxis | dict[str, tp.Any] | None = None,
    *,
    mesh: OptionalMesh = None,
    mode: str | int | object = MODE_TRAIN,
    shape: tuple[int, ...] | object = NOT_GIVEN,
    overrides: LogicalAxisRules | None = None,
) -> Iterator[Mapping[str, str | None]]:
    """Open a spectrax logical-axis-rules scope derived from EasyDeL sharding config."""
    resolver = coerce_runtime_sharding_resolver(value, mesh=mesh)
    with resolver.logical_axis_rules(mode=mode, shape=shape, overrides=overrides) as active_rules:
        yield active_rules


def replicated_named_sharding(mesh: MeshLike) -> jax.sharding.NamedSharding:
    """Return a fully-replicated ``NamedSharding`` for the given mesh.

    Centralised constructor for the ``NamedSharding(mesh, PartitionSpec())``
    pattern that was being hand-rolled at ~8 sites across the codebase.
    Accepts either a JAX :class:`jax.sharding.Mesh` or a spectrax
    :class:`SpxMesh` / :class:`MpMdMesh`.
    """
    jax_mesh = mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh
    return jax.sharding.NamedSharding(jax_mesh, PartitionSpec())


def replicate_on_array_mesh(value: jax.Array, reference: jax.Array) -> jax.Array:
    """Replicate ``value`` on ``reference``'s named-sharding mesh."""
    reference_mesh = resolve_array_mesh(reference)
    if reference_mesh is None:
        return value
    target = replicated_named_sharding(reference_mesh)
    current = getattr(value, "sharding", None)
    if isinstance(current, NamedSharding) and mesh_matches(current.mesh, reference_mesh) and current.spec == target.spec:
        return value
    return jax.device_put(value, target)


def final_stage_replicated_sharding(
    model: tp.Any,
    mesh: OptionalMesh,
    fallback_sharding: jax.sharding.Sharding,
) -> jax.sharding.Sharding:
    """Return replicated sharding on the model's final PP stage when mesh is MPMD."""
    if not is_mpmd_mesh(mesh):
        return fallback_sharding
    text_config = model.config.get_text_config()
    total_layers = int(getattr(text_config, "num_hidden_layers", 1))
    if hasattr(model, "_layer_physical_stage_assignment"):
        stage = model._layer_physical_stage_assignment(total_layers - 1, total_layers)
    else:
        mpmd_dim = int(getattr(mesh, "mpmd_dim", 1))
        stage = (mpmd_dim - 1, mpmd_dim)
    stage_mesh = resolve_stage_mesh(mesh, stage=stage)
    if stage_mesh is None:
        return fallback_sharding
    return replicated_named_sharding(stage_mesh)


def specs_to_named_sharding(tree: tp.Any, mesh: OptionalMesh = None) -> tp.Any:
    """Convert a PyTree of PartitionSpecs into a PyTree of NamedShardings."""
    active_mesh = mesh or pxla.thread_resources.env.physical_mesh
    jax_mesh, _ = _resolve_named_sharding_mesh(active_mesh)
    return jax.tree_util.tree_map(lambda spec: NamedSharding(spec=spec, mesh=jax_mesh), tree)


__all__ = [
    "CANONICAL_MESH_AXIS_NAMES",
    "AxisPolicy",
    "LogicalAxisRules",
    "MeshLike",
    "OptionalMesh",
    "RuntimeShardingResolver",
    "StageMesh",
    "TensorLayout",
    "axis_index",
    "coerce_axis_policy",
    "coerce_partition_spec",
    "coerce_runtime_sharding_resolver",
    "final_stage_replicated_sharding",
    "is_mpmd_mesh",
    "is_valid_mesh",
    "logical_axis_rules",
    "mesh_axis_size",
    "mesh_matches",
    "mesh_partition_product",
    "metadata_for_layout",
    "normalize_axis_names",
    "partition_spec_for_mesh",
    "pick_array_mesh",
    "pick_mesh",
    "replicate_on_array_mesh",
    "replicated_named_sharding",
    "resolve_array_mesh",
    "resolve_safe_sharding",
    "resolve_stage_cache_mesh",
    "resolve_stage_mesh",
    "sanitize_partition_spec_for_shape",
    "sanitize_partition_specs_for_shape_tree",
    "sanitize_sharding_axes_for_shape",
    "sharding_for_layout",
    "specs_to_named_sharding",
]
