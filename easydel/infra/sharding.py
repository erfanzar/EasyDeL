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
import spectrax as spx
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
from spectrax.sharding.mesh import SpxMesh

_HAS_SPECTRAX_MESH_TYPES = True

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
    return tuple(_normalize_axis_entry(axis) for axis in axes)


def _normalize_logical_axis_rules(overrides: LogicalAxisRules | None) -> tuple[tuple[str, str | None], ...]:
    if overrides is None:
        return ()
    items = overrides.items() if isinstance(overrides, Mapping) else overrides
    normalized: list[tuple[str, str | None]] = []
    for logical_name, mesh_axis in items:
        normalized.append((str(logical_name), None if mesh_axis is None else str(mesh_axis)))
    return tuple(normalized)


def _simple_rule_value(value: tp.Any) -> str | None | object:
    normalized = _normalize_axis_entry(value)
    if normalized is None:
        return None
    if isinstance(normalized, tuple):
        return _UNSUPPORTED_SIMPLE_RULE
    return normalized


def _mesh_axis_size(mesh: Mesh, axis_name: str) -> int:
    try:
        return int(mesh.shape[axis_name])
    except Exception:
        return 1


def mesh_partition_product(mesh: Mesh, axis_spec: AxisEntry) -> int:
    """Return shard multiplicity implied by a PartitionSpec entry."""
    if axis_spec is None:
        return 1
    if isinstance(axis_spec, (tuple, list)):
        product = 1
        for axis_name in axis_spec:
            if axis_name is None:
                continue
            product *= _mesh_axis_size(mesh, axis_name)
        return int(product)
    return _mesh_axis_size(mesh, axis_spec)


_mesh_partition_product = mesh_partition_product


def _resolve_named_sharding_mesh(
    mesh: Mesh | tp.Any,
) -> tuple[Mesh, tp.Any | None]:
    if _HAS_SPECTRAX_MESH_TYPES and isinstance(mesh, SpxMesh):
        return mesh.jax_mesh, mesh.mpmd_mesh
    if _HAS_SPECTRAX_MESH_TYPES and isinstance(mesh, MpMdMesh):
        return mesh.jax_mesh, mesh
    return mesh, None


def _stage_local_mesh(
    mesh: Mesh | tp.Any,
    metadata: dict[str, tp.Any] | None,
) -> Mesh | tp.Any:
    assignment = metadata_stage_assignment(metadata)
    if assignment is None:
        if _HAS_SPECTRAX_MESH_TYPES and isinstance(mesh, SpxMesh):
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


def _is_mpmd_mesh(mesh: Mesh | tp.Any | None) -> bool:
    if mesh is None or not _HAS_SPECTRAX_MESH_TYPES:
        return False
    if isinstance(mesh, SpxMesh):
        return bool(mesh.is_mpmd)
    return isinstance(mesh, MpMdMesh)


def sanitize_partition_spec_for_shape(
    spec: PartitionSpec,
    shape: tuple[int, ...],
    mesh: Mesh,
) -> PartitionSpec:
    """Drop non-divisible sharding axes for a concrete tensor shape."""
    axes = list(tuple(spec))
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
        return spec
    return PartitionSpec(*axes)


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
        object.__setattr__(self, "axes", _normalize_axes(self.axes))

    @classmethod
    def from_any(cls, value: tp.Any) -> TensorLayout | None:
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
        return {"axes": self.axes, "mode": self.mode}

    def as_spectrax_sharding(self) -> spx.Sharding:
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
    axis_names = _sharding_axis_names_from_metadata(metadata)
    if axis_names is None:
        return False
    try:
        return any(isinstance(axis, list | tuple) for axis in axis_names if axis is not None)
    except TypeError:
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class AxisPolicy:
    """Immutable semantic-axis configuration for EasyDeL."""

    partition_axis: PartitionAxis = dataclasses.field(default_factory=PartitionAxis, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition_axis", _coerce_partition_axis(self.partition_axis))

    def __getattr__(self, name: str) -> tp.Any:
        return getattr(self.partition_axis, name)

    @classmethod
    def from_partition_axis(cls, value: PartitionAxis | dict[str, tp.Any] | None) -> AxisPolicy:
        return cls(partition_axis=_coerce_partition_axis(value))

    @classmethod
    def from_dict(cls, value: dict[str, tp.Any]) -> AxisPolicy:
        return cls.from_partition_axis(value)

    @classmethod
    def from_any(cls, value: AxisPolicy | PartitionAxis | dict[str, tp.Any] | None) -> AxisPolicy:
        return coerce_axis_policy(value)

    def to_partition_axis(self) -> PartitionAxis:
        return copy.deepcopy(self.partition_axis)

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            field.name: copy.deepcopy(getattr(self.partition_axis, field.name))
            for field in dataclasses.fields(self.partition_axis)
        }

    def resolve_axis(
        self,
        axes: tp.Sequence[str | None],
        mode: str,
    ) -> list[tp.Any]:
        return self.partition_axis.resolve_axis(axes=axes, mode=mode)

    def resolve_spec(
        self,
        axes: tp.Sequence[str | None],
        mode: str,
    ) -> PartitionSpec:
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
    """Lower semantic sharding declarations to concrete JAX shardings."""

    axis_policy: AxisPolicy
    mesh: Mesh | tp.Any | None = None

    def with_mesh(self, mesh: Mesh | tp.Any | None) -> RuntimeShardingResolver:
        return RuntimeShardingResolver(axis_policy=self.axis_policy, mesh=mesh)

    @property
    def paxis(self) -> PartitionAxis:
        return self.axis_policy.to_partition_axis()

    def _resolve_mode(
        self,
        *,
        mode: str | int | object,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        layout_mode: str | int = MODE_TRAIN,
    ) -> str:
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
        mesh: Mesh | tp.Any | None = None,
    ) -> PartitionSpec:
        active_mesh = self.mesh if mesh is None else mesh
        if active_mesh is None or shape is NOT_GIVEN:
            return spec
        base_mesh, _ = _resolve_named_sharding_mesh(active_mesh)
        return sanitize_partition_spec_for_shape(spec=spec, shape=shape, mesh=base_mesh)

    def _resolve_axis_name_entry(self, axis: tp.Any, mode: str) -> AxisEntry:
        normalized = _normalize_axis_entry(axis)
        if normalized is None:
            return None

        logical_rules = dict(self.logical_axis_rule_pairs(mode=mode))

        def _resolve_one(name: str) -> str | None:
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
        return PartitionSpec(*(self._resolve_axis_name_entry(axis, mode) for axis in axes))

    def partition_spec_for_layout(
        self,
        layout: tp.Any,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: Mesh | tp.Any | None = None,
    ) -> PartitionSpec:
        return self.resolve_layout(layout=layout, mode=mode, shape=shape, mesh=mesh)

    def resolve_layout(
        self,
        layout: tp.Any,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: Mesh | tp.Any | None = None,
    ) -> PartitionSpec:
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
        mesh: Mesh | tp.Any | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> NamedSharding:
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
        mesh: Mesh | tp.Any | None = None,
    ) -> PartitionSpec | None:
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
        mesh: Mesh | tp.Any | None = None,
    ) -> PartitionSpec | None:
        return self.resolve_metadata(metadata=metadata, mode=mode, shape=shape, mesh=mesh)

    def named_sharding_for_metadata(
        self,
        metadata: dict[str, tp.Any] | None,
        *,
        mode: str | int | object = NOT_GIVEN,
        shape: tuple[int, ...] | object = NOT_GIVEN,
        mesh: Mesh | tp.Any | None = None,
    ) -> NamedSharding | None:
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
        mesh: Mesh | tp.Any | None = None,
    ) -> PartitionSpec | None:
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
        mesh: Mesh | tp.Any | None = None,
    ) -> NamedSharding:
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
        mesh: Mesh | tp.Any | None = None,
    ) -> NamedSharding | None:
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
    mesh: Mesh | tp.Any | None = None,
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
    mesh: Mesh | tp.Any | None = None,
    mode: str | int | object = MODE_TRAIN,
    shape: tuple[int, ...] | object = NOT_GIVEN,
    overrides: LogicalAxisRules | None = None,
) -> Iterator[Mapping[str, str | None]]:
    """Open a spectrax logical-axis-rules scope derived from EasyDeL sharding config."""
    resolver = coerce_runtime_sharding_resolver(value, mesh=mesh)
    with resolver.logical_axis_rules(mode=mode, shape=shape, overrides=overrides) as active_rules:
        yield active_rules


def replicated_named_sharding(mesh: tp.Any) -> jax.sharding.NamedSharding:
    """Return a fully-replicated ``NamedSharding`` for the given mesh.

    Centralised constructor for the ``NamedSharding(mesh, PartitionSpec())``
    pattern that was being hand-rolled at ~8 sites across the codebase.
    Accepts either a JAX :class:`jax.sharding.Mesh` or a spectrax
    :class:`SpxMesh` / :class:`MpMdMesh`.
    """
    jax_mesh = mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh
    return jax.sharding.NamedSharding(jax_mesh, PartitionSpec())


__all__ = [
    "CANONICAL_MESH_AXIS_NAMES",
    "AxisPolicy",
    "LogicalAxisRules",
    "RuntimeShardingResolver",
    "TensorLayout",
    "coerce_axis_policy",
    "coerce_runtime_sharding_resolver",
    "logical_axis_rules",
    "mesh_partition_product",
    "metadata_for_layout",
    "replicated_named_sharding",
    "sanitize_partition_spec_for_shape",
    "sharding_for_layout",
]
