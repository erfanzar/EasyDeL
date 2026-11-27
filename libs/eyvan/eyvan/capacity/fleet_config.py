# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declarative fleet configuration used by the Eyvan operator CLI."""

from __future__ import annotations

import re
import typing as tp
import warnings
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, model_json
from eyvan.capacity.drivers.base import canonical_execution_backend, capacity_backend_agent_family
from eyvan.capacity.hardware_catalog import shape_for, validate_accelerator_write_boundary
from eyvan.capacity.providers.gcp.queued_resources import bootstrap_egress_choices
from eyvan.records import (
    AvailabilityPolicy,
    CanonicalCapacityClass,
    CapacityClass,
    CapacityPoolRecord,
    JsonDocument,
    NativeRuntimeConfig,
    ResourceLimits,
    WarmWindow,
    validate_capacity_pool_quota_groups,
)

_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_WORKER_SCALING_ALIASES = {
    "min_workers": "min_cells",
    "max_workers": "max_cells",
    "initial_workers": "initial_cells",
}
#: Largest pool any fleet may declare. Also what `max_workers: 0` resolves
#: to, so "unbounded" has one spelling the whole package agrees on.
MAX_POOL_WORKERS = 10_000
_MAX_POOL_WORKERS = MAX_POOL_WORKERS
_RESOURCE_KEY_ALIASES = {
    "cpu": "cpu",
    "gpu": "gpu",
    "memory": "memory_gib",
    "memory_gib": "memory_gib",
    "tpu": "tpu",
    "custom": "custom",
}


def _resource_values(resources: ResourceLimits) -> dict[str, float]:
    return {
        "cpu": resources.cpu,
        "memory_gib": resources.memory_gib,
        "gpu": resources.gpu,
        "tpu": resources.tpu,
        **resources.custom,
    }


def _cell_prefix(fleet_name: str, node_type: str) -> str:
    return f"{fleet_name}-{node_type}"[:84].rstrip("-")


def validate_fleet_name(name: str) -> str:
    """Validate and return a fleet name safe for IDs and URL paths."""

    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError("fleet name must contain only lowercase letters, digits, and hyphens")
    return name


def _json_document(value: JsonDocument) -> JsonObject:
    """Convert a validated Pydantic JSON document into Eyvan's JSON algebra."""

    document = dict(value)
    if not is_json_object(document):
        raise ValueError("Pydantic JSON document contains a non-JSON value")
    return document


def _json_object(value: tp.Mapping[str, JsonValue]) -> JsonObject:
    """Copy an already JSON-shaped mapping into a concrete JSON object."""

    document = dict(value)
    if not is_json_object(document):
        raise ValueError("JSON mapping contains a non-JSON value")
    return document


def _normalize_resource_mapping(value: JsonValue, *, field_name: str) -> JsonValue:
    """Normalize public resource keys without changing arbitrary custom names."""

    if not is_json_object(value):
        return value
    normalized: JsonObject = {}
    sources: dict[str, str] = {}
    for raw_key, amount in value.items():
        key = _RESOURCE_KEY_ALIASES.get(raw_key.casefold(), raw_key)
        if key in normalized:
            previous = sources[key]
            raise ValueError(f"{field_name} has conflicting resource keys {previous!r} and {raw_key!r}")
        normalized[key] = amount
        sources[key] = raw_key
    return normalized


def _normalize_resource_fields(value: JsonObject, *, pool_name: str) -> JsonObject:
    """Normalize resource mappings shared by public and legacy fleet inputs."""

    normalized = dict(value)
    for field_name in ("resources", "allocatable", "max_replica_resources"):
        if field_name in normalized:
            normalized[field_name] = _normalize_resource_mapping(
                normalized[field_name], field_name=f"pool {pool_name!r} {field_name}"
            )
    return normalized


def _pool_kind_from_resources(value: tp.Mapping[str, JsonValue]) -> str | None:
    """Infer a public pool kind from its advertised built-in resources."""

    resources = value.get("resources")
    if not is_json_object(resources):
        return None
    accelerated = [kind for kind in ("gpu", "tpu") if kind in resources]
    if len(accelerated) > 1:
        raise ValueError("a pool cannot advertise both GPU and TPU resources")
    if accelerated:
        return accelerated[0]
    return "cpu" if "cpu" in resources else None


def _pool_accelerator_type(pool: JsonObject, *, name: str) -> str:
    """Return the accelerator shape a pool runs, which need not be its name.

    A pool is keyed by its accelerator type by default. Allowing a separate name
    lets a fleet hold multiple pools of the same shape, so workloads can race
    zones with different Spot availability.

    So a pool may state ``accelerator_type`` explicitly and take a free-form
    name, letting ``v6e-8-us-east5-b`` and ``v6e-8-us-central2-b`` coexist.
    """

    declared = pool.get("accelerator_type")
    if declared is None:
        return name
    if not isinstance(declared, str) or not declared or declared != declared.strip():
        raise ValueError(f"pool {name!r} accelerator_type must be a non-empty trimmed string when set")
    return declared


def _apply_catalog_shape(pool: JsonObject, *, accelerator_type: str) -> None:
    """Fill every omitted admission resource for a known accelerator shape.

    The shape is the pool's ``accelerator_type`` when it states one, and its
    mapping key otherwise. An explicit positive CPU or memory value remains an
    intentional schedulable override; omitted values inherit the catalog.
    TPU count itself is a shape invariant and is checked later by
    ``TpuNodeType.validate_tpu_resources``.

    Silence for an unknown type is deliberate only at this normalization step.
    The typed node validator then requires a complete positive manual resource
    vector. That keeps new hardware usable without turning a provider-discovered
    name into a zero-memory schedulable pool.
    """

    shape = shape_for(accelerator_type)
    if shape is None or shape.kind != "tpu":
        return
    resources = pool.setdefault("resources", {})
    if not is_json_object(resources):
        return
    resources.setdefault("cpu", shape.host_cpu)
    resources.setdefault("memory_gib", shape.host_memory_gib)
    resources.setdefault("tpu", shape.schedulable_tpu)


def _expand_pool_zones(name: str, value: JsonValue) -> list[tuple[str, JsonValue]]:
    """Split one pool declaring several zones into one pool per zone.

    Racing zones improves the chance of obtaining Spot capacity. The scheduler
    treats interchangeable pools of one shape as cross-zone alternates; expansion
    avoids repeating the same declaration with a different name for each zone.

    The expansion is mechanical and visible: ``v6e-8`` with three zones becomes
    ``v6e-8-us-east5-a`` and its siblings, each pinned to one zone and each
    naming ``v6e-8`` as its shape. They are therefore identical in admission
    width and topology, which is exactly the condition the submission path
    requires before it will offer them as alternates for each other.

    Args:
        name: The pool's mapping key.
        value: Its declaration.

    Returns:
        One ``(name, declaration)`` pair per zone, or the original pair
        unchanged when the pool names no zone list.

    Raises:
        ValueError: When ``zones`` is not a list of distinct non-empty strings,
            or is combined with a single ``zone``. Two ways of saying where a
            pool runs is one way too many.
    """

    if not is_json_object(value) or "zones" not in value:
        return [(name, value)]
    declared = value["zones"]
    if not isinstance(declared, list) or not declared:
        raise ValueError(f"pool {name!r} zones must be a non-empty list of zone names")
    zones = []
    for zone in declared:
        if not isinstance(zone, str) or not zone or zone != zone.strip():
            raise ValueError(f"pool {name!r} zones must contain non-empty trimmed zone names")
        zones.append(zone)
    if len(set(zones)) != len(zones):
        raise ValueError(f"pool {name!r} lists a zone twice")
    if value.get("zone") is not None:
        raise ValueError(f"pool {name!r} sets both zone and zones; use one")
    shape = value.get("accelerator_type")
    expanded: list[tuple[str, JsonValue]] = []
    for zone in zones:
        pool = dict(value)
        pool.pop("zones")
        pool["zone"] = zone
        # Each sibling keeps the shared shape, so the fleet holds one kind of
        # hardware in several places rather than several kinds of hardware.
        pool["accelerator_type"] = shape if isinstance(shape, str) and shape else name
        expanded.append((f"{name}-{zone}", pool))
    return expanded


def _normalize_public_pool(name: str, value: JsonValue) -> JsonValue:
    """Compile one concise ``pools`` entry into the internal node-type shape."""

    if not is_json_object(value):
        return value
    pool = _normalize_resource_fields(dict(value), pool_name=name)
    if "label" in pool:
        label = pool.pop("label")
        if not isinstance(label, str) or label != name:
            raise ValueError(f"pool {name!r} label must exactly match its mapping key")
    if "slices" in pool:
        raise ValueError(
            f"pool {name!r} uses removed field 'slices'; a pool defines one complete TPU worker, "
            "use TpuConfig(pods=N) to request N workers"
        )
    provider_options = pool.get("provider_options")
    if is_json_object(provider_options):
        provider_options = dict(provider_options)
        if "node_count" in provider_options:
            raise ValueError(
                f"pool {name!r} cannot set provider_options.node_count; Eyvan creates one queued resource per worker"
            )
        if "node_prefix" in provider_options:
            raise ValueError(f"pool {name!r} cannot set provider_options.node_prefix for a one-worker pool")
        pool["provider_options"] = provider_options

    accelerator_type = _pool_accelerator_type(pool, name=name)
    _apply_catalog_shape(pool, accelerator_type=accelerator_type)

    kind = pool.get("kind")
    if isinstance(kind, str):
        kind = kind.strip().lower()
        pool["kind"] = kind
    if kind is None:
        kind = _pool_kind_from_resources(pool)
        if kind is None:
            raise ValueError(f"pool {name!r} must declare kind or advertise CPU, GPU, or TPU resources")
        pool["kind"] = kind
    if kind in {"gpu", "tpu"}:
        # The shape a pool runs, which defaults to its name but may be stated
        # explicitly so several pools of one shape can coexist in a fleet --
        # each pinned to a different zone, so a workload has somewhere else to
        # go when one zone keeps reclaiming.
        pool["accelerator_type"] = accelerator_type
    return pool


def _normalize_legacy_node_type(name: str, value: JsonValue) -> JsonValue:
    """Accept only legacy node types that already represent one worker allocation."""

    if not is_json_object(value):
        return value
    node_type = _normalize_resource_fields(dict(value), pool_name=name)
    if "slices" in node_type:
        slices = node_type.pop("slices")
        if isinstance(slices, bool) or not isinstance(slices, int) or slices != 1:
            raise ValueError(
                f"legacy node_types.{name!r}.slices must be 1; migrate multi-slice capacity to independent pools"
            )
    if node_type.get("kind") == "tpu" and isinstance(accelerator_type := node_type.get("accelerator_type"), str):
        _apply_catalog_shape(node_type, accelerator_type=accelerator_type)
    provider_options = node_type.get("provider_options")
    if is_json_object(provider_options):
        provider_options = dict(provider_options)
        if "node_count" in provider_options:
            node_count = provider_options.pop("node_count")
            if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count != 1:
                raise ValueError(
                    f"legacy node_types.{name!r}.provider_options.node_count must be 1; "
                    "migrate multi-node capacity to independent pools"
                )
        if "node_prefix" in provider_options:
            raise ValueError(
                f"legacy node_types.{name!r}.provider_options.node_prefix is not supported for a one-worker pool"
            )
        node_type["provider_options"] = provider_options
    return node_type


class FleetConfigModel(BaseModel):
    """Strict immutable base for operator configuration models."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProviderConfig(FleetConfigModel):
    """Default provider identity shared by every node type in a fleet."""

    type: str = Field(min_length=1)
    project: str | None = Field(default=None, min_length=1)
    region: str = Field(min_length=1)
    zone: str | None = Field(default=None, min_length=1)


class AutoscalingDefaults(FleetConfigModel):
    """Fleet-wide defaults for server-side capacity pool reconciliation."""

    idle_timeout_s: int = Field(default=3600, ge=0)
    queue_timeout_s: float | None = Field(default=None, gt=0)
    provisioning_timeout_s: int = Field(default=1800, gt=0)
    upscaling_speed: float = Field(default=1.0, gt=0)
    launch_batch: int = Field(default=5, ge=1)


class NodeTypeBase(FleetConfigModel):
    """Fields shared by CPU, GPU, and TPU capacity templates.

    Worker bounds also accept the operator spellings ``min_workers``,
    ``initial_workers``, and ``max_workers``. ``max_workers: 0`` means
    "unbounded": validation rewrites it to the maximum supported pool
    ceiling of 10,000 cells and emits a warning so the alias is never
    applied silently.
    """

    resources: ResourceLimits
    allocatable: ResourceLimits | None = None
    min_cells: int = Field(default=0, ge=0)
    #: Recurring windows during which the pool holds a raised cell floor.
    #:
    #: Honoured by the planner since ``WarmWindow`` was written; until now every
    #: writer was a test, so the feature existed and no operator could reach it.
    warm_windows: tuple[WarmWindow, ...] = ()
    max_cells: int = Field(default=1, ge=0, le=10_000)
    initial_cells: int | None = Field(default=None, ge=0)
    idle_timeout_s: int | None = Field(default=None, ge=0)
    queue_timeout_s: float | None = Field(default=None, gt=0)
    provisioning_timeout_s: int | None = Field(default=None, gt=0)
    upscaling_speed: float | None = Field(default=None, gt=0)
    launch_batch: int | None = Field(default=None, ge=1)
    capacity_backend: str | None = Field(default=None, min_length=1)
    provider_project: str | None = Field(default=None, min_length=1)
    region: str | None = Field(default=None, min_length=1)
    zone: str | None = Field(default=None, min_length=1)
    capacity_class: CanonicalCapacityClass | None = None
    executor: str | None = Field(default=None, min_length=1)
    agent_endpoint: str | None = Field(default=None, min_length=1)
    provider_options: JsonDocument = Field(default_factory=dict)
    runtime: NativeRuntimeConfig | None = None
    #: How this pool backs off when the provider refuses it, overriding the
    #: fleet's. Whole-value rather than merged field by field: a backoff curve
    #: is one shape, and half of somebody else's is not a curve anybody chose.
    availability_policy: AvailabilityPolicy | None = None
    #: How long the provider holds this pool's capacity before taking it back,
    #: whatever the job is doing. Required for `capacity_class: flex_start`.
    max_hold_s: int | None = Field(default=None, ge=60)
    unit_prices: dict[str, float] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    topology: dict[str, str] = Field(default_factory=dict)
    max_replica_resources: ResourceLimits | None = None
    max_gang_replicas: int | None = Field(default=None, ge=1)
    #: Let several jobs that each ask for some TPU chips (``TpuConfig(chips=...)``)
    #: share this pool's cells, each seeing only its own chips. Off by default.
    chip_sharing: bool = False
    quota_group: str | None = Field(default=None, max_length=192)
    quota_limit: int | None = Field(default=None, ge=0)
    quota_units_per_cell: int | None = Field(default=None, ge=1)

    def _warn_about_unpriced_resources(self) -> None:
        """Warn when a pool advertises resources it declares no unit price for.

        ``ResourceVector.dot`` reads prices with ``prices.get(name)``, which returns
        zero for an absent key, so an unpriced pool's cells have an hourly cost of
        exactly zero. ``Scheduler._budget_allows`` is a *hard admission gate*, so such
        a pool admits unlimited work while reporting budget compliance and never moving
        ``BudgetRecord.spent``.

        A warning rather than an error, and emitted here rather than in the scheduler,
        for two reasons: unpriced pools are legitimate wherever no money budget is set
        (local and manual development fleets never price anything), and the scheduler is
        a pure placement function that should not refuse a placement because an operator
        omitted an input it never sees. ``eyvan doctor`` upgrades this to a failure when
        a hard money budget actually applies, which is the only context that makes the
        omission dangerous.

        A declared zero counts as priced, matching ``CapacityCell.prices_cover``:
        a zero is a statement about cost, not an omission.
        """

        priced = set(self.unit_prices)
        unpriced = sorted(
            name
            for name, amount in _resource_values(self.resources).items()
            if amount and name not in priced and name != "custom"
        )
        if unpriced and priced:
            # Partial pricing is unambiguously a mistake: the operator meant to price
            # this pool and missed a resource, so the cost is silently understated.
            warnings.warn(
                f"pool declares unit prices but none for {', '.join(unpriced)}; its cost will be understated "
                "and any hard money budget will under-count it",
                UserWarning,
                stacklevel=2,
            )

    def _validate_host_admission_resources(self, *, kind: str) -> None:
        """Require positive CPU and memory on every schedulable host vector."""

        vectors = (("resources", self.resources), ("allocatable", self.allocatable or self.resources))
        for field_name, resources in vectors:
            if resources.cpu <= 0:
                raise ValueError(f"{kind} node types require {field_name}.cpu > 0")
            if resources.memory_gib <= 0:
                raise ValueError(f"{kind} node types require {field_name}.memory_gib > 0")

    @model_validator(mode="before")
    @classmethod
    def normalize_worker_scaling_aliases(cls, value: tp.Any) -> tp.Any:
        """Map operator-facing worker bounds to the internal cell fields."""

        if not is_json_object(value):
            return value
        normalized: JsonObject = dict(value)
        for worker_field, cell_field in _WORKER_SCALING_ALIASES.items():
            if worker_field not in normalized:
                continue
            worker_value = normalized[worker_field]
            if worker_field == "max_workers" and not isinstance(worker_value, bool) and worker_value == 0:
                worker_value = _MAX_POOL_WORKERS
                bounded_by_quota_group = (
                    bool(normalized.get("quota_group")) and normalized.get("quota_limit") is not None
                )
                if not bounded_by_quota_group:
                    warnings.warn(
                        f"max_workers: 0 means unbounded and was normalized to the maximum pool ceiling of "
                        f"{_MAX_POOL_WORKERS} cells; spell max_workers: {_MAX_POOL_WORKERS} to silence this warning",
                        UserWarning,
                        stacklevel=2,
                    )
            if cell_field in normalized and normalized[cell_field] != worker_value:
                raise ValueError(f"{worker_field} conflicts with {cell_field}")
            normalized[cell_field] = worker_value
            normalized.pop(worker_field)
        return normalized

    @property
    def desired_initial_cells(self) -> int:
        """Return the number of cells materialized by the first ``up``."""

        return self.min_cells if self.initial_cells is None else self.initial_cells

    @property
    def cell_accelerator_type(self) -> str:
        """Return the provider shape stored in the existing cell API."""

        raise NotImplementedError

    @property
    def default_capacity_backend(self) -> str:
        """Return the built-in capacity backend for this resource kind."""

        raise NotImplementedError

    @property
    def compiled_provider_options(self) -> JsonObject:
        """Return provider-specific creation options for this template."""

        return _json_document(self.provider_options)

    @model_validator(mode="after")
    def validate_scaling_bounds(self) -> NodeTypeBase:
        desired = self.desired_initial_cells
        if self.min_cells > self.max_cells:
            raise ValueError("min_cells must not exceed max_cells")
        for window in self.warm_windows:
            if window.min_cells > self.max_cells:
                # Caught here rather than at 09:00 on a Monday. A floor above the
                # ceiling is a window that can never be satisfied, and the
                # planner would clamp it silently -- so the operator would see a
                # pool that quietly ignored the number they wrote.
                raise ValueError(f"warm window min_cells {window.min_cells} exceeds max_cells {self.max_cells}")
        if desired < self.min_cells:
            raise ValueError("initial_cells must be at least min_cells")
        if desired > self.max_cells:
            raise ValueError("initial_cells must not exceed max_cells")
        quota_fields = (self.quota_group, self.quota_limit, self.quota_units_per_cell)
        if any(value is not None for value in quota_fields):
            if self.quota_group is None or self.quota_limit is None or self.quota_units_per_cell is None:
                raise ValueError("quota_group, quota_limit, and quota_units_per_cell must be configured together")
            if not self.quota_group or self.quota_group != self.quota_group.strip():
                raise ValueError("quota_group must be non-empty and trimmed")
            if desired * self.quota_units_per_cell > self.quota_limit:
                raise ValueError("initial quota demand exceeds quota_limit")
        if any(price < 0 for price in self.unit_prices.values()):
            raise ValueError("unit prices must be non-negative")
        self._warn_about_unpriced_resources()
        capacity = _resource_values(self.resources)
        if self.allocatable is not None:
            allocatable = _resource_values(self.allocatable)
            exceeded = [name for name, amount in allocatable.items() if amount > capacity.get(name, 0)]
            if exceeded:
                raise ValueError(f"allocatable {exceeded[0]} must fit within resources")
        if self.max_replica_resources is not None:
            allocatable = _resource_values(self.allocatable or self.resources)
            maximum = _resource_values(self.max_replica_resources)
            exceeded = [name for name, amount in maximum.items() if amount > allocatable.get(name, 0)]
            if exceeded:
                raise ValueError(f"max per-replica {exceeded[0]} must fit within allocatable resources")
        return self


class CpuNodeType(NodeTypeBase):
    """A CPU worker template."""

    kind: Literal["cpu"]
    machine_type: str = Field(min_length=1)

    @property
    def cell_accelerator_type(self) -> str:
        return "cpu"

    @property
    def default_capacity_backend(self) -> str:
        return "gcp-compute"

    @property
    def compiled_provider_options(self) -> JsonObject:
        options = super().compiled_provider_options
        options["machine_type"] = self.machine_type
        return options

    @model_validator(mode="after")
    def validate_cpu_resources(self) -> CpuNodeType:
        self._validate_host_admission_resources(kind="CPU")
        if self.resources.gpu or self.resources.tpu:
            raise ValueError("CPU node types cannot advertise GPU or TPU resources")
        return self


class GpuNodeType(NodeTypeBase):
    """A GPU worker template with its host machine shape."""

    kind: Literal["gpu"]
    accelerator_type: str = Field(min_length=1)
    machine_type: str = Field(min_length=1)

    @property
    def cell_accelerator_type(self) -> str:
        return self.accelerator_type

    @property
    def default_capacity_backend(self) -> str:
        return "gcp-compute"

    @property
    def compiled_provider_options(self) -> JsonObject:
        options = super().compiled_provider_options
        options["machine_type"] = self.machine_type
        options["gpu_type"] = self.accelerator_type
        options["gpu_count"] = self.resources.gpu
        return options

    @model_validator(mode="after")
    def validate_gpu_resources(self) -> GpuNodeType:
        self._validate_host_admission_resources(kind="GPU")
        if self.resources.gpu <= 0:
            raise ValueError("GPU node types require resources.gpu > 0")
        if self.allocatable is not None and self.allocatable.gpu <= 0:
            raise ValueError("GPU node types require allocatable.gpu > 0 when allocatable is set")
        if self.resources.tpu or (self.allocatable is not None and self.allocatable.tpu):
            raise ValueError("GPU node types cannot advertise TPU resources")
        return self


class TpuNodeType(NodeTypeBase):
    """A complete TPU slice worker template."""

    kind: Literal["tpu"]
    accelerator_type: str = Field(min_length=1)

    @property
    def cell_accelerator_type(self) -> str:
        return self.accelerator_type

    @property
    def default_capacity_backend(self) -> str:
        shape = shape_for(self.accelerator_type)
        return "gcp-tpu" if shape is None else shape.capacity_backend

    @property
    def compiled_provider_options(self) -> JsonObject:
        shape = shape_for(self.accelerator_type)
        selected_backend = (self.capacity_backend or self.default_capacity_backend).strip().lower().replace("_", "-")
        if (
            shape is not None
            and shape.machine_type is not None
            and selected_backend in {"compute", "gcp", "gcp-compute"}
        ):
            options = super().compiled_provider_options
            # An integrated Compute Engine TPU VM. Its machine type, chip count
            # and image all come from the one catalog entry rather than a second
            # dictionary that had to agree with it.
            options.update(
                {
                    "machine_type": shape.machine_type,
                    "tpu_type": shape.tpu_type,
                    "tpu_count": shape.chips,
                    "image_family": shape.image_family,
                    "image_project": shape.image_project,
                }
            )
            return options
        options = super().compiled_provider_options
        # A pool worker is always one queued resource.  The public fleet
        # schema removes any singleton node_count compatibility spelling too.
        options.pop("node_count", None)
        return options

    @model_validator(mode="after")
    def validate_tpu_resources(self) -> TpuNodeType:
        shape = shape_for(self.accelerator_type)
        allocatable = self.allocatable or self.resources
        if shape is None and any(
            value <= 0
            for value in (
                self.resources.cpu,
                self.resources.memory_gib,
                allocatable.cpu,
                allocatable.memory_gib,
            )
        ):
            raise ValueError(
                f"unknown TPU shape {self.accelerator_type!r} requires explicit positive CPU and memory_gib "
                "in resources and allocatable when allocatable is set"
            )
        self._validate_host_admission_resources(kind="TPU")
        if self.resources.tpu <= 0:
            raise ValueError("TPU node types require resources.tpu > 0")
        if self.allocatable is not None and self.allocatable.tpu <= 0:
            raise ValueError("TPU node types require allocatable.tpu > 0 when allocatable is set")
        if self.resources.gpu or (self.allocatable is not None and self.allocatable.gpu):
            raise ValueError("TPU node types cannot advertise GPU resources")
        configured_nodes = self.provider_options.get("node_count", 1)
        if isinstance(configured_nodes, bool) or not isinstance(configured_nodes, int) or configured_nodes != 1:
            raise ValueError("TPU fleet workers cannot set provider_options.node_count above one")
        if self.provider_options.get("node_prefix") is not None:
            raise ValueError("TPU fleet workers cannot set provider_options.node_prefix")
        # A known shape fixes accelerator count. An unknown type is intentionally
        # allowed only after the complete manual host vector above has passed;
        # that keeps the catalog extensible without guessing discovered hardware.
        if shape is not None:
            for field_name, resources in (("resources", self.resources), ("allocatable", allocatable)):
                if resources.tpu != shape.schedulable_tpu:
                    raise ValueError(
                        f"TPU node type {self.accelerator_type!r} requires {field_name}.tpu == {shape.schedulable_tpu}"
                    )
        return self


NodeType = Annotated[CpuNodeType | GpuNodeType | TpuNodeType, Field(discriminator="kind")]


class EyvanFleetConfig(FleetConfigModel):
    """A Ray-up-style desired fleet containing one or more node pools."""

    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=63)
    provider: ProviderConfig
    agent_endpoint: str | None = Field(default=None, min_length=1)
    executor: str = Field(default="eyvan-native", min_length=1)
    capacity_class: CanonicalCapacityClass = CapacityClass.ON_DEMAND
    autoscaling: AutoscalingDefaults = Field(default_factory=AutoscalingDefaults)
    runtime: NativeRuntimeConfig = Field(default_factory=NativeRuntimeConfig)
    availability_policy: AvailabilityPolicy = Field(default_factory=AvailabilityPolicy)
    labels: dict[str, str] = Field(default_factory=dict)
    pools: dict[str, NodeType] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_pool_schema(cls, value: tp.Any) -> tp.Any:
        """Accept concise keyed pools and narrowly migrate one-slice node types."""

        if not is_json_object(value):
            return value
        normalized: JsonObject = dict(value)
        has_pools = "pools" in normalized
        has_node_types = "node_types" in normalized
        if has_pools and has_node_types:
            raise ValueError("fleet configuration must use pools or legacy node_types, not both")
        if has_pools:
            pools = normalized["pools"]
            if is_json_object(pools):
                expanded: JsonObject = {}
                for name, pool in pools.items():
                    for zoned_name, zoned in _expand_pool_zones(name, pool):
                        if zoned_name in expanded:
                            raise ValueError(f"pool {zoned_name!r} is declared twice after expanding zones")
                        expanded[zoned_name] = _normalize_public_pool(zoned_name, zoned)
                normalized["pools"] = expanded
        elif has_node_types:
            node_types = normalized.pop("node_types")
            if is_json_object(node_types):
                normalized["pools"] = {
                    name: _normalize_legacy_node_type(name, node_type) for name, node_type in node_types.items()
                }
            else:
                normalized["pools"] = node_types
        return normalized

    @property
    def node_types(self) -> dict[str, NodeType]:
        """Return the legacy read-only alias for :attr:`pools`."""

        return self.pools

    @model_validator(mode="after")
    def validate_names_and_provider(self) -> EyvanFleetConfig:
        validate_fleet_name(self.name)
        invalid = [name for name in self.node_types if not _NAME_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid node type name: {invalid[0]!r}")
        prefixes = [_cell_prefix(self.name, node_type) for node_type in self.node_types]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("fleet and node type names collide after cell ID normalization")
        if self.provider.type == "gcp":
            incomplete = [
                name
                for name, node_type in self.node_types.items()
                if not (node_type.provider_project or self.provider.project)
                or not (node_type.zone or self.provider.zone)
            ]
            if incomplete:
                raise ValueError(f"GCP node type {incomplete[0]!r} requires a provider project and zone")
        elif any(node_type.capacity_backend is None for node_type in self.node_types.values()):
            raise ValueError("non-GCP node types must set capacity_backend explicitly")
        endpoint_executors: dict[str, str] = {}
        # One cell agent serves exactly one capacity backend: CellAgentService reads
        # settings.capacity_backend once at construction. Two node types with different
        # backends behind one endpoint used to validate cleanly and then fail forever --
        # every capacity.apply for the losing pool is rejected inside the driver's own
        # _validate_desired and retried with backoff, with an opaque ValueError.
        endpoint_capacity_backends: dict[str, str] = {}
        for name, node_type in self.node_types.items():
            endpoint = node_type.agent_endpoint or self.agent_endpoint
            endpoint_key = endpoint or "<platform-default-agent>"
            executor = canonical_execution_backend(node_type.executor or self.executor)
            capacity_backend = (node_type.capacity_backend or node_type.default_capacity_backend).strip().lower()
            capacity_backend = capacity_backend.replace("_", "-")
            validate_accelerator_write_boundary(
                provider=self.provider.type,
                capacity_backend=capacity_backend,
                accelerator_type=node_type.cell_accelerator_type,
                capacity=node_type.resources,
                allocatable=node_type.allocatable or node_type.resources,
                boundary=f"fleet node type {name!r}",
            )
            native_legacy_tpu = capacity_backend in {"gcp-tpu", "tpu"} or (
                capacity_backend == "gcp" and node_type.default_capacity_backend == "gcp-tpu"
            )
            compute_capacity = capacity_backend in {"gcp-compute", "compute"} or (
                capacity_backend == "gcp" and node_type.default_capacity_backend == "gcp-compute"
            )
            if executor == "eyvan-native" and (compute_capacity or native_legacy_tpu):
                options = node_type.compiled_provider_options
                if not isinstance(options.get("bootstrap_egress"), str) or not options["bootstrap_egress"].strip():
                    raise ValueError(
                        f"node type {name!r} requires provider_options.bootstrap_egress for eyvan-native bootstrap "
                        f"(one of: {bootstrap_egress_choices()})"
                    )
                private_option = "internal_ip_only" if compute_capacity else "internal_ips"
                egress = options["bootstrap_egress"].lower().replace("_", "-")
                if options.get(private_option, False) and egress == "public-internet":
                    raise ValueError(
                        f"node type {name!r} cannot combine provider_options.{private_option}=true with "
                        "bootstrap_egress=public-internet; set it to false for a public IP, or keep it true "
                        "and select cloud-nat (requires configured NAT) or private-google-access "
                        "(requires staged dependencies)"
                    )
            runtime = self._resolved_runtime(node_type)
            if not runtime.is_empty and executor != "eyvan-native":
                raise ValueError(f"node type {name!r} runtime requires executor eyvan-native")
            if not runtime.is_empty and node_type.provider_options.get("startup_script") is not None:
                raise ValueError("use runtime.init_commands instead of provider_options.startup_script")
            if runtime.vm_image is not None:
                if self.provider.type != "gcp" or not compute_capacity:
                    raise ValueError(f"node type {name!r} runtime.vm_image requires the gcp-compute capacity backend")
                supplied_image = next(
                    (key for key in ("image", "image_family", "image_project") if key in node_type.provider_options),
                    None,
                )
                if supplied_image is not None:
                    raise ValueError(
                        f"node type {name!r} cannot combine runtime.vm_image with provider_options.{supplied_image}"
                    )
            previous = endpoint_executors.setdefault(endpoint_key, executor)
            if previous != executor:
                raise ValueError(
                    f"node type {name!r} selects executor {executor!r}, but agent endpoint {endpoint_key!r} "
                    f"already serves {previous!r}"
                )
            backend_family = capacity_backend_agent_family(capacity_backend)
            previous_backend = endpoint_capacity_backends.setdefault(endpoint_key, backend_family)
            if previous_backend != backend_family:
                raise ValueError(
                    f"node type {name!r} selects capacity backend {capacity_backend!r}, but agent endpoint "
                    f"{endpoint_key!r} already serves {previous_backend!r}; give each capacity backend its "
                    "own cell agent endpoint"
                )
        quota_records: list[CapacityPoolRecord] = []
        for pool in self.pool_payloads():
            # Fleet YAML may defer agent routing to the platform. The quota
            # contract does not depend on that endpoint, but the durable pool
            # record still requires one after API resolution.
            quota_payload = dict(pool)
            quota_payload["agent_endpoint"] = quota_payload["agent_endpoint"] or "http://platform-agent.invalid"
            quota_records.append(CapacityPoolRecord.model_validate(quota_payload))
        validate_capacity_pool_quota_groups(quota_records)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> EyvanFleetConfig:
        """Load and validate a fleet configuration from YAML."""

        config_path = Path(path).expanduser()
        with config_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not is_json_object(raw):
            raise ValueError("fleet configuration must be a YAML mapping")
        return cls.model_validate(raw)

    def desired_cells(self) -> list[JsonObject]:
        """Compile initial capacity into deterministic cell API payloads."""

        cells: list[JsonObject] = []
        for node_type_name, node_type in sorted(self.node_types.items()):
            template = self._cell_template(node_type_name, node_type)
            for index in range(1, node_type.desired_initial_cells + 1):
                cells.append({"id": f"{_cell_prefix(self.name, node_type_name)}-{index:04d}", **template})
        return cells

    def pool_payloads(self) -> list[JsonObject]:
        """Compile reusable capacity-pool payloads for an autoscaling API."""

        pools: list[JsonObject] = []
        for node_type_name, node_type in sorted(self.node_types.items()):
            template = self._cell_template(node_type_name, node_type)
            pools.append(
                {
                    "id": f"{self.name}-{node_type_name}",
                    "fleet_name": self.name,
                    "node_type": node_type_name,
                    "provider": template["provider"],
                    "capacity_backend": template["capacity_backend"],
                    "provider_project": template["provider_project"],
                    "region": template["region"],
                    "zone": template["zone"],
                    "capacity": template["capacity"],
                    "allocatable": template["allocatable"],
                    "unit_prices": template["unit_prices"],
                    "accelerator_type": template["accelerator_type"],
                    "capacity_class": template["capacity_class"],
                    "executor": template["executor"],
                    "agent_endpoint": template["endpoint"],
                    "provider_options": template["provider_options"],
                    "runtime": template["runtime"],
                    "availability_policy": template["availability_policy"],
                    "max_hold_s": node_type.max_hold_s,
                    "labels": template["labels"],
                    "topology": template["topology"],
                    "max_replica_resources": template["max_replica_resources"],
                    "max_gang_replicas": template["max_gang_replicas"],
                    # Only when enabled, so a compiled pool document is unchanged
                    # for every fleet that does not opt in.
                    **({"chip_sharing": True} if node_type.chip_sharing else {}),
                    "quota_group": node_type.quota_group,
                    "quota_limit": node_type.quota_limit,
                    "quota_units_per_cell": node_type.quota_units_per_cell,
                    "min_cells": node_type.min_cells,
                    "warm_windows": [model_json(window) for window in node_type.warm_windows],
                    "max_cells": node_type.max_cells,
                    "initial_cells": node_type.desired_initial_cells,
                    "idle_timeout_s": node_type.idle_timeout_s
                    if node_type.idle_timeout_s is not None
                    else self.autoscaling.idle_timeout_s,
                    "queue_timeout_s": node_type.queue_timeout_s
                    if node_type.queue_timeout_s is not None
                    else self.autoscaling.queue_timeout_s,
                    "provisioning_timeout_s": node_type.provisioning_timeout_s
                    if node_type.provisioning_timeout_s is not None
                    else self.autoscaling.provisioning_timeout_s,
                    "upscaling_speed": node_type.upscaling_speed
                    if node_type.upscaling_speed is not None
                    else self.autoscaling.upscaling_speed,
                    "launch_batch": node_type.launch_batch
                    if node_type.launch_batch is not None
                    else self.autoscaling.launch_batch,
                    "enabled": True,
                }
            )
        return pools

    def _cell_template(self, node_type_name: str, node_type: NodeType) -> JsonObject:
        resources = model_json(node_type.resources)
        allocatable = model_json(node_type.allocatable or node_type.resources)
        runtime = self._resolved_runtime(node_type)
        capacity_class = node_type.capacity_class
        if capacity_class is None and node_type.kind == "tpu" and "capacity_class" not in self.model_fields_set:
            # TPU pools use Spot unless an operator explicitly sets a pool or
            # fleet-wide purchase class. CPU/GPU preserve the fleet default.
            capacity_class = CapacityClass.SPOT
        labels = {
            **self.labels,
            **node_type.labels,
            "eyvan_fleet": self.name,
            "eyvan_pool": f"{self.name}-{node_type_name}",
            "eyvan_node_type": node_type_name,
            "eyvan_resource_kind": node_type.kind,
            "eyvan_managed_by": "fleet-config",
        }
        executor = canonical_execution_backend(node_type.executor or self.executor)
        max_gang_replicas = node_type.max_gang_replicas
        if executor in {"native-local", "eyvan-native"} and max_gang_replicas is None:
            max_gang_replicas = 1
        return {
            "provider": self.provider.type,
            "capacity_backend": node_type.capacity_backend or node_type.default_capacity_backend,
            "provider_project": node_type.provider_project or self.provider.project,
            "region": node_type.region or self.provider.region,
            "zone": node_type.zone or self.provider.zone,
            "capacity": resources,
            "allocatable": allocatable,
            "unit_prices": _json_object(node_type.unit_prices),
            "accelerator_type": node_type.cell_accelerator_type,
            "capacity_class": (capacity_class or self.capacity_class).value,
            "executor": executor,
            "endpoint": node_type.agent_endpoint or self.agent_endpoint,
            "pool_id": f"{self.name}-{node_type_name}",
            "provider_options": self._compiled_provider_options(node_type, runtime),
            "runtime": model_json(runtime),
            "availability_policy": model_json(self._resolved_availability_policy(node_type)),
            "labels": _json_object(labels),
            "topology": _json_object(node_type.topology),
            "max_replica_resources": (
                model_json(node_type.max_replica_resources) if node_type.max_replica_resources is not None else None
            ),
            "max_gang_replicas": max_gang_replicas,
        }

    def _resolved_availability_policy(self, node_type: NodeType) -> AvailabilityPolicy:
        """Return the pool's own curve, or the fleet's when it declares none."""

        return node_type.availability_policy or self.availability_policy

    def _resolved_runtime(self, node_type: NodeType) -> NativeRuntimeConfig:
        """Return fleet defaults overlaid by one node-type runtime."""

        return self.runtime.merged(node_type.runtime)

    @staticmethod
    def _compiled_provider_options(
        node_type: NodeType,
        runtime: NativeRuntimeConfig,
    ) -> JsonObject:
        """Compile host-image intent into the provider's existing VM option."""

        options = node_type.compiled_provider_options
        if runtime.vm_image is None:
            return options
        # A v5p-8 Compute profile contributes an image family automatically;
        # runtime.vm_image deliberately replaces that generated default.
        for key in ("image", "image_family", "image_project"):
            options.pop(key, None)
        options["image"] = runtime.vm_image
        return options


def validate_generated_fleet(document: JsonObject) -> EyvanFleetConfig:
    """Validate a fleet document Eyvan wrote, without the advice for hand-written ones.

    ``max_workers: 0`` earns a warning above, and that warning is good: it tells
    somebody writing YAML by hand that they have asked for an unbounded pool and
    how to say so deliberately. It has no audience when Eyvan generated the
    document itself -- from the operator's own autoscale profile -- and it
    arrives as a pydantic stack line printed through the middle of a rendered
    table, advice about a file the reader never opened.

    Scoped to documents this package produced. A file a person wrote still gets
    the warning on the path that reads it, which is the only place it was for.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="max_workers: 0 means unbounded", category=UserWarning)
        return EyvanFleetConfig.model_validate(document)


__all__ = (
    "AutoscalingDefaults",
    "CpuNodeType",
    "EyvanFleetConfig",
    "GpuNodeType",
    "NodeType",
    "ProviderConfig",
    "TpuNodeType",
    "validate_fleet_name",
)
