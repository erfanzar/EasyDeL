# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Immutable domain objects for Eyvan scheduling decisions."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import IntEnum

from eyvan._internal.compat import StrEnum

Scalar: tp.TypeAlias = int | float | Decimal
StringPairs: tp.TypeAlias = tp.Mapping[str, str] | tp.Iterable[tuple[str, str]]


def _as_decimal(value: Scalar, *, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite decimal value") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _non_empty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, trimmed string")
    return value


def _canonical_pairs(values: StringPairs, *, field_name: str) -> tuple[tuple[str, str], ...]:
    items = values.items() if isinstance(values, tp.Mapping) else values
    result: dict[str, str] = {}
    for key, value in items:
        key = _non_empty(key, field_name=f"{field_name} key")
        value = _non_empty(value, field_name=f"{field_name}[{key!r}]")
        if key in result and result[key] != value:
            raise ValueError(f"{field_name} contains conflicting values for {key!r}")
        result[key] = value
    return tuple(sorted(result.items()))


@dataclass(frozen=True, slots=True, init=False)
class ResourceVector:
    """A canonical, non-negative vector of named scalar resources.

    Values are stored as :class:`~decimal.Decimal` so quota, budget, and
    fairness decisions do not depend on binary floating-point rounding.

    Args:
        values: Optional mapping or iterable of ``(resource, quantity)`` pairs.
        **resources: Additional named resource quantities.
    """

    _items: tuple[tuple[str, Decimal], ...]

    def __init__(
        self,
        values: tp.Mapping[str, Scalar] | tp.Iterable[tuple[str, Scalar]] | None = None,
        /,
        **resources: Scalar,
    ) -> None:
        source = values.items() if isinstance(values, tp.Mapping) else (values or ())
        normalized: dict[str, Decimal] = {}
        for name, value in source:
            self._insert(normalized, name, value)
        for name, value in resources.items():
            if name in normalized:
                raise ValueError(f"resource {name!r} was provided more than once")
            self._insert(normalized, name, value)
        object.__setattr__(self, "_items", tuple(sorted(normalized.items())))

    @staticmethod
    def _insert(target: dict[str, Decimal], name: str, value: Scalar) -> None:
        name = _non_empty(name, field_name="resource name")
        if name in target:
            raise ValueError(f"resource {name!r} was provided more than once")
        quantity = _as_decimal(value, field_name=f"resource {name!r}")
        if quantity < 0:
            raise ValueError(f"resource {name!r} must be non-negative")
        target[name] = quantity

    def __iter__(self) -> tp.Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return any(value > 0 for _, value in self._items)

    def __getitem__(self, name: str) -> Decimal:
        for candidate, value in self._items:
            if candidate == name:
                return value
        raise KeyError(name)

    def get(self, name: str, default: Scalar = 0) -> Decimal:
        """Return one resource quantity, or ``default`` when absent."""
        try:
            return self[name]
        except KeyError:
            return _as_decimal(default, field_name="resource default")

    def items(self) -> tuple[tuple[str, Decimal], ...]:
        """Return the canonical resource items."""
        return self._items

    def as_dict(self) -> dict[str, Decimal]:
        """Return a mutable copy of the resource mapping."""
        return dict(self._items)

    def __add__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        names = set(self) | set(other)
        return ResourceVector((name, self.get(name) + other.get(name)) for name in names)

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        names = set(self) | set(other)
        differences = {name: self.get(name) - other.get(name) for name in names}
        negative = sorted(name for name, value in differences.items() if value < 0)
        if negative:
            raise ValueError(f"resource subtraction would make {', '.join(negative)} negative")
        return ResourceVector(differences)

    def scale(self, multiplier: Scalar) -> ResourceVector:
        """Return a vector with every quantity multiplied by ``multiplier``."""
        factor = _as_decimal(multiplier, field_name="resource multiplier")
        if factor < 0:
            raise ValueError("resource multiplier must be non-negative")
        return ResourceVector((name, value * factor) for name, value in self._items)

    def fits_within(self, capacity: ResourceVector) -> bool:
        """Return whether every requested quantity exists in ``capacity``."""
        return all(value <= capacity.get(name) for name, value in self._items)

    def within_limits(self, limits: ResourceVector) -> bool:
        """Return whether quantities constrained by ``limits`` stay below them.

        A missing limit means that resource is unconstrained. This differs from
        :meth:`fits_within`, where a missing capacity means zero capacity.
        """
        return all(self.get(name) <= limit for name, limit in limits.items())

    def dominant_share(self, denominator: ResourceVector) -> Decimal:
        """Return the largest resource share relative to ``denominator``."""
        if not self:
            return Decimal(0)
        shares: list[Decimal] = []
        for name, quantity in self._items:
            if quantity == 0:
                continue
            capacity = denominator.get(name)
            if capacity <= 0:
                return Decimal("Infinity")
            shares.append(quantity / capacity)
        return max(shares, default=Decimal(0))

    def dot(self, prices: ResourceVector) -> Decimal:
        """Return the scalar product with per-resource ``prices``."""
        return sum((quantity * prices.get(name) for name, quantity in self._items), Decimal(0))

    @classmethod
    def total(cls, vectors: tp.Iterable[ResourceVector]) -> ResourceVector:
        """Sum an iterable of vectors."""
        result = cls()
        for vector in vectors:
            result += vector
        return result


class PriorityBand(IntEnum):
    """Strict workload priority bands, ordered from lowest to highest."""

    BATCH = 0
    INTERACTIVE = 1
    PRODUCTION = 2


class CapacityClass(StrEnum):
    """Commercial and interruption semantics of a capacity pool."""

    RESERVED = "reserved"
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    #: Queued until the provider can satisfy it, then held for a bounded time
    #: and returned whether or not the job is finished.
    #:
    #: The fourth thing capacity can be, and neither of the other three: unlike
    #: spot it is not taken back mid-run, and unlike on-demand it does not
    #: arrive immediately or stay indefinitely. GCP sells it as Dynamic Workload
    #: Scheduler flex-start; the name here describes the *shape* -- a wait, then
    #: a window -- because a provider's product name in the scheduler is the
    #: leak this project keeps having to remove.
    FLEX_START = "flex_start"

    @classmethod
    def parse(cls, value: CapacityClass | str) -> CapacityClass:
        """Parse a boundary value into its stable canonical representation.

        ``on-demand``, ``on demand``, and ``ondemand`` are accepted aliases;
        serialization always emits ``on_demand``.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("capacity class must be a string or CapacityClass")
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == "ondemand":
            normalized = cls.ON_DEMAND.value
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(candidate.value for candidate in cls)
            raise ValueError(f"unknown capacity class {value!r}; expected one of: {choices}") from exc


class WorkerResourceScope(StrEnum):
    """Describe whether a workload gang is one worker or one complete atomic cell."""

    PER_WORKER = "per_worker"
    ATOMIC_CELL = "atomic_cell"


class PlacementMode(StrEnum):
    """How a workload's workers may be distributed over capacity cells.

    ATOMIC is the exact-shape contract: one cell whose worker count equals the
    workload's, one provider resource, one hardware world. ELASTIC is the
    node-pool contract: workers are packed from any combination of cells of the
    right accelerator type -- warm cells first, provisioning only the deficit --
    and the workload runs as one lease group with one hardware world per cell.

    ELASTIC requires ``WorkerResourceScope.PER_WORKER``; a legacy gang that
    encodes a whole atomic cell has no per-worker shape to pack from.
    """

    ATOMIC = "atomic"
    ELASTIC = "elastic"


class CheckpointMode(StrEnum):
    """Checkpoint behavior available to a workload."""

    NONE = "none"
    ON_PREEMPTION = "on_preemption"
    PERIODIC = "periodic"


class PreemptionActionKind(StrEnum):
    """Action required before reclaiming a victim lease."""

    CHECKPOINT_AND_STOP = "checkpoint_and_stop"
    TERMINATE = "terminate"


class InfeasibilityCode(StrEnum):
    """Stable scheduler rejection categories."""

    NO_CELLS = "no_cells"
    HARD_CONSTRAINT = "hard_constraint"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    CAPACITY_CLASS_MISMATCH = "capacity_class_mismatch"
    RUNTIME_UNSUPPORTED = "runtime_unsupported"
    REPLICA_COUNT_UNSUPPORTED = "replica_count_unsupported"
    REPLICA_TOO_LARGE = "replica_too_large"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    QUOTA_EXCEEDED = "quota_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNPRICED_CELL = "unpriced_cell"
    PREEMPTION_UNSAFE = "preemption_unsafe"


@dataclass(frozen=True, slots=True)
class IdentityScope:
    """A tenant, project, or principal scope in the identity hierarchy."""

    tenant_id: str
    project_id: str | None = None
    principal_id: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.tenant_id, field_name="tenant_id")
        if self.project_id is not None:
            _non_empty(self.project_id, field_name="project_id")
        if self.principal_id is not None:
            _non_empty(self.principal_id, field_name="principal_id")
            if self.project_id is None:
                raise ValueError("principal scope requires a project_id")

    @property
    def depth(self) -> int:
        """Return hierarchy depth: tenant=1, project=2, principal=3."""
        return 1 + int(self.project_id is not None) + int(self.principal_id is not None)

    def contains(self, identity: WorkloadIdentity) -> bool:
        """Return whether ``identity`` belongs to this scope."""
        return (
            self.tenant_id == identity.tenant_id
            and (self.project_id is None or self.project_id == identity.project_id)
            and (self.principal_id is None or self.principal_id == identity.principal_id)
        )


@dataclass(frozen=True, slots=True)
class WorkloadIdentity:
    """Fully qualified identity charged for a workload."""

    tenant_id: str
    project_id: str
    principal_id: str

    def __post_init__(self) -> None:
        _non_empty(self.tenant_id, field_name="tenant_id")
        _non_empty(self.project_id, field_name="project_id")
        _non_empty(self.principal_id, field_name="principal_id")

    @property
    def scopes(self) -> tuple[IdentityScope, IdentityScope, IdentityScope]:
        """Return tenant, project, and principal scopes in order."""
        return (
            IdentityScope(self.tenant_id),
            IdentityScope(self.tenant_id, self.project_id),
            IdentityScope(self.tenant_id, self.project_id, self.principal_id),
        )


@dataclass(frozen=True, slots=True)
class Quota:
    """Resource limits and fair-share weight for one identity scope."""

    scope: IdentityScope
    limit: ResourceVector
    weight: Decimal = Decimal(1)

    def __post_init__(self) -> None:
        weight = _as_decimal(self.weight, field_name="quota weight")
        if weight <= 0:
            raise ValueError("quota weight must be positive")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class Budget:
    """Monetary and accelerator-hour budgets for one identity scope.

    Accelerator-hour vectors are keyed by the resource names used in workload
    requests, for example ``tpu`` or ``gpu_h100``. Unlisted resource names are
    not constrained by this budget.
    """

    scope: IdentityScope
    limit: Decimal
    spent: Decimal = Decimal(0)
    currency: str = "USD"
    accelerator_hour_limits: ResourceVector = field(default_factory=ResourceVector)
    accelerator_hours_spent: ResourceVector = field(default_factory=ResourceVector)
    hard_limit: bool = True

    def __post_init__(self) -> None:
        limit = _as_decimal(self.limit, field_name="budget limit")
        spent = _as_decimal(self.spent, field_name="budget spent")
        if limit < 0 or spent < 0:
            raise ValueError("budget values must be non-negative")
        currency = _non_empty(self.currency, field_name="budget currency").upper()
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "spent", spent)
        object.__setattr__(self, "currency", currency)

    @property
    def remaining(self) -> Decimal:
        """Return unspent budget, clamped at zero."""
        return max(self.limit - self.spent, Decimal(0))


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Controls whether and how a workload may be safely preempted."""

    mode: CheckpointMode = CheckpointMode.NONE
    grace_period_seconds: int = 0
    checkpoint_interval_seconds: int | None = None
    allow_forced_termination: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CheckpointMode(self.mode))
        if self.grace_period_seconds < 0:
            raise ValueError("grace_period_seconds must be non-negative")
        if self.checkpoint_interval_seconds is not None and self.checkpoint_interval_seconds <= 0:
            raise ValueError("checkpoint_interval_seconds must be positive")
        if self.mode is CheckpointMode.PERIODIC and self.checkpoint_interval_seconds is None:
            raise ValueError("periodic checkpointing requires checkpoint_interval_seconds")

    @property
    def checkpointable(self) -> bool:
        """Return whether a preemption checkpoint can be requested."""
        return self.mode is not CheckpointMode.NONE


@dataclass(frozen=True, slots=True)
class ResourceGang:
    """An all-or-nothing group of identical resource replicas."""

    gang_id: str
    replicas: int
    resources_per_replica: ResourceVector
    required_topology: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.gang_id, field_name="gang_id")
        if self.replicas <= 0:
            raise ValueError("gang replicas must be positive")
        if not self.resources_per_replica:
            raise ValueError("gang resources_per_replica must not be empty")
        object.__setattr__(
            self,
            "required_topology",
            _canonical_pairs(self.required_topology, field_name="required_topology"),
        )

    @property
    def total_resources(self) -> ResourceVector:
        """Return resources required by all replicas."""
        return self.resources_per_replica.scale(self.replicas)


_ALL_CAPACITY_CLASSES = frozenset(CapacityClass)
_DEFAULT_CAPACITY_PREFERENCE = (
    CapacityClass.RESERVED,
    CapacityClass.ON_DEMAND,
    CapacityClass.SPOT,
)


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """A globally schedulable workload and its immutable policy inputs."""

    workload_id: str
    identity: WorkloadIdentity
    gangs: tuple[ResourceGang, ...]
    priority: PriorityBand = PriorityBand.INTERACTIVE
    attempt: int = 1
    generation: int = 1
    queue_position: int = 0
    runtime: str | None = None
    required_labels: tuple[tuple[str, str], ...] = ()
    preferred_labels: tuple[tuple[str, str], ...] = ()
    allowed_label_values: tuple[tuple[str, tuple[str, ...]], ...] = ()
    denied_label_values: tuple[tuple[str, tuple[str, ...]], ...] = ()
    preferred_cells: tuple[str, ...] = ()
    allowed_capacity_classes: frozenset[CapacityClass] = field(default_factory=lambda: _ALL_CAPACITY_CLASSES)
    capacity_class_preference: tuple[CapacityClass, ...] = _DEFAULT_CAPACITY_PREFERENCE
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    estimated_duration_seconds: int = 3600
    revocable: bool = True
    cell_group_count: int = 1
    worker_count: int = 1
    worker_resource_scope: WorkerResourceScope = WorkerResourceScope.ATOMIC_CELL
    placement: PlacementMode = PlacementMode.ATOMIC
    allowed_label_combinations: tuple[tuple[tuple[str, str], ...], ...] = ()
    #: A share of one TPU host: this many chips, optionally exactly these
    #: global indexes. ``None`` is a whole-cell workload, as before.
    tpu_chip_count: int | None = None
    tpu_chip_indexes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _non_empty(self.workload_id, field_name="workload_id")
        if self.tpu_chip_count is not None:
            if self.tpu_chip_count <= 0:
                raise ValueError("tpu_chip_count must be positive")
            if self.worker_count != 1 or self.cell_group_count != 1 or self.placement is not PlacementMode.ATOMIC:
                raise ValueError("a TPU chip workload is one atomic single-worker placement")
            if self.tpu_chip_indexes is not None and len(self.tpu_chip_indexes) != self.tpu_chip_count:
                raise ValueError("tpu_chip_indexes must name exactly tpu_chip_count chips")
        elif self.tpu_chip_indexes is not None:
            raise ValueError("tpu_chip_indexes require tpu_chip_count")
        if self.runtime is not None:
            _non_empty(self.runtime, field_name="runtime")
        object.__setattr__(self, "placement", PlacementMode(self.placement))
        if not self.gangs:
            raise ValueError("workload must contain at least one resource gang")
        gang_ids = [gang.gang_id for gang in self.gangs]
        if len(gang_ids) != len(set(gang_ids)):
            raise ValueError("workload gang_id values must be unique")
        if self.attempt <= 0:
            raise ValueError("workload attempt must be positive")
        if self.generation <= 0:
            raise ValueError("workload generation must be positive")
        if self.queue_position < 0:
            raise ValueError("queue_position must be non-negative")
        if self.estimated_duration_seconds <= 0:
            raise ValueError("estimated_duration_seconds must be positive")
        if self.cell_group_count <= 0:
            raise ValueError("cell_group_count must be positive")
        if isinstance(self.worker_count, bool) or not isinstance(self.worker_count, int) or self.worker_count <= 0:
            raise ValueError("worker_count must be a positive integer")

        object.__setattr__(self, "priority", PriorityBand(self.priority))
        object.__setattr__(self, "worker_resource_scope", WorkerResourceScope(self.worker_resource_scope))
        object.__setattr__(self, "required_labels", _canonical_pairs(self.required_labels, field_name="required_labels"))
        object.__setattr__(
            self,
            "preferred_labels",
            _canonical_pairs(self.preferred_labels, field_name="preferred_labels"),
        )
        object.__setattr__(
            self,
            "allowed_label_values",
            self._canonical_label_sets(self.allowed_label_values, field_name="allowed_label_values"),
        )
        object.__setattr__(
            self,
            "denied_label_values",
            self._canonical_label_sets(self.denied_label_values, field_name="denied_label_values"),
        )

        object.__setattr__(
            self,
            "allowed_label_combinations",
            tuple(
                sorted(
                    {
                        _canonical_pairs(combination, field_name="allowed_label_combinations")
                        for combination in self.allowed_label_combinations
                    }
                )
            ),
        )

        preferred_cells = tuple(self.preferred_cells)
        if len(preferred_cells) != len(set(preferred_cells)):
            raise ValueError("preferred_cells must not contain duplicates")
        for cell_id in preferred_cells:
            _non_empty(cell_id, field_name="preferred cell id")
        object.__setattr__(self, "preferred_cells", preferred_cells)

        allowed = frozenset(CapacityClass.parse(value) for value in self.allowed_capacity_classes)
        if not allowed:
            raise ValueError("allowed_capacity_classes must not be empty")
        preference = tuple(CapacityClass.parse(value) for value in self.capacity_class_preference)
        if len(preference) != len(set(preference)):
            raise ValueError("capacity_class_preference must not contain duplicates")
        effective_preference = tuple(value for value in preference if value in allowed)
        effective_preference += tuple(sorted(allowed - set(effective_preference), key=lambda value: value.value))
        object.__setattr__(self, "allowed_capacity_classes", allowed)
        object.__setattr__(self, "capacity_class_preference", effective_preference)

    @staticmethod
    def _canonical_label_sets(
        values: tp.Iterable[tuple[str, tp.Iterable[str]]],
        *,
        field_name: str,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        normalized: dict[str, tuple[str, ...]] = {}
        for key, candidates in values:
            key = _non_empty(key, field_name=f"{field_name} key")
            items = tuple(sorted({_non_empty(item, field_name=f"{field_name}[{key!r}]") for item in candidates}))
            if not items:
                raise ValueError(f"{field_name}[{key!r}] must not be empty")
            if key in normalized and normalized[key] != items:
                raise ValueError(f"{field_name} contains conflicting values for {key!r}")
            normalized[key] = items
        return tuple(sorted(normalized.items()))

    @property
    def total_resources(self) -> ResourceVector:
        """Return aggregate resources for every gang."""
        return ResourceVector.total(gang.total_resources for gang in self.gangs)

    @property
    def allocation_resources(self) -> ResourceVector:
        """Return resources reserved by one physical capacity cell.

        Legacy records encode the whole atomic cell directly in ``gang``.
        New TPU multislice records encode one complete worker and multiply that
        typed worker shape only for capacity, quota, and cost accounting.
        """

        resources = self.total_resources
        if self.worker_resource_scope is WorkerResourceScope.PER_WORKER:
            return resources.scale(self.worker_count)
        return resources


@dataclass(frozen=True, slots=True)
class CapacityCell:
    """A failure-isolated placement cell with one allocatable resource pool.

    ``autoscale_request_workload_id`` reserves freshly requested capacity for
    the workload that caused the autoscaler to create it. The reservation is
    cleared when its first lease is committed, after which the cell is normal
    fleet capacity.
    """

    cell_id: str
    capacity: ResourceVector
    capacity_class: CapacityClass
    labels: tuple[tuple[str, str], ...] = ()
    topology: tuple[tuple[str, str], ...] = ()
    runtimes: frozenset[str] = field(default_factory=lambda: frozenset({"eyvan-native"}))
    unit_prices: ResourceVector = field(default_factory=ResourceVector)
    route_priority: int = 0
    startup_latency_seconds: int = 0
    max_replica_resources: ResourceVector | None = None
    max_gang_replicas: int | None = None
    exclusive_workloads: bool = False
    autoscale_request_workload_id: str | None = None
    enabled: bool = True
    worker_count: int = 1
    #: Whether chip-sized workloads may share this (otherwise exclusive) cell,
    #: each holding disjoint TPU chips. Needs ``chips_per_host``.
    chip_sharing: bool = False
    chips_per_host: int | None = None

    def __post_init__(self) -> None:
        _non_empty(self.cell_id, field_name="cell_id")
        if self.chips_per_host is not None and self.chips_per_host <= 0:
            raise ValueError("chips_per_host must be positive")
        if not self.capacity:
            raise ValueError("cell capacity must not be empty")
        object.__setattr__(self, "capacity_class", CapacityClass.parse(self.capacity_class))
        object.__setattr__(self, "labels", _canonical_pairs(self.labels, field_name="cell labels"))
        object.__setattr__(self, "topology", _canonical_pairs(self.topology, field_name="cell topology"))
        runtimes = frozenset(_non_empty(runtime, field_name="cell runtime") for runtime in self.runtimes)
        if not runtimes:
            raise ValueError("cell runtimes must not be empty")
        object.__setattr__(self, "runtimes", runtimes)
        if self.route_priority < 0:
            raise ValueError("route_priority must be non-negative")
        if self.startup_latency_seconds < 0:
            raise ValueError("startup_latency_seconds must be non-negative")
        if isinstance(self.worker_count, bool) or not isinstance(self.worker_count, int) or self.worker_count <= 0:
            raise ValueError("worker_count must be a positive integer")
        if self.max_replica_resources is not None and not self.max_replica_resources.fits_within(self.capacity):
            raise ValueError("max_replica_resources must fit within cell capacity")
        if self.max_gang_replicas is not None and self.max_gang_replicas <= 0:
            raise ValueError("max_gang_replicas must be positive")
        if self.autoscale_request_workload_id is not None:
            _non_empty(self.autoscale_request_workload_id, field_name="autoscale_request_workload_id")

    def hourly_cost(self, resources: ResourceVector) -> Decimal:
        """Return this cell's hourly price for ``resources``."""
        return resources.dot(self.unit_prices)

    @property
    def declares_prices(self) -> bool:
        """Return whether the operator declared a price model at all.

        A ``unit_prices`` vector with no entries means pricing is unknown,
        not free. Declared zeros are knowledge: they make the model known
        even though every price on it is zero.
        """
        return len(self.unit_prices._items) > 0

    def prices_cover(self, resources: ResourceVector) -> bool:
        """Return whether every requested resource has a *declared* unit price.

        By contract a cell's ``unit_prices`` are the operator's complete
        statement of what the runtime charges for; unlisted resources are
        uncharged, so a partial vector still yields a deterministic cost
        estimate -- but not a complete one, which the scheduler surfaces as
        ``cost_unknown`` so fully priced cells win cost tie-breaks. Declared
        zeros count as covered; missing names do not (issue-register C10). A
        zero-quantity entry in the *request* is skipped: a resource the
        workload does not consume cannot make the cost unknown.
        """
        return all(name in self.unit_prices for name, quantity in resources.items() if quantity)


@dataclass(frozen=True, slots=True)
class CapacityLease:
    """Capacity reserved for one workload attempt."""

    lease_id: str
    workload_id: str
    attempt: int
    identity: WorkloadIdentity
    cell_id: str
    resources: ResourceVector
    priority: PriorityBand
    hourly_cost: Decimal
    reserved_cost: Decimal
    resource_hours_reserved: ResourceVector
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    revocable: bool = True
    fencing_epoch: int = 0
    #: Global TPU chip indexes held on a chip-sharing cell; empty means the whole cell.
    tpu_chips: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        chips = tuple(sorted(self.tpu_chips))
        if len(set(chips)) != len(chips) or any(chip < 0 for chip in chips):
            raise ValueError("lease tpu_chips must be unique non-negative indexes")
        object.__setattr__(self, "tpu_chips", chips)
        _non_empty(self.lease_id, field_name="lease_id")
        _non_empty(self.workload_id, field_name="workload_id")
        _non_empty(self.cell_id, field_name="cell_id")
        if self.attempt <= 0:
            raise ValueError("lease attempt must be positive")
        if not self.resources:
            raise ValueError("lease resources must not be empty")
        if self.fencing_epoch < 0:
            raise ValueError("fencing_epoch must be non-negative")
        object.__setattr__(self, "priority", PriorityBand(self.priority))
        hourly_cost = _as_decimal(self.hourly_cost, field_name="lease hourly_cost")
        reserved_cost = _as_decimal(self.reserved_cost, field_name="lease reserved_cost")
        if hourly_cost < 0 or reserved_cost < 0:
            raise ValueError("lease costs must be non-negative")
        object.__setattr__(self, "hourly_cost", hourly_cost)
        object.__setattr__(self, "reserved_cost", reserved_cost)


@dataclass(frozen=True, slots=True, order=True)
class PlacementScore:
    """An explainable lexicographic cell score; lower values are preferred.

    ``cost_unknown`` sits immediately before ``estimated_cost`` so a cell with
    undeclared prices loses the cost tie-break to every priced cell outright,
    and two unpriced cells tie into the later keys instead of both reading as
    free. Unknown is not zero.
    """

    preferred_cell_rank: int
    preferred_label_misses: int
    route_priority: int
    capacity_class_rank: int
    startup_latency_seconds: int
    cost_unknown: bool
    estimated_cost: Decimal
    fractional_waste: Decimal
    cell_id: str


@dataclass(frozen=True, slots=True)
class CellInfeasibility:
    """One cell-specific reason a workload could not be placed."""

    cell_id: str
    code: InfeasibilityCode
    message: str


@dataclass(frozen=True, slots=True)
class Infeasibility:
    """Structured explanation for a workload rejected in one scheduling pass."""

    workload_id: str
    code: InfeasibilityCode
    message: str
    cells: tuple[CellInfeasibility, ...] = ()


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    """A new lease that can be committed without preemption."""

    workload_id: str
    lease: CapacityLease
    score: PlacementScore
    member_leases: tuple[CapacityLease, ...] = ()

    def __post_init__(self) -> None:
        """Keep the leader lease distinct from every additional group member."""

        if any(member.lease_id == self.lease.lease_id for member in self.member_leases):
            raise ValueError("placement member leases must not repeat the leader lease")
        lease_ids = [self.lease.lease_id, *(member.lease_id for member in self.member_leases)]
        if len(lease_ids) != len(set(lease_ids)):
            raise ValueError("placement member leases must be unique")
        if any(member.workload_id != self.workload_id for member in self.member_leases):
            raise ValueError("placement member leases must belong to the placement workload")

    @property
    def all_leases(self) -> tuple[CapacityLease, ...]:
        """Return every cell reservation in this atomic placement."""

        return (self.lease, *self.member_leases)


@dataclass(frozen=True, slots=True)
class PreemptionVictim:
    """A lower-priority lease and the action required to reclaim it."""

    lease_id: str
    workload_id: str
    action: PreemptionActionKind
    grace_period_seconds: int


@dataclass(frozen=True, slots=True)
class PreemptionPlan:
    """A replacement lease plus ordered victim actions needed to realize it."""

    workload_id: str
    cell_id: str
    victims: tuple[PreemptionVictim, ...]
    replacement_lease: CapacityLease
    score: PlacementScore


@dataclass(frozen=True, slots=True)
class SchedulingResult:
    """Complete deterministic output from one scheduler reconciliation pass."""

    placements: tuple[PlacementDecision, ...]
    preemptions: tuple[PreemptionPlan, ...]
    infeasibilities: tuple[Infeasibility, ...]
    considered_order: tuple[str, ...]
