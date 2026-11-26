# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Transactional application service for Eyvan control-plane operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import typing as tp
import uuid
from dataclasses import replace
from datetime import datetime, timedelta

from pydantic import BaseModel

from eyvan._internal.identifiers import stable_id
from eyvan._internal.json_types import JsonObject, JsonValue, model_json
from eyvan.artifacts.managed import (
    CONTENT_ADDRESSED_KINDS,
    MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE,
    InvalidManagedArtifactURI,
    ManagedArtifactConflictError,
    ManagedArtifactDeleteInconclusiveError,
    ManagedArtifactDeleteStatus,
    ManagedArtifactInventoryError,
    ManagedArtifactInventoryStore,
    ManagedArtifactKind,
    ManagedArtifactObjectMetadata,
    ManagedArtifactOperationInconclusiveError,
    ManagedArtifactRef,
    ManagedArtifactVersionedStore,
    ManagedArtifactVersionMismatchError,
    ManagedArtifactWriteInconclusiveError,
    is_managed_uri,
    parse_managed_uri,
)
from eyvan.artifacts.managed import result_uri as managed_result_uri
from eyvan.capacity.drivers.base import backend_capabilities, providers_hosting_multi_worker_cells
from eyvan.capacity.planning import (
    POOL_ALTERNATE_TEMPLATES_LABEL,
    POOL_ALTERNATES_LABEL,
    POOL_LABEL,
    POOL_TEMPLATE_LABEL,
    candidate_pool_templates,
    candidate_zone_report,
    cell_matches_template,
    cell_provides_pool_supply,
)
from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type
from eyvan.domain import costing
from eyvan.domain.availability import p95_latency
from eyvan.domain.carbon import CarbonSource, CarbonUnavailable, NoCarbonSource, fresh_reading
from eyvan.domain.models import PlacementMode, SchedulingResult
from eyvan.domain.spend import spend_of
from eyvan.records import (
    BILLABLE_CELL_STATES,
    CALLABLE_SIGNING_KEY_ENV,
    DEFAULT_MAX_RETRIES,
    MAX_GCP_TPU_MULTISLICE_WORKERS,
    REMOTE_CALLABLE_SIGNATURE_ERROR,
    REMOTE_CALLABLE_SIGNING_KEY_ERROR,
    WORKER_COUNT_LABEL,
    ArtifactCollection,
    ArtifactHold,
    ArtifactInventoryEntry,
    ArtifactInventoryReport,
    ArtifactInventoryStatus,
    ArtifactReferenceRecord,
    ArtifactUsage,
    AuditEvent,
    AutoscaleBlockReason,
    BudgetRecord,
    BudgetStatus,
    CallableExecution,
    CandidateZone,
    CapacityClass,
    CapacityPoolRecord,
    CarbonReport,
    CellBillingAccrual,
    CellBillingReport,
    CellRecord,
    CellRejection,
    CellState,
    CommandExecution,
    CoordinationKind,
    EffectiveQuota,
    ExpansionAdmissionRecord,
    ExpansionAdmissionState,
    ExpansionMode,
    FederatedBudgetStatus,
    GcpQueueAdmissionDenialTarget,
    IdempotencyRecord,
    LaunchLatencyReport,
    LeaseRecord,
    LeaseState,
    ManagedArtifactRecord,
    ManagedArtifactState,
    OrganizationRecord,
    OutboxRecord,
    OutboxState,
    Permission,
    PolicyRecord,
    PoolEstimate,
    PoolLaunchLatency,
    PoolWarmth,
    Principal,
    PriorityBand,
    ProjectRecord,
    ProjectUsageReport,
    QueueTrend,
    QueueTrendPoint,
    QuotaRecord,
    RegionCarbon,
    ResourceGang,
    ResourceHours,
    ResourceLimits,
    Role,
    RoleGrant,
    RoleGrantRecord,
    ScopeQuota,
    ServiceDecisionKind,
    ServiceDecisionRecord,
    ServiceObservation,
    ServiceReplicaHealthState,
    SubjectUsageRecord,
    UserProgressReport,
    WarmthReport,
    WebhookRecord,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadDispatchReservation,
    WorkloadEstimate,
    WorkloadExplanation,
    WorkloadRecord,
    WorkloadSpendRecord,
    WorkloadState,
    WorkspaceRef,
    artifact_reference_id,
    expansion_admission_id,
    resource_quantities,
    utcnow,
    validate_capacity_pool_quota_groups,
)
from eyvan.secrets import matching_secret_reference_root
from eyvan.server.persistence.base import ReadOnlyQueryResult, Repository, RepositorySession
from eyvan.server.service.authz import Authorizer
from eyvan.server.service.batch import (
    DEFAULT_MAX_FAN_OUT,
    BatchRequest,
    child_idempotency_key,
    expand,
    task_index,
)
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.dag import DagNode
from eyvan.server.service.dag import validate as dag_validate
from eyvan.server.service.errors import (
    AuthorizationError,
    CapacityClassUnavailableError,
    ConflictError,
    InputError,
    NoMatchingCapacityPoolError,
    NotFoundError,
    UnavailableError,
)
from eyvan.server.service.expansion import (
    AdmissionPlan,
    ExpansionLimits,
    plan_expansion,
    quota_headroom_children,
)
from eyvan.server.service.federation import (
    BudgetGrantRecord,
    GrantState,
    PlanePeerRecord,
    QuotaGrantRecord,
    exceeds_quota_headroom,
    federated_spend,
    issuable_amount,
    outstanding_total,
    quota_headroom,
)
from eyvan.server.service.groups import MAX_GROUP_MEMBERS, GroupRole
from eyvan.server.service.groups import member_names as group_member_names
from eyvan.server.service.groups import validate as group_validate
from eyvan.server.service.locality import build_index as locality_index
from eyvan.server.service.locality import preferred_cells_for
from eyvan.server.service.pricing_resolver import accelerator_unit_price
from eyvan.server.service.scheduling import SchedulingService
from eyvan.server.service.services import (
    CHILD_REVISION,
    ServicePolicy,
    replica_health_state,
    scaled_service_metadata,
    service_metadata,
    service_policy,
)
from eyvan.server.service.state import (
    ACTIVE_LEASE_STATES,
    TERMINAL_WORKLOAD_STATES,
    InvalidTransition,
    ensure_lease_transition,
    ensure_workload_transition,
)
from eyvan.server.service.subjobs import (
    MAX_JOB_TREE_DEPTH,
    MAX_LIVE_CHILDREN,
    JobTreeTooDeepError,
    JobTreeTooWideError,
    collect_descendants,
    parent_depth,
    validate_parent,
)
from eyvan.server.service.webhooks import validate_target as validate_webhook_target
from eyvan.worker.runtime.node_health import HostFaultProof, parse_host_fault_marker

#: How much of a statement the audit ledger keeps. Enough to recognise the
#: query during a review, bounded so one pasted statement cannot dominate the
#: ledger.
_QUERY_AUDIT_STATEMENT_LIMIT = 4000


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _reserved_resource_hours(reserved: tp.Mapping[str, tp.Any]) -> ResourceHours:
    """Return one placement's reserved resource-hours as the durable vector.

    The scheduler speaks in a flat mapping and the lease record in a typed
    vector with a ``custom`` bag; this is the one conversion between them on the
    service side.
    """

    values = {name: float(quantity) for name, quantity in reserved.items()}
    known = {name: values.pop(name, 0.0) for name in ("cpu", "memory_gib", "gpu", "tpu")}
    return ResourceHours(**known, custom=values)


class WorkloadTransitionUpdates(tp.TypedDict, total=False):
    """Fields that may change alongside a workload state transition."""

    remote_id: str | None
    checkpoint_uri: str | None
    failure_reason: str | None
    preemption_id: str | None
    scheduling_reason: str | None
    autoscale_block_reason: AutoscaleBlockReason | None
    queue_admission_denial: GcpQueueAdmissionDenialTarget | None


def _fingerprint(value: BaseModel | tp.Mapping[str, JsonValue]) -> str:
    payload = model_json(value) if isinstance(value, BaseModel) else dict(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _flex_start_refusal(pool: CapacityPoolRecord, *, estimated_duration_seconds: int, checkpointable: bool) -> str:
    """Say why a job cannot run inside this pool's hold window, or nothing.

    Flex-start capacity arrives after a wait and is returned after a bounded
    window, finished or not. A job longer than the window will be interrupted:
    if it can checkpoint it resumes on the next grant, and if it cannot it is
    killed having produced nothing -- at hour four of a four-hour window, after
    the wait, which is the most expensive possible time to learn this.
    """

    if pool.max_hold_s is None or estimated_duration_seconds <= pool.max_hold_s:
        return ""
    if checkpointable:
        return ""
    return (
        f"this pool holds capacity for {pool.max_hold_s}s and the job estimates {estimated_duration_seconds}s; "
        "it would be stopped before finishing. Make it checkpointable so it resumes on the next grant, "
        "shorten the estimate, or use a pool that holds capacity until you release it"
    )


def _multislice_refusal(pool: CapacityPoolRecord) -> str:
    """Say why one pool cannot host several slices at once, or nothing if it can.

    The same conditions ``WorkloadDefinition`` enforces, checked against the pool
    a caller actually named so the refusal can point at it. Several workers
    allocated together needs a backend that allocates a gang of machines
    atomically, the native executor to place ranks across them, and accelerator
    chips to be worth doing.

    Asked of the capability table rather than of a provider's name. This used to
    read ``pool.provider != "gcp"``, which is the leak the durable model had
    already been cured of one layer down: a second cloud implementing atomic
    gang allocation would have passed the record validator and been refused
    here, by the submission path, for being the wrong brand.

    A pool naming a backend is judged on that backend, which is the precise
    question. One that names none falls back to the provider-level answer the
    spec validator uses, so pools predating ``capacity_backend`` keep working
    rather than losing multislice to a field they never set.
    """

    if pool.executor != "eyvan-native":
        return "it does not run the eyvan-native executor"
    builds_gangs = (
        backend_capabilities(pool.capacity_backend).multi_worker_cells
        if pool.capacity_backend
        else pool.provider in providers_hosting_multi_worker_cells()
    )
    if not builds_gangs:
        named = pool.capacity_backend or pool.provider
        return f"its {named} capacity backend allocates one machine per cell"
    if pool.allocatable.tpu <= 0 or pool.allocatable.gpu > 0:
        return "it provides no TPU resource"
    return ""


#: Label naming the pool a workload was placed against. Written to both the
#: spec's required labels and its metadata when a job is submitted to a pool.
_POOL_LABEL = "eyvan_pool"


def _declares_a_price(unit_prices: tp.Mapping[str, float]) -> bool:
    """Return whether a pool actually declares a positive price for anything."""

    return any(value > 0 for value in unit_prices.values())


def _on_demand_price(
    shape: str | None,
    pools: tp.Sequence[CapacityPoolRecord],
    supplied: float | None,
) -> tuple[float | None, str]:
    """Find an on-demand hourly price for ``shape``, or say why there is none.

    Eyvan carries no price catalogue and no spot-discount constant, so this is
    either a price an operator declared on an on-demand pool of the same shape
    or a number the caller passed. Anything else would be invented, and an
    invented saving is the figure a team puts in a budget.

    The class is compared as the enum. The client-side version of this compared
    against the string ``"on-demand"`` while the enum's value is ``on_demand``,
    so it never matched: the comparison this whole feature exists for could not
    fire, and every run reported the alternative as unknown.
    """

    if supplied is not None:
        return supplied, "from --on-demand-price"
    if shape is None:
        return None, "this run's shape is not recorded"
    for pool in pools:
        if pool.accelerator_type != shape or pool.capacity_class is not CapacityClass.ON_DEMAND:
            continue
        for dimension in ("tpu", "gpu"):
            value = pool.unit_prices.get(dimension)
            if value is not None and value > 0:
                return float(value), f"declared by pool {pool.id}"
    return None, f"no on-demand pool for {shape} declares a price; pass --on-demand-price to compare"


def storage_scope(org_id: str, project_id: str) -> str:
    payload = json.dumps((org_id, project_id), ensure_ascii=True, separators=(",", ":")).encode()
    return f"scope_{hashlib.sha256(payload).hexdigest()}"


_scope = storage_scope


def _validate_idempotency_key(key: str) -> str:
    key = key.strip()
    if not key:
        raise InputError("Idempotency-Key must not be empty")
    if len(key) > 200:
        raise InputError("Idempotency-Key must be at most 200 characters")
    return key


def _resource_quantities(resources) -> dict[str, float]:
    return {
        "cpu": float(resources.cpu),
        "memory_gib": float(resources.memory_gib),
        "gpu": float(resources.gpu),
        "tpu": float(resources.tpu),
        **{name: float(value) for name, value in resources.custom.items()},
    }


def _dump_limits(limits: ResourceLimits) -> JsonObject:
    """Render a limit vector for a fingerprint or an audit payload.

    Only the non-zero dimensions, so re-issuing the same grant produces the same
    fingerprint whether or not the caller spelled out the zeros.
    """

    return {name: quantity for name, quantity in sorted(resource_quantities(limits).items()) if quantity}


def _capacity_apply_payload(cell: CellRecord) -> JsonObject:
    """Build the durable agent request for one registered cell generation."""

    return {
        "cell_id": cell.id,
        "provider": cell.provider,
        "capacity_backend": cell.capacity_backend,
        "project": cell.provider_project,
        "region": cell.region,
        "zone": cell.zone,
        "accelerator_type": cell.accelerator_type,
        "capacity_class": cell.capacity_class.value,
        "executor": cell.executor,
        "agent_endpoint": cell.endpoint,
        "pool_id": cell.pool_id,
        "provider_options": cell.provider_options,
        "runtime": cell.runtime.model_dump(mode="json"),
    }


def _changed_cell_capacity_identity(existing: CellRecord, requested: CellRecord) -> tuple[str, ...]:
    """Return immutable capacity identity fields changed by an upsert."""

    changed: list[str] = []
    if existing.provider != requested.provider:
        changed.append("provider")
    if existing.capacity_backend != requested.capacity_backend:
        changed.append("capacity_backend")
    if existing.provider_project != requested.provider_project:
        changed.append("provider_project")
    if existing.region != requested.region:
        changed.append("region")
    if existing.zone != requested.zone:
        changed.append("zone")
    if existing.accelerator_type != requested.accelerator_type:
        changed.append("accelerator_type")
    if existing.capacity_class != requested.capacity_class:
        changed.append("capacity_class")
    if existing.executor != requested.executor:
        changed.append("executor")
    if existing.pool_id != requested.pool_id:
        changed.append("pool_id")
    if existing.provider_options != requested.provider_options:
        changed.append("provider_options")
    if existing.runtime != requested.runtime:
        changed.append("runtime")
    return tuple(changed)


_CELL_LIFECYCLE_TOPIC_BY_STATE: tp.Final[dict[CellState, str]] = {
    CellState.DRAINING: "capacity.drain",
    CellState.DELETING: "capacity.release",
}
_LIVE_OUTBOX_STATES: tp.Final[frozenset[OutboxState]] = frozenset({OutboxState.PENDING, OutboxState.PROCESSING})
_ACTIVE_CAPACITY_APPLY_STATES: tp.Final[frozenset[OutboxState]] = frozenset(
    {OutboxState.PENDING, OutboxState.PROCESSING, OutboxState.FAILED}
)


def _is_active_capacity_apply(command: OutboxRecord, cell: CellRecord) -> bool:
    """Return whether an outbox command still owns this cell's current provisioning generation."""

    return (
        command.topic == "capacity.apply"
        and command.aggregate_type == "cell"
        and command.aggregate_id == cell.id
        and command.generation == cell.generation
        and command.state in _ACTIVE_CAPACITY_APPLY_STATES
    )


def _host_fault_proof(cell: CellRecord, workload: WorkloadRecord) -> HostFaultProof | None:
    """Return this execution's host-fault record on ``cell``, if it has one.

    Narrow on purpose. The marker names the workload and the generation it was
    observed for, and only that execution's own fault says anything about that
    execution's process: a fault recorded for a previous tenant of the same cell
    record is evidence about a machine that has since been replaced.

    ``None`` means *no proof*, never *no fault* -- an older record, a truncated
    detail, or a control plane that has never seen node health all land here, and
    the caller answers all of them by giving the host up, which is what it did
    before any of this existed.
    """

    proof = parse_host_fault_marker(cell.unavailable_reason, cell.unavailable_detail)
    if proof is None or proof.workload_id != workload.id or proof.generation != workload.generation:
        return None
    return proof


def _decimal_sum(*values: float) -> float:
    """Deprecated alias; every money computation lives in ``eyvan.domain.costing``."""

    return costing.total(*values)


logger = logging.getLogger("eyvan.server.service.control_plane")

_EXPLAIN_MAX_CELLS = 32

#: Ledger rows one queue-trend query reads. Bounded because this backs a dashboard
#: panel that polls: an unbounded scan of the fastest-growing table in the system
#: would make an idle browser tab the heaviest query the control plane serves.
_QUEUE_TREND_MAX_EVENTS = 5_000
#: Prevent one dashboard or CLI query from allocating an arbitrarily large
#: response. Forty-eight half-hour buckets over a day is already denser than the
#: shipped chart renders; callers that need raw time series should use metrics.
QUEUE_TREND_MAX_BUCKETS = 48
#: A sub-minute trend is scheduler jitter, not a useful operations window; a
#: window wider than 31 days belongs in retained metrics rather than a polling
#: audit-ledger query.
QUEUE_TREND_MIN_WINDOW_S = 60.0
QUEUE_TREND_MAX_WINDOW_S = 31 * 86_400.0

#: How long an indexed artifact is protected from collection regardless of its
#: references. Closes a race that is easy to miss: an upload indexes the object
#: *before* any workload can name it -- the submitter needs the URI to build the
#: spec -- so a freshly uploaded callable is indistinguishable from abandoned
#: garbage until the submission lands. One day is far longer than that window and
#: costs only delayed reclamation.
DEFAULT_ARTIFACT_GRACE_PERIOD_S = 86_400.0
DEFAULT_ARTIFACT_RECOVERY_GRACE_S = 900.0

#: URI prefix of the managed result kind. Compared as a prefix rather than parsed:
#: the supersession sweep runs over every reference of every workload, and the only
#: question it asks is which kind the URI names.
_MANAGED_RESULT_PREFIX = f"eyvan://{ManagedArtifactKind.RESULTS.value}/"


def _is_finite_nonnegative_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, float) and math.isfinite(value) and value >= 0


def _artifact_record_update(
    record: ManagedArtifactRecord,
    updates: tp.Mapping[str, tp.Any],
) -> ManagedArtifactRecord:
    """Apply one lifecycle transition with full Pydantic invariant checks."""

    return ManagedArtifactRecord.model_validate({**record.model_dump(mode="python"), **updates})


def _explain_base(workload: WorkloadRecord) -> dict[str, tp.Any]:
    """Return the durable explanation fields shared by every scheduling verdict."""

    return {
        "workload_id": workload.id,
        "state": workload.state,
        "scheduling_reason": workload.scheduling_reason,
        "autoscale_block_reason": workload.autoscale_block_reason,
        "queue_admission_denial": workload.queue_admission_denial,
    }


def _explain_from_scheduling_result(workload: WorkloadRecord, result: SchedulingResult) -> WorkloadExplanation:
    """Project one pending workload's outcome out of a whole scheduling pass."""

    base = _explain_base(workload)
    placement = next((candidate for candidate in result.placements if candidate.workload_id == workload.id), None)
    if placement is not None:
        return WorkloadExplanation(
            **base,
            placeable=True,
            placement_cell_id=placement.lease.cell_id,
            message="the scheduler would place this workload now",
        )
    preemption = next((candidate for candidate in result.preemptions if candidate.workload_id == workload.id), None)
    if preemption is not None:
        victims = ", ".join(sorted({victim.workload_id for victim in preemption.victims}))
        return WorkloadExplanation(
            **base,
            placeable=True,
            placement_cell_id=preemption.cell_id,
            message=f"the scheduler would place this workload by preempting: {victims}",
        )
    infeasibility = next(
        (candidate for candidate in result.infeasibilities if candidate.workload_id == workload.id), None
    )
    if infeasibility is None:
        return WorkloadExplanation(
            **base,
            placeable=False,
            message="the scheduler did not consider this workload in the current pass",
        )
    rejected = sorted(infeasibility.cells, key=lambda cell: cell.cell_id)
    return WorkloadExplanation(
        **base,
        placeable=False,
        code=infeasibility.code,
        message=infeasibility.message,
        cells=tuple(
            CellRejection(cell_id=cell.cell_id, code=cell.code, message=cell.message)
            for cell in rejected[:_EXPLAIN_MAX_CELLS]
        ),
        considered_cell_count=len(rejected),
    )


#: How long a cancellation must have been failing before it counts as stuck
#: rather than in flight.
#:
#: Matches ``STUCK_OUTBOX_AFTER_S``, the point at which I3 reports the same
#: command. An operator should not be able to force-finalize something the
#: invariants still call healthy -- two thresholds that disagree would mean the
#: break-glass and the alarm describe different fleets.
STUCK_CANCEL_AFTER_S = 900.0


def _pool_declaration(pool: CapacityPoolRecord) -> JsonObject:
    """Return only what a fleet config declares about one pool.

    Generation, timestamps and ``initialized_at`` are the plane's bookkeeping,
    not the operator's declaration, so two applies of the same YAML must compare
    equal through this even though the stored records differ.
    """

    return tp.cast(
        "JsonObject",
        pool.model_copy(
            update={
                "generation": 1,
                "created_at": datetime.min,
                "updated_at": datetime.min,
                "initialized_at": None,
                "enabled": True,
            }
        ).model_dump(mode="json"),
    )


__all__ = (
    "ACTIVE_LEASE_STATES",
    "BILLABLE_CELL_STATES",
    "CALLABLE_SIGNING_KEY_ENV",
    "CHILD_REVISION",
    "CONTENT_ADDRESSED_KINDS",
    "DEFAULT_ARTIFACT_GRACE_PERIOD_S",
    "DEFAULT_ARTIFACT_RECOVERY_GRACE_S",
    "DEFAULT_MAX_FAN_OUT",
    "DEFAULT_MAX_RETRIES",
    "MAX_GCP_TPU_MULTISLICE_WORKERS",
    "MAX_GROUP_MEMBERS",
    "MAX_JOB_TREE_DEPTH",
    "MAX_LIVE_CHILDREN",
    "MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE",
    "POOL_ALTERNATES_LABEL",
    "POOL_ALTERNATE_TEMPLATES_LABEL",
    "POOL_LABEL",
    "POOL_TEMPLATE_LABEL",
    "QUEUE_TREND_MAX_BUCKETS",
    "QUEUE_TREND_MAX_WINDOW_S",
    "QUEUE_TREND_MIN_WINDOW_S",
    "REMOTE_CALLABLE_SIGNATURE_ERROR",
    "REMOTE_CALLABLE_SIGNING_KEY_ERROR",
    "STUCK_CANCEL_AFTER_S",
    "TERMINAL_WORKLOAD_STATES",
    "WORKER_COUNT_LABEL",
    "_ACTIVE_CAPACITY_APPLY_STATES",
    "_CELL_LIFECYCLE_TOPIC_BY_STATE",
    "_EXPLAIN_MAX_CELLS",
    "_LIVE_OUTBOX_STATES",
    "_MANAGED_RESULT_PREFIX",
    "_POOL_LABEL",
    "_QUERY_AUDIT_STATEMENT_LIMIT",
    "_QUEUE_TREND_MAX_EVENTS",
    "AdmissionPlan",
    "ArtifactCollection",
    "ArtifactHold",
    "ArtifactInventoryEntry",
    "ArtifactInventoryReport",
    "ArtifactInventoryStatus",
    "ArtifactReferenceRecord",
    "ArtifactUsage",
    "AuditEvent",
    "AuthorizationError",
    "Authorizer",
    "AutoscaleBlockReason",
    "BaseModel",
    "BatchRequest",
    "BudgetGrantRecord",
    "BudgetRecord",
    "BudgetStatus",
    "CallableExecution",
    "CallableProjectTrustPolicy",
    "CandidateZone",
    "CapacityClass",
    "CapacityClassUnavailableError",
    "CapacityPoolRecord",
    "CarbonReport",
    "CarbonSource",
    "CarbonUnavailable",
    "CellBillingAccrual",
    "CellBillingReport",
    "CellRecord",
    "CellRejection",
    "CellState",
    "CommandExecution",
    "ConflictError",
    "CoordinationKind",
    "DagNode",
    "EffectiveQuota",
    "ExpansionAdmissionRecord",
    "ExpansionAdmissionState",
    "ExpansionLimits",
    "ExpansionMode",
    "FederatedBudgetStatus",
    "GcpQueueAdmissionDenialTarget",
    "GrantState",
    "GroupRole",
    "HostFaultProof",
    "IdempotencyRecord",
    "InputError",
    "InvalidManagedArtifactURI",
    "InvalidTransition",
    "JobTreeTooDeepError",
    "JobTreeTooWideError",
    "JsonObject",
    "JsonValue",
    "LaunchLatencyReport",
    "LeaseRecord",
    "LeaseState",
    "ManagedArtifactConflictError",
    "ManagedArtifactDeleteInconclusiveError",
    "ManagedArtifactDeleteStatus",
    "ManagedArtifactInventoryError",
    "ManagedArtifactInventoryStore",
    "ManagedArtifactKind",
    "ManagedArtifactObjectMetadata",
    "ManagedArtifactOperationInconclusiveError",
    "ManagedArtifactRecord",
    "ManagedArtifactRef",
    "ManagedArtifactState",
    "ManagedArtifactVersionMismatchError",
    "ManagedArtifactVersionedStore",
    "ManagedArtifactWriteInconclusiveError",
    "NoCarbonSource",
    "NoMatchingCapacityPoolError",
    "NotFoundError",
    "OrganizationRecord",
    "OutboxRecord",
    "OutboxState",
    "Permission",
    "PlacementMode",
    "PlanePeerRecord",
    "PolicyRecord",
    "PoolEstimate",
    "PoolLaunchLatency",
    "PoolWarmth",
    "Principal",
    "PriorityBand",
    "ProjectRecord",
    "ProjectUsageReport",
    "QueueTrend",
    "QueueTrendPoint",
    "QuotaGrantRecord",
    "QuotaRecord",
    "ReadOnlyQueryResult",
    "RegionCarbon",
    "Repository",
    "RepositorySession",
    "ResourceGang",
    "ResourceHours",
    "ResourceLimits",
    "Role",
    "RoleGrant",
    "RoleGrantRecord",
    "SchedulingResult",
    "SchedulingService",
    "ScopeQuota",
    "ServiceDecisionKind",
    "ServiceDecisionRecord",
    "ServiceObservation",
    "ServicePolicy",
    "ServiceReplicaHealthState",
    "SubjectUsageRecord",
    "UnavailableError",
    "UserProgressReport",
    "WarmthReport",
    "WebhookRecord",
    "WorkerResourceScope",
    "WorkloadDefinition",
    "WorkloadDispatchReservation",
    "WorkloadEstimate",
    "WorkloadExplanation",
    "WorkloadRecord",
    "WorkloadSpendRecord",
    "WorkloadState",
    "WorkloadTransitionUpdates",
    "WorkspaceRef",
    "_artifact_record_update",
    "_capacity_apply_payload",
    "_changed_cell_capacity_identity",
    "_decimal_sum",
    "_declares_a_price",
    "_dump_limits",
    "_explain_base",
    "_explain_from_scheduling_result",
    "_fingerprint",
    "_flex_start_refusal",
    "_host_fault_proof",
    "_is_active_capacity_apply",
    "_is_finite_nonnegative_number",
    "_multislice_refusal",
    "_new_id",
    "_on_demand_price",
    "_pool_declaration",
    "_reserved_resource_hours",
    "_resource_quantities",
    "_scope",
    "_validate_idempotency_key",
    "accelerator_unit_price",
    "artifact_reference_id",
    "asyncio",
    "backend_capabilities",
    "candidate_pool_templates",
    "candidate_zone_report",
    "cell_matches_template",
    "cell_provides_pool_supply",
    "child_idempotency_key",
    "collect_descendants",
    "costing",
    "dag_validate",
    "datetime",
    "ensure_lease_transition",
    "ensure_workload_transition",
    "exceeds_quota_headroom",
    "expand",
    "expansion_admission_id",
    "federated_spend",
    "fresh_reading",
    "group_member_names",
    "group_validate",
    "hashlib",
    "is_managed_uri",
    "issuable_amount",
    "json",
    "locality_index",
    "logger",
    "logging",
    "managed_result_uri",
    "matching_secret_reference_root",
    "math",
    "model_json",
    "outstanding_total",
    "p95_latency",
    "parent_depth",
    "parse_host_fault_marker",
    "parse_managed_uri",
    "parse_tpu_accelerator_type",
    "plan_expansion",
    "preferred_cells_for",
    "providers_hosting_multi_worker_cells",
    "quota_headroom",
    "quota_headroom_children",
    "replace",
    "replica_health_state",
    "resource_quantities",
    "scaled_service_metadata",
    "secrets",
    "service_metadata",
    "service_policy",
    "spend_of",
    "stable_id",
    "storage_scope",
    "task_index",
    "timedelta",
    "tp",
    "utcnow",
    "uuid",
    "validate_capacity_pool_quota_groups",
    "validate_parent",
    "validate_webhook_target",
)
