# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Isolated adapter from persisted API records to the pure core scheduler."""

from __future__ import annotations

import typing as tp
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from uuid import uuid4

from eyvan.domain import (
    Budget,
    CapacityCell,
    CapacityClass,
    CapacityLease,
    CheckpointMode,
    CheckpointPolicy,
    IdentityScope,
    PlacementMode,
    Quota,
    ResourceGang,
    ResourceVector,
    Scheduler,
    SchedulingResult,
    WorkerResourceScope,
    WorkloadIdentity,
    WorkloadSpec,
)
from eyvan.domain import (
    PriorityBand as CorePriorityBand,
)
from eyvan.domain.availability import p95_latency
from eyvan.records import (
    POOL_ALTERNATE_TEMPLATES_LABEL,
    POOL_ALTERNATES_LABEL,
    POOL_LABEL,
    POOL_TEMPLATE_LABEL,
    WORKER_COUNT_LABEL,
    BudgetRecord,
    CellRecord,
    CellState,
    LeaseRecord,
    PolicyRecord,
    PriorityBand,
    QuotaRecord,
    WorkloadRecord,
    WorkloadState,
    resource_quantities,
    utcnow,
)
from eyvan.records import (
    CapacityClass as ApiCapacityClass,
)
from eyvan.records import (
    ResourceVector as ApiResourceVector,
)
from eyvan.server.service.federation import (
    _MONEY_PRECISION,
    _QUOTA_PRECISION,
    BudgetGrantRecord,
    QuotaGrantRecord,
    _issuable_amount_exact,
    _outstanding_quota_exact,
    _quota_headroom_exact,
)
from eyvan.server.service.state import ACTIVE_LEASE_STATES


class SchedulingService(tp.Protocol):
    """Pure global scheduling boundary injected into reconciliation services."""

    def schedule(
        self,
        workloads: tp.Iterable[WorkloadRecord],
        cells: tp.Iterable[CellRecord],
        *,
        leases: tp.Iterable[LeaseRecord] = (),
        budgets: tp.Iterable[BudgetRecord] = (),
        quotas: tp.Iterable[QuotaRecord] = (),
        policies: tp.Iterable[PolicyRecord] = (),
        grants: tp.Iterable[BudgetGrantRecord] = (),
        quota_grants: tp.Iterable[QuotaGrantRecord] = (),
        plane_id: str = "",
        now: datetime | None = None,
    ) -> SchedulingResult: ...


def _resources(value: ApiResourceVector) -> ResourceVector:
    items = {
        "cpu": value.cpu,
        "memory_gib": value.memory_gib,
        "gpu": value.gpu,
        "tpu": value.tpu,
        **value.custom,
    }
    return ResourceVector({key: quantity for key, quantity in items.items() if quantity})


def _priority(value: PriorityBand) -> CorePriorityBand:
    return CorePriorityBand[value.name]


def _checkpoint(record: WorkloadRecord, landed_capacity_class: ApiCapacityClass | None = None) -> CheckpointPolicy:
    spec = record.spec
    return CheckpointPolicy(
        mode=CheckpointMode.ON_PREEMPTION if spec.checkpointable and spec.checkpoint_hook else CheckpointMode.NONE,
        grace_period_seconds=spec.checkpoint_grace_seconds,
        # Spot capacity is interruptible by definition: the provider may reclaim
        # it at any moment with no checkpoint, so Eyvan reclaiming it for a
        # higher-priority workload breaks no promise the user was given.
        #
        # This must be set explicitly. The scheduler requires a victim to be
        # `checkpointable or allow_forced_termination` (core/scheduler.py), so
        # leaving it at its default made every non-checkpointable spot lease
        # revocable-but-never-eligible: the request surfaced as
        # PREEMPTION_UNSAFE and that capacity was never reclaimed at all.
        # On-demand capacity is deliberately excluded -- it is only preemptible
        # when the workload can checkpoint.
        #
        # The class that decides is the one the lease *landed* on -- the same
        # class ``revocable`` keys on. With a portfolio, a workload that
        # requested on-demand can land on a spot cell; keying this off the
        # requested class left that lease revocable but still ineligible, and
        # the scheduler answered PREEMPTION_UNSAFE for capacity that was
        # reclaimable by definition. ``None`` falls back to the requested
        # class, right for a workload that has not landed anywhere yet.
        allow_forced_termination=(landed_capacity_class or spec.capacity_class) is ApiCapacityClass.SPOT,
    )


def _pool_choices(
    required_labels: dict[str, str],
) -> tuple[list[tuple[str, tuple[str, ...]]], tuple[tuple[tuple[str, str], ...], ...]]:
    """Turn a cross-zone workload's pool pins into "one of" choices, in place.

    Submission records a cross-zone workload as ``eyvan_pool`` (the first
    choice), ``eyvan_pool_template`` (that pool's template) and the alternates
    as a comma list. These cannot all be required equality labels: no cell
    carries the alternates list, and an alternate's template differs from the
    first choice's template.

    The pool pin becomes "any candidate pool", with correlated pool/template
    combinations -- never the Cartesian product of two independent label sets.
    This refuses cells built from a different candidate pool's template. A workload stored before
    the alternates' templates were recorded keeps its strict first-choice
    template pin, so its first choice works again and nothing is loosened.
    """

    alternates_value = required_labels.pop(POOL_ALTERNATES_LABEL, "")
    templates_value = required_labels.pop(POOL_ALTERNATE_TEMPLATES_LABEL, "")
    alternates = [value.strip() for value in alternates_value.split(",") if value.strip()]
    primary = required_labels.get(POOL_LABEL)
    if not alternates or primary is None:
        return [], ()
    pools = tuple(dict.fromkeys((primary, *alternates)))
    required_labels.pop(POOL_LABEL)
    choices: list[tuple[str, tuple[str, ...]]] = [(POOL_LABEL, pools)]
    templates = [value.strip() for value in templates_value.split(",") if value.strip()]
    primary_template = required_labels.get(POOL_TEMPLATE_LABEL)
    if primary_template is not None and len(templates) == len(alternates):
        required_labels.pop(POOL_TEMPLATE_LABEL)
        combinations = tuple(
            ((POOL_LABEL, pool), (POOL_TEMPLATE_LABEL, template))
            for pool, template in zip((primary, *alternates), (primary_template, *templates), strict=True)
        )
        return choices, combinations
    return choices, ()


def to_core_workload(record: WorkloadRecord, policy: PolicyRecord | None = None) -> WorkloadSpec:
    """Convert one persisted workload without sharing mutable API types."""

    spec = record.spec
    required_labels = {
        **spec.required_labels,
        **({"provider": spec.provider} if spec.provider else {}),
        **({"region": spec.region} if spec.region else {}),
        **({"accelerator_type": spec.accelerator_type} if spec.accelerator_type else {}),
    }
    pool_choices, pool_combinations = _pool_choices(required_labels)
    if pool_choices:
        # The candidate pools are named outright, and each carries its own
        # region. Keeping the first choice's region as a required label would
        # reject every alternate in another region.
        required_labels.pop("region", None)
    if spec.placement is PlacementMode.ELASTIC:
        # The exact-shape pin, dropped: elastic workloads pack cells of any
        # worker count up to the request, and the pin lives in the elastic path
        # itself (``_schedule_elastic`` skips cells larger than the remaining
        # need) instead of rejecting every warm cell on a label.
        required_labels.pop(WORKER_COUNT_LABEL, None)
    else:
        required_labels[WORKER_COUNT_LABEL] = str(spec.worker_count)
    portfolio = tuple(CapacityClass.parse(entry.value) for entry in spec.capacity_class_portfolio)
    allowed_label_values: list[tuple[str, tuple[str, ...]]] = []
    # Workload choices describe preferences, never additional project authority.
    # Region choices outside the policy are denied below; keeping the original
    # list also represents a disjoint intersection without an invalid empty
    # allowed-label set.
    for label, choices in (
        ("region", spec.allowed_regions),
        ("zone", spec.allowed_zones),
        ("accelerator_type", spec.allowed_accelerator_types),
    ):
        if choices:
            allowed_label_values.append((label, choices))
    allowed_label_values.extend(pool_choices)
    if policy is not None:
        if not spec.provider and policy.allowed_providers:
            allowed_label_values.append(("provider", policy.allowed_providers))
        # A cross-zone workload has had its region pin dropped (its candidate
        # pools may span regions), so the spec's region no longer stands in for
        # the policy check it used to satisfy: apply the policy directly.
        if not spec.allowed_regions and policy.allowed_regions:
            allowed_label_values.append(("region", policy.allowed_regions))
    denied_label_values: list[tuple[str, tuple[str, ...]]] = []
    if policy is not None:
        if policy.denied_accelerator_types:
            denied_label_values.append(("accelerator_type", policy.denied_accelerator_types))
        if policy.allowed_regions:
            forbidden_regions = tuple(region for region in spec.allowed_regions if region not in policy.allowed_regions)
            if forbidden_regions:
                denied_label_values.append(("region", forbidden_regions))
    return WorkloadSpec(
        workload_id=record.id,
        identity=WorkloadIdentity(record.org_id, record.project_id, record.created_by),
        gangs=(
            ResourceGang(
                gang_id=f"{record.id}-0",
                replicas=spec.gang.replicas,
                resources_per_replica=_resources(spec.gang.per_replica),
                required_topology=tuple(spec.gang.required_topology.items()),
            ),
        ),
        priority=_priority(spec.priority_band),
        attempt=record.attempt,
        generation=record.generation,
        queue_position=1000 - spec.intra_band_priority,
        runtime=spec.required_executor,
        required_labels=tuple(required_labels.items()),
        preferred_labels=tuple(spec.preferred_labels.items()),
        allowed_label_values=tuple(allowed_label_values),
        allowed_label_combinations=pool_combinations,
        denied_label_values=tuple(denied_label_values),
        preferred_cells=spec.preferred_cells,
        # The portfolio, not the single requested class. The core scheduler has
        # always ranked these -- ``capacity_class_rank`` is the fourth key of the
        # lexicographic PlacementScore -- and this adapter collapsing it to one
        # entry was the only thing making the mechanism unreachable.
        allowed_capacity_classes=frozenset(portfolio),
        capacity_class_preference=portfolio,
        checkpoint_policy=_checkpoint(record),
        estimated_duration_seconds=spec.estimated_duration_seconds,
        # A multi-cell gang is preemptible as one unit: the scheduler takes
        # every sibling lease or none (``_preemption_candidate``), and
        # ``NativeGangExecutor.checkpoint`` runs the hook on every cell under one
        # shared checkpoint ID. Exposing a single member as an individual victim
        # was the hazard, not multi-cell membership itself.
        revocable=(spec.capacity_class is ApiCapacityClass.SPOT or spec.checkpointable)
        and (policy is None or policy.preemption_enabled),
        cell_group_count=spec.cell_group_count,
        worker_count=spec.worker_count,
        worker_resource_scope=spec.worker_resource_scope,
        placement=spec.placement,
        tpu_chip_count=None if spec.tpu_chips is None else spec.tpu_chips.count,
        tpu_chip_indexes=None if spec.tpu_chips is None else spec.tpu_chips.indexes,
    )


def to_core_cell(record: CellRecord) -> CapacityCell:
    """Convert an inventory record to scheduler-visible allocatable capacity."""

    labels = {
        **record.labels,
        WORKER_COUNT_LABEL: str(record.worker_count),
        "provider": record.provider,
        "region": record.region,
        **({"zone": record.zone} if record.zone else {}),
        **({"accelerator_type": record.accelerator_type} if record.accelerator_type else {}),
    }
    evidence = record.availability.success_evidence + record.availability.failure_evidence
    # A cell without availability history has no failure signal; treat it as
    # fully reliable instead of dividing by zero.
    success_probability = record.availability.success_evidence / evidence if evidence else 1.0
    latency_p95 = p95_latency(record.availability.provision_latencies_s) or 0
    interruption_rate = record.availability.interruptions / max(record.availability.successful_hours, 1)
    return CapacityCell(
        cell_id=record.id,
        capacity=_resources(record.allocatable),
        capacity_class=CapacityClass.parse(record.capacity_class.value),
        labels=tuple(labels.items()),
        topology=tuple(record.topology.items()),
        runtimes=frozenset({record.executor}),
        unit_prices=ResourceVector(record.unit_prices),
        route_priority=round((1 - success_probability) * 1000 + interruption_rate * 100),
        startup_latency_seconds=round(latency_p95),
        max_replica_resources=_resources(record.max_replica_resources) if record.max_replica_resources else None,
        max_gang_replicas=(
            record.max_gang_replicas
            if record.max_gang_replicas is not None
            else 1
            if record.executor in {"native-local", "eyvan-native"}
            else None
        ),
        exclusive_workloads=record.executor in {"native-local", "eyvan-native"},
        autoscale_request_workload_id=record.autoscale_request_workload_id,
        enabled=record.state is CellState.READY,
        worker_count=record.worker_count,
        chip_sharing=record.chip_sharing,
        chips_per_host=_chips_per_host(record) if record.chip_sharing else None,
    )


def _chips_per_host(record: CellRecord) -> int | None:
    """Return the cell's TPU chips per host from the hardware catalog, or ``None`` when unknown."""

    from eyvan.capacity.hardware_catalog import shape_for

    shape = shape_for(record.accelerator_type or "") if record.accelerator_type else None
    return None if shape is None else shape.chips_per_host


def landed_capacity_classes(cells: tp.Iterable[CellRecord]) -> dict[str, ApiCapacityClass]:
    """Map cell id to capacity class, for resolving what a lease actually landed on.

    A named function rather than an inline comprehension because it is the whole
    of the wiring that makes the landed class reachable: ``LeaseRecord`` carries no
    capacity class, so without this join ``to_core_lease`` has nothing to consult
    and silently falls back to the *requested* class -- which is the bug.
    """

    return {cell.id: cell.capacity_class for cell in cells}


def to_core_lease(
    record: LeaseRecord,
    workload: WorkloadRecord,
    policy: PolicyRecord | None = None,
    landed_capacity_class: ApiCapacityClass | None = None,
    landed_worker_count: int | None = None,
) -> CapacityLease:
    """Convert a durable lease, preserving exact budget reservation dimensions.

    ``landed_capacity_class`` is the class of the cell the lease actually landed
    on, and it decides ``revocable`` *and* ``allow_forced_termination``. With a
    portfolio the *requested* class is no longer the answer: a workload that
    lists ``spot`` in its portfolio and lands on an **on-demand** cell would
    otherwise be marked force-terminable -- a silent promise violation, since
    nothing about that cell is preemptible. The reverse matters too: a workload
    that requested on-demand and landed on a **spot** cell is preemptable
    whether or not it can checkpoint, and keying the checkpoint policy off the
    requested class left it revocable but never an eligible victim
    (PREEMPTION_UNSAFE on capacity reclaimable by definition).

    ``None`` falls back to the requested class, which is exactly right for the
    single-class case and is what every pre-portfolio caller means.
    """

    resources = _resources(record.resources.per_replica).scale(record.resources.replicas)
    if workload.spec.worker_resource_scope is WorkerResourceScope.PER_WORKER:
        # Persisted member leases retain the per-worker gang. An elastic
        # member holds only its cell's workers, not the entire pack's width.
        worker_count = workload.spec.worker_count
        if workload.spec.placement is PlacementMode.ELASTIC and landed_worker_count is not None:
            worker_count = landed_worker_count
        resources = resources.scale(worker_count)
    return CapacityLease(
        lease_id=record.id,
        workload_id=record.workload_id,
        attempt=workload.attempt,
        identity=WorkloadIdentity(record.org_id, record.project_id, workload.created_by),
        cell_id=record.cell_id,
        resources=resources,
        priority=_priority(workload.spec.priority_band),
        hourly_cost=record.hourly_cost,
        reserved_cost=record.reserved_cost,
        resource_hours_reserved=_resources(record.resource_hours_reserved),
        checkpoint_policy=_checkpoint(workload, landed_capacity_class),
        # Gangs are preemptible as one unit; see to_core_workload.
        revocable=(
            (landed_capacity_class or workload.spec.capacity_class) is ApiCapacityClass.SPOT
            or workload.spec.checkpointable
        )
        and (policy is None or policy.preemption_enabled),
        fencing_epoch=record.generation,
        tpu_chips=record.tpu_chips or (),
    )


_PERIOD_SECONDS: tp.Final[dict[str, float]] = {
    "hourly": 3_600.0,
    "daily": 86_400.0,
    "weekly": 604_800.0,
    "monthly": 2_592_000.0,
    "quarterly": 7_776_000.0,
    "yearly": 31_536_000.0,
}


def rolled_budget(record: BudgetRecord, now: datetime) -> BudgetRecord:
    """Reset a budget whose declared period has elapsed.

    Port of the control plane's ``_rolled_budget`` (``_settle_lease`` rolls only
    where money moves): ``spent`` is a monotone total over the declared window,
    so admission reading it raw compares spend from a crossed period against a
    fresh limit -- a hard budget then refuses every placement forever once the
    window closes with no in-flight lease to trigger a settlement. Mirrors the
    control-plane semantics exactly: unknown periods never roll, a missing
    ``period_started_at`` is stamped without a reset, whole elapsed periods
    advance at once so the window stays aligned to its origin, and per-period
    alerts re-arm.
    """

    length = _PERIOD_SECONDS.get(record.period.strip().lower())
    started = record.period_started_at
    if length is not None and started is None:
        # Initializing a window is not a reset: all retained spend belongs to
        # its opening, not to settlements whose clocks happen to be later.
        return record.model_copy(
            update={
                "period_started_at": now,
                "settlement_period_id": uuid4().hex,
                "period_opening_spent": record.spent,
            }
        )
    if length is None:
        return record
    assert started is not None
    if (now - started).total_seconds() < length:
        return record
    elapsed_periods = int((now - started).total_seconds() // length)
    return record.model_copy(
        update={
            "spent": 0.0,
            "accelerator_hours_spent": {},
            "alerted_thresholds": (),
            "period_started_at": started + timedelta(seconds=length * elapsed_periods),
            "settlement_period_id": uuid4().hex,
            "period_opening_spent": 0.0,
        }
    )


def to_core_budget(record: BudgetRecord, *, now: datetime | None = None) -> Budget:
    """Convert monetary and accelerator-hour hard limits.

    The record's ``spent`` is only meaningful inside its declared period, so the
    window is rolled (:func:`rolled_budget`) before conversion. Callers that
    already hold a clock should pass it as ``now``; defaulting to the current
    time keeps the pure read path honest without inventing a second notion of
    "now" per plane.
    """

    record = rolled_budget(record, now or utcnow())
    return Budget(
        scope=IdentityScope(record.org_id, record.project_id),
        limit=record.limit,
        spent=record.spent,
        currency=record.currency,
        accelerator_hour_limits=ResourceVector(record.accelerator_hour_limits),
        accelerator_hours_spent=ResourceVector(record.accelerator_hours_spent),
        hard_limit=record.hard_limit,
    )


def to_core_quota(record: QuotaRecord) -> Quota:
    """Convert a hierarchical API quota to the exact core vector contract."""

    quantities = {
        "cpu": record.limits.cpu,
        "memory_gib": record.limits.memory_gib,
        "gpu": record.limits.gpu,
        "tpu": record.limits.tpu,
        **record.limits.custom,
    }
    return Quota(
        scope=IdentityScope(record.org_id, record.project_id, record.principal_id),
        limit=ResourceVector({name: quantities[name] for name in record.constrained_resources}),
        weight=record.weight,
    )


def federated_budgets(
    records: tp.Sequence[BudgetRecord],
    grants: tp.Sequence[BudgetGrantRecord],
    *,
    plane_id: str,
    now: datetime,
) -> list[Budget]:
    """Bound this plane's admission by the grants it holds, not by a local counter.

    ``BudgetRecord.spent`` is one durable counter in one database. A tenant with
    projects on three planes therefore had three independent budgets, each
    enforcing its own limit and each reporting full compliance while the tenant
    spent three times what its operator set. Grants existed to fix that and
    nothing consulted them: ``list_budget_grants`` was called from issuance,
    status and expiry, and never from admission. A grant was recorded and
    licensed nothing.

    Three rules, and the second and third are the ones that matter:

    * A plane with no inbound grant history for a tenant is unchanged. An
      inactive-only inbound scope retains zero authority, never a local fallback.
    * A grantee's effective limit is the **tighter** of its grant total and any
      local limit. A grant may not raise a ceiling a local operator set, and a
      local record may not exceed the slice the home plane issued.
    * A grant is a **hard** ceiling even when the local budget is soft.
      Otherwise a grantee overspends its slice and the bound the whole mechanism
      provides is void.

    A grantee without a local budget record gets a hard zero monetary allowance.
    A grant conveys authority but is not a durable consumption ledger: settlement
    cannot accumulate spend without a BudgetRecord. Refuse positive-cost admission
    until the operator supplies durable local accounting, rather than synthesize
    fresh spent=0 on every pass. This does not reconstruct legacy consumption or
    define a new cross-plane grant/window accounting protocol.
    """

    outgoing: dict[tuple[str, str], list[BudgetGrantRecord]] = {}
    if plane_id:
        for grant in grants:
            if grant.home_plane_id == plane_id:
                outgoing.setdefault((grant.org_id, grant.project_id), []).append(grant)
    # The home uses exactly the issuance partition: local spend/reservations
    # plus retained issued rights. The core accounts active leases separately.
    # Retirement and period changes never free unproven peer consumption.
    granted: dict[tuple[str, str], Decimal] = {}
    # The empty check is a short-circuit, not the safety property. What actually
    # keeps a non-federated plane from matching is ``grantee_plane_id``'s
    # ``min_length=1``: no grant can be addressed to "".
    if plane_id:
        for grant in grants:
            if grant.grantee_plane_id == plane_id:
                key = (grant.org_id, grant.project_id)
                # Retained inactive records distinguish a scope that requires
                # inbound authority from a genuinely never-federated scope.
                granted.setdefault(key, Decimal(0))
                if grant.active_at(now):
                    granted[key] += Decimal(str(grant.amount))
    budgets: list[Budget] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.org_id, record.project_id)
        seen.add(key)
        local = to_core_budget(record, now=now)
        issued = outgoing.get(key)
        if issued:
            local = replace(local, limit=_issuable_amount_exact(record.limit, issued, now=now), hard_limit=True)
        allowance = granted.get(key)
        if allowance is not None:
            local = replace(local, limit=min(allowance, local.limit), hard_limit=True)
        budgets.append(local)
    for key in sorted(granted.keys() | outgoing.keys()):
        if key in seen:
            continue
        org_id, project_id = key
        budgets.append(
            Budget(
                scope=IdentityScope(org_id, project_id),
                limit=0.0,
                spent=0.0,
                currency="USD",
                hard_limit=True,
            )
        )
    return budgets


def federated_quotas(
    records: tp.Sequence[QuotaRecord],
    grants: tp.Sequence[QuotaGrantRecord],
    *,
    plane_id: str,
    now: datetime,
    resource_names: tp.Iterable[str] = (),
) -> list[Quota]:
    """Bound this plane's placement by the quota slices it holds.

    The capacity half of :func:`federated_budgets`, and it exists for the same
    reason: ``QuotaRecord.limits`` is one vector in one database, so a tenant with
    projects on three planes could run three times its accelerator ceiling with
    every plane locally compliant. Money and chips are separate limits, and the
    scarce thing in a shared fleet is usually chips.

    The same three rules:

    * A plane with no inbound quota grant history for a tenant is unchanged.
      Retained inactive grants close admission instead of removing the ceiling.
    * The effective limit is the **tighter** of the grant and any local quota,
      per dimension. A grant may not raise a ceiling a local operator set, and a
      local record may not exceed the slice the home plane issued.
    * A grantee with grants and no local quota record gets the grant as its
      limit. Without that, the plane with the least local configuration would be
      the one with no ceiling -- exactly backwards.

    Only project-scoped records are federated. Principal-scoped quotas subdivide
    a project's share *within* one plane, and a cross-plane grant has nothing to
    say about how a tenant splits its own slice between its own people.
    """

    dimensions = {"cpu", "memory_gib", "gpu", "tpu", *resource_names}
    for record in records:
        dimensions.update(record.limits.custom)
    for grant in grants:
        dimensions.update(grant.limits.custom)
    outgoing: dict[tuple[str, str], list[QuotaGrantRecord]] = {}
    if plane_id:
        for grant in grants:
            if grant.home_plane_id == plane_id:
                outgoing.setdefault((grant.org_id, grant.project_id), []).append(grant)
    granted: dict[tuple[str, str], dict[str, Decimal]] = {}
    # An inactive-only scope retains explicit zero authority, including custom
    # resources present in the scheduling snapshot.
    if plane_id:
        for grant in grants:
            if grant.grantee_plane_id != plane_id:
                continue
            key = (grant.org_id, grant.project_id)
            totals = granted.setdefault(key, dict.fromkeys(sorted(dimensions), Decimal(0)))
            if not grant.active_at(now):
                continue
            with localcontext(prec=_QUOTA_PRECISION):
                for name, quantity in resource_quantities(grant.limits).items():
                    totals[name] += Decimal(str(quantity))
    quotas: list[Quota] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        if record.principal_id is not None or record.project_id is None:
            quotas.append(to_core_quota(record))
            continue
        key = (record.org_id, record.project_id)
        allowance = granted.get(key)
        seen.add(key)
        local = to_core_quota(record)
        issued = outgoing.get(key)
        if issued:
            retained = _outstanding_quota_exact(issued, now=now)
            headroom = _quota_headroom_exact(record.limits, issued, now=now)
            home_limits = local.limit.as_dict()
            # Issuance uses declared quantities, even for locally unconstrained
            # dimensions. Those delegated dimensions must bind the home too;
            # otherwise a sparse local quota bypasses the same partition.
            for name, quantity in retained.items():
                if quantity > 0:
                    remaining = headroom.get(name, Decimal(0))
                    home_limits[name] = min(home_limits.get(name, remaining), remaining)
            local = replace(local, limit=ResourceVector(home_limits))
        if allowance is None:
            quotas.append(local)
            continue
        # Local absence is unrestricted; grant absence is zero authority.
        # Enforce the grant independently of the local quota's sparse keys.
        local_limits = local.limit.as_dict()
        bounded = {
            name: min(local_limits.get(name, Decimal(str(quantity))), Decimal(str(quantity)))
            for name, quantity in allowance.items()
        }
        quotas.append(replace(local, limit=ResourceVector(bounded)))
    for key in sorted(granted.keys() | outgoing.keys()):
        if key in seen:
            continue
        org_id, project_id = key
        allowance = dict.fromkeys(sorted(dimensions), Decimal(0)) if key in outgoing else granted[key]
        quotas.append(Quota(scope=IdentityScope(org_id, project_id), limit=ResourceVector(dict(allowance)), weight=1.0))
    return quotas


class CoreSchedulingService:
    """Adapter around :class:`eyvan.domain.Scheduler`."""

    def __init__(self, scheduler: Scheduler | None = None):
        self.scheduler = scheduler or Scheduler()

    def schedule(
        self,
        workloads: tp.Iterable[WorkloadRecord],
        cells: tp.Iterable[CellRecord],
        *,
        leases: tp.Iterable[LeaseRecord] = (),
        budgets: tp.Iterable[BudgetRecord] = (),
        quotas: tp.Iterable[QuotaRecord] = (),
        policies: tp.Iterable[PolicyRecord] = (),
        grants: tp.Iterable[BudgetGrantRecord] = (),
        quota_grants: tp.Iterable[QuotaGrantRecord] = (),
        plane_id: str = "",
        now: datetime | None = None,
    ) -> SchedulingResult:
        workload_records = list(workloads)
        workload_by_id = {workload.id: workload for workload in workload_records}
        # `schedulable` excludes coordinator records -- a batch parent owns its
        # children and derives its state from them, it does not run.
        pending = [
            workload for workload in workload_records if workload.state is WorkloadState.PENDING and workload.schedulable
        ]
        active_leases = [lease for lease in leases if lease.state in ACTIVE_LEASE_STATES]
        # ``schedule`` already receives the cells; a lease record carries no
        # capacity class of its own, so this join is the only place the landed
        # class is available at conversion time.
        cell_records = list(cells)
        class_by_cell = landed_capacity_classes(cell_records)
        workers_by_cell = {cell.id: cell.worker_count for cell in cell_records}
        orphans = [lease.id for lease in active_leases if lease.workload_id not in workload_by_id]
        if orphans:
            raise ValueError(f"active lease {orphans[0]!r} references a missing workload")
        policy_by_scope = {(policy.org_id, policy.project_id): policy for policy in policies}
        budget_grants = list(grants)
        quota_grant_records = list(quota_grants)
        quota_records = list(quotas)
        moment = now or utcnow()
        with localcontext() as context:
            # Keep the complete retained-rights partition through core addition
            # and comparison too; a default-context sum can erase a tiny active
            # reservation even when issuance computed its headroom exactly.
            # Local quota reservations require the same exact arithmetic even
            # before a scope has issued or received any federation grants.
            if quota_records or (plane_id and (budget_grants or quota_grant_records)):
                context.prec = max(context.prec, _MONEY_PRECISION, _QUOTA_PRECISION)
            return self.scheduler.schedule(
                (
                    to_core_workload(workload, policy_by_scope.get((workload.org_id, workload.project_id)))
                    for workload in pending
                ),
                (to_core_cell(cell) for cell in cell_records),
                leases=(
                    to_core_lease(
                        lease,
                        workload_by_id[lease.workload_id],
                        policy_by_scope.get((lease.org_id, lease.project_id)),
                        class_by_cell.get(lease.cell_id),
                        workers_by_cell.get(lease.cell_id),
                    )
                    for lease in active_leases
                ),
                quotas=federated_quotas(
                    quota_records,
                    quota_grant_records,
                    plane_id=plane_id,
                    now=moment,
                    resource_names=(
                        name for workload in workload_records for name in workload.spec.gang.per_replica.custom
                    ),
                ),
                budgets=federated_budgets(list(budgets), budget_grants, plane_id=plane_id, now=moment),
            )
