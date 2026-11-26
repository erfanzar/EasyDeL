# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic, side-effect-free Eyvan scheduling policy."""

from __future__ import annotations

import hashlib
import heapq
import typing as tp
from dataclasses import dataclass
from decimal import Decimal

from .models import (
    Budget,
    CapacityCell,
    CapacityLease,
    CellInfeasibility,
    IdentityScope,
    Infeasibility,
    InfeasibilityCode,
    PlacementDecision,
    PlacementMode,
    PlacementScore,
    PreemptionActionKind,
    PreemptionPlan,
    PreemptionVictim,
    PriorityBand,
    Quota,
    ResourceVector,
    SchedulingResult,
    WorkerResourceScope,
    WorkloadIdentity,
    WorkloadSpec,
)
from .tpu_chips import pick_chips

_EMPTY_RESOURCES = ResourceVector()


def _scope_key(scope: IdentityScope) -> tuple[str, str | None, str | None]:
    """Return the structural dedup key for a scope, independent of ``repr``."""

    return (scope.tenant_id, scope.project_id, scope.principal_id)


_PreemptionSortKey: tp.TypeAlias = tuple[
    int,
    int,
    int,
    int,
    int,
    bool,
    Decimal,
    int,
    int,
    Decimal,
    tuple[str, ...],
    str,
]
_VictimSortKey: tp.TypeAlias = tuple[int, int, Decimal, int, Decimal, str]


class _LeaseAccounting:
    """Incremental indexes over the scheduler's active lease snapshot."""

    def __init__(self, leases: tp.Iterable[CapacityLease], *, track_budgets: bool) -> None:
        self._leases_by_id: dict[str, CapacityLease] = {}
        self._leases_by_cell: dict[str, dict[str, CapacityLease]] = {}
        # A workload holding leases on several cells is a multi-cell gang, and
        # preemption has to take all of them or none. Derived from the lease set
        # rather than read off the spec: a *multislice* workload holds one lease
        # covering N workers, so lease count is exactly the discriminator, and it
        # is available here where the workload record is not.
        self._leases_by_workload: dict[str, dict[str, CapacityLease]] = {}
        self._usage_by_scope: dict[IdentityScope, ResourceVector] = {}
        self._usage_by_cell: dict[str, ResourceVector] = {}
        self._reserved_cost_by_scope: dict[IdentityScope, Decimal] = {}
        self._resource_hours_by_scope: dict[IdentityScope, ResourceVector] = {}
        self._track_budgets = track_budgets
        for lease in leases:
            self.add(lease)

    def add(self, lease: CapacityLease) -> None:
        self._leases_by_id[lease.lease_id] = lease
        self._leases_by_cell.setdefault(lease.cell_id, {})[lease.lease_id] = lease
        self._leases_by_workload.setdefault(lease.workload_id, {})[lease.lease_id] = lease
        self._usage_by_cell[lease.cell_id] = self._usage_by_cell.get(lease.cell_id, _EMPTY_RESOURCES) + lease.resources
        for scope in lease.identity.scopes:
            self._usage_by_scope[scope] = self._usage_by_scope.get(scope, _EMPTY_RESOURCES) + lease.resources
            if self._track_budgets:
                self._reserved_cost_by_scope[scope] = (
                    self._reserved_cost_by_scope.get(scope, Decimal(0)) + lease.reserved_cost
                )
                self._resource_hours_by_scope[scope] = (
                    self._resource_hours_by_scope.get(scope, _EMPTY_RESOURCES) + lease.resource_hours_reserved
                )

    def usage(self, scope: IdentityScope) -> ResourceVector:
        return self._usage_by_scope.get(scope, _EMPTY_RESOURCES)

    def reserved_cost(self, scope: IdentityScope) -> Decimal:
        return self._reserved_cost_by_scope.get(scope, Decimal(0))

    def resource_hours(self, scope: IdentityScope) -> ResourceVector:
        return self._resource_hours_by_scope.get(scope, _EMPTY_RESOURCES)

    def remaining(self, cell: CapacityCell) -> ResourceVector:
        used = self._usage_by_cell.get(cell.cell_id, _EMPTY_RESOURCES)
        names = set(cell.capacity) | set(used)
        return ResourceVector((name, max(cell.capacity.get(name) - used.get(name), Decimal(0))) for name in names)

    def leases_for_workload(self, workload_id: str) -> tuple[CapacityLease, ...]:
        """Return every lease one workload holds, across all of its cells."""

        return tuple(self._leases_by_workload.get(workload_id, {}).values())

    def leases_for_cell(self, cell_id: str) -> tp.ValuesView[CapacityLease]:
        return self._leases_by_cell.get(cell_id, {}).values()


_HeapEntry: tp.TypeAlias = tuple[Decimal, str, str, str, int, IdentityScope]
_WorkloadEntry: tp.TypeAlias = tuple[int, str, int, WorkloadSpec]


class _HierarchicalDRFQueue:
    """Incremental tenant/project/principal DRF queue with lazy heap updates."""

    def __init__(
        self,
        workloads: tp.Iterable[WorkloadSpec],
        share: tp.Callable[[IdentityScope], Decimal],
    ) -> None:
        self._share = share
        self._workloads: dict[IdentityScope, list[_WorkloadEntry]] = {}
        self._parent: dict[IdentityScope, IdentityScope | None] = {}
        self._active_children: dict[IdentityScope | None, set[IdentityScope]] = {}
        self._heaps: dict[IdentityScope | None, list[_HeapEntry]] = {}
        self._versions: dict[IdentityScope, int] = {}
        self._next_version = 0

        for workload in workloads:
            tenant, project, principal = workload.identity.scopes
            self._parent[tenant] = None
            self._parent[project] = tenant
            self._parent[principal] = project
            self._active_children.setdefault(None, set()).add(tenant)
            self._active_children.setdefault(tenant, set()).add(project)
            self._active_children.setdefault(project, set()).add(principal)
            self._workloads.setdefault(principal, []).append(
                (workload.queue_position, workload.workload_id, workload.attempt, workload)
            )

        for workload_heap in self._workloads.values():
            heapq.heapify(workload_heap)
        for scope in self._parent:
            self._push(scope)

    def __bool__(self) -> bool:
        return bool(self._active_children.get(None))

    def pop(self) -> WorkloadSpec:
        tenant = self._peek_child(None)
        project = self._peek_child(tenant)
        principal = self._peek_child(project)
        workload = heapq.heappop(self._workloads[principal])[3]
        if not self._workloads[principal]:
            self._deactivate(principal)
        return workload

    def refresh_identities(self, identities: tp.Iterable[WorkloadIdentity]) -> None:
        scopes = {scope for identity in identities for scope in identity.scopes}
        for scope in scopes:
            if scope in self._versions:
                self._push(scope)

    def _push(self, scope: IdentityScope) -> None:
        self._next_version += 1
        version = self._next_version
        self._versions[scope] = version
        heapq.heappush(
            self._heaps.setdefault(self._parent[scope], []),
            (
                self._share(scope),
                scope.tenant_id,
                scope.project_id or "",
                scope.principal_id or "",
                version,
                scope,
            ),
        )

    def _peek_child(self, parent: IdentityScope | None) -> IdentityScope:
        heap = self._heaps[parent]
        while True:
            *_, version, scope = heap[0]
            if self._versions.get(scope) == version:
                return scope
            heapq.heappop(heap)

    def _deactivate(self, scope: IdentityScope) -> None:
        self._versions.pop(scope)
        parent = self._parent[scope]
        siblings = self._active_children[parent]
        siblings.remove(scope)
        if not siblings and parent is not None:
            self._deactivate(parent)


@dataclass(frozen=True, slots=True)
class _PreemptionCandidate:
    plan: PreemptionPlan
    victim_priority_sum: int
    victim_count: int

    @property
    def sort_key(self) -> _PreemptionSortKey:
        score = self.plan.score
        return (
            score.preferred_cell_rank,
            score.preferred_label_misses,
            score.route_priority,
            score.capacity_class_rank,
            score.startup_latency_seconds,
            # Unknown pricing sorts after every known price, not as free.
            score.cost_unknown,
            score.estimated_cost,
            self.victim_priority_sum,
            self.victim_count,
            score.fractional_waste,
            tuple(victim.lease_id for victim in self.plan.victims),
            score.cell_id,
        )


class Scheduler:
    """Apply Eyvan's global admission, fairness, and placement policy.

    The scheduler is deliberately pure: it reads immutable snapshots and
    returns proposed leases and preemption plans. Callers persist decisions
    and realize them through fenced reconciliation.

    Args:
        currency: Currency represented by cell unit prices.
    """

    def __init__(self, *, currency: str = "USD") -> None:
        if not currency or currency != currency.strip():
            raise ValueError("currency must be a non-empty, trimmed string")
        self._currency = currency.upper()

    def schedule(
        self,
        workloads: tp.Iterable[WorkloadSpec],
        cells: tp.Iterable[CapacityCell],
        *,
        leases: tp.Iterable[CapacityLease] = (),
        quotas: tp.Iterable[Quota] = (),
        budgets: tp.Iterable[Budget] = (),
    ) -> SchedulingResult:
        """Schedule a snapshot of pending workloads.

        Priority bands are strict. Within a band, hierarchical weighted DRF
        chooses a tenant, then project, then principal. Successful decisions
        update provisional lease accounting before the next choice.

        Args:
            workloads: Pending immutable workload specs.
            cells: Current capacity-cell snapshots.
            leases: Active capacity leases.
            quotas: Resource limits and fair-share weights by identity scope.
            budgets: Monetary reservation budgets by identity scope.

        Returns:
            Deterministically ordered placements, preemptions, and rejections.

        Raises:
            ValueError: If snapshot identifiers or policy scopes conflict.
        """
        pending = list(workloads)
        cell_list = list(cells)
        active_leases = list(leases)
        quota_list = list(quotas)
        budget_list = list(budgets)
        self._require_unique((cell.cell_id for cell in cell_list), "cell_id")
        self._require_unique((_scope_key(quota.scope) for quota in quota_list), "quota scope")
        self._require_unique((_scope_key(budget.scope) for budget in budget_list), "budget scope")
        cell_by_id = {cell.cell_id: cell for cell in cell_list}
        quota_by_scope = {quota.scope: quota for quota in quota_list}
        budget_by_scope = {budget.scope: budget for budget in budget_list}
        self._validate_snapshot(pending, cell_by_id, active_leases, quota_by_scope, budget_by_scope)
        accounting = _LeaseAccounting(active_leases, track_budgets=bool(budget_by_scope))

        placements: list[PlacementDecision] = []
        preemptions: list[PreemptionPlan] = []
        infeasibilities: list[Infeasibility] = []
        considered_order: list[str] = []
        aggregate_capacity = ResourceVector.total(cell.capacity for cell in cell_by_id.values() if cell.enabled)
        pending_by_priority: dict[PriorityBand, list[WorkloadSpec]] = {priority: [] for priority in PriorityBand}
        for workload in pending:
            pending_by_priority[workload.priority].append(workload)

        for priority in sorted(PriorityBand, reverse=True):
            queue = _HierarchicalDRFQueue(
                pending_by_priority[priority],
                lambda scope: self._weighted_dominant_share(
                    scope,
                    accounting,
                    quota_by_scope,
                    aggregate_capacity,
                ),
            )
            while queue:
                workload = queue.pop()
                considered_order.append(workload.workload_id)
                decision = self._schedule_one(
                    workload,
                    cell_by_id,
                    accounting,
                    quota_by_scope,
                    budget_by_scope,
                )

                if isinstance(decision, PlacementDecision):
                    placements.append(decision)
                    for lease in decision.all_leases:
                        accounting.add(lease)
                    queue.refresh_identities((decision.lease.identity,))
                elif isinstance(decision, PreemptionPlan):
                    preemptions.append(decision)
                    # Reclaimed capacity is not schedulable until the durable
                    # preemption command settles its victims in a later cycle,
                    # but the replacement lease is committed now: charge its
                    # quota, budget, and dominant share before the next choice.
                    accounting.add(decision.replacement_lease)
                    queue.refresh_identities((decision.replacement_lease.identity,))
                    cell_by_id.pop(decision.cell_id, None)
                else:
                    infeasibilities.append(decision)

        return SchedulingResult(
            placements=tuple(placements),
            preemptions=tuple(preemptions),
            infeasibilities=tuple(infeasibilities),
            considered_order=tuple(considered_order),
        )

    def _validate_snapshot(
        self,
        workloads: list[WorkloadSpec],
        cell_by_id: dict[str, CapacityCell],
        leases: list[CapacityLease],
        quota_by_scope: dict[IdentityScope, Quota],
        budget_by_scope: dict[IdentityScope, Budget],
    ) -> None:
        self._require_unique((workload.workload_id for workload in workloads), "workload_id")
        self._require_unique((lease.lease_id for lease in leases), "lease_id")

        workload_attempts = {(workload.workload_id, workload.attempt) for workload in workloads}
        active_attempts = {(lease.workload_id, lease.attempt) for lease in leases}
        duplicate_attempts = sorted(workload_attempts & active_attempts)
        if duplicate_attempts:
            raise ValueError(f"pending workload already has an active lease: {duplicate_attempts[0]!r}")

        unknown_cells = sorted({lease.cell_id for lease in leases} - set(cell_by_id))
        if unknown_cells:
            raise ValueError(f"lease references unknown cell {unknown_cells[0]!r}")
        usage_by_cell: dict[str, ResourceVector] = {}
        for lease in leases:
            usage_by_cell[lease.cell_id] = usage_by_cell.get(lease.cell_id, _EMPTY_RESOURCES) + lease.resources
        overcommitted = sorted(
            cell_id for cell_id, used in usage_by_cell.items() if not used.fits_within(cell_by_id[cell_id].capacity)
        )
        if overcommitted:
            raise ValueError(f"active leases overcommit cell {overcommitted[0]!r}")
        wrong_currency = sorted({budget.currency for budget in budget_by_scope.values()} - {self._currency})
        if wrong_currency:
            raise ValueError(
                f"budget currency {wrong_currency[0]!r} does not match scheduler currency {self._currency!r}"
            )

    @staticmethod
    def _require_unique(values: tp.Iterable[tp.Hashable], field_name: str) -> None:
        seen: set[tp.Hashable] = set()
        for value in values:
            if value in seen:
                raise ValueError(f"duplicate {field_name} {value!r}")
            seen.add(value)

    def _weighted_dominant_share(
        self,
        scope: IdentityScope,
        accounting: _LeaseAccounting,
        quotas: dict[IdentityScope, Quota],
        aggregate_capacity: ResourceVector,
    ) -> Decimal:
        usage = accounting.usage(scope)
        quota = quotas.get(scope)
        denominator_values = aggregate_capacity.as_dict()
        weight = Decimal(1)
        if quota is not None:
            denominator_values.update(quota.limit.as_dict())
            weight = quota.weight
        denominator = ResourceVector(denominator_values)
        return usage.dominant_share(denominator) / weight

    def _schedule_one(
        self,
        workload: WorkloadSpec,
        cells: dict[str, CapacityCell],
        accounting: _LeaseAccounting,
        quotas: dict[IdentityScope, Quota],
        budgets: dict[IdentityScope, Budget],
    ) -> PlacementDecision | PreemptionPlan | Infeasibility:
        if workload.cell_group_count > 1:
            return self._schedule_cell_group(workload, cells, accounting, quotas, budgets)
        if workload.placement is PlacementMode.ELASTIC:
            return self._schedule_elastic(workload, cells, accounting, quotas, budgets)
        if not cells:
            return Infeasibility(
                workload_id=workload.workload_id,
                code=InfeasibilityCode.NO_CELLS,
                message="no capacity cells are registered",
            )

        requested = workload.allocation_resources
        quota_failure = self._quota_infeasibility(workload, requested, accounting, quotas)

        static_failures: list[CellInfeasibility] = []
        feasible_cells: list[CapacityCell] = []
        for cell in sorted(cells.values(), key=lambda candidate: candidate.cell_id):
            failure = self._static_cell_failure(workload, requested, cell)
            if failure is None:
                feasible_cells.append(cell)
            else:
                static_failures.append(failure)
        if not feasible_cells:
            return self._summarize_static_failures(workload, static_failures)

        best_placement: tuple[PlacementScore, CapacityCell, Decimal, Decimal, ResourceVector, tuple[int, ...]] | None = (
            None
        )
        dynamic_failures: list[CellInfeasibility] = []
        budget_allowed_cells: list[CapacityCell] = []
        for cell in feasible_cells:
            hourly_cost, reserved_cost, resource_hours = self._cost(workload, requested, cell)
            if self._unpriced_under_hard_budget(workload, cell, budgets):
                dynamic_failures.append(
                    CellInfeasibility(
                        cell_id=cell.cell_id,
                        code=InfeasibilityCode.UNPRICED_CELL,
                        message=(
                            "cell declares no unit prices at all, so its cost is unknown; "
                            "a hard budget cannot be enforced against unknown prices"
                        ),
                    )
                )
                continue
            if budgets and not self._budget_allows(
                workload.identity,
                reserved_cost,
                resource_hours,
                accounting,
                budgets,
            ):
                dynamic_failures.append(
                    CellInfeasibility(
                        cell_id=cell.cell_id,
                        code=InfeasibilityCode.BUDGET_EXCEEDED,
                        message=f"reservation cost {reserved_cost} exceeds an ancestor budget",
                    )
                )
                continue
            budget_allowed_cells.append(cell)
            remaining = accounting.remaining(cell)
            busy_reason, chips = self._cell_occupancy(workload, cell, accounting)
            if busy_reason is None and requested.fits_within(remaining) and quota_failure is None:
                score = self._placement_score(workload, requested, cell, remaining, reserved_cost)
                if best_placement is None or score < best_placement[0]:
                    best_placement = (score, cell, hourly_cost, reserved_cost, resource_hours, chips)
            else:
                code = (
                    InfeasibilityCode.QUOTA_EXCEEDED
                    if quota_failure is not None
                    else InfeasibilityCode.INSUFFICIENT_CAPACITY
                )
                message = (
                    quota_failure.message
                    if quota_failure is not None
                    else busy_reason
                    if busy_reason is not None
                    else f"gang requires {requested.as_dict()}, available {remaining.as_dict()}"
                )
                dynamic_failures.append(
                    CellInfeasibility(
                        cell_id=cell.cell_id,
                        code=code,
                        message=message,
                    )
                )
        if best_placement is not None:
            score, cell, hourly_cost, reserved_cost, resource_hours, chips = best_placement
            lease = self._make_lease(
                workload, requested, cell, hourly_cost, reserved_cost, resource_hours, tpu_chips=chips
            )
            return PlacementDecision(workload.workload_id, lease, score)

        preemption_plans: list[_PreemptionCandidate] = []
        unsafe_preemption = False
        # A chip share waits for its chips; it never stops another job to get them.
        for cell in () if workload.tpu_chip_count is not None else budget_allowed_cells:
            candidate, unsafe = self._preemption_candidate(workload, requested, cell, accounting, quotas)
            unsafe_preemption |= unsafe
            if candidate is not None:
                preemption_plans.append(candidate)
        if preemption_plans:
            return min(preemption_plans, key=lambda candidate: candidate.sort_key).plan

        if not budget_allowed_cells:
            if dynamic_failures and all(failure.code is InfeasibilityCode.UNPRICED_CELL for failure in dynamic_failures):
                code = InfeasibilityCode.UNPRICED_CELL
                message = (
                    "every feasible cell lacks declared prices entirely, so costs are unknown; "
                    "hard budgets cannot be enforced against unknown prices"
                )
            else:
                code = InfeasibilityCode.BUDGET_EXCEEDED
                message = "every feasible cell exceeds an ancestor budget"
        elif quota_failure is not None:
            code = InfeasibilityCode.QUOTA_EXCEEDED
            message = quota_failure.message
        elif unsafe_preemption:
            code = InfeasibilityCode.PREEMPTION_UNSAFE
            message = "capacity exists only behind lower-priority leases that cannot be safely preempted"
        else:
            code = InfeasibilityCode.INSUFFICIENT_CAPACITY
            message = "no feasible cell has enough unleased or safely reclaimable capacity for the whole gang"
        return Infeasibility(
            workload_id=workload.workload_id,
            code=code,
            message=message,
            cells=tuple(sorted(dynamic_failures, key=lambda failure: failure.cell_id)),
        )

    def _schedule_elastic(
        self,
        workload: WorkloadSpec,
        cells: dict[str, CapacityCell],
        accounting: _LeaseAccounting,
        quotas: dict[IdentityScope, Quota],
        budgets: dict[IdentityScope, Budget],
    ) -> PlacementDecision | Infeasibility:
        """Pack one workload's workers from cells of any matching size.

        The elastic answer to "pods 2 -> 4": the two warm slices are leased as
        they are and only the deficit is provisioned, instead of rebuilding the
        whole allocation at the new shape. Cells are taken biggest-first so a
        packed placement spans as few hardware worlds as the warm supply
        allows. All-or-nothing is preserved exactly as in the atomic group
        path: nothing is leased unless the full worker count is covered, so a
        deficit that never grants holds no warm cell hostage.
        """

        if workload.worker_resource_scope is not WorkerResourceScope.PER_WORKER:
            return Infeasibility(
                workload_id=workload.workload_id,
                code=InfeasibilityCode.HARD_CONSTRAINT,
                message=(
                    "elastic placement requires a per-worker resource scope; "
                    "a gang that encodes a whole atomic cell has no per-worker shape to pack"
                ),
            )
        if not cells:
            return Infeasibility(
                workload_id=workload.workload_id,
                code=InfeasibilityCode.NO_CELLS,
                message="no capacity cells are registered",
            )

        per_worker = workload.total_resources
        need = workload.worker_count
        requested_total = per_worker.scale(need)
        quota_failure = self._quota_infeasibility(workload, requested_total, accounting, quotas)
        static_failures: list[CellInfeasibility] = []
        dynamic_failures: list[CellInfeasibility] = []
        candidates: list[tuple[CapacityCell, Decimal, Decimal, ResourceVector, PlacementScore]] = []

        for cell in sorted(cells.values(), key=lambda candidate: candidate.cell_id):
            per_cell = per_worker.scale(cell.worker_count)
            failure = self._static_cell_failure(workload, per_cell, cell, require_worker_count_equality=False)
            if failure is not None:
                static_failures.append(failure)
                continue
            if cell.worker_count > need:
                # A cell larger than the remaining need is not elastic supply;
                # leasing it would bill for workers nobody asked for. It stays
                # warm for a workload of its own shape.
                continue
            remaining = accounting.remaining(cell)
            exclusive_busy = cell.exclusive_workloads and any(accounting.leases_for_cell(cell.cell_id))
            if exclusive_busy or not per_cell.fits_within(remaining):
                dynamic_failures.append(
                    CellInfeasibility(
                        cell_id=cell.cell_id,
                        code=InfeasibilityCode.INSUFFICIENT_CAPACITY,
                        message=(
                            "cell permits exactly one active workload"
                            if exclusive_busy
                            else f"packed workers require {per_cell.as_dict()}, available {remaining.as_dict()}"
                        ),
                    )
                )
                continue
            hourly_cost, reserved_cost, resource_hours = self._cost(workload, per_cell, cell)
            if self._unpriced_under_hard_budget(workload, cell, budgets):
                dynamic_failures.append(
                    CellInfeasibility(
                        cell_id=cell.cell_id,
                        code=InfeasibilityCode.UNPRICED_CELL,
                        message="cell declares no unit prices at all, so its cost is unknown",
                    )
                )
                continue
            candidates.append(
                (
                    cell,
                    hourly_cost,
                    reserved_cost,
                    resource_hours,
                    self._placement_score(workload, per_cell, cell, remaining, reserved_cost),
                )
            )

        if quota_failure is not None:
            return quota_failure

        ordered = sorted(candidates, key=lambda item: (-item[0].worker_count, item[4], item[0].cell_id))
        # Keep the fewest-cell subset for every reachable worker total. Greedy
        # biggest-first gets stuck on 3+2+2 for a four-worker request, and can
        # choose more worlds than necessary even when it does find a full pack.
        # Snapshot each pass so a cell is never used twice. Equal-size answers
        # retain the earlier, deterministically preferred candidate sequence.
        packs: dict[int, tuple[int, ...]] = {0: ()}
        for index, candidate in enumerate(ordered):
            width = candidate[0].worker_count
            for total, members in tuple(packs.items()):
                combined = total + width
                if combined > need:
                    continue
                previous = packs.get(combined)
                if previous is None or len(members) + 1 < len(previous):
                    packs[combined] = (*members, index)
        packed = max(packs)
        selected = [ordered[index] for index in packs[packed]]

        if packed < need:
            if not candidates and static_failures:
                return self._summarize_static_failures(workload, static_failures)
            return Infeasibility(
                workload_id=workload.workload_id,
                code=InfeasibilityCode.INSUFFICIENT_CAPACITY,
                message=(
                    f"elastic placement requires {need} workers; "
                    f"warm cells cover {packed} and no cell fits the remaining {need - packed}"
                ),
                cells=tuple(sorted((*static_failures, *dynamic_failures), key=lambda failure: failure.cell_id)),
            )

        total_reserved_cost = sum((candidate[2] for candidate in selected), Decimal(0))
        total_resource_hours = ResourceVector.total(candidate[3] for candidate in selected)
        if budgets and not self._budget_allows(
            workload.identity,
            total_reserved_cost,
            total_resource_hours,
            accounting,
            budgets,
        ):
            # The preferred fewest-cell pack may be unaffordable even though
            # another exact pack fits. Preserve the cheapest pack at each width
            # and cardinality, then choose the fewest affordable worlds. Equal
            # worker totals have identical resource-hours, independent of cells.
            affordable: dict[tuple[int, int], tuple[Decimal, tuple[int, ...]]] = {(0, 0): (Decimal(0), ())}
            for index, candidate in enumerate(ordered):
                for (total, count), (cost, members) in tuple(affordable.items()):
                    combined = total + candidate[0].worker_count
                    if combined > need:
                        continue
                    key = (combined, count + 1)
                    price = cost + candidate[2]
                    prior_affordable = affordable.get(key)
                    if prior_affordable is None or price < prior_affordable[0]:
                        affordable[key] = (price, (*members, index))
            alternatives = sorted(
                (count, cost, members) for (total, count), (cost, members) in affordable.items() if total == need
            )
            for _count, cost, members in alternatives:
                if self._budget_allows(workload.identity, cost, total_resource_hours, accounting, budgets):
                    selected = [ordered[index] for index in members]
                    total_reserved_cost = cost
                    break
            else:
                return Infeasibility(
                    workload_id=workload.workload_id,
                    code=InfeasibilityCode.BUDGET_EXCEEDED,
                    message=f"elastic reservation cost {total_reserved_cost} exceeds an ancestor budget",
                )

        leases = tuple(
            self._make_lease(
                workload,
                per_worker.scale(cell.worker_count),
                cell,
                hourly_cost,
                reserved_cost,
                resource_hours,
            )
            for cell, hourly_cost, reserved_cost, resource_hours, _score in selected
        )
        leader_score = selected[0][4]
        score = PlacementScore(
            preferred_cell_rank=leader_score.preferred_cell_rank,
            preferred_label_misses=leader_score.preferred_label_misses,
            route_priority=leader_score.route_priority,
            capacity_class_rank=leader_score.capacity_class_rank,
            startup_latency_seconds=leader_score.startup_latency_seconds,
            cost_unknown=any(candidate[4].cost_unknown for candidate in selected),
            estimated_cost=total_reserved_cost,
            fractional_waste=sum((candidate[4].fractional_waste for candidate in selected), Decimal(0)),
            cell_id=leader_score.cell_id,
        )
        return PlacementDecision(
            workload_id=workload.workload_id,
            lease=leases[0],
            score=score,
            member_leases=leases[1:],
        )

    def _schedule_cell_group(
        self,
        workload: WorkloadSpec,
        cells: dict[str, CapacityCell],
        accounting: _LeaseAccounting,
        quotas: dict[IdentityScope, Quota],
        budgets: dict[IdentityScope, Budget],
    ) -> PlacementDecision | Infeasibility:
        """Place one workload across distinct full-capacity cells atomically.

        A TPU pod group is not a larger provider resource. Each member is a
        normal full pool worker, and every member must be selected before any
        lease is emitted. The existing preemption algorithm is intentionally
        excluded here because it only fences one cell at a time.
        """

        if not cells:
            return Infeasibility(
                workload_id=workload.workload_id,
                code=InfeasibilityCode.NO_CELLS,
                message="no capacity cells are registered",
            )

        requested = workload.allocation_resources
        requested_total = requested.scale(workload.cell_group_count)
        quota_failure = self._quota_infeasibility(workload, requested_total, accounting, quotas)
        static_failures: list[CellInfeasibility] = []
        dynamic_failures: list[CellInfeasibility] = []
        candidates: list[tuple[CapacityCell, Decimal, Decimal, ResourceVector, PlacementScore]] = []

        for cell in sorted(cells.values(), key=lambda candidate: candidate.cell_id):
            failure = self._static_cell_failure(workload, requested, cell)
            if failure is not None:
                static_failures.append(failure)
                continue
            remaining = accounting.remaining(cell)
            exclusive_busy = cell.exclusive_workloads and any(accounting.leases_for_cell(cell.cell_id))
            if exclusive_busy or not requested.fits_within(remaining):
                dynamic_failures.append(
                    CellInfeasibility(
                        cell_id=cell.cell_id,
                        code=InfeasibilityCode.INSUFFICIENT_CAPACITY,
                        message=(
                            "cell permits exactly one active workload"
                            if exclusive_busy
                            else f"pod requires {requested.as_dict()}, available {remaining.as_dict()}"
                        ),
                    )
                )
                continue
            hourly_cost, reserved_cost, resource_hours = self._cost(workload, requested, cell)
            if self._unpriced_under_hard_budget(workload, cell, budgets):
                dynamic_failures.append(
                    CellInfeasibility(
                        cell_id=cell.cell_id,
                        code=InfeasibilityCode.UNPRICED_CELL,
                        message=(
                            "cell declares no unit prices at all, so its cost is unknown; "
                            "a hard budget cannot be enforced against unknown prices"
                        ),
                    )
                )
                continue
            candidates.append(
                (
                    cell,
                    hourly_cost,
                    reserved_cost,
                    resource_hours,
                    self._placement_score(workload, requested, cell, remaining, reserved_cost),
                )
            )

        if quota_failure is not None:
            return quota_failure
        if len(candidates) < workload.cell_group_count:
            if not candidates and static_failures:
                return self._summarize_static_failures(workload, static_failures)
            if (
                not candidates
                and dynamic_failures
                and all(failure.code is InfeasibilityCode.UNPRICED_CELL for failure in dynamic_failures)
            ):
                return Infeasibility(
                    workload_id=workload.workload_id,
                    code=InfeasibilityCode.UNPRICED_CELL,
                    message=(
                        "every feasible cell lacks declared prices entirely, so costs are unknown; "
                        "hard budgets cannot be enforced against unknown prices"
                    ),
                    cells=tuple(sorted(dynamic_failures, key=lambda failure: failure.cell_id)),
                )
            return Infeasibility(
                workload_id=workload.workload_id,
                code=InfeasibilityCode.INSUFFICIENT_CAPACITY,
                message=(
                    f"pod group requires {workload.cell_group_count} distinct cells, "
                    f"but only {len(candidates)} are currently free"
                ),
                cells=tuple(sorted((*static_failures, *dynamic_failures), key=lambda failure: failure.cell_id)),
            )

        ordered = sorted(candidates, key=lambda candidate: candidate[4])
        selected = ordered[: workload.cell_group_count]
        total_reserved_cost = sum((candidate[2] for candidate in selected), Decimal(0))
        total_resource_hours = ResourceVector.total(candidate[3] for candidate in selected)
        if budgets and not self._budget_allows(
            workload.identity,
            total_reserved_cost,
            total_resource_hours,
            accounting,
            budgets,
        ):
            # Every member reserves the same resources for the same duration,
            # so exact-cardinality groups have identical resource-hours. Keep
            # each preferred candidate only if the cheapest remaining members
            # can complete an affordable group. This finds the lexicographically
            # preferred feasible sequence without enumerating combinations or
            # turning soft preferences into hard placement constraints.
            selected = []
            for index, candidate in enumerate(ordered):
                remaining_count = workload.cell_group_count - len(selected) - 1
                completion = heapq.nsmallest(remaining_count, ordered[index + 1 :], key=lambda item: item[2])
                if len(completion) < remaining_count:
                    break
                completion_cost = sum((item[2] for item in (*selected, candidate, *completion)), Decimal(0))
                if self._budget_allows(workload.identity, completion_cost, total_resource_hours, accounting, budgets):
                    selected.append(candidate)
                    if len(selected) == workload.cell_group_count:
                        break
            if len(selected) != workload.cell_group_count:
                return Infeasibility(
                    workload_id=workload.workload_id,
                    code=InfeasibilityCode.BUDGET_EXCEEDED,
                    message=f"pod group reservation cost {total_reserved_cost} exceeds an ancestor budget",
                )
            total_reserved_cost = sum((candidate[2] for candidate in selected), Decimal(0))

        leases = tuple(
            self._make_lease(workload, requested, cell, hourly_cost, reserved_cost, resource_hours)
            for cell, hourly_cost, reserved_cost, resource_hours, _score in selected
        )
        leader_score = selected[0][4]
        score = PlacementScore(
            preferred_cell_rank=leader_score.preferred_cell_rank,
            preferred_label_misses=leader_score.preferred_label_misses,
            route_priority=leader_score.route_priority,
            capacity_class_rank=leader_score.capacity_class_rank,
            startup_latency_seconds=leader_score.startup_latency_seconds,
            cost_unknown=any(candidate[4].cost_unknown for candidate in selected),
            estimated_cost=total_reserved_cost,
            fractional_waste=sum((candidate[4].fractional_waste for candidate in selected), Decimal(0)),
            cell_id=leader_score.cell_id,
        )
        return PlacementDecision(
            workload_id=workload.workload_id,
            lease=leases[0],
            score=score,
            member_leases=leases[1:],
        )

    @staticmethod
    def _static_cell_failure(
        workload: WorkloadSpec,
        requested: ResourceVector,
        cell: CapacityCell,
        *,
        require_worker_count_equality: bool = True,
    ) -> CellInfeasibility | None:
        if not cell.enabled:
            return CellInfeasibility(cell.cell_id, InfeasibilityCode.HARD_CONSTRAINT, "cell is disabled")
        if workload.tpu_chip_count is not None and (not cell.chip_sharing or cell.chips_per_host is None):
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.HARD_CONSTRAINT,
                "cell does not allow TPU chip sharing (set chip_sharing: true on its pool)",
            )
        if require_worker_count_equality and workload.worker_count != cell.worker_count:
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.HARD_CONSTRAINT,
                f"atomic worker count {cell.worker_count} does not match workload worker count {workload.worker_count}",
            )
        if cell.autoscale_request_workload_id is not None and cell.autoscale_request_workload_id != workload.workload_id:
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.HARD_CONSTRAINT,
                "cell is reserved for another autoscale request",
            )
        if workload.runtime is not None and workload.runtime not in cell.runtimes:
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.RUNTIME_UNSUPPORTED,
                f"runtime {workload.runtime!r} is not supported",
            )
        if cell.capacity_class not in workload.allowed_capacity_classes:
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.CAPACITY_CLASS_MISMATCH,
                f"capacity class {cell.capacity_class.value!r} is not allowed",
            )

        labels = dict(cell.labels)
        missing_labels = [(key, value) for key, value in workload.required_labels if labels.get(key) != value]
        if missing_labels:
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.HARD_CONSTRAINT,
                f"required labels do not match: {missing_labels!r}",
            )
        if workload.allowed_label_combinations and not any(
            all(labels.get(key) == value for key, value in combination)
            for combination in workload.allowed_label_combinations
        ):
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.HARD_CONSTRAINT,
                "cell does not match an allowed label combination",
            )
        disallowed = [
            (key, labels.get(key), values)
            for key, values in workload.allowed_label_values
            if labels.get(key) not in values
        ]
        denied = [(key, labels.get(key)) for key, values in workload.denied_label_values if labels.get(key) in values]
        if disallowed or denied:
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.HARD_CONSTRAINT,
                f"cell violates allowed or denied label policy: allowed={disallowed!r}, denied={denied!r}",
            )

        topology = dict(cell.topology)
        for gang in workload.gangs:
            if cell.max_gang_replicas is not None and gang.replicas > cell.max_gang_replicas:
                return CellInfeasibility(
                    cell.cell_id,
                    InfeasibilityCode.REPLICA_COUNT_UNSUPPORTED,
                    f"gang {gang.gang_id!r} has {gang.replicas} replicas; cell supports at most "
                    f"{cell.max_gang_replicas}",
                )
            mismatch = [(key, value) for key, value in gang.required_topology if topology.get(key) != value]
            if mismatch:
                return CellInfeasibility(
                    cell.cell_id,
                    InfeasibilityCode.TOPOLOGY_MISMATCH,
                    f"gang {gang.gang_id!r} topology does not match: {mismatch!r}",
                )
            if cell.max_replica_resources is not None and not gang.resources_per_replica.fits_within(
                cell.max_replica_resources
            ):
                return CellInfeasibility(
                    cell.cell_id,
                    InfeasibilityCode.REPLICA_TOO_LARGE,
                    f"gang {gang.gang_id!r} has a replica larger than the cell's placement unit",
                )

        if not requested.fits_within(cell.capacity):
            return CellInfeasibility(
                cell.cell_id,
                InfeasibilityCode.INSUFFICIENT_CAPACITY,
                "the whole gang is larger than the cell's total capacity",
            )
        return None

    @staticmethod
    def _summarize_static_failures(
        workload: WorkloadSpec,
        failures: list[CellInfeasibility],
    ) -> Infeasibility:
        codes = {failure.code for failure in failures}
        code = next(iter(codes)) if len(codes) == 1 else InfeasibilityCode.HARD_CONSTRAINT
        return Infeasibility(
            workload_id=workload.workload_id,
            code=code,
            message="no cell satisfies all hard workload constraints",
            cells=tuple(failures),
        )

    @staticmethod
    def _quota_infeasibility(
        workload: WorkloadSpec,
        requested: ResourceVector,
        accounting: _LeaseAccounting,
        quotas: dict[IdentityScope, Quota],
        victims: tp.Iterable[CapacityLease] = (),
    ) -> Infeasibility | None:
        victim_list = tuple(victims)
        for scope in workload.identity.scopes:
            quota = quotas.get(scope)
            if quota is None:
                continue
            reclaimed = ResourceVector.total(lease.resources for lease in victim_list if scope in lease.identity.scopes)
            projected = accounting.usage(scope) - reclaimed + requested
            if not projected.within_limits(quota.limit):
                return Infeasibility(
                    workload_id=workload.workload_id,
                    code=InfeasibilityCode.QUOTA_EXCEEDED,
                    message=f"projected usage {projected.as_dict()} exceeds quota at {scope!r}",
                )
        return None

    @staticmethod
    def _unpriced_under_hard_budget(
        workload: WorkloadSpec,
        cell: CapacityCell,
        budgets: dict[IdentityScope, Budget],
    ) -> bool:
        """Return whether placing on ``cell`` would defeat a hard budget.

        A hard budget is only enforceable against a cell that declares a
        price model at all: an unpriced cell computes zero reservation cost
        for everything, which used to sail through the gate, and unknown is
        not zero. A *partially* priced cell is different by contract -- its
        ``unit_prices`` are the operator's complete statement of what the
        runtime charges for (unlisted resources are uncharged), so its cost
        estimate stays deterministic and the budget gate stays meaningful;
        the ``cost_unknown`` score flag still prefers fully priced cells
        when costs tie-break.
        """
        if not budgets:
            return False
        hard_scopes = {scope for scope, budget in budgets.items() if budget.hard_limit}
        if not hard_scopes.intersection(workload.identity.scopes):
            return False
        return not cell.declares_prices

    @staticmethod
    def _budget_allows(
        identity: WorkloadIdentity,
        reserved_cost: Decimal,
        resource_hours: ResourceVector,
        accounting: _LeaseAccounting,
        budgets: dict[IdentityScope, Budget],
    ) -> bool:
        for scope in identity.scopes:
            budget = budgets.get(scope)
            if budget is None or not budget.hard_limit:
                continue
            existing = accounting.reserved_cost(scope)
            if budget.spent + existing + reserved_cost > budget.limit:
                return False
            existing_hours = accounting.resource_hours(scope)
            projected_hours = budget.accelerator_hours_spent + existing_hours + resource_hours
            if not projected_hours.within_limits(budget.accelerator_hour_limits):
                return False
        return True

    @staticmethod
    def _cost(
        workload: WorkloadSpec,
        requested: ResourceVector,
        cell: CapacityCell,
    ) -> tuple[Decimal, Decimal, ResourceVector]:
        hourly = cell.hourly_cost(requested)
        duration_hours = Decimal(workload.estimated_duration_seconds) / Decimal(3600)
        reserved = hourly * duration_hours
        resource_hours = requested.scale(duration_hours)
        return hourly, reserved, resource_hours

    def _make_lease(
        self,
        workload: WorkloadSpec,
        requested: ResourceVector,
        cell: CapacityCell,
        hourly_cost: Decimal,
        reserved_cost: Decimal,
        resource_hours: ResourceVector,
        *,
        tpu_chips: tuple[int, ...] = (),
    ) -> CapacityLease:
        lease_key = f"{workload.workload_id}\0{workload.attempt}\0{workload.generation}\0{cell.cell_id}".encode()
        return CapacityLease(
            lease_id=f"lease_{hashlib.sha256(lease_key).hexdigest()[:32]}",
            workload_id=workload.workload_id,
            attempt=workload.attempt,
            identity=workload.identity,
            cell_id=cell.cell_id,
            resources=requested,
            priority=workload.priority,
            hourly_cost=hourly_cost,
            reserved_cost=reserved_cost,
            resource_hours_reserved=resource_hours,
            checkpoint_policy=workload.checkpoint_policy,
            revocable=workload.revocable,
            tpu_chips=tpu_chips,
        )

    @staticmethod
    def _cell_occupancy(
        workload: WorkloadSpec,
        cell: CapacityCell,
        accounting: _LeaseAccounting,
    ) -> tuple[str | None, tuple[int, ...]]:
        """Return why ``cell`` cannot take this workload now (or ``None``) and the chips it would get.

        A whole-cell workload keeps the exclusive rule: one workload per native
        cell. A chip workload may join a chip-sharing cell whose every lease
        holds chips, and is granted chips disjoint from all of them.
        """

        leases = tuple(accounting.leases_for_cell(cell.cell_id))
        if workload.tpu_chip_count is None:
            if cell.exclusive_workloads and leases:
                return "cell permits exactly one active workload", ()
            return None, ()
        if any(not lease.tpu_chips for lease in leases):
            return "cell is held whole by another workload", ()
        if cell.chips_per_host is None:
            return "cell does not allow TPU chip sharing", ()
        total = int(cell.capacity.get("tpu"))
        used = {chip for lease in leases for chip in lease.tpu_chips}
        chips = pick_chips(
            frozenset(range(total)) - used,
            workload.tpu_chip_count,
            workload.tpu_chip_indexes,
            total_chips=total,
            chips_per_host=cell.chips_per_host,
        )
        if chips is not None:
            return None, chips
        if workload.tpu_chip_indexes is not None:
            held = sorted(set(workload.tpu_chip_indexes) & used)
            return f"TPU chips {held} are held by another workload", ()
        return f"no host has {workload.tpu_chip_count} free TPU chips", ()

    @staticmethod
    def _placement_score(
        workload: WorkloadSpec,
        requested: ResourceVector,
        cell: CapacityCell,
        remaining_before: ResourceVector,
        estimated_cost: Decimal,
    ) -> PlacementScore:
        if workload.preferred_cells:
            try:
                preferred_cell_rank = workload.preferred_cells.index(cell.cell_id)
            except ValueError:
                preferred_cell_rank = len(workload.preferred_cells)
        else:
            preferred_cell_rank = 0

        labels = dict(cell.labels)
        preferred_label_misses = sum(labels.get(key) != value for key, value in workload.preferred_labels)
        capacity_class_rank = workload.capacity_class_preference.index(cell.capacity_class)
        remaining_after = remaining_before - requested
        fractional_waste = sum(
            (remaining_after.get(name) / capacity for name, capacity in cell.capacity.items() if capacity > 0),
            Decimal(0),
        )
        return PlacementScore(
            preferred_cell_rank=preferred_cell_rank,
            preferred_label_misses=preferred_label_misses,
            route_priority=cell.route_priority,
            capacity_class_rank=capacity_class_rank,
            startup_latency_seconds=cell.startup_latency_seconds,
            cost_unknown=not cell.prices_cover(requested),
            estimated_cost=estimated_cost,
            fractional_waste=fractional_waste,
            cell_id=cell.cell_id,
        )

    def _preemption_candidate(
        self,
        workload: WorkloadSpec,
        requested: ResourceVector,
        cell: CapacityCell,
        accounting: _LeaseAccounting,
        quotas: dict[IdentityScope, Quota],
    ) -> tuple[_PreemptionCandidate | None, bool]:
        cell_leases = tuple(accounting.leases_for_cell(cell.cell_id))
        exclusive_busy = cell.exclusive_workloads and bool(cell_leases)
        remaining = _EMPTY_RESOURCES if exclusive_busy else accounting.remaining(cell)
        deficit = self._deficit(requested, remaining)
        lower_priority = [lease for lease in cell_leases if lease.priority < workload.priority and lease.revocable]
        eligible = [lease for lease in lower_priority if self._victim_eligible(lease, workload.priority)]
        eligible.sort(key=lambda lease: self._victim_sort_key(lease, deficit))

        reclaimed = ResourceVector()
        selected: list[CapacityLease] = []
        chosen_ids: set[str] = set()
        for lease in eligible:
            if lease.lease_id in chosen_ids:
                continue
            # One workload attempt holding several cells is one gang: stopping
            # a strict subset strands its surviving members. Other attempts are
            # separate executions, not siblings. All members must independently
            # satisfy the same priority and safe-stop policy as local victims.
            victims = tuple(
                sibling
                for sibling in accounting.leases_for_workload(lease.workload_id)
                if sibling.attempt == lease.attempt
            )
            if any(not self._victim_eligible(victim, workload.priority) for victim in victims):
                continue
            for victim in victims:
                if victim.lease_id in chosen_ids:
                    continue
                chosen_ids.add(victim.lease_id)
                selected.append(victim)
                # Only capacity freed *on this cell* helps this placement fit;
                # a sibling stopped on another cell frees nothing here.
                if victim.cell_id == cell.cell_id:
                    reclaimed += victim.resources
            # An exclusive cell is only free once *its own* leases are all
            # victims. ``selected`` may now also hold gang siblings on other
            # cells, so count what this cell actually gives up.
            selected_here = sum(1 for victim in selected if victim.cell_id == cell.cell_id)
            if (
                (not exclusive_busy or selected_here == len(cell_leases))
                and requested.fits_within(remaining + reclaimed)
                and self._quota_infeasibility(
                    workload,
                    requested,
                    accounting,
                    quotas,
                    selected,
                )
                is None
            ):
                break
        quota_satisfied = (
            self._quota_infeasibility(
                workload,
                requested,
                accounting,
                quotas,
                selected,
            )
            is None
        )
        replacement_fits = requested.fits_within(remaining + reclaimed) and (
            not exclusive_busy or sum(1 for victim in selected if victim.cell_id == cell.cell_id) == len(cell_leases)
        )
        if not replacement_fits or not quota_satisfied:
            all_reclaimable = ResourceVector.total(lease.resources for lease in lower_priority)
            unsafe = (
                requested.fits_within(remaining + all_reclaimable)
                and (not exclusive_busy or len(lower_priority) == len(cell_leases))
                and self._quota_infeasibility(
                    workload,
                    requested,
                    accounting,
                    quotas,
                    lower_priority,
                )
                is None
                and len(eligible) < len(lower_priority)
            )
            return None, unsafe

        victim_actions = tuple(
            PreemptionVictim(
                lease_id=lease.lease_id,
                workload_id=lease.workload_id,
                action=(
                    PreemptionActionKind.CHECKPOINT_AND_STOP
                    if lease.checkpoint_policy.checkpointable
                    else PreemptionActionKind.TERMINATE
                ),
                grace_period_seconds=lease.checkpoint_policy.grace_period_seconds,
            )
            for lease in selected
        )
        hourly_cost, reserved_cost, resource_hours = self._cost(workload, requested, cell)
        replacement = self._make_lease(
            workload,
            requested,
            cell,
            hourly_cost,
            reserved_cost,
            resource_hours,
        )
        score = self._placement_score(workload, requested, cell, remaining + reclaimed, reserved_cost)
        plan = PreemptionPlan(
            workload_id=workload.workload_id,
            cell_id=cell.cell_id,
            victims=victim_actions,
            replacement_lease=replacement,
            score=score,
        )
        return (
            _PreemptionCandidate(
                plan=plan,
                victim_priority_sum=sum(int(lease.priority) for lease in selected),
                victim_count=len(selected),
            ),
            False,
        )

    @staticmethod
    def _victim_eligible(lease: CapacityLease, priority: PriorityBand) -> bool:
        return (
            lease.priority < priority
            and lease.revocable
            and (lease.checkpoint_policy.checkpointable or lease.checkpoint_policy.allow_forced_termination)
        )

    @staticmethod
    def _deficit(requested: ResourceVector, available: ResourceVector) -> ResourceVector:
        return ResourceVector(
            (name, max(quantity - available.get(name), Decimal(0))) for name, quantity in requested.items()
        )

    @staticmethod
    def _victim_sort_key(lease: CapacityLease, deficit: ResourceVector) -> _VictimSortKey:
        coverage = sum(
            (min(lease.resources.get(name), quantity) / quantity for name, quantity in deficit.items() if quantity > 0),
            Decimal(0),
        )
        action_rank = 0 if lease.checkpoint_policy.checkpointable else 1
        return (
            int(lease.priority),
            action_rank,
            -coverage,
            lease.checkpoint_policy.grace_period_seconds,
            lease.reserved_cost,
            lease.lease_id,
        )
