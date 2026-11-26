# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed, read-only autoscaling progress for one workload."""

from __future__ import annotations

from dataclasses import dataclass

from eyvan.capacity import PENDING_CAPACITY_STATUSES, QUEUED_CAPACITY_STATUSES, CapacityStatus
from eyvan.capacity.acquisition_backoff import is_acquisition_backoff_reason
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.domain.models import PlacementMode
from eyvan.records import (
    ACQUISITION_EXHAUSTED_SCHEDULING_REASON,
    AutoscaleBlockReason,
    CapacityRetryProgress,
    CellRecord,
    CellState,
    OutboxRecord,
    OutboxState,
    Principal,
    WorkloadCapacityPhaseProgress,
    WorkloadCapacityRequest,
    WorkloadCapacitySummary,
    WorkloadPoolProgress,
    WorkloadProgress,
    WorkloadProgressPhase,
    WorkloadRecord,
    WorkloadState,
    candidate_pool_templates,
    scale_resources,
)
from eyvan.server.persistence.base import Repository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.provider_diagnostics import safe_provider_diagnostic
from eyvan.server.service.state import ACTIVE_LEASE_STATES, TERMINAL_WORKLOAD_STATES

_POOL_LABEL = "eyvan_pool"
_POOL_TEMPLATE_LABEL = "eyvan_pool_template"


@dataclass(frozen=True, slots=True)
class WorkloadProgressResolver:
    """Resolve a workload's exact physical capacity request and current supply."""

    service: ControlPlaneService
    repository: Repository

    async def resolve(self, principal: Principal, workload_id: str) -> WorkloadProgress:
        """Return one authorized workload's pool, capacity, and retry progress."""

        workload = await self.service.get_workload(principal, workload_id)
        pool_id = workload.spec.required_labels.get(_POOL_LABEL)
        async with self.repository.transaction() as transaction:
            pool = None if pool_id is None else await transaction.get_capacity_pool(pool_id)
            request_cells = await transaction.list_cells_for_autoscale_request(workload.id)
            leases = await transaction.list_leases(workload_id=workload.id)
            leased_cells: list[CellRecord] = []
            for lease in leases:
                if lease.state not in ACTIVE_LEASE_STATES:
                    continue
                cell = await transaction.get_cell(lease.cell_id)
                if cell is not None:
                    leased_cells.append(cell)

            cells_by_id = {cell.id: cell for cell in request_cells}
            for cell in leased_cells:
                cells_by_id[cell.id] = cell
            if workload.spec.placement is PlacementMode.ELASTIC:
                # Elastic packs same-type cells of any size, so the exact-shape
                # filter above would report a pending pack as 0/N even while
                # warm cells cover most of it. Count what the scheduler would
                # actually pack: this workload's own cells plus free warm ones.
                fleet_cells = await transaction.list_cells()
                fleet_leases = await transaction.list_leases()
                leased_by_anyone = frozenset(
                    lease.cell_id for lease in fleet_leases if lease.state in ACTIVE_LEASE_STATES
                )
                leased_by_workload = frozenset(
                    lease.cell_id
                    for lease in fleet_leases
                    if lease.state in ACTIVE_LEASE_STATES and lease.workload_id == workload.id
                )
                matching_cells = tuple(
                    sorted(
                        _elastic_pack_candidates(workload, pool_id, fleet_cells, leased_by_anyone, leased_by_workload),
                        key=lambda candidate: candidate.id,
                    )
                )
            else:
                matching_cells = tuple(
                    cell
                    for cell in sorted(cells_by_id.values(), key=lambda candidate: candidate.id)
                    if _matches_workload_capacity(cell, workload, pool_id)
                )
            # Name the zone that is actually building this workload's capacity.
            # A raced workload's first choice may have refused it outright;
            # presenting that zone would direct diagnostics to the wrong place.
            leading = _leading_pool_id(matching_cells, pool_id)
            if leading is not None and leading != pool_id:
                pool = await transaction.get_capacity_pool(leading) or pool
            commands = await transaction.list_outbox_for_aggregates(
                "cell",
                {cell.id: cell.generation for cell in matching_cells},
                topic="capacity.apply",
            )
            # Read in the same transaction as everything else on this route. The
            # dashboard and ``client/callables.py`` already poll it, so a training
            # loop's own progress needs no new polling loop anywhere.
            reports = await transaction.list_progress_reports(workload.id)

        capacity = _capacity_summary(matching_cells)
        retry = _with_lost_cell_diagnostic(_capacity_retry(matching_cells, commands), matching_cells)
        resources_per_cell = workload.spec.resources_per_cell
        return WorkloadProgress(
            workload_id=workload.id,
            workload_state=workload.state,
            phase=_phase(workload, capacity, retry),
            # The scheduler rewrites ``scheduling_reason`` every pass, usually
            # to a generic "no cell satisfies" -- which, for a workload the
            # planner has stopped acquiring for, hides the one sentence that
            # names the remedy.
            scheduling_reason=(
                ACQUISITION_EXHAUSTED_SCHEDULING_REASON
                if workload.autoscale_block_reason is AutoscaleBlockReason.CAPACITY_ACQUISITION_EXHAUSTED
                else workload.scheduling_reason
            ),
            failure_reason=workload.failure_reason,
            autoscale_block_reason=workload.autoscale_block_reason,
            queue_admission_denial=workload.queue_admission_denial,
            request=WorkloadCapacityRequest(
                cells=workload.spec.cell_group_count,
                workers_per_cell=workload.spec.worker_count,
                resources_per_cell=resources_per_cell,
                total_resources=scale_resources(resources_per_cell, workload.spec.cell_group_count),
            ),
            pool=None
            if pool is None
            else WorkloadPoolProgress(
                id=pool.id,
                fleet_name=pool.fleet_name,
                node_type=pool.node_type,
                provider=pool.provider,
                provider_project=pool.provider_project,
                region=pool.region,
                zone=pool.zone,
                accelerator_type=pool.accelerator_type,
                capacity_class=pool.capacity_class,
            ),
            capacity=capacity,
            retry=retry,
            user_progress=tuple(report for report in reports if report.attempt == workload.attempt),
        )


def _matches_workload_capacity(cell: CellRecord, workload: WorkloadRecord, pool_id: str | None) -> bool:
    """Return whether a live cell is an exact physical candidate for a pinned workload.

    Any of the workload's candidate pools counts, not only its first choice: a
    racing cross-zone workload is provisioning in every zone it asked, and
    reporting only the first would show "0 cells" beside capacity being built
    for it elsewhere.
    """

    if pool_id is None or cell.state is CellState.DELETED:
        return False
    if cell.worker_count != workload.spec.worker_count:
        return False
    candidates = candidate_pool_templates(workload.spec.required_labels) or {pool_id: None}
    if cell.pool_id not in candidates:
        return False
    template = candidates[cell.pool_id]
    return template is None or cell.labels.get(_POOL_TEMPLATE_LABEL) == template


def _elastic_pack_candidates(
    workload: WorkloadRecord,
    pool_id: str | None,
    cells: tuple[CellRecord, ...] | list[CellRecord],
    leased_by_anyone: frozenset[str],
    leased_by_workload: frozenset[str],
) -> tuple[CellRecord, ...]:
    """Return the cells an elastic placement would pack right now.

    Cells this workload already owns or leases always count. Free warm cells
    then supply the largest subset that fits the remaining workers, preferring
    fewer cells as the scheduler does. Never overshoot: an oversized pack is
    not dispatchable. A snapshot, not a reservation: a warm cell stolen by
    another workload before the commit stops counting on the next poll.
    """

    if pool_id is None:
        return ()
    need = workload.spec.worker_count
    candidates = candidate_pool_templates(workload.spec.required_labels) or {pool_id: None}
    own: list[CellRecord] = []
    warm: list[CellRecord] = []
    for cell in cells:
        if cell.pool_id not in candidates or cell.state is CellState.DELETED or cell.worker_count > need:
            continue
        template = candidates[cell.pool_id]
        if template is not None and cell.labels.get(_POOL_TEMPLATE_LABEL) != template:
            continue
        if cell.id in leased_by_workload:
            own.append(cell)
        elif cell.id not in leased_by_anyone:
            if cell.autoscale_request_workload_id == workload.id:
                own.append(cell)
            elif cell.state is CellState.READY and cell.autoscale_request_workload_id is None:
                warm.append(cell)
    remaining = need - sum(cell.worker_count for cell in own)
    ordered = sorted(warm, key=lambda candidate: (-candidate.worker_count, candidate.id))
    # Keep the fewest cells at each reachable width, just as the scheduler
    # does. Greedy either overshoots (2+2 for three) or misses an exact subset
    # (3+2+2 for four). Snapshot each pass so a cell is never counted twice.
    packs: dict[int, tuple[int, ...]] = {0: ()}
    for index, cell in enumerate(ordered):
        for total, members in tuple(packs.items()):
            combined = total + cell.worker_count
            if combined > remaining:
                continue
            previous = packs.get(combined)
            if previous is None or len(members) + 1 < len(previous):
                packs[combined] = (*members, index)
    return (*own, *(ordered[index] for index in packs[max(packs)]))


def _capacity_summary(cells: tuple[CellRecord, ...]) -> WorkloadCapacitySummary:
    """Aggregate matching cells without exposing unrelated fleet inventory."""

    grouped: dict[tuple[CellState, CapacityStatus | None], tuple[int, int]] = {}
    ready_cells = ready_workers = 0
    requesting_cells = requesting_workers = 0
    waiting_cells = waiting_workers = 0
    provisioning_cells = provisioning_workers = 0
    lost_cells = lost_workers = 0

    for cell in cells:
        key = (cell.state, cell.provider_phase)
        previous_cells, previous_workers = grouped.get(key, (0, 0))
        grouped[key] = (previous_cells + 1, previous_workers + cell.worker_count)

        if cell.state is CellState.READY:
            ready_cells += 1
            ready_workers += cell.worker_count
        if cell.provider_phase is CapacityStatus.REQUESTING:
            requesting_cells += 1
            requesting_workers += cell.worker_count
        if cell.provider_phase in QUEUED_CAPACITY_STATUSES:
            waiting_cells += 1
            waiting_workers += cell.worker_count
        if _is_provisioning(cell):
            provisioning_cells += 1
            provisioning_workers += cell.worker_count
        if cell.state is CellState.UNAVAILABLE or cell.provider_phase is CapacityStatus.PROVIDER_ERROR:
            lost_cells += 1
            lost_workers += cell.worker_count

    phases = tuple(
        WorkloadCapacityPhaseProgress(
            state=state,
            provider_phase=provider_phase,
            cells=cell_count,
            workers=worker_count,
        )
        for (state, provider_phase), (cell_count, worker_count) in sorted(
            grouped.items(), key=lambda item: (item[0][0].value, "" if item[0][1] is None else item[0][1].value)
        )
    )
    active_cells = len(cells)
    active_workers = sum(cell.worker_count for cell in cells)
    return WorkloadCapacitySummary(
        active_cells=active_cells,
        active_workers=active_workers,
        ready_cells=ready_cells,
        ready_workers=ready_workers,
        requesting_cells=requesting_cells,
        requesting_workers=requesting_workers,
        waiting_cells=waiting_cells,
        waiting_workers=waiting_workers,
        provisioning_cells=provisioning_cells,
        provisioning_workers=provisioning_workers,
        lost_cells=lost_cells,
        lost_workers=lost_workers,
        observed_tpu_vms=_observed_tpu_vms(cells),
        phases=phases,
    )


def _leading_pool_id(cells: tuple[CellRecord, ...], first_choice: str | None) -> str | None:
    """Return the pool of the cell furthest along, preferring the first choice on a tie."""

    def progress(cell: CellRecord) -> int:
        if cell.state is CellState.READY:
            return 4
        if _is_provisioning(cell) or cell.provider_phase in {CapacityStatus.ACTIVE, CapacityStatus.EXECUTION_CONNECTING}:
            return 3
        if cell.provider_phase in QUEUED_CAPACITY_STATUSES:
            return 2
        if cell.provider_phase is CapacityStatus.REQUESTING:
            return 1
        return 0

    ranked = [(progress(cell), cell.pool_id == first_choice, cell.pool_id) for cell in cells if cell.pool_id]
    if not ranked or max(ranked)[0] == 0:
        return first_choice
    return max(ranked)[2]


def _observed_tpu_vms(cells: tuple[CellRecord, ...]) -> int | None:
    """Return a total only when GCP observed every matching TPU allocation."""

    if not cells:
        return None
    total = 0
    for cell in cells:
        count = cell.observed_tpu_vms
        if count is None:
            return None
        total += count
    return total


def _is_provisioning(cell: CellRecord) -> bool:
    """Return whether one registered cell is actively acquiring or connecting capacity."""

    if cell.state is not CellState.REGISTERING:
        return False
    return (
        cell.provider_phase is not None
        and cell.provider_phase is not CapacityStatus.REQUESTING
        and (cell.provider_phase in PENDING_CAPACITY_STATUSES and cell.provider_phase not in QUEUED_CAPACITY_STATUSES)
    )


def _capacity_retry(
    cells: tuple[CellRecord, ...],
    commands: list[OutboxRecord],
) -> CapacityRetryProgress | None:
    """Return the newest current-generation capacity mutation for matching cells."""

    generation_by_cell = {cell.id: cell.generation for cell in cells}
    relevant = tuple(
        command
        for command in commands
        if command.topic == "capacity.apply"
        and command.aggregate_type == "cell"
        and generation_by_cell.get(command.aggregate_id) == command.generation
    )
    if not relevant:
        return None
    command = max(relevant, key=lambda candidate: (candidate.updated_at, candidate.created_at, candidate.id))
    return CapacityRetryProgress(
        state=command.state,
        attempts=command.attempts,
        next_attempt_at=command.available_at if command.state is OutboxState.FAILED else None,
        diagnostic=safe_provider_diagnostic(command.last_error),
        failure_code=command.failure_code,
    )


def _with_lost_cell_diagnostic(
    retry: CapacityRetryProgress | None,
    cells: tuple[CellRecord, ...],
) -> CapacityRetryProgress | None:
    """Carry a lost cell's own cause when the capacity command has nothing to say.

    A capacity apply can *succeed* and the cell still be lost minutes later --
    a Spot insert that fails after the create call returned, a reclamation. The
    command then holds no error, but the cell's record can still explain why
    capacity was lost. Only fills a gap: a command's own diagnostic wins, and no
    retry record is invented where no command exists.
    """

    if retry is None or retry.diagnostic is not None:
        return retry
    lost = [
        cell
        for cell in cells
        if (cell.state is CellState.UNAVAILABLE or cell.provider_phase is CapacityStatus.PROVIDER_ERROR)
        and cell.unavailable_detail
    ]
    if not lost:
        return retry
    newest = max(lost, key=lambda cell: (cell.updated_at, cell.id))
    diagnostic = safe_provider_diagnostic(newest.unavailable_detail)
    if diagnostic is None:
        return retry
    return retry.model_copy(
        update={"diagnostic": diagnostic, "failure_code": retry.failure_code or newest.provider_failure_code}
    )


def _phase(
    workload: WorkloadRecord,
    capacity: WorkloadCapacitySummary,
    retry: CapacityRetryProgress | None,
) -> WorkloadProgressPhase:
    """Map durable workload and capacity state to one user-facing phase."""

    # A queue-admission denial explains two situations and no others: a workload
    # still waiting because of it, and a terminal workload that died of it.
    #
    # It used to short-circuit ahead of every state, and nothing ever clears the
    # field -- so a workload that was denied once, then acquired capacity and ran,
    # reported "queue admission denied" while it was running and after it
    # succeeded. The advisory outlived the condition it described.
    denied = workload.autoscale_block_reason is AutoscaleBlockReason.QUEUE_ADMISSION_DENIED
    if denied and workload.state in TERMINAL_WORKLOAD_STATES:
        return WorkloadProgressPhase.QUEUE_ADMISSION_DENIED
    if workload.state in TERMINAL_WORKLOAD_STATES:
        return WorkloadProgressPhase.TERMINAL
    if workload.state is WorkloadState.CANCELLING:
        return WorkloadProgressPhase.CANCELLING
    if workload.state is WorkloadState.CHECKPOINTING:
        return WorkloadProgressPhase.CHECKPOINTING
    if workload.state is WorkloadState.RUNNING:
        return WorkloadProgressPhase.RUNNING
    if workload.state is WorkloadState.DISPATCHING:
        return WorkloadProgressPhase.DISPATCHING
    # One zone's denial does not describe a cross-zone workload that another zone
    # is now acquiring for. Active acquisition elsewhere takes precedence over
    # a previous candidate's denial.
    acquiring_elsewhere = bool(
        capacity.requesting_workers
        or capacity.waiting_workers
        or capacity.provisioning_workers
        or capacity.ready_workers
        or any(
            entry.provider_phase in {CapacityStatus.ACTIVE, CapacityStatus.EXECUTION_CONNECTING}
            for entry in capacity.phases
        )
    )
    if denied and not acquiring_elsewhere:
        return WorkloadProgressPhase.QUEUE_ADMISSION_DENIED
    if workload.preemption_id is not None:
        return WorkloadProgressPhase.PREEMPTED
    if workload.state is WorkloadState.LEASED:
        return WorkloadProgressPhase.WAITING_FOR_DISPATCH
    if workload.autoscale_block_reason is AutoscaleBlockReason.SHARED_TPU_QUOTA:
        return WorkloadProgressPhase.QUOTA_BLOCKED
    if workload.autoscale_block_reason is AutoscaleBlockReason.CAPACITY_ACQUISITION_EXHAUSTED:
        # The planner has stopped asking every zone this workload accepts, so
        # do not describe it as requesting capacity. Use an existing phase: a
        # new wire value would break every older SDK mid-wait.
        return WorkloadProgressPhase.PROVIDER_FAILURE
    if workload.state is WorkloadState.ADMITTED:
        return WorkloadProgressPhase.WAITING_FOR_SCHEDULER
    requested_workers = workload.spec.cell_group_count * workload.spec.worker_count
    if capacity.ready_workers >= requested_workers:
        return WorkloadProgressPhase.READY_FOR_DISPATCH
    if is_acquisition_backoff_reason(workload.scheduling_reason) and not (
        capacity.waiting_workers
        or capacity.provisioning_workers
        or capacity.ready_workers
        or any(
            entry.provider_phase in {CapacityStatus.ACTIVE, CapacityStatus.EXECUTION_CONNECTING}
            for entry in capacity.phases
        )
    ):
        # Every pool this workload accepts has refused its last requests. A
        # fresh request in flight, or none at all while the pool waits out its
        # backoff, must not hide the repeated failures as ordinary acquisition.
        # Use an existing phase so older SDKs keep decoding it; scheduling_reason carries
        # the last error and when the next attempt happens.
        return WorkloadProgressPhase.PROVIDER_FAILURE
    if capacity.requesting_workers:
        return WorkloadProgressPhase.REQUESTING_CAPACITY
    if any(entry.provider_phase is CapacityStatus.EXECUTION_CONNECTING for entry in capacity.phases):
        return WorkloadProgressPhase.NATIVE_WORKERS_CONNECTING
    if any(entry.provider_phase is CapacityStatus.ACTIVE for entry in capacity.phases):
        return WorkloadProgressPhase.CAPACITY_ALLOCATED
    if capacity.provisioning_workers:
        return WorkloadProgressPhase.PROVISIONING
    if capacity.waiting_workers:
        return WorkloadProgressPhase.WAITING_FOR_CAPACITY
    if any(entry.provider_phase is CapacityStatus.HALTED_QUOTA for entry in capacity.phases):
        return WorkloadProgressPhase.QUOTA_BLOCKED
    if retry is not None and retry.failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED:
        return WorkloadProgressPhase.QUEUE_ADMISSION_DENIED
    if capacity.lost_workers or (retry is not None and retry.diagnostic is not None):
        return WorkloadProgressPhase.PROVIDER_FAILURE
    return WorkloadProgressPhase.WAITING_FOR_AUTOSCALER


__all__ = ("WorkloadProgressResolver",)
