# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Operator-driven fleet, pool, and cell administration."""

from __future__ import annotations

import asyncio
import time
import typing as tp
import uuid
from datetime import datetime

from eyvan._internal.build_identity import build_identity
from eyvan._internal.json_types import JsonObject
from eyvan.capacity import (
    CapacityStatus,
)
from eyvan.capacity.acquisition_backoff import (
    PoolAcquisitionBackoff,
    acquisition_backoff_reason,
    is_acquisition_backoff_reason,
    workload_acquisition_backoff,
)
from eyvan.capacity.audit import CapacityLifecycleEvent, CapacityReplacementReason
from eyvan.capacity.autoscale.pool_capacity import ClusterCapacityPass
from eyvan.capacity.planning import (
    CapacityPlanner,
    CapacitySimulation,
    PoolScalePlan,
    cell_matches_pool,
    cell_provides_pool_supply,
    cell_requires_replacement,
    next_pool_cell_index,
    pool_cell_id,
)
from eyvan.domain import (
    InfeasibilityCode,
    PlacementMode,
    SchedulingResult,
)
from eyvan.records import (
    AuditEvent,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    Principal,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller._support import (
    CapacityApplySuperseded,
    _replacement_reason,
    logger,
)
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.errors import ConflictError, NotFoundError
from eyvan.server.service.state import ACTIVE_LEASE_STATES

#: States in which a workload has capacity requested for it and no lease yet.
#: The same pair ``server/invariants.py`` means by "not yet running", and the
#: window in which a reservation on a freshly requested cell is the only thing
#: stopping another workload from taking it.
_WAITING_FOR_CAPACITY = frozenset({WorkloadState.PENDING, WorkloadState.ADMITTED})

#: Audit actions for a capacity pool whose acquisitions keep failing.
ACQUISITION_BACKOFF_ACTION = "capacity_pool.acquisition_backoff"
ACQUISITION_BACKOFF_CLEARED_ACTION = "capacity_pool.acquisition_backoff_cleared"
#: How far back a newly elected leader looks for the last announcement.
_BACKOFF_EVENT_SCAN_LIMIT = 50


def _still_demanded(cell: CellRecord, pending_specs: tp.Sequence[WorkloadDefinition]) -> bool:
    """Return whether any pending workload could still lease this cell.

    Conservative on purpose -- draining less is always safe. An atomic
    workload demands its exact worker count; an elastic one can pack any
    cell up to its request. Pool pinning is ignored here so a cell another
    fleet's workload might want is left alone.
    """

    for spec in pending_specs:
        if spec.placement is PlacementMode.ELASTIC:
            if spec.worker_count >= cell.worker_count:
                return True
        elif spec.worker_count == cell.worker_count:
            return True
    return False


class _ControllerFleets:
    """Operator-driven fleet, pool, and cell administration.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def reconcile_if_leader(self) -> SchedulingResult | None:
        """Run a manual reconciliation only on the elected scheduler replica.

        Leadership is re-checked *before every phase*, not once at the top. A
        full cycle spans many provider round trips, so a replica can lose the
        advisory lock partway through; continuing would let two leaders mint
        autoscaler cell ids from the same ``len(managed) + 1`` sequence and
        collide. Losing leadership mid-cycle is a normal outcome, so it returns
        ``None`` rather than raising.
        """

        if not await self.leader.acquire():
            return None
        async with self._leader_cycle_lock:
            for phase in (
                self.reconcile_unmanaged_capacity_recoveries_once,
                self.reconcile_capacity_pools_once,
                self._observe_cells_once,
                self.quarantine_expired_leases_once,
                self.observe_workloads_once,
            ):
                if not await self.leader.acquire():
                    return None
                await phase()
            if not await self.leader.acquire():
                return None
            return await self.reconcile_once()

    async def disable_fleet(
        self,
        principal: Principal,
        fleet_name: str,
        *,
        idempotency_key: str,
    ) -> tuple[list[CapacityPoolRecord], bool]:
        """Persist fleet shutdown before any newly admitted capacity operation can start."""

        async with self._fleet_capacity_gate(fleet_name).closing():
            return await self.service.disable_fleet(principal, fleet_name, idempotency_key=idempotency_key)

    async def apply_fleet(
        self,
        principal: Principal,
        fleet_name: str,
        pools: tp.Sequence[CapacityPoolRecord],
        *,
        idempotency_key: str,
    ) -> tuple[list[CapacityPoolRecord], bool]:
        """Apply one fleet after closing its capacity-operation admission gate."""

        async with self._fleet_capacity_gate(fleet_name).closing():
            return await self.service.apply_fleet(principal, fleet_name, pools, idempotency_key=idempotency_key)

    async def set_fleet_init_script(
        self,
        principal: Principal,
        fleet_name: str,
        script: str,
        *,
        idempotency_key: str,
    ) -> tuple[JsonObject, bool]:
        """Change a fleet's init script after closing its capacity-operation admission gate."""

        async with self._fleet_capacity_gate(fleet_name).closing():
            return await self.service.set_fleet_init_script(
                principal, fleet_name, script, idempotency_key=idempotency_key
            )

    async def upsert_capacity_pool(
        self,
        principal: Principal,
        pool: CapacityPoolRecord,
        *,
        idempotency_key: str,
    ) -> tuple[CapacityPoolRecord, bool]:
        """Update one pool after closing its fleet's capacity-operation admission gate."""

        async with self._fleet_capacity_gate(pool.fleet_name).closing():
            return await self.service.upsert_capacity_pool(principal, pool, idempotency_key=idempotency_key)

    async def disable_capacity_pool(
        self,
        principal: Principal,
        pool_id: str,
        *,
        idempotency_key: str,
    ) -> tuple[CapacityPoolRecord, bool]:
        """Disable one pool after closing its fleet's capacity-operation admission gate."""

        async with self.repository.transaction() as transaction:
            pool = await transaction.get_capacity_pool(pool_id)
        if pool is None:
            return await self.service.disable_capacity_pool(principal, pool_id, idempotency_key=idempotency_key)
        async with self._fleet_capacity_gate(pool.fleet_name).closing():
            return await self.service.disable_capacity_pool(principal, pool_id, idempotency_key=idempotency_key)

    async def drain_cell(
        self,
        principal: Principal,
        cell_id: str,
        *,
        idempotency_key: str,
        expected_workload: WorkloadRecord | None = None,
        expected_cell_generation: int | None = None,
        expected_pool_generation: int | None = None,
        require_empty: bool = False,
    ) -> tuple[CellRecord, bool]:
        """Persist a cell drain while serializing that cell's future capacity work."""

        cell = await self.service.get_cell(principal, cell_id)
        async with self._capacity_operation(cell):
            return await self.service.drain_cell(
                principal,
                cell_id,
                idempotency_key=idempotency_key,
                expected_workload=expected_workload,
                expected_cell_generation=expected_cell_generation,
                expected_pool_generation=expected_pool_generation,
                require_empty=require_empty,
            )

    async def delete_cell(
        self,
        principal: Principal,
        cell_id: str,
        *,
        idempotency_key: str,
        expected_cell_generation: int | None = None,
        expected_pool_generation: int | None = None,
    ) -> tuple[CellRecord, bool]:
        """Persist a cell release while serializing that cell's future capacity work."""

        cell = await self.service.get_cell(principal, cell_id)
        async with self._capacity_operation(cell):
            return await self.service.delete_cell(
                principal,
                cell_id,
                idempotency_key=idempotency_key,
                expected_cell_generation=expected_cell_generation,
                expected_pool_generation=expected_pool_generation,
            )

    async def recycle_stale_build_cells_once(self) -> int:
        """Drain every schedulable cell whose agent runs a different build.

        The dispatch guard catches these one submission at a time, which left
        the recycling to whoever submitted next -- and an idle stale cell
        billed until someone did, because a cell the current build cannot use
        is still a cell the pool counts and pays for. This sweep is the same
        check without the submission: each stale cell is drained, the
        autoscaler deletes it once its leases are gone, and the replacement is
        provisioned from the wheel the plane stages now.

        Returns the number of cells drained, so the phase's effect is visible
        without reading the audit trail -- and so a fleet that recycles on
        every pass is noticeable as a number, not as churn.
        """

        from eyvan._internal.build_identity import build_identity

        head_digest = build_identity().source_digest
        if not head_digest:
            return 0  # a source tree with no identity cannot judge anyone
        async with self.repository.transaction() as transaction:
            # A cell hosting an active lease is mid-flight, not stale inventory:
            # draining it strands the workload between a dead agent and a lease
            # nothing will requeue, because the dispatch guard only judges
            # builds at dispatch time. The sweep catches the cell on a later
            # pass, once the lease releases.
            leased_ids = {
                lease.cell_id for lease in await transaction.list_leases() if lease.state in ACTIVE_LEASE_STATES
            }
            cells = [
                cell
                for cell in await transaction.list_cells()
                if cell.state is CellState.READY and cell.id not in leased_ids
            ]
        # Observation found a backing-off agent unreachable and will say when it
        # answers again; asking it here too only repeats the refusal and the log
        # line, every interval.
        cells = [cell for cell in cells if not self._agent_backoff.backing_off(cell)]
        gate = asyncio.Semaphore(max(1, self.settings.cell_reconcile_concurrency))

        async def read(cell: CellRecord) -> list[tuple[CellRecord, str]] | None:
            # Each cell on its own, with an unreadable agent skipped rather than
            # raised: the dispatch guard must refuse what it cannot prove, but a
            # sweep that aborted on one unreachable endpoint would hold the
            # *other* stale cells billing for a full interval. Unreadable is a
            # third state, not a mismatch, and the next pass asks again.
            #
            # Concurrently, bounded like observation: asked one at a time, this
            # phase cost the sum of every agent and worker round trip (8.5 s live)
            # and every phase after it -- scheduling and observation included --
            # waited.
            async with gate:
                try:
                    return await self._stale_build_report((cell,), head_digest)
                except Exception as exc:
                    logger.info("cell %s build could not be read; left for the next pass: %s", cell.id, exc)
                    return None

        reports = await asyncio.gather(*(read(cell) for cell in cells))
        drained = 0
        for cell, stale in zip(cells, reports, strict=True):
            if not stale:
                continue
            _cell, description = stale[0]
            try:
                await self.drain_cell(
                    self._principal,
                    cell.id,
                    idempotency_key=f"stale-build:{cell.id}:g{cell.generation}",
                )
            except ConflictError:
                continue  # already draining or deleting; the cause is leaving either way
            drained += 1
            logger.info(
                "cell %s runs %s but this plane stages %s; drained for replacement",
                cell.id,
                description,
                head_digest[:12],
            )
        return drained

    async def _autoscale_upsert_cell(
        self,
        pool: CapacityPoolRecord,
        cell_id: str,
        worker_count: int,
        *,
        autoscale_request_workload_id: str | None = None,
        idempotency_key: str,
    ) -> tuple[CellRecord | None, bool]:
        """Upsert one planned cell only while its pool template remains current."""

        async with self._fleet_capacity_gate(pool.fleet_name).mutation():
            async with self.repository.transaction() as transaction:
                current_pool = await transaction.get_capacity_pool(pool.id)
            if current_pool is None or not current_pool.enabled or current_pool.generation != pool.generation:
                return None, False
            return await self.service.upsert_cell(
                self._principal,
                current_pool.make_cell(
                    cell_id,
                    worker_count=worker_count,
                    autoscale_request_workload_id=autoscale_request_workload_id,
                ),
                idempotency_key=idempotency_key,
            )

    async def _autoscale_requeue_unprovisioned_cell(
        self,
        pool: CapacityPoolRecord,
        cell: CellRecord,
    ) -> tuple[CellRecord | None, bool]:
        """Restore a current-template registering cell whose apply command was superseded."""

        async with self._fleet_capacity_gate(pool.fleet_name).mutation():
            async with self.repository.transaction() as transaction:
                current_pool = await transaction.get_capacity_pool(pool.id)
                current_cell = await transaction.get_cell(cell.id)
            if (
                current_pool is None
                or not current_pool.enabled
                or current_pool.generation != pool.generation
                or current_cell is None
                or current_cell.generation != cell.generation
                or not cell_matches_pool(current_cell, current_pool)
                or current_cell.state is not CellState.REGISTERING
                or current_cell.provider_phase is not None
            ):
                return None, False
            return await self.service.requeue_unprovisioned_cell(
                self._principal,
                current_cell.id,
                expected_generation=current_cell.generation,
            )

    async def reconcile_once(self) -> SchedulingResult:
        """Schedule one durable snapshot and commit its plans."""

        started = time.perf_counter()
        result = await self.service.schedule_once(self._principal)
        if self.metrics is not None:
            self.metrics.scheduler_duration.observe(time.perf_counter() - started)
            for _ in result.placements:
                self.metrics.scheduler_decisions.labels(outcome="placed", reason="ready_capacity").inc()
            for _ in result.preemptions:
                self.metrics.scheduler_decisions.labels(outcome="preempt", reason="priority").inc()
            for failure in result.infeasibilities:
                self.metrics.scheduler_decisions.labels(outcome="pending", reason=failure.code.value).inc()
        for decision in result.placements:
            if not await self.leader.acquire():
                return result
            await self._commit_placement(decision)
        for plan in result.preemptions:
            if not await self.leader.acquire():
                return result
            await self._commit_preemption(plan)
        if result.infeasibilities:
            if not await self.leader.acquire():
                return result
            await self._record_infeasibilities(result)
        return result

    async def _release_inactive_autoscale_request_ownership(
        self,
        cells: list[CellRecord],
    ) -> list[CellRecord]:
        """Return source-attributed capacity to shared supply after its owner stops waiting.

        A fresh autoscaled cell is reserved only while the workload that caused
        it to be requested is still waiting for capacity. Lease creation clears
        the reservation atomically. This recovery path handles cancellation,
        terminal failure, and deleted legacy workloads before a lease exists.
        On federated planes it also releases waiting owners that the current
        budget or quota bounds demonstrably refuse; unknown capacity does not
        justify releasing a reservation.

        Waiting is ``PENDING`` *or* ``ADMITTED``, matching what
        ``server/invariants.py`` already means by "not yet running". Both are
        states in which a workload has capacity requested for it and no lease,
        so releasing an admitted one would hand its slice to somebody else while
        it is still queued for it. Every path today admits and leases in one
        transaction, so this changes nothing that currently happens -- it is the
        precondition for requesting several zones at once, where waiting with
        capacity already granted is the normal case rather than a race window.
        """

        candidates = tuple(cell for cell in cells if cell.autoscale_request_workload_id is not None)
        if not candidates:
            return cells

        released_by_id: dict[str, CellRecord] = {}
        async with self.repository.transaction() as transaction:
            refused_owners: set[str] = set()
            leased_ids: set[str] = set()
            if self.service.plane_id and self.service.scheduler is not None:
                grants = await transaction.list_budget_grants()
                quota_grants = await transaction.list_quota_grants()
                if grants or quota_grants:
                    # Waiting does not license capacity that current federation
                    # bounds refuse. Rejudge against current records in this
                    # transaction before returning its reservation to shared supply.
                    current_workloads = await transaction.list_workloads()
                    current_cells = await transaction.list_cells()
                    current_leases = await transaction.list_leases()
                    leased_ids = {lease.cell_id for lease in current_leases if lease.state in ACTIVE_LEASE_STATES}
                    simulation = CapacitySimulation.build(
                        tuple(await transaction.list_capacity_pools()),
                        tuple(current_cells),
                        tuple(current_workloads),
                        tuple(current_leases),
                        utcnow(),
                        policies=tuple(await transaction.list_policies()),
                    )
                    result = self.service.scheduler.schedule(
                        current_workloads,
                        simulation.cells,
                        leases=current_leases,
                        budgets=await transaction.list_budgets(),
                        quotas=await transaction.list_quotas(),
                        policies=await transaction.list_policies(),
                        grants=grants,
                        quota_grants=quota_grants,
                        plane_id=self.service.plane_id,
                    )
                    refused_owners = {
                        failure.workload_id
                        for failure in result.infeasibilities
                        if failure.code in {InfeasibilityCode.BUDGET_EXCEEDED, InfeasibilityCode.QUOTA_EXCEEDED}
                    }
            for snapshot in candidates:
                current = await transaction.get_cell(snapshot.id)
                if (
                    current is None
                    or current.generation != snapshot.generation
                    or current.autoscale_request_workload_id != snapshot.autoscale_request_workload_id
                ):
                    continue
                owner_id = current.autoscale_request_workload_id
                assert owner_id is not None
                owner = await transaction.get_workload(owner_id)
                if owner is not None and owner.state in _WAITING_FOR_CAPACITY:
                    if owner_id not in refused_owners or current.id in leased_ids:
                        continue
                released = current.model_copy(
                    update={
                        "autoscale_request_workload_id": None,
                        "updated_at": utcnow(),
                    }
                )
                await transaction.put_cell(released)
                released_by_id[released.id] = released
                logger.info(
                    "releasing autoscale reservation on cell %s because workload %s is no longer waiting or admissible",
                    released.id,
                    owner_id,
                )
        return [released_by_id.get(cell.id, cell) for cell in cells]

    def _cluster_capacity(self) -> ClusterCapacityPass:
        """Return the pool cluster-capacity pass, built from the environment once.

        Lazy rather than constructed with the controller so that enabling an
        observed cluster as a plan input stays one file. A pass assigned before
        the first reconciliation -- which is what a test does, and what an
        embedder with its own Kubernetes credentials would do -- is used
        unchanged.
        """

        existing: ClusterCapacityPass | None = getattr(self, "_cluster_capacity_pass", None)
        if existing is None:
            existing = ClusterCapacityPass.from_env()
            self._cluster_capacity_pass = existing
        return existing

    async def reconcile_capacity_pools_once(self) -> dict[str, int]:
        """Reconcile declarative pools against global submitted workload demand."""

        if self.service.scheduler is None:
            return {"launched": 0, "drained": 0, "deleted": 0}
        async with self.repository.transaction() as transaction:
            pools = await transaction.list_capacity_pools()
            if not pools:
                return {"launched": 0, "drained": 0, "deleted": 0}
            workloads = await transaction.list_workloads()
            cells = await transaction.list_cells()
            leases = await transaction.list_leases()
            budgets = await transaction.list_budgets()
            quotas = await transaction.list_quotas()
            policies = await transaction.list_policies()
            grants = await transaction.list_budget_grants() if self.service.plane_id else []
            quota_grants = await transaction.list_quota_grants() if self.service.plane_id else []

        cells = await self._release_inactive_autoscale_request_ownership(cells)
        pools_by_id = {pool.id: pool for pool in pools}
        now = utcnow()
        refreshed_cells: list[CellRecord] = []
        for cell in cells:
            pool = pools_by_id.get(cell.pool_id or "")
            # The state tests come first on purpose: both template checks mint a
            # whole CellRecord, and tombstones -- kept forever as id evidence --
            # are nearly every cell a long-lived plane holds. Every term is a pure
            # test, so the order changes the cost, never the answer.
            if (
                pool is not None
                and pool.enabled
                and cell.state not in {CellState.DRAINING, CellState.DELETING, CellState.DELETED}
                and not cell_requires_replacement(cell, pool)
                and not cell_matches_pool(cell, pool)
            ):
                try:
                    updated, _ = await self._autoscale_upsert_cell(
                        pool,
                        cell.id,
                        cell.worker_count,
                        autoscale_request_workload_id=cell.autoscale_request_workload_id,
                        idempotency_key=f"autoscale:update:{cell.id}:g{pool.generation}",
                    )
                    if updated is not None:
                        cell = updated
                except ConflictError as exc:
                    logger.info("cell %s metadata update is waiting for active leases: %s", cell.id, exc)
            if (
                pool is not None
                and pool.enabled
                and cell.state is CellState.REGISTERING
                and cell.provider_phase is None
                and cell_matches_pool(cell, pool)
            ):
                try:
                    repaired, _ = await self._autoscale_requeue_unprovisioned_cell(pool, cell)
                    if repaired is not None:
                        cell = repaired
                except ConflictError as exc:
                    logger.info("cell %s provisioning recovery became stale: %s", cell.id, exc)
            refreshed_cells.append(cell)
        cells = refreshed_cells
        cells_by_pool: dict[str, list[CellRecord]] = {pool.id: [] for pool in pools}
        for cell in cells:
            pool_id = cell.pool_id
            if pool_id is not None and pool_id in cells_by_pool:
                cells_by_pool[pool_id].append(cell)

        pool_records = tuple(pools)
        simulation = CapacitySimulation.build(
            pool_records, tuple(cells), tuple(workloads), tuple(leases), now, policies=tuple(policies)
        )
        simulated = self.service.scheduler.schedule(
            workloads,
            simulation.cells,
            leases=leases,
            budgets=budgets,
            quotas=quotas,
            policies=policies,
            grants=grants,
            quota_grants=quota_grants,
            plane_id=self.service.plane_id,
        )
        demand_by_pool = simulation.demand(
            pool_records,
            tuple(leases),
            simulated,
            tuple(workloads),
            now,
            race_width=max(int(self.settings.capacity_race_width), 1),
        )
        planner = CapacityPlanner(pool_records, cells_by_pool, now)
        plans = planner.plan(demand_by_pool)
        await self._persist_shrink_clocks(pool_records, plans, now)
        await self._record_autoscale_quota_blocks(tuple(workloads), planner.quota_blocked_workload_ids)
        await self._record_acquisition_exhausted(tuple(workloads), simulation.acquisition_exhausted_workload_ids)
        backoffs = planner.acquisition_backoffs
        await self._announce_acquisition_backoffs(pool_records, backoffs, plans, now)
        await self._record_acquisition_backoff_reasons(tuple(workloads), backoffs, now)

        # Kueue's own view of the cluster, read once per Kubernetes pool. Taken
        # here on purpose: every transaction above is closed and the launch loop
        # below has not opened one, so a slow API server delays a pass instead of
        # holding a lease. Off unless an operator enabled it, in which case this
        # returns an empty mapping and issues no request at all.
        cluster_capacity = await asyncio.to_thread(
            self._cluster_capacity().advise_launches,
            [(pool, plans[pool.id].launch_worker_counts) for pool in pool_records],
        )

        launched = 0
        drained = 0
        deleted = 0
        active_lease_cells = {lease.cell_id for lease in leases if lease.state in ACTIVE_LEASE_STATES}
        cell_ids = {cell.id for cell in cells}

        for pool in sorted(pools, key=lambda item: item.id):
            plan = plans[pool.id]
            live = list(plan.live)
            matching_supply = list(plan.matching_supply)
            target = plan.target_worker_units
            # Monotonic, from every cell this pool has ever had -- including
            # deleted ones, whose records are the evidence their index is spent.
            # `len(managed) + 1` walked backwards on deletion and handed the same
            # id out twice; the second holder then inherited the first's provider
            # resource under that name.
            next_index = next_pool_cell_index(pool, cells)
            advice = cluster_capacity.get(pool.id)
            # An unknown cluster keeps the planner's own number. Only a fresh,
            # exact reading that proved the cluster admits fewer cells than were
            # planned withholds any, and it withholds from the tail so the
            # workload-bearing launches the planner ordered first survive.
            allowed = len(plan.launch_worker_counts) if advice is None else advice.allowed_cells
            if advice is not None and advice.withheld_cells:
                logger.info(
                    "pool %s is holding back %s of %s planned cell(s): %s",
                    pool.id,
                    advice.withheld_cells,
                    advice.requested_cells,
                    advice.reason,
                )
            for worker_count, autoscale_request_workload_id in zip(
                plan.launch_worker_counts[:allowed],
                plan.launch_workload_ids[:allowed],
                strict=True,
            ):
                # `pool_cell_id` carries a per-pool-record epoch. Without it the
                # counter restarts at 0001 on a control plane with an empty
                # database and collides with the agent's durable cell tombstones,
                # which fence every apply forever.
                cell_id = pool_cell_id(pool, next_index)
                while cell_id in cell_ids:
                    next_index += 1
                    cell_id = pool_cell_id(pool, next_index)
                created, replayed = await self._autoscale_upsert_cell(
                    pool,
                    cell_id,
                    worker_count,
                    autoscale_request_workload_id=autoscale_request_workload_id,
                    idempotency_key=f"autoscale:create:{cell_id}:g{pool.generation}",
                )
                if created is None:
                    continue
                if not replayed:
                    self._capacity_lifecycle.event(
                        CapacityLifecycleEvent.PLANNED,
                        created,
                        reason=f"autoscaler launch for {worker_count} worker(s), target {target}",
                    )
                cell_ids.add(cell_id)
                next_index += 1
                launched += 1

            # The resize leftover: an atomic launch of W workers can never use a
            # smaller idle cell in this pool, so one that nothing else demands is
            # known-unusable now -- draining it now, rather than at the idle
            # timeout, is the difference between paying for the old shape for the
            # whole provisioning window or not. Warm cells a pending workload
            # could still lease (exact shape, or any size an elastic one can
            # pack), leased cells, and in-flight acquisitions are untouched.
            workload_by_id = {workload.id: workload for workload in workloads}
            pending_specs = [workload.spec for workload in workloads if workload.state is WorkloadState.PENDING]
            for worker_count, owner_id in zip(
                plan.launch_worker_counts[:allowed],
                plan.launch_workload_ids[:allowed],
                strict=True,
            ):
                owner = workload_by_id.get(owner_id)
                if owner is None or owner.spec.placement is PlacementMode.ELASTIC:
                    continue
                for cell in matching_supply:
                    if (
                        cell.state is CellState.READY
                        and cell.worker_count < worker_count
                        and cell.id not in active_lease_cells
                        and cell.autoscale_request_workload_id is None
                        and cell.id not in plan.protected_cell_ids
                        and not _still_demanded(cell, pending_specs)
                    ):
                        try:
                            await self.drain_cell(
                                self._principal,
                                cell.id,
                                idempotency_key=f"superseded-shape:{cell.id}:g{cell.generation}",
                            )
                        except ConflictError:
                            continue  # already draining or deleting; either way it is leaving
                        drained += 1
                        logger.info(
                            "cell %s (%d workers) drained: superseded by an atomic %d-worker launch in pool %s",
                            cell.id,
                            cell.worker_count,
                            worker_count,
                            pool.id,
                        )

            if pool.enabled and pool.initialized_at is None:
                async with self.repository.transaction() as transaction:
                    current = await transaction.get_capacity_pool(pool.id)
                    if (
                        current is not None
                        and current.enabled
                        and current.generation == pool.generation
                        and current.initialized_at is None
                    ):
                        # Advice can withhold launches and a stale upsert can be
                        # skipped. Only durable, current-template supply fulfills
                        # the initial target, never the planner's hypothetical batch.
                        supply = sum(
                            cell.worker_count
                            for cell in await transaction.list_cells()
                            if cell.pool_id == current.id and cell_provides_pool_supply(cell, current, now)
                        )
                        if supply >= current.initial_target:
                            await transaction.put_capacity_pool(current.model_copy(update={"initialized_at": now}))

            draining = [cell for cell in live if cell.state is CellState.DRAINING]
            for cell in draining:
                if cell.id in active_lease_cells:
                    continue
                try:
                    await self.delete_cell(
                        self._principal,
                        cell.id,
                        idempotency_key=f"autoscale:delete:{cell.id}:g{cell.generation + 1}",
                        expected_cell_generation=cell.generation,
                        expected_pool_generation=pool.generation,
                    )
                except ConflictError:
                    continue
                deleted += 1

            current_candidates = [
                cell
                for cell in matching_supply
                if cell.state not in {CellState.DRAINING, CellState.DELETING} and cell.id not in active_lease_cells
            ]
            protected = set(plan.protected_cell_ids)

            updates: list[tuple[str, int, datetime | None]] = []
            stale_candidates = [
                cell
                for cell in live
                if cell not in matching_supply
                and cell.state not in {CellState.DRAINING, CellState.DELETING}
                and cell.id not in active_lease_cells
            ]
            ready_drain_budget = max(
                sum(cell.worker_count for cell in live if cell.state is CellState.READY) - target,
                0,
            )
            to_drain: list[CellRecord] = []
            replacement_reasons: dict[str, CapacityReplacementReason] = {}
            for cell in stale_candidates:
                if pool.enabled and cell.state is CellState.READY:
                    if cell.id in plan.forced_drain_cell_ids:
                        pass
                    elif ready_drain_budget > 0:
                        ready_drain_budget = max(ready_drain_budget - cell.worker_count, 0)
                    else:
                        continue
                to_drain.append(cell)
                replacement_reasons[cell.id] = _replacement_reason(cell, pool)
            idle_timeout_s = pool.idle_timeout_s if pool.enabled else 0
            for cell in current_candidates:
                if cell.id in plan.forced_drain_cell_ids:
                    to_drain.append(cell)
                    replacement_reasons[cell.id] = CapacityReplacementReason.SHARED_QUOTA_REBALANCE
                    continue
                if cell.id in protected:
                    # On a held shrink pass the shape is protected by the
                    # hysteresis alone, and clearing idle clocks here would
                    # erase the very clocks the post-window fall drains
                    # against -- with the hold re-arming on every pass, no
                    # clock ever survived long enough to time out. Cells
                    # protected because demand wants them still get the reset.
                    if not plan.shrink_held and cell.idle_since is not None:
                        updates.append((cell.id, cell.generation, None))
                    continue
                idle_since = cell.idle_since or now
                if cell.idle_since is None:
                    updates.append((cell.id, cell.generation, idle_since))
                if (now - idle_since).total_seconds() >= idle_timeout_s:
                    to_drain.append(cell)
            if updates:
                async with self.repository.transaction() as transaction:
                    for cell_id, generation, idle_since in updates:
                        current = await transaction.get_cell(cell_id)
                        if current is None or current.generation != generation:
                            continue
                        # Only the two fields this loop decided, applied to the
                        # record as it is *now*. Writing back the whole snapshot
                        # read before the transaction discarded anything that had
                        # changed since without bumping the generation -- which is
                        # exactly what a concurrent `capacity.apply` result does
                        # when it moves a cell REGISTERING -> READY. The capacity
                        # arrived, and the idle sweep put it back to registering,
                        # so a fleet could hold provisioned cells it never used.
                        await transaction.put_cell(
                            current.model_copy(update={"idle_since": idle_since, "updated_at": now})
                        )
            for cell in {candidate.id: candidate for candidate in to_drain}.values():
                try:
                    draining_cell, replayed = await self.drain_cell(
                        self._principal,
                        cell.id,
                        idempotency_key=f"autoscale:drain:{cell.id}:g{cell.generation}",
                        expected_cell_generation=cell.generation,
                        expected_pool_generation=pool.generation,
                        require_empty=True,
                    )
                except ConflictError:
                    continue
                # "Never delivered capacity" is the question, and `state` answers a
                # different one -- what the cell is *now*. A reclaimed spot cell is
                # draining or NEEDS_RECREATE by the time it reaches here, so a cell
                # that delivered and was later preempted must not be scored as a
                # STOCKOUT against a zone that supplied capacity.
                #
                # `provision_latencies_s` is appended only on the SUCCESS
                # transition into READY, so a non-empty list is the durable record
                # that this cell once delivered.
                never_delivered = not cell.availability.provision_latencies_s
                if not replayed and cell.state is not CellState.READY and never_delivered:
                    # This cell is being reclaimed without ever having delivered
                    # capacity -- it exceeded queue_timeout_s or provisioning_timeout_s
                    # and stopped counting as supply. Nothing else records that: the
                    # observation paths key off the provider's reported state, and a
                    # cell parked in WAITING keeps reporting a *pending* status, which
                    # leaves its belief untouched. Without this the autoscaler never
                    # learns which zone is empty and keeps choosing it first forever.
                    # `replayed` makes it once per (cell, generation).
                    await self._record_capacity_stockout(cell)
                replacement_reason = replacement_reasons.get(cell.id)
                if replacement_reason is not None and not replayed:
                    self._capacity_lifecycle.event(
                        CapacityLifecycleEvent.REPLACEMENT,
                        draining_cell,
                        previous=cell,
                        reason=replacement_reason.value,
                    )
                drained += 1

        return {"launched": launched, "drained": drained, "deleted": deleted}

    async def _record_acquisition_backoff_reasons(
        self,
        workloads: tuple[WorkloadRecord, ...],
        backoffs: dict[str, PoolAcquisitionBackoff],
        now: datetime,
    ) -> None:
        """Name a repeated acquisition failure on each pending workload it is holding up.

        Without this a workload whose pool kept failing read "requesting one
        elastic cell" -- the plane was asking, but every answer had been no and
        nothing said so. Written only when the sentence changes, and only where
        no stronger reason (quota, exhaustion, a queue denial) already owns the
        field; cleared again, exactly, once the streak stops describing the pool.
        """

        updates: list[tuple[str, str | None]] = []
        for workload in workloads:
            if workload.state is not WorkloadState.PENDING:
                continue
            streak = workload_acquisition_backoff(workload, backoffs, now)
            current = workload.scheduling_reason
            if streak is not None and workload.autoscale_block_reason is None:
                reason = acquisition_backoff_reason(streak, now)
                if reason != current:
                    updates.append((workload.id, reason))
            elif is_acquisition_backoff_reason(current):
                updates.append((workload.id, None))
        if not updates:
            return
        async with self.repository.transaction() as transaction:
            for workload_id, reason in updates:
                current_workload = await transaction.get_workload(workload_id)
                if current_workload is None or current_workload.state is not WorkloadState.PENDING:
                    continue
                if reason is None and not is_acquisition_backoff_reason(current_workload.scheduling_reason):
                    continue
                if reason is not None and current_workload.autoscale_block_reason is not None:
                    continue
                await transaction.put_workload(
                    current_workload.model_copy(update={"scheduling_reason": reason, "updated_at": utcnow()})
                )

    async def _announced_acquisition_backoff(self, pool_id: str) -> str | None:
        """Return the streak last announced for ``pool_id``, from memory or the audit ledger.

        The ledger is the durable answer, so a restarted or newly elected leader
        neither repeats an announcement nor forgets to say a streak ended.
        """

        cache = self._acquisition_backoff_announcements
        if pool_id in cache:
            return cache[pool_id]
        async with self.repository.transaction() as transaction:
            events = await transaction.list_events(
                entity_type="capacity_pool",
                entity_id=pool_id,
                newest_first=True,
                limit=_BACKOFF_EVENT_SCAN_LIMIT,
            )
        announced: str | None = None
        for event in events:
            if event.action == ACQUISITION_BACKOFF_ACTION:
                streak = event.payload.get("streak")
                announced = streak if isinstance(streak, str) else None
                break
            if event.action == ACQUISITION_BACKOFF_CLEARED_ACTION:
                break
        cache[pool_id] = announced
        return announced

    async def _announce_acquisition_backoffs(
        self,
        pools: tuple[CapacityPoolRecord, ...],
        backoffs: dict[str, PoolAcquisitionBackoff],
        plans: dict[str, PoolScalePlan],
        now: datetime,
    ) -> None:
        """Say once per failed incarnation that a pool is backing off, and once when it stops.

        A pool that keeps failing used to be visible only as an ``unhealthy``
        lifecycle line per cell, indistinguishable from one bad machine. This
        names the pool, the run length, the last error and when it will be
        asked again -- as an audit event, so it survives log rotation, and as a
        warning. Bounded by construction: one announcement per failed cell, and
        failed cells are what the backoff spaces out.
        """

        for pool in pools:
            backoff = backoffs.get(pool.id)
            repeated = backoff is not None and backoff.failed_attempts >= 2 and backoff.recent(now)
            streak = f"{backoff.last_failed_cell_id}:{backoff.failed_attempts}" if repeated and backoff else None
            announced = await self._announced_acquisition_backoff(pool.id)
            if streak == announced:
                continue
            if streak is not None and backoff is not None:
                withheld = len(plans[pool.id].withheld_worker_counts) if pool.id in plans else 0
                payload: JsonObject = {
                    "streak": streak,
                    "failed_attempts": backoff.failed_attempts,
                    "backoff_s": backoff.backoff_s,
                    "next_attempt_at": backoff.next_attempt_at.isoformat(),
                    "last_failure_at": backoff.last_failure_at.isoformat(),
                    "last_failed_cell_id": backoff.last_failed_cell_id,
                    "failure_code": None if backoff.failure_code is None else backoff.failure_code.value,
                    "last_error": backoff.last_error,
                    "withheld_launches": withheld,
                }
                action = ACQUISITION_BACKOFF_ACTION
                logger.warning(
                    "capacity pool %s (fleet %s) failed %d capacity request(s) in a row; "
                    "next attempt after %s (backoff %.0fs): %s",
                    pool.id,
                    pool.fleet_name,
                    backoff.failed_attempts,
                    backoff.next_attempt_at.isoformat(),
                    backoff.backoff_s,
                    backoff.last_error or "no diagnostic recorded",
                )
            else:
                payload = {"previous_streak": announced}
                action = ACQUISITION_BACKOFF_CLEARED_ACTION
                logger.info("capacity pool %s (fleet %s) is no longer backing off", pool.id, pool.fleet_name)
            async with self.repository.transaction() as transaction:
                await transaction.append_event(
                    AuditEvent(
                        id=f"evt_{uuid.uuid4().hex}",
                        actor=self._principal.subject,
                        entity_type="capacity_pool",
                        entity_id=pool.id,
                        action=action,
                        payload=payload,
                    )
                )
            self._acquisition_backoff_announcements[pool.id] = streak

    async def _persist_shrink_clocks(
        self,
        pools: tuple[CapacityPoolRecord, ...],
        plans: dict[str, PoolScalePlan],
        now: datetime,
    ) -> None:
        """Persist what the planner decided about each pool's shrink clock.

        The planner reads ``demand_below_supply_since`` and decides; this is the
        write side. A pool whose record moved underneath the pass (an operator
        edit bumped its generation) is skipped -- the next pass re-judges from
        fresh state rather than stamping a stale view.
        """

        for pool in pools:
            plan = plans[pool.id]
            if not plan.shrink_since_changed:
                continue
            async with self.repository.transaction() as transaction:
                current = await transaction.get_capacity_pool(pool.id)
                if current is None or current.generation != pool.generation:
                    continue
                await transaction.put_capacity_pool(
                    current.model_copy(update={"demand_below_supply_since": plan.shrink_since_update})
                )

    async def reconcile_unmanaged_capacity_recoveries_once(self) -> int:
        """Resume manual cells whose agent fence conflict survived a controller interruption."""

        async with self.repository.transaction() as transaction:
            candidates = [
                cell
                for cell in await transaction.list_cells()
                if cell.pool_id is None
                and cell.state is CellState.UNAVAILABLE
                and cell.provider_phase is CapacityStatus.NEEDS_RECREATE
            ]
        recovered = 0
        for cell in candidates:
            try:
                recreated, replayed = await self.service.recreate_cell(
                    self._principal,
                    cell.id,
                    expected_generation=cell.generation,
                    idempotency_key=f"agent-fence-recreate:{cell.id}:g{cell.generation}",
                )
            except (ConflictError, NotFoundError) as exc:
                logger.debug("manual cell %s cannot yet recover from an agent fence conflict: %s", cell.id, exc)
                continue
            if not replayed:
                self._capacity_lifecycle.transition(
                    cell,
                    recreated,
                    reason="resuming manual cell after an agent fence conflict",
                )
                recovered += 1
        return recovered

    async def _require_current_pool_template(
        self,
        transaction: RepositorySession,
        cell: CellRecord,
    ) -> None:
        """Reject capacity work whose managed pool was disabled, removed, or replaced."""

        if cell.pool_id is None:
            return
        pool = await transaction.get_capacity_pool(cell.pool_id)
        if pool is None:
            raise CapacityApplySuperseded(f"cell {cell.id!r} belongs to a removed capacity pool")
        if not pool.enabled:
            raise CapacityApplySuperseded(f"cell {cell.id!r} belongs to disabled fleet {pool.fleet_name!r}")
        if not cell_matches_pool(cell, pool):
            raise CapacityApplySuperseded(f"cell {cell.id!r} no longer matches pool {pool.id!r}'s active template")

    async def capacity_orphans(self) -> dict[str, object]:
        """Return cells an agent still holds that no control-plane record claims.

        A fleet-state document outlives the control plane that wrote it. When a
        control plane's database is replaced, its cells are left behind in the
        agent -- and any left ``desired_state: up`` are re-ensured forever, so
        the agent keeps provisioning capacity nobody asked for, invisible to the
        control plane that would otherwise own it.

        Read-only, and deliberately not automatic. One agent can legitimately be
        shared by more than one control plane, so a record this control plane
        does not recognise is not necessarily unowned -- reclaiming it is an
        operator decision, and this is the information that decision needs.
        """

        async with self.repository.transaction() as transaction:
            cells = await transaction.list_cells()
        known = {cell.id for cell in cells}
        endpoints: dict[str, CellRecord] = {}
        for cell in cells:
            if cell.endpoint and cell.state is not CellState.DELETED:
                endpoints.setdefault(cell.endpoint, cell)
        orphans: list[dict[str, str | int | None]] = []
        unreachable: list[dict[str, str]] = []
        for endpoint, sample in sorted(endpoints.items()):
            try:
                inventory = await asyncio.to_thread(self._agent_factory(sample).inventory)
            except Exception as exc:
                logger.warning("cell agent %s inventory is unavailable: %s", endpoint, exc)
                unreachable.append({"endpoint": endpoint, "detail": str(exc)})
                continue
            for entry in inventory:
                name = entry.get("name")
                if isinstance(name, str) and name not in known:
                    orphans.append({**entry, "endpoint": endpoint})
        # An agent that could not be reached has reported nothing, which is not
        # the same as reporting nothing to report. Returning a bare list made the
        # two identical, so `eyvan capacity orphans` printed "no orphaned cell
        # records" after contacting no agent at all -- including in the case this
        # command exists for, where the control plane's database was replaced and
        # therefore knows no cells, no endpoints, and scans nobody.
        return {
            "orphans": orphans,
            "scanned_endpoints": sorted(endpoints),
            "unreachable": unreachable,
        }

    async def fleet_builds(self) -> dict[str, object]:
        """Report which build each agent runs, beside this plane's own.

        Every component of this package reports one version string, so that
        string cannot answer the question an upgrade actually raises. A proof has
        already been run against a stale agent and read as evidence, and nothing
        in the plane could have shown that at the time.

        Read-only and operator-triggered, like the orphan scan it is modelled on:
        one agent-build request per distinct endpoint and a worker-build request
        per live cell, each failure recorded rather than failing the sweep.
        Three outcomes per
        agent and they stay distinct -- a digest, ``null`` from an agent that
        predates the route, or an entry under ``unreachable``. Collapsing the
        last two would report a plane nobody could reach as one in agreement.

        Only endpoints that hold a cell are scanned, because that is where the
        agent identity is discoverable from a durable record. A pool with no cell
        is named in ``unscanned_pools`` rather than passed over in silence.
        """

        async with self.repository.transaction() as transaction:
            cells = await transaction.list_cells()
            pools = await transaction.list_capacity_pools()
        endpoints: dict[str, list[CellRecord]] = {}
        for cell in cells:
            if cell.endpoint and cell.state is not CellState.DELETED:
                endpoints.setdefault(cell.endpoint, []).append(cell)
        scanned_pools = {cell.pool_id for cell in cells if cell.endpoint and cell.state is not CellState.DELETED}

        agents: list[dict[str, object]] = []
        unreachable: list[dict[str, str]] = []
        for endpoint, endpoint_cells in sorted(endpoints.items()):
            sample = endpoint_cells[0]
            try:
                reported = await asyncio.to_thread(self._agent_factory(sample).build)
            except Exception as exc:
                logger.warning("cell agent %s could not report its build: %s", endpoint, exc)
                unreachable.append({"endpoint": endpoint, "detail": str(exc)})
                continue
            # Agent identity belongs to the endpoint; worker identity belongs to
            # a cell. One shared agent can own cells running different wheels.
            for cell in sorted(endpoint_cells, key=lambda item: item.id):
                try:
                    workers = await asyncio.to_thread(self._agent_factory(cell).worker_builds, cell.id)
                except Exception as exc:
                    # The agent answered about itself, so this is not an unreachable
                    # agent. Recorded against the cell instead, because "its workers
                    # could not be asked" is a smaller fact than "the agent is gone".
                    logger.info("cell %s could not report its worker builds: %s", cell.id, exc)
                    workers = None
                agents.append(
                    {
                        "endpoint": endpoint,
                        "cell_id": cell.id,
                        "pool_id": cell.pool_id,
                        "build": None if reported is None else reported.model_dump(mode="json"),
                        "workers": workers,
                    }
                )

        plane = build_identity()
        # An unreadable known endpoint is an unknown comparison, not missing
        # inventory. Proven skew still takes precedence over these unknowns.
        matches: list[bool | None] = [None for _ in unreachable]
        for entry in agents:
            matches.append(plane.matches(entry["build"] or {}))
            workers = entry["workers"]
            if workers is None:
                # The agent could not be asked about its workers at all.
                matches.append(None)
                continue
            # An empty list is not an unknown: this cell's backend runs the
            # workload in the agent process, whose build is the line above.
            for worker in workers if isinstance(workers, list) else ():
                # An entry that is not a document is an agent this plane cannot
                # read, which is an unknown rank rather than an exception out of
                # an operator's read-only command.
                matches.append(plane.matches(worker.get("build") or {}) if isinstance(worker, dict) else None)
        return {
            "plane": plane.as_dict(),
            "agents": agents,
            "unreachable": unreachable,
            "unscanned_pools": sorted(pool.id for pool in pools if pool.id not in scanned_pools),
            # Three answers, for the same reason every component has three. False
            # only when something provably runs other code; None while any of
            # them could not say, because "nothing disagreed" is not agreement.
            # Workers count here as much as agents: they execute the workload,
            # and a cell outlives the head that staged its wheel.
            "in_agreement": (
                False if any(match is False for match in matches) else None if not matches or None in matches else True
            ),
        }
