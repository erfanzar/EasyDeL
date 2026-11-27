# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import statistics
from datetime import UTC

from eyvan.server.service._control_plane.shared import (
    ACTIVE_LEASE_STATES,
    BILLABLE_CELL_STATES,
    TERMINAL_WORKLOAD_STATES,
    CarbonReport,
    CarbonUnavailable,
    CellBillingAccrual,
    CellBillingReport,
    CellRecord,
    CellState,
    LaunchLatencyReport,
    Permission,
    PoolLaunchLatency,
    PoolWarmth,
    Principal,
    ProjectUsageReport,
    RegionCarbon,
    SubjectUsageRecord,
    WarmthReport,
    cell_provides_pool_supply,
    costing,
    datetime,
    fresh_reading,
    p95_latency,
    utcnow,
)


class ReportingMixin:
    """Transactional reporting operations."""

    async def accrue_cell_billing(self, *, now: datetime | None = None) -> CellBillingAccrual:
        """Advance every billable cell's clock and accrue its idle time.

        Eyvan has always charged *leases*: a workload runs, its lease settles, the
        project's budget moves. A provisioned cell holding no lease costs exactly
        as much and appeared in no ledger at all, so a fleet could burn a TPU pod
        for a week between jobs while every budget reported compliance.

        Two properties make this safe to run from an at-least-once leader loop:

        The clock charges ``now - billing_ticked_at``, not one fixed interval, so
        a cycle that runs late, twice, or after a failover charges the elapsed
        time once rather than the number of times the loop happened to fire.

        Only a cell the provider is actually charging for accrues. A queued TPU
        request is a record Eyvan holds and a bill nobody is paying; billing
        starts when the resource exists and stops when it is released.
        """

        moment = now or utcnow()
        accrued_seconds = 0.0
        idle_seconds = 0.0
        accrued_cost = 0.0
        idle_cost = 0.0
        counted = 0
        async with self.repository.transaction() as tx:
            leased_cells = {lease.cell_id for lease in await tx.list_leases() if lease.state in ACTIVE_LEASE_STATES}
            for cell in await tx.list_cells():
                billable = cell.state in BILLABLE_CELL_STATES
                if not billable:
                    if cell.billing_started_at is None:
                        continue
                    # The provider resource is gone. Close the clock rather than
                    # leaving a started_at that a later re-READY would charge from,
                    # which would bill for the whole gap the cell did not exist.
                    await tx.put_cell(cell.model_copy(update={"billing_started_at": None, "billing_ticked_at": None}))
                    continue
                if cell.billing_started_at is None:
                    # First tick for this incarnation: start the clock, charge
                    # nothing. Charging from created_at would bill for the queue
                    # wait, which is the one interval that is genuinely free.
                    await tx.put_cell(
                        cell.model_copy(update={"billing_started_at": moment, "billing_ticked_at": moment})
                    )
                    continue
                elapsed = max(0.0, (moment - (cell.billing_ticked_at or cell.billing_started_at)).total_seconds())
                if elapsed == 0.0:
                    continue
                idle = cell.id not in leased_cells
                hourly = cell.hourly_cost()
                cost = costing.cost_for_hours(hourly, elapsed / 3_600)
                await tx.put_cell(
                    cell.model_copy(
                        update={
                            "billing_ticked_at": moment,
                            "billed_seconds": costing.total(cell.billed_seconds, elapsed),
                            "idle_seconds": costing.total(cell.idle_seconds, elapsed if idle else 0.0),
                        }
                    )
                )
                counted += 1
                accrued_seconds += elapsed
                accrued_cost = costing.total(accrued_cost, cost)
                if idle:
                    idle_seconds += elapsed
                    idle_cost = costing.total(idle_cost, cost)
        return CellBillingAccrual(
            cells=counted,
            accrued_seconds=accrued_seconds,
            idle_seconds=idle_seconds,
            accrued_cost=accrued_cost,
            idle_cost=idle_cost,
        )

    async def carbon_report(self, principal: Principal, *, now: datetime | None = None) -> CarbonReport:
        """Report grid intensity for the regions the fleet actually occupies.

        Only regions with a pool, because a global table of grid intensities is a
        dataset rather than a report -- and one whose accuracy nobody here could
        vouch for.

        Every entry today carries ``no_provider``: none ships. Reporting the
        absence is the deliverable. A carbon feature whose unconfigured state is
        an empty response is one an operator cannot distinguish from a broken
        one, and a carbon feature whose unconfigured state is *zero* is one that
        claims every grid is clean.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            pools = await tx.list_capacity_pools()
        regions: list[RegionCarbon] = []
        for region in sorted({pool.region for pool in pools if pool.region}):
            reading = fresh_reading(self.carbon.read(region), now=moment)
            if isinstance(reading, CarbonUnavailable):
                regions.append(RegionCarbon(region=region, reason=reading.reason.value, detail=reading.detail))
                continue
            regions.append(
                RegionCarbon(
                    region=region,
                    intensity=reading.grams_co2e_per_kwh,
                    source=reading.source,
                    observed_at=reading.observed_at,
                )
            )
        return CarbonReport(observed_at=moment, regions=tuple(regions))

    async def project_usage(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        since: datetime | None = None,
    ) -> ProjectUsageReport:
        """Report what each person in a project used, and what it cost.

        Cost was settled per lease and reported per pool, so a shared project
        could see its bill and never its breakdown -- and "who is using the
        v5p" is the first question anybody sharing a fleet asks.

        Nothing new is measured. A lease already carries its settled cost and
        accelerator hours; the workload it belongs to already carries who
        submitted it. This is the join.

        Read permission on the project, deliberately, and not MANAGE_PROJECT:
        this answers how a shared budget is being spent, and the people sharing
        it are exactly who should be able to see that without being made
        administrators first.

        A timezone-less ``since`` cutoff is interpreted as UTC, consistent with
        the stored workload timestamps. Offset-aware cutoffs retain their instant.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        if since is not None and since.utcoffset() is None:
            since = since.replace(tzinfo=UTC)
        async with self.repository.transaction() as tx:
            workloads = await tx.list_workloads(org_id=org_id, project_id=project_id)
            leases = await tx.list_leases(org_id=org_id, project_id=project_id)

        submitted_by = {workload.id: workload for workload in workloads}
        totals: dict[str, dict[str, float]] = {}
        seen: dict[str, dict[str, datetime]] = {}

        def bucket(subject: str) -> dict[str, float]:
            return totals.setdefault(
                subject,
                {
                    "workloads": 0.0,
                    "settled_workloads": 0.0,
                    "leases": 0.0,
                    "accelerator_hours": 0.0,
                    "running_seconds": 0.0,
                    "queued_seconds": 0.0,
                    "settled_cost": 0.0,
                    "unpriced_leases": 0.0,
                },
            )

        for workload in workloads:
            if since is not None and workload.created_at < since:
                continue
            row = bucket(workload.created_by)
            row["workloads"] += 1
            if workload.state in TERMINAL_WORKLOAD_STATES:
                row["settled_workloads"] += 1
            stamps = seen.setdefault(workload.created_by, {})
            first = stamps.get("first")
            last = stamps.get("last")
            stamps["first"] = workload.created_at if first is None else min(first, workload.created_at)
            stamps["last"] = workload.created_at if last is None else max(last, workload.created_at)

        for lease in leases:
            workload = submitted_by.get(lease.workload_id)
            if workload is None:
                # A lease whose workload has been collected. Charging it to
                # nobody loses the cost; charging it to a guess is worse.
                continue
            if since is not None and workload.created_at < since:
                continue
            row = bucket(workload.created_by)
            row["leases"] += 1
            # A mapping of resource -> hours, not a scalar. Summed across
            # resources: a gang holding TPU and CPU settles both, and taking
            # only one would under-report the person who used the accelerator.
            row["accelerator_hours"] += sum((lease.accelerator_hours_settled or {}).values())
            row["running_seconds"] += lease.running_seconds or 0.0
            row["queued_seconds"] += lease.queued_seconds or 0.0
            settled = lease.settled_cost
            if settled is None or (settled == 0 and (lease.hourly_cost or 0) == 0):
                # Zero because the pool declares no unit price, not because the
                # lease was free. Counted separately so a heavy user of unpriced
                # capacity does not read as an idle one.
                row["unpriced_leases"] += 1
            row["settled_cost"] += settled or 0.0

        subjects = tuple(
            sorted(
                (
                    SubjectUsageRecord(
                        subject=subject,
                        org_id=org_id,
                        project_id=project_id,
                        workloads=int(row["workloads"]),
                        settled_workloads=int(row["settled_workloads"]),
                        leases=int(row["leases"]),
                        accelerator_hours=row["accelerator_hours"],
                        running_seconds=row["running_seconds"],
                        queued_seconds=row["queued_seconds"],
                        settled_cost=row["settled_cost"],
                        unpriced_leases=int(row["unpriced_leases"]),
                        first_submitted_at=seen.get(subject, {}).get("first"),
                        last_submitted_at=seen.get(subject, {}).get("last"),
                    )
                    for subject, row in totals.items()
                ),
                key=lambda record: (-record.settled_cost, -record.accelerator_hours, record.subject),
            )
        )
        return ProjectUsageReport(
            org_id=org_id,
            project_id=project_id,
            observed_at=utcnow(),
            subjects=subjects,
        )

    async def warmth_report(self, principal: Principal, *, now: datetime | None = None) -> WarmthReport:
        """Report every pool's warm floor and whether it is currently held.

        Uses the same ``cell_provides_pool_supply`` predicate the planner uses to
        decide what counts toward a floor. Reporting on a different definition
        than the one that acts would produce the worst possible answer: a report
        that says the floor is met while the planner keeps launching.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            pools = await tx.list_capacity_pools()
            cells = await tx.list_cells()
            leased = {lease.cell_id for lease in await tx.list_leases() if lease.state in ACTIVE_LEASE_STATES}
        warmth: list[PoolWarmth] = []
        for pool in pools:
            supplying = [
                cell
                for cell in cells
                if cell.pool_id == pool.id
                and cell.state is not CellState.DELETED
                and cell_provides_pool_supply(cell, pool, moment)
            ]
            floor = pool.warm_floor(moment)
            # Only the cells the floor actually reserves count; anything above it
            # is ordinary demand-driven capacity and belongs to no warm promise.
            # Ready ones first, so a floor that has one ready cell and one still
            # being requested reports the ready one as warm.
            reserved = sorted(supplying, key=lambda cell: cell.state is not CellState.READY)[: max(floor, 0)]
            # The planner counts an in-flight request as supply so it does not
            # launch twice, and that is right for planning. It is not warmth: a
            # Spot request still in the queue cannot serve a workload when its
            # warm window opens. Only ready capacity fulfills the warm promise.
            warm = [cell for cell in reserved if cell.state is CellState.READY]
            warmth.append(
                PoolWarmth(
                    pool_id=pool.id,
                    floor=floor,
                    window_active=any(window.covers(moment) for window in pool.warm_windows),
                    warm_cells=len(warm),
                    provisioning_cells=len(reserved) - len(warm),
                    warm_idle_cells=sum(1 for cell in warm if cell.id not in leased),
                    hourly_cost=costing.total(*(cell.hourly_cost() for cell in warm)),
                )
            )
        warmth.sort(key=lambda entry: entry.pool_id)
        return WarmthReport(observed_at=moment, pools=tuple(warmth))

    async def launch_latency_report(self, principal: Principal, *, now: datetime | None = None) -> LaunchLatencyReport:
        """Report how long each pool takes to hand over usable capacity.

        Read from the same ``provision_latencies_s`` samples the scheduler ranks
        on, rather than from a second time series. A separate store would drift
        from the numbers that actually decide placement, and the first symptom
        would be a dashboard disagreeing with a scheduling verdict -- which is
        worse than no dashboard.

        Percentiles are computed over the pool's cells pooled together, because
        an operator picks a *pool* and lives with whichever cell it hands them.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            pools = await tx.list_capacity_pools()
            cells = await tx.list_cells()
        by_pool: dict[str, list[CellRecord]] = {}
        for cell in cells:
            if cell.pool_id is not None and cell.state is not CellState.DELETED:
                by_pool.setdefault(cell.pool_id, []).append(cell)
        entries: list[PoolLaunchLatency] = []
        for pool in pools:
            members = by_pool.get(pool.id, [])
            samples = sorted(latency for cell in members for latency in cell.availability.provision_latencies_s)
            entries.append(
                PoolLaunchLatency(
                    pool_id=pool.id,
                    accelerator_type=pool.accelerator_type,
                    p95_seconds=p95_latency(samples),
                    median_seconds=statistics.median(samples) if samples else None,
                    slowest_seconds=samples[-1] if samples else None,
                    samples=len(samples),
                    cells_in_cooldown=sum(
                        1
                        for cell in members
                        if cell.availability.cooldown_until is not None and cell.availability.cooldown_until > moment
                    ),
                )
            )
        entries.sort(key=lambda entry: entry.pool_id)
        return LaunchLatencyReport(observed_at=moment, pools=tuple(entries))

    async def cell_billing_report(self, principal: Principal, *, pool_id: str | None = None) -> CellBillingReport:
        """Report accrued and idle cell cost, optionally for one pool.

        The idle figure is the one an operator cannot get anywhere else: it is
        capacity the fleet paid for and no lease used, and no lease-derived ledger
        can produce it, because an idle cell has no lease to settle.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            cells = [cell for cell in await tx.list_cells() if pool_id is None or cell.pool_id == pool_id]
        billed = sum(cell.billed_seconds for cell in cells)
        idle = sum(cell.idle_seconds for cell in cells)
        accrued_cost = costing.total(
            *(costing.cost_for_hours(cell.hourly_cost(), cell.billed_seconds / 3_600) for cell in cells)
        )
        idle_cost = costing.total(
            *(costing.cost_for_hours(cell.hourly_cost(), cell.idle_seconds / 3_600) for cell in cells)
        )
        return CellBillingReport(
            pool_id=pool_id,
            cells=len(cells),
            billed_seconds=billed,
            idle_seconds=idle,
            accrued_cost=accrued_cost,
            idle_cost=idle_cost,
            unpriced_cells=sum(1 for cell in cells if not cell.unit_prices),
        )
