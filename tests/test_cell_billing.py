# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The cell billing clock: what a fleet costs when nothing is running on it.

Eyvan has always charged *leases* -- a workload runs, its lease settles, the
project's budget moves. A provisioned cell holding no lease costs the provider
exactly as much and appeared in no ledger at all, so a fleet could burn a TPU
pod for a week between jobs while every budget reported full compliance.

Three properties are asserted, and each is what makes the number trustworthy
rather than merely present:

**Elapsed time, not tick count.** The clock charges ``now - last_tick``, so an
at-least-once leader loop that runs late, twice, or after a failover charges the
same amount as one that ran exactly on schedule.

**Queue time is free.** A queued TPU request is a record Eyvan holds and a bill
nobody is paying. Billing starts when the provider resource exists.

**Idle is measured, not inferred.** No lease-derived ledger can produce it: an
idle cell has no lease to settle, which is precisely why it was invisible.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    BILLABLE_CELL_STATES,
    CapacityClass,
    CellRecord,
    CellState,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


def _cell(
    cell_id: str = "cell-a",
    *,
    state: CellState = CellState.READY,
    prices: dict[str, float] | None = None,
    pool_id: str | None = None,
    **overrides: object,
) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="manual",
        capacity_backend="manual",
        region="us-east5",
        state=state,
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=4),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=4),
        unit_prices=prices if prices is not None else {"tpu": 2.0},
        capacity_class=CapacityClass.ON_DEMAND,
        pool_id=pool_id,
        **overrides,
    )


async def _put(service: ControlPlaneService, *records: CellRecord) -> None:
    async with service.repository.transaction() as tx:
        for record in records:
            await tx.put_cell(record)


def test_hourly_cost_multiplies_capacity_by_declared_prices() -> None:
    assert _cell(prices={"tpu": 2.0}).hourly_cost() == 8.0
    assert _cell(prices={"cpu": 0.5, "tpu": 1.0}).hourly_cost() == 8.0


def test_an_unpriced_cell_costs_zero_and_that_is_the_defect_not_a_discount() -> None:
    """D7's shape: no prices dots to zero, and a report must surface it."""

    assert _cell(prices={}).hourly_cost() == 0.0


@pytest.mark.asyncio
async def test_the_first_pass_starts_the_clock_without_charging_for_queue_time() -> None:
    service = await _service()
    old = utcnow() - timedelta(hours=10)
    await _put(service, _cell(created_at=old, updated_at=old))

    accrual = await service.accrue_cell_billing()

    assert accrual.cells == 0, "nothing is charged before the clock starts"
    assert accrual.accrued_cost == 0.0
    async with service.repository.transaction() as tx:
        stored = await tx.get_cell("cell-a")
    assert stored is not None
    assert stored.billing_started_at is not None
    assert stored.billed_seconds == 0.0


@pytest.mark.asyncio
async def test_the_second_pass_charges_the_elapsed_interval() -> None:
    service = await _service()
    await _put(service, _cell())
    start = utcnow()
    await service.accrue_cell_billing(now=start)

    accrual = await service.accrue_cell_billing(now=start + timedelta(hours=2))

    assert accrual.cells == 1
    assert accrual.accrued_seconds == pytest.approx(7200.0)
    assert accrual.accrued_cost == pytest.approx(16.0), "2h of an 8/hour cell"


@pytest.mark.asyncio
async def test_charging_follows_elapsed_time_not_the_number_of_passes() -> None:
    """A leader loop that fires twice as often must not bill twice as much."""

    slow = await _service()
    fast = await _service()
    await _put(slow, _cell())
    await _put(fast, _cell())
    start = utcnow()
    await slow.accrue_cell_billing(now=start)
    await fast.accrue_cell_billing(now=start)

    await slow.accrue_cell_billing(now=start + timedelta(hours=4))
    for hour in range(1, 5):
        await fast.accrue_cell_billing(now=start + timedelta(hours=hour))

    async with slow.repository.transaction() as tx:
        slow_cell = await tx.get_cell("cell-a")
    async with fast.repository.transaction() as tx:
        fast_cell = await tx.get_cell("cell-a")
    assert slow_cell is not None and fast_cell is not None
    assert slow_cell.billed_seconds == pytest.approx(fast_cell.billed_seconds)


@pytest.mark.asyncio
async def test_a_registering_cell_is_not_billed() -> None:
    """A queued TPU request is a record Eyvan holds, not a bill anyone is paying."""

    assert CellState.REGISTERING not in BILLABLE_CELL_STATES
    service = await _service()
    await _put(service, _cell(state=CellState.REGISTERING))
    start = utcnow()

    await service.accrue_cell_billing(now=start)
    accrual = await service.accrue_cell_billing(now=start + timedelta(hours=5))

    assert accrual.cells == 0
    async with service.repository.transaction() as tx:
        stored = await tx.get_cell("cell-a")
    assert stored is not None
    assert stored.billing_started_at is None
    assert stored.billed_seconds == 0.0


@pytest.mark.asyncio
async def test_a_deleted_cell_stops_the_clock_rather_than_leaving_it_running() -> None:
    """Otherwise a cell that becomes READY again bills for the gap it did not exist."""

    service = await _service()
    await _put(service, _cell())
    start = utcnow()
    await service.accrue_cell_billing(now=start)
    await service.accrue_cell_billing(now=start + timedelta(hours=1))

    async with service.repository.transaction() as tx:
        live = await tx.get_cell("cell-a")
        assert live is not None
        await tx.put_cell(live.model_copy(update={"state": CellState.DELETED}))
    await service.accrue_cell_billing(now=start + timedelta(hours=2))

    async with service.repository.transaction() as tx:
        stopped = await tx.get_cell("cell-a")
    assert stopped is not None
    assert stopped.billing_started_at is None
    assert stopped.billed_seconds == pytest.approx(3600.0), "the hour it was alive, and no more"

    async with service.repository.transaction() as tx:
        await tx.put_cell(stopped.model_copy(update={"state": CellState.READY}))
    await service.accrue_cell_billing(now=start + timedelta(hours=9))
    await service.accrue_cell_billing(now=start + timedelta(hours=10))

    async with service.repository.transaction() as tx:
        revived = await tx.get_cell("cell-a")
    assert revived is not None
    assert revived.billed_seconds == pytest.approx(7200.0), "1h alive + 1h alive again, not the 7h gap"


@pytest.mark.asyncio
async def test_a_cell_with_no_active_lease_accrues_idle_time() -> None:
    service = await _service()
    await _put(service, _cell())
    start = utcnow()
    await service.accrue_cell_billing(now=start)

    accrual = await service.accrue_cell_billing(now=start + timedelta(hours=3))

    assert accrual.idle_seconds == pytest.approx(10800.0)
    assert accrual.idle_cost == pytest.approx(24.0)


@pytest.mark.asyncio
async def test_a_leased_cell_accrues_cost_but_not_idle_time() -> None:
    service = await _service()
    await _put(service, _cell())
    async with service.repository.transaction() as tx:
        await tx.put_lease(
            LeaseRecord(
                id="lease-1",
                workload_id="wl-1",
                cell_id="cell-a",
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                fencing_token="token-1",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )
    start = utcnow()
    await service.accrue_cell_billing(now=start)

    accrual = await service.accrue_cell_billing(now=start + timedelta(hours=3))

    assert accrual.accrued_seconds == pytest.approx(10800.0)
    assert accrual.idle_seconds == 0.0, "a leased cell is paid for and used"


@pytest.mark.asyncio
async def test_the_report_separates_idle_cost_from_total_and_names_unpriced_cells() -> None:
    service = await _service()
    await _put(service, _cell("cell-a", pool_id="pool-1"), _cell("cell-b", prices={}, pool_id="pool-1"))
    start = utcnow()
    await service.accrue_cell_billing(now=start)
    await service.accrue_cell_billing(now=start + timedelta(hours=1))

    report = await service.cell_billing_report(_admin())

    assert report.cells == 2
    assert report.billed_seconds == pytest.approx(7200.0), "two cells, one hour each"
    assert report.idle_seconds == pytest.approx(7200.0)
    assert report.accrued_cost == pytest.approx(8.0), "only the priced cell contributes"
    assert report.idle_cost == pytest.approx(8.0)
    assert report.unpriced_cells == 1, "an unpriced cell must be visible, not silently free"


@pytest.mark.asyncio
async def test_the_report_can_be_scoped_to_one_pool() -> None:
    service = await _service()
    await _put(service, _cell("cell-a", pool_id="pool-1"), _cell("cell-b", pool_id="pool-2"))
    start = utcnow()
    await service.accrue_cell_billing(now=start)
    await service.accrue_cell_billing(now=start + timedelta(hours=1))

    report = await service.cell_billing_report(_admin(), pool_id="pool-1")

    assert report.cells == 1
    assert report.accrued_cost == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_a_clock_that_is_never_ticked_twice_charges_nothing() -> None:
    """Idempotence at zero elapsed: a repeated pass at the same instant is free."""

    service = await _service()
    await _put(service, _cell())
    start = utcnow()
    await service.accrue_cell_billing(now=start)

    repeated = await service.accrue_cell_billing(now=start)

    assert repeated.cells == 0
    assert repeated.accrued_cost == 0.0
