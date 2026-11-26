# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Warm-floor reporting: is the promise held, and what does holding it cost?

A warm window is a promise an operator wrote and nothing reported on. Two things
went wrong invisibly. The floor is not met, so the 09:00 provisioning stampede
the window exists to prevent happens anyway. Or the floor *is* met and those
cells sit unused -- which is correct, and which looked identical to waste in
every other view the package offers.

That second case is the one that matters. Warm capacity is deliberately idle, so
without a report naming it, an operator reading the fleet's idle cost cannot tell
reserved capacity from bad packing, and the only safe reaction to an unexplained
idle number is to delete the warm floor -- undoing the fix it paid for.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from eyvan._internal.compat import UTC
from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
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
    WarmWindow,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

#: A Wednesday, inside a weekday 08:00-18:00 window.
_WORKDAY = datetime(2026, 7, 29, 10, 0, tzinfo=UTC)
#: The same Wednesday, before it opens.
_NIGHT = datetime(2026, 7, 29, 3, 0, tzinfo=UTC)


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


def _pool(
    pool_id: str = "research/v5p-16",
    *,
    min_cells: int = 0,
    windows: tuple[WarmWindow, ...] = (),
    max_cells: int = 10,
) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="v5p-16",
        provider="manual",
        capacity_backend="manual",
        region="us-east5",
        agent_endpoint="http://agent:8080",
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        unit_prices={"tpu": 2.0},
        accelerator_type="v5p-16",
        capacity_class=CapacityClass.ON_DEMAND,
        min_cells=min_cells,
        warm_windows=windows,
        max_cells=max_cells,
        initialized_at=utcnow(),
    )


def _cell(cell_id: str, pool: CapacityPoolRecord | None = None) -> CellRecord:
    """Build a cell the way the autoscaler does: from the pool's own template.

    Hand-building one is how a test ends up asserting against a cell the planner
    would not count, since ``cell_matches_template`` compares every provisionable
    dimension and any drift makes the cell invisible to its own pool.
    """

    return (pool or _pool()).make_cell(cell_id, worker_count=1).model_copy(update={"state": CellState.READY})


def _workday_window(min_cells: int = 4) -> WarmWindow:
    return WarmWindow(min_cells=min_cells, start_hour=8, end_hour=18, days=(1, 2, 3, 4, 5))


async def _seed(service: ControlPlaneService, pool: CapacityPoolRecord, *cells: CellRecord) -> None:
    async with service.repository.transaction() as tx:
        await tx.put_capacity_pool(pool)
        for cell in cells:
            await tx.put_cell(cell)


@pytest.mark.asyncio
async def test_a_window_raises_the_reported_floor_only_while_it_covers_now() -> None:
    service = await _service()
    await _seed(service, _pool(windows=(_workday_window(4),)))

    day = await service.warmth_report(_admin(), now=_WORKDAY)
    night = await service.warmth_report(_admin(), now=_NIGHT)

    assert day.pools[0].floor == 4
    assert day.pools[0].window_active is True
    assert night.pools[0].floor == 0
    assert night.pools[0].window_active is False


@pytest.mark.asyncio
async def test_a_standing_minimum_is_reported_without_claiming_a_window() -> None:
    """ "Why is this pool not empty at 03:00" has two different answers."""

    service = await _service()
    await _seed(service, _pool(min_cells=2))

    report = await service.warmth_report(_admin(), now=_NIGHT)

    assert report.pools[0].floor == 2
    assert report.pools[0].window_active is False


@pytest.mark.asyncio
async def test_an_unmet_floor_is_surfaced_as_unsatisfied() -> None:
    service = await _service()
    pool = _pool(windows=(_workday_window(4),))
    await _seed(service, pool, _cell("cell-a", pool))

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert report.pools[0].warm_cells == 1
    assert report.pools[0].satisfied is False
    assert [pool.pool_id for pool in report.unsatisfied] == ["research/v5p-16"]


@pytest.mark.asyncio
async def test_a_met_floor_is_satisfied_and_reports_its_hourly_cost() -> None:
    service = await _service()
    pool = _pool(windows=(_workday_window(2),))
    await _seed(service, pool, _cell("cell-a", pool), _cell("cell-b", pool))

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert report.pools[0].satisfied is True
    assert report.pools[0].warm_cells == 2
    assert report.pools[0].hourly_cost == pytest.approx(32.0), "two 16/hour cells"
    assert report.warm_hourly_cost == pytest.approx(32.0)


@pytest.mark.asyncio
async def test_capacity_above_the_floor_is_not_counted_as_warm() -> None:
    """Demand-driven cells belong to no warm promise, so charging them to one lies."""

    service = await _service()
    pool = _pool(windows=(_workday_window(1),))
    await _seed(service, pool, _cell("cell-a", pool), _cell("cell-b", pool), _cell("cell-c", pool))

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert report.pools[0].warm_cells == 1
    assert report.pools[0].hourly_cost == pytest.approx(16.0), "one cell's worth, not three"


@pytest.mark.asyncio
async def test_warm_idle_separates_reserved_capacity_from_wasted_capacity() -> None:
    """The join with the billing clock: this is the idle that is supposed to exist."""

    service = await _service()
    pool = _pool(windows=(_workday_window(2),))
    await _seed(service, pool, _cell("cell-a", pool), _cell("cell-b", pool))
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

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert report.pools[0].warm_cells == 2
    assert report.pools[0].warm_idle_cells == 1, "one warm cell is working, one is reserved"


@pytest.mark.asyncio
async def test_a_pool_with_no_floor_is_reported_as_permitted_to_scale_to_zero() -> None:
    """Absent is not the same as zero: an operator needs to see the pool listed."""

    service = await _service()
    await _seed(service, _pool())

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert len(report.pools) == 1
    assert report.pools[0].floor == 0
    assert report.pools[0].satisfied is True, "no floor is trivially held"
    assert report.unsatisfied == ()


@pytest.mark.asyncio
async def test_a_deleted_cell_does_not_count_toward_a_floor() -> None:
    """Otherwise a fleet reads as warm while nothing is actually running."""

    service = await _service()
    pool = _pool(windows=(_workday_window(1),))
    await _seed(service, pool, _cell("cell-a", pool))
    async with service.repository.transaction() as tx:
        live = await tx.get_cell("cell-a")
        assert live is not None
        await tx.put_cell(live.model_copy(update={"state": CellState.DELETED}))

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert report.pools[0].warm_cells == 0
    assert report.pools[0].satisfied is False


@pytest.mark.asyncio
async def test_pools_are_reported_in_a_stable_order() -> None:
    service = await _service()
    async with service.repository.transaction() as tx:
        for name in ("research/c", "research/a", "research/b"):
            await tx.put_capacity_pool(_pool(name))

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert [pool.pool_id for pool in report.pools] == ["research/a", "research/b", "research/c"]


@pytest.mark.asyncio
async def test_a_requested_cell_is_provisioning_not_warm() -> None:
    """Count in-flight requests as provisioning, not warm capacity, even when the planner counts them as supply."""

    service = await _service()
    pool = _pool(windows=(_workday_window(2),))
    ready = _cell("cell-a", pool)
    requested = _cell("cell-b", pool).model_copy(update={"state": CellState.REGISTERING})
    await _seed(service, pool, requested, ready)

    report = await service.warmth_report(_admin(), now=_WORKDAY)

    assert report.pools[0].floor == 2
    assert report.pools[0].warm_cells == 1
    assert report.pools[0].provisioning_cells == 1
    assert report.pools[0].satisfied is False
