# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The number the scheduler has always ranked on, finally shown to someone.

``startup_latency_seconds`` is a p95 over each cell's ``provision_latencies_s``,
computed on every scheduling pass and consumed by the placement score. Nothing
ever surfaced it, so "why did my job land in the slow region" had no answer that
did not involve reading the scheduler.

Read from the *same* samples rather than a second time series, because a separate
store would drift and the first symptom would be a dashboard disagreeing with a
scheduling verdict -- worse than no dashboard.

The property that carries the design is that every figure is reported with its
sample count. A p95 over three provisions is not a p95, and a pool that has never
provisioned successfully is not fast: it is unmeasured, and the two are the same
shape in a table sorted ascending by latency.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CapacityBeliefState,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    Principal,
    ResourceVector,
    Role,
    RoleGrant,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _cell(cell_id: str, *, pool: str, latencies: tuple[float, ...] = (), cooldown_s: float | None = None) -> CellRecord:
    from datetime import timedelta

    # The cooldown is stamped when the fixture is built, not when the module is
    # imported: the report compares it against the wall clock, and between
    # collection-time import and this test's execution the window can quietly
    # close (a slept machine alone is enough).
    return CellRecord(
        id=cell_id,
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        endpoint="http://agent:8090",
        pool_id=pool,
        availability=CapacityBeliefState(
            provision_latencies_s=latencies,
            cooldown_until=None if cooldown_s is None else utcnow() + timedelta(seconds=cooldown_s),
        ),
    )


def _pool(pool_id: str = "research/v5p", *, accelerator: str | None = "v5p-8") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="tpu",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        accelerator_type=accelerator,
        agent_endpoint="http://agent:8090",
    )


async def _service(*records: object) -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    async with repository.transaction() as tx:
        for record in records:
            if isinstance(record, CapacityPoolRecord):
                await tx.put_capacity_pool(record)
            else:
                await tx.put_cell(record)  # type: ignore[arg-type]
    return ControlPlaneService(repository=repository)


@pytest.mark.asyncio
async def test_a_pool_reports_the_percentile_the_scheduler_ranks_on() -> None:
    """Same samples, same nearest-rank p95 -- not a parallel calculation."""

    from eyvan.domain.availability import p95_latency

    samples = (10.0, 20.0, 30.0, 40.0, 900.0)
    service = await _service(_pool(), _cell("cell-a", pool="research/v5p", latencies=samples))

    report = await service.launch_latency_report(_admin())

    assert report.pools[0].p95_seconds == p95_latency(samples)
    assert report.pools[0].slowest_seconds == 900.0
    assert report.pools[0].samples == 5


@pytest.mark.asyncio
async def test_a_pool_pools_its_cells() -> None:
    """An operator picks a pool and lives with whichever cell it hands them."""

    service = await _service(
        _pool(),
        _cell("cell-a", pool="research/v5p", latencies=(10.0, 10.0)),
        _cell("cell-b", pool="research/v5p", latencies=(500.0,)),
    )

    report = await service.launch_latency_report(_admin())

    assert report.pools[0].samples == 3
    assert report.pools[0].slowest_seconds == 500.0


@pytest.mark.asyncio
async def test_an_unmeasured_pool_is_not_reported_as_instant() -> None:
    """Zero and None are different answers, and one of them is dangerous.

    A pool that has never provisioned successfully sorts to the top of a table
    ordered by latency, where it reads as the fastest option available.
    """

    service = await _service(_pool(), _cell("cell-a", pool="research/v5p"))

    report = await service.launch_latency_report(_admin())

    assert report.pools[0].p95_seconds is None
    assert report.pools[0].samples == 0
    assert report.unmeasured == report.pools


@pytest.mark.asyncio
async def test_a_pool_with_no_cells_at_all_still_appears() -> None:
    """Omitting it would make an unused pool indistinguishable from a deleted one."""

    service = await _service(_pool())

    report = await service.launch_latency_report(_admin())

    assert [pool.pool_id for pool in report.pools] == ["research/v5p"]


@pytest.mark.asyncio
async def test_cells_in_cooldown_are_counted() -> None:
    """A pool whose latency looks excellent because it stopped being asked.

    Cooldown suppresses provisioning attempts, so the samples stop arriving and
    the reported figure freezes at whatever it was before the failures began.
    Without this count the report would be quietly stale in exactly the case an
    operator most needs it to be honest.
    """

    service = await _service(
        _pool(),
        _cell("cell-a", pool="research/v5p", latencies=(10.0,), cooldown_s=600),
        _cell("cell-b", pool="research/v5p", latencies=(10.0,)),
    )

    report = await service.launch_latency_report(_admin())

    assert report.pools[0].cells_in_cooldown == 1


@pytest.mark.asyncio
async def test_a_deleted_cell_does_not_drag_the_figure() -> None:
    """Released capacity says nothing about how long the pool takes today."""

    gone = _cell("cell-old", pool="research/v5p", latencies=(9_000.0,)).model_copy(update={"state": CellState.DELETED})
    service = await _service(_pool(), gone, _cell("cell-a", pool="research/v5p", latencies=(10.0,)))

    report = await service.launch_latency_report(_admin())

    assert report.pools[0].samples == 1
    assert report.pools[0].slowest_seconds == 10.0


@pytest.mark.asyncio
async def test_reading_it_requires_fleet_scope() -> None:
    """Provisioning latency is fleet capacity detail, like warmth and billing."""

    from eyvan.server.service.errors import AuthorizationError

    service = await _service(_pool())

    with pytest.raises(AuthorizationError):
        await service.launch_latency_report(
            Principal(subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="r"),))
        )


def test_the_command_is_attached() -> None:
    """A report with no client is a report nobody reads."""

    from eyvan.sdk import cli

    assert "launch-latency" in cli.main.commands["capacity"].commands
