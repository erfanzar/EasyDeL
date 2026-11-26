# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable asset-eviction exclusion at admission, launch and response writeback."""

from __future__ import annotations

import asyncio
import threading

import pytest

from eyvan.records import CellStageState, LeaseState, WorkloadState
from eyvan.server.controller.staging import CellStageUnsupportedError
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.stage import (
    CellStageInconclusiveError,
    CellStageMeasurement,
    CellStageRequest,
    CellStageStaleError,
    StageOutcome,
)
from tests.test_controller import GangFakeAgent
from tests.test_locality_stage import (
    NOW,
    _cell,
    _controller,
    _Executor,
    _lease,
    _policy,
    _seed,
    _stage,
    _stages,
    _workload,
)


def _members(count: int):
    workload = _workload("work", "asset", state=WorkloadState.PENDING)
    if count > 1:
        workload = workload.model_copy(
            update={
                "spec": workload.spec.model_copy(update={"cell_group_count": count, "required_executor": "eyvan-native"})
            }
        )
    cells = [_cell(f"cell-{index}") for index in range(count)]
    leases = [
        _lease(f"lease-{index}", workload_id=workload.id, cell_id=cell.id, state=LeaseState.PENDING).model_copy(
            update={"group_id": "group" if count > 1 else None, "group_index": index}
        )
        for index, cell in enumerate(cells)
    ]
    return workload, cells, leases


async def _admit(controller, leases):
    if len(leases) == 1:
        return await controller.service.create_lease(controller._principal, leases[0], idempotency_key="admit")
    return await controller.service.create_lease_group(controller._principal, leases, idempotency_key="admit")


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_eviction_reservation_refuses_atomic_lease_admission(count: int) -> None:
    repository = InMemoryRepository()
    controller = _controller(repository, None)
    controller.service.scheduler = None
    workload, cells, leases = _members(count)
    await _seed(
        repository,
        cells=cells,
        workloads=[workload],
        stages=[_stage(cells[-1].id, "asset", state=CellStageState.EVICTING)],
    )
    with pytest.raises(ConflictError, match="eviction"):
        await _admit(controller, leases)
    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
        assert stored is not None and stored.state is WorkloadState.PENDING
        assert await tx.list_leases() == []
        assert await tx.list_outbox() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_dispatch_refuses_eviction_without_reservation_or_agent_submit(count: int) -> None:
    repository = InMemoryRepository()
    agent = GangFakeAgent()
    controller = _controller(repository, None, agent_factory=lambda _cell: agent)
    controller.service.scheduler = None
    workload, cells, leases = _members(count)
    await _seed(repository, cells=cells, workloads=[workload])
    await _admit(controller, leases)
    # Model an alternate admission path/legacy row: launch must independently
    # revalidate the current-generation durable reservation before any RPC.
    await _seed(repository, stages=[_stage(cells[-1].id, "asset", state=CellStageState.EVICTING)])
    await controller.dispatch_outbox_once()
    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
        assert stored is not None and stored.dispatch_reservation is None
        assert stored.state is WorkloadState.LEASED
    assert agent.submissions == []
    assert agent.gang_submissions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("kind", ["different-asset", "old-generation"])
async def test_unrelated_eviction_does_not_block_lease_admission(count: int, kind: str) -> None:
    repository = InMemoryRepository()
    controller = _controller(repository, None)
    controller.service.scheduler = None
    workload, cells, leases = _members(count)
    if kind == "old-generation":
        cells = [cell.model_copy(update={"generation": 2}) for cell in cells]
        leases = [lease.model_copy(update={"cell_generation": 2}) for lease in leases]
    await _seed(
        repository,
        cells=cells,
        workloads=[workload],
        stages=[
            _stage(
                cells[-1].id,
                "other" if kind == "different-asset" else "asset",
                generation=1,
                state=CellStageState.EVICTING,
            )
        ],
    )
    await _admit(controller, leases)
    async with repository.transaction() as tx:
        assert len(await tx.list_leases()) == count
        stored = await tx.get_workload(workload.id)
        assert stored is not None and stored.state is WorkloadState.LEASED


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "unsupported", "stale", "inconclusive"])
async def test_stale_eviction_response_cannot_modify_successor_row(outcome: str) -> None:
    repository = InMemoryRepository()
    entered, release = threading.Event(), threading.Event()

    class _BarrierExecutor(_Executor):
        def evict(self, request):
            entered.set()
            if not release.wait(timeout=10):
                raise OSError("test did not release eviction")
            errors = {
                "unsupported": CellStageUnsupportedError("unsupported"),
                "stale": CellStageStaleError("fenced"),
                "inconclusive": CellStageInconclusiveError("unknown"),
            }
            if outcome in errors:
                raise errors[outcome]

    controller = _controller(repository, _BarrierExecutor(), policy=_policy(budget=1, per_action=1))
    await _seed(repository, cells=[_cell("cell")], stages=[_stage("cell", "asset", reserved=8)])
    eviction = asyncio.create_task(controller.evict_cell_stages_once(dry_run=False, now=NOW))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        prior = (await _stages(repository))[0]
        successor = prior.model_copy(update={"detail": "eviction:successor-operation", "reserved_bytes": 9})
        await _seed(repository, stages=[successor])
    finally:
        release.set()
        await eviction
    assert await _stages(repository) == [successor]


@pytest.mark.asyncio
async def test_inconclusive_eviction_remains_admission_blocking_and_not_retried() -> None:
    repository = InMemoryRepository()
    executor = _Executor(evict_fail={"asset": OSError("remote delete may still be running")})
    controller = _controller(repository, executor, policy=_policy(budget=1, per_action=1))
    controller.service.scheduler = None
    workload, cells, leases = _members(1)
    await _seed(repository, cells=cells, stages=[_stage(cells[0].id, "asset", reserved=8)])
    await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    row = (await _stages(repository))[0]
    assert row.state is CellStageState.EVICTING
    assert row.reserved_bytes == 8 and row.confirmed_at is None
    assert "inconclusive" in row.detail
    await _seed(repository, workloads=[workload])
    with pytest.raises(ConflictError, match="eviction"):
        await _admit(controller, leases)
    plans = await controller.evict_cell_stages_once(dry_run=False, now=NOW)
    assert len(executor.evicted) == 1
    assert plans[0].victims == ()
    assert plans[0].shortfall_bytes == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("writeback", ["reserve", "confirm", "unknown", "blocked"])
async def test_staging_writeback_cannot_erase_unresolved_eviction(writeback: str) -> None:
    repository = InMemoryRepository()
    controller = _controller(repository, None)
    row = _stage("cell", "asset", state=CellStageState.EVICTING).model_copy(
        update={"detail": "eviction:current-operation: in progress"}
    )
    await _seed(repository, cells=[_cell("cell")], stages=[row])
    request = CellStageRequest("cell", "asset", 1, "cell:cell:1", row.reserved_bytes)
    if writeback == "reserve":
        reserved = await controller._reserve_stage(request, measured=CellStageMeasurement(1), policy=_policy(), now=NOW)
        assert reserved is None
    elif writeback == "confirm":
        await controller._confirm_stage(request, written=CellStageMeasurement(1), now=NOW, outcome=StageOutcome())
    else:
        await controller._record_stage(
            request,
            state=CellStageState.UNKNOWN if writeback == "unknown" else CellStageState.BLOCKED,
            reserved_bytes=1,
            detail="stale staging response",
            now=NOW,
        )
    assert await _stages(repository) == [row]
