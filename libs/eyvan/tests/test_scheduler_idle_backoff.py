# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An idle single-process plane runs leader cycles less often and wakes on any commit."""

from __future__ import annotations

import asyncio

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CommandExecution,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController, LeaderPhase
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service import ControlPlaneService

_TICK_S = 0.01


class _CountingController(EyvanController):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cycles = 0
        self.cycle_started = asyncio.Event()

    def _leader_phases(self) -> tuple[LeaderPhase, ...]:
        async def count() -> None:
            self.cycles += 1
            self.cycle_started.set()

        return (LeaderPhase("count", count),)


def _controller(repository, *, idle_s: float) -> _CountingController:
    return _CountingController(
        repository,
        ControlPlaneService(repository),
        settings=EyvanSettings(scheduler_interval_s=_TICK_S, scheduler_idle_interval_s=idle_s),
    )


async def _run_for(controller: _CountingController, seconds: float) -> int:
    stop = asyncio.Event()
    task = asyncio.create_task(controller._run_scheduler_loop(stop))
    try:
        await asyncio.sleep(seconds)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2)
    return controller.cycles


def _cell(cell_id: str) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="gcp",
        region="us-east5",
        provider_project="provider-project",
        zone="us-east5-a",
        capacity={"tpu": 8},
        allocatable={"tpu": 8},
        accelerator_type="v5p-8",
        capacity_class="spot",
        endpoint="http://agent:8090",
    )


async def test_an_idle_loop_backs_off() -> None:
    repository = InMemoryRepository()
    cycles = await _run_for(_controller(repository, idle_s=5.0), 0.3)
    assert cycles <= 2, f"an idle plane must not cycle every {_TICK_S}s; ran {cycles} cycles"


async def test_zero_idle_interval_keeps_the_fixed_cadence() -> None:
    repository = InMemoryRepository()
    cycles = await _run_for(_controller(repository, idle_s=0.0), 0.3)
    assert cycles >= 5, f"the previous cadence must remain available; ran {cycles} cycles"


async def test_any_commit_wakes_an_idle_loop() -> None:
    repository = InMemoryRepository()
    controller = _controller(repository, idle_s=30.0)
    stop = asyncio.Event()
    task = asyncio.create_task(controller._run_scheduler_loop(stop))
    try:
        await asyncio.wait_for(controller.cycle_started.wait(), timeout=1)
        await asyncio.sleep(0.1)
        before = controller.cycles
        controller.cycle_started.clear()
        async with repository.transaction() as tx:
            await tx.put_cell(_cell("woken"))
        await asyncio.wait_for(controller.cycle_started.wait(), timeout=1)
        assert controller.cycles == before + 1
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2)


async def test_stop_ends_an_idle_wait_promptly() -> None:
    repository = InMemoryRepository()
    controller = _controller(repository, idle_s=60.0)
    stop = asyncio.Event()
    task = asyncio.create_task(controller._run_scheduler_loop(stop))
    await asyncio.wait_for(controller.cycle_started.wait(), timeout=1)
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=1)


async def test_a_live_workload_keeps_the_fixed_cadence() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as tx:
        await tx.put_workload(
            WorkloadRecord(
                id="wl-live",
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                state=WorkloadState.RUNNING,
                lease_id="lease-1",
                lease_ids=("lease-1",),
                remote_id="remote-1",
                created_by="alice",
            )
        )
    cycles = await _run_for(_controller(repository, idle_s=5.0), 0.3)
    assert cycles >= 5, f"a running workload's lease is judged by the clock; ran {cycles} cycles"


class _OpaqueRepository:
    """A repository without an in-process revision, like the SQL backend."""

    def __init__(self, inner: InMemoryRepository) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        if name in {"committed_revision", "wait_for_commit"}:
            raise AttributeError(name)
        return getattr(self._inner, name)


async def test_a_backend_without_a_revision_keeps_the_fixed_cadence() -> None:
    repository = _OpaqueRepository(InMemoryRepository())
    controller = _controller(repository, idle_s=5.0)
    assert controller._committed_revision() is None
    cycles = await _run_for(controller, 0.3)
    assert cycles >= 5


async def test_wait_for_commit_reports_commits_and_timeouts() -> None:
    repository = InMemoryRepository()
    revision = repository.committed_revision
    assert await repository.wait_for_commit(revision, timeout=0.02) is False

    async def commit_soon() -> None:
        await asyncio.sleep(0.02)
        async with repository.transaction() as tx:
            await tx.put_cell(_cell("committed"))

    committer = asyncio.create_task(commit_soon())
    assert await repository.wait_for_commit(revision, timeout=1) is True
    await committer
    assert repository.committed_revision == revision + 1
    assert await repository.wait_for_commit(revision, timeout=0) is True, "an already-moved revision returns at once"


@pytest.mark.parametrize("raw", ["0", "45"])
def test_idle_interval_is_parsed_and_validated(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    assert EyvanSettings().scheduler_idle_interval_s == 15.0
    monkeypatch.setenv("EYVAN_SCHEDULER_IDLE_INTERVAL_S", raw)
    assert EyvanSettings.from_env().scheduler_idle_interval_s == float(raw)
    with pytest.raises(ValueError, match="scheduler_idle_interval_s must be finite and non-negative"):
        EyvanSettings(scheduler_idle_interval_s=-1.0)
