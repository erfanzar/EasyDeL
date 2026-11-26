# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observing and finishing a job must not cost the sum of its hosts.

Archive rank output concurrently so terminal settlement does not block other
workloads for the sum of per-host latencies, including on wide slices.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
import typing as tp

import httpx
import pytest

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.worker.runtime.native_slice import _EXEC_CONCURRENCY_LIMIT, NativeSliceExecutor


class _InFlight:
    """Counts concurrent requests across every host's transport."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
            self.paths.append(request.url.path)
        try:
            time.sleep(self.hold_s)
            if request.url.path.endswith("/log-page"):
                return httpx.Response(200, json={"records": [], "next_cursor": ""})
            return httpx.Response(200, json={"remote_id": "slice-job", "status": "RUNNING"})
        finally:
            with self.lock:
                self.current -= 1


def _slice(hosts: int, counter: _InFlight) -> tuple[NativeSliceExecutor, list[httpx.Client]]:
    endpoints = tuple(f"http://worker-{index}:8100" for index in range(hosts))
    clients = [httpx.Client(base_url=endpoint, transport=httpx.MockTransport(counter.handler)) for endpoint in endpoints]
    return NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients), clients  # pragma: allowlist secret


@pytest.mark.parametrize("hosts", [2, 16, 64, 128])
@pytest.mark.parametrize("read", ["status", "log_page"])
def test_slice_reads_ask_every_host_at_once_within_the_thread_bound(hosts: int, read: str) -> None:
    counter = _InFlight(hold_s=0.02)
    executor, clients = _slice(hosts, counter)
    try:
        if read == "status":
            result = executor.status("slice-job")
            assert result["status"] is ExecutionWorkloadStatus.RUNNING
            # Answers stay in rank order whatever order the hosts finished in.
            assert [worker["rank"] for worker in result["workers"]] == list(range(hosts))
        else:
            page = executor.log_page("slice-job")
            assert not page.unavailable
    finally:
        for client in clients:
            client.close()

    assert len(counter.paths) == hosts
    assert counter.peak <= _EXEC_CONCURRENCY_LIMIT
    # Concurrent, not one host at a time: with the hold above, a sequential walk
    # can never have more than one request in flight.
    assert counter.peak == min(hosts, _EXEC_CONCURRENCY_LIMIT)


async def _dispatched(interval: float) -> tuple[tp.Any, tp.Any, tp.Any, tp.Any, tp.Any]:
    """A controller with one workload dispatched to a fake agent's cell."""

    from eyvan.config import EyvanSettings
    from eyvan.records import CellRecord, CellState, CommandExecution, ResourceVector, Role, WorkloadDefinition
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService
    from tests.test_controller import FakeAgent, principal

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=dataclasses.replace(EyvanSettings(), scheduler_interval_s=interval),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type]
    )
    admin = principal(Role.FLEET_ADMIN)
    operator = principal(Role.PROJECT_OPERATOR)
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="p",
            region="us-central2",
            zone="us-central2-b",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            allocatable=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            accelerator_type="v5p-8",
            capacity_class="spot",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 2, "memory_gib": 8, "tpu": 8}},
            capacity_class="spot",
        ),
        idempotency_key="submit",
    )
    await controller.dispatch_outbox_once()
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    return controller, service, agent, admin, workload


@pytest.mark.asyncio
async def test_a_slow_terminal_archive_does_not_hold_the_observation_pass() -> None:
    from eyvan.records import WorkloadState

    interval = 0.05
    controller, service, agent, admin, workload = await _dispatched(interval)
    release = asyncio.Event()
    archived: list[str] = []

    async def slow_archive(current: tp.Any) -> None:
        archived.append(current.id)
        await release.wait()

    controller._archive_terminal_logs = slow_archive  # type: ignore[method-assign]
    agent.remote_status = ExecutionWorkloadStatus.SUCCEEDED

    started = time.monotonic()
    await controller.observe_workloads_once()
    # Bounded by the scheduler interval, not by however long the archive takes.
    assert time.monotonic() - started < interval + 1.0
    assert archived == [workload.id]
    assert (await service.get_workload(admin, workload.id)).state is not WorkloadState.SUCCEEDED
    # Settlement holds the same per-attempt lock the live-log pass takes.
    assert controller._log_archive_lock(workload.id, workload.attempt).locked()

    # A later pass does not start a second settlement of the same attempt.
    await controller.observe_workloads_once()
    assert archived == [workload.id]

    release.set()
    await asyncio.gather(*list(controller._settling_workloads.values()))
    assert (await service.get_workload(admin, workload.id)).state is WorkloadState.SUCCEEDED
    assert not controller._settling_workloads
    await controller.close()


@pytest.mark.asyncio
async def test_closing_the_controller_drains_or_cancels_in_flight_settlements() -> None:
    from eyvan.config import EyvanSettings
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=dataclasses.replace(EyvanSettings(), cell_reconcile_interval_s=0.05),
    )
    never = asyncio.Event()
    task = asyncio.ensure_future(never.wait())
    controller._settling_workloads[("wl-1", 1)] = task  # type: ignore[assignment]

    await asyncio.wait_for(controller.close(), timeout=5)

    assert task.cancelled()


@pytest.mark.asyncio
async def test_a_background_phase_does_not_hold_the_cycle_and_never_overlaps_itself() -> None:
    from eyvan.server.controller._support import LeaderPhase

    controller, *_ = await _dispatched(0.05)
    release = asyncio.Event()
    started: list[float] = []
    inline: list[float] = []

    async def slow() -> None:
        started.append(time.monotonic())
        await release.wait()

    async def quick() -> None:
        inline.append(time.monotonic())

    controller._leader_phases = lambda: (  # type: ignore[method-assign]
        LeaderPhase("slow-housekeeping", slow, background=True),
        LeaderPhase("schedule", quick),
    )
    for _ in range(3):
        await asyncio.wait_for(controller._run_leader_phases(now=0.0, due={}), timeout=1.0)
        await asyncio.sleep(0)

    # The phase after it ran every cycle; the slow one started exactly once.
    assert len(inline) == 3
    assert len(started) == 1

    release.set()
    await controller._background_phases["slow-housekeeping"]
    await controller._run_leader_phases(now=0.0, due={})
    await asyncio.sleep(0)
    assert len(started) == 2
    await controller.close()


@pytest.mark.asyncio
async def test_one_slow_cell_observation_does_not_hold_the_observe_cells_phase() -> None:
    """Keep a slow cell's runtime probe from blocking the observation phase for other cells."""

    interval = 0.05
    controller, _service, agent, _admin, _workload = await _dispatched(interval)
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    original = agent.reconcile

    def slow(cell_id: str, request: tp.Any) -> tp.Any:
        calls.append(cell_id)
        entered.set()
        assert release.wait(10)
        return original(cell_id, request)

    agent.reconcile = slow
    try:
        started = time.monotonic()
        assert await controller._observe_cells_once() == 0
        # Bounded by the scheduler interval, not by the agent's answer.
        assert time.monotonic() - started < interval + 1.0
        assert entered.wait(5)
        # A later pass does not start a second observation of the same cell.
        await controller._observe_cells_once()
        assert calls == ["cell-a"]
    finally:
        release.set()
    await asyncio.gather(*list(controller._observing_cells.values()))
    assert not controller._observing_cells
    # Once the agent answers promptly again, the pass completes inside itself.
    agent.reconcile = original
    assert await controller._observe_cells_once() == 1
    await controller.close()


@pytest.mark.asyncio
async def test_a_live_log_pass_waits_for_the_terminal_archive_of_the_same_attempt() -> None:
    from eyvan.records import WorkloadState

    controller, service, _agent, admin, workload = await _dispatched(0.05)
    await controller.observe_workloads_once()
    running = await service.get_workload(admin, workload.id)
    assert running.state is WorkloadState.RUNNING
    chunked: list[str] = []

    async def record_chunk(current: tp.Any, *, limit: int) -> bool:
        del limit
        chunked.append(current.id)
        return True

    controller.artifact_access = object()
    controller._archive_log_chunk = record_chunk  # type: ignore[method-assign]
    lock = controller._log_archive_lock(running.id, running.attempt)
    await lock.acquire()
    try:
        task = asyncio.ensure_future(controller.archive_running_logs_once())
        await asyncio.sleep(0.05)
        assert chunked == []
        assert not task.done()
    finally:
        lock.release()

    assert await asyncio.wait_for(task, timeout=5) == 1
    assert chunked == [running.id]
    await controller.close()
