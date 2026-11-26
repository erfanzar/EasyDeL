# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A stale build recycles its cell instead of failing the workload that found it.

The dispatch guard refuses cells running a different build than the plane
stages, and used to settle that refusal as a FAILED workload whose message
named two commands for the operator to paste. But the operator here is the
plane: it staged the wheel, so it can drain the cell itself, and the
autoscaler behind a drain already deletes the emptied cell and provisions a
replacement from the wheel staged now.

These tests pin the whole hand-off. The drain is committed before anything
else and reaches the agent through the same durable outbox ``eyvan
cell-drain`` uses; the workload returns to PENDING to meet the replacement;
neither retry budget pays, because both default to zero and a recycle is
fleet maintenance the workload did not cause; and the attempt bound keeps a
broken staging path -- one whose replacements keep arriving stale -- from
reading as a queue rather than failing as the deployment fault it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from types import SimpleNamespace

import pytest
from test_controller import FakeAgent, principal

import eyvan._internal.build_identity as build_identity_module
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    ResourceGang,
    ResourceVector,
    Role,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.controller.workload_commands import _STALE_BUILD_REQUEUE_ATTEMPT_LIMIT
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.client import AgentBuild


def _plane_digest(monkeypatch: pytest.MonkeyPatch, digest: str) -> None:
    monkeypatch.setattr(
        build_identity_module,
        "build_identity",
        lambda: SimpleNamespace(source_digest=digest),
    )


class _BuildReportingAgent(FakeAgent):
    def __init__(self, source_digest: str):
        super().__init__()
        self._source_digest = source_digest

    def build(self):
        return AgentBuild(
            version="0.0.1",
            origin="wheel",
            origin_detail="",
            source_digest=self._source_digest,
            digest_detail="",
            file_count=1,
            python="3.13",
            platform="linux",
        )

    def worker_builds(self, cell_id):
        return [{"source_digest": self._source_digest}]


class _UnreadableAgent(FakeAgent):
    def build(self):
        raise RuntimeError("no route to the agent")


async def _leased_workload(
    agent: FakeAgent,
) -> tuple[
    InMemoryRepository,
    EyvanController,
    WorkloadRecord,
    OutboxRecord,
]:
    """One leased workload and its undispatched command, on a stale cell."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda _cell: agent,
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=1, memory_gib=1),
                allocatable=ResourceVector(cpu=1, memory_gib=1),
            )
        )
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="queued", execution=CommandExecution(command="python train.py")),
        idempotency_key="queued",
    )
    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [workload.id]
    async with repository.transaction() as transaction:
        leased = await transaction.get_workload(workload.id)
        assert leased is not None and leased.lease_id is not None
        commands = await transaction.list_outbox_for_aggregates(
            "workload",
            {leased.id: leased.generation},
            topic="workload.dispatch",
        )
    assert len(commands) == 1
    # `dispatch_outbox_once` claims a command before delivering it; calling the
    # dispatch path directly has to stage the same claim or it reads as stale.
    claimed = commands[0].model_copy(
        update={
            "state": OutboxState.PROCESSING,
            "attempts": 1,
            "claimed_by": controller.worker_id,
            "claimed_until": utcnow() + timedelta(seconds=60),
        }
    )
    async with repository.transaction() as transaction:
        await transaction.put_outbox(claimed)
    return repository, controller, leased, claimed


async def _cell_outbox(repository: InMemoryRepository, cell_id: str, topic: str) -> Sequence[str]:
    async with repository.transaction() as transaction:
        cell = await transaction.get_cell(cell_id)
        assert cell is not None
        commands = await transaction.list_outbox_for_aggregates(
            "cell",
            {cell.id: cell.generation},
            topic=topic,
        )
    return [command.aggregate_id for command in commands]


@pytest.mark.asyncio
async def test_a_stale_cell_is_drained_and_the_workload_waits_for_its_replacement(monkeypatch) -> None:
    _plane_digest(monkeypatch, "a" * 64)
    agent = _BuildReportingAgent("b" * 64)
    repository, controller, leased, command = await _leased_workload(agent)

    await controller._dispatch_workload(command)

    async with repository.transaction() as transaction:
        workload = await transaction.get_workload(leased.id)
        cell = await transaction.get_cell("cell-a")
        lease = await transaction.get_lease(leased.lease_id)
    assert workload is not None and cell is not None and lease is not None
    assert workload.state is WorkloadState.PENDING, "the workload waits for replacement capacity, it does not die"
    assert workload.attempt == leased.attempt + 1
    assert workload.capacity_loss_count == 0, "a recycle is maintenance, not a capacity loss"
    assert workload.preemption_count == 0
    assert "stale build" in (workload.scheduling_reason or "")
    assert cell.state is CellState.DRAINING, "the stale cell is on its way out before the workload requeues"
    assert lease.state is LeaseState.RELEASED
    assert agent.submissions == [], "nothing was dispatched onto the stale build"
    # The drain is the durable one `eyvan cell-drain` writes, not an in-memory
    # side effect: the agent is told through the outbox, so a crash between
    # these steps replays into a cell that is already draining.
    assert list(await _cell_outbox(repository, "cell-a", "capacity.drain")) == ["cell-a"]


@pytest.mark.asyncio
async def test_a_workload_past_the_recycle_limit_fails_with_the_commands_it_can_paste(monkeypatch) -> None:
    _plane_digest(monkeypatch, "a" * 64)
    agent = _BuildReportingAgent("b" * 64)
    repository, controller, leased, command = await _leased_workload(agent)
    async with repository.transaction() as transaction:
        await transaction.put_workload(leased.model_copy(update={"attempt": _STALE_BUILD_REQUEUE_ATTEMPT_LIMIT}))

    await controller._dispatch_workload(command)

    async with repository.transaction() as transaction:
        workload = await transaction.get_workload(leased.id)
        cell = await transaction.get_cell("cell-a")
    assert workload is not None and cell is not None
    assert workload.state is WorkloadState.FAILED
    assert workload.failure_reason is not None
    assert "eyvan cell-drain cell-a" in workload.failure_reason
    # The cause is still removed even when the workload dies: the drain is
    # committed before the bound is consulted.
    assert cell.state is CellState.DRAINING


@pytest.mark.asyncio
async def test_the_sweep_drains_stale_cells_and_leaves_matching_ones_alone(monkeypatch) -> None:
    _plane_digest(monkeypatch, "a" * 64)
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda cell: (
            _BuildReportingAgent("b" * 64) if cell.id == "cell-skewed" else _BuildReportingAgent("a" * 64)
        ),
    )
    async with repository.transaction() as transaction:
        for cell_id in ("cell-skewed", "cell-honest"):
            await transaction.put_cell(
                CellRecord(
                    id=cell_id,
                    provider="local",
                    region="local",
                    state=CellState.READY,
                    capacity=ResourceVector(cpu=1, memory_gib=1),
                    allocatable=ResourceVector(cpu=1, memory_gib=1),
                )
            )

    drained = await controller.recycle_stale_build_cells_once()

    assert drained == 1
    async with repository.transaction() as transaction:
        skewed = await transaction.get_cell("cell-skewed")
        honest = await transaction.get_cell("cell-honest")
    assert skewed is not None and skewed.state is CellState.DRAINING
    assert honest is not None and honest.state is CellState.READY
    assert list(await _cell_outbox(repository, "cell-skewed", "capacity.drain")) == ["cell-skewed"]
    assert list(await _cell_outbox(repository, "cell-honest", "capacity.drain")) == []


@pytest.mark.asyncio
async def test_the_sweep_leaves_a_stale_cell_hosting_an_active_lease_for_a_later_pass(monkeypatch) -> None:
    """Draining a leased cell strands its workload between a dead agent and a lease
    nothing requeues -- the dispatch guard only judges builds at dispatch time, so a
    running job on a stale build must finish on it and be recycled after."""

    _plane_digest(monkeypatch, "a" * 64)
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda cell: _BuildReportingAgent("b" * 64),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-leased",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=1, memory_gib=1),
                allocatable=ResourceVector(cpu=1, memory_gib=1),
            )
        )
        await transaction.put_lease(
            LeaseRecord(
                id="lease-live",
                workload_id="wl-live",
                cell_id="cell-leased",
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                resources=ResourceGang(per_replica=ResourceVector(cpu=1)),
                generation=1,
                cell_generation=1,
                fencing_token="token-live",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )

    drained = await controller.recycle_stale_build_cells_once()

    assert drained == 0
    async with repository.transaction() as transaction:
        cell = await transaction.get_cell("cell-leased")
    assert cell is not None and cell.state is CellState.READY
    assert list(await _cell_outbox(repository, "cell-leased", "capacity.drain")) == []


@pytest.mark.asyncio
async def test_the_sweep_skips_an_agent_it_cannot_read_and_keeps_recycling_the_rest(monkeypatch) -> None:
    _plane_digest(monkeypatch, "a" * 64)
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda cell: _UnreadableAgent() if cell.id == "cell-mute" else _BuildReportingAgent("b" * 64),
    )
    async with repository.transaction() as transaction:
        for cell_id in ("cell-mute", "cell-skewed"):
            await transaction.put_cell(
                CellRecord(
                    id=cell_id,
                    provider="local",
                    region="local",
                    state=CellState.READY,
                    capacity=ResourceVector(cpu=1, memory_gib=1),
                    allocatable=ResourceVector(cpu=1, memory_gib=1),
                )
            )

    drained = await controller.recycle_stale_build_cells_once()

    assert drained == 1
    async with repository.transaction() as transaction:
        mute = await transaction.get_cell("cell-mute")
        skewed = await transaction.get_cell("cell-skewed")
    assert mute is not None and mute.state is CellState.READY, "unreadable is a third state, not a mismatch"
    assert skewed is not None and skewed.state is CellState.DRAINING
