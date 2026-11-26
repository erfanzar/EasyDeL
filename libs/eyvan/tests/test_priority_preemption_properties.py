# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded controller priority-preemption histories, not physical stop/failover proof."""

from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from test_controller import FakeAgent
from test_group_lease_lifecycle import _preemption
from test_running_retry_properties import _snapshot

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.records import LeaseRecord, LeaseState, OutboxState, WorkloadState, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


@pytest.fixture(params=["memory", "local", "postgres"])
async def repository(request, tmp_path):
    engine = admin = None
    schema = "priority_" + uuid4().hex
    if request.param == "memory":
        repo = InMemoryRepository()
    elif request.param == "local":
        repo = LocalJsonRepository(tmp_path / "state.json")
    else:
        url = request.getfixturevalue("postgres_url")
        admin = create_async_engine(url)
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
        await create_schema(engine)
        repo = SqlAlchemyRepository(engine)
    await repo.open()
    try:
        yield repo
    finally:
        await repo.close()
        if engine is not None:
            await engine.dispose()
        if admin is not None:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await admin.dispose()


class PerCellAgent(FakeAgent):
    def __init__(self, status):
        super().__init__()
        self.remote_status = ExecutionWorkloadStatus[status]
        self.observations = []

    def workload(self, cell_id, remote_id):
        self.observations.append((cell_id, remote_id))
        return super().workload(cell_id, remote_id)


def _controller(repository, agents):
    return EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        agent_factory=lambda cell: agents[cell.id],
    )


async def _requeue(repository, command_id):
    """Model redelivery by resetting only delivery state/claim fields."""
    async with repository.transaction() as tx:
        command = await tx.get_outbox(command_id)
        assert command is not None
        await tx.put_outbox(
            command.model_copy(update={"state": OutboxState.PENDING, "claimed_by": None, "claimed_until": None})
        )


async def _history(repository, statuses, replays):
    _, command = await _preemption(repository)
    agents = {f"cell-{suffix}": PerCellAgent(status) for suffix, status in zip("ab", statuses, strict=True)}
    async with repository.transaction() as tx:
        victim = await tx.get_workload("victim")
        assert victim is not None
        await tx.put_workload(victim.model_copy(update={"remote_id": "remote-victim"}))
    # The helper commits a real scheduler plan, but leaves a manually held claim.
    # From here every execution/settlement goes through actual outbox dispatch.
    await _requeue(repository, command.id)
    before = await _snapshot(repository, "victim")
    assert await _controller(repository, agents).dispatch_outbox_once() == 1
    after = await _snapshot(repository, "victim")
    expected = (
        WorkloadState.FAILED
        if "FAILED" in statuses
        else WorkloadState.SUCCEEDED
        if all(status == "SUCCEEDED" for status in statuses)
        else WorkloadState.PENDING
    )
    workload = after["workload"]
    assert workload.state is expected
    assert workload.generation == workload.attempt == (2 if expected is WorkloadState.PENDING else 1)
    assert workload.preemption_id is None
    assert workload.failure_count == workload.capacity_loss_count == workload.preemption_count == 0
    if expected is WorkloadState.PENDING:
        assert workload.lease_id is None and workload.lease_ids == () and workload.remote_id is None
    assert set(after["leases"]) == {"lease-a", "lease-b"}
    assert all(lease.state is LeaseState.RELEASED and lease.settled_at for lease in after["leases"].values())
    settlements = [event for event in after["events"] if event.action == "lease.settled"]
    assert len(settlements) == 2
    transitions = [
        event
        for event in after["events"]
        if event.action
        in {"workload.preempted", "workload.succeeded_during_preemption", "workload.failed_during_preemption"}
    ]
    assert len(transitions) == 1
    assert after["cells"] == before["cells"]
    # This fixture has no priced reservations; do not claim money conservation.
    assert after["budget"] == before["budget"]
    for suffix, status in zip("ab", statuses, strict=True):
        agent = agents[f"cell-{suffix}"]
        assert agent.observations and set(agent.observations) == {(f"cell-{suffix}", "remote-victim")}
        assert len(agent.stops) == (1 if status == "RUNNING" else 0)
        for cell_id, remote_id, request in agent.stops:
            assert (cell_id, remote_id) == (f"cell-{suffix}", "remote-victim")
            assert request.mutation.generation == 1
            assert request.mutation.fencing_token == f"fence-{suffix}"
            assert request.mutation.idempotency_key == f"{command.idempotency_key}:lease-{suffix}"
    finished = after["outbox"][command.id]
    assert finished.state is OutboxState.SUCCEEDED and finished.payload["execution_completed"] is True
    calls = {cell: (list(agent.observations), list(agent.stops)) for cell, agent in agents.items()}
    for index in range(replays):
        await _requeue(repository, command.id)
        assert await _controller(repository, agents).dispatch_outbox_once() == 1
        replayed = await _snapshot(repository, "victim")
        for key in ("workload", "leases", "events", "cells", "budget"):
            assert replayed[key] == after[key]
        assert {cell: (agent.observations, agent.stops) for cell, agent in agents.items()} == calls
        replay_command = replayed["outbox"][command.id]
        assert replay_command.payload == finished.payload
        assert replay_command.state is OutboxState.SUCCEEDED
        assert replay_command.attempts == finished.attempts + index + 1


@settings(max_examples=18, deadline=None)
@given(
    statuses=st.tuples(
        st.sampled_from(["RUNNING", "SUCCEEDED", "FAILED"]), st.sampled_from(["RUNNING", "SUCCEEDED", "FAILED"])
    ),
    replays=st.integers(min_value=1, max_value=3),
)
async def test_generated_priority_preemption_history(statuses, replays):
    await _history(InMemoryRepository(), statuses, replays)


@pytest.mark.parametrize(
    "statuses", [("RUNNING", "RUNNING"), ("SUCCEEDED", "SUCCEEDED"), ("SUCCEEDED", "FAILED"), ("SUCCEEDED", "RUNNING")]
)
async def test_priority_preemption_history_on_repository(repository, statuses):
    await _history(repository, statuses, 2)


async def test_stale_priority_plan_cannot_touch_successor(repository):
    _, command = await _preemption(repository)
    # Seed a coherent concurrently committed successor, not a claim to have run
    # a concurrent controller. Preserve the not-started plan and its old victims.
    async with repository.transaction() as tx:
        victim = await tx.get_workload("victim")
        assert victim is not None
        for suffix in "ab":
            old = await tx.get_lease(f"lease-{suffix}")
            assert old is not None
            await tx.put_lease(
                LeaseRecord.model_validate(
                    old.model_dump()
                    | {
                        "state": LeaseState.RELEASED,
                        "settled_at": utcnow(),
                        "settlement_reason": "preempted",
                    }
                )
            )
            successor = LeaseRecord.model_validate(
                old.model_dump()
                | {
                    "id": f"successor-{suffix}",
                    "generation": 2,
                    "state": LeaseState.ACTIVE,
                    "group_id": "successor-group",
                    "fencing_token": f"successor-fence-{suffix}",
                    "revocation_reason": None,
                    "settled_at": None,
                }
            )
            await tx.put_lease(successor)
        await tx.put_workload(
            victim.model_copy(
                update={
                    "generation": 2,
                    "attempt": 2,
                    "state": WorkloadState.RUNNING,
                    "lease_id": "successor-a",
                    "lease_ids": ("successor-a", "successor-b"),
                    "remote_id": "successor-remote",
                    "preemption_id": None,
                }
            )
        )
    await _requeue(repository, command.id)
    before = await _snapshot(repository, "victim")
    agents = {f"cell-{suffix}": PerCellAgent("RUNNING") for suffix in "ab"}
    assert await _controller(repository, agents).dispatch_outbox_once() == 1
    after = await _snapshot(repository, "victim")
    for key in ("workload", "leases", "events", "cells", "budget"):
        assert after[key] == before[key]
    assert all(not agent.observations and not agent.stops and not agent.submissions for agent in agents.values())
    assert after["outbox"][command.id].state is OutboxState.SUCCEEDED
    assert after["outbox"][command.id].payload == before["outbox"][command.id].payload
