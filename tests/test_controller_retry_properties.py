# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Controller/outbox histories with an agent double, not hardware execution."""

from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_controller import FakeAgent
from test_group_lease_lifecycle import repository as _repository_fixture
from test_running_retry_properties import ADMIN, _seed, _snapshot

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.records import LeaseRecord, LeaseState, OutboxState, ResourceHours, WorkloadState, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

repository = _repository_fixture


async def _dispatch_history(repository, failures: int, rate: int) -> None:
    key = f"dispatch-{failures}-{rate}"
    workload_id, cell = await _seed(repository, key, failures)
    cell = type(cell).model_validate(
        cell.model_dump()
        | {
            "provider": "gcp",
            "provider_project": "test-project",
            "region": "us-central2",
            "zone": "us-central2-b",
            "endpoint": "http://agent:8090",
        }
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
    initial = await _snapshot(repository, workload_id)

    class OwnedAgent(FakeAgent):
        def workload(self, cell_id, remote_id):
            submitted_cell, request = self.submissions[-1]
            assert cell_id == submitted_cell
            assert remote_id == request.spec["submission_id"]
            return super().workload(cell_id, remote_id)

    agent = OwnedAgent()
    released = {}
    remote_ids = set()

    for attempt in range(failures + 1):
        # Reconstruct both objects: persisted state, not controller-local state,
        # must carry admission, ownership and accounting between attempts.
        service = ControlPlaneService(repository)
        controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
        before = await _snapshot(repository, workload_id)
        workload = before["workload"]
        assert workload.state is WorkloadState.PENDING
        lease = LeaseRecord(
            id=f"lease-{key}-{attempt}",
            workload_id=workload_id,
            cell_id=cell.id,
            org_id=workload.org_id,
            project_id=workload.project_id,
            resources=workload.spec.gang,
            generation=workload.generation,
            cell_generation=cell.generation,
            fencing_token=f"fence-{key}-{attempt}",
            hourly_cost=rate,
            reserved_cost=rate,
            resource_hours_reserved=ResourceHours(tpu=2),
            expires_at=utcnow() + timedelta(hours=2),
        )
        await service.create_lease(ADMIN, lease, idempotency_key=f"admit-{lease.id}")
        assert await controller.dispatch_outbox_once() >= 1
        dispatched = await _snapshot(repository, workload_id)
        assert dispatched["workload"].state is WorkloadState.DISPATCHING
        active = dispatched["leases"][lease.id]
        assert active.state is LeaseState.ACTIVE and active.run_started_at is not None
        commands = [
            command
            for command in dispatched["outbox"].values()
            if command.topic == "workload.dispatch" and command.generation == workload.generation
        ]
        assert len(commands) == 1 and commands[0].state is OutboxState.SUCCEEDED
        assert len(agent.submissions) == attempt + 1
        submitted_cell, request = agent.submissions[-1]
        assert submitted_cell == cell.id
        remote_id = request.spec["submission_id"]
        assert remote_id not in remote_ids
        assert dispatched["workload"].remote_id == remote_id
        assert dispatched["workload"].lease_id == lease.id
        assert workload.remote_id is None
        remote_ids.add(remote_id)
        agent.remote_status = ExecutionWorkloadStatus.RUNNING
        assert await controller.observe_workloads_once() == 1
        assert (await _snapshot(repository, workload_id))["workload"].state is WorkloadState.RUNNING

        # Only timestamps are backdated; dispatch and running transitions above
        # are performed by the real controller using the agent's observation.
        async with repository.transaction() as tx:
            active = await tx.get_lease(lease.id)
            started = utcnow() - timedelta(hours=1)
            await tx.put_lease(active.model_copy(update={"created_at": started, "run_started_at": started}))
        agent.remote_status = ExecutionWorkloadStatus.FAILED if attempt < failures else ExecutionWorkloadStatus.SUCCEEDED
        assert await controller.observe_workloads_once() == 1
        after = await _snapshot(repository, workload_id)
        settled = after["leases"][lease.id]
        assert settled.state is LeaseState.RELEASED and settled.settled_at is not None
        assert rate <= settled.settled_cost <= rate * 1.01
        assert 2 <= settled.accelerator_hours_settled["tpu"] <= 2.02
        released[lease.id] = settled
        assert after["leases"] == released
        budget = after["budget"]
        assert budget.revision == initial["budget"].revision + attempt + 1
        assert budget.spent == pytest.approx(17 + sum(item.settled_cost for item in released.values()))
        assert budget.accelerator_hours_spent["tpu"] == pytest.approx(
            11 + sum(item.accelerator_hours_settled["tpu"] for item in released.values())
        )
        assert after["cells"][cell.id] == initial["cells"][cell.id]
        dispatches = [command for command in after["outbox"].values() if command.topic == "workload.dispatch"]
        assert len(dispatches) == len(released)
        assert {command.generation for command in dispatches} == {item.generation for item in released.values()}
        for command in dispatches:
            historical = released[command.payload["lease_id"]]
            assert command.aggregate_id == command.payload["workload_id"] == workload_id
            assert command.payload["cell_id"] == historical.cell_id
            assert command.generation == historical.generation
            assert command.state is OutboxState.SUCCEEDED and command.attempts == 1
            assert command.claimed_by is None and command.claimed_until is None
        lease_events = [event for event in after["events"] if event.action == "lease.settled"]
        budget_events = [event for event in after["events"] if event.action == "budget.lease_settled"]
        assert len(lease_events) == len(budget_events) == len(released)
        assert {event.entity_id for event in lease_events} == set(released)
        assert {event.payload["lease_id"] for event in budget_events} == set(released)
        for event in lease_events:
            historical = released[event.entity_id]
            assert event.payload["settled_cost"] == historical.settled_cost
            assert event.payload["accelerator_hours_settled"] == historical.accelerator_hours_settled
        for event in budget_events:
            historical = released[event.payload["lease_id"]]
            assert event.payload["cost_increment"] == historical.settled_cost
            assert event.payload["accelerator_hours_increment"] == historical.accelerator_hours_settled
        current = after["workload"]
        assert current.failure_count == min(attempt + 1, failures)
        assert current.capacity_loss_count == 0 and current.preemption_count == 0
        if attempt < failures:
            assert current.state is WorkloadState.PENDING
            assert current.generation == workload.generation + 1
            assert current.attempt == workload.attempt + 1
            assert current.lease_id is None and current.lease_ids == ()
        else:
            assert current.state is WorkloadState.SUCCEEDED
        # Repeated polling of an already settled attempt cannot charge again.
        assert await controller.observe_workloads_once() == 0
        assert await _snapshot(repository, workload_id) == after


@pytest.mark.asyncio
@settings(max_examples=15, deadline=None)
@given(failures=st.integers(min_value=0, max_value=2), rate=st.integers(min_value=0, max_value=10))
async def test_generated_controller_retry_history(failures: int, rate: int) -> None:
    await _dispatch_history(InMemoryRepository(), failures, rate)


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["memory", "postgres"], indirect=True)
@pytest.mark.parametrize("failures,rate", [(0, 0), (1, 7), (2, 0), (2, 3)])
async def test_controller_retry_history_on_repository(repository, failures: int, rate: int) -> None:
    await _dispatch_history(repository, failures, rate)


@pytest.mark.asyncio
@pytest.mark.parametrize("failures,rate", [(0, 0), (1, 7), (2, 0), (2, 3)])
async def test_controller_retry_history_on_local_repository(tmp_path, failures: int, rate: int) -> None:
    # Exercise the shipped local backend, not PostgreSQL-specific outbox
    # claiming through the SQLAlchemy-on-SQLite service-contract fixture.
    from eyvan.server.persistence.local import LocalJsonRepository

    repository = LocalJsonRepository(tmp_path / "state.json")
    await repository.open()
    try:
        await _dispatch_history(repository, failures, rate)
    finally:
        await repository.close()
