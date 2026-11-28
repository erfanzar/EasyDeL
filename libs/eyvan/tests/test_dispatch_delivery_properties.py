# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded submit-RPC retries through real outbox and reservation transactions."""

from contextlib import asynccontextmanager
from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_controller import FakeAgent
from test_group_lease_lifecycle import repository as _repository_fixture
from test_running_retry_properties import ADMIN, _seed, _snapshot

from eyvan.records import LeaseRecord, LeaseState, OutboxState, ResourceHours, WorkloadState, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.service import ControlPlaneService

repository = _repository_fixture


class TransactionProbe:
    """Delegate storage unchanged, observing the public transaction boundary."""

    def __init__(self, repository):
        self.repository = repository
        self.active_transactions = 0

    def __getattr__(self, name):
        return getattr(self.repository, name)

    @asynccontextmanager
    async def transaction(self):
        self.active_transactions += 1
        try:
            async with self.repository.transaction() as tx:
                yield tx
        finally:
            self.active_transactions -= 1


async def _delivery_history(repository, failures: int, rate: int, accepted_before_failure: bool = False) -> None:
    repository = TransactionProbe(repository)
    key = f"delivery-{failures}-{rate}"
    workload_id, cell = await _seed(repository, key, 0)
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
        workload = await tx.get_workload(workload_id)
    lease = LeaseRecord(
        id=f"lease-{key}",
        workload_id=workload_id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        resources=workload.spec.gang,
        generation=workload.generation,
        cell_generation=cell.generation,
        fencing_token=f"fence-{key}",
        hourly_cost=rate,
        reserved_cost=rate,
        resource_hours_reserved=ResourceHours(tpu=2),
        expires_at=utcnow() + timedelta(hours=2),
    )
    await ControlPlaneService(repository).create_lease(ADMIN, lease, idempotency_key=f"admit-{key}")
    initial = await _snapshot(repository, workload_id)
    (command_id,) = initial["outbox"]

    class FailingSubmit(FakeAgent):
        def __init__(self):
            super().__init__()
            self.requests = []
            self.accepted = {}

        def submit(self, cell_id, request):
            assert repository.active_transactions == 0
            self.requests.append((cell_id, request))
            key = request.mutation.idempotency_key
            failing = len(self.requests) <= failures
            if accepted_before_failure or not failing:
                if key not in self.accepted:
                    self.accepted[key] = super().submit(cell_id, request)
            if failing:
                # The controller sees the same exception whether admission never
                # happened or the worker admitted it and lost its response.
                raise RuntimeError("deliberate submit failure")
            return self.accepted[key]

    agent = FailingSubmit()
    fence = None
    for delivery in range(failures + 1):
        # Reconstruct the controller between deliveries; reservation and retry
        # state must come from storage rather than process-local counters.
        controller = EyvanController(repository, ControlPlaneService(repository), agent_factory=lambda _cell: agent)
        await controller.dispatch_outbox_once()
        after = await _snapshot(repository, workload_id)
        assert len(agent.requests) == delivery + 1
        assert len(after["outbox"]) == 1
        command = after["outbox"][command_id]
        assert command.attempts == delivery + 1
        assert command.claimed_by is None and command.claimed_until is None
        assert after["budget"] == initial["budget"]
        assert after["cells"] == initial["cells"]
        current = after["workload"]
        assert current.generation == workload.generation and current.attempt == workload.attempt
        assert current.failure_count == current.capacity_loss_count == current.preemption_count == 0
        assert current.lease_id == lease.id
        reserved = [event for event in after["events"] if event.action == "workload.dispatch_reserved"]
        failed = [event for event in after["events"] if event.action == "workload.dispatch_reservation_failed"]
        dispatched = [event for event in after["events"] if event.action == "workload.dispatched"]
        assert len(reserved) == delivery + 1
        assert not failed
        reservation = current.dispatch_reservation
        assert reservation is not None
        fence = fence or reservation.fence
        assert reservation.fence == fence
        assert reservation.outbox_id == command.id
        assert reservation.outbox_attempt == command.attempts
        assert reservation.generation == current.generation
        assert reservation.workload_attempt == current.attempt
        assert reservation.lease_ids == (lease.id,)
        assert command.payload["dispatch_reservation"] == reservation.model_dump(mode="json")
        assert not any(event.action == "lease.settled" for event in after["events"])
        if delivery < failures:
            assert command.state is OutboxState.FAILED
            assert "deliberate submit failure" in command.last_error
            assert current.state is WorkloadState.LEASED and current.remote_id is None
            assert after["leases"] == initial["leases"]
            assert not dispatched
            assert len(agent.submissions) == int(accepted_before_failure)
            # Advance only the retry clock; do not alter states/claims/attempts.
            async with repository.transaction() as tx:
                await tx.put_outbox(command.model_copy(update={"available_at": utcnow() - timedelta(seconds=1)}))
        else:
            assert command.state is OutboxState.SUCCEEDED and command.last_error is None
            assert current.state is WorkloadState.DISPATCHING
            assert current.remote_id == agent.requests[-1][1].spec["submission_id"]
            assert len(dispatched) == len(agent.submissions) == 1
            active = after["leases"][lease.id]
            assert active.state is LeaseState.ACTIVE and active.run_started_at is not None
            assert active.settled_at is None and active.settled_cost == 0
            assert await controller.dispatch_outbox_once() == 0
            assert await _snapshot(repository, workload_id) == after
    assert {identifier for identifier, _ in agent.requests} == {cell.id}
    assert len({request.spec["submission_id"] for _, request in agent.requests}) == 1
    assert len({request.mutation.idempotency_key for _, request in agent.requests}) == 1
    assert len(agent.submissions) == len(agent.accepted) == 1


@pytest.mark.asyncio
@settings(max_examples=15, deadline=None)
@given(
    failures=st.integers(min_value=0, max_value=3),
    rate=st.integers(min_value=0, max_value=10),
    accepted_before_failure=st.booleans(),
)
async def test_generated_dispatch_delivery_history(failures: int, rate: int, accepted_before_failure: bool) -> None:
    await _delivery_history(InMemoryRepository(), failures, rate, accepted_before_failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["memory", "postgres"], indirect=True)
@pytest.mark.parametrize("failures,rate", [(0, 0), (1, 7), (3, 0), (3, 5)])
@pytest.mark.parametrize("accepted_before_failure", [False, True])
async def test_dispatch_delivery_history_on_repository(
    repository, failures: int, rate: int, accepted_before_failure: bool
) -> None:
    await _delivery_history(repository, failures, rate, accepted_before_failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("failures,rate", [(0, 0), (1, 7), (3, 0), (3, 5)])
@pytest.mark.parametrize("accepted_before_failure", [False, True])
async def test_dispatch_delivery_history_on_local_repository(
    tmp_path, failures: int, rate: int, accepted_before_failure: bool
) -> None:
    repository = LocalJsonRepository(tmp_path / "state.json")
    await repository.open()
    try:
        await _delivery_history(repository, failures, rate, accepted_before_failure)
    finally:
        await repository.close()
