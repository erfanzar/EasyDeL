# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Administrative reclaim: checkpoint, stop and requeue one running workload."""

from __future__ import annotations

import pytest

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseState,
    OutboxState,
    Principal,
    PriorityBand,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.models import AgentObservation, WorkloadObservation


def principal(role: Role) -> Principal:
    if role is Role.FLEET_ADMIN:
        return Principal(subject="admin", grants=(RoleGrant(role=role),))
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=role, org_id="acme", project_id="research"),),
    )


class FakeAgent:
    """Cell-agent double: checkpoint hooks succeed, ``stop`` makes a run STOPPED."""

    def __init__(self) -> None:
        self.submissions: list[tuple] = []
        self.remote_status = ExecutionWorkloadStatus.RUNNING
        self.remote_states: dict[str, ExecutionWorkloadStatus] = {}
        self.checkpoints: list[tuple] = []
        self.stops: list[tuple] = []
        self.retires: list[tuple] = []
        self.gang_submissions: list[tuple] = []
        self.gang_stops: list[tuple] = []
        self.gang_remote_status = ExecutionWorkloadStatus.RUNNING

    def apply_capacity(self, cell_id, request):
        return AgentObservation(cell_id=cell_id, state="healthy", generation=request.mutation.generation)

    def submit(self, cell_id, request):
        self.submissions.append((cell_id, request))
        return WorkloadObservation(remote_id=request.spec["submission_id"], status=ExecutionWorkloadStatus.SUBMITTED)

    def workload(self, cell_id, remote_id):
        del cell_id
        if remote_id.startswith("checkpoint-"):
            return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUCCEEDED)
        if remote_id in self.remote_states:
            return WorkloadObservation(remote_id=remote_id, status=self.remote_states[remote_id])
        return WorkloadObservation(remote_id=remote_id, status=self.remote_status)

    def checkpoint(self, cell_id, remote_id, request):
        self.checkpoints.append((cell_id, remote_id, request))
        return WorkloadObservation(
            remote_id=f"checkpoint-{remote_id}",
            status=ExecutionWorkloadStatus.CHECKPOINTING,
            checkpoint_uri=request.output_uri,
        )

    def stop(self, cell_id, remote_id, request):
        self.stops.append((cell_id, remote_id, request))
        self.remote_states[remote_id] = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    def retire(self, cell_id, remote_id, request):
        self.retires.append((cell_id, remote_id, request))
        self.remote_states[remote_id] = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPED)

    def submit_gang(self, coordinator_cell_id, request):
        self.gang_submissions.append((coordinator_cell_id, request))
        return WorkloadObservation(
            remote_id=f"gang-{request.spec['submission_id']}", status=ExecutionWorkloadStatus.SUBMITTED
        )

    def gang_workload(self, coordinator_cell_id, workload_id, remote_id):
        del coordinator_cell_id, workload_id
        return WorkloadObservation(remote_id=remote_id, status=self.gang_remote_status)

    def stop_gang(self, coordinator_cell_id, workload_id, remote_id, request):
        self.gang_stops.append((coordinator_cell_id, workload_id, remote_id, request))
        self.gang_remote_status = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)


class FailedCheckpointAgent(FakeAgent):
    def workload(self, cell_id, remote_id):
        if remote_id.startswith("checkpoint-"):
            return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.FAILED)
        return super().workload(cell_id, remote_id)


ADMIN = principal(Role.FLEET_ADMIN)
OPERATOR = principal(Role.PROJECT_OPERATOR)


def _spec(*, checkpointable: bool, cell_group_count: int = 1) -> WorkloadDefinition:
    extra = (
        {
            "checkpointable": True,
            "checkpoint_hook": "python checkpoint.py",
            "checkpoint_output_uri": "gs://checkpoints/batch",
        }
        if checkpointable
        else {}
    )
    return WorkloadDefinition(
        name="batch",
        execution=CommandExecution(command="python batch.py"),
        gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
        priority_band=PriorityBand.BATCH,
        cell_group_count=cell_group_count,
        # Multi-cell pod groups dispatch only through the native executor.
        required_executor="eyvan-native" if cell_group_count > 1 else None,
        **extra,
    )


async def _running(
    agent: FakeAgent, *, checkpointable: bool = True, cell_group_count: int = 1
) -> tuple[InMemoryRepository, ControlPlaneService, EyvanController, WorkloadRecord]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(checkpoint_grace_s=2, preemption_stop_grace_s=2),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
    )
    async with repository.transaction() as transaction:
        for index in range(cell_group_count):
            await transaction.put_cell(
                CellRecord(
                    id=f"cell-{index}",
                    provider="local",
                    region="local",
                    state=CellState.READY,
                    capacity=ResourceVector(tpu=8),
                    allocatable=ResourceVector(tpu=8),
                    endpoint="http://agent:8090",
                )
            )
    workload, _ = await service.submit_workload(
        OPERATOR,
        org_id="acme",
        project_id="research",
        spec=_spec(checkpointable=checkpointable, cell_group_count=cell_group_count),
        idempotency_key="batch",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    running = await _workload(repository, workload.id)
    assert running.state is WorkloadState.RUNNING
    assert running.remote_id is not None
    return repository, service, controller, running


async def _workload(repository: InMemoryRepository, workload_id: str) -> WorkloadRecord:
    async with repository.transaction() as transaction:
        workload = await transaction.get_workload(workload_id)
    assert workload is not None
    return workload


async def _leases(repository: InMemoryRepository, workload_id: str):
    async with repository.transaction() as transaction:
        return [lease for lease in await transaction.list_leases() if lease.workload_id == workload_id]


async def _reclaim_commands(repository: InMemoryRepository):
    async with repository.transaction() as transaction:
        return [command for command in await transaction.list_outbox() if command.topic == "workload.reclaim"]


async def _actions(repository: InMemoryRepository, workload_id: str) -> list[str]:
    async with repository.transaction() as transaction:
        events = await transaction.list_events(entity_type="workload", entity_id=workload_id, limit=1000)
    return [event.action for event in events]


@pytest.mark.asyncio
async def test_reclaim_checkpoints_then_stops_then_requeues_at_a_new_generation() -> None:
    agent = FakeAgent()
    repository, service, controller, running = await _running(agent)

    requested, replayed = await service.reclaim_workload(
        ADMIN, running.id, idempotency_key="reclaim-1", reason="make room for eval"
    )
    assert not replayed
    assert requested.state is WorkloadState.CHECKPOINTING
    assert {lease.state for lease in await _leases(repository, running.id)} == {LeaseState.REVOKING}
    assert len(await _reclaim_commands(repository)) == 1

    # An observation pass between enqueue and dispatch must not settle or recover anything.
    await controller.observe_workloads_once()
    in_flight = await _workload(repository, running.id)
    assert (in_flight.state, in_flight.generation) == (WorkloadState.CHECKPOINTING, running.generation)
    assert {lease.state for lease in await _leases(repository, running.id)} == {LeaseState.REVOKING}
    assert not agent.checkpoints and not agent.stops

    assert await controller.dispatch_outbox_once() == 1

    assert len(agent.checkpoints) == 1
    assert len(agent.stops) == 1
    assert agent.stops[0][1] == running.remote_id
    resumed = await _workload(repository, running.id)
    assert resumed.state is WorkloadState.PENDING
    assert resumed.generation == running.generation + 1
    assert resumed.attempt == running.attempt + 1
    assert resumed.checkpoint_uri == "gs://checkpoints/batch"
    assert resumed.lease_id is None and resumed.remote_id is None
    assert resumed.preemption_count == running.preemption_count + 1
    assert resumed.capacity_loss_count == running.capacity_loss_count
    assert resumed.scheduling_reason == "reclaimed by admin: make room for eval; waiting to resume"
    leases = await _leases(repository, running.id)
    assert leases and all(lease.state is LeaseState.RELEASED for lease in leases)
    assert all(lease.settled_at is not None for lease in leases)
    commands = await _reclaim_commands(repository)
    assert [command.state for command in commands] == [OutboxState.SUCCEEDED]
    actions = await _actions(repository, running.id)
    assert "workload.reclaim_requested" in actions
    assert "workload.reclaim_checkpointed" in actions
    assert "workload.reclaimed" in actions

    # Runnable again: the next scheduling pass places it.
    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [running.id]


@pytest.mark.asyncio
async def test_non_checkpointable_reclaim_skips_the_hook_but_still_stops_and_requeues() -> None:
    agent = FakeAgent()
    repository, service, controller, running = await _running(agent, checkpointable=False)

    requested, _ = await service.reclaim_workload(ADMIN, running.id, idempotency_key="reclaim", reason="rebalance")
    assert requested.state is WorkloadState.CANCELLING
    assert await controller.dispatch_outbox_once() == 1

    assert agent.checkpoints == []
    assert len(agent.stops) == 1
    resumed = await _workload(repository, running.id)
    assert (resumed.state, resumed.generation) == (WorkloadState.PENDING, running.generation + 1)
    assert resumed.checkpoint_uri is None
    assert resumed.preemption_count == running.preemption_count + 1
    assert all(lease.state is LeaseState.RELEASED for lease in await _leases(repository, running.id))
    assert "workload.reclaimed" in await _actions(repository, running.id)


@pytest.mark.asyncio
async def test_a_stopped_observation_during_reclaim_is_not_settled_as_cancelled() -> None:
    agent = FakeAgent()
    repository, service, controller, running = await _running(agent, checkpointable=False)
    await service.reclaim_workload(ADMIN, running.id, idempotency_key="reclaim", reason="rebalance")

    # The remote process is already gone: an ordinary CANCELLING workload would
    # be settled CANCELLED by this observation, ending the job.
    assert running.remote_id is not None
    agent.remote_states[running.remote_id] = ExecutionWorkloadStatus.STOPPED
    await controller.observe_workloads_once()
    in_flight = await _workload(repository, running.id)
    assert (in_flight.state, in_flight.generation) == (WorkloadState.CANCELLING, running.generation)
    assert all(lease.state is LeaseState.REVOKING for lease in await _leases(repository, running.id))

    assert await controller.dispatch_outbox_once() == 1
    resumed = await _workload(repository, running.id)
    assert resumed.state is WorkloadState.PENDING
    assert agent.stops == []  # already terminal; nothing left to stop


@pytest.mark.asyncio
async def test_reclaim_requires_fleet_administration() -> None:
    _repository, service, _controller, running = await _running(FakeAgent())
    with pytest.raises(AuthorizationError):
        await service.reclaim_workload(OPERATOR, running.id, idempotency_key="reclaim", reason="mine now")


@pytest.mark.asyncio
async def test_reclaim_refuses_a_workload_that_is_not_running() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    pending, _ = await service.submit_workload(
        OPERATOR, org_id="acme", project_id="research", spec=_spec(checkpointable=True), idempotency_key="p"
    )
    with pytest.raises(ConflictError):
        await service.reclaim_workload(ADMIN, pending.id, idempotency_key="reclaim", reason="nothing to take")
    assert await _reclaim_commands(repository) == []


@pytest.mark.asyncio
async def test_reclaim_replay_returns_the_workload_without_a_second_command() -> None:
    repository, service, _controller, running = await _running(FakeAgent())
    first, first_replayed = await service.reclaim_workload(ADMIN, running.id, idempotency_key="same", reason="r")
    second, second_replayed = await service.reclaim_workload(ADMIN, running.id, idempotency_key="same", reason="r")
    assert (first_replayed, second_replayed) == (False, True)
    assert second.id == first.id and second.state is first.state
    assert len(await _reclaim_commands(repository)) == 1
    with pytest.raises(ConflictError):
        await service.reclaim_workload(ADMIN, running.id, idempotency_key="same", reason="different")
    with pytest.raises(ConflictError):
        await service.reclaim_workload(ADMIN, running.id, idempotency_key="other", reason="r")


@pytest.mark.asyncio
async def test_a_workload_that_succeeded_before_the_stop_is_settled_succeeded() -> None:
    agent = FakeAgent()
    repository, service, controller, running = await _running(agent)
    await service.reclaim_workload(ADMIN, running.id, idempotency_key="reclaim", reason="late")
    agent.remote_status = ExecutionWorkloadStatus.SUCCEEDED

    # Observation leaves the reclaim-owned outcome to the command.
    await controller.observe_workloads_once()
    assert (await _workload(repository, running.id)).state is WorkloadState.CHECKPOINTING

    assert await controller.dispatch_outbox_once() == 1
    settled = await _workload(repository, running.id)
    assert settled.state is WorkloadState.SUCCEEDED
    assert settled.generation == running.generation
    assert settled.preemption_count == running.preemption_count
    assert agent.checkpoints == [] and agent.stops == []
    assert all(lease.state is LeaseState.RELEASED for lease in await _leases(repository, running.id))
    actions = await _actions(repository, running.id)
    assert "workload.succeeded_during_reclaim" in actions
    assert "workload.reclaimed" not in actions


@pytest.mark.asyncio
async def test_failed_checkpoint_aborts_the_reclaim_and_leaves_the_workload_running() -> None:
    agent = FailedCheckpointAgent()
    repository, service, controller, running = await _running(agent)
    await service.reclaim_workload(ADMIN, running.id, idempotency_key="reclaim", reason="try")

    assert await controller.dispatch_outbox_once() == 1

    restored = await _workload(repository, running.id)
    assert restored.state is WorkloadState.RUNNING
    assert (restored.generation, restored.attempt) == (running.generation, running.attempt)
    assert restored.remote_id == running.remote_id
    assert restored.preemption_count == running.preemption_count
    leases = await _leases(repository, running.id)
    assert [(lease.state, lease.revocation_reason) for lease in leases] == [(LeaseState.ACTIVE, None)]
    assert len(agent.checkpoints) == 1
    assert agent.stops == []
    assert [retire[1] for retire in agent.retires] == [f"checkpoint-{running.remote_id}"]
    [command] = await _reclaim_commands(repository)
    assert command.state is OutboxState.SUCCEEDED
    assert "checkpoint failed" in str(command.payload["aborted_reason"])
    actions = await _actions(repository, running.id)
    assert "workload.reclaim_aborted" in actions
    assert "workload.reclaimed" not in actions

    # Still observed as a healthy run, and reclaimable again.
    await controller.observe_workloads_once()
    assert (await _workload(repository, running.id)).state is WorkloadState.RUNNING
    again, replayed = await service.reclaim_workload(ADMIN, running.id, idempotency_key="again", reason="try")
    assert not replayed and again.state is WorkloadState.CHECKPOINTING


@pytest.mark.asyncio
async def test_a_user_cancel_takes_over_an_in_flight_reclaim() -> None:
    agent = FakeAgent()
    repository, service, controller, running = await _running(agent)
    await service.reclaim_workload(ADMIN, running.id, idempotency_key="reclaim", reason="r")
    cancelled, _ = await service.cancel_workload(OPERATOR, running.id, idempotency_key="cancel")
    assert cancelled.state is WorkloadState.CANCELLING

    await controller.dispatch_outbox_once()
    await controller.dispatch_outbox_once()

    final = await _workload(repository, running.id)
    assert final.state is WorkloadState.CANCELLED
    assert agent.checkpoints == []
    # Retired as stale (the controller finishes stale commands as SUCCEEDED),
    # never requeueing the workload the cancel now owns.
    [command] = await _reclaim_commands(repository)
    assert command.state is OutboxState.SUCCEEDED
    assert "workload.reclaimed" not in await _actions(repository, running.id)


@pytest.mark.asyncio
async def test_reclaim_of_a_multi_cell_group_stops_the_whole_gang_and_requeues() -> None:
    agent = FakeAgent()
    repository, service, controller, running = await _running(agent, checkpointable=True, cell_group_count=2)
    assert len(running.lease_ids) == 2

    requested, _ = await service.reclaim_workload(ADMIN, running.id, idempotency_key="reclaim", reason="gang")
    # Checkpoint hooks are not supported across cells: stop without one.
    assert requested.state is WorkloadState.CANCELLING
    await controller.observe_workloads_once()
    assert (await _workload(repository, running.id)).state is WorkloadState.CANCELLING

    assert await controller.dispatch_outbox_once() == 1
    assert agent.checkpoints == []
    assert len(agent.gang_stops) == 1
    resumed = await _workload(repository, running.id)
    assert (resumed.state, resumed.generation, resumed.lease_ids) == (
        WorkloadState.PENDING,
        running.generation + 1,
        (),
    )
    leases = await _leases(repository, running.id)
    assert len(leases) == 2 and all(lease.state is LeaseState.RELEASED for lease in leases)
