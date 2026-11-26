# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.artifacts.managed import (
    ArtifactAction,
    ArtifactTokenSigner,
    LocalManagedArtifactStore,
    callable_uri,
    result_uri,
)
from eyvan.artifacts.transfers import ManagedArtifactAccess
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, CapacityStatus, ExecutionWorkloadStatus
from eyvan.config import EyvanSettings
from eyvan.domain import InfeasibilityCode
from eyvan.records import (
    AutoscaleBlockReason,
    BudgetRecord,
    CallableExecution,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    PolicyRecord,
    Principal,
    PriorityBand,
    ResourceVector,
    Role,
    RoleGrant,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController, LeaderPhase, StaleOutboxCommand
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.client import CellAgentBusyError, CellAgentClient, CellAgentStaleMutationError
from eyvan.worker.agent.models import AgentObservation, CapacityRequest, WorkloadMutationRequest, WorkloadObservation


def principal(role: Role) -> Principal:
    if role is Role.FLEET_ADMIN:
        return Principal(subject="admin", grants=(RoleGrant(role=role),))
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=role, org_id="acme", project_id="research"),),
    )


def _payload_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    assert is_json_object(value)
    return value


def _preemption_victim_payloads(payload: JsonObject) -> list[JsonObject]:
    value = payload.get("victims")
    assert isinstance(value, list)
    assert all(is_json_object(item) for item in value)
    return [item for item in value if is_json_object(item)]


class FakeAgent:
    def __init__(self):
        self.capacity_requests = []
        self.submissions = []
        self.remote_status = ExecutionWorkloadStatus.RUNNING
        self.remote_message: str = ""
        self.checkpoints = []
        self.stops = []
        self.retires = []
        self.drains = []
        self.releases = []
        self.capacity_state = CapacityStatus.HEALTHY
        self.observed_tpu_vms: int | None = None
        self.runtime_process_count: int | None = None
        self.capacity_failure_code = None
        self.metadata: dict = {}
        self.remote_states: dict[str, ExecutionWorkloadStatus] = {}

    def apply_capacity(self, cell_id, request):
        self.capacity_requests.append((cell_id, request))
        return AgentObservation(
            cell_id=cell_id,
            state=self.capacity_state,
            generation=request.mutation.generation,
            observed_tpu_vms=self.observed_tpu_vms,
            runtime_process_count=self.runtime_process_count,
        )

    def reconcile(self, cell_id, request):
        return AgentObservation(
            cell_id=cell_id,
            state=self.capacity_state,
            generation=request.mutation.generation,
            observed_tpu_vms=self.observed_tpu_vms,
            runtime_process_count=self.runtime_process_count,
            failure_code=self.capacity_failure_code,
            metadata=self.metadata,
        )

    def submit(self, cell_id, request):
        self.submissions.append((cell_id, request))
        return WorkloadObservation(
            remote_id=request.spec["submission_id"],
            status=ExecutionWorkloadStatus.SUBMITTED,
        )

    def workload(self, cell_id, remote_id):
        del cell_id
        if remote_id.startswith("checkpoint-"):
            return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUCCEEDED)
        if remote_id in self.remote_states:
            return WorkloadObservation(remote_id=remote_id, status=self.remote_states[remote_id])
        return WorkloadObservation(remote_id=remote_id, status=self.remote_status, message=self.remote_message)

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

    def drain(self, cell_id, request):
        self.drains.append((cell_id, request))
        return AgentObservation(
            cell_id=cell_id,
            state=CapacityStatus.DRAINING,
            generation=request.mutation.generation,
        )

    def release(self, cell_id, request):
        self.releases.append((cell_id, request))
        return AgentObservation(
            cell_id=cell_id,
            state=CapacityStatus.RELEASED,
            generation=request.mutation.generation,
        )


class BlockingCapacityAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.apply_entered = threading.Event()
        self.allow_apply = threading.Event()

    def apply_capacity(self, cell_id, request):
        self.apply_entered.set()
        if not self.allow_apply.wait(timeout=5):
            raise TimeoutError("test did not release capacity apply")
        return super().apply_capacity(cell_id, request)


class BlockingProvisioningAgent(CellAgentClient):
    """Hold a typed capacity apply until the test releases the provider call."""

    def __init__(self) -> None:
        self.apply_entered = threading.Event()
        self.allow_apply = threading.Event()
        self.apply_calls = 0
        self.applied_cell_ids: list[str] = []

    def apply_capacity(self, cell_id: str, request: CapacityRequest) -> AgentObservation:
        self.apply_calls += 1
        self.applied_cell_ids.append(cell_id)
        self.apply_entered.set()
        if not self.allow_apply.wait(timeout=5):
            raise TimeoutError("test did not release capacity apply")
        return AgentObservation(
            cell_id=cell_id,
            state=CapacityStatus.WAITING,
            generation=request.mutation.generation,
        )

    def close(self) -> None:
        return None


class StaleSubmissionAgent(FakeAgent):
    """Reject one workload submission because the agent owns a newer fence."""

    def submit(self, cell_id, request):
        del cell_id, request
        raise CellAgentStaleMutationError("agent fence belongs to a newer workload generation")


class OneShotStaleCapacityLifecycleAgent(FakeAgent):
    """Reject each lifecycle mutation once, then accept its replacement fence."""

    def __init__(self) -> None:
        super().__init__()
        self.drain_generations: list[int] = []
        self.release_generations: list[int] = []

    def drain(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        self.drain_generations.append(request.mutation.generation)
        if len(self.drain_generations) == 1:
            raise CellAgentStaleMutationError("agent fence belongs to a newer drain generation")
        return super().drain(cell_id, request)

    def release(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        self.release_generations.append(request.mutation.generation)
        if len(self.release_generations) == 1:
            raise CellAgentStaleMutationError("agent fence belongs to a newer release generation")
        return super().release(cell_id, request)


class BlockingObservationClient(CellAgentClient):
    def __init__(self) -> None:
        self.reconcile_entered = threading.Event()
        self.allow_reconcile = threading.Event()
        self.reconcile_calls = 0

    def reconcile(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        self.reconcile_calls += 1
        self.reconcile_entered.set()
        if not self.allow_reconcile.wait(timeout=5):
            raise TimeoutError("test did not release capacity reconciliation")
        return AgentObservation(
            cell_id=cell_id,
            state=CapacityStatus.WAITING,
            generation=request.mutation.generation,
        )

    def close(self) -> None:
        return None


class ConcurrentBlockingObservationClient(CellAgentClient):
    """Hold multiple cell reconciliations until the test releases them together."""

    def __init__(self, expected_calls: int) -> None:
        self.expected_calls = expected_calls
        self.reconcile_entered = threading.Event()
        self.allow_reconcile = threading.Event()
        self.reconcile_calls = 0
        self._lock = threading.Lock()

    def reconcile(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        with self._lock:
            self.reconcile_calls += 1
            if self.reconcile_calls == self.expected_calls:
                self.reconcile_entered.set()
        if not self.allow_reconcile.wait(timeout=5):
            raise TimeoutError("test did not release capacity reconciliations")
        return AgentObservation(
            cell_id=cell_id,
            state=CapacityStatus.WAITING,
            generation=request.mutation.generation,
        )

    def close(self) -> None:
        return None


class GangFakeAgent(FakeAgent):
    """Cell-agent double that exposes the native multi-cell group contract."""

    def __init__(self):
        super().__init__()
        self.gang_submissions = []
        self.gang_status_calls = []
        self.gang_stops = []
        self.gang_remote_status = ExecutionWorkloadStatus.RUNNING

    def submit_gang(self, coordinator_cell_id, request):
        self.gang_submissions.append((coordinator_cell_id, request))
        return WorkloadObservation(
            remote_id=f"gang-{request.spec['submission_id']}",
            status=ExecutionWorkloadStatus.SUBMITTED,
        )

    def gang_workload(self, coordinator_cell_id, workload_id, remote_id):
        self.gang_status_calls.append((coordinator_cell_id, workload_id, remote_id))
        return WorkloadObservation(remote_id=remote_id, status=self.gang_remote_status)

    def stop_gang(self, coordinator_cell_id, workload_id, remote_id, request):
        self.gang_stops.append((coordinator_cell_id, workload_id, remote_id, request))
        self.gang_remote_status = ExecutionWorkloadStatus.STOPPED
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)


async def snapshot(repository: InMemoryRepository):
    async with repository.transaction() as transaction:
        return (
            await transaction.list_workloads(),
            await transaction.list_cells(),
            await transaction.list_leases(),
            await transaction.list_outbox(),
        )


async def _prepare_undispatched_leased_workload() -> tuple[
    InMemoryRepository,
    EyvanController,
    WorkloadRecord,
    LeaseRecord,
    OutboxRecord,
]:
    """Create one leased workload before its dispatch command is delivered."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
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
        assert leased is not None
        assert leased.lease_id is not None
        lease = await transaction.get_lease(leased.lease_id)
        assert lease is not None
        commands = await transaction.list_outbox_for_aggregates(
            "workload",
            {leased.id: leased.generation},
            topic="workload.dispatch",
        )
    assert len(commands) == 1
    return repository, controller, leased, lease, commands[0]


@pytest.mark.asyncio
async def test_controller_reconciles_capacity_dispatches_and_releases_workload() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(lease_ttl_s=120),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
    )
    admin = principal(Role.FLEET_ADMIN)
    operator = principal(Role.PROJECT_OPERATOR)
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        state=CellState.REGISTERING,
        capacity=ResourceVector(cpu=8, memory_gib=64, tpu=8),
        allocatable=ResourceVector(cpu=8, memory_gib=64, tpu=8),
        accelerator_type="v5p-8",
        capacity_class="spot",
        endpoint="http://agent:8090",
        runtime={"env": {"HF_HOME": "/var/lib/eyvan/workload/hf", "WANDB_MODE": "offline"}},
    )
    await service.upsert_cell(admin, cell, idempotency_key="cell-create")
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={
                "replicas": 1,
                "per_replica": {"cpu": 2, "memory_gib": 8, "tpu": 8},
            },
            capacity_class="spot",
            environment={"WANDB_MODE": "online"},
        ),
        idempotency_key="submit",
    )

    assert await controller.dispatch_outbox_once() == 1
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.READY
    assert cells[0].availability.success_evidence > cells[0].availability.failure_evidence
    assert commands[0].state is OutboxState.SUCCEEDED
    assert fake_agent.capacity_requests[0][1].desired["project"] == "gcp-project-a"
    assert fake_agent.capacity_requests[0][1].desired["runtime"]["env"]["HF_HOME"] == "/var/lib/eyvan/workload/hf"

    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [workload.id]
    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    assert workloads[0].state is WorkloadState.DISPATCHING
    assert leases[0].state is LeaseState.ACTIVE
    assert all(command.state is OutboxState.SUCCEEDED for command in commands)
    assert fake_agent.submissions[0][1].spec["entrypoint"] == "python train.py"
    environment = fake_agent.submissions[0][1].spec["environment"]
    assert environment["HF_HOME"] == "/var/lib/eyvan/workload/hf"
    assert environment["WANDB_MODE"] == "online"
    assert environment["EYVAN_WORKLOAD_ID"] == workload.id
    assert fake_agent.submissions[0][1].spec["gang"] == {
        "replicas": 1,
        "per_replica": {"cpu": 2.0, "memory_gib": 8.0, "gpu": 0, "tpu": 8, "custom": {}},
        "required_topology": {},
        "placement_strategy": "PACK",
    }

    assert await controller.observe_workloads_once() == 1
    async with repository.transaction() as transaction:
        active = (await transaction.list_leases())[0]
        await transaction.put_lease(active.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}))
    assert await controller.quarantine_expired_leases_once() == 1
    _, _, quarantined, _ = await snapshot(repository)
    assert quarantined[0].state is LeaseState.QUARANTINED
    blocked, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=workload.spec.model_copy(update={"name": "blocked"}),
        idempotency_key="submit-blocked",
    )
    blocked_result = await controller.reconcile_once()
    assert not blocked_result.placements
    assert [failure.workload_id for failure in blocked_result.infeasibilities] == [blocked.id]

    fake_agent.remote_status = ExecutionWorkloadStatus.SUCCEEDED
    assert await controller.observe_workloads_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert workloads[0].state is WorkloadState.SUCCEEDED
    assert leases[0].state is LeaseState.RELEASED

    await service.drain_cell(admin, "cell-a", idempotency_key="drain")
    await controller.dispatch_outbox_once()
    await service.delete_cell(admin, "cell-a", idempotency_key="delete")
    await controller.dispatch_outbox_once()
    _, cells, _, _ = await snapshot(repository)
    assert cells[0].state is CellState.DELETED
    assert fake_agent.drains and fake_agent.releases


@pytest.mark.asyncio
async def test_stale_agent_workload_fence_settles_dispatch_without_retrying() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = StaleSubmissionAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    operator = principal(Role.PROJECT_OPERATOR)
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=16, tpu=8),
        allocatable=ResourceVector(cpu=8, memory_gib=16, tpu=8),
        accelerator_type="v5p-8",
        endpoint="http://agent:8090",
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 2, "tpu": 8}},
        ),
        idempotency_key="submit",
    )

    scheduled = await controller.reconcile_once()
    assert [placement.workload_id for placement in scheduled.placements] == [workload.id]
    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    assert workloads[0].state is WorkloadState.FAILED
    assert workloads[0].remote_id is None
    assert workloads[0].failure_reason == (
        "cell-agent dispatch fence conflict: agent fence belongs to a newer workload generation"
    )
    assert leases[0].state is LeaseState.RELEASED
    assert leases[0].settlement_reason == "failed"
    dispatch = next(command for command in commands if command.topic == "workload.dispatch")
    assert dispatch.state is OutboxState.DISCARDED
    assert dispatch.last_error == "agent fence belongs to a newer workload generation"
    assert [command for command in commands if command.topic == "workload.dispatch"] == [dispatch]


@pytest.mark.asyncio
async def test_stale_capacity_lifecycle_mutations_are_refenced_before_retrying() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = OneShotStaleCapacityLifecycleAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    admin = principal(Role.FLEET_ADMIN)
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=16, tpu=8),
        allocatable=ResourceVector(cpu=8, memory_gib=16, tpu=8),
        accelerator_type="v5p-8",
        endpoint="http://agent:8090",
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)

    drained, _ = await service.drain_cell(admin, cell.id, idempotency_key="drain")
    assert drained.generation == cell.generation
    assert await controller.dispatch_outbox_once() == 1

    _, cells, _, commands = await snapshot(repository)
    refenced_drain = cells[0]
    assert (refenced_drain.state, refenced_drain.generation) == (CellState.DRAINING, cell.generation + 1)
    drain_commands = [command for command in commands if command.topic == "capacity.drain"]
    assert len(drain_commands) == 2
    stale_drain = next(command for command in drain_commands if command.generation == cell.generation)
    fresh_drain = next(command for command in drain_commands if command.generation == refenced_drain.generation)
    assert stale_drain.state is OutboxState.DISCARDED
    assert stale_drain.last_error == "agent fence belongs to a newer drain generation"
    assert fresh_drain.state is OutboxState.PENDING

    assert await controller.dispatch_outbox_once() == 1
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.DRAINING
    assert next(command for command in commands if command.id == fresh_drain.id).state is OutboxState.SUCCEEDED
    assert agent.drain_generations == [cell.generation, refenced_drain.generation]

    deleting, _ = await service.delete_cell(admin, cell.id, idempotency_key="delete")
    assert deleting.generation == refenced_drain.generation + 1
    assert await controller.dispatch_outbox_once() == 1

    _, cells, _, commands = await snapshot(repository)
    refenced_release = cells[0]
    assert (refenced_release.state, refenced_release.generation) == (CellState.DELETING, deleting.generation + 1)
    release_commands = [command for command in commands if command.topic == "capacity.release"]
    assert len(release_commands) == 2
    stale_release = next(command for command in release_commands if command.generation == deleting.generation)
    fresh_release = next(command for command in release_commands if command.generation == refenced_release.generation)
    assert stale_release.state is OutboxState.DISCARDED
    assert stale_release.last_error == "agent fence belongs to a newer release generation"
    assert fresh_release.state is OutboxState.PENDING

    assert await controller.dispatch_outbox_once() == 1
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.DELETED
    assert next(command for command in commands if command.id == fresh_release.id).state is OutboxState.SUCCEEDED
    assert agent.release_generations == [deleting.generation, refenced_release.generation]


@pytest.mark.asyncio
async def test_controller_dispatches_and_settles_one_atomic_multi_worker_tpu_cell() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    agent = GangFakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(lease_ttl_s=120),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type]
    )
    admin = principal(Role.FLEET_ADMIN)
    operator = principal(Role.PROJECT_OPERATOR)
    pool = CapacityPoolRecord(
        id="research/v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=0,
        max_cells=4,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool-v5p-16")
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            pool.make_cell("cell-a", worker_count=2).model_copy(update={"state": CellState.READY})
        )

    workload, replayed = await service.submit_pool_callable(
        operator,
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        execution=CallableExecution(
            artifact_uri="gs://trusted/train.pkl",
            sha256="a" * 64,
            python="3.12",
        ),
        idempotency_key="two-v5p-16-pods",
    )
    assert not replayed
    assert workload.spec.cell_group_count == 1
    assert workload.spec.worker_count == 2
    assert workload.spec.worker_resource_scope is WorkerResourceScope.PER_WORKER
    assert workload.spec.gang.per_replica.tpu == 8
    assert workload.spec.resources_per_cell.tpu == 16

    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [workload.id]

    queued, _, leases, commands = await snapshot(repository)
    current = next(item for item in queued if item.id == workload.id)
    assert current.state is WorkloadState.LEASED
    assert current.lease_ids == ()
    assert current.lease_id == leases[0].id
    assert len(leases) == 1
    assert leases[0].cell_id == "cell-a"
    assert leases[0].group_id is None
    assert leases[0].group_index == 0
    assert leases[0].resources.per_replica.tpu == 8
    assert leases[0].resource_hours_reserved.tpu == 16
    dispatch = next(command for command in commands if command.topic == "workload.dispatch")
    assert dispatch.payload["lease_id"] == current.lease_id
    assert dispatch.payload["cell_id"] == "cell-a"
    assert "lease_ids" not in dispatch.payload
    assert "cell_ids" not in dispatch.payload

    assert await controller.dispatch_outbox_once() == 1
    assert len(agent.submissions) == 1
    submitted_cell_id, request = agent.submissions[0]
    assert submitted_cell_id == "cell-a"
    assert agent.gang_submissions == []
    assert request.spec["gang"]["replicas"] == 1
    assert request.spec["gang"]["per_replica"]["tpu"] == 8

    dispatched, _, active_leases, _ = await snapshot(repository)
    active = next(item for item in dispatched if item.id == workload.id)
    assert active.state is WorkloadState.DISPATCHING
    assert all(lease.state is LeaseState.ACTIVE for lease in active_leases)

    assert await controller.observe_workloads_once() == 1
    running, _, running_leases, _ = await snapshot(repository)
    assert next(item for item in running if item.id == workload.id).state is WorkloadState.RUNNING
    assert all(lease.state is LeaseState.ACTIVE for lease in running_leases)

    agent.remote_status = ExecutionWorkloadStatus.SUCCEEDED
    assert await controller.observe_workloads_once() == 1
    settled, _, settled_leases, _ = await snapshot(repository)
    assert next(item for item in settled if item.id == workload.id).state is WorkloadState.SUCCEEDED
    assert all(lease.state is LeaseState.RELEASED for lease in settled_leases)
    assert agent.gang_status_calls == []


@pytest.mark.asyncio
async def test_controller_cancels_a_dispatched_multi_worker_cell_with_one_native_stop() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    agent = GangFakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(lease_ttl_s=120),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type]
    )
    admin = principal(Role.FLEET_ADMIN)
    operator = principal(Role.PROJECT_OPERATOR)
    pool = CapacityPoolRecord(
        id="research/v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=0,
        max_cells=4,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="cancel-pool-v5p-16")
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            pool.make_cell("cell-a", worker_count=2).model_copy(update={"state": CellState.READY})
        )

    workload, _ = await service.submit_pool_callable(
        operator,
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        execution=CallableExecution(
            artifact_uri="gs://trusted/train.pkl",
            sha256="a" * 64,
            python="3.12",
        ),
        idempotency_key="cancel-two-v5p-16-pods",
    )
    assert len((await controller.reconcile_once()).placements) == 1
    assert await controller.dispatch_outbox_once() == 1

    dispatched, _, active_leases, _ = await snapshot(repository)
    active = next(item for item in dispatched if item.id == workload.id)
    assert active.state is WorkloadState.DISPATCHING
    assert active.lease_ids == ()
    assert active.lease_id == active_leases[0].id
    assert len(active_leases) == 1
    assert all(lease.state is LeaseState.ACTIVE for lease in active_leases)

    cancelling, replayed = await service.cancel_workload(
        operator,
        workload.id,
        idempotency_key="cancel-dispatched-two-pod-group",
    )
    assert not replayed
    assert cancelling.state is WorkloadState.CANCELLING
    assert await controller.dispatch_outbox_once() == 1

    assert len(agent.stops) == 1
    stopped_cell_id, stopped_remote_id, request = agent.stops[0]
    assert (stopped_cell_id, stopped_remote_id) == ("cell-a", active.remote_id)
    assert request.mutation.fencing_token == active_leases[0].fencing_token
    assert agent.gang_stops == []

    cancelled, _, released_leases, _ = await snapshot(repository)
    assert next(item for item in cancelled if item.id == workload.id).state is WorkloadState.CANCELLED
    assert all(lease.state is LeaseState.RELEASED for lease in released_leases)


@pytest.mark.asyncio
async def test_controller_resolves_managed_artifacts_at_dispatch_without_putting_tokens_in_command(tmp_path) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    fake_agent = FakeAgent()
    signer = ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=3600, max_ttl_seconds=3600)
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path / "artifacts"),
        signer=signer,
        public_url="http://127.0.0.1:8080",
        token_ttl_seconds=3600,
    )
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(dev_auth_enabled=True),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
        artifact_access=access,
    )
    cell = CellRecord(
        id="cell-managed",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
    digest = "a" * 64
    requested_result = result_uri("call-managed")
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="managed-callable",
            execution=CallableExecution(
                artifact_uri=callable_uri(digest),
                sha256=digest,
                python="3.13",
                result_uri=requested_result,
            ),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
        ),
        idempotency_key="managed-callable",
    )

    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [workload.id]
    assert await controller.dispatch_outbox_once() == 1

    request = fake_agent.submissions[0][1]
    workload_execution = workload.spec.execution
    assert isinstance(workload_execution, CallableExecution)
    managed_result = workload_execution.result_uri
    assert managed_result == result_uri(f"{workload.id}-g1")
    entrypoint = request.spec["entrypoint"]
    environment = request.spec["environment"]
    assert "--uri-env EYVAN_MANAGED_CALLABLE_URL --uri-token-env EYVAN_MANAGED_CALLABLE_TOKEN" in entrypoint
    assert "--result-uri-env EYVAN_MANAGED_RESULT_URL --result-token-env EYVAN_MANAGED_RESULT_TOKEN" in entrypoint
    assert "token=" not in entrypoint
    assert environment["EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP"] == "1"
    callable_query = parse_qs(urlsplit(environment["EYVAN_MANAGED_CALLABLE_URL"]).query)
    assert "token" not in callable_query
    signer.verify(
        environment["EYVAN_MANAGED_CALLABLE_TOKEN"],
        ArtifactAction.READ,
        callable_uri(digest),
        org_id="acme",
        project_id="research",
        workload_id=workload.id,
    )
    result_query = parse_qs(urlsplit(environment["EYVAN_MANAGED_RESULT_URL"]).query)
    assert "token" not in result_query
    signer.verify(
        environment["EYVAN_MANAGED_RESULT_TOKEN"],
        ArtifactAction.WRITE,
        managed_result,
        org_id="acme",
        project_id="research",
        workload_id=workload.id,
    )


@pytest.mark.asyncio
async def test_drain_wins_over_an_inflight_capacity_apply_writeback() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = BlockingCapacityAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    admin = principal(Role.FLEET_ADMIN)
    cell, _ = await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )

    apply = asyncio.create_task(controller.dispatch_outbox_once())
    assert await asyncio.to_thread(fake_agent.apply_entered.wait, 2)
    drained, _ = await service.drain_cell(admin, cell.id, idempotency_key="drain")
    assert drained.state is CellState.DRAINING
    fake_agent.allow_apply.set()
    assert await apply == 1

    _, cells, _, _ = await snapshot(repository)
    assert cells[0].state is CellState.DRAINING
    assert await controller.dispatch_outbox_once() == 1
    _, cells, _, _ = await snapshot(repository)
    assert cells[0].state is CellState.DRAINING
    assert len(fake_agent.drains) == 1


@pytest.mark.asyncio
async def test_capacity_apply_persists_requesting_while_agent_call_is_in_flight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = BlockingProvisioningAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    caplog.set_level(logging.INFO, logger="eyvan.server.controller")
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )

    delivery = asyncio.create_task(controller.dispatch_outbox_once())
    assert await asyncio.to_thread(agent.apply_entered.wait, 2)
    assert not delivery.done()

    _, cells, _, commands = await snapshot(repository)
    assert len(cells) == 1
    assert len(commands) == 1
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].provider_phase is CapacityStatus.REQUESTING
    assert cells[0].provider_phase_started_at is not None
    requesting_started_at = cells[0].provider_phase_started_at
    assert commands[0].state is OutboxState.PROCESSING
    assert commands[0].claimed_by == controller.worker_id
    assert any(
        "capacity lifecycle event=requesting" in record.getMessage()
        and "provider_state=REQUESTING" in record.getMessage()
        for record in caplog.records
    )

    agent.allow_apply.set()
    assert await delivery == 1

    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].provider_phase is CapacityStatus.WAITING
    assert cells[0].provider_phase_started_at is not None
    assert cells[0].provider_phase_started_at > requesting_started_at
    assert commands[0].state is OutboxState.SUCCEEDED
    assert agent.apply_calls == 1


@pytest.mark.asyncio
async def test_long_capacity_apply_renews_claim_before_another_controller_can_reclaim() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    settings = EyvanSettings(outbox_claim_ttl_s=1)
    agent = BlockingProvisioningAgent()
    controller_a = EyvanController(repository, service, settings=settings, agent_factory=lambda _cell: agent)
    controller_b = EyvanController(repository, service, settings=settings, agent_factory=lambda _cell: agent)
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )

    delivery = asyncio.create_task(controller_a.dispatch_outbox_once())
    assert await asyncio.to_thread(agent.apply_entered.wait, 2)
    _, _, _, initial_commands = await snapshot(repository)
    assert len(initial_commands) == 1
    initial_claimed_until = initial_commands[0].claimed_until
    assert initial_claimed_until is not None

    try:
        await asyncio.sleep(1.2)
        _, cells, _, commands = await snapshot(repository)
        assert len(cells) == 1
        assert len(commands) == 1
        renewed_command = commands[0]
        assert renewed_command.state is OutboxState.PROCESSING
        assert renewed_command.claimed_by == controller_a.worker_id
        assert renewed_command.claimed_until is not None
        assert renewed_command.claimed_until > initial_claimed_until
        assert renewed_command.claimed_until > utcnow()
        assert await controller_b.dispatch_outbox_once() == 0
    finally:
        agent.allow_apply.set()
        delivery_result = await delivery

    assert delivery_result == 1
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].provider_phase is CapacityStatus.WAITING
    assert commands[0].state is OutboxState.SUCCEEDED
    assert agent.apply_calls == 1


@pytest.mark.asyncio
async def test_outbox_batch_dispatch_never_starts_a_later_command_after_another_controller_claims_it() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    # Concurrency pinned to one: this test is about claim fencing and hand-off
    # after a TTL expires, and it was written when delivery happened to be serial.
    # With the default concurrency both cells are claimed and delivered at once,
    # which is the point of that change and not the subject of this test.
    settings = EyvanSettings(outbox_claim_ttl_s=1, outbox_dispatch_concurrency=1)
    first_agent = BlockingProvisioningAgent()
    second_agent = BlockingProvisioningAgent()
    controller_a = EyvanController(repository, service, settings=settings, agent_factory=lambda _cell: first_agent)
    controller_b = EyvanController(repository, service, settings=settings, agent_factory=lambda _cell: second_agent)
    admin = principal(Role.FLEET_ADMIN)
    for cell_id in ("cell-a", "cell-b"):
        await service.upsert_cell(
            admin,
            CellRecord(
                id=cell_id,
                provider="gcp",
                provider_project="gcp-project-a",
                region="us-central2",
                zone="us-central2-b",
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                accelerator_type="v5p-8",
                endpoint="http://agent:8090",
            ),
            idempotency_key=f"{cell_id}-create",
        )

    delivery_a = asyncio.create_task(controller_a.dispatch_outbox_once(limit=2))
    delivery_b: asyncio.Task[int] | None = None
    try:
        assert await asyncio.to_thread(first_agent.apply_entered.wait, 2)
        _, _, _, commands = await snapshot(repository)
        commands_by_cell = {command.aggregate_id: command for command in commands}
        assert commands_by_cell["cell-a"].state is OutboxState.PROCESSING
        assert commands_by_cell["cell-a"].claimed_by == controller_a.worker_id
        assert commands_by_cell["cell-b"].state is OutboxState.PENDING
        assert commands_by_cell["cell-b"].claimed_by is None

        await asyncio.sleep(1.2)
        delivery_b = asyncio.create_task(controller_b.dispatch_outbox_once())
        assert await asyncio.to_thread(second_agent.apply_entered.wait, 2)
        assert second_agent.applied_cell_ids == ["cell-b"]

        first_agent.allow_apply.set()
        assert await delivery_a == 1
        assert first_agent.applied_cell_ids == ["cell-a"]
    finally:
        first_agent.allow_apply.set()
        second_agent.allow_apply.set()
        await delivery_a
        if delivery_b is not None:
            await delivery_b

    _, _, _, commands = await snapshot(repository)
    assert all(command.state is OutboxState.SUCCEEDED for command in commands)


@pytest.mark.asyncio
async def test_fleet_down_is_not_blocked_by_inflight_observation_and_prevents_a_second_reconcile() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = BlockingObservationClient()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    admin = principal(Role.FLEET_ADMIN)
    pool = CapacityPoolRecord(
        id="research-v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=1,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool-create")
    async with repository.transaction() as transaction:
        await transaction.put_cell(pool.make_cell("cell-a").model_copy(update={"state": CellState.READY}))

    observation = asyncio.create_task(controller.observe_cells_once())
    assert await asyncio.to_thread(agent.reconcile_entered.wait, 2)
    down = asyncio.create_task(controller.disable_fleet(admin, "research", idempotency_key="fleet-down"))
    pools, replayed = await asyncio.wait_for(down, timeout=1)
    assert not replayed
    assert all(not current.enabled for current in pools)
    assert agent.reconcile_calls == 1

    assert await controller.observe_cells_once() == 0
    assert agent.reconcile_calls == 1
    agent.allow_reconcile.set()
    assert await observation == 1


@pytest.mark.asyncio
async def test_observation_rechecks_capacity_apply_started_after_its_snapshot() -> None:
    class SnapshotPausedController(EyvanController):
        def __init__(self, repository: InMemoryRepository, service: ControlPlaneService, agent: CellAgentClient) -> None:
            super().__init__(repository, service, agent_factory=lambda _cell: agent)
            self.snapshot_complete = asyncio.Event()
            self.continue_observation = asyncio.Event()

        async def _observe_cell_once(self, cell: CellRecord, now: datetime) -> bool:
            self.snapshot_complete.set()
            await self.continue_observation.wait()
            return await super()._observe_cell_once(cell, now)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = BlockingObservationClient()
    agent.allow_reconcile.set()
    controller = SnapshotPausedController(repository, service, agent)
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v5p-8",
        endpoint="http://agent:8090",
        provider_phase=CapacityStatus.WAITING,
        provider_phase_started_at=utcnow(),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)

    observation = asyncio.create_task(controller.observe_cells_once())
    await asyncio.wait_for(controller.snapshot_complete.wait(), timeout=1)
    marked_at = utcnow()
    async with repository.transaction() as transaction:
        current = await transaction.get_cell(cell.id)
        assert current is not None
        await transaction.put_cell(
            current.model_copy(
                update={
                    "provider_phase": CapacityStatus.REQUESTING,
                    "provider_phase_started_at": marked_at,
                    "updated_at": marked_at,
                }
            )
        )
        await transaction.put_outbox(
            OutboxRecord(
                id="capacity-apply-cell-a",
                topic="capacity.apply",
                aggregate_type="cell",
                aggregate_id=cell.id,
                idempotency_key="capacity:cell-a:1",
                generation=cell.generation,
                payload={"provider_rpc_started": True},
                state=OutboxState.PROCESSING,
                claimed_by="other-controller",
                claimed_until=marked_at + timedelta(seconds=60),
            )
        )
    controller.continue_observation.set()

    assert await observation == 0
    assert agent.reconcile_calls == 0


@pytest.mark.asyncio
async def test_same_fleet_observations_remain_concurrent_while_fleet_lifecycle_is_open() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = ConcurrentBlockingObservationClient(expected_calls=2)
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(cell_reconcile_concurrency=2),
        agent_factory=lambda _cell: agent,
    )
    admin = principal(Role.FLEET_ADMIN)
    pool = CapacityPoolRecord(
        id="research-v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=2,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool-create")
    async with repository.transaction() as transaction:
        await transaction.put_cell(pool.make_cell("cell-a").model_copy(update={"state": CellState.READY}))
        await transaction.put_cell(pool.make_cell("cell-b").model_copy(update={"state": CellState.READY}))

    observation = asyncio.create_task(controller.observe_cells_once())
    assert await asyncio.to_thread(agent.reconcile_entered.wait, 2)
    assert agent.reconcile_calls == 2
    agent.allow_reconcile.set()
    assert await observation == 2


@pytest.mark.asyncio
async def test_failed_outbox_delivery_is_retried_with_backoff() -> None:
    class FailingAgent(FakeAgent):
        def apply_capacity(self, cell_id, request):
            raise RuntimeError(f"provider unavailable for {cell_id}/{request.mutation.generation}")

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, agent_factory=lambda _cell: FailingAgent())  # type: ignore[arg-type]
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    await controller.dispatch_outbox_once()

    _, cells, _, commands = await snapshot(repository)
    assert commands[0].state is OutboxState.FAILED
    assert commands[0].attempts == 1
    assert "provider unavailable" in (commands[0].last_error or "")
    assert commands[0].failure_code is None
    assert commands[0].claimed_by is None
    assert cells[0].state is CellState.UNAVAILABLE
    assert cells[0].provider_phase is CapacityStatus.PROVIDER_ERROR
    assert cells[0].provider_phase_started_at is not None
    assert cells[0].availability.cooldown_until is not None


@pytest.mark.asyncio
async def test_superseded_capacity_apply_keeps_the_original_provider_diagnostic() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            502,
            json={"detail": "GCP queued-resource admission denied", "code": "queue_admission_denied"},
        )

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = CellAgentClient(
        "http://agent",
        client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)),
    )
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    assert await controller.dispatch_outbox_once() == 1
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.UNAVAILABLE
    assert commands[0].state is OutboxState.DISCARDED
    diagnostic = commands[0].last_error
    assert diagnostic == "GCP queued-resource admission denied"
    assert commands[0].failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED

    assert await controller.dispatch_outbox_once() == 0
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.UNAVAILABLE
    assert commands[0].state is OutboxState.DISCARDED
    assert commands[0].last_error == diagnostic
    assert commands[0].failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
    assert calls == 1
    agent.close()


@pytest.mark.asyncio
async def test_queue_admission_denial_terminally_blocks_owned_autoscale_workload() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={"detail": "GCP queued-resource admission denied", "code": "queue_admission_denied"},
        )

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    agent = CellAgentClient(
        "http://agent",
        client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)),
    )
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    pool = CapacityPoolRecord(
        id="research/v6e-16",
        fleet_name="research",
        node_type="v6e-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-central1",
        zone="us-central1-c",
        capacity=ResourceVector(cpu=64, tpu=16),
        allocatable=ResourceVector(cpu=64, tpu=16),
        accelerator_type="v6e-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=0,
        max_cells=4,
    )
    await service.apply_fleet(principal(Role.FLEET_ADMIN), "research", (pool,), idempotency_key="fleet-v1")
    workload, _ = await service.submit_pool_callable(
        principal(Role.PROJECT_OPERATOR),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v6e-16",
        pods=2,
        execution=CallableExecution(
            artifact_uri="gs://trusted/train.pkl",
            sha256="a" * 64,
            python="3.12",
        ),
        idempotency_key="v6e-16-queue-denied",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    assert await controller.dispatch_outbox_once() == 1

    workloads, cells, _, commands = await snapshot(repository)
    current = next(candidate for candidate in workloads if candidate.id == workload.id)
    assert current.state is WorkloadState.FAILED
    assert current.autoscale_block_reason is AutoscaleBlockReason.QUEUE_ADMISSION_DENIED
    assert current.queue_admission_denial is not None
    assert current.queue_admission_denial.provider_project == "gcp-project-a"
    assert current.queue_admission_denial.zone == "us-central1-c"
    assert current.queue_admission_denial.accelerator_type == "v6e-16"
    assert current.failure_reason == "GCP queued-resource admission denied"
    assert current.scheduling_reason is None
    assert cells[0].provider_failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
    assert cells[0].unavailable_reason == "queue_admission_denied"
    apply = next(command for command in commands if command.topic == "capacity.apply")
    assert apply.state is OutboxState.DISCARDED
    assert apply.failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED

    result = await controller.reconcile_once()
    assert result.infeasibilities == ()
    scaled = await controller.reconcile_capacity_pools_once()
    assert scaled["launched"] == 0

    await service.apply_fleet(
        principal(Role.FLEET_ADMIN),
        "research",
        (pool.model_copy(update={"provider_project": "gcp-project-b", "region": "us-east5", "zone": "us-east5-a"}),),
        idempotency_key="fleet-v2",
    )
    workloads, _, _, _ = await snapshot(repository)
    reloaded = next(candidate for candidate in workloads if candidate.id == workload.id)
    assert reloaded.queue_admission_denial is not None
    assert reloaded.queue_admission_denial.provider_project == "gcp-project-a"
    assert reloaded.queue_admission_denial.zone == "us-central1-c"
    assert reloaded.queue_admission_denial.accelerator_type == "v6e-16"
    agent.close()


@pytest.mark.asyncio
async def test_cell_agent_stale_capacity_mutation_queues_replacement_without_degrading_capacity_belief() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"detail": "agent fence belongs to a newer generation", "code": "stale_mutation"},
        )

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = CellAgentClient(
        "http://agent",
        client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)),
    )
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    cell, _ = await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    assert await controller.dispatch_outbox_once() == 1
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].id == cell.id
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].generation == cell.generation + 1
    assert cells[0].provider_phase is None
    assert cells[0].unavailable_reason is None
    assert cells[0].availability.consecutive_failures == cell.availability.consecutive_failures
    assert cells[0].availability.cooldown_until == cell.availability.cooldown_until
    assert cells[0].availability.interruptions > cell.availability.interruptions
    discarded = next(command for command in commands if command.generation == cell.generation)
    replacement = next(command for command in commands if command.generation == cell.generation + 1)
    assert discarded.state is OutboxState.DISCARDED
    assert discarded.last_error == "agent fence belongs to a newer generation"
    assert replacement.topic == "capacity.apply"
    assert replacement.state is OutboxState.PENDING
    agent.close()


@pytest.mark.asyncio
async def test_autoscaler_repairs_a_discarded_capacity_apply_for_the_current_pool_template() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    admin = principal(Role.FLEET_ADMIN)
    pool = CapacityPoolRecord(
        id="research-v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        min_cells=1,
        max_cells=1,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool-create")
    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    replacement = pool.model_copy(update={"labels": {"template": "replacement"}})
    await service.apply_fleet(admin, pool.fleet_name, [replacement], idempotency_key="fleet-apply")

    assert await controller.dispatch_outbox_once() == 1
    _, _, _, commands = await snapshot(repository)
    stale_apply = next(command for command in commands if command.topic == "capacity.apply")
    assert stale_apply.state is OutboxState.DISCARDED
    assert agent.capacity_requests == []

    await controller.reconcile_capacity_pools_once()

    _, cells, _, commands = await snapshot(repository)
    current_template_cells = [
        cell
        for cell in cells
        if cell.pool_id == replacement.id
        and cell.state is CellState.REGISTERING
        and cell.provider_phase is None
        and cell.labels.get("template") == "replacement"
    ]
    assert current_template_cells
    for cell in current_template_cells:
        assert cell.generation > stale_apply.generation
        assert any(
            command.topic == "capacity.apply"
            and command.aggregate_id == cell.id
            and command.generation == cell.generation
            and command.state is OutboxState.PENDING
            for command in commands
        )


@pytest.mark.asyncio
async def test_capacity_apply_discards_its_writeback_when_another_controller_updates_the_pool() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = BlockingCapacityAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    admin = principal(Role.FLEET_ADMIN)
    pool = CapacityPoolRecord(
        id="research-v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=1,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool-create")
    cell, _ = await service.upsert_cell(admin, pool.make_cell("cell-a"), idempotency_key="cell-create")

    dispatch = asyncio.create_task(controller.dispatch_outbox_once())
    assert await asyncio.to_thread(agent.apply_entered.wait, 2)
    replacement = pool.model_copy(update={"labels": {"template": "replacement"}})
    await service.apply_fleet(admin, pool.fleet_name, [replacement], idempotency_key="replica-b-fleet-apply")
    agent.allow_apply.set()
    assert await dispatch == 1

    _, cells, _, commands = await snapshot(repository)
    assert len(cells) == 1
    assert cells[0].id == cell.id
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].provider_phase is CapacityStatus.REQUESTING
    assert commands[0].state is OutboxState.DISCARDED
    assert commands[0].payload["provider_rpc_started"] is True
    assert len(agent.capacity_requests) == 1


@pytest.mark.asyncio
async def test_reclaimed_pre_rpc_capacity_apply_clears_requesting_before_template_replacement() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = FakeAgent()
    controller_a = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    controller_b = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    admin = principal(Role.FLEET_ADMIN)
    pool = CapacityPoolRecord(
        id="research-v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(cpu=32, tpu=8),
        allocatable=ResourceVector(cpu=32, tpu=8),
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        min_cells=1,
        max_cells=1,
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool-create")
    cell, _ = await service.upsert_cell(admin, pool.make_cell("cell-a"), idempotency_key="cell-create")

    claimed = await repository.claim_outbox(worker_id=controller_a.worker_id, limit=1, lease_seconds=60)
    assert len(claimed) == 1
    await controller_a._mark_capacity_apply_requesting(claimed[0], cell)

    _, marked_cells, _, marked_commands = await snapshot(repository)
    assert marked_cells[0].provider_phase is CapacityStatus.REQUESTING
    assert marked_commands[0].payload["provider_rpc_started"] is False

    replacement = pool.model_copy(update={"labels": {"template": "replacement"}})
    await controller_a.apply_fleet(admin, pool.fleet_name, [replacement], idempotency_key="fleet-apply")
    async with repository.transaction() as transaction:
        current_pool = await transaction.get_capacity_pool(pool.id)
        current_cell = await transaction.get_cell(cell.id)
        assert current_pool is not None
        assert current_cell is not None
        assert current_pool.labels == replacement.labels
        assert current_cell.labels != replacement.labels
        interrupted = await transaction.get_outbox(claimed[0].id)
        assert interrupted is not None
        await transaction.put_outbox(
            interrupted.model_copy(
                update={
                    "state": OutboxState.PENDING,
                    "claimed_by": None,
                    "claimed_until": None,
                    "available_at": utcnow(),
                    "updated_at": utcnow(),
                }
            )
        )

    assert await controller_b.dispatch_outbox_once() == 1
    _, recovered_cells, _, recovered_commands = await snapshot(repository)
    assert recovered_cells[0].generation == cell.generation
    assert recovered_cells[0].provider_phase is None
    assert recovered_commands[0].state is OutboxState.DISCARDED
    assert agent.capacity_requests == []

    await controller_b.reconcile_capacity_pools_once()

    _, requeued_cells, _, requeued_commands = await snapshot(repository)
    assert len(requeued_cells) == 1
    requeued = requeued_cells[0]
    assert requeued.generation == cell.generation + 1
    assert requeued.labels == replacement.make_cell(requeued.id, worker_count=requeued.worker_count).labels
    assert requeued.provider_phase is None
    assert any(
        command.topic == "capacity.apply"
        and command.aggregate_id == requeued.id
        and command.generation == requeued.generation
        and command.state is OutboxState.PENDING
        for command in requeued_commands
    )


@pytest.mark.asyncio
async def test_busy_fleet_lease_retries_without_degrading_capacity() -> None:
    class BusyAgent(FakeAgent):
        def apply_capacity(self, cell_id, request):
            raise CellAgentBusyError("Eyvan fleet mutation lease is busy")

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, agent_factory=lambda _cell: BusyAgent())  # type: ignore[arg-type]
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    await controller.dispatch_outbox_once()

    _, cells, _, commands = await snapshot(repository)
    assert commands[0].state is OutboxState.FAILED
    assert commands[0].attempts == 0
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].availability.cooldown_until is None


@pytest.mark.asyncio
async def test_capacity_recreation_requeues_dispatched_workload_with_retry_budget() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, agent_factory=lambda _cell: FakeAgent())  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.REGISTERING,
        capacity=ResourceVector(tpu=4),
        allocatable=ResourceVector(tpu=4),
        accelerator_type="v5p-8",
        capacity_class="spot",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"tpu": 4}},
            capacity_class="spot",
            max_retries=1,
        ),
        state=WorkloadState.DISPATCHING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        cell_generation=cell.generation,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    assert await controller.observe_workloads_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert (workloads[0].state, workloads[0].attempt, workloads[0].generation) == (
        WorkloadState.PENDING,
        2,
        2,
    )
    assert workloads[0].lease_id is None
    assert workloads[0].remote_id is None
    assert workloads[0].scheduling_reason == "execution cell capacity was recreated; retrying capacity loss (1 of 1)"
    assert (leases[0].state, leases[0].settlement_reason) == (LeaseState.RELEASED, "preempted")


@pytest.mark.asyncio
async def test_native_worker_readiness_loss_does_not_recover_a_running_workload() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        provider_phase=CapacityStatus.HEALTHY,
        capacity=ResourceVector(tpu=4),
        allocatable=ResourceVector(tpu=4),
        accelerator_type="v5p-8",
        capacity_class="spot",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"tpu": 4}},
            capacity_class="spot",
            max_retries=1,
        ),
        state=WorkloadState.RUNNING,
        lease_id="lease-a",
        remote_id="native-a1",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        cell_generation=cell.generation,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    fake_agent.capacity_state = CapacityStatus.EXECUTION_CONNECTING
    assert await controller.observe_cells_once() == 1
    assert await controller.observe_workloads_once() == 1

    workloads, cells, leases, _ = await snapshot(repository)
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].provider_phase is CapacityStatus.EXECUTION_CONNECTING
    assert (workloads[0].state, workloads[0].attempt, workloads[0].generation) == (WorkloadState.RUNNING, 1, 1)
    assert leases[0].state is LeaseState.ACTIVE


@pytest.mark.asyncio
async def test_indeterminate_workload_observation_fails_closed_without_recovering_lease() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.remote_status = ExecutionWorkloadStatus.INDETERMINATE
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=4),
        allocatable=ResourceVector(cpu=4),
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.RUNNING,
        lease_id="lease-a",
        remote_id="remote-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.QUARANTINED,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    assert await controller.observe_workloads_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert workloads[0].state is WorkloadState.RUNNING
    assert leases[0].state is LeaseState.QUARANTINED
    assert fake_agent.stops == []


@pytest.mark.asyncio
async def test_remote_program_failure_remains_terminal_with_capacity_retry_budget() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.remote_status = ExecutionWorkloadStatus.FAILED
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        capacity=ResourceVector(tpu=4),
        allocatable=ResourceVector(tpu=4),
        accelerator_type="v5p-8",
        capacity_class="spot",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"tpu": 4}},
            capacity_class="spot",
            max_retries=3,
        ),
        state=WorkloadState.RUNNING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        cell_generation=cell.generation,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    assert await controller.observe_workloads_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert (workloads[0].state, workloads[0].attempt) == (WorkloadState.FAILED, 1)
    assert leases[0].state is LeaseState.RELEASED
    assert len(fake_agent.stops) == 1
    stopped_cell, stopped_remote_id, stop_request = fake_agent.stops[0]
    assert (stopped_cell, stopped_remote_id) == ("cell-a", "ray-job-a")
    assert stop_request.mutation.generation == workload.generation
    assert stop_request.mutation.fencing_token == lease.fencing_token


@pytest.mark.asyncio
async def test_a_failed_workload_records_why_it_failed() -> None:
    """Persist the executor's failure reason before the cell and its local diagnostics can be released."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.remote_status = ExecutionWorkloadStatus.FAILED
    fake_agent.remote_message = "ModuleNotFoundError: No module named 'eyvan'"
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        capacity=ResourceVector(tpu=4),
        allocatable=ResourceVector(tpu=4),
        accelerator_type="v5p-8",
        capacity_class="spot",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"tpu": 4}},
            capacity_class="spot",
        ),
        state=WorkloadState.RUNNING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        cell_generation=cell.generation,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    assert await controller.observe_workloads_once() == 1

    workloads, _, _leases, _ = await snapshot(repository)
    assert workloads[0].state is WorkloadState.FAILED
    assert workloads[0].failure_reason is not None
    assert "No module named" in workloads[0].failure_reason


@pytest.mark.asyncio
async def test_a_failure_with_no_diagnostic_still_says_so() -> None:
    """An empty reason and "the backend told us nothing" must not look alike."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.remote_status = ExecutionWorkloadStatus.FAILED
    fake_agent.remote_message = ""
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        capacity=ResourceVector(tpu=4),
        allocatable=ResourceVector(tpu=4),
        accelerator_type="v5p-8",
        capacity_class="spot",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"tpu": 4}},
            capacity_class="spot",
        ),
        state=WorkloadState.RUNNING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        cell_generation=cell.generation,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    assert await controller.observe_workloads_once() == 1

    workloads, _, _leases, _ = await snapshot(repository)
    assert workloads[0].failure_reason
    assert "without a diagnostic" in workloads[0].failure_reason


@pytest.mark.asyncio
async def test_definitive_halted_cell_fails_non_checkpointable_workload() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="project-a",
        region="us-central2",
        zone="us-central2-b",
        state=CellState.READY,
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v6e-8",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.RUNNING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    fake_agent.capacity_state = CapacityStatus.HALTED_QUOTA
    assert await controller.observe_cells_once() == 1
    assert await controller.observe_workloads_once() == 1

    workloads, cells, leases, _ = await snapshot(repository)
    assert cells[0].unavailable_reason == "HALTED_QUOTA"
    assert workloads[0].state is WorkloadState.FAILED
    assert leases[0].state is LeaseState.RELEASED


@pytest.mark.asyncio
async def test_workload_observer_does_not_settle_quarantined_preemption() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.remote_status = ExecutionWorkloadStatus.STOPPED
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8),
        allocatable=ResourceVector(cpu=8),
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.CHECKPOINTING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.QUARANTINED,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        revocation_reason="preemption",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    assert await controller.observe_workloads_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert workloads[0].state is WorkloadState.CHECKPOINTING
    assert leases[0].state is LeaseState.QUARANTINED


@pytest.mark.asyncio
async def test_stale_running_observation_cannot_undo_same_generation_preemption() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingAgent(FakeAgent):
        def workload(self, cell_id, remote_id):
            started.set()
            assert release.wait(timeout=2)
            return super().workload(cell_id, remote_id)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = BlockingAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=1, memory_gib=1),
        allocatable=ResourceVector(cpu=1, memory_gib=1),
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.DISPATCHING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    observation = asyncio.create_task(controller.observe_workloads_once())
    assert await asyncio.to_thread(started.wait, 2)
    async with repository.transaction() as transaction:
        await transaction.put_workload(workload.model_copy(update={"state": WorkloadState.CHECKPOINTING}))
        await transaction.put_lease(
            lease.model_copy(update={"state": LeaseState.REVOKING, "revocation_reason": "preemption"})
        )
    release.set()
    assert await observation == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert workloads[0].state is WorkloadState.CHECKPOINTING
    assert leases[0].state is LeaseState.REVOKING


@pytest.mark.asyncio
async def test_user_cancel_takes_ownership_of_a_revoking_lease() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=1, memory_gib=1),
        allocatable=ResourceVector(cpu=1, memory_gib=1),
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.CHECKPOINTING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.REVOKING,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        revocation_reason="preemption",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    cancelled, _ = await service.cancel_workload(
        principal(Role.PROJECT_OPERATOR),
        workload.id,
        idempotency_key="cancel",
    )
    assert cancelled.state is WorkloadState.CANCELLING
    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert workloads[0].state is WorkloadState.CANCELLED
    assert leases[0].state is LeaseState.RELEASED
    assert leases[0].revocation_reason is None


@pytest.mark.asyncio
async def test_draining_cell_keeps_running_workloads_observable_and_controllable() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=1, memory_gib=1),
        allocatable=ResourceVector(cpu=1, memory_gib=1),
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.RUNNING,
        lease_id="lease-a",
        remote_id="ray-job-a",
        created_by="alice",
    )
    lease = LeaseRecord(
        id="lease-a",
        workload_id=workload.id,
        cell_id=cell.id,
        cell_generation=cell.generation,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)

    drained, _ = await service.drain_cell(
        principal(Role.FLEET_ADMIN),
        cell.id,
        idempotency_key="drain",
    )
    assert (drained.state, drained.generation) == (CellState.DRAINING, cell.generation)
    assert await controller.dispatch_outbox_once() == 1
    assert await controller.observe_workloads_once() == 1
    await service.cancel_workload(
        principal(Role.PROJECT_OPERATOR),
        workload.id,
        idempotency_key="cancel",
    )
    assert await controller.dispatch_outbox_once() == 1

    workloads, cells, leases, _ = await snapshot(repository)
    assert workloads[0].state is WorkloadState.CANCELLED
    assert cells[0].state is CellState.DRAINING
    assert leases[0].state is LeaseState.RELEASED


@pytest.mark.asyncio
async def test_stale_cancel_command_cannot_stop_a_new_attempt() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    cell = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="project-a",
        region="us-central2",
        zone="us-central2-b",
        state=CellState.READY,
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v6e-8",
        endpoint="http://agent:8090",
    )
    workload = WorkloadRecord(
        id="workload-a",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.RUNNING,
        attempt=2,
        generation=2,
        lease_id="lease-new",
        remote_id="new-attempt",
        created_by="alice",
    )
    lease_template = {
        "workload_id": workload.id,
        "cell_id": cell.id,
        "org_id": workload.org_id,
        "project_id": workload.project_id,
        "resources": {"replicas": 1, "per_replica": {"tpu": 8}},
        "expires_at": datetime.now(UTC) + timedelta(minutes=5),
    }
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_workload(workload)
        await transaction.put_lease(
            LeaseRecord(
                id="lease-old",
                state=LeaseState.RELEASED,
                generation=1,
                fencing_token="old-token",
                **lease_template,
            )
        )
        await transaction.put_lease(
            LeaseRecord(
                id="lease-new",
                state=LeaseState.ACTIVE,
                generation=2,
                fencing_token="new-token",
                **lease_template,
            )
        )
        await transaction.put_outbox(
            OutboxRecord(
                id="cancel-old",
                topic="workload.cancel",
                aggregate_type="workload",
                aggregate_id=workload.id,
                idempotency_key="cancel:workload-a:1",
                generation=1,
                payload={
                    "workload_id": workload.id,
                    "lease_id": "lease-old",
                    "cell_id": cell.id,
                    "remote_id": "old-attempt",
                },
            )
        )

    assert await controller.dispatch_outbox_once() == 1

    current_workloads, _, _, commands = await snapshot(repository)
    assert current_workloads[0].remote_id == "new-attempt"
    assert fake_agent.stops == []
    assert commands[0].state is OutboxState.SUCCEEDED


@pytest.mark.asyncio
async def test_manual_reconcile_is_rejected_on_a_follower() -> None:
    class Follower:
        async def acquire(self) -> bool:
            return False

        async def release(self) -> None:
            return None

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, leader=Follower())

    assert await controller.reconcile_if_leader() is None


@pytest.mark.asyncio
async def test_lost_leader_does_not_replace_a_pending_workloads_newer_reason() -> None:
    """A stale scheduling snapshot must not overwrite a successor's status."""

    class Follower:
        async def acquire(self) -> bool:
            return False

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, leader=Follower())
    workload = WorkloadRecord(
        id="wl-pending",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="pending", execution=CommandExecution(command="echo pending")),
        state=WorkloadState.PENDING,
        created_by="alice",
        scheduling_reason="newer leader reason",
    )
    async with repository.transaction() as transaction:
        await transaction.put_workload(workload)

    result = await controller.reconcile_once()

    assert result.infeasibilities
    async with repository.transaction() as transaction:
        current = await transaction.get_workload(workload.id)
    assert current is not None
    assert current.scheduling_reason == "newer leader reason"


@pytest.mark.asyncio
async def test_the_daemon_stops_mid_cycle_when_it_loses_leadership() -> None:
    """A full leader cycle spans many provider round trips.

    This replica can lose the advisory lock partway through, and continuing
    after that lets two leaders mint autoscaler cell ids from the same
    ``len(managed) + 1`` sequence and collide. The re-check has to be on the
    production scheduler loop's cycle -- putting it only on the manual
    ``/reconcile`` endpoint leaves the daemon acquiring once and running every
    phase unchecked.
    """

    class LosesLeadershipAfter:
        def __init__(self, grants: int) -> None:
            self.remaining = grants
            self.acquisitions = 0

        async def acquire(self) -> bool:
            self.acquisitions += 1
            self.remaining -= 1
            return self.remaining >= 0

        async def release(self) -> None:
            return None

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    # One grant for the loop's own gate, two for the first two phases.
    leader = LosesLeadershipAfter(grants=3)
    controller = EyvanController(repository, service, leader=leader)
    ran: list[str] = []

    async def tracked(name):
        async def phase():
            ran.append(name)

        return phase

    for name in (
        "reconcile_unmanaged_capacity_recoveries_once",
        "reconcile_capacity_pools_once",
        "quarantine_expired_leases_once",
        "observe_workloads_once",
        "reconcile_once",
    ):
        setattr(controller, name, await tracked(name))

    assert await leader.acquire() is True  # the loop's own gate
    # Every interval-gated phase is parked far in the future, so only the
    # every-cycle phases are eligible and the leadership check is what stops them.
    parked = {phase.name: 1e9 for phase in controller._leader_phases()}
    await controller._run_leader_phases(now=0.0, due=parked)

    # The two phases granted leadership ran; everything after the loss did not.
    assert ran == ["reconcile_unmanaged_capacity_recoveries_once", "reconcile_capacity_pools_once"]


@pytest.mark.asyncio
async def test_a_failing_retention_prune_cannot_pin_the_leader_loop() -> None:
    """The deadline must advance before the call, not after a success.

    Advancing only on success makes a persistently failing prune re-run on
    every single tick, so one broken table turns the leader loop into a hot
    retry against the database.
    """

    class Leader:
        async def acquire(self) -> bool:
            return True

        async def release(self) -> None:
            return None

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, leader=Leader())
    attempts = 0

    async def failing_prune():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("retention is wedged")

    controller.prune_retained_state_once = failing_prune

    due = await controller._run_leader_phases(
        now=100.0,
        due={"observe-cells": 1e9, "retention": 0.0},
    )

    assert attempts == 1
    # The next attempt is a full interval away despite the failure.
    assert due["retention"] == 100.0 + controller.settings.retention_interval_s


@pytest.mark.asyncio
async def test_observation_waits_for_initial_capacity_apply() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.capacity_state = CapacityStatus.ABSENT
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v6e-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    assert await controller.observe_cells_once() == 0
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].provider_phase is None
    assert commands[0].state is OutboxState.PENDING

    fake_agent.capacity_state = CapacityStatus.WAITING
    assert await controller.dispatch_outbox_once() == 1
    _, cells, _, commands = await snapshot(repository)
    assert cells[0].state is CellState.REGISTERING
    assert cells[0].provider_phase is CapacityStatus.WAITING
    assert commands[0].state is OutboxState.SUCCEEDED


@pytest.mark.asyncio
async def test_controller_advances_waiting_capacity_to_ready() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.capacity_state = CapacityStatus.WAITING
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v6e-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )
    await controller.dispatch_outbox_once()
    _, waiting_cells, _, _ = await snapshot(repository)
    assert waiting_cells[0].availability.success_evidence == 1
    fake_agent.capacity_state = CapacityStatus.HEALTHY
    fake_agent.observed_tpu_vms = 8
    fake_agent.runtime_process_count = 8

    assert await controller.observe_cells_once() == 1

    _, cells, _, _ = await snapshot(repository)
    assert cells[0].state is CellState.READY
    assert cells[0].observed_tpu_vms == 8
    assert cells[0].runtime_process_count == 8
    assert cells[0].last_heartbeat_at is not None

    # A later loss yields no live endpoint set. The rank width remains the last
    # exact provider observation rather than collapsing and renumbering logs.
    fake_agent.capacity_state = CapacityStatus.UNAVAILABLE
    fake_agent.observed_tpu_vms = None
    fake_agent.runtime_process_count = None
    assert await controller.observe_cells_once() == 1
    _, lost_cells, _, _ = await snapshot(repository)
    assert lost_cells[0].runtime_process_count == 8


@pytest.mark.asyncio
async def test_metadata_capacity_update_does_not_reprovision_the_cell() -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowAgent(FakeAgent):
        def apply_capacity(self, cell_id, request):
            started.set()
            assert release.wait(timeout=2)
            return super().apply_capacity(cell_id, request)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = SlowAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    original = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v6e-8",
        endpoint="http://agent:8090",
    )
    await service.upsert_cell(principal(Role.FLEET_ADMIN), original, idempotency_key="cell-v1")

    dispatch = asyncio.create_task(controller.dispatch_outbox_once())
    assert await asyncio.to_thread(started.wait, 2)
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        original.model_copy(update={"capacity": ResourceVector(tpu=32), "allocatable": ResourceVector(tpu=32)}),
        idempotency_key="cell-v2",
    )
    release.set()
    await dispatch

    _, cells, _, commands = await snapshot(repository)
    assert (cells[0].generation, cells[0].state, cells[0].capacity.tpu) == (1, CellState.READY, 32)
    assert [(command.generation, command.state) for command in commands] == [(1, OutboxState.SUCCEEDED)]


@pytest.mark.asyncio
async def test_metadata_update_preserves_provider_observed_runtime_process_count() -> None:
    """A declarative update cannot reset active log rank width to legacy one."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    fake_agent.runtime_process_count = 2
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    original = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v5p-16",
        endpoint="http://agent:8090",
    )
    await service.upsert_cell(principal(Role.FLEET_ADMIN), original, idempotency_key="cell-v1")
    await controller.dispatch_outbox_once()

    updated, _ = await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        original.model_copy(update={"labels": {"team": "research"}}),
        idempotency_key="cell-v2",
    )

    assert updated.runtime_process_count == 2
    assert updated.generation == 1


@pytest.mark.asyncio
async def test_agent_endpoint_change_advances_incarnation_and_reapplies_route() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    routed_endpoints: list[str | None] = []
    fake_agent = FakeAgent()

    def agent_factory(cell):
        routed_endpoints.append(cell.endpoint)
        return fake_agent

    controller = EyvanController(repository, service, agent_factory=agent_factory)  # type: ignore[arg-type]
    original = CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v6e-8",
        endpoint="http://old-agent:8090",
    )
    await service.upsert_cell(principal(Role.FLEET_ADMIN), original, idempotency_key="cell-v1")
    await controller.dispatch_outbox_once()

    updated, _ = await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        original.model_copy(update={"endpoint": "http://new-agent:8090"}),
        idempotency_key="cell-v2",
    )

    assert (updated.generation, updated.state) == (2, CellState.REGISTERING)
    await controller.dispatch_outbox_once()
    _, cells, _, commands = await snapshot(repository)
    assert (cells[0].generation, cells[0].state) == (2, CellState.READY)
    assert routed_endpoints == ["http://old-agent:8090", "http://new-agent:8090"]
    assert [command.state for command in commands] == [OutboxState.SUCCEEDED, OutboxState.SUCCEEDED]
    assert fake_agent.capacity_requests[-1][1].desired["agent_endpoint"] == "http://new-agent:8090"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_retries", "expected_state"),
    [(1, WorkloadState.PENDING), (0, WorkloadState.FAILED)],
)
@pytest.mark.parametrize(
    ("provider_state", "expected_cell_state"),
    [
        (CapacityStatus.WAITING, CellState.REGISTERING),
        (CapacityStatus.NEEDS_RECREATE, CellState.UNAVAILABLE),
    ],
)
async def test_lost_spot_cell_recovers_from_durable_checkpoint_only_within_retry_budget(
    max_retries: int,
    expected_state: WorkloadState,
    provider_state: CapacityStatus,
    expected_cell_state: CellState,
) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v6e-8",
            capacity_class="spot",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )
    await controller.dispatch_outbox_once()
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="resumable",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            capacity_class="spot",
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/resumable",
            max_retries=max_retries,
        ),
        idempotency_key="workload",
    )
    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [workload.id]
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    async with repository.transaction() as transaction:
        running = await transaction.get_workload(workload.id)
        assert running is not None
        await transaction.put_workload(running.model_copy(update={"checkpoint_uri": "gs://checkpoints/resumable/42"}))

    fake_agent.capacity_state = provider_state
    assert await controller.observe_cells_once() == 1
    assert await controller.observe_workloads_once() == 1

    workloads, cells, leases, _ = await snapshot(repository)
    assert cells[0].state is expected_cell_state
    assert workloads[0].state is expected_state
    if expected_state is WorkloadState.PENDING:
        assert (workloads[0].attempt, workloads[0].generation) == (2, 2)
        assert workloads[0].checkpoint_uri == "gs://checkpoints/resumable/42"
    else:
        assert (workloads[0].attempt, workloads[0].generation) == (1, 1)
        assert workloads[0].failure_reason
    assert leases[0].state is LeaseState.RELEASED
    assert leases[0].settlement_reason == ("preempted" if expected_state is WorkloadState.PENDING else "failed")


@pytest.mark.asyncio
async def test_higher_priority_workload_checkpoint_preempts_whole_gang() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(checkpoint_grace_s=2),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
    )
    admin = principal(Role.FLEET_ADMIN)
    operator = principal(Role.PROJECT_OPERATOR)
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )
    await controller.dispatch_outbox_once()
    low, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
        ),
        idempotency_key="low",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    high, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )

    result = await controller.reconcile_once()
    assert result.preemptions[0].workload_id == high.id
    async with repository.transaction() as transaction:
        revoking = (await transaction.list_leases())[0]
        assert revoking.state is LeaseState.REVOKING
        await transaction.put_lease(revoking.model_copy(update={"state": LeaseState.QUARANTINED}))
    await controller.dispatch_outbox_once()

    workloads, _, leases, commands = await snapshot(repository)
    low_record = next(item for item in workloads if item.id == low.id)
    assert (low_record.state, low_record.attempt, low_record.checkpoint_uri) == (
        WorkloadState.PENDING,
        2,
        "gs://checkpoints/batch",
    )
    assert leases[0].state is LeaseState.RELEASED
    assert fake_agent.checkpoints and fake_agent.stops
    assert all(command.state is OutboxState.SUCCEEDED for command in commands)

    resumed_result = await controller.reconcile_once()
    assert resumed_result.placements[0].workload_id == high.id


@pytest.mark.asyncio
async def test_cancelled_preemption_target_restores_victim_before_external_effects() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    admin = principal(Role.FLEET_ADMIN)
    operator = principal(Role.PROJECT_OPERATOR)
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="gcp-project-a",
            region="us-central2",
            zone="us-central2-b",
            capacity=ResourceVector(tpu=8),
            allocatable=ResourceVector(tpu=8),
            accelerator_type="v5p-8",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )
    await controller.dispatch_outbox_once()
    low, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
        ),
        idempotency_key="low",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    high, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )
    assert (await controller.reconcile_once()).preemptions
    await service.cancel_workload(operator, high.id, idempotency_key="cancel-high")

    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    low_record = next(item for item in workloads if item.id == low.id)
    high_record = next(item for item in workloads if item.id == high.id)
    assert low_record.state is WorkloadState.RUNNING
    assert high_record.state is WorkloadState.CANCELLED
    assert leases[0].state is LeaseState.ACTIVE
    assert leases[0].revocation_reason is None
    assert not fake_agent.checkpoints and not fake_agent.stops
    assert commands[-1].state is OutboxState.SUCCEEDED


@pytest.mark.asyncio
async def test_preemption_rollback_rearms_an_invalidated_unstarted_dispatch() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    operator = principal(Role.PROJECT_OPERATOR)
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                endpoint="http://agent:8090",
            )
        )
    low, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
        ),
        idempotency_key="low",
    )
    assert (await controller.reconcile_once()).placements
    high, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )
    assert (await controller.reconcile_once()).preemptions
    await service.cancel_workload(operator, high.id, idempotency_key="cancel-high")

    assert await controller.dispatch_outbox_once() == 2
    assert not fake_agent.submissions
    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    restored = next(item for item in workloads if item.id == low.id)
    assert restored.state is WorkloadState.DISPATCHING
    assert leases[0].state is LeaseState.ACTIVE
    assert len(fake_agent.submissions) == 1


@pytest.mark.asyncio
async def test_stale_preemption_can_form_a_new_plan_after_policy_revision() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    operator = principal(Role.PROJECT_OPERATOR)
    admin = principal(Role.PROJECT_ADMIN)
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                endpoint="http://agent:8090",
            )
        )
    low, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
        ),
        idempotency_key="low",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    high, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )
    assert (await controller.reconcile_once()).preemptions
    await service.put_policy(
        admin,
        PolicyRecord(org_id="acme", project_id="research", preemption_enabled=False, updated_by="admin"),
        idempotency_key="disable-preemption",
    )
    await controller.dispatch_outbox_once()
    workloads, _, leases, commands = await snapshot(repository)
    assert next(item for item in workloads if item.id == low.id).state is WorkloadState.RUNNING
    target = next(item for item in workloads if item.id == high.id)
    assert target.preemption_id is None
    assert leases[0].state is LeaseState.ACTIVE
    first_plan = next(command for command in commands if command.topic == "workload.preempt")
    assert first_plan.state is OutboxState.SUCCEEDED

    await service.put_policy(
        admin,
        PolicyRecord(org_id="acme", project_id="research", preemption_enabled=True, updated_by="admin"),
        idempotency_key="enable-preemption",
    )
    assert (await controller.reconcile_once()).preemptions
    _, _, _, commands = await snapshot(repository)
    plan_ids = [command.id for command in commands if command.topic == "workload.preempt"]
    assert len(plan_ids) == 2
    assert len(set(plan_ids)) == 2


@pytest.mark.asyncio
async def test_failed_preemption_checkpoint_aborts_before_stopping_victim() -> None:
    class FailedCheckpointAgent(FakeAgent):
        def workload(self, cell_id, remote_id):
            if remote_id.startswith("checkpoint-"):
                return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.FAILED)
            return super().workload(cell_id, remote_id)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FailedCheckpointAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    operator = principal(Role.PROJECT_OPERATOR)
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                endpoint="http://agent:8090",
            )
        )
    low, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
        ),
        idempotency_key="low",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    high, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )
    assert (await controller.reconcile_once()).preemptions

    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    restored = next(workload for workload in workloads if workload.id == low.id)
    target = next(workload for workload in workloads if workload.id == high.id)
    command = next(command for command in commands if command.topic == "workload.preempt")
    assert restored.state is WorkloadState.RUNNING
    assert restored.preemption_id is None
    assert target.preemption_id is None
    assert leases[0].state is LeaseState.ACTIVE
    assert command.state is OutboxState.SUCCEEDED
    assert command.payload["execution_completed"]
    assert command.payload["aborted_reason"]
    assert fake_agent.checkpoints
    assert not fake_agent.stops
    cancelled, _ = await service.cancel_workload(operator, high.id, idempotency_key="cancel-after-abort")
    assert cancelled.state is WorkloadState.CANCELLED


@pytest.mark.asyncio
async def test_aborted_checkpoint_phase_retires_the_hook_it_left_running() -> None:
    """A hook that outlived its grace is still holding the victim's accelerators.

    The abort rolls the victim back to RUNNING, so an un-retired hook would then
    be sharing the TPU with the very trainer it was meant to checkpoint, and
    could still write output under a preemption plan that no longer exists.
    """

    class HangingCheckpointAgent(FakeAgent):
        def workload(self, cell_id, remote_id):
            if remote_id.startswith("checkpoint-") and remote_id not in self.remote_states:
                # Never terminal: the hook is still running when grace expires.
                return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.RUNNING)
            return super().workload(cell_id, remote_id)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = HangingCheckpointAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: fake_agent)  # type: ignore[arg-type]
    operator = principal(Role.PROJECT_OPERATOR)
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                endpoint="http://agent:8090",
            )
        )
    low, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
            checkpoint_grace_seconds=1,
        ),
        idempotency_key="low",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    _high, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )
    assert (await controller.reconcile_once()).preemptions
    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    restored = next(workload for workload in workloads if workload.id == low.id)
    command = next(command for command in commands if command.topic == "workload.preempt")
    assert restored.state is WorkloadState.RUNNING
    assert leases[0].state is LeaseState.ACTIVE
    assert command.payload["execution_completed"]

    assert fake_agent.checkpoints, "the hook must have been launched for this to be a real abort"
    assert not fake_agent.stops, "the victim itself is rolled back, not stopped"
    hook_id = f"checkpoint-{fake_agent.checkpoints[0][1]}"
    assert [remote_id for _cell, remote_id, _request in fake_agent.retires] == [hook_id]
    # The hook is fenced as the victim's lease was, or the agent would reject it.
    assert fake_agent.retires[0][2].mutation.fencing_token == leases[0].fencing_token


@pytest.mark.asyncio
async def test_started_multi_victim_preemption_finishes_as_one_committed_plan() -> None:
    class BlockingStopAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.stop_entered = threading.Event()
            self.allow_stop = threading.Event()

        def stop(self, cell_id, remote_id, request):
            if not self.stops:
                self.stop_entered.set()
                if not self.allow_stop.wait(timeout=5):
                    raise TimeoutError("test did not release the first preemption stop")
            return super().stop(cell_id, remote_id, request)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = BlockingStopAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(checkpoint_grace_s=2),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
    )
    research_operator = principal(Role.PROJECT_OPERATOR)
    other_operator = Principal(
        subject="bob",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="other"),),
    )
    other_admin = Principal(
        subject="bob-admin",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="other"),),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                executor="ray-jobs",
                endpoint="http://agent:8090",
            )
        )

    low_ids: list[str] = []
    for index, (project_id, operator) in enumerate(
        (("research", research_operator), ("other", other_operator)),
        start=1,
    ):
        low, _ = await service.submit_workload(
            operator,
            org_id="acme",
            project_id=project_id,
            spec=WorkloadDefinition(
                name=f"batch-{index}",
                execution=CommandExecution(command="python batch.py"),
                gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 4}},
                priority_band=PriorityBand.BATCH,
                checkpointable=True,
                checkpoint_hook="python checkpoint.py",
                checkpoint_output_uri=f"gs://checkpoints/batch-{index}",
            ),
            idempotency_key=f"low-{index}",
        )
        low_ids.append(low.id)
    assert len((await controller.reconcile_once()).placements) == 2
    assert await controller.dispatch_outbox_once() == 2
    assert await controller.observe_workloads_once() == 2

    high, _ = await service.submit_workload(
        research_operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )
    result = await controller.reconcile_once()
    assert len(result.preemptions) == 1
    assert len(result.preemptions[0].victims) == 2
    _, _, _, commands = await snapshot(repository)
    plans = [command for command in commands if command.topic == "workload.preempt"]
    assert len(plans) == 1
    assert len(plans[0].payload["victims"]) == 2

    delivery = asyncio.create_task(controller.dispatch_outbox_once())
    assert await asyncio.to_thread(fake_agent.stop_entered.wait, 2)
    with pytest.raises(ConflictError, match="already executing"):
        await service.cancel_workload(research_operator, high.id, idempotency_key="cancel-started-high")
    await service.put_policy(
        other_admin,
        PolicyRecord(org_id="acme", project_id="other", preemption_enabled=False, updated_by="bob-admin"),
        idempotency_key="disable-other-preemption",
    )
    fake_agent.allow_stop.set()
    assert await delivery == 1

    workloads, _, leases, _ = await snapshot(repository)
    preempted = [workload for workload in workloads if workload.id in low_ids]
    assert all(workload.state is WorkloadState.PENDING and workload.attempt == 2 for workload in preempted)
    assert all(lease.state is LeaseState.RELEASED for lease in leases)
    assert len(fake_agent.checkpoints) == 2
    assert len(fake_agent.stops) == 2
    target = next(workload for workload in workloads if workload.id == high.id)
    assert target.preemption_id is None
    cancelled, _ = await service.cancel_workload(
        research_operator,
        high.id,
        idempotency_key="cancel-after-completion",
    )
    assert cancelled.state is WorkloadState.CANCELLED


def _persisted_preemption_command(payload: JsonObject) -> OutboxRecord:
    command = OutboxRecord(
        id="cmd-preemption-payload",
        topic="workload.preempt",
        aggregate_type="workload",
        aggregate_id="target-workload",
        idempotency_key="cmd-preemption-payload",
        generation=3,
        payload=payload,
    )
    return OutboxRecord.model_validate_json(command.model_dump_json())


def _valid_preemption_victim_payload() -> JsonObject:
    return {
        "workload_id": "victim-workload",
        "lease_id": "victim-lease",
        "cell_id": "cell-a",
        "generation": 2,
        "checkpoint": True,
        "grace_period_seconds": 30,
        "hook": "python checkpoint.py",
        "output_uri": "gs://checkpoints/victim-workload",
        "workload_state": WorkloadState.RUNNING.value,
        "lease_state": LeaseState.ACTIVE.value,
    }


def test_preemption_victim_payload_parses_once_into_typed_durable_fields() -> None:
    command = _persisted_preemption_command({"victims": [_valid_preemption_victim_payload()]})

    (victim,) = EyvanController._preemption_victims(command)

    assert victim.workload_id == "victim-workload"
    assert victim.lease_id == "victim-lease"
    assert victim.cell_id == "cell-a"
    assert victim.generation == 2
    assert victim.checkpoint is True
    assert victim.grace_period_seconds == 30
    assert victim.hook == "python checkpoint.py"
    assert victim.output_uri == "gs://checkpoints/victim-workload"
    assert victim.workload_state is WorkloadState.RUNNING
    assert victim.lease_state is LeaseState.ACTIVE


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("workload_id", ""),
        ("generation", "2"),
        ("checkpoint", 1),
        ("grace_period_seconds", "30"),
        ("workload_state", WorkloadState.SUCCEEDED.value),
        ("lease_state", LeaseState.RELEASED.value),
    ],
)
def test_preemption_victim_rejects_malformed_persisted_fields(field: str, malformed: str | int) -> None:
    victim = _valid_preemption_victim_payload()
    victim[field] = malformed
    command = _persisted_preemption_command({"victims": [victim]})

    with pytest.raises(StaleOutboxCommand, match="invalid victim"):
        EyvanController._preemption_victims(command)


def test_preemption_progress_rejects_malformed_persisted_maps() -> None:
    checkpoint_command = _persisted_preemption_command({"checkpoint_uris": {"victim-workload": 7}})
    with pytest.raises(StaleOutboxCommand, match="invalid checkpoint URIs"):
        EyvanController._preemption_checkpoint_uris(checkpoint_command)

    outcome_command = _persisted_preemption_command({"victim_outcomes": {"victim-lease": "not-an-outcome"}})
    with pytest.raises(StaleOutboxCommand, match="invalid victim outcomes"):
        EyvanController._preemption_victim_outcomes(outcome_command)

    stopped_command = _persisted_preemption_command({"stopped_victims": ["victim-lease", 7]})
    with pytest.raises(StaleOutboxCommand, match="invalid stopped-victim set"):
        EyvanController._preemption_stopped_victims(stopped_command)


async def _prepare_two_victim_preemption(
    fake_agent: FakeAgent,
    *,
    settings: EyvanSettings | None = None,
) -> tuple[
    InMemoryRepository,
    ControlPlaneService,
    EyvanController,
    list[str],
    str,
    OutboxRecord,
]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=settings or EyvanSettings(checkpoint_grace_s=2, preemption_stop_grace_s=2),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
    )
    research_operator = principal(Role.PROJECT_OPERATOR)
    other_operator = Principal(
        subject="bob",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="other"),),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                executor="ray-jobs",
                endpoint="http://agent:8090",
            )
        )

    low_ids: list[str] = []
    for index, (project_id, operator) in enumerate(
        (("research", research_operator), ("other", other_operator)),
        start=1,
    ):
        low, _ = await service.submit_workload(
            operator,
            org_id="acme",
            project_id=project_id,
            spec=WorkloadDefinition(
                name=f"batch-{index}",
                execution=CommandExecution(command="python batch.py"),
                gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 4}},
                priority_band=PriorityBand.BATCH,
                checkpointable=True,
                checkpoint_hook="python checkpoint.py",
                checkpoint_output_uri=f"gs://checkpoints/batch-{index}",
            ),
            idempotency_key=f"low-{index}",
        )
        low_ids.append(low.id)
    assert len((await controller.reconcile_once()).placements) == 2
    assert await controller.dispatch_outbox_once() == 2
    assert await controller.observe_workloads_once() == 2

    high, _ = await service.submit_workload(
        research_operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high",
    )
    result = await controller.reconcile_once()
    assert len(result.preemptions) == 1
    assert len(result.preemptions[0].victims) == 2
    _, _, _, commands = await snapshot(repository)
    plans = [command for command in commands if command.topic == "workload.preempt"]
    assert len(plans) == 1
    return repository, service, controller, low_ids, high.id, plans[0]


@pytest.mark.asyncio
async def test_reclaimed_preemption_claim_does_not_roll_back_stopped_victims() -> None:
    class BlockingSecondStopAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.second_stop_entered = threading.Event()
            self.allow_second_stop = threading.Event()

        def stop(self, cell_id, remote_id, request):
            self.stops.append((cell_id, remote_id, request))
            if len(self.stops) == 2:
                self.second_stop_entered.set()
                if not self.allow_second_stop.wait(timeout=5):
                    raise TimeoutError("test did not release the second preemption stop")
            self.remote_states[remote_id] = ExecutionWorkloadStatus.STOPPED
            return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    fake_agent = BlockingSecondStopAgent()
    settings = EyvanSettings(checkpoint_grace_s=2, preemption_stop_grace_s=2, outbox_claim_ttl_s=1)
    repository, service, controller_a, low_ids, _, plan = await _prepare_two_victim_preemption(
        fake_agent,
        settings=settings,
    )
    controller_b = EyvanController(
        repository,
        service,
        settings=settings,
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
    )

    delivery_a = asyncio.create_task(controller_a.dispatch_outbox_once())
    assert await asyncio.to_thread(fake_agent.second_stop_entered.wait, 2)
    first_lease_id = str(_preemption_victim_payloads(plan.payload)[0]["lease_id"])
    async with repository.transaction() as transaction:
        current = await transaction.get_outbox(plan.id)
        assert current is not None
        assert current.payload["victim_outcomes"] == {first_lease_id: "preempted"}
        await transaction.put_outbox(
            current.model_copy(update={"claimed_until": datetime.now(UTC) - timedelta(seconds=1)})
        )

    try:
        assert await controller_b.dispatch_outbox_once() == 1
    finally:
        fake_agent.allow_second_stop.set()
    assert await delivery_a == 1

    workloads, _, leases, commands = await snapshot(repository)
    victims = [workload for workload in workloads if workload.id in low_ids]
    current = next(command for command in commands if command.id == plan.id)
    assert current.state is OutboxState.SUCCEEDED
    assert current.payload["execution_completed"]
    assert all(workload.state is WorkloadState.PENDING and workload.attempt == 2 for workload in victims)
    assert all(lease.state is LeaseState.RELEASED for lease in leases)


@pytest.mark.asyncio
async def test_preemption_target_cannot_place_on_a_newly_ready_cell() -> None:
    fake_agent = FakeAgent()
    repository, _, controller, _, high_id, plan = await _prepare_two_victim_preemption(fake_agent)
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-b",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                endpoint="http://agent-b:8090",
            )
        )

    result = await controller.reconcile_once()

    workloads, _, _, commands = await snapshot(repository)
    target = next(workload for workload in workloads if workload.id == high_id)
    assert all(placement.workload_id != high_id for placement in result.placements)
    assert target.state is WorkloadState.PENDING
    assert target.lease_id is None
    assert target.preemption_id == plan.id
    assert not any(command.topic == "workload.dispatch" and command.aggregate_id == high_id for command in commands)


@pytest.mark.asyncio
async def test_naturally_succeeded_preemption_victim_remains_succeeded() -> None:
    fake_agent = FakeAgent()
    repository, _, controller, low_ids, _, plan = await _prepare_two_victim_preemption(fake_agent)
    succeeded_id = str(_preemption_victim_payloads(plan.payload)[0]["workload_id"])
    async with repository.transaction() as transaction:
        succeeded = await transaction.get_workload(succeeded_id)
        assert succeeded is not None and succeeded.remote_id is not None
        succeeded_remote_id = succeeded.remote_id
    fake_agent.remote_states[succeeded_remote_id] = ExecutionWorkloadStatus.SUCCEEDED

    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    by_workload = {workload.id: workload for workload in workloads}
    by_lease = {lease.workload_id: lease for lease in leases}
    assert by_workload[succeeded_id].state is WorkloadState.SUCCEEDED
    assert by_lease[succeeded_id].state is LeaseState.RELEASED
    assert by_lease[succeeded_id].settlement_reason == "succeeded"
    other_id = next(workload_id for workload_id in low_ids if workload_id != succeeded_id)
    assert by_workload[other_id].state is WorkloadState.PENDING
    assert succeeded_remote_id not in [remote_id for _, remote_id, _ in fake_agent.stops]
    command = next(command for command in commands if command.id == plan.id)
    assert _payload_object(command.payload, "victim_outcomes")[by_lease[succeeded_id].id] == "succeeded"


@pytest.mark.asyncio
async def test_lost_preemption_victim_is_settled_without_stop_timeout() -> None:
    fake_agent = FakeAgent()
    repository, _, controller, low_ids, _, plan = await _prepare_two_victim_preemption(fake_agent)
    lost_id = str(_preemption_victim_payloads(plan.payload)[0]["workload_id"])
    async with repository.transaction() as transaction:
        lost = await transaction.get_workload(lost_id)
        assert lost is not None and lost.remote_id is not None
        lost_remote_id = lost.remote_id
    fake_agent.remote_states[lost_remote_id] = ExecutionWorkloadStatus.LOST

    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    by_workload = {workload.id: workload for workload in workloads}
    by_lease = {lease.workload_id: lease for lease in leases}
    current = next(command for command in commands if command.id == plan.id)
    assert current.state is OutboxState.SUCCEEDED
    assert current.payload["execution_completed"]
    assert by_workload[lost_id].state is WorkloadState.PENDING
    assert by_workload[lost_id].attempt == 2
    assert by_lease[lost_id].state is LeaseState.RELEASED
    assert by_lease[lost_id].settlement_reason == "preempted"
    assert lost_remote_id not in [remote_id for _, remote_id, _ in fake_agent.stops]
    assert _payload_object(current.payload, "victim_outcomes")[by_lease[lost_id].id] == "preempted"
    assert all(by_workload[workload_id].state is WorkloadState.PENDING for workload_id in low_ids)


@pytest.mark.asyncio
async def test_preemption_retry_skips_a_victim_with_durable_stop_progress() -> None:
    class FailSecondStopOnceAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.failed = False

        def stop(self, cell_id, remote_id, request):
            self.stops.append((cell_id, remote_id, request))
            if len(self.stops) == 2 and not self.failed:
                self.failed = True
                raise RuntimeError("second victim stop failed")
            self.remote_states[remote_id] = ExecutionWorkloadStatus.STOPPED
            return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    fake_agent = FailSecondStopOnceAgent()
    repository, _, controller, low_ids, _, plan = await _prepare_two_victim_preemption(fake_agent)
    assert await controller.dispatch_outbox_once() == 1

    async with repository.transaction() as transaction:
        failed = await transaction.get_outbox(plan.id)
        assert failed is not None
        assert failed.state is OutboxState.FAILED
        assert len(failed.payload["victim_outcomes"]) == 1
        await transaction.put_outbox(failed.model_copy(update={"available_at": datetime.now(UTC)}))
    first_remote_id = fake_agent.stops[0][1]
    second_remote_id = fake_agent.stops[1][1]

    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    victims = [workload for workload in workloads if workload.id in low_ids]
    current = next(command for command in commands if command.id == plan.id)
    stopped_remote_ids = [remote_id for _, remote_id, _ in fake_agent.stops]
    assert current.state is OutboxState.SUCCEEDED
    assert current.payload["execution_completed"]
    assert stopped_remote_ids.count(first_remote_id) == 1
    assert stopped_remote_ids.count(second_remote_id) == 2
    assert all(workload.state is WorkloadState.PENDING and workload.attempt == 2 for workload in victims)
    assert all(lease.state is LeaseState.RELEASED for lease in leases)


@pytest.mark.asyncio
async def test_fleet_admin_force_finalizes_missing_preemption_outcomes() -> None:
    class FailSecondStopAgent(FakeAgent):
        def stop(self, cell_id, remote_id, request):
            self.stops.append((cell_id, remote_id, request))
            if len(self.stops) == 2:
                raise RuntimeError("second victim stop outcome is unknown")
            self.remote_states[remote_id] = ExecutionWorkloadStatus.STOPPED
            return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    fake_agent = FailSecondStopAgent()
    repository, service, controller, low_ids, high_id, plan = await _prepare_two_victim_preemption(fake_agent)
    assert await controller.dispatch_outbox_once() == 1

    target, replayed = await service.force_finalize_preemption(
        principal(Role.FLEET_ADMIN),
        plan.id,
        reason="operator verified the second Ray job is terminal",
        idempotency_key="force-finalize",
    )
    assert target.id == high_id
    assert not replayed
    async with repository.transaction() as transaction:
        forced = await transaction.get_outbox(plan.id)
        assert forced is not None
        expected_lease_ids = {str(victim["lease_id"]) for victim in _preemption_victim_payloads(plan.payload)}
        assert forced.state is OutboxState.FAILED
        assert forced.payload["victim_outcomes"] == dict.fromkeys(expected_lease_ids, "preempted")
        assert set(forced.payload["stopped_victims"]) == expected_lease_ids
        assert forced.payload["force_finalize_actor"] == "admin"

    assert await controller.dispatch_outbox_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    victims = [workload for workload in workloads if workload.id in low_ids]
    target = next(workload for workload in workloads if workload.id == high_id)
    current = next(command for command in commands if command.id == plan.id)
    assert current.state is OutboxState.SUCCEEDED
    assert current.payload["execution_completed"]
    assert target.preemption_id is None
    assert all(workload.state is WorkloadState.PENDING and workload.attempt == 2 for workload in victims)
    assert all(lease.state is LeaseState.RELEASED for lease in leases)


@pytest.mark.asyncio
async def test_controller_enforces_configured_cell_price_against_project_budget() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="gcp",
                provider_project="gcp-project-a",
                region="us-central2",
                zone="us-central2-b",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                unit_prices={"tpu": 2.4},
                accelerator_type="v5p-8",
            )
        )
    await service.put_budget(
        principal(Role.PROJECT_ADMIN),
        BudgetRecord(org_id="acme", project_id="research", limit=10, updated_by="admin"),
        idempotency_key="budget",
    )
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="too-expensive",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            estimated_duration_seconds=3600,
        ),
        idempotency_key="workload",
    )

    result = await controller.reconcile_once()

    assert not result.placements
    assert result.infeasibilities[0].workload_id == workload.id
    assert result.infeasibilities[0].code is InfeasibilityCode.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_fractional_accelerator_hours_are_persisted_on_lease_commit() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=0, memory_gib=0, tpu=1),
                allocatable=ResourceVector(cpu=0, memory_gib=0, tpu=1),
            )
        )
    await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="half-hour",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 1}},
            estimated_duration_seconds=1800,
        ),
        idempotency_key="half-hour",
    )

    await controller.reconcile_once()

    _, _, leases, _ = await snapshot(repository)
    assert leases[0].resource_hours_reserved.tpu == 0.5


@pytest.mark.asyncio
async def test_undispatched_lease_requeues_without_budget_charge_after_cell_loss() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
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
    leased_workloads, _, pending_leases, _ = await snapshot(repository)
    assert leased_workloads[0].state is WorkloadState.LEASED
    assert pending_leases[0].state is LeaseState.PENDING
    async with repository.transaction() as transaction:
        cell = await transaction.get_cell("cell-a")
        assert cell is not None
        await transaction.put_cell(cell.model_copy(update={"state": CellState.REGISTERING}))

    assert await controller.observe_workloads_once() == 1

    workloads, _, leases, _ = await snapshot(repository)
    assert (workloads[0].state, workloads[0].attempt, workloads[0].generation) == (
        WorkloadState.PENDING,
        1,
        workload.generation + 1,
    )
    assert leases[0].state is LeaseState.RELEASED
    assert leases[0].settlement_reason == "unstarted"
    assert leases[0].settled_cost == 0


@pytest.mark.asyncio
async def test_observer_requeues_quarantined_undispatched_lease_after_dispatch_retires() -> None:
    repository, controller, leased, lease, command = await _prepare_undispatched_leased_workload()
    async with repository.transaction() as transaction:
        current_lease = await transaction.get_lease(lease.id)
        current_command = await transaction.get_outbox(command.id)
        assert current_lease is not None
        assert current_command is not None
        await transaction.put_lease(current_lease.model_copy(update={"expires_at": utcnow() - timedelta(seconds=1)}))
        await transaction.put_outbox(current_command.model_copy(update={"state": OutboxState.DISCARDED}))

    assert await controller.quarantine_expired_leases_once() == 1
    assert await controller.observe_workloads_once() == 1

    workloads, _, leases, commands = await snapshot(repository)
    recovered = next(workload for workload in workloads if workload.id == leased.id)
    released = next(candidate for candidate in leases if candidate.id == lease.id)
    retired = next(candidate for candidate in commands if candidate.id == command.id)
    assert (recovered.state, recovered.generation, recovered.lease_id, recovered.remote_id) == (
        WorkloadState.PENDING,
        leased.generation + 1,
        None,
        None,
    )
    assert (released.state, released.settlement_reason, released.settled_cost) == (
        LeaseState.RELEASED,
        "unstarted",
        0,
    )
    assert retired.state is OutboxState.DISCARDED

    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [leased.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatch_state", (OutboxState.PENDING, OutboxState.PROCESSING, OutboxState.FAILED))
async def test_observer_keeps_quarantined_undispatched_lease_while_dispatch_is_live(
    dispatch_state: OutboxState,
) -> None:
    repository, controller, leased, lease, command = await _prepare_undispatched_leased_workload()
    async with repository.transaction() as transaction:
        current_lease = await transaction.get_lease(lease.id)
        current_command = await transaction.get_outbox(command.id)
        assert current_lease is not None
        assert current_command is not None
        await transaction.put_lease(current_lease.model_copy(update={"expires_at": utcnow() - timedelta(seconds=1)}))
        await transaction.put_outbox(current_command.model_copy(update={"state": dispatch_state}))

    assert await controller.quarantine_expired_leases_once() == 1
    assert await controller.observe_workloads_once() == 0

    workloads, _, leases, _ = await snapshot(repository)
    current_workload = next(workload for workload in workloads if workload.id == leased.id)
    current_lease = next(candidate for candidate in leases if candidate.id == lease.id)
    assert (
        current_workload.state,
        current_workload.generation,
        current_workload.lease_id,
        current_workload.remote_id,
    ) == (
        WorkloadState.LEASED,
        leased.generation,
        lease.id,
        None,
    )
    assert current_lease.state is LeaseState.QUARANTINED


@pytest.mark.asyncio
async def test_adding_a_leader_phase_does_not_change_the_runner_signature() -> None:
    """The collision this registry exists to prevent.

    Five planned changes each add a leader phase. While the runner took one
    positional deadline per interval-gated phase, every pair of those changes
    conflicted on the same signature. Deadlines are now keyed by phase name, so a new
    phase adds a dict entry and nothing else.
    """

    class Leader:
        async def acquire(self) -> bool:
            return True

        async def release(self) -> None:
            return None

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, leader=Leader())
    ran: list[str] = []

    async def noop() -> None:
        ran.append("extra")

    original = controller._leader_phases
    controller._leader_phases = lambda: (  # type: ignore[method-assign]
        *original(),
        LeaderPhase("extra", noop, interval_s=60),
    )

    due = await controller._run_leader_phases(now=1_000.0, due={})

    assert "extra" in ran
    assert due["extra"] == 1_060.0
    # The pre-existing phases still carry their own independent deadlines.
    assert due["retention"] == 1_000.0 + controller.settings.retention_interval_s


@pytest.mark.asyncio
async def test_an_interval_phase_is_skipped_until_it_comes_due() -> None:
    class Leader:
        async def acquire(self) -> bool:
            return True

        async def release(self) -> None:
            return None

    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository), leader=Leader())
    ran: list[str] = []

    async def tick() -> None:
        ran.append("tick")

    controller._leader_phases = lambda: (LeaderPhase("tick", tick, interval_s=100),)  # type: ignore[method-assign]

    due = await controller._run_leader_phases(now=0.0, due={"tick": 50.0})
    assert ran == []
    assert due["tick"] == 50.0

    due = await controller._run_leader_phases(now=60.0, due=due)
    assert ran == ["tick"]
    assert due["tick"] == 160.0


@pytest.mark.asyncio
async def test_a_zero_interval_disables_only_the_phases_that_document_it() -> None:
    """Zero means "disabled" for retention and "every cycle" for cell reconcile.

    Collapsing those two meanings would silently stop cell observation on any
    deployment that set the interval to zero.
    """

    class Leader:
        async def acquire(self) -> bool:
            return True

        async def release(self) -> None:
            return None

    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository), leader=Leader())
    ran: list[str] = []

    async def disabled() -> None:
        ran.append("disabled")

    async def every_cycle() -> None:
        ran.append("every-cycle")

    controller._leader_phases = lambda: (  # type: ignore[method-assign]
        LeaderPhase("disabled", disabled, interval_s=0, zero_interval_disables=True),
        LeaderPhase("every-cycle", every_cycle, interval_s=0),
    )

    await controller._run_leader_phases(now=0.0, due={})

    assert ran == ["every-cycle"]


def _always_leader():
    class Leader:
        async def acquire(self) -> bool:
            return True

        async def release(self) -> None:
            return None

    return Leader()


def test_every_declared_prerequisite_names_a_phase_that_exists() -> None:
    """A typo would silently remove the guard rather than break anything.

    ``requires`` only ever *blocks*, so a name matching no phase can never be in
    the failed set and the phase simply always runs -- losing the one protection
    that justified containing failures everywhere else.
    """

    repository = InMemoryRepository()
    phases = EyvanController(repository, ControlPlaneService(repository))._leader_phases()
    names = {phase.name for phase in phases}

    dangling = {(phase.name, required) for phase in phases for required in phase.requires if required not in names}

    assert dangling == set()


def test_every_prerequisite_runs_before_the_phase_that_names_it() -> None:
    """The failed set is per-cycle, so a later phase's failure is invisible here.

    Declaring one would read as a guard and be none, which is worse than not
    declaring it at all.
    """

    repository = InMemoryRepository()
    phases = EyvanController(repository, ControlPlaneService(repository))._leader_phases()
    position = {phase.name: index for index, phase in enumerate(phases)}

    inverted = {
        (phase.name, required)
        for phase in phases
        for required in phase.requires
        if position[required] > position[phase.name]
    }

    assert inverted == set()


def test_the_phases_that_give_capacity_back_depend_on_nothing() -> None:
    """The property the redesign exists for, stated against the real list.

    Reclaiming is safe against any view, and not reclaiming is what strands a
    running accelerator. A prerequisite added to one of these would reintroduce
    the failure that cost eight minutes of v5p-16 by a different route.
    """

    repository = InMemoryRepository()
    phases = {
        phase.name: phase for phase in EyvanController(repository, ControlPlaneService(repository))._leader_phases()
    }

    for name in ("settle-orphans", "settle-orphan-leases", "settle-coordinators", "cell-billing"):
        assert phases[name].requires == (), name


@pytest.mark.asyncio
async def test_one_failing_phase_does_not_stop_the_rest_of_the_cycle() -> None:
    """The property a live v5p-16 multislice paid for.

    A failing phase used to propagate out of the whole cycle, and the phases
    that release capacity and settle billing sit in the middle of the list --
    so one failure upstream of them left an accelerator running that nothing
    was going to stop paying for. Reclaiming has to be reachable from any
    failure, because not reclaiming is the expensive mistake.
    """

    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository), leader=_always_leader())
    ran: list[str] = []

    async def boom() -> None:
        raise RuntimeError("phase failed")

    async def settle() -> None:
        ran.append("settle-orphan-leases")

    async def bill() -> None:
        ran.append("cell-billing")

    controller._leader_phases = lambda: (  # type: ignore[method-assign]
        LeaderPhase("observe-cells", boom),
        LeaderPhase("settle-orphan-leases", settle),
        LeaderPhase("cell-billing", bill),
    )

    await controller._run_leader_phases(now=0.0, due={})

    assert ran == ["settle-orphan-leases", "cell-billing"]


@pytest.mark.asyncio
async def test_a_phase_that_commits_work_is_skipped_when_its_view_is_stale() -> None:
    """The safety the old stop-the-cycle behaviour actually bought, kept.

    Placing a workload decides where real money goes. Doing that against a cell
    inventory this cycle failed to refresh is the mistake worth refusing -- so a
    committing phase names what it reads, and is skipped rather than stopped as
    a side effect of something unrelated.
    """

    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository), leader=_always_leader())
    ran: list[str] = []

    async def boom() -> None:
        raise RuntimeError("observation failed")

    async def scheduled() -> None:
        ran.append("schedule")

    async def settle() -> None:
        ran.append("settle-orphan-leases")

    controller._leader_phases = lambda: (  # type: ignore[method-assign]
        LeaderPhase("observe-cells", boom),
        LeaderPhase("settle-orphan-leases", settle),
        LeaderPhase("schedule", scheduled, requires=("observe-cells", "observe-workloads")),
    )

    await controller._run_leader_phases(now=0.0, due={})

    assert ran == ["settle-orphan-leases"]


@pytest.mark.asyncio
async def test_a_committing_phase_runs_when_the_failure_was_somewhere_else() -> None:
    """Only the observations it named gate it, not any failure anywhere."""

    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository), leader=_always_leader())
    ran: list[str] = []

    async def boom() -> None:
        raise RuntimeError("unrelated")

    async def scheduled() -> None:
        ran.append("schedule")

    controller._leader_phases = lambda: (  # type: ignore[method-assign]
        LeaderPhase("capacity-pools", boom),
        LeaderPhase("observe-cells", lambda: _done()),
        LeaderPhase("observe-workloads", lambda: _done()),
        LeaderPhase("schedule", scheduled, requires=("observe-cells", "observe-workloads")),
    )

    await controller._run_leader_phases(now=0.0, due={})

    assert ran == ["schedule"]


@pytest.mark.asyncio
async def test_a_skipped_phase_says_which_observation_it_was_waiting_on(caplog) -> None:
    """An operator reading the log must not have to infer the dependency."""

    import logging

    repository = InMemoryRepository()
    controller = EyvanController(repository, ControlPlaneService(repository), leader=_always_leader())

    async def boom() -> None:
        raise RuntimeError("observation failed")

    async def scheduled() -> None:  # pragma: no cover - it must not run
        raise AssertionError("scheduled against a stale view")

    controller._leader_phases = lambda: (  # type: ignore[method-assign]
        LeaderPhase("observe-cells", boom),
        LeaderPhase("schedule", scheduled, requires=("observe-cells",)),
    )

    with caplog.at_level(logging.WARNING):
        await controller._run_leader_phases(now=0.0, due={})

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "schedule" in message
    assert "observe-cells" in message


async def _done() -> None:
    return None


@pytest.mark.asyncio
async def test_a_plain_http_plane_settles_the_attempt_instead_of_holding_the_accelerator(tmp_path) -> None:
    """Settle a non-loopback plaintext callback refusal instead of retrying while holding an accelerator."""

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    fake_agent = FakeAgent()
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path / "artifacts"),
        signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=3600, max_ttl_seconds=3600),
        # What a head actually writes: internal address, no TLS.
        public_url="http://10.128.0.11:8080",
        token_ttl_seconds=3600,
    )
    controller = EyvanController(
        repository,
        service,
        # dev auth is refused on a network bind, so a head can never have it.
        settings=EyvanSettings(dev_auth_enabled=False),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
        artifact_access=access,
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-managed",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
            )
        )
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="callable",
            execution=CallableExecution(
                artifact_uri=callable_uri("a" * 64),
                sha256="a" * 64,
                python="3.13",
                result_uri=result_uri("call-insecure"),
            ),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
        ),
        idempotency_key="insecure-http",
    )

    result = await controller.reconcile_once()
    assert [placement.workload_id for placement in result.placements] == [workload.id]
    await controller.dispatch_outbox_once()

    async with repository.transaction() as transaction:
        settled = await transaction.get_workload(workload.id)
        leases = await transaction.list_leases()

    # Settled with a reason, not retried forever.
    assert settled.state is WorkloadState.FAILED
    assert "clear text" in (settled.failure_reason or "")
    assert "EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP" in (settled.failure_reason or "")
    # And the capacity is given back rather than held by a lease that never resolves.
    assert [lease for lease in leases if lease.state is LeaseState.PENDING] == []
    assert fake_agent.submissions == []


@pytest.mark.asyncio
async def test_an_operator_who_accepts_the_risk_can_run_callables_on_a_private_plane(tmp_path) -> None:
    """The refusal must be escapable, or a VPC-only head can never run a callable."""

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    fake_agent = FakeAgent()
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path / "artifacts"),
        signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=3600, max_ttl_seconds=3600),
        public_url="http://10.128.0.11:8080",
        token_ttl_seconds=3600,
    )
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(dev_auth_enabled=False, allow_insecure_artifact_http=True),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
        artifact_access=access,
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-managed",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
            )
        )
    await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="callable",
            execution=CallableExecution(
                artifact_uri=callable_uri("a" * 64),
                sha256="a" * 64,
                python="3.13",
                result_uri=result_uri("call-consented"),
            ),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
        ),
        idempotency_key="insecure-http-consented",
    )

    assert await controller.reconcile_once() is not None
    assert await controller.dispatch_outbox_once() == 1

    request = fake_agent.submissions[0][1]
    assert request.spec["environment"]["EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP"] == "1"


@pytest.mark.asyncio
async def test_a_multi_cell_gang_can_actually_be_preempted() -> None:
    """The pure scheduler named every sibling; the controller threw the plan away.

    A workload holding a lease *group* records the group in `lease_ids` and only
    the leader in `lease_id`. The commit guard tested `workload.lease_id !=
    lease.id`, so the first sibling failed it and the whole plan was abandoned
    before anything was written -- no outbox command, no revocation, the victim
    still LEASED and the high-priority workload waiting forever. Metrics counted
    a preemption on every cycle; none ever happened.

    `test_gang_preemption_is_atomic` covers the scheduler, which emits the
    siblings correctly, so it stayed green throughout. The gap was between the
    plan and its commit, and only a controller-level test can see it.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(checkpoint_grace_s=2),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
    )
    admin = principal(Role.FLEET_ADMIN)
    operator = principal(Role.PROJECT_OPERATOR)
    for name in ("cell-a", "cell-b"):
        await service.upsert_cell(
            admin,
            CellRecord(
                id=name,
                provider="gcp",
                provider_project="gcp-project-a",
                region="us-central2",
                zone="us-central2-b",
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                accelerator_type="v5p-8",
                endpoint="http://agent:8090",
            ),
            idempotency_key=name,
        )
    await controller.dispatch_outbox_once()

    # A gang spanning both cells: this is the shape that could never be preempted.
    low, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch-gang",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            cell_group_count=2,
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
        ),
        idempotency_key="low-gang",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()

    async with repository.transaction() as transaction:
        held = await transaction.get_workload(low.id)
        leases = await transaction.list_leases()
    # The premise: it really is holding two leases, with one of them the leader.
    assert len(held.lease_ids) == 2, f"expected a two-cell gang, got {held.lease_ids}"
    assert held.lease_id in held.lease_ids
    assert len(leases) == 2

    high, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="production",
            execution=CommandExecution(command="python production.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            priority_band=PriorityBand.PRODUCTION,
        ),
        idempotency_key="high-gang",
    )

    result = await controller.reconcile_once()

    assert [plan.workload_id for plan in result.preemptions] == [high.id]
    async with repository.transaction() as transaction:
        target = await transaction.get_workload(high.id)
        revoking = await transaction.list_leases()
    # Committed, not discarded: the plan is durable and every sibling is revoking.
    assert target.preemption_id is not None
    assert {lease.state for lease in revoking} == {LeaseState.REVOKING}


@pytest.mark.asyncio
async def test_workload_logs_answer_when_the_capacity_is_gone_instead_of_500ing() -> None:
    """The orphan sweep settles a workload whose leases are already gone.

    `workload_logs` then indexed an empty member list and raised, so the route
    answered HTTP 500 on exactly the records an operator reads logs for -- the
    ones something went wrong with. `/log-page` handled the same state cleanly,
    so the two surfaces disagreed about the same workload.

    Also pins the two answers that used to be an identical empty string: logs
    suppressed because the workload mounts secrets, and a workload that was never
    dispatched at all. "Suppressed", "never ran" and "produced nothing" are three
    different facts.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    operator = principal(Role.PROJECT_OPERATOR)

    undispatched, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="never-ran", execution=CommandExecution(command="python x.py")),
        idempotency_key="never-ran",
    )
    text = await controller.workload_logs(undispatched.id)
    assert "not been dispatched" in text

    # Written directly: a secret mount needs a project policy allowing its
    # scheme, and this test is about what `workload_logs` answers, not about
    # secret admission.
    async with repository.transaction() as transaction:
        base = await transaction.get_workload(undispatched.id)
        secretive = base.model_copy(
            update={
                "id": "wl_secretive",
                "spec": base.spec.model_copy(update={"secret_mounts": {"TOKEN": "gcp-secret://project/token"}}),
            }
        )
        await transaction.put_workload(secretive)

    text = await controller.workload_logs(secretive.id)
    assert "suppressed" in text
    assert "secrets" in text

    # A workload that ran and whose capacity has since been released: remote_id
    # set, no leases left. This is the shape the orphan sweep produces.
    async with repository.transaction() as transaction:
        record = await transaction.get_workload(undispatched.id)
        await transaction.put_workload(record.model_copy(update={"remote_id": "remote-1"}))

    text = await controller.workload_logs(undispatched.id)
    assert "released" in text


@pytest.mark.asyncio
async def test_a_slow_command_does_not_block_delivery_for_unrelated_cells() -> None:
    """One preemption's checkpoint used to stall every mutation on the replica.

    Delivery was strictly serial, and a command can legitimately take minutes: a
    preemption checkpoints a real training step and waits for it. Everything
    behind it queued -- capacity applies, dispatches, cancellations -- so one slow
    victim stalled a whole fleet's mutations. Nothing caught it because
    correctness was never in question; only throughput was.

    Ordering is still required *within* an aggregate, so this asserts the
    property that actually matters: two different cells make progress at the same
    time.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    blocking = BlockingProvisioningAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(outbox_dispatch_concurrency=4),
        agent_factory=lambda _cell: blocking,
    )
    admin = principal(Role.FLEET_ADMIN)
    for cell_id in ("cell-a", "cell-b"):
        await service.upsert_cell(
            admin,
            CellRecord(
                id=cell_id,
                provider="gcp",
                provider_project="gcp-project-a",
                region="us-central2",
                zone="us-central2-b",
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                accelerator_type="v5p-8",
                endpoint="http://agent:8090",
            ),
            idempotency_key=f"{cell_id}-create",
        )

    delivery = asyncio.create_task(controller.dispatch_outbox_once(limit=2))
    try:
        # The first apply blocks. Serially, the second cell would not be touched
        # until it returns; both should now be in flight.
        assert await asyncio.to_thread(blocking.apply_entered.wait, 2)
        for _ in range(50):
            if len(blocking.applied_cell_ids) >= 2:
                break
            await asyncio.sleep(0.02)
        assert sorted(blocking.applied_cell_ids) == ["cell-a", "cell-b"]
    finally:
        blocking.allow_apply.set()
        await delivery


async def test_dispatch_refuses_a_cell_running_a_different_build(monkeypatch) -> None:
    """A cell outliving its head must not receive work it will execute wrongly."""

    from types import SimpleNamespace

    import eyvan._internal.build_identity as build_identity_module

    monkeypatch.setattr(
        build_identity_module,
        "build_identity",
        lambda: SimpleNamespace(source_digest="a" * 64),
    )

    from eyvan.worker.agent.client import AgentBuild

    class SkewedAgent(FakeAgent):
        def build(self):
            return AgentBuild(
                version="0.0.1",
                origin="wheel",
                origin_detail="",
                source_digest="b" * 64,
                digest_detail="",
                file_count=1,
                python="3.13",
                platform="linux",
            )

        def worker_builds(self, cell_id):
            return [{"source_digest": "c" * 64}]

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = SkewedAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda _cell: fake_agent,
    )
    cell = CellRecord(
        id="cell-skewed",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
    )

    from eyvan.server.controller.workload_commands import PermanentDispatchRefusal

    with pytest.raises(PermanentDispatchRefusal) as refused:
        await controller._refuse_stale_cell_builds((cell,))
    message = str(refused.value)
    assert "bbbbbbbbbbbb" in message and "cccccccccccc" in message and "aaaaaaaaaaaa" in message
    # A refusal that says what to do without naming the commands still sends
    # the operator hunting through docs mid-incident. Both verbs and the cell
    # id must ride in the message itself, ready to paste.
    assert "eyvan cell-drain cell-skewed" in message
    assert "eyvan cell-delete cell-skewed" in message
    assert fake_agent.submissions == []


async def test_a_matching_cell_build_dispatches(monkeypatch) -> None:
    """The guard reads builds but never refuses an honest fleet."""

    from types import SimpleNamespace

    import eyvan._internal.build_identity as build_identity_module

    monkeypatch.setattr(
        build_identity_module,
        "build_identity",
        lambda: SimpleNamespace(source_digest="a" * 64),
    )

    from eyvan.worker.agent.client import AgentBuild

    class HonestAgent(FakeAgent):
        def build(self):
            return AgentBuild(
                version="0.0.1",
                origin="wheel",
                origin_detail="",
                source_digest="a" * 64,
                digest_detail="",
                file_count=1,
                python="3.13",
                platform="linux",
            )

        def worker_builds(self, cell_id):
            return [{"source_digest": "a" * 64}]

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda _cell: HonestAgent(),
    )
    cell = CellRecord(
        id="cell-honest",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
    )
    await controller._refuse_stale_cell_builds((cell,))


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow


def _denial_harness():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={"detail": "GCP queued-resource admission denied", "code": "queue_admission_denied"},
        )

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    agent = CellAgentClient(
        "http://agent",
        client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)),
    )
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    return repository, service, controller, agent


def _denial_pool(fleet: str, zone: str) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=f"{fleet}/v6e-16-{zone}",
        fleet_name=fleet,
        node_type=f"v6e-16-{zone}",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-central1",
        zone=zone,
        capacity=ResourceVector(cpu=64, tpu=16),
        allocatable=ResourceVector(cpu=64, tpu=16),
        accelerator_type="v6e-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=0,
        max_cells=4,
    )


async def _submit_denied_callable(service, fleet: str, key: str):
    workload, _ = await service.submit_pool_callable(
        principal(Role.PROJECT_OPERATOR),
        fleet_name=fleet,
        org_id="acme",
        project_id="research",
        accelerator_type="v6e-16",
        pods=1,
        execution=CallableExecution(artifact_uri="gs://trusted/train.pkl", sha256="a" * 64, python="3.12"),
        idempotency_key=key,
    )
    return workload


@pytest.mark.asyncio
async def test_an_unrelated_fleets_identical_pool_does_not_keep_a_denied_workload_waiting() -> None:
    """A workload pinned to fleet A's pool can never use fleet B's.

    Alternative pools must be usable by this workload, not merely match its
    shape and region, so an unrelated fleet cannot suppress the refusal verdict.
    """

    repository, service, controller, agent = _denial_harness()
    await service.apply_fleet(
        principal(Role.FLEET_ADMIN), "research", (_denial_pool("research", "us-central1-c"),), idempotency_key="a"
    )
    await service.apply_fleet(
        principal(Role.FLEET_ADMIN), "other", (_denial_pool("other", "us-central1-c"),), idempotency_key="b"
    )
    workload = await _submit_denied_callable(service, "research", "pinned-denied")

    assert (await controller.reconcile_capacity_pools_once())["launched"] == 1
    await controller.dispatch_outbox_once()

    workloads, _, _, _ = await snapshot(repository)
    current = next(candidate for candidate in workloads if candidate.id == workload.id)
    assert current.state is WorkloadState.FAILED
    assert current.failure_reason == "GCP queued-resource admission denied"
    agent.close()


@pytest.mark.asyncio
async def test_a_cross_zone_workload_denied_in_every_zone_is_failed_not_left_waiting() -> None:
    """Two candidates that have *both* refused must not vouch for each other."""

    repository, service, controller, agent = _denial_harness()
    await service.apply_fleet(
        principal(Role.FLEET_ADMIN),
        "research",
        (_denial_pool("research", "us-central1-b"), _denial_pool("research", "us-central1-c")),
        idempotency_key="a",
    )
    workload = await _submit_denied_callable(service, "research", "cross-zone-denied")

    for _ in range(4):
        await controller.reconcile_capacity_pools_once()
        await controller.dispatch_outbox_once()
        workloads, _, _, _ = await snapshot(repository)
        current = next(candidate for candidate in workloads if candidate.id == workload.id)
        if current.state is WorkloadState.FAILED:
            break

    assert current.state is WorkloadState.FAILED
    assert current.autoscale_block_reason is AutoscaleBlockReason.QUEUE_ADMISSION_DENIED
    _, cells, _, _ = await snapshot(repository)
    denied_zones = sorted(
        cell.zone for cell in cells if cell.provider_failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
    )
    assert denied_zones == ["us-central1-b", "us-central1-c"], "both zones were asked before giving up"
    agent.close()


def _racing_harness(race_width: int, denied_zones: frozenset[str] = frozenset()):
    import dataclasses

    from eyvan.config import EyvanSettings

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if any(zone in body for zone in denied_zones):
            return httpx.Response(
                502, json={"detail": "GCP queued-resource admission denied", "code": "queue_admission_denied"}
            )
        return httpx.Response(202, json={"cell_id": "x", "state": "waiting", "generation": 1})

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    agent = CellAgentClient(
        "http://agent", client=httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    )
    settings = dataclasses.replace(EyvanSettings.from_env(), capacity_race_width=race_width)
    controller = EyvanController(repository, service, settings=settings, agent_factory=lambda _cell: agent)
    return repository, service, controller, agent


async def _live_zones(repository) -> list[str]:
    _, cells, _, _ = await snapshot(repository)
    return sorted(cell.zone for cell in cells if cell.state is not CellState.DELETED)


@pytest.mark.asyncio
async def test_a_race_keeps_its_width_while_the_first_zone_is_still_only_a_request() -> None:
    """Keep the configured race width until a zone grants capacity, not merely accepts a request."""

    zones = ("us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f")
    repository, service, controller, agent = _racing_harness(3, denied_zones=frozenset({"us-central1-b"}))
    await service.apply_fleet(
        principal(Role.FLEET_ADMIN),
        "research",
        tuple(_denial_pool("research", zone) for zone in zones),
        idempotency_key="a",
    )
    await _submit_denied_callable(service, "research", "racer")

    await controller.reconcile_capacity_pools_once()
    assert await _live_zones(repository) == ["us-central1-a", "us-central1-b", "us-central1-c"]
    while await controller.dispatch_outbox_once():
        pass

    # us-central1-b refused; the race tops back up to three zones with the one not yet asked.
    await controller.reconcile_capacity_pools_once()
    await controller.reconcile_capacity_pools_once()
    live = await _live_zones(repository)
    assert "us-central1-f" in live
    assert live.count("us-central1-a") == 1 and live.count("us-central1-c") == 1, (
        "no second cell in a zone already asked"
    )
    agent.close()


@pytest.mark.asyncio
async def test_a_launch_the_cell_rejects_as_invalid_fails_the_attempt_with_its_reason() -> None:
    """Fail an invalid launch with the worker's reason and release its lease rather than retry validation."""

    from eyvan.worker.agent.client import CellAgentInvalidConfigurationError

    class RefusingAgent(FakeAgent):
        def submit(self, cell_id, request):
            self.submissions.append((cell_id, request))
            raise CellAgentInvalidConfigurationError(
                "native-systemd environment variable 'EXAMPLE' must not contain a newline"
            )

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = RefusingAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)  # type: ignore[arg-type]
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
            )
        )
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="refused",
            execution=CommandExecution(command="true"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
        ),
        idempotency_key="refused-launch",
    )

    await controller.reconcile_once()
    await controller.dispatch_outbox_once()

    async with repository.transaction() as transaction:
        settled = await transaction.get_workload(workload.id)
        leases = await transaction.list_leases()

    assert settled.state is WorkloadState.FAILED
    assert "refused this launch as invalid" in (settled.failure_reason or "")
    assert "must not contain a newline" in (settled.failure_reason or "")
    assert [lease for lease in leases if lease.state is LeaseState.PENDING] == []
    assert len(agent.submissions) == 1
