# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execution ownership must survive a delayed controller observation."""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta

import pytest
from test_controller import GangFakeAgent, principal
from test_node_health_replacement import TPU_INIT, FailingAgent, _attempt, _cell, _host_fault_events

from eyvan.capacity import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellState,
    CommandExecution,
    LeaseState,
    PriorityBand,
    Role,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.runtime.node_health import HOST_FAULT_HOST_QUARANTINED, HostReplacementBudget


class BlockingFaultAgent(FailingAgent):
    def __init__(self) -> None:
        super().__init__(TPU_INIT)
        self.started = threading.Event()
        self.release = threading.Event()

    def workload(self, cell_id, remote_id):
        self.started.set()
        assert self.release.wait(10), "test did not release observation"
        return super().workload(cell_id, remote_id)


async def _settle_and_replace(controller, *, replace: bool) -> None:
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
    await controller.service.settle_terminal_workload(
        controller._principal,
        workload.id,
        WorkloadState.SUCCEEDED,
        action="workload.succeeded",
        expected_generation=workload.generation,
        expected_state=workload.state,
        expected_lease_id=workload.lease_id,
    )
    if replace:
        async with controller.repository.transaction() as transaction:
            cell = await transaction.get_cell("cell-a")
            await transaction.put_cell(
                cell.model_copy(update={"state": CellState.UNAVAILABLE, "provider_phase": CapacityStatus.NEEDS_RECREATE})
            )
        await controller.service.recreate_cell(
            controller._principal, "cell-a", expected_generation=1, idempotency_key="replacement"
        )
    else:
        successor_workload, _ = await controller.service.submit_workload(
            principal(Role.PROJECT_OPERATOR),
            org_id="acme",
            project_id="research",
            spec=WorkloadDefinition(
                name="healthy-successor",
                execution=CommandExecution(command="echo successor"),
                gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            ),
            idempotency_key="healthy-successor",
        )
        successor = EyvanController(
            controller.repository,
            controller.service,
            settings=controller.settings,
            agent_factory=controller._agent_factory,
        )
        assert (await successor.reconcile_once()).placements
        await successor.dispatch_outbox_once()
        async with controller.repository.transaction() as transaction:
            current = await transaction.get_workload(successor_workload.id)
            current_lease = await transaction.get_lease(current.lease_id)
        assert current.remote_id
        assert current_lease.state is LeaseState.ACTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("replace", [False, True], ids=["settled-same-machine", "recreated-machine"])
@pytest.mark.parametrize("pause", ["rpc", "attribution"])
async def test_stale_host_fault_never_quarantines_settled_or_recreated_capacity(replace, pause, monkeypatch) -> None:
    agent = BlockingFaultAgent()
    controller = await _attempt(agent, [_cell("cell-a")], budget=HostReplacementBudget(max_replacements=2))
    if pause == "rpc":
        observation = asyncio.create_task(controller.observe_workloads_once())
        try:
            assert await asyncio.to_thread(agent.started.wait, 5)
            await _settle_and_replace(controller, replace=replace)
        finally:
            agent.release.set()
        await observation
    else:
        agent.release.set()
        original = controller._quarantine_faulted_cell

        async def race(*args, **kwargs):
            await _settle_and_replace(controller, replace=replace)
            return await original(*args, **kwargs)

        monkeypatch.setattr(controller, "_quarantine_faulted_cell", race)
        await controller.observe_workloads_once()
    async with controller.repository.transaction() as transaction:
        cell = await transaction.get_cell("cell-a")
        workload = await transaction.get_workload("wl-1")
    assert cell.generation == (2 if replace else 1)
    assert cell.state is (CellState.REGISTERING if replace else CellState.READY)
    assert cell.unavailable_reason is None
    assert workload.state is WorkloadState.SUCCEEDED
    assert not [event for event in await _host_fault_events(controller) if event.action == HOST_FAULT_HOST_QUARANTINED]


class BlockingHealthyAgent(GangFakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def _wait(self):
        if self.block:
            self.started.set()
            assert self.release.wait(10), "test did not release observation"

    def workload(self, cell_id, remote_id):
        self._wait()
        return super().workload(cell_id, remote_id)

    def gang_workload(self, cell_id, workload_id, remote_id):
        self._wait()
        return super().gang_workload(cell_id, workload_id, remote_id)


async def _public_batch_attempt(agent, *, group_count):
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(
        repository, service, settings=EyvanSettings(checkpoint_grace_s=2), agent_factory=lambda _cell: agent
    )
    async with repository.transaction() as transaction:
        for index in range(group_count):
            await transaction.put_cell(_cell(f"cell-{index}").model_copy(update={"executor": "eyvan-native"}))
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch",
            required_executor="eyvan-native",
            execution=CommandExecution(command="python batch.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
            cell_group_count=group_count,
            priority_band=PriorityBand.BATCH,
            checkpointable=True,
            checkpoint_hook="python checkpoint.py",
            checkpoint_output_uri="gs://checkpoints/batch",
        ),
        idempotency_key="batch",
    )
    assert (await controller.reconcile_once()).placements
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    async with repository.transaction() as transaction:
        current = await transaction.get_workload(workload.id)
        assert current.state is WorkloadState.RUNNING
        leases = await transaction.list_leases()
        assert len(leases) == group_count
        expired = utcnow() - timedelta(seconds=10)
        for lease in leases:
            await transaction.put_lease(lease.model_copy(update={"expires_at": expired}))
    assert await controller.quarantine_expired_leases_once() == group_count
    return controller, workload.id, expired


@pytest.mark.asyncio
@pytest.mark.parametrize("group_count", [1, 2], ids=["leader", "leader-and-sibling"])
@pytest.mark.parametrize("expire_preemption", [False, True], ids=["revoking-control", "requarantined"])
async def test_delayed_healthy_observation_cannot_recover_preemption_owned_leases(
    group_count, expire_preemption
) -> None:
    agent = BlockingHealthyAgent()
    controller, workload_id, expired = await _public_batch_attempt(agent, group_count=group_count)
    agent.block = True
    observation = asyncio.create_task(controller.observe_workloads_once())
    try:
        assert await asyncio.to_thread(agent.started.wait, 5)
        successor = EyvanController(
            controller.repository, controller.service, settings=controller.settings, agent_factory=lambda _cell: agent
        )
        high, _ = await controller.service.submit_workload(
            principal(Role.PROJECT_OPERATOR),
            org_id="acme",
            project_id="research",
            spec=WorkloadDefinition(
                name="urgent",
                execution=CommandExecution(command="python urgent.py"),
                gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
                priority_band=PriorityBand.PRODUCTION,
            ),
            idempotency_key="urgent",
        )
        assert (await successor.reconcile_once()).preemptions
        if expire_preemption:
            assert await successor.quarantine_expired_leases_once() == group_count
    finally:
        agent.release.set()
    await observation
    async with controller.repository.transaction() as transaction:
        current = await transaction.get_workload(workload_id)
        leases = await transaction.list_leases()
        recovered = await transaction.list_events(entity_type="lease", limit=100)
    assert current.state is WorkloadState.CHECKPOINTING
    assert {lease.state for lease in leases} == {LeaseState.QUARANTINED if expire_preemption else LeaseState.REVOKING}
    assert {lease.revocation_reason for lease in leases} == {"preemption"}
    assert {lease.expires_at for lease in leases} == {expired}
    assert not [event for event in recovered if event.action == "lease.recovered"]
    # The real plan remains rollback-eligible, rather than stranding its victim.
    await controller.service.cancel_workload(principal(Role.PROJECT_OPERATOR), high.id, idempotency_key="cancel-urgent")
    agent.block = False
    await successor.dispatch_outbox_once()
    async with controller.repository.transaction() as transaction:
        current = await transaction.get_workload(workload_id)
        leases = await transaction.list_leases()
        high = await transaction.get_workload(high.id)
    assert current.state is WorkloadState.RUNNING
    assert high.state is WorkloadState.CANCELLED
    assert {lease.revocation_reason for lease in leases} == {None}


@pytest.mark.asyncio
@pytest.mark.parametrize("group_count", [1, 2])
async def test_unchanged_healthy_attempt_recovers_and_renews_quarantine(group_count) -> None:
    controller, workload_id, expired = await _public_batch_attempt(BlockingHealthyAgent(), group_count=group_count)
    await controller.observe_workloads_once()
    async with controller.repository.transaction() as transaction:
        leases = await transaction.list_leases()
        events = await transaction.list_events(entity_type="lease", limit=100)
        workload = await transaction.get_workload(workload_id)
    assert workload.state is WorkloadState.RUNNING
    assert {lease.state for lease in leases} == {LeaseState.ACTIVE}
    assert all(lease.expires_at > expired for lease in leases)
    assert len([event for event in events if event.action == "lease.recovered"]) == group_count


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["no-lease-ids", "lease-row", "cell-row"])
async def test_process_status_of_unhosted_attempt_is_a_readable_http_diagnostic(missing, tmp_path) -> None:
    import httpx

    from eyvan.records import LeaseRecord, WorkloadRecord
    from eyvan.server.api.app import create_app

    repository = InMemoryRepository()
    app = create_app(
        repository=repository,
        settings=EyvanSettings(
            dev_auth_enabled=True, artifact_token_key="k" * 32, artifact_store_uri=str(tmp_path / "artifacts")
        ),
    )
    workload = WorkloadRecord(
        id="wl-orphan",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="orphan", execution=CommandExecution(command="echo hi")),
        created_by="alice",
        state=WorkloadState.RUNNING,
        remote_id="retained-remote",
        lease_id=None if missing == "no-lease-ids" else "missing-lease",
        updated_at=utcnow() - timedelta(hours=1),
    )
    async with repository.transaction() as transaction:
        await transaction.put_workload(workload)
        if missing == "cell-row":
            await transaction.put_lease(
                LeaseRecord(
                    id="missing-lease",
                    workload_id=workload.id,
                    cell_id="missing-cell",
                    org_id="acme",
                    project_id="research",
                    state=LeaseState.ACTIVE,
                    resources=workload.spec.gang,
                    fencing_token="historical-fence",
                    expires_at=utcnow() + timedelta(minutes=5),
                )
            )
    if missing != "no-lease-ids":
        assert await app.state.controller.settle_orphaned_workloads_once() == 1
        async with repository.transaction() as transaction:
            settled = await transaction.get_workload(workload.id)
        assert settled.state is WorkloadState.FAILED
        assert settled.remote_id == "retained-remote"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/workloads/{workload.id}/process-status",
            headers={
                "X-Eyvan-User": "alice",
                "X-Eyvan-Roles": "viewer",
                "X-Eyvan-Org": "acme",
                "X-Eyvan-Project": "research",
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["observed"] is False
    assert response.json()["detail"] == "workload no longer holds a cell"


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse", ["successor", "replacement"])
async def test_delayed_unreadable_execution_does_not_drain_healthy_successor(reuse) -> None:
    from eyvan.capacity.drivers.base import ExecutionWorkloadStatus

    agent = BlockingHealthyAgent()
    controller, workload_id, _expired = await _public_batch_attempt(agent, group_count=1)
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload(workload_id)
    controller._indeterminate_since[(workload.id, workload.generation)] = utcnow() - timedelta(seconds=901)
    agent.remote_status = ExecutionWorkloadStatus.INDETERMINATE
    agent.block = True
    observation = asyncio.create_task(controller.observe_workloads_once())
    try:
        assert await asyncio.to_thread(agent.started.wait, 5)
        await controller.service.settle_terminal_workload(
            controller._principal,
            workload.id,
            WorkloadState.SUCCEEDED,
            action="workload.succeeded",
            expected_generation=workload.generation,
            expected_state=workload.state,
            expected_lease_id=workload.lease_id,
        )
        successor = EyvanController(
            controller.repository, controller.service, settings=controller.settings, agent_factory=lambda _cell: agent
        )
        if reuse == "replacement":
            async with controller.repository.transaction() as transaction:
                cell = await transaction.get_cell("cell-0")
                await transaction.put_cell(
                    cell.model_copy(
                        update={"state": CellState.UNAVAILABLE, "provider_phase": CapacityStatus.NEEDS_RECREATE}
                    )
                )
            await controller.service.recreate_cell(
                controller._principal, "cell-0", expected_generation=1, idempotency_key="replace-unreadable"
            )
            await successor.dispatch_outbox_once()
            await successor.observe_cells_once()
        else:
            successor_workload, _ = await controller.service.submit_workload(
                principal(Role.PROJECT_OPERATOR),
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(
                    name="successor",
                    required_executor="eyvan-native",
                    execution=CommandExecution(command="echo successor"),
                    gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
                ),
                idempotency_key="successor",
            )
            assert (await successor.reconcile_once()).placements
            await successor.dispatch_outbox_once()
            async with controller.repository.transaction() as transaction:
                current_successor = await transaction.get_workload(successor_workload.id)
                successor_lease = await transaction.get_lease(current_successor.lease_id)
            assert current_successor.remote_id
            assert successor_lease.state is LeaseState.ACTIVE
        async with controller.repository.transaction() as transaction:
            before = await transaction.get_cell("cell-0")
        assert before.state is CellState.READY
    finally:
        agent.release.set()
    await observation
    async with controller.repository.transaction() as transaction:
        cell = await transaction.get_cell("cell-0")
        commands = await transaction.list_outbox()
    assert cell.generation == (2 if reuse == "replacement" else 1)
    assert cell.state is CellState.READY
    assert not [command for command in commands if command.topic == "capacity.drain"]
    assert (workload.id, workload.generation) not in controller._indeterminate_since


class RecordingExecAgent:
    def __init__(self, cell_ids):
        from eyvan.worker.runtime.native_slice import NativeSliceExecutor

        self.calls = []
        self.executors = {}
        for cell_id in cell_ids:

            class Worker:
                def __init__(self, recorder, owner, local_rank):
                    self.recorder = recorder
                    self.owner = owner
                    self.local_rank = local_rank

                def exec_in_workload(self, remote_id, command, *, timeout_s):
                    self.recorder.calls.append((self.owner, self.local_rank, tuple(command)))
                    return {"exit_code": 0, "stdout": "local recording worker"}

            executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
            executor.endpoints = ("http://recording-worker-0", "http://recording-worker-1")
            executor._workers = (Worker(self, cell_id, 0), Worker(self, cell_id, 1))
            self.executors[cell_id] = executor

    def exec_in_workload(self, cell_id, remote_id, command, *, timeout_s, **kwargs):
        return self.executors[cell_id].exec_in_workload(remote_id, command, timeout_s=timeout_s, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("cell_count,rank", [(1, 0), (1, 1), (2, 1)])
async def test_exec_selects_one_global_runtime_rank_not_one_cell(cell_count, rank) -> None:
    from test_attempt_topology import _running_attempt, _tpu_cell

    cell_ids = [f"cell-{index}" for index in range(cell_count)]
    agent = RecordingExecAgent(cell_ids)
    controller = await _running_attempt(agent, [_tpu_cell(cell_id, 2) for cell_id in cell_ids])
    document = await controller.workload_exec("wl-1", ["id"], rank=rank)
    assert agent.calls == [("cell-0", rank, ("id",))]
    assert document["results"][0]["rank"] == rank


@pytest.mark.asyncio
async def test_exec_without_rank_preserves_all_inner_runtime_hosts() -> None:
    from test_attempt_topology import _running_attempt, _tpu_cell

    agent = RecordingExecAgent(["cell-0", "cell-1"])
    controller = await _running_attempt(agent, [_tpu_cell("cell-0", 2), _tpu_cell("cell-1", 2)])
    await controller.workload_exec("wl-1", ["id"])
    assert agent.calls == [(cell, rank, ("id",)) for cell in ("cell-0", "cell-1") for rank in (0, 1)]


class NonDestructiveDrainAgent(BlockingHealthyAgent):
    """Draining leaves the execution alive; only release proves simulated host loss."""

    def drain(self, cell_id, request):
        self.capacity_state = CapacityStatus.DRAINING
        return super().drain(cell_id, request)

    def release(self, cell_id, request):
        self.capacity_state = CapacityStatus.RELEASED
        return super().release(cell_id, request)


@pytest.mark.asyncio
async def test_unreadable_escalation_keeps_leases_until_host_loss_is_observed() -> None:
    from eyvan.capacity.drivers.base import ExecutionWorkloadStatus

    agent = NonDestructiveDrainAgent()
    controller, workload_id, _expired = await _public_batch_attempt(agent, group_count=1)
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload(workload_id)
        before_leases = await transaction.list_leases()
    controller._indeterminate_since[(workload.id, workload.generation)] = utcnow() - timedelta(seconds=901)
    agent.remote_status = ExecutionWorkloadStatus.INDETERMINATE
    await controller.observe_workloads_once()
    await controller.dispatch_outbox_once()
    assert len(agent.drains) == 1
    for _ in range(4):
        await controller.observe_workloads_once()
        await controller.observe_cells_once()
        await controller.quarantine_expired_leases_once()
        await controller.settle_orphaned_workloads_once()
        await controller.settle_orphaned_leases_once()
        await controller.reconcile_unmanaged_capacity_recoveries_once()
        await controller.dispatch_outbox_once()
    async with controller.repository.transaction() as transaction:
        current = await transaction.get_workload(workload_id)
        leases = await transaction.list_leases()
        cell = await transaction.get_cell("cell-0")
    # A non-destructive drain is not evidence of host loss. Unknown execution
    # remains fenced rather than being released to manufacture reusable capacity.
    with pytest.raises(ConflictError, match="active lease"):
        await controller.delete_cell(controller._principal, cell.id, idempotency_key="unsafe-generic-delete")
    assert agent.releases == []
    assert agent.capacity_state is CapacityStatus.DRAINING
    assert cell.state is CellState.DRAINING
    assert current.state is WorkloadState.RUNNING
    assert current.generation == workload.generation
    assert current.lease_id == workload.lease_id
    assert current.remote_id == workload.remote_id
    assert {lease.state for lease in leases} == {LeaseState.QUARANTINED}
    assert [(lease.id, lease.fencing_token, lease.expires_at) for lease in leases] == [
        (lease.id, lease.fencing_token, lease.expires_at) for lease in before_leases
    ]
