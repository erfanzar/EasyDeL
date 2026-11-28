# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cancellation must recover an ambiguous admission before releasing its leases."""

from datetime import timedelta

import pytest
from test_controller import principal

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxState,
    ResourceGang,
    ResourceVector,
    Role,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.service import ControlPlaneService
from eyvan.worker.agent.client import CellAgentUnavailableError
from eyvan.worker.agent.models import WorkloadObservation


class AdmittedResponseLoss:
    """Model an admitted execution independently of the controller's durable state."""

    def __init__(self, error_type):
        self.error_type = error_type
        self.requests = []
        self.running = set()
        self.launches = 0

    def submit(self, cell_id, request):
        self.requests.append(request)
        remote_id = request.spec["submission_id"]
        if len(self.requests) == 1:
            self.running.add(remote_id)
            self.launches += 1
            raise self.error_type("response lost after remote admission")
        assert remote_id in self.running, "recovery must address the original admission"
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.RUNNING)

    submit_gang = submit

    def stop(self, cell_id, remote_id, request):
        self.running.remove(remote_id)
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPED)

    def stop_gang(self, cell_id, workload_id, remote_id, request):
        return self.stop(cell_id, remote_id, request)


async def _admit(repository, members):
    service = ControlPlaneService(repository)
    operator = principal(Role.PROJECT_OPERATOR)
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="ambiguous-admission",
            execution=CommandExecution(command="true"),
            required_executor="eyvan-native",
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=0)),
            cell_group_count=members,
        ),
        idempotency_key="submit",
    )
    cells = tuple(
        CellRecord(
            id=f"cell-{index}",
            provider="gcp",
            provider_project="test-project",
            region="us-central2",
            zone="us-central2-b",
            endpoint="http://agent",
            executor="eyvan-native",
            state=CellState.READY,
            capacity=workload.spec.gang.per_replica,
            allocatable=workload.spec.gang.per_replica,
        )
        for index in range(members)
    )
    async with repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)
    leases = tuple(
        LeaseRecord(
            id=f"lease-{index}",
            workload_id=workload.id,
            cell_id=cell.id,
            org_id=workload.org_id,
            project_id=workload.project_id,
            resources=workload.spec.gang,
            generation=workload.generation,
            cell_generation=cell.generation,
            fencing_token=f"fence-{index}",
            group_id="admission-group" if members > 1 else None,
            group_index=index,
            expires_at=utcnow() + timedelta(hours=1),
        )
        for index, cell in enumerate(cells)
    )
    admin = principal(Role.FLEET_ADMIN)
    if members > 1:
        await service.create_lease_group(admin, leases, idempotency_key="lease")
    else:
        await service.create_lease(admin, leases[0], idempotency_key="lease")
    return service, operator, workload, leases


@pytest.mark.parametrize("backend", ["memory", "local"])
@pytest.mark.parametrize("members", [1, 2], ids=["single", "gang"])
@pytest.mark.parametrize("error_type", [CellAgentUnavailableError, RuntimeError])
async def test_cancel_during_ambiguous_dispatch_backoff_retains_execution_ownership(
    tmp_path, backend, members, error_type
):
    repository = InMemoryRepository() if backend == "memory" else LocalJsonRepository(tmp_path / "state.json")
    await repository.open()
    try:
        service, operator, workload, leases = await _admit(repository, members)
        agent = AdmittedResponseLoss(error_type)

        def controller():
            return EyvanController(repository, ControlPlaneService(repository), agent_factory=lambda _cell: agent)

        assert await controller().dispatch_outbox_once() == 1
        async with repository.transaction() as tx:
            dispatch = next(command for command in await tx.list_outbox() if command.topic == "workload.dispatch")
        assert dispatch.state is OutboxState.FAILED
        assert agent.running and agent.launches == 1
        await service.cancel_workload(operator, workload.id, idempotency_key="cancel")
        # Make the recovery unavailable independently of elapsed test time. Only
        # the ready cancellation can be claimed in this next delivery pass.
        async with repository.transaction() as tx:
            await tx.put_outbox(dispatch.model_copy(update={"available_at": utcnow() + timedelta(hours=1)}))
        assert await controller().dispatch_outbox_once() == 1
        async with repository.transaction() as tx:
            current = await tx.get_workload(workload.id)
            held = tuple([await tx.get_lease(lease.id) for lease in leases])
            cancel = next(command for command in await tx.list_outbox() if command.topic == "workload.cancel")
        assert current.state is WorkloadState.CANCELLING, "lost admission response must not permit terminal settlement"
        assert all(lease.state is LeaseState.PENDING and lease.settled_at is None for lease in held)
        assert cancel.state is OutboxState.FAILED
        reservation = current.dispatch_reservation
        assert reservation is not None and agent.running
        assert dispatch.payload["dispatch_reservation"] == reservation.model_dump(mode="json")
        # Reclaim with a new controller, recovering the same remote execution and
        # capability fence while cancellation stays in backoff.
        async with repository.transaction() as tx:
            await tx.put_outbox(dispatch.model_copy(update={"available_at": utcnow() - timedelta(seconds=1)}))
            await tx.put_outbox(cancel.model_copy(update={"available_at": utcnow() + timedelta(hours=1)}))
        assert await controller().dispatch_outbox_once() == 1
        async with repository.transaction() as tx:
            recovered = await tx.get_workload(workload.id)
            active = tuple([await tx.get_lease(lease.id) for lease in leases])
        assert recovered.state is WorkloadState.CANCELLING
        assert recovered.remote_id in agent.running
        assert recovered.dispatch_reservation.fence == reservation.fence, (
            "recovery must not revoke the admitted capability"
        )
        assert all(lease.state is LeaseState.ACTIVE for lease in active)
        assert agent.launches == 1
        async with repository.transaction() as tx:
            await tx.put_outbox(cancel.model_copy(update={"available_at": utcnow() - timedelta(seconds=1)}))
        assert await controller().dispatch_outbox_once() == 1
        async with repository.transaction() as tx:
            terminal = await tx.get_workload(workload.id)
            released = tuple([await tx.get_lease(lease.id) for lease in leases])
            events = await tx.list_events()
        assert terminal.state is WorkloadState.CANCELLED
        assert all(lease.state is LeaseState.RELEASED and lease.settled_at is not None for lease in released)
        assert not agent.running
        assert sum(event.action == "workload.cancelled" for event in events) == 1
    finally:
        await repository.close()


async def test_original_secret_capability_survives_ambiguous_dispatch_recovery(tmp_path):
    from test_secret_dispatch_transactions import _plane, _seed_dispatch

    repository = InMemoryRepository()
    service, signer, agent, controller = await _plane(repository, tmp_path, fail=True)
    seeded, _leases, command = await _seed_dispatch(repository)
    assert await controller.dispatch_outbox_once() == 1
    original_token = agent.tokens[0]
    claims = signer.verify(original_token, workload_id=seeded.id, org_id="acme", project_id="research")
    async with repository.transaction() as tx:
        failed = await tx.get_outbox(command.id)
        await tx.put_outbox(failed.model_copy(update={"available_at": utcnow() - timedelta(seconds=1)}))
    agent.fail_after_materialization = False
    assert await controller.dispatch_outbox_once() == 1
    # Validate the bytes issued to the original admission, not merely equality
    # of request arguments: the authorization layer must still accept them.
    assert await service.authorize_secret_mount(
        org_id=claims.org_id,
        project_id=claims.project_id,
        workload_id=claims.workload_id,
        attempt=claims.attempt,
        generation=claims.generation,
        dispatch_fence=claims.dispatch_fence,
        name=claims.name,
    ) == ("file:///operator/secrets/hf-token", "file:///operator/secrets")


async def test_response_loss_cancellation_stops_a_real_local_child(tmp_path, monkeypatch):
    import asyncio
    import time

    from fastapi.testclient import TestClient
    from test_agent import FakeCapacity, StaticResolver
    from test_dispatch_response_loss import _admit_response_workload, _owned_admissions, _pid_exists
    from test_native_remote_worker import wait_for

    from eyvan.capacity.drivers.base import MutationContext
    from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
    from eyvan.config import EyvanSettings
    from eyvan.worker.agent.app import create_agent_app
    from eyvan.worker.agent.client import CellAgentClient
    from eyvan.worker.agent.service import CellAgentService
    from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
    from eyvan.worker.runtime.native_worker import create_native_worker_app

    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "1")
    repository = InMemoryRepository()
    pid_file = tmp_path / "child.pid"
    source = (
        f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    workload, _cell, lease = await _admit_response_workload(repository, source)
    cleanup = MutationContext(idempotency_key="cleanup", generation=lease.generation, fencing_token=lease.fencing_token)
    worker_app = create_native_worker_app(state_dir=tmp_path / "native", secret="worker-test-secret")
    with TestClient(worker_app) as worker_http:
        native = NativeRemoteExecutor("http://testserver", secret="worker-test-secret", client=worker_http)
        settings = EyvanSettings(agent_shared_secret="agent-test-secret")
        agent_service = CellAgentService(
            capacity=FakeCapacity(),
            state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
            executors=StaticResolver(native),
            settings=settings,
        )
        with _owned_admissions(create_agent_app(agent_service, settings), native, cleanup, lose=True) as transport:
            client = CellAgentClient("http://agent", client=transport)
            controller = EyvanController(repository, ControlPlaneService(repository), agent_factory=lambda _cell: client)
            assert await controller.dispatch_outbox_once() == 1
            remote_id = transport.admitted[0]
            await asyncio.to_thread(wait_for, native, remote_id, {ExecutionWorkloadStatus.RUNNING})
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert pid_file.exists()
            pid = int(pid_file.read_text())
            assert _pid_exists(pid)
            await controller.service.cancel_workload(
                principal(Role.PROJECT_OPERATOR), workload.id, idempotency_key="cancel"
            )
            async with repository.transaction() as tx:
                dispatch = next(command for command in await tx.list_outbox() if command.topic == "workload.dispatch")
                await tx.put_outbox(dispatch.model_copy(update={"available_at": utcnow() + timedelta(hours=1)}))
            assert await controller.dispatch_outbox_once() == 1
            async with repository.transaction() as tx:
                current = await tx.get_workload(workload.id)
                held = await tx.get_lease(lease.id)
                cancel = next(command for command in await tx.list_outbox() if command.topic == "workload.cancel")
            assert current.state is WorkloadState.CANCELLING
            assert held.state is LeaseState.PENDING and held.settled_at is None
            assert native.status(remote_id).status is ExecutionWorkloadStatus.RUNNING and _pid_exists(pid)
            transport.lose = False
            async with repository.transaction() as tx:
                await tx.put_outbox(dispatch.model_copy(update={"available_at": utcnow() - timedelta(seconds=1)}))
                await tx.put_outbox(cancel.model_copy(update={"available_at": utcnow() + timedelta(hours=1)}))
            assert await controller.dispatch_outbox_once() == 1
            async with repository.transaction() as tx:
                await tx.put_outbox(cancel.model_copy(update={"available_at": utcnow() - timedelta(seconds=1)}))
            assert await controller.dispatch_outbox_once() == 1
            async with repository.transaction() as tx:
                terminal = await tx.get_workload(workload.id)
                released = await tx.get_lease(lease.id)
            assert terminal.state is WorkloadState.CANCELLED and released.state is LeaseState.RELEASED
            await asyncio.to_thread(wait_for, native, remote_id, {ExecutionWorkloadStatus.STOPPED})
            deadline = time.monotonic() + 2
            while _pid_exists(pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert not _pid_exists(pid), "terminal cancellation must not leave the admitted child alive"
