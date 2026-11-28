# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lost admission responses across real agent/native HTTP handlers and local execution."""

import asyncio
import os
import shlex
import sys
import time
from contextlib import contextmanager
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from test_agent import FakeCapacity, StaticResolver
from test_native_remote_worker import wait_for
from test_priority_preemption_properties import repository as _repository_fixture
from test_running_retry_properties import ADMIN, OPERATOR, _snapshot

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.service import ControlPlaneService
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_worker import create_native_worker_app

repository = _repository_fixture


class LoseAdmissionResponse(TestClient):
    """Run the actual handler, then withhold its successful response from the caller."""

    def __init__(self, app, *, lose):
        super().__init__(app, headers={"Authorization": "Bearer agent-test-secret"})
        self.lose = lose
        self.admitted = []

    def request(self, method, url, **kwargs):
        response = super().request(method, url, **kwargs)
        if method.upper() == "POST" and str(url).endswith("/workloads"):
            assert response.status_code == 200, response.text
            self.admitted.append(response.json()["remote_id"])
            if self.lose:
                raise httpx.ReadTimeout("deliberately lost response after admission", request=response.request)
        return response


async def _admit_response_workload(repository, source):
    resources = ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=0))
    service = ControlPlaneService(repository)
    workload, _ = await service.submit_workload(
        OPERATOR,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="response-loss",
            execution=CommandExecution(command=shlex.join([sys.executable, "-c", source])),
            gang=resources,
        ),
        idempotency_key="submit-response-loss",
    )
    cell = CellRecord(
        id="response-cell",
        provider="gcp",
        provider_project="test-project",
        region="us-central2",
        zone="us-central2-b",
        endpoint="http://agent",
        state=CellState.READY,
        capacity=resources.per_replica,
        allocatable=resources.per_replica,
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
    lease = LeaseRecord(
        id="response-lease",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id="acme",
        project_id="research",
        resources=resources,
        generation=workload.generation,
        cell_generation=cell.generation,
        fencing_token="response-fence",
        expires_at=utcnow() + timedelta(hours=1),
    )
    await service.create_lease(ADMIN, lease, idempotency_key="admit-response-loss")
    return workload, cell, lease


@pytest.mark.parametrize("losses", [1, 3])
async def test_admission_response_loss_does_not_launch_twice(repository, tmp_path, losses):
    marker = tmp_path / "executions.txt"
    source = f"from pathlib import Path; p=Path({str(marker)!r}); p.open('a').write('executed\\n')"
    workload, _cell, lease = await _admit_response_workload(repository, source)
    initial = await _snapshot(repository, workload.id)
    (command_id,) = initial["outbox"]
    admitted = []
    for delivery in range(losses + 1):
        # Recreate all three service objects against the same durable state.
        # The native worker runs a real local child; no fake executor deduplicates it.
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
            cleanup_context = MutationContext(
                idempotency_key="test-cleanup",
                generation=lease.generation,
                fencing_token=lease.fencing_token,
            )
            with _owned_admissions(
                create_agent_app(agent_service, settings),
                native,
                cleanup_context,
                lose=delivery < losses,
            ) as agent_http:
                client = CellAgentClient("http://agent", client=agent_http)
                controller = EyvanController(
                    repository, ControlPlaneService(repository), agent_factory=lambda _cell, client=client: client
                )
                assert await controller.dispatch_outbox_once() == 1
                assert len(agent_http.admitted) == 1
                admitted.extend(agent_http.admitted)
                await asyncio.to_thread(wait_for, native, admitted[-1], {ExecutionWorkloadStatus.SUCCEEDED})
                assert marker.read_text() == "executed\n"
        after = await _snapshot(repository, workload.id)
        command = after["outbox"][command_id]
        current = after["workload"]
        assert current.generation == current.attempt == 1
        assert current.failure_count == current.capacity_loss_count == current.preemption_count == 0
        assert after["cells"] == initial["cells"] and after["budget"] == initial["budget"]
        assert not any(event.action == "lease.settled" for event in after["events"])
        if delivery < losses:
            assert command.state is OutboxState.FAILED
            assert current.state is WorkloadState.LEASED and current.remote_id is None
            assert after["leases"] == initial["leases"]
            async with repository.transaction() as tx:
                await tx.put_outbox(command.model_copy(update={"available_at": utcnow() - timedelta(seconds=1)}))
        else:
            assert command.state is OutboxState.SUCCEEDED
            assert current.state is WorkloadState.DISPATCHING and current.remote_id == admitted[-1]
            assert after["leases"][lease.id].state is LeaseState.ACTIVE
            assert current.dispatch_reservation is not None
            assert current.dispatch_reservation.outbox_id == command.id
    assert len(set(admitted)) == 1
    assert marker.read_text() == "executed\n"


@pytest.fixture(autouse=True)
def bounded_owned_execution_cleanup(monkeypatch):
    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "1")


@contextmanager
def _owned_admissions(app, native, context, *, lose):
    with LoseAdmissionResponse(app, lose=lose) as transport:
        failure = None
        try:
            yield transport
        except BaseException as exc:
            failure = exc
            raise
        finally:
            for remote_id in dict.fromkeys(transport.admitted):
                try:
                    native.retire(remote_id, context)
                except Exception as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"owned execution cleanup failed: {cleanup_error}")


def _pid_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_post_admission_failure_retires_the_owned_child(tmp_path, monkeypatch):
    from test_native_remote_worker import command, spec

    from eyvan.worker.agent.models import MutationModel, WorkloadRequest

    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "1")
    pid_file = tmp_path / "child.pid"
    source = (
        f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    context = MutationContext(idempotency_key="cleanup", generation=1, fencing_token="cleanup-fence")
    app = create_native_worker_app(state_dir=tmp_path / "native", secret="worker-test-secret")
    with TestClient(app) as worker_http:
        native = NativeRemoteExecutor("http://testserver", secret="worker-test-secret", client=worker_http)
        settings = EyvanSettings(agent_shared_secret="agent-test-secret")
        service = CellAgentService(
            capacity=FakeCapacity(),
            state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
            executors=StaticResolver(native),
            settings=settings,
        )
        try:
            with pytest.raises(RuntimeError, match="post-admission assertion"):
                with _owned_admissions(create_agent_app(service, settings), native, context, lose=False) as transport:
                    client = CellAgentClient("http://agent", client=transport)
                    client.submit(
                        "cell",
                        WorkloadRequest(
                            mutation=MutationModel(
                                idempotency_key="cleanup", generation=1, fencing_token="cleanup-fence"
                            ),
                            spec=spec(command(source), submission_id="cleanup"),
                        ),
                    )
                    deadline = time.monotonic() + 8
                    while not pid_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    assert pid_file.exists(), "child failed to start"
                    assert _pid_exists(int(pid_file.read_text()))
                    raise RuntimeError("post-admission assertion")
            assert native.status("cleanup").status is ExecutionWorkloadStatus.STOPPED
            deadline = time.monotonic() + 2
            while _pid_exists(int(pid_file.read_text())) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not _pid_exists(int(pid_file.read_text()))
        finally:
            # Even the intentionally failing RED probe must not leak this child.
            native.retire("cleanup", context)
