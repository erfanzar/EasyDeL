# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A late admission response cannot overwrite a newer dispatch claim owner."""

import asyncio
import threading
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from test_agent import FakeCapacity, StaticResolver
from test_dispatch_response_loss import _admit_response_workload, _owned_admissions
from test_native_remote_worker import wait_for
from test_priority_preemption_properties import repository as _repository_fixture
from test_running_retry_properties import _snapshot

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.records import LeaseState, OutboxState, WorkloadState, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.service import ControlPlaneService
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_worker import create_native_worker_app

repository = _repository_fixture


class HoldAdmittedResponse:
    def __init__(self, client):
        self.client = client
        self.admitted = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.remote = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def submit(self, cell_id, request):
        self.remote = self.client.submit(cell_id, request)
        self.admitted.set()
        if not self.release.wait(timeout=15):
            raise RuntimeError("test admission barrier timed out")
        self.returned.set()
        return self.remote


async def _drain_dispatch(controller, delivery, timeout_s=10):
    try:
        if delivery is not None:
            await asyncio.wait_for(asyncio.gather(delivery, return_exceptions=True), timeout_s)
    finally:
        if controller.inner_dispatch is not None:
            await asyncio.wait_for(asyncio.gather(controller.inner_dispatch, return_exceptions=True), 10)


async def test_outer_cleanup_timeout_still_joins_inner_dispatch():
    outer_cancelled = asyncio.Event()
    release_inner = asyncio.Event()

    async def outer():
        try:
            await asyncio.Event().wait()
        finally:
            outer_cancelled.set()

    inner = asyncio.create_task(release_inner.wait())
    delivery = asyncio.create_task(outer())

    class Owner:
        inner_dispatch = inner

    cleanup = asyncio.create_task(_drain_dispatch(Owner(), delivery, timeout_s=0.01))
    try:
        await asyncio.wait_for(outer_cancelled.wait(), 2)
        completed, _ = await asyncio.wait((cleanup,), timeout=0.02)
        assert not completed, "cleanup must join the inner task even after outer timeout"
        assert not inner.done()
    finally:
        release_inner.set()
        results = await asyncio.wait_for(asyncio.gather(cleanup, inner, delivery, return_exceptions=True), 2)
    assert isinstance(results[0], TimeoutError), "the original cleanup timeout must remain observable"
    assert inner.done() and not inner.cancelled()


class ObservedDispatchController(EyvanController):
    """Observe completion without replacing the real dispatch/renewal protocol."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_dispatch = None
        self.inner_finished = asyncio.Event()

    async def _dispatch(self, command):
        self.inner_dispatch = asyncio.current_task()
        try:
            return await super()._dispatch(command)
        finally:
            self.inner_finished.set()


@pytest.mark.parametrize("lose_renewal", [False, True], ids=["late-response", "renewal-lost"])
async def test_late_dispatch_response_preserves_the_new_claim_owner(repository, tmp_path, monkeypatch, lose_renewal):
    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "1")
    marker = tmp_path / "executions.txt"
    finish = tmp_path / "finish"
    source = (
        "import time\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).open('a').write('executed\\n')\n"
        f"while not Path({str(finish)!r}).exists(): time.sleep(0.02)\n"
    )
    workload, _cell, lease = await _admit_response_workload(repository, source)
    initial = await _snapshot(repository, workload.id)
    (command_id,) = initial["outbox"]
    cleanup = MutationContext(
        idempotency_key="test-cleanup", generation=lease.generation, fencing_token=lease.fencing_token
    )
    worker_app = create_native_worker_app(state_dir=tmp_path / "native", secret="worker-test-secret")
    with TestClient(worker_app) as worker_http:
        native = NativeRemoteExecutor("http://testserver", secret="worker-test-secret", client=worker_http)
        settings = EyvanSettings(agent_shared_secret="agent-test-secret")

        def agent_app():
            service = CellAgentService(
                capacity=FakeCapacity(),
                state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
                executors=StaticResolver(native),
                settings=settings,
            )
            return create_agent_app(service, settings)

        # Both ownership scopes span the entire history, including the late response.
        with (
            _owned_admissions(agent_app(), native, cleanup, lose=False) as first_http,
            _owned_admissions(agent_app(), native, cleanup, lose=False) as second_http,
        ):
            held = HoldAdmittedResponse(CellAgentClient("http://agent", client=first_http))
            second_client = CellAgentClient("http://agent", client=second_http)
            # Keep renewal cadence (30s) longer than the bounded response barrier.
            controller_settings = EyvanSettings(outbox_claim_ttl_s=90)
            first = ObservedDispatchController(
                repository,
                ControlPlaneService(repository),
                settings=controller_settings,
                agent_factory=lambda _cell: held,
            )
            renewal_allowed = asyncio.Event()
            renewal_attempts = 0
            if lose_renewal:
                # Gate the accelerated heartbeat until the deliberate expiry is committed.
                # Otherwise it races admission's serializable transaction on PostgreSQL,
                # testing an unrelated serialization conflict instead of claim refusal.
                real_renewal = first._renew_outbox_claim

                async def renew_after_expiry(command):
                    nonlocal renewal_attempts
                    if renewal_attempts:
                        await renewal_allowed.wait()
                    renewal_attempts += 1
                    return await real_renewal(command)

                monkeypatch.setattr(first, "_renew_outbox_claim", renew_after_expiry)
                monkeypatch.setattr(first, "_outbox_claim_heartbeat_s", lambda: 0.02)
            second = ObservedDispatchController(
                repository,
                ControlPlaneService(repository),
                settings=controller_settings,
                agent_factory=lambda _cell: second_client,
            )
            assert first.worker_id != second.worker_id
            delivery = asyncio.create_task(first.dispatch_outbox_once())
            second_delivery = None
            try:
                assert await asyncio.to_thread(held.admitted.wait, 8), "first admission did not reach the barrier"
                assert held.remote is not None
                remote_id = held.remote.remote_id
                await asyncio.to_thread(wait_for, native, remote_id, {ExecutionWorkloadStatus.RUNNING})
                reserved = await _snapshot(repository, workload.id)
                old_reservation = reserved["workload"].dispatch_reservation
                assert old_reservation is not None
                command = reserved["outbox"][command_id]
                assert command.state is OutboxState.PROCESSING and command.claimed_by == first.worker_id
                # Deterministic lost-renewal fault, not a wall-clock sleep or leadership claim.
                async with repository.transaction() as tx:
                    await tx.put_outbox(command.model_copy(update={"claimed_until": utcnow() - timedelta(seconds=1)}))
                if lose_renewal:
                    assert renewal_attempts == 1, "heartbeat must not race admission or deliberate expiry"
                    renewal_allowed.set()
                    assert await asyncio.wait_for(delivery, 8) == 1
                    assert renewal_attempts == 2, "the real renewal must inspect and refuse the expired claim"
                    assert first.inner_dispatch is not None and not first.inner_dispatch.done()
                    assert not first.inner_finished.is_set() and not held.returned.is_set()
                    assert native.status(remote_id).status is ExecutionWorkloadStatus.RUNNING
                second_delivery = asyncio.create_task(second.dispatch_outbox_once())
                assert await asyncio.wait_for(second_delivery, 8) == 1
                winner = await _snapshot(repository, workload.id)
                current = winner["workload"]
                assert current.state is WorkloadState.DISPATCHING and current.remote_id == remote_id
                assert current.dispatch_reservation is not None
                assert current.dispatch_reservation != old_reservation
                assert current.dispatch_reservation.outbox_id == command_id
                assert current.generation == current.attempt == 1
                assert winner["leases"][lease.id].state is LeaseState.ACTIVE
                assert winner["outbox"][command_id].state is OutboxState.SUCCEEDED
                assert first_http.admitted == second_http.admitted == [remote_id]
                assert native.status(remote_id).status is ExecutionWorkloadStatus.RUNNING
                held.release.set()
                assert await asyncio.wait_for(delivery, 8) == 1
                await asyncio.wait_for(first.inner_finished.wait(), 8)
                assert first.inner_dispatch.done(), "outer completion does not prove inner dispatch finished"
                assert held.returned.is_set(), "the late response must return, not time out at the test barrier"
                assert await _snapshot(repository, workload.id) == winner
                assert native.status(remote_id).status is ExecutionWorkloadStatus.RUNNING
                finish.touch()
                await asyncio.to_thread(wait_for, native, remote_id, {ExecutionWorkloadStatus.SUCCEEDED})
                assert marker.read_text() == "executed\n"
                assert sum(event.action == "lease.activated" for event in winner["events"]) == 1
                assert not any(event.action == "lease.settled" for event in winner["events"])
                assert winner["cells"] == initial["cells"]
            finally:
                renewal_allowed.set()
                held.release.set()
                try:
                    await _drain_dispatch(first, delivery)
                finally:
                    await _drain_dispatch(second, second_delivery)
