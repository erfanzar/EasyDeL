# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dispatch/secret integration at the repository transaction boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from test_priority_preemption_properties import repository as _repository_fixture

from eyvan.artifacts.managed import ArtifactTokenSigner, LocalManagedArtifactStore
from eyvan.artifacts.secret_capability import SecretCapabilitySigner, SecretTokenClaims
from eyvan.artifacts.transfers import ManagedArtifactAccess
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    PolicyRecord,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, ConflictError
from eyvan.worker.agent.client import CellAgentInvalidConfigurationError
from eyvan.worker.agent.models import WorkloadObservation
from eyvan.worker.runtime.secret_materialization import materialize_secrets

repository = _repository_fixture


_KEY = b"d" * 32
_VALUE = b"dispatch-only-secret-value"
_REFERENCE = "file:///operator/secrets/hf-token"
_ROOT = "file:///operator/secrets"
_ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
_OPERATOR = Principal(
    subject="alice",
    grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
)


class _ReentrantSecretAgent:
    """Redeem the submitted capability synchronously, just like the real runtime."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        service: ControlPlaneService,
        signer: SecretCapabilitySigner,
        *,
        fail_after_materialization: bool = False,
        block_after_materialization: bool = False,
    ) -> None:
        self.loop = loop
        self.service = service
        self.signer = signer
        self.fail_after_materialization = fail_after_materialization
        self.block_after_materialization = block_after_materialization
        self.materialized = threading.Event()
        self.release = threading.Event()
        self.values: list[dict[str, str]] = []
        self.tokens: list[str] = []
        self.claims: list[SecretTokenClaims] = []
        self.submissions = 0
        self.gang_submissions = 0
        self.stops: list[str] = []
        self.gang_stops: list[str] = []
        self.requests = []
        self.executions = {}

    def _await(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=2)

    def _materialize(self, request) -> None:
        environment = request.spec["environment"]

        def fetch(url: str, token: str) -> bytes:
            query = parse_qs(urlsplit(url).query)
            workload_id = query["workload_id"][0]
            scope = self._await(self.service.secret_redemption_scope(workload_id))
            assert scope == ("acme", "research")
            claims = self.signer.verify(
                token,
                workload_id=workload_id,
                org_id=scope[0],
                project_id=scope[1],
            )
            reference, root = self._await(
                self.service.authorize_secret_mount(
                    org_id=scope[0],
                    project_id=scope[1],
                    workload_id=workload_id,
                    attempt=claims.attempt,
                    generation=claims.generation,
                    dispatch_fence=claims.dispatch_fence,
                    name=claims.name,
                )
            )
            assert (reference, root) == (_REFERENCE, _ROOT)
            self.tokens.append(token)
            self.claims.append(claims)
            return _VALUE

        self.values.append(materialize_secrets(environment, fetch=fetch))
        self.materialized.set()
        if self.block_after_materialization and not self.release.wait(timeout=3):
            raise TimeoutError("test did not release reentrant submit")
        if self.fail_after_materialization:
            token = self.tokens[-1]
            raise RuntimeError(f"provider reflected {_VALUE.decode()} and {token}")

    def _submit(self, request, key):
        self.requests.append(request)
        # The worker's durable idempotency contract is keyed by the mutation,
        # not by delivery count. An accepted execution survives a lost response.
        if key not in self.executions:
            self.executions[key] = WorkloadObservation(
                remote_id=request.spec["submission_id"],
                status=ExecutionWorkloadStatus.SUBMITTED,
            )
            self._materialize(request)
        return self.executions[key]

    def submit(self, cell_id, request):
        del cell_id
        self.submissions += 1
        return self._submit(request, request.mutation.idempotency_key)

    def submit_gang(self, coordinator_cell_id, request):
        del coordinator_cell_id
        self.gang_submissions += 1
        return self._submit(request, tuple(member.mutation.idempotency_key for member in request.members))

    def stop(self, cell_id, remote_id, request):
        del cell_id, request
        self.stops.append(remote_id)
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    def stop_gang(self, coordinator_cell_id, workload_id, remote_id, request):
        del coordinator_cell_id, workload_id, request
        self.gang_stops.append(remote_id)
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)


async def _plane(repository, tmp_path, *, fail: bool = False, block: bool = False):
    service = ControlPlaneService(repository)
    signer = SecretCapabilitySigner(_KEY, ttl_seconds=3600)
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path / "artifacts"),
        signer=ArtifactTokenSigner(_KEY, default_ttl_seconds=3600, max_ttl_seconds=3600),
        public_url="http://127.0.0.1:8000",
        token_ttl_seconds=3600,
    )
    agent = _ReentrantSecretAgent(
        asyncio.get_running_loop(),
        service,
        signer,
        fail_after_materialization=fail,
        block_after_materialization=block,
    )
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(dev_auth_enabled=True, lease_ttl_s=60),
        agent_factory=lambda _cell: agent,
        artifact_access=access,
        secret_capability=signer,
    )
    return service, signer, agent, controller


async def _seed_dispatch(
    repository,
    *,
    gang: bool = False,
) -> tuple[WorkloadRecord, tuple[LeaseRecord, ...], OutboxRecord]:
    lease_ids = ("lease-a", "lease-b") if gang else ("lease-a",)
    cell_ids = ("cell-a", "cell-b") if gang else ("cell-a",)
    spec = WorkloadDefinition(
        name="secret-job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        required_executor="eyvan-native",
        cell_group_count=len(cell_ids),
        secret_mounts={"HF_TOKEN": _REFERENCE},
    )
    workload = WorkloadRecord(
        id="wl-secret-dispatch",
        org_id="acme",
        project_id="research",
        spec=spec,
        state=WorkloadState.LEASED,
        lease_id=lease_ids[0],
        lease_ids=lease_ids,
        created_by="alice",
    )
    leases = tuple(
        LeaseRecord(
            id=lease_id,
            workload_id=workload.id,
            cell_id=cell_id,
            cell_generation=1,
            org_id=workload.org_id,
            project_id=workload.project_id,
            state=LeaseState.PENDING,
            resources=spec.gang,
            generation=workload.generation,
            fencing_token=f"fence-{cell_id}",
            group_id="gang-a" if gang else None,
            group_index=index,
            expires_at=utcnow() + timedelta(minutes=5),
        )
        for index, (lease_id, cell_id) in enumerate(zip(lease_ids, cell_ids, strict=True))
    )
    command = OutboxRecord(
        id="cmd-secret-dispatch",
        topic="workload.dispatch",
        aggregate_type="workload",
        aggregate_id=workload.id,
        idempotency_key="dispatch-secret-job",
        generation=workload.generation,
        payload={
            "workload_id": workload.id,
            "lease_id": lease_ids[0],
            "lease_ids": list(lease_ids),
            "cell_id": cell_ids[0],
        },
    )
    async with repository.transaction() as tx:
        await tx.put_policy(
            PolicyRecord(
                org_id=workload.org_id,
                project_id=workload.project_id,
                allowed_secret_references=(_ROOT,),
                updated_by="root",
            )
        )
        for cell_id in cell_ids:
            await tx.put_cell(
                CellRecord(
                    id=cell_id,
                    provider="local",
                    region="local",
                    state=CellState.READY,
                    capacity=ResourceVector(cpu=4, memory_gib=16),
                    allocatable=ResourceVector(cpu=4, memory_gib=16),
                    executor="eyvan-native",
                    endpoint="http://agent:8090",
                )
            )
        await tx.put_workload(workload)
        for lease in leases:
            await tx.put_lease(lease)
        await tx.put_outbox(command)
    return workload, leases, command


async def _snapshot(repository, workload_id: str):
    async with repository.transaction() as tx:
        workload = await tx.get_workload(workload_id)
        leases = await tx.list_leases(workload_id=workload_id)
        commands = await tx.list_outbox(limit=100)
        events = await tx.list_events(limit=200)
    assert workload is not None
    return workload, leases, commands, events


async def _assert_success(repository, tmp_path, *, gang: bool = False) -> None:
    service, _signer, agent, controller = await _plane(repository, tmp_path)
    seeded, _leases, _command = await _seed_dispatch(repository, gang=gang)

    assert await asyncio.wait_for(controller.dispatch_outbox_once(), timeout=3) == 1

    workload, leases, commands, events = await _snapshot(repository, seeded.id)
    assert workload.state is WorkloadState.DISPATCHING
    assert workload.remote_id == f"{seeded.id}-a1-g1"
    assert workload.dispatch_reservation is not None
    assert all(lease.state is LeaseState.ACTIVE for lease in leases)
    assert agent.values == [{"HF_TOKEN": _VALUE.decode()}]
    assert agent.gang_submissions == int(gang)
    assert agent.submissions == int(not gang)
    dispatch = next(command for command in commands if command.topic == "workload.dispatch")
    assert dispatch.state is OutboxState.SUCCEEDED

    actions = [event.action for event in events]
    assert actions.index("workload.dispatch_reserved") < actions.index("secret.issued")
    assert actions.index("secret.issued") < actions.index("secret.read")
    assert actions.index("secret.read") < actions.index("workload.dispatched")
    assert actions.count("lease.activated") == len(leases)

    durable = json.dumps(
        {
            "workload": workload.model_dump(mode="json"),
            "leases": [lease.model_dump(mode="json") for lease in leases],
            "commands": [command.model_dump(mode="json") for command in commands],
            "events": [event.model_dump(mode="json") for event in events],
        },
        sort_keys=True,
    )
    assert _VALUE.decode() not in durable
    assert agent.tokens[0] not in durable
    assert not any(_VALUE.decode() in str(event.payload) for event in events)

    claims = agent.claims[0]
    assert claims.generation == workload.generation
    assert claims.dispatch_fence == workload.dispatch_reservation.fence
    assert await service.secret_redemption_scope(workload.id) == ("acme", "research")


async def test_reentrant_secret_materialization_does_not_deadlock_memory(tmp_path) -> None:
    await _assert_success(InMemoryRepository(), tmp_path)


async def test_reentrant_gang_dispatch_commits_every_member_atomically(tmp_path) -> None:
    await _assert_success(InMemoryRepository(), tmp_path, gang=True)


@pytest.mark.parametrize("repository", ["memory", "postgres", "local"], indirect=True)
@pytest.mark.parametrize("gang", [False, True])
async def test_unknown_submit_retains_the_issuance_and_retry_reuses_its_fence(repository, tmp_path, gang) -> None:
    service, _signer, agent, controller = await _plane(repository, tmp_path, fail=True)
    seeded, leases, _command = await _seed_dispatch(repository, gang=gang)

    assert await asyncio.wait_for(controller.dispatch_outbox_once(), timeout=3) == 1
    failed_workload, failed_leases, commands, events = await _snapshot(repository, seeded.id)
    failed_command = next(command for command in commands if command.topic == "workload.dispatch")
    old_claims = agent.claims[0]
    old_token = agent.tokens[0]

    assert failed_workload.state is WorkloadState.LEASED
    reservation = failed_workload.dispatch_reservation
    assert reservation is not None
    assert reservation.fence == old_claims.dispatch_fence
    assert all(lease.state is LeaseState.PENDING for lease in failed_leases)
    assert failed_command.state is OutboxState.FAILED
    assert failed_command.payload["dispatch_reservation"] == reservation.model_dump(mode="json")
    assert len(agent.executions) == 1
    # The message is withheld; the class and the Eyvan frame that raised it are
    # code identifiers, kept so a suppressed failure can still be located.
    assert failed_command.last_error.startswith("output suppressed for secret-bearing execution (RuntimeError in eyvan.")
    assert _VALUE.decode() not in failed_command.last_error
    assert old_token not in failed_command.last_error
    assert "workload.dispatch_reservation_failed" not in [event.action for event in events]
    with pytest.raises(ConflictError, match="dispatch may have started"):
        lease = leases[0]
        await service.requeue_unstarted_workload(
            _ADMIN,
            seeded.id,
            scheduling_reason="submit acceptance is still unknown",
            expected_generation=seeded.generation,
            expected_lease_id=lease.id,
            expected_cell_generation=lease.cell_generation,
            expected_cell_state=CellState.READY,
            expected_unavailable_reason=None,
            expected_cell_id=lease.cell_id,
        )

    durable_failure = json.dumps(
        {
            "workload": failed_workload.model_dump(mode="json"),
            "leases": [lease.model_dump(mode="json") for lease in failed_leases],
            "command": failed_command.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in events],
        }
    )
    assert _VALUE.decode() not in durable_failure
    assert old_token not in durable_failure
    if isinstance(repository, LocalJsonRepository):
        await repository.close()
        await repository.open()
        restored, restored_leases, restored_commands, _ = await _snapshot(repository, seeded.id)
        assert restored == failed_workload
        assert restored_leases == failed_leases
        assert next(command for command in restored_commands if command.id == failed_command.id) == failed_command

    async with repository.transaction() as tx:
        current = await tx.get_outbox(failed_command.id)
        assert current is not None
        await tx.put_outbox(current.model_copy(update={"available_at": utcnow()}))
    agent.fail_after_materialization = False
    successor = EyvanController(
        repository,
        service,
        settings=controller.settings,
        agent_factory=lambda _cell: agent,
        artifact_access=controller.artifact_access,
        secret_capability=controller.secret_capability,
    )
    assert await asyncio.wait_for(successor.dispatch_outbox_once(), timeout=3) == 1

    retried_workload, retried_leases, commands, events = await _snapshot(repository, seeded.id)
    assert retried_workload.state is WorkloadState.DISPATCHING
    assert all(lease.state is LeaseState.ACTIVE for lease in retried_leases)
    assert len(agent.claims) == len(agent.values) == 1
    replay_claims = _signer.verify(
        agent.requests[1].spec["environment"]["EYVAN_SECRET_0_TOKEN"],
        workload_id=seeded.id,
        org_id="acme",
        project_id="research",
    )
    assert replay_claims.dispatch_fence == old_claims.dispatch_fence
    assert replay_claims.attempt == old_claims.attempt
    assert replay_claims.generation == old_claims.generation
    assert await service.authorize_secret_mount(
        org_id="acme",
        project_id="research",
        workload_id=seeded.id,
        attempt=old_claims.attempt,
        generation=old_claims.generation,
        dispatch_fence=old_claims.dispatch_fence,
        name=old_claims.name,
    ) == (_REFERENCE, _ROOT)
    assert len(agent.executions) == 1
    assert len({request.spec["submission_id"] for request in agent.requests}) == 1
    assert await successor.dispatch_outbox_once() == 0
    issuances = [event for event in events if event.action == "secret.issued"]
    assert [event.payload["outbox_attempt"] for event in issuances] == [1, 2]
    assert next(command for command in commands if command.topic == "workload.dispatch").state is OutboxState.SUCCEEDED
    assert [lease.id for lease in retried_leases] == [lease.id for lease in leases]


@pytest.mark.parametrize("repository", ["memory", "postgres", "local"], indirect=True)
@pytest.mark.parametrize("gang", [False, True])
async def test_definitive_preacceptance_refusal_revokes_and_settles(repository, tmp_path, gang) -> None:
    service, signer, agent, controller = await _plane(repository, tmp_path)
    seeded, _leases, _command = await _seed_dispatch(repository, gang=gang)
    captured = []

    def refuse(*args):
        request = args[-1]
        captured.append(request.spec["environment"]["EYVAN_SECRET_0_TOKEN"])
        # Unlike an arbitrary submit exception this is the wire client's typed
        # validation verdict, before the backend creates any execution.
        raise CellAgentInvalidConfigurationError("launch rejected before admission")

    agent.submit = refuse
    agent.submit_gang = refuse
    assert await controller.dispatch_outbox_once() == 1
    workload, leases, commands, events = await _snapshot(repository, seeded.id)
    assert workload.state is WorkloadState.FAILED
    assert workload.dispatch_reservation is None and workload.remote_id is None
    assert all(lease.state is LeaseState.RELEASED for lease in leases)
    dispatch = next(command for command in commands if command.topic == "workload.dispatch")
    assert dispatch.state is OutboxState.SUCCEEDED
    assert "dispatch_reservation" not in dispatch.payload
    assert not agent.executions and not agent.values
    assert [event.action for event in events].count("workload.dispatch_reservation_failed") == 1
    claims = signer.verify(captured[0], workload_id=seeded.id, org_id="acme", project_id="research")
    with pytest.raises(AuthorizationError):
        await service.authorize_secret_mount(
            org_id="acme",
            project_id="research",
            workload_id=seeded.id,
            attempt=claims.attempt,
            generation=claims.generation,
            dispatch_fence=claims.dispatch_fence,
            name=claims.name,
        )
    assert await controller.dispatch_outbox_once() == 0


@pytest.mark.parametrize("repository", ["memory", "postgres", "local"], indirect=True)
@pytest.mark.parametrize("gang", [False, True])
@pytest.mark.parametrize("refusal_stage", ["validation", "preflight", "stale_build"])
async def test_replay_refusal_cannot_disprove_prior_unknown_acceptance(repository, tmp_path, gang, refusal_stage):
    from eyvan.server.controller.workload_commands import StaleBuildDispatchRefusal, UnreachableCallbackRefusal

    _service, _signer, agent, controller = await _plane(repository, tmp_path, fail=True)
    seeded, _leases, _command = await _seed_dispatch(repository, gang=gang)
    assert await controller.dispatch_outbox_once() == 1
    unknown, pending, commands, _ = await _snapshot(repository, seeded.id)
    assert unknown.dispatch_reservation is not None and len(agent.executions) == 1
    dispatch = next(command for command in commands if command.topic == "workload.dispatch")
    async with repository.transaction() as tx:
        await tx.put_outbox(dispatch.model_copy(update={"available_at": utcnow()}))
    submit, submit_gang = agent.submit, agent.submit_gang
    preflight = controller._refuse_unreachable_callback
    builds = controller._refuse_stale_cell_builds

    async def refuse_stale_builds(cells):
        raise StaleBuildDispatchRefusal("this replay sees a changed build", cells)

    def refuse(*_args):
        if refusal_stage == "validation":
            raise CellAgentInvalidConfigurationError("this replay was not admitted")
        raise UnreachableCallbackRefusal("this replica cannot render a reachable callback")

    if refusal_stage == "validation":
        agent.submit = agent.submit_gang = refuse
    elif refusal_stage == "stale_build":
        controller._refuse_stale_cell_builds = refuse_stale_builds
    else:
        controller._refuse_unreachable_callback = refuse
    assert await controller.dispatch_outbox_once() == 1
    still_unknown, retained_leases, commands, _ = await _snapshot(repository, seeded.id)
    dispatch = next(command for command in commands if command.id == dispatch.id)
    assert still_unknown.state is WorkloadState.LEASED
    assert still_unknown.remote_id is None
    assert still_unknown.dispatch_reservation is not None
    assert still_unknown.dispatch_reservation.fence == unknown.dispatch_reservation.fence
    assert retained_leases == pending
    assert dispatch.state is OutboxState.FAILED
    assert dispatch.payload["dispatch_reservation"] == still_unknown.dispatch_reservation.model_dump(mode="json")
    assert len(agent.executions) == 1
    assert not any(command.topic == "capacity.drain" for command in commands)
    async with repository.transaction() as tx:
        assert all(cell.state is CellState.READY for cell in await tx.list_cells())
    agent.submit, agent.submit_gang = submit, submit_gang
    agent.fail_after_materialization = False
    controller._refuse_unreachable_callback = preflight
    controller._refuse_stale_cell_builds = builds
    async with repository.transaction() as tx:
        await tx.put_outbox(dispatch.model_copy(update={"available_at": utcnow()}))
    assert await controller.dispatch_outbox_once() == 1
    recovered, active, commands, _ = await _snapshot(repository, seeded.id)
    assert recovered.state is WorkloadState.DISPATCHING
    assert recovered.dispatch_reservation.fence == unknown.dispatch_reservation.fence
    assert all(lease.state is LeaseState.ACTIVE for lease in active)
    assert len(agent.executions) == len(agent.values) == 1
    assert next(command for command in commands if command.id == dispatch.id).state is OutboxState.SUCCEEDED


async def test_failed_gang_submit_keeps_every_member_pending(tmp_path) -> None:
    repository = InMemoryRepository()
    _service, _signer, agent, controller = await _plane(repository, tmp_path, fail=True)
    seeded, _leases, _command = await _seed_dispatch(repository, gang=True)

    assert await asyncio.wait_for(controller.dispatch_outbox_once(), timeout=3) == 1

    workload, leases, commands, events = await _snapshot(repository, seeded.id)
    dispatch = next(command for command in commands if command.topic == "workload.dispatch")
    assert workload.state is WorkloadState.LEASED
    assert workload.dispatch_reservation is not None
    assert [lease.state for lease in leases] == [LeaseState.PENDING, LeaseState.PENDING]
    assert dispatch.state is OutboxState.FAILED
    assert dispatch.payload["dispatch_reservation"] == workload.dispatch_reservation.model_dump(mode="json")
    assert len(agent.executions) == 1
    assert agent.gang_submissions == 1
    assert [event.action for event in events].count("lease.activated") == 0


@pytest.mark.parametrize("repository", ["memory", "postgres", "local"], indirect=True)
@pytest.mark.parametrize("gang", [False, True])
async def test_cancellation_waits_for_unknown_acceptance_then_stops_recovered_execution(repository, tmp_path, gang):
    service, _signer, agent, controller = await _plane(repository, tmp_path, fail=True)
    seeded, _leases, dispatch = await _seed_dispatch(repository, gang=gang)
    assert await controller.dispatch_outbox_once() == 1
    failed, pending, commands, _ = await _snapshot(repository, seeded.id)
    assert failed.dispatch_reservation is not None and failed.remote_id is None
    assert all(lease.state is LeaseState.PENDING for lease in pending)
    async with repository.transaction() as tx:
        current = next(command for command in commands if command.id == dispatch.id)
        await tx.put_outbox(current.model_copy(update={"available_at": utcnow() + timedelta(hours=1)}))
    await service.cancel_workload(_OPERATOR, seeded.id, idempotency_key="cancel-unknown-admission")
    successor = EyvanController(
        repository,
        service,
        settings=controller.settings,
        agent_factory=lambda _cell: agent,
        artifact_access=controller.artifact_access,
        secret_capability=controller.secret_capability,
    )
    assert await successor.dispatch_outbox_once() == 1
    waiting, pending, commands, _ = await _snapshot(repository, seeded.id)
    cancel = next(command for command in commands if command.topic == "workload.cancel")
    assert cancel.state is OutboxState.FAILED
    assert waiting.state is WorkloadState.CANCELLING
    assert waiting.dispatch_reservation == failed.dispatch_reservation
    assert all(lease.state is LeaseState.PENDING for lease in pending)
    assert not agent.stops and not agent.gang_stops
    agent.fail_after_materialization = False
    async with repository.transaction() as tx:
        current = await tx.get_outbox(dispatch.id)
        await tx.put_outbox(current.model_copy(update={"available_at": utcnow()}))
    assert await successor.dispatch_outbox_once() == 1
    recovered, active, commands, _ = await _snapshot(repository, seeded.id)
    assert recovered.state is WorkloadState.CANCELLING
    assert recovered.remote_id == next(iter(agent.executions.values())).remote_id
    assert recovered.dispatch_reservation.fence == failed.dispatch_reservation.fence
    assert all(lease.state is LeaseState.ACTIVE for lease in active)
    async with repository.transaction() as tx:
        current = await tx.get_outbox(cancel.id)
        await tx.put_outbox(current.model_copy(update={"available_at": utcnow()}))
    assert await successor.dispatch_outbox_once() == 1
    cancelled, released, commands, _ = await _snapshot(repository, seeded.id)
    assert cancelled.state is WorkloadState.CANCELLED and cancelled.dispatch_reservation is None
    assert all(lease.state is LeaseState.RELEASED for lease in released)
    assert agent.gang_stops == ([recovered.remote_id] if gang else [])
    assert agent.stops == ([] if gang else [recovered.remote_id])
    assert len(agent.executions) == len(agent.values) == 1
    assert all(command.state is OutboxState.SUCCEEDED for command in commands)
    assert await successor.dispatch_outbox_once() == 0


@pytest.mark.parametrize("repository", ["memory", "postgres", "local"], indirect=True)
@pytest.mark.parametrize("gang", [False, True])
async def test_successor_claim_fences_a_late_submit_response_without_refencing_secrets(repository, tmp_path, gang):
    service, _signer, agent, controller = await _plane(repository, tmp_path, block=True)
    seeded, _leases, dispatch = await _seed_dispatch(repository, gang=gang)
    first_delivery = asyncio.create_task(controller.dispatch_outbox_once())
    try:
        assert await asyncio.to_thread(agent.materialized.wait, 2)
        inflight, pending, commands, _ = await _snapshot(repository, seeded.id)
        assert inflight.dispatch_reservation is not None
        assert all(lease.state is LeaseState.PENDING for lease in pending)
        async with repository.transaction() as tx:
            current = next(command for command in commands if command.id == dispatch.id)
            await tx.put_outbox(current.model_copy(update={"claimed_until": utcnow() - timedelta(seconds=1)}))
        successor = EyvanController(
            repository,
            service,
            settings=controller.settings,
            agent_factory=lambda _cell: agent,
            artifact_access=controller.artifact_access,
            secret_capability=controller.secret_capability,
        )
        assert await successor.dispatch_outbox_once() == 1
        finalized = await _snapshot(repository, seeded.id)
        workload, active, commands, _ = finalized
        assert workload.state is WorkloadState.DISPATCHING
        assert workload.dispatch_reservation.fence == inflight.dispatch_reservation.fence
        assert workload.dispatch_reservation.outbox_attempt == 2
        assert all(lease.state is LeaseState.ACTIVE for lease in active)
        assert next(command for command in commands if command.id == dispatch.id).state is OutboxState.SUCCEEDED
        agent.release.set()
        assert await asyncio.wait_for(first_delivery, timeout=3) == 1
        # A late response owned by attempt 1 must not rewrite attempt 2's
        # workload/outbox, timestamps, lease activation or issuance events.
        assert await _snapshot(repository, seeded.id) == finalized
        assert len(agent.executions) == len(agent.values) == len(agent.claims) == 1
        assert await successor.dispatch_outbox_once() == 0
    finally:
        agent.release.set()
        await asyncio.wait_for(first_delivery, timeout=3)


async def test_cancellation_and_requeue_cannot_overtake_an_inflight_submit(tmp_path) -> None:
    repository = InMemoryRepository()
    service, _signer, agent, controller = await _plane(repository, tmp_path, block=True)
    seeded, leases, _command = await _seed_dispatch(repository)
    lease = leases[0]

    delivery = asyncio.create_task(controller.dispatch_outbox_once())
    assert await asyncio.to_thread(agent.materialized.wait, 2)

    with pytest.raises(ConflictError, match="dispatch may have started"):
        await service.requeue_unstarted_workload(
            _ADMIN,
            seeded.id,
            scheduling_reason="cell changed while dispatching",
            expected_generation=seeded.generation,
            expected_lease_id=lease.id,
            expected_cell_generation=lease.cell_generation,
            expected_cell_state=CellState.READY,
            expected_unavailable_reason=None,
            expected_cell_id=lease.cell_id,
        )

    cancelling, _ = await service.cancel_workload(
        _OPERATOR,
        seeded.id,
        idempotency_key="cancel-inflight-dispatch",
    )
    assert cancelling.state is WorkloadState.CANCELLING
    second_controller = EyvanController(
        repository,
        service,
        settings=controller.settings,
        agent_factory=lambda _cell: agent,
        artifact_access=controller.artifact_access,
        secret_capability=controller.secret_capability,
    )
    assert await second_controller.dispatch_outbox_once() == 1
    during_cancel, during_leases, commands, _events = await _snapshot(repository, seeded.id)
    cancel_command = next(command for command in commands if command.topic == "workload.cancel")
    assert during_cancel.state is WorkloadState.CANCELLING
    assert during_cancel.remote_id is None
    assert during_cancel.dispatch_reservation is not None
    assert all(candidate.state is LeaseState.PENDING for candidate in during_leases)
    assert cancel_command.state is OutboxState.FAILED
    assert not agent.stops

    agent.release.set()
    assert await asyncio.wait_for(delivery, timeout=3) == 1
    resolved, active_leases, commands, _events = await _snapshot(repository, seeded.id)
    assert resolved.state is WorkloadState.CANCELLING
    assert resolved.remote_id == f"{seeded.id}-a1-g1"
    assert all(candidate.state is LeaseState.ACTIVE for candidate in active_leases)

    async with repository.transaction() as tx:
        current_cancel = await tx.get_outbox(cancel_command.id)
        assert current_cancel is not None
        await tx.put_outbox(current_cancel.model_copy(update={"available_at": utcnow()}))
    assert await second_controller.dispatch_outbox_once() == 1
    cancelled, released_leases, _commands, _events = await _snapshot(repository, seeded.id)
    assert cancelled.state is WorkloadState.CANCELLED
    assert cancelled.dispatch_reservation is None
    assert all(candidate.state is LeaseState.RELEASED for candidate in released_leases)
    assert agent.stops == [f"{seeded.id}-a1-g1"]


async def test_reentrant_secret_materialization_does_not_deadlock_postgres(postgres_url: str, tmp_path) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema

    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        await _assert_success(repository, tmp_path)
    finally:
        await repository.close()
        await engine.dispose()


async def test_postgres_fences_two_replicas_beginning_one_credential_rotation(postgres_url: str) -> None:
    """The rotation fence on the backend production actually uses.

    Lives beside the other secret-path PostgreSQL contracts rather than with the
    rest of the rotation tests, because this is the file the scheduled database
    job runs. ``latest_credential_rotation`` reads the newest generation
    ``FOR UPDATE``, so a second replica blocks, re-reads a row that is now in
    flight, and refuses; without the lock both read "nothing in flight" and both
    insert. The row and its check constraints are asserted in SQL for the same
    reason -- an operator diagnosing a stuck rotation reads the table, not the
    model.
    """

    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import create_async_engine

    from eyvan.identity.rotation import CredentialRotationInterleaved
    from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
    from eyvan.server.service.credential_rotation import SECRET_CAPABILITY_CREDENTIAL, CredentialRotationService

    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        first_replica = CredentialRotationService(repository)
        second_replica = CredentialRotationService(repository)

        started = await first_replica.begin(
            credential=SECRET_CAPABILITY_CREDENTIAL,
            fence="fence-a",
            overlap_s=300,
        )
        with pytest.raises(CredentialRotationInterleaved):
            await second_replica.begin(
                credential=SECRET_CAPABILITY_CREDENTIAL,
                fence="fence-b",
                overlap_s=300,
            )

        committed = await first_replica.commit(
            credential=SECRET_CAPABILITY_CREDENTIAL,
            generation=started.generation,
            fence="fence-a",
        )
        assert committed.previous_not_after is not None

        async with engine.begin() as connection:
            rows = (
                await connection.execute(
                    text("SELECT credential, generation, state FROM credential_rotations ORDER BY generation")
                )
            ).all()
            with pytest.raises(IntegrityError):
                # Generation 1 is the material a deployment started with, which
                # nothing rotated into; the constraint is what stops a row from
                # claiming otherwise.
                await connection.execute(
                    text(
                        "INSERT INTO credential_rotations "
                        "(credential, generation, state, fence, overlap_s, data) "
                        "VALUES ('other', 1, 'pending', 'f', 0, '{}')"
                    )
                )
        assert [tuple(row) for row in rows] == [(SECRET_CAPABILITY_CREDENTIAL, 2, "committed")]
    finally:
        await repository.close()
        await engine.dispose()
