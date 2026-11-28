# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A remotely accepted dispatch retains its secret authority after response loss."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from test_secret_dispatch_transactions import _KEY, _REFERENCE, _ROOT, _VALUE, _plane, _seed_dispatch, _snapshot

from eyvan.records import LeaseState, OutboxState, WorkloadDispatchReservation, WorkloadState, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service.errors import AuthorizationError
from eyvan.worker.agent.client import CellAgentUnavailableError


@pytest.fixture(params=["memory", "local", "postgres"])
async def repository(request, tmp_path):
    engine = admin = None
    schema = "retained_secret_" + uuid4().hex
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


@pytest.mark.parametrize("gang", [False, True], ids=["single", "gang"])
async def test_response_loss_retains_only_exact_dispatch_secret_authority(repository, tmp_path, monkeypatch, gang):
    service, signer, agent, controller = await _plane(repository, tmp_path)
    materialize = agent._materialize
    admitted = {}

    def lose_response(request):
        materialize(request)
        admitted[request.spec["submission_id"]] = dict(request.spec["environment"])
        raise CellAgentUnavailableError(f"response lost after {_VALUE.decode()} using {agent.tokens[-1]}")

    monkeypatch.setattr(agent, "_materialize", lose_response)
    seeded, _, _ = await _seed_dispatch(repository, gang=gang)

    # The agent has already resolved the secret and admitted the execution;
    # its submit response, not its acceptance, is what the controller loses.
    assert await asyncio.wait_for(controller.dispatch_outbox_once(), timeout=5) == 1
    workload, leases, commands, events = await _snapshot(repository, seeded.id)
    command = next(item for item in commands if item.topic == "workload.dispatch")
    assert agent.values == [{"HF_TOKEN": _VALUE.decode()}]
    assert agent.submissions == int(not gang)
    assert agent.gang_submissions == int(gang)
    assert workload.state is WorkloadState.LEASED and workload.remote_id is None
    reservation = workload.dispatch_reservation
    assert reservation is not None
    assert reservation.lease_ids == tuple(lease.id for lease in leases)
    assert WorkloadDispatchReservation.model_validate(command.payload["dispatch_reservation"]) == reservation
    assert all(lease.state is LeaseState.PENDING for lease in leases)
    assert command.state is OutboxState.FAILED
    assert command.claimed_by is None and command.claimed_until is None
    assert not any(event.action == "workload.dispatch_reservation_failed" for event in events)
    token = agent.tokens[0]
    assert admitted[f"{workload.id}-a{workload.attempt}-g{workload.generation}"]["EYVAN_SECRET_0_TOKEN"] == token
    claims = signer.verify(token, workload_id=workload.id, org_id=workload.org_id, project_id=workload.project_id)
    authority = dict(
        org_id=workload.org_id,
        project_id=workload.project_id,
        workload_id=workload.id,
        attempt=claims.attempt,
        generation=claims.generation,
        dispatch_fence=claims.dispatch_fence,
        name=claims.name,
    )

    async def redeem():
        assert await service.authorize_secret_mount(**authority) == (_REFERENCE, _ROOT)

    await redeem()  # RED: a failed outbox's cleared claim wrongly revokes this token.
    for updates in (
        {"state": OutboxState.PENDING},
        {
            "state": OutboxState.PROCESSING,
            "claimed_by": "expired-delivery-owner",
            "claimed_until": utcnow() - timedelta(seconds=1),
        },
    ):
        async with repository.transaction() as tx:
            await tx.put_outbox(command.model_copy(update=updates))
        await redeem()
    async with repository.transaction() as tx:
        await tx.put_outbox(command)

    # Quarantine and a new delivery claim do not supersede remote admission.
    async with repository.transaction() as tx:
        for lease in leases:
            await tx.put_lease(lease.model_copy(update={"state": LeaseState.QUARANTINED}))
        await tx.put_outbox(command.model_copy(update={"available_at": utcnow()}))
    await redeem()
    # Transient transport failures refund the initial delivery attempt. This
    # counter is not the frozen workload attempt or reservation's identity.
    assert command.attempts == reservation.outbox_attempt - 1
    reclaimed = await repository.claim_outbox_id(worker_id=controller.worker_id, command_id=command.id, lease_seconds=60)
    assert reclaimed is not None
    await redeem()
    assert await controller._finish_outbox(reclaimed, error="response lost")
    async with repository.transaction() as tx:
        failed = await tx.get_outbox(command.id)
        assert failed is not None
        await tx.put_outbox(failed.model_copy(update={"available_at": utcnow()}))
    reclaimed = await repository.claim_outbox_id(worker_id=controller.worker_id, command_id=command.id, lease_seconds=60)
    assert reclaimed is not None and reclaimed.attempts == reservation.outbox_attempt + 1
    await redeem()
    assert await controller._finish_outbox(reclaimed, error="response lost", transient=True)
    await redeem()

    # A signed token is still subordinate to exact durable ownership. Each
    # mutation is restored so none of these denials depends on a prior denial.
    for updates in (
        {"dispatch_reservation": None},
        {"dispatch_reservation": reservation.model_copy(update={"fence": "successor-fence"})},
        {"generation": workload.generation + 1},
        {"attempt": workload.attempt + 1},
        {"lease_id": "successor-lease", "lease_ids": ("successor-lease",)},
        {"state": WorkloadState.CANCELLING},
        {"state": WorkloadState.CANCELLED},
    ):
        async with repository.transaction() as tx:
            await tx.put_workload(workload.model_copy(update=updates))
        with pytest.raises(AuthorizationError):
            await service.authorize_secret_mount(**authority)
        async with repository.transaction() as tx:
            await tx.put_workload(workload)
    for updates in (
        {"aggregate_id": "other-workload"},
        {"aggregate_type": "cell"},
        {"topic": "workload.cancel"},
        {"generation": workload.generation + 1},
        {"state": OutboxState.DISCARDED},
        {"state": OutboxState.SUCCEEDED},
        {"payload": {**command.payload, "dispatch_reservation": None}},
        {
            "payload": {
                **command.payload,
                "dispatch_reservation": reservation.model_copy(
                    update={"outbox_attempt": reservation.outbox_attempt + 1}
                ).model_dump(mode="json"),
            }
        },
    ):
        async with repository.transaction() as tx:
            await tx.put_outbox(command.model_copy(update=updates))
        with pytest.raises(AuthorizationError):
            await service.authorize_secret_mount(**authority)
        async with repository.transaction() as tx:
            await tx.put_outbox(command)
    for updates in (
        {"state": LeaseState.RELEASED},
        {"workload_id": "other-workload"},
        {"generation": workload.generation + 1},
        {"expires_at": utcnow() - timedelta(seconds=1)},
    ):
        async with repository.transaction() as tx:
            await tx.put_lease(leases[-1].model_copy(update=updates))
        with pytest.raises(AuthorizationError):
            await service.authorize_secret_mount(**authority)
        async with repository.transaction() as tx:
            await tx.put_lease(leases[-1])
    for updates in ({"org_id": "other"}, {"project_id": "other"}, {"name": "UNDECLARED"}):
        with pytest.raises(AuthorizationError):
            await service.authorize_secret_mount(**{**authority, **updates})
    async with repository.transaction() as tx:
        policy = await tx.get_policy(workload.org_id, workload.project_id)
        assert policy is not None
        await tx.put_policy(policy.model_copy(update={"allowed_secret_references": ()}))
    with pytest.raises(AuthorizationError):
        await service.authorize_secret_mount(**authority)
    async with repository.transaction() as tx:
        await tx.put_policy(policy)

    # Idempotent recovery keeps the signed fence; definitive refusal must not.
    monkeypatch.setattr(agent, "_materialize", materialize)
    async with repository.transaction() as tx:
        for lease in leases:
            await tx.put_lease(lease)
        await tx.put_outbox(command.model_copy(update={"available_at": utcnow()}))
    assert await asyncio.wait_for(controller.dispatch_outbox_once(), timeout=5) == 1
    recovered, active, commands, events = await _snapshot(repository, workload.id)
    assert recovered.state is WorkloadState.DISPATCHING
    assert recovered.remote_id is not None
    assert recovered.dispatch_reservation.fence == claims.dispatch_fence
    assert all(lease.state is LeaseState.ACTIVE for lease in active)
    assert next(item for item in commands if item.id == command.id).state is OutboxState.SUCCEEDED
    await redeem()
    assert any(event.action == "secret.read_denied" for event in events)
    recorded = json.dumps([event.model_dump(mode="json") for event in events])
    assert token not in recorded and _VALUE.decode() not in recorded and _REFERENCE not in recorded
    assert _KEY.decode() not in recorded
