# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PostgreSQL serialization proofs for artifact lifecycle races."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from datetime import timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri, result_uri
from eyvan.records import (
    ArtifactReferenceRecord,
    CallableExecution,
    ManagedArtifactState,
    Principal,
    QuotaRecord,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    artifact_reference_id,
)
from eyvan.server.persistence.base import TransientRepositoryError
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import ConflictError, InputError


class BlockingFirstPutStore(LocalManagedArtifactStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.first_put_started = threading.Event()
        self.allow_first_put = threading.Event()
        self._put_lock = threading.Lock()
        self._put_calls = 0

    def put_versioned(self, uri, payload, *, org_id, project_id):
        with self._put_lock:
            self._put_calls += 1
            call_number = self._put_calls
        if call_number == 1:
            self.first_put_started.set()
            if not self.allow_first_put.wait(timeout=3):
                raise RuntimeError("stale upload barrier timed out")
        return super().put_versioned(uri, payload, org_id=org_id, project_id=project_id)


class BlockingDeleteStore(LocalManagedArtifactStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.delete_started = threading.Event()
        self.allow_delete = threading.Event()
        self.delete_calls = 0
        self.block_delete = False

    def delete_versioned(self, uri, *, org_id, project_id, object_version):
        self.delete_calls += 1
        self.delete_started.set()
        if self.block_delete and not self.allow_delete.wait(timeout=3):
            raise RuntimeError("delete barrier timed out")
        return super().delete_versioned(
            uri,
            org_id=org_id,
            project_id=project_id,
            object_version=object_version,
        )


@pytest.fixture
async def repository(postgres_url: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)
    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        yield repository
    finally:
        await engine.dispose()


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )


def _fleet_admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _service(repository, store) -> ControlPlaneService:
    return ControlPlaneService(
        repository,
        artifact_store=store,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
    )


def _spec(uri: str) -> WorkloadDefinition:
    digest = uri.rsplit("/", 1)[-1]
    return WorkloadDefinition(
        name="postgres-artifact-race",
        execution=CallableExecution(kind="callable", artifact_uri=uri, sha256=digest, python="3.11"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1)),
    )


@pytest.mark.asyncio
async def test_stale_upload_cannot_delete_the_local_version_activated_by_its_postgres_retry(
    repository,
    tmp_path,
) -> None:
    store = BlockingFirstPutStore(tmp_path / "objects")
    service = _service(repository, store)
    payload = b"postgres-stale-upload-retry"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    stale_upload = asyncio.create_task(
        service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=uri,
            payload=payload,
        )
    )
    assert await asyncio.to_thread(store.first_put_started.wait, 2)
    async with repository.transaction() as transaction:
        first_reservation = await transaction.get_managed_artifact("acme", "research", uri)
    assert first_reservation is not None
    assert first_reservation.state is ManagedArtifactState.PENDING_UPLOAD

    recovered = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=first_reservation.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert recovered["released"] == 1
    retry = await service.upload_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        payload=payload,
    )
    assert retry.state is ManagedArtifactState.ACTIVE
    assert retry.record_generation == first_reservation.record_generation + 1

    store.allow_first_put.set()
    with pytest.raises(ConflictError, match="activation is stale"):
        await stale_upload

    async with repository.transaction() as transaction:
        final = await transaction.get_managed_artifact("acme", "research", uri)
    assert final is not None
    assert final.state is ManagedArtifactState.ACTIVE
    assert final.record_generation == retry.record_generation
    assert final.object_version == retry.object_version
    assert (
        await service.read_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=uri,
        )
        == payload
    )


@pytest.mark.asyncio
async def test_late_stale_payload_is_fully_charged_after_postgres_retry_conflict(repository, tmp_path) -> None:
    store = BlockingFirstPutStore(tmp_path / "objects")
    service = _service(repository, store)
    stale_payload = b"a" * 100
    retry_payload = b"b"
    uri = result_uri("postgres-late-stale-payload")
    stale_upload = asyncio.create_task(
        service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=uri,
            payload=stale_payload,
        )
    )
    assert await asyncio.to_thread(store.first_put_started.wait, 2)
    async with repository.transaction() as transaction:
        first_reservation = await transaction.get_managed_artifact("acme", "research", uri)
    assert first_reservation is not None
    released = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=first_reservation.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert released["released"] == 1
    retry_reservation = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(retry_payload),
        sha256=hashlib.sha256(retry_payload).hexdigest(),
    )

    store.allow_first_put.set()
    with pytest.raises(ConflictError, match="is stale"):
        await stale_upload
    with pytest.raises(ConflictError, match="physical object conflicts"):
        await service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=uri,
            payload=retry_payload,
        )

    async with repository.transaction() as transaction:
        final = await transaction.get_managed_artifact("acme", "research", uri)
    assert final is not None
    assert final.state is ManagedArtifactState.QUARANTINED
    assert final.record_generation == retry_reservation.record_generation
    assert final.size_bytes == len(stale_payload)
    assert store.get(uri, org_id="acme", project_id="research") == stale_payload
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == len(stale_payload)


@pytest.mark.asyncio
async def test_concurrent_upload_reservations_cannot_both_exceed_one_quota(repository, tmp_path) -> None:
    service = _service(repository, LocalManagedArtifactStore(tmp_path / "objects"))
    await service.put_quota(
        _fleet_admin(),
        quota=QuotaRecord(
            org_id="acme",
            project_id="research",
            limits=ResourceVector(cpu=10),
            artifact_bytes_limit=10,
            updated_by="root",
        ),
        idempotency_key="postgres-artifact-quota",
    )
    payloads = (b"aaaaaa", b"bbbbbb")

    async def upload(payload: bytes):
        return await service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=callable_uri(hashlib.sha256(payload).hexdigest()),
            payload=payload,
        )

    outcomes: list[object] = list(
        await asyncio.gather(*(upload(payload) for payload in payloads), return_exceptions=True)
    )
    # SERIALIZABLE may abort both racers at commit (seen on a loaded CI runner:
    # zero winners). That is safe, and TransientRepositoryError is the
    # documented retry signal, so retry the way a client would and then check
    # the real invariant: the quota admits exactly one of the two uploads.
    for index, payload in enumerate(payloads):
        for _ in range(5):
            if not isinstance(outcomes[index], TransientRepositoryError):
                break
            try:
                outcomes[index] = await upload(payload)
            except BaseException as exc:  # recorded, then asserted below
                outcomes[index] = exc
    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1, outcomes
    losers = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(losers) == 1, outcomes
    assert isinstance(losers[0], InputError), outcomes
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == 6


@pytest.mark.asyncio
async def test_reference_commit_before_claim_never_allows_physical_delete(repository, tmp_path) -> None:
    store = BlockingDeleteStore(tmp_path / "objects")
    service = _service(repository, store)
    payload = b"reference-wins"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    locked = asyncio.Event()
    release = asyncio.Event()

    async def commit_reference() -> None:
        async with repository.transaction() as transaction:
            record = await transaction.get_managed_artifact("acme", "research", uri)
            assert record is not None
            locked.set()
            await release.wait()
            await transaction.put_artifact_reference(
                ArtifactReferenceRecord(
                    id=artifact_reference_id("acme", "research", uri, "wl-reference-wins"),
                    uri=uri,
                    workload_id="wl-reference-wins",
                    org_id="acme",
                    project_id="research",
                )
            )

    reference_task = asyncio.create_task(commit_reference())
    await locked.wait()
    collection_task = asyncio.create_task(
        service.collect_artifacts_once(
            org_id="acme",
            project_id="research",
            grace_period_s=0,
            dry_run=False,
            now=(active.activated_at or active.updated_at) + timedelta(seconds=1),
        )
    )
    await asyncio.sleep(0.05)
    release.set()
    await reference_task
    try:
        first = await collection_task
    except TransientRepositoryError:
        first = None
    assert store.delete_calls == 0

    retry = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(active.activated_at or active.updated_at) + timedelta(seconds=2),
    )
    assert retry.referenced == 1
    assert store.delete_calls == 0
    assert store.exists(uri, org_id="acme", project_id="research")
    if first is not None:
        assert first.referenced == 1


@pytest.mark.asyncio
async def test_claim_commit_rejects_reference_hold_and_reupload_until_delete_finishes(repository, tmp_path) -> None:
    store = BlockingDeleteStore(tmp_path / "objects")
    service = _service(repository, store)
    payload = b"claim-wins"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    store.block_delete = True
    collection_task = asyncio.create_task(
        service.collect_artifacts_once(
            org_id="acme",
            project_id="research",
            grace_period_s=0,
            dry_run=False,
            now=(active.activated_at or active.updated_at) + timedelta(seconds=1),
        )
    )
    assert await asyncio.to_thread(store.delete_started.wait, 2)

    with pytest.raises(ConflictError, match="deleting"):
        await service.submit_workload(
            _principal(),
            org_id="acme",
            project_id="research",
            spec=_spec(uri),
            idempotency_key="postgres-claim-reference",
        )
    with pytest.raises(ConflictError, match="deleting"):
        await service.hold_artifact(_principal(), org_id="acme", project_id="research", uri=uri, reason="claim wins")
    with pytest.raises(ConflictError, match="deletion"):
        await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)

    store.allow_delete.set()
    result = await collection_task
    assert result.collected_uris == (uri,)
