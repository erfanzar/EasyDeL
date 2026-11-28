# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generation and provider-version fences for managed artifact lifecycle I/O."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from datetime import timedelta

import pytest

from eyvan.artifacts.managed import (
    LocalManagedArtifactStore,
    ManagedArtifactConflictError,
    ManagedArtifactDeleteInconclusiveError,
    ManagedArtifactDeleteResult,
    ManagedArtifactDeleteStatus,
    ManagedArtifactObjectMetadata,
    ManagedArtifactRead,
    ManagedArtifactVersionMismatchError,
    ManagedArtifactWrite,
    callable_uri,
    result_uri,
    storage_scope,
)
from eyvan.records import (
    CallableExecution,
    ManagedArtifactState,
    Principal,
    QuotaRecord,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import ConflictError, InputError, UnavailableError


class VersionedMemoryStore:
    """Deterministic exact-current store fake with optional delete barriers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[tuple[str, str, str], tuple[bytes, int]] = {}
        self._next_version = 1
        self.delete_started = threading.Event()
        self.allow_delete = threading.Event()
        self.block_delete = False
        self.fail_delete_before_mutation = False
        self.fail_delete_after_mutation = False
        self.io_guard = lambda: None

    @staticmethod
    def _key(uri: str, org_id: str, project_id: str) -> tuple[str, str, str]:
        return org_id, project_id, uri

    @staticmethod
    def _metadata(uri: str, payload: bytes, version: int) -> ManagedArtifactObjectMetadata:
        return ManagedArtifactObjectMetadata(
            uri=uri,
            size_bytes=len(payload),
            object_version=f"fake:{version}",
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def put_versioned(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> ManagedArtifactWrite:
        self.io_guard()
        key = self._key(uri, org_id, project_id)
        with self._lock:
            current = self._objects.get(key)
            if current is not None:
                existing, version = current
                if existing != payload:
                    raise ManagedArtifactConflictError(
                        "different bytes",
                        current_metadata=self._metadata(uri, existing, version),
                    )
                return ManagedArtifactWrite(self._metadata(uri, existing, version), created=False)
            version = self._next_version
            self._next_version += 1
            self._objects[key] = payload, version
            return ManagedArtifactWrite(self._metadata(uri, payload, version), created=True)

    def get_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str | None = None,
    ) -> ManagedArtifactRead:
        self.io_guard()
        with self._lock:
            current = self._objects.get(self._key(uri, org_id, project_id))
            if current is None:
                raise FileNotFoundError(uri)
            payload, version = current
            metadata = self._metadata(uri, payload, version)
            if object_version is not None and metadata.object_version != object_version:
                raise ManagedArtifactVersionMismatchError("different version", current_metadata=metadata)
            return ManagedArtifactRead(payload=payload, metadata=metadata)

    def delete_versioned(
        self, uri: str, *, org_id: str, project_id: str, object_version: str
    ) -> ManagedArtifactDeleteResult:
        self.io_guard()
        self.delete_started.set()
        if self.block_delete and not self.allow_delete.wait(timeout=3):
            raise RuntimeError("delete barrier timed out")
        if self.fail_delete_before_mutation:
            self.fail_delete_before_mutation = False
            raise ManagedArtifactDeleteInconclusiveError("request did not reach storage")
        key = self._key(uri, org_id, project_id)
        with self._lock:
            current = self._objects.get(key)
            if current is None:
                return ManagedArtifactDeleteResult(ManagedArtifactDeleteStatus.ABSENT)
            payload, version = current
            metadata = self._metadata(uri, payload, version)
            if metadata.object_version != object_version:
                return ManagedArtifactDeleteResult(
                    ManagedArtifactDeleteStatus.VERSION_MISMATCH,
                    current_metadata=metadata,
                )
            del self._objects[key]
        if self.fail_delete_after_mutation:
            self.fail_delete_after_mutation = False
            raise ManagedArtifactDeleteInconclusiveError("lost response")
        return ManagedArtifactDeleteResult(ManagedArtifactDeleteStatus.DELETED)

    def put(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> str:
        return self.put_versioned(uri, payload, org_id=org_id, project_id=project_id).metadata.uri

    def get(self, uri: str, *, org_id: str, project_id: str) -> bytes:
        return self.get_versioned(uri, org_id=org_id, project_id=project_id).payload

    def exists(self, uri: str, *, org_id: str, project_id: str) -> bool:
        with self._lock:
            return self._key(uri, org_id, project_id) in self._objects

    def delete(self, uri: str, *, org_id: str, project_id: str) -> bool:
        with self._lock:
            current = self._objects.get(self._key(uri, org_id, project_id))
        if current is None:
            return False
        result = self.delete_versioned(
            uri,
            org_id=org_id,
            project_id=project_id,
            object_version=f"fake:{current[1]}",
        )
        return result.status is ManagedArtifactDeleteStatus.DELETED

    def replace_out_of_band(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> None:
        key = self._key(uri, org_id, project_id)
        with self._lock:
            version = self._next_version
            self._next_version += 1
            self._objects[key] = payload, version


async def _service() -> tuple[ControlPlaneService, VersionedMemoryStore]:
    repository = InMemoryRepository()
    await repository.open()
    store = VersionedMemoryStore()
    return (
        ControlPlaneService(
            repository,
            artifact_store=store,
            callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
        ),
        store,
    )


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )


def _spec(uri: str) -> WorkloadDefinition:
    digest = uri.rsplit("/", 1)[-1]
    return WorkloadDefinition(
        name="uses-artifact",
        execution=CallableExecution(kind="callable", artifact_uri=uri, sha256=digest, python="3.11"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1)),
    )


async def _record(service: ControlPlaneService, uri: str):
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", uri)
    assert record is not None
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0, True, "1"))
async def test_lifecycle_time_windows_reject_non_finite_or_non_numeric_values(value: object) -> None:
    service, _store = await _service()
    with pytest.raises(InputError, match="finite and non-negative"):
        await service.reconcile_artifact_lifecycle_once(
            org_id="acme",
            project_id="research",
            recovery_grace_s=value,  # type: ignore[arg-type]
        )
    with pytest.raises(InputError, match="finite and non-negative"):
        await service.collect_artifacts_once(
            org_id="acme",
            project_id="research",
            grace_period_s=value,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_per_kind_retention_window_rejects_nan() -> None:
    service, _store = await _service()
    with pytest.raises(InputError, match="retention windows"):
        await service.collect_artifacts_once(
            org_id="acme",
            project_id="research",
            retention_by_kind={"logs": float("nan")},
        )


@pytest.mark.asyncio
async def test_no_physical_store_io_runs_while_the_repository_transaction_is_locked() -> None:
    service, store = await _service()

    def assert_unlocked() -> None:
        assert not service.repository._transaction_lock.locked()

    store.io_guard = assert_unlocked
    payload = b"transaction-boundary"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    assert await service.read_managed_artifact(org_id="acme", project_id="research", uri=uri) == payload
    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(active.activated_at or utcnow()) + timedelta(seconds=1),
    )
    assert result.collected_uris == (uri,)


@pytest.mark.asyncio
async def test_local_read_permission_failure_is_unavailable_not_absent_or_corrupt(tmp_path) -> None:
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(
        repository,
        artifact_store=store,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
    )
    payload = b"permission-failure"
    digest = hashlib.sha256(payload).hexdigest()
    uri = callable_uri(digest)
    active = await service.upload_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        payload=payload,
    )
    target = next((store.root / "scopes").rglob(f"{digest}.pkl"))
    target.chmod(0)
    try:
        with pytest.raises(UnavailableError, match="read is inconclusive"):
            await service.read_managed_artifact(
                org_id="acme",
                project_id="research",
                uri=uri,
            )
    finally:
        target.chmod(0o400)

    unchanged = await _record(service, uri)
    assert unchanged.state is ManagedArtifactState.ACTIVE
    assert unchanged.record_generation == active.record_generation
    assert unchanged.object_version == active.object_version


@pytest.mark.asyncio
async def test_reserve_write_activate_and_exact_read_are_distinct_boundaries() -> None:
    service, store = await _service()
    payload = b"payload"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert reserved.state is ManagedArtifactState.PENDING_UPLOAD
    with pytest.raises(ConflictError, match="pending_upload"):
        await service.read_managed_artifact(org_id="acme", project_id="research", uri=uri)

    written = store.put_versioned(uri, payload, org_id="acme", project_id="research")
    activated = await service.activate_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        record_generation=reserved.record_generation,
        metadata=written.metadata,
    )
    assert activated.state is ManagedArtifactState.ACTIVE
    assert activated.object_version == written.metadata.object_version
    assert await service.read_managed_artifact(org_id="acme", project_id="research", uri=uri) == payload


@pytest.mark.asyncio
async def test_pending_upload_cannot_be_referenced_or_held_before_activation() -> None:
    service, _store = await _service()
    payload = b"not-active"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ConflictError, match="pending_upload"):
        await service.submit_workload(
            _principal(),
            org_id="acme",
            project_id="research",
            spec=_spec(uri),
            idempotency_key="pending-reference",
        )
    with pytest.raises(ConflictError, match="pending_upload"):
        await service.hold_artifact(_principal(), org_id="acme", project_id="research", uri=uri, reason="too early")


@pytest.mark.asyncio
async def test_recovery_treats_pending_and_tombstone_read_races_as_inconclusive() -> None:
    service, store = await _service()
    payload = b"changing-during-recovery"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    original_get = store.get_versioned

    def changing(*args, **kwargs):
        del args, kwargs
        raise ManagedArtifactVersionMismatchError("changed during exact observation")

    store.get_versioned = changing  # type: ignore[method-assign]
    pending_result = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=reserved.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert pending_result["inconclusive"] == 1
    assert (await _record(service, uri)).state is ManagedArtifactState.PENDING_UPLOAD

    store.get_versioned = original_get  # type: ignore[method-assign]
    released = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=reserved.updated_at + timedelta(seconds=3),
        recovery_grace_s=1,
    )
    assert released["released"] == 1
    tombstone = await _record(service, uri)
    store.get_versioned = changing  # type: ignore[method-assign]
    tombstone_result = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=tombstone.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert tombstone_result["inconclusive"] == 1
    assert (await _record(service, uri)).state is ManagedArtifactState.DELETED


@pytest.mark.asyncio
async def test_expired_pending_absence_releases_quota_as_a_tombstone() -> None:
    service, _store = await _service()
    payload = b"pending"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    result = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=reserved.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    record = await _record(service, uri)
    assert result["released"] == 1
    assert record.state is ManagedArtifactState.DELETED
    assert record.size_bytes == 0
    assert record.record_generation == reserved.record_generation


@pytest.mark.asyncio
async def test_stale_upload_cannot_delete_the_provider_version_activated_by_its_retry() -> None:
    service, store = await _service()
    payload = b"stale-upload-retry"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    original_put = store.put_versioned
    first_put_started = threading.Event()
    allow_first_put = threading.Event()
    put_calls = 0

    def pause_first_put(
        value_uri: str,
        value_payload: bytes,
        *,
        org_id: str,
        project_id: str,
    ) -> ManagedArtifactWrite:
        nonlocal put_calls
        put_calls += 1
        if put_calls == 1:
            first_put_started.set()
            if not allow_first_put.wait(timeout=3):
                raise RuntimeError("stale upload barrier timed out")
        return original_put(value_uri, value_payload, org_id=org_id, project_id=project_id)

    store.put_versioned = pause_first_put  # type: ignore[method-assign]
    stale_upload = asyncio.create_task(
        service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=uri,
            payload=payload,
        )
    )
    assert await asyncio.to_thread(first_put_started.wait, 3)
    first_reservation = await _record(service, uri)
    assert first_reservation.state is ManagedArtifactState.PENDING_UPLOAD

    recovered = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=first_reservation.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert recovered["released"] == 1
    assert (await _record(service, uri)).state is ManagedArtifactState.DELETED

    retry = await service.upload_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        payload=payload,
    )
    assert retry.state is ManagedArtifactState.ACTIVE
    assert retry.record_generation == first_reservation.record_generation + 1
    allow_first_put.set()
    with pytest.raises(ConflictError, match="activation is stale"):
        await stale_upload

    final = await _record(service, uri)
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
async def test_late_stale_payload_charges_its_full_size_when_the_retry_is_quarantined() -> None:
    service, store = await _service()
    stale_payload = b"a" * 100
    retry_payload = b"b"
    uri = result_uri("late-stale-payload")
    original_put = store.put_versioned
    first_put_started = threading.Event()
    allow_first_put = threading.Event()
    put_calls = 0

    def pause_first_put(
        value_uri: str,
        value_payload: bytes,
        *,
        org_id: str,
        project_id: str,
    ) -> ManagedArtifactWrite:
        nonlocal put_calls
        put_calls += 1
        if put_calls == 1:
            first_put_started.set()
            if not allow_first_put.wait(timeout=3):
                raise RuntimeError("stale payload barrier timed out")
        return original_put(value_uri, value_payload, org_id=org_id, project_id=project_id)

    store.put_versioned = pause_first_put  # type: ignore[method-assign]
    stale_upload = asyncio.create_task(
        service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=uri,
            payload=stale_payload,
        )
    )
    assert await asyncio.to_thread(first_put_started.wait, 3)
    first_reservation = await _record(service, uri)
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
    assert retry_reservation.record_generation == first_reservation.record_generation + 1

    allow_first_put.set()
    with pytest.raises(ConflictError, match="is stale"):
        await stale_upload
    with pytest.raises(ConflictError, match="physical object conflicts"):
        await service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=uri,
            payload=retry_payload,
        )

    quarantined = await _record(service, uri)
    assert quarantined.state is ManagedArtifactState.QUARANTINED
    assert quarantined.record_generation == retry_reservation.record_generation
    assert quarantined.size_bytes == len(stale_payload)
    assert store.get(uri, org_id="acme", project_id="research") == stale_payload
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == len(stale_payload)


@pytest.mark.asyncio
async def test_recovery_activates_an_exact_write_left_after_the_first_commit() -> None:
    service, store = await _service()
    payload = b"recover"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    written = store.put_versioned(uri, payload, org_id="acme", project_id="research")
    result = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=reserved.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    record = await _record(service, uri)
    assert result["activated"] == 1
    assert record.state is ManagedArtifactState.ACTIVE
    assert record.object_version == written.metadata.object_version


@pytest.mark.asyncio
async def test_recovery_quarantines_a_pending_physical_mismatch() -> None:
    service, store = await _service()
    intended = b"intended"
    uri = callable_uri(hashlib.sha256(intended).hexdigest())
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(intended),
        sha256=hashlib.sha256(intended).hexdigest(),
    )
    store.replace_out_of_band(uri, b"different", org_id="acme", project_id="research")
    result = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=reserved.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert result["quarantined"] == 1
    assert (await _record(service, uri)).state is ManagedArtifactState.QUARANTINED


@pytest.mark.asyncio
async def test_a_durable_delete_claim_rejects_reference_hold_and_reupload_races() -> None:
    service, store = await _service()
    payload = b"race"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    store.block_delete = True
    collection = asyncio.create_task(
        service.collect_artifacts_once(
            org_id="acme",
            project_id="research",
            principal=_principal(),
            grace_period_s=0,
            dry_run=False,
            now=(active.activated_at or utcnow()) + timedelta(seconds=1),
        )
    )
    assert await asyncio.to_thread(store.delete_started.wait, 2)
    claimed = await _record(service, uri)
    assert claimed.state is ManagedArtifactState.DELETING

    with pytest.raises(ConflictError, match="deleting"):
        await service.submit_workload(
            _principal(),
            org_id="acme",
            project_id="research",
            spec=_spec(uri),
            idempotency_key="race-reference",
        )
    with pytest.raises(ConflictError, match="deleting"):
        await service.hold_artifact(_principal(), org_id="acme", project_id="research", uri=uri, reason="race")
    with pytest.raises(ConflictError, match="deletion"):
        await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)

    store.allow_delete.set()
    result = await collection
    assert result.collected_uris == (uri,)
    assert (await _record(service, uri)).state is ManagedArtifactState.DELETED


@pytest.mark.asyncio
async def test_a_committed_reference_or_hold_wins_before_the_collector_claim() -> None:
    service, store = await _service()
    payload = b"protected"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(uri),
        idempotency_key="protected-reference",
    )
    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(active.activated_at or utcnow()) + timedelta(seconds=1),
    )
    assert result.referenced == 1
    assert store.exists(uri, org_id="acme", project_id="research")

    await service.cancel_workload(_principal(), workload.id, idempotency_key="cancel-protected")
    await service.settle_terminal_workload(
        Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),)),
        workload_id=workload.id,
        target=WorkloadState.CANCELLED,
    )
    await service.hold_artifact(_principal(), org_id="acme", project_id="research", uri=uri, reason="hold wins")
    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(active.activated_at or utcnow()) + timedelta(seconds=2),
    )
    assert result.held == 1
    assert store.exists(uri, org_id="acme", project_id="research")


@pytest.mark.asyncio
async def test_delete_failure_before_mutation_recovers_the_same_exact_version() -> None:
    service, store = await _service()
    payload = b"before-delete"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    store.fail_delete_before_mutation = True
    first = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(active.activated_at or utcnow()) + timedelta(seconds=1),
    )
    assert first.collected_uris == ()
    claim = await _record(service, uri)
    assert claim.state is ManagedArtifactState.DELETING
    assert store.exists(uri, org_id="acme", project_id="research")

    recovered = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=claim.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert recovered["deleted"] == 1
    assert not store.exists(uri, org_id="acme", project_id="research")
    assert (await _record(service, uri)).state is ManagedArtifactState.DELETED


@pytest.mark.asyncio
async def test_delete_response_loss_recovers_absence_without_unconditional_retry() -> None:
    service, store = await _service()
    payload = b"response-loss"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    store.fail_delete_after_mutation = True
    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(active.activated_at or utcnow()) + timedelta(seconds=1),
    )
    assert result.collected_uris == ()
    claim = await _record(service, uri)
    assert claim.state is ManagedArtifactState.DELETING

    recovered = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=claim.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert recovered["deleted"] == 1
    assert (await _record(service, uri)).state is ManagedArtifactState.DELETED


@pytest.mark.asyncio
async def test_stale_delete_version_never_removes_a_reuploaded_generation() -> None:
    service, store = await _service()
    first_payload = b"first"
    uri = callable_uri(hashlib.sha256(first_payload).hexdigest())
    first = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=first_payload)
    await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(first.activated_at or utcnow()) + timedelta(seconds=1),
    )
    tombstone = await _record(service, uri)
    second = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=first_payload)
    stale = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=first.object_version or "",
    )
    assert stale.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert store.exists(uri, org_id="acme", project_id="research")
    assert second.record_generation > tombstone.record_generation


@pytest.mark.asyncio
async def test_delete_version_mismatch_charges_the_larger_quarantined_replacement() -> None:
    service, store = await _service()
    payload = b"a"
    replacement = b"b" * 100
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    active = await service.upload_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        payload=payload,
    )
    store.replace_out_of_band(uri, replacement, org_id="acme", project_id="research")

    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        dry_run=False,
        now=(active.activated_at or utcnow()) + timedelta(seconds=1),
    )

    assert result.collected_uris == ()
    quarantined = await _record(service, uri)
    assert quarantined.state is ManagedArtifactState.QUARANTINED
    assert quarantined.size_bytes == len(replacement)
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == len(replacement)


@pytest.mark.asyncio
async def test_out_of_band_active_replacement_quarantines_instead_of_returning_bytes() -> None:
    service, store = await _service()
    payload = b"original"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    store.replace_out_of_band(uri, b"replacement", org_id="acme", project_id="research")
    with pytest.raises(ConflictError, match="outside the lifecycle"):
        await service.read_managed_artifact(org_id="acme", project_id="research", uri=uri)
    quarantined = await _record(service, uri)
    assert quarantined.state is ManagedArtifactState.QUARANTINED
    assert quarantined.size_bytes == len(b"replacement")
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == len(b"replacement")


@pytest.mark.asyncio
async def test_late_invalid_callable_bytes_after_tombstone_are_quarantined_and_charged(tmp_path) -> None:
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(
        repository,
        artifact_store=store,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
    )
    intended = b"a"
    replacement = b"b" * 100
    digest = hashlib.sha256(intended).hexdigest()
    uri = callable_uri(digest)
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(intended),
        sha256=digest,
    )
    released = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=reserved.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert released["released"] == 1
    target = store.root / "scopes" / storage_scope("acme", "research") / "callables" / f"{digest}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(replacement)
    target.chmod(0o400)

    recovered = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=(await _record(service, uri)).updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )

    assert recovered["quarantined"] == 1
    quarantined = await _record(service, uri)
    assert quarantined.state is ManagedArtifactState.QUARANTINED
    assert quarantined.size_bytes == len(replacement)
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == len(replacement)


@pytest.mark.asyncio
async def test_late_bytes_after_expired_reservation_are_quarantined_not_deleted() -> None:
    service, store = await _service()
    payload = b"late"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=reserved.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    store.put_versioned(uri, payload, org_id="acme", project_id="research")
    tombstone = await _record(service, uri)
    result = await service.reconcile_artifact_lifecycle_once(
        org_id="acme",
        project_id="research",
        now=tombstone.updated_at + timedelta(seconds=2),
        recovery_grace_s=1,
    )
    assert result["quarantined"] == 1
    assert (await _record(service, uri)).state is ManagedArtifactState.QUARANTINED
    assert store.exists(uri, org_id="acme", project_id="research")


@pytest.mark.asyncio
async def test_concurrent_different_uploads_serialize_the_shared_quota_reservation() -> None:
    service, _store = await _service()
    await service.put_quota(
        Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),)),
        quota=QuotaRecord(
            org_id="acme",
            project_id="research",
            limits=ResourceVector(cpu=10),
            artifact_bytes_limit=10,
            updated_by="root",
        ),
        idempotency_key="artifact-quota",
    )
    payloads = (b"aaaaaa", b"bbbbbb")

    async def upload(payload: bytes):
        return await service.upload_managed_artifact(
            org_id="acme",
            project_id="research",
            uri=callable_uri(hashlib.sha256(payload).hexdigest()),
            payload=payload,
        )

    outcomes = await asyncio.gather(*(upload(payload) for payload in payloads), return_exceptions=True)
    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, InputError) for outcome in outcomes) == 1
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == 6


@pytest.mark.asyncio
async def test_quarantined_metadata_is_never_read_as_if_it_were_active() -> None:
    service, _store = await _service()
    uri = callable_uri("a" * 64)
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=1, sha256="a" * 64)
    with pytest.raises(ConflictError, match="quarantined"):
        await service.read_managed_artifact(org_id="acme", project_id="research", uri=uri)
