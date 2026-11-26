# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scoped, unique protection counters must not enlarge the deletion scan."""

from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest
from sqlalchemy import insert, text, update
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri
from eyvan.records import ArtifactHold, ArtifactReferenceRecord, ManagedArtifactRecord, ManagedArtifactState, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import (
    ArtifactReferenceRow,
    ManagedArtifactRow,
    SqlAlchemyRepository,
    create_schema,
)
from eyvan.server.service import ControlPlaneService


@pytest.fixture(params=["memory", "local_json", "sqlite", "postgres"])
async def repository(request, tmp_path):
    if request.param == "memory":
        repo = InMemoryRepository()
    elif request.param == "local_json":
        repo = LocalJsonRepository(tmp_path / "state.json")
    else:
        admin = None
        if request.param == "postgres":
            url = request.getfixturevalue("postgres_url")
            schema = "artifact_counters_" + uuid.uuid4().hex
            admin = create_async_engine(url)
            async with admin.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
        else:
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'counters.db'}")
        await create_schema(engine)
        repo = SqlAlchemyRepository(engine)
    await repo.open()
    try:
        yield repo
    finally:
        await repo.close()
        if request.param in {"sqlite", "postgres"}:
            await engine.dispose()
            if admin is not None:
                async with admin.begin() as connection:
                    await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                await admin.dispose()


def _record(name, now, *, org="acme", project="research", kind="callable", hold=None, recent=False):
    digest = hashlib.sha256(name.encode()).hexdigest()
    old = now - timedelta(days=2)
    return ManagedArtifactRecord(
        uri=callable_uri(digest),
        kind=kind,
        org_id=org,
        project_id=project,
        state=ManagedArtifactState.ACTIVE,
        size_bytes=1,
        sha256=digest,
        object_version="seed-version",
        created_at=old,
        updated_at=old,
        activated_at=old,
        upload_started_at=now if recent else old,
        hold=hold,
        quarantine_reason=None,
    )


async def _seed(repository, records, references=()):
    async with repository.transaction() as tx:
        for record in records:
            await tx.put_managed_artifact(record)
        for ref in references:
            await tx.put_artifact_reference(ref)


def _ref(record, workload, *, released_at=None):
    return ArtifactReferenceRecord(
        id=workload,
        uri=record.uri,
        workload_id=workload,
        org_id=record.org_id,
        project_id=record.project_id,
        released_at=released_at,
    )


async def test_scoped_unique_categories_prefer_hold_then_reference_then_grace(repository, tmp_path):
    now = utcnow()
    hold = ArtifactHold(reason="protected", held_by="alice")
    expired = ArtifactHold(reason="expired", held_by="alice", expires_at=now)
    records = [
        _record("all-three", now, hold=hold, recent=True),
        _record("reference-and-grace", now, recent=True),
        _record("recent-upload", now, recent=True),
        _record("retain-forever", now, kind="workspace"),
        _record("expired-hold", now, hold=expired),
        _record("released-reference", now),
        _record("all-three", now, org="other", hold=hold),
        _record("all-three", now, project="other", hold=hold),
    ]
    references = [
        _ref(records[0], "held-live"),
        _ref(records[1], "live-a"),
        _ref(records[1], "live-b"),
        _ref(records[5], "released", released_at=now),
        _ref(records[6], "other-org"),
        _ref(records[7], "other-project"),
    ]
    # Foreign live references sharing a URI must not protect this scope's tail.
    references.append(_ref(records[4].model_copy(update={"project_id": "elsewhere"}), "foreign-only"))
    await _seed(repository, records, references)
    if isinstance(repository, LocalJsonRepository):
        await repository.close()
        await repository.open()
    service = ControlPlaneService(repository, artifact_store=LocalManagedArtifactStore(tmp_path / "bytes"))
    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        now=now,
        grace_period_s=60,
        retention_by_kind={"workspace": None},
    )
    assert (result.held, result.referenced, result.within_grace) == (1, 1, 2)
    assert set(result.collected_uris) == {records[4].uri, records[5].uri}
    assert result.capped is False


async def test_more_than_scan_limit_protections_are_counted_without_decoding_or_starving_tail(
    repository,
    tmp_path,
    monkeypatch,
):
    now = utcnow()
    hold = ArtifactHold(reason="keep", held_by="alice", expires_at=now + timedelta(hours=1))
    records = [_record(f"protected-{index}", now, hold=hold) for index in range(10_002)]
    tail = _record("collectible-tail", now)
    if isinstance(repository, SqlAlchemyRepository):
        # Bulk fixture insertion; the production queries must return aggregates,
        # not these 10,002 JSON records.
        async with repository.engine.begin() as connection:
            await connection.execute(
                insert(ManagedArtifactRow),
                [
                    {
                        "org_id": r.org_id,
                        "project_id": r.project_id,
                        "uri": r.uri,
                        "kind": r.kind,
                        "state": r.state.value,
                        "record_generation": r.record_generation,
                        "object_version": r.object_version,
                        "size_bytes": r.size_bytes,
                        "created_at": r.created_at,
                        "updated_at": r.updated_at,
                        "data": r.model_dump(mode="json"),
                    }
                    for r in [*records, tail]
                ],
            )
        import eyvan.server.persistence.sqlalchemy as sql_module

        decode = sql_module._decode_managed_artifact_rows

        def bounded_decode(rows):
            rows = list(rows)
            assert len(rows) <= 1, "protection statistics must not decode the protected inventory"
            return decode(rows)

        monkeypatch.setattr(sql_module, "_decode_managed_artifact_rows", bounded_decode)
    else:
        await _seed(repository, [*records, tail])
    service = ControlPlaneService(repository, artifact_store=LocalManagedArtifactStore(tmp_path / "bytes"))
    result = await service.collect_artifacts_once(org_id="acme", project_id="research", now=now, limit=1)
    assert result.held == 10_002
    assert (result.referenced, result.within_grace) == (0, 0)
    assert result.collected_uris == (tail.uri,)
    assert result.capped is False


@pytest.mark.parametrize("repository", ["sqlite", "postgres"], indirect=True)
@pytest.mark.parametrize("damage", ["future_field", "wrong_scope"])
async def test_unreadable_typed_live_references_count_once_and_preserve_bytes(repository, tmp_path, damage):
    payload = b"counter-corrupt-reference"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    store = LocalManagedArtifactStore(tmp_path / "bytes")
    service = ControlPlaneService(repository, artifact_store=store)
    record = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    references = [_ref(record, "live-a"), _ref(record, "live-b")]
    await _seed(repository, [], references)
    damaged = references[0].model_dump(mode="json")
    damaged.update({"future_contract": True} if damage == "future_field" else {"project_id": "wrong"})
    async with repository.engine.begin() as connection:
        await connection.execute(
            update(ArtifactReferenceRow).where(ArtifactReferenceRow.workload_id == "live-a").values(data=damaged)
        )
    result = await service.collect_artifacts_once(org_id="acme", project_id="research", grace_period_s=0, dry_run=False)
    assert result.referenced == 1
    assert result.collected_uris == ()
    assert store.get(uri, org_id="acme", project_id="research") == payload
    # The destructive revalidation API must still refuse the unreadable row.
    with pytest.raises(RuntimeError, match="active artifact reference"):
        async with repository.transaction() as tx:
            await tx.list_artifact_references(org_id="acme", project_id="research", uri=uri, active_only=True)


@pytest.mark.parametrize("add_hold", [True, False])
async def test_new_simultaneous_protections_are_revalidated_without_double_counting(
    repository, tmp_path, monkeypatch, add_hold
):
    now = utcnow()
    payload = b"concurrent-counter-protection"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    store = LocalManagedArtifactStore(tmp_path / "bytes")
    service = ControlPlaneService(repository, artifact_store=store)
    record = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    old = now - timedelta(days=2)
    record = record.model_copy(update={"created_at": old, "activated_at": old, "upload_started_at": old})
    already_held = _record("existing-held", now, hold=ArtifactHold(reason="keep", held_by="alice"))
    await _seed(repository, [record, already_held])
    transaction = repository.transaction
    initial_completed = False

    @asynccontextmanager
    async def protect_after_snapshot():
        nonlocal initial_completed
        async with transaction() as tx:
            yield tx
        if not initial_completed:
            initial_completed = True
            async with transaction() as tx:
                current = await tx.get_managed_artifact("acme", "research", uri)
                await tx.put_managed_artifact(
                    current.model_copy(
                        update={
                            "hold": ArtifactHold(reason="concurrent", held_by="alice") if add_hold else None,
                            "upload_started_at": now,
                        }
                    )
                )
                await tx.put_artifact_reference(_ref(record, "concurrent-live"))

    monkeypatch.setattr(repository, "transaction", protect_after_snapshot)
    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=60,
        now=now,
        dry_run=False,
    )
    assert (result.held, result.referenced, result.within_grace) == ((2, 0, 0) if add_hold else (1, 1, 0))
    assert result.collected_uris == ()
    assert store.get(uri, org_id="acme", project_id="research") == payload


async def test_only_active_inventory_contributes_to_scoped_protection_counts(repository, tmp_path):
    now = utcnow()
    hold = ArtifactHold(reason="keep", held_by="alice")
    records = []
    for state in ManagedArtifactState:
        record = _record(state.value, now, hold=hold, recent=True)
        updates = {"state": state}
        if state is ManagedArtifactState.PENDING_UPLOAD:
            updates.update(object_version=None, activated_at=None)
        elif state is ManagedArtifactState.DELETING:
            updates.update(deletion_claim_id="claim", deletion_started_at=now)
        elif state is ManagedArtifactState.DELETED:
            updates.update(size_bytes=0, deleted_at=now)
        elif state is ManagedArtifactState.QUARANTINED:
            updates.update(quarantine_reason="not active")
        records.append(ManagedArtifactRecord.model_validate(record.model_dump() | updates))
    await _seed(repository, records, [_ref(record, record.state.value) for record in records])
    service = ControlPlaneService(repository, artifact_store=LocalManagedArtifactStore(tmp_path / "bytes"))
    result = await service.collect_artifacts_once(org_id="acme", project_id="research", now=now)
    assert (result.held, result.referenced, result.within_grace) == (1, 0, 0)
    assert result.collected_uris == tuple(
        record.uri for record in records if record.state is ManagedArtifactState.DELETING
    )


@pytest.mark.parametrize("repository", ["sqlite", "postgres"], indirect=True)
async def test_unreadable_reference_published_after_selection_refuses_collection(repository, tmp_path, monkeypatch):
    payload = b"corrupt-live-reference-race"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    store = LocalManagedArtifactStore(tmp_path / "bytes")
    service = ControlPlaneService(repository, artifact_store=store)
    record = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    transaction = repository.transaction
    snapshot_completed = False

    @asynccontextmanager
    async def corrupt_after_snapshot():
        nonlocal snapshot_completed
        async with transaction() as tx:
            yield tx
        if not snapshot_completed:
            snapshot_completed = True
            reference = _ref(record, "corrupt-racing-reference")
            async with transaction() as tx:
                await tx.put_artifact_reference(reference)
            async with repository.engine.begin() as connection:
                await connection.execute(
                    update(ArtifactReferenceRow).values(
                        data=reference.model_dump(mode="json") | {"future_contract": True}
                    )
                )

    monkeypatch.setattr(repository, "transaction", corrupt_after_snapshot)
    with pytest.raises(RuntimeError, match="active artifact reference"):
        await service.collect_artifacts_once(org_id="acme", project_id="research", grace_period_s=0, dry_run=False)
    assert store.get(uri, org_id="acme", project_id="research") == payload
    async with transaction() as tx:
        assert await tx.get_managed_artifact("acme", "research", uri) == record


@pytest.mark.parametrize("repository", ["sqlite"], indirect=True)
async def test_protection_released_between_queries_then_restored_is_never_counted_twice(
    repository, tmp_path, monkeypatch
):
    from eyvan.server.persistence.sqlalchemy import SqlAlchemySession

    payload = b"protection-restored-between-queries"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    store = LocalManagedArtifactStore(tmp_path / "bytes")
    service = ControlPlaneService(repository, artifact_store=store)
    record = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    held_record = record.model_copy(update={"hold": ArtifactHold(reason="keep", held_by="alice")})
    await _seed(repository, [held_record])
    transaction = repository.transaction
    count = SqlAlchemySession.count_managed_artifact_protections
    released = False
    first_transaction = True

    async def release_after_count(self, **kwargs):
        nonlocal released
        result = await count(self, **kwargs)
        if not released:
            released = True
            async with transaction() as tx:
                await tx.put_managed_artifact(record)
        return result

    @asynccontextmanager
    async def restore_after_snapshot():
        nonlocal first_transaction
        async with transaction() as tx:
            yield tx
        if first_transaction:
            first_transaction = False
            async with transaction() as tx:
                await tx.put_managed_artifact(held_record)

    monkeypatch.setattr(SqlAlchemySession, "count_managed_artifact_protections", release_after_count)
    monkeypatch.setattr(repository, "transaction", restore_after_snapshot)
    result = await service.collect_artifacts_once(org_id="acme", project_id="research", grace_period_s=0, dry_run=False)
    assert result.held == 1
    assert result.collected_uris == ()
    assert store.get(uri, org_id="acme", project_id="research") == payload
