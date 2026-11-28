# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Eligible-before-LIMIT artifact queries across shipped repository backends."""

from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri
from eyvan.records import ArtifactHold, ArtifactReferenceRecord, ManagedArtifactRecord, ManagedArtifactState, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService


@pytest.fixture(params=["memory", "local", "postgres"])
async def repository(request, tmp_path):
    if request.param != "postgres":
        repo = InMemoryRepository() if request.param == "memory" else LocalJsonRepository(tmp_path / "state.json")
        await repo.open()
        try:
            yield repo
        finally:
            await repo.close()
        return
    url = request.getfixturevalue("postgres_url")
    schema = f"artifact_gc_{uuid.uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        await create_schema(engine)
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


def _record(name, old, **updates):
    digest = hashlib.sha256(name.encode()).hexdigest()
    return ManagedArtifactRecord(
        uri=callable_uri(digest),
        kind="callable",
        org_id="acme",
        project_id="research",
        state=ManagedArtifactState.ACTIVE,
        size_bytes=10,
        sha256=digest,
        object_version=f"version:{name}",
        quarantine_reason=None,
        created_at=old,
        activated_at=old,
        updated_at=old,
        **updates,
    )


async def test_query_filters_every_protection_before_limit_and_preserves_rows(repository):
    moment = utcnow()
    old = moment - timedelta(days=30)
    held = _record("held", old, hold=ArtifactHold(reason="keep", held_by="alice"))
    expiring = _record(
        "future-hold", old, hold=ArtifactHold(reason="keep", held_by="alice", expires_at=moment + timedelta(seconds=1))
    )
    referenced = _record("live-reference", old)
    never = _record("never-retain", old).model_copy(update={"kind": "workspace"})
    fresh_activation = _record("fresh-activation", old).model_copy(update={"activated_at": moment})
    fresh_upload = _record("fresh-upload", old, upload_started_at=moment)
    eligible = _record("eligible", old + timedelta(days=1))
    expired_hold = _record(
        "expired-hold", old + timedelta(days=2), hold=ArtifactHold(reason="expired", held_by="alice", expires_at=moment)
    )
    records = [held, expiring, referenced, never, fresh_activation, fresh_upload, eligible, expired_hold]
    async with repository.transaction() as tx:
        for record in records:
            await tx.put_managed_artifact(record)
        await tx.put_artifact_reference(
            ArtifactReferenceRecord(
                id="live",
                org_id="acme",
                project_id="research",
                uri=referenced.uri,
                workload_id="wl-live",
            )
        )
        # Same URI in a different tenant cannot protect this tenant's object.
        await tx.put_artifact_reference(
            ArtifactReferenceRecord(
                id="foreign",
                org_id="other",
                project_id="research",
                uri=eligible.uri,
                workload_id="wl-foreign",
            )
        )
        await tx.put_artifact_reference(
            ArtifactReferenceRecord(
                id="released",
                org_id="acme",
                project_id="research",
                uri=expired_hold.uri,
                workload_id="wl-done",
                released_at=moment,
            )
        )
    async with repository.transaction() as tx:
        rows = await tx.list_collectable_managed_artifacts(
            org_id="acme",
            project_id="research",
            now=moment,
            grace_period_s=86400,
            retention_by_kind={"workspace": None},
            limit=1,
        )
        assert rows == [eligible]
        all_eligible = await tx.list_collectable_managed_artifacts(
            org_id="acme",
            project_id="research",
            now=moment,
            grace_period_s=86400,
            retention_by_kind={"workspace": None},
            limit=2,
        )
        assert all_eligible == [eligible, expired_hold]
        for record in records:
            assert await tx.get_managed_artifact(record.org_id, record.project_id, record.uri) == record


async def test_collection_reaches_tail_and_separately_counts_protected_prefix(repository, tmp_path):
    moment = utcnow()
    old = moment - timedelta(days=30)
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    payload = b"collectable tail"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    async with repository.transaction() as tx:
        tail = await tx.get_managed_artifact("acme", "research", uri)
        assert tail is not None
        tail_time = old + timedelta(days=1)
        await tx.put_managed_artifact(
            tail.model_copy(
                update={
                    "created_at": tail_time,
                    "activated_at": tail_time,
                    "upload_started_at": tail_time,
                    "updated_at": tail_time,
                }
            )
        )
        protected = [
            _record(f"prefix-{index}", old, hold=ArtifactHold(reason="keep", held_by="alice")) for index in range(20)
        ]
        for record in protected:
            await tx.put_managed_artifact(record)
    preview = await service.collect_artifacts_once(org_id="acme", project_id="research", now=moment, limit=1)
    assert preview.collected_uris == (uri,)
    assert (preview.held, preview.referenced, preview.within_grace) == (20, 0, 0)
    assert not preview.capped
    assert store.get(uri, org_id="acme", project_id="research") == payload
    collected = await service.collect_artifacts_once(
        org_id="acme", project_id="research", now=moment, limit=1, dry_run=False
    )
    assert collected.collected_uris == (uri,)
    assert not store.exists(uri, org_id="acme", project_id="research")
    async with repository.transaction() as tx:
        for record in protected:
            assert await tx.get_managed_artifact(record.org_id, record.project_id, record.uri) == record
        final = await tx.get_managed_artifact("acme", "research", uri)
        assert final.state is ManagedArtifactState.DELETED


@pytest.mark.parametrize("protection", ["hold", "reference", "retention"])
async def test_collection_revalidates_protection_and_counts_only_checked_batch(
    repository, tmp_path, monkeypatch, protection
):
    moment = utcnow()
    old = moment - timedelta(days=30)
    store = LocalManagedArtifactStore(tmp_path / "objects")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    payload = b"protection raced with eligible selection"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    async with repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", uri)
        assert record is not None
        await tx.put_managed_artifact(
            record.model_copy(
                update={
                    "created_at": old,
                    "activated_at": old,
                    "upload_started_at": old,
                    "updated_at": old,
                }
            )
        )
    original_transaction = repository.transaction
    transaction_count = 0

    @asynccontextmanager
    async def racing_transaction():
        nonlocal transaction_count
        transaction_count += 1
        if transaction_count == 2:
            # The candidate query transaction has closed. Publish the concurrent
            # protection in a genuine separate transaction before claim locking.
            async with original_transaction() as tx:
                current = await tx.get_managed_artifact("acme", "research", uri)
                if protection == "hold":
                    await tx.put_managed_artifact(
                        current.model_copy(
                            update={
                                "hold": ArtifactHold(reason="concurrent hold", held_by="alice"),
                            }
                        )
                    )
                elif protection == "reference":
                    await tx.put_artifact_reference(
                        ArtifactReferenceRecord(
                            id="raced-reference",
                            org_id="acme",
                            project_id="research",
                            uri=uri,
                            workload_id="wl-live",
                        )
                    )
                else:
                    await tx.put_managed_artifact(current.model_copy(update={"activated_at": moment}))
        async with original_transaction() as tx:
            yield tx

    monkeypatch.setattr(repository, "transaction", racing_transaction)
    result = await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        now=moment,
        dry_run=False,
    )
    assert result.collected_uris == ()
    assert (result.held, result.referenced, result.within_grace) == {
        "hold": (1, 0, 0),
        "reference": (0, 1, 0),
        "retention": (0, 0, 1),
    }[protection]
    assert store.get(uri, org_id="acme", project_id="research") == payload
    async with original_transaction() as tx:
        current = await tx.get_managed_artifact("acme", "research", uri)
        assert current.state is ManagedArtifactState.ACTIVE
        assert current.object_version == record.object_version
        assert current.record_generation == record.record_generation
        events = await tx.list_events(limit=100)
        assert not any(event.action in {"artifact.collected", "artifact.deletion_claimed"} for event in events)
