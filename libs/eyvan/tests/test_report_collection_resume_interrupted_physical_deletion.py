# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Report-only collection must not resume an interrupted physical deletion."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from eyvan.artifacts.managed import LocalManagedArtifactStore, ManagedArtifactDeleteInconclusiveError, callable_uri
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import ArtifactHold, ManagedArtifactRecord, ManagedArtifactState, Principal, Role, RoleGrant, utcnow
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


async def _snapshot(service, uri):
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", uri)
        events = await tx.list_events(org_id="acme", project_id="research", limit=1000)
    assert record is not None
    return record, events


async def _interrupt(service, store, monkeypatch):
    payload = b"writer-087 interrupted local deletion"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", uri)
        assert record is not None
        old = utcnow() - timedelta(days=30)
        await tx.put_managed_artifact(
            record.model_copy(
                update={"created_at": old, "activated_at": old, "upload_started_at": old, "updated_at": old}
            )
        )
    delete = store.delete_versioned
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(kwargs["object_version"])
        if len(calls) == 1:
            raise ManagedArtifactDeleteInconclusiveError("injected before local storage mutation")
        return delete(*args, **kwargs)

    monkeypatch.setattr(store, "delete_versioned", fail_once)
    result = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)
    assert result.collected_uris == ()
    record, _events = await _snapshot(service, uri)
    assert record.state is ManagedArtifactState.DELETING
    assert store.get(uri, org_id="acme", project_id="research") == payload
    assert len(calls) == 1
    return uri, payload, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("service-default", "service-explicit", "http-default", "http-explicit"))
@pytest.mark.parametrize("recovery", ("collection", "lifecycle"))
async def test_dry_run_preserves_interrupted_claim_bytes_and_audit(tmp_path, monkeypatch, surface, recovery):
    settings = EyvanSettings(
        dev_auth_enabled=True,
        repository_backend=RepositoryBackend.MEMORY,
        artifact_store_uri=str(tmp_path / "managed"),
        artifact_token_key="w" * 32,
    )
    repository = InMemoryRepository()
    await repository.open()
    app = create_app(settings=settings, repository=repository)
    service = app.state.service
    store = service.artifact_store
    assert isinstance(store, LocalManagedArtifactStore)
    uri, payload, calls = await _interrupt(service, store, monkeypatch)
    before = await _snapshot(service, uri)
    principal = Principal(
        subject="alice", grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),)
    )
    if surface.startswith("http"):
        params = {"org_id": "acme", "project_id": "research"}
        if surface.endswith("explicit"):
            params["dry_run"] = "true"
        response = await asyncio.to_thread(
            TestClient(app).post,
            "/api/v1/artifacts/collect",
            params=params,
            headers={
                "X-Eyvan-User": "alice",
                "X-Eyvan-Roles": "admin",
                "X-Eyvan-Org": "acme",
                "X-Eyvan-Project": "research",
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
    else:
        kwargs = {"dry_run": True} if surface.endswith("explicit") else {}
        preview = await service.collect_artifacts_once(
            org_id="acme", project_id="research", principal=principal, **kwargs
        )
        result = preview.model_dump(mode="json")
    assert result["dry_run"] is True
    assert store.exists(uri, org_id="acme", project_id="research"), "a report-only pass removed local bytes"
    assert store.get(uri, org_id="acme", project_id="research") == payload
    assert await _snapshot(service, uri) == before, "a report-only pass changed the claim or audit history"
    assert len(calls) == 1, "a report-only pass performed physical deletion I/O"
    assert result["collected_uris"] == [uri]
    assert result["reclaimed_bytes"] == len(payload)

    if recovery == "collection":
        resumed = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)
        assert resumed.collected_uris == (uri,)
    else:
        resumed = await service.reconcile_artifact_lifecycle_once(
            org_id="acme", project_id="research", recovery_grace_s=0
        )
        assert resumed["deleted"] == 1
    assert len(calls) == 2
    assert calls[0] == calls[1] == before[0].object_version
    assert not store.exists(uri, org_id="acme", project_id="research")
    after, events = await _snapshot(service, uri)
    assert after.state is ManagedArtifactState.DELETED
    assert after.charged_bytes == 0
    assert len(events) == len(before[1]) + 1
    assert events[-1].action == "artifact.collected"


@pytest.mark.asyncio
async def test_protected_scan_prefix_does_not_starve_collectable_tail(tmp_path):
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "starvation")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    moment = utcnow()
    old = moment - timedelta(days=30)
    payload = b"writer-087 unprotected tail"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    protected = []
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
        for index in range(10_000):
            record = ManagedArtifactRecord(
                uri=callable_uri(hashlib.sha256(f"held-{index}".encode()).hexdigest()),
                kind="callable",
                org_id="acme",
                project_id="research",
                state=ManagedArtifactState.ACTIVE,
                size_bytes=32,
                sha256=hashlib.sha256(f"held-{index}".encode()).hexdigest(),
                object_version=f"held:{index}",
                created_at=old,
                activated_at=old,
                updated_at=old,
                quarantine_reason=None,
                hold=ArtifactHold(reason="indefinitely protected", held_by="alice", created_at=old),
            )
            protected.append(record)
            await tx.put_managed_artifact(record)
    results = []
    for _ in range(3):
        results.append(
            await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False, now=moment)
        )
    async with repository.transaction() as tx:
        for before in protected:
            assert await tx.get_managed_artifact("acme", "research", before.uri) == before
        after = await tx.get_managed_artifact("acme", "research", uri)
    assert after is not None
    assert after.state is ManagedArtifactState.DELETED, (
        "bounded passes never reached collectable tail: "
        f"{[(result.held, result.capped, result.collected_uris) for result in results]}"
    )
    assert not store.exists(uri, org_id="acme", project_id="research")
