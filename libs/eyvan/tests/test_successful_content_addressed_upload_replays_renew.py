# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Successful content-addressed upload replays must renew submission grace."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from datetime import timedelta

import pytest

from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri, workspace_uri
from eyvan.records import (
    CallableExecution,
    ManagedArtifactState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.control_plane import DEFAULT_ARTIFACT_GRACE_PERIOD_S
from eyvan.server.service.errors import ConflictError


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["callable", "workspace"])
async def test_identical_old_upload_replay_protects_the_upload_to_submission_interval(tmp_path, kind: str) -> None:
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    service = ControlPlaneService(
        repository,
        artifact_store=store,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
    )
    payload = b"replayed-content-addressed-payload"
    digest = hashlib.sha256(payload).hexdigest()
    uri = callable_uri(digest) if kind == "callable" else workspace_uri(digest)
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    if kind == "workspace":
        await service.upload_managed_artifact(
            org_id="acme", project_id="research", uri=callable_uri(digest), payload=payload
        )
    aged_at = utcnow() - timedelta(seconds=DEFAULT_ARTIFACT_GRACE_PERIOD_S + 3600)
    async with repository.transaction() as tx:
        original = await tx.get_managed_artifact("acme", "research", uri)
        assert original is not None
        original = original.model_copy(
            update={
                "created_at": aged_at,
                "activated_at": aged_at,
                "updated_at": aged_at,
                "upload_started_at": aged_at,
            }
        )
        await tx.put_managed_artifact(original)

    replay = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    assert replay.state is ManagedArtifactState.ACTIVE
    assert replay.created_at == original.created_at
    assert replay.activated_at == original.activated_at
    assert replay.record_generation == original.record_generation
    assert replay.object_version == original.object_version
    assert replay.sha256 == original.sha256
    assert replay.size_bytes == original.size_bytes

    collection = await service.collect_artifacts_once(org_id="acme", project_id="research", dry_run=False)
    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )
    spec = WorkloadDefinition(
        name="uses-replayed-artifact",
        execution=CallableExecution(
            kind="callable",
            artifact_uri=callable_uri(digest),
            sha256=digest,
            workspace_uri=uri if kind == "workspace" else None,
            workspace_sha256=digest if kind == "workspace" else None,
            python="3.11",
        ),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1)),
    )
    workload, replayed = await service.submit_workload(
        principal,
        org_id="acme",
        project_id="research",
        spec=spec,
        idempotency_key=f"replay-grace-{kind}",
    )
    assert not replayed
    assert workload.spec.execution.artifact_uri == spec.execution.artifact_uri
    assert workload.spec.execution.workspace_uri == spec.execution.workspace_uri
    assert collection.collected_uris == ()
    assert collection.within_grace == (2 if kind == "workspace" else 1)
    assert (
        store.get_versioned(uri, org_id="acme", project_id="research", object_version=replay.object_version).payload
        == payload
    )
    async with repository.transaction() as tx:
        references = await tx.list_artifact_references(org_id="acme", project_id="research", uri=uri, active_only=True)
        assert [reference.workload_id for reference in references] == [workload.id]


async def test_replay_refuses_when_collector_wins_after_physical_verification(tmp_path, monkeypatch) -> None:
    repository = InMemoryRepository()
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    service = ControlPlaneService(repository, artifact_store=store)
    payload = b"replay-delete-race"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    original = await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    entered = threading.Event()
    release = threading.Event()
    get = store.get_versioned

    def delayed(*args, **kwargs):
        observed = get(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return observed

    monkeypatch.setattr(store, "get_versioned", delayed)
    replay = asyncio.create_task(
        service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        collection = await service.collect_artifacts_once(
            org_id="acme", project_id="research", grace_period_s=0, dry_run=False
        )
        assert collection.collected_uris == (uri,)
    finally:
        release.set()
    with pytest.raises(ConflictError, match="lifecycle changed"):
        await replay
    async with repository.transaction() as tx:
        tombstone = await tx.get_managed_artifact("acme", "research", uri)
    assert tombstone.state is ManagedArtifactState.DELETED
    assert tombstone.upload_started_at == original.upload_started_at
    assert not store.exists(uri, org_id="acme", project_id="research")
