# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed spec-level workspaces retain the same durable claims as legacy ones."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pytest

from eyvan.artifacts.managed import LocalManagedArtifactStore, workspace_uri
from eyvan.records import (
    CallableExecution,
    CommandExecution,
    ManagedArtifactState,
    Principal,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    WorkspaceRef,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import ConflictError

_PAYLOAD = b"owned workspace source bundle"
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()
_URI = workspace_uri(_DIGEST)
_PRINCIPAL = Principal(
    subject="alice",
    grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
)


def _spec(kind: str) -> WorkloadDefinition:
    workspace = WorkspaceRef(uri=_URI, sha256=_DIGEST)
    if kind == "command":
        execution = CommandExecution(command="python train.py")
    else:
        execution = CallableExecution(
            artifact_uri="gs://owned-test/callable.pkl",
            sha256="a" * 64,
            python="3.11",
            workspace_uri=_URI if kind == "legacy-callable" else None,
            workspace_sha256=_DIGEST if kind == "legacy-callable" else None,
        )
    return WorkloadDefinition(
        name="workspace-owner",
        execution=execution,
        workspace=None if kind == "legacy-callable" else workspace,
        max_retries=1,
    )


async def _service(tmp_path: Path) -> tuple[ControlPlaneService, LocalManagedArtifactStore]:
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    service = ControlPlaneService(
        repository,
        artifact_store=store,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
    )
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=_URI, payload=_PAYLOAD)
    return service, store


async def _references(service: ControlPlaneService, workload_id: str, *, active_only: bool = True):
    async with service.repository.transaction() as tx:
        return await tx.list_artifact_references(
            org_id="acme", project_id="research", workload_id=workload_id, active_only=active_only
        )


async def _collect(service: ControlPlaneService):
    return await service.collect_artifacts_once(
        org_id="acme",
        project_id="research",
        grace_period_s=0,
        now=utcnow() + timedelta(hours=1),
        dry_run=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("command", "spec-callable", "legacy-callable"))
async def test_pending_and_retried_workspace_is_kept_until_terminal_release(tmp_path: Path, kind: str) -> None:
    service, store = await _service(tmp_path)
    workload, replayed = await service.submit_workload(
        _PRINCIPAL, org_id="acme", project_id="research", spec=_spec(kind), idempotency_key="submit"
    )
    assert not replayed
    assert workload.state is WorkloadState.PENDING

    collection = await _collect(service)
    assert store.exists(_URI, org_id="acme", project_id="research"), collection
    assert collection.collected_uris == ()
    references = await _references(service, workload.id)
    assert [reference.uri for reference in references] == [_URI]
    reference_id = references[0].id

    cancelled, _ = await service.cancel_workload(_PRINCIPAL, workload.id, idempotency_key="cancel-1")
    assert cancelled.state is WorkloadState.CANCELLED
    assert await _references(service, workload.id) == []
    history = await _references(service, workload.id, active_only=False)
    assert len(history) == 1 and history[0].released_at is not None

    retried, _ = await service.retry_workload(_PRINCIPAL, workload.id, idempotency_key="retry")
    assert retried.generation == workload.generation + 1
    assert retried.state is WorkloadState.PENDING
    references = await _references(service, workload.id)
    assert len(references) == 1 and references[0].id == reference_id
    assert references[0].released_at is None
    collection = await _collect(service)
    assert collection.collected_uris == ()
    assert store.get(_URI, org_id="acme", project_id="research") == _PAYLOAD

    await service.cancel_workload(_PRINCIPAL, workload.id, idempotency_key="cancel-2")
    collection = await _collect(service)
    assert collection.collected_uris == (_URI,)
    assert not store.exists(_URI, org_id="acme", project_id="research")
    before = await service.get_workload(_PRINCIPAL, workload.id)
    # A new submission cannot claim bytes that the collector already deleted.
    with pytest.raises(ConflictError, match="deleted"):
        await service.submit_workload(
            _PRINCIPAL, org_id="acme", project_id="research", spec=_spec(kind), idempotency_key="submit-after-delete"
        )
    assert await service.get_workload(_PRINCIPAL, workload.id) == before
    assert len(await service.list_workloads(_PRINCIPAL, org_id="acme", project_id="research")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("command", "spec-callable"))
@pytest.mark.parametrize("state", (ManagedArtifactState.DELETING, ManagedArtifactState.QUARANTINED))
async def test_nonactive_spec_workspace_refuses_admission_atomically(tmp_path: Path, kind: str, state) -> None:
    service, store = await _service(tmp_path)
    async with service.repository.transaction() as tx:
        artifact = await tx.get_managed_artifact("acme", "research", _URI)
        assert artifact is not None
        await tx.put_managed_artifact(artifact.model_copy(update={"state": state}))

    with pytest.raises(ConflictError, match=state.value):
        await service.submit_workload(
            _PRINCIPAL, org_id="acme", project_id="research", spec=_spec(kind), idempotency_key="submit"
        )
    assert await service.list_workloads(_PRINCIPAL, org_id="acme", project_id="research") == []
    async with service.repository.transaction() as tx:
        assert await tx.list_artifact_references(org_id="acme", project_id="research") == []
    assert store.get(_URI, org_id="acme", project_id="research") == _PAYLOAD
