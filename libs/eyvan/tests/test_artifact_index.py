# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The managed-artifact index: byte accounting, quotas, and the reverse index.

The control plane could always serve artifacts and never knew how many existed
or what they cost. Three properties are asserted here, and each one is what a
future collector would have to trust:

**Bytes are recorded at store time**, because they cannot be recovered later
without reading every object back.

**A content-addressed URI is charged once per project.** The physical store is
tenant-scoped, so identical digests in two projects are independent bytes and
ledger rows; re-uploading within one project still replaces its one charge.

**"Unreferenced" is computable.** A reference is created with the workload that
names an artifact and released when that workload can no longer run -- and the
released row is *kept*, because the history of who referenced what is what makes
a later deletion auditable.

Nothing in this module or the code it covers deletes an object. That is
deliberate: the index has to exist and be trusted before a collector may be
written against it.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from eyvan.artifacts.managed import callable_uri, result_uri, workspace_uri
from eyvan.config import EyvanSettings
from eyvan.records import (
    ArtifactReferenceRecord,
    CallableExecution,
    CommandExecution,
    ManagedArtifactRecord,
    ManagedArtifactState,
    Principal,
    QuotaRecord,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    artifact_reference_id,
    utcnow,
)
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import InputError, NotFoundError

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _admin() -> Principal:
    return Principal(
        subject="root",
        grants=(RoleGrant(role=Role.FLEET_ADMIN), RoleGrant(role=Role.ORG_ADMIN, org_id="acme")),
    )


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(
        repository=repository,
        callable_trust=CallableProjectTrustPolicy.from_projects(("acme/research",)),
    )


async def _activate_indexed(service: ControlPlaneService, uri: str) -> None:
    async with service.repository.transaction() as tx:
        record = await tx.get_managed_artifact("acme", "research", uri)
        assert record is not None
        now = utcnow()
        active = ManagedArtifactRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "state": ManagedArtifactState.ACTIVE,
                "sha256": record.sha256 or hashlib.sha256(uri.encode()).hexdigest(),
                "object_version": f"test:{uri}",
                "activated_at": now,
                "updated_at": now,
                "quarantine_reason": None,
            }
        )
        await tx.put_managed_artifact(active)


#: Clients ask for a managed result by naming any eyvan://results URI; the server
#: replaces it with one it owns, per generation.
_REQUESTED_RESULT = result_uri("requested")


def _callable_spec(*, name: str = "train", result: str | None = None, max_retries: int = 0) -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        max_retries=max_retries,
        execution=CallableExecution(
            kind="callable",
            artifact_uri=callable_uri(_DIGEST),
            sha256=_DIGEST,
            python="3.11",
            workspace_uri=workspace_uri(_OTHER_DIGEST),
            workspace_sha256=_OTHER_DIGEST,
            result_uri=result,
        ),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
    )


@pytest.mark.asyncio
async def test_indexing_records_the_bytes_a_stored_object_occupies() -> None:
    service = await _service()

    record = await service.index_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=callable_uri(_DIGEST),
        size_bytes=4096,
        sha256=_DIGEST,
    )

    assert record.size_bytes == 4096
    assert record.kind == "callables"
    assert record.sha256 == _DIGEST
    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == 4096
    assert usage.artifact_count == 1


@pytest.mark.asyncio
async def test_reindexing_one_uri_replaces_its_bytes_rather_than_adding_them() -> None:
    """The store holds one object per URI, so the ledger must hold one charge."""

    service = await _service()
    uri = callable_uri(_DIGEST)

    await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=4096)
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=4096)

    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == 4096, "a re-upload of the same digest is not new bytes"
    assert usage.artifact_count == 1


@pytest.mark.asyncio
async def test_reindexing_preserves_the_original_creation_time() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)

    first = await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=10)
    second = await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=20)

    assert second.created_at == first.created_at, "the object has existed since the first write"
    assert second.updated_at >= first.updated_at


@pytest.mark.asyncio
async def test_a_project_artifact_quota_refuses_a_write_that_would_exceed_it() -> None:
    service = await _service()
    await service.put_quota(
        _admin(),
        quota=QuotaRecord(
            org_id="acme",
            project_id="research",
            limits=ResourceVector(cpu=100.0),
            artifact_bytes_limit=10_000,
            updated_by="root",
        ),
        idempotency_key="quota-1",
    )
    await service.index_managed_artifact(
        org_id="acme", project_id="research", uri=callable_uri(_DIGEST), size_bytes=8_000
    )

    with pytest.raises(InputError, match="artifact quota"):
        await service.index_managed_artifact(
            org_id="acme", project_id="research", uri=workspace_uri(_OTHER_DIGEST), size_bytes=4_000
        )

    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.stored_bytes == 8_000, "a refused write may not be charged"
    assert usage.quota_bytes == 10_000


@pytest.mark.asyncio
async def test_overwriting_an_indexed_uri_is_measured_against_its_own_current_size() -> None:
    """Otherwise a re-upload double-counts and fails at half the real limit."""

    service = await _service()
    await service.put_quota(
        _admin(),
        quota=QuotaRecord(
            org_id="acme",
            project_id="research",
            limits=ResourceVector(cpu=100.0),
            artifact_bytes_limit=10_000,
            updated_by="root",
        ),
        idempotency_key="quota-1",
    )
    uri = callable_uri(_DIGEST)
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=9_000)

    record = await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=9_500)

    assert record.size_bytes == 9_000
    assert record.state is ManagedArtifactState.QUARANTINED


@pytest.mark.asyncio
async def test_an_unquotaed_project_is_unlimited_not_zero() -> None:
    service = await _service()

    await service.index_managed_artifact(
        org_id="acme", project_id="research", uri=callable_uri(_DIGEST), size_bytes=10**9
    )

    usage = await service.artifact_usage(_principal(), org_id="acme", project_id="research")
    assert usage.quota_bytes is None
    assert usage.stored_bytes == 10**9


@pytest.mark.asyncio
async def test_an_org_quota_applies_to_a_project_that_has_none_of_its_own() -> None:
    service = await _service()
    await service.put_quota(
        _admin(),
        quota=QuotaRecord(
            org_id="acme",
            limits=ResourceVector(cpu=100.0),
            artifact_bytes_limit=1_000,
            updated_by="root",
        ),
        idempotency_key="quota-org",
    )

    with pytest.raises(InputError, match="artifact quota"):
        await service.index_managed_artifact(
            org_id="acme", project_id="research", uri=callable_uri(_DIGEST), size_bytes=2_000
        )


@pytest.mark.asyncio
async def test_a_project_quota_overrides_the_org_quota_for_that_project() -> None:
    """The narrower grant is the one an operator wrote about this project."""

    service = await _service()
    for index, quota in enumerate(
        (
            QuotaRecord(org_id="acme", limits=ResourceVector(cpu=100.0), artifact_bytes_limit=1_000, updated_by="root"),
            QuotaRecord(
                org_id="acme",
                project_id="research",
                limits=ResourceVector(cpu=100.0),
                artifact_bytes_limit=50_000,
                updated_by="root",
            ),
        )
    ):
        await service.put_quota(_admin(), quota=quota, idempotency_key=f"quota-{index}")

    record = await service.index_managed_artifact(
        org_id="acme", project_id="research", uri=callable_uri(_DIGEST), size_bytes=20_000
    )

    assert record.size_bytes == 20_000


@pytest.mark.asyncio
async def test_submitting_a_callable_workload_claims_every_artifact_it_names() -> None:
    """The reverse index nothing could previously answer: who uses this callable?"""

    service = await _service()
    for uri in (callable_uri(_DIGEST), workspace_uri(_OTHER_DIGEST)):
        await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=1)
        await _activate_indexed(service, uri)

    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(),
        idempotency_key="submit-1",
    )

    references = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST)
    )
    assert [reference.workload_id for reference in references] == [workload.id]
    workspace_references = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=workspace_uri(_OTHER_DIGEST)
    )
    assert [reference.workload_id for reference in workspace_references] == [workload.id]


@pytest.mark.asyncio
async def test_the_allocated_result_uri_is_claimed_before_anything_writes_it() -> None:
    """An allocated-but-unwritten result is owned, not free to reuse."""

    service = await _service()
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(result=_REQUESTED_RESULT),
        idempotency_key="submit-1",
    )
    execution = workload.spec.execution
    assert isinstance(execution, CallableExecution)
    assert execution.result_uri is not None
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=execution.result_uri, size_bytes=1)

    references = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=execution.result_uri
    )

    assert [reference.workload_id for reference in references] == [workload.id]


@pytest.mark.asyncio
async def test_a_command_workload_claims_nothing() -> None:
    service = await _service()
    spec = WorkloadDefinition(
        name="train",
        execution=CommandExecution(kind="command", command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
    )

    assert ControlPlaneService.referenced_artifact_uris(spec) == ()

    workload, _ = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=spec, idempotency_key="submit-1"
    )
    async with service.repository.transaction() as tx:
        assert await tx.list_artifact_references(org_id="acme", project_id="research", workload_id=workload.id) == []


@pytest.mark.asyncio
async def test_an_unmanaged_result_uri_is_not_claimed() -> None:
    """A gs:// result lives in the operator's bucket; Eyvan does not index it."""

    spec = _callable_spec(result="gs://customer-bucket/results/run-1.pkl")

    uris = ControlPlaneService.referenced_artifact_uris(spec)

    assert "gs://customer-bucket/results/run-1.pkl" not in uris
    assert callable_uri(_DIGEST) in uris


@pytest.mark.asyncio
async def test_a_replayed_submission_does_not_inflate_the_reference_set() -> None:
    """References count workloads, not submissions, so the id is derived."""

    service = await _service()
    for _ in range(3):
        await service.submit_workload(
            _principal(),
            org_id="acme",
            project_id="research",
            spec=_callable_spec(),
            idempotency_key="submit-1",
        )
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=callable_uri(_DIGEST), size_bytes=1)

    references = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST)
    )

    assert len(references) == 1


@pytest.mark.asyncio
async def test_two_workloads_sharing_one_callable_are_both_recorded() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=1)
    await _activate_indexed(service, uri)
    submitted = [
        (
            await service.submit_workload(
                _principal(),
                org_id="acme",
                project_id="research",
                spec=_callable_spec(name=f"train-{index}"),
                idempotency_key=f"submit-{index}",
            )
        )[0]
        for index in range(2)
    ]

    references = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST)
    )

    assert sorted(reference.workload_id for reference in references) == sorted(item.id for item in submitted)


@pytest.mark.asyncio
async def test_a_terminal_workload_releases_its_claims_without_erasing_them() -> None:
    """The released row is what proves the artifact was once in use."""

    service = await _service()
    uri = callable_uri(_DIGEST)
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=1)
    await _activate_indexed(service, uri)
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(),
        idempotency_key="submit-1",
    )
    await service.cancel_workload(_principal(), workload.id, idempotency_key="cancel-1")
    await service.settle_terminal_workload(_admin(), workload_id=workload.id, target=WorkloadState.CANCELLED)

    live = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST), active_only=True
    )
    all_references = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST)
    )

    assert live == [], "a terminal workload no longer holds its artifacts"
    assert len(all_references) == 1, "the history of who referenced what is kept"
    assert all_references[0].released_at is not None


@pytest.mark.asyncio
async def test_one_workload_terminating_does_not_release_a_siblings_claim() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=1)
    await _activate_indexed(service, uri)
    first, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(name="a"),
        idempotency_key="submit-a",
    )
    second, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(name="b"),
        idempotency_key="submit-b",
    )
    await service.cancel_workload(_principal(), first.id, idempotency_key="cancel-1")
    await service.settle_terminal_workload(_admin(), workload_id=first.id, target=WorkloadState.CANCELLED)

    live = await service.artifact_references(
        _principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST), active_only=True
    )

    assert [reference.workload_id for reference in live] == [second.id]


@pytest.mark.asyncio
async def test_references_for_an_unindexed_artifact_are_a_not_found_not_an_empty_list() -> None:
    """Empty and unknown are different answers, and a collector must not confuse them."""

    service = await _service()

    with pytest.raises(NotFoundError):
        await service.artifact_references(_principal(), org_id="acme", project_id="research", uri=callable_uri(_DIGEST))


@pytest.mark.asyncio
async def test_reading_an_artifact_stamps_its_last_access() -> None:
    service = await _service()
    uri = callable_uri(_DIGEST)
    record = await service.index_managed_artifact(org_id="acme", project_id="research", uri=uri, size_bytes=1)
    assert record.last_accessed_at is None
    await _activate_indexed(service, uri)

    await service.record_artifact_access(
        org_id="acme",
        project_id="research",
        workload_id="wl-1",
        kind="callables",
        object_id=_DIGEST,
        action="artifact.read",
    )

    artifacts = await service.list_managed_artifacts(_principal(), org_id="acme", project_id="research")
    assert artifacts[0].last_accessed_at is not None


@pytest.mark.asyncio
async def test_reading_an_unindexed_artifact_is_not_an_error() -> None:
    """An artifact stored before this index existed has no row; a read must still work."""

    service = await _service()

    await service.record_artifact_access(
        org_id="acme",
        project_id="research",
        workload_id="wl-1",
        kind="results",
        object_id="run-1",
        action="artifact.read",
    )

    assert await service.list_managed_artifacts(_principal(), org_id="acme", project_id="research") == []


def test_a_reference_id_is_derived_from_the_complete_scope_it_records() -> None:
    first = artifact_reference_id("acme", "research", callable_uri(_DIGEST), "wl-1")

    assert first == artifact_reference_id("acme", "research", callable_uri(_DIGEST), "wl-1")
    assert first != artifact_reference_id("acme", "other", callable_uri(_DIGEST), "wl-1")
    assert first != artifact_reference_id("other-org", "research", callable_uri(_DIGEST), "wl-1")
    assert first != artifact_reference_id("acme", "research", callable_uri(_DIGEST), "wl-2")
    assert first != artifact_reference_id("acme", "research", result_uri("run-1"), "wl-1")


def test_the_records_reject_a_negative_size_and_a_blank_scope() -> None:
    with pytest.raises(ValueError):
        ManagedArtifactRecord(
            uri=callable_uri(_DIGEST), kind="callables", org_id="acme", project_id="research", size_bytes=-1
        )
    with pytest.raises(ValueError):
        ArtifactReferenceRecord(
            id="ref-1", uri=callable_uri(_DIGEST), workload_id="wl-1", org_id=" ", project_id="research"
        )


def test_the_upload_route_indexes_the_bytes_it_stores(tmp_path) -> None:
    client = TestClient(
        create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_store_uri=str(tmp_path / "artifacts"),
                trusted_callable_projects=("acme/research",),
            )
        )
    )
    headers = {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": "operator",
    }
    payload = b"x" * 512
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()

    uploaded = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers=headers,
        content=payload,
    )
    assert uploaded.status_code == 200, uploaded.text

    usage = client.get("/api/v1/artifacts/usage", params={"org_id": "acme", "project_id": "research"}, headers=headers)
    assert usage.status_code == 200, usage.text
    assert usage.json()["stored_bytes"] == 512
    listed = client.get("/api/v1/artifacts", params={"org_id": "acme", "project_id": "research"}, headers=headers)
    assert [item["sha256"] for item in listed.json()] == [digest]


@pytest.mark.asyncio
async def test_a_retry_claims_the_new_result_uri_its_generation_mints() -> None:
    """Each attempt owns a different result address, so each must be claimed.

    ``result_spec_for_generation`` rewrites ``result_uri`` per generation. Recording
    references only at submit would leave every attempt after the first owning an
    artifact the index believes nobody references.
    """

    service = await _service()
    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_callable_spec(result=_REQUESTED_RESULT, max_retries=2),
        idempotency_key="submit-1",
    )
    first_execution = workload.spec.execution
    assert isinstance(first_execution, CallableExecution)
    await service.transition_workload(_admin(), workload_id=workload.id, target=WorkloadState.FAILED)

    retried, _ = await service.retry_workload(_principal(), workload.id, idempotency_key="retry-1")

    second_execution = retried.spec.execution
    assert isinstance(second_execution, CallableExecution)
    assert second_execution.result_uri != first_execution.result_uri
    async with service.repository.transaction() as tx:
        claimed = {
            reference.uri
            for reference in await tx.list_artifact_references(
                org_id="acme", project_id="research", workload_id=workload.id
            )
        }
    assert first_execution.result_uri in claimed
    assert second_execution.result_uri in claimed, "the retry's own result address is unindexed"
