# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Restart and ownership coverage for Eyvan's durable local control plane."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import eyvan.server.persistence.local as local_persistence
from eyvan._internal.json_types import JsonObject, model_json
from eyvan.artifacts.managed import ArtifactAction, result_uri
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import (
    ArtifactReferenceRecord,
    AuditEvent,
    BudgetRecord,
    CapacityPoolRecord,
    CellRecord,
    CommandExecution,
    IdempotencyRecord,
    LeaseRecord,
    ManagedArtifactRecord,
    ManagedArtifactState,
    OrganizationRecord,
    OutboxRecord,
    OutboxState,
    PolicyRecord,
    Principal,
    ProjectRecord,
    QuotaRecord,
    ResourceVector,
    Role,
    RoleGrant,
    RoleGrantRecord,
    UserProgressReport,
    WebhookRecord,
    WorkloadDefinition,
    WorkloadRecord,
)
from eyvan.server.api.app import create_app
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.persistence.local import (
    LOCAL_CONTROL_PLANE_MAX_BYTES,
    LOCAL_CONTROL_PLANE_SCHEMA_VERSION,
    LOCAL_CONTROL_PLANE_WEBHOOK_SCHEMA_VERSION,
    LocalControlPlaneAmbiguousCommitError,
    LocalControlPlaneDocument,
    LocalControlPlaneError,
    LocalControlPlaneInUseError,
    LocalControlPlaneSchemaError,
    LocalControlPlaneStore,
    LocalControlPlaneV6Document,
    LocalJsonRepository,
)
from eyvan.server.persistence.state import ControlPlaneState
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.federation import BudgetGrantRecord, GrantState, PlanePeerRecord, QuotaGrantRecord


def _principal() -> Principal:
    return Principal(subject="alice", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _workload_spec() -> WorkloadDefinition:
    return WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py"))


def _workload_request() -> JsonObject:
    return {
        "org_id": "acme",
        "project_id": "research",
        "spec": model_json(_workload_spec()),
    }


def _headers() -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": "operator",
    }


_FIXED_TIME = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
_ARTIFACT_PAYLOAD = b"artifact-payload-that-must-remain-outside-control-plane-json"
_ARTIFACT_DIGEST = hashlib.sha256(_ARTIFACT_PAYLOAD).hexdigest()
_V1_FIXTURE = Path(__file__).parent / "fixtures" / "local_control_plane_v1.json"


def _audit_event(event_id: str, action: str = "test.event") -> AuditEvent:
    return AuditEvent(
        id=event_id,
        timestamp=_FIXED_TIME,
        actor="alice",
        org_id="acme",
        project_id="research",
        entity_type="workload",
        entity_id="wl-1",
        action=action,
    )


def _managed_artifact(uri: str, state: ManagedArtifactState, *, project_id: str = "research") -> ManagedArtifactRecord:
    common = {
        "uri": uri,
        "kind": "callable",
        "org_id": "acme",
        "project_id": project_id,
        "state": state,
        "record_generation": 3,
        "sha256": _ARTIFACT_DIGEST,
        "created_at": _FIXED_TIME,
        "updated_at": _FIXED_TIME,
        "quarantine_reason": None,
    }
    if state is ManagedArtifactState.PENDING_UPLOAD:
        return ManagedArtifactRecord(size_bytes=len(_ARTIFACT_PAYLOAD), upload_started_at=_FIXED_TIME, **common)
    if state is ManagedArtifactState.ACTIVE:
        return ManagedArtifactRecord(
            size_bytes=len(_ARTIFACT_PAYLOAD),
            object_version=f"version:{project_id}",
            activated_at=_FIXED_TIME,
            **common,
        )
    if state is ManagedArtifactState.DELETING:
        return ManagedArtifactRecord(
            size_bytes=len(_ARTIFACT_PAYLOAD),
            object_version="version:deleting",
            deletion_claim_id="claim-1",
            deletion_started_at=_FIXED_TIME,
            **common,
        )
    if state is ManagedArtifactState.DELETED:
        return ManagedArtifactRecord(
            size_bytes=0,
            deleted_at=_FIXED_TIME,
            superseded_at=_FIXED_TIME,
            superseded_by_uri="managed://callable/replacement",
            **common,
        )
    return ManagedArtifactRecord(
        size_bytes=len(_ARTIFACT_PAYLOAD),
        quarantine_reason="physical generation needs operator repair",
        **{key: value for key, value in common.items() if key != "quarantine_reason"},
    )


def _complete_control_plane_state() -> ControlPlaneState:
    workload = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=_workload_spec(),
        created_by="alice",
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    cell = CellRecord(
        id="cell-1",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=8, memory_gib=32),
        allocatable=ResourceVector(cpu=7, memory_gib=30),
    )
    pool = CapacityPoolRecord(
        id="pool-1",
        fleet_name="local",
        node_type="cpu-8",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=8, memory_gib=32),
        allocatable=ResourceVector(cpu=8, memory_gib=32),
        agent_endpoint="http://127.0.0.1:8090",
    )
    lease = LeaseRecord(
        id="lease-1",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        resources=workload.spec.gang,
        fencing_token="fence-1",
        expires_at=_FIXED_TIME + timedelta(hours=1),
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    policy = PolicyRecord(org_id="acme", project_id="research", updated_by="alice", updated_at=_FIXED_TIME)
    quota = QuotaRecord(
        org_id="acme",
        project_id="research",
        principal_id="alice",
        limits=ResourceVector(cpu=16, memory_gib=64),
        updated_by="alice",
        updated_at=_FIXED_TIME,
    )
    budget = BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=100.0,
        spent=12.5,
        updated_by="alice",
        updated_at=_FIXED_TIME,
    )
    role_grant = RoleGrantRecord(
        subject="alice",
        grant=RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),
        granted_by="root",
        granted_at=_FIXED_TIME,
    )
    organization = OrganizationRecord(
        id="acme",
        owner="alice",
        created_by="root",
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    project = ProjectRecord(
        org_id="acme",
        id="research",
        owner="alice",
        created_by="root",
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    progress = UserProgressReport(rank=7, attempt=2, step=12, total=20, reported_at=_FIXED_TIME)
    peer = PlanePeerRecord(
        plane_id="peer-west",
        endpoint="https://peer-west.example.invalid",
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    budget_grant = BudgetGrantRecord(
        id="budget-grant-1",
        org_id="acme",
        project_id="research",
        home_plane_id="home",
        grantee_plane_id="peer-west",
        amount=20.0,
        state=GrantState.REVOKED,
        issued_at=_FIXED_TIME,
        expires_at=_FIXED_TIME + timedelta(hours=1),
    )
    quota_grant = QuotaGrantRecord(
        id="quota-grant-1",
        org_id="acme",
        project_id="research",
        home_plane_id="home",
        grantee_plane_id="peer-west",
        limits=ResourceVector(tpu=8),
        state=GrantState.EXPIRED,
        issued_at=_FIXED_TIME,
        expires_at=_FIXED_TIME + timedelta(hours=1),
    )
    artifacts = (
        _managed_artifact("managed://callable/pending", ManagedArtifactState.PENDING_UPLOAD),
        _managed_artifact("managed://callable/shared", ManagedArtifactState.ACTIVE),
        _managed_artifact("managed://callable/deleting", ManagedArtifactState.DELETING),
        _managed_artifact("managed://callable/deleted", ManagedArtifactState.DELETED),
        _managed_artifact("managed://callable/quarantined", ManagedArtifactState.QUARANTINED),
        _managed_artifact("managed://callable/shared", ManagedArtifactState.ACTIVE, project_id="other"),
    )
    references = (
        ArtifactReferenceRecord(
            id="ref-display-collision",
            uri="managed://callable/shared",
            workload_id="wl-1",
            org_id="acme",
            project_id="research",
            created_at=_FIXED_TIME,
        ),
        ArtifactReferenceRecord(
            id="ref-released",
            uri="managed://callable/deleted",
            workload_id="wl-1",
            org_id="acme",
            project_id="research",
            created_at=_FIXED_TIME,
            released_at=_FIXED_TIME + timedelta(minutes=1),
        ),
        ArtifactReferenceRecord(
            id="ref-display-collision",
            uri="managed://callable/shared",
            workload_id="wl-other",
            org_id="acme",
            project_id="other",
            created_at=_FIXED_TIME,
        ),
    )
    idempotency = IdempotencyRecord(
        scope="acme/research/alice",
        operation="workload.submit",
        key="submit-1",
        fingerprint="fingerprint-1",
        resource_type="workload",
        resource_id=workload.id,
        created_at=_FIXED_TIME,
    )
    outbox = OutboxRecord(
        id="outbox-1",
        topic="capacity.apply",
        aggregate_type="capacity_pool",
        aggregate_id=pool.id,
        idempotency_key="apply-1",
        generation=1,
        state=OutboxState.SUCCEEDED,
        attempts=1,
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    event = _audit_event("event-41").model_copy(update={"sequence": 41})
    return ControlPlaneState(
        workloads={workload.id: workload},
        cells={cell.id: cell},
        capacity_pools={pool.id: pool},
        leases={lease.id: lease},
        policies={(policy.org_id, policy.project_id): policy},
        quotas={(quota.org_id, quota.project_id or "", quota.principal_id or ""): quota},
        budgets={(budget.org_id, budget.project_id): budget},
        role_grants={role_grant.key: role_grant},
        organizations={organization.id: organization},
        projects={project.key: project},
        progress_reports={workload.id: {progress.rank: progress}},
        plane_peers={peer.plane_id: peer},
        budget_grants={budget_grant.id: budget_grant},
        quota_grants={quota_grant.id: quota_grant},
        managed_artifacts={(record.org_id, record.project_id, record.uri): record for record in artifacts},
        artifact_references={
            (record.org_id, record.project_id, record.uri, record.workload_id): record for record in references
        },
        idempotency={(idempotency.scope, idempotency.operation, idempotency.key): idempotency},
        outbox={outbox.id: outbox},
        events=[event],
        next_event_sequence=50,
        revision=17,
    )


def _settings(tmp_path: Path) -> EyvanSettings:
    profile = tmp_path / "profile"
    return EyvanSettings(
        repository_backend=RepositoryBackend.LOCAL,
        local_state_path=profile / "control-plane.json",
        local_artifact_token_key_path=profile / "artifact-token.key",
        artifact_store_uri=str(profile / "managed-artifacts"),
        controller_enabled=False,
        dev_auth_enabled=True,
    )


@pytest.mark.asyncio
async def test_local_repository_read_only_transaction_does_not_create_or_rewrite_state(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    repository = LocalJsonRepository(path)
    try:
        async with repository.transaction() as transaction:
            assert await transaction.list_workloads() == []
        assert not path.exists()

        async with repository.transaction() as transaction:
            await transaction.put_workload(
                WorkloadRecord(
                    id="wl-1",
                    org_id="acme",
                    project_id="research",
                    spec=_workload_spec(),
                    created_by="alice",
                )
            )
        original = path.read_bytes()

        async with repository.transaction() as transaction:
            assert (await transaction.get_workload("wl-1")) is not None

        assert path.read_bytes() == original
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_local_repository_persists_workload_and_idempotency_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    first_repository = LocalJsonRepository(path)
    first_service = ControlPlaneService(first_repository)

    try:
        created, replayed = await first_service.submit_workload(
            _principal(),
            org_id="acme",
            project_id="research",
            spec=_workload_spec(),
            idempotency_key="submit-1",
        )
        assert replayed is False
    finally:
        await first_repository.close()

    second_repository = LocalJsonRepository(path)
    second_service = ControlPlaneService(second_repository)
    try:
        replay, replayed = await second_service.submit_workload(
            _principal(),
            org_id="acme",
            project_id="research",
            spec=_workload_spec(),
            idempotency_key="submit-1",
        )
        restored = await second_service.get_workload(_principal(), created.id)
        events = await second_repository.wait_for_events(after_sequence=0)
    finally:
        await second_repository.close()

    assert replayed is True
    assert replay.id == created.id
    assert restored == created
    assert [(event.sequence, event.action) for event in events] == [(1, "workload.submitted")]


@pytest.mark.asyncio
async def test_local_repository_reclaims_processing_outbox_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    first_repository = LocalJsonRepository(path)
    try:
        await first_repository.open()
        async with first_repository.transaction() as transaction:
            await transaction.put_outbox(
                OutboxRecord(
                    id="command-1",
                    topic="capacity.apply",
                    aggregate_type="capacity_pool",
                    aggregate_id="pool-1",
                    idempotency_key="apply-1",
                    generation=1,
                )
            )
        claimed = await first_repository.claim_outbox(worker_id="first-server", lease_seconds=3600)
        assert [record.id for record in claimed] == ["command-1"]
        assert claimed[0].state is OutboxState.PROCESSING
    finally:
        await first_repository.close()

    second_repository = LocalJsonRepository(path)
    try:
        await second_repository.open()
        reclaimed = await second_repository.claim_outbox(worker_id="second-server", lease_seconds=3600)
    finally:
        await second_repository.close()

    assert len(reclaimed) == 1
    assert reclaimed[0].id == "command-1"
    assert reclaimed[0].attempts == 2
    assert reclaimed[0].claimed_by == "second-server"


@pytest.mark.asyncio
async def test_local_repository_rejects_a_second_process_owner(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    owner = LocalJsonRepository(path)
    contender = LocalJsonRepository(path, lease_timeout_s=0)
    await owner.open()
    try:
        with pytest.raises(LocalControlPlaneInUseError, match=rf"already owned by another server \(pid {os.getpid()}\)"):
            await contender.open()
    finally:
        await contender.close()
        await owner.close()


@pytest.mark.asyncio
async def test_local_repository_waits_out_a_previous_owners_shutdown(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    owner = LocalJsonRepository(path)
    await owner.open()

    async def stop_owner() -> None:
        await asyncio.sleep(0.2)
        await owner.close()

    stopping = asyncio.create_task(stop_owner())
    try:
        replacement = LocalJsonRepository(path, lease_timeout_s=10)
        await replacement.open()
        await replacement.close()
    finally:
        await stopping


@pytest.mark.asyncio
async def test_replace_directory_fsync_failure_publishes_next_cursor_and_poisons_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "control-plane.json"
    repository = LocalJsonRepository(path)
    await repository.open()
    try:

        def fail_directory_fsync() -> None:
            raise OSError("simulated state directory fsync failure")

        with monkeypatch.context() as context:
            context.setattr(repository.store, "_fsync_parent", fail_directory_fsync)
            with pytest.raises(LocalControlPlaneAmbiguousCommitError, match="durability is unknown"):
                async with repository.transaction() as transaction:
                    first_event = await transaction.append_event(_audit_event("ambiguous-event"))

            assert first_event.sequence == 1
            assert [event.sequence for event in repository._state.events] == [1]
            assert repository._state.next_event_sequence == 2
            assert repository.store.commit_status_ambiguous is True

            with pytest.raises(LocalControlPlaneAmbiguousCommitError, match="ambiguous prior commit"):
                async with repository.transaction():
                    pytest.fail("a poisoned repository must refuse transactions before yielding")
            with pytest.raises(LocalControlPlaneAmbiguousCommitError, match="reload"):
                repository.store.save(ControlPlaneState())

        on_disk = LocalControlPlaneStore(path).load()
        assert [(event.sequence, event.id) for event in on_disk.events] == [(1, "ambiguous-event")]
        assert on_disk.next_event_sequence == 2
    finally:
        await repository.close()

    restarted = LocalJsonRepository(path)
    try:
        async with restarted.transaction() as transaction:
            second_event = await transaction.append_event(_audit_event("post-reload-event"))
    finally:
        await restarted.close()

    assert second_event.sequence == 2
    final = LocalControlPlaneStore(path).load()
    assert [(event.sequence, event.id) for event in final.events] == [
        (1, "ambiguous-event"),
        (2, "post-reload-event"),
    ]
    assert final.next_event_sequence == 3


def test_local_app_reuses_state_and_artifact_key_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    bind = ServerBindPolicy.from_host("127.0.0.1")
    first_app = create_app(settings=settings, server_bind=bind)

    with TestClient(first_app) as first_client:
        submitted = first_client.post(
            "/api/v1/workloads",
            headers={**_headers(), "Idempotency-Key": "submit-1"},
            json=_workload_request(),
        )
        assert submitted.status_code == 201
        submitted_workload = WorkloadRecord.model_validate(submitted.json())
        workload_id = submitted_workload.id
        token = first_app.state.artifact_access.signer.issue(
            ArtifactAction.WRITE,
            result_uri("restart-result"),
            org_id="acme",
            project_id="research",
            workload_id=workload_id,
            expires=False,
        )

    second_app = create_app(settings=settings, server_bind=bind)
    with TestClient(second_app) as second_client:
        restored = second_client.get(f"/api/v1/workloads/{workload_id}", headers=_headers())
        restored_workload = WorkloadRecord.model_validate(restored.json())
        claims = second_app.state.artifact_access.signer.verify(
            token,
            ArtifactAction.WRITE,
            result_uri("restart-result"),
            org_id="acme",
            project_id="research",
            workload_id=workload_id,
        )

    assert restored.status_code == 200
    assert restored_workload.id == workload_id
    assert claims.workload_id == workload_id
    assert stat.S_IMODE(settings.local_state_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(settings.local_artifact_token_key_path.stat().st_mode) == 0o600


def test_v2_document_round_trips_every_collection_in_deterministic_identity_order() -> None:
    state = _complete_control_plane_state()

    document = LocalControlPlaneDocument.from_state(state)
    restored = document.to_state()

    assert restored == state
    assert LocalControlPlaneDocument.from_state(restored).model_dump(mode="json", by_alias=True) == document.model_dump(
        mode="json", by_alias=True
    )
    assert document.version == LOCAL_CONTROL_PLANE_SCHEMA_VERSION == 2
    assert [(item.workload_id, item.report.rank) for item in document.progress_reports] == [("wl-1", 7)]
    assert [(item.org_id, item.project_id, item.uri) for item in document.managed_artifacts] == sorted(
        (item.org_id, item.project_id, item.uri) for item in document.managed_artifacts
    )
    assert len([item for item in document.managed_artifacts if item.uri == "managed://callable/shared"]) == 2
    assert len([item for item in document.artifact_references if item.id == "ref-display-collision"]) == 2


@pytest.mark.asyncio
async def test_v2_repository_reopens_every_aggregate_lifecycle_tombstone_and_reference(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    expected = _complete_control_plane_state()
    LocalControlPlaneStore(path).save(expected)

    raw_payload = path.read_bytes()
    raw = json.loads(raw_payload)
    assert raw["version"] == 2
    assert set(
        (
            "progress_reports",
            "plane_peers",
            "budget_grants",
            "quota_grants",
            "managed_artifacts",
            "artifact_references",
        )
    ).issubset(raw)
    assert _ARTIFACT_PAYLOAD not in raw_payload

    first = LocalJsonRepository(path)
    try:
        await first.open()
        assert first.store.load() == expected
        async with first.transaction() as transaction:
            research_artifacts = await transaction.list_managed_artifacts(
                org_id="acme", project_id="research", limit=100
            )
            references = await transaction.list_artifact_references(
                org_id="acme", project_id="research", active_only=False
            )
            event = await transaction.append_event(_audit_event("event-50", "test.after_first_restart"))
        assert {record.state for record in research_artifacts} == set(ManagedArtifactState)
        assert {record.released_at is None for record in references} == {False, True}
        assert event.sequence == 50
    finally:
        await first.close()

    second = LocalJsonRepository(path)
    try:
        await second.open()
        async with second.transaction() as transaction:
            assert (await transaction.get_workload("wl-1")) == expected.workloads["wl-1"]
            assert (await transaction.get_cell("cell-1")) == expected.cells["cell-1"]
            assert (await transaction.get_capacity_pool("pool-1")) == expected.capacity_pools["pool-1"]
            assert (await transaction.get_lease("lease-1")) == expected.leases["lease-1"]
            assert (await transaction.get_policy("acme", "research")) == expected.policies[("acme", "research")]
            assert (await transaction.get_quota("acme", "research", "alice")) == expected.quotas[
                ("acme", "research", "alice")
            ]
            assert (await transaction.get_budget("acme", "research")) == expected.budgets[("acme", "research")]
            assert await transaction.list_role_grants(subject="alice") == list(expected.role_grants.values())
            assert (await transaction.get_organization("acme")) == expected.organizations["acme"]
            assert (await transaction.get_project("acme", "research")) == expected.projects[("acme", "research")]
            assert await transaction.list_progress_reports("wl-1") == list(expected.progress_reports["wl-1"].values())
            assert (await transaction.get_plane_peer("peer-west")) == expected.plane_peers["peer-west"]
            assert (await transaction.get_budget_grant("budget-grant-1")) == expected.budget_grants["budget-grant-1"]
            assert (await transaction.get_quota_grant("quota-grant-1")) == expected.quota_grants["quota-grant-1"]
            assert (await transaction.get_idempotency("acme/research/alice", "workload.submit", "submit-1")) == next(
                iter(expected.idempotency.values())
            )
            assert (await transaction.get_outbox("outbox-1")) == expected.outbox["outbox-1"]
            events = await transaction.list_events(after_sequence=0, limit=100)
            event = await transaction.append_event(_audit_event("event-51", "test.after_second_restart"))
        assert [item.sequence for item in events] == [41, 50]
        assert event.sequence == 51
    finally:
        await second.close()

    final_state = LocalControlPlaneStore(path).load()
    assert [event.sequence for event in final_state.events] == [41, 50, 51]
    assert final_state.next_event_sequence == 52
    assert not LocalControlPlaneStore(path).v1_backup_path.exists()


def _webhook_record() -> WebhookRecord:
    return WebhookRecord(
        id="wh-1",
        org_id="acme",
        project_id="research",
        url="https://hooks.example.com/eyvan",
        secret="s" * 32,
        entity_types=("workload",),
        actions=("workload.settled",),
        delivered_sequence=41,
        consecutive_failures=2,
        disabled_reason="receiver kept returning 503",
        enabled=False,
        created_by="alice",
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )


@pytest.mark.asyncio
async def test_local_repository_persists_webhook_subscriptions_and_cursors_across_restart(tmp_path: Path) -> None:
    """A dropped subscription silently stops every push integration a project
    depends on, and a dropped delivery cursor replays already-delivered audit
    history into it after one restart."""

    path = tmp_path / "control-plane.json"
    first = LocalJsonRepository(path)
    try:
        await first.open()
        async with first.transaction() as transaction:
            await transaction.put_webhook(_webhook_record())
    finally:
        await first.close()

    document, _snapshot = LocalControlPlaneStore(path)._read_document()
    assert isinstance(document, LocalControlPlaneV6Document)
    assert document.version == LOCAL_CONTROL_PLANE_WEBHOOK_SCHEMA_VERSION == 6

    second = LocalJsonRepository(path)
    try:
        await second.open()
        async with second.transaction() as transaction:
            restored = await transaction.get_webhook("wh-1")
    finally:
        await second.close()

    assert restored == _webhook_record()


def test_an_older_document_without_a_webhooks_field_loads_with_an_empty_collection(tmp_path: Path) -> None:
    """Migration is by absence, not refusal: a v5 document simply has no
    subscriptions, and the first committed webhook moves the file to v6."""

    path = tmp_path / "control-plane.json"
    path.write_text(json.dumps({"schema": local_persistence.LOCAL_CONTROL_PLANE_SCHEMA, "version": 5}))
    path.chmod(0o600)

    state = LocalControlPlaneStore(path).load()

    assert state.webhooks == {}


@pytest.mark.asyncio
async def test_actual_v1_fixture_loads_with_empty_v2_defaults_and_upgrades_without_reusing_sequences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control-plane.json"
    original = _V1_FIXTURE.read_bytes()
    shutil.copyfile(_V1_FIXTURE, path)
    path.chmod(0o600)
    store = LocalControlPlaneStore(path)

    legacy_state = store.load()
    assert list(legacy_state.workloads) == ["wl-v1-fixture"]
    assert legacy_state.next_event_sequence == 11
    assert legacy_state.progress_reports == {}
    assert legacy_state.plane_peers == {}
    assert legacy_state.budget_grants == {}
    assert legacy_state.quota_grants == {}
    assert legacy_state.managed_artifacts == {}
    assert legacy_state.artifact_references == {}
    assert not store.v1_backup_path.exists()

    first = LocalJsonRepository(path)
    try:
        async with first.transaction() as transaction:
            first_event = await transaction.append_event(_audit_event("v1-upgrade-event"))
    finally:
        await first.close()

    assert first_event.sequence == 11
    assert json.loads(path.read_text())["version"] == 2
    assert store.v1_backup_path.read_bytes() == original
    assert stat.S_IMODE(store.v1_backup_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    second = LocalJsonRepository(path)
    try:
        async with second.transaction() as transaction:
            second_event = await transaction.append_event(_audit_event("v2-restart-event"))
    finally:
        await second.close()

    assert second_event.sequence == 12
    assert store.v1_backup_path.read_bytes() == original


@pytest.mark.parametrize("backup_origin", ("preexisting", "link-collision"))
def test_exact_v1_backup_file_and_parent_are_resynced_in_order_before_replace_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backup_origin: str,
) -> None:
    path = tmp_path / "control-plane.json"
    original = _V1_FIXTURE.read_bytes()
    shutil.copyfile(_V1_FIXTURE, path)
    path.chmod(0o600)
    store = LocalControlPlaneStore(path)
    state = store.load()
    backup_path = store.v1_backup_path
    if backup_origin == "preexisting":
        backup_path.write_bytes(original)
        backup_path.chmod(0o600)

    real_fsync = os.fsync
    real_link = os.link
    real_replace = os.replace
    fail_first_parent_fsync = True
    injected_collision = False
    events: list[str] = []

    def collide_with_exact_backup(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal injected_collision
        if backup_origin == "link-collision" and Path(destination) == backup_path and not injected_collision:
            injected_collision = True
            backup_path.write_bytes(original)
            backup_path.chmod(0o600)
            raise FileExistsError("simulated exact backup link collision")
        real_link(source, destination, follow_symlinks=follow_symlinks)

    def record_fsync(descriptor: int) -> None:
        nonlocal fail_first_parent_fsync
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            if fail_first_parent_fsync:
                fail_first_parent_fsync = False
                events.append("backup-parent-fsync-failed")
                raise OSError("simulated first backup directory fsync failure")
            events.append("parent-fsync")
        elif backup_path.exists():
            backup_metadata = backup_path.stat()
            if (metadata.st_dev, metadata.st_ino) == (backup_metadata.st_dev, backup_metadata.st_ino):
                events.append("backup-file-fsync")
        real_fsync(descriptor)

    def record_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        if Path(destination) == path:
            events.append("state-replace")
        real_replace(source, destination)

    with monkeypatch.context() as context:
        context.setattr(os, "fsync", record_fsync)
        context.setattr(os, "link", collide_with_exact_backup)
        context.setattr(os, "replace", record_replace)
        with pytest.raises(LocalControlPlaneError, match="directory fsync failure"):
            store.save(state)
        assert path.read_bytes() == original

        store.save(state)

    assert injected_collision is (backup_origin == "link-collision")
    assert events == [
        "backup-file-fsync",
        "backup-parent-fsync-failed",
        "backup-file-fsync",
        "parent-fsync",
        "state-replace",
        "parent-fsync",
    ]
    assert backup_path.read_bytes() == original
    assert json.loads(path.read_text())["version"] == 2


def test_v1_backup_is_fsynced_before_the_v2_replace_and_survives_an_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "control-plane.json"
    original = _V1_FIXTURE.read_bytes()
    shutil.copyfile(_V1_FIXTURE, path)
    path.chmod(0o600)
    store = LocalControlPlaneStore(path)
    state = store.load()
    parent_fsyncs = 0
    real_fsync_parent = store._fsync_parent

    def record_parent_fsync() -> None:
        nonlocal parent_fsyncs
        real_fsync_parent()
        parent_fsyncs += 1

    def interrupt_replace(source: str | bytes | os.PathLike[str] | os.PathLike[bytes], destination: object) -> None:
        assert Path(destination) == path
        assert parent_fsyncs == 1
        assert store.v1_backup_path.read_bytes() == original
        assert stat.S_IMODE(store.v1_backup_path.stat().st_mode) == 0o600
        raise OSError("simulated crash before v2 replace")

    with monkeypatch.context() as context:
        context.setattr(store, "_fsync_parent", record_parent_fsync)
        context.setattr(os, "replace", interrupt_replace)
        with pytest.raises(LocalControlPlaneError, match="simulated crash"):
            store.save(state)

    assert path.read_bytes() == original
    assert store.v1_backup_path.read_bytes() == original
    store.save(state)
    assert json.loads(path.read_text())["version"] == 2
    assert store.v1_backup_path.read_bytes() == original


@pytest.mark.parametrize(
    ("backup_payload", "backup_mode", "error"),
    (
        (_V1_FIXTURE.read_bytes(), 0o644, "group- or world-readable"),
        (b'{"stale": true}\n', 0o600, "does not match"),
    ),
)
def test_v1_upgrade_refuses_unsafe_or_stale_existing_backup(
    tmp_path: Path,
    backup_payload: bytes,
    backup_mode: int,
    error: str,
) -> None:
    path = tmp_path / "control-plane.json"
    original = _V1_FIXTURE.read_bytes()
    shutil.copyfile(_V1_FIXTURE, path)
    path.chmod(0o600)
    store = LocalControlPlaneStore(path)
    store.v1_backup_path.write_bytes(backup_payload)
    store.v1_backup_path.chmod(backup_mode)

    with pytest.raises(LocalControlPlaneError, match=error):
        store.save(store.load())

    assert path.read_bytes() == original
    assert json.loads(path.read_text())["version"] == 1


def test_v1_upgrade_refuses_a_non_regular_backup_path(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    original = _V1_FIXTURE.read_bytes()
    shutil.copyfile(_V1_FIXTURE, path)
    path.chmod(0o600)
    store = LocalControlPlaneStore(path)
    target = tmp_path / "backup-target"
    target.write_bytes(original)
    target.chmod(0o600)
    store.v1_backup_path.symlink_to(target)

    with pytest.raises(LocalControlPlaneError, match="regular file"):
        store.save(store.load())

    assert path.read_bytes() == original


def test_v1_upgrade_accepts_an_exact_owner_only_preexisting_backup_without_overwriting_it(tmp_path: Path) -> None:
    path = tmp_path / "control-plane.json"
    original = _V1_FIXTURE.read_bytes()
    shutil.copyfile(_V1_FIXTURE, path)
    path.chmod(0o600)
    store = LocalControlPlaneStore(path)
    store.v1_backup_path.write_bytes(original)
    store.v1_backup_path.chmod(0o600)
    before = store.v1_backup_path.stat()

    store.save(store.load())

    after = store.v1_backup_path.stat()
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (before.st_dev, before.st_ino, before.st_mtime_ns)
    assert store.v1_backup_path.read_bytes() == original
    assert json.loads(path.read_text())["version"] == 2


def test_v1_upgrade_revalidates_exact_pathname_after_backup_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "control-plane.json"
    original = _V1_FIXTURE.read_bytes()
    shutil.copyfile(_V1_FIXTURE, path)
    path.chmod(0o600)
    store = LocalControlPlaneStore(path)
    state = store.load()
    real_ensure_backup = store._ensure_v1_backup

    def preserve_then_swap_revision(payload: bytes) -> None:
        real_ensure_backup(payload)
        replacement_document = json.loads(original)
        replacement_document["revision"] = 999
        replacement = path.with_name("replacement-v1.json")
        replacement.write_text(json.dumps(replacement_document, indent=2, sort_keys=True) + "\n")
        replacement.chmod(0o600)
        os.replace(replacement, path)

    monkeypatch.setattr(store, "_ensure_v1_backup", preserve_then_swap_revision)

    with pytest.raises(LocalControlPlaneError, match="changed after its backup was secured"):
        store.save(state)

    assert json.loads(path.read_text())["revision"] == 999
    assert json.loads(path.read_text())["version"] == 1
    assert store.v1_backup_path.read_bytes() == original


def test_local_store_rejects_oversize_read_and_bounds_streamed_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized_path = tmp_path / "oversized.json"
    with oversized_path.open("wb") as stream:
        stream.truncate(LOCAL_CONTROL_PLANE_MAX_BYTES + 1)
    oversized_path.chmod(0o600)

    with pytest.raises(LocalControlPlaneSchemaError, match=f"{LOCAL_CONTROL_PLANE_MAX_BYTES}-byte"):
        LocalControlPlaneStore(oversized_path).load()

    output_path = tmp_path / "bounded-save.json"
    with monkeypatch.context() as context:
        context.setattr(local_persistence, "LOCAL_CONTROL_PLANE_MAX_BYTES", 256)
        with pytest.raises(LocalControlPlaneError, match="256-byte local JSON limit"):
            LocalControlPlaneStore(output_path).save(ControlPlaneState())

    assert not output_path.exists()
    assert list(tmp_path.glob(".bounded-save.json.*.tmp")) == []


@pytest.mark.parametrize(
    "payload",
    (
        b'{"schema":"eyvan.control-plane","version":2,"unexpected":' + b"[" * 101 + b"0" + b"]" * 101 + b"}",
        b'{"schema":"eyvan.control-plane","version":2,"revision":' + b"9" * 10_000 + b"}",
        b'{"schema":"eyvan.control-plane","version":2,"revision":NaN}',
    ),
    ids=("pathological-depth", "pathological-integer", "non-finite-constant"),
)
def test_local_store_normalizes_pathological_json_decoder_failures(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "control-plane.json"
    path.write_bytes(payload)
    path.chmod(0o600)

    with pytest.raises(LocalControlPlaneSchemaError):
        LocalControlPlaneStore(path).load()


@pytest.mark.parametrize(
    "payload",
    (
        b"{",
        b"\xff",
        '{"schema":"eyvan.control-plane","version":2,"revision":0}'.encode("utf-16"),
        b"[]\n",
        # Versions 3 to 8 each became real schemas when a durable collection
        # arrived that the previous one could not carry -- the frozen attempt
        # topologies, then the credential rotation and cell stage ledgers, then
        # the service replica health and decision logs, then the webhook
        # subscriptions and delivery cursors, then team accounts, then team
        # requests, then assistant providers, then assistant conversations, then
        # research plans. The unsupported-version case moves up each time rather
        # than disappearing: an unknown version must still be refused by
        # version, not by silently dropping the state it describes.
        b'{"schema":"eyvan.control-plane","version":12}\n',
        b'{"version":2}\n',
        b'{"schema":"eyvan.control-plane","version":{},"revision":0}\n',
        b'{"schema":"eyvan.control-plane","version":2,"unexpected":true}\n',
    ),
)
def test_local_store_rejects_malformed_foreign_or_unsupported_documents(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "control-plane.json"
    path.write_bytes(payload)
    path.chmod(0o600)

    with pytest.raises(LocalControlPlaneSchemaError):
        LocalControlPlaneStore(path).load()


@pytest.mark.parametrize(
    ("collection", "identity_error"),
    (
        ("progress_reports", "workload/rank"),
        ("plane_peers", "plane IDs"),
        ("budget_grants", "budget grants"),
        ("quota_grants", "quota grants"),
        ("managed_artifacts", "organization/project/URI"),
        ("artifact_references", "organization/project/URI/workload"),
        ("events", "audit events"),
    ),
)
def test_v2_document_rejects_duplicate_complete_identities(
    tmp_path: Path,
    collection: str,
    identity_error: str,
) -> None:
    path = tmp_path / "control-plane.json"
    raw = LocalControlPlaneDocument.from_state(_complete_control_plane_state()).model_dump(mode="json", by_alias=True)
    records = raw[collection]
    assert isinstance(records, list)
    records.append(records[0])
    path.write_text(json.dumps(raw))
    path.chmod(0o600)

    with pytest.raises(LocalControlPlaneSchemaError, match=identity_error):
        LocalControlPlaneStore(path).load()
