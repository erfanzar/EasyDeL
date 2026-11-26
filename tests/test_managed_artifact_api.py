# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import hashlib
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from eyvan.artifacts.managed import ArtifactAction, InvalidArtifactToken, callable_uri, result_uri, storage_scope
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import CallableExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.worker.runtime.results import dump_result


def _headers(*, org_id: str = "acme", project_id: str = "research") -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Roles": "operator",
        "X-Eyvan-Org": org_id,
        "X-Eyvan-Project": project_id,
    }


def _route(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.path}?{parsed.query}"


def _app(tmp_path, **updates):
    settings = EyvanSettings(
        dev_auth_enabled=True,
        trusted_callable_projects=("acme/research",),
        public_url="https://eyvan.test",
        artifact_store_uri=str(tmp_path / "managed"),
        artifact_token_key="k" * 32,
        artifact_token_ttl_s=3600,
        **updates,
    )
    return create_app(settings=settings)


def _register_result_target(app, workload_id: str, uri: str) -> WorkloadRecord:
    workload = WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="managed-callable",
            execution=CallableExecution(
                artifact_uri=callable_uri("0" * 64),
                sha256="0" * 64,
                python="3.13",
                result_uri=uri,
            ),
        ),
        state=WorkloadState.RUNNING,
        created_by="alice",
    )

    async def store() -> None:
        async with app.state.service.repository.transaction() as transaction:
            await transaction.put_workload(workload)

    asyncio.run(store())
    return workload


def test_managed_callable_and_result_round_trip_with_scoped_transfer_tokens(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    callable_payload = b"serialized callable"
    digest = hashlib.sha256(callable_payload).hexdigest()

    uploaded = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers={**_headers(), "Content-Type": "application/octet-stream"},
        content=callable_payload,
    )

    assert uploaded.status_code == 200
    assert uploaded.json()["uri"] == callable_uri(digest)
    access = app.state.artifact_access
    download = access.callable_download_transfer(
        callable_uri(digest),
        org_id="acme",
        project_id="research",
        workload_id="wl-managed",
    )
    downloaded = client.get(_route(download.url), headers={"X-Eyvan-Artifact-Token": download.token})
    assert downloaded.status_code == 200
    assert downloaded.content == callable_payload
    assert downloaded.headers["Cache-Control"] == "private, no-store"

    result_id = "call-managed-result"
    result_payload = dump_result({"loss": 0.25})
    _register_result_target(app, "wl-managed", result_uri(result_id))
    upload = access.result_upload_transfer(
        result_uri(result_id),
        org_id="acme",
        project_id="research",
        workload_id="wl-managed",
    )
    assert "token" not in parse_qs(urlsplit(upload.url).query)
    stored = client.put(
        _route(upload.url),
        headers={"Content-Type": "application/octet-stream", "X-Eyvan-Artifact-Token": upload.token},
        content=result_payload,
    )
    assert stored.status_code == 200

    result = client.get(
        f"/api/v1/artifacts/results/{result_id}",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers(),
    )
    assert result.status_code == 200
    assert result.content == result_payload


def test_local_artifact_permission_failure_returns_503_without_changing_lifecycle(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    payload = b"permission-failure"
    digest = hashlib.sha256(payload).hexdigest()
    uri = callable_uri(digest)
    uploaded = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers={**_headers(), "Content-Type": "application/octet-stream"},
        content=payload,
    )
    assert uploaded.status_code == 200
    access = app.state.artifact_access
    target = next((access.store.root / "scopes").rglob(f"{digest}.pkl"))
    transfer = access.callable_download_transfer(
        uri,
        org_id="acme",
        project_id="research",
        workload_id="wl-permission",
    )
    target.chmod(0)
    try:
        response = client.get(
            _route(transfer.url),
            headers={"X-Eyvan-Artifact-Token": transfer.token},
        )
    finally:
        target.chmod(0o400)

    assert response.status_code == 503
    assert "Permission" not in response.text
    assert str(target) not in response.text

    async def state() -> str:
        async with app.state.service.repository.transaction() as transaction:
            record = await transaction.get_managed_artifact("acme", "research", uri)
        assert record is not None
        return record.state.value

    assert asyncio.run(state()) == "active"


def test_content_addressed_upload_mismatch_is_rejected_before_reservation(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    intended_digest = hashlib.sha256(b"intended").hexdigest()
    uri = callable_uri(intended_digest)

    response = client.put(
        f"/api/v1/artifacts/callables/{intended_digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers={**_headers(), "Content-Type": "application/octet-stream"},
        content=b"different",
    )

    assert response.status_code == 400

    async def missing() -> bool:
        async with app.state.service.repository.transaction() as transaction:
            return await transaction.get_managed_artifact("acme", "research", uri) is None

    assert asyncio.run(missing()) is True


def test_managed_artifact_tokens_and_tenant_storage_do_not_cross_scopes(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    result_id = "private-result"
    _register_result_target(app, "wl-private", result_uri(result_id))
    upload = app.state.artifact_access.result_upload_transfer(
        result_uri(result_id),
        org_id="acme",
        project_id="research",
        workload_id="wl-private",
    )
    tampered = _route(upload.url).replace("workload_id=wl-private", "workload_id=wl-other")

    rejected = client.put(tampered, headers={"X-Eyvan-Artifact-Token": upload.token}, content=b"secret")
    assert rejected.status_code == 403

    stored = client.put(
        _route(upload.url),
        headers={"X-Eyvan-Artifact-Token": upload.token},
        content=b"secret",
    )
    assert stored.status_code == 200
    cross_scope = client.get(
        f"/api/v1/artifacts/results/{result_id}",
        params={"org_id": "other", "project_id": "other"},
        headers=_headers(org_id="other", project_id="other"),
    )
    assert cross_scope.status_code == 404


def test_stale_result_transfer_cannot_write_after_workload_generation_changes(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    old_uri = result_uri("wl-stale-g1")
    new_uri = result_uri("wl-stale-g2")
    workload = _register_result_target(app, "wl-stale", old_uri)
    stale = app.state.artifact_access.result_upload_transfer(
        old_uri,
        org_id="acme",
        project_id="research",
        workload_id=workload.id,
    )

    async def advance_generation() -> None:
        execution = workload.spec.execution
        assert isinstance(execution, CallableExecution)
        updated = workload.model_copy(
            update={
                "generation": 2,
                "spec": workload.spec.model_copy(
                    update={"execution": execution.model_copy(update={"result_uri": new_uri})}
                ),
            }
        )
        async with app.state.service.repository.transaction() as transaction:
            await transaction.put_workload(updated)

    asyncio.run(advance_generation())

    rejected = client.put(
        _route(stale.url),
        headers={"X-Eyvan-Artifact-Token": stale.token},
        content=b"stale",
    )
    assert rejected.status_code == 403
    assert "not current" in rejected.json()["detail"]


def test_managed_callable_upload_requires_trust_and_enforces_size_limit(tmp_path) -> None:
    app = _app(tmp_path, artifact_max_callable_bytes=4)
    client = TestClient(app)
    payload = b"12345"
    digest = hashlib.sha256(payload).hexdigest()

    too_large = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers(),
        content=payload,
    )
    assert too_large.status_code == 400
    assert "4-byte limit" in too_large.json()["detail"]

    chunked = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers(),
        content=iter((b"123", b"45")),
    )
    assert chunked.status_code == 400
    assert "4-byte limit" in chunked.json()["detail"]

    untrusted = client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "other"},
        headers=_headers(project_id="other"),
        content=payload[:4],
    )
    assert untrusted.status_code == 403


def test_non_local_repository_requires_an_explicit_artifact_token_key(tmp_path) -> None:
    settings = EyvanSettings(
        dev_auth_enabled=True,
        repository_backend=RepositoryBackend.POSTGRES,
        artifact_store_uri=str(tmp_path / "managed"),
    )

    with pytest.raises(ValueError, match="EYVAN_ARTIFACT_TOKEN_KEY"):
        create_app(settings=settings, repository=InMemoryRepository())


def test_in_memory_repository_mints_an_ephemeral_per_process_artifact_token_key(tmp_path) -> None:
    def build_app():
        return create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_store_uri=str(tmp_path / "managed"),
            )
        )

    first = build_app()
    second = build_app()
    token = first.state.artifact_access.signer.issue(
        ArtifactAction.WRITE,
        result_uri("ephemeral-result"),
        org_id="acme",
        project_id="research",
        workload_id="wl-ephemeral",
        expires=False,
    )

    with pytest.raises(InvalidArtifactToken):
        second.state.artifact_access.signer.verify(
            token,
            ArtifactAction.WRITE,
            result_uri("ephemeral-result"),
            org_id="acme",
            project_id="research",
            workload_id="wl-ephemeral",
        )


def _audit_actions(app, entity_type: str = "artifact") -> list[tuple[str, str, str]]:
    """Return (actor, action, entity_id) for every audit event of one entity type."""

    async def read():
        async with app.state.service.repository.transaction() as transaction:
            return await transaction.list_events(after_sequence=0, limit=1000)

    return [
        (event.actor, event.action, event.entity_id) for event in asyncio.run(read()) if event.entity_type == entity_type
    ]


def test_every_capability_authenticated_transfer_is_audited(tmp_path) -> None:
    """These three routes authenticate a *workload*, not a principal.

    They served content with no audit trail at all, so a leaked or stale transfer
    token left no record of having been used -- and "audit every access" cannot be
    built for secrets on a pattern that audits nothing.
    """

    app = _app(tmp_path)
    client = TestClient(app)
    payload = b"serialized callable"
    digest = hashlib.sha256(payload).hexdigest()
    client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers={**_headers(), "Content-Type": "application/octet-stream"},
        content=payload,
    )
    access = app.state.artifact_access
    download = access.callable_download_transfer(
        callable_uri(digest), org_id="acme", project_id="research", workload_id="wl-audit"
    )

    assert client.get(_route(download.url), headers={"X-Eyvan-Artifact-Token": download.token}).status_code == 200

    events = _audit_actions(app)
    assert ("workload:wl-audit", "artifact.read", f"callables/{callable_uri(digest)}") in events
    # The actor is the workload, not the operator who happened to upload it.
    transfer_events = [event for event in events if event[1] == "artifact.read"]
    assert all(actor.startswith("workload:") for actor, _action, _entity in transfer_events)


def test_a_refused_transfer_token_is_audited_too(tmp_path) -> None:
    """A rejection is audited in the target workload's durable tenant scope.

    The unauthenticated request cannot supply its own audit-ledger scope, so
    the fixture must contain the workload whose transfer is being refused.
    """

    app = _app(tmp_path)
    client = TestClient(app)
    _register_result_target(app, "wl-audit", result_uri("call-denial-audit"))
    payload = b"serialized callable"
    digest = hashlib.sha256(payload).hexdigest()
    client.put(
        f"/api/v1/artifacts/callables/{digest}",
        params={"org_id": "acme", "project_id": "research"},
        headers={**_headers(), "Content-Type": "application/octet-stream"},
        content=payload,
    )
    access = app.state.artifact_access
    download = access.callable_download_transfer(
        callable_uri(digest), org_id="acme", project_id="research", workload_id="wl-audit"
    )

    refused = client.get(_route(download.url), headers={"X-Eyvan-Artifact-Token": "v1.forged.signature"})

    assert refused.status_code == 403
    assert any(action == "artifact.read_denied" for _actor, action, _entity in _audit_actions(app))


def test_a_granted_result_upload_cannot_commit_unaudited(tmp_path) -> None:
    """The write audit shares the authorization read's transaction.

    A separate transaction could commit the upload's authorization and then fail to
    record it, which is precisely the gap an auditor would care about.
    """

    app = _app(tmp_path)
    client = TestClient(app)
    result_id = "call-audited-result"
    _register_result_target(app, "wl-audit", result_uri(result_id))
    upload = app.state.artifact_access.result_upload_transfer(
        result_uri(result_id), org_id="acme", project_id="research", workload_id="wl-audit"
    )

    stored = client.put(
        _route(upload.url),
        headers={"Content-Type": "application/octet-stream", "X-Eyvan-Artifact-Token": upload.token},
        content=dump_result({"loss": 0.25}),
    )

    assert stored.status_code == 200
    assert any(action == "artifact.write" for _actor, action, _entity in _audit_actions(app))


def test_a_stale_result_upload_is_audited_as_denied(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    result_id = "call-stale-result"
    upload = app.state.artifact_access.result_upload_transfer(
        result_uri(result_id), org_id="acme", project_id="research", workload_id="wl-missing"
    )

    refused = client.put(
        _route(upload.url),
        headers={"Content-Type": "application/octet-stream", "X-Eyvan-Artifact-Token": upload.token},
        content=dump_result({"loss": 0.25}),
    )

    assert refused.status_code == 403
    assert any(action == "artifact.write_denied" for _actor, action, _entity in _audit_actions(app))


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    (
        ("POST", "/api/v1/artifacts/holds", {"json": {"uri": result_uri("legacy"), "reason": "keep"}}),
        ("DELETE", "/api/v1/artifacts/holds", {"params": {"uri": result_uri("legacy")}}),
        ("GET", "/api/v1/artifacts/references", {"params": {"uri": result_uri("legacy")}}),
    ),
)
def test_legacy_unscoped_artifact_lifecycle_calls_receive_actionable_upgrade_required(
    tmp_path,
    method: str,
    path: str,
    request_kwargs: dict[str, object],
) -> None:
    client = TestClient(_app(tmp_path))

    response = client.request(method, path, headers=_headers(), **request_kwargs)

    assert response.status_code == 426
    assert response.json()["code"] == "artifact_scope_required"
    assert response.json()["context"] == {"required_fields": ["org_id", "project_id"]}
    assert "upgrade" in response.json()["detail"]
    assert response.headers["X-Eyvan-Required-Client-Feature"] == "scoped-artifact-identity"


def test_non_scope_validation_errors_remain_ordinary_422_responses(tmp_path) -> None:
    client = TestClient(_app(tmp_path))

    response = client.post(
        "/api/v1/artifacts/holds",
        headers=_headers(),
        json={"org_id": "acme", "project_id": "research", "uri": result_uri("invalid-body")},
    )

    assert response.status_code == 422
    assert "code" not in response.json()


def test_inventory_api_is_dry_run_first_and_reindex_requires_scope_acknowledgement(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    uri = result_uri("physical-only")
    store = app.state.artifact_access.store
    store.put(uri, b"physical", org_id="acme", project_id="research")

    dry_run = client.get(
        "/api/v1/artifacts/inventory",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers(),
    )

    assert dry_run.status_code == 200
    assert dry_run.json()["dry_run"] is True
    assert dry_run.json()["entries"] == [
        {
            "uri": uri,
            "status": "physical_only",
            "indexed_size_bytes": None,
            "observed_size_bytes": len(b"physical"),
            "lifecycle_state": None,
            "indexed_object_version": None,
            "observed_object_version": dry_run.json()["entries"][0]["observed_object_version"],
            "reindexed": False,
        }
    ]

    admin_headers = {**_headers(), "X-Eyvan-Roles": "project_admin"}
    refused = client.post(
        "/api/v1/artifacts/inventory/reindex",
        headers=admin_headers,
        json={
            "org_id": "acme",
            "project_id": "research",
            "limit": 100,
            "acknowledge_scope": "acme/other",
        },
    )
    assert refused.status_code == 400
    assert "acknowledge_scope" in refused.json()["detail"]

    repaired = client.post(
        "/api/v1/artifacts/inventory/reindex",
        headers=admin_headers,
        json={
            "org_id": "acme",
            "project_id": "research",
            "limit": 100,
            "acknowledge_scope": "acme/research",
        },
    )
    assert repaired.status_code == 200
    assert repaired.json()["reindexed_count"] == 1
    assert repaired.json()["entries"][0]["reindexed"] is True
    assert store.get(uri, org_id="acme", project_id="research") == b"physical"
    assert any(action == "artifact.inventory_reindexed" for _actor, action, _entity in _audit_actions(app))


def test_inventory_reindex_fails_closed_on_content_addressed_physical_mismatch(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    intended = b"intended"
    digest = hashlib.sha256(intended).hexdigest()
    uri = callable_uri(digest)
    store = app.state.artifact_access.store
    target = store.root / "scopes" / storage_scope("acme", "research") / "callables" / f"{digest}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"wrong-physical-bytes")
    target.chmod(0o400)

    response = client.post(
        "/api/v1/artifacts/inventory/reindex",
        headers={**_headers(), "X-Eyvan-Roles": "project_admin"},
        json={
            "org_id": "acme",
            "project_id": "research",
            "limit": 100,
            "acknowledge_scope": "acme/research",
        },
    )

    assert response.status_code == 503
    assert response.json()["code"] == "artifact_inventory_inconclusive"

    async def missing() -> bool:
        async with app.state.service.repository.transaction() as transaction:
            return await transaction.get_managed_artifact("acme", "research", uri) is None

    assert asyncio.run(missing()) is True


def test_inventory_api_fails_closed_instead_of_treating_parse_failure_as_an_empty_scope(tmp_path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    uri = result_uri("valid")
    store = app.state.artifact_access.store
    store.put(uri, b"physical", org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "unexpected.txt"
    target.write_bytes(b"unknown")

    response = client.get(
        "/api/v1/artifacts/inventory",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers(),
    )

    assert response.status_code == 503
    assert response.json()["code"] == "artifact_inventory_inconclusive"

    async def missing() -> bool:
        async with app.state.service.repository.transaction() as transaction:
            return await transaction.get_managed_artifact("acme", "research", uri) is None

    assert asyncio.run(missing()) is True
