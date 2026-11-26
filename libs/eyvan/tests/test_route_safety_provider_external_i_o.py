# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Keep provider and external I/O outside route-level transactions."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.identity.rotation import CredentialRotationBlocked
from eyvan.records import CommandExecution, Principal, Role, RoleGrant, WorkloadDefinition, WorkloadRecord
from eyvan.server.api.app import create_app
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.credential_rotation import SECRET_CAPABILITY_CREDENTIAL, CredentialRotationService


def _settings(tmp_path) -> EyvanSettings:
    return EyvanSettings(
        dev_auth_enabled=True,
        controller_enabled=False,
        artifact_token_key="a" * 64,
        artifact_store_uri=str(tmp_path / "artifacts"),
    )


def _headers(role="fleet_admin") -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": role,
    }


def _issue(app) -> str:
    return app.state.secret_capability.issue(
        workload_id="wl-1",
        org_id="acme",
        project_id="research",
        attempt=1,
        generation=1,
        dispatch_fence="dispatch-fence",
        name="TOKEN",
    )


def _verify(app, token) -> None:
    app.state.secret_capability.verify(token, workload_id="wl-1", org_id="acme", project_id="research")


async def test_unsupported_rotation_refuses_without_diverging_replicas_or_ledger(tmp_path) -> None:
    repository = InMemoryRepository()
    settings = _settings(tmp_path)
    first = create_app(settings=settings, repository=repository)
    other = create_app(settings=settings, repository=repository)
    old = _issue(first)
    with TestClient(first) as client:
        assert client.post("/api/v1/secrets/rotate", headers=_headers("viewer"), json={}).status_code == 403
        response = client.post("/api/v1/secrets/rotate", headers=_headers(), json={"overlap_s": 0})
        assert response.status_code == 503, response.text
        assert "shared" in response.json()["detail"]
        assert await first.state.credential_rotation.current(SECRET_CAPABILITY_CREDENTIAL) is None
    replacement = create_app(settings=settings, repository=repository)
    new = _issue(first)
    for app in (first, other, replacement):
        assert app.state.secret_capability_credential.generation == 1
        _verify(app, old)
        _verify(app, new)


async def test_unsupported_rotation_cannot_install_before_a_durable_commit_failure(tmp_path, monkeypatch) -> None:
    repository = LocalJsonRepository(tmp_path / "state.json")
    app = create_app(settings=_settings(tmp_path), repository=repository)
    original_persist = repository._persist

    def fail_committed_snapshot(state):
        if any(row.state.value == "committed" for row in state.credential_rotations.values()):
            raise RuntimeError("injected durable commit failure")
        original_persist(state)

    monkeypatch.setattr(repository, "_persist", fail_committed_snapshot)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/secrets/rotate", headers=_headers(), json={"overlap_s": 0})
        current = await app.state.credential_rotation.current(SECRET_CAPABILITY_CREDENTIAL)
        assert response.status_code == 503, (
            response.status_code,
            current.state.value if current else None,
            app.state.secret_capability_credential.rotation_state,
            bool(_issue(app)),
        )
        assert current is None
        assert app.state.secret_capability_credential.generation == 1
        _verify(app, _issue(app))


@pytest.mark.parametrize("outcome", ["pending", "committed", "inconclusive"])
async def test_startup_refuses_unrecoverable_historical_rotation_before_serving(tmp_path, outcome) -> None:
    path = tmp_path / "state.json"
    repository = LocalJsonRepository(path)
    await repository.open()
    ledger = CredentialRotationService(repository)
    begun = await ledger.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence", overlap_s=0)
    if outcome != "pending":
        finalize = ledger.commit if outcome == "committed" else ledger.record_inconclusive
        await finalize(credential=SECRET_CAPABILITY_CREDENTIAL, generation=begun.generation, fence="fence")
    await repository.close()
    reopened = LocalJsonRepository(path)
    replacement = create_app(settings=_settings(tmp_path), repository=reopened)
    with pytest.raises(CredentialRotationBlocked, match="shared"):
        with TestClient(replacement):
            pytest.fail("a historical volatile rotation must not serve using the original key")
    await reopened.open()
    assert (await CredentialRotationService(reopened).current(SECRET_CAPABILITY_CREDENTIAL)).state.value == outcome
    await reopened.close()


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_documentation_has_compatible_scoped_csp_without_weakening_dashboard(tmp_path, path) -> None:
    with TestClient(create_app(settings=_settings(tmp_path))) as client:
        response = client.get(path)
        dashboard = client.get("/")
    assert response.status_code == 200
    policy = response.headers["Content-Security-Policy"]
    script_policy = next(part.strip() for part in policy.split(";") if part.strip().startswith("script-src "))
    if "https://cdn.jsdelivr.net/" in response.text:
        assert "https://cdn.jsdelivr.net" in script_policy
    if "<script>" in response.text:
        assert "'unsafe-inline'" in script_policy
    assert "script-src 'self';" in dashboard.headers["Content-Security-Policy"]
    assert "https://cdn.jsdelivr.net" not in dashboard.headers["Content-Security-Policy"]
    assert response.headers["X-Frame-Options"] == dashboard.headers["X-Frame-Options"] == "DENY"


async def _log_app(tmp_path, page):
    repository = InMemoryRepository()
    async with repository.transaction() as tx:
        await tx.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id="acme",
                project_id="research",
                created_by="alice",
                spec=WorkloadDefinition(name="logs", execution=CommandExecution(command="echo hi")),
            )
        )
    app = create_app(settings=_settings(tmp_path), repository=repository)

    async def scripted_page(workload_id, *, after="", limit=200):
        return page

    app.state.controller.workload_log_page = scripted_page
    return app


@pytest.mark.parametrize(
    "metadata",
    [
        {"unavailable": True, "detail": "rank 0 unreachable", "truncated_ranks": [0]},
        {"unavailable": False, "truncated_ranks": [1], "missing_ranks": [1], "topology_state": "changed"},
        {"unavailable": True, "detail": "output suppressed for secret-bearing execution"},
        {"unavailable": False, "completeness_unknown": True, "topology_state": "unknown"},
        {"unavailable": False, "detail": "", "truncated_ranks": [], "missing_ranks": [], "topology_state": "frozen"},
    ],
)
async def test_log_status_opt_in_preserves_empty_partial_and_suppressed_page_diagnostics(tmp_path, metadata) -> None:
    records = [{"message": "surviving rank", "rank": 0}] if metadata.get("truncated_ranks") == [1] else []
    page = {"attempt": 1, "records": records, "next_cursor": "1:0=c1" if records else "", **metadata}
    app = await _log_app(tmp_path, page)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/workloads/wl-1/logs/stream",
            params={"follow": "false", "include_status": "true"},
            headers=_headers("operator"),
        )
        legacy = client.get(
            "/api/v1/workloads/wl-1/logs/stream",
            params={"follow": "false"},
            headers=_headers("operator"),
        )
    assert response.status_code == 200
    frames = [frame for frame in response.text.split("\n\n") if "data: " in frame]
    statuses = [frame for frame in frames if "event: log.page_status\n" in frame]
    assert len(statuses) == 1, response.text
    document = json.loads(statuses[0].split("data: ", 1)[1])
    for key, value in metadata.items():
        assert document[key] == value
    assert document["attempt"] == 1
    assert response.text.count("event: log.record") == len(records)
    assert "log.page_status" not in legacy.text
    if records:
        assert frames[-1] == statuses[0]
        assert response.text.index("surviving rank") < response.text.index("id: 1:0=c1")
        assert "id: " not in frames[0]
    else:
        assert "id: " not in response.text
        assert legacy.text == ""


async def test_followed_log_status_deduplicates_quiet_polls_and_announces_recovery(tmp_path, monkeypatch) -> None:
    page = {"attempt": 1, "records": [], "next_cursor": "", "unavailable": True, "detail": "unreachable"}
    app = await _log_app(tmp_path, page)
    polls = 0

    async def evolving_page(workload_id, *, after="", limit=200):
        nonlocal polls
        polls += 1
        return page if polls < 3 else {**page, "unavailable": False, "detail": ""}

    class ConnectedRequest:
        async def is_disconnected(self):
            return polls >= 3

    # Avoid waiting for the normal poll interval; no provider/agent read occurs.
    monkeypatch.setattr("eyvan.server.api.app._LOG_STREAM_POLL_S", 0)
    app.state.controller.workload_log_page = evolving_page
    route = next(
        route for route in app.routes if getattr(route, "path", None) == "/api/v1/workloads/{workload_id}/logs/stream"
    )
    principal = Principal(
        subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
    )
    response = await route.endpoint(
        raw_request=ConnectedRequest(),
        workload_id="wl-1",
        principal=principal,
        after="",
        last_event_id=None,
        follow=True,
        limit=200,
        include_status=True,
    )
    text = "".join([frame async for frame in response.body_iterator])
    statuses = [
        json.loads(frame.split("data: ", 1)[1]) for frame in text.split("\n\n") if "event: log.page_status" in frame
    ]
    assert [status["unavailable"] for status in statuses] == [True, False]
    assert text.count(": keep-alive") == 2
    assert "id: " not in text


async def test_log_status_clipping_is_explicit_and_does_not_copy_unknown_page_fields(tmp_path) -> None:
    page = {
        "attempt": 1,
        "records": [],
        "next_cursor": "",
        "unavailable": True,
        "detail": "x" * 2048,
        "truncated_ranks": list(range(1001)),
        "missing_ranks": list(range(1001)),
        "untrusted_extra": "must-not-be-forwarded",
    }
    app = await _log_app(tmp_path, page)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/workloads/wl-1/logs/stream",
            params={"follow": "false", "include_status": "true"},
            headers=_headers("operator"),
        )
    document = json.loads(response.text.split("data: ", 1)[1])
    assert document["status_truncated"] is True
    assert len(document["detail"]) == 1024
    assert len(document["truncated_ranks"]) == len(document["missing_ranks"]) == 1000
    assert "untrusted_extra" not in document
    assert "must-not-be-forwarded" not in response.text
