# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Authenticated, project-scoped reads through the tracker proxy."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api.app import create_app


class _Verifier:
    async def verify(self, token: str) -> Principal:
        return Principal(
            subject="alice", grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),)
        )


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return create_app(
        settings=EyvanSettings(
            controller_enabled=False,
            artifact_store_uri=str(tmp_path / "artifacts"),
            tracker_url="https://tracker.example",
            tracker_token="tracker-secret",
        ),
        verifier=_Verifier(),
    )


def test_proxy_auth_and_scope_before_network(app) -> None:
    requests = []
    app.state.tracking_transport = httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json={})
    )
    with TestClient(app) as client:
        assert client.get("/api/v1/tracking/health").status_code == 401
        assert (
            client.get(
                "/api/v1/tracking/runs?org_id=acme&project_id=other", headers={"Authorization": "Bearer user"}
            ).status_code
            == 403
        )
        assert client.get("/api/v1/tracking/health", headers={"Authorization": "Bearer user"}).status_code == 200
    assert len(requests) == 1


def test_proxy_preserves_repeated_keys_token_and_json_status(app) -> None:
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(404, json={"detail": "run absent"})

    app.state.tracking_transport = httpx.MockTransport(upstream)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/tracking/runs/run-1/series?org_id=acme&project_id=research&key=loss&key=reward&points=21",
            headers={"Authorization": "Bearer user"},
        )
    assert response.status_code == 404
    assert response.json() == {"detail": "run absent"}
    assert requests[0].url.path == "/api/v1/tracking/runs/run-1/series"
    assert requests[0].url.params.get_list("key") == ["loss", "reward"]
    assert requests[0].url.params["points"] == "21"
    assert requests[0].headers["Authorization"] == "Bearer tracker-secret"
    assert "Bearer user" not in requests[0].headers.values()


@pytest.mark.parametrize(
    "path",
    [
        "runs",
        "runs/run-1",
        "runs/run-1/series",
        "runs/run-1/histograms",
        "runs/run-1/episodes",
        "runs/run-1/episodes/curve",
        "runs/run-1/episodes/breakdown",
        "runs/run-1/rollouts",
        "runs/run-1/rollouts/a1-r0-abc-0-0",
        "runs/run-1/completions",
        "runs/run-1/completions/stats",
        "runs/run-1/completions/a1-r0-abc-0-0",
        "runs/run-1/tables",
        "runs/run-1/media",
    ],
)
def test_proxy_all_data_routes_are_registered(app, path) -> None:
    app.state.tracking_transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/tracking/{path}?org_id=acme&project_id=research", headers={"Authorization": "Bearer u"}
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


_DIGEST = "a" * 64


def test_proxy_blob_passes_bytes_with_security_headers(app) -> None:
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(200, content=b"\x89PNG-bytes", headers={"Content-Type": "image/png"})

    app.state.tracking_transport = httpx.MockTransport(upstream)
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/tracking/runs/run-1/blobs/{_DIGEST}?org_id=acme&project_id=research",
            headers={"Authorization": "Bearer u"},
        )
    assert response.status_code == 200
    assert response.content == b"\x89PNG-bytes"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert response.headers["content-disposition"] == "inline"
    assert requests[0].url.path == f"/api/v1/tracking/runs/run-1/blobs/{_DIGEST}"
    assert requests[0].headers["Authorization"] == "Bearer tracker-secret"


@pytest.mark.parametrize("content_type", ["text/html", "image/svg+xml", "application/javascript", ""])
def test_proxy_blob_refuses_types_outside_the_allowlist(app, content_type) -> None:
    headers = {"Content-Type": content_type} if content_type else {}
    app.state.tracking_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"<script>alert(1)</script>", headers=headers)
    )
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/tracking/runs/run-1/blobs/{_DIGEST}?org_id=acme&project_id=research",
            headers={"Authorization": "Bearer u"},
        )
    assert response.status_code == 502
    assert b"<script>" not in response.content


def test_proxy_blob_authorizes_scope_and_validates_digest_before_network(app) -> None:
    requests = []
    app.state.tracking_transport = httpx.MockTransport(
        lambda request: (
            requests.append(request) or httpx.Response(200, content=b"x", headers={"Content-Type": "image/png"})
        )
    )
    with TestClient(app) as client:
        foreign = client.get(
            f"/api/v1/tracking/runs/run-1/blobs/{_DIGEST}?org_id=acme&project_id=other",
            headers={"Authorization": "Bearer u"},
        )
        malformed = client.get(
            "/api/v1/tracking/runs/run-1/blobs/NOT-A-DIGEST?org_id=acme&project_id=research",
            headers={"Authorization": "Bearer u"},
        )
    assert foreign.status_code == 403
    assert malformed.status_code == 422
    assert requests == []


def test_proxy_blob_passes_tracker_errors_through_as_json(app) -> None:
    app.state.tracking_transport = httpx.MockTransport(
        lambda request: httpx.Response(404, json={"detail": "blob not available"})
    )
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/tracking/runs/run-1/blobs/{_DIGEST}?org_id=acme&project_id=research",
            headers={"Authorization": "Bearer u"},
        )
    assert response.status_code == 404
    assert response.json() == {"detail": "blob not available"}


def test_proxy_without_tracker_is_actionable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    app = create_app(
        settings=EyvanSettings(controller_enabled=False, artifact_store_uri=str(tmp_path / "artifacts")),
        verifier=_Verifier(),
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/tracking/health", headers={"Authorization": "Bearer u"})
    assert response.status_code == 503
    assert "eyvan track up --local" in response.json()["detail"]


def test_proxy_unreachable_does_not_expose_token(app) -> None:
    def upstream(request):
        raise httpx.ConnectError("secret-containing diagnostic", request=request)

    app.state.tracking_transport = httpx.MockTransport(upstream)
    with TestClient(app) as client:
        response = client.get("/api/v1/tracking/health", headers={"Authorization": "Bearer u"})
    assert response.status_code == 502
    assert "secret" not in response.text
    assert "eyvan track status" in response.json()["detail"]

    def timed_out(request):
        raise httpx.ReadTimeout("secret-timeout", request=request)

    app.state.tracking_transport = httpx.MockTransport(timed_out)
    with TestClient(app) as client:
        timeout_resp = client.get("/api/v1/tracking/health", headers={"Authorization": "Bearer u"})
    assert timeout_resp.status_code == 502
    assert "secret" not in timeout_resp.text
    assert "timed out" in timeout_resp.json()["detail"]


@pytest.mark.parametrize("token", ["local-secret", ""])
def test_proxy_resolves_private_local_record_and_refuses_invalid_token(tmp_path, monkeypatch, token) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".eyvan" / "tracker"
    root.mkdir(parents=True, mode=0o700)
    (root / "token").write_text(token)
    (root / "token").chmod(0o600)
    record = {
        "url": "http://127.0.0.1:8095",
        "pid": 98765,
        "store": str(tmp_path / "runs"),
        "db": str(root / "index.sqlite3"),
        "log": str(root / "tracker.log"),
        "token_path": str(root / "token"),
    }
    (root / "record.json").write_text(json.dumps(record))
    (root / "record.json").chmod(0o600)
    app = create_app(
        settings=EyvanSettings(controller_enabled=False, artifact_store_uri=str(tmp_path / "artifacts")),
        verifier=_Verifier(),
    )
    requests = []
    app.state.tracking_transport = httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json={"runs": []})
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/tracking/runs?org_id=acme&project_id=research", headers={"Authorization": "Bearer u"}
        )
    if token:
        assert response.status_code == 200
        assert requests[0].headers["Authorization"] == "Bearer local-secret"
        assert requests[0].url.host == "127.0.0.1"
    else:
        assert response.status_code == 503
        assert requests == []


def test_proxy_authorized_duplicate_scope_is_the_only_forwarded_scope(app) -> None:
    requests = []
    app.state.tracking_transport = httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json={"runs": []})
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/tracking/runs?org_id=foreign&org_id=acme&project_id=foreign&project_id=research",
            headers={"Authorization": "Bearer u"},
        )
    assert response.status_code == 200
    assert requests[0].url.params.get_list("org_id") == ["acme"]
    assert requests[0].url.params.get_list("project_id") == ["research"]


def test_explicit_tracker_wins_over_local_record_and_invalid_json_is_502(app, monkeypatch) -> None:
    from eyvan.server.api import tracking_proxy

    def forbidden_local_read():
        raise AssertionError("explicit URL must not read the local record")

    monkeypatch.setattr(tracking_proxy, "read_tracker_record", forbidden_local_read)
    app.state.tracking_transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON"))
    with TestClient(app) as client:
        response = client.get("/api/v1/tracking/health", headers={"Authorization": "Bearer u"})
    assert response.status_code == 502
