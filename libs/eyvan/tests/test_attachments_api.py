# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local attachment jobs use the CLI without real SSH or provider traffic."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api.app import create_app
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.local import attach_job

ADMIN = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "fleet_admin"}
BODY = {"target": "alice@203.0.113.20", "kind": "ssh", "fleet": "research"}
PLAN = {
    "dry_run": True,
    "fleet": "research",
    "fleet_path": "/private/research.yaml",
    "fleet_yaml": (
        "name: research\ncapacity_class: spot\npools:\n  v5p-8:\n"
        "    accelerator_type: v5p-8\n    provider_options:\n      endpoint_mode: tunnel\n"
    ),
    "endpoints": ["http://127.0.0.1:18001"],
    "actions": [{"step": "install", "summary": "Install the worker", "argv": ["ssh", "host"], "stdin": True}],
    "limitations": [],
}
RESULT = {"fleet": "research", "ready": True, "cells": [], "submit": 'init(fleet="research")', "limitations": []}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def unexpected_process(*args, **kwargs):
        pytest.fail("attachment tests must replace every CLI subprocess; no SSH/cloud traffic is permitted")

    monkeypatch.setattr(attach_job.subprocess, "run", unexpected_process)
    monkeypatch.setattr(attach_job.subprocess, "Popen", unexpected_process)


def client(*, local: bool = True, development: bool = True) -> TestClient:
    class Verifier:
        async def verify(self, token: str) -> Principal:
            return Principal(subject="alice", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    settings = EyvanSettings(
        dev_auth_enabled=development and local,
        allow_insecure_plaintext_bind=True,
        metrics_token="metrics-test",
        artifact_token_key="k" * 32,
    )
    app = create_app(
        settings=settings,
        verifier=Verifier(),
        server_bind=ServerBindPolicy.from_host("127.0.0.1" if local else "0.0.0.0"),
    )
    return TestClient(app, base_url="http://127.0.0.1:8181")


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/v1/attachments/plan"),
        ("post", "/api/v1/attachments"),
        ("get", "/api/v1/attachments"),
        ("get", "/api/v1/attachments/" + "a" * 32),
        ("get", "/api/v1/attachments/discovered"),
    ],
)
def test_auth_required(method: str, path: str) -> None:
    http = client()
    kwargs = {"json": BODY} if method == "post" else {}
    assert http.request(method, path, **kwargs).status_code == 401
    headers = {
        "X-Eyvan-User": "bob",
        "X-Eyvan-Roles": "project_viewer",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
    }
    assert http.request(method, path, headers=headers, **kwargs).status_code == 403


def test_public_dashboard_config_describes_local_attachment_capability() -> None:
    response = client().get("/api/v1/dashboard/config")
    assert response.status_code == 200
    assert response.json() == {
        "issuer": None,
        "client_id": None,
        "audience": "eyvan",
        "development_auth": "explicit",
        "local_attachments_available": True,
        "default_org": "default",
        "default_project": "default",
    }
    assert response.json()["local_attachments_available"] is True


@pytest.mark.parametrize("local", [False, True])
def test_nonlocal_or_nondev_refused(local: bool) -> None:
    http = client(local=local, development=False)
    assert http.get("/api/v1/dashboard/config").json()["local_attachments_available"] is False
    response = http.post("/api/v1/attachments/plan", json=BODY, headers={"Authorization": "Bearer test"})
    assert response.status_code == 403
    assert "local loopback development plane" in response.json()["detail"]


@pytest.mark.parametrize(
    "target", ["-oProxyCommand=evil", "host;touch x", "host$(id)", "user@host/path", "", "host\nother"]
)
def test_invalid_target(target: str) -> None:
    assert client().post("/api/v1/attachments/plan", json={**BODY, "target": target}, headers=ADMIN).status_code == 422


@pytest.mark.parametrize(
    "fields",
    [
        {"fleet": "../escape"},
        {"kind": "tpu", "target": "--flag"},
        {"project": "x;evil"},
        {"zone": "-evil"},
        {"token": "secret"},
    ],
)
def test_invalid_options(fields: dict) -> None:
    assert client().post("/api/v1/attachments/plan", json={**BODY, **fields}, headers=ADMIN).status_code == 422


@pytest.mark.parametrize("name", ["team_spot2", "lab_spot_m", "v5p-trainer-1"])
def test_real_tpu_names_with_underscores_are_accepted(name: str) -> None:
    # Real Cloud TPU node names carry underscores; the request model used to 422 them.
    request = attach_job.AttachmentRequest(
        target=name, kind="tpu", project="example-research-project", zone="us-central2-b"
    )
    assert request.target == name
    assert "--tpu" in (argv := attach_job.cli_argv({"request": request.model_dump(), "url": "http://x", "user": "u"}))
    assert argv[argv.index("--tpu") + 1] == name


def test_plan_success_and_explicit_plane_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(PLAN), "Read-only probe\n")

    monkeypatch.setattr(attach_job.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/not/inherited")
    response = client().post("/api/v1/attachments/plan", json=BODY, headers={**ADMIN, "Host": "attacker.example"})
    assert response.status_code == 200
    assert response.json()["fleet"] == "research"
    assert response.json()["summary"] == {
        "shape": "v5p-8",
        "hosts": 1,
        "capacity_class": "spot",
        "endpoint_mode": "tunnel",
    }
    argv, kwargs = calls[0]
    assert argv[argv.index("--url") + 1] == "http://127.0.0.1:8181"
    assert "--dry-run" in argv and "--json" in argv
    assert kwargs["timeout"] == 120
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert "PYTHONPATH" not in kwargs["env"]


@pytest.mark.parametrize(
    "code,out,err,expected",
    [(1, "", "phase\nError: SSH probe refused\n", "SSH probe refused"), (0, "not-json", "", "invalid JSON")],
)
def test_plan_failure(monkeypatch: pytest.MonkeyPatch, code: int, out: str, err: str, expected: str) -> None:
    monkeypatch.setattr(
        attach_job.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, code, out, err)
    )
    response = client().post("/api/v1/attachments/plan", json=BODY, headers=ADMIN)
    assert response.status_code == 502
    assert expected in response.json()["detail"]


def test_plan_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 120, stderr="not exposed")

    monkeypatch.setattr(attach_job.subprocess, "run", timeout)
    response = client().post("/api/v1/attachments/plan", json=BODY, headers=ADMIN)
    assert response.status_code == 504
    assert "120" in response.json()["detail"]


def test_job_survives_fresh_app_and_keeps_owner_only_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spawned = []

    def spawn(argv, **kwargs):
        spawned.append((argv, kwargs))
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(attach_job.subprocess, "Popen", spawn)
    monkeypatch.setenv("EYVAN_TOKEN", "secret-env-not-for-disk")
    response = client().post("/api/v1/attachments", json=BODY, headers=ADMIN)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "running"
    argv, kwargs = spawned[0]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert "secret-env-not-for-disk" not in " ".join(argv)
    directory = tmp_path / ".eyvan" / "attachments" / job["id"]
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(file.stat().st_mode) == 0o600 for file in directory.iterdir())
    assert "secret-env-not-for-disk" not in (directory / "request.json").read_text()

    def run_cli(argv, **kwargs):
        assert "--dry-run" not in argv
        assert "EYVAN_TOKEN" not in kwargs["env"]
        kwargs["stdout"].write(json.dumps(RESULT))
        kwargs["stderr"].write("Installing worker on 16 hosts\nOpening tunnel\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(attach_job.subprocess, "run", run_cli)
    assert attach_job.run_job(directory) == 0
    fresh = client()
    restored = fresh.get(f"/api/v1/attachments/{job['id']}", headers=ADMIN).json()
    assert restored["status"] == "succeeded"
    assert restored["result"] == RESULT
    assert restored["progress"] == ["Installing worker on 16 hosts", "Opening tunnel"]
    assert restored["finished_at"] and restored["started_at"]
    assert fresh.get("/api/v1/attachments", headers=ADMIN).json()["jobs"][0] == restored
    assert "secret-env-not-for-disk" not in json.dumps(restored)


def test_failed_job_reports_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attach_job.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=os.getpid()))
    job = client().post("/api/v1/attachments", json=BODY, headers=ADMIN).json()
    directory = Path.home() / ".eyvan" / "attachments" / job["id"]

    def run(argv, **kwargs):
        kwargs["stderr"].write("Error: Opening an SSH tunnel failed\n")
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(attach_job.subprocess, "run", run)
    assert attach_job.run_job(directory) == 1
    restored = client().get(f"/api/v1/attachments/{job['id']}", headers=ADMIN).json()
    assert restored["status"] == "failed"
    assert restored["error"] == "Opening an SSH tunnel failed"
    assert restored["result"] is None


def _discovery_client(tmp_path: Path, snapshot=None) -> TestClient:
    from eyvan.capacity.providers.gcp.tpu_discovery import DiscoveryIndex

    http = client()
    index = DiscoveryIndex(tmp_path / "discovered.json")
    if snapshot is not None:
        index.save(snapshot)
    http.app.state.discovery_index = index
    # The route must never reach GCP from a test; the periodic refresher owns that.
    http.app.state.discovery_refresh = False
    return http


def test_discovered_tpus_before_the_first_listing(tmp_path: Path) -> None:
    response = _discovery_client(tmp_path).get("/api/v1/attachments/discovered", headers=ADMIN)
    assert response.status_code == 200
    assert response.json() == {"discovered_at": None, "tpus": [], "skipped": {}, "projects": []}


def test_discovered_tpus_mark_connected_fleets_and_keep_skips(tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.tpu_discovery import DiscoveredTpu, DiscoverySnapshot

    tpu = DiscoveredTpu(
        fleet="tpu-train-a-1a2b3c4d",
        project="demo-research",
        zone="us-central1-a",
        name="train-a",
        node_id="1",
        accelerator_type="v5p-8",
        capacity_class="on_demand",
        state="READY",
        hosts=1,
    )
    snapshot = DiscoverySnapshot(
        discovered_at=1000.0, tpus=(tpu,), skipped={"locked": "no permission to list TPU nodes"}, projects=("a",)
    )
    body = _discovery_client(tmp_path, snapshot).get("/api/v1/attachments/discovered", headers=ADMIN).json()
    assert body["discovered_at"] == 1000.0
    assert body["skipped"] == {"locked": "no permission to list TPU nodes"}
    assert [(t["fleet"], t["connected"]) for t in body["tpus"]] == [("tpu-train-a-1a2b3c4d", False)]
    assert "version" not in body


def test_missing_job_and_path_traversal() -> None:
    assert client().get("/api/v1/attachments/" + "b" * 32, headers=ADMIN).status_code == 404
    assert client().get("/api/v1/attachments/not-an-id", headers=ADMIN).status_code == 422
