# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cold credential failures must reach quota fallback and the unavailable panel."""

from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from eyvan.capacity.providers.gcp.quota_discovery import GcpTpuQuotaDiscovery
from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

_SECRET = "synthetic-bearer-secret-do-not-report"
_FAILURES = ("missing-binary", "no-active-account", "timeout")


def _offline_gcloud(monkeypatch: pytest.MonkeyPatch, failure: str, *, fallback_succeeds: bool) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert command[0] == "gcloud"
        is_token = command[1:] == ["auth", "print-access-token"]
        if not is_token:
            assert command[1:5] == ["beta", "quotas", "info", "list"]
            if fallback_succeeds:
                return subprocess.CompletedProcess(command, 0, "[]", "")
        if failure == "missing-binary":
            raise FileNotFoundError(2, "synthetic missing gcloud")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=_SECRET)
        assert failure == "no-active-account"
        return subprocess.CompletedProcess(command, 1, _SECRET, "no active account selected")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize("failure", _FAILURES)
def test_cold_quota_token_failure_reaches_successful_cli_fallback(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    calls = _offline_gcloud(monkeypatch, failure, fallback_succeeds=True)

    # Use the production default catalog, token helper, and subprocess runner.
    rows = GcpTpuQuotaDiscovery().discover(project="project-a")

    assert rows == ()  # The successful CLI response really was an empty quota list.
    assert [command[1:5] for command in calls] == [
        ["auth", "print-access-token"],
        ["beta", "quotas", "info", "list"],
    ]


@pytest.mark.parametrize("failure", _FAILURES)
def test_default_quota_endpoint_reports_unavailable_when_both_paths_fail(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    calls = _offline_gcloud(monkeypatch, failure, fallback_succeeds=False)

    # No injected discovery or converted exception: exercise the default reader
    # through both cached purchase-class reads and the real HTTP route.
    with TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True))) as client:
        response = client.get(
            "/api/v1/capacity/quota",
            params={"project": "project-a"},
            headers={
                "X-Eyvan-User": "alice",
                "X-Eyvan-Org": "acme",
                "X-Eyvan-Project": "research",
                "X-Eyvan-Roles": "fleet_admin",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["entries"] == []
    assert {item["capacity_class"] for item in payload["failures"]} == {"spot", "on_demand"}
    assert all("gcloud beta quotas info list failed:" in item["reason"] for item in payload["failures"])
    assert _SECRET not in response.text
    assert sum(command[1:] == ["auth", "print-access-token"] for command in calls) == 2
    assert sum(command[1:5] == ["beta", "quotas", "info", "list"] for command in calls) == 2
