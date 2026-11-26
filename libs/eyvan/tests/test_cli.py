# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import typing as tp
from datetime import datetime
from pathlib import Path
from unittest import mock

import httpx
import pytest
import yaml
from click.testing import CliRunner
from fastapi.testclient import TestClient

import eyvan.sdk.cli as cli
import eyvan.sdk.client.identity as identity_module
import eyvan.sdk.client.transport as client_transport
import eyvan.sdk.commands.autoscale as autoscale_commands
import eyvan.sdk.commands.fleet_status as fleet_status_rendering
import eyvan.sdk.commands.fleets as fleet_commands
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, is_json_value
from eyvan.capacity.autoscale import (
    DEFAULT_TPU_PROFILES,
    AutoscaleConfigCache,
    AutoscaleDefaults,
    AutoscaleRequest,
    AutoscaleZoneRequest,
    QuotaSnapshot,
    generate_autoscale_config,
    generate_zone_autoscale_config,
    tpu_profile_from_accelerator_type,
)
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.capacity.providers.gcp.quota_discovery import TpuQuotaAvailability
from eyvan.config import EyvanSettings
from eyvan.records import CapacityClass, CapacityPoolRecord, CellRecord, CellState, utcnow
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands import workloads as workload_commands
from eyvan.sdk.contexts import get_active_context
from eyvan.server.api.app import create_app
from eyvan.server.api.schemas import (
    FleetApplyRequest,
    FleetStatusResponse,
    PoolWorkloadSubmitRequest,
    WorkloadSubmitRequest,
)
from eyvan.server.local.auth import LocalDevelopmentCredentialStore
from eyvan.server.local.daemon import LocalDaemonResult, LocalDaemonState

QueryValue: tp.TypeAlias = str | int | float | bool | None


class RequestArguments(tp.TypedDict, total=False):
    json: JsonObject
    headers: dict[str, str]
    params: dict[str, QueryValue]


FLEET_YAML = """\
version: 1
name: dev-fleet
provider:
  type: gcp
  project: gcp-project
  region: us-central1
  zone: us-central1-a
agent_endpoint: https://agent.example.com
node_types:
  cpu-workers:
    kind: cpu
    machine_type: n2-standard-8
    resources:
      cpu: 8
      memory_gib: 32
    min_workers: 1
    max_workers: 4
    provider_options:
      service_account: eyvan-worker@gcp-project.iam.gserviceaccount.com
      bootstrap_egress: private-google-access
  gpu-workers:
    kind: gpu
    machine_type: g2-standard-8
    accelerator_type: nvidia-l4
    resources:
      cpu: 8
      memory_gib: 32
      gpu: 1
    min_workers: 0
    initial_workers: 1
    max_workers: 8
    provider_options:
      service_account: eyvan-worker@gcp-project.iam.gserviceaccount.com
      bootstrap_egress: private-google-access
"""


class FakeClient:
    def __init__(self, status: JsonObject | None = None, *, fleets: tuple[str, ...] = ("dev-fleet",)):
        self.status = status or {"name": "dev-fleet", "pools": [], "cells": []}
        self.fleets = fleets
        self.calls: list[tuple[str, str, RequestArguments]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, QueryValue] | None = None,
        json: JsonObject | None = None,
        headers: dict[str, str] | None = None,
    ) -> JsonValue:
        arguments: RequestArguments = {}
        if params is not None:
            arguments["params"] = params
        if json is not None:
            arguments["json"] = json
        if headers is not None:
            arguments["headers"] = headers
        self.calls.append((method, path, arguments))
        if method == "PUT" and path.startswith("/api/v1/fleets/"):
            if json is None:
                raise AssertionError("fleet apply requires JSON")
            return json["pools"]
        if method == "GET" and path == "/api/v1/fleets":
            return list(self.fleets)
        if method == "GET" and path == "/api/v1/fleets/dev-fleet":
            return self.status
        if method == "GET" and path == "/api/v1/workloads":
            return []
        if method == "GET" and path.startswith("/api/v1/workloads/"):
            return {"id": path.rsplit("/", 1)[-1], "state": "pending"}
        if method == "DELETE" and path == "/api/v1/fleets/dev-fleet":
            return [{"id": "dev-fleet-cpu-workers", "enabled": False}]
        if method == "POST" and path == "/api/v1/workloads":
            if json is None:
                raise AssertionError("workload submit requires JSON")
            return json
        if method == "POST" and path == "/api/v1/workloads/estimate":
            if json is None:
                raise AssertionError("workload estimate requires JSON")
            return {
                "org_id": json["org_id"],
                "project_id": json["project_id"],
                "estimated_duration_seconds": 3600,
                "candidates": [],
            }
        if method == "POST" and path == "/api/v1/batches":
            if json is None:
                raise AssertionError("batch submit requires JSON")
            count = json["count"]
            assert isinstance(count, int)
            return {
                "parent": {"id": "wl-parent", "state": "pending", "schedulable": False},
                "tasks": [{"id": f"wl-task-{index}", "state": "pending"} for index in range(count)],
            }
        if method == "POST" and path == "/api/v1/dags":
            if json is None:
                raise AssertionError("dag submit requires JSON")
            nodes = json["nodes"]
            assert isinstance(nodes, list)
            return {
                "parent": {"id": "wl-dag", "state": "pending", "schedulable": False},
                "nodes": [{"id": f"wl-node-{index}", "state": "pending"} for index in range(len(nodes))],
            }
        if method == "POST" and path.startswith("/api/v1/fleets/") and path.endswith("/workloads"):
            if json is None:
                raise AssertionError("pool workload submit requires JSON")
            return json
        raise AssertionError(f"unexpected request: {method} {path}")

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        """Record refreshed headers without performing real HTTP I/O."""

        self.identity_headers = dict(headers)


@pytest.fixture(autouse=True)
def _isolated_context_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "eyvan-config.json"))
    credential_path = tmp_path / "local-development-credential"

    def credential_store(_path: Path | str) -> LocalDevelopmentCredentialStore:
        return LocalDevelopmentCredentialStore(credential_path)

    monkeypatch.setattr(identity_module, "LocalDevelopmentCredentialStore", credential_store)

    def unavailable_daemon(endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.SKIPPED,
            pid=None,
            log_path=None,
        )

    monkeypatch.setattr(fleet_commands, "ensure_loopback_control_plane", unavailable_daemon)
    monkeypatch.setattr(fleet_commands, "has_gcp_native_runtime_profile", lambda: False)
    # ensure_local_runtime also asks whether a plane already serves a runtime
    # profile, which reads this user's real ~/.eyvan daemon records. A plane
    # running on the developer's machine must not change which branch these
    # tests exercise.
    monkeypatch.setattr(fleet_commands, "serving_runtime_profile_path", lambda *_args, **_kwargs: None)

    class ReadyNativeGcpRuntime:
        """Keep generic CLI tests independent from a private runtime profile."""

        @classmethod
        def for_endpoint(cls, endpoint: str | ControlPlaneEndpoint) -> ReadyNativeGcpRuntime:
            del endpoint
            return cls()

        def ensure(self, endpoint: str | ControlPlaneEndpoint) -> None:
            del endpoint

    monkeypatch.setattr(fleet_commands, "GcpNativeRuntimeSupervisor", ReadyNativeGcpRuntime)


def test_cli_help_imports_in_a_clean_process() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "eyvan.sdk.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Commands:" in result.stdout
    assert "autoscale" in result.stdout


def _invoke(monkeypatch, tmp_path, fake: FakeClient, *args: str):
    config = tmp_path / "fleet.yaml"
    config.write_text(FLEET_YAML)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    return CliRunner().invoke(cli.main, ["--user", "alice", "--role", "fleet_admin", *args, str(config)])


def _select_plain_fleet_status_rendering(monkeypatch) -> None:
    """Exercise the human view without inheriting terminal color from the test runner."""

    console_factory = fleet_status_rendering.fleet_status_console
    monkeypatch.setattr(fleet_status_rendering, "is_interactive_output", lambda: True)
    monkeypatch.setattr(
        fleet_status_rendering,
        "fleet_status_console",
        lambda: console_factory(force_terminal=False, environment={}),
    )


def _json_argument(call: tuple[str, str, RequestArguments]) -> JsonObject:
    payload = call[2].get("json")
    if payload is None:
        raise AssertionError("request has no JSON body")
    return payload


def _header_arguments(call: tuple[str, str, RequestArguments]) -> dict[str, str]:
    headers = call[2].get("headers")
    if headers is None:
        raise AssertionError("request has no headers")
    return headers


def _save_autoscale_defaults(cache_dir: Path) -> AutoscaleConfigCache:
    cache = AutoscaleConfigCache(cache_dir)
    cache.save_defaults(
        AutoscaleDefaults(
            project="project-a",
            fleet_prefix="research",
        )
    )
    return cache


def test_use_persists_active_context_and_context_displays_it(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    runner = CliRunner()

    selected = runner.invoke(
        cli.main,
        [
            "--url",
            "https://eyvan.example.com/control/",
            "use",
            "research",
            "--org",
            "acme",
            "--project",
            "training",
            "--json",
        ],
    )
    shown = runner.invoke(cli.main, ["context", "--json"])

    expected = {
        "active_context": "research",
        "base_url": "https://eyvan.example.com/control",
        "fleet": "research",
        "org_id": "acme",
        "project_id": "training",
    }
    assert selected.exit_code == 0, selected.output
    assert shown.exit_code == 0, shown.output
    assert json.loads(selected.output) == expected
    assert json.loads(shown.output) == expected
    config_path = Path(os.environ["EYVAN_CONFIG_FILE"])
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert "token" not in config_path.read_text(encoding="utf-8")


def test_use_applies_documented_scope_defaults(monkeypatch) -> None:
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())

    selected = CliRunner().invoke(cli.main, ["use", "research", "--json"])

    assert selected.exit_code == 0, selected.output
    assert json.loads(selected.output) == {
        "active_context": "research",
        "base_url": "http://127.0.0.1:8080",
        "fleet": "research",
        "org_id": "default",
        "project_id": "default",
    }


def test_active_context_supplies_run_defaults_and_environment_overrides_them(monkeypatch) -> None:
    fake = FakeClient()
    constructed: list[tuple[str, dict[str, str]]] = []

    def client(base_url: str, *, headers: dict[str, str]) -> FakeClient:
        constructed.append((base_url, headers))
        return fake

    monkeypatch.setattr(cli, "EyvanClient", client)
    runner = CliRunner()
    selected = runner.invoke(
        cli.main,
        [
            "--url",
            "https://stored.example.com",
            "use",
            "dev-fleet",
            "--org",
            "stored-org",
            "--project",
            "stored-project",
        ],
    )
    assert selected.exit_code == 0, selected.output

    from_context = runner.invoke(
        cli.main,
        ["--user", "alice", "run", "--no-wait", "dev-fleet/cpu-workers", "--", "python", "train.py"],
    )
    from_environment = runner.invoke(
        cli.main,
        ["--user", "alice", "run", "--no-wait", "dev-fleet/cpu-workers", "--", "python", "eval.py"],
        env={
            "EYVAN_URL": "https://override.example.com",
            "EYVAN_ORG": "override-org",
            "EYVAN_PROJECT": "override-project",
        },
    )

    assert from_context.exit_code == 0, from_context.output
    assert from_environment.exit_code == 0, from_environment.output
    assert fake.calls[-2][2]["json"]["org_id"] == "stored-org"
    assert fake.calls[-2][2]["json"]["project_id"] == "stored-project"
    assert fake.calls[-1][2]["json"]["org_id"] == "override-org"
    assert fake.calls[-1][2]["json"]["project_id"] == "override-project"
    assert constructed[-2][0] == "https://stored.example.com"
    assert constructed[-1][0] == "https://override.example.com"
    assert constructed[-1][1]["X-Eyvan-Org"] == "override-org"


def test_loopback_control_plane_uses_local_development_identity_by_default(monkeypatch) -> None:
    fake = FakeClient()
    constructed: list[tuple[str, dict[str, str]]] = []

    def client(base_url: str, *, headers: dict[str, str]) -> FakeClient:
        constructed.append((base_url, headers))
        return fake

    monkeypatch.delenv("EYVAN_TOKEN", raising=False)
    monkeypatch.delenv("EYVAN_DEV_USER", raising=False)
    monkeypatch.setattr(identity_module.getpass, "getuser", lambda: "erfan")
    monkeypatch.setattr(cli, "EyvanClient", client)

    result = CliRunner().invoke(cli.main, ["status", "dev-fleet"])

    assert result.exit_code == 0, result.output
    assert constructed == [
        (
            "http://127.0.0.1:8080",
            {"X-Eyvan-Roles": "fleet_admin", "X-Eyvan-User": "erfan"},
        )
    ]


def test_the_development_identity_notice_names_the_plane_the_role_and_the_way_out(monkeypatch) -> None:
    """It prints above every local command, so it has to earn its line.

    Three things make it worth keeping: which plane answered (two development
    planes on one machine differ only by port), what rights the command ran
    with, and the flag that changes them. An earlier version named none of the
    first and buried the third behind "override it".
    """

    fake = FakeClient()
    monkeypatch.delenv("EYVAN_TOKEN", raising=False)
    monkeypatch.delenv("EYVAN_DEV_USER", raising=False)
    monkeypatch.setattr(cli, "EyvanClient", lambda base_url, *, headers: fake)
    monkeypatch.setattr(cli, "stream_is_terminal", lambda _stream: True)

    result = CliRunner().invoke(cli.main, ["--url", "http://127.0.0.1:8081", "status", "dev-fleet"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8081" in result.output
    assert "fleet_admin" in result.output
    assert "--role" in result.output
    # One line, not a paragraph. It is chrome above real output, and a notice
    # that wraps is one people stop reading.
    notice = next(line for line in result.output.splitlines() if "fleet_admin" in line)
    assert len(notice) <= 100


def test_the_development_identity_notice_stays_off_a_pipe(monkeypatch) -> None:
    """Redirected output is somebody's script reading it, not somebody reading it."""

    fake = FakeClient()
    monkeypatch.delenv("EYVAN_TOKEN", raising=False)
    monkeypatch.delenv("EYVAN_DEV_USER", raising=False)
    monkeypatch.setattr(cli, "EyvanClient", lambda base_url, *, headers: fake)
    monkeypatch.setattr(cli, "stream_is_terminal", lambda _stream: False)

    result = CliRunner().invoke(cli.main, ["status", "dev-fleet"])

    assert result.exit_code == 0, result.output
    assert "fleet_admin" not in result.output


def test_remote_control_plane_never_receives_synthesized_development_identity(monkeypatch) -> None:
    fake = FakeClient()
    constructed: list[tuple[str, dict[str, str]]] = []

    def client(base_url: str, *, headers: dict[str, str]) -> FakeClient:
        constructed.append((base_url, headers))
        return fake

    monkeypatch.delenv("EYVAN_TOKEN", raising=False)
    monkeypatch.delenv("EYVAN_DEV_USER", raising=False)
    monkeypatch.setattr(identity_module.getpass, "getuser", lambda: "erfan")
    monkeypatch.setattr(cli, "EyvanClient", client)

    result = CliRunner().invoke(
        cli.main,
        ["--url", "https://eyvan.example.com", "status", "dev-fleet"],
    )

    assert result.exit_code == 0, result.output
    assert constructed == [("https://eyvan.example.com", {})]


def test_bearer_token_takes_precedence_over_loopback_development_identity(monkeypatch) -> None:
    fake = FakeClient()
    constructed: list[tuple[str, dict[str, str]]] = []

    def client(base_url: str, *, headers: dict[str, str]) -> FakeClient:
        constructed.append((base_url, headers))
        return fake

    monkeypatch.setattr(identity_module.getpass, "getuser", lambda: "erfan")
    monkeypatch.setattr(cli, "EyvanClient", client)

    result = CliRunner().invoke(
        cli.main,
        ["--token", "test-token", "status", "dev-fleet"],
    )

    assert result.exit_code == 0, result.output
    assert constructed == [("http://127.0.0.1:8080", {"Authorization": "Bearer test-token"})]


@pytest.mark.parametrize(
    ("base_url", "expected_trust_env"),
    (("http://localhost:8080", False), ("http://127.0.0.2:8080", False), ("https://eyvan.example.com", True)),
)
def test_api_client_never_uses_environment_proxies_for_loopback(
    monkeypatch, base_url: str, expected_trust_env: bool
) -> None:
    constructed: list[bool] = []
    httpx_client = httpx.Client

    def client(
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        trust_env: bool,
    ) -> httpx.Client:
        constructed.append(trust_env)
        return httpx_client(base_url=base_url, headers=headers, timeout=timeout, trust_env=trust_env)

    monkeypatch.setattr(client_transport.httpx, "Client", client)

    cli.EyvanClient(base_url, headers={"X-Eyvan-User": "erfan"})

    assert constructed == [expected_trust_env]


def test_cli_reports_an_unavailable_control_plane_without_a_transport_traceback(monkeypatch) -> None:
    def fail_request(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    raw_client = httpx.Client(
        base_url="http://127.0.0.1:8080",
        transport=httpx.MockTransport(fail_request),
    )
    monkeypatch.setattr(client_transport.httpx, "Client", lambda **_kwargs: raw_client)

    result = CliRunner().invoke(cli.main, ["cells"])

    assert result.exit_code != 0
    assert "Eyvan control plane is unavailable at http://127.0.0.1:8080." in result.output
    assert "eyvan-server --log-level info" in result.output
    assert "Traceback" not in result.output


def test_cli_client_maps_a_non_json_success_body_to_a_typed_api_error(monkeypatch) -> None:
    def non_json_response(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy   maintenance\npage</html>")

    raw_client = httpx.Client(
        base_url="http://127.0.0.1:8080",
        transport=httpx.MockTransport(non_json_response),
    )
    monkeypatch.setattr(client_transport.httpx, "Client", lambda **_kwargs: raw_client)
    client = cli.EyvanClient("http://127.0.0.1:8080", headers={})

    with pytest.raises(cli.EyvanApiError) as raised:
        client.request("GET", "/api/v1/fleets")

    assert raised.value.status_code == 200
    assert "non-JSON response body" in raised.value.message
    assert "proxy maintenance page" in raised.value.message


def test_context_without_active_selection_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())

    result = CliRunner().invoke(cli.main, ["context"])

    assert result.exit_code != 0
    assert "eyvan use FLEET" in result.output


def test_use_switches_to_an_existing_named_context_without_overwriting_it(monkeypatch) -> None:
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())
    runner = CliRunner()

    research = runner.invoke(
        cli.main,
        ["--url", "https://research.example.com", "use", "research", "--org", "acme", "--project", "training"],
    )
    serving = runner.invoke(
        cli.main,
        ["--url", "https://serving.example.com", "use", "serving", "--org", "prod", "--project", "inference"],
    )
    selected = runner.invoke(cli.main, ["use", "research", "--json"])

    assert research.exit_code == 0, research.output
    assert serving.exit_code == 0, serving.output
    assert selected.exit_code == 0, selected.output
    assert json.loads(selected.output) == {
        "active_context": "research",
        "base_url": "https://research.example.com",
        "fleet": "research",
        "org_id": "acme",
        "project_id": "training",
    }


def test_applying_a_fleet_does_not_borrow_the_active_fleets_endpoint(monkeypatch, tmp_path) -> None:
    """`up` overwrote the applied fleet's saved URL with whichever one was active.

    `activate_fleet` took the *session* URL unconditionally while its sibling
    `select_context` preferred the fleet's own saved value -- opposite precedence,
    same file. With another fleet selected, applying `dev-fleet` repointed it at
    the selected fleet's address, which for a head is a private VPC IP only the
    VPC can route. Nothing said the endpoint had moved.
    """

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())
    runner = CliRunner()

    # dev-fleet is reachable at its own address...
    saved = runner.invoke(
        cli.main, ["--url", "https://dev.example.com", "use", "dev-fleet", "--org", "acme", "--project", "training"]
    )
    # ...and something else is the active context when `up` runs.
    other = runner.invoke(
        cli.main, ["--url", "http://10.128.0.7:8080", "use", "other-fleet", "--org", "acme", "--project", "training"]
    )
    assert saved.exit_code == 0, saved.output
    assert other.exit_code == 0, other.output

    fake = FakeClient()
    applied = _invoke(monkeypatch, tmp_path, fake, "up")
    assert applied.exit_code == 1, applied.output
    assert "different saved control-plane endpoint" in applied.output
    assert fake.calls == [], "an implicit apply must not write to the active fleet's unrelated plane"

    stored = json.loads((tmp_path / "eyvan-config.json").read_text())

    assert stored["contexts"]["dev-fleet"]["base_url"] == "https://dev.example.com"


def test_up_follows_a_fleet_to_its_own_local_plane_and_ensures_it(monkeypatch, tmp_path) -> None:
    """Two loopback planes differ only by port; `up` must not make the operator run `eyvan use`."""

    fake = FakeClient()
    client_urls: list[str] = []

    def client(url: str, **_kwargs: object) -> FakeClient:
        client_urls.append(ControlPlaneEndpoint.parse(url).url)
        return fake

    monkeypatch.setattr(cli, "EyvanClient", client)
    runner = CliRunner()
    saved = runner.invoke(cli.main, ["--url", "http://127.0.0.1:8080", "use", "dev-fleet"])
    other = runner.invoke(cli.main, ["--url", "http://127.0.0.1:18280", "use", "other-fleet"])
    assert saved.exit_code == 0, saved.output
    assert other.exit_code == 0, other.output

    ensured: list[ControlPlaneEndpoint] = []

    def ensure(endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        ensured.append(configured_endpoint)
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.STARTED,
            pid=1234,
            log_path=tmp_path / "control-plane.log",
        )

    monkeypatch.setattr(fleet_commands, "ensure_loopback_control_plane", ensure)
    client_urls.clear()
    config = tmp_path / "fleet.yaml"
    config.write_text(FLEET_YAML)
    applied = CliRunner().invoke(cli.main, ["--user", "alice", "--role", "fleet_admin", "up", str(config)])

    assert applied.exit_code == 0, applied.output
    assert "lives on local plane http://127.0.0.1:8080" in applied.output
    assert ensured == [ControlPlaneEndpoint.parse("http://127.0.0.1:8080")]
    assert client_urls[-1] == ControlPlaneEndpoint.parse("http://127.0.0.1:8080").url
    assert fake.calls[0][:2] == ("PUT", "/api/v1/fleets/dev-fleet")
    stored = json.loads((tmp_path / "eyvan-config.json").read_text())
    assert stored["active_context"] == "dev-fleet"
    assert stored["contexts"]["dev-fleet"]["base_url"] == ControlPlaneEndpoint.parse("http://127.0.0.1:8080").url


def test_up_still_refuses_a_loopback_session_for_a_fleet_saved_on_a_remote_plane(monkeypatch, tmp_path) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    runner = CliRunner()
    assert runner.invoke(cli.main, ["--url", "https://dev.example.com", "use", "dev-fleet"]).exit_code == 0
    assert runner.invoke(cli.main, ["--url", "http://127.0.0.1:18280", "use", "other-fleet"]).exit_code == 0

    applied = _invoke(monkeypatch, tmp_path, fake, "up")

    assert applied.exit_code == 1, applied.output
    assert "different saved control-plane endpoint" in applied.output
    assert fake.calls == []


def test_up_applies_all_initial_cells_with_distinct_operation_keys(monkeypatch, tmp_path) -> None:
    fake = FakeClient()

    first = _invoke(monkeypatch, tmp_path, fake, "up")
    second = _invoke(monkeypatch, tmp_path, fake, "up")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    payload = json.loads(first.output)
    assert [pool["node_type"] for pool in payload] == ["cpu-workers", "gpu-workers"]
    puts = [call for call in fake.calls if call[:2] == ("PUT", "/api/v1/fleets/dev-fleet")]
    assert len(puts) == 2
    apply_request = FleetApplyRequest.model_validate(_json_argument(puts[0]))
    assert apply_request.pools[0].capacity.cpu == 8
    assert apply_request.pools[1].capacity.gpu == 1
    assert _header_arguments(puts[0])["Idempotency-Key"] != _header_arguments(puts[1])["Idempotency-Key"]


def test_up_accepts_explicit_idempotency_key_for_an_ambiguous_retry(monkeypatch, tmp_path) -> None:
    fake = FakeClient()

    result = _invoke(monkeypatch, tmp_path, fake, "up", "--idempotency-key", "retry-fleet-apply-1")

    assert result.exit_code == 0, result.output
    assert _header_arguments(fake.calls[0])["Idempotency-Key"] == "retry-fleet-apply-1"


def test_up_ensures_the_configured_loopback_control_plane_before_fleet_apply(monkeypatch, tmp_path) -> None:
    fake = FakeClient()
    ensured: list[ControlPlaneEndpoint] = []

    def ensure(endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        ensured.append(configured_endpoint)
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.STARTED,
            pid=1234,
            log_path=tmp_path / "control-plane.log",
        )

    monkeypatch.setattr(fleet_commands, "ensure_loopback_control_plane", ensure)

    result = _invoke(monkeypatch, tmp_path, fake, "up")

    assert result.exit_code == 0, result.output
    assert ensured == [ControlPlaneEndpoint.parse("http://127.0.0.1:8080")]
    assert fake.calls[0][:2] == ("PUT", "/api/v1/fleets/dev-fleet")


def test_up_does_not_supervise_a_remote_control_plane(monkeypatch, tmp_path) -> None:
    fake = FakeClient()
    ensured: list[ControlPlaneEndpoint] = []
    config_path = tmp_path / "fleet.yaml"
    config_path.write_text(FLEET_YAML, encoding="utf-8")

    def ensure(endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        ensured.append(configured_endpoint)
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.SKIPPED,
            pid=None,
            log_path=None,
        )

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(fleet_commands, "ensure_loopback_control_plane", ensure)

    result = CliRunner().invoke(
        cli.main,
        [
            "--url",
            "https://eyvan.example.com",
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert ensured == []
    assert fake.calls[0][:2] == ("PUT", "/api/v1/fleets/dev-fleet")


def test_up_dry_run_and_config_validate_do_not_call_api(monkeypatch, tmp_path) -> None:
    fake = FakeClient()

    dry_run = _invoke(monkeypatch, tmp_path, fake, "up", "--dry-run", "--json")
    validated = _invoke(monkeypatch, tmp_path, fake, "config", "validate")

    assert dry_run.exit_code == 0, dry_run.output
    assert validated.exit_code == 0, validated.output
    assert len(json.loads(dry_run.output)["desired_initial_cells"]) == 2
    assert json.loads(validated.output)["valid"] is True
    assert fake.calls == []


@pytest.mark.parametrize(
    "autoscale_args",
    [
        ("--quota-based",),
        ("--select", "research-v5p-16-us-east5-a"),
        ("--zone", "us-east5-a"),
        ("--tpu-generation", "v5p"),
        ("--defaults", "autoscale-defaults.yaml"),
        ("--cache-dir", "autoscale"),
    ],
)
def test_up_rejects_autoscale_options_with_an_explicit_config(monkeypatch, tmp_path, autoscale_args) -> None:
    fake = FakeClient()

    result = _invoke(monkeypatch, tmp_path, fake, "up", *autoscale_args)

    assert result.exit_code != 0
    assert "requires omitting CONFIG_PATH" in result.output
    assert fake.calls == []


def test_autoscale_generate_discovers_dynamic_types_and_caches_one_zone_fleet(monkeypatch, tmp_path) -> None:
    class Discovery:
        def discover(self, **kwargs):
            assert kwargs["accelerator_types"] == ()
            assert kwargs["zones"] == ("us-central1-a",)
            assert kwargs["generations"] == ()
            assert kwargs["preemptible"] is True
            return (
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v5p-16",
                    generation="v5p",
                    quota_tpus=1024,
                    available_tpus=1024,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v5p-8",
                    generation="v5p",
                    quota_tpus=1024,
                    available_tpus=1024,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v5p-512",
                    generation="v5p",
                    quota_tpus=1024,
                    available_tpus=1024,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v6e-64",
                    generation="v6e",
                    quota_tpus=512,
                    available_tpus=512,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v6e-preemptible",
                ),
            )

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)
    cache_dir = tmp_path / "autoscale"
    _save_autoscale_defaults(cache_dir)

    result = CliRunner().invoke(
        cli.main,
        [
            "autoscale",
            "generate",
            "--zone",
            "us-central1-a",
            "--quota-based",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [entry["fleet"] for entry in payload["generated"]] == ["research-us-central1-a"]
    # Every discovered type, not the two the catalog once had rows for.
    assert sorted(pool["accelerator_type"] for pool in payload["generated"][0]["pools"]) == [
        "v5p-16",
        "v5p-512",
        "v6e-64",
    ]
    # `shared` is now true for v5p: two pools draw on the one generation quota,
    # which is exactly what it reports and what it could not report while only
    # one v5p type per zone was ever generated.
    assert payload["generated"][0]["generation_quotas"] == [
        {
            "available_tpus": 1024,
            "generation": "v5p",
            "preemptible": True,
            "quota_tpus": 1024,
            "shared": True,
            "source_metric": "tpu.googleapis.com/tpu-v5p-preemptible",
            "zone": "us-central1-a",
        },
        {
            "available_tpus": 512,
            "generation": "v6e",
            "preemptible": True,
            "quota_tpus": 512,
            "shared": False,
            "source_metric": "tpu.googleapis.com/tpu-v6e-preemptible",
            "zone": "us-central1-a",
        },
    ]
    assert "quota" not in payload["generated"][0]
    # Only the deliberate exclusion remains. v5p-512 and v6e-64 used to be here
    # too, refused for having no catalog row of their own.
    assert payload["skipped"] == [
        {
            "generation": "v5p",
            "reason": "compute-backed TPU VM is excluded from queued-resource autoscale",
            "scope": "generation",
            "type_count": 1,
            "zone": "us-central1-a",
        },
    ]
    entries = AutoscaleConfigCache(cache_dir).list()
    assert [entry.config.name for entry in entries] == ["research-us-central1-a"]
    assert all(pool.capacity_class is CapacityClass.SPOT for pool in entries[0].config.pools.values())


def test_autoscale_generate_renders_a_compact_terminal_summary(monkeypatch, tmp_path) -> None:
    class Discovery:
        def discover(self, **_kwargs):
            return (
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v5p-16",
                    generation="v5p",
                    quota_tpus=1024,
                    available_tpus=1024,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v5p-512",
                    generation="v5p",
                    quota_tpus=1024,
                    available_tpus=1024,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v5p-8",
                    generation="v5p",
                    quota_tpus=1024,
                    available_tpus=1024,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
                TpuQuotaAvailability(
                    zone="us-central1-a",
                    accelerator_type="v6e-16",
                    generation="v6e",
                    quota_tpus=512,
                    available_tpus=512,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v6e-preemptible",
                ),
            )

    cache_dir = tmp_path / "autoscale"
    _save_autoscale_defaults(cache_dir)
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)
    monkeypatch.setattr(autoscale_commands, "is_interactive_output", lambda: True)

    result = CliRunner().invoke(
        cli.main,
        ["autoscale", "generate", "--zone", "us-central1-a", "--cache-dir", str(cache_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Eyvan TPU Autoscale" in result.output
    assert "us-central1-a" in result.output
    assert "1024 available" in result.output
    assert "512 available" in result.output
    assert "research-us-central1-a.yaml" in result.output
    assert "Skipped candidates" in result.output
    assert "1 type across 1 zone" in result.output
    assert "compute-backed TPU VM is excluded" in result.output
    assert '"generated"' not in result.output
    assert '"accelerator_type"' not in result.output


def test_autoscale_generate_json_forces_machine_readable_output(monkeypatch, tmp_path) -> None:
    class Discovery:
        def discover(self, **_kwargs):
            return (
                TpuQuotaAvailability(
                    zone="us-east5-a",
                    accelerator_type="v5p-16",
                    generation="v5p",
                    quota_tpus=32,
                    available_tpus=32,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
            )

    cache_dir = tmp_path / "autoscale"
    _save_autoscale_defaults(cache_dir)
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)
    monkeypatch.setattr(autoscale_commands, "is_interactive_output", lambda: True)

    result = CliRunner().invoke(
        cli.main,
        ["autoscale", "generate", "--json", "--zone", "us-east5-a", "--cache-dir", str(cache_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [entry["fleet"] for entry in payload["generated"]] == ["research-us-east5-a"]
    assert "Eyvan TPU Autoscale" not in result.output


def test_autoscale_init_prompts_once_and_persists_a_yaml_profile(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "autoscale"
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(autoscale_commands, "gcloud_default_project", lambda: "project-a")

    result = CliRunner().invoke(
        cli.main,
        ["autoscale", "init", "--cache-dir", str(cache_dir)],
        input="\nresearch\n\n2\n4\ny\n",
    )

    assert result.exit_code == 0, result.output
    defaults = AutoscaleConfigCache(cache_dir).load_defaults()
    assert defaults == AutoscaleDefaults(
        project="project-a",
        fleet_prefix="research",
        min_workers=2,
        max_workers=4,
    )
    assert "defaults.yaml" in result.output
    assert "Eyvan agent endpoint" not in result.output
    assert "Worker service account" not in result.output
    assert "Review autoscale defaults" in result.output
    assert '"defaults"' not in result.output


def test_autoscale_init_rejects_a_gcs_uri_and_reprompts_for_project_id(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "autoscale"
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(autoscale_commands, "gcloud_default_project", lambda: "example-project-1")

    result = CliRunner().invoke(
        cli.main,
        ["autoscale", "init", "--cache-dir", str(cache_dir)],
        input="gs://example-bucket/eyvan\nexample-project-1\nresearch\n\n0\n0\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert "not bucket URI" in result.output
    assert AutoscaleConfigCache(cache_dir).load_defaults().project == "example-project-1"


def test_autoscale_init_repairs_an_invalid_profile_written_by_the_old_wizard(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "autoscale"
    cache = AutoscaleConfigCache(cache_dir)
    cache.defaults_path.parent.mkdir(parents=True)
    cache.defaults_path.write_text(
        "\n".join(
            (
                "version: 1",
                "project: gs://example-bucket/eyvan",
                "worker_service_account: eyvan-worker@gs://example-bucket/eyvan.iam.gserviceaccount.com",
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(autoscale_commands, "gcloud_default_project", lambda: "example-project-1")

    result = CliRunner().invoke(
        cli.main,
        ["autoscale", "init", "--cache-dir", str(cache_dir)],
        input="\nresearch\n\n0\n0\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert "Repair saved defaults" not in result.output
    assert "GCS bucket URI" not in result.output
    assert cache.load_defaults().project == "example-project-1"
    assert "worker_service_account" not in cache.defaults_path.read_text(encoding="utf-8")


def test_autoscale_init_json_keeps_stdout_machine_readable(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "autoscale"
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(autoscale_commands, "gcloud_default_project", lambda: "project-a")

    result = CliRunner().invoke(
        cli.main,
        ["autoscale", "init", "--json", "--cache-dir", str(cache_dir)],
        input="\nresearch\n\n0\n0\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["defaults"]["project"] == "project-a"
    assert "worker_service_account" not in json.loads(result.stdout)["defaults"]
    assert "Review autoscale defaults" in result.stderr


def test_autoscale_generate_requires_yaml_defaults_without_a_terminal(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: False)

    result = CliRunner().invoke(
        cli.main,
        ["autoscale", "generate", "--zone", "us-east5-a", "--cache-dir", str(tmp_path / "autoscale")],
    )

    assert result.exit_code != 0
    assert "eyvan autoscale init" in result.output


def test_autoscale_commands_expose_yaml_defaults_instead_of_identity_flags() -> None:
    runner = CliRunner()
    autoscale_help = runner.invoke(cli.main, ["autoscale", "--help"])
    generate_help = runner.invoke(cli.main, ["autoscale", "generate", "--help"])
    up_help = runner.invoke(cli.main, ["up", "--help"])

    assert autoscale_help.exit_code == 0, autoscale_help.output
    assert generate_help.exit_code == 0, generate_help.output
    assert up_help.exit_code == 0, up_help.output
    for output in (generate_help.output, up_help.output):
        assert "--defaults" in output
        assert "--gcp-project" not in output
        assert "--agent-endpoint" not in output
        assert "--service-account" not in output
    assert "--json" in generate_help.output
    assert "--json" in up_help.output
    assert "clear-queue-denial" not in autoscale_help.output


def test_autoscale_generate_without_zone_or_generation_discovers_cache_candidates(monkeypatch, tmp_path) -> None:
    class Discovery:
        def discover(self, **kwargs):
            assert kwargs["zones"] == ()
            assert kwargs["generations"] == ()
            assert kwargs["preemptible"] is True
            return (
                TpuQuotaAvailability(
                    zone="us-east5-b",
                    accelerator_type="v6e-8",
                    generation="v6e",
                    quota_tpus=512,
                    available_tpus=512,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v6e-preemptible",
                ),
            )

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)
    cache_dir = tmp_path / "autoscale"
    _save_autoscale_defaults(cache_dir)

    result = CliRunner().invoke(
        cli.main,
        [
            "autoscale",
            "generate",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [entry["fleet"] for entry in payload["generated"]] == ["research-us-east5-b"]
    assert payload["generated"][0]["pools"][0]["max_workers"] == 64


def test_up_without_config_applies_selected_cached_autoscale_fleet(monkeypatch, tmp_path) -> None:
    fake = FakeClient()
    cache_dir = tmp_path / "autoscale"
    request = AutoscaleRequest(
        project="project-a",
        zone="us-east5-a",
        profile=DEFAULT_TPU_PROFILES[0],
        fleet_prefix="research",
    )
    AutoscaleConfigCache(cache_dir).store(generate_autoscale_config(request))
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--cache-dir",
            str(cache_dir),
            "--select",
            "research-v5p-16-us-east5-a",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.calls[0][:2] == ("PUT", "/api/v1/fleets/research-v5p-16-us-east5-a")
    request = FleetApplyRequest.model_validate(_json_argument(fake.calls[0]))
    assert request.pools[0].capacity_class is CapacityClass.SPOT
    active = get_active_context(Path(os.environ["EYVAN_CONFIG_FILE"]))
    assert active is not None
    assert active.autoscale_cache_dir == str(cache_dir.resolve())


def test_up_filters_a_cached_queue_denied_tpu_pool_before_applying(monkeypatch, tmp_path) -> None:
    v5p_16 = tpu_profile_from_accelerator_type("v5p-16")
    v6e_16 = tpu_profile_from_accelerator_type("v6e-16")
    assert v5p_16 is not None
    assert v6e_16 is not None
    cache_dir = tmp_path / "autoscale"
    cache = AutoscaleConfigCache(cache_dir)
    cache.store(
        generate_zone_autoscale_config(
            AutoscaleZoneRequest(
                project="project-a",
                zone="us-east5-a",
                profiles=(v5p_16, v6e_16),
                fleet_prefix="research",
            )
        )
    )
    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--cache-dir",
            str(cache_dir),
            "--select",
            "research-us-east5-a",
        ],
    )

    assert result.exit_code == 0, result.output
    request = FleetApplyRequest.model_validate(_json_argument(fake.calls[0]))
    assert [pool.accelerator_type for pool in request.pools] == ["v5p-16"]


def test_up_cached_selector_groups_tpu_families_by_zone(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "autoscale"
    v5p_16 = tpu_profile_from_accelerator_type("v5p-16")
    v5p_32 = tpu_profile_from_accelerator_type("v5p-32")
    assert v5p_16 is not None
    assert v5p_32 is not None
    AutoscaleConfigCache(cache_dir).store(
        generate_zone_autoscale_config(
            AutoscaleZoneRequest(
                project="project-a",
                zone="us-central1-a",
                profiles=(v5p_16, v5p_32),
                fleet_prefix="research",
            ),
            quotas=(
                QuotaSnapshot("us-central1-a", "v5p-16", "v5p", 1024, available_tpus=1024),
                QuotaSnapshot("us-central1-a", "v5p-32", "v5p", 1024, available_tpus=1024),
            ),
        )
    )
    AutoscaleConfigCache(cache_dir).store(
        generate_zone_autoscale_config(
            AutoscaleZoneRequest(
                project="project-a",
                zone="us-east5-a",
                profiles=(v5p_16,),
                fleet_prefix="research",
            )
        )
    )
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--dry-run",
            "--json",
            "--cache-dir",
            str(cache_dir),
        ],
        input="1\n",
    )

    assert result.exit_code == 0, result.output
    assert "Cached Eyvan TPU zones" in result.output
    assert "us-central1-a" in result.output
    assert "1024 available" in result.output
    assert json.loads(result.output[result.output.index("{") :])["fleet"] == "research-us-central1-a"


def test_quota_based_up_shows_preemptible_quota_and_generates_only_matching_config(monkeypatch, tmp_path) -> None:
    fake = FakeClient()

    class Discovery:
        def discover(self, **kwargs):
            assert kwargs["preemptible"] is True
            assert kwargs["accelerator_types"] == ()
            return (
                TpuQuotaAvailability(
                    zone="us-east5-a",
                    accelerator_type="v5p-16",
                    generation="v5p",
                    quota_tpus=32,
                    available_tpus=32,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
            )

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)
    cache_dir = tmp_path / "autoscale"
    _save_autoscale_defaults(cache_dir)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--quota-based",
            "--preemptible",
            "--dry-run",
            "--json",
            "--zone",
            "us-east5-a",
            "--tpu-generation",
            "v5p",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pools"][0]["max_cells"] == 2
    assert payload["autoscale"]["cached"] is False
    assert payload["autoscale"]["quota"] == [
        {
            "accelerator_type": "v5p-16",
            "available_tpus": 32,
            "generation": "v5p",
            "metric": "tpu.googleapis.com/tpu-v5p-preemptible",
            "preemptible": True,
            "quota_tpus": 32,
            "zone": "us-east5-a",
        }
    ]
    assert AutoscaleConfigCache(cache_dir).list() == ()
    assert not (cache_dir / "index.json").exists()
    assert fake.calls == []


def test_quota_based_up_refuses_a_fully_consumed_quota_ceiling(monkeypatch, tmp_path) -> None:
    """A provider limit is not a launchable pool when live balance is zero."""

    fake = FakeClient()

    class Discovery:
        def discover(self, **kwargs):
            assert kwargs["preemptible"] is True
            return (
                TpuQuotaAvailability(
                    zone="us-east5-a",
                    accelerator_type="v5p-16",
                    generation="v5p",
                    quota_tpus=2048,
                    available_tpus=0,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
            )

    cache_dir = tmp_path / "autoscale"
    _save_autoscale_defaults(cache_dir)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--quota-based",
            "--preemptible",
            "--dry-run",
            "--zone",
            "us-east5-a",
            "--tpu-generation",
            "v5p",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code != 0
    assert "no eligible preemptible TPU worker pools have usable live quota" in result.output
    assert "v5p-16" not in result.output
    assert AutoscaleConfigCache(cache_dir).list() == ()
    assert fake.calls == []


def test_quota_based_up_names_the_zone_rejection_when_warm_workers_exceed_quota(monkeypatch, tmp_path) -> None:
    """A zone dropped for its warm-worker floor must say so, not just "no usable quota"."""

    fake = FakeClient()

    class Discovery:
        def discover(self, **kwargs):
            return (
                TpuQuotaAvailability(
                    zone="us-east5-a",
                    accelerator_type="v5p-16",
                    generation="v5p",
                    quota_tpus=32,
                    available_tpus=32,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
            )

    cache_dir = tmp_path / "autoscale"
    AutoscaleConfigCache(cache_dir).save_defaults(
        AutoscaleDefaults(project="project-a", fleet_prefix="research", min_workers=100)
    )
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--quota-based",
            "--preemptible",
            "--dry-run",
            "--zone",
            "us-east5-a",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code != 0
    assert "no eligible preemptible TPU worker pools have usable live quota" in result.output
    assert "us-east5-a:" in result.output
    assert "min_workers" in result.output
    assert "eyvan autoscale init" in result.output
    assert fake.calls == []


def test_quota_based_up_renders_a_bounded_terminal_summary(monkeypatch, tmp_path) -> None:
    fake = FakeClient()

    class Discovery:
        def discover(self, **kwargs):
            return tuple(
                TpuQuotaAvailability(
                    zone="us-east5-a",
                    accelerator_type=accelerator_type,
                    generation="v6e",
                    quota_tpus=1536,
                    available_tpus=1536,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v6e-preemptible",
                )
                for accelerator_type in ("v6e-8", "v6e-16")
            )

    cache_dir = tmp_path / "autoscale"
    _save_autoscale_defaults(cache_dir)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: True)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--quota-based",
            "--preemptible",
            "--zone",
            "us-east5-a",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Fleet applied" in result.output
    assert "research-us-east5-a" in result.output
    assert "v6e: 2 types" in result.output
    assert "Spot / preemptible" in result.output
    assert "1536 available of 1536 TPU units" in result.output
    assert '"accelerator_type"' not in result.output
    assert '"generated"' not in result.output
    assert len(result.output) < 2_000
    assert fake.calls[0][:2] == ("PUT", "/api/v1/fleets/research-us-east5-a")


def test_quota_based_dry_run_does_not_save_prompted_defaults(monkeypatch, tmp_path) -> None:
    fake = FakeClient()

    class Discovery:
        def discover(self, **kwargs):
            return (
                TpuQuotaAvailability(
                    zone="us-east5-a",
                    accelerator_type="v5p-16",
                    generation="v5p",
                    quota_tpus=32,
                    available_tpus=32,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v5p-preemptible",
                ),
            )

    cache_dir = tmp_path / "autoscale"
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", Discovery)
    monkeypatch.setattr(autoscale_commands, "is_interactive", lambda: True)
    monkeypatch.setattr(autoscale_commands, "gcloud_default_project", lambda: "project-a")

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "up",
            "--quota-based",
            "--dry-run",
            "--zone",
            "us-east5-a",
            "--tpu-generation",
            "v5p",
            "--cache-dir",
            str(cache_dir),
        ],
        input="\n\n\n\n0\n0\n",
    )

    assert result.exit_code == 0, result.output
    assert "nothing was saved" in result.output
    assert not cache_dir.exists()
    assert fake.calls == []


def test_status_uses_server_side_fleet_aggregate(monkeypatch, tmp_path) -> None:
    fake = FakeClient(
        {
            "name": "dev-fleet",
            "pools": [{"id": "dev-fleet-cpu-workers", "desired_cells": 2, "ready_cells": 1}],
            "cells": [{"id": "dev-fleet-cpu-workers-0000", "state": "ready"}],
            "pending_demand": {"cpu": 8},
        }
    )

    result = _invoke(monkeypatch, tmp_path, fake, "status")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == fake.status
    assert fake.calls[0][:2] == ("GET", "/api/v1/fleets/dev-fleet")


def test_status_backfills_a_gcp_queue_admission_denial_into_autoscale_cache(monkeypatch, tmp_path) -> None:
    pool = _fleet_status_pool()
    denied = pool.make_cell("dev-fleet-v6e-8-0001").model_copy(
        update={
            "provider_failure_code": CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
            "provider_failure_occurred_at": utcnow(),
            "provider_failure_event_id": "queue-denial-dev-fleet-v6e-8-0001",
            "provider_failure_generation": 1,
            "state": CellState.UNAVAILABLE,
        }
    )
    fake = FakeClient(FleetStatusResponse(name="dev-fleet", pools=[pool], cells=[denied]).model_dump(mode="json"))
    cache_dir = tmp_path / "autoscale"

    result = _invoke(monkeypatch, tmp_path, fake, "status", "--cache-dir", str(cache_dir))

    assert result.exit_code == 0, result.output
    assert AutoscaleConfigCache(cache_dir).blocks_queue_admission_denial(
        project="gcp-project",
        zone="us-central1-c",
        accelerator_type="v6e-8",
        capacity_class=CapacityClass.SPOT,
    )


def test_status_keeps_the_server_response_when_local_denial_cache_is_invalid(monkeypatch, tmp_path) -> None:
    class FailingAutoscaleConfigCache:
        """Raise from the optional cache write while status remains authoritative."""

        calls = 0

        def __init__(self, root: Path | str | None = None) -> None:
            del root

        def record_queue_admission_denial(
            self,
            *,
            project: str,
            zone: str,
            accelerator_type: str,
            capacity_class: CapacityClass,
            observed_at: datetime | None = None,
        ) -> None:
            del project, zone, accelerator_type, capacity_class, observed_at
            type(self).calls += 1
            raise ValueError("invalid local denial cache")

    pool = _fleet_status_pool()
    denied = pool.make_cell("dev-fleet-v6e-8-0001").model_copy(
        update={
            "provider_failure_code": CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
            "provider_failure_occurred_at": utcnow(),
            "provider_failure_event_id": "queue-denial-dev-fleet-v6e-8-0001",
            "provider_failure_generation": 1,
            "state": CellState.UNAVAILABLE,
        }
    )
    payload = FleetStatusResponse(name="dev-fleet", pools=[pool], cells=[denied]).model_dump(mode="json")
    fake = FakeClient(payload)
    monkeypatch.setattr(fleet_commands, "AutoscaleConfigCache", FailingAutoscaleConfigCache)

    result = _invoke(monkeypatch, tmp_path, fake, "status")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == payload
    assert FailingAutoscaleConfigCache.calls == 1


def test_status_uses_the_active_context_autoscale_cache_directory(monkeypatch, tmp_path) -> None:
    from eyvan.sdk.contexts import EyvanContext, set_active_context

    pool = _fleet_status_pool()
    denied = pool.make_cell("dev-fleet-v6e-8-0001").model_copy(
        update={
            "provider_failure_code": CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
            "provider_failure_occurred_at": utcnow(),
            "provider_failure_event_id": "queue-denial-dev-fleet-v6e-8-0001",
            "provider_failure_generation": 1,
            "state": CellState.UNAVAILABLE,
        }
    )
    fake = FakeClient(FleetStatusResponse(name="dev-fleet", pools=[pool], cells=[denied]).model_dump(mode="json"))
    cache_dir = tmp_path / "selected-autoscale"
    set_active_context(
        EyvanContext(
            fleet="dev-fleet",
            org_id="default",
            project_id="default",
            autoscale_cache_dir=str(cache_dir),
        ),
        path=Path(os.environ["EYVAN_CONFIG_FILE"]),
    )

    result = _invoke(monkeypatch, tmp_path, fake, "status")

    assert result.exit_code == 0, result.output
    assert AutoscaleConfigCache(cache_dir).blocks_queue_admission_denial(
        project="gcp-project",
        zone="us-central1-c",
        accelerator_type="v6e-8",
        capacity_class=CapacityClass.SPOT,
    )


def _fleet_status_pool(
    *,
    min_cells: int = 0,
    initial_cells: int | None = None,
    custom_resources: dict[str, float] | None = None,
) -> CapacityPoolRecord:
    custom = custom_resources or {}
    return CapacityPoolRecord(
        id="dev-fleet-v6e-8",
        fleet_name="dev-fleet",
        node_type="v6e-8",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project",
        region="us-central1",
        zone="us-central1-c",
        capacity={"cpu": 32, "tpu": 8, "custom": custom},
        allocatable={"cpu": 32, "tpu": 8, "custom": custom},
        accelerator_type="v6e-8",
        capacity_class=CapacityClass.SPOT,
        agent_endpoint="http://127.0.0.1:8090",
        provider_options={"bootstrap_egress": "public-internet"},
        quota_group="gcp-tpu-spot:gcp-project:us-central1-c:v6e",
        quota_limit=1536,
        quota_units_per_cell=8,
        min_cells=min_cells,
        initial_cells=initial_cells,
        max_cells=10_000,
    )


def _fleet_status_document() -> JsonObject:
    pool = _fleet_status_pool()
    ready = pool.make_cell("dev-fleet-v6e-8-0001").model_copy(update={"state": CellState.READY})
    unavailable = pool.make_cell("dev-fleet-v6e-8-0002").model_copy(
        update={"state": CellState.UNAVAILABLE, "unavailable_reason": "Spot capacity was reclaimed"}
    )
    return FleetStatusResponse(name="dev-fleet", pools=[pool], cells=[ready, unavailable]).model_dump(mode="json")


def test_status_renders_compact_tty_summary_without_platform_internals(monkeypatch, tmp_path) -> None:
    fake = FakeClient(_fleet_status_document())
    _select_plain_fleet_status_rendering(monkeypatch)

    result = _invoke(monkeypatch, tmp_path, fake, "status")

    assert result.exit_code == 0, result.output
    assert "Eyvan fleet" in result.output
    assert "dev-fleet" in result.output
    assert "us-central1-c" in result.output
    assert "Spot / preemptible" in result.output
    assert "v6e-8" in result.output
    assert "8 TPU, 32 CPU" in result.output
    assert "1 ready / 2 active" in result.output
    # A pool legitimately capped at the pydantic max_cells bound of 10000 must
    # render the real number, not an "unbounded" label.
    assert "0 .. 10000" in result.output
    assert "unbounded" not in result.output
    assert "v6e | Spot / preemptible | us-central1-c: 16 / 1536 TPU" in result.output
    assert "Spot capacity" in result.output
    assert "reclaimed" in result.output
    assert "agent_endpoint" not in result.output
    assert "provider_options" not in result.output
    assert "quota_group" not in result.output
    assert max(len(line) for line in result.output.splitlines()) <= 80


def test_status_marks_an_empty_warm_pool_under_provisioned(monkeypatch, tmp_path) -> None:
    pool = _fleet_status_pool(min_cells=2, initial_cells=3)
    document = FleetStatusResponse(name="dev-fleet", pools=[pool], cells=[]).model_dump(mode="json")
    fake = FakeClient(document)
    _select_plain_fleet_status_rendering(monkeypatch)

    result = _invoke(monkeypatch, tmp_path, fake, "status")

    assert result.exit_code == 0, result.output
    assert "under-provisioned" in result.output
    assert "0 ready / 0 active / 3 target" in result.output
    assert "idle" not in result.output
    assert max(len(line) for line in result.output.splitlines()) <= 80

    initialized_pool = pool.model_copy(update={"initialized_at": pool.created_at})
    initialized_document = FleetStatusResponse(name="dev-fleet", pools=[initialized_pool], cells=[]).model_dump(
        mode="json"
    )
    initialized_result = _invoke(monkeypatch, tmp_path, FakeClient(initialized_document), "status")

    assert initialized_result.exit_code == 0, initialized_result.output
    assert "0 ready / 0 active / 2 target" in initialized_result.output
    assert "3 target" not in initialized_result.output
    assert max(len(line) for line in initialized_result.output.splitlines()) <= 80


def test_status_treats_server_values_as_literal_text(monkeypatch, tmp_path) -> None:
    pool = _fleet_status_pool(custom_resources={"[/]": 1})
    cell = CellRecord.model_validate(
        pool.make_cell("[/][bold red]cell[/]")
        .model_copy(
            update={
                "state": CellState.UNAVAILABLE,
                "unavailable_reason": "[/][bold red]literal reason[/]",
            }
        )
        .model_dump(mode="json")
    )
    document = FleetStatusResponse(name="dev-fleet", pools=[pool], cells=[cell]).model_dump(mode="json")
    fake = FakeClient(document)
    _select_plain_fleet_status_rendering(monkeypatch)

    result = _invoke(monkeypatch, tmp_path, fake, "status")

    assert result.exit_code == 0, result.output
    assert "1 [/]" in result.output
    assert "[/][bold red]cell[/]" in result.output
    assert "[/][bold red]literal" in result.output
    assert "reason[/]" in result.output
    assert max(len(line) for line in result.output.splitlines()) <= 80


def test_status_json_overrides_interactive_rendering(monkeypatch, tmp_path) -> None:
    document = _fleet_status_document()
    fake = FakeClient(document)
    _select_plain_fleet_status_rendering(monkeypatch)

    result = _invoke(monkeypatch, tmp_path, fake, "status", "--json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == document


def test_status_orders_worker_shapes_by_numeric_size(monkeypatch, tmp_path) -> None:
    status = FleetStatusResponse.model_validate(_fleet_status_document())
    template = status.pools[0]
    sizes = (128, 1, 16, 4, 8)
    pools = [
        template.model_copy(
            update={
                "id": f"dev-fleet-v6e-{size}",
                "node_type": f"v6e-{size}",
                "accelerator_type": f"v6e-{size}",
            }
        )
        for size in sizes
    ]
    fake = FakeClient(FleetStatusResponse(name="dev-fleet", pools=pools, cells=[]).model_dump(mode="json"))
    _select_plain_fleet_status_rendering(monkeypatch)

    result = _invoke(monkeypatch, tmp_path, fake, "status")

    assert result.exit_code == 0, result.output
    rendered_types = [line.strip().split()[0] for line in result.output.splitlines() if line.strip().startswith("v6e-")]
    assert rendered_types == ["v6e-1", "v6e-4", "v6e-8", "v6e-16", "v6e-128"]


def test_status_help_documents_full_json_mode(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(cli.main, ["status", "--help"])

    assert result.exit_code == 0, result.output
    assert "--json" in result.output
    assert "full server-side fleet record" in result.output


def test_status_accepts_server_fleet_name_without_local_config(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        ["--user", "alice", "--role", "fleet_admin", "status", "dev-fleet"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["name"] == "dev-fleet"
    assert fake.calls[0][:2] == ("GET", "/api/v1/fleets/dev-fleet")


def test_status_rejects_unsafe_fleet_name(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(cli.main, ["status", "../other"])

    assert result.exit_code != 0
    assert "fleet name must contain only" in result.output
    assert fake.calls == []


def test_down_atomically_disables_pools_before_server_side_drain(monkeypatch, tmp_path) -> None:
    fake = FakeClient()

    result = _invoke(monkeypatch, tmp_path, fake, "down", "--yes", "--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == [{"enabled": False, "id": "dev-fleet-cpu-workers"}]
    assert [call[:2] for call in fake.calls] == [
        ("DELETE", "/api/v1/fleets/dev-fleet"),
    ]


def test_config_validation_error_is_actionable(monkeypatch, tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("version: 1\nname: BAD NAME\nnode_types: {}\n")
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: FakeClient())

    result = CliRunner().invoke(cli.main, ["config", "validate", str(config)])

    assert result.exit_code != 0
    assert "invalid fleet configuration" in result.output


def test_workload_submit_sends_capacity_selectors(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--org",
            "acme",
            "--project",
            "research",
            "workload",
            "submit",
            "--name",
            "tpu-train",
            "--command",
            "python train.py",
            "--tpu",
            "8",
            "--provider",
            "gcp",
            "--region",
            "us-east5",
            "--accelerator-type",
            "v5p-8",
            "--required-executor",
            "ray-jobs",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1
    method, path, _arguments = fake.calls[0]
    assert (method, path) == ("POST", "/api/v1/workloads")
    request = WorkloadSubmitRequest.model_validate(_json_argument(fake.calls[0]))
    assert request.spec.model_dump(mode="json", exclude_unset=True) == {
        "name": "tpu-train",
        "execution": {"kind": "command", "command": "python train.py"},
        "secret_mounts": {},
        "gang": {
            "replicas": 1,
            "per_replica": {"cpu": 1.0, "memory_gib": 1.0, "gpu": 0, "tpu": 8},
        },
        "capacity_class": "on_demand",
        "accelerator_type": "v5p-8",
        "provider": "gcp",
        "region": "us-east5",
        "required_executor": "ray-jobs",
        # Priority is sent explicitly rather than left to the server default:
        # without it the CLI could not express a job that preempts batch work,
        # even though the scheduler ranks on exactly these two fields.
        "priority_band": "interactive",
        "intra_band_priority": 0,
        "checkpointable": False,
        "checkpoint_hook": None,
        "checkpoint_output_uri": None,
        "max_retries": 0,
        "max_failure_retries": 0,
        "max_preemption_retries": 10,
        # Sent explicitly like every other option-backed field, so the wire says
        # the capability was declined rather than merely omitted. The placement
        # breadth below reads the same way: empty is this job naming no
        # alternative, which is what makes --region a pin.
        "can_submit_children": False,
        "additional_capacity_classes": [],
        "allowed_regions": [],
        "allowed_zones": [],
        "allowed_accelerator_types": [],
        "named_assets": [],
    }


def test_workload_submit_batch_posts_one_array_request(monkeypatch) -> None:
    """One CLI call, one request: the fan-out happens server-side in one transaction."""

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--org",
            "acme",
            "--project",
            "research",
            "workload",
            "submit-batch",
            "--name",
            "eval",
            "--command",
            "python eval.py",
            "--count",
            "64",
            "--name-prefix",
            "sweep",
            "--tpu",
            "8",
            "--accelerator-type",
            "v5p-8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1, "a 64-task array must not become 64 client requests"
    method, path, _arguments = fake.calls[0]
    assert (method, path) == ("POST", "/api/v1/batches")
    payload = _json_argument(fake.calls[0])
    assert payload["count"] == 64
    assert payload["name_prefix"] == "sweep"
    assert payload["template"]["gang"]["per_replica"]["tpu"] == 8


def test_a_batch_above_the_fan_out_ceiling_is_refused_by_the_cli(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "workload",
            "submit-batch",
            "--name",
            "eval",
            "--command",
            "python eval.py",
            "--count",
            "5",
            "--max-fan-out",
            "999999999",
        ],
    )

    assert result.exit_code != 0
    assert fake.calls == [], "nothing may reach the control plane"


def test_workload_list_uses_the_active_scope_without_duplicate_flags(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    runner = CliRunner()
    activated = runner.invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--org",
            "acme",
            "--project",
            "research",
            "use",
            "dev-fleet",
        ],
    )
    assert activated.exit_code == 0, activated.output
    fake.calls.clear()

    result = runner.invoke(cli.main, ["--user", "alice", "workload", "list"])

    assert result.exit_code == 0, result.output
    assert fake.calls == [
        (
            "GET",
            "/api/v1/workloads",
            {
                "params": {"org_id": "acme", "project_id": "research"},
                "headers": {
                    "X-Eyvan-User": "alice",
                    "X-Eyvan-Roles": "operator",
                    "X-Eyvan-Org": "acme",
                    "X-Eyvan-Project": "research",
                },
            },
        )
    ]


def test_workload_subcommand_scope_updates_real_development_auth_headers(monkeypatch, tmp_path) -> None:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_store_uri=str(tmp_path / "artifacts")))
    config_path = tmp_path / "context.json"
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(config_path))

    with TestClient(app) as api:

        class AsgiCliClient:
            def __init__(self, _base_url: str, *, headers: dict[str, str]):
                self.headers = headers

            def request(
                self,
                method: str,
                path: str,
                *,
                params: tp.Mapping[str, QueryValue] | None = None,
                json: JsonObject | None = None,
                headers: tp.Mapping[str, str] | None = None,
            ) -> JsonValue:
                response = api.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers={**self.headers, **dict(headers or {})},
                )
                if response.is_error:
                    raise cli.EyvanApiError(response.status_code, response.text)
                payload = response.json()
                if not is_json_value(payload):
                    raise AssertionError("test API returned a non-JSON value")
                return payload

            def request_text(
                self,
                method: str,
                path: str,
                *,
                params: tp.Mapping[str, QueryValue] | None = None,
                headers: tp.Mapping[str, str] | None = None,
            ) -> str:
                response = api.request(
                    method,
                    path,
                    params=params,
                    headers={**self.headers, **dict(headers or {})},
                )
                response.raise_for_status()
                return response.text

        monkeypatch.setattr(cli, "EyvanClient", AsgiCliClient)
        result = CliRunner().invoke(
            cli.main,
            ["--user", "alice", "workload", "list", "--org", "acme", "--project", "research"],
        )

    assert result.exit_code == 0, result.output
    assert "[]" in result.output


def test_run_sends_only_command_intent_and_logical_pool_target(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--org",
            "acme",
            "--project",
            "research",
            "run",
            "--no-wait",
            "dev-fleet/gpu-workers",
            "--name",
            "train",
            "--env",
            "RUN_ID=experiment-1",
            "--idempotency-key",
            "run-1",
            "--",
            "python",
            "-m",
            "train",
            "--config",
            "two words.yaml",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1
    method, path, _arguments = fake.calls[0]
    assert (method, path) == (
        "POST",
        "/api/v1/fleets/dev-fleet/node-types/gpu-workers/workloads",
    )
    request = PoolWorkloadSubmitRequest.model_validate(_json_argument(fake.calls[0]))
    assert request.model_dump(mode="json") == {
        "org_id": "acme",
        "project_id": "research",
        "command": "python -m train --config 'two words.yaml'",
        # Shape and placement a caller may state, both at their defaults here:
        # one slice, and no zone pinned, so the fleet still chooses.
        "pods": 1,
        "zone": None,
        "name": "train",
        "environment": {"RUN_ID": "experiment-1"},
        "secret_mounts": {},
        "metadata": {},
        "max_retries": 3,
        "max_failure_retries": 0,
        "max_preemption_retries": 10,
        # Absent unless `--workspace` is passed: `eyvan run` must keep sending
        # pure command intent, with the working tree an explicit opt-in rather
        # than something a submission carries by surprise.
        "workspace": None,
        # Job intent, not placement, so these do not cross the boundary this
        # test draws: the pool still supplies provider, region, accelerator and
        # gang. Whether a job can be checkpointed is a fact about the job.
        "checkpointable": False,
        "checkpoint_hook": None,
        "checkpoint_output_uri": None,
        "can_submit_children": False,
        # Scheduling intent the pool cannot supply: without these a long
        # training run submitted here sat in the interactive band.
        "priority_band": "interactive",
        "intra_band_priority": 0,
        "checkpoint_grace_seconds": 0,
        "estimated_duration_seconds": 3600,
        "named_assets": [],
        # Job intent too: what a job needs installed is a fact about the job,
        # and empty is what one that asked for nothing sends.
        "dependencies": [],
    }
    assert (
        not {
            "provider",
            "region",
            "accelerator_type",
            "capacity_class",
            "required_executor",
            "cpu",
            "memory_gib",
            "gpu",
            "tpu",
        }
        & _json_argument(fake.calls[0]).keys()
    )
    assert _header_arguments(fake.calls[0])["Idempotency-Key"] == "run-1"


def test_run_prints_durable_job_controls_without_changing_json_stdout(monkeypatch) -> None:
    class SubmissionClient(FakeClient):
        def request(
            self,
            method: str,
            path: str,
            *,
            params: dict[str, QueryValue] | None = None,
            json: JsonObject | None = None,
            headers: dict[str, str] | None = None,
        ) -> JsonValue:
            response = super().request(method, path, params=params, json=json, headers=headers)
            if method == "POST" and path.endswith("/workloads"):
                if not is_json_object(response):
                    raise AssertionError("run submission did not return a JSON object")
                return {"id": "wl-cli-run", **response}
            return response

    fake = SubmissionClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--org",
            "acme",
            "--project",
            "research",
            "run",
            "--no-wait",
            "dev-fleet/gpu-workers",
            "--",
            "python",
            "train.py",
        ],
        env={"NO_COLOR": "1", "FORCE_COLOR": "1"},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == "wl-cli-run"
    assert result.stderr.splitlines() == [
        "Eyvan: job wl-cli-run submitted to fleet dev-fleet.",
        "  allocation  elastic",
        "  this job keeps running if this client exits.",
        "  status  eyvan workload status wl-cli-run",
        "  logs    eyvan workload logs wl-cli-run",
        "  cancel  eyvan workload cancel wl-cli-run",
    ]


def test_workload_status_reads_the_durable_workload_record(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    # `--json` for the record. Without it the command summarises for a person,
    # which is the same fetch rendered differently -- and the route is what this
    # test is about.
    result = CliRunner().invoke(cli.main, ["--user", "alice", "workload", "status", "wl-cli-status", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"id": "wl-cli-status", "state": "pending"}
    assert fake.calls == [("GET", "/api/v1/workloads/wl-cli-status", {})]


def test_workload_status_summarises_the_same_record_without_json(monkeypatch) -> None:
    """One fetch, two renderings. A status that dumps the durable record makes
    somebody read sixty keys to learn a state."""

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(cli.main, ["--user", "alice", "workload", "status", "wl-cli-status"])

    assert result.exit_code == 0, result.output
    assert "wl-cli-status" in result.stdout
    assert "pending" in result.stdout
    assert fake.calls == [("GET", "/api/v1/workloads/wl-cli-status", {})]


def test_run_uses_environment_scope_and_preserves_help_as_command_input(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        ["--user", "alice", "run", "--no-wait", "dev-fleet/cpu-workers", "--", "python", "train.py", "--help"],
        env={"EYVAN_ORG": "acme", "EYVAN_PROJECT": "research"},
    )

    assert result.exit_code == 0, result.output
    assert fake.calls[0][2]["json"]["command"] == "python train.py --help"


@pytest.mark.parametrize("target", ["/node", "fleet/", "fleet/node/extra", "../node", "Fleet/node"])
def test_run_rejects_invalid_target_without_calling_api(monkeypatch, target: str) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        ["--org", "acme", "--project", "research", "run", "--no-wait", target, "--", "python", "train.py"],
    )

    assert result.exit_code != 0
    assert "run target" in result.output
    assert fake.calls == []


def test_run_uses_documented_scope_defaults(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        ["run", "--no-wait", "dev-fleet/cpu-workers", "--", "python", "train.py"],
    )

    assert result.exit_code == 0, result.output
    assert fake.calls[0][2]["json"]["org_id"] == "default"
    assert fake.calls[0][2]["json"]["project_id"] == "default"


def test_run_requires_a_command_without_calling_api(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(
        cli.main,
        ["--org", "acme", "--project", "research", "run", "--no-wait", "dev-fleet/cpu-workers"],
    )

    assert result.exit_code != 0
    assert fake.calls == []


def test_run_infers_the_only_submit_capable_fleet(monkeypatch) -> None:
    fake = FakeClient(fleets=("research",))
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(cli.main, ["run", "--no-wait", "v5p-16", "--", "python", "train.py"])

    assert result.exit_code == 0, result.output
    assert [call[:2] for call in fake.calls] == [
        ("GET", "/api/v1/fleets"),
        ("POST", "/api/v1/fleets/research/node-types/v5p-16/workloads"),
    ]


def test_run_requires_an_applied_fleet(monkeypatch) -> None:
    fake = FakeClient(fleets=())
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(cli.main, ["run", "--no-wait", "v5p-16", "--", "python", "train.py"])

    assert result.exit_code != 0
    assert "run `eyvan up` first" in result.output
    assert [call[:2] for call in fake.calls] == [("GET", "/api/v1/fleets")]


def test_run_requires_selection_when_multiple_fleets_are_applied(monkeypatch) -> None:
    fake = FakeClient(fleets=("research", "serving"))
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(cli.main, ["run", "--no-wait", "v5p-16", "--", "python", "train.py"])

    assert result.exit_code != 0
    assert "multiple fleets are applied: research, serving" in result.output
    assert "eyvan use FLEET" in result.output
    assert [call[:2] for call in fake.calls] == [("GET", "/api/v1/fleets")]


def test_run_rejects_a_malformed_fleet_list(monkeypatch) -> None:
    class MalformedFleetClient(FakeClient):
        def request(
            self,
            method: str,
            path: str,
            *,
            params: dict[str, QueryValue] | None = None,
            json: JsonObject | None = None,
            headers: dict[str, str] | None = None,
        ) -> JsonValue:
            if method == "GET" and path == "/api/v1/fleets":
                self.calls.append((method, path, {}))
                return {"fleet": "research"}
            return super().request(method, path, params=params, json=json, headers=headers)

    fake = MalformedFleetClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)

    result = CliRunner().invoke(cli.main, ["run", "--no-wait", "v5p-16", "--", "python", "train.py"])

    assert result.exit_code != 0
    assert "invalid fleet-name list" in result.output
    assert [call[:2] for call in fake.calls] == [("GET", "/api/v1/fleets")]


def test_run_ignores_an_active_fleet_from_another_endpoint(monkeypatch) -> None:
    fake = FakeClient(fleets=("serving",))
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    runner = CliRunner()
    selected = runner.invoke(
        cli.main,
        ["--url", "https://research.example.com", "use", "research"],
    )

    result = runner.invoke(
        cli.main,
        ["--url", "https://serving.example.com", "run", "--no-wait", "v5p-16", "--", "python", "serve.py"],
    )

    assert selected.exit_code == 0, selected.output
    assert result.exit_code == 0, result.output
    assert fake.calls[-2][0:2] == ("GET", "/api/v1/fleets")
    assert fake.calls[-1][0:2] == ("POST", "/api/v1/fleets/serving/node-types/v5p-16/workloads")


def test_up_activates_fleet_for_bare_run_target(monkeypatch, tmp_path) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    config_path = tmp_path / "fleet.yaml"
    config_path.write_text(FLEET_YAML, encoding="utf-8")
    runner = CliRunner()

    applied = runner.invoke(cli.main, ["up", str(config_path)])
    submitted = runner.invoke(cli.main, ["run", "--no-wait", "gpu-workers", "--", "python", "train.py"])

    assert applied.exit_code == 0, applied.output
    assert submitted.exit_code == 0, submitted.output
    assert fake.calls[-1][:2] == (
        "POST",
        "/api/v1/fleets/dev-fleet/node-types/gpu-workers/workloads",
    )
    assert all(call[1] != "/api/v1/fleets" for call in fake.calls)


def test_submit_dag_sends_one_request_carrying_every_edge(monkeypatch, tmp_path) -> None:
    """The pipeline commits in one transaction, so it must be one client call."""

    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "nodes:\n"
        "  - name: prepare\n"
        "    command: python prepare.py\n"
        "    resources: {cpu: 2, memory_gib: 8}\n"
        "  - name: train\n"
        "    command: python train.py\n"
        "    depends_on: [prepare]\n"
        "    replicas: 4\n"
        "    accelerator_type: v5p-16\n"
        "    resources: {tpu: 4}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--org",
            "acme",
            "--project",
            "research",
            "workload",
            "submit-dag",
            "--file",
            str(pipeline),
            "--name-prefix",
            "nightly",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1
    method, path, _arguments = fake.calls[0]
    assert (method, path) == ("POST", "/api/v1/dags")
    payload = _json_argument(fake.calls[0])
    assert payload["name_prefix"] == "nightly"
    assert [node["name"] for node in payload["nodes"]] == ["prepare", "train"]
    assert payload["nodes"][1]["depends_on"] == ["prepare"]
    assert payload["nodes"][1]["template"]["gang"]["replicas"] == 4
    assert payload["nodes"][1]["template"]["accelerator_type"] == "v5p-16"


def test_a_malformed_pipeline_file_never_reaches_the_control_plane(monkeypatch, tmp_path) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("nodes:\n  - name: prepare\n", encoding="utf-8")

    result = CliRunner().invoke(cli.main, ["--user", "alice", "workload", "submit-dag", "--file", str(pipeline)])

    assert result.exit_code != 0
    assert "command" in result.output
    assert fake.calls == [], "nothing may reach the control plane"


# ------------------------------------------------- naming hardware instead of a pool


def _run_args(*args: str) -> list[str]:
    return ["--user", "alice", "--role", "project_operator", "run", *args]


def test_naming_a_shape_leaves_every_positional_to_the_command() -> None:
    """`--tpu v6e-8 python train.py` must not place work on a pool named "python".

    TARGET is a positional in front of a variadic COMMAND, so click hands the
    command's first word to the target. When the hardware is named by flag there
    is no pool argument and that word belongs to the command.
    """

    seen: dict[str, object] = {}

    def capture(self, **kwargs):
        seen.update(kwargs)

    with mock.patch.object(workload_commands.WorkloadCommandService, "run", capture):
        result = CliRunner().invoke(cli.main, _run_args("--tpu", "v6e-8", "python", "train.py"))

    assert result.exit_code == 0, result.output
    assert seen["target_value"] == "v6e-8"
    assert seen["command"] == ("python", "train.py")


def test_naming_a_pool_still_pins_that_pool() -> None:
    """The original form keeps working, so no existing script breaks."""

    seen: dict[str, object] = {}

    def capture(self, **kwargs):
        seen.update(kwargs)

    with mock.patch.object(workload_commands.WorkloadCommandService, "run", capture):
        result = CliRunner().invoke(cli.main, _run_args("research/v6e-8", "python", "train.py"))

    assert result.exit_code == 0, result.output
    assert seen["target_value"] == "research/v6e-8"
    assert seen["command"] == ("python", "train.py")


def test_run_refuses_two_kinds_of_hardware() -> None:
    result = CliRunner().invoke(cli.main, _run_args("--tpu", "v6e-8", "--gpu", "nvidia-tesla-a100", "echo", "hi"))

    assert result.exit_code != 0
    assert "one kind of hardware" in result.output


def test_a_field_the_plane_does_not_know_is_named_as_a_version_skew() -> None:
    """FastAPI's answer is right and completely unhelpful on its own.

    The operator reads that `dependencies` are not permitted, goes looking for
    the flag they used, and finds it documented and working. The cause is never
    in the request: the plane is older than the client that sent it.
    """

    from eyvan.sdk.commands.http import _name_a_version_skew

    detail = [{"type": "extra_forbidden", "loc": ["body", "dependencies"], "msg": "Extra inputs are not permitted"}]

    named = _name_a_version_skew(422, detail)

    assert named is not None
    assert "dependencies" in named
    assert "older than this CLI" in named


def test_an_ordinary_validation_failure_still_says_what_was_wrong() -> None:
    """Most 422s are the caller's fault, and blaming the plane for those would
    send somebody to upgrade a server over a typo."""

    from eyvan.sdk.commands.http import _name_a_version_skew

    detail = [{"type": "missing", "loc": ["body", "command"], "msg": "Field required"}]

    assert _name_a_version_skew(422, detail) is None
    assert _name_a_version_skew(404, [{"type": "extra_forbidden", "loc": ["body", "x"]}]) is None


def test_a_schema_warning_names_the_file_rather_than_pydantics_internals(tmp_path) -> None:
    """The advice is worth keeping; the frame it arrived in is not.

    Python's default handler prints the line inside pydantic that called
    `warn` -- a path under site-packages that tells the reader nothing about
    which file to edit.
    """

    config = tmp_path / "fleet.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "warned",
                "provider": {"type": "local", "region": "local"},
                "executor": "native-local",
                "capacity_class": "on_demand",
                "pools": {
                    f"cpu-{index}": {
                        "kind": "cpu",
                        "capacity_backend": "local",
                        "machine_type": "laptop",
                        "min_workers": 0,
                        "max_workers": 0,
                        "resources": {"cpu": 1, "memory_gib": 1},
                        "agent_endpoint": "http://127.0.0.1:8095",
                    }
                    for index in range(3)
                },
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli.main, ["config", "validate", str(config)])

    assert result.exit_code == 0, result.output
    assert "site-packages" not in result.output
    assert str(config) in result.output
    # Once, not once per pool: the schema warns as it validates each one, and
    # three copies of one sentence bury the file name the line exists to add.
    assert result.output.count("max_workers: 0 means unbounded") == 1


def test_session_reresolves_loopback_credentials_minted_after_it_resolved(tmp_path, monkeypatch) -> None:
    """First run: a session resolved before the plane must present what the plane minted.

    The session's identity is fixed before the command body runs, and on a
    machine with no development credential the plane started by that same
    command mints one and requires it. ``refresh_loopback_identity`` is the
    seam: a no-op while nothing changed, a full re-resolution once the
    credential exists, and never a downgrade of a token identity.
    """

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.sdk.client.identity import ClientIdentity, resolve_control_plane_identity
    from eyvan.sdk.commands.state import CliSession, ContextOverrides
    from eyvan.sdk.contexts import EyvanContextConfig
    from eyvan.server.local.auth import LocalDevelopmentCredentialStore

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8098")
    header_updates: list[dict[str, str]] = []

    class _Client:
        def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
            header_updates.append(dict(headers))

    def _session(identity: ClientIdentity) -> CliSession:
        return CliSession(
            client=_Client(),
            identity=identity,
            context_store=None,
            context_config=EyvanContextConfig(),
            base_url=endpoint.url,
            org_id=None,
            project_id=None,
            overrides=ContextOverrides(),
        )

    session = _session(resolve_control_plane_identity(endpoint))
    assert session.identity.local_development_credential is None

    # Nothing has been minted yet: refreshing must not invent identity headers.
    session.refresh_loopback_identity()
    assert header_updates == []

    # The plane starts and mints the credential it will demand. This file's
    # autouse fixture redirects the store to a per-test path; mint there,
    # exactly where the redirected resolution reads from.
    LocalDevelopmentCredentialStore(tmp_path / "local-development-credential").load_or_create()

    session.refresh_loopback_identity()
    assert session.identity.local_development_credential is not None
    assert header_updates, "the client must be told about the new credential"
    assert all("X-Eyvan-Local-Credential" in headers for headers in header_updates)

    # A token identity outranks development headers and is never re-resolved.
    token_session = _session(ClientIdentity(token="operator-token"))
    token_session.refresh_loopback_identity()
    assert token_session.identity.token == "operator-token"

    # A remote endpoint has no local development material to pick up.
    remote = _session(ClientIdentity(token="operator-token"))
    remote.base_url = "https://eyvan.example.com"
    remote.refresh_loopback_identity()
    assert header_updates[-1] is not None


def test_ensuring_a_local_runtime_refreshes_the_session_after_the_plane_exists() -> None:
    """The refresh must follow the ensure, or the session keeps its stale identity."""

    from eyvan.sdk.commands.fleets import FleetCommandService

    events: list[str] = []

    class _Session:
        base_url = "http://127.0.0.1:8080"

        def refresh_loopback_identity(self) -> None:
            events.append("refresh")

    class _Runtime:
        def ensure(self, endpoint: object) -> None:
            events.append("runtime")

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    config = EyvanFleetConfig.model_validate(
        {
            "version": 1,
            "name": "research",
            "provider": {"type": "onprem-ssh", "region": "local"},
            "pools": {
                "workers": {
                    "kind": "cpu",
                    "machine_type": "n2-standard-8",
                    "resources": {"cpu": 4, "memory_gib": 16},
                    "capacity_backend": "onprem",
                    "min_cells": 0,
                    "max_cells": 1,
                }
            },
        }
    )
    service = FleetCommandService(
        session=_Session(),
        loopback_control_plane=lambda endpoint: events.append("plane"),
        native_gcp_profile_available=lambda: False,
    )
    service.ensure_local_runtime(config)
    assert events == ["plane", "refresh"], "the session must re-resolve after the plane exists"

    events.clear()
    native = FleetCommandService(
        session=_Session(),
        gcp_runtime=_Runtime(),
        native_gcp_profile_available=lambda: True,
    )
    native.ensure_local_runtime(config)
    assert events == ["runtime", "refresh"], "the native runtime lane refreshes on the same rule"


def test_builds_disagreement_names_the_recycle_commands(monkeypatch, tmp_path) -> None:
    """Proven skew is only useful if the exit says how to leave it.

    ``capacity builds`` exists to catch a stale agent before it executes a
    workload wrongly. Catching it and then printing a bare condition hands the
    operator the same mid-incident doc hunt the dispatch refusal used to.
    """

    class SkewedPlane(FakeClient):
        def request(self, method, path, **kwargs):
            if method == "GET" and path == "/api/v1/capacity/builds":
                return {
                    "plane": {"version": "0.0.1", "source_digest": "sha256:" + "a" * 64},
                    "agents": [{"endpoint": "http://agent:8090", "build": {"source_digest": "sha256:" + "b" * 64}}],
                    "unreachable": [],
                    "unscanned_pools": [],
                    "in_agreement": False,
                }
            return super().request(method, path, **kwargs)

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: SkewedPlane())
    result = CliRunner().invoke(cli.main, ["--user", "alice", "--role", "fleet_admin", "capacity", "builds"])

    assert result.exit_code != 0
    assert "cell-drain" in result.output, "the recycle verbs must ride in the error"
    assert "cell-delete" in result.output
    assert "runtime deploy" in result.output, "the one-command alternative must appear too"
