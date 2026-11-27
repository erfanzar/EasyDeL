# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for Wave 1 CLI, SDK, and configuration fixes."""

from __future__ import annotations

import asyncio
import json
import typing as tp
from pathlib import Path

import httpx
import pytest
import yaml
from click.testing import CliRunner

from eyvan.capacity.autoscale import AutoscaleDefaults
from eyvan.capacity.providers.gcp import accelerator_catalog
from eyvan.sdk.client.async_client import AsyncEyvanClient
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.client.jobs import CpuConfig, EyvanPlacementError, GpuConfig, TpuConfig, placement_error_labels
from eyvan.sdk.commands import autoscale as autoscale_commands
from eyvan.sdk.commands import gcp as gcp_commands
from eyvan.sdk.commands import gcp_connect as gcp_connect_commands
from eyvan.sdk.commands.capacity import capacity
from eyvan.sdk.commands.fleets import down_fleet
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContext, EyvanContextConfig, FileContextStore
from eyvan.sdk.workload_summary import format_user_progress
from eyvan.sdk.workload_summary import render as render_workload_summary


class _StubClient:
    def __init__(self, responses: dict[tuple[str, str], tp.Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, tp.Any] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, tp.Any] | None = None,
        json: tp.Any = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> tp.Any:
        del json, headers
        self.calls.append((method, path, None if params is None else dict(params)))
        if (method, path) in self.responses:
            return self.responses[(method, path)]
        raise AssertionError(f"unexpected request {method} {path}")

    def request_text(self, method: str, path: str, *, params=None, headers=None) -> str:
        del method, path, params, headers
        return ""

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        del headers


def _session(tmp_path: Path, client: _StubClient) -> CliSession:
    ctx = EyvanContext(fleet="research", org_id="acme", project_id="ml")
    return CliSession(
        client=tp.cast(tp.Any, client),
        identity=ClientIdentity(dev_user="alice"),
        context_store=FileContextStore(tmp_path / "contexts.json"),
        context_config=EyvanContextConfig(active_context=ctx.fleet, contexts={ctx.fleet: ctx}),
        base_url="http://127.0.0.1:8080",
        org_id="acme",
        project_id="ml",
        overrides=ContextOverrides(),
    )


def test_eyvan_down_accepts_fleet_name_and_yaml_config(tmp_path: Path) -> None:
    client = _StubClient(
        {
            ("GET", "/api/v1/fleets/research-v6e"): {"name": "research-v6e", "pools": [{"id": "p1"}]},
            ("DELETE", "/api/v1/fleets/research-v6e"): {"fleet": "research-v6e", "disabled_pools": 1},
        }
    )
    session = _session(tmp_path, client)

    dry = CliRunner().invoke(down_fleet, ["research-v6e", "--dry-run"], obj=session)
    assert dry.exit_code == 0, dry.output
    assert "research-v6e" in dry.output
    assert client.calls == [("GET", "/api/v1/fleets/research-v6e", None)]

    real = CliRunner().invoke(down_fleet, ["research-v6e", "--yes"], obj=session)
    assert real.exit_code == 0, real.output
    assert client.calls[-1] == ("DELETE", "/api/v1/fleets/research-v6e", None)

    cfg = tmp_path / "fleet.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "yaml-fleet",
                "provider": {"type": "local", "region": "local"},
                "pools": {
                    "cpu": {
                        "kind": "cpu",
                        "machine_type": "local-cpu",
                        "capacity_backend": "manual",
                        "resources": {"CPU": 2, "memory_gib": 4},
                        "min_workers": 0,
                        "max_workers": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    client.responses[("DELETE", "/api/v1/fleets/yaml-fleet")] = {"fleet": "yaml-fleet", "disabled_pools": 1}
    from_file = CliRunner().invoke(down_fleet, [str(cfg), "--yes"], obj=session)
    assert from_file.exit_code == 0, from_file.output
    assert client.calls[-1] == ("DELETE", "/api/v1/fleets/yaml-fleet", None)


def test_capacity_fleets_uses_tenant_scoped_route_and_pool_disable_deletes_pool(tmp_path: Path) -> None:
    client = _StubClient(
        {
            ("GET", "/api/v1/capacity"): {
                "pools": [
                    {
                        "pool_id": "p-v6e",
                        "fleet": "research",
                        "accelerator_type": "v6e-8",
                        "zone": "us-east5-a",
                        "enabled": True,
                    }
                ]
            },
            ("GET", "/api/v1/capacity-pools"): [
                {
                    "id": "p-admin",
                    "fleet_name": "admin-fleet",
                    "accelerator_type": "v5p-8",
                    "zone": "us-east5-a",
                    "enabled": True,
                }
            ],
            ("DELETE", "/api/v1/capacity-pools/p-v6e"): {"id": "p-v6e", "enabled": False},
        }
    )
    session = _session(tmp_path, client)

    # Default uses tenant-scoped GET /api/v1/capacity (works for non-admin submitters).
    scoped = CliRunner().invoke(capacity, ["fleets", "--json"], obj=session)
    assert scoped.exit_code == 0, scoped.output
    assert client.calls[-1] == ("GET", "/api/v1/capacity", {"org_id": "acme", "project_id": "ml"})
    assert json.loads(scoped.output)[0]["fleet"] == "research"

    # --all uses admin GET /api/v1/capacity-pools without bogus query params.
    admin_all = CliRunner().invoke(capacity, ["fleets", "--all", "--json"], obj=session)
    assert admin_all.exit_code == 0, admin_all.output
    assert client.calls[-1] == ("GET", "/api/v1/capacity-pools", None)
    assert json.loads(admin_all.output)[0]["fleet"] == "admin-fleet"

    # capacity pool disable calls DELETE /api/v1/capacity-pools/{pool_id}
    disabled = CliRunner().invoke(capacity, ["pool", "disable", "p-v6e", "--yes", "--json"], obj=session)
    assert disabled.exit_code == 0, disabled.output
    assert client.calls[-1] == ("DELETE", "/api/v1/capacity-pools/p-v6e", None)
    assert json.loads(disabled.output)["enabled"] is False


def test_autoscale_init_prompt_preserves_runtime_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = AutoscaleDefaults(
        project="proj-a",
        fleet_prefix="research",
        bootstrap_egress="public-internet",
        min_workers=0,
        max_workers=2,
        runtime_python="/opt/venv/bin/python",
        runtime_init_commands=("echo init",),
        runtime_container_image="us-docker.pkg.dev/proj-a/eyvan/worker:v1",
    )
    prompt_answers = iter(["proj-b", "team", "3"])
    int_answers = iter([1, 4])
    monkeypatch.setattr(autoscale_commands.Prompt, "ask", lambda *_a, **_k: next(prompt_answers))
    monkeypatch.setattr(autoscale_commands.IntPrompt, "ask", lambda *_a, **_k: next(int_answers))
    monkeypatch.setattr(autoscale_commands, "gcloud_default_project", lambda: None)

    updated = autoscale_commands._prompt_autoscale_defaults(existing, review=False)
    assert updated.project == "proj-b"
    assert updated.fleet_prefix == "team"
    assert updated.min_workers == 1
    assert updated.max_workers == 4
    assert updated.runtime_python == "/opt/venv/bin/python"
    assert updated.runtime_init_commands == ("echo init",)
    assert updated.runtime_container_image == "us-docker.pkg.dev/proj-a/eyvan/worker:v1"


def test_gcp_account_env_and_local_agent_endpoint_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    seen_env: list[dict[str, str]] = []

    original_run = subprocess.run

    def fake_run(argv, **kwargs):
        if argv and argv[0] == "gcloud":
            seen_env.append(dict(kwargs["env"]))
            return subprocess.CompletedProcess(argv, 0, stdout="tok-123\n", stderr="")
        return original_run(argv, **kwargs)

    monkeypatch.setattr(accelerator_catalog.subprocess, "run", fake_run)
    assert accelerator_catalog.gcloud_access_token(account="ops@example.com") == "tok-123"
    assert seen_env[-1]["CLOUDSDK_CORE_ACCOUNT"] == "ops@example.com"

    # CLI flags are registered on browse, connect, and sync.
    for cmd_name in ("browse", "connect", "sync"):
        cmd = gcp_commands.gcp.commands[cmd_name]
        assert any("--account" in param.opts for param in cmd.params), cmd_name

    # _resolved_local_agent_endpoint reads loopback platform default_agent_endpoint first.
    profile_path = tmp_path / "runtime-profile.json"
    platform_path = tmp_path / "platform.yaml"
    platform_path.write_text(
        "version: 1\nagents:\n  default:\n    endpoint: http://127.0.0.1:9099\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EYVAN_PLATFORM_CONFIG_PATH", str(platform_path))
    assert gcp_connect_commands._resolved_local_agent_endpoint(profile_path).url == "http://127.0.0.1:9099"

    # Non-loopback default_agent_endpoint must NOT receive local agent_shared_secret; falls back safely.
    platform_path.write_text(
        "version: 1\nagents:\n  default:\n    endpoint: https://attacker.example:9099\n",
        encoding="utf-8",
    )
    fake_home = tmp_path / "home"
    daemon_dir = fake_home / ".eyvan"
    daemon_dir.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv("HOME", str(fake_home))

    # Non-dict JSON and world-writable/symlink PID files are safely ignored (no AttributeError or redirect).
    malformed = daemon_dir / "cell-agent-0001.pid"
    malformed.write_text("[]", encoding="utf-8")
    malformed.chmod(0o600)
    world_writable = daemon_dir / "cell-agent-0002.pid"
    world_writable.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:9111",
                "pid": 1,
                "runtime_profile_path": str(profile_path.resolve()),
            }
        ),
        encoding="utf-8",
    )
    world_writable.chmod(0o666)
    assert gcp_connect_commands._resolved_local_agent_endpoint(profile_path).url == "http://127.0.0.1:8090"

    # Valid owner-only (0o600) loopback PID record matching profile_path with live PID is used.
    import os

    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.server.local.daemon import LocalDaemonRecord, write_owner_only_record

    valid_pid = daemon_dir / "cell-agent-0003.pid"
    write_owner_only_record(
        valid_pid,
        LocalDaemonRecord(
            endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:9123"),
            pid=os.getpid(),
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    assert gcp_connect_commands._resolved_local_agent_endpoint(profile_path).url == "http://127.0.0.1:9123"


def test_async_client_list_capacity_and_workload_log_page() -> None:
    seen: list[tuple[str, str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/api/v1/capacity":
            return httpx.Response(
                200,
                json={
                    "pools": [
                        {
                            "fleet": "research",
                            "pool": "p1",
                            "accelerator_type": "v6e-8",
                            "capacity_class": "spot",
                            "region": "us-east5",
                            "zone": "us-east5-a",
                            "min_cells": 0,
                            "max_cells": 4,
                            "enabled": True,
                        }
                    ]
                },
            )
        if request.url.path == "/api/v1/workloads/wl-1/log-page":
            return httpx.Response(200, json={"entries": [{"line": "ok"}], "next_cursor": "c2"})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    async def exercise() -> tuple[tp.Any, tp.Any]:
        async with AsyncEyvanClient(
            "http://127.0.0.1:8080",
            identity=ClientIdentity(dev_user="alice"),
            client=httpx.AsyncClient(base_url="http://127.0.0.1:8080", transport=httpx.MockTransport(handler)),
        ) as client:
            cap = await client.list_capacity(org_id="acme", project_id="ml")
            page = await client.workload_log_page("wl-1", after="c1", limit=50)
            return cap, page

    cap, page = asyncio.run(exercise())
    assert len(cap) == 1 and cap[0].fleet == "research" and cap[0].pool == "p1"
    assert page == {"entries": [{"line": "ok"}], "next_cursor": "c2"}
    assert seen == [
        ("GET", "/api/v1/capacity", {"org_id": "acme", "project_id": "ml"}),
        ("GET", "/api/v1/workloads/wl-1/log-page", {"after": "c1", "limit": "50"}),
    ]


def test_accelerator_config_validation_and_placement_error_labels() -> None:
    with pytest.raises(ValueError, match="max_preemption_retries"):
        TpuConfig("v6e-8", max_preemption_retries=-1)
    with pytest.raises(ValueError, match="max_preemption_retries"):
        TpuConfig("v6e-8", max_preemption_retries=1001)
    with pytest.raises(ValueError, match="max_preemption_retries"):
        TpuConfig("v6e-8", max_preemption_retries=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dependencies"):
        GpuConfig("h100", dependencies="torch")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dependencies"):
        CpuConfig(dependencies=b"numpy")  # type: ignore[arg-type]

    cfg = CpuConfig(dependencies=["numpy", "scipy"])
    assert cfg.dependencies == ("numpy", "scipy")
    assert placement_error_labels(cfg) == ("CPU", "CpuConfig")
    assert placement_error_labels(GpuConfig("h100")) == ("GPU", "GpuConfig")
    assert placement_error_labels(TpuConfig("v6e-8")) == ("TPU", "TpuConfig")

    err = EyvanPlacementError(
        requested_accelerator_type="cpu",
        fleet="local",
        accelerator_label="CPU",
        config_name="CpuConfig",
        available_accelerator_types=("cpu-4",),
    )
    assert "CPU 'cpu'" in str(err) and "CpuConfig(" in str(err)


def test_workload_summary_renders_user_progress() -> None:
    formatted = format_user_progress(
        [
            {"rank": 0, "step": 10, "total": 40, "phase": "train", "message": "loss=0.5"},
            {"rank": 1, "step": 10, "total": 40, "message": "ok"},
        ]
    )
    assert any("10/40" in line and "25%" in line and "loss=0.5" in line for line in formatted)
    lines = render_workload_summary(
        {
            "id": "wl-1",
            "state": "RUNNING",
            "spec": {"name": "train", "accelerator_type": "v6e-8", "capacity_class": "spot"},
            "user_progress": [{"rank": 0, "step": 10, "total": 40, "phase": "train"}],
        }
    )
    assert any("progress" in line and "10/40" in line for line in lines)


def test_cli_skips_context_load_and_banner_for_race_and_subcommand_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from eyvan.sdk import cli

    fake_home = tmp_path / "broken-config-home"
    bad_cfg = fake_home / ".config" / "eyvan" / "config.json"
    bad_cfg.parent.mkdir(parents=True)
    bad_cfg.write_text("{not-valid-json", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))

    runner = CliRunner()
    for argv in (
        ["race", "--help"],
        ["race", "sweep", "--help"],
        ["capacity", "--help"],
        ["run", "--help"],
        ["workload", "submit", "--help"],
    ):
        res = runner.invoke(cli.main, argv)
        assert res.exit_code == 0, (argv, res.output)
        assert "invalid Eyvan context configuration" not in res.output
        assert "development plane" not in res.output


def test_ai_zone_suffix_accepted_across_gcp_validators() -> None:
    from eyvan.capacity.autoscale import models as autoscale_models
    from eyvan.sdk.commands import head as head_commands
    from eyvan.server.local.attach_job import AttachmentRequest

    zone = "us-south1-ai1b"
    assert gcp_connect_commands._ZONE.fullmatch(zone) is not None
    assert head_commands._ZONE_PATTERN.fullmatch(zone) is not None
    assert autoscale_models._ZONE.fullmatch(zone) is not None
    req = AttachmentRequest(target="tpu_node_1", kind="tpu", project="example-project-a", zone=zone)
    assert req.zone == zone


def test_capacity_quota_supports_account_and_offline_project_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from eyvan.capacity.providers.gcp.quota_discovery import TpuQuotaAvailability
    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.sdk.client.transport import EyvanUnavailableError
    from eyvan.sdk.commands import capacity as capacity_commands

    seen_accounts: list[str | None] = []

    class _FakeDiscovery:
        def __init__(self, *, account: str | None = None) -> None:
            seen_accounts.append(account)

        def discover(self, *, project: str, zones=(), preemptible: bool = True):
            del project, zones
            if not preemptible:
                return ()
            return (
                TpuQuotaAvailability(
                    zone="us-central2-b",
                    accelerator_type="v6e-4",
                    generation="v6e",
                    quota_tpus=512,
                    available_tpus=512,
                    preemptible=True,
                    metric="tpu.googleapis.com/tpu-v6e-preemptible",
                ),
            )

    class _OfflineClient(_StubClient):
        def request(self, method: str, path: str, *, params=None, json=None, headers=None):
            del path, params, json, headers
            raise EyvanUnavailableError(ControlPlaneEndpoint.parse("http://127.0.0.1:8080"), method)

    monkeypatch.setattr(capacity_commands, "GcpTpuQuotaDiscovery", _FakeDiscovery)
    session = _session(tmp_path, _OfflineClient({}))

    # Offline fallback when --project is passed and plane is unavailable.
    offline = CliRunner().invoke(capacity, ["quota", "--project", "my-proj", "--json"], obj=session)
    assert offline.exit_code == 0, offline.output
    doc = json.loads(offline.stdout)
    assert doc["status"] == "ok" and doc["project"] == "my-proj"
    assert doc["entries"][0]["zone"] == "us-central2-b" and doc["entries"][0]["free_chips"] == 512
    assert seen_accounts == [None]

    # Explicit --account bypasses control plane and passes account to GcpTpuQuotaDiscovery.
    acct = CliRunner().invoke(
        capacity,
        ["quota", "--project", "my-proj", "--account", "ops@example.com", "--json"],
        obj=session,
    )
    assert acct.exit_code == 0, acct.output
    assert seen_accounts == [None, "ops@example.com"]
    assert json.loads(acct.stdout)["entries"][0]["accelerator_type"] == "v6e-4"

    # Without --project, offline control plane still surfaces unavailability.
    no_proj = CliRunner().invoke(capacity, ["quota", "--json"], obj=session)
    assert no_proj.exit_code != 0


def test_autoscale_generate_and_quota_discovery_forward_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    from eyvan.capacity.providers.gcp import quota_discovery as qd
    from eyvan.capacity.providers.gcp import quota_usage as qu

    assert any("--account" in param.opts for param in autoscale_commands.generate_autoscale.params)

    seen_envs: list[dict[str, str]] = []

    def fake_run(argv, **kwargs):
        seen_envs.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(argv, 0, stdout="tok\n", stderr="")

    monkeypatch.setattr(qd.subprocess, "run", fake_run)
    monkeypatch.setattr(qu.subprocess, "run", fake_run)

    discovery = qd.GcpTpuQuotaDiscovery(account="ops@example.com")
    discovery._runner(("version",))
    assert seen_envs[-1]["CLOUDSDK_CORE_ACCOUNT"] == "ops@example.com"
    assert isinstance(discovery._usage_discovery, qu.GcpTpuQuotaUsageDiscovery)
    discovery._usage_discovery._token_runner(("auth", "print-access-token"))
    assert seen_envs[-1]["CLOUDSDK_CORE_ACCOUNT"] == "ops@example.com"


def test_cli_reports_local_development_credential_error_as_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from eyvan.sdk import cli
    from eyvan.server.local.auth import LocalDevelopmentCredentialError

    monkeypatch.setenv("HOME", str(tmp_path))

    def fail_identity(*_a, **_k):
        raise LocalDevelopmentCredentialError("local Eyvan credential directory must have mode 0700")

    monkeypatch.setattr(cli, "resolve_control_plane_identity", fail_identity)
    result = CliRunner().invoke(cli.main, ["capacity", "fleets"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error: local Eyvan credential directory must have mode 0700" in result.output


def test_generated_autoscale_yaml_with_quota_group_does_not_warn_on_max_workers_zero(tmp_path: Path) -> None:
    import warnings

    from eyvan.capacity.autoscale import (
        DEFAULT_TPU_PROFILES,
        AutoscaleZoneRequest,
        QuotaSnapshot,
        generate_zone_autoscale_config,
        render_autoscale_yaml,
    )
    from eyvan.capacity.fleet_config import EyvanFleetConfig

    v6e_profiles = tuple(p for p in DEFAULT_TPU_PROFILES if p.generation == "v6e")[:2]
    assert len(v6e_profiles) == 2
    quotas = tuple(
        QuotaSnapshot(
            zone="us-central2-b",
            accelerator_type=p.accelerator_type,
            generation="v6e",
            quota_tpus=512,
            available_tpus=512,
            preemptible=True,
        )
        for p in v6e_profiles
    )
    generated = generate_zone_autoscale_config(
        AutoscaleZoneRequest(
            project="example-project-a",
            zone="us-central2-b",
            profiles=v6e_profiles,
            fleet_prefix="eyvan-ex",
            min_workers=0,
            max_workers=0,
        ),
        quotas=quotas,
    )
    yaml_path = tmp_path / "autoscale.yaml"
    yaml_path.write_text(render_autoscale_yaml(generated), encoding="utf-8")

    # Loading Eyvan's own quota-bounded autoscale YAML (e.g. via `eyvan down`) must not emit UserWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        loaded = EyvanFleetConfig.from_yaml(yaml_path)
    assert loaded.name == generated.config.name

    # A hand-written pool with max_workers: 0 and no quota_group/quota_limit still warns.
    hand_written = tmp_path / "hand.yaml"
    hand_written.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "hand-fleet",
                "provider": {"type": "local", "region": "local"},
                "pools": {
                    "cpu": {
                        "kind": "cpu",
                        "machine_type": "local-cpu",
                        "capacity_backend": "manual",
                        "resources": {"CPU": 2, "memory_gib": 4},
                        "max_workers": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="max_workers: 0 means unbounded"):
        EyvanFleetConfig.from_yaml(hand_written)


def test_capacity_quota_truncates_overlong_gcloud_failure_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Offline and server quota discovery truncate >2048-char provider errors instead of crashing Pydantic."""

    from fastapi.testclient import TestClient

    from eyvan.capacity.providers.gcp.quota_discovery import GcloudQuotaError
    from eyvan.config import EyvanSettings
    from eyvan.records import CapacityClass
    from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
    from eyvan.sdk.client.transport import EyvanUnavailableError
    from eyvan.sdk.commands import capacity as capacity_commands
    from eyvan.server.api.app import create_app
    from eyvan.server.api.schemas import TPU_QUOTA_FAILURE_REASON_MAX_LENGTH, TpuQuotaFailure
    from eyvan.server.persistence import InMemoryRepository

    long_error = "ERROR: gcloud quota lookup failed: " + ("x" * 3500) + " tail-marker"

    # Direct helper bounds both overlong and blank reasons.
    truncated = TpuQuotaFailure.from_reason(capacity_class=CapacityClass.SPOT, reason=long_error)
    assert len(truncated.reason) <= TPU_QUOTA_FAILURE_REASON_MAX_LENGTH
    assert truncated.reason.endswith("...")
    assert truncated.reason.startswith("ERROR: gcloud quota lookup failed: ")

    blank = TpuQuotaFailure.from_reason(capacity_class=CapacityClass.ON_DEMAND, reason="   ")
    assert blank.reason == "provider quota discovery failed"

    class _FailingDiscovery:
        def __init__(self, *, account: str | None = None) -> None:
            del account

        def discover(self, *, project: str, zones=(), preemptible: bool = True):
            del project, zones, preemptible
            raise GcloudQuotaError(long_error)

    class _OfflineClient(_StubClient):
        def request(self, method: str, path: str, *, params=None, json=None, headers=None):
            del path, params, json, headers
            raise EyvanUnavailableError(ControlPlaneEndpoint.parse("http://127.0.0.1:8080"), method)

    monkeypatch.setattr(capacity_commands, "GcpTpuQuotaDiscovery", _FailingDiscovery)
    session = _session(tmp_path, _OfflineClient({}))

    offline = CliRunner().invoke(capacity, ["quota", "--project", "my-proj", "--json"], obj=session)
    assert offline.exit_code == 0, offline.output
    doc = json.loads(offline.stdout)
    assert doc["status"] == "unavailable"
    assert len(doc["failures"]) == 2
    for failure in doc["failures"]:
        assert len(failure["reason"]) <= TPU_QUOTA_FAILURE_REASON_MAX_LENGTH
        assert failure["reason"].endswith("...")
        assert failure["reason"].startswith("ERROR: gcloud quota lookup failed: ")

    # Server GET /api/v1/capacity/quota must handle the same >3000-char exception cleanly.
    repository = InMemoryRepository()
    settings = EyvanSettings(
        dev_auth_enabled=True,
        artifact_store_uri=str(tmp_path / "artifacts"),
        controller_enabled=False,
    )
    app = create_app(settings=settings, repository=repository, quota_discovery=_FailingDiscovery())
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/capacity/quota",
            params={"project": "my-proj"},
            headers={"X-Eyvan-User": "alice", "X-Eyvan-Roles": "fleet_admin"},
        )
    assert response.status_code == 200, response.text
    server_doc = response.json()
    assert server_doc["status"] == "unavailable"
    assert len(server_doc["failures"]) == 2
    for failure in server_doc["failures"]:
        assert len(failure["reason"]) <= TPU_QUOTA_FAILURE_REASON_MAX_LENGTH
        assert failure["reason"].endswith("...")
