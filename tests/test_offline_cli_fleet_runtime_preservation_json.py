# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline CLI regressions for fleet runtime preservation and JSON output."""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.autoscale import AutoscaleDefaults, AutoscaleGenerationResult, CachedAutoscaleConfig
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands import fleets as fleet_commands
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContextConfig, FileContextStore


class _OfflineClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, JsonObject | None]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        json: JsonObject | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        self.calls.append((method, path, json))
        if method == "PUT" and path == "/api/v1/fleets/proof-fleet":
            return {"name": "proof-fleet", "pools": []}
        if method == "GET" and path == "/api/v1/capacity":
            return {"pools": []}
        raise AssertionError(f"unexpected offline API request: {method} {path}")


def _session(tmp_path: Path) -> CliSession:
    return CliSession(
        client=_OfflineClient(),
        identity=ClientIdentity(),
        context_store=FileContextStore(tmp_path / "context.json"),
        context_config=EyvanContextConfig(),
        base_url="https://plane.example.test",
        org_id="proof-org",
        project_id="proof-project",
        overrides=ContextOverrides(org_id="proof-org", project_id="proof-project"),
    )


def _fleet(fleet_python: str | None = None) -> EyvanFleetConfig:
    runtime: dict[str, tp.Any] = {"init_commands": ["echo fleet-bootstrap"]}
    if fleet_python is not None:
        runtime["python"] = fleet_python
    return EyvanFleetConfig.model_validate(
        {
            "version": 1,
            "name": "proof-fleet",
            "provider": {
                "type": "gcp",
                "project": "proof-project",
                "region": "us-central1",
                "zone": "us-central1-a",
            },
            "agent_endpoint": "https://agent.example.test",
            "executor": "eyvan-native",
            "runtime": runtime,
            "node_types": {
                "tpu": {
                    "kind": "tpu",
                    "accelerator_type": "v5p-8",
                    "resources": {"tpu": 4},
                    "capacity_class": "spot",
                    "min_workers": 0,
                    "max_workers": 1,
                    "provider_options": {"bootstrap_egress": "public-internet"},
                    "runtime": {
                        "python": "3.11",
                        "env": {"POOL_SETTING": "preserve-me"},
                        "init_commands": ["echo pool-bootstrap"],
                    },
                }
            },
        }
    )


@pytest.mark.parametrize("fleet_python", [None, "3.13"])
@pytest.mark.parametrize("append_command", [False, True])
def test_up_dry_run_preserves_pool_python_override(
    tmp_path: Path, fleet_python: str | None, append_command: bool
) -> None:
    config = _fleet(fleet_python)
    before = config.model_dump(mode="json")
    config_path = tmp_path / "fleet.yaml"
    config_path.write_text(yaml.safe_dump(before), encoding="utf-8")
    args = [str(config_path), "--dry-run", "--json"]
    if append_command:
        args.extend(["--init-command", "echo cli-bootstrap"])
    session = _session(tmp_path)

    result = CliRunner().invoke(fleet_commands.up_fleet, args, obj=session)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    expected_runtime = config.pool_payloads()[0]["runtime"]
    assert isinstance(expected_runtime, dict)
    expected_runtime = dict(expected_runtime)
    if append_command:
        expected_runtime["init_commands"] = [*expected_runtime["init_commands"], "echo cli-bootstrap"]
    assert payload["pools"][0]["runtime"] == expected_runtime
    assert payload["pools"][0]["runtime"]["python"] == "3.11"
    assert config.model_dump(mode="json") == before
    assert session.client.calls == []


def test_init_command_update_does_not_mutate_source_runtime() -> None:
    config = _fleet("3.13")
    before = config.model_dump(mode="json")

    updated = fleet_commands._with_init_commands(config, ("echo cli-bootstrap",))

    assert config.model_dump(mode="json") == before
    original = config.pools["tpu"].runtime
    actual = updated.pools["tpu"].runtime
    assert original is not None and actual is not None
    expected = original.model_dump(mode="json")
    expected["init_commands"].append("echo cli-bootstrap")
    assert actual.model_dump(mode="json") == expected


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
def test_region_up_emits_one_json_document_or_human_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool, as_json: bool
) -> None:
    config = _fleet()
    cache_dir = tmp_path / "autoscale"
    entry = CachedAutoscaleConfig(path=cache_dir / "proof-fleet.yaml", config=config, quota=None, generated_at=None)
    generation = AutoscaleGenerationResult(entries=(entry,), quotas=(), skipped=(), cached=not dry_run)
    defaults = AutoscaleDefaults(project="proof-project")
    monkeypatch.setattr(fleet_commands, "_zones_for_region", lambda region, project: ("us-central1-a",))
    monkeypatch.setattr(
        fleet_commands.autoscale_commands,
        "load_autoscale_defaults",
        lambda cache, defaults_path, *, persist_if_prompted: defaults,
    )
    monkeypatch.setattr(fleet_commands.autoscale_commands, "generate_autoscale_entries", lambda **kwargs: generation)
    monkeypatch.setattr(fleet_commands.autoscale_commands, "is_interactive", lambda: False)
    session = _session(tmp_path)
    args = ["--region", "us-central1", "--cache-dir", str(cache_dir)]
    if as_json:
        args.append("--json")
    if dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(fleet_commands.up_fleet, args, obj=session)

    assert result.exit_code == 0, result.output
    if as_json:
        # Parse all stdout, not just the first document or a selected line.
        payload = json.loads(result.stdout)
        assert payload["autoscale"] == generation.as_dict()
        if dry_run:
            assert payload["fleet"] == config.name
        else:
            assert payload == {"apply": {"name": config.name, "pools": []}, "autoscale": generation.as_dict()}
        assert "Server is running" not in result.stdout
    elif not dry_run:
        assert "No enabled capacity pools available." in result.stdout
        assert "Server is running. Submit with:" in result.stdout
        assert "eyvan up --region us-central1" in result.stdout
    else:
        assert "Would apply fleet proof-fleet" in result.stdout
        assert "Server is running" not in result.stdout
    if dry_run:
        assert session.client.calls == []
        assert not session.context_store.path.exists()
    else:
        assert session.context_config.active_context == config.name
        assert session.context_store.path.exists()
        assert session.client.calls[0][:2] == ("PUT", "/api/v1/fleets/proof-fleet")
        applied = session.client.calls[0][2]
        assert applied is not None
        assert applied["pools"][0]["runtime"]["python"] == "3.11"
        if not as_json:
            assert session.client.calls[1][:2] == ("GET", "/api/v1/capacity")
