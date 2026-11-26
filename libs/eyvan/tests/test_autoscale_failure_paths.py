# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest
from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.autoscale as autoscale_commands
from eyvan.capacity.autoscale import AutoscaleConfigCache, AutoscaleDefaults
from eyvan.capacity.providers.gcp.quota_discovery import TpuQuotaAvailability


class _NoEligiblePoolDiscovery:
    """Return a successful provider refresh with only an ineligible worker."""

    def discover(
        self,
        *,
        project: str,
        zones: Iterable[str] = (),
        generations: Iterable[str] = (),
        accelerator_types: Iterable[str] = (),
        preemptible: bool = True,
    ) -> tuple[TpuQuotaAvailability, ...]:
        del project, zones, generations, accelerator_types, preemptible
        return (
            TpuQuotaAvailability(
                zone="us-east5-a",
                accelerator_type="v5p-16",
                generation="v5p",
                quota_tpus=15,
                available_tpus=15,
                preemptible=True,
                metric="tpu.googleapis.com/tpu-v5p-preemptible",
            ),
        )


def _cache_with_defaults(cache_dir: Path) -> AutoscaleConfigCache:
    cache = AutoscaleConfigCache(cache_dir)
    cache.save_defaults(AutoscaleDefaults(project="project-a", fleet_prefix="research"))
    return cache


def _configure_local_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "eyvan-config.json"))
    monkeypatch.setattr(autoscale_commands, "GcpTpuQuotaDiscovery", _NoEligiblePoolDiscovery)


def test_autoscale_generate_empty_result_keeps_json_stdout_and_reports_failure_on_stderr(monkeypatch, tmp_path) -> None:
    _configure_local_cli(monkeypatch, tmp_path)
    cache_dir = tmp_path / "autoscale"
    _cache_with_defaults(cache_dir)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "autoscale",
            "generate",
            "--json",
            "--zone",
            "us-east5-a",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["generated"] == []
    assert payload["skipped"] == [
        {
            "generation": "v5p",
            "reason": "available shared Spot quota cannot fit one complete worker",
            "scope": "generation",
            "type_count": 1,
            "zone": "us-east5-a",
        }
    ]
    assert result.stderr == "No eligible preemptible TPU worker pools were generated; inspect the emitted diagnostics.\n"


def test_autoscale_generate_empty_terminal_view_omits_empty_config_table(monkeypatch, tmp_path) -> None:
    _configure_local_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(autoscale_commands, "is_interactive_output", lambda: True)
    cache_dir = tmp_path / "autoscale"
    _cache_with_defaults(cache_dir)

    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "alice",
            "--role",
            "fleet_admin",
            "autoscale",
            "generate",
            "--zone",
            "us-east5-a",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "Eyvan TPU Autoscale" not in result.stdout
    assert "Skipped candidates" in result.stdout
    assert "available shared" in result.stdout
    assert result.stderr == "No eligible preemptible TPU worker pools were generated; inspect the emitted diagnostics.\n"
