# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Keep workstation owner credentials out of remote dashboard-dev launchers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.dashboard as dashboard_commands
from eyvan.server.local.auth import LocalDevelopmentCredentialStore


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    """Exercise the real CLI without starting npm or accessing operator state."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "discover_node_context", lambda: None)
    monkeypatch.setattr(cli, "discover_node_token_or_warn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(dashboard_commands.shutil, "which", lambda _name: "/fake/npm")
    launched: list[dict[str, str]] = []

    def run(args, *, cwd, env, check):
        assert args[:3] == ["/fake/npm", "run", "dev"]
        assert (Path(cwd) / "package.json").is_file()
        assert check is True
        launched.append(env)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(dashboard_commands.subprocess, "run", run)
    return tmp_path, launched


@pytest.mark.parametrize("url", ["https://remote.invalid:443", "http://127.0.0.1.remote.invalid:8099"])
@pytest.mark.parametrize("credential_exists", [False, True])
@pytest.mark.parametrize("inherited", [False, True])
def test_remote_dashboard_launch_has_no_workstation_owner_credential(
    launcher, monkeypatch, url, credential_exists, inherited
) -> None:
    home, launched = launcher
    if credential_exists:
        credential_path = home / ".eyvan" / "local-development-credential"
        credential_path.parent.mkdir()
        credential_path.write_text("fake-file-owner-credential", encoding="utf-8")
    if inherited:
        monkeypatch.setenv("VITE_EYVAN_LOCAL_CREDENTIAL", "fake-inherited-owner-credential")
    else:
        monkeypatch.delenv("VITE_EYVAN_LOCAL_CREDENTIAL", raising=False)

    result = CliRunner().invoke(cli.main, ["--url", url, "dashboard-dev", "--no-open"])

    assert result.exit_code == 0, result.output
    assert len(launched) == 1
    assert launched[0]["EYVAN_DASHBOARD_API"] == url
    assert launched[0]["VITE_EYVAN_DEV_AUTH"] == "true"
    assert "VITE_EYVAN_LOCAL_CREDENTIAL" not in launched[0]
    assert "fake-file-owner-credential" not in result.output
    assert "fake-inherited-owner-credential" not in result.output


@pytest.mark.parametrize("url", ["http://127.0.0.1:8099", "http://localhost:8099", "http://[::1]:8099"])
def test_loopback_dashboard_launch_still_receives_its_owner_credential(launcher, monkeypatch, url) -> None:
    _home, launched = launcher
    credential = LocalDevelopmentCredentialStore().load_or_create().value
    monkeypatch.setenv("VITE_EYVAN_LOCAL_CREDENTIAL", "fake-inherited-owner-credential")

    result = CliRunner().invoke(cli.main, ["--url", url, "dashboard-dev", "--no-open"])

    assert result.exit_code == 0, result.output
    assert len(launched) == 1
    assert launched[0]["EYVAN_DASHBOARD_API"] == url
    assert launched[0]["VITE_EYVAN_LOCAL_CREDENTIAL"] == credential
    assert credential not in result.output


def test_remote_print_env_does_not_offer_a_local_credential(launcher) -> None:
    home, launched = launcher
    credential_path = home / ".eyvan" / "local-development-credential"
    credential_path.parent.mkdir()
    credential_path.write_text("fake-file-owner-credential", encoding="utf-8")

    result = CliRunner().invoke(cli.main, ["--url", "https://remote.invalid", "dashboard-dev", "--print-env"])

    assert result.exit_code == 0, result.output
    assert "EYVAN_DASHBOARD_API=https://remote.invalid" in result.output
    assert "VITE_EYVAN_LOCAL_CREDENTIAL" not in result.output
    assert launched == []
