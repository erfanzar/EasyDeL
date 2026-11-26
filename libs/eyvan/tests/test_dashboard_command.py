# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Installed-wheel and source-development dashboard command contracts."""

from __future__ import annotations

import typing as tp
from urllib.parse import quote

from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.dashboard as dashboard_commands
from eyvan.sdk.cli import main
from eyvan.sdk.commands.http import EyvanApiError
from eyvan.server.local.auth import LocalDevelopmentCredentialStore


def _env(output: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in output.strip().splitlines() if "=" in line)


class _RaisingClient:
    """A client whose GET / answers with one fixed API error."""

    def __init__(self, status_code: int) -> None:
        self._error = EyvanApiError(status_code, "no such route")

    def request(self, method: str, path: str, **_kwargs: tp.Any) -> tp.Any:
        del method, path
        raise self._error

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        del headers


def test_a_404_on_the_probe_names_the_missing_dashboard(monkeypatch) -> None:
    """The wheel-install answer: the plane is up, its API answers, it hosts no UI.

    The probe used to read ``.status_code`` off a decoded JSON document, so the
    404 always drowned in the generic handler and the operator got the bare 404
    in a browser instead of this guidance.
    """

    opened: list[str] = []
    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: _RaisingClient(404))

    result = CliRunner().invoke(main, ["--url", "https://eyvan.example.com", "dashboard"])

    assert result.exit_code != 0
    assert "hosts no dashboard" in result.output
    assert "dashboard-dev" in result.output
    assert opened == []


def test_any_other_probe_failure_still_opens_the_dashboard(monkeypatch) -> None:
    """Only a 404 means "no dashboard here"; a 403 is a credential question."""

    opened: list[str] = []
    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: _RaisingClient(403))

    result = CliRunner().invoke(main, ["--url", "https://eyvan.example.com", "dashboard", "--no-open"])

    assert result.exit_code == 0, result.output
    assert opened == []


def test_dashboard_opens_the_hosted_control_plane_url(monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)

    result = CliRunner().invoke(main, ["--url", "https://eyvan.example.com", "dashboard"])

    assert result.exit_code == 0, result.output
    assert opened == ["https://eyvan.example.com/"]
    assert "https://eyvan.example.com/" in result.output


def test_dashboard_is_independent_of_source_checkout(monkeypatch) -> None:
    """An isolated wheel has no sibling dashboard/package.json directory."""

    def no_sources() -> None:
        raise AssertionError("the hosted command must not look for dashboard sources")

    monkeypatch.setattr(dashboard_commands, "_dashboard_root", no_sources)
    result = CliRunner().invoke(
        main,
        ["--url", "https://eyvan.example.com/control-plane", "dashboard", "--no-open"],
    )

    assert result.exit_code == 0, result.output
    assert "https://eyvan.example.com/control-plane/" in result.output


def test_dashboard_dev_targets_this_session_not_a_hardcoded_port() -> None:
    """`eyvan runtime` picks a per-profile port; 8080 is frequently someone else."""

    result = CliRunner().invoke(
        main,
        [
            "--url",
            "http://127.0.0.1:8099",
            "--org",
            "acme",
            "--project",
            "research",
            "dashboard-dev",
            "--print-env",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _env(result.output)["EYVAN_DASHBOARD_API"] == "http://127.0.0.1:8099"


def test_development_auth_is_enabled_explicitly() -> None:
    """Without the flag `apiAuthHeaders()` returns {} and every request 401s."""

    result = CliRunner().invoke(main, ["--url", "http://127.0.0.1:8099", "dashboard-dev", "--print-env"])

    assert _env(result.output)["VITE_EYVAN_DEV_AUTH"] == "true"


def test_the_scope_comes_from_the_session() -> None:
    """A dashboard pointed at a different project shows an empty, plausible page."""

    result = CliRunner().invoke(
        main,
        [
            "--url",
            "http://127.0.0.1:8099",
            "--org",
            "acme",
            "--project",
            "research",
            "dashboard-dev",
            "--print-env",
        ],
    )

    environment = _env(result.output)
    assert environment["VITE_EYVAN_ORG"] == "acme"
    assert environment["VITE_EYVAN_PROJECT"] == "research"


def test_the_role_carries_over_so_the_fleet_page_is_populated() -> None:
    """Cells, pools and leases all require manage_fleet; `viewer` sees nothing."""

    result = CliRunner().invoke(
        main,
        ["--url", "http://127.0.0.1:8099", "--role", "fleet_admin", "dashboard-dev", "--print-env"],
    )

    assert _env(result.output)["VITE_EYVAN_DEV_ROLE"] == "fleet_admin"


def test_the_credential_is_never_printed(tmp_path) -> None:
    """`--print-env` output is for pasting into a shell, and shells keep history."""

    credential = tmp_path / "credential"
    credential.write_text("super-secret-value", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "--url",
            "http://127.0.0.1:8099",
            "dashboard-dev",
            "--print-env",
            "--credential-path",
            str(credential),
        ],
    )

    assert "super-secret-value" not in result.output
    assert _env(result.output)["VITE_EYVAN_LOCAL_CREDENTIAL"] == "<redacted>"


def test_a_missing_credential_is_omitted_rather_than_sent_empty(tmp_path) -> None:
    """An empty header is a request that fails confusingly instead of loudly."""

    result = CliRunner().invoke(
        main,
        [
            "--url",
            "http://127.0.0.1:8099",
            "dashboard-dev",
            "--print-env",
            "--credential-path",
            str(tmp_path / "absent"),
        ],
    )

    assert "VITE_EYVAN_LOCAL_CREDENTIAL" not in _env(result.output)


def test_dashboard_dev_fails_explicitly_without_source_files(monkeypatch) -> None:
    def missing() -> None:
        raise dashboard_commands.click.ClickException(
            "'eyvan dashboard-dev' requires a source checkout; 'eyvan dashboard' opens the hosted production UI"
        )

    monkeypatch.setattr(dashboard_commands, "_dashboard_root", missing)
    result = CliRunner().invoke(main, ["dashboard-dev", "--print-env"])

    assert result.exit_code != 0
    assert "requires a source checkout" in result.output
    assert "opens the hosted production UI" in result.output


def test_a_loopback_plane_is_opened_with_the_credential_the_browser_cannot_read(tmp_path, monkeypatch) -> None:
    """A browser cannot read the 0600 file a loopback plane demands; a terminal can.

    The plane refuses its own dashboard's calls without the owner-only
    credential, and the only sound way into the browser is through the owner's
    terminal: the value rides in the URL fragment, which no server ever
    receives, and the page strips it on arrival.
    """

    monkeypatch.setenv("HOME", str(tmp_path))
    credential = LocalDevelopmentCredentialStore().load_or_create().value
    opened: list[str] = []
    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)

    result = CliRunner().invoke(main, ["--url", "http://127.0.0.1:8099", "dashboard"])

    assert result.exit_code == 0, result.output
    expected = f"http://127.0.0.1:8099/#eyvan-local-credential={quote(credential)}"
    assert opened == [expected]
    assert expected in result.output


def test_a_loopback_plane_with_no_credential_yet_gets_a_plain_url(tmp_path, monkeypatch) -> None:
    """Nothing has been issued, so there is nothing to demand and nothing to bridge."""

    monkeypatch.setenv("HOME", str(tmp_path))
    opened: list[str] = []
    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)

    result = CliRunner().invoke(main, ["--url", "http://127.0.0.1:8099", "dashboard"])

    assert result.exit_code == 0, result.output
    assert opened == ["http://127.0.0.1:8099/"]
    assert "eyvan-local-credential" not in result.output


def test_the_local_credential_never_leaves_this_machine(tmp_path, monkeypatch) -> None:
    """A remote plane gets a plain URL even when a local credential exists.

    The credential authenticates this machine's owner to this machine's plane;
    printing it into a URL for a different host would hand it to whatever is
    listening there and paste it into a foreign terminal's scrollback.
    """

    monkeypatch.setenv("HOME", str(tmp_path))
    credential = LocalDevelopmentCredentialStore().load_or_create().value
    opened: list[str] = []
    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)

    result = CliRunner().invoke(main, ["--url", "https://eyvan.example.com", "dashboard"])

    assert result.exit_code == 0, result.output
    assert opened == ["https://eyvan.example.com/"]
    assert "eyvan-local-credential" not in result.output
    assert credential not in result.output
