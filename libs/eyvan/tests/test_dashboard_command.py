# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Installed-wheel and source-development dashboard command contracts."""

from __future__ import annotations

import typing as tp
from urllib.parse import quote

import httpx
from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.dashboard as dashboard_commands
from eyvan.sdk.cli import main
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands.http import EyvanApiError, EyvanCliUnavailableError
from eyvan.server.local.auth import LocalDevelopmentCredentialStore
from eyvan.server.local.daemon import LocalDaemonResult, LocalDaemonState
from eyvan.server.local.tracker import LocalTrackerError, TrackerRecord


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


def _healthy_tracker_record(tmp_path) -> TrackerRecord:
    return TrackerRecord(
        url="http://127.0.0.1:8095",
        pid=4242,
        store=str(tmp_path / "tracking"),
        db=str(tmp_path / "tracker" / "index.sqlite3"),
        log=str(tmp_path / "tracker" / "tracker.log"),
        token_path=str(tmp_path / "tracker" / "token"),
    )


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
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: _RaisingClient(200))
    monkeypatch.setattr(dashboard_commands, "_ensure_local_tracker", lambda: (_healthy_tracker_record(tmp_path), False))

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
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: _RaisingClient(200))
    monkeypatch.setattr(dashboard_commands, "_ensure_local_tracker", lambda: (_healthy_tracker_record(tmp_path), False))

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

    def forbidden(*_args, **_kwargs):
        raise AssertionError("remote dashboard must never auto-start loopback services")

    monkeypatch.setattr(dashboard_commands, "_ensure_loopback_plane", forbidden)
    monkeypatch.setattr(dashboard_commands, "_ensure_local_tracker", forbidden)

    result = CliRunner().invoke(main, ["--url", "https://eyvan.example.com", "dashboard"])

    assert result.exit_code == 0, result.output
    assert opened == ["https://eyvan.example.com/"]
    assert "eyvan-local-credential" not in result.output
    assert credential not in result.output


def test_dashboard_auto_bootstraps_unreachable_loopback_plane_tracker_and_credential(tmp_path, monkeypatch) -> None:
    """One-command `eyvan dashboard` boots an unreachable loopback plane, refreshes credential, and starts tracker."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EYVAN_TRACKER_URL", raising=False)
    opened: list[str] = []
    plane_starts: list[str] = []
    tracker_starts: list[bool] = []
    minted_credential: list[str] = []
    probe_calls = 0

    class _BootstrappingClient:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def request(self, method: str, path: str, **_kwargs: tp.Any) -> tp.Any:
            nonlocal probe_calls
            del method, path
            probe_calls += 1
            if probe_calls == 1:
                raise EyvanCliUnavailableError(ControlPlaneEndpoint.parse("http://127.0.0.1:8099"), "GET")
            raise EyvanApiError(200, "control plane returned a non-JSON response body: '<html>'")

        def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
            self.headers = dict(headers)

    def fake_ensure_plane(endpoint: ControlPlaneEndpoint) -> LocalDaemonResult:
        plane_starts.append(endpoint.url)
        minted = LocalDevelopmentCredentialStore().load_or_create().value
        minted_credential.append(minted)
        return LocalDaemonResult(endpoint=endpoint, state=LocalDaemonState.STARTED, pid=1234, log_path=None)

    def fake_ensure_tracker() -> tuple[TrackerRecord, bool]:
        tracker_starts.append(True)
        return _healthy_tracker_record(tmp_path), True

    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: _BootstrappingClient())
    monkeypatch.setattr(dashboard_commands, "_ensure_loopback_plane", fake_ensure_plane)
    monkeypatch.setattr(dashboard_commands, "_ensure_local_tracker", fake_ensure_tracker)

    result = CliRunner().invoke(main, ["--url", "http://127.0.0.1:8099", "dashboard"])

    assert result.exit_code == 0, result.output
    assert plane_starts == ["http://127.0.0.1:8099"]
    assert tracker_starts == [True]
    assert "Started local Eyvan control plane at http://127.0.0.1:8099" in result.output
    assert "Started local Eyvan run tracker at http://127.0.0.1:8095" in result.output
    expected = f"http://127.0.0.1:8099/#eyvan-local-credential={quote(minted_credential[0])}"
    assert opened == [expected]


def test_dashboard_no_auto_start_and_nonfatal_tracker_warning(tmp_path, monkeypatch) -> None:
    """`--no-auto-start` skips bootstrap; tracker errors warn without blocking dashboard launch."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EYVAN_TRACKER_URL", raising=False)
    opened: list[str] = []

    class _ConnectErrorClient:
        def request(self, method: str, path: str, **_kwargs: tp.Any) -> tp.Any:
            del method, path
            raise httpx.ConnectError("connection refused")

        def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
            del headers

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must not auto-start when --no-auto-start is passed")

    monkeypatch.setattr(dashboard_commands.click, "launch", opened.append)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: _ConnectErrorClient())
    monkeypatch.setattr(dashboard_commands, "_ensure_loopback_plane", forbidden)
    monkeypatch.setattr(dashboard_commands, "_ensure_local_tracker", forbidden)

    no_auto = CliRunner().invoke(main, ["--url", "http://127.0.0.1:8099", "dashboard", "--no-auto-start"])
    assert no_auto.exit_code == 0, no_auto.output
    assert opened == ["http://127.0.0.1:8099/"]

    # Now enable auto-start with failing tracker: plane boots, tracker warns, dashboard still opens!
    opened.clear()
    monkeypatch.setattr(
        dashboard_commands,
        "_ensure_loopback_plane",
        lambda ep: LocalDaemonResult(endpoint=ep, state=LocalDaemonState.REUSED, pid=1234, log_path=None),
    )

    def broken_tracker():
        raise LocalTrackerError("port 8095 in use")

    monkeypatch.setattr(dashboard_commands, "_ensure_local_tracker", broken_tracker)
    with_warn = CliRunner().invoke(main, ["--url", "http://127.0.0.1:8099", "dashboard"])
    assert with_warn.exit_code == 0, with_warn.output
    assert "Warning: could not start local run tracker" in with_warn.output
    assert opened == ["http://127.0.0.1:8099/"]
