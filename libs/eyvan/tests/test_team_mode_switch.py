# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team mode is switched on by ``eyvan-server --team`` or ``team: true`` in the platform file, never by env.

Every surface that asks "is this a team plane" -- the server, ``bootstrap-admin``
and ``eyvan doctor`` -- resolves the answer the same way, so a team mode that only
the platform file turned on is refused, bootstrapped and preflighted identically.
"""

from __future__ import annotations

import os
import sys
import typing as tp
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from test_startup_preflight import _requirement_names, _session

from eyvan._internal.platform import EyvanPlatformConfig, load_platform_config, resolve_team_mode
from eyvan.config import EyvanSettings
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands.doctor import SERVER_STARTUP_CHECK, CheckStatus, DoctorCommandService, doctor
from eyvan.server.api import app as app_module
from eyvan.server.api import bootstrap_admin as bootstrap_module
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.api.server_startup import ServerStartupSettings, server_environment_settings
from eyvan.server.local import daemon
from eyvan.server.persistence.memory import InMemoryRepository

_NETWORK = ServerBindPolicy.from_host("0.0.0.0")

#: Cleared before each case so the developer's own exported environment cannot decide an outcome.
_VARIABLES = (
    "EYVAN_TEAM_MODE",
    "EYVAN_TLS_CERT_FILE",
    "EYVAN_TLS_KEY_FILE",
    "EYVAN_INSECURE_PLAINTEXT_BIND",
    "EYVAN_REPOSITORY_BACKEND",
    "EYVAN_METRICS_TOKEN",
    "EYVAN_ARTIFACT_TOKEN_KEY",
    "EYVAN_DEV_AUTH_ENABLED",
    "EYVAN_PUBLIC_URL",
    "EYVAN_CONTROL_PLANE_CA_FILE",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point every surface at a platform file under tmp_path (absent until a case writes it)."""

    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    platform = tmp_path / "platform.yaml"
    monkeypatch.setenv("EYVAN_PLATFORM_CONFIG_PATH", str(platform))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "artifacts"))
    monkeypatch.setattr("eyvan.sdk.commands.doctor.stale_components", lambda **_kwargs: ())
    monkeypatch.setattr("eyvan.sdk.commands.doctor_preflights.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "eyvan.sdk.commands.doctor_preflights.DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH",
        tmp_path / "no-runtime-profile.json",
    )
    return platform


def _write_team_file(platform: Path, team: bool = True) -> None:
    platform.write_text(f"version: 1\nteam: {'true' if team else 'false'}\n", encoding="utf-8")


def _network_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Satisfy every network-bind requirement except a usable public URL."""

    monkeypatch.setenv("EYVAN_TLS_CERT_FILE", str(tmp_path / "cert.pem"))
    monkeypatch.setenv("EYVAN_TLS_KEY_FILE", str(tmp_path / "key.pem"))
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "postgres")
    monkeypatch.setenv("EYVAN_METRICS_TOKEN", "scrape-token")
    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_KEY", "k" * 32)


# ------------------------------------------------------------- platform file


def test_the_platform_file_accepts_a_boolean_team_key() -> None:
    assert EyvanPlatformConfig.from_document({"version": 1, "team": True}).team is True
    assert EyvanPlatformConfig.from_document({"team": False}).team is False
    with_agent = EyvanPlatformConfig.from_document(
        {"version": 1, "team": True, "agents": {"default": {"endpoint": "https://agent.example.com"}}}
    )
    assert with_agent == EyvanPlatformConfig(default_agent_endpoint="https://agent.example.com", team=True)


@pytest.mark.parametrize("value", ["true", "yes", 1, 0, None, [], {}])
def test_the_team_key_must_be_a_real_boolean(value: tp.Any) -> None:
    with pytest.raises(ValueError, match="team must be true or false"):
        EyvanPlatformConfig.from_document({"version": 1, "team": value})


def test_unknown_platform_keys_are_still_refused() -> None:
    with pytest.raises(ValueError, match="unknown Eyvan platform field 'team_mode'"):
        EyvanPlatformConfig.from_document({"version": 1, "team_mode": True})


def test_existing_platform_documents_parse_unchanged(tmp_path: Path) -> None:
    assert EyvanPlatformConfig.from_document({}) == EyvanPlatformConfig()
    path = tmp_path / "old.yaml"
    path.write_text("version: 1\nagents:\n  default:\n    endpoint: https://agent.example.com\n", encoding="utf-8")
    assert load_platform_config(path) == EyvanPlatformConfig(default_agent_endpoint="https://agent.example.com")
    assert load_platform_config(path).team is False


def test_a_quoted_team_value_in_a_real_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "quoted.yaml"
    path.write_text('version: 1\nteam: "true"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="team must be true or false"):
        load_platform_config(path)


# ------------------------------------------------------------------ resolver


@pytest.mark.parametrize(
    ("flag", "file_team", "expected"),
    [(False, False, False), (True, False, True), (False, True, True), (True, True, True)],
    ids=["neither", "flag-only", "file-only", "both"],
)
def test_resolve_team_mode(flag: bool, file_team: bool, expected: bool) -> None:
    resolved = resolve_team_mode(EyvanSettings(), flag=flag, platform=EyvanPlatformConfig(team=file_team))
    assert resolved.team_mode is expected


def test_resolve_team_mode_keeps_explicit_settings() -> None:
    resolved = resolve_team_mode(EyvanSettings(team_mode=True), flag=False, platform=EyvanPlatformConfig())
    assert resolved.team_mode is True


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_the_removed_environment_variable_refuses_instead_of_silently_ignoring(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A 0.0.3 plane still exporting it must not come back with team accounts quietly off."""

    monkeypatch.setenv("EYVAN_TEAM_MODE", value)
    # The settings object never reads it: it turns nothing on.
    assert EyvanSettings.from_env().team_mode is False
    for resolve in (
        server_environment_settings,
        lambda: ServerStartupSettings.from_environment(ServerBindPolicy.from_host("127.0.0.1")),
        lambda: ServerStartupSettings.from_environment(_NETWORK, team=True),
    ):
        with pytest.raises(ValueError, match="--team") as refused:
            resolve()
        assert "team: true" in str(refused.value)


def test_server_environment_settings_reads_the_file_and_the_flag(_isolated: Path) -> None:
    assert server_environment_settings().team_mode is False
    assert server_environment_settings(team=True).team_mode is True
    _write_team_file(_isolated)
    assert server_environment_settings().team_mode is True


# ----------------------------------------------------------- startup policy


def test_a_file_only_team_plane_is_refused_without_a_public_url_on_a_network_bind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: Path
) -> None:
    _network_ready(monkeypatch, tmp_path)
    # Without team mode the same environment starts: the public URL rule is team-only.
    assert ServerStartupSettings.from_environment(_NETWORK).settings.team_mode is False

    _write_team_file(_isolated)
    with pytest.raises(ValueError, match="EYVAN_PUBLIC_URL") as refused:
        ServerStartupSettings.from_environment(_NETWORK)
    assert _requirement_names(str(refused.value)) == ("EYVAN_PUBLIC_URL",)

    monkeypatch.setenv("EYVAN_PUBLIC_URL", "https://plane.example.com")
    assert ServerStartupSettings.from_environment(_NETWORK).settings.team_mode is True


def test_team_false_in_the_file_leaves_team_mode_off(monkeypatch: pytest.MonkeyPatch, tmp_path, _isolated) -> None:
    _network_ready(monkeypatch, tmp_path)
    _write_team_file(_isolated, team=False)
    assert ServerStartupSettings.from_environment(_NETWORK).settings.team_mode is False


# ------------------------------------------------------------- eyvan-server


class _Served:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, application: object, **options: object) -> None:
        self.calls.append((application, options))


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> _Served:
    served = _Served()
    monkeypatch.setattr(sys, "argv", ["eyvan-server", *argv])
    monkeypatch.setattr(app_module.uvicorn, "run", served)
    app_module.main()
    return served


def _served_app(served: _Served) -> FastAPI:
    assert len(served.calls) == 1
    application, options = served.calls[0]
    # An app instance, not an import-string factory that would re-read the environment.
    assert isinstance(application, FastAPI)
    assert "factory" not in options and "reload" not in options
    return application


@pytest.mark.parametrize("flag", ["--team", "--insecure-plaintext"])
def test_reload_refuses_flags_it_cannot_carry(monkeypatch: pytest.MonkeyPatch, capsys, flag: str) -> None:
    served = _Served()
    monkeypatch.setattr(sys, "argv", ["eyvan-server", "--reload", flag])
    monkeypatch.setattr(app_module.uvicorn, "run", served)
    with pytest.raises(SystemExit) as exited:
        app_module.main()
    assert exited.value.code == 2
    assert served.calls == []
    assert "--reload does not support --team or --insecure-plaintext" in capsys.readouterr().err


def test_the_team_flag_serves_a_team_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "memory")
    application = _served_app(_run_main(monkeypatch, "--team"))
    assert application.state.server_startup.settings.team_mode is True
    assert application.state.team_mode is True


def test_the_platform_file_alone_serves_a_team_plane(monkeypatch: pytest.MonkeyPatch, _isolated: Path) -> None:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "memory")
    _write_team_file(_isolated)
    application = _served_app(_run_main(monkeypatch))
    assert application.state.team_mode is True


def test_neither_flag_nor_file_serves_a_plain_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "memory")
    application = _served_app(_run_main(monkeypatch))
    assert application.state.team_mode is False


def test_eyvan_server_refuses_to_start_with_the_removed_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "memory")
    monkeypatch.setenv("EYVAN_TEAM_MODE", "1")
    served = _Served()
    monkeypatch.setattr(sys, "argv", ["eyvan-server", "--team"])
    monkeypatch.setattr(app_module.uvicorn, "run", served)
    with pytest.raises(SystemExit) as exited:
        app_module.main()
    assert "EYVAN_TEAM_MODE" in str(exited.value.code) and "--team" in str(exited.value.code)
    assert served.calls == []


def test_a_file_only_team_plane_refuses_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: Path
) -> None:
    _network_ready(monkeypatch, tmp_path)
    _write_team_file(_isolated)
    served = _Served()
    monkeypatch.setattr(sys, "argv", ["eyvan-server", "--host", "0.0.0.0"])
    monkeypatch.setattr(app_module.uvicorn, "run", served)
    with pytest.raises(SystemExit) as exited:
        app_module.main()
    assert "EYVAN_PUBLIC_URL" in str(exited.value.code)
    assert served.calls == []


def test_insecure_plaintext_takes_effect_without_writing_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "postgres")
    monkeypatch.setenv("EYVAN_METRICS_TOKEN", "scrape-token")
    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_KEY", "k" * 32)
    monkeypatch.setenv("EYVAN_CONTROLLER_ENABLED", "0")
    monkeypatch.setattr(app_module, "create_async_engine", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "SqlAlchemyRepository", lambda _engine: InMemoryRepository())
    before = dict(os.environ)

    application = _served_app(_run_main(monkeypatch, "--host", "0.0.0.0", "--insecure-plaintext"))

    assert dict(os.environ) == before
    assert "EYVAN_INSECURE_PLAINTEXT_BIND" not in os.environ
    settings = application.state.server_startup.settings
    assert settings.allow_insecure_plaintext_bind is True


def test_a_network_bind_without_the_opt_in_still_refuses_plaintext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "postgres")
    monkeypatch.setenv("EYVAN_METRICS_TOKEN", "scrape-token")
    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_KEY", "k" * 32)
    served = _Served()
    monkeypatch.setattr(sys, "argv", ["eyvan-server", "--host", "0.0.0.0"])
    monkeypatch.setattr(app_module.uvicorn, "run", served)
    with pytest.raises(SystemExit) as exited:
        app_module.main()
    assert "EYVAN_TLS_CERT_FILE" in str(exited.value.code)
    assert served.calls == []


# ----------------------------------------------------------- bootstrap-admin


def _bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> int:
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "local")
    monkeypatch.setenv("EYVAN_PUBLIC_URL", "https://plane.example.com")
    return bootstrap_module.main(
        ["--username", "erfan", "--local-state-path", str(tmp_path / "control-plane.json"), "--json", *extra]
    )


def test_bootstrap_admin_refuses_a_plane_without_team_mode(monkeypatch, tmp_path: Path, capsys) -> None:
    assert _bootstrap(monkeypatch, tmp_path) == 1
    captured = capsys.readouterr()
    assert "--team" in captured.err and "team: true" in captured.err
    assert "EYVAN_TEAM_MODE" not in captured.err
    assert "eyvinv_" not in captured.out


def test_bootstrap_admin_refuses_while_the_removed_variable_is_set(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("EYVAN_TEAM_MODE", "1")
    assert _bootstrap(monkeypatch, tmp_path, "--team") == 1
    captured = capsys.readouterr()
    assert "EYVAN_TEAM_MODE" in captured.err and "--team" in captured.err
    assert "eyvinv_" not in captured.out


def test_bootstrap_admin_honours_the_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    assert _bootstrap(monkeypatch, tmp_path, "--team") == 0
    assert capsys.readouterr().out.count("eyvinv_") == 1


def test_bootstrap_admin_honours_the_platform_file(monkeypatch, tmp_path: Path, capsys, _isolated: Path) -> None:
    _write_team_file(_isolated)
    assert _bootstrap(monkeypatch, tmp_path) == 0
    assert capsys.readouterr().out.count("eyvinv_") == 1


def test_bootstrap_admin_reports_a_malformed_platform_file(monkeypatch, tmp_path, capsys, _isolated: Path) -> None:
    _isolated.write_text("team: 1\n", encoding="utf-8")
    assert _bootstrap(monkeypatch, tmp_path, "--team") == 1
    assert "team must be true or false" in capsys.readouterr().err


# -------------------------------------------------------------------- doctor


def _doctor_startup(*, team: bool = False):
    report = DoctorCommandService(_session(), server_bind_host="0.0.0.0", team=team).report()
    return next(check for check in report.checks if check.name == SERVER_STARTUP_CHECK)


def _server_names(*, team: bool = False) -> tuple[str, ...]:
    try:
        ServerStartupSettings.from_environment(_NETWORK, team=team)
    except ValueError as error:
        return _requirement_names(str(error))
    return ()


def test_doctor_honours_the_team_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _network_ready(monkeypatch, tmp_path)
    plain = _doctor_startup()
    assert plain.status is CheckStatus.PASS

    team = _doctor_startup(team=True)
    assert team.status is CheckStatus.FAIL
    assert _requirement_names(team.remedy) == ("EYVAN_PUBLIC_URL",) == _server_names(team=True)


def test_doctor_honours_the_platform_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated: Path) -> None:
    _network_ready(monkeypatch, tmp_path)
    _write_team_file(_isolated)
    check = _doctor_startup()
    assert check.status is CheckStatus.FAIL
    assert _requirement_names(check.remedy) == ("EYVAN_PUBLIC_URL",) == _server_names()


def test_doctor_fails_while_the_removed_environment_variable_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _network_ready(monkeypatch, tmp_path)
    monkeypatch.setenv("EYVAN_TEAM_MODE", "1")
    check = _doctor_startup()
    assert check.status is CheckStatus.FAIL
    assert "EYVAN_TEAM_MODE" in f"{check.detail} {check.remedy}"


def test_doctor_cli_takes_team(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _network_ready(monkeypatch, tmp_path)
    result = CliRunner().invoke(doctor, ["--server-host", "0.0.0.0", "--team"], obj=_session())
    assert result.exit_code == 1
    assert "EYVAN_PUBLIC_URL" in result.stdout
    assert CliRunner().invoke(doctor, ["--server-host", "0.0.0.0"], obj=_session()).exit_code == 0


def test_doctor_reports_a_malformed_platform_file(monkeypatch, tmp_path: Path, _isolated: Path) -> None:
    _network_ready(monkeypatch, tmp_path)
    _isolated.write_text("team: yes-please\n", encoding="utf-8")
    check = _doctor_startup()
    assert check.status is CheckStatus.FAIL
    assert "platform file does not parse" in check.detail


# -------------------------------------------------------------- local daemon


def _command(tmp_path: Path, **options: tp.Any) -> daemon.LocalDaemonCommand:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19085")
    target = daemon.LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    paths = daemon.LocalDaemonPaths(tmp_path / "state")
    return daemon.LocalDaemonCommand(target, paths.log_path(target), paths.profile(target), **options)


@pytest.mark.parametrize("runtime_profile", [False, True])
def test_the_daemon_command_passes_team_and_stays_owned(tmp_path: Path, runtime_profile: bool) -> None:
    profile = (tmp_path / "runtime.json").resolve() if runtime_profile else None
    plain = _command(tmp_path, runtime_profile_path=profile)
    team = _command(tmp_path, runtime_profile_path=profile, team=True)

    assert "--team" not in plain.argv
    assert team.argv.count("--team") == 1
    assert team.argv[-2:] == ("--log-level", "info")
    assert tuple(a for a in team.argv if a != "--team") == plain.argv
    for argv in (plain.argv, team.argv):
        assert daemon._matches_local_daemon_process(
            argv,
            plain.target,
            runtime_profile_path=None if profile is None else str(profile),
            require_local_daemon=True,
        )


def test_a_team_local_daemon_serves_team_routes(tmp_path: Path) -> None:
    profile = tmp_path / "profiles" / "19086"
    kwargs: dict[str, tp.Any] = {
        "server_bind": ServerBindPolicy.from_host("127.0.0.1"),
        "port": 19086,
        "public_url": "http://127.0.0.1:19086",
        "local_state_path": profile / "control-plane.json",
        "local_artifact_token_key_path": profile / "artifact-token.key",
        "artifact_store_uri": str(profile / "managed-artifacts"),
    }
    assert app_module.create_local_daemon_app(**kwargs, team=True).state.team_mode is True
    assert app_module.create_local_daemon_app(**kwargs).state.team_mode is False


def test_the_local_daemon_refuses_while_the_removed_variable_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    profile = tmp_path / "profiles" / "19087"
    monkeypatch.setenv("EYVAN_TEAM_MODE", "1")
    served = _Served()
    monkeypatch.setattr(app_module.uvicorn, "run", served)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eyvan-server",
            "--host",
            "127.0.0.1",
            "--port",
            "19087",
            "--local-daemon",
            "--team",
            "--public-url",
            "http://127.0.0.1:19087",
            "--local-state-path",
            str(profile / "control-plane.json"),
            "--local-artifact-token-key-path",
            str(profile / "artifact-token.key"),
            "--artifact-store-uri",
            str(profile / "managed-artifacts"),
        ],
    )
    with pytest.raises(SystemExit) as exited:
        app_module.main()
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "EYVAN_TEAM_MODE" in err and "--team" in err
    assert served.calls == []
    assert not (profile / "control-plane.json").exists()
