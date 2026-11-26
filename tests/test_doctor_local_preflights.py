# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The five local preflight checks doctor gained: each answers, none guesses.

gcloud authentication, the Secret Manager bind permission, the artifact
store, the runtime profile, and control-plane port ownership all ask about
this machine, so they must answer even when no control plane is running —
and each outcome is actionable: a failure names the command that fixes it,
and an unanswerable check is a skip with the reason, never a silent pass.
"""

from __future__ import annotations

import typing as tp

import pytest

import eyvan.sdk.commands.doctor as doctor_module
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands import doctor_preflights as preflights
from eyvan.sdk.commands.doctor import CheckStatus, DoctorCommandService
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContextConfig
from tests.test_doctor import _healthy, _Store


class _FakeClient:
    def __init__(self, handler: tp.Callable[[str], tp.Any]) -> None:
        self._handler = handler

    def request(self, method: str, path: str, **_kwargs: tp.Any) -> tp.Any:
        del method
        return self._handler(path)


def _session(base_url: str = "http://127.0.0.1:8080", handler: tp.Callable[[str], tp.Any] | None = None) -> CliSession:
    return CliSession(
        client=_FakeClient(handler or _healthy()),
        identity=ClientIdentity(),
        context_store=_Store(),
        context_config=EyvanContextConfig(),
        base_url=base_url,
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
    )


def _check(report_checks: tp.Sequence[tp.Any], name: str) -> tp.Any:
    return next(check for check in report_checks if check.name == name)


@pytest.fixture(autouse=True)
def _isolated_local_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        preflights,
        "DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH",
        tmp_path / "absent.json",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "artifacts"))
    monkeypatch.setattr(preflights.socket.socket, "connect_ex", lambda _self, _address: 111)
    monkeypatch.setattr(preflights.socket.socket, "bind", lambda _self, _address: None)
    # Every test opts into the one boundary it exercises. The remaining
    # preflights must not invoke this developer's real cloud credentials.
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        preflights.GcloudNativeWorkerSecretStore,
        "caller_bind_state",
        lambda _self, *, project: (False, None),
    )
    # The doctor's own source-drift check must not read this machine's state.
    monkeypatch.setattr(doctor_module, "stale_components", lambda **_kwargs: ())


def test_gcloud_authentication_fails_with_the_login_remedy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: "/usr/bin/gcloud")
    monkeypatch.setattr(preflights, "gcloud_run", lambda _argv: (False, "reauth required"))

    check = _check(DoctorCommandService(_session()).report().checks, "gcloud-authentication")

    assert check.status is CheckStatus.FAIL
    assert "reauth required" in check.detail
    assert "gcloud auth application-default login" in check.remedy


def test_gcloud_authentication_without_gcloud_is_not_a_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: None)

    check = _check(DoctorCommandService(_session()).report().checks, "gcloud-authentication")

    # Not applicable is not sick: a GCP-less laptop must stay green. Doctor's
    # `ok` treats SKIP as not-ok, so the check passes with the reason in the
    # detail — the same discipline as the kueue check.
    assert check.status is CheckStatus.PASS
    assert "not applicable" in check.detail
    assert "not installed" in check.detail


def test_gcloud_authentication_passes_with_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: "/usr/bin/gcloud")
    monkeypatch.setattr(
        preflights,
        "gcloud_run",
        lambda argv: (True, "my-project") if "get-value" in argv else (True, "TOKEN"),
    )

    check = _check(DoctorCommandService(_session()).report().checks, "gcloud-authentication")

    assert check.status is CheckStatus.PASS
    assert "my-project" in check.detail
    # The token itself is a credential: it must never appear in a diagnostic.
    assert "TOKEN" not in check.detail


def test_secret_manager_bind_uses_the_existing_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: "/usr/bin/gcloud")
    monkeypatch.setattr(preflights, "gcloud_run", lambda _argv: (True, "my-project"))
    monkeypatch.setattr(
        preflights.GcloudNativeWorkerSecretStore,
        "caller_bind_state",
        lambda _self, *, project: (False, "missing: secretmanager.secrets.setIamPolicy — ask a project owner"),
    )

    check = _check(DoctorCommandService(_session()).report().checks, "secret-manager-bind-permission")

    assert check.status is CheckStatus.FAIL
    assert "setIamPolicy" in check.detail


def test_secret_manager_bind_unknown_is_a_warning_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: "/usr/bin/gcloud")
    monkeypatch.setattr(preflights, "gcloud_run", lambda _argv: (True, "my-project"))
    monkeypatch.setattr(
        preflights.GcloudNativeWorkerSecretStore,
        "caller_bind_state",
        lambda _self, *, project: (False, None),
    )

    check = _check(DoctorCommandService(_session()).report().checks, "secret-manager-bind-permission")

    assert check.status is CheckStatus.WARN
    assert "could not determine" in check.detail
    # Unknown must not hijack `next_step` from a concrete finding behind it.
    assert check.remedy == ""


def test_artifact_store_local_path_must_be_creatable(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # The plane creates the store directory on startup; a missing directory
    # under a writable parent is the normal pre-first-run state, not a fault.
    missing = tmp_path / "not-created-yet"
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(missing))

    check = _check(DoctorCommandService(_session()).report().checks, "artifact-store")

    assert check.status is CheckStatus.PASS
    assert "creates it on startup" in check.detail


def test_artifact_store_local_path_under_an_unwritable_parent_fails(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    blocked = tmp_path / "read-only-parent" / "managed"
    blocked.parent.mkdir()
    blocked.parent.chmod(0o500)
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(blocked))

    check = _check(DoctorCommandService(_session()).report().checks, "artifact-store")

    assert check.status is CheckStatus.FAIL
    assert str(blocked) in check.detail


def test_artifact_store_local_writable_path_passes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path))

    check = _check(DoctorCommandService(_session()).report().checks, "artifact-store")

    assert check.status is CheckStatus.PASS


def test_artifact_store_gs_bucket_failure_names_the_create_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: "/usr/bin/gcloud")
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", "gs://artifacts-missing.example/eyvan")
    monkeypatch.setattr(preflights, "gcloud_run", lambda _argv: (False, "bucket not found"))

    check = _check(DoctorCommandService(_session()).report().checks, "artifact-store")

    assert check.status is CheckStatus.FAIL
    assert "gs://artifacts-missing.example" in check.detail
    assert "gcloud storage buckets create" in check.remedy


def test_configured_gcs_without_gcloud_is_unknown_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflights.shutil, "which", lambda _name: None)
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", "gs://configured-artifacts/eyvan")

    check = _check(DoctorCommandService(_session()).report().checks, "artifact-store")

    assert check.status is CheckStatus.WARN
    assert "configured-artifacts" in check.detail
    assert "not applicable" not in check.detail
    assert "gcloud" in check.remedy


def test_runtime_profile_absent_is_not_a_finding(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(preflights, "DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH", tmp_path / "absent.json")

    check = _check(DoctorCommandService(_session()).report().checks, "runtime-profile")

    assert check.status is CheckStatus.PASS
    assert "not applicable" in check.detail
    assert "eyvan runtime init" in check.detail


def test_control_plane_port_free_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket as socket_module

    monkeypatch.setattr(socket_module.socket, "connect_ex", lambda _self, _address: 111)

    check = _check(DoctorCommandService(_session()).report().checks, "control-plane-port")

    assert check.status is CheckStatus.PASS
    assert "free" in check.detail


def test_control_plane_port_squatted_by_a_non_eyvan_process_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket as socket_module

    monkeypatch.setattr(socket_module.socket, "connect_ex", lambda _self, _address: 0)

    class _RefusedResponse:
        status_code = 503
        text = ""

    monkeypatch.setattr(preflights.httpx, "get", lambda _url, **_kwargs: _RefusedResponse())

    check = _check(DoctorCommandService(_session()).report().checks, "control-plane-port")

    assert check.status is CheckStatus.FAIL
    assert "not an Eyvan control plane" in check.detail
    assert "lsof" in check.remedy


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8080", "http://[::1]:8080", "https://[::1]"])
def test_control_plane_port_health_probe_preserves_valid_authority(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.setattr(preflights.socket.socket, "connect_ex", lambda _self, _address: 0)
    observed = []

    def health(url: str, *, timeout: float, trust_env: bool, **_tls: tp.Any) -> tp.Any:
        assert timeout == 3.0
        assert trust_env is False
        # Use the real URL parser: an unbracketed IPv6 authority raises
        # InvalidURL before any network request could be issued.
        request = preflights.httpx.Request("GET", url)
        observed.append(request.url)
        return preflights.httpx.Response(200, json={"controller": "api_only"}, request=request)

    monkeypatch.setattr(preflights.httpx, "get", health)

    name, status, detail, remedy = preflights.check_control_plane_port(base_url)

    assert name == "control-plane-port"
    assert status == "pass", (detail, remedy)
    assert observed == [preflights.httpx.URL(base_url + "/api/v1/health")]


def test_control_plane_port_remote_endpoint_is_not_a_finding() -> None:
    check = _check(
        DoctorCommandService(_session("https://plane.example.com")).report().checks,
        "control-plane-port",
    )

    assert check.status is CheckStatus.PASS
    assert "not applicable" in check.detail
    assert "remote" in check.detail


def test_local_preflights_run_even_when_the_plane_is_unreachable() -> None:
    # The whole point of this check family: they answer before anything is
    # started. An unreachable endpoint must not silence them.

    def unreachable(_path: str) -> tp.Any:
        raise ConnectionRefusedError("connection refused")

    names = [check.name for check in DoctorCommandService(_session(handler=unreachable)).report().checks]

    for expected in (
        "gcloud-authentication",
        "secret-manager-bind-permission",
        "artifact-store",
        "runtime-profile",
        "control-plane-port",
    ):
        assert expected in names
