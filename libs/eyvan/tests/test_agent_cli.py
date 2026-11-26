# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal runtime-profile startup coverage for the Eyvan cell-agent CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI

from eyvan.capacity.providers.gcp.runtime.profile import (
    DEFAULT_GCP_NATIVE_FLEET_STATE_PATH,
    GcpNativeRuntimeProfileStore,
)
from eyvan.worker.agent.cli import _settings_for_runtime_profile, main


def test_agent_cli_loads_native_gcp_settings_from_the_owner_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EYVAN_FLEET_STATE_URI", raising=False)
    store = GcpNativeRuntimeProfileStore(tmp_path / "profile" / "runtime.json")
    profile = store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )

    settings = _settings_for_runtime_profile(store.path)

    assert settings.agent_shared_secret == profile.agent_shared_secret
    assert settings.native_worker_root_secret == profile.native_worker_root_secret
    # The profile-owned local agent keeps its fleet-state cache on its own
    # filesystem; only an explicit EYVAN_FLEET_STATE_URI moves it to a bucket.
    assert settings.fleet_state_uri == str(DEFAULT_GCP_NATIVE_FLEET_STATE_PATH)
    assert settings.native_worker_package == profile.native_worker_wheel_uri
    assert settings.native_worker_artifact_root_uri == profile.artifact_root_uri


def _clear_agent_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYVAN_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", raising=False)


def _stub_agent_factory(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Capture the allow_unauthenticated flag the CLI hands to the app factory."""

    import eyvan.worker.agent.cli as cli

    allowed: list[bool] = []

    def fake_create_agent_app(service=None, settings=None, *, allow_unauthenticated: bool = False) -> FastAPI:
        allowed.append(allow_unauthenticated)
        return FastAPI()

    monkeypatch.setattr(cli, "create_agent_app", fake_create_agent_app)
    return allowed


def test_agent_cli_refuses_to_start_without_a_shared_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_auth_env(monkeypatch)
    monkeypatch.setattr("sys.argv", ["eyvan-agent"])
    served: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: served.append(app))

    with pytest.raises(ValueError, match="EYVAN_AGENT_SHARED_SECRET"):
        main()

    assert served == []


def test_agent_cli_defaults_to_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_auth_env(monkeypatch)
    monkeypatch.setenv("EYVAN_AGENT_SHARED_SECRET", "secret")
    monkeypatch.setattr("sys.argv", ["eyvan-agent"])
    allowed = _stub_agent_factory(monkeypatch)
    served: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: served.append(app))

    main()

    assert allowed == [False]
    assert len(served) == 1


def test_agent_cli_allow_unauthenticated_flag_starts_without_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_auth_env(monkeypatch)
    monkeypatch.setattr("sys.argv", ["eyvan-agent", "--host", "127.0.0.1", "--allow-unauthenticated"])
    allowed = _stub_agent_factory(monkeypatch)
    served: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: served.append(app))

    main()

    assert allowed == [True]
    assert len(served) == 1


def test_agent_cli_allow_unauthenticated_env_starts_without_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_auth_env(monkeypatch)
    monkeypatch.setenv("EYVAN_AGENT_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr("sys.argv", ["eyvan-agent", "--host", "127.0.0.1"])
    allowed = _stub_agent_factory(monkeypatch)
    served: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: served.append(app))

    main()

    assert allowed == [True]
    assert len(served) == 1
