# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Clients follow the runtime profile a loopback plane was deployed with.

When ``eyvan runtime deploy --profile PATH`` selects a nondefault profile,
``eyvan.init()`` and ``eyvan doctor`` must use that serving profile rather than
suggest a restart under, or report details from, the default profile.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import eyvan
import eyvan.sdk.client.sync as sync_module
import eyvan.sdk.execution as execution_module
from eyvan.artifacts import GCSArtifactStore
from eyvan.capacity.providers.gcp.runtime.profile import (
    DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH,
    GcpNativeRuntimeProfile,
    GcpNativeRuntimeProfileStore,
)
from eyvan.capacity.providers.gcp.runtime.supervisor import (
    GcpNativeRuntimeSupervisor,
    GcpRuntimeResult,
    LocalAgentResult,
    LocalAgentState,
    has_gcp_native_runtime_profile,
)
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands import doctor_preflights
from eyvan.sdk.commands.doctor_preflights import check_runtime_profile, serving_or_default_runtime_profile
from eyvan.server.local.daemon import (
    LocalDaemonPaths,
    LocalDaemonRecord,
    LocalDaemonResult,
    LocalDaemonState,
    LocalDaemonTarget,
    runtime_profile_identifier,
    serving_runtime_profile_path,
    write_owner_only_record,
)
from eyvan.worker.agent.models import AgentCapabilities

ENDPOINT = ControlPlaneEndpoint.parse("http://127.0.0.1:19281")


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(execution_module, "_context", None)
    return home


def _profile(path: Path, *, project_id: str, root: str, digest: str) -> GcpNativeRuntimeProfile:
    return GcpNativeRuntimeProfileStore(path).create(
        project_id=project_id,
        artifact_root_uri=root,
        native_worker_wheel_uri=f"{root}/runtime/wheels/{digest * 64}/eyvan-0.1.0-py3-none-any.whl",
    )


def _record(
    endpoint: ControlPlaneEndpoint,
    *,
    pid: int,
    profile_path: Path | None,
    record_endpoint: ControlPlaneEndpoint | None = None,
) -> Path:
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    path = LocalDaemonPaths().record_path(target)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_owner_only_record(
        path,
        LocalDaemonRecord(
            endpoint=record_endpoint or endpoint,
            pid=pid,
            runtime_profile_path=runtime_profile_identifier(profile_path),
        ),
    )
    return path


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_a_live_record_names_the_profile_its_plane_serves(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign" / "runtime.json"
    _record(ENDPOINT, pid=os.getpid(), profile_path=campaign)

    assert serving_runtime_profile_path(ENDPOINT) == campaign.resolve()
    assert serving_runtime_profile_path(ENDPOINT.url) == campaign.resolve()


@pytest.mark.parametrize(
    "case",
    ["no-record", "dead-pid", "no-profile", "other-endpoint", "remote-endpoint"],
)
def test_anything_short_of_a_live_owned_record_keeps_the_default(tmp_path: Path, case: str) -> None:
    campaign = tmp_path / "campaign" / "runtime.json"
    endpoint = ENDPOINT
    if case == "dead-pid":
        _record(ENDPOINT, pid=_dead_pid(), profile_path=campaign)
    elif case == "no-profile":
        _record(ENDPOINT, pid=os.getpid(), profile_path=None)
    elif case == "other-endpoint":
        _record(
            ENDPOINT,
            pid=os.getpid(),
            profile_path=campaign,
            record_endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:19282"),
        )
    elif case == "remote-endpoint":
        endpoint = ControlPlaneEndpoint.parse("https://eyvan.example.com")

    assert serving_runtime_profile_path(endpoint) is None
    assert serving_or_default_runtime_profile(endpoint.url) == doctor_preflights.DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH


def test_init_uses_the_profile_the_deployed_plane_serves(
    tmp_path: Path, _isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _profile(
        _isolated_home / ".eyvan" / "gcp-native-runtime.json",
        project_id="default-project",
        root="gs://default-artifacts/eyvan",
        digest="a",
    )
    campaign_path = tmp_path / "campaign" / "runtime.json"
    campaign = _profile(campaign_path, project_id="campaign-project", root="gs://campaign-artifacts/run", digest="b")
    _record(ENDPOINT, pid=os.getpid(), profile_path=campaign_path)
    ensured: list[tuple[ControlPlaneEndpoint, Path | None]] = []

    def ensure(endpoint: str | ControlPlaneEndpoint, *, profile_path: Path | None = None) -> GcpRuntimeResult:
        parsed = ControlPlaneEndpoint.parse(endpoint)
        ensured.append((parsed, profile_path))
        loaded = GcpNativeRuntimeProfileStore(profile_path).load() if profile_path is not None else None
        assert loaded is not None
        return GcpRuntimeResult(
            profile=loaded,
            control_plane=LocalDaemonResult(
                endpoint=parsed, state=LocalDaemonState.REUSED, pid=os.getpid(), log_path=None
            ),
            agent=LocalAgentResult(
                endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
                state=LocalAgentState.REUSED,
                pid=os.getpid(),
                log_path=tmp_path / "cell-agent.log",
                capabilities=AgentCapabilities(
                    provider="gcp",
                    executor="eyvan-native",
                    accelerator_types=("tpu",),
                    capacity_classes=("spot",),
                    supports_checkpoint=True,
                    supports_callables=True,
                    supports_exec=True,
                    max_gang_replicas=1,
                    resource_enforcement="none",
                ),
            ),
        )

    class FakeClient:
        def __init__(self, *, base_url: str, identity: ClientIdentity) -> None:
            assert base_url == ENDPOINT.url

    monkeypatch.setattr(execution_module, "ensure_gcp_native_runtime", ensure)
    monkeypatch.setattr(sync_module, "EyvanClient", FakeClient)

    context = eyvan.init(ENDPOINT.url)

    assert ensured == [(ENDPOINT, campaign_path.resolve())]
    assert isinstance(context.artifact_store, GCSArtifactStore)
    assert context.artifact_store.root_uri == campaign.artifact_root_uri


def test_doctor_reports_the_serving_profile_not_the_default(tmp_path: Path, _isolated_home: Path) -> None:
    _profile(
        _isolated_home / ".eyvan" / "gcp-native-runtime.json",
        project_id="default-project",
        root="gs://default-artifacts/eyvan",
        digest="c",
    )
    campaign_path = tmp_path / "campaign" / "runtime.json"
    _profile(campaign_path, project_id="campaign-project", root="gs://campaign-artifacts/run", digest="d")
    _record(ENDPOINT, pid=os.getpid(), profile_path=campaign_path)

    name, status, detail, _remedy = check_runtime_profile(ENDPOINT.url)

    assert (name, status) == ("runtime-profile", "pass")
    assert "campaign-project" in detail and "gs://campaign-artifacts/run" in detail
    assert "default-project" not in detail
    # Without an endpoint the check keeps reading the default profile.
    assert "default-project" in check_runtime_profile()[2]


def test_the_supervisor_every_command_uses_follows_the_serving_profile(tmp_path: Path) -> None:
    """``eyvan up`` and ``init()`` both build their supervisor through ``for_endpoint``."""

    campaign_path = tmp_path / "campaign" / "runtime.json"

    assert (
        GcpNativeRuntimeSupervisor.for_endpoint(ENDPOINT).profile_store.path
        == DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH.expanduser()
    )
    assert has_gcp_native_runtime_profile(ENDPOINT) is False

    _profile(campaign_path, project_id="campaign-project", root="gs://campaign-artifacts/run", digest="e")
    _record(ENDPOINT, pid=os.getpid(), profile_path=campaign_path)

    assert GcpNativeRuntimeSupervisor.for_endpoint(ENDPOINT).profile_store.path == campaign_path.resolve()
    # The default profile does not exist, yet a plane serving another one is a
    # native runtime all the same.
    assert has_gcp_native_runtime_profile(ENDPOINT) is True
    assert has_gcp_native_runtime_profile() is False


def test_a_live_record_names_the_build_its_plane_was_launched_from() -> None:
    """The agent is judged against this, not against whichever CLI is running."""

    from eyvan.server.local.daemon import serving_source_digest

    target = LocalDaemonTarget.from_endpoint(ENDPOINT)
    assert target is not None
    path = LocalDaemonPaths().record_path(target)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_owner_only_record(path, LocalDaemonRecord(endpoint=ENDPOINT, pid=os.getpid(), source_digest="sha256:plane"))

    assert serving_source_digest(ENDPOINT) == "sha256:plane"


def test_a_record_from_before_the_field_or_a_dead_plane_names_no_build() -> None:
    from eyvan.server.local.daemon import serving_source_digest

    _record(ENDPOINT, pid=os.getpid(), profile_path=None)
    assert serving_source_digest(ENDPOINT) is None

    target = LocalDaemonTarget.from_endpoint(ENDPOINT)
    assert target is not None
    write_owner_only_record(
        LocalDaemonPaths().record_path(target),
        LocalDaemonRecord(endpoint=ENDPOINT, pid=_dead_pid(), source_digest="sha256:plane"),
    )
    assert serving_source_digest(ENDPOINT) is None
