# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Redeploying every local Eyvan component as one operation.

A source checkout deploys three separate surfaces -- the control plane, the cell
agent, and the staged worker wheel -- and each was updated by its own command.
Forgetting one produced a fleet that worked and was wrong: the workload
succeeds, on code nobody is looking at. It happened repeatedly during this
package's own development, most expensively when a fix lived in the cell agent
while only the TPU hosts were redeployed, so a live preemption test reproduced a
bug that had already been fixed.

Reporting the drift was not enough, because a report still relies on somebody
running three commands in the right order. ``runtime deploy`` does all three and
then *verifies* the postcondition, refusing to report success while anything is
still older than the source. That is what turns a convention into a guarantee,
so the verification is what these tests are mostly about.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile, GcpNativeRuntimeProfileStore
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands import runtime as runtime_commands
from eyvan.sdk.commands.runtime import GcpRuntimeDeployError, LocalAgentState, RuntimeRestartResult
from eyvan.server.local.daemon import LocalDaemonResult, LocalDaemonState

ENDPOINT = ControlPlaneEndpoint.parse("http://127.0.0.1:19099")
STAGED_WHEEL = f"gs://bucket/eyvan/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl"


def _profile(wheel_uri: str | None = None) -> GcpNativeRuntimeProfile:
    return GcpNativeRuntimeProfile(
        project_id="example-project-1",
        artifact_root_uri="gs://bucket/eyvan",
        agent_shared_secret="agent-secret-" + "a" * 32,
        native_worker_root_secret="worker-secret-" + "b" * 32,
        native_worker_wheel_uri=wheel_uri,
    )


def _restart_result() -> RuntimeRestartResult:
    return RuntimeRestartResult(
        endpoint=ENDPOINT,
        control_plane=LocalDaemonResult(endpoint=ENDPOINT, state=LocalDaemonState.STARTED, pid=4242, log_path=None),
        agent_endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
        agent_state=LocalAgentState.STARTED,
        trusted_callable_projects=("acme/research",),
    )


@pytest.fixture
def deployed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Wire a runtime whose wheel staging and restarts are observable."""

    store = GcpNativeRuntimeProfileStore(tmp_path / "runtime.json")
    store.create(
        project_id="example-project-1",
        artifact_root_uri="gs://bucket/eyvan",
        native_worker_wheel_uri=None,
    )
    calls: dict[str, int] = {"staged": 0, "restarted": 0}

    class _Bootstrap:
        def __init__(self, _builder) -> None:
            pass

        def stage_native_worker_wheel(self, profile: GcpNativeRuntimeProfile) -> GcpNativeRuntimeProfile:
            calls["staged"] += 1
            return replace(profile, native_worker_wheel_uri=STAGED_WHEEL)

    class _Builder:
        def build(self) -> Path:
            raise AssertionError("the fake bootstrap stages without building")

        def close(self) -> None:
            pass

    def _restart(*, endpoint: ControlPlaneEndpoint, profile_path: Path) -> RuntimeRestartResult:
        del endpoint, profile_path
        calls["restarted"] += 1
        return _restart_result()

    monkeypatch.setattr(runtime_commands, "GcpRuntimeBootstrap", _Bootstrap)
    monkeypatch.setattr(runtime_commands, "native_worker_wheel_builder", _Builder)
    monkeypatch.setattr(runtime_commands, "_restart_native_runtime", _restart)
    monkeypatch.setattr(runtime_commands, "stale_components", lambda **_kwargs: ())
    return store, calls


def test_one_command_stages_the_wheel_and_restarts_both_daemons(deployed) -> None:
    """The whole point: no surface is left to a separate command."""

    store, calls = deployed

    result = runtime_commands._deploy_native_runtime(endpoint=ENDPOINT, profile_path=store.path)

    assert calls == {"staged": 1, "restarted": 1}
    assert result.wheel_uri == STAGED_WHEEL
    assert store.load().native_worker_wheel_uri == STAGED_WHEEL


def test_the_staged_wheel_is_persisted_before_the_restart(deployed, monkeypatch: pytest.MonkeyPatch) -> None:
    """The restarted daemons must read the wheel this deploy just staged.

    Restarting first would boot the control plane against the previous pin and
    leave it a generation behind the workers -- the drift this command exists to
    remove, reintroduced by ordering.
    """

    store, _calls = deployed
    seen: list[str | None] = []

    def _restart(*, endpoint: ControlPlaneEndpoint, profile_path: Path) -> RuntimeRestartResult:
        del endpoint
        seen.append(GcpNativeRuntimeProfileStore(profile_path).load().native_worker_wheel_uri)
        return _restart_result()

    monkeypatch.setattr(runtime_commands, "_restart_native_runtime", _restart)

    runtime_commands._deploy_native_runtime(endpoint=ENDPOINT, profile_path=store.path)

    assert seen == [STAGED_WHEEL]


def test_a_component_left_stale_fails_the_deploy(deployed, monkeypatch: pytest.MonkeyPatch) -> None:
    """The postcondition is the guarantee; without it this is just a convenience.

    Reporting success with a stale component recreates exactly the failure the
    command was written to remove.
    """

    store, _calls = deployed
    monkeypatch.setattr(
        runtime_commands,
        "stale_components",
        lambda **_kwargs: ("cell agent (http://127.0.0.1:8090)",),
    )

    with pytest.raises(GcpRuntimeDeployError, match="still run older code"):
        runtime_commands._deploy_native_runtime(endpoint=ENDPOINT, profile_path=store.path)


def test_an_uninitialized_runtime_says_what_to_run(tmp_path: Path) -> None:
    """Deploy cannot invent credentials; it must name the command that creates them."""

    with pytest.raises(GcpRuntimeDeployError, match="runtime init"):
        runtime_commands._deploy_native_runtime(endpoint=ENDPOINT, profile_path=tmp_path / "absent.json")


def test_staging_that_returns_no_wheel_is_refused(deployed, monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile pinned to nothing would deploy workers with no Eyvan at all."""

    store, _calls = deployed

    class _NoWheel:
        def __init__(self, _builder) -> None:
            pass

        def stage_native_worker_wheel(self, profile: GcpNativeRuntimeProfile) -> GcpNativeRuntimeProfile:
            return profile

    monkeypatch.setattr(runtime_commands, "GcpRuntimeBootstrap", _NoWheel)

    with pytest.raises(GcpRuntimeDeployError, match="without a worker wheel URI"):
        runtime_commands._deploy_native_runtime(endpoint=ENDPOINT, profile_path=store.path)


def test_the_deploy_document_carries_no_profile_secret(deployed) -> None:
    """Automation reads this; the profile holds two shared secrets."""

    store, _calls = deployed

    document = runtime_commands._deploy_native_runtime(endpoint=ENDPOINT, profile_path=store.path).as_document()

    encoded = str(document)
    assert "agent-secret-" not in encoded
    assert "worker-secret-" not in encoded
    assert document["stale_components"] == []
