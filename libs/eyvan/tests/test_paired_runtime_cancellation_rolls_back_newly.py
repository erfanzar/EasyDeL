# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Paired runtime cancellation rolls back only the newly owned processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import eyvan.sdk.commands.runtime as runtime_commands
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfileStore
from eyvan.capacity.providers.gcp.runtime.supervisor import LocalAgentSupervisor
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.server.local.daemon import (
    LocalDaemonError,
    LocalDaemonPaths,
    LocalDaemonRecord,
    LocalDaemonResult,
    LocalDaemonState,
    LocalDaemonTarget,
    read_owner_only_record,
    write_owner_only_record,
)


@dataclass
class OwnedProcess:
    """A launch handle whose termination is observable without signalling the OS."""

    pid: int = 9011
    exit_code: int | None = None
    terminations: int = 0

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminations += 1
        self.exit_code = 0

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        assert self.exit_code is not None
        return self.exit_code


class OwnedLauncher:
    """Return an exact child handle instead of launching a daemon."""

    def __init__(self, process: OwnedProcess) -> None:
        self.process = process

    def start(self, command: object) -> OwnedProcess:
        return self.process


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_paired_restart_cleans_both_owned_replacements_on_readiness_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[BaseException]
) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "private" / "runtime.json")
    store.create(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        native_worker_wheel_uri=(
            f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'e' * 64}/eyvan-0.1.0-py3-none-any.whl"
        ),
    )
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19093")
    plane_result = LocalDaemonResult(
        endpoint=endpoint, state=LocalDaemonState.STARTED, pid=9010, log_path=tmp_path / "plane.log"
    )
    plane = OwnedProcess(pid=9010)
    agent = OwnedProcess()
    agent_paths = LocalDaemonPaths(tmp_path / "agent")
    interruption = error_type("readiness interrupted")
    events: list[str] = []

    class ReplacementPlane:
        def restart(self, configured_endpoint: ControlPlaneEndpoint, *, runtime_profile_path: Path) -> LocalDaemonResult:
            assert configured_endpoint == endpoint
            assert runtime_profile_path == store.path
            events.append("plane-started")
            return plane_result

        def stop_started(self, result: LocalDaemonResult, *, runtime_profile_path: Path) -> None:
            assert result is plane_result
            assert runtime_profile_path == store.path
            events.append("plane-rolled-back")
            plane.terminate()

    def interrupt_readiness(duration: float) -> None:
        assert duration > 0
        events.append("agent-interrupted")
        raise interruption

    supervisor = LocalAgentSupervisor(
        paths=agent_paths,
        launcher=OwnedLauncher(agent),
        capability_probe=lambda endpoint, secret: None,
        clock=lambda: 0.0,
        sleeper=interrupt_readiness,
    )
    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", ReplacementPlane)
    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", lambda: supervisor)

    with pytest.raises(error_type) as raised:
        runtime_commands._restart_native_runtime(endpoint=endpoint, profile_path=store.path)

    assert raised.value is interruption
    assert agent.terminations == 1
    assert agent.poll() == 0
    target = LocalDaemonTarget.from_endpoint(supervisor.endpoint)
    assert target is not None
    assert read_owner_only_record(agent_paths.agent_record_path(target)) is None
    assert plane.terminations == 1
    assert plane.poll() == 0
    assert events == ["plane-started", "agent-interrupted", "plane-rolled-back"]


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [LocalDaemonError, RuntimeError, KeyboardInterrupt])
def test_rollback_failure_does_not_replace_original_runtime_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[BaseException],
    cleanup_type: type[BaseException],
) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "private" / "runtime.json")
    store.create(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        native_worker_wheel_uri=(
            f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'e' * 64}/eyvan-0.1.0-py3-none-any.whl"
        ),
    )
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19093")
    plane_result = LocalDaemonResult(
        endpoint=endpoint, state=LocalDaemonState.STARTED, pid=9010, log_path=tmp_path / "plane.log"
    )
    interruption = error_type("interrupted")
    cleanup_attempts: list[LocalDaemonResult] = []

    class ReplacementPlane:
        def restart(self, configured_endpoint: ControlPlaneEndpoint, *, runtime_profile_path: Path) -> LocalDaemonResult:
            return plane_result

        def stop_started(self, result: LocalDaemonResult, *, runtime_profile_path: Path) -> None:
            assert runtime_profile_path == store.path
            cleanup_attempts.append(result)
            raise cleanup_type("cleanup detail")

    class InterruptedAgent:
        def restart(self, *args: object) -> None:
            raise interruption

    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", ReplacementPlane)
    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", InterruptedAgent)

    with pytest.raises(error_type) as raised:
        runtime_commands._restart_native_runtime(endpoint=endpoint, profile_path=store.path)

    assert raised.value is interruption
    assert cleanup_attempts == [plane_result]
    assert any("could not stop" in note for note in interruption.__notes__)
    assert "cleanup detail" not in " ".join(interruption.__notes__)


def test_agent_readiness_interruption_preserves_a_different_pid_record(tmp_path: Path) -> None:
    """Peer supervisor repair protects exact-child cleanup, not a later owner."""

    store = GcpNativeRuntimeProfileStore(tmp_path / "private" / "runtime.json")
    profile = store.create(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        native_worker_wheel_uri=(
            f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'e' * 64}/eyvan-0.1.0-py3-none-any.whl"
        ),
    )
    paths = LocalDaemonPaths(tmp_path / "agent")
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8090")
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    record_path = paths.agent_record_path(target)
    process = OwnedProcess()
    interruption = KeyboardInterrupt("readiness interrupted")
    replacement = LocalDaemonRecord(endpoint=endpoint, pid=9012, runtime_profile_path=str(store.path))

    def interrupt_readiness(duration: float) -> None:
        write_owner_only_record(record_path, replacement)
        raise interruption

    supervisor = LocalAgentSupervisor(
        paths=paths,
        endpoint=endpoint,
        launcher=OwnedLauncher(process),
        capability_probe=lambda endpoint, secret: None,
        clock=lambda: 0.0,
        sleeper=interrupt_readiness,
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        supervisor.restart(store.path, profile)

    assert raised.value is interruption
    assert process.poll() == 0
    assert process.terminations == 1
    assert read_owner_only_record(record_path) == replacement
