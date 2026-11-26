# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exact-child cleanup and endpoint exclusion regressions for GCP supervision."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from eyvan.capacity.providers.gcp.runtime import supervisor as module
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile
from eyvan.server.local.daemon import (
    LocalDaemonPaths,
    LocalDaemonRecord,
    read_owner_only_record,
    write_owner_only_record,
)
from eyvan.worker.agent.models import AgentBuild, AgentCapabilities


def _profile() -> GcpNativeRuntimeProfile:
    return GcpNativeRuntimeProfile(
        project_id="test-project",
        artifact_root_uri="gs://test-artifacts/runtime",
        agent_shared_secret="test-agent-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_root_secret="test-worker-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_wheel_uri=f"gs://test-artifacts/runtime/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl",
    )


def _capabilities() -> AgentCapabilities:
    return AgentCapabilities(
        provider="gcp",
        executor="eyvan-native",
        accelerator_types=("tpu",),
        capacity_classes=("spot",),
        supports_checkpoint=True,
        supports_callables=True,
        supports_exec=True,
        max_gang_replicas=1,
        resource_enforcement="none",
    )


def _build(digest: str) -> AgentBuild:
    return AgentBuild(version="0.0.0", origin="source", source_digest=digest, python="3.11", platform="test")


def _lock_held(path: Path) -> bool:
    """Observe real flock exclusion through a separate file description."""
    if not path.exists():
        return False
    descriptor = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


@dataclass
class Process:
    pid: int
    exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.exit_code = 0

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        assert self.exit_code is not None
        return self.exit_code


@dataclass
class Launcher:
    lock_path: Path
    processes: list[Process] = field(default_factory=list)
    launch_locks: list[bool] = field(default_factory=list)

    def start(self, command: module.LocalAgentCommand) -> Process:
        self.launch_locks.append(_lock_held(self.lock_path))
        process = Process(pid=41000 + len(self.processes))
        self.processes.append(process)
        return process

    def capabilities(self, endpoint, secret) -> AgentCapabilities | None:
        return _capabilities() if any(process.poll() is None for process in self.processes) else None

    def alive(self, pid: int) -> bool:
        return any(process.pid == pid and process.poll() is None for process in self.processes)


def _supervisor(tmp_path: Path) -> tuple[module.LocalAgentSupervisor, Launcher]:
    paths = LocalDaemonPaths(tmp_path / "owned-runtime")
    target = module.LocalDaemonTarget.from_endpoint(
        module.ControlPlaneEndpoint.parse(module.DEFAULT_LOCAL_GCP_AGENT_URL)
    )
    assert target is not None
    launcher = Launcher(paths.agent_lock_path(target))
    supervisor = module.LocalAgentSupervisor(
        paths=paths,
        launcher=launcher,
        capability_probe=launcher.capabilities,
        build_probe=lambda endpoint, secret: None,
        process_liveness=launcher.alive,
    )
    return supervisor, launcher


def test_restart_serializes_replacement_against_ensure(monkeypatch, tmp_path: Path) -> None:
    supervisor, launcher = _supervisor(tmp_path)
    real_lock = module.owner_only_startup_lock
    raced = False
    contender_results = []
    profile_path = tmp_path / "profile.json"
    profile = _profile()

    @contextmanager
    def lock_with_contender(path):
        nonlocal raced
        with real_lock(path):
            yield
        if not raced:
            raced = True
            contender_results.append(supervisor.ensure(profile_path, profile))

    monkeypatch.setattr(module, "owner_only_startup_lock", lock_with_contender)
    result = supervisor.restart(profile_path, profile)

    assert len(launcher.processes) == 1, "restart must not launch over the competing ensure's replacement"
    assert launcher.launch_locks == [True]
    record = read_owner_only_record(supervisor.paths.agent_record_path(supervisor._target()))
    assert record is not None and record.pid == result.pid == contender_results[0].pid
    assert contender_results[0].state is module.LocalAgentState.REUSED
    assert launcher.alive(record.pid)


def test_stale_agent_stop_and_record_mutation_hold_endpoint_lock(monkeypatch, tmp_path: Path) -> None:
    supervisor, launcher = _supervisor(tmp_path)
    profile_path = tmp_path / "profile.json"
    old = Process(pid=40000)
    launcher.processes.append(old)
    record_path = supervisor.paths.agent_record_path(supervisor._target())
    write_owner_only_record(
        record_path,
        LocalDaemonRecord(endpoint=supervisor.endpoint, pid=old.pid, runtime_profile_path=str(profile_path.resolve())),
    )
    monkeypatch.setattr(module, "build_identity", lambda: _build("current-build"))
    stop_locks = []
    clear_locks = []
    real_clear = module.clear_owner_only_record

    def stop(pid):
        stop_locks.append(_lock_held(launcher.lock_path))
        assert pid == old.pid
        old.terminate()

    def clear(path, *, pid):
        clear_locks.append(_lock_held(launcher.lock_path))
        real_clear(path, pid=pid)

    def build(endpoint, secret):
        return _build("old-build") if old.poll() is None else None

    supervisor = module.LocalAgentSupervisor(
        paths=supervisor.paths,
        launcher=launcher,
        capability_probe=launcher.capabilities,
        build_probe=build,
        process_liveness=launcher.alive,
        process_argument_reader=lambda pid: (
            sys.executable,
            "-m",
            "eyvan.worker.agent.cli",
            "--host",
            "127.0.0.1",
            "--port",
            "8090",
            "--runtime-profile",
            str(profile_path.resolve()),
        ),
        process_terminator=stop,
    )
    monkeypatch.setattr(module, "clear_owner_only_record", clear)
    result = supervisor.ensure(profile_path, _profile())

    assert stop_locks == [True], "stale-code replacement must not signal outside the endpoint lock"
    assert clear_locks == [True]
    assert launcher.launch_locks == [True]
    record = read_owner_only_record(record_path)
    assert record is not None and record.pid == result.pid != old.pid
    assert old.poll() is not None


@pytest.mark.parametrize("failure", ["capabilities", "probe", "interrupt"])
def test_failed_new_child_readiness_cleans_up_exact_attempt(tmp_path: Path, failure: str) -> None:
    supervisor, launcher = _supervisor(tmp_path)
    original = KeyboardInterrupt("cancelled startup") if failure == "interrupt" else RuntimeError("probe failed")

    def probe(endpoint, secret):
        if not launcher.processes or failure == "interrupt":
            return None
        if failure == "capabilities":
            return _capabilities().model_copy(update={"provider": "other"})
        raise original

    def sleeper(duration):
        raise original

    supervisor = module.LocalAgentSupervisor(
        paths=supervisor.paths,
        launcher=launcher,
        capability_probe=probe,
        build_probe=lambda endpoint, secret: None,
        process_liveness=launcher.alive,
        sleeper=sleeper,
    )
    expected = module.LocalAgentConfigurationError if failure == "capabilities" else type(original)
    with pytest.raises(expected) as caught:
        supervisor.ensure(tmp_path / "profile.json", _profile())

    if failure != "capabilities":
        assert caught.value is original, "cleanup must preserve the initiating exception"
    assert len(launcher.processes) == 1
    assert launcher.processes[0].poll() is not None, "the just-launched child must not leak after readiness fails"
    assert read_owner_only_record(supervisor.paths.agent_record_path(supervisor._target())) is None


def test_failed_existing_agent_readiness_preserves_its_child_and_record(tmp_path: Path) -> None:
    supervisor, launcher = _supervisor(tmp_path)
    profile_path = tmp_path / "profile.json"
    existing = Process(pid=42000)
    launcher.processes.append(existing)
    record_path = supervisor.paths.agent_record_path(supervisor._target())
    record = LocalDaemonRecord(
        endpoint=supervisor.endpoint, pid=existing.pid, runtime_profile_path=str(profile_path.resolve())
    )
    write_owner_only_record(record_path, record)
    probes = 0

    def probe(endpoint, secret):
        nonlocal probes
        probes += 1
        if probes == 1:
            return None
        return _capabilities().model_copy(update={"provider": "other"})

    supervisor = module.LocalAgentSupervisor(
        paths=supervisor.paths,
        launcher=launcher,
        capability_probe=probe,
        build_probe=lambda endpoint, secret: None,
        process_liveness=launcher.alive,
    )
    with pytest.raises(module.LocalAgentConfigurationError, match="provider 'gcp'"):
        supervisor.ensure(profile_path, _profile())

    assert existing.poll() is None
    assert launcher.launch_locks == []
    assert read_owner_only_record(record_path) == record


@dataclass
class StubbornProcess(Process):
    failure: str = "signals"
    terminate_count: int = 0
    kill_count: int = 0

    def terminate(self) -> None:
        self.terminate_count += 1
        if self.failure == "signals":
            raise OSError("signal refused")

    def kill(self) -> None:
        self.kill_count += 1
        if self.failure == "signals":
            raise OSError("signal refused")

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired("owned-test-child", timeout)


@dataclass
class CleanupLauncher(Launcher):
    failure: str = "signals"

    def start(self, command: module.LocalAgentCommand) -> Process:
        process = super().start(command)
        replacement = StubbornProcess(pid=process.pid, failure=self.failure)
        self.processes[-1] = replacement
        return replacement


@pytest.mark.parametrize("readiness", ["timeout", "build-skew"])
@pytest.mark.parametrize("stop_failure", ["signals", "waits", "none"])
def test_failed_startup_retains_ownership_until_child_exit(
    monkeypatch, tmp_path: Path, readiness: str, stop_failure: str
) -> None:
    supervisor, cooperative = _supervisor(tmp_path)
    launcher = cooperative if stop_failure == "none" else CleanupLauncher(cooperative.lock_path, failure=stop_failure)
    monkeypatch.setattr(module, "build_identity", lambda: _build("current-build"))
    now = 0.0

    def sleeper(duration):
        nonlocal now
        now += duration

    supervisor = module.LocalAgentSupervisor(
        paths=supervisor.paths,
        launcher=launcher,
        capability_probe=launcher.capabilities if readiness == "build-skew" else lambda endpoint, secret: None,
        build_probe=lambda endpoint, secret: _build("wrong-build") if launcher.processes else None,
        process_liveness=launcher.alive,
        startup_timeout_s=0.1,
        poll_interval_s=0.1,
        clock=lambda: now,
        sleeper=sleeper,
    )
    expected = module.LocalAgentStartupError if readiness == "timeout" else module.LocalAgentConfigurationError
    with pytest.raises(expected) as caught:
        supervisor.ensure(tmp_path / "profile.json", _profile())

    process = launcher.processes[0]
    record = read_owner_only_record(supervisor.paths.agent_record_path(supervisor._target()))
    if stop_failure == "none":
        assert process.poll() is not None
        assert record is None
    else:
        assert process.poll() is None
        assert record is not None and record.pid == process.pid, "an unexited child must retain its ownership evidence"
        assert process.terminate_count == process.kill_count == 1
        assert any("did not exit" in note and "ownership record was retained" in note for note in caught.value.__notes__)
    assert launcher.launch_locks == [True]
    assert "timeout" in str(caught.value) if readiness == "timeout" else "different build" in str(caught.value)


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_real_owned_child_exits_after_readiness_exception(tmp_path: Path, exception_type) -> None:
    """Exercise actual OS child termination, without binding sockets or starting Eyvan services."""
    supervisor, _ = _supervisor(tmp_path)
    original = exception_type("readiness failed")
    processes = []
    record_path = supervisor.paths.agent_record_path(supervisor._target())

    class RealLauncher:
        def start(self, command):
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", "import time; time.sleep(60)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            processes.append(process)
            return process

    def probe(endpoint, secret):
        if processes:
            record = read_owner_only_record(record_path)
            assert record is not None and record.pid == processes[0].pid
            assert processes[0].poll() is None
            raise original
        return None

    supervisor = module.LocalAgentSupervisor(
        paths=supervisor.paths,
        launcher=RealLauncher(),
        capability_probe=probe,
        build_probe=lambda endpoint, secret: None,
    )
    try:
        with pytest.raises(exception_type) as caught:
            supervisor.ensure(tmp_path / "profile.json", _profile())
        assert caught.value is original
        assert len(processes) == 1
        assert processes[0].poll() is not None
        assert read_owner_only_record(record_path) is None
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
