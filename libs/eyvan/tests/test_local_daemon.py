# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.server.local.daemon import (
    DEFAULT_LOCAL_MANAGED_ARTIFACTS_PATH,
    LocalDaemonCommand,
    LocalDaemonConfigurationError,
    LocalDaemonPaths,
    LocalDaemonProfile,
    LocalDaemonRecord,
    LocalDaemonResult,
    LocalDaemonStartupError,
    LocalDaemonState,
    LocalDaemonStateError,
    LocalDaemonSupervisor,
    LocalDaemonTarget,
    SubprocessLocalDaemonLauncher,
    write_owner_only_record,
)
from eyvan.server.local.profile import DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH, DEFAULT_LOCAL_CONTROL_PLANE_PATH


@dataclass
class FakeProcess:
    """A deterministic process handle for daemon-supervisor tests."""

    pid: int = 4242
    exit_code: int | None = None
    terminated: bool = False
    killed: bool = False

    def poll(self) -> int | None:
        """Return the configured process state."""

        return self.exit_code

    def terminate(self) -> None:
        """Record a graceful stop request."""

        self.terminated = True
        self.exit_code = 0

    def kill(self) -> None:
        """Record a forced stop request."""

        self.killed = True
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        """Return the configured exit status after a stop request."""

        del timeout
        return self.exit_code if self.exit_code is not None else 0


@dataclass
class FakeLauncher:
    """Captures daemon commands without creating a real subprocess."""

    process: FakeProcess
    commands: list[LocalDaemonCommand] = field(default_factory=list)

    def start(self, command: LocalDaemonCommand) -> FakeProcess:
        """Return the configured process handle."""

        self.commands.append(command)
        return self.process


@dataclass
class HealthResponses:
    """Return a finite ordered sequence of health-check outcomes."""

    values: list[bool]
    endpoints: list[ControlPlaneEndpoint] = field(default_factory=list)

    def __call__(self, endpoint: ControlPlaneEndpoint) -> bool:
        """Return the next configured health-check result."""

        self.endpoints.append(endpoint)
        if not self.values:
            raise AssertionError("daemon supervisor performed an unexpected health check")
        return self.values.pop(0)


@dataclass
class ProcessLiveness:
    """Return an ordered sequence of PID liveness observations."""

    values: list[bool]
    pids: list[int] = field(default_factory=list)

    def __call__(self, pid: int) -> bool:
        """Record a liveness probe and return its configured result."""

        self.pids.append(pid)
        if not self.values:
            raise AssertionError("daemon restart performed an unexpected PID liveness check")
        return self.values.pop(0)


@dataclass
class ProcessTerminator:
    """Record explicit graceful termination requests without signaling real processes."""

    pids: list[int] = field(default_factory=list)

    def __call__(self, pid: int) -> None:
        """Record one requested process termination."""

        self.pids.append(pid)


@dataclass
class ProcessKiller:
    """Record forced termination requests without signalling a real process."""

    pids: list[int] = field(default_factory=list)

    def __call__(self, pid: int) -> None:
        """Record one forced process termination."""

        self.pids.append(pid)


@dataclass
class AdvancingClock:
    """Advance a deterministic monotonic clock through the supervisor's sleeps."""

    value: float = 0.0

    def __call__(self) -> float:
        """Return the current deterministic timestamp."""

        return self.value

    def sleep(self, duration: float) -> None:
        """Advance time instead of blocking the test process."""

        self.value += duration


def _target(url: str) -> LocalDaemonTarget:
    endpoint = ControlPlaneEndpoint.parse(url)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    return target


def _supervisor(tmp_path: Path, launcher: FakeLauncher, health: HealthResponses) -> LocalDaemonSupervisor:
    return LocalDaemonSupervisor(
        paths=LocalDaemonPaths(tmp_path / "eyvan"),
        launcher=launcher,
        health_probe=health,
    )


@pytest.mark.parametrize(
    "url",
    (
        "https://eyvan.example.com",
        "http://localhost:8080",
        "https://127.0.0.1:8080",
        "http://127.0.0.1:8080/control",
    ),
)
def test_supervisor_never_launches_a_non_literal_http_loopback_endpoint(tmp_path: Path, url: str) -> None:
    launcher = FakeLauncher(FakeProcess())
    health = HealthResponses([])

    result = _supervisor(tmp_path, launcher, health).ensure(url)

    assert result == LocalDaemonResult(
        endpoint=ControlPlaneEndpoint.parse(url),
        state=LocalDaemonState.SKIPPED,
        pid=None,
        log_path=None,
    )
    assert launcher.commands == []
    assert health.endpoints == []


def test_supervisor_reuses_a_healthy_literal_loopback_control_plane(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19080")
    launcher = FakeLauncher(FakeProcess())
    health = HealthResponses([True])

    result = _supervisor(tmp_path, launcher, health).ensure(endpoint)

    assert result.endpoint == endpoint
    assert result.state is LocalDaemonState.REUSED
    assert result.pid is None
    assert result.log_path == LocalDaemonPaths(tmp_path / "eyvan").log_path(_target(endpoint.url))
    assert launcher.commands == []
    assert health.endpoints == [endpoint]


def test_default_loopback_profile_preserves_the_existing_control_plane_locations(tmp_path: Path) -> None:
    target = _target("http://127.0.0.1:8080")

    profile = LocalDaemonPaths(tmp_path / "eyvan").profile(target)

    assert profile == LocalDaemonProfile(
        endpoint=target.endpoint,
        state_path=DEFAULT_LOCAL_CONTROL_PLANE_PATH,
        artifact_token_key_path=DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH,
        artifact_store_path=DEFAULT_LOCAL_MANAGED_ARTIFACTS_PATH,
    )
    assert profile.artifact_store_uri == "~/.eyvan/managed-artifacts"


def test_custom_loopback_profile_is_isolated_by_endpoint(tmp_path: Path) -> None:
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    target = _target("http://127.0.0.1:19080")

    profile = paths.profile(target)
    expected_directory = paths.profile_directory(target)

    assert profile == LocalDaemonProfile(
        endpoint=target.endpoint,
        state_path=expected_directory / "control-plane.json",
        artifact_token_key_path=expected_directory / "artifact-token.key",
        artifact_store_path=expected_directory / "managed-artifacts",
    )
    assert profile.state_path != DEFAULT_LOCAL_CONTROL_PLANE_PATH
    assert profile.artifact_token_key_path != DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH
    assert profile.artifact_store_uri == str(expected_directory / "managed-artifacts")


def test_supervisor_starts_literal_loopback_server_and_persists_owner_only_pid_record(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19081")
    launcher = FakeLauncher(FakeProcess(pid=9876))
    health = HealthResponses([False, False, True])
    paths = LocalDaemonPaths(tmp_path / "eyvan")

    result = LocalDaemonSupervisor(paths=paths, launcher=launcher, health_probe=health).ensure(endpoint)

    target = _target(endpoint.url)
    record_path = paths.record_path(target)
    profile = paths.profile(target)
    assert result == LocalDaemonResult(
        endpoint=endpoint,
        state=LocalDaemonState.STARTED,
        pid=9876,
        log_path=paths.log_path(target),
    )
    # record_path is passed so the server can claim ownership as the process
    # that actually binds; the supervisor's own pre-write is only a first guess.
    assert launcher.commands == [
        LocalDaemonCommand(
            target=target,
            log_path=paths.log_path(target),
            profile=profile,
            record_path=record_path,
        )
    ]
    assert launcher.commands[0].argv == (
        sys.executable,
        "-m",
        "eyvan.server.api.app",
        "--host",
        "127.0.0.1",
        "--port",
        "19081",
        "--local-daemon",
        "--public-url",
        endpoint.url,
        "--local-state-path",
        str(profile.state_path),
        "--local-artifact-token-key-path",
        str(profile.artifact_token_key_path),
        "--artifact-store-uri",
        profile.artifact_store_uri,
        "--local-daemon-record-path",
        str(record_path),
        "--log-level",
        "info",
    )
    assert json.loads(record_path.read_text(encoding="utf-8")) == {
        "endpoint": endpoint.url,
        "pid": 9876,
    }
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.resolved_directory.stat().st_mode) == 0o700


def test_a_child_record_published_after_the_parents_read_is_not_overwritten(tmp_path: Path, monkeypatch) -> None:
    """The child can publish between the parent's ownership read and its fallback write.

    The parent must re-read under the publication lock the child shares, or its
    unknown-digest fallback replaces the child's authoritative build digest.
    """

    from eyvan.server.local import daemon as daemon_module

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:18080")
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    record_path = paths.record_path(target)
    launcher = FakeLauncher(FakeProcess(pid=9877))
    child_record = LocalDaemonRecord(
        endpoint=endpoint,
        pid=launcher.process.pid,
        runtime_profile_path=None,
        source_digest="sha256:child-build",
    )
    real_read = daemon_module.read_owner_only_record
    published: list[bool] = []

    def read_then_child_publishes(path: Path):
        value = real_read(path)
        if path == record_path and launcher.commands and not published and value is None:
            # The child's own startup lands right after the parent's post-launch read.
            published.append(True)
            daemon_module.write_owner_only_record(path, child_record)
        return value

    monkeypatch.setattr(daemon_module, "read_owner_only_record", read_then_child_publishes)
    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=launcher,
        health_probe=HealthResponses([False, False, True]),
    )

    supervisor.ensure(endpoint)

    assert published, "the interleaving under test never happened"
    record = real_read(record_path)
    assert record is not None
    assert record.pid == launcher.process.pid
    assert record.source_digest == "sha256:child-build"


@pytest.mark.parametrize("child_publishes", [True, False])
def test_startup_preserves_child_digest_or_writes_unknown(tmp_path, child_publishes) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19081")
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    record_path = paths.record_path(_target(endpoint.url))

    class PublishingLauncher(FakeLauncher):
        def start(self, command):
            process = super().start(command)
            assert command.record_path == record_path
            if child_publishes:
                write_owner_only_record(
                    record_path, LocalDaemonRecord(endpoint=endpoint, pid=process.pid, source_digest="child-build")
                )
            return process

    launcher = PublishingLauncher(FakeProcess(pid=9876))
    supervisor = LocalDaemonSupervisor(
        paths=paths, launcher=launcher, health_probe=HealthResponses([False, False, True])
    )
    assert supervisor.ensure(endpoint).state is LocalDaemonState.STARTED
    document = json.loads(record_path.read_text())
    assert document["pid"] == launcher.process.pid
    assert document.get("source_digest") == ("child-build" if child_publishes else None)


def test_supervisor_surfaces_server_exit_with_the_log_path_and_clears_its_pid_record(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19082")
    launcher = FakeLauncher(FakeProcess(pid=2222, exit_code=17))
    health = HealthResponses([False, False, False])
    paths = LocalDaemonPaths(tmp_path / "eyvan")

    with pytest.raises(LocalDaemonStartupError) as raised:
        LocalDaemonSupervisor(paths=paths, launcher=launcher, health_probe=health).ensure(endpoint)

    assert "server process exited with status 17" in str(raised.value)
    assert str(paths.log_path(_target(endpoint.url))) in str(raised.value)
    assert not paths.record_path(_target(endpoint.url)).exists()


def test_restart_replaces_a_verified_legacy_owned_daemon_with_the_runtime_profile(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19084")
    target = _target(endpoint.url)
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    write_owner_only_record(paths.record_path(target), LocalDaemonRecord(endpoint=endpoint, pid=7812))
    launcher = FakeLauncher(FakeProcess(pid=7813))
    health = HealthResponses([False, False, True])
    liveness = ProcessLiveness([True, True, False])
    terminator = ProcessTerminator()
    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=launcher,
        health_probe=health,
        process_argument_reader=lambda pid: (
            (
                sys.executable,
                "-m",
                "eyvan.server.api.app",
                "--host",
                "127.0.0.1",
                "--port",
                "19084",
                "--log-level",
                "info",
            )
            if pid == 7812
            else None
        ),
        process_liveness=liveness,
        process_terminator=terminator,
    )

    result = supervisor.restart(endpoint, runtime_profile_path=profile_path)

    assert result == LocalDaemonResult(
        endpoint=endpoint,
        state=LocalDaemonState.STARTED,
        pid=7813,
        log_path=paths.log_path(target),
    )
    assert terminator.pids == [7812]
    assert launcher.commands == [
        LocalDaemonCommand(
            target=target,
            log_path=paths.log_path(target),
            profile=paths.profile(target),
            runtime_profile_path=profile_path,
            record_path=paths.record_path(target),
        )
    ]
    assert json.loads(paths.record_path(target).read_text(encoding="utf-8")) == {
        "endpoint": endpoint.url,
        "pid": 7813,
        "runtime_profile_path": str(profile_path.resolve()),
    }


def test_restart_forces_an_owned_daemon_that_does_not_finish_gracefully(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19089")
    target = _target(endpoint.url)
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    write_owner_only_record(
        paths.record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=7822,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    launcher = FakeLauncher(FakeProcess(pid=7823))
    liveness = ProcessLiveness([True, True, True, True, False])
    terminator = ProcessTerminator()
    killer = ProcessKiller()
    clock = AdvancingClock()
    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=launcher,
        health_probe=HealthResponses([False, False, True]),
        startup_timeout_s=1.0,
        poll_interval_s=1.0,
        clock=clock,
        sleeper=clock.sleep,
        process_argument_reader=lambda pid: (
            (
                sys.executable,
                "-m",
                "eyvan.server.api.app",
                "--host",
                "127.0.0.1",
                "--port",
                "19089",
                "--local-daemon",
                "--runtime-profile",
                str(profile_path.resolve()),
                "--log-level",
                "info",
            )
            if pid == 7822
            else None
        ),
        process_liveness=liveness,
        process_terminator=terminator,
        process_killer=killer,
    )

    result = supervisor.restart(endpoint, runtime_profile_path=profile_path)

    assert result.state is LocalDaemonState.STARTED
    assert terminator.pids == [7822]
    assert killer.pids == [7822]


def test_stop_started_rolls_back_only_the_matching_profile_daemon(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19088")
    target = _target(endpoint.url)
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    write_owner_only_record(
        paths.record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=7820,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    terminator = ProcessTerminator()
    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=FakeLauncher(FakeProcess(pid=7821)),
        health_probe=HealthResponses([]),
        process_argument_reader=lambda pid: (
            (
                sys.executable,
                "-m",
                "eyvan.server.api.app",
                "--host",
                "127.0.0.1",
                "--port",
                "19088",
                "--local-daemon",
                "--runtime-profile",
                str(profile_path.resolve()),
                "--log-level",
                "info",
            )
            if pid == 7820
            else None
        ),
        process_liveness=ProcessLiveness([True, True, False]),
        process_terminator=terminator,
    )

    supervisor.stop_started(
        LocalDaemonResult(
            endpoint=endpoint,
            state=LocalDaemonState.STARTED,
            pid=7820,
            log_path=paths.log_path(target),
        ),
        runtime_profile_path=profile_path,
    )

    assert terminator.pids == [7820]
    assert not paths.record_path(target).exists()


def test_restart_refuses_a_live_daemon_owned_by_a_different_runtime_profile(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19085")
    target = _target(endpoint.url)
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    requested_profile = tmp_path / "runtime" / "requested.json"
    other_profile = tmp_path / "runtime" / "other.json"
    write_owner_only_record(
        paths.record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=7814,
            runtime_profile_path=str(other_profile.resolve()),
        ),
    )
    terminator = ProcessTerminator()
    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=FakeLauncher(FakeProcess(pid=7815)),
        health_probe=HealthResponses([]),
        process_argument_reader=lambda pid: None,
        process_liveness=ProcessLiveness([True]),
        process_terminator=terminator,
    )

    # The refusal now names both profiles and the remedy rather than only the
    # fact of a mismatch; the behaviour it guards is unchanged.
    with pytest.raises(LocalDaemonConfigurationError, match="not the selected"):
        supervisor.restart(endpoint, runtime_profile_path=requested_profile)

    assert terminator.pids == []


def test_restart_refuses_to_signal_a_pid_that_does_not_match_eyvan_api_command(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19086")
    target = _target(endpoint.url)
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    write_owner_only_record(paths.record_path(target), LocalDaemonRecord(endpoint=endpoint, pid=7816))
    terminator = ProcessTerminator()
    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=FakeLauncher(FakeProcess(pid=7817)),
        health_probe=HealthResponses([]),
        process_argument_reader=lambda pid: (sys.executable, "-m", "unrelated.server") if pid == 7816 else None,
        process_liveness=ProcessLiveness([True, True]),
        process_terminator=terminator,
    )

    with pytest.raises(LocalDaemonConfigurationError, match="not the expected Eyvan control-plane process"):
        supervisor.restart(endpoint, runtime_profile_path=profile_path)

    assert terminator.pids == []


def test_restart_rejects_a_group_readable_pid_record_before_inspecting_its_pid(tmp_path: Path) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19087")
    target = _target(endpoint.url)
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    record_path = paths.record_path(target)
    write_owner_only_record(record_path, LocalDaemonRecord(endpoint=endpoint, pid=7818))
    os.chmod(record_path, 0o640)
    terminator = ProcessTerminator()
    supervisor = LocalDaemonSupervisor(
        paths=paths,
        launcher=FakeLauncher(FakeProcess(pid=7819)),
        health_probe=HealthResponses([]),
        process_argument_reader=lambda pid: None,
        process_liveness=ProcessLiveness([]),
        process_terminator=terminator,
    )

    with pytest.raises(LocalDaemonStateError, match="not group- or world-accessible"):
        supervisor.restart(endpoint, runtime_profile_path=tmp_path / "runtime" / "gcp-native-runtime.json")

    assert terminator.pids == []


def test_subprocess_launcher_rejects_a_symlinked_log_path(tmp_path: Path) -> None:
    target = _target("http://127.0.0.1:19083")
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    log_path = paths.log_path(target)
    log_path.parent.mkdir(mode=0o700, parents=True)
    replacement = tmp_path / "replacement.log"
    replacement.write_text("do not follow", encoding="utf-8")
    log_path.symlink_to(replacement)

    with pytest.raises(LocalDaemonStateError, match="daemon log"):
        SubprocessLocalDaemonLauncher().start(
            LocalDaemonCommand(
                target=target,
                log_path=log_path,
                profile=paths.profile(target),
                record_path=paths.record_path(target),
            )
        )


def _profile_aware_supervisor(tmp_path: Path, health: HealthResponses) -> LocalDaemonSupervisor:
    return LocalDaemonSupervisor(
        paths=LocalDaemonPaths(tmp_path / "eyvan"),
        launcher=FakeLauncher([]),
        health_probe=health,
    )


@pytest.mark.parametrize("mismatch", [None, "profile", "endpoint"], ids=["matching", "other-profile", "other-endpoint"])
def test_stale_owner_record_respects_endpoint_and_profile(tmp_path: Path, monkeypatch, mismatch) -> None:
    """Preserve intentional healthy-listener reuse without relying on /tmp aliases."""

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8123")
    profile = (tmp_path / "profile.json").resolve()
    recorded_profile = profile if mismatch != "profile" else profile.with_name("other-profile.json")
    recorded_endpoint = endpoint if mismatch != "endpoint" else ControlPlaneEndpoint.parse("http://127.0.0.1:8124")
    supervisor = _profile_aware_supervisor(tmp_path, HealthResponses([True]))
    monkeypatch.setattr("eyvan.server.local.daemon.process_is_alive", lambda _pid: False)
    record_path = supervisor.paths.record_path(_target(endpoint.url))
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_owner_only_record(
        record_path,
        LocalDaemonRecord(endpoint=recorded_endpoint, pid=2**31 - 1, runtime_profile_path=str(recorded_profile)),
    )
    original_record = record_path.read_bytes()

    if mismatch is None:
        result = supervisor.ensure(endpoint, runtime_profile_path=profile)
        assert result.state is LocalDaemonState.REUSED
        assert result.endpoint == endpoint and result.pid is None
    else:
        with pytest.raises(LocalDaemonConfigurationError) as excinfo:
            supervisor.ensure(endpoint, runtime_profile_path=profile)
        message = str(excinfo.value)
        assert str(record_path) in message
        if mismatch == "profile":
            assert "is not running" in message
            assert "different runtime profile" in message
        else:
            assert "names endpoint" in message and recorded_endpoint.url in message
    assert record_path.read_bytes() == original_record, "reuse/refusal must not invent PID ownership"


def test_absent_pid_record_names_the_path_it_looked_for(tmp_path: Path) -> None:
    """With nothing listening on record, the operator needs the path to check."""

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8124")
    supervisor = _profile_aware_supervisor(tmp_path, HealthResponses([True]))
    record_path = supervisor.paths.record_path(_target(endpoint.url))

    with pytest.raises(LocalDaemonConfigurationError) as excinfo:
        supervisor.ensure(endpoint, runtime_profile_path=Path("/tmp/profile.json"))

    message = str(excinfo.value)
    assert "no owner-only PID record exists" in message
    assert str(record_path) in message


def test_the_server_is_told_where_to_claim_ownership(tmp_path: Path) -> None:
    """Only the serving process knows which PID ends up holding the port.

    The supervisor writes the record first, from the PID it launched, so a
    process that dies before binding is still cleaned up. But a record naming
    any other process makes every later native command refuse with "not owned by
    the selected native GCP runtime profile", and the only way out was editing
    the record's JSON by hand.
    """

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19099")
    launcher = FakeLauncher(FakeProcess(pid=5150))
    # Two probes before launch: ensure() checks for an existing daemon once
    # before taking the startup lock and once inside it.
    supervisor = _supervisor(tmp_path, launcher, HealthResponses([False, False, True]))
    target = _target(endpoint.url)

    supervisor.ensure(endpoint)

    argv = launcher.commands[0].argv
    assert "--local-daemon-record-path" in argv
    assert argv[argv.index("--local-daemon-record-path") + 1] == str(
        LocalDaemonPaths(tmp_path / "eyvan").record_path(target)
    )
