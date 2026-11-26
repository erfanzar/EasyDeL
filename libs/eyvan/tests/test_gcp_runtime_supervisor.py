# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for private local API-and-agent supervision of native GCP runtime profiles."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile, GcpNativeRuntimeProfileStore
from eyvan.capacity.providers.gcp.runtime.supervisor import (
    DEFAULT_LOCAL_GCP_AGENT_URL,
    GcpNativeRuntimeSupervisor,
    GcpRuntimeError,
    GcpRuntimeProfileNotPreparedError,
    LocalAgentCommand,
    LocalAgentConfigurationError,
    LocalAgentResult,
    LocalAgentState,
    LocalAgentSupervisor,
)
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.server.local.daemon import (
    LocalDaemonPaths,
    LocalDaemonRecord,
    LocalDaemonResult,
    LocalDaemonState,
    LocalDaemonTarget,
    read_owner_only_record,
    write_owner_only_record,
)
from eyvan.worker.agent.models import AgentBuild, AgentCapabilities

_DEFAULT_WHEEL_URI = f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl"


def _capabilities() -> AgentCapabilities:
    """Return the authenticated native GCP capability contract expected by the supervisor."""

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


def _profile(*, wheel_uri: str | None = _DEFAULT_WHEEL_URI) -> GcpNativeRuntimeProfile:
    """Build one valid in-memory profile without exposing its random persisted secrets."""

    return GcpNativeRuntimeProfile(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        agent_shared_secret="agent-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_root_secret="worker-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_wheel_uri=wheel_uri,
    )


@dataclass
class FakeProcess:
    """Minimal deterministic local process used by agent supervision tests."""

    pid: int = 9123
    exit_code: int | None = None
    terminated: bool = False
    killed: bool = False

    def poll(self) -> int | None:
        """Return the configured process exit state."""

        return self.exit_code

    def terminate(self) -> None:
        """Record a graceful stop."""

        self.terminated = True
        self.exit_code = 0

    def kill(self) -> None:
        """Record a forced stop."""

        self.killed = True
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        """Return the current process exit code."""

        del timeout
        return self.exit_code if self.exit_code is not None else 0


@dataclass
class FakeAgentLauncher:
    """Capture local-agent commands without spawning an operating-system process."""

    process: FakeProcess = field(default_factory=FakeProcess)
    commands: list[LocalAgentCommand] = field(default_factory=list)

    def start(self, command: LocalAgentCommand) -> FakeProcess:
        """Record one launch request and return the deterministic process handle."""

        self.commands.append(command)
        return self.process


@dataclass
class AgentProcessLiveness:
    """Return ordered liveness observations without probing real operating-system PIDs."""

    values: list[bool]
    pids: list[int] = field(default_factory=list)

    def __call__(self, pid: int) -> bool:
        """Record one liveness observation and return its configured result."""

        self.pids.append(pid)
        if not self.values:
            raise AssertionError("cell-agent supervisor performed an unexpected PID liveness check")
        return self.values.pop(0)


@dataclass
class AgentProcessTerminator:
    """Record graceful termination requests without signaling a real process."""

    pids: list[int] = field(default_factory=list)

    def __call__(self, pid: int) -> None:
        """Record one requested termination."""

        self.pids.append(pid)


@dataclass
class AgentProcessKiller:
    """Record forced agent termination requests without signalling a real process."""

    pids: list[int] = field(default_factory=list)

    def __call__(self, pid: int) -> None:
        """Record one forced termination."""

        self.pids.append(pid)


@dataclass
class AdvancingClock:
    """Advance deterministic supervisor time without blocking a test."""

    value: float = 0.0

    def __call__(self) -> float:
        """Return the current deterministic timestamp."""

        return self.value

    def sleep(self, duration: float) -> None:
        """Advance time by one requested sleep interval."""

        self.value += duration


@dataclass
class CapabilityResponses:
    """Return an ordered sequence of authenticated capability observations."""

    values: list[AgentCapabilities | None]
    requests: list[tuple[ControlPlaneEndpoint, str]] = field(default_factory=list)

    def __call__(self, endpoint: ControlPlaneEndpoint, secret: str) -> AgentCapabilities | None:
        """Record one authenticated probe and return its configured result."""

        self.requests.append((endpoint, secret))
        if not self.values:
            raise AssertionError("unexpected cell-agent capability probe")
        return self.values.pop(0)


def test_agent_command_passes_only_the_private_profile_path_not_its_secrets(tmp_path: Path) -> None:
    target = LocalDaemonTarget.from_endpoint(ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL))
    assert target is not None
    profile_path = tmp_path / "runtime.json"
    command = LocalAgentCommand(target=target, log_path=tmp_path / "agent.log", runtime_profile_path=profile_path)

    assert command.argv == (
        sys.executable,
        "-m",
        "eyvan.worker.agent.cli",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--runtime-profile",
        str(profile_path),
        "--log-level",
        "info",
    )


def test_agent_supervisor_starts_authenticated_matching_agent_and_persists_profile_identity(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    launcher = FakeAgentLauncher()
    capabilities = _capabilities()
    probe = CapabilityResponses([None, None, capabilities])
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    supervisor = LocalAgentSupervisor(paths=paths, launcher=launcher, capability_probe=probe)

    result = supervisor.ensure(profile_path, profile)

    target = LocalDaemonTarget.from_endpoint(ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL))
    assert target is not None
    assert result == LocalAgentResult(
        endpoint=ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL),
        state=LocalAgentState.STARTED,
        pid=launcher.process.pid,
        log_path=paths.agent_log_path(target),
        capabilities=capabilities,
    )
    assert launcher.commands == [
        LocalAgentCommand(
            target=target,
            log_path=paths.agent_log_path(target),
            runtime_profile_path=profile_path,
        )
    ]
    assert json.loads(paths.agent_record_path(target).read_text(encoding="utf-8")) == {
        "endpoint": DEFAULT_LOCAL_GCP_AGENT_URL,
        "pid": launcher.process.pid,
        "runtime_profile_path": str(profile_path.resolve()),
    }
    assert all(secret == profile.agent_shared_secret for _, secret in probe.requests)


def test_agent_supervisor_rejects_an_unowned_agent_that_accepts_the_secret(tmp_path: Path) -> None:
    profile = _profile()
    probe = CapabilityResponses([_capabilities()])
    supervisor = LocalAgentSupervisor(
        paths=LocalDaemonPaths(tmp_path / "eyvan"),
        launcher=FakeAgentLauncher(),
        capability_probe=probe,
    )

    with pytest.raises(LocalAgentConfigurationError, match="not owned"):
        supervisor.ensure(tmp_path / "runtime.json", profile)


def test_agent_supervisor_preserves_normal_reuse_for_a_matching_owned_agent(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9130,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    launcher = FakeAgentLauncher()
    capabilities = _capabilities()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=launcher,
        capability_probe=CapabilityResponses([capabilities]),
        process_liveness=AgentProcessLiveness([True]),
    )

    result = supervisor.ensure(profile_path, profile)

    assert result == LocalAgentResult(
        endpoint=endpoint,
        state=LocalAgentState.REUSED,
        pid=9130,
        log_path=paths.agent_log_path(target),
        capabilities=capabilities,
    )
    assert launcher.commands == []


def test_agent_supervisor_restart_replaces_only_a_verified_owned_agent(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9131,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    launcher = FakeAgentLauncher(process=FakeProcess(pid=9132))
    capabilities = _capabilities()
    liveness = AgentProcessLiveness([True, True, False])
    terminator = AgentProcessTerminator()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=launcher,
        capability_probe=CapabilityResponses([capabilities]),
        process_argument_reader=lambda pid: (
            (
                sys.executable,
                "-m",
                "eyvan.worker.agent.cli",
                "--host",
                "127.0.0.1",
                "--port",
                "8090",
                "--runtime-profile",
                str(profile_path.resolve()),
                "--log-level",
                "info",
            )
            if pid == 9131
            else None
        ),
        process_liveness=liveness,
        process_terminator=terminator,
    )

    result = supervisor.restart(profile_path, profile)

    assert result == LocalAgentResult(
        endpoint=endpoint,
        state=LocalAgentState.STARTED,
        pid=9132,
        log_path=paths.agent_log_path(target),
        capabilities=capabilities,
    )
    assert terminator.pids == [9131]
    assert launcher.commands == [
        LocalAgentCommand(
            target=target,
            log_path=paths.agent_log_path(target),
            runtime_profile_path=profile_path.resolve(),
        )
    ]
    assert json.loads(paths.agent_record_path(target).read_text(encoding="utf-8")) == {
        "endpoint": endpoint.url,
        "pid": 9132,
        "runtime_profile_path": str(profile_path.resolve()),
    }


def test_agent_restart_forces_an_owned_agent_that_does_not_finish_gracefully(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9135,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    capabilities = _capabilities()
    liveness = AgentProcessLiveness([True, True, True, True, False])
    terminator = AgentProcessTerminator()
    killer = AgentProcessKiller()
    clock = AdvancingClock()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=FakeAgentLauncher(process=FakeProcess(pid=9136)),
        capability_probe=CapabilityResponses([capabilities]),
        startup_timeout_s=1.0,
        poll_interval_s=1.0,
        clock=clock,
        sleeper=clock.sleep,
        process_argument_reader=lambda pid: (
            (
                sys.executable,
                "-m",
                "eyvan.worker.agent.cli",
                "--host",
                "127.0.0.1",
                "--port",
                "8090",
                "--runtime-profile",
                str(profile_path.resolve()),
                "--log-level",
                "info",
            )
            if pid == 9135
            else None
        ),
        process_liveness=liveness,
        process_terminator=terminator,
        process_killer=killer,
    )

    result = supervisor.restart(profile_path, profile)

    assert result.state is LocalAgentState.STARTED
    assert terminator.pids == [9135]
    assert killer.pids == [9135]


def test_agent_supervisor_restart_refuses_an_unowned_agent_that_accepts_the_profile_secret(tmp_path: Path) -> None:
    profile = _profile()
    launcher = FakeAgentLauncher()
    supervisor = LocalAgentSupervisor(
        paths=LocalDaemonPaths(tmp_path / "eyvan"),
        launcher=launcher,
        capability_probe=CapabilityResponses([_capabilities()]),
    )

    with pytest.raises(LocalAgentConfigurationError, match="not owned"):
        supervisor.restart(tmp_path / "runtime.json", profile)

    assert launcher.commands == []


def test_agent_supervisor_restart_refuses_a_different_profile_without_signaling_it(tmp_path: Path) -> None:
    requested_profile_path = tmp_path / "runtime" / "requested.json"
    other_profile_path = tmp_path / "runtime" / "other.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9133,
            runtime_profile_path=str(other_profile_path.resolve()),
        ),
    )
    terminator = AgentProcessTerminator()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=FakeAgentLauncher(),
        capability_probe=CapabilityResponses([]),
        process_liveness=AgentProcessLiveness([True]),
        process_terminator=terminator,
    )

    with pytest.raises(LocalAgentConfigurationError, match="different runtime profile"):
        supervisor.restart(requested_profile_path, profile)

    assert terminator.pids == []


def test_agent_supervisor_restart_refuses_a_reused_pid_that_is_not_an_eyvan_agent(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9134,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    terminator = AgentProcessTerminator()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=FakeAgentLauncher(),
        capability_probe=CapabilityResponses([]),
        process_argument_reader=lambda pid: (sys.executable, "-m", "unrelated.service") if pid == 9134 else None,
        process_liveness=AgentProcessLiveness([True, True]),
        process_terminator=terminator,
    )

    with pytest.raises(LocalAgentConfigurationError, match="not the expected Eyvan cell-agent process"):
        supervisor.restart(profile_path, profile)

    assert terminator.pids == []


@dataclass
class FakeControlPlaneSupervisor:
    """Record profile-aware API startup without starting a real local server."""

    calls: list[tuple[ControlPlaneEndpoint, Path | None]] = field(default_factory=list)
    stop_calls: list[tuple[LocalDaemonResult, Path]] = field(default_factory=list)

    def ensure(
        self,
        endpoint: str | ControlPlaneEndpoint,
        *,
        runtime_profile_path: Path | None = None,
    ) -> LocalDaemonResult:
        """Return a deterministic started API result for the supplied profile path."""

        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        self.calls.append((configured_endpoint, runtime_profile_path))
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.STARTED,
            pid=7001,
            log_path=Path("/tmp/eyvan-control-plane.log"),
        )

    def stop_started(self, result: LocalDaemonResult, *, runtime_profile_path: Path) -> None:
        """Record rollback of an API process started by this fake supervisor."""

        self.stop_calls.append((result, runtime_profile_path))


@dataclass
class FakeCellAgentSupervisor:
    """Record private profile handoff without starting a real cell-agent process."""

    calls: list[tuple[Path, GcpNativeRuntimeProfile]] = field(default_factory=list)
    reference_digests: list[str | None] = field(default_factory=list)

    def ensure(
        self, profile_path: Path, profile: GcpNativeRuntimeProfile, *, reference_digest: str | None = None
    ) -> LocalAgentResult:
        """Return a deterministic authenticated local-agent result."""

        self.calls.append((profile_path, profile))
        self.reference_digests.append(reference_digest)
        endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
        return LocalAgentResult(
            endpoint=endpoint,
            state=LocalAgentState.STARTED,
            pid=7002,
            log_path=Path("/tmp/eyvan-cell-agent.log"),
            capabilities=_capabilities(),
        )


@dataclass
class FailingCellAgentSupervisor:
    """Fail deterministically after the control plane starts."""

    calls: list[tuple[Path, GcpNativeRuntimeProfile]] = field(default_factory=list)

    def ensure(
        self, profile_path: Path, profile: GcpNativeRuntimeProfile, *, reference_digest: str | None = None
    ) -> LocalAgentResult:
        """Record the handoff and simulate an authenticated agent startup failure."""

        self.calls.append((profile_path, profile))
        raise LocalAgentConfigurationError("the local cell agent rejected the selected runtime profile")


def test_runtime_supervisor_loads_one_profile_for_both_private_processes(tmp_path: Path) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "runtime" / "gcp-native-runtime.json")
    profile = store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'b' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )
    control_plane = FakeControlPlaneSupervisor()
    cell_agent = FakeCellAgentSupervisor()
    supervisor = GcpNativeRuntimeSupervisor(
        profile_store=store,
        control_plane=control_plane,
        agent=cell_agent,
    )

    result = supervisor.ensure("http://127.0.0.1:19090")

    assert result.profile == profile
    assert control_plane.calls == [(ControlPlaneEndpoint.parse("http://127.0.0.1:19090"), store.path)]
    assert cell_agent.calls == [(store.path, profile)]


def test_runtime_supervisor_requires_a_staged_worker_wheel_before_starting_processes(tmp_path: Path) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "runtime" / "gcp-native-runtime.json")
    store.create(project_id="eyvan-project", artifact_root_uri="gs://eyvan-artifacts/eyvan")
    control_plane = FakeControlPlaneSupervisor()
    cell_agent = FakeCellAgentSupervisor()
    supervisor = GcpNativeRuntimeSupervisor(
        profile_store=store,
        control_plane=control_plane,
        agent=cell_agent,
    )

    with pytest.raises(GcpRuntimeProfileNotPreparedError, match="eyvan runtime init"):
        supervisor.ensure("http://127.0.0.1:19090")

    assert control_plane.calls == []
    assert cell_agent.calls == []


@pytest.mark.parametrize("endpoint", ("http://localhost:19090", "https://example.com"))
def test_runtime_supervisor_rejects_nonliteral_or_remote_endpoints_without_starting_processes(
    tmp_path: Path,
    endpoint: str,
) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "runtime" / "gcp-native-runtime.json")
    store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'c' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )
    control_plane = FakeControlPlaneSupervisor()
    cell_agent = FakeCellAgentSupervisor()
    supervisor = GcpNativeRuntimeSupervisor(
        profile_store=store,
        control_plane=control_plane,
        agent=cell_agent,
    )

    with pytest.raises(GcpRuntimeError, match="literal HTTP loopback"):
        supervisor.ensure(endpoint)

    assert control_plane.calls == []
    assert cell_agent.calls == []


def test_runtime_supervisor_rolls_back_a_new_control_plane_when_agent_startup_fails(tmp_path: Path) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "runtime" / "gcp-native-runtime.json")
    profile = store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'d' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )
    control_plane = FakeControlPlaneSupervisor()
    cell_agent = FailingCellAgentSupervisor()
    supervisor = GcpNativeRuntimeSupervisor(
        profile_store=store,
        control_plane=control_plane,
        agent=cell_agent,
    )

    with pytest.raises(LocalAgentConfigurationError, match="rejected"):
        supervisor.ensure("http://127.0.0.1:19090")

    assert cell_agent.calls == [(store.path, profile)]
    assert len(control_plane.stop_calls) == 1
    stopped, profile_path = control_plane.stop_calls[0]
    assert stopped.state is LocalDaemonState.STARTED
    assert profile_path == store.path


@dataclass
class BuildResponses:
    """Return an ordered sequence of agent build identities."""

    values: list[AgentBuild | None]

    def __call__(self, endpoint: ControlPlaneEndpoint, secret: str) -> AgentBuild | None:
        """Return the next configured build identity."""

        del endpoint, secret
        if not self.values:
            raise AssertionError("unexpected cell-agent build probe")
        return self.values.pop(0)


def _agent_build(source_digest: str | None) -> AgentBuild:
    return AgentBuild(
        version="0.0.0",
        origin="source",
        source_digest=source_digest,
        python="3.13.0",
        platform="test",
    )


@pytest.mark.parametrize("caller_digest", ["plane-build", None, "caller-build"])
def test_missing_agent_startup_rejects_known_caller_skew(monkeypatch, tmp_path, caller_digest) -> None:
    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build(caller_digest))
    launcher = FakeAgentLauncher()
    builds = BuildResponses([None, None, _agent_build("plane-build")])
    supervisor = LocalAgentSupervisor(
        paths=LocalDaemonPaths(tmp_path),
        launcher=launcher,
        capability_probe=CapabilityResponses([None, None, _capabilities()]),
        build_probe=builds,
        process_liveness=AgentProcessLiveness([]),
    )
    if caller_digest == "caller-build":
        with pytest.raises(LocalAgentConfigurationError, match="runtime deploy --yes"):
            supervisor.ensure(tmp_path / "profile.json", _profile(), reference_digest="plane-build")
        assert launcher.commands == []
    else:
        result = supervisor.ensure(tmp_path / "profile.json", _profile(), reference_digest="plane-build")
        assert result.state is LocalAgentState.STARTED
        assert builds.values == []


@pytest.mark.parametrize("remote_digest", ["plane-build", None, "unexpected-build"])
def test_started_agent_build_is_checked_and_mismatch_cleaned_up(monkeypatch, tmp_path, remote_digest) -> None:
    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build("plane-build"))
    launcher = FakeAgentLauncher()
    builds = BuildResponses([None, None, _agent_build(remote_digest)])
    supervisor = LocalAgentSupervisor(
        paths=LocalDaemonPaths(tmp_path),
        launcher=launcher,
        capability_probe=CapabilityResponses([None, None, _capabilities()]),
        build_probe=builds,
        process_liveness=AgentProcessLiveness([]),
    )
    if remote_digest == "unexpected-build":
        with pytest.raises(LocalAgentConfigurationError, match="runtime deploy --yes"):
            supervisor.ensure(tmp_path / "profile.json", _profile(), reference_digest="plane-build")
        assert launcher.process.terminated
        assert read_owner_only_record(supervisor.paths.agent_record_path(supervisor._target())) is None
    else:
        result = supervisor.ensure(tmp_path / "profile.json", _profile(), reference_digest="plane-build")
        assert result.state is LocalAgentState.STARTED
        assert not launcher.process.terminated
    assert builds.values == []


def test_agent_supervisor_replaces_an_owned_agent_running_different_code(monkeypatch, tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9140,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build("current-build-digest"), raising=False)
    launcher = FakeAgentLauncher(process=FakeProcess(pid=9141))
    capabilities = _capabilities()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=launcher,
        capability_probe=CapabilityResponses([None, None, capabilities]),
        # The stale agent is stopped, absence rechecked, then the new build verified.
        build_probe=BuildResponses([_agent_build("stale-agent-digest"), None, _agent_build("current-build-digest")]),
        process_argument_reader=lambda pid: (
            sys.executable,
            "-m",
            "eyvan.worker.agent.cli",
            "--host",
            "127.0.0.1",
            "--port",
            "8090",
            "--runtime-profile",
            str(profile_path),
            "--log-level",
            "info",
        ),
        process_liveness=AgentProcessLiveness([True, True, True, False]),
        process_terminator=AgentProcessTerminator(),
        process_killer=AgentProcessKiller(),
        clock=AdvancingClock(),
        sleeper=lambda duration: None,
    )

    result = supervisor.ensure(profile_path, profile)

    assert result.state is LocalAgentState.STARTED
    assert result.pid == 9141
    assert len(launcher.commands) == 1


def test_agent_supervisor_refuses_to_replace_an_agent_it_does_not_own(monkeypatch, tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build("current-build-digest"), raising=False)
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=FakeAgentLauncher(),
        capability_probe=CapabilityResponses([_capabilities()]),
        build_probe=BuildResponses([_agent_build("stale-agent-digest")]),
        process_liveness=AgentProcessLiveness([]),
    )

    with pytest.raises(LocalAgentConfigurationError, match="different code"):
        supervisor.ensure(tmp_path / "runtime.json", profile)


def test_agent_supervisor_reuses_an_owned_agent_running_this_build(monkeypatch, tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9142,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build("current-build-digest"), raising=False)
    capabilities = _capabilities()
    launcher = FakeAgentLauncher()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=launcher,
        capability_probe=CapabilityResponses([capabilities]),
        # Two build probes: the pre-lock replacement check, then the reuse
        # path re-verifying the running agent's digest before returning it.
        build_probe=BuildResponses([_agent_build("current-build-digest"), _agent_build("current-build-digest")]),
        process_liveness=AgentProcessLiveness([True]),
    )

    result = supervisor.ensure(profile_path, profile)

    assert result.state is LocalAgentState.REUSED
    assert launcher.commands == []


def test_agent_supervisor_replaces_a_stale_agent_relaunched_after_the_prelock_digest_check(
    monkeypatch, tmp_path: Path
) -> None:
    """A stale relaunch in the window before the startup lock must not be reused.

    A concurrent supervisor running older code can write a live record for a
    stale agent between this process's pre-lock digest probe and its lock. The
    in-lock re-check must stop that agent and start a fresh one instead of
    handing the stale build back as a REUSED result.
    """

    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(
            endpoint=endpoint,
            pid=9143,
            runtime_profile_path=str(profile_path.resolve()),
        ),
    )
    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build("current-build-digest"), raising=False)
    launcher = FakeAgentLauncher(process=FakeProcess(pid=9144))
    capabilities = _capabilities()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=launcher,
        # Probes 1-2 (pre-lock reuse check, then one wait iteration): the
        # stale agent is not answering yet. Probe 3: it came up ready and is
        # replaced. Probe 4 (in-lock reuse check): it was stopped and cleared.
        # Probe 5: the freshly started agent is ready.
        capability_probe=CapabilityResponses([None, None, capabilities, None, capabilities]),
        # Probe 1 (pre-lock): nothing digest-divergent is running. Probes 2-3
        # (in-lock re-check, then its replacement stop): the concurrent
        # relaunch answers with a stale digest. Probe 4: it is gone. Probe 5: new build.
        build_probe=BuildResponses(
            [
                _agent_build("current-build-digest"),
                _agent_build("stale-agent-digest"),
                _agent_build("stale-agent-digest"),
                None,
                _agent_build("current-build-digest"),
            ]
        ),
        process_argument_reader=lambda pid: (
            sys.executable,
            "-m",
            "eyvan.worker.agent.cli",
            "--host",
            "127.0.0.1",
            "--port",
            "8090",
            "--runtime-profile",
            str(profile_path),
            "--log-level",
            "info",
        ),
        process_liveness=AgentProcessLiveness([True, True, True, True, False]),
        process_terminator=AgentProcessTerminator(),
        process_killer=AgentProcessKiller(),
        clock=AdvancingClock(),
        sleeper=lambda duration: None,
    )

    result = supervisor.ensure(profile_path, profile)

    assert result.state is LocalAgentState.STARTED
    assert result.pid == 9144
    assert len(launcher.commands) == 1


def test_stop_ends_a_recorded_cell_agent_and_refuses_foreign_processes(tmp_path: Path) -> None:
    """``ensure`` without a ``stop`` is how an agent outlives its plane's profile.

    A cell agent recorded under a profile directory keeps its port and its
    log long after the plane -- and the tests, or the operator session --
    that started it are gone; two such orphans were found still running
    days later. The stop demands the same ownership proof as a restart:
    the record names this endpoint, and the recorded PID still runs the
    expected Eyvan cell agent under its recorded runtime profile.
    """

    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    profile_id = str((tmp_path / "runtime" / "gcp-native-runtime.json").resolve())
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    record_path = paths.agent_record_path(target)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_owner_only_record(record_path, LocalDaemonRecord(endpoint=endpoint, pid=9177, runtime_profile_path=profile_id))

    terminated: set[int] = set()
    graceful: list[int] = []

    def _liveness(pid: int) -> bool:
        return pid not in terminated

    def _terminate(pid: int) -> None:
        graceful.append(pid)
        terminated.add(pid)

    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=FakeAgentLauncher(),
        capability_probe=CapabilityResponses([]),
        process_argument_reader=lambda pid: (
            (
                sys.executable,
                "-m",
                "eyvan.worker.agent.cli",
                "--host",
                "127.0.0.1",
                "--port",
                "8090",
                "--runtime-profile",
                profile_id,
                "--log-level",
                "info",
            )
            if pid == 9177
            else None
        ),
        process_liveness=_liveness,
        process_terminator=_terminate,
        process_killer=AgentProcessKiller(),
    )

    supervisor.stop()

    assert graceful == [9177], "the recorded agent must be gracefully terminated, exactly once"
    assert read_owner_only_record(record_path) is None, "stopping must clear the record it owned"
    # A cleared record makes a second stop a silent no-op, not an error.
    supervisor.stop()

    # A foreign command under a recorded PID is refused, not killed.
    write_owner_only_record(record_path, LocalDaemonRecord(endpoint=endpoint, pid=9178, runtime_profile_path=profile_id))
    foreign = LocalAgentSupervisor(
        paths=paths,
        launcher=FakeAgentLauncher(),
        capability_probe=CapabilityResponses([]),
        process_argument_reader=lambda pid: (sys.executable, "-m", "unrelated.service") if pid == 9178 else None,
        process_liveness=lambda _pid: True,
        process_terminator=_terminate,
        process_killer=AgentProcessKiller(),
    )
    with pytest.raises(LocalAgentConfigurationError, match="refusing to stop it"):
        foreign.stop()
    assert 9178 not in graceful
    assert read_owner_only_record(record_path) is not None, "a refusal must leave the record in place"


# --------------------------------------------------------------------------- one runtime, one build


def _plane_store(tmp_path: Path) -> GcpNativeRuntimeProfileStore:
    store = GcpNativeRuntimeProfileStore(tmp_path / "runtime" / "gcp-native-runtime.json")
    store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'b' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )
    return store


def test_the_agent_is_judged_against_the_serving_plane_not_the_calling_cli(monkeypatch, tmp_path: Path) -> None:
    """Compare the agent with the serving plane's build so an edited CLI cannot split the runtime's identity."""

    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "serving_source_digest", lambda endpoint: "sha256:plane")
    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build("sha256:edited-cli"), raising=False)
    cell_agent = FakeCellAgentSupervisor()
    supervisor = GcpNativeRuntimeSupervisor(
        profile_store=_plane_store(tmp_path),
        control_plane=FakeControlPlaneSupervisor(),
        agent=cell_agent,
    )

    with pytest.warns(RuntimeWarning, match="eyvan runtime deploy"):
        supervisor.ensure("http://127.0.0.1:19090")

    assert cell_agent.reference_digests == ["sha256:plane"]


def test_a_plane_without_a_recorded_build_keeps_the_callers_reference(monkeypatch, tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "serving_source_digest", lambda endpoint: None)
    cell_agent = FakeCellAgentSupervisor()
    supervisor = GcpNativeRuntimeSupervisor(
        profile_store=_plane_store(tmp_path),
        control_plane=FakeControlPlaneSupervisor(),
        agent=cell_agent,
    )

    supervisor.ensure("http://127.0.0.1:19090")

    assert cell_agent.reference_digests == [None]


def test_an_agent_matching_the_plane_is_reused_though_the_cli_differs(monkeypatch, tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module

    profile_path = tmp_path / "runtime" / "gcp-native-runtime.json"
    profile = _profile()
    paths = LocalDaemonPaths(tmp_path / "eyvan")
    endpoint = ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    write_owner_only_record(
        paths.agent_record_path(target),
        LocalDaemonRecord(endpoint=endpoint, pid=9142, runtime_profile_path=str(profile_path.resolve())),
    )
    monkeypatch.setattr(supervisor_module, "build_identity", lambda: _agent_build("sha256:edited-cli"), raising=False)
    launcher = FakeAgentLauncher()
    supervisor = LocalAgentSupervisor(
        paths=paths,
        launcher=launcher,
        capability_probe=CapabilityResponses([_capabilities()]),
        build_probe=BuildResponses([_agent_build("sha256:plane"), _agent_build("sha256:plane")]),
        process_liveness=AgentProcessLiveness([True]),
    )

    result = supervisor.ensure(profile_path, profile, reference_digest="sha256:plane")

    assert result.state is LocalAgentState.REUSED
    assert launcher.commands == []
