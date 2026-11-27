# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Focused coverage for one-time native GCP runtime CLI setup."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from click.testing import CliRunner

import eyvan.sdk.cli as cli
import eyvan.sdk.commands.runtime as runtime_commands
from eyvan.capacity.autoscale import AutoscaleConfigCache, AutoscaleDefaults
from eyvan.capacity.providers.gcp.runtime.bootstrap import (
    GcpRuntimeBootstrapStep,
    GcpRuntimePreflight,
    GcpRuntimePreflightError,
)
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile, GcpNativeRuntimeProfileStore
from eyvan.capacity.providers.gcp.runtime.supervisor import LocalAgentResult, LocalAgentState
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.server.local.daemon import LocalDaemonResult, LocalDaemonState
from eyvan.worker.agent.models import AgentCapabilities


def _profile(*, wheel_uri: str | None = None) -> GcpNativeRuntimeProfile:
    """Build one profile with recognizable secret values for output-safety tests."""

    return GcpNativeRuntimeProfile(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        agent_shared_secret="agent-secret-" + "a" * 32,
        native_worker_root_secret="worker-secret-" + "b" * 32,
        native_worker_wheel_uri=wheel_uri,
    )


@dataclass
class FakeBootstrap:
    """In-memory GCP preparation boundary for service-level coverage."""

    profiles: list[GcpNativeRuntimeProfile] = field(default_factory=list)
    ensured_buckets: list[str] = field(default_factory=list)

    def preflight(self, profile: GcpNativeRuntimeProfile) -> GcpRuntimePreflight:
        self.profiles.append(profile)
        return GcpRuntimePreflight(
            project_id=profile.project_id,
            artifact_bucket_uri="gs://eyvan-runtime-artifacts",
            fleet_state_uri=profile.fleet_state_uri,
            checks=(
                GcpRuntimeBootstrapStep.PROJECT_ACCESS,
                GcpRuntimeBootstrapStep.STORAGE_ACCESS,
                GcpRuntimeBootstrapStep.TPU_ACCESS,
            ),
        )

    def ensure_artifact_bucket(self, profile: GcpNativeRuntimeProfile) -> str:
        bucket_uri = profile.artifact_root_uri.split("/", 3)[2]
        self.ensured_buckets.append(f"gs://{bucket_uri}")
        return f"gs://{bucket_uri}"

    def stage_native_worker_wheel(self, profile: GcpNativeRuntimeProfile) -> GcpNativeRuntimeProfile:
        self.profiles.append(profile)
        return replace(
            profile,
            native_worker_wheel_uri=(
                f"{profile.artifact_root_uri}/runtime/wheels/{'c' * 64}/eyvan-0.1.0-py3-none-any.whl"
            ),
        )


def test_initializer_persists_only_the_staged_profile_and_returns_nonsecret_summary(tmp_path: Path) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "profile" / "runtime.json")
    bootstrap = FakeBootstrap()

    result = runtime_commands.GcpRuntimeInitializer(store, bootstrap).initialize(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan/",
    )

    assert result.profile_path == store.path
    assert store.load() == result.profile
    # Preflight and staging see the profile; the bucket is ensured in between,
    # so a first run never depends on a hand-created bucket.
    assert bootstrap.ensured_buckets == ["gs://eyvan-runtime-artifacts"]
    assert len(bootstrap.profiles) == 2
    assert bootstrap.profiles[0].native_worker_wheel_uri is None
    assert result.profile.native_worker_wheel_uri == (
        f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'c' * 64}/eyvan-0.1.0-py3-none-any.whl"
    )
    document = result.as_document()
    encoded = json.dumps(document)
    assert "agent-secret-" not in encoded
    assert "worker-secret-" not in encoded
    assert document["checks"] == ["project_access", "storage_access", "tpu_access"]


@dataclass
class MissingWheelBootstrap:
    """Invalid bootstrap result used to keep the result contract explicit."""

    def preflight(self, profile: GcpNativeRuntimeProfile) -> GcpRuntimePreflight:
        return GcpRuntimePreflight(
            project_id=profile.project_id,
            artifact_bucket_uri="gs://eyvan-runtime-artifacts",
            fleet_state_uri=profile.fleet_state_uri,
            checks=(),
        )

    def ensure_artifact_bucket(self, profile: GcpNativeRuntimeProfile) -> str:
        return profile.artifact_root_uri

    def stage_native_worker_wheel(self, profile: GcpNativeRuntimeProfile) -> GcpNativeRuntimeProfile:
        return profile


def test_initializer_rejects_a_bootstrapper_that_does_not_return_a_pinned_wheel(tmp_path: Path) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "profile" / "runtime.json")

    with pytest.raises(runtime_commands.GcpRuntimeInitializationError, match="without a worker wheel URI"):
        runtime_commands.GcpRuntimeInitializer(store, MissingWheelBootstrap()).initialize(
            project_id="example-project-1",
            artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        )


@dataclass
class FakeUvBuildRunner:
    """Create one local wheel where the builder requested it."""

    calls: list[tuple[str, ...]] = field(default_factory=list)

    def __call__(
        self,
        args: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(args)
        assert cwd == Path(args[-1])
        assert check is False
        assert capture_output is True
        assert timeout == 12.0
        output_directory = Path(args[4])
        (output_directory / "eyvan-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")


def test_uv_wheel_builder_keeps_the_local_wheel_temporary(tmp_path: Path) -> None:
    package_root = tmp_path / "eyvan-package"
    package_root.mkdir()
    runner = FakeUvBuildRunner()
    builder = runtime_commands.UvNativeWorkerWheelBuilder(package_root=package_root, runner=runner, timeout_s=12)

    wheel = builder.build()
    output_directory = wheel.parent

    assert wheel.read_bytes() == b"wheel"
    assert runner.calls == [
        ("uv", "build", "--wheel", "--out-dir", str(output_directory), str(package_root)),
    ]
    builder.close()
    assert not output_directory.exists()


def _result(profile_path: Path) -> runtime_commands.RuntimeInitializationResult:
    """Return a success result for CLI rendering without invoking gcloud or uv."""

    profile = _profile(
        wheel_uri=f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'d' * 64}/eyvan-0.1.0-py3-none-any.whl"
    )
    return runtime_commands.RuntimeInitializationResult(
        profile_path=profile_path,
        profile=profile,
        preflight=GcpRuntimePreflight(
            project_id=profile.project_id,
            artifact_bucket_uri="gs://eyvan-runtime-artifacts",
            fleet_state_uri=profile.fleet_state_uri,
            checks=(GcpRuntimeBootstrapStep.PROJECT_ACCESS,),
        ),
    )


def test_runtime_init_json_uses_explicit_public_inputs_and_never_emits_profile_secrets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "private" / "runtime.json"
    calls: list[tuple[str, str, Path]] = []

    def initialize(
        *,
        project_id: str,
        artifact_root_uri: str,
        profile_path: Path,
    ) -> runtime_commands.RuntimeInitializationResult:
        calls.append((project_id, artifact_root_uri, profile_path))
        return _result(profile_path)

    monkeypatch.setattr(runtime_commands, "_initialize_native_runtime", initialize)

    invoked = CliRunner().invoke(
        runtime_commands.runtime_group,
        [
            "init",
            "--project",
            "example-project-1",
            "--artifact-root",
            "gs://eyvan-runtime-artifacts/eyvan/",
            "--profile",
            str(profile_path),
            "--yes",
            "--json",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    assert calls == [("example-project-1", "gs://eyvan-runtime-artifacts/eyvan", profile_path)]
    payload = json.loads(invoked.output)
    assert payload["profile_path"] == str(profile_path)
    assert payload["native_worker_wheel_uri"].endswith(f"{'d' * 64}/eyvan-0.1.0-py3-none-any.whl")
    assert "agent-secret-" not in invoked.output
    assert "worker-secret-" not in invoked.output


def test_runtime_init_derives_an_eyvan_managed_root_in_noninteractive_mode(monkeypatch, tmp_path: Path) -> None:
    """No --artifact-root names the Eyvan-managed bucket, which init itself creates."""

    calls: list[tuple[str, str, Path]] = []

    def initialize(
        *,
        project_id: str,
        artifact_root_uri: str,
        profile_path: Path,
    ) -> runtime_commands.RuntimeInitializationResult:
        calls.append((project_id, artifact_root_uri, profile_path))
        return _result(profile_path)

    monkeypatch.setattr(runtime_commands, "_initialize_native_runtime", initialize)

    invoked = CliRunner().invoke(
        runtime_commands.runtime_group,
        ["init", "--project", "example-project-1", "--profile", str(tmp_path / "runtime.json"), "--yes"],
    )

    assert invoked.exit_code == 0, invoked.output
    assert calls == [("example-project-1", "gs://eyvan-example-project-1", tmp_path / "runtime.json")]


def test_runtime_group_is_registered_without_service_account_or_agent_endpoint_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))

    invoked = CliRunner().invoke(cli.main, ["runtime", "init", "--help"])

    assert invoked.exit_code == 0, invoked.output
    assert "--artifact-root" in invoked.output
    assert "service account" not in invoked.output.lower()
    assert "agent endpoint" not in invoked.output.lower()


def test_runtime_project_suggestion_prefers_the_autoscale_yaml_project(monkeypatch, tmp_path: Path) -> None:
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.save_defaults(
        AutoscaleDefaults(
            project="example-project-1",
            fleet_prefix="research",
            bootstrap_egress="public-internet",
        )
    )
    monkeypatch.setattr(runtime_commands, "gcloud_default_project", lambda: "another-project")

    suggested = runtime_commands.suggested_runtime_project(defaults_path=None, cache_dir=tmp_path / "autoscale")

    assert suggested == "example-project-1"


def test_runtime_init_translates_typed_gcp_preflight_failures(monkeypatch, tmp_path: Path) -> None:
    def initialize(
        *,
        project_id: str,
        artifact_root_uri: str,
        profile_path: Path,
    ) -> runtime_commands.RuntimeInitializationResult:
        raise GcpRuntimePreflightError(
            GcpRuntimeBootstrapStep.STORAGE_ACCESS,
            "PERMISSION_DENIED: bucket access is missing",
        )

    monkeypatch.setattr(runtime_commands, "_initialize_native_runtime", initialize)

    invoked = CliRunner().invoke(
        runtime_commands.runtime_group,
        [
            "init",
            "--project",
            "example-project-1",
            "--artifact-root",
            "gs://eyvan-runtime-artifacts/eyvan",
            "--profile",
            str(tmp_path / "private" / "runtime.json"),
            "--yes",
        ],
    )

    assert invoked.exit_code != 0
    assert "could not initialize native GCP runtime" in invoked.output
    assert "storage access failed" in invoked.output
    assert "PERMISSION_DENIED" in invoked.output


def _restart_result(endpoint: ControlPlaneEndpoint) -> runtime_commands.RuntimeRestartResult:
    """Return a non-secret restarted-runtime summary without starting local processes."""

    return runtime_commands.RuntimeRestartResult(
        endpoint=endpoint,
        control_plane=LocalDaemonResult(
            endpoint=endpoint,
            state=LocalDaemonState.STARTED,
            pid=9001,
            log_path=Path("/tmp/eyvan-control-plane.log"),
        ),
        agent_endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
        agent_state=LocalAgentState.STARTED,
    )


def test_runtime_restart_requires_explicit_confirmation_outside_a_terminal(tmp_path: Path) -> None:
    invoked = CliRunner().invoke(
        runtime_commands.runtime_group,
        ["restart", "--profile", str(tmp_path / "private" / "runtime.json")],
    )

    assert invoked.exit_code != 0
    assert "runtime restart requires --yes outside a terminal" in invoked.output


def test_runtime_restart_uses_the_root_cli_url_and_emits_no_profile_credentials(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "private" / "runtime.json"
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19091")
    calls: list[tuple[ControlPlaneEndpoint, Path]] = []

    def restart(*, endpoint: ControlPlaneEndpoint, profile_path: Path) -> runtime_commands.RuntimeRestartResult:
        calls.append((endpoint, profile_path))
        return _restart_result(endpoint)

    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    monkeypatch.setattr(runtime_commands, "_restart_native_runtime", restart)

    invoked = CliRunner().invoke(
        cli.main,
        [
            "--url",
            endpoint.url,
            "runtime",
            "restart",
            "--profile",
            str(profile_path),
            "--yes",
            "--json",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    assert calls == [(endpoint, profile_path)]
    payload = json.loads(invoked.output)
    assert payload == {
        "agent": {"endpoint": "http://127.0.0.1:8090", "state": "started"},
        # The replacement inherits the invoking environment, so a restart from a
        # shell without EYVAN_TRUSTED_CALLABLE_PROJECTS silently narrows what the
        # control plane will accept. The document reports that, and still carries
        # no profile credential.
        "callable_trust_warning": runtime_commands.RuntimeRestartResult(
            endpoint=endpoint,
            control_plane=_restart_result(endpoint).control_plane,
            agent_endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
            agent_state=runtime_commands.LocalAgentState.STARTED,
        ).callable_trust_warning,
        "trusted_callable_projects": [],
        "control_plane": {"endpoint": endpoint.url, "pid": 9001, "state": "started"},
        "endpoint": endpoint.url,
    }
    assert "agent-secret-" not in invoked.output
    assert "worker-secret-" not in invoked.output


def test_restart_service_requires_an_initialized_profile_before_stopping_a_daemon(tmp_path: Path) -> None:
    with pytest.raises(runtime_commands.GcpRuntimeRestartError, match="run `eyvan runtime init`"):
        runtime_commands._restart_native_runtime(
            endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:19092"),
            profile_path=tmp_path / "private" / "runtime.json",
        )


def test_restart_service_replaces_the_profile_owned_cell_agent(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "private" / "runtime.json"
    store = GcpNativeRuntimeProfileStore(profile_path)
    profile = store.create(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        native_worker_wheel_uri=(
            f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'e' * 64}/eyvan-0.1.0-py3-none-any.whl"
        ),
    )
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19093")
    control_plane_result = LocalDaemonResult(
        endpoint=endpoint,
        state=LocalDaemonState.STARTED,
        pid=9010,
        log_path=Path("/tmp/eyvan-control-plane.log"),
    )
    control_plane_calls: list[tuple[ControlPlaneEndpoint, Path]] = []
    agent_calls: list[tuple[Path, GcpNativeRuntimeProfile]] = []

    class RestartControlPlane:
        """Capture the exact verified API restart requested by the runtime command."""

        def restart(
            self,
            configured_endpoint: ControlPlaneEndpoint,
            *,
            runtime_profile_path: Path,
        ) -> LocalDaemonResult:
            control_plane_calls.append((configured_endpoint, runtime_profile_path))
            return control_plane_result

        def stop_started(self, result: LocalDaemonResult, *, runtime_profile_path: Path) -> None:
            raise AssertionError(f"unexpected rollback for {result} and {runtime_profile_path}")

    class RestartAgent:
        """Capture the forced agent replacement requested by the runtime command."""

        def restart(
            self,
            configured_profile_path: Path,
            configured_profile: GcpNativeRuntimeProfile,
        ) -> LocalAgentResult:
            agent_calls.append((configured_profile_path, configured_profile))
            return LocalAgentResult(
                endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
                state=LocalAgentState.STARTED,
                pid=9011,
                log_path=Path("/tmp/eyvan-cell-agent.log"),
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
            )

    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", RestartControlPlane)
    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", RestartAgent)

    result = runtime_commands._restart_native_runtime(endpoint=endpoint, profile_path=profile_path)

    assert result.control_plane == control_plane_result
    assert result.agent_endpoint == ControlPlaneEndpoint.parse("http://127.0.0.1:8090")
    assert result.agent_state is LocalAgentState.STARTED
    assert control_plane_calls == [(endpoint, store.path)]
    assert agent_calls == [(store.path, profile)]


def test_runtime_stop_stops_a_supervised_loopback_plane_and_emits_json(monkeypatch, tmp_path: Path) -> None:
    """`eyvan runtime stop` wires to LocalDaemonSupervisor.stop() and respects root --url."""

    from eyvan.server.local.daemon import LocalDaemonPaths, LocalDaemonRecord, LocalDaemonTarget, write_owner_only_record

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19094")
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    paths = LocalDaemonPaths(directory=tmp_path / ".eyvan")
    write_owner_only_record(paths.record_path(target), LocalDaemonRecord(endpoint=endpoint, pid=43210))
    stopped_endpoints: list[ControlPlaneEndpoint] = []

    class FakeSupervisor:
        def __init__(self) -> None:
            self.paths = paths

        def process_liveness(self, pid: int) -> bool:
            return pid == 43210

        def health_probe(self, _endpoint: ControlPlaneEndpoint) -> bool:
            return False

        def stop(self, target_endpoint: ControlPlaneEndpoint) -> LocalDaemonResult:
            stopped_endpoints.append(target_endpoint)
            return LocalDaemonResult(
                endpoint=target_endpoint,
                state=LocalDaemonState.STOPPED,
                pid=None,
                log_path=paths.log_path(target),
            )

    class IdleAgentSupervisor:
        def __init__(self) -> None:
            self.paths = paths
            self.endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8090")

        def process_liveness(self, _pid: int) -> bool:
            return False

        def stop(self, *, runtime_profile_path: Path | None = None) -> int | None:
            raise AssertionError(f"plain control plane without profile must not stop agent ({runtime_profile_path})")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", FakeSupervisor)
    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", IdleAgentSupervisor)

    invoked = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop", "--json"])

    assert invoked.exit_code == 0, invoked.output
    assert stopped_endpoints == [endpoint]
    assert json.loads(invoked.output) == {
        "agent": {"endpoint": "http://127.0.0.1:8090", "pid": None, "state": "not_running"},
        "control_plane": {"endpoint": endpoint.url, "pid": 43210, "state": "stopped"},
        "endpoint": endpoint.url,
        "running_unsupervised": False,
        "stopped": True,
    }


def test_runtime_stop_reports_cleanly_when_no_supervised_plane_is_running(monkeypatch, tmp_path: Path) -> None:
    """Stopping an idle endpoint exits 0 with a clear message and stopped=False in JSON."""

    from eyvan.server.local.daemon import LocalDaemonPaths, LocalDaemonTarget

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19095")
    target = LocalDaemonTarget.from_endpoint(endpoint)
    assert target is not None
    paths = LocalDaemonPaths(directory=tmp_path / ".eyvan")

    class IdleSupervisor:
        def __init__(self) -> None:
            self.paths = paths

        def process_liveness(self, _pid: int) -> bool:
            return False

        def health_probe(self, _endpoint: ControlPlaneEndpoint) -> bool:
            return False

        def stop(self, target_endpoint: ControlPlaneEndpoint) -> LocalDaemonResult:
            return LocalDaemonResult(
                endpoint=target_endpoint,
                state=LocalDaemonState.STOPPED,
                pid=None,
                log_path=paths.log_path(target),
            )

    class IdleAgentSupervisor:
        def __init__(self) -> None:
            self.paths = paths
            self.endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8090")

        def process_liveness(self, _pid: int) -> bool:
            return False

        def stop(self, *, runtime_profile_path: Path | None = None) -> int | None:
            return None

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", IdleSupervisor)
    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", IdleAgentSupervisor)

    text_result = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop"])
    assert text_result.exit_code == 0, text_result.output
    assert "No supervised control plane is running" in text_result.output
    assert "No supervised cell agent is running" in text_result.output

    json_result = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop", "--json"])
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output) == {
        "agent": {"endpoint": "http://127.0.0.1:8090", "pid": None, "state": "not_running"},
        "control_plane": {"endpoint": endpoint.url, "pid": None, "state": "not_running"},
        "endpoint": endpoint.url,
        "running_unsupervised": False,
        "stopped": False,
    }


def test_runtime_stop_stops_both_control_plane_and_matching_cell_agent_idempotently(monkeypatch, tmp_path: Path) -> None:
    """`eyvan runtime stop` stops both the plane and its profile-matched cell agent, and is idempotent."""

    from eyvan.server.local.daemon import (
        LocalDaemonPaths,
        LocalDaemonRecord,
        LocalDaemonTarget,
        clear_owner_only_record,
        write_owner_only_record,
    )

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19096")
    agent_endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8090")
    cp_target = LocalDaemonTarget.from_endpoint(endpoint)
    agent_target = LocalDaemonTarget.from_endpoint(agent_endpoint)
    assert cp_target is not None and agent_target is not None
    paths = LocalDaemonPaths(directory=tmp_path / ".eyvan")
    custom_profile = (tmp_path / "custom-runtime.json").resolve()
    write_owner_only_record(
        paths.record_path(cp_target),
        LocalDaemonRecord(endpoint=endpoint, pid=44001, runtime_profile_path=str(custom_profile)),
    )
    write_owner_only_record(
        paths.agent_record_path(agent_target),
        LocalDaemonRecord(endpoint=agent_endpoint, pid=44002, runtime_profile_path=str(custom_profile)),
    )
    alive = {44001, 44002}
    stopped_agent_profiles: list[Path | None] = []

    class FakeControlPlaneSupervisor:
        def __init__(self) -> None:
            self.paths = paths

        def process_liveness(self, pid: int) -> bool:
            return pid in alive

        def health_probe(self, _endpoint: ControlPlaneEndpoint) -> bool:
            return False

        def stop(self, target_endpoint: ControlPlaneEndpoint) -> LocalDaemonResult:
            if 44001 in alive:
                alive.remove(44001)
                clear_owner_only_record(paths.record_path(cp_target), pid=44001)
            return LocalDaemonResult(
                endpoint=target_endpoint,
                state=LocalDaemonState.STOPPED,
                pid=None,
                log_path=paths.log_path(cp_target),
            )

    class FakeAgentSupervisor:
        def __init__(self) -> None:
            self.paths = paths
            self.endpoint = agent_endpoint

        def process_liveness(self, pid: int) -> bool:
            return pid in alive

        def stop(self, *, runtime_profile_path: Path | None = None) -> int | None:
            stopped_agent_profiles.append(runtime_profile_path)
            if 44002 in alive and runtime_profile_path is not None and runtime_profile_path.resolve() == custom_profile:
                alive.remove(44002)
                clear_owner_only_record(paths.agent_record_path(agent_target), pid=44002)
                return 44002
            return None

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", FakeControlPlaneSupervisor)
    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", FakeAgentSupervisor)

    first = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop", "--json"])
    assert first.exit_code == 0, first.output
    assert stopped_agent_profiles == [custom_profile]
    assert json.loads(first.output) == {
        "agent": {"endpoint": agent_endpoint.url, "pid": 44002, "state": "stopped"},
        "control_plane": {"endpoint": endpoint.url, "pid": 44001, "state": "stopped"},
        "endpoint": endpoint.url,
        "running_unsupervised": False,
        "stopped": True,
    }

    # Second call is idempotent when both are already stopped.
    second = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop"])
    assert second.exit_code == 0, second.output
    assert "No supervised control plane is running" in second.output
    assert "No supervised cell agent is running" in second.output


def test_runtime_stop_stops_orphaned_cell_agent_and_respects_profile_matching(monkeypatch, tmp_path: Path) -> None:
    """When only the cell agent is running, stop cleans it up if its profile matches and skips other profiles."""

    from eyvan.server.local.daemon import (
        LocalDaemonPaths,
        LocalDaemonRecord,
        LocalDaemonTarget,
        clear_owner_only_record,
        write_owner_only_record,
    )

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19097")
    agent_endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:8090")
    cp_target = LocalDaemonTarget.from_endpoint(endpoint)
    agent_target = LocalDaemonTarget.from_endpoint(agent_endpoint)
    assert cp_target is not None and agent_target is not None
    paths = LocalDaemonPaths(directory=tmp_path / ".eyvan")
    other_profile = (tmp_path / "other-profile.json").resolve()
    write_owner_only_record(
        paths.agent_record_path(agent_target),
        LocalDaemonRecord(endpoint=agent_endpoint, pid=45002, runtime_profile_path=str(other_profile)),
    )
    alive = {45002}

    class IdleControlPlaneSupervisor:
        def __init__(self) -> None:
            self.paths = paths

        def process_liveness(self, _pid: int) -> bool:
            return False

        def health_probe(self, _endpoint: ControlPlaneEndpoint) -> bool:
            return False

        def stop(self, target_endpoint: ControlPlaneEndpoint) -> LocalDaemonResult:
            return LocalDaemonResult(
                endpoint=target_endpoint,
                state=LocalDaemonState.STOPPED,
                pid=None,
                log_path=paths.log_path(cp_target),
            )

    class ProfileScopedAgentSupervisor:
        def __init__(self) -> None:
            self.paths = paths
            self.endpoint = agent_endpoint

        def process_liveness(self, pid: int) -> bool:
            return pid in alive

        def stop(self, *, runtime_profile_path: Path | None = None) -> int | None:
            if 45002 in alive and runtime_profile_path is not None and runtime_profile_path.resolve() == other_profile:
                alive.remove(45002)
                clear_owner_only_record(paths.agent_record_path(agent_target), pid=45002)
                return 45002
            return None

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", IdleControlPlaneSupervisor)
    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", ProfileScopedAgentSupervisor)

    # Without --profile, an agent owned by a different profile is skipped and reported clearly.
    skipped_json = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop", "--json"])
    assert skipped_json.exit_code == 0, skipped_json.output
    assert json.loads(skipped_json.output) == {
        "agent": {"endpoint": agent_endpoint.url, "pid": None, "state": "skipped"},
        "control_plane": {"endpoint": endpoint.url, "pid": None, "state": "not_running"},
        "endpoint": endpoint.url,
        "running_unsupervised": False,
        "stopped": False,
    }
    skipped_text = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop"])
    assert skipped_text.exit_code == 0, skipped_text.output
    assert "belongs to a different runtime profile" in skipped_text.output
    assert 45002 in alive

    # Passing the matching --profile stops the orphaned cell agent and reports it in text and JSON.
    stopped_text = CliRunner().invoke(
        cli.main, ["--url", endpoint.url, "runtime", "stop", "--profile", str(other_profile)]
    )
    assert stopped_text.exit_code == 0, stopped_text.output
    assert "No supervised control plane is running" in stopped_text.output
    assert f"Stopped supervised cell agent at {agent_endpoint.url} (pid 45002)" in stopped_text.output
    assert 45002 not in alive

    # Re-create an orphaned cell agent under the default profile: plain `eyvan runtime stop --json` stops it.
    default_profile = runtime_commands.DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH.expanduser().resolve()
    write_owner_only_record(
        paths.agent_record_path(agent_target),
        LocalDaemonRecord(endpoint=agent_endpoint, pid=45003, runtime_profile_path=str(default_profile)),
    )
    alive.add(45003)

    class DefaultProfileAgentSupervisor(ProfileScopedAgentSupervisor):
        def stop(self, *, runtime_profile_path: Path | None = None) -> int | None:
            if (
                45003 in alive
                and runtime_profile_path is not None
                and runtime_profile_path.expanduser().resolve() == default_profile
            ):
                alive.remove(45003)
                clear_owner_only_record(paths.agent_record_path(agent_target), pid=45003)
                return 45003
            return None

    monkeypatch.setattr(runtime_commands, "LocalAgentSupervisor", DefaultProfileAgentSupervisor)
    stopped_default_json = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop", "--json"])
    assert stopped_default_json.exit_code == 0, stopped_default_json.output
    assert json.loads(stopped_default_json.output) == {
        "agent": {"endpoint": agent_endpoint.url, "pid": 45003, "state": "stopped"},
        "control_plane": {"endpoint": endpoint.url, "pid": None, "state": "not_running"},
        "endpoint": endpoint.url,
        "running_unsupervised": False,
        "stopped": True,
    }
    assert 45003 not in alive


def test_runtime_stop_refuses_mismatched_profile_against_live_control_plane(monkeypatch, tmp_path: Path) -> None:
    """Passing a different --profile than the live control plane's profile refuses before stopping anything."""

    from eyvan.server.local.daemon import LocalDaemonPaths, LocalDaemonRecord, LocalDaemonTarget, write_owner_only_record

    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19098")
    cp_target = LocalDaemonTarget.from_endpoint(endpoint)
    assert cp_target is not None
    paths = LocalDaemonPaths(directory=tmp_path / ".eyvan")
    profile_a = (tmp_path / "profile-a.json").resolve()
    profile_b = (tmp_path / "profile-b.json").resolve()
    write_owner_only_record(
        paths.record_path(cp_target),
        LocalDaemonRecord(endpoint=endpoint, pid=46001, runtime_profile_path=str(profile_a)),
    )

    class GuardedSupervisor:
        def __init__(self) -> None:
            self.paths = paths

        def process_liveness(self, pid: int) -> bool:
            return pid == 46001

        def health_probe(self, _endpoint: ControlPlaneEndpoint) -> bool:
            return True

        def stop(self, _endpoint: ControlPlaneEndpoint) -> LocalDaemonResult:
            raise AssertionError("must not stop a control plane belonging to another profile")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    monkeypatch.setattr(runtime_commands, "LocalDaemonSupervisor", GuardedSupervisor)

    result = CliRunner().invoke(cli.main, ["--url", endpoint.url, "runtime", "stop", "--profile", str(profile_b)])
    assert result.exit_code != 0
    assert "could not stop local control plane" in result.output
    assert "refusing to stop it" in result.output


def test_runtime_stop_rejects_non_loopback_endpoints_cleanly(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))

    result = CliRunner().invoke(cli.main, ["--url", "https://plane.example.com", "runtime", "stop"])

    assert result.exit_code != 0
    assert "could not stop local control plane" in result.output
