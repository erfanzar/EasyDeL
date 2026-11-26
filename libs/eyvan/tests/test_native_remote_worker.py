# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import stat
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import eyvan.worker.runtime.native_worker as native_worker
from eyvan._internal.access_logging import uvicorn_log_config
from eyvan._internal.json_types import JsonObject
from eyvan.capacity.drivers.base import (
    CapacityStatus,
    CellCapabilities,
    CellObservation,
    ExecutionWorkloadStatus,
    MutationContext,
    StaleMutationError,
    UnsupportedCapabilityError,
)
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.records import NativeContainerConfig
from eyvan.worker.agent.models import MutationModel, WorkloadRequest
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_worker import create_native_worker_app
from eyvan.worker.runtime.stage_executor import AssetStageVerdict, LocalAssetStageExecutor
from eyvan.worker.runtime.systemd_transient import SystemdCommandResult, SystemdTransientServiceExecutor


def context(*, key: str = "submit-1", generation: int = 1, token: str = "token-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def command(source: str) -> str:
    return shlex.join((sys.executable, "-c", source))


def spec(entrypoint: str, *, submission_id: str, workload_id: str | None = None) -> JsonObject:
    document: JsonObject = {
        "entrypoint": entrypoint,
        "submission_id": submission_id,
        "environment": {},
        "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
        "metadata": {"eyvan_workload_id": workload_id or submission_id},
    }
    return document


def wait_for(
    executor: NativeRemoteExecutor,
    remote_id: str,
    expected: set[ExecutionWorkloadStatus],
    *,
    timeout: float = 8,
) -> ExecutionWorkloadStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = executor.status(remote_id).status
        if status in expected:
            return status
        time.sleep(0.02)
    pytest.fail(f"execution {remote_id!r} did not reach {sorted(expected)}; got {executor.status(remote_id)!r}")


def remote_executor(tmp_path: Path, *, secret: str = "worker-secret") -> tuple[TestClient, NativeRemoteExecutor]:
    app = create_native_worker_app(state_dir=tmp_path / "worker", secret=secret)
    transport = TestClient(app)
    return transport, NativeRemoteExecutor("http://testserver", secret=secret, client=transport)


def test_native_worker_requires_bearer_auth_and_exposes_health(tmp_path: Path) -> None:
    app = create_native_worker_app(state_dir=tmp_path / "worker", secret="worker-secret")
    with TestClient(app) as transport:
        assert transport.get("/healthz").json() == {"status": "ok", "runtime": "native-worker"}
        assert transport.get("/api/v1/workloads/missing").status_code == 401
        assert (
            transport.get(
                "/api/v1/workloads/missing",
                headers={"Authorization": "Bearer wrong-secret"},
            ).status_code
            == 401
        )
        # A non-ASCII header must be a refused credential (401), not the 500
        # the TypeError inside compare_digest used to raise. Sent as the raw
        # latin-1 bytes a caller puts on the wire, which the ASGI server
        # decodes to str.
        assert (
            transport.get(
                "/api/v1/workloads/missing",
                headers={"Authorization": "Bearer wörker-secret".encode("latin-1")},
            ).status_code
            == 401
        )
        assert NativeRemoteExecutor("http://testserver", secret="worker-secret", client=transport).health() is True
        assert NativeRemoteExecutor("http://testserver", secret="wrong-secret", client=transport).health() is False


def test_native_worker_refuses_an_unauthenticated_service(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty bearer secret"):
        create_native_worker_app(state_dir=tmp_path / "worker")
    with pytest.raises(ValueError, match="non-empty bearer secret"):
        NativeRemoteExecutor("http://native-worker")


def test_native_worker_requires_a_logical_workload_id(tmp_path: Path) -> None:
    app = create_native_worker_app(state_dir=tmp_path / "worker", secret="worker-secret")
    with TestClient(app) as transport:
        response = transport.post(
            "/api/v1/workloads",
            headers={"Authorization": "Bearer worker-secret"},
            json=WorkloadRequest(
                mutation=MutationModel(idempotency_key="missing-workload", generation=1, fencing_token="token-1"),
                spec={
                    "entrypoint": command("print('missing')"),
                    "submission_id": "missing-workload",
                    "environment": {},
                    "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
                    "metadata": {},
                },
            ).model_dump(mode="json"),
        )

    assert response.status_code == 422
    assert "eyvan_workload_id" in response.json()["detail"]


def test_native_worker_main_uses_cli_and_environment(monkeypatch, tmp_path: Path) -> None:
    app = object()
    created: list[dict] = []
    served: list[dict] = []
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_SECRET", "environment-secret")
    monkeypatch.setattr(
        native_worker,
        "create_native_worker_app",
        lambda **kwargs: created.append(kwargs) or app,
    )
    monkeypatch.setattr("uvicorn.run", lambda selected, **kwargs: served.append({"app": selected, **kwargs}))

    assert native_worker.main(("--host", "127.0.0.1", "--port", "9123", "--state-dir", str(tmp_path))) == 0
    assert created == [
        {
            "state_dir": str(tmp_path),
            "secret": "environment-secret",
            "systemd_transient": False,
            "systemd_exit_type_cgroup": False,
            "runtime_env": None,
            "runtime_container": None,
            "container_artifact_mounts": (),
        }
    ]
    # The access log is uvicorn's own graph with Eyvan's redacting access
    # formatter, so a log-page request does not write its cursor to this host.
    assert served == [
        {"app": app, "host": "127.0.0.1", "port": 9123, "log_config": uvicorn_log_config()},
    ]


def test_native_worker_reads_the_bootstrap_systemd_compatibility_flag(monkeypatch, tmp_path: Path) -> None:
    app = object()
    created: list[dict] = []
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_SECRET", "environment-secret")
    monkeypatch.setenv("EYVAN_SYSTEMD_EXIT_TYPE_CGROUP", "1")
    monkeypatch.setattr(native_worker, "create_native_worker_app", lambda **kwargs: created.append(kwargs) or app)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)

    assert native_worker.main(("--systemd-transient", "--state-dir", str(tmp_path))) == 0

    assert created[0]["systemd_transient"] is True
    assert created[0]["systemd_exit_type_cgroup"] is True


def test_native_worker_selects_systemd_transient_executor_only_when_requested(tmp_path: Path) -> None:
    local = create_native_worker_app(state_dir=tmp_path / "local", secret="worker-secret")
    production = create_native_worker_app(
        state_dir=tmp_path / "production",
        secret="worker-secret",
        systemd_transient=True,
    )

    assert local.state.executor.__class__.__name__ == "LocalProcessExecutor"
    assert isinstance(production.state.executor, SystemdTransientServiceExecutor)


def test_native_worker_selects_the_typed_oci_executor(monkeypatch, tmp_path: Path) -> None:
    import eyvan.worker.runtime.container as container_runtime

    created: list[dict] = []

    class FakeContainerExecutor:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

    monkeypatch.setattr(container_runtime, "DockerContainerExecutor", FakeContainerExecutor)
    container = NativeContainerConfig(
        image="registry.example.com/easydel:latest",
        pull_policy="never",
    )

    app = create_native_worker_app(
        state_dir=tmp_path / "production",
        secret="worker-secret",
        systemd_transient=True,
        runtime_env={"WANDB_MODE": "offline"},
        runtime_container=container,
        container_artifact_mounts=("/opt/eyvan-runtime/easydel.whl",),
    )

    assert isinstance(app.state.executor, FakeContainerExecutor)
    assert created[0]["container"] == container
    assert created[0]["artifact_mounts"] == ("/opt/eyvan-runtime/easydel.whl",)
    assert created[0]["workload_directory"] == "/var/lib/eyvan/workload"
    assert created[0]["base_environment"]["WANDB_MODE"] == "offline"


def test_native_worker_requires_systemd_for_an_oci_runtime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="container runtime requires systemd_transient"):
        create_native_worker_app(
            state_dir=tmp_path / "worker",
            secret="worker-secret",
            runtime_container=NativeContainerConfig(image="registry.example.com/easydel:latest"),
        )


def test_native_worker_main_loads_the_typed_oci_runtime_file(monkeypatch, tmp_path: Path) -> None:
    app = object()
    created: list[dict] = []
    runtime_path = tmp_path / "container-runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "artifact_mounts": ["/opt/eyvan-runtime/easydel.whl"],
                "container": {
                    "image": "registry.example.com/easydel:latest",
                    "pull_policy": "if-not-present",
                    "privileged": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_SECRET", "environment-secret")
    monkeypatch.setattr(native_worker, "create_native_worker_app", lambda **kwargs: created.append(kwargs) or app)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)

    assert (
        native_worker.main(
            (
                "--systemd-transient",
                "--state-dir",
                str(tmp_path / "state"),
                "--container-runtime-file",
                str(runtime_path),
            )
        )
        == 0
    )

    assert created[0]["runtime_container"].image == "registry.example.com/easydel:latest"
    assert created[0]["runtime_container"].privileged is True
    assert created[0]["container_artifact_mounts"] == ("/opt/eyvan-runtime/easydel.whl",)


def test_native_worker_applies_fleet_runtime_environment_below_job_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app = create_native_worker_app(
        state_dir=tmp_path / "production",
        secret="worker-secret",
        systemd_transient=True,
        runtime_env={
            "HF_HOME": "/var/lib/eyvan/workload/hf",
            "WANDB_MODE": "offline",
        },
    )
    executor = app.state.executor
    assert isinstance(executor, SystemdTransientServiceExecutor)
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> SystemdCommandResult:
        commands.append(command)
        return SystemdCommandResult(returncode=0)

    monkeypatch.setattr(executor, "_command_runner", run)
    executor.submit(
        {
            **spec("python train.py", submission_id="runtime-environment"),
            "environment": {"WANDB_MODE": "online", "WORKLOAD_ONLY": "set-by-job"},
        },
        context(),
    )
    launch = next(command for command in commands if command[0] == "systemd-run")

    # Workload environment values must not be visible in argv.
    assert not any(option.startswith("--setenv=") for option in launch)
    env_file_option = next(option for option in launch if option.startswith("--property=EnvironmentFile="))
    env_path = Path(env_file_option.removeprefix("--property=EnvironmentFile="))
    assert not env_file_option.startswith("--property=EnvironmentFile=-")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    env = dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines() if line)
    assert env["HF_HOME"] == "/var/lib/eyvan/workload/hf"
    assert env["WANDB_MODE"] == "online"
    assert "WANDB_MODE=offline" not in env_path.read_text(encoding="utf-8")
    assert env["WORKLOAD_ONLY"] == "set-by-job"


@pytest.mark.parametrize(
    ("runtime_env", "job_env", "expected"),
    [
        # Default: a directory the workload user owns, never the shared /tmp one.
        ({}, {}, "/var/lib/eyvan/workload/tpu_logs"),
        ({"TPU_LOG_DIR": "/srv/fleet-logs"}, {}, "/srv/fleet-logs"),
        ({"TPU_LOG_DIR": "/srv/fleet-logs"}, {"TPU_LOG_DIR": "/srv/job-logs"}, "/srv/job-logs"),
    ],
)
def test_native_worker_points_libtpu_logs_at_a_directory_the_workload_user_owns(
    monkeypatch, tmp_path: Path, runtime_env: dict[str, str], job_env: dict[str, str], expected: str
) -> None:
    monkeypatch.delenv("TPU_LOG_DIR", raising=False)
    monkeypatch.delenv("EYVAN_NATIVE_WORKLOAD_DIRECTORY", raising=False)
    app = create_native_worker_app(
        state_dir=tmp_path / "production",
        secret="worker-secret",
        systemd_transient=True,
        runtime_env=runtime_env,
    )
    executor = app.state.executor
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> SystemdCommandResult:
        commands.append(command)
        return SystemdCommandResult(returncode=0)

    monkeypatch.setattr(executor, "_command_runner", run)
    executor.submit({**spec("python train.py", submission_id="tpu-logs"), "environment": job_env}, context())
    launch = next(command for command in commands if command[0] == "systemd-run")
    env_file_option = next(option for option in launch if option.startswith("--property=EnvironmentFile="))
    env_path = Path(env_file_option.removeprefix("--property=EnvironmentFile="))
    env = dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines() if line)
    assert env["TPU_LOG_DIR"] == expected


def test_native_worker_rejects_invalid_fleet_runtime_environment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="environment"):
        create_native_worker_app(
            state_dir=tmp_path / "production",
            secret="worker-secret",
            systemd_transient=True,
            runtime_env={"NOT-AN-ENVIRONMENT-NAME": "value"},
        )


def test_native_remote_executor_runs_stops_and_reads_logs(tmp_path: Path) -> None:
    transport, executor = remote_executor(tmp_path)
    with transport:
        completed = executor.submit(spec(command("print('native-ready')"), submission_id="completed"), context())

        assert completed == "completed"
        assert wait_for(executor, completed, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
        assert executor.logs(completed).strip() == "native-ready"

        running_context = context(key="running", generation=2, token="token-2")
        running = executor.submit(
            spec(command("import time; print('started', flush=True); time.sleep(30)"), submission_id="running"),
            running_context,
        )
        assert wait_for(executor, running, {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING

        executor.stop(running, running_context)
        assert wait_for(executor, running, {ExecutionWorkloadStatus.STOPPED}) is ExecutionWorkloadStatus.STOPPED
        executor.retire(running, running_context)


def test_native_worker_fences_logical_workloads_and_retires_a_superseded_attempt(tmp_path: Path) -> None:
    transport, executor = remote_executor(tmp_path)
    with transport:
        first_context = context(key="first", generation=1, token="token-1")
        first = executor.submit(
            spec(
                command("import time; print('first', flush=True); time.sleep(30)"),
                submission_id="logical-a1",
                workload_id="logical-workload",
            ),
            first_context,
        )
        assert wait_for(executor, first, {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING

        second_context = context(key="second", generation=2, token="token-2")
        second = executor.submit(
            spec(
                command("print('second')"),
                submission_id="logical-a2",
                workload_id="logical-workload",
            ),
            second_context,
        )

        assert wait_for(executor, first, {ExecutionWorkloadStatus.STOPPED}) is ExecutionWorkloadStatus.STOPPED
        assert wait_for(executor, second, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
        with pytest.raises(StaleMutationError, match="older than logical workload"):
            executor.submit(
                spec(
                    command("print('stale')"),
                    submission_id="logical-delayed-a1",
                    workload_id="logical-workload",
                ),
                first_context,
            )


def test_native_worker_does_not_leak_its_service_secret_to_workloads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EYVAN_NATIVE_WORKER_SECRET", "service-secret")
    transport, executor = remote_executor(tmp_path)
    with transport:
        remote_id = executor.submit(
            spec(command("import os; print(os.getenv('EYVAN_NATIVE_WORKER_SECRET', 'absent'))"), submission_id="secret"),
            context(),
        )

        assert wait_for(executor, remote_id, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
        assert executor.logs(remote_id).strip() == "absent"


def test_cell_agent_dispatches_to_the_remote_native_worker_not_its_own_host(tmp_path: Path) -> None:
    class Capacity:
        def capabilities(self) -> CellCapabilities:
            return CellCapabilities(
                provider="gcp-compute",
                executor="eyvan-native",
                accelerator_types=("tpu",),
                capacity_classes=("spot",),
                max_gang_replicas=1,
            )

        def observe(self, cell_id: str) -> CellObservation:
            return CellObservation(
                cell_id,
                CapacityStatus.HEALTHY,
                1,
                execution_endpoint="http://native-worker",
            )

    class Resolver:
        def __init__(self, executor: NativeRemoteExecutor):
            self.executor = executor

        def resolve(self, _cell_id: str, _observation: CellObservation) -> NativeRemoteExecutor:
            return self.executor

    worker_dir = tmp_path / "remote-worker"
    app = create_native_worker_app(state_dir=worker_dir, secret="worker-secret")
    with TestClient(app) as transport:
        remote = NativeRemoteExecutor("http://native-worker", secret="worker-secret", client=transport)
        service = CellAgentService(
            capacity=Capacity(),  # type: ignore[arg-type]
            state=FleetStateStore(LocalBackend(tmp_path / "agent-state.json")),
            executors=Resolver(remote),  # type: ignore[arg-type]
        )
        request = WorkloadRequest(
            mutation=MutationModel(idempotency_key="dispatch", generation=1, fencing_token="fence"),
            spec=spec(command("print('remote-cell')"), submission_id="remote-cell"),
        )

        observation = service.submit("cell-a", request)

        assert observation.remote_id == "remote-cell"
        assert wait_for(remote, "remote-cell", {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
        assert service.logs("cell-a", "remote-cell").strip() == "remote-cell"
        assert (worker_dir / "jobs").exists()


def test_native_remote_checkpoint_and_stale_mutation_mapping(tmp_path: Path) -> None:
    transport, executor = remote_executor(tmp_path)
    with transport:
        owner = context(generation=7, token="token-7")
        remote_id = executor.submit(spec(command("print('source')"), submission_id="source"), owner)
        assert wait_for(executor, remote_id, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED

        with pytest.raises(StaleMutationError, match="generation 8 does not own"):
            executor.stop(remote_id, context(key="stale", generation=8, token="token-8"))

        checkpoint_context = context(key="checkpoint-1", generation=7, token="token-7")
        checkpoint_id = executor.checkpoint(
            remote_id,
            hook=command("print('checkpoint')"),
            output_uri="gs://checkpoints/source",
            context=checkpoint_context,
        )

        assert checkpoint_id == "checkpoint-1"
        assert (
            wait_for(executor, checkpoint_id, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
        )
        assert executor.logs(checkpoint_id).strip() == "checkpoint"


def test_native_worker_main_warns_when_the_secret_file_cannot_be_removed(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = object()
    created: list[dict] = []
    secret_path = tmp_path / "worker-secret"
    secret_path.write_text("file-secret", encoding="utf-8")
    monkeypatch.setattr(native_worker, "create_native_worker_app", lambda **kwargs: created.append(kwargs) or app)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)

    real_unlink = Path.unlink

    def guarded_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.resolve() == secret_path.resolve():
            raise OSError("device busy")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)

    with caplog.at_level(logging.WARNING, logger="eyvan.worker.runtime.native_worker"):
        assert native_worker.main(("--secret-file", str(secret_path))) == 0

    assert created[0]["secret"] == "file-secret"
    warning = next(
        record for record in caplog.records if record.levelno == logging.WARNING and "secret file" in record.message
    )
    assert str(secret_path) in warning.message


# --------------------------------------------------------------------- managed staging


def _stage_worker(
    tmp_path: Path,
    *,
    allowed: tuple[str, ...] = (),
    secret: str = "worker-secret",
    client_secret: str | None = None,
) -> tuple[TestClient, NativeRemoteExecutor]:
    """A native worker holding the real staging executor, and the driver that calls it."""

    stager = LocalAssetStageExecutor(root=tmp_path / "worker-disk", allowed_prefixes=allowed) if allowed else None
    app = create_native_worker_app(state_dir=tmp_path / "worker", secret=secret, stager=stager)
    transport = TestClient(app)
    return transport, NativeRemoteExecutor("http://testserver", secret=client_secret or secret, client=transport)


def test_the_native_worker_measures_stages_and_evicts_over_one_route(tmp_path: Path) -> None:
    """The worker half of managed staging, driven the way the agent drives it.

    Every number here is read back off the worker's disk rather than taken from
    the request: the size, the digest, and the path the workload will open. An
    executor that reported what it *expected* would be handing the control plane
    evidence it then charges a budget and steers placement on.
    """

    payload = b"shard-0000" * 96
    source = tmp_path / "assets" / "shard.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    _transport, driver = _stage_worker(tmp_path, allowed=(str(tmp_path / "assets"),))

    measured = driver.measure("cell-a", str(source), max_bytes=1 << 20)
    staged = driver.stage("cell-a", str(source), max_bytes=1 << 20)
    landed = Path(staged.path or "")
    evicted = driver.evict("cell-a", str(source))

    assert measured.verdict is AssetStageVerdict.OK
    assert measured.size_bytes == len(payload)
    assert staged.verdict is AssetStageVerdict.OK
    assert staged.size_bytes == len(payload)
    assert staged.digest == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert landed.is_relative_to(tmp_path / "worker-disk")
    assert evicted.verdict is AssetStageVerdict.OK
    assert not landed.exists()


def test_a_native_worker_without_declared_sources_has_no_staging_executor(tmp_path: Path) -> None:
    """Empty is the off switch, and it is absence rather than refusal.

    A refusal is terminal: it blocks the stage permanently, and nothing unblocks
    it short of replacing the cell. Answering that way would poison the ledger
    for every asset in the fleet over one unset environment variable.
    """

    transport, driver = _stage_worker(tmp_path)

    response = transport.post(
        "/api/v1/cells/cell-a/stage-actions",
        headers={"Authorization": "Bearer worker-secret"},
        json={"action": "stage", "asset": "/data/pile", "max_bytes": 1024},
    )
    with pytest.raises(UnsupportedCapabilityError, match="no staging executor"):
        driver.stage("cell-a", "/data/pile", max_bytes=1024)

    assert response.status_code == 501
    assert "EYVAN_CELL_STAGE_SOURCES" in response.json()["detail"]


def test_a_native_worker_that_predates_the_route_has_no_staging_executor(tmp_path: Path) -> None:
    """Rolling upgrade: workers first, then agents. A 404 is absence, not failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    driver = NativeRemoteExecutor(
        "http://worker",
        secret="worker-secret",
        client=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handler)),
    )

    for call in (
        lambda: driver.measure("cell-a", "/data/pile", max_bytes=1024),
        lambda: driver.stage("cell-a", "/data/pile", max_bytes=1024),
        lambda: driver.evict("cell-a", "/data/pile"),
    ):
        with pytest.raises(UnsupportedCapabilityError, match="no staging executor"):
            call()


def test_a_credential_the_native_worker_rejects_is_unknown_rather_than_refused(tmp_path: Path) -> None:
    """A worker that would not talk has said nothing about the asset.

    The bearer credential is derived per provider generation, so a worker
    rebuilt under a new one answers 401 to a driver still holding the old.
    Recording that as a refusal would block the asset permanently on evidence
    about a credential, and the block would outlive the rebuild that caused it.
    """

    _transport, driver = _stage_worker(
        tmp_path,
        allowed=(str(tmp_path),),
        client_secret="a-previous-generations-secret",
    )

    result = driver.stage("cell-a", str(tmp_path / "assets" / "shard.bin"), max_bytes=1024)

    assert result.verdict is AssetStageVerdict.INCONCLUSIVE
    assert result.size_bytes == 0
    assert result.digest is None
    assert "401" in result.detail


def test_a_request_the_native_worker_rejects_is_refused(tmp_path: Path) -> None:
    """The worker read it and said no, so the same bytes get the same answer tomorrow."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "unprocessable"})

    driver = NativeRemoteExecutor(
        "http://worker",
        secret="worker-secret",
        client=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handler)),
    )

    result = driver.stage("cell-a", "/data/pile", max_bytes=1024)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "422" in result.detail


def test_a_native_worker_failure_or_unreadable_answer_is_inconclusive(tmp_path: Path) -> None:
    """Neither a 5xx nor a body that will not parse is evidence about a disk."""

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    def garbled(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="{not json")

    results = [
        NativeRemoteExecutor(
            "http://worker",
            secret="worker-secret",
            client=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handler)),
        ).measure("cell-a", "/data/pile", max_bytes=1024)
        for handler in (failing, garbled)
    ]

    assert [result.verdict for result in results] == [AssetStageVerdict.INCONCLUSIVE] * 2
    assert all(result.size_bytes == 0 and result.digest is None for result in results)


def test_the_native_worker_builds_its_staging_executor_from_its_own_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The allowlist and the directory belong to the host that opens the file.

    On a Compute fleet those two variables are declared on the worker VM, so
    this is the binding a real fleet actually uses -- the injected executor
    every other test here passes is a convenience, not the shipped path.
    """

    payload = b"declared-on-the-worker"
    source = tmp_path / "assets" / "shard.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    monkeypatch.setenv("EYVAN_CELL_STAGE_SOURCES", str(tmp_path / "assets"))
    monkeypatch.setenv("EYVAN_CELL_STAGE_DIR", str(tmp_path / "declared-disk"))
    app = create_native_worker_app(state_dir=tmp_path / "worker", secret="worker-secret")
    with TestClient(app) as transport:
        driver = NativeRemoteExecutor("http://testserver", secret="worker-secret", client=transport)
        staged = driver.stage("cell-a", str(source), max_bytes=1 << 20)

    assert staged.verdict is AssetStageVerdict.OK
    assert Path(staged.path or "").read_bytes() == payload
    assert Path(staged.path or "").is_relative_to((tmp_path / "declared-disk").resolve())


def test_a_launch_the_worker_rejects_is_a_value_error_not_a_server_fault(tmp_path) -> None:
    """Return invalid launches as typed configuration errors so the plane does not retry them as server faults."""

    from fastapi.testclient import TestClient

    from eyvan.worker.runtime.native_worker import create_native_worker_app

    class RejectingExecutor:
        """Rejects every launch the way the systemd executor rejects a newline."""

        def submit(self, spec_document, context):
            del spec_document, context
            raise ValueError("native-systemd environment variable 'EXAMPLE' must not contain a newline")

    app = create_native_worker_app(RejectingExecutor(), secret="worker-test-secret")  # type: ignore[arg-type]
    invalid = spec(command("print(1)"), submission_id="refused")
    with TestClient(app) as worker_http:
        response = worker_http.post(
            "/api/v1/workloads",
            headers={"Authorization": "Bearer worker-test-secret"},
            json={
                "mutation": {"idempotency_key": "refused", "generation": 1, "fencing_token": "fence"},
                "spec": invalid,
            },
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_configuration"

        native = NativeRemoteExecutor("http://testserver", secret="worker-test-secret", client=worker_http)
        with pytest.raises(ValueError, match="must not contain a newline"):
            native.submit(invalid, MutationContext(idempotency_key="refused-2", generation=1, fencing_token="fence"))
