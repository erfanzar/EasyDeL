# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for Wave 2B live TPU runtime, controller, and service fixes."""

from __future__ import annotations

import typing as tp
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eyvan.artifacts.managed import (
    ArtifactTokenSigner,
    LocalManagedArtifactStore,
)
from eyvan.artifacts.transfers import ManagedArtifactAccess, progress_report_uri
from eyvan.capacity.drivers.base import (
    CapacityProviderFailureCode,
    CapacityStatus,
    CellCapabilities,
    MutationContext,
)
from eyvan.config import EyvanSettings
from eyvan.records import (
    CallableExecution,
    CapacityBeliefState,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    NativeContainerConfig,
    OutboxRecord,
    OutboxState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.api.app import create_app
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.provider_diagnostics import safe_provider_diagnostic
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.models import AgentObservation, WorkloadMutationRequest
from eyvan.worker.agent.service import CellAgentService, UnsupportedCapabilityError
from eyvan.worker.runtime.container import DockerContainerExecutor
from eyvan.worker.runtime.systemd_transient import SystemdCommandResult, SystemdTransientServiceExecutor


def _simulate_systemd_replace_env_arg(arg: str, env: dict[str, str] | None = None) -> str:
    """Faithful subset of systemd's ``replace_env_arg()`` for verifying ``ExecStart`` escaping."""

    values = env or {}
    out: list[str] = []
    index = 0
    length = len(arg)
    while index < length:
        char = arg[index]
        if char != "$":
            out.append(char)
            index += 1
            continue
        if index + 1 < length and arg[index + 1] == "$":
            out.append("$")
            index += 2
            continue
        if index + 1 < length and arg[index + 1] == "{":
            close = arg.find("}", index + 2)
            if close != -1:
                out.append(values.get(arg[index + 2 : close], ""))
                index = close + 1
                continue
        end = index + 1
        while end < length and (arg[end].isalnum() or arg[end] == "_"):
            end += 1
        if end > index + 1:
            out.append(values.get(arg[index + 1 : end], ""))
            index = end
        else:
            out.append("$")
            index += 1
    return "".join(out)


def _mutation(key: str = "mut-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=1, fencing_token="fence-1")


# ---------------------------------------------------------------------------
# 1. P0: systemd-run ExecStart $VAR / ${...} escaping
# ---------------------------------------------------------------------------


def test_systemd_transient_escapes_dollar_signs_for_launch_and_exec(tmp_path: Path) -> None:
    """systemd-run ExecStart argv must escape ``$`` -> ``$$`` so bash syntax survives PID 1."""

    seen: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> SystemdCommandResult:
        seen.append(command)
        if command[0] == "systemd-run":
            return SystemdCommandResult(returncode=0, stdout="ok\n")
        if command[:2] == ("systemctl", "show"):
            return SystemdCommandResult(
                returncode=0,
                stdout="LoadState=loaded\nActiveState=active\nSubState=running\nControlGroup=/system.slice/x.service\n",
            )
        return SystemdCommandResult(returncode=0)

    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=runner)
    original_entrypoint = 'UV=(uv); echo "${#UV[@]}" "${UV[@]}" "$FOO" "$$" "%h"'
    remote_id = executor.submit(
        {
            "entrypoint": original_entrypoint,
            "submission_id": "setup-dollar-job",
            "environment": {},
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
        },
        _mutation("launch-dollar"),
    )

    launch = next(cmd for cmd in seen if cmd[0] == "systemd-run")
    assert launch[-3:-1] == ("/bin/sh", "-c")
    assert launch[-1] == original_entrypoint.replace("$", "$$")
    assert "%h" in launch[-1]
    assert "%%h" not in launch[-1]
    assert _simulate_systemd_replace_env_arg(launch[-1], {"FOO": "WRONG"}) == original_entrypoint

    exec_cmd = ["sh", "-c", 'echo "${#UV[@]}" "$FOO" "$$" "%h"']
    doc = executor.exec_in_workload(remote_id, exec_cmd, timeout_s=15.0)
    exec_run = [cmd for cmd in seen if cmd[0] == "systemd-run"][-1]
    sep = exec_run.index("--")
    escaped_argv = list(exec_run[sep + 1 :])
    assert escaped_argv == [part.replace("$", "$$") for part in exec_cmd]
    assert [_simulate_systemd_replace_env_arg(part, {"FOO": "WRONG"}) for part in escaped_argv] == exec_cmd
    # The API response preserves the user's original unescaped command.
    assert doc["command"] == exec_cmd


def test_docker_container_executor_escapes_dollar_signs_for_systemd_run(tmp_path: Path) -> None:
    """DockerContainerExecutor launches via systemd-run too and must escape ``$`` in entrypoint."""

    from test_container_executor import FakeDockerSystemd

    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = DockerContainerExecutor(
        tmp_path,
        container=NativeContainerConfig(image="registry.example/easydel:latest"),
        command_runner=daemon,
    )
    original_entrypoint = 'set -euo pipefail; UV=(uv); echo "${#UV[@]}" "$FOO" "$$" "%h"'
    executor.submit(
        {
            "entrypoint": original_entrypoint,
            "submission_id": "docker-dollar-job",
            "environment": {},
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
        },
        _mutation("docker-dollar"),
    )

    launch = next(cmd for cmd in daemon.commands if cmd[0] == "systemd-run")
    assert launch[-1] == original_entrypoint.replace("$", "$$")
    assert _simulate_systemd_replace_env_arg(launch[-1], {"FOO": "WRONG"}) == original_entrypoint


# ---------------------------------------------------------------------------
# 2. P1: EYVAN_PROGRESS_URL on GCS-backed (gs://) fleets + reachability gate
# ---------------------------------------------------------------------------


def _callable_workload(*, result_uri: str, generation: int = 1, workload_id: str = "wl-gcs-1") -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        created_by="alice",
        attempt=1,
        generation=generation,
        state=WorkloadState.RUNNING,
        spec=WorkloadDefinition(
            name="train",
            execution=CallableExecution(
                artifact_uri="gs://bucket/eyvan/callables/abc.pkl",
                sha256="a" * 64,
                python="3.11",
                format_version=1,
                result_uri=result_uri,
            ),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        ),
    )


def _controller_for_progress(
    tmp_path: Path,
    *,
    public_url: str,
    dev_auth_enabled: bool = False,
    allow_insecure_artifact_http: bool = False,
) -> EyvanController:
    repository = InMemoryRepository()
    settings = EyvanSettings(
        public_url=public_url,
        dev_auth_enabled=dev_auth_enabled,
        allow_insecure_artifact_http=allow_insecure_artifact_http,
        artifact_store_uri=str(tmp_path / "artifacts"),
        controller_enabled=False,
    )
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path / "artifacts"),
        signer=ArtifactTokenSigner(b"test-progress-signing-key-32bytes!"),
        public_url=public_url,
        token_ttl_seconds=3600,
    )
    return EyvanController(
        repository=repository,
        service=ControlPlaneService(repository),
        settings=settings,
        artifact_access=access,
    )


def test_gcs_callable_progress_callback_respects_reachability_and_transport_policy(tmp_path: Path) -> None:
    """gs:// callables get EYVAN_PROGRESS_URL iff the plane URL is reachable and permitted by transport policy."""

    workload = _callable_workload(result_uri="gs://bucket/eyvan/results/wl-gcs-1.pkl")

    # 1. Reachable HTTPS plane + remote GCP cell -> injects URL and token without insecure-HTTP flag.
    https_ctrl = _controller_for_progress(tmp_path / "https", public_url="https://plane.example.com")
    _entry, env_https = https_ctrl._render_execution(workload, runtime_environment={}, cell_provider="gcp")
    assert env_https["EYVAN_PROGRESS_URL"] == (
        "https://plane.example.com/api/v1/workloads/wl-gcs-1/progress-reports?org_id=acme&project_id=research"
    )
    assert env_https["EYVAN_MANAGED_RESULT_TOKEN"]
    assert "EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP" not in env_https

    # 2. Loopback plane + remote GCP cell -> omitted so remote TPU does not stall 10s per report_progress().
    loopback_ctrl = _controller_for_progress(
        tmp_path / "loopback",
        public_url="http://127.0.0.1:8080",
        dev_auth_enabled=True,
    )
    _entry, env_remote_loopback = loopback_ctrl._render_execution(workload, runtime_environment={}, cell_provider="gcp")
    assert "EYVAN_PROGRESS_URL" not in env_remote_loopback
    assert "EYVAN_MANAGED_RESULT_TOKEN" not in env_remote_loopback

    # 3. Loopback plane + local cell -> permitted when dev_auth_enabled.
    _entry, env_local_loopback = loopback_ctrl._render_execution(
        workload,
        runtime_environment={},
        cell_provider="local",
    )
    assert env_local_loopback["EYVAN_PROGRESS_URL"].startswith("http://127.0.0.1:8080/api/v1/workloads/wl-gcs-1/")
    assert env_local_loopback["EYVAN_MANAGED_RESULT_TOKEN"]
    assert env_local_loopback["EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP"] == "1"

    # 4. Plain HTTP non-loopback without opt-in -> omitted (never fails gs:// dispatch or leaks token in clear text).
    plain_denied = _controller_for_progress(
        tmp_path / "http-denied",
        public_url="http://10.0.0.5:8080",
        dev_auth_enabled=False,
        allow_insecure_artifact_http=False,
    )
    _entry, env_http_denied = plain_denied._render_execution(workload, runtime_environment={}, cell_provider="gcp")
    assert "EYVAN_PROGRESS_URL" not in env_http_denied
    assert "EYVAN_MANAGED_RESULT_TOKEN" not in env_http_denied

    # 5. Plain HTTP non-loopback WITH allow_insecure_artifact_http=True -> permitted + flag propagated.
    plain_allowed = _controller_for_progress(
        tmp_path / "http-allowed",
        public_url="http://10.0.0.5:8080",
        dev_auth_enabled=False,
        allow_insecure_artifact_http=True,
    )
    _entry, env_http_allowed = plain_allowed._render_execution(workload, runtime_environment={}, cell_provider="gcp")
    assert env_http_allowed["EYVAN_PROGRESS_URL"].startswith("http://10.0.0.5:8080/api/v1/workloads/wl-gcs-1/")
    assert env_http_allowed["EYVAN_MANAGED_RESULT_TOKEN"]
    assert env_http_allowed["EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP"] == "1"


@pytest.mark.asyncio
async def test_gcs_callable_progress_token_authenticates_revokes_on_retry_and_cannot_write_managed_store(
    tmp_path: Path,
) -> None:
    """End-to-end API route verification for progress-report tokens issued on gs:// result URIs."""

    repository = InMemoryRepository()
    await repository.open()
    workload = _callable_workload(result_uri="gs://bucket/eyvan/results/wl-gcs-e2e.pkl", workload_id="wl-gcs-e2e")
    async with repository.transaction() as tx:
        await tx.put_workload(workload)

    settings = EyvanSettings(
        public_url="https://plane.example.com",
        dev_auth_enabled=True,
        artifact_store_uri=str(tmp_path / "artifacts"),
        artifact_token_key="test-gcs-progress-route-key-32b!!",
        controller_enabled=False,
    )
    app = create_app(settings=settings, repository=repository)
    transfer = app.state.artifact_access.progress_report_transfer(
        org_id=workload.org_id,
        project_id=workload.project_id,
        workload_id=workload.id,
        generation=workload.generation,
        result_uri=workload.spec.execution.result_uri,
    )

    with TestClient(app) as client:
        # 1. Valid progress report is accepted and stored.
        posted = client.post(
            "/api/v1/workloads/wl-gcs-e2e/progress-reports?org_id=acme&project_id=research",
            headers={"X-Eyvan-Artifact-Token": transfer.token},
            json={"rank": 0, "attempt": 1, "step": 25, "total": 100, "message": "epoch 1", "metrics": {"loss": 0.4}},
        )
        assert posted.status_code == 200, posted.text
        assert posted.json()["step"] == 25

        # 2. The progress token CANNOT be abused to upload into the managed artifact store on a gs:// workload.
        scoped_id = progress_report_uri(workload.id, workload.generation).rsplit("/", 1)[1]
        abused = client.put(
            f"/api/v1/artifacts/results/{scoped_id}/content?org_id=acme&project_id=research&workload_id=wl-gcs-e2e",
            headers={"X-Eyvan-Artifact-Token": transfer.token},
            content=b"forged-result",
        )
        assert abused.status_code == 403, abused.text

        # 3. Bumping generation revokes the old progress token immediately.
        async with repository.transaction() as tx:
            current = await tx.get_workload(workload.id)
            assert current is not None
            await tx.put_workload(current.model_copy(update={"generation": 2, "attempt": 2}))

        stale = client.post(
            "/api/v1/workloads/wl-gcs-e2e/progress-reports?org_id=acme&project_id=research",
            headers={"X-Eyvan-Artifact-Token": transfer.token},
            json={"rank": 0, "attempt": 2, "step": 30, "total": 100},
        )
        assert stale.status_code == 403, stale.text


# ---------------------------------------------------------------------------
# 3. P1: CONFIGURATION non-retryable, IAM diagnostic preserved, explain_workload filters DELETED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outbox_discards_configuration_failures_instead_of_retrying_infinitely(tmp_path: Path) -> None:
    """Permanent CapacityProviderFailureCode.CONFIGURATION must settle to DISCARDED on first failure."""

    repository = InMemoryRepository()
    await repository.open()
    now = datetime.now(UTC)
    command = OutboxRecord(
        id="outbox-iam-1",
        topic="cell.apply",
        aggregate_type="cell",
        aggregate_id="cell-1",
        idempotency_key="apply:cell-1:g1",
        generation=1,
        state=OutboxState.PROCESSING,
        attempts=1,
        claimed_by="ctrl-1",
        claimed_until=now + timedelta(seconds=60),
    )
    async with repository.transaction() as tx:
        await tx.put_outbox(command)

    controller = EyvanController(
        repository=repository,
        service=ControlPlaneService(repository),
        settings=EyvanSettings(artifact_store_uri=str(tmp_path / "artifacts"), controller_enabled=False),
    )
    controller.worker_id = "ctrl-1"
    completed = await controller._finish_outbox(
        command,
        error=(
            "the active gcloud account cannot bind native-worker secrets (missing: secretmanager.secrets.setIamPolicy)"
        ),
        failure_code=CapacityProviderFailureCode.CONFIGURATION,
    )
    assert completed is True

    async with repository.transaction() as tx:
        stored = await tx.get_outbox(command.id)
    assert stored is not None
    assert stored.state is OutboxState.DISCARDED
    assert stored.failure_code is CapacityProviderFailureCode.CONFIGURATION


def test_safe_provider_diagnostic_preserves_secret_manager_iam_preflight_refusals() -> None:
    """Actionable IAM remediation must not be collapsed into the generic controller-logs fallback."""

    denial = (
        "the active gcloud account cannot bind native-worker secrets in project 'example-project-a' "
        "(missing: secretmanager.secrets.setIamPolicy). Eyvan-native bootstrap needs a one-time grant; "
        "ask a project owner to run: gcloud projects add-iam-policy-binding example-project-a "
        "--member=user:<your-account> --role=roles/secretmanager.admin"
    )
    rendered = safe_provider_diagnostic(denial)
    assert rendered is not None
    assert "secretmanager.secrets.setIamPolicy" in rendered
    assert "roles/secretmanager.admin" in rendered

    grant_line = "testIamPermissions failed (missing: secretmanager.secrets.create); grant roles/secretmanager.admin"
    assert safe_provider_diagnostic(grant_line) == grant_line


@pytest.mark.asyncio
async def test_explain_workload_excludes_deleted_cells_from_considered_set() -> None:
    """Historical DELETED cells must not inflate considered_cell_count or hide active cells."""

    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    async with repository.transaction() as tx:
        for index in range(50):
            await tx.put_cell(
                CellRecord(
                    id=f"cell-deleted-{index:02d}",
                    provider="gcp",
                    provider_project="example-project-a",
                    region="us-central2",
                    zone="us-central2-b",
                    state=CellState.DELETED,
                    capacity=ResourceVector(cpu=4, memory_gib=8),
                    allocatable=ResourceVector(cpu=4, memory_gib=8),
                )
            )
        await tx.put_cell(
            CellRecord(
                id="cell-active-small",
                provider="gcp",
                provider_project="example-project-a",
                region="us-central2",
                zone="us-central2-b",
                state=CellState.READY,
                capacity=ResourceVector(cpu=2, memory_gib=4),
                allocatable=ResourceVector(cpu=2, memory_gib=4),
            )
        )

    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    workload, _ = await service.submit_workload(
        principal,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="big-job",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=64, memory_gib=64)),
        ),
        idempotency_key="explain-deleted-filter",
    )

    explanation = await service.explain_workload(principal, workload.id)
    assert explanation.placeable is False
    assert explanation.considered_cell_count == 1
    assert explanation.truncated_cell_count == 0
    assert [cell.cell_id for cell in explanation.cells] == ["cell-active-small"]


# ---------------------------------------------------------------------------
# 4. P1: First-READY launch latency only + true even-length median
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_latency_is_recorded_only_on_first_ready_and_uses_true_even_median(tmp_path: Path) -> None:
    """Later READY flaps must not append bogus lifetime latencies, and even-length median averages middle two."""

    created_at = datetime.now(UTC) - timedelta(seconds=600)
    repository = InMemoryRepository()
    await repository.open()
    pool = CapacityPoolRecord(
        id="research/v6e-4",
        fleet_name="research",
        provider="gcp",
        provider_project="example-project-a",
        region="us-central2",
        zone="us-central2-b",
        node_type="v6e-4",
        accelerator_type="v6e-4",
        agent_endpoint="http://127.0.0.1:19091",
        capacity=ResourceVector(cpu=4, memory_gib=16, tpu=4),
        allocatable=ResourceVector(cpu=4, memory_gib=16, tpu=4),
    )
    cell = CellRecord(
        id="cell-flap",
        pool_id=pool.id,
        provider="gcp",
        provider_project="example-project-a",
        region="us-central2",
        zone="us-central2-b",
        endpoint="http://127.0.0.1:19091",
        state=CellState.REGISTERING,
        provider_phase=CapacityStatus.PROVISIONING,
        created_at=created_at,
        updated_at=created_at,
        capacity=ResourceVector(cpu=4, memory_gib=16, tpu=4),
        allocatable=ResourceVector(cpu=4, memory_gib=16, tpu=4),
    )
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(pool)
        await tx.put_cell(cell)

    observations = [
        AgentObservation(cell_id=cell.id, state=CapacityStatus.HEALTHY, generation=1, detail="ready-1"),
        AgentObservation(cell_id=cell.id, state=CapacityStatus.UNAVAILABLE, generation=1, detail="transient-flap"),
        AgentObservation(cell_id=cell.id, state=CapacityStatus.HEALTHY, generation=1, detail="ready-2"),
    ]

    class _StubAgentClient:
        def reconcile(self, _cell_id: str, _request: WorkloadMutationRequest) -> AgentObservation:
            return observations.pop(0)

    service = ControlPlaneService(repository)
    controller = EyvanController(
        repository=repository,
        service=service,
        settings=EyvanSettings(artifact_store_uri=str(tmp_path / "artifacts"), controller_enabled=False),
        agent_factory=lambda _cell: _StubAgentClient(),  # type: ignore[arg-type,return-value]
    )

    # First transition REGISTERING -> READY at +2s records the true launch latency once.
    await controller._observe_cell_once(cell, created_at + timedelta(seconds=2))
    async with repository.transaction() as tx:
        after_first_ready = await tx.get_cell(cell.id)
    assert after_first_ready is not None
    assert after_first_ready.state is CellState.READY
    assert after_first_ready.billing_started_at == created_at + timedelta(seconds=2)
    assert after_first_ready.availability.provision_latencies_s == (2.0,)

    # Flap to UNAVAILABLE at +200s, then recover back to READY after cooldown -> must NOT append a second latency!
    await controller._observe_cell_once(after_first_ready, created_at + timedelta(seconds=200))
    async with repository.transaction() as tx:
        while_unavailable = await tx.get_cell(cell.id)
    assert while_unavailable is not None
    assert while_unavailable.state is CellState.UNAVAILABLE

    recovery_at = max(
        created_at + timedelta(seconds=300),
        (while_unavailable.availability.cooldown_until or created_at) + timedelta(seconds=10),
    )
    await controller._observe_cell_once(while_unavailable, recovery_at)
    async with repository.transaction() as tx:
        after_recovery = await tx.get_cell(cell.id)
    assert after_recovery is not None
    assert after_recovery.state is CellState.READY
    assert after_recovery.availability.provision_latencies_s == (2.0,)
    assert after_recovery.availability.consecutive_failures == 0
    assert after_recovery.availability.cooldown_until is None

    # Add a second cell with latency 229.34s so the pool has an even-length sample list [2.0, 229.34].
    async with repository.transaction() as tx:
        await tx.put_cell(
            cell.model_copy(
                update={
                    "id": "cell-second",
                    "state": CellState.READY,
                    "availability": CapacityBeliefState(provision_latencies_s=(229.34,)),
                }
            )
        )
    admin = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    report = await service.launch_latency_report(admin)
    assert len(report.pools) == 1
    assert report.pools[0].samples == 2
    assert report.pools[0].median_seconds == pytest.approx((2.0 + 229.34) / 2.0)


# ---------------------------------------------------------------------------
# 5. P2: Exec refusal message names EYVAN_CAPACITY_SUPPORTS_EXEC=1
# ---------------------------------------------------------------------------


def test_exec_refusal_message_names_agent_env_var() -> None:
    """When supports_exec is False, the error tells the operator the exact agent knob."""

    class _CapacityWithoutExec:
        def capabilities(self) -> CellCapabilities:
            return CellCapabilities(
                provider="gcp",
                executor="eyvan-native",
                accelerator_types=("v6e-4",),
                capacity_classes=(),
                supports_exec=False,
            )

    agent = CellAgentService.__new__(CellAgentService)
    agent.settings = EyvanSettings()
    agent.capacity = _CapacityWithoutExec()  # type: ignore[assignment]
    agent.execution_backend = "eyvan-native"
    agent.executors = None  # type: ignore[assignment]
    with pytest.raises(UnsupportedCapabilityError, match="EYVAN_CAPACITY_SUPPORTS_EXEC=1"):
        agent.exec_in_workload("cell-1", "job-1", ["hostname"])


# ---------------------------------------------------------------------------
# 6. P0 Wave 4: Stage large / %-bearing --setup JSON + unloadable active/exited
# ---------------------------------------------------------------------------


def test_systemd_transient_stages_easydel_setup_json_and_sets_safe_description(tmp_path: Path) -> None:
    """Large or ``%``-bearing ``--setup`` JSON is staged to ``--setup-file`` and cleaned on terminal state."""

    import json
    import shlex
    import stat

    from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
    from eyvan.records import WorkloadSetup
    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands
    from eyvan.worker.runtime import dependency_entry

    easydel_script = (Path(__file__).resolve().parents[1] / "eyvan" / "sdk" / "client" / "easydel_setup.sh").read_text(
        encoding="utf-8"
    )
    assert "%" in easydel_script and len(easydel_script) > 10_000
    setup = WorkloadSetup(script=easydel_script, args=("--repo", "https://example.com/easydel.git", "--ref", "a" * 40))
    entrypoint = _ControllerWorkloadCommands._dependency_entrypoint(
        (),
        "python -c 'print(\"$FOO\")'",
        setup,
    )

    state_dir = tmp_path / "state"
    workload_dir = tmp_path / "workload"
    workload_dir.mkdir(mode=0o750)
    seen: list[tuple[str, ...]] = []
    unit_state = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "Result": "success",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "ControlGroup": "/system.slice/x.service",
    }

    def runner(command: tuple[str, ...]) -> SystemdCommandResult:
        seen.append(command)
        if command[:2] == ("systemctl", "show"):
            return SystemdCommandResult(
                returncode=0,
                stdout="".join(f"{k}={v}\n" for k, v in unit_state.items()),
            )
        return SystemdCommandResult(returncode=0)

    executor = SystemdTransientServiceExecutor(
        state_dir,
        command_runner=runner,
        boot_id_reader=lambda: "boot-a",
        workload_directory=workload_dir,
        use_exit_type_cgroup=False,
    )
    remote_id = executor.submit(
        {
            "entrypoint": entrypoint,
            "submission_id": "wl-easydel-setup",
            "environment": {},
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
        },
        _mutation("launch-easydel-setup"),
    )

    launch = next(cmd for cmd in seen if cmd[0] == "systemd-run")
    unit_name = next(part.removeprefix("--unit=") for part in launch if part.startswith("--unit="))
    assert f"--description=Eyvan workload {unit_name}" in launch
    assert launch[-3:-1] == ("/bin/sh", "-c")
    exec_arg = launch[-1]
    assert "%" not in exec_arg
    assert "--setup " not in exec_arg
    assert len(exec_arg) < 512

    unescaped = _simulate_systemd_replace_env_arg(exec_arg, {"FOO": "WRONG"})
    parts = shlex.split(unescaped)
    assert "--setup-file" in parts
    setup_file = Path(parts[parts.index("--setup-file") + 1])
    assert setup_file.parent == workload_dir / ".eyvan-setup"
    assert setup_file.is_file()
    assert stat.S_IMODE(setup_file.stat().st_mode) == 0o644
    assert json.loads(setup_file.read_text(encoding="utf-8")) == setup.model_dump(mode="json")
    # Dollar escaping for the remaining command still survives PID 1 env expansion.
    assert parts[-3:] == ["python", "-c", 'print("$FOO")']

    # While RUNNING, the staged setup file stays on disk so dependency_entry can still read it.
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.RUNNING
    assert setup_file.is_file()

    # dependency_entry consumes --setup-file end-to-end and resolves the exact script and args.
    calls: list[tuple[list[str], object]] = []

    class _Executed(BaseException):
        pass

    def _fake_prepare(requirements: tp.Sequence[str], **options: tp.Any) -> Path:
        calls.append((list(requirements), options["setup"]))
        return tmp_path

    def _fake_execvp(_prog: str, argv: tp.Sequence[str]) -> None:
        raise _Executed(list(argv))

    orig_prepare = dependency_entry.prepare
    orig_execvp = dependency_entry.os.execvp
    try:
        dependency_entry.prepare = _fake_prepare  # type: ignore[assignment]
        dependency_entry.os.execvp = _fake_execvp  # type: ignore[assignment]
        with pytest.raises(_Executed) as executed:
            dependency_entry.main(["--setup-file", str(setup_file), "--", "python", "train.py"])
    finally:
        dependency_entry.prepare = orig_prepare  # type: ignore[assignment]
        dependency_entry.os.execvp = orig_execvp  # type: ignore[assignment]
    assert executed.value.args[0] == ["python", "train.py"]
    assert calls == [([], dependency_entry.SetupStep(script=easydel_script, args=setup.args))]

    # Once terminal, the staged setup file is cleaned up.
    unit_state.update({"SubState": "exited", "ExecMainCode": "1", "ExecMainStatus": "0", "ControlGroup": ""})
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED
    assert not setup_file.exists()


def test_dependency_entry_setup_file_rejects_conflicts_and_invalid_files(tmp_path: Path, capsys) -> None:
    """``dependency_entry --setup-file`` validates exclusivity, readability, and JSON object shape."""

    from eyvan.worker.runtime import dependency_entry

    with pytest.raises(SystemExit) as conflict:
        dependency_entry.main(["--setup", '{"script":"true"}', "--setup-file", str(tmp_path / "s.json"), "--", "true"])
    assert conflict.value.code == 2

    assert dependency_entry.main(["--setup-file", str(tmp_path / "missing.json"), "--", "true"]) == 1
    assert "could not read --setup-file" in capsys.readouterr().err

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    assert dependency_entry.main(["--setup-file", str(bad_json), "--", "true"]) == 1
    assert "--setup-file is not valid JSON" in capsys.readouterr().err

    non_obj = tmp_path / "list.json"
    non_obj.write_text("[]", encoding="utf-8")
    assert dependency_entry.main(["--setup-file", str(non_obj), "--", "true"]) == 1
    assert "--setup-file must be a JSON object" in capsys.readouterr().err


def test_docker_container_executor_stages_large_or_percent_setup_into_oci_bridge(tmp_path: Path) -> None:
    """DockerContainerExecutor stages ``%``-bearing setup JSON into the read-only OCI mount."""

    import json
    import shlex

    from test_container_executor import FakeDockerSystemd

    from eyvan.records import WorkloadSetup
    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands
    from eyvan.worker.runtime.oci_callable_stager import OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY

    setup = WorkloadSetup(script='local_dir="${line%%;*}"\nprintf "%s" "$local_dir"\n', args=("--ref", "abc"))
    entrypoint = _ControllerWorkloadCommands._dependency_entrypoint((), "python train.py", setup)

    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = DockerContainerExecutor(
        tmp_path,
        container=NativeContainerConfig(image="registry.example/easydel:latest"),
        command_runner=daemon,
    )
    executor.submit(
        {
            "entrypoint": entrypoint,
            "submission_id": "docker-percent-setup",
            "environment": {},
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
        },
        _mutation("docker-percent-setup"),
    )

    launch = next(cmd for cmd in daemon.commands if cmd[0] == "systemd-run")
    unit_name = next(part.removeprefix("--unit=") for part in launch if part.startswith("--unit="))
    assert f"--description=Eyvan workload {unit_name}" in launch
    assert "%" not in launch[-1]
    parts = shlex.split(launch[-1])
    assert parts[:3] == ["python", "-I", f"{OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY}/dependency_entry.py"]
    container_setup_path = parts[parts.index("--setup-file") + 1]
    assert container_setup_path.startswith(f"{OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY}/setup/")
    host_setup_path = tmp_path / "oci-callable-runner" / "setup" / Path(container_setup_path).name
    assert host_setup_path.is_file()
    assert json.loads(host_setup_path.read_text(encoding="utf-8")) == setup.model_dump(mode="json")


@pytest.mark.parametrize("load_state", ["bad-setting", "error", "masked"])
@pytest.mark.parametrize("exec_main_code", ["1", "exited"])
def test_unloadable_cleanly_exited_unit_reports_succeeded_and_honors_cgroup_check(
    tmp_path: Path,
    load_state: str,
    exec_main_code: str,
) -> None:
    """``LoadState=bad-setting`` + ``ActiveState=active/SubState=exited`` must never hang in RUNNING."""

    from eyvan.capacity.drivers.base import ExecutionWorkloadStatus

    props = {
        "LoadState": load_state,
        "ActiveState": "active",
        "SubState": "running",
        "Result": "success",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "ControlGroup": "/system.slice/wl.service",
    }
    cgroup_alive = {"value": True}

    def runner(command: tuple[str, ...]) -> SystemdCommandResult:
        if command[:2] == ("systemctl", "show"):
            return SystemdCommandResult(returncode=0, stdout="".join(f"{k}={v}\n" for k, v in props.items()))
        return SystemdCommandResult(returncode=0)

    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=runner,
        boot_id_reader=lambda: "boot-a",
        cgroup_processes_reader=lambda _cg: cgroup_alive["value"],
        use_exit_type_cgroup=False,
    )
    remote_id = executor.submit(
        {
            "entrypoint": "python train.py",
            "submission_id": f"wl-{load_state}-{exec_main_code}",
            "environment": {},
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
        },
        _mutation(f"mut-{load_state}-{exec_main_code}"),
    )

    # While still SubState=running under bad-setting, stay INDETERMINATE (recoverable, not latched).
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE

    # Once main process exits cleanly but systemd 249 cgroup still holds children, report RUNNING.
    props.update({"SubState": "exited", "ExecMainCode": exec_main_code, "ExecMainStatus": "0"})
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.RUNNING

    # Once cgroup empties (or ControlGroup is cleared), transition cleanly to SUCCEEDED.
    cgroup_alive["value"] = False
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED


@pytest.mark.parametrize(
    ("result", "exec_main_code", "exec_main_status"),
    [
        ("exit-code", "1", "1"),
        ("success", "1", "1"),
        ("success", "2", "0"),
        ("signal", "killed", "9"),
    ],
)
def test_unloadable_exited_unit_with_non_clean_exit_remains_indeterminate(
    tmp_path: Path,
    result: str,
    exec_main_code: str,
    exec_main_status: str,
) -> None:
    """An unloadable ``active/exited`` unit without clean exit evidence must never report SUCCEEDED."""

    from eyvan.capacity.drivers.base import ExecutionWorkloadStatus

    props = {
        "LoadState": "bad-setting",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": result,
        "ExecMainCode": exec_main_code,
        "ExecMainStatus": exec_main_status,
        "ControlGroup": "",
    }

    def runner(command: tuple[str, ...]) -> SystemdCommandResult:
        if command[:2] == ("systemctl", "show"):
            return SystemdCommandResult(returncode=0, stdout="".join(f"{k}={v}\n" for k, v in props.items()))
        return SystemdCommandResult(returncode=0)

    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=runner,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(
        {
            "entrypoint": "python train.py",
            "submission_id": "wl-bad-exited",
            "environment": {},
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
        },
        _mutation("mut-bad-exited"),
    )

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE
