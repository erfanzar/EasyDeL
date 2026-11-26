# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from eyvan.artifacts import LocalArtifactStore
from eyvan.capacity.autoscale import AutoscaleConfigCache
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.records import (
    AutoscaleBlockReason,
    CallableExecution,
    CapacityClass,
    CapacityRetryProgress,
    GcpQueueAdmissionDenialTarget,
    OutboxState,
    ResourceGang,
    ResourceVector,
    WorkloadCapacityRequest,
    WorkloadCapacitySummary,
    WorkloadDefinition,
    WorkloadPoolProgress,
    WorkloadProgress,
    WorkloadProgressPhase,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk.client.callables import EyvanCallableSubmitter
from eyvan.sdk.client.jobs import CallableSubmission, EyvanJobError, TpuConfig
from eyvan.sdk.client.sync import EyvanClient
from eyvan.sdk.client.transport import EyvanHTTPError
from eyvan.sdk.job_receipt import JobTerminalOutput
from eyvan.worker.runtime.callables import dump_callable


def _plain_output() -> JobTerminalOutput:
    """Select the redirected public rendering contract independently of process color variables."""

    return JobTerminalOutput(force_terminal=False, environment={})


def _workload(
    execution: CallableExecution,
    state: WorkloadState,
    failure_reason: str | None = None,
    autoscale_block_reason: AutoscaleBlockReason | None = None,
    queue_admission_denial: GcpQueueAdmissionDenialTarget | None = None,
) -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-v6e-16",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="tests.train",
            execution=execution,
            gang=ResourceGang(per_replica=ResourceVector(tpu=32)),
            provider="gcp",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
            required_executor="eyvan-native",
            worker_count=2,
        ),
        state=state,
        failure_reason=failure_reason,
        autoscale_block_reason=autoscale_block_reason,
        queue_admission_denial=queue_admission_denial,
        created_by="alice",
    )


def _progress(
    *,
    phase: WorkloadProgressPhase,
    workload_state: WorkloadState = WorkloadState.PENDING,
    lost_workers: int = 0,
    requesting_workers: int = 0,
    provisioning_workers: int = 0,
    observed_tpu_vms: int | None = None,
    retry: CapacityRetryProgress | None = None,
    queue_admission_denial: GcpQueueAdmissionDenialTarget | None = None,
) -> WorkloadProgress:
    return WorkloadProgress(
        workload_id="wl-v6e-16",
        workload_state=workload_state,
        phase=phase,
        scheduling_reason=None,
        failure_reason=None,
        queue_admission_denial=queue_admission_denial,
        request=WorkloadCapacityRequest(
            cells=1,
            workers_per_cell=2,
            resources_per_cell=ResourceVector(tpu=32),
            total_resources=ResourceVector(tpu=32),
        ),
        pool=WorkloadPoolProgress(
            id="research-us-central1-c-v6e-16",
            fleet_name="research-us-central1-c",
            node_type="v6e-16",
            provider="gcp",
            provider_project="project-a",
            region="us-central1",
            zone="us-central1-c",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
        ),
        capacity=WorkloadCapacitySummary(
            active_cells=0,
            active_workers=0,
            ready_cells=0,
            ready_workers=0,
            requesting_cells=0,
            requesting_workers=requesting_workers,
            waiting_cells=0,
            waiting_workers=0,
            provisioning_cells=0,
            provisioning_workers=provisioning_workers,
            lost_cells=0,
            lost_workers=lost_workers,
            observed_tpu_vms=observed_tpu_vms,
        ),
        retry=retry,
    )


@pytest.mark.parametrize(
    ("tpu", "request_shape"),
    (
        (TpuConfig(type="v6e-16"), "v6e-16 x 1 worker"),
        (TpuConfig(type="v6e-16", pods=2), "v6e-16 x 2 workers"),
    ),
)
def test_callable_submission_prints_durable_job_controls(tmp_path, capsys, tpu: TpuConfig, request_shape: str) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-receipt"),
    )
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        raise AssertionError((request.method, request.url.path))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter[None](client, store, terminal_output=_plain_output())

    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-receipt",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=tpu,
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )

    assert job.id == "wl-v6e-16"
    assert capsys.readouterr().err.splitlines() == [
        "Eyvan: job wl-v6e-16 submitted to fleet research-us-central1-c.",
        f"  request  {request_shape}",
        "  allocation  elastic",
        "  this job keeps running if this client exits.",
        "  status  eyvan workload status wl-v6e-16",
        "  logs    eyvan workload logs wl-v6e-16",
        "  cancel  eyvan workload cancel wl-v6e-16",
    ]
    assert requests == [("POST", "/api/v1/fleets/research-us-central1-c/callables")]
    client.close()


def test_callable_submission_does_not_block_an_explicit_target_after_discovery_denial(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-explicit-retry"),
    )
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    cache.record_queue_admission_denial(
        project="project-a",
        zone="us-central1-c",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        raise AssertionError((request.method, request.url.path))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter[None](
        client,
        store,
        terminal_output=_plain_output(),
        queue_admission_cache=cache,
    )

    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-explicit-retry",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16"),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )

    assert job.id == "wl-v6e-16"
    assert requests == [("POST", "/api/v1/fleets/research-us-central1-c/callables")]
    client.close()


def test_callable_result_renders_scale_from_zero_and_provider_diagnostic_once(tmp_path, monkeypatch, capsys) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-v6e-16"),
    )
    diagnostic = "ERROR: queue admission policy rejected the requested worker shape"
    queued = _progress(phase=WorkloadProgressPhase.WAITING_FOR_AUTOSCALER)
    denied_target = GcpQueueAdmissionDenialTarget(
        provider_project="project-a",
        zone="us-central1-c",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        occurred_at=utcnow() - timedelta(minutes=5),
        event_id="queue-denial-wl-v6e-16",
    )
    provider_failure = _progress(
        phase=WorkloadProgressPhase.QUEUE_ADMISSION_DENIED,
        retry=CapacityRetryProgress(
            state=OutboxState.FAILED,
            attempts=1,
            next_attempt_at=None,
            diagnostic=diagnostic,
            failure_code=CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
        ),
        queue_admission_denial=denied_target,
    )
    states = iter((WorkloadState.PENDING, WorkloadState.PENDING, WorkloadState.FAILED))
    progress_updates = iter((queued, provider_failure, provider_failure))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16":
            state = next(states)
            return httpx.Response(
                200,
                json=_workload(
                    execution,
                    state,
                    diagnostic if state is WorkloadState.FAILED else None,
                    AutoscaleBlockReason.QUEUE_ADMISSION_DENIED if state is WorkloadState.FAILED else None,
                    denied_target if state is WorkloadState.FAILED else None,
                ).model_dump(mode="json"),
            )
        if request.url.path == "/api/v1/workloads/wl-v6e-16/progress":
            return httpx.Response(200, json=next(progress_updates).model_dump(mode="json"))
        raise AssertionError((request.method, request.url.path))

    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _seconds: None)
    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    submitter = EyvanCallableSubmitter[None](
        client,
        store,
        terminal_output=_plain_output(),
        queue_admission_cache=cache,
    )
    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-v6e-16",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16", pods=2),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )
    capsys.readouterr()

    with pytest.raises(EyvanJobError, match="queue admission policy rejected"):
        job.result(timeout=1)

    output = capsys.readouterr().err
    assert output.splitlines() == [
        "Eyvan: requesting | v6e-16 x 2 workers | us-central1-c",
        "  0/2 workers ready | requesting one atomic allocation",
        "Eyvan: blocked | v6e-16 x 2 workers | us-central1-c",
        "  GCP denied queue admission; no queued resource was created",
        f"  {diagnostic}",
    ]
    assert "TPU devices" not in output
    assert cache.blocks_queue_admission_denial(
        project="project-a",
        zone="us-central1-c",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    client.close()


def test_callable_wait_caches_a_terminal_queue_denial_without_progress_polling(tmp_path, monkeypatch) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-terminal-denial"),
    )
    denied_target = GcpQueueAdmissionDenialTarget(
        provider_project="project-a",
        zone="us-central1-c",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        occurred_at=utcnow() - timedelta(minutes=5),
        event_id="queue-denial-wl-v6e-16",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16":
            workload = _workload(
                execution,
                WorkloadState.FAILED,
                "GCP denied queue admission",
                AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
                denied_target,
            )
            return httpx.Response(200, json=workload.model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16/progress":
            raise AssertionError("a terminal queue-admission denial must not depend on progress polling")
        raise AssertionError((request.method, request.url.path))

    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _seconds: None)
    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    submitter = EyvanCallableSubmitter[None](
        client,
        store,
        terminal_output=_plain_output(),
        queue_admission_cache=cache,
    )
    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-terminal-denial",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16", pods=2),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )

    completion = job.wait(timeout=1)

    assert completion.state is WorkloadState.FAILED
    assert completion.succeeded is False
    assert completion.message == "GCP denied queue admission"

    assert cache.blocks_queue_admission_denial(
        project="project-a",
        zone="us-central1-c",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    client.close()


def test_callable_result_preserves_queue_denial_when_local_cache_target_is_invalid(tmp_path, monkeypatch) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-invalid-cache-target"),
    )
    denied_target = GcpQueueAdmissionDenialTarget(
        provider_project="not a valid GCP project",
        zone="us-central1-c",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16":
            return httpx.Response(
                200,
                json=_workload(
                    execution,
                    WorkloadState.FAILED,
                    "GCP denied queue admission",
                    AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
                    denied_target,
                ).model_dump(mode="json"),
            )
        raise AssertionError((request.method, request.url.path))

    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _seconds: None)
    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter[None](
        client,
        store,
        terminal_output=_plain_output(),
        queue_admission_cache=AutoscaleConfigCache(tmp_path / "autoscale"),
    )
    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-invalid-cache-target",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16", pods=2),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )

    with pytest.raises(EyvanJobError, match="GCP denied queue admission"):
        job.result(timeout=1)

    client.close()


def test_callable_result_does_not_hide_progress_authorization_errors(tmp_path, capsys) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-progress-auth"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16":
            return httpx.Response(200, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16/progress":
            return httpx.Response(401, json={"detail": "X-Eyvan-User is required"})
        raise AssertionError((request.method, request.url.path))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter[None](client, store, terminal_output=_plain_output())
    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-progress-auth",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16", pods=2),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )
    capsys.readouterr()

    with pytest.raises(EyvanHTTPError, match="X-Eyvan-User is required"):
        job.result(timeout=1)

    assert capsys.readouterr().err == ""
    client.close()


@pytest.mark.parametrize("progress_failure", ("forbidden", "not_found", "unavailable"))
def test_callable_result_uses_the_durable_workload_fallback_when_progress_is_temporarily_unavailable(
    tmp_path,
    monkeypatch,
    capsys,
    progress_failure: str,
) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-progress-fallback"),
    )
    states = iter((WorkloadState.PENDING, WorkloadState.CANCELLED))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16":
            return httpx.Response(200, json=_workload(execution, next(states)).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16/progress":
            if progress_failure == "forbidden":
                return httpx.Response(403, json={"detail": "project read permission required"})
            if progress_failure == "not_found":
                return httpx.Response(404, json={"detail": "workload progress has not been initialized"})
            raise httpx.ConnectError("control plane restarted")
        raise AssertionError((request.method, request.url.path))

    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _seconds: None)
    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter[None](client, store, terminal_output=_plain_output())
    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-progress-fallback",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16", pods=2),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )
    capsys.readouterr()

    with pytest.raises(EyvanJobError, match="cancelled"):
        job.result(timeout=1)

    assert capsys.readouterr().err.splitlines() == [
        "Eyvan: queued",
        "  waiting for matching capacity",
        "Eyvan: cancelled",
        "  job cancelled",
    ]
    client.close()


@pytest.mark.parametrize(
    (
        "state",
        "phase",
        "lost_workers",
        "requesting_workers",
        "provisioning_workers",
        "expected_state",
        "expected_summary",
    ),
    (
        (
            WorkloadState.CHECKPOINTING,
            WorkloadProgressPhase.CHECKPOINTING,
            0,
            0,
            0,
            "checkpointing",
            "writing checkpoint before releasing capacity",
        ),
        (
            WorkloadState.CANCELLING,
            WorkloadProgressPhase.CANCELLING,
            0,
            0,
            0,
            "cancelling",
            "releasing workload capacity",
        ),
        (
            WorkloadState.LEASED,
            WorkloadProgressPhase.WAITING_FOR_DISPATCH,
            0,
            0,
            0,
            "dispatching",
            "capacity leased | waiting for dispatch",
        ),
        (
            WorkloadState.PENDING,
            WorkloadProgressPhase.PROVISIONING,
            0,
            0,
            2,
            "provisioning",
            "0/2 workers ready | GCP is provisioning the atomic allocation",
        ),
        (
            WorkloadState.PENDING,
            WorkloadProgressPhase.CAPACITY_ALLOCATED,
            0,
            0,
            0,
            "allocated",
            "0/2 workers ready | GCP allocated capacity; TPU workers are starting",
        ),
        (
            WorkloadState.PENDING,
            WorkloadProgressPhase.NATIVE_WORKERS_CONNECTING,
            0,
            0,
            0,
            "connecting",
            "0/2 workers ready | GCP reported 8 TPU VMs; native workers are connecting",
        ),
        (
            WorkloadState.PENDING,
            WorkloadProgressPhase.PREEMPTED,
            1,
            0,
            0,
            "preempted",
            "Spot capacity lost: 1 worker | re-requesting 2 workers",
        ),
        (
            WorkloadState.PENDING,
            WorkloadProgressPhase.REQUESTING_CAPACITY,
            0,
            2,
            0,
            "requesting",
            "0/2 workers ready | sending one atomic allocation request to GCP",
        ),
    ),
)
def test_callable_result_renders_nonterminal_states(
    tmp_path,
    monkeypatch,
    capsys,
    state: WorkloadState,
    phase: WorkloadProgressPhase,
    lost_workers: int,
    requesting_workers: int,
    provisioning_workers: int,
    expected_state: str,
    expected_summary: str,
) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-control-state"),
    )
    states = iter((state, WorkloadState.CANCELLED))
    progress_updates = iter(
        (
            _progress(
                phase=phase,
                workload_state=state,
                lost_workers=lost_workers,
                requesting_workers=requesting_workers,
                provisioning_workers=provisioning_workers,
                observed_tpu_vms=8 if phase is WorkloadProgressPhase.NATIVE_WORKERS_CONNECTING else None,
            ),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16":
            return httpx.Response(200, json=_workload(execution, next(states)).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16/progress":
            return httpx.Response(200, json=next(progress_updates).model_dump(mode="json"))
        raise AssertionError((request.method, request.url.path))

    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _seconds: None)
    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter[None](client, store, terminal_output=_plain_output())
    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-control-state",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16", pods=2),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )
    capsys.readouterr()

    with pytest.raises(EyvanJobError, match="cancelled"):
        job.result(timeout=1)

    output = capsys.readouterr().err
    assert output.splitlines() == [
        f"Eyvan: {expected_state} | v6e-16 x 2 workers | us-central1-c",
        f"  {expected_summary}",
        "Eyvan: cancelled",
        "  job cancelled",
    ]
    assert "TPU devices" not in output
    assert "32 TPU" not in output
    client.close()


def test_callable_progress_wait_short_circuits_progress_polling_on_a_terminal_observation(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=artifact.uri,
        sha256=artifact.sha256,
        python=artifact.python,
        result_uri=store.allocate_result_uri("call-terminal-observation"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "POST":
            return httpx.Response(201, json=_workload(execution, WorkloadState.PENDING).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16":
            return httpx.Response(200, json=_workload(execution, WorkloadState.SUCCEEDED).model_dump(mode="json"))
        if request.url.path == "/api/v1/workloads/wl-v6e-16/progress":
            raise AssertionError("a terminal workload observation must not poll the progress endpoint")
        raise AssertionError((request.method, request.url.path))

    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _seconds: None)
    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter[None](client, store, terminal_output=_plain_output())
    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-terminal-observation",
            fleet="research-us-central1-c",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v6e-16", pods=2),
            artifact=artifact,
            result_uri=execution.result_uri or "",
        )
    )
    capsys.readouterr()

    completion = job.wait(timeout=1, progress=True)

    assert completion.state is WorkloadState.SUCCEEDED
    assert completion.succeeded is True
    assert capsys.readouterr().err.splitlines() == [
        "Eyvan: succeeded",
        "  job completed",
    ]
    client.close()


@pytest.mark.parametrize(
    ("phase", "cells", "requesting", "provisioning", "expected"),
    (
        (WorkloadProgressPhase.WAITING_FOR_AUTOSCALER, 1, 0, 0, "0/2 workers ready | requesting one elastic cell"),
        (WorkloadProgressPhase.WAITING_FOR_AUTOSCALER, 2, 0, 0, "0/4 workers ready | requesting 2 elastic cells"),
        (
            WorkloadProgressPhase.REQUESTING_CAPACITY,
            2,
            2,
            0,
            "0/4 workers ready | sending 2 elastic cell requests to GCP",
        ),
        (WorkloadProgressPhase.PROVISIONING, 2, 0, 2, "0/4 workers ready | GCP is provisioning 2 elastic cells"),
    ),
)
def test_an_elastic_job_is_never_narrated_as_one_atomic_allocation(
    phase: WorkloadProgressPhase, cells: int, requesting: int, provisioning: int, expected: str
) -> None:
    """Describe elastic cell requests consistently without calling them an atomic allocation."""

    from eyvan.records import PlacementMode
    from eyvan.sdk.client.callables import _CallableProgressRenderer

    progress = _progress(phase=phase, requesting_workers=requesting, provisioning_workers=provisioning)
    progress = progress.model_copy(update={"request": progress.request.model_copy(update={"cells": cells})})
    renderer = _CallableProgressRenderer(None, _plain_output(), None)  # type: ignore[arg-type]

    elastic = renderer._progress_update(progress, placement=PlacementMode.ELASTIC)
    atomic = renderer._progress_update(progress, placement=PlacementMode.ATOMIC)

    assert elastic.summary == expected
    assert "atomic" not in elastic.summary
    assert "atomic allocation" in atomic.summary
