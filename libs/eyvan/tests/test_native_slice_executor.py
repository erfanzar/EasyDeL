# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts for the native multi-host TPU slice execution adapter."""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.worker.runtime._multihost_site import GangInitializationPolicy
from eyvan.worker.runtime.native_slice import JaxClusterDetectionMode, NativeSliceExecutor


def _context(*, key: str = "submit-1", generation: int = 1, token: str = "token-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _client(endpoint: str, handler: tp.Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url=endpoint, transport=httpx.MockTransport(handler))


def _json_object(value: JsonValue) -> JsonObject:
    assert is_json_object(value)
    return value


def _json_string(value: JsonValue) -> str:
    assert isinstance(value, str)
    return value


def _spec(*, submission_id: str = "slice-job", environment: dict[str, str] | None = None) -> JsonObject:
    environment_document: JsonObject = dict(environment or {})
    document: JsonObject = {
        "entrypoint": "python train.py --steps 4",
        "submission_id": submission_id,
        "environment": environment_document,
        "gang": {"replicas": 1, "per_replica": {"tpu": 8}},
        "metadata": {"eyvan_workload_id": submission_id},
    }
    return document


def _payload(request: httpx.Request) -> JsonObject:
    value = json.loads(request.content)
    assert is_json_object(value)
    return value


def test_native_slice_requires_authenticated_endpoints() -> None:
    clients = [
        _client(
            "http://worker-a:8100",
            lambda _request: httpx.Response(200, json={"status": "ok", "runtime": "native-worker"}),
        ),
        _client(
            "http://worker-b:8100",
            lambda _request: httpx.Response(200, json={"status": "ok", "runtime": "native-worker"}),
        ),
    ]
    try:
        with pytest.raises(ValueError, match="bearer secret"):
            NativeSliceExecutor(("http://worker-a:8100", "http://worker-b:8100"), secret="", clients=clients)
        with pytest.raises(ValueError, match="endpoint"):
            NativeSliceExecutor(("http://worker-a:8100", " "), secret="worker-secret", clients=clients)
    finally:
        for client in clients:
            client.close()


def test_native_slice_health_requires_every_authenticated_worker() -> None:
    requests: list[httpx.Request] = []

    def healthy(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok", "runtime": "native-worker"})

    def unavailable(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, json={"status": "starting"})

    clients = [_client("http://worker-a:8100", healthy), _client("http://worker-b:8100", unavailable)]
    try:
        executor = NativeSliceExecutor(
            ("http://worker-a:8100", "http://worker-b:8100"),
            secret="worker-secret",
            clients=clients,
        )

        assert executor.health() is False
        assert [(request.url.host, request.url.path) for request in requests] == [
            ("worker-a", "/api/v1/health"),
            ("worker-b", "/api/v1/health"),
        ]
        assert {request.headers["Authorization"] for request in requests} == {"Bearer worker-secret"}
    finally:
        for client in clients:
            client.close()


def test_native_slice_submit_fans_out_a_deterministic_jax_environment() -> None:
    submissions: list[tuple[str, JsonObject]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/workloads"
        assert request.headers["Authorization"] == "Bearer worker-secret"
        submissions.append((request.url.host, _payload(request)))
        return httpx.Response(200, json={"remote_id": "slice-job"})

    endpoints = ("http://10.12.0.10:8100", "http://10.12.0.11:8100")
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    original_environment = {"USER_FLAG": "kept", "PYTHONPATH": "/workspace"}
    workload = _spec(environment=original_environment)
    try:
        executor = NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients)

        assert executor.submit(workload, _context()) == "slice-job"

        assert workload["environment"] == original_environment
        assert [host for host, _ in submissions] == ["10.12.0.10", "10.12.0.11"]
        coordinator_addresses = {
            _json_string(_json_object(_json_object(payload["spec"])["environment"])["JAX_COORDINATOR_ADDRESS"])
            for _, payload in submissions
        }
        assert coordinator_addresses == {"10.12.0.10:12345"}
        for rank, (_, payload) in enumerate(submissions):
            assert payload["mutation"] == {
                "idempotency_key": "submit-1",
                "generation": 1,
                "fencing_token": "token-1",
            }
            spec = _json_object(payload["spec"])
            assert spec["submission_id"] == "slice-job"
            environment = _json_object(spec["environment"])
            assert environment["USER_FLAG"] == "kept"
            assert environment["EYVAN_JAX_CLUSTER_DETECTION"] == "explicit"
            assert environment["JAX_PROCESS_ID"] == str(rank)
            assert environment["JAX_PROCESS_INDEX"] == str(rank)
            assert environment["JAX_PROCESS_COUNT"] == "2"
            assert environment["JAX_COORDINATOR_PORT"] == "12345"
            assert environment["EYVAN_JAX_MULTIHOST"] == "1"
            assert environment["PYTHONPATH"].endswith("/workspace")
            assert environment["PYTHONPATH"] != "/workspace"
    finally:
        for client in clients:
            client.close()


def test_native_multislice_defers_jax_topology_to_gce_metadata() -> None:
    submissions: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        submissions.append(_payload(request))
        return httpx.Response(200, json={"remote_id": "slice-job"})

    endpoints = ("http://worker-a:8100", "http://worker-b:8100")
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    workload = _spec(
        environment={
            "GCE_METADATA_IP": "attacker.invalid",
            "JAX_COORDINATOR_ADDRESS": "stale:12345",
            "JAX_COORDINATOR_BIND_ADDRESS": "127.0.0.1:12345",
            "JAX_LOCAL_DEVICE_IDS": "0",
            "JAX_PARTITION_INDEX": "9",
            "JAX_PROCESS_COUNT": "99",
            "JAX_PROCESS_ID": "98",
            "JAX_SLICE_INDEX": "9",
            "MEGASCALE_NUM_SLICES": "99",
            "TPU_SKIP_MDS_QUERY": "1",
        }
    )
    try:
        executor = NativeSliceExecutor(
            endpoints,
            secret="worker-secret",
            clients=clients,
            jax_cluster_detection=JaxClusterDetectionMode.GCE_TPU,
        )

        assert executor.submit(workload, _context()) == "slice-job"

        for payload in submissions:
            environment = _json_object(_json_object(payload["spec"])["environment"])
            assert environment["EYVAN_JAX_CLUSTER_DETECTION"] == "gcetpu"
            assert environment["JAX_COORDINATOR_PORT"] == "12345"
            assert "GCE_METADATA_IP" not in environment
            assert "JAX_COORDINATOR_ADDRESS" not in environment
            assert "JAX_COORDINATOR_BIND_ADDRESS" not in environment
            assert "JAX_LOCAL_DEVICE_IDS" not in environment
            assert "JAX_PARTITION_INDEX" not in environment
            assert "JAX_PROCESS_COUNT" not in environment
            assert "JAX_PROCESS_ID" not in environment
            assert "JAX_PROCESS_INDEX" not in environment
            assert "JAX_SLICE_INDEX" not in environment
            assert "MEGASCALE_NUM_SLICES" not in environment
            assert "TPU_SKIP_MDS_QUERY" not in environment
    finally:
        for client in clients:
            client.close()


def test_native_slice_retires_started_workers_when_a_fanout_submission_fails() -> None:
    first_requests: list[httpx.Request] = []
    second_requests: list[httpx.Request] = []

    def first_worker(request: httpx.Request) -> httpx.Response:
        first_requests.append(request)
        if request.url.path == "/api/v1/workloads":
            return httpx.Response(200, json={"remote_id": "slice-partial"})
        # Asserted after the call, not here: the rollback loop catches every
        # exception a retire raises, so an assertion in this handler could
        # never fail the test.
        return httpx.Response(200, json={})

    def second_worker(request: httpx.Request) -> httpx.Response:
        second_requests.append(request)
        assert request.method == "POST"
        if request.url.path == "/api/v1/workloads":
            # This represents an ambiguous transport failure: the native
            # worker may have persisted the request before the reply broke.
            return httpx.Response(503, json={"detail": "worker unavailable"})
        assert request.url.path == "/api/v1/workloads/slice-partial/retire"
        return httpx.Response(200, json={})

    clients = [
        _client("http://worker-a:8100", first_worker),
        _client("http://worker-b:8100", second_worker),
    ]
    try:
        executor = NativeSliceExecutor(
            ("http://worker-a:8100", "http://worker-b:8100"),
            secret="worker-secret",
            clients=clients,
        )

        with pytest.raises(httpx.HTTPStatusError):
            executor.submit(_spec(submission_id="slice-partial"), _context())

        assert [request.url.path for request in first_requests] == [
            "/api/v1/workloads",
            "/api/v1/workloads/slice-partial/retire",
        ]
        assert [request.url.path for request in second_requests] == [
            "/api/v1/workloads",
            "/api/v1/workloads/slice-partial/retire",
        ]
        retire = first_requests[1]
        assert retire.method == "POST"
        mutation = _payload(retire)["mutation"]
        # The fence is the submission's; the key is the rollback's own.
        assert (mutation["generation"], mutation["fencing_token"]) == (1, "token-1")
        assert mutation["idempotency_key"].startswith("rollback:")
        assert mutation["idempotency_key"].endswith(":submit-1")
        keys = {_payload(request)["mutation"]["idempotency_key"] for request in (first_requests[1], second_requests[1])}
        assert len(keys) == 2, "each rank's rollback carries its own key"
    finally:
        for client in clients:
            client.close()


def test_native_slice_checkpoint_cleanup_rolls_back_under_its_own_key() -> None:
    first_requests: list[httpx.Request] = []

    def first_worker(request: httpx.Request) -> httpx.Response:
        first_requests.append(request)
        if request.url.path.endswith("/checkpoint"):
            return httpx.Response(200, json={"remote_id": "ckpt-1"})
        assert request.url.path == "/api/v1/workloads/ckpt-1/retire"
        return httpx.Response(200, json={})

    def second_worker(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/checkpoint")
        # The second rank fails after the first rank's hook already launched:
        # cleanup must retire rank 0 under a key that is not the checkpoint's
        # own, or the cleanup cannot be told apart from a checkpoint replay.
        return httpx.Response(503, json={"detail": "worker unavailable"})

    clients = [
        _client("http://worker-a:8100", first_worker),
        _client("http://worker-b:8100", second_worker),
    ]
    try:
        executor = NativeSliceExecutor(
            ("http://worker-a:8100", "http://worker-b:8100"),
            secret="worker-secret",
            clients=clients,
        )

        with pytest.raises(httpx.HTTPStatusError):
            executor.checkpoint("slice-x", hook="python hook.py", output_uri=None, context=_context(key="ckpt-1"))

        assert [request.url.path for request in first_requests] == [
            "/api/v1/workloads/slice-x/checkpoint",
            "/api/v1/workloads/ckpt-1/retire",
        ]
        mutation = _payload(first_requests[1])["mutation"]
        # The fence is the checkpoint's; the key is the rollback's own.
        assert (mutation["generation"], mutation["fencing_token"]) == (1, "token-1")
        assert mutation["idempotency_key"].startswith("rollback:")
        assert mutation["idempotency_key"].endswith(":ckpt-1")
    finally:
        for client in clients:
            client.close()


def test_native_slice_surfaces_retire_failures_and_still_retires_remaining_ranks() -> None:
    retire_attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if request.url.path == "/api/v1/workloads":
            if host == "worker-c":
                return httpx.Response(503, json={"detail": "worker unavailable"})
            return httpx.Response(200, json={"remote_id": "slice-partial"})
        assert request.url.path == "/api/v1/workloads/slice-partial/retire"
        retire_attempts.append(host)
        if host == "worker-b":
            return httpx.Response(500, json={"detail": "retire exploded"})
        return httpx.Response(200, json={})

    endpoints = ("http://worker-a:8100", "http://worker-b:8100", "http://worker-c:8100")
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients)

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            executor.submit(_spec(submission_id="slice-partial"), _context())

        message = str(excinfo.value)
        assert "partial-launch cleanup could not retire" in message
        assert "rank 1 (http://worker-b:8100): HTTPStatusError" in message
        # One failing retire must not stop cleanup of the other attempted ranks.
        assert retire_attempts == ["worker-c", "worker-b", "worker-a"]
    finally:
        for client in clients:
            client.close()


def test_native_slice_aggregates_worker_statuses_and_fans_out_stop_and_logs() -> None:
    states = ["RUNNING", "SUCCEEDED"]
    stopped: list[httpx.Request] = []

    def first_worker(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/workloads/slice-job":
            return httpx.Response(200, json={"remote_id": "slice-job", "status": states[0], "message": "rank zero"})
        if request.method == "GET" and request.url.path == "/api/v1/workloads/slice-job/process-status":
            return httpx.Response(200, json={"observed": True, "exit_status": 1})
        if request.method == "GET" and request.url.path == "/api/v1/workloads/slice-job/logs":
            return httpx.Response(200, text="rank zero log\n")
        assert request.method == "POST"
        assert request.url.path == "/api/v1/workloads/slice-job/stop"
        stopped.append(request)
        return httpx.Response(200, json={})

    def second_worker(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/workloads/slice-job":
            return httpx.Response(200, json={"remote_id": "slice-job", "status": states[1], "message": "rank one"})
        if request.method == "GET" and request.url.path == "/api/v1/workloads/slice-job/process-status":
            return httpx.Response(200, json={"observed": True, "exit_status": 0})
        if request.method == "GET" and request.url.path == "/api/v1/workloads/slice-job/logs":
            return httpx.Response(200, text="rank one log\n")
        assert request.method == "POST"
        assert request.url.path == "/api/v1/workloads/slice-job/stop"
        stopped.append(request)
        return httpx.Response(200, json={})

    endpoints = ("http://worker-a:8100", "http://worker-b:8100")
    clients = [_client(endpoints[0], first_worker), _client(endpoints[1], second_worker)]
    try:
        # Fatality off keeps this test about aggregation alone; with the fatal
        # hook on, a FAILED aggregate also polls each rank's process status,
        # which is covered by the cold-gang-initialization suite.
        executor = NativeSliceExecutor(
            endpoints,
            secret="worker-secret",
            clients=clients,
            initialization=GangInitializationPolicy(fatal=False),
        )

        partial = executor.status("slice-job")
        assert partial["remote_id"] == "slice-job"
        assert partial["status"] is ExecutionWorkloadStatus.RUNNING
        assert partial["workers"][0]["status"] is ExecutionWorkloadStatus.RUNNING
        assert partial["workers"][1]["status"] is ExecutionWorkloadStatus.SUCCEEDED
        assert [worker["endpoint"] for worker in partial["workers"]] == list(endpoints)

        states[0] = "FAILED"
        failed = executor.status("slice-job")
        assert failed["status"] is ExecutionWorkloadStatus.FAILED
        assert failed["workers"][0]["status"] is ExecutionWorkloadStatus.FAILED
        assert failed["workers"][1]["status"] is ExecutionWorkloadStatus.SUCCEEDED

        states[:] = ["NOT_FOUND", "RUNNING"]
        lost = executor.status("slice-job")
        assert lost["status"] is ExecutionWorkloadStatus.LOST

        states[:] = ["STOPPED", "RUNNING"]
        stopped_peer = executor.status("slice-job")
        assert stopped_peer["status"] is ExecutionWorkloadStatus.LOST

        states[:] = ["SUCCEEDED", "RUNNING"]
        completed_peer = executor.status("slice-job")
        assert completed_peer["status"] is ExecutionWorkloadStatus.RUNNING

        logs = executor.logs("slice-job")
        assert "rank 0" in logs.lower()
        assert "rank 1" in logs.lower()
        assert endpoints[0] in logs
        assert endpoints[1] in logs
        assert logs.index("rank zero log") < logs.index("rank one log")

        executor.stop("slice-job", _context(key="stop-1", generation=2, token="token-2"))
        assert len(stopped) == 2
        for request in stopped:
            assert request.headers["Authorization"] == "Bearer worker-secret"
            assert _payload(request)["mutation"] == {
                "idempotency_key": "stop-1",
                "generation": 2,
                "fencing_token": "token-2",
            }

        # Multi-host checkpointing is now supported: the hook fans across every
        # rank under one shared logical ID, and each worker inherits its own JAX
        # identity from that host's source job. An empty hook is still refused
        # before any rank is contacted.
        with pytest.raises(ValueError, match="must be non-empty"):
            executor.checkpoint(
                "slice-job",
                hook="   ",
                output_uri="gs://bucket/checkpoint",
                context=_context(),
            )
    finally:
        for client in clients:
            client.close()


def test_native_slice_reports_unreachable_workers_and_preserves_surviving_logs() -> None:
    def healthy_worker(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/workloads/slice-job":
            return httpx.Response(200, json={"remote_id": "slice-job", "status": "RUNNING"})
        if request.url.path == "/api/v1/workloads/slice-job/logs":
            return httpx.Response(200, text="surviving rank log")
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    def unreachable_worker(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    endpoints = ("http://worker-a:8100", "http://worker-b:8100")
    clients = [_client(endpoints[0], healthy_worker), _client(endpoints[1], unreachable_worker)]
    try:
        executor = NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients)

        report = executor.status("slice-job")

        # A connection failure leaves rank status unknown; it cannot establish execution failure.
        assert report["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert report["workers"][0]["status"] is ExecutionWorkloadStatus.RUNNING
        assert report["workers"][1]["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert "connection refused" in report["workers"][1]["message"]
        logs = executor.logs("slice-job")
        assert "surviving rank log" in logs
        assert "native worker is unreachable" in logs
    finally:
        for client in clients:
            client.close()


def test_native_slice_marks_a_malformed_worker_observation_indeterminate() -> None:
    def healthy_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "slice-job", "status": "RUNNING"})

    def malformed_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "slice-job", "message": "status was omitted"})

    endpoints = ("http://worker-a:8100", "http://worker-b:8100")
    clients = [_client(endpoints[0], healthy_worker), _client(endpoints[1], malformed_worker)]
    try:
        report = NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients).status("slice-job")

        assert report["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert report["workers"][1]["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert "omitted status" in report["workers"][1]["message"]
    finally:
        for client in clients:
            client.close()


def test_native_slice_adds_a_reason_to_an_indeterminate_worker_status() -> None:
    def healthy_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "slice-job", "status": "RUNNING"})

    def indeterminate_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "slice-job", "status": "INDETERMINATE"})

    endpoints = ("http://worker-a:8100", "http://worker-b:8100")
    clients = [_client(endpoints[0], healthy_worker), _client(endpoints[1], indeterminate_worker)]
    try:
        report = NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients).status("slice-job")

        assert report["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert report["workers"][1]["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert "without a reason" in report["workers"][1]["message"]
    finally:
        for client in clients:
            client.close()


def test_native_slice_rejects_an_unsupported_worker_status() -> None:
    def healthy_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "slice-job", "status": "RUNNING"})

    def unsupported_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "slice-job", "status": "MYSTERY"})

    endpoints = ("http://worker-a:8100", "http://worker-b:8100")
    clients = [_client(endpoints[0], healthy_worker), _client(endpoints[1], unsupported_worker)]
    try:
        report = NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients).status("slice-job")

        assert report["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert report["workers"][1]["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert "MYSTERY" in report["workers"][1]["message"]
    finally:
        for client in clients:
            client.close()
