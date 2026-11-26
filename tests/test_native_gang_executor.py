# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts for native execution across independently fenced TPU cells."""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor


def _context(*, key: str, generation: int, token: str) -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _client(endpoint: str, handler: tp.Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url=endpoint, transport=httpx.MockTransport(handler))


def _json_object(value: JsonValue) -> JsonObject:
    assert is_json_object(value)
    return value


def _json_string(value: JsonValue) -> str:
    assert isinstance(value, str)
    return value


def _payload(request: httpx.Request) -> JsonObject:
    value = json.loads(request.content)
    assert is_json_object(value)
    return value


def _spec(*, submission_id: str = "gang-job", environment: dict[str, str] | None = None) -> JsonObject:
    environment_document: JsonObject = dict(environment or {})
    document: JsonObject = {
        "entrypoint": "python train.py --steps 4",
        "submission_id": submission_id,
        "environment": environment_document,
        "metadata": {"eyvan_workload_id": submission_id},
    }
    return document


def _members() -> tuple[NativeGangMember, NativeGangMember]:
    return (
        NativeGangMember(
            cell_id="cell-b",
            endpoints=("http://10.12.0.20:8100", "http://10.12.0.21:8100"),
            secret="worker-secret-b",
            context=_context(key="submit-b", generation=7, token="fence-b"),
        ),
        NativeGangMember(
            cell_id="cell-a",
            endpoints=("http://10.12.0.10:8100", "http://10.12.0.11:8100"),
            secret="worker-secret-a",
            context=_context(key="submit-a", generation=7, token="fence-a"),
        ),
    )


def test_native_gang_submit_uses_global_ranks_and_per_cell_credentials() -> None:
    submissions: list[tuple[str, httpx.Request, JsonObject]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/workloads"
        submissions.append((request.url.host, request, _payload(request)))
        return httpx.Response(200, json={"remote_id": "gang-job"})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    original_environment = {"USER_FLAG": "kept", "PYTHONPATH": "/workspace"}
    workload = _spec(environment=original_environment)
    try:
        executor = NativeGangExecutor(_members(), clients=clients)

        assert executor.submit(workload) == "gang-job"

        assert workload["environment"] == original_environment
        assert [host for host, _, _ in submissions] == ["10.12.0.10", "10.12.0.11", "10.12.0.20", "10.12.0.21"]
        coordinator_addresses = {
            _json_string(_json_object(_json_object(payload["spec"])["environment"])["JAX_COORDINATOR_ADDRESS"])
            for _, _, payload in submissions
        }
        assert coordinator_addresses == {"10.12.0.10:12345"}
        for rank, (host, request, payload) in enumerate(submissions):
            expected_cell = "cell-a" if host.startswith("10.12.0.1") else "cell-b"
            expected_secret = "worker-secret-a" if expected_cell == "cell-a" else "worker-secret-b"
            expected_mutation = {
                "idempotency_key": "submit-a" if expected_cell == "cell-a" else "submit-b",
                "generation": 7,
                "fencing_token": "fence-a" if expected_cell == "cell-a" else "fence-b",
            }
            assert request.headers["Authorization"] == f"Bearer {expected_secret}"
            assert payload["mutation"] == expected_mutation
            environment = _json_object(_json_object(payload["spec"])["environment"])
            assert environment["USER_FLAG"] == "kept"
            assert environment["EYVAN_DISTRIBUTED_PROCESS_COUNT"] == "4"
            assert environment["EYVAN_DISTRIBUTED_PROCESS_ID"] == str(rank)
            assert environment["EYVAN_REPLICA_COUNT"] == "4"
            assert environment["EYVAN_REPLICA_INDEX"] == str(rank)
            assert environment["JAX_PROCESS_COUNT"] == "4"
            assert environment["JAX_PROCESS_ID"] == str(rank)
            assert environment["JAX_PROCESS_INDEX"] == str(rank)
            assert environment["JAX_PARTITION_INDEX"] == ("0" if expected_cell == "cell-a" else "1")
            assert environment["JAX_COORDINATOR_PORT"] == "12345"
            assert environment["EYVAN_GANG_CELL_ID"] == expected_cell
            assert environment["EYVAN_GANG_CELL_INDEX"] == ("0" if expected_cell == "cell-a" else "1")
            assert environment["PYTHONPATH"].endswith("/workspace")
            assert environment["PYTHONPATH"] != "/workspace"
    finally:
        for client in clients:
            client.close()


def test_native_gang_structured_logs_use_the_same_global_rank_order() -> None:
    from eyvan.worker.runtime.log_records import split_log_cursor
    from eyvan.worker.runtime.registry import StructuredLogExecutionDriver

    requested: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/workloads/gang-job/log-page"
        host = request.url.host
        requested[host] = request.url.params.get("after", "")
        cursor = f"s={host}|position=1"
        return httpx.Response(
            200,
            json={
                "records": [{"message": host, "stream": "stdout", "rank": 0, "cursor": cursor}],
                "next_cursor": cursor,
                "unavailable": False,
                "detail": "",
                "truncated_ranks": [],
            },
        )

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeGangExecutor(_members(), clients=clients)
        assert isinstance(executor, StructuredLogExecutionDriver)
        first = executor.log_page("gang-job")

        assert [(record.rank, record.message) for record in first.records] == [
            (0, "10.12.0.10"),
            (1, "10.12.0.11"),
            (2, "10.12.0.20"),
            (3, "10.12.0.21"),
        ]
        _attempt, cursors = split_log_cursor(first.next_cursor)
        assert cursors == {
            rank: f"s={endpoint.split('//')[1].split(':')[0]}|position=1" for rank, endpoint in enumerate(endpoints)
        }

        executor.log_page("gang-job", after=first.next_cursor)
        assert requested == {
            endpoint.split("//")[1].split(":")[0]: cursors[rank] for rank, endpoint in enumerate(endpoints)
        }
    finally:
        for client in clients:
            client.close()


def test_native_gang_empty_and_unreachable_ranks_retain_their_previous_cursors() -> None:
    """A quiet or lost rank must not restart from the beginning on the next poll."""

    from collections import Counter

    from eyvan.worker.runtime.log_records import split_log_cursor

    calls: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls[host] += 1
        if calls[host] == 1:
            cursor = f"s={host}|position=1"
            return httpx.Response(
                200,
                json={"records": [{"message": host}], "next_cursor": cursor, "unavailable": False},
            )
        if host == "10.12.0.10":
            return httpx.Response(200, json={"records": [], "next_cursor": "", "unavailable": False})
        if host == "10.12.0.11":
            raise httpx.ConnectError("host disappeared", request=request)
        cursor = f"s={host}|position=2"
        return httpx.Response(200, json={"records": [{"message": host}], "next_cursor": cursor})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeGangExecutor(_members(), clients=clients)
        first = executor.log_page("gang-job")
        _attempt, first_positions = split_log_cursor(first.next_cursor)

        second = executor.log_page("gang-job", after=first.next_cursor)
        _attempt, second_positions = split_log_cursor(second.next_cursor)

        assert second_positions[0] == first_positions[0]
        assert second_positions[1] == first_positions[1]
        assert second_positions[2].endswith("position=2")
        assert second_positions[3].endswith("position=2")
        assert second.truncated_ranks == (1,)
    finally:
        for client in clients:
            client.close()


def test_native_gang_retires_every_attempted_worker_with_its_cell_fence() -> None:
    calls: list[tuple[str, str, httpx.Request]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append((host, request.url.path, request))
        if request.url.path == "/api/v1/workloads":
            if host == "10.12.0.20":
                # A server-side failure can be ambiguous: the worker might
                # have persisted the launch before its reply reached Eyvan.
                return httpx.Response(503, json={"detail": "worker unavailable"})
            return httpx.Response(200, json={"remote_id": "gang-partial"})
        assert request.url.path == "/api/v1/workloads/gang-partial/retire"
        assert request.method == "POST"
        return httpx.Response(200, json={})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeGangExecutor(_members(), clients=clients)

        with pytest.raises(httpx.HTTPStatusError):
            executor.submit(_spec(submission_id="gang-partial"))

        assert [(host, path) for host, path, _ in calls] == [
            ("10.12.0.10", "/api/v1/workloads"),
            ("10.12.0.11", "/api/v1/workloads"),
            ("10.12.0.20", "/api/v1/workloads"),
            ("10.12.0.20", "/api/v1/workloads/gang-partial/retire"),
            ("10.12.0.11", "/api/v1/workloads/gang-partial/retire"),
            ("10.12.0.10", "/api/v1/workloads/gang-partial/retire"),
        ]
        for host, path, request in calls:
            expected_cell = "cell-a" if host.startswith("10.12.0.1") else "cell-b"
            expected_secret = "worker-secret-a" if expected_cell == "cell-a" else "worker-secret-b"
            expected_mutation = {
                "idempotency_key": "submit-a" if expected_cell == "cell-a" else "submit-b",
                "generation": 7,
                "fencing_token": "fence-a" if expected_cell == "cell-a" else "fence-b",
            }
            assert request.headers["Authorization"] == f"Bearer {expected_secret}"
            if path.endswith("/retire"):
                mutation = _payload(request)["mutation"]
                # Same fence, but a key of its own: a rollback that reused the
                # submission key could not be told apart from a stale replay
                # once the dispatch retry relaunched under that key.
                assert mutation["generation"] == expected_mutation["generation"]
                assert mutation["fencing_token"] == expected_mutation["fencing_token"]
                assert mutation["idempotency_key"].startswith("rollback:")
                assert mutation["idempotency_key"].endswith(f":{expected_mutation['idempotency_key']}")
        retire_keys = [
            _payload(request)["mutation"]["idempotency_key"] for _, path, request in calls if "retire" in path
        ]
        assert len(set(retire_keys)) == len(retire_keys), retire_keys
    finally:
        for client in clients:
            client.close()


def test_native_gang_checkpoint_cleanup_rolls_back_under_each_cell_fence() -> None:
    calls: list[tuple[str, str, httpx.Request]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append((host, request.url.path, request))
        if request.url.path.endswith("/checkpoint"):
            if host == "10.12.0.20":
                return httpx.Response(503, json={"detail": "worker unavailable"})
            return httpx.Response(200, json={"remote_id": "ckpt-shared"})
        assert request.url.path == "/api/v1/workloads/ckpt-shared/retire"
        return httpx.Response(200, json={})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeGangExecutor(_members(), clients=clients)

        with pytest.raises(httpx.HTTPStatusError):
            executor.checkpoint(
                "gang-x",
                hook="python hook.py",
                output_uri=None,
                context=_context(key="ckpt-shared", generation=7, token="fence-a"),
            )

        retires = [(host, request) for host, path, request in calls if path.endswith("/retire")]
        retired_hosts = {host for host, _request in retires}
        assert {"10.12.0.10", "10.12.0.11"} <= retired_hosts
        for host, request in retires:
            expected_token = "fence-a" if host.startswith("10.12.0.1") else "fence-b"
            mutation = _payload(request)["mutation"]
            # Each rank is retired under its own cell fence with a rollback key
            # of its own -- the shared checkpoint key would collide with the
            # checkpoint command's own claim on the worker.
            assert (mutation["generation"], mutation["fencing_token"]) == (7, expected_token)
            assert mutation["idempotency_key"].startswith("rollback:")
            assert mutation["idempotency_key"].endswith(":ckpt-shared")
        keys = [_payload(request)["mutation"]["idempotency_key"] for _host, request in retires]
        assert len(set(keys)) == len(keys), "each rank's rollback carries its own key"
    finally:
        for client in clients:
            client.close()


def test_native_gang_surfaces_retire_failures_and_still_retires_remaining_ranks() -> None:
    retire_attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if request.url.path == "/api/v1/workloads":
            if host == "10.12.0.21":
                return httpx.Response(503, json={"detail": "worker unavailable"})
            return httpx.Response(200, json={"remote_id": "gang-partial"})
        assert request.url.path == "/api/v1/workloads/gang-partial/retire"
        retire_attempts.append(host)
        if host == "10.12.0.20":
            return httpx.Response(500, json={"detail": "retire exploded"})
        return httpx.Response(200, json={})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeGangExecutor(_members(), clients=clients)

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            executor.submit(_spec(submission_id="gang-partial"))

        message = str(excinfo.value)
        assert "partial-launch cleanup could not retire" in message
        assert "rank 2 (http://10.12.0.20:8100): HTTPStatusError" in message
        # One failing retire must not stop cleanup of the other attempted ranks.
        assert retire_attempts == ["10.12.0.21", "10.12.0.20", "10.12.0.11", "10.12.0.10"]
    finally:
        for client in clients:
            client.close()


def test_native_gang_strips_caller_distributed_topology_overrides() -> None:
    submissions: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        submissions.append(_payload(request))
        return httpx.Response(200, json={"remote_id": "gang-job"})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
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
            "USER_FLAG": "kept",
        }
    )
    try:
        executor = NativeGangExecutor(_members(), clients=clients)

        assert executor.submit(workload) == "gang-job"

        assert len(submissions) == 4
        for payload in submissions:
            environment = _json_object(_json_object(payload["spec"])["environment"])
            assert environment["USER_FLAG"] == "kept"
            assert environment["JAX_COORDINATOR_ADDRESS"] == "10.12.0.10:12345"
            assert environment["JAX_PROCESS_COUNT"] == "4"
            assert environment["JAX_PROCESS_ID"] != "98"
            assert environment["JAX_PARTITION_INDEX"] != "9"
            assert "GCE_METADATA_IP" not in environment
            assert "JAX_COORDINATOR_BIND_ADDRESS" not in environment
            assert "JAX_LOCAL_DEVICE_IDS" not in environment
            assert "JAX_SLICE_INDEX" not in environment
            assert "MEGASCALE_NUM_SLICES" not in environment
            assert "TPU_SKIP_MDS_QUERY" not in environment
    finally:
        for client in clients:
            client.close()


def test_native_gang_aggregates_status_across_cells() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/workloads/gang-job":
            state = "RUNNING" if request.url.host == "10.12.0.10" else "SUCCEEDED"
            return httpx.Response(200, json={"remote_id": "gang-job", "status": state})
        if request.url.path == "/api/v1/workloads/gang-job/logs":
            return httpx.Response(200, text=f"logs from {request.url.host}")
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeGangExecutor(_members(), clients=clients)

        report = executor.status("gang-job")

        assert report["status"] is ExecutionWorkloadStatus.RUNNING
        assert [worker["rank"] for worker in report["workers"]] == [0, 1, 2, 3]
        assert [worker["cell_id"] for worker in report["workers"]] == ["cell-a", "cell-a", "cell-b", "cell-b"]
        assert report["workers"][0]["status"] is ExecutionWorkloadStatus.RUNNING
        assert all(worker["status"] is ExecutionWorkloadStatus.SUCCEEDED for worker in report["workers"][1:])
        logs = executor.logs("gang-job")
        assert "rank 0 cell cell-a" in logs
        assert "rank 3 cell cell-b" in logs
    finally:
        for client in clients:
            client.close()


def test_native_gang_marks_a_malformed_worker_observation_indeterminate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/v1/workloads/gang-job":
            raise AssertionError(f"unexpected request: {request.method} {request.url.path}")
        if request.url.host == "10.12.0.20":
            return httpx.Response(200, json={"remote_id": "gang-job", "message": "status was omitted"})
        return httpx.Response(200, json={"remote_id": "gang-job", "status": "RUNNING"})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        report = NativeGangExecutor(_members(), clients=clients).status("gang-job")

        assert report["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert report["workers"][2]["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert "omitted status" in report["workers"][2]["message"]
    finally:
        for client in clients:
            client.close()


def test_a_transport_failure_observing_one_rank_is_not_that_ranks_death() -> None:
    """One unreadable endpoint must not settle a gang whose ranks are alive.

    Observed on a live 4x v6e-8: every rank completed its callable and exited
    0, and the fastest rank's vanished HTTP endpoint -- a closed keep-alive to
    the observer -- was read as LOST. The aggregate failed the workload and
    tore down three still-running peers, discarding a finished job. A rank the
    observer cannot *ask* is INDETERMINATE: the next poll usually answers, and
    a host that stays unreadable escalates through the indeterminate grace
    path instead of by killing its peers.
    """

    def healthy_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "gang-job", "status": "RUNNING"})

    def unreachable_worker(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [
        _client(endpoints[0], healthy_worker),
        _client(endpoints[1], healthy_worker),
        _client(endpoints[2], unreachable_worker),
        _client(endpoints[3], healthy_worker),
    ]
    try:
        report = NativeGangExecutor(_members(), clients=clients).status("gang-job")

        assert report["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert report["workers"][2]["status"] is ExecutionWorkloadStatus.INDETERMINATE
        assert "connection refused" in report["workers"][2]["message"]
    finally:
        for client in clients:
            client.close()


def test_a_rank_that_answers_succeeded_after_an_unreadable_poll_converges() -> None:
    """Converge after an unreadable rank later reports success while its peers are still running."""

    polls: list[str] = []

    def flaky_then_finished(request: httpx.Request) -> httpx.Response:
        polls.append(request.url.path)
        if len(polls) == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json={"remote_id": "gang-job", "status": "SUCCEEDED"})

    def healthy_worker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "gang-job", "status": "RUNNING"})

    endpoints = (
        "http://10.12.0.10:8100",
        "http://10.12.0.11:8100",
        "http://10.12.0.20:8100",
        "http://10.12.0.21:8100",
    )
    clients = [
        _client(endpoints[0], healthy_worker),
        _client(endpoints[1], healthy_worker),
        _client(endpoints[2], flaky_then_finished),
        _client(endpoints[3], healthy_worker),
    ]
    try:
        executor = NativeGangExecutor(_members(), clients=clients)
        first = executor.status("gang-job")
        assert first["status"] is ExecutionWorkloadStatus.INDETERMINATE

        second = executor.status("gang-job")

        assert second["status"] is ExecutionWorkloadStatus.RUNNING, "a finished rank converges"
        assert second["workers"][2]["status"] is ExecutionWorkloadStatus.SUCCEEDED
    finally:
        for client in clients:
            client.close()


def test_a_closed_keepalive_is_healed_by_one_retry_not_read_as_rank_loss() -> None:
    """A pooled connection the server closed is the observer's problem.

    The poll interval can outlive the worker's keep-alive idle timeout; the
    first request on the stale socket dies with ``RemoteProtocolError`` before
    any byte reaches the server. A read is provably safe to retry once, and
    transparently healing it is what keeps that race from ever reaching the
    status mapping.
    """

    calls: list[str] = []

    def stale_then_healthy(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response", request=request)
        return httpx.Response(200, json={"remote_id": "wl-retry", "status": "RUNNING"})

    client = _client("http://10.12.0.20:8100", stale_then_healthy)
    executor = NativeRemoteExecutor("http://10.12.0.20:8100", secret="s", client=client)
    try:
        observation = executor.status("wl-retry")

        assert observation.status is ExecutionWorkloadStatus.RUNNING
        assert len(calls) == 2, "exactly one healing retry"
    finally:
        client.close()
