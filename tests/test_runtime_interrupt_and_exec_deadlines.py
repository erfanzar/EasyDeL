# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline interrupt rollback and shared-budget native diagnostic contracts."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from eyvan._internal.correlation import traced_operation
from eyvan._internal.trace_context import remote_trace_parent
from eyvan.capacity.drivers.base import MutationContext
from eyvan.capacity.providers.gcp.runtime import supervisor as supervisor_module
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile
from eyvan.server.local.daemon import LocalDaemonError, LocalDaemonResult, LocalDaemonState
from eyvan.worker.runtime import native_slice
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.native_slice import NativeSliceExecutor


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("state", [LocalDaemonState.STARTED, LocalDaemonState.REUSED])
@pytest.mark.parametrize("cleanup_type", [None, LocalDaemonError, KeyboardInterrupt])
def test_ensure_rolls_back_only_its_plane_and_preserves_original(monkeypatch, tmp_path, error_type, state, cleanup_type):
    original = error_type("agent startup interrupted")
    stopped = []
    endpoint = supervisor_module.ControlPlaneEndpoint.parse("http://127.0.0.1:19090")
    plane = LocalDaemonResult(endpoint=endpoint, state=state, pid=7001, log_path=Path("/offline/plane.log"))
    profile = GcpNativeRuntimeProfile(
        project_id="offline-project",
        artifact_root_uri="gs://offline-artifacts/eyvan",
        agent_shared_secret="agent-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_root_secret="worker-secret-abcdefghijklmnopqrstuvwxyz",
        native_worker_wheel_uri=f"gs://offline-artifacts/eyvan/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl",
    )
    path = tmp_path / "profile.json"

    def stop(result, *, runtime_profile_path):
        stopped.append((result, runtime_profile_path))
        if cleanup_type is not None:
            raise cleanup_type("rollback refused")

    def start_agent(*args, **kwargs):
        raise original

    monkeypatch.setattr(supervisor_module, "serving_source_digest", lambda _: None)
    monkeypatch.setattr(supervisor_module, "build_identity", lambda: SimpleNamespace(source_digest=None))
    runtime = supervisor_module.GcpNativeRuntimeSupervisor(
        profile_store=SimpleNamespace(path=path, load=lambda: profile),
        control_plane=SimpleNamespace(ensure=lambda *a, **k: plane, stop_started=stop),
        agent=SimpleNamespace(ensure=start_agent),
    )
    with pytest.raises(BaseException) as raised:
        runtime.ensure(endpoint)
    ordinary_double_failure = (
        error_type is RuntimeError and cleanup_type is LocalDaemonError and state is LocalDaemonState.STARTED
    )
    if ordinary_double_failure:
        assert isinstance(raised.value, supervisor_module.GcpRuntimeError)
        assert raised.value.__cause__ is original
        assert "rollback refused" in str(raised.value)
    else:
        assert raised.value is original
        if cleanup_type is not None and state is LocalDaemonState.STARTED:
            assert any("rollback" in note for note in original.__notes__)
    assert stopped == ([(plane, path)] if state is LocalDaemonState.STARTED else [])


class _Clock:
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def advance_to(self, value):
        with self.lock:
            self.value = max(self.value, value)


@contextmanager
def _executor(kind, hosts, handler, *, owned=False):
    endpoints = tuple(f"http://worker-{rank:03d}:8100" for rank in range(hosts))
    clients = [httpx.Client(base_url=endpoint, transport=httpx.MockTransport(handler)) for endpoint in endpoints]
    by_endpoint = dict(zip(endpoints, clients, strict=True))
    # Exercise owned-client deadline derivation without opening sockets. Injected
    # clients deliberately retain caller policy in NativeRemoteExecutor.
    with patch("eyvan.worker.runtime.native_remote.httpx.Client", side_effect=lambda **kw: by_endpoint[kw["base_url"]]):
        if kind == "slice":
            executor = NativeSliceExecutor(endpoints, secret="offline-secret", clients=None if owned else clients)
        else:
            members = [
                NativeGangMember(
                    cell_id=cell,
                    endpoints=group,
                    secret="offline-secret",
                    context=MutationContext(idempotency_key="exec", generation=1, fencing_token="fence"),
                )
                for cell, group in [("cell-a", endpoints[: hosts // 2]), ("cell-b", endpoints[hosts // 2 :])]
            ]
            executor = NativeGangExecutor(members, clients=None if owned else clients)
    try:
        yield executor
    finally:
        executor.close()
        for client in clients:
            client.close()


def _rank(request):
    return int(request.url.host.split("-")[1])


@pytest.mark.parametrize("kind", ["slice", "gang"])
def test_two_450_second_commands_share_one_600_second_budget(monkeypatch, kind):
    """A real barrier proves overlap; virtual durations avoid long sleeps."""
    clock = _Clock()
    monkeypatch.setattr(native_slice, "monotonic", clock, raising=False)
    barrier = threading.Barrier(2, timeout=0.25)
    budgets = []
    guard = threading.Lock()

    def command(request):
        body = json.loads(request.content)
        with guard:
            budgets.append(body["timeout_s"])
        barrier.wait()
        clock.advance_to(450)
        return httpx.Response(200, json={"exit_code": 0, "stdout": str(_rank(request))})

    with _executor(kind, 2, command) as executor:
        result = executor.exec_in_workload("workload", ["diagnose"], timeout_s=600)
    assert [entry["rank"] for entry in result["ranks"]] == [0, 1]
    assert [entry["result"]["stdout"] for entry in result["ranks"]] == ["0", "1"]
    assert budgets == [600, 600]
    assert clock() == 450


@pytest.mark.parametrize("kind", ["slice", "gang"])
def test_wider_than_concurrency_bound_uses_remaining_budget_and_preserves_context(monkeypatch, kind):
    clock = _Clock()
    monkeypatch.setattr(native_slice, "monotonic", clock, raising=False)
    # 40 hosts exceed the chosen 16-thread bound. Hold the first wave until all
    # slots are occupied, then advance virtual time before queued ranks start.
    first_wave = threading.Barrier(16, action=lambda: clock.advance_to(450), timeout=1)
    lock = threading.Lock()
    requests = []
    thread_ids = set()
    parent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"

    def command(request):
        body = json.loads(request.content)
        with lock:
            index = len(requests)
            requests.append((_rank(request), body, request.headers.get("traceparent"), request.extensions["timeout"]))
            thread_ids.add(threading.get_ident())
        if index < 16:
            first_wave.wait()
        return httpx.Response(200, json={"exit_code": 0, "stdout": str(_rank(request))})

    with traced_operation("offline-exec", "correlation"), remote_trace_parent(parent, enabled=True):
        with _executor(kind, 40, command, owned=True) as executor:
            result = executor.exec_in_workload("workload", ["diagnose", "--safe"], timeout_s=600)
    assert len(requests) == 40
    assert len(thread_ids) <= 16
    assert [entry["rank"] for entry in result["ranks"]] == list(range(40))
    assert [entry["result"]["stdout"] for entry in result["ranks"]] == [str(rank) for rank in range(40)]
    assert sorted(body["timeout_s"] for _, body, _, _ in requests) == [150] * 24 + [600] * 16
    assert all(body["command"] == ["diagnose", "--safe"] and trace == parent for _, body, trace, _ in requests)
    assert all(timeout["read"] == body["timeout_s"] + 30 for _, body, _, timeout in requests)
    assert not any(t.name.startswith("eyvan-exec") for t in threading.enumerate())


@pytest.mark.parametrize("kind", ["slice", "gang"])
def test_deadline_expiry_does_not_dispatch_or_hide_queued_ranks(monkeypatch, kind):
    clock = _Clock()
    monkeypatch.setattr(native_slice, "monotonic", clock, raising=False)
    first_wave = threading.Barrier(16, action=lambda: clock.advance_to(600), timeout=1)
    requests = []
    lock = threading.Lock()

    def command(request):
        with lock:
            requests.append(_rank(request))
        first_wave.wait()
        return httpx.Response(200, json={"exit_code": 124, "stderr": "command timed out"})

    with _executor(kind, 40, command) as executor:
        result = executor.exec_in_workload("workload", ["diagnose"], timeout_s=600)
    assert len(requests) == 16
    assert len(result["ranks"]) == 40
    for entry in result["ranks"]:
        if entry["rank"] in requests:
            assert entry["result"]["exit_code"] == 124
        else:
            assert entry["result"]["observed"] is False
            assert "deadline" in entry["result"]["detail"]


@pytest.mark.parametrize("kind", ["slice", "gang"])
def test_failures_are_ordered_and_do_not_drop_other_ranks(kind):
    def command(request):
        rank = _rank(request)
        if rank == 0:
            raise httpx.ReadTimeout("offline timeout", request=request)
        if rank == 1:
            return httpx.Response(503)
        if rank == 2:
            raise RuntimeError("offline failure")
        return httpx.Response(200, json={"exit_code": 0})

    with _executor(kind, 4, command) as executor:
        result = executor.exec_in_workload("workload", ["diagnose"], timeout_s=600)
    assert [entry["rank"] for entry in result["ranks"]] == list(range(4))
    assert all(entry["result"]["observed"] is False for entry in result["ranks"][:3])
    assert result["ranks"][3]["result"]["exit_code"] == 0


@pytest.mark.parametrize("kind", ["slice", "gang"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_cancellation_propagates_and_drains_started_threads(kind, error_type):
    original = error_type("offline cancellation")
    entered = threading.Barrier(2, timeout=0.25)
    finished = threading.Event()

    def command(request):
        entered.wait()
        if _rank(request) == 0:
            raise original
        finished.set()
        return httpx.Response(200, json={"exit_code": 0})

    with _executor(kind, 2, command) as executor:
        with pytest.raises(error_type) as raised:
            executor.exec_in_workload("workload", ["diagnose"], timeout_s=600)
    assert raised.value is original
    assert finished.is_set()
    assert not any(t.name.startswith("eyvan-exec") for t in threading.enumerate())


@pytest.mark.parametrize("kind", ["slice", "gang"])
@pytest.mark.parametrize("response_time", [600, 630, 631])
def test_response_margin_is_shared_and_late_success_is_not_manufactured(monkeypatch, kind, response_time):
    clock = _Clock()
    monkeypatch.setattr(native_slice, "monotonic", clock)
    entered = threading.Barrier(2, action=lambda: clock.advance_to(response_time), timeout=1)

    def command(request):
        entered.wait()
        return httpx.Response(200, json={"exit_code": 0})

    with _executor(kind, 2, command, owned=True) as executor:
        result = executor.exec_in_workload("workload", ["diagnose"], timeout_s=600)
    for entry in result["ranks"]:
        if response_time <= 630:
            assert entry["result"]["exit_code"] == 0
        else:
            assert entry["result"]["observed"] is False
            assert "deadline" in entry["result"]["detail"]
    assert not any(t.name.startswith("eyvan-exec") for t in threading.enumerate())


def test_one_host_keeps_exact_command_budget_and_result():
    # A gang requires at least two cells; the one-host contract is the slice path.
    kind = "slice"
    asked = []

    def command(request):
        asked.append(json.loads(request.content))
        return httpx.Response(200, json={"exit_code": 7, "stdout": "unchanged"})

    with _executor(kind, 1, command) as executor:
        result = executor.exec_in_workload("workload", ["diagnose"], timeout_s=600)
    assert asked == [{"command": ["diagnose"], "timeout_s": 600}]
    assert result["ranks"] == [
        {"rank": 0, "endpoint": "http://worker-000:8100", "result": {"exit_code": 7, "stdout": "unchanged"}}
    ]


def test_exact_slice_rank_contacts_no_other_host_and_keeps_budget():
    asked = []

    def command(request):
        asked.append((_rank(request), json.loads(request.content)))
        return httpx.Response(200, json={"exit_code": 0})

    with _executor("slice", 40, command) as executor:
        result = executor.exec_in_workload_rank("workload", ["diagnose"], timeout_s=600, runtime_rank=37)
        for invalid in [True, -1, 40, "0"]:
            with pytest.raises(ValueError, match="runtime rank"):
                executor.exec_in_workload("workload", ["diagnose"], runtime_rank=invalid)
    assert asked == [(37, {"command": ["diagnose"], "timeout_s": 600})]
    assert [entry["rank"] for entry in result["ranks"]] == [37]
