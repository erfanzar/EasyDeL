# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Native-worker exec HTTP budgets, measured without network or timed sleeps."""

from __future__ import annotations

import json

import httpx
import pytest

from eyvan.capacity.drivers.base import MutationContext
from eyvan.worker.agent.models import ExecRequest
from eyvan.worker.runtime import native_remote, native_slice
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_slice import NativeSliceExecutor


@pytest.fixture
def owned_transport(monkeypatch):
    """Keep the production-owned client configuration, replacing only its I/O."""

    client_class = httpx.Client
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        assert request.url.path == "/api/v1/workloads/remote-1/exec"
        assert request.headers["Authorization"] == "Bearer worker-secret"
        assert body["command"] == ["diagnostic", "--dump"]
        # Model a silent command finishing inside its requested execution bound.
        # This is transport-configuration evidence, not a real delayed command.
        silent_s = body["timeout_s"] * 0.75
        if request.extensions["timeout"]["read"] <= silent_s:
            raise httpx.ReadTimeout("HTTP budget expired before diagnostic completion", request=request)
        return httpx.Response(200, json={"exit_code": 0, "stdout": "diagnostic output", "stderr": ""})

    def create_client(**kwargs):
        return client_class(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(native_remote.httpx, "Client", create_client)
    return seen


@pytest.mark.parametrize("budget_s", [0.1, 30.0, 60.0, 600.0])
def test_owned_exec_transport_covers_valid_command_budget(owned_transport, budget_s: float) -> None:
    request = ExecRequest(command=("diagnostic", "--dump"), timeout_s=budget_s)
    executor = NativeRemoteExecutor("http://worker-0", secret="worker-secret")
    try:
        result = executor.exec_in_workload("remote-1", request.command, timeout_s=request.timeout_s)
    finally:
        executor.close()

    assert result == {"exit_code": 0, "stdout": "diagnostic output", "stderr": ""}
    assert len(owned_transport) == 1
    assert json.loads(owned_transport[0].content)["timeout_s"] == budget_s
    assert owned_transport[0].extensions["timeout"]["read"] == budget_s + 30.0


@pytest.mark.parametrize("topology", ["slice", "gang"])
@pytest.mark.parametrize("budget_s", [60.0, 600.0])
def test_owned_rank_exec_preserves_output_with_long_command_budget(
    owned_transport, monkeypatch, topology: str, budget_s: float
) -> None:
    # Every rank starts one second after the shared deadline was established.
    # A fresh full timeout per rank would incorrectly extend that deadline.
    first_tick = iter((1000.0,))
    monkeypatch.setattr(native_slice, "monotonic", lambda: next(first_tick, 1001.0))
    remaining_s = budget_s - 1.0
    if topology == "slice":
        executor = NativeSliceExecutor(("http://worker-0", "http://worker-1"), secret="worker-secret")
        rank_count = 2
    else:
        executor = NativeGangExecutor(
            tuple(
                NativeGangMember(
                    cell_id=f"cell-{cell}",
                    endpoints=(f"http://worker-{cell}-0", f"http://worker-{cell}-1"),
                    secret="worker-secret",
                    context=MutationContext(idempotency_key="exec-test", generation=1, fencing_token="fence"),
                )
                for cell in range(2)
            )
        )
        rank_count = 4
    try:
        result = executor.exec_in_workload("remote-1", ["diagnostic", "--dump"], timeout_s=budget_s)
    finally:
        executor.close()

    assert len(result["ranks"]) == rank_count
    assert [rank["rank"] for rank in result["ranks"]] == list(range(rank_count))
    assert all(
        rank["result"] == {"exit_code": 0, "stdout": "diagnostic output", "stderr": ""} for rank in result["ranks"]
    )
    assert len(owned_transport) == rank_count
    assert all(json.loads(request.content)["timeout_s"] == remaining_s for request in owned_transport)
    assert all(request.extensions["timeout"]["read"] == remaining_s + 30.0 for request in owned_transport)


def test_exec_preserves_caller_supplied_client_timeout_and_ownership() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"exit_code": 0, "stdout": "custom transport", "stderr": ""})

    with httpx.Client(base_url="http://custom-worker", timeout=7.0, transport=httpx.MockTransport(handle)) as client:
        executor = NativeRemoteExecutor("http://custom-worker", secret="worker-secret", client=client)
        result = executor.exec_in_workload("remote-1", ["diagnostic", "--dump"], timeout_s=60.0)
        executor.close()
        assert not client.is_closed

    assert result["stdout"] == "custom transport"
    assert len(seen) == 1
    assert seen[0].extensions["timeout"]["read"] == 7.0
    assert json.loads(seen[0].content)["timeout_s"] == 60.0
