# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cold-gang initialization: who owns the deadline, and what a lost gang says.

A gang that never formed and a workload that failed are the same terminal state
and used to carry the same terminal message, while calling for opposite operator
actions -- lengthen the window and resubmit, or go and read the user's
traceback. These exercise the two halves that make them distinguishable: the
deadline Eyvan injects into every rank, and the reason it reads back from the
hosts afterwards.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CellObservation, ExecutionWorkloadStatus, MutationContext
from eyvan.config import EyvanSettings
from eyvan.worker.runtime._multihost_site import (
    FATAL_INITIALIZATION_ENV,
    GANG_INITIALIZATION_DEADLINE_EXIT_CODE,
    GANG_INITIALIZATION_ERROR_EXIT_CODE,
    GANG_NEVER_FORMED_REASON,
    HEARTBEAT_TIMEOUT_ENV,
    INITIALIZATION_TIMEOUT_ENV,
    GangInitializationPolicy,
    gang_formation_failure_reason,
)
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_slice import NativeSliceExecutor
from eyvan.worker.runtime.registry import ExecutionDriverResolver

_GANG_ENDPOINTS = (
    "http://10.12.0.10:8100",
    "http://10.12.0.11:8100",
    "http://10.12.0.20:8100",
    "http://10.12.0.21:8100",
)
_SLICE_ENDPOINTS = ("http://10.12.0.10:8100", "http://10.12.0.11:8100")


def _context(*, key: str = "submit-1", generation: int = 1, token: str = "fence-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _client(endpoint: str, handler: tp.Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url=endpoint, transport=httpx.MockTransport(handler))


def _json_object(value: JsonValue) -> JsonObject:
    assert is_json_object(value)
    return value


def _payload(request: httpx.Request) -> JsonObject:
    value = json.loads(request.content)
    assert is_json_object(value)
    return value


def _spec(*, environment: dict[str, str] | None = None) -> JsonObject:
    document: JsonObject = {
        "entrypoint": "python train.py",
        "submission_id": "gang-job",
        "environment": dict(environment or {}),
        "metadata": {"eyvan_workload_id": "gang-job"},
    }
    return document


def _members() -> tuple[NativeGangMember, NativeGangMember]:
    return (
        NativeGangMember(
            cell_id="cell-a",
            endpoints=(_GANG_ENDPOINTS[0], _GANG_ENDPOINTS[1]),
            secret="worker-secret-a",
            context=_context(key="submit-a", token="fence-a"),
        ),
        NativeGangMember(
            cell_id="cell-b",
            endpoints=(_GANG_ENDPOINTS[2], _GANG_ENDPOINTS[3]),
            secret="worker-secret-b",
            context=_context(key="submit-b", token="fence-b"),
        ),
    )


def _submitted_environments(
    endpoints: tp.Sequence[str],
    submit: tp.Callable[[tp.Sequence[httpx.Client]], None],
) -> list[dict[str, str]]:
    """Run one submission against mock workers and return each rank's environment."""

    environments: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        spec = _json_object(_payload(request)["spec"])
        environment = _json_object(spec["environment"])
        environments.append({name: value for name, value in environment.items() if isinstance(value, str)})
        return httpx.Response(200, json={"remote_id": "gang-job"})

    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        submit(clients)
    finally:
        for client in clients:
            client.close()
    return environments


def _terminal_worker(
    *,
    exit_status: int | None,
    message: str = "systemd service exited with result exit-code",
    process_status_error: bool = False,
) -> tp.Callable[[httpx.Request], httpx.Response]:
    """Answer as a host whose workload has failed, with the exit status given."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/process-status"):
            if process_status_error:
                raise httpx.ConnectError("host disappeared", request=request)
            if exit_status is None:
                return httpx.Response(
                    200,
                    json={"remote_id": "gang-job", "observed": False, "detail": "systemd unit is absent"},
                )
            return httpx.Response(
                200,
                json={
                    "remote_id": "gang-job",
                    "unit_name": "eyvan-gang-job.service",
                    "observed": True,
                    "active_state": "failed",
                    "result": "exit-code",
                    "exec_main_status": exit_status,
                },
            )
        return httpx.Response(
            200,
            json={"remote_id": "gang-job", "status": ExecutionWorkloadStatus.FAILED.value, "message": message},
        )

    return handler


def _gang_status(
    handlers: tp.Sequence[tp.Callable[[httpx.Request], httpx.Response]],
    *,
    initialization: GangInitializationPolicy,
) -> JsonObject:
    """Observe one gang through freshly built clients and a fresh executor."""

    clients = [_client(endpoint, handler) for endpoint, handler in zip(_GANG_ENDPOINTS, handlers, strict=True)]
    try:
        executor = NativeGangExecutor(_members(), clients=clients, initialization=initialization)
        return dict(executor.status("gang-job"))
    finally:
        for client in clients:
            client.close()


def test_the_gang_deadline_reaches_every_rank_of_a_multi_cell_gang() -> None:
    """The deadline is only real if it is in the environment each host launches."""

    policy = GangInitializationPolicy(initialization_timeout_s=1800, heartbeat_timeout_s=100, fatal=True)
    environments = _submitted_environments(
        _GANG_ENDPOINTS,
        lambda clients: NativeGangExecutor(_members(), clients=clients, initialization=policy).submit(_spec()),
    )

    assert len(environments) == len(_GANG_ENDPOINTS)
    for environment in environments:
        assert environment[INITIALIZATION_TIMEOUT_ENV] == "1800"
        assert environment[HEARTBEAT_TIMEOUT_ENV] == "100"
        assert environment[FATAL_INITIALIZATION_ENV] == "1"


def test_the_gang_deadline_reaches_every_host_of_one_slice() -> None:
    """A slice is one cell, but its hosts join the same barrier a gang does."""

    policy = GangInitializationPolicy(initialization_timeout_s=1200, heartbeat_timeout_s=900)
    environments = _submitted_environments(
        _SLICE_ENDPOINTS,
        lambda clients: NativeSliceExecutor(
            _SLICE_ENDPOINTS,
            secret="worker-secret",
            clients=list(clients),
            initialization=policy,
        ).submit(_spec(), _context()),
    )

    # Fatality is on by default: a rank whose gang never formed must not run on
    # as an unwitting single-process job.
    assert len(environments) == len(_SLICE_ENDPOINTS)
    for environment in environments:
        assert environment[INITIALIZATION_TIMEOUT_ENV] == "1200"
        assert environment[HEARTBEAT_TIMEOUT_ENV] == "900"
        assert environment[FATAL_INITIALIZATION_ENV] == "1"


def test_a_caller_cannot_shorten_the_gang_deadline_through_its_own_environment() -> None:
    """A one-second barrier submitted by a caller would lose every gang it touched.

    Eyvan owns this variable for the same reason it owns the topology: the
    control plane is the thing that knows what the fleet's cold start costs, and
    a submitter that gets it wrong takes down hosts it does not own.
    """

    hostile = {
        INITIALIZATION_TIMEOUT_ENV: "1",
        HEARTBEAT_TIMEOUT_ENV: "1",
        FATAL_INITIALIZATION_ENV: "1",
        "USER_FLAG": "kept",
    }
    policy = GangInitializationPolicy(initialization_timeout_s=900, heartbeat_timeout_s=600)

    gang = _submitted_environments(
        _GANG_ENDPOINTS,
        lambda clients: NativeGangExecutor(_members(), clients=clients, initialization=policy).submit(
            _spec(environment=dict(hostile))
        ),
    )
    slice_ = _submitted_environments(
        _SLICE_ENDPOINTS,
        lambda clients: NativeSliceExecutor(
            _SLICE_ENDPOINTS,
            secret="worker-secret",
            clients=list(clients),
            initialization=policy,
        ).submit(_spec(environment=dict(hostile)), _context()),
    )

    for environment in (*gang, *slice_):
        assert environment[INITIALIZATION_TIMEOUT_ENV] == "900"
        assert environment[HEARTBEAT_TIMEOUT_ENV] == "600"
        assert environment[FATAL_INITIALIZATION_ENV] == "1"
        assert environment["USER_FLAG"] == "kept"


def test_a_gang_that_never_formed_reports_a_distinct_terminal_reason() -> None:
    """Every rank waited the deadline out, so no rank ever ran the workload."""

    policy = GangInitializationPolicy(initialization_timeout_s=1800, fatal=True)
    report = _gang_status(
        [_terminal_worker(exit_status=GANG_INITIALIZATION_DEADLINE_EXIT_CODE) for _ in _GANG_ENDPOINTS],
        initialization=policy,
    )

    assert report["status"] is ExecutionWorkloadStatus.FAILED
    message = report["message"]
    assert isinstance(message, str)
    assert message.startswith(f"{GANG_NEVER_FORMED_REASON}:")
    assert "rank(s) 0, 1, 2, 3 waited the full 1800s initialization deadline" in message
    # The per-rank detail the operator still needs is kept, not replaced.
    assert "rank 0=FAILED: systemd service exited with result exit-code" in message


def test_a_workload_failure_is_not_relabelled_as_a_lost_gang() -> None:
    """The distinction is worthless if an ordinary crash also claims it."""

    policy = GangInitializationPolicy(initialization_timeout_s=1800, fatal=True)
    report = _gang_status(
        [_terminal_worker(exit_status=1) for _ in _GANG_ENDPOINTS],
        initialization=policy,
    )

    assert report["status"] is ExecutionWorkloadStatus.FAILED
    message = report["message"]
    assert isinstance(message, str)
    assert GANG_NEVER_FORMED_REASON not in message
    assert message.startswith("rank 0=FAILED")


def test_one_rank_that_never_joined_names_the_whole_gang_as_never_formed() -> None:
    """Fate sharing: the peers died of its absence, not of their own fault.

    Ranks that were already registered are killed by the coordination service
    when one task never arrives, so their exit statuses say nothing useful. The
    single rank that reports why is the whole diagnosis.
    """

    policy = GangInitializationPolicy(initialization_timeout_s=600, fatal=True)
    report = _gang_status(
        [
            _terminal_worker(exit_status=GANG_INITIALIZATION_ERROR_EXIT_CODE),
            _terminal_worker(exit_status=1),
            _terminal_worker(exit_status=1),
            _terminal_worker(exit_status=None),
        ],
        initialization=policy,
    )

    message = report["message"]
    assert isinstance(message, str)
    assert message.startswith(f"{GANG_NEVER_FORMED_REASON}:")
    assert "rank(s) 0 failed to join before the 600s deadline" in message


def test_an_unreadable_host_does_not_become_a_verdict_about_the_gang() -> None:
    """A host nobody can reach is unknown, and unknown is not "the gang formed"."""

    policy = GangInitializationPolicy(initialization_timeout_s=600, fatal=True)
    report = _gang_status(
        [_terminal_worker(exit_status=None, process_status_error=True) for _ in _GANG_ENDPOINTS],
        initialization=policy,
    )

    message = report["message"]
    assert isinstance(message, str)
    assert GANG_NEVER_FORMED_REASON not in message
    assert message.startswith("rank 0=FAILED")


def test_the_lost_gang_reason_is_identical_after_a_controller_restart() -> None:
    """It is re-derived from the hosts, so nothing about it lives in memory.

    A restarted controller keeps no executor, no timer and no cached
    observation. If the reason came from any of those, a restart mid-failure
    would settle the same workload with a different account of why it died.
    """

    policy = GangInitializationPolicy(initialization_timeout_s=1800, fatal=True)
    handlers = [_terminal_worker(exit_status=GANG_INITIALIZATION_DEADLINE_EXIT_CODE) for _ in _GANG_ENDPOINTS]

    before = _gang_status(handlers, initialization=policy)
    after = _gang_status(handlers, initialization=policy)

    assert before["message"] == after["message"]
    assert before["status"] == after["status"]
    message = before["message"]
    assert isinstance(message, str)
    assert message.startswith(f"{GANG_NEVER_FORMED_REASON}:")


def test_a_nonfatal_gang_asks_no_host_for_an_exit_status() -> None:
    """With no fatal hook no rank can carry the evidence, so nothing is asked for.

    The cost argument survives the fatal-by-default flip: an operator who turns
    fatality off gives up the per-rank evidence, and the controller must not
    pay one request per host to learn nothing.
    """

    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return _terminal_worker(exit_status=GANG_INITIALIZATION_DEADLINE_EXIT_CODE)(request)

    report = _gang_status(
        [handler for _ in _GANG_ENDPOINTS],
        initialization=GangInitializationPolicy(fatal=False),
    )

    message = report["message"]
    assert isinstance(message, str)
    assert GANG_NEVER_FORMED_REASON not in message
    assert [path for path in paths if path.endswith("/process-status")] == []


def test_a_slice_that_never_formed_reports_the_same_distinct_reason() -> None:
    """One cell, several hosts, the same barrier and the same operator question."""

    policy = GangInitializationPolicy(initialization_timeout_s=900, fatal=True)
    clients = [
        _client(endpoint, _terminal_worker(exit_status=GANG_INITIALIZATION_DEADLINE_EXIT_CODE))
        for endpoint in _SLICE_ENDPOINTS
    ]
    try:
        executor = NativeSliceExecutor(
            _SLICE_ENDPOINTS,
            secret="worker-secret",
            clients=clients,
            initialization=policy,
        )
        report = executor.status("gang-job")
    finally:
        for client in clients:
            client.close()

    assert report["status"] is ExecutionWorkloadStatus.FAILED
    assert report["message"].startswith(f"{GANG_NEVER_FORMED_REASON}:")
    assert "rank(s) 0, 1 waited the full 900s initialization deadline" in report["message"]


def test_a_single_host_workload_carries_no_gang_deadline() -> None:
    """One host forms no gang, so nothing about gangs may reach its environment.

    The single-cell path is a different executor entirely, and this is the
    assertion that keeps it that way: an ordinary job must not inherit a
    coordination deadline, a fatal hook, or the multihost marker that turns them
    on.
    """

    submitted: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        submitted.append(_payload(request))
        return httpx.Response(200, json={"remote_id": "single-job"})

    client = _client("http://10.12.0.10:8100", handler)
    try:
        executor = NativeRemoteExecutor("http://10.12.0.10:8100", secret="worker-secret", client=client)
        executor.submit({"submission_id": "single-job", "environment": {"USER_FLAG": "kept"}}, _context())
    finally:
        client.close()

    environment = _json_object(_json_object(submitted[0]["spec"])["environment"])
    assert environment == {"USER_FLAG": "kept"}


def test_the_configured_deadline_reaches_the_executor_the_agent_builds(tmp_path) -> None:
    """A setting that reaches no executor is a knob wired to nothing."""

    settings = EyvanSettings(
        gang_init_deadline_s=1800,
        gang_heartbeat_timeout_s=100,
        gang_init_fatal=True,
    )
    resolver = ExecutionDriverResolver(
        "eyvan-native",
        state_dir=tmp_path,
        native_worker_token="agent-global-secret",
        gang_initialization=settings.gang_initialization_policy(),
    )
    observation = CellObservation(
        "cell-a",
        CapacityStatus.HEALTHY,
        1,
        metadata={
            "management_uid": "cell-uid",
            "provider_generation": 1,
            "native_worker_endpoints": list(_SLICE_ENDPOINTS),
        },
    )
    try:
        driver = resolver.resolve("cell-a", observation)

        assert isinstance(driver, NativeSliceExecutor)
        assert driver.initialization == GangInitializationPolicy(
            initialization_timeout_s=1800,
            heartbeat_timeout_s=100,
            fatal=True,
        )
    finally:
        resolver.close()


def test_an_unconfigured_deployment_keeps_the_deadline_it_ships_with() -> None:
    """Rolling upgrade: nothing changes for a fleet that configures nothing."""

    assert EyvanSettings().gang_initialization_policy() == GangInitializationPolicy()


def test_the_deadline_is_configurable_without_rebuilding_a_wheel(monkeypatch) -> None:
    """Iris needed 1800 on v5p-64. That must be a setting, not a release."""

    monkeypatch.setenv("EYVAN_GANG_INIT_DEADLINE_S", "1800")
    monkeypatch.setenv("EYVAN_GANG_HEARTBEAT_TIMEOUT_S", "1200")
    monkeypatch.setenv("EYVAN_GANG_INIT_FATAL", "true")

    policy = EyvanSettings.from_env().gang_initialization_policy()

    assert policy.environment() == {
        INITIALIZATION_TIMEOUT_ENV: "1800",
        HEARTBEAT_TIMEOUT_ENV: "1200",
        FATAL_INITIALIZATION_ENV: "1",
    }


def test_a_deadline_of_zero_is_refused_where_it_is_configured() -> None:
    """Zero means "abort every gang on arrival", which no operator ever means."""

    with pytest.raises(ValueError, match="positive"):
        EyvanSettings(gang_init_deadline_s=0).gang_initialization_policy()


def test_the_terminal_reason_names_both_kinds_of_rank_it_found() -> None:
    """The two kinds carry different instructions, so the reason keeps them apart."""

    reason = gang_formation_failure_reason(deadline_s=1800, deadline_ranks=(3, 1), error_ranks=(0,))

    assert reason.startswith(f"{GANG_NEVER_FORMED_REASON}:")
    assert "rank(s) 1, 3 waited the full 1800s initialization deadline" in reason
    assert "rank(s) 0 failed to join before the 1800s deadline" in reason
