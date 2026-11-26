# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Follower repairs exercised against production archive windows and CLI reads."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from click.testing import CliRunner

from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
from eyvan.sdk import cli
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.sync import UNREACHABLE_GRACE_S
from eyvan.sdk.client.transport import EyvanUnavailableError, SyncApiTransport
from eyvan.sdk.commands.http import EyvanApiError, EyvanClient
from eyvan.sdk.follow import WorkloadFollower
from eyvan.sdk.job_receipt import JobTerminalOutput
from eyvan.server.controller.observation import _ArchiveReplay, _ControllerObservation, _log_page_document
from eyvan.worker.runtime.log_records import LogRecord

WORKLOAD_ID = "wl_" + "6" * 32


def _workload(state: WorkloadState = WorkloadState.SUCCEEDED) -> WorkloadRecord:
    return WorkloadRecord(
        id=WORKLOAD_ID,
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=WorkloadDefinition(name="archive", execution=CommandExecution(command="echo done")),
        state=state,
        remote_id="retained-execution",
    )


class _ArchivePlane:
    """No server/storage I/O: real immutable archive pagination and wire projection."""

    def __init__(self, records: tuple[LogRecord, ...], *, states=None, truncated=(), detail=""):
        self.workload = _workload()
        self.states = list(states or [WorkloadState.SUCCEEDED])
        self.replay = _ArchiveReplay(ranks={0: records}, truncated=frozenset(truncated), detail=detail)
        self.observation = _ControllerObservation.__new__(_ControllerObservation)
        self.requests: list[tuple[str, str]] = []
        self.cursors: list[str] = []

    def request(self, method, path, *, params=None, **kwargs):
        self.requests.append((method, path))
        if not path.endswith("/log-page"):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return self.workload.model_copy(update={"state": state}).model_dump(mode="json")
        after = str((params or {}).get("after", ""))
        self.cursors.append(after)
        changed, positions = self.observation._decoded_archive_cursor(after, self.workload)
        page = self.observation._archive_window(
            self.workload, self.replay, positions=positions, position_gap="", limit=(params or {}).get("limit", 200)
        )
        return _log_page_document(
            page, attempt=self.workload.attempt, attempt_changed=changed, next_cursor=page.next_cursor
        )


def _records(count: int, *, old: bool = False, message: str | None = None) -> tuple[LogRecord, ...]:
    at = datetime.now(UTC) - timedelta(hours=1) if old else datetime.now(UTC)
    return tuple(
        LogRecord(message=message or f"line-{index}", timestamp=at, cursor=f"journal-{index}") for index in range(count)
    )


def _follower(plane, stdout, stderr, **kwargs):
    return WorkloadFollower(
        client=plane, output=JobTerminalOutput(writer=stderr.append), write=stdout.append, sleep=lambda _: None, **kwargs
    )


def _since(monkeypatch, plane):
    monkeypatch.setattr(cli, "EyvanClient", lambda *args, **kwargs: plane)
    return CliRunner().invoke(cli.main, ["--user", "alice", "workload", "logs", WORKLOAD_ID, "--since", "10m"])


@pytest.mark.parametrize("count", [25_400, 25_801])
def test_terminal_archive_is_exhausted_before_success(count):
    plane = _ArchivePlane(_records(count))
    stdout, stderr = [], []
    outcome = _follower(plane, stdout, stderr).follow(WORKLOAD_ID)
    assert outcome.exit_code == 0
    assert stdout == [f"line-{index}" for index in range(count)]
    assert "succeeded" in "\n".join(stderr)


def test_since_fetches_past_the_live_poll_budget(monkeypatch):
    recent = LogRecord(message="RECENT RESULT", timestamp=datetime.now(UTC), cursor="recent")
    plane = _ArchivePlane((*_records(12_800, old=True), recent), states=[WorkloadState.RUNNING])
    result = _since(monkeypatch, plane)
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "RECENT RESULT"
    assert all(path.endswith("/log-page") for _, path in plane.requests)


def test_live_following_retains_the_per_poll_fairness_budget():
    plane = _ArchivePlane(_records(25_801), states=[WorkloadState.RUNNING, WorkloadState.SUCCEEDED])
    stdout, stderr, pages_at_sleep = [], [], []
    follower = _follower(plane, stdout, stderr)
    follower.sleep = lambda _: pages_at_sleep.append(len(plane.cursors))
    assert follower.follow(WORKLOAD_ID).exit_code == 0
    assert pages_at_sleep == [64]
    assert stdout[-1] == "line-25800"
    assert len(stdout) == 25_801


@pytest.mark.parametrize("one_shot", [False, True])
def test_filtered_raw_pages_do_not_hide_later_user_output(monkeypatch, one_shot):
    chatter = _records(200 if one_shot else 400, message="eyvan-gang-init-joined rank=unknown")
    plane = _ArchivePlane((*chatter, LogRecord(message="FINAL RESULT", timestamp=datetime.now(UTC), cursor="final")))
    if one_shot:
        result = _since(monkeypatch, plane)
        assert result.exit_code == 0, result.output
        assert result.stdout.strip() == "FINAL RESULT"
    else:
        stdout, stderr = [], []
        assert _follower(plane, stdout, stderr).follow(WORKLOAD_ID).exit_code == 0
        assert stdout == ["FINAL RESULT"]
    assert len(plane.cursors) >= (3 if one_shot else 4)


@pytest.mark.parametrize("one_shot", [False, True])
def test_missing_replica_artifact_access_is_reported_on_stderr(monkeypatch, one_shot):
    import asyncio

    class MissingReplica(_ArchivePlane):
        def request(self, method, path, **kwargs):
            if not path.endswith("/log-page"):
                return super().request(method, path, **kwargs)
            self.observation.artifact_access = None
            page = asyncio.run(self.observation._archived_log_page(self.workload))
            return _log_page_document(page, attempt=self.workload.attempt)

    plane = MissingReplica(())
    if one_shot:
        result = _since(monkeypatch, plane)
        assert result.exit_code == 0, result.output
        assert not result.stdout
        assert "logs are unavailable" in result.stderr
    else:
        stdout, stderr = [], []
        follower = _follower(plane, stdout, stderr)
        assert follower.follow(WORKLOAD_ID).exit_code == 0
        follower._drain(WORKLOAD_ID)
        assert not stdout
        assert sum("logs are unavailable" in line for line in stderr) == 1


@pytest.mark.parametrize("one_shot", [False, True])
def test_partial_archive_notes_truncated_ranks_without_echoing_untrusted_detail(monkeypatch, one_shot):
    plane = _ArchivePlane(_records(401), truncated=(1,), detail="rank missing: token=DO-NOT-PRINT\x1b[31m")
    if one_shot:
        result = _since(monkeypatch, plane)
        assert result.exit_code == 0, result.output
        assert "line-400" in result.stdout
        stdout, stderr = result.stdout, result.stderr
    else:
        written, notes = [], []
        assert _follower(plane, written, notes).follow(WORKLOAD_ID).exit_code == 0
        stdout, stderr = "\n".join(written), "\n".join(notes)
    assert "logs are incomplete for ranks: 1" in stderr
    assert stderr.count("logs are incomplete") == 1
    assert "DO-NOT-PRINT" not in stderr
    assert "logs are incomplete" not in stdout


@pytest.mark.parametrize("endpoint", ["state", "log-page"])
def test_follow_recovers_injected_connection_loss_through_real_cli_adapter(endpoint):
    plane = _ArchivePlane(_records(401), states=[WorkloadState.RUNNING, WorkloadState.SUCCEEDED])
    endpoint_calls = 0
    requests = []

    def handler(request):
        nonlocal endpoint_calls
        requests.append(request)
        target = request.url.path.endswith("/log-page") == (endpoint == "log-page")
        if target:
            endpoint_calls += 1
            # Lose the first state read, or a later log read after printing page 1.
            if endpoint_calls == (1 if endpoint == "state" else 2):
                raise httpx.ConnectError("injected local restart", request=request)
        params = {"after": request.url.params.get("after", ""), "limit": 200}
        return httpx.Response(200, json=plane.request(request.method, request.url.path, params=params))

    with httpx.Client(base_url="http://127.0.0.1:8080", transport=httpx.MockTransport(handler)) as http:
        adapter = EyvanClient.__new__(EyvanClient)
        adapter._transport = SyncApiTransport("http://127.0.0.1:8080", client=http)
        stdout, stderr = [], []
        assert _follower(adapter, stdout, stderr).follow(WORKLOAD_ID).exit_code == 0
    assert stdout == [f"line-{index}" for index in range(401)]
    assert all(request.method == "GET" for request in requests)
    if endpoint == "log-page":
        pages = [request for request in requests if request.url.path.endswith("/log-page")]
        assert pages[1].url.params["after"] == pages[2].url.params["after"] != ""


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize("endpoint", ["state", "log-page"])
def test_follow_outage_grace_is_bounded_even_when_the_other_endpoint_answers(monkeypatch, endpoint):
    clock = _Clock()
    monkeypatch.setattr("eyvan.sdk.follow.time.monotonic", lambda: clock.now)

    class Gone(_ArchivePlane):
        def request(self, method, path, **kwargs):
            self.requests.append((method, path))
            if path.endswith("/log-page") == (endpoint == "log-page"):
                raise EyvanUnavailableError(ControlPlaneEndpoint("http://127.0.0.1:8080"), method)
            return super().request(method, path, **kwargs)

    plane = Gone(())
    follower = _follower(plane, [], [], poll_interval=45)
    follower.sleep = clock.sleep
    with pytest.raises(EyvanUnavailableError):
        follower.follow(WORKLOAD_ID)
    assert clock.now == UNREACHABLE_GRACE_S
    assert len(clock.sleeps) == 2
    assert all(method == "GET" for method, _ in plane.requests)


@pytest.mark.parametrize("endpoint", ["state", "log-page"])
def test_follow_success_resets_outage_grace(monkeypatch, endpoint):
    clock = _Clock()
    monkeypatch.setattr("eyvan.sdk.follow.time.monotonic", lambda: clock.now)

    class Blips(_ArchivePlane):
        calls = 0

        def request(self, method, path, **kwargs):
            if path.endswith("/log-page") == (endpoint == "log-page"):
                self.calls += 1
                if self.calls in {1, 3, 4}:
                    raise EyvanUnavailableError(ControlPlaneEndpoint("http://127.0.0.1:8080"), method)
            return super().request(method, path, **kwargs)

    plane = Blips(_records(401), states=[WorkloadState.RUNNING, WorkloadState.SUCCEEDED])
    stdout, stderr = [], []
    follower = _follower(plane, stdout, stderr, poll_interval=45)
    follower.sleep = clock.sleep
    assert follower.follow(WORKLOAD_ID).exit_code == 0
    assert clock.now >= UNREACHABLE_GRACE_S
    assert stdout == [f"line-{index}" for index in range(401)]


@pytest.mark.parametrize("endpoint", ["state", "log-page"])
@pytest.mark.parametrize("status", [401, 403, 404])
def test_follow_never_forgives_authorization_or_missing_workloads(endpoint, status):
    plane = _ArchivePlane(())
    refusals = []

    def handler(request):
        if request.url.path.endswith("/log-page") == (endpoint == "log-page"):
            refusals.append(request)
            return httpx.Response(status, json={"detail": "refused"})
        return httpx.Response(200, json=plane.request(request.method, request.url.path))

    with httpx.Client(base_url="http://127.0.0.1:8080", transport=httpx.MockTransport(handler)) as http:
        adapter = EyvanClient.__new__(EyvanClient)
        adapter._transport = SyncApiTransport("http://127.0.0.1:8080", client=http)
        with pytest.raises(EyvanApiError) as caught:
            _follower(adapter, [], []).follow(WORKLOAD_ID)
    assert caught.value.status_code == status
    assert len(refusals) == 1


@pytest.mark.parametrize("newly_terminal", [False, True])
@pytest.mark.parametrize("count", [3_000, 3_400])
def test_related_callable_peer_repair_exhausts_actual_archive_before_wait_returns(monkeypatch, newly_terminal, count):
    from eyvan.sdk.client.callables import _CallableProgressRenderer
    from eyvan.sdk.client.sync import EyvanClient as TypedClient

    states = [WorkloadState.RUNNING, WorkloadState.SUCCEEDED] if newly_terminal else [WorkloadState.SUCCEEDED]
    plane = _ArchivePlane(_records(count), states=states)
    monkeypatch.setattr("eyvan.sdk.client.sync.time.sleep", lambda _: None)

    def handler(request):
        if request.url.path.endswith("/progress"):
            return httpx.Response(404, json={"detail": "legacy progress endpoint"})
        params = {"after": request.url.params.get("after", ""), "limit": 200}
        return httpx.Response(200, json=plane.request(request.method, request.url.path, params=params))

    transcript = []
    with httpx.Client(base_url="http://127.0.0.1:8080", transport=httpx.MockTransport(handler)) as http:
        client = TypedClient(client=http)
        renderer = _CallableProgressRenderer(client, JobTerminalOutput(writer=transcript.append), None)
        outcome = client.wait_workload(WORKLOAD_ID, on_update=renderer)
    assert outcome.state is WorkloadState.SUCCEEDED
    assert [line for line in transcript if line.startswith("line-")] == [f"line-{index}" for index in range(count)]
    verdict = next(index for index, line in enumerate(transcript) if "succeeded" in line)
    assert transcript.index(f"line-{count - 1}") < verdict


@pytest.mark.parametrize("cyclic", [False, True])
def test_exhaustive_drain_stops_on_nonadvancing_or_cyclic_cursors(cyclic):
    class Stalled(_ArchivePlane):
        def request(self, method, path, **kwargs):
            if not path.endswith("/log-page"):
                return super().request(method, path, **kwargs)
            after = (kwargs.get("params") or {}).get("after", "")
            self.cursors.append(after)
            return {
                "records": [{"message": "one page", "rank": 0}],
                "next_cursor": "first" if not after or (cyclic and after == "second") else "second" if cyclic else after,
            }

    plane = Stalled(())
    stdout, stderr = [], []
    follower = _follower(plane, stdout, stderr)
    assert follower._drain(WORKLOAD_ID) is False
    assert len(plane.cursors) == (3 if cyclic else 2)
    assert len(stdout) == len(plane.cursors)
    assert sum("cursor did not advance" in line for line in stderr) == 1


def test_drain_distinguishes_live_budget_from_raw_archive_exhaustion():
    plane = _ArchivePlane(_records(401))
    stdout, stderr = [], []
    follower = _follower(plane, stdout, stderr, max_pages_per_poll=1)
    assert follower._drain(WORKLOAD_ID, bounded=True) is False
    assert len(stdout) == 200
    assert follower._drain(WORKLOAD_ID) is True
    assert len(stdout) == 401
