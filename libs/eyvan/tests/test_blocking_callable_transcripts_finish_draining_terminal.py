# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Blocking callable transcripts must finish draining terminal log archives."""

from __future__ import annotations

import httpx
import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
from eyvan.sdk.client.callables import _CallableProgressRenderer, _SdkJobBackend
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.jobs import EyvanJob
from eyvan.sdk.client.sync import EyvanClient
from eyvan.sdk.client.transport import EyvanHTTPError, EyvanUnavailableError
from eyvan.sdk.job_receipt import JobTerminalOutput


def _workload(state: WorkloadState = WorkloadState.SUCCEEDED) -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-terminal-transcript",
        org_id="acme",
        project_id="models",
        spec=WorkloadDefinition(name="transcript", execution=CommandExecution(command="true")),
        state=state,
        remote_id="execution-1",
        created_by="tester",
    )


class _PageClient(EyvanClient):
    """Local archive pages beneath the real SDK blocking-wait implementation."""

    def __init__(self, pages: dict[str, JsonObject], *, error: Exception | None = None) -> None:
        self.pages = pages
        self.error = error
        self.reads: list[str] = []

    def get_workload(self, workload_id: str) -> WorkloadRecord:
        assert workload_id == "wl-terminal-transcript"
        return _workload()

    def get_workload_progress(self, workload_id: str):
        raise EyvanUnavailableError(ControlPlaneEndpoint("http://plane"), "GET")

    def workload_log_page(self, workload_id: str, *, after: str, limit: int = 200) -> JsonObject:
        assert workload_id == "wl-terminal-transcript"
        assert limit == 200
        self.reads.append(after)
        assert len(self.reads) <= 100, "terminal pagination must stop on non-progressing cursors"
        if self.error is not None:
            raise self.error
        return self.pages.get(after, {"records": [], "next_cursor": after})


def _result(client: _PageClient) -> list[str]:
    lines: list[str] = []
    backend = _SdkJobBackend(
        client,
        terminal_output=JobTerminalOutput(writer=lines.append),
        queue_admission_recorder=None,
    )
    job = EyvanJob(
        id="wl-terminal-transcript",
        result_uri="memory://result",
        backend=backend,
        result_reader=lambda uri: "decoded-result",
        progress_waiter=backend.wait_with_progress,
    )
    assert job.result() == "decoded-result"
    return lines


def _backlog() -> dict[str, JsonObject]:
    return {
        "" if start == 0 else str(start): {
            "records": [{"rank": 0, "message": f"user-line-{index}"} for index in range(start, start + 200)],
            "next_cursor": str(start + 200),
            "expected_ranks": 1,
        }
        for start in range(0, 3400, 200)
    }


def test_result_attached_after_completion_renders_the_entire_terminal_backlog() -> None:
    client = _PageClient(_backlog())

    lines = _result(client)

    user_lines = [line for line in lines if line.startswith("user-line-")]
    assert user_lines == [f"user-line-{index}" for index in range(3400)]
    assert lines.index("user-line-3399") < lines.index("  job completed")
    assert client.reads[-1] == "3400"  # Empty raw archive EOF, with its unchanged cursor.


@pytest.mark.parametrize("has_first_user_page", [False, True])
def test_a_filtered_raw_page_does_not_hide_later_terminal_user_output(has_first_user_page: bool) -> None:
    pages: dict[str, JsonObject] = {
        "before-chatter" if has_first_user_page else "": {
            "records": [{"rank": 0, "message": "eyvan-gang-init-joined rank=unknown"} for _ in range(200)],
            "next_cursor": "after-chatter",
        },
        "after-chatter": {
            "records": [{"rank": 0, "message": "the final user line"}],
            "next_cursor": "end",
        },
    }
    if has_first_user_page:
        pages[""] = {"records": [{"rank": 0, "message": "the first user line"}], "next_cursor": "before-chatter"}
    client = _PageClient(pages)

    lines = _result(client)

    assert "the final user line" in lines
    assert lines.index("the final user line") < lines.index("  job completed")
    assert not any("eyvan-gang-init-joined" in line for line in lines)
    assert client.reads[-1] == "end"


@pytest.mark.parametrize("state", [WorkloadState.FAILED, WorkloadState.CANCELLED])
def test_unsuccessful_terminal_verdict_also_follows_the_complete_transcript(state: WorkloadState) -> None:
    client = _PageClient(_backlog())
    lines: list[str] = []
    renderer = _CallableProgressRenderer(client, JobTerminalOutput(writer=lines.append), None)

    renderer(_workload(state))

    assert "user-line-3399" in lines
    verdict_index = next(index for index, line in enumerate(lines) if line.startswith("Eyvan"))
    assert lines.index("user-line-3399") < verdict_index


def test_live_polls_stay_bounded_but_terminal_poll_drains_the_tail() -> None:
    client = _PageClient(_backlog())
    lines: list[str] = []
    renderer = _CallableProgressRenderer(client, JobTerminalOutput(writer=lines.append), None)

    renderer(_workload(WorkloadState.RUNNING))

    assert len(client.reads) == 16
    assert "user-line-3199" in lines
    assert "user-line-3200" not in lines

    renderer(_workload())

    assert "user-line-3399" in lines
    assert lines.count("user-line-0") == 1
    assert lines.index("user-line-3399") < lines.index("  job completed")


def test_unchanged_nonempty_cursor_does_not_repeat_output() -> None:
    client = _PageClient(
        {
            "": {"records": [{"rank": 0, "message": "first"}], "next_cursor": "stuck"},
            "stuck": {"records": [{"rank": 0, "message": "last"}], "next_cursor": "stuck"},
        }
    )

    lines = _result(client)

    assert lines.count("last") == 1
    assert "  job completed" in lines
    assert client.reads == ["", "stuck"]


def test_terminal_cursor_cycle_does_not_replay_pages_forever() -> None:
    client = _PageClient(
        {
            "": {"records": [{"rank": 0, "message": "first"}], "next_cursor": "a"},
            "a": {"records": [{"rank": 0, "message": "second"}], "next_cursor": "b"},
            "b": {"records": [{"rank": 0, "message": "third"}], "next_cursor": "a"},
        }
    )

    lines = _result(client)

    assert [line for line in lines if line in {"first", "second", "third"}] == ["first", "second", "third"]
    assert "  job completed" in lines
    assert client.reads == ["", "a", "b"]


@pytest.mark.parametrize("unavailable", [False, True])
def test_terminal_log_read_errors_remain_best_effort(unavailable: bool) -> None:
    error = (
        EyvanUnavailableError(ControlPlaneEndpoint("http://plane"), "GET")
        if unavailable
        else EyvanHTTPError(
            response=httpx.Response(404, request=httpx.Request("GET", "http://plane/api")),
            detail="archive unavailable",
        )
    )

    lines = _result(_PageClient({}, error=error))

    assert "  job completed" in lines


def test_a_page_without_a_next_cursor_still_renders_its_output() -> None:
    client = _PageClient({"": {"records": [{"rank": 0, "message": "last"}]}})

    lines = _result(client)

    assert lines.index("last") < lines.index("  job completed")
    assert client.reads == [""]
