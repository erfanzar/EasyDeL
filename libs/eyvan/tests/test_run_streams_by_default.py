# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan run` attaches to the job it submitted.

Submitting used to print an id and stop, so the shortest way to see whether a
job worked was two more commands and a guess at when to run them. Ray streams
and `--no-wait` opts out; this is the same contract, plus an exit status that
makes `eyvan run ... && next-thing` mean what it looks like it means.
"""

from __future__ import annotations

import json
import typing as tp
from datetime import UTC, datetime, timedelta

from click.testing import CliRunner

from eyvan._internal.json_types import JsonValue
from eyvan.records import (
    CommandExecution,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.sdk import cli
from eyvan.sdk.follow import EXIT_CANCELLED, EXIT_DETACHED, EXIT_FAILED, EXIT_SUCCEEDED, WorkloadFollower
from eyvan.sdk.job_receipt import JobTerminalOutput

ORG = "acme"
PROJECT = "research"


def _record(state: WorkloadState, *, failure_reason: str | None = None) -> dict:
    return WorkloadRecord(
        id="wl-1",
        org_id=ORG,
        project_id=PROJECT,
        spec=WorkloadDefinition(name="t", execution=CommandExecution(command="python train.py")),
        state=state,
        failure_reason=failure_reason,
        created_by="alice",
    ).model_dump(mode="json")


def _page(
    lines: tp.Sequence[tuple[int, str]],
    *,
    cursor: str,
    expected_ranks: int | None = None,
    attempt_changed: bool = False,
    age_s: float = 0.0,
) -> dict:
    stamped = (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat()
    return {
        "attempt": 1,
        "attempt_changed": attempt_changed,
        "records": [
            {"message": message, "rank": rank, "stream": "stdout", "timestamp": stamped} for rank, message in lines
        ],
        "next_cursor": cursor,
        "unavailable": False,
        "detail": "",
        "truncated_ranks": [],
        "expected_ranks": expected_ranks,
    }


class _Plane:
    """A control plane that hands out scripted states and log pages in order."""

    def __init__(self, states: tp.Sequence[dict], pages: tp.Mapping[str, dict]) -> None:
        self.states = list(states)
        self.pages = dict(pages)
        self.seen_cursors: list[str] = []
        self.paths: list[str] = []

    def request(self, method: str, path: str, *, params=None, json=None, headers=None) -> JsonValue:
        del method, json, headers
        self.paths.append(path)
        if path.endswith("/log-page"):
            cursor = "" if params is None else str(params.get("after", ""))
            self.seen_cursors.append(cursor)
            return self.pages.get(cursor, _page((), cursor=cursor))
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def _follower(plane: _Plane, written: list[str]) -> WorkloadFollower:
    return WorkloadFollower(
        client=plane,
        output=JobTerminalOutput(writer=lambda line: written.append(line)),
        write=written.append,
        sleep=lambda _seconds: None,
    )


# --------------------------------------------------------------------------- the follower


def test_a_followed_job_prints_its_output_and_exits_zero() -> None:
    """The point: one command submits, shows the run, and ends when it does."""

    plane = _Plane(
        states=[_record(WorkloadState.RUNNING), _record(WorkloadState.SUCCEEDED)],
        pages={
            "": _page([(0, "epoch 0")], cursor="c1"),
            "c1": _page([(0, "epoch 1")], cursor="c2"),
        },
    )
    written: list[str] = []

    outcome = _follower(plane, written).follow("wl-1")

    assert outcome.state is WorkloadState.SUCCEEDED
    assert outcome.exit_code == EXIT_SUCCEEDED
    assert [line for line in written if line.startswith("epoch")] == ["epoch 0", "epoch 1"]


def test_a_failed_job_makes_the_command_fail() -> None:
    """`eyvan run ... && deploy` must not deploy after a failed training run."""

    plane = _Plane(states=[_record(WorkloadState.FAILED, failure_reason="exit status 1")], pages={})
    written: list[str] = []

    outcome = _follower(plane, written).follow("wl-1")

    assert outcome.exit_code == EXIT_FAILED
    assert any("exit status 1" in line for line in written)


def test_a_cancelled_job_is_distinguishable_from_a_failed_one() -> None:
    """Cancelled is a decision someone made; failed is the job's own verdict."""

    plane = _Plane(states=[_record(WorkloadState.CANCELLED)], pages={})

    assert _follower(plane, []).follow("wl-1").exit_code == EXIT_CANCELLED


def test_output_is_never_replayed_across_polls() -> None:
    """The cursor is what stops a follower re-printing the whole log every second."""

    plane = _Plane(
        states=[_record(WorkloadState.RUNNING), _record(WorkloadState.RUNNING), _record(WorkloadState.SUCCEEDED)],
        pages={"": _page([(0, "only once")], cursor="c1")},
    )
    written: list[str] = []

    _follower(plane, written).follow("wl-1")

    assert [line for line in written if line == "only once"] == ["only once"]
    assert plane.seen_cursors[0] == ""
    assert set(plane.seen_cursors[1:]) == {"c1"}


def test_a_multi_rank_job_attributes_every_line_to_its_rank() -> None:
    """Which rank said it is the question a merged stream cannot answer."""

    plane = _Plane(
        states=[_record(WorkloadState.SUCCEEDED)],
        pages={"": _page([(0, "rank zero"), (1, "rank one")], cursor="c1", expected_ranks=2)},
    )
    written: list[str] = []

    _follower(plane, written).follow("wl-1")

    # Ray-style attribution: a rank with no provenance reads "(rank N)".
    assert "(rank 0) rank zero" in written
    assert "(rank 1) rank one" in written


def test_a_single_rank_job_is_not_cluttered_with_prefixes() -> None:
    """A one-process job's output should look like the program's own."""

    plane = _Plane(
        states=[_record(WorkloadState.SUCCEEDED)],
        pages={"": _page([(0, "plain line")], cursor="c1", expected_ranks=1)},
    )
    written: list[str] = []

    _follower(plane, written).follow("wl-1")

    assert "plain line" in written


def test_a_retry_is_announced_rather_than_appended_silently() -> None:
    """Lines after a retry are a different run; presenting them as one is a lie."""

    plane = _Plane(
        states=[_record(WorkloadState.RUNNING), _record(WorkloadState.SUCCEEDED)],
        pages={
            "": _page([(0, "attempt one")], cursor="c1"),
            "c1": _page([(0, "attempt two")], cursor="c2", attempt_changed=True),
        },
    )
    written: list[str] = []

    _follower(plane, written).follow("wl-1")

    assert any("restarted" in line for line in written)


def test_the_tail_is_read_after_the_job_is_already_terminal() -> None:
    """Settlement archives the last lines before it publishes the terminal state.

    A follower that stopped the moment it saw SUCCEEDED would drop exactly the
    output explaining how the run ended.
    """

    plane = _Plane(
        states=[_record(WorkloadState.SUCCEEDED)],
        pages={
            "": _page([(0, "second to last")], cursor="c1"),
            "c1": _page([(0, "the final line")], cursor="c2"),
        },
    )
    written: list[str] = []

    _follower(plane, written).follow("wl-1")

    assert "the final line" in written


def test_interrupting_detaches_and_says_the_job_is_still_running() -> None:
    """Ctrl-C must not cancel. The job is durable and outlives its submitter."""

    class _Interrupting(_Plane):
        def request(self, method, path, **kwargs):
            raise KeyboardInterrupt

    written: list[str] = []
    outcome = _follower(_Interrupting([], {}), written).follow("wl-1")

    assert outcome.detached
    assert outcome.exit_code == EXIT_DETACHED
    assert any("still running" in line for line in written)
    assert any("eyvan workload logs wl-1 --follow" in line for line in written)


def test_an_unparseable_record_keeps_streaming_rather_than_ending_the_run() -> None:
    """A version skew must not be reported as a finished job."""

    class _Skewed(_Plane):
        def __init__(self) -> None:
            super().__init__(states=[], pages={"": _page([(0, "still here")], cursor="c1")})
            self.polls = 0

        def request(self, method, path, *, params=None, json=None, headers=None):
            if path.endswith("/log-page"):
                return super().request(method, path, params=params)
            self.polls += 1
            if self.polls > 2:
                return _record(WorkloadState.SUCCEEDED)
            return {"id": "wl-1", "unknown_field_from_a_newer_plane": True}

    written: list[str] = []
    outcome = _follower(_Skewed(), written).follow("wl-1")

    assert outcome.state is WorkloadState.SUCCEEDED
    assert "still here" in written


# --------------------------------------------------------------------------- the command


class _SubmittingPlane(_Plane):
    """A plane that also accepts the submission `eyvan run` makes."""

    def request(self, method, path, *, params=None, json=None, headers=None):
        if method == "POST" and "workloads" in path:
            return _record(WorkloadState.PENDING)
        return super().request(method, path, params=params, json=json, headers=headers)


def _invoke(monkeypatch, plane: _Plane, *args: str):
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: plane)
    monkeypatch.setattr("eyvan.sdk.follow.time.sleep", lambda _seconds: None)
    return CliRunner().invoke(cli.main, ["--user", "alice", "--org", ORG, "--project", PROJECT, "run", *args])


def test_run_streams_the_job_instead_of_printing_an_id(monkeypatch) -> None:
    """The ten minutes that decide whether anyone uses this: one command."""

    plane = _SubmittingPlane(
        states=[_record(WorkloadState.SUCCEEDED)],
        pages={"": _page([(0, "training started")], cursor="c1")},
    )

    result = _invoke(monkeypatch, plane, "dev-fleet/gpu-workers", "--", "python", "train.py")

    assert result.exit_code == EXIT_SUCCEEDED, result.output
    assert "training started" in result.stdout


def test_a_failed_job_gives_the_command_a_failing_status(monkeypatch) -> None:
    plane = _SubmittingPlane(states=[_record(WorkloadState.FAILED, failure_reason="boom")], pages={})

    result = _invoke(monkeypatch, plane, "dev-fleet/gpu-workers", "--", "python", "train.py")

    assert result.exit_code == EXIT_FAILED


def test_no_wait_keeps_the_submission_document_on_stdout(monkeypatch) -> None:
    """The scripting contract. stdout stays parseable JSON and nothing else."""

    plane = _SubmittingPlane(states=[_record(WorkloadState.RUNNING)], pages={})

    result = _invoke(monkeypatch, plane, "--no-wait", "dev-fleet/gpu-workers", "--", "python", "train.py")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == "wl-1"
    assert not any(path.endswith("/log-page") for path in plane.paths)


def test_streaming_keeps_stdout_free_of_the_submission_document(monkeypatch) -> None:
    """`eyvan run ... > out.txt` should hold the job's output, not a receipt."""

    plane = _SubmittingPlane(
        states=[_record(WorkloadState.SUCCEEDED)],
        pages={"": _page([(0, "the only thing on stdout")], cursor="c1")},
    )

    result = _invoke(monkeypatch, plane, "dev-fleet/gpu-workers", "--", "python", "train.py")

    assert result.stdout.strip() == "the only thing on stdout"


def test_since_shows_only_the_recent_window() -> None:
    """The tail is what a reattaching operator wants, not an hour of history."""

    plane = _Plane(
        states=[_record(WorkloadState.SUCCEEDED)],
        pages={"": {**_page([(0, "an hour ago")], cursor="c1", age_s=3600)}},
    )
    written: list[str] = []
    follower = _follower(plane, written)
    follower.since = timedelta(minutes=10)

    follower.follow("wl-1")

    assert "an hour ago" not in written


def test_since_keeps_a_line_it_cannot_place_in_time() -> None:
    """Dropping an undated line would silently shorten the window asked for."""

    page = _page([(0, "undated")], cursor="c1")
    for record in page["records"]:
        record.pop("timestamp")
    plane = _Plane(states=[_record(WorkloadState.SUCCEEDED)], pages={"": page})
    written: list[str] = []
    follower = _follower(plane, written)
    follower.since = timedelta(minutes=10)

    follower.follow("wl-1")

    assert "undated" in written


def test_workload_logs_can_follow_a_running_job(monkeypatch) -> None:
    """The command the detach message tells people to run has to exist."""

    plane = _SubmittingPlane(
        states=[_record(WorkloadState.SUCCEEDED)],
        pages={"": _page([(0, "reattached output")], cursor="c1")},
    )
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: plane)
    monkeypatch.setattr("eyvan.sdk.follow.time.sleep", lambda _seconds: None)

    result = CliRunner().invoke(cli.main, ["--user", "alice", "workload", "logs", "wl-1", "--follow"])

    assert result.exit_code == EXIT_SUCCEEDED, result.output
    assert "reattached output" in result.stdout


def test_since_without_follow_dumps_the_window_once(monkeypatch) -> None:
    """``--since`` is an output filter; alone it must not attach to the job.

    It used to imply ``--follow`` and poll until a running workload stopped --
    a hang where a dump was asked for.
    """

    stale = _page([(0, "an hour ago")], cursor="c1", age_s=3600)
    recent = _page([(0, "just now")], cursor="c1")
    window = {**stale, "records": stale["records"] + recent["records"]}
    plane = _Plane(states=[_record(WorkloadState.RUNNING)], pages={"": window})
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: plane)

    result = CliRunner().invoke(cli.main, ["--user", "alice", "workload", "logs", "wl-1", "--since", "10m"])

    assert result.exit_code == 0, result.output
    assert "just now" in result.stdout
    assert "an hour ago" not in result.stdout
    # One drain, no state polling: following the workload would read
    # /api/v1/workloads/wl-1 and wait for it to stop.
    assert "/api/v1/workloads/wl-1" not in plane.paths


def test_follower_falls_back_to_plain_logs_on_501_and_streams_without_replaying() -> None:
    """When /log-page returns 501, fall back to /logs so stdout/stderr are not dropped."""

    from eyvan.sdk.commands.http import EyvanApiError

    class _PlainLogsPlane:
        def __init__(self) -> None:
            self.states = [_record(WorkloadState.RUNNING), _record(WorkloadState.SUCCEEDED)]
            self.log_snapshots = ["line one\npartial", "line one\npartial finished\nline two\n"]
            self.log_page_calls = 0

        def request(self, method: str, path: str, *, params=None, json=None, headers=None) -> JsonValue:
            del method, params, json, headers
            if path.endswith("/log-page"):
                self.log_page_calls += 1
                raise EyvanApiError(501, "execution backend 'native-local' does not expose structured logs")
            return self.states.pop(0) if len(self.states) > 1 else self.states[0]

        def request_text(self, method: str, path: str, *, params=None, headers=None) -> str:
            del method, params, headers
            assert path.endswith("/logs")
            return self.log_snapshots.pop(0) if len(self.log_snapshots) > 1 else self.log_snapshots[0]

    plane = _PlainLogsPlane()
    stdout: list[str] = []
    stderr: list[str] = []
    follower = WorkloadFollower(
        client=plane,
        output=JobTerminalOutput(writer=stderr.append),
        write=stdout.append,
        sleep=lambda _seconds: None,
    )

    outcome = follower.follow("wl-1")

    assert outcome.exit_code == EXIT_SUCCEEDED
    assert stdout == ["line one", "partial finished", "line two"]
    assert plane.log_page_calls == 1
    assert not any("logs are unavailable" in line for line in stderr)


def test_follower_falls_back_to_plain_logs_when_log_page_reports_unsupported_backend() -> None:
    """When the controller wraps an agent's 501 into an unavailable page, still fall back to /logs."""

    class _WrappedUnsupportedPlane:
        def __init__(self) -> None:
            self.states = [_record(WorkloadState.SUCCEEDED)]

        def request(self, method: str, path: str, *, params=None, json=None, headers=None) -> JsonValue:
            del method, params, json, headers
            if path.endswith("/log-page"):
                return {
                    "attempt": 1,
                    "attempt_changed": False,
                    "records": [],
                    "next_cursor": "",
                    "unavailable": True,
                    "detail": "execution backend 'native-local' does not expose structured logs",
                    "truncated_ranks": [0],
                }
            return self.states[0]

        def request_text(self, method: str, path: str, *, params=None, headers=None) -> str:
            del method, params, headers
            assert path.endswith("/logs")
            return "RUN_HUMANIZE=4.1 kB\n"

    stdout: list[str] = []
    stderr: list[str] = []
    follower = WorkloadFollower(
        client=_WrappedUnsupportedPlane(),
        output=JobTerminalOutput(writer=stderr.append),
        write=stdout.append,
        sleep=lambda _seconds: None,
    )

    outcome = follower.follow("wl-1")

    assert outcome.exit_code == EXIT_SUCCEEDED
    assert stdout == ["RUN_HUMANIZE=4.1 kB"]
    assert not any("logs are unavailable" in line for line in stderr)
