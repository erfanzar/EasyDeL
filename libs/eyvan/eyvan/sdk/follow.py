# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Attach to a running workload: its output on stdout, its progress on stderr."""

from __future__ import annotations

import time
import typing as tp
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

from eyvan._internal.json_types import JsonValue, is_json_object
from eyvan.records import WorkloadRecord, WorkloadState
from eyvan.sdk.client.sync import UNREACHABLE_GRACE_S
from eyvan.sdk.client.transport import EyvanUnavailableError
from eyvan.sdk.job_receipt import JobLifecycleUpdate, JobTerminalOutput
from eyvan.sdk.lifecycle import TERMINAL_STATES, workload_lifecycle_update
from eyvan.worker.runtime.log_records import routine_unit_chatter

#: What the process exits with once the workload it followed has stopped.
#:
#: Derived from the durable state, not from the job's own status byte: nothing
#: in the control plane records one today, and inventing a number that looked
#: like the process's would be worse than a state nobody can misread.
EXIT_SUCCEEDED = 0
EXIT_FAILED = 1
EXIT_CANCELLED = 2
#: The follower was interrupted. The job is unaffected -- it is durable and
#: keeps running -- so this is "you stopped watching", not "it stopped".
EXIT_DETACHED = 130

_EXIT_STATUS: dict[WorkloadState, int] = {
    WorkloadState.SUCCEEDED: EXIT_SUCCEEDED,
    WorkloadState.FAILED: EXIT_FAILED,
    WorkloadState.CANCELLED: EXIT_CANCELLED,
}


@dataclass(frozen=True, slots=True)
class FollowOutcome:
    """How a followed workload ended, and what the process should exit with."""

    state: WorkloadState | None
    exit_code: int
    detached: bool = False


@dataclass(frozen=True, slots=True)
class _LogWindow:
    """One page of output as the follower needs it."""

    lines: tuple[tuple[int, str, datetime | None], ...] = ()
    #: Parallel to ``lines``: ``(pid, host)`` provenance when the archive
    #: reports it, else ``(None, None)``. Ray-style prefixes are built from it.
    provenance: tuple[tuple[int | None, str | None], ...] = ()
    cursor: str = ""
    attempt_changed: bool = False
    expected_ranks: int | None = None
    detail: str = ""
    unavailable: bool = False
    #: Exhaustion is about the raw page, not the transcript after filtering.
    raw_record_count: int = 0
    truncated_ranks: tuple[int, ...] = ()

    @classmethod
    def parse(cls, payload: JsonValue) -> _LogWindow:
        """Read a control-plane log page, keeping only what a follower shows."""

        if not is_json_object(payload):
            return cls()
        raw = payload.get("records")
        lines: list[tuple[int, str, datetime | None]] = []
        provenance: list[tuple[int | None, str | None]] = []
        for record in raw if isinstance(raw, list) else []:
            if not is_json_object(record):
                continue
            message = record.get("message")
            if not isinstance(message, str) or routine_unit_chatter(message):
                continue
            pid = record.get("pid")
            host = record.get("host")
            provenance.append(
                (
                    pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
                    host if isinstance(host, str) else None,
                )
            )
            rank = record.get("rank")
            lines.append(
                (
                    rank if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0 else 0,
                    message,
                    _timestamp(record.get("timestamp")),
                )
            )
        cursor = payload.get("next_cursor")
        expected = payload.get("expected_ranks")
        detail = payload.get("detail")
        truncated = payload.get("truncated_ranks")
        ranks = (
            tuple(
                sorted(
                    {rank for rank in truncated if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0}
                )
            )
            if isinstance(truncated, list)
            else ()
        )
        return cls(
            lines=tuple(lines),
            provenance=tuple(provenance),
            cursor=cursor if isinstance(cursor, str) else "",
            attempt_changed=bool(payload.get("attempt_changed")),
            expected_ranks=expected if isinstance(expected, int) and not isinstance(expected, bool) else None,
            detail=detail if isinstance(detail, str) else "",
            unavailable=bool(payload.get("unavailable")),
            raw_record_count=len(raw) if isinstance(raw, list) else 0,
            truncated_ranks=ranks,
        )


#: How a duration is spelled on the command line. Iris uses --since-seconds;
#: a suffix reads better and "10m" is what somebody means when they type it.
_DURATION_UNITS: dict[str, int] = {"s": 1, "m": 60, "h": 3600, "d": 86_400}


def parse_duration(value: str) -> timedelta:
    """Parse ``30s``, ``10m``, ``2h`` or ``1d`` into a duration.

    Raises:
        ValueError: When the spelling is not one of those. Refused rather than
            guessed: reading "10" as ten seconds when somebody meant minutes
            hides most of what they asked to see and looks like a quiet log.
    """

    text = value.strip().lower()
    if len(text) < 2 or text[-1] not in _DURATION_UNITS or not text[:-1].isdigit():
        units = ", ".join(f"10{suffix}" for suffix in _DURATION_UNITS)
        raise ValueError(f"{value!r} is not a duration; write one of {units}")
    return timedelta(seconds=int(text[:-1]) * _DURATION_UNITS[text[-1]])


def _timestamp(value: object) -> datetime | None:
    """Read one record's ISO timestamp, or ``None`` when it carries none.

    A record with no readable timestamp cannot be placed in time, and is shown
    by ``--since`` rather than hidden: dropping it would silently shorten the
    window a caller asked to see.
    """

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class _Transport(tp.Protocol):
    """The control-plane calls a follower makes."""

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        json: tp.Any = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        """Send a JSON request and return its decoded response."""

        ...


def _ray_style_prefix(rank: int, provenance: tuple[int | None, str | None]) -> str:
    """Ray's worker-log prefix: ``(Worker pid=4102, ip=10.130.0.138) ``.

    Whatever the archive reports goes in; what it does not is left out rather
    than invented -- a rank with no pid reads ``(rank 2)``, bare if neither.
    """

    pid, host = provenance
    parts: list[str] = []
    if pid is not None:
        parts.append(f"pid={pid}")
    if host is not None:
        parts.append(host)
    if not parts:
        return f"(rank {rank}) "
    inner = ", ".join(parts)
    return f"(Worker {inner}) "


@dataclass(slots=True)
class WorkloadFollower:
    """Print a workload's output as it appears and end when the workload does.

    The two streams are separate on purpose. Output goes to **stdout**, because
    it is the job's own and redirecting it should capture that and nothing else.
    Progress goes to **stderr**, because "queued, provisioning, running" is
    about the fleet rather than about the run.

    Logs are read through the ordinary resumable cursor, so the handover from
    the live cell to the archive happens underneath this loop rather than in it:
    a job whose slice is reclaimed keeps streaming from what was archived, and a
    finished job's tail is served from an archive that settlement commits before
    it marks the workload terminal. That ordering is why one final drain after a
    terminal state is enough, rather than a guess at how long to keep asking.
    """

    client: _Transport
    output: JobTerminalOutput = field(default_factory=JobTerminalOutput)
    write: tp.Callable[[str], None] = print
    poll_interval: float = 2.0
    page_limit: int = 200
    sleep: tp.Callable[[float], None] = time.sleep
    #: Pages drained per poll. A bound rather than "until empty": a job printing
    #: faster than this client reads would otherwise hold the loop forever and
    #: the lifecycle line would stop moving while the fleet changed underneath.
    max_pages_per_poll: int = 64
    #: Show only output newer than this. A client-side window over the whole
    #: journal, because a log position is an opaque backend cursor and no
    #: backend Eyvan reads can seek by time. It bounds what is *printed*, never
    #: what is fetched -- worth saying, because the difference is the whole cost.
    since: timedelta | None = None
    _cursor: str = ""
    _last_update: JobLifecycleUpdate | None = None
    _rank_prefixes: bool = False
    _reported_notes: set[str] = field(default_factory=set)

    def follow(self, workload_id: str) -> FollowOutcome:
        """Stream one workload to completion and report how it ended."""

        try:
            return self._follow(workload_id)
        except KeyboardInterrupt:
            # Detaching, not cancelling. The workload is durable and nothing
            # here owns it; a client that cancelled on Ctrl-C would destroy a
            # long training run because someone wanted their terminal back.
            self.output.detached(workload_id)
            return FollowOutcome(state=None, exit_code=EXIT_DETACHED, detached=True)
        finally:
            # However this ends -- completion, Ctrl-C, or an error on the way
            # out -- the terminal must not be left with a live region nothing
            # is repainting any more.
            self._settle_output()

    def _follow(self, workload_id: str) -> FollowOutcome:
        while True:
            workload = self._workload(workload_id)
            terminal = workload is not None and workload.state in TERMINAL_STATES
            # Live polls are fair to lifecycle updates; terminal and one-shot
            # reads have no later poll to recover output left behind a budget.
            self._drain(workload_id, bounded=not terminal)
            if terminal:
                assert workload is not None
                # Settlement archives this attempt's remaining output *before*
                # it commits the terminal state, so drain the available archive
                # to exhaustion before reporting its durable outcome.
                self._emit(workload)
                return FollowOutcome(state=workload.state, exit_code=_EXIT_STATUS[workload.state])
            if workload is not None:
                self._emit(workload)
            self.sleep(self.poll_interval)

    def _workload(self, workload_id: str) -> WorkloadRecord | None:
        """Read the durable record, or ``None`` when it cannot be parsed.

        A record this client cannot parse is a version skew, and the answer to
        one is to keep streaming rather than to declare the job over: the logs
        are the reason to be here and they are still readable.
        """

        payload = self._request(f"/api/v1/workloads/{workload_id}")
        if not is_json_object(payload):
            return None
        try:
            return WorkloadRecord.model_validate(payload)
        except Exception:
            return None

    def _drain(self, workload_id: str, *, bounded: bool = False) -> bool:
        """Read available output, returning whether the page stream was exhausted.

        The one-shot CLI caller uses an exhaustive drain by default. Live
        following explicitly bounds each batch so progress is still polled.
        """

        pages = 0
        seen_cursors: set[str] = set()
        while not bounded or pages < self.max_pages_per_poll:
            if self._cursor in seen_cursors:
                self._note("log reading stopped because its cursor did not advance; output may be incomplete")
                return False
            seen_cursors.add(self._cursor)
            pages += 1
            window = _LogWindow.parse(
                self._request(
                    f"/api/v1/workloads/{workload_id}/log-page",
                    params={"after": self._cursor, "limit": self.page_limit},
                )
            )
            if window.attempt_changed:
                # A retry. The previous attempt's positions address journals
                # that belong to an execution that no longer exists, so the
                # boundary is announced rather than papered over: the lines
                # after it are a different run of the same job.
                self.output.note("the job restarted; following the new attempt from its first line")
                # The new attempt's first page may carry no cursor at all;
                # keeping the previous attempt's position would resend it every
                # poll and re-announce the restart forever.
                self._cursor = ""
            # Never echo arbitrary server detail: it can contain credentials or
            # terminal controls. Structured gap metadata is safe to describe.
            if window.unavailable:
                self._note("job logs are unavailable; output may be incomplete")
            if window.truncated_ranks:
                ranks = ", ".join(str(rank) for rank in window.truncated_ranks[:20])
                suffix = f" (and {len(window.truncated_ranks) - 20} more)" if len(window.truncated_ranks) > 20 else ""
                self._note(f"job logs are incomplete for ranks: {ranks}{suffix}")
            if window.expected_ranks is not None and window.expected_ranks > 1:
                self._rank_prefixes = True
            if not self._rank_prefixes and any(rank for rank, _message, _at in window.lines):
                self._rank_prefixes = True
            cutoff = None if self.since is None else datetime.now(UTC) - self.since
            for index, (rank, message, at) in enumerate(window.lines):
                if cutoff is not None and at is not None and at < cutoff:
                    continue
                # The job has something to say, so the waiting line has done its
                # job; settle it first or the log scrolls through the region the
                # live display is still repainting.
                self._settle_output()
                prefix = _ray_style_prefix(rank, window.provenance[index]) if self._rank_prefixes else ""
                if prefix:
                    from eyvan.sdk.terminal import cyan

                    self.write(f"{cyan(prefix)}{message}")
                else:
                    self.write(message)
            if window.cursor:
                self._cursor = window.cursor
            if not window.raw_record_count:
                return not window.unavailable
        return False

    def _request(
        self, path: str, *, params: tp.Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonValue:
        """Give each durable read the SDK wait's bounded transport-outage grace.

        Retry the same request/cursor without replaying already printed output.
        Only unavailability is forgiven: authorization and missing records are
        authoritative refusals, not a control-plane restart. A successful read
        ends this outage, so the next read receives a fresh grace window.
        """

        unreachable_since: float | None = None
        while True:
            try:
                return self.client.request("GET", path, params=params)
            except EyvanUnavailableError:
                now = time.monotonic()
                if unreachable_since is None:
                    unreachable_since = now
                remaining = UNREACHABLE_GRACE_S - (now - unreachable_since)
                if remaining <= 0:
                    raise
                self.sleep(min(self.poll_interval, remaining))

    def _note(self, message: str) -> None:
        """Show a credential-safe log-read diagnostic once per attachment."""

        if message not in self._reported_notes:
            self._settle_output()
            self.output.note(message)
            self._reported_notes.add(message)

    def _settle_output(self) -> None:
        """Stop any in-place status the output is drawing, if it draws one.

        Optional on the protocol: a plain writer has nothing to settle, and the
        follower must not require every output to grow a method it cannot use.
        """

        settle = getattr(self.output, "settle", None)
        if callable(settle):
            settle()

    def _emit(self, workload: WorkloadRecord) -> None:
        """Show one lifecycle change, never the same one twice."""

        update = workload_lifecycle_update(workload)
        if update == self._last_update:
            return
        if self._last_update is not None and update.diagnostic == self._last_update.diagnostic:
            update = replace(update, diagnostic=None)
        if update == self._last_update:
            return
        self.output.lifecycle(update)
        self._last_update = update


__all__ = (
    "EXIT_CANCELLED",
    "EXIT_DETACHED",
    "EXIT_FAILED",
    "EXIT_SUCCEEDED",
    "FollowOutcome",
    "WorkloadFollower",
)
