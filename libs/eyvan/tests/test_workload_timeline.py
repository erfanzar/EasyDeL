# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One ordered narrative per workload, and a time window over its output.

"It failed" and "it failed here" are different answers, and every fact behind
the second was already in the audit ledger. What was missing was reading the
events together: eight minutes queued, thirty provisioning, two preemptions is
obvious in one column and invisible across three commands.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from click.testing import CliRunner

from eyvan.records import AuditEvent
from eyvan.sdk import cli
from eyvan.sdk.follow import parse_duration
from eyvan.sdk.timeline import render, timeline

START = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def _event(sequence: int, action: str, *, after: int = 0, new_state: str | None = None, payload=None) -> AuditEvent:
    return AuditEvent(
        id=f"evt-{sequence}",
        sequence=sequence,
        timestamp=START + timedelta(seconds=after),
        actor="system",
        org_id="acme",
        project_id="research",
        entity_type="workload",
        entity_id="wl-1",
        action=action,
        new_state=new_state,
        payload=payload or {},
    )


# --------------------------------------------------------------------------- the narrative


def test_a_run_reads_as_a_sequence_with_elapsed_times() -> None:
    """The whole point: where the time went, in one column."""

    moments = timeline(
        [
            _event(1, "workload.submitted", new_state="pending"),
            _event(2, "workload.admitted", after=59, new_state="admitted"),
            _event(3, "workload.dispatched", after=130, new_state="dispatching"),
            _event(4, "workload.succeeded", after=828, new_state="succeeded"),
        ]
    )

    assert [moment.elapsed for moment in moments] == ["0:00:00", "0:00:59", "0:02:10", "0:13:48"]
    assert [moment.headline for moment in moments] == [
        "submitted",
        "admitted",
        "dispatched to its cells",
        "succeeded",
    ]


def test_order_comes_from_the_ledger_sequence_not_the_clock() -> None:
    """Two events committed together share a timestamp.

    Sorting those by clock is a coin flip, and a timeline that puts "dispatched"
    before "leased" is worse than no timeline.
    """

    moments = timeline(
        [
            _event(3, "workload.dispatched", new_state="dispatching"),
            _event(1, "workload.submitted", new_state="pending"),
            _event(2, "workload.leased", new_state="leased"),
        ]
    )

    assert [moment.headline for moment in moments] == ["submitted", "leased capacity", "dispatched to its cells"]


def test_an_unrecognised_action_is_shown_rather_than_dropped() -> None:
    """An event nobody thought to phrase is the one worth seeing."""

    moments = timeline([_event(1, "workload.something_new_happened", new_state="running")])

    assert "something new happened" in render(moments)[0]


def test_a_bare_state_change_reads_as_the_state() -> None:
    """ "state changed (running)" says the same thing twice and buries the word."""

    lines = render(timeline([_event(1, "workload.state_changed", new_state="running")]))

    assert "running" in lines[0]
    assert "state changed" not in lines[0]


def test_a_failure_carries_its_reason_onto_the_line() -> None:
    """The line an operator opened the timeline to read."""

    lines = render(
        timeline(
            [
                _event(1, "workload.submitted", new_state="pending"),
                _event(
                    2,
                    "workload.failed",
                    after=10,
                    new_state="failed",
                    payload={"failure_reason": "exit status 1"},
                ),
            ]
        )
    )

    assert "exit status 1" in lines[-1]


def test_an_empty_ledger_says_so_rather_than_printing_nothing() -> None:
    """A blank answer is indistinguishable from a broken command."""

    assert "nothing has been recorded" in render(timeline([]))[0]


# --------------------------------------------------------------------------- the command


class _Plane:
    def __init__(self, events: list[dict], workload: dict | None = None) -> None:
        self.events = events
        self.workload = workload or {"id": "wl-1", "lease_ids": []}
        self.queries: list[dict] = []

    def request(self, method, path, *, params=None, json=None, headers=None):
        del method, json, headers
        if path == "/api/v1/events":
            self.queries.append(dict(params or {}))
            entity = (params or {}).get("entity_type")
            return [event for event in self.events if event["entity_type"] == entity]
        return self.workload

    def request_text(self, method, path, *, params=None, headers=None):
        del method, path, params, headers
        return ""

    def replace_identity_headers(self, headers) -> None:
        del headers


def test_the_command_reads_the_workload_and_its_leases(monkeypatch) -> None:
    """ "capacity leased" and "capacity released" are recorded against the lease."""

    events = [
        {**_event(1, "workload.submitted", new_state="pending").model_dump(mode="json")},
        {
            **_event(2, "lease.settled", after=30).model_dump(mode="json"),
            "entity_type": "lease",
            "entity_id": "lease-1",
        },
    ]
    plane = _Plane(events, workload={"id": "wl-1", "lease_ids": ["lease-1"]})
    monkeypatch.setattr(cli, "EyvanClient", lambda *_a, **_k: plane)

    result = CliRunner().invoke(
        cli.main, ["--user", "alice", "--org", "acme", "--project", "research", "workload", "timeline", "wl-1"]
    )

    assert result.exit_code == 0, result.output
    assert "submitted" in result.output
    assert "capacity released" in result.output
    assert {query["entity_type"] for query in plane.queries} == {"workload", "lease"}


# --------------------------------------------------------------------------- --since


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30s", timedelta(seconds=30)),
        ("10m", timedelta(minutes=10)),
        ("2h", timedelta(hours=2)),
        ("1d", timedelta(days=1)),
    ],
)
def test_a_duration_is_read_as_written(text: str, expected: timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["10", "m", "ten minutes", "-5m", "10w", ""])
def test_an_unspelled_duration_is_refused_rather_than_guessed(text: str) -> None:
    """Reading "10" as seconds when somebody meant minutes hides what they asked for."""

    with pytest.raises(ValueError):
        parse_duration(text)
