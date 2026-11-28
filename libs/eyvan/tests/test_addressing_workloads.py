# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Naming a job, and naming a batch of them.

A workload is addressed by ``wl_`` and thirty-two hex characters, which nobody
types from memory and no sweep has in common. Iris addresses tasks by a path
and stops them with ``--prefix``; this is the same ergonomics without inventing
a second identity system: a name, a unique id prefix, or a prefix that matches
a set on purpose.

The set operations are the dangerous half. A mistyped prefix on a cancel ends
other people's training runs, so these tests are mostly about what has to be
true before anything is sent.
"""

from __future__ import annotations

from click.testing import CliRunner

from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
from eyvan.sdk import cli


def _record(workload_id: str, name: str, state: WorkloadState = WorkloadState.RUNNING) -> dict:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name=name, execution=CommandExecution(command="python train.py")),
        state=state,
        created_by="alice",
    ).model_dump(mode="json")


class _Plane:
    """A control plane holding a fixed set of workloads, recording mutations."""

    def __init__(self, *workloads: dict) -> None:
        self.workloads = list(workloads)
        self.calls: list[tuple[str, str]] = []

    def request(self, method, path, *, params=None, json=None, headers=None):
        del params, json, headers
        self.calls.append((method, path))
        if method == "GET" and path == "/api/v1/workloads":
            return self.workloads
        if method == "POST":
            return self.workloads[0] if self.workloads else {}
        return {}

    def request_text(self, method, path, *, params=None, headers=None):
        del method, path, params, headers
        return ""

    def replace_identity_headers(self, headers) -> None:
        del headers


def _invoke(monkeypatch, plane: _Plane, *args: str):
    monkeypatch.setattr(cli, "EyvanClient", lambda *_a, **_k: plane)
    return CliRunner().invoke(cli.main, ["--user", "alice", "--org", "acme", "--project", "research", *args])


def _mutations(plane: _Plane, action: str) -> list[str]:
    return [path for method, path in plane.calls if method == "POST" and path.endswith(f"/{action}")]


SWEEP = (
    _record("wl_" + "a" * 32, "sweep-lr-001"),
    _record("wl_" + "b" * 32, "sweep-lr-002"),
    _record("wl_" + "c" * 32, "baseline"),
)


# --------------------------------------------------------------------------- naming one


def test_a_full_id_is_used_without_a_lookup(monkeypatch) -> None:
    """The normal path stays one request, and works in a scope you cannot list."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "wl_" + "a" * 32)

    assert result.exit_code == 0, result.output
    assert _mutations(plane, "cancel") == [f"/api/v1/workloads/wl_{'a' * 32}/cancel"]
    assert ("GET", "/api/v1/workloads") not in plane.calls


def test_a_name_addresses_the_one_workload_carrying_it(monkeypatch) -> None:
    """The name is the thing a person chose; the id is the thing a machine did."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "baseline")

    assert result.exit_code == 0, result.output
    assert _mutations(plane, "cancel") == [f"/api/v1/workloads/wl_{'c' * 32}/cancel"]


def test_an_ambiguous_reference_names_the_candidates_instead_of_choosing(monkeypatch) -> None:
    """Picking one would cancel the wrong training run and say it succeeded."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "sweep-lr-")

    assert result.exit_code != 0
    assert "sweep-lr-001" in result.output and "sweep-lr-002" in result.output
    assert _mutations(plane, "cancel") == []


def test_an_unmatched_reference_is_left_for_the_route_to_judge(monkeypatch) -> None:
    """This client does not own the question of whether a workload exists."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "wl_unknown")

    assert result.exit_code == 0, result.output
    assert _mutations(plane, "cancel") == ["/api/v1/workloads/wl_unknown/cancel"]


# --------------------------------------------------------------------------- naming a set


def test_a_prefix_applies_to_every_match(monkeypatch) -> None:
    """What a sweep is for: one command over the batch you submitted."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "--prefix", "sweep-lr-", "--yes")

    assert result.exit_code == 0, result.output
    assert sorted(_mutations(plane, "cancel")) == sorted(
        [f"/api/v1/workloads/wl_{'a' * 32}/cancel", f"/api/v1/workloads/wl_{'b' * 32}/cancel"]
    )


def test_a_prefix_matching_several_is_refused_until_confirmed(monkeypatch) -> None:
    """A mistyped prefix on a cancel is this CLI's most expensive mistake."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "--prefix", "sweep-lr-")

    assert result.exit_code != 0
    assert "2 workloads" in result.output
    assert "sweep-lr-001" in result.output, "show what would be cancelled"
    assert _mutations(plane, "cancel") == []


def test_a_prefix_matching_one_needs_no_confirmation(monkeypatch) -> None:
    """Confirmation for a single job is friction with nothing behind it."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "--prefix", "baseline")

    assert result.exit_code == 0, result.output
    assert _mutations(plane, "cancel") == [f"/api/v1/workloads/wl_{'c' * 32}/cancel"]


def test_an_empty_prefix_is_refused(monkeypatch) -> None:
    """Everything starts with the empty string, and this argument is a cancel."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "--prefix", "", "--yes")

    assert result.exit_code != 0
    assert _mutations(plane, "cancel") == []


def test_a_prefix_matching_nothing_says_so(monkeypatch) -> None:
    """Silence would read as "done" for a command that did nothing."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "--prefix", "nothing-", "--yes")

    assert result.exit_code != 0
    assert "nothing-" in result.output


def test_an_id_and_a_prefix_together_are_refused(monkeypatch) -> None:
    """Two different sets of work; guessing which one was meant is not an option."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "cancel", "wl_" + "a" * 32, "--prefix", "sweep-")

    assert result.exit_code != 0
    assert _mutations(plane, "cancel") == []


def test_kick_takes_a_prefix_too(monkeypatch) -> None:
    """A zone going bad under a sweep is exactly the shape kick exists for."""

    plane = _Plane(*SWEEP)

    result = _invoke(monkeypatch, plane, "workload", "kick", "--prefix", "sweep-lr-", "--yes")

    assert result.exit_code == 0, result.output
    assert len(_mutations(plane, "kick")) == 2


def test_retry_takes_a_prefix_too(monkeypatch) -> None:
    plane = _Plane(*[_record(item["id"], item["spec"]["name"], WorkloadState.FAILED) for item in SWEEP])

    result = _invoke(monkeypatch, plane, "workload", "retry", "--prefix", "sweep-lr-", "--yes")

    assert result.exit_code == 0, result.output
    assert len(_mutations(plane, "retry")) == 2
