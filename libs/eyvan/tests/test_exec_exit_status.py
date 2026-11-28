# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan workload exec`` exits non-zero when any rank did not run the command.

Scripts must be able to distinguish a refused diagnostic from a successful one
without parsing the response document.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from eyvan.sdk import cli
from eyvan.sdk.commands.workloads import _failed_exec_ranks

WORKLOAD_ID = "wl_" + "e" * 32


class _Plane:
    def __init__(self, document: dict) -> None:
        self.document = document

    def request(self, method, path, *, params=None, json=None, headers=None):
        del method, path, params, json, headers
        return self.document

    def request_text(self, method, path, *, params=None, headers=None):
        del method, path, params, headers
        return ""

    def replace_identity_headers(self, headers) -> None:
        del headers


def _exec(monkeypatch, document: dict):
    monkeypatch.setattr(cli, "EyvanClient", lambda *_a, **_k: _Plane(document))
    return CliRunner().invoke(cli.main, ["workload", "exec", WORKLOAD_ID, "--json", "--", "hostname"])


def _document(*results: dict) -> dict:
    return {"workload_id": WORKLOAD_ID, "attempt": 1, "results": list(results)}


def test_every_rank_succeeding_exits_zero(monkeypatch) -> None:
    result = _exec(monkeypatch, _document({"rank": 0, "exit_code": 0}, {"rank": 1, "exit_code": 0}))

    assert result.exit_code == 0, result.output


def test_a_refused_rank_exits_non_zero_and_still_prints_the_document(monkeypatch) -> None:
    refused = {"rank": 0, "cell_id": "c1", "error": "this fleet does not permit exec"}

    result = _exec(monkeypatch, _document(refused))

    assert result.exit_code == 1
    assert "this fleet does not permit exec" in result.output
    assert "exec failed on rank(s) 0" in result.output


@pytest.mark.parametrize(
    ("results", "failed"),
    [
        (({"rank": 0, "exit_code": 0}, {"rank": 1, "exit_code": 3}), [1]),
        (({"rank": 0, "error": "unreachable"}, {"rank": 1, "exit_code": 0}), [0]),
        # Suppressed output withholds the exit code on purpose; that is not a failure.
        (({"rank": 0, "exit_code": None, "output_suppressed": True},), []),
        # A slice answers per host; one unreachable host fails the rank that owns it.
        (
            (
                {
                    "rank": 0,
                    "ranks": [
                        {"rank": 0, "result": {"exit_code": 0}},
                        {"rank": 1, "result": {"observed": False, "detail": "native worker is unreachable"}},
                    ],
                },
            ),
            [0],
        ),
        (({"rank": 0, "ranks": [{"rank": 0, "result": {"exit_code": 0}}]},), []),
    ],
)
def test_failed_ranks_are_named(results, failed) -> None:
    assert _failed_exec_ranks(_document(*results)) == failed


def test_secret_mount_exec_refusal_exits_nonzero(monkeypatch) -> None:
    result = _exec(
        monkeypatch,
        {"workload_id": WORKLOAD_ID, "attempt": 1, "results": [], "output_suppressed": True},
    )
    assert result.exit_code == 1
    assert "exec is not available for workloads with secret mounts; nothing ran" in result.stderr


def test_per_rank_suppressed_exec_remains_success(monkeypatch) -> None:
    result = _exec(monkeypatch, _document({"rank": 0, "exit_code": None, "output_suppressed": True}))
    assert result.exit_code == 0


def test_an_unexpected_document_is_not_called_a_failure() -> None:
    assert _failed_exec_ranks({"workload_id": WORKLOAD_ID}) == []
    assert _failed_exec_ranks([]) == []


def test_human_exec_renders_stdout_stderr_exit_codes_and_errors(monkeypatch) -> None:
    document = _document(
        {"rank": 0, "exit_code": 0, "stdout": "host-0\n", "stderr": "warn-0\n"},
        {
            "rank": 1,
            "ranks": [
                {"rank": 0, "result": {"exit_code": 0, "stdout": "slice-host-0\n"}},
                {"rank": 1, "result": {"observed": False, "detail": "worker unreachable"}},
            ],
        },
        {"rank": 2, "error": "exec refused"},
    )
    monkeypatch.setattr(cli, "EyvanClient", lambda *_a, **_k: _Plane(document))
    result = CliRunner().invoke(cli.main, ["workload", "exec", WORKLOAD_ID, "--", "hostname"])

    assert result.exit_code == 1
    assert "[rank 0] exit=0" in result.output
    assert "host-0" in result.output
    assert "warn-0" in result.output
    assert "[rank 1.0] exit=0" in result.output
    assert "slice-host-0" in result.output
    assert "[rank 1.1] unobserved: worker unreachable" in result.output
    assert "[rank 2] error: exec refused" in result.output
    assert "exec failed on rank(s) 1, 2" in result.output
