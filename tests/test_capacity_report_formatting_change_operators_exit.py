# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capacity report formatting must not change the operator's exit-status gate."""

from __future__ import annotations

import copy
import json

import pytest
from click.testing import CliRunner

from eyvan.sdk import cli


class _ReportClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.paths: list[str] = []

    def request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
        assert method == "GET"
        self.paths.append(path)
        return self.payload

    def replace_identity_headers(self, headers: object) -> None:
        pass


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize(
    ("command", "agreement", "scanned", "unreachable", "expected_exit"),
    [
        ("builds", False, True, False, 1),
        ("builds", None, True, False, 1),
        ("builds", None, True, True, 1),
        ("builds", True, True, False, 0),
        # Preserve the server's agreement verdict, not a stricter client policy.
        ("builds", True, True, True, 0),
        ("orphans", None, False, False, 1),
        ("orphans", None, True, True, 1),
        ("orphans", None, True, False, 0),
    ],
    ids=[
        "skew",
        "unknown",
        "unknown-unreachable",
        "agreement",
        "partial-agreement",
        "unscanned",
        "unreachable",
        "scanned",
    ],
)
def test_registered_capacity_report_exit_status_is_format_independent(
    monkeypatch: pytest.MonkeyPatch,
    as_json: bool,
    command: str,
    agreement: bool | None,
    scanned: bool,
    unreachable: bool,
    expected_exit: int,
) -> None:
    failures = [{"endpoint": "http://agent:8090", "detail": "unavailable"}] if unreachable else []
    if command == "builds":
        payload: dict[str, object] = {
            "plane": {"version": "0.0.1", "source_digest": "sha256:plane"},
            "agents": [
                {
                    "endpoint": "http://agent:8090",
                    "pool_id": "pool",
                    "cell_id": "cell",
                    "build": {"source_digest": "sha256:old" if agreement is False else "sha256:plane"},
                    "workers": None if agreement is None else [],
                }
            ],
            "unreachable": failures,
            "unscanned_pools": [],
            "in_agreement": agreement,
        }
    else:
        payload = {
            "orphans": [],
            "scanned_endpoints": ["http://agent:8090"] if scanned else [],
            "unreachable": failures,
        }
    original = copy.deepcopy(payload)
    client = _ReportClient(payload)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: client)
    args = ["--user", "alice", "--org", "acme", "--project", "research", "capacity", command]
    if as_json:
        args.append("--json")

    result = CliRunner().invoke(cli.main, args)

    assert result.exit_code == expected_exit, result.output
    assert client.paths == [f"/api/v1/capacity/{command}"]
    assert payload == original
    if as_json:
        assert json.loads(result.stdout) == original
    elif command == "builds" and agreement is False:
        assert "DIFFERENT BUILD" in result.stdout
    elif command == "builds" and agreement is None:
        assert "could not be shown to agree" in result.stderr
    elif command == "orphans" and not scanned:
        assert "nothing was scanned" in result.stderr
    elif command == "orphans" and unreachable:
        assert "could not be reached" in result.stderr
    elif command == "orphans":
        assert "no orphaned cell records" in result.stdout
