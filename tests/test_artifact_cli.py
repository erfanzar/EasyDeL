# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI contract for dry-run-first artifact inventory."""

from __future__ import annotations

from dataclasses import dataclass

from click.testing import CliRunner

from eyvan.sdk.commands.artifacts import artifacts


@dataclass(frozen=True)
class _Scope:
    org_id: str
    project_id: str


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
        self.calls.append((method, path, kwargs))
        return {"dry_run": method == "GET", "entries": []}


class _Session:
    def __init__(self) -> None:
        self.client = _Client()

    def workload_scope(self, *, org_id: str | None, project_id: str | None) -> _Scope:
        return _Scope(org_id or "context-org", project_id or "context-project")


def test_inventory_cli_is_report_only_by_default_and_forwards_the_bounded_cursor() -> None:
    session = _Session()

    result = CliRunner().invoke(
        artifacts,
        [
            "inventory",
            "--org",
            "acme",
            "--project",
            "research",
            "--cursor",
            "eyvan://results/run-1",
            "--limit",
            "25",
        ],
        obj=session,
    )

    assert result.exit_code == 0, result.output
    assert session.client.calls == [
        (
            "GET",
            "/api/v1/artifacts/inventory",
            {
                "params": {
                    "org_id": "acme",
                    "project_id": "research",
                    "cursor": "eyvan://results/run-1",
                    "limit": 25,
                }
            },
        )
    ]


def test_inventory_cli_refuses_reindex_until_the_exact_scope_is_acknowledged() -> None:
    session = _Session()

    refused = CliRunner().invoke(
        artifacts,
        [
            "inventory",
            "--org",
            "acme",
            "--project",
            "research",
            "--reindex-missing",
            "--acknowledge-scope",
            "acme/other",
        ],
        obj=session,
    )

    assert refused.exit_code != 0
    assert "run the dry-run inventory first" in refused.output
    assert session.client.calls == []

    accepted = CliRunner().invoke(
        artifacts,
        [
            "inventory",
            "--org",
            "acme",
            "--project",
            "research",
            "--reindex-missing",
            "--acknowledge-scope",
            "acme/research",
        ],
        obj=session,
    )

    assert accepted.exit_code == 0, accepted.output
    assert session.client.calls == [
        (
            "POST",
            "/api/v1/artifacts/inventory/reindex",
            {
                "json": {
                    "org_id": "acme",
                    "project_id": "research",
                    "cursor": None,
                    "limit": 100,
                    "acknowledge_scope": "acme/research",
                }
            },
        )
    ]
