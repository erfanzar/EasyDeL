# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI coverage for cell, lease, and preemption operation commands."""

from __future__ import annotations

import json
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from click.testing import CliRunner

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.operations import (
    delete_cell,
    drain_cell,
    force_finalize_preemption,
    list_cells,
    list_leases,
)
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContextConfig, FileContextStore

QueryValue: tp.TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """One captured control-plane request."""

    method: str
    path: str
    params: dict[str, QueryValue] | None
    json: JsonObject | None
    headers: dict[str, str] | None


@dataclass
class FakeOperationsClient:
    """Fake API client returning canned payloads while recording request shapes."""

    responses: list[JsonValue]
    requests: list[RecordedRequest] = field(default_factory=list)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, QueryValue] | None = None,
        json: JsonObject | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        """Record the request and return the next canned response."""

        self.requests.append(
            RecordedRequest(
                method=method,
                path=path,
                params=dict(params) if params is not None else None,
                json=json,
                headers=dict(headers) if headers is not None else None,
            )
        )
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {path}")
        return self.responses.pop(0)

    def request_text(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, QueryValue] | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> str:
        """Reject text endpoints because operation commands use JSON only."""

        del method, path, params, headers
        raise AssertionError("operation commands must not request text")

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        """Reject identity refreshes because operation commands never re-authenticate."""

        del headers
        raise AssertionError("operation commands must not refresh identity")


def _session(tmp_path: Path, client: FakeOperationsClient) -> CliSession:
    """Build one CLI session wired to the fake control-plane client."""

    return CliSession(
        client=client,
        identity=ClientIdentity(dev_user="alice"),
        context_store=FileContextStore(tmp_path / "contexts.json"),
        context_config=EyvanContextConfig(),
        base_url="http://127.0.0.1:8080",
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
    )


def _only_request(client: FakeOperationsClient) -> RecordedRequest:
    assert len(client.requests) == 1
    return client.requests[0]


def _idempotency_key(request: RecordedRequest) -> str:
    assert request.headers is not None
    return request.headers["Idempotency-Key"]


def test_cells_lists_registered_cells_as_json(tmp_path: Path) -> None:
    cells = [
        {"id": "cell-b", "state": "draining"},
        {"id": "cell-a", "state": "ready"},
    ]
    client = FakeOperationsClient(responses=[cells])

    result = CliRunner().invoke(list_cells, ["--json"], obj=_session(tmp_path, client))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == cells
    assert _only_request(client) == RecordedRequest("GET", "/api/v1/cells", None, None, None)


def test_cells_renders_a_table_for_a_person(tmp_path: Path) -> None:
    """The same fetch, read rather than parsed. A cell record is thirty keys and
    a fleet of them is a screenful per machine."""

    client = FakeOperationsClient(responses=[[{"id": "cell-a", "state": "ready", "zone": "us-east5-a"}]])

    result = CliRunner().invoke(list_cells, [], obj=_session(tmp_path, client))

    assert result.exit_code == 0, result.output
    assert "cell-a" in result.output
    assert "us-east5-a" in result.output
    assert '"id"' not in result.output, "the record is what --json is for"


def test_cell_drain_posts_with_a_generated_operation_scoped_idempotency_key(tmp_path: Path) -> None:
    client = FakeOperationsClient(responses=[{"id": "cell-9", "state": "draining"}])

    result = CliRunner().invoke(drain_cell, ["cell-9"], obj=_session(tmp_path, client))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"id": "cell-9", "state": "draining"}
    request = _only_request(client)
    assert request.method == "POST"
    assert request.path == "/api/v1/cells/cell-9/drain"
    assert request.params is None
    assert request.json is None
    key = _idempotency_key(request)
    assert key.startswith("cli-drain-")
    suffix = key.removeprefix("cli-drain-")
    assert len(suffix) == 32
    int(suffix, 16)


def test_cell_drain_forwards_an_explicit_idempotency_key(tmp_path: Path) -> None:
    client = FakeOperationsClient(responses=[{"id": "cell-9", "state": "draining"}])

    result = CliRunner().invoke(
        drain_cell,
        ["cell-9", "--idempotency-key", "retry-drain-17"],
        obj=_session(tmp_path, client),
    )

    assert result.exit_code == 0, result.output
    assert _idempotency_key(_only_request(client)) == "retry-drain-17"


def test_repeated_drains_never_reuse_a_generated_idempotency_key(tmp_path: Path) -> None:
    client = FakeOperationsClient(responses=[{"state": "draining"}, {"state": "draining"}])
    session = _session(tmp_path, client)

    first = CliRunner().invoke(drain_cell, ["cell-9"], obj=session)
    second = CliRunner().invoke(drain_cell, ["cell-9"], obj=session)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert len(client.requests) == 2
    assert _idempotency_key(client.requests[0]) != _idempotency_key(client.requests[1])


def test_cell_delete_issues_an_idempotent_delete(tmp_path: Path) -> None:
    client = FakeOperationsClient(responses=[{"id": "cell-9", "state": "deleted"}])

    result = CliRunner().invoke(
        delete_cell,
        ["cell-9", "--idempotency-key", "retry-delete-3"],
        obj=_session(tmp_path, client),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"id": "cell-9", "state": "deleted"}
    request = _only_request(client)
    assert request.method == "DELETE"
    assert request.path == "/api/v1/cells/cell-9"
    assert request.json is None
    assert _idempotency_key(request) == "retry-delete-3"


def test_leases_scopes_the_query_to_org_and_project(tmp_path: Path) -> None:
    leases = [{"id": "lease-1", "cell_id": "cell-9"}]
    client = FakeOperationsClient(responses=[leases])

    result = CliRunner().invoke(
        list_leases,
        ["--org", "acme", "--project", "research"],
        obj=_session(tmp_path, client),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == leases
    request = _only_request(client)
    assert request.method == "GET"
    assert request.path == "/api/v1/leases"
    assert request.params == {"org_id": "acme", "project_id": "research"}
    assert request.headers is None


def test_leases_requires_an_explicit_org_and_project(tmp_path: Path) -> None:
    client = FakeOperationsClient(responses=[])

    result = CliRunner().invoke(list_leases, ["--org", "acme"], obj=_session(tmp_path, client))

    assert result.exit_code != 0
    assert "--project" in result.output
    assert client.requests == []


def test_preemption_force_finalize_posts_the_reason_with_idempotency(tmp_path: Path) -> None:
    client = FakeOperationsClient(responses=[{"command_id": "cmd-1", "state": "finalized"}])

    result = CliRunner().invoke(
        force_finalize_preemption,
        ["cmd-1", "--reason", "victims stopped", "--idempotency-key", "retry-finalize-5"],
        obj=_session(tmp_path, client),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"command_id": "cmd-1", "state": "finalized"}
    request = _only_request(client)
    assert request.method == "POST"
    assert request.path == "/api/v1/preemptions/cmd-1/force-finalize"
    assert request.json == {"reason": "victims stopped"}
    assert _idempotency_key(request) == "retry-finalize-5"


def test_preemption_force_finalize_generates_a_scoped_key_and_requires_a_reason(tmp_path: Path) -> None:
    client = FakeOperationsClient(responses=[{"command_id": "cmd-2", "state": "finalized"}])

    missing_reason = CliRunner().invoke(force_finalize_preemption, ["cmd-2"], obj=_session(tmp_path, client))
    finalized = CliRunner().invoke(
        force_finalize_preemption,
        ["cmd-2", "--reason", "operator attestation"],
        obj=_session(tmp_path, client),
    )

    assert missing_reason.exit_code != 0
    assert "--reason" in missing_reason.output
    assert finalized.exit_code == 0, finalized.output
    request = _only_request(client)
    assert _idempotency_key(request).startswith("cli-force-finalize-")
