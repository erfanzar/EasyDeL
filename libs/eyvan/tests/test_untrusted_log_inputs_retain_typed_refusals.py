# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Untrusted log inputs retain typed refusals and readable messages."""

from __future__ import annotations

import asyncio
import base64

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
from eyvan.server.api.app import create_app
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.worker.runtime.log_records import (
    ARCHIVE_OFFSET_PREFIX,
    ARCHIVE_POSITION_MALFORMED,
    CURSOR_INVALID_ATTEMPT,
    CURSOR_INVALID_RANK,
    CURSOR_MALFORMED,
    ArchiveGap,
    ArchiveOffset,
    CursorFresh,
    CursorInvalid,
    CursorResume,
    decode_log_cursor,
    log_page_from_json,
    resolve_archive_position,
    split_log_cursor,
)


def _deep_cursor() -> str:
    raw = b'{"version":2,"attempt":1,"cursors":' + b"[" * 20_000 + b"0" + b"]" * 20_000 + b"}"
    return "eyvan-log-v2." + base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize(
    ("cursor", "reason"),
    [("9" * 5000 + ":c1", CURSOR_INVALID_ATTEMPT), ("1:" + "9" * 5000 + "=c1", CURSOR_INVALID_RANK)],
    ids=["attempt", "rank"],
)
def test_long_decimal_cursor_is_a_typed_refusal(cursor: str, reason: str) -> None:
    outcome = decode_log_cursor(cursor)
    assert isinstance(outcome, CursorInvalid)
    assert outcome.reason == reason


def test_long_archive_offset_is_a_typed_gap() -> None:
    outcome = resolve_archive_position(ARCHIVE_OFFSET_PREFIX + "9" * 5000, ["c1"], rank=0)
    assert isinstance(outcome, ArchiveGap)
    assert outcome.reason == ARCHIVE_POSITION_MALFORMED


def test_excessively_nested_json_cursor_is_a_typed_refusal() -> None:
    outcome = decode_log_cursor(_deep_cursor())
    assert isinstance(outcome, CursorInvalid)
    assert outcome.reason == CURSOR_MALFORMED


def test_ordinary_cursor_and_archive_controls() -> None:
    assert decode_log_cursor("") == CursorFresh()
    outcome = decode_log_cursor("1:c1")
    assert isinstance(outcome, CursorResume)
    assert outcome.attempt == 1
    assert outcome.cursors == {0: "c1"}
    assert resolve_archive_position(ARCHIVE_OFFSET_PREFIX + "1", ["c1"], rank=0) == ArchiveOffset(rank=0, offset=1)


@pytest.mark.parametrize("route", ["log-page", "logs/stream"])
@pytest.mark.parametrize(
    "cursor",
    ["9" * 5000 + ":c1", "1:" + "9" * 5000 + "=c1", _deep_cursor()],
    ids=["attempt", "rank", "deep-json"],
)
def test_authorized_log_routes_refuse_hostile_cursor_before_serving(route: str, cursor: str) -> None:
    app = create_app(
        settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32), repository=InMemoryRepository()
    )

    async def store() -> None:
        async with app.state.service.repository.transaction() as transaction:
            await transaction.put_workload(
                WorkloadRecord(
                    id="wl-writer-115",
                    org_id="acme",
                    project_id="research",
                    spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                    created_by="alice",
                    state=WorkloadState.RUNNING,
                    attempt=1,
                    remote_id="remote-115",
                )
            )

    asyncio.run(store())
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get(
        f"/api/v1/workloads/wl-writer-115/{route}",
        params={"after": cursor, "follow": "false"},
        headers={
            "X-Eyvan-User": "alice",
            "X-Eyvan-Roles": "operator",
            "X-Eyvan-Org": "acme",
            "X-Eyvan-Project": "research",
        },
    )
    assert response.status_code == 400, response.text
    assert isinstance(response.json()["detail"], str)
    assert "log cursor" in response.json()["detail"]


_MALFORMED_PAGES = [
    {"next_cursor": "advanced"},
    {"records": "bad", "next_cursor": "advanced"},
    {"records": ["not-an-object"], "next_cursor": "advanced"},
    {"records": [{"message": "readable"}, "bad"], "next_cursor": "advanced"},
    {"records": [{}], "next_cursor": "advanced"},
    {"records": [{"message": ["bad"]}], "next_cursor": "advanced"},
]
_MALFORMED_IDS = ["missing-records", "wrong-records-type", "bad-entry", "mixed", "missing-message", "bad-message"]


@pytest.mark.parametrize("payload", _MALFORMED_PAGES, ids=_MALFORMED_IDS)
def test_malformed_wire_page_refuses_unsafe_advance(payload: dict) -> None:
    from eyvan.server.controller.observation import _log_page_from_document

    page = log_page_from_json(payload)
    assert page.unavailable is True
    assert page.next_cursor == ""
    assert page.detail
    adapted = _log_page_from_document(payload, rank=3, runtime_process_count=2)
    assert adapted.unavailable is True
    assert adapted.next_cursor == ""
    assert adapted.truncated_ranks == (3, 4)
    assert adapted.detail


async def _page_with_agent(payload: dict, *, after: str = "1:before") -> dict:
    from datetime import timedelta

    from eyvan.records import CellRecord, CellState, LeaseRecord, LeaseState, ResourceGang, ResourceVector, utcnow
    from eyvan.server.controller import EyvanController
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    class Agent:
        def log_page(self, cell_id, remote_id, *, after="", limit=200):
            assert (cell_id, remote_id, after) == ("cell-115", "remote-115", "before")
            return payload

    repository = InMemoryRepository()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=EyvanSettings(),
        agent_factory=lambda _cell: Agent(),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-115",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8),
                allocatable=ResourceVector(cpu=8),
                endpoint="http://agent:8090",
            )
        )
        await transaction.put_lease(
            LeaseRecord(
                id="lease-115",
                workload_id="wl-115",
                cell_id="cell-115",
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1)),
                generation=1,
                fencing_token="token-115",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-115",
                org_id="acme",
                project_id="research",
                created_by="alice",
                spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                state=WorkloadState.RUNNING,
                attempt=1,
                lease_id="lease-115",
                remote_id="remote-115",
            )
        )
    return await controller.workload_log_page("wl-115", after=after)


@pytest.mark.parametrize("payload", _MALFORMED_PAGES, ids=_MALFORMED_IDS)
async def test_controller_preserves_previous_cell_cursor_for_malformed_reply(payload: dict) -> None:
    page = await _page_with_agent(payload)
    assert split_log_cursor(page["next_cursor"]) == (1, {0: "before"})
    assert page["unavailable"] is True
    assert page["truncated_ranks"] == [0]
    assert page["detail"]


async def test_valid_empty_page_can_advance_while_invalid_rank_cannot() -> None:
    valid = await _page_with_agent({"records": [], "next_cursor": "advanced"})
    assert split_log_cursor(valid["next_cursor"]) == (1, {0: "advanced"})
    assert valid["unavailable"] is False
    invalid = await _page_with_agent({"records": [{"message": "bad rank", "rank": -1}], "next_cursor": "advanced"})
    assert split_log_cursor(invalid["next_cursor"]) == (1, {0: "before"})
    assert invalid["truncated_ranks"] == [0]


_MIXED_TIMESTAMPS = [
    {"message": "naive", "timestamp": "2026-08-01T12:00:01", "cursor": "c-naive"},
    {"message": "aware", "timestamp": "2026-08-01T12:00:02+02:00", "cursor": "c-aware"},
    {"message": "missing", "cursor": "c-missing"},
    {"message": "malformed", "timestamp": "not-a-date", "cursor": "c-malformed"},
]


def test_timestamp_boundary_preserves_unknown_provenance_and_all_messages() -> None:
    from datetime import timedelta

    from eyvan.server.controller.observation import _log_page_from_document
    from eyvan.worker.runtime.log_records import merge_pages

    page = _log_page_from_document({"records": _MIXED_TIMESTAMPS}, rank=0)
    merged = merge_pages({0: page})
    assert {record.message for record in merged.records} == {"naive", "aware", "missing", "malformed"}
    by_message = {record.message: record for record in merged.records}
    assert by_message["naive"].timestamp is None
    assert by_message["missing"].timestamp is None
    assert by_message["malformed"].timestamp is None
    assert by_message["aware"].timestamp.utcoffset() == timedelta(hours=2)
    assert merged.unavailable is False


async def test_live_controller_merges_naive_aware_and_missing_timestamps() -> None:
    page = await _page_with_agent({"records": _MIXED_TIMESTAMPS, "next_cursor": "advanced"})
    assert {record["message"] for record in page["records"]} == {"naive", "aware", "missing", "malformed"}
    assert next(record for record in page["records"] if record["message"] == "naive")["timestamp"] is None
    assert split_log_cursor(page["next_cursor"]) == (1, {0: "advanced"})
    assert page["unavailable"] is False


def test_archive_controller_window_merges_timestamp_variants_without_losing_messages() -> None:
    from eyvan.server.controller.observation import _archived_rank_records, _ArchiveReplay

    app = create_app(
        settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32), repository=InMemoryRepository()
    )
    workload = WorkloadRecord(
        id="wl-115",
        org_id="acme",
        project_id="research",
        created_by="alice",
        attempt=1,
        spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
    )
    replay = _ArchiveReplay(
        ranks={rank: _archived_rank_records([entry], rank=rank) for rank, entry in enumerate(_MIXED_TIMESTAMPS)}
    )
    page = app.state.controller._archive_window(workload, replay, positions={}, position_gap="", limit=200)
    assert {record.message for record in page.records} == {"naive", "aware", "missing", "malformed"}
    assert next(record for record in page.records if record.message == "naive").timestamp is None
    assert page.unavailable is False
    assert split_log_cursor(page.next_cursor) == (1, {rank: ARCHIVE_OFFSET_PREFIX + "1" for rank in range(4)})
