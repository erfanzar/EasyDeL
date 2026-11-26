# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Existing bounded SSE lookback compatibility and genuine stored rollback gaps.

Repository allocation is commit-safe; the production concurrency proof lives in
``test_commit_safe_audit_cursors_actual_repository``. Explicit stored sequence gaps here pin
legacy lookback bytes without pretending StateSession accepts caller sequences.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import AuditEvent
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def _headers() -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": "operator",
    }


def _event(event_id: str, sequence: int) -> AuditEvent:
    return AuditEvent(
        id=event_id,
        actor="alice",
        org_id="acme",
        project_id="research",
        entity_type="workload",
        entity_id="wl-1",
        action="workload.submitted",
        sequence=sequence,
    )


def _stream(client: TestClient, *, after: int) -> list[dict]:
    response = client.get(
        "/api/v1/events/stream",
        params={"org_id": "acme", "project_id": "research", "after_sequence": after, "follow": False},
        headers=_headers(),
    )
    assert response.status_code == 200
    return [json.loads(block.split("data: ", 1)[1]) for block in response.text.split("\n\n") if "data: " in block]


def _client(repository: InMemoryRepository, tmp_path) -> TestClient:
    return TestClient(
        create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True,
                controller_enabled=False,
                artifact_store_uri=str(tmp_path / "artifacts"),
                artifact_token_key="a" * 32,
            ),
            repository=repository,
            service=ControlPlaneService(repository),
        )
    )


async def test_stored_gap_preserves_existing_bounded_reconnect_lookback(tmp_path) -> None:
    """Stored sequences, not supplied append sequences, exercise lookback compatibility."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        transaction._state.next_event_sequence = 101
        stored = await transaction.append_event(_event("evt-101", 101))
        assert stored.sequence == 101

    with _client(repository, tmp_path) as client:
        first = _stream(client, after=0)
        assert [event["id"] for event in first] == ["evt-101"]

        second = _stream(client, after=101)

    assert [event["sequence"] for event in second] == [101]
    assert [event["id"] for event in second] == ["evt-101"]


async def test_lookback_does_not_redeliver_within_one_stream(tmp_path) -> None:
    """Re-scanning behind the cursor must not duplicate what was already sent."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        for index in range(1, 6):
            await transaction.append_event(_event(f"evt-{index}", index))

    with _client(repository, tmp_path) as client:
        delivered = _stream(client, after=0)

    ids = [event["id"] for event in delivered]
    assert ids == sorted(ids, key=lambda value: int(value.split("-")[1]))
    assert len(ids) == len(set(ids)), "lookback re-reads rows; the follower must dedupe them"
