# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retention: nothing used to be deleted, so these tables grew forever.

``session.delete`` appeared exactly once in the whole package (``delete_quota``).
``audit_events``, ``outbox``, and ``idempotency_keys`` therefore grew without
bound in every backend -- and an idempotency key was honored forever, so a
caller reusing one months later silently replayed the original response instead
of acting.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    AuditEvent,
    IdempotencyRecord,
    OutboxRecord,
    OutboxState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository


def _outbox(record_id: str, state: OutboxState, age_days: int) -> OutboxRecord:
    stamp = utcnow() - timedelta(days=age_days)
    return OutboxRecord(
        id=record_id,
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id="cell-a",
        idempotency_key=f"capacity:{record_id}",
        generation=1,
        state=state,
        payload={},
        created_at=stamp,
        updated_at=stamp,
    )


def _event(event_id: str, sequence: int, age_days: int) -> AuditEvent:
    return AuditEvent(
        id=event_id,
        actor="alice",
        org_id="acme",
        project_id="research",
        entity_type="workload",
        entity_id="wl-1",
        action="workload.submitted",
        sequence=sequence,
        timestamp=utcnow() - timedelta(days=age_days),
    )


def _idempotency(key: str, age_days: int) -> IdempotencyRecord:
    return IdempotencyRecord(
        scope="scope_a",
        operation="workload.submit",
        key=key,
        fingerprint="f" * 64,
        resource_type="workload",
        resource_id="wl-1",
        created_at=utcnow() - timedelta(days=age_days),
    )


async def test_only_settled_outbox_commands_are_pruned() -> None:
    """PENDING, PROCESSING, and FAILED rows are all still owed delivery."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_outbox(_outbox("done", OutboxState.SUCCEEDED, age_days=30))
        await transaction.put_outbox(_outbox("dropped", OutboxState.DISCARDED, age_days=30))
        await transaction.put_outbox(_outbox("waiting", OutboxState.PENDING, age_days=30))
        await transaction.put_outbox(_outbox("running", OutboxState.PROCESSING, age_days=30))
        await transaction.put_outbox(_outbox("retrying", OutboxState.FAILED, age_days=30))

    async with repository.transaction() as transaction:
        removed = await transaction.prune_outbox(before=utcnow() - timedelta(days=7))

    assert removed == 2
    async with repository.transaction() as transaction:
        surviving = {record.id for record in await transaction.list_outbox(limit=100)}
    assert surviving == {"waiting", "running", "retrying"}


async def test_recent_settled_commands_are_kept() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_outbox(_outbox("fresh", OutboxState.SUCCEEDED, age_days=1))

    async with repository.transaction() as transaction:
        assert await transaction.prune_outbox(before=utcnow() - timedelta(days=7)) == 0


async def test_pruning_audit_events_preserves_the_sequence_cursor_contract() -> None:
    """Sequences come from a monotonic counter, not list position.

    Deleting old rows must therefore leave surviving sequences untouched, so a
    follower's ``sequence > cursor`` cursor keeps working across a prune.
    """

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        for index in range(1, 6):
            await transaction.append_event(_event(f"evt-{index}", index, age_days=100 if index <= 2 else 1))

    async with repository.transaction() as transaction:
        removed = await transaction.prune_events(before=utcnow() - timedelta(days=30))

    assert removed == 2
    async with repository.transaction() as transaction:
        survivors = await transaction.list_events(after_sequence=0, limit=100)
    assert [event.sequence for event in survivors] == [3, 4, 5]

    # A new event must continue the counter rather than reuse a freed sequence.
    async with repository.transaction() as transaction:
        appended = await transaction.append_event(_event("evt-new", 0, age_days=0))
    assert appended.sequence > 5


async def test_expired_idempotency_keys_stop_replaying() -> None:
    """An indefinitely honored key silently replays instead of acting."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_idempotency(_idempotency("old-key", age_days=90))
        await transaction.put_idempotency(_idempotency("fresh-key", age_days=1))

    async with repository.transaction() as transaction:
        removed = await transaction.prune_idempotency(before=utcnow() - timedelta(days=30))

    assert removed == 1
    async with repository.transaction() as transaction:
        assert await transaction.get_idempotency("scope_a", "workload.submit", "old-key") is None
        assert await transaction.get_idempotency("scope_a", "workload.submit", "fresh-key") is not None


async def test_pruning_is_batched_so_one_pass_cannot_stall_the_loop() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        for index in range(10):
            await transaction.put_outbox(_outbox(f"done-{index}", OutboxState.SUCCEEDED, age_days=30))

    async with repository.transaction() as transaction:
        first = await transaction.prune_outbox(before=utcnow() - timedelta(days=7), limit=4)
    async with repository.transaction() as transaction:
        second = await transaction.prune_outbox(before=utcnow() - timedelta(days=7), limit=4)

    assert first == 4
    assert second == 4
    async with repository.transaction() as transaction:
        assert len(await transaction.list_outbox(limit=100)) == 2


async def test_pruning_the_local_backend_leaves_a_document_that_still_loads(tmp_path) -> None:
    """The local store is the *default* backend, and it serializes to one file.

    Its document-level invariant used to require sequences ``1..len(events)``,
    which pruning necessarily breaks -- so the write that followed a successful
    prune raised, the transaction rolled back, and nothing was ever deleted on
    the daemon most users actually run.
    """

    from eyvan.server.persistence.local import LocalJsonRepository

    path = tmp_path / "control-plane.json"
    repository = LocalJsonRepository(path)
    try:
        async with repository.transaction() as transaction:
            for index in range(1, 6):
                await transaction.append_event(_event(f"evt-{index}", index, age_days=100 if index <= 2 else 1))

        async with repository.transaction() as transaction:
            removed = await transaction.prune_events(before=utcnow() - timedelta(days=30))
        assert removed == 2
    finally:
        await repository.close()

    # The proof is a restart: a document that cannot be re-read is not pruned.
    reopened = LocalJsonRepository(path)
    try:
        async with reopened.transaction() as transaction:
            survivors = await transaction.list_events(after_sequence=0, limit=100)
            appended = await transaction.append_event(_event("evt-new", 0, age_days=0))
    finally:
        await reopened.close()

    assert [event.sequence for event in survivors] == [3, 4, 5]
    assert appended.sequence > 5


async def test_a_broken_retention_window_does_not_stop_the_others(tmp_path) -> None:
    """Aborting on the first failure is how "keys honored forever" stayed live.

    The windows are independent tables. When the audit prune raised, the loop
    never reached idempotency, so the headline replay bug the feature exists to
    fix was still present on any backend where audit pruning failed.
    """

    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.state import StateSession
    from eyvan.server.service import ControlPlaneService

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_outbox(_outbox("done", OutboxState.SUCCEEDED, age_days=90))
        await transaction.put_idempotency(_idempotency("old-key", age_days=90))

    async def exploding_prune_events(self, *, before, limit=1000):
        raise RuntimeError("audit table is wedged")

    original = StateSession.prune_events
    StateSession.prune_events = exploding_prune_events  # type: ignore[method-assign]
    try:
        controller = EyvanController(repository, ControlPlaneService(repository))
        with pytest.raises(RuntimeError, match="audit table is wedged"):
            await controller.prune_retained_state_once()
    finally:
        StateSession.prune_events = original  # type: ignore[method-assign]

    # Both surrounding windows still ran: outbox before the failure, and
    # idempotency -- which is ordered *after* it -- despite the raise.
    async with repository.transaction() as transaction:
        assert await transaction.list_outbox(limit=100) == []
        assert await transaction.get_idempotency("scope_a", "workload.submit", "old-key") is None


@pytest.mark.parametrize("field", ("outbox_retention_s", "audit_retention_s", "idempotency_retention_s"))
def test_a_negative_retention_window_is_rejected_at_startup(field: str) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        EyvanSettings(**{field: -1})


def test_zero_retention_disables_pruning_for_that_table() -> None:
    """Zero preserves the historical never-delete behaviour deliberately."""

    settings = EyvanSettings(audit_retention_s=0)

    assert settings.audit_retention_s == 0
    assert settings.outbox_retention_s > 0


def test_retention_windows_are_configurable_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_AUDIT_RETENTION_S", "1234")
    monkeypatch.setenv("EYVAN_RETENTION_BATCH_LIMIT", "7")

    settings = EyvanSettings.from_env()

    assert settings.audit_retention_s == 1234
    assert settings.retention_batch_limit == 7
