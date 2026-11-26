# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Same-aggregate delivery ordering must hold across replicas, not just within one.

Ordering a cell's drain behind its apply lived only in a process-local
``asyncio.Lock`` map, so it never reached the active-active case: nothing
stopped replica B from claiming and delivering a cell's next command while
replica A was minutes inside the current one. The claimable predicate now also
withholds a command while its aggregate has another row PROCESSING under a live
claim, so a peer's claim fences its own aggregate.
"""

from __future__ import annotations

import typing as tp
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from eyvan.records import OutboxRecord, OutboxState, utcnow
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema


@pytest.fixture
async def repository(tmp_path: tp.Any) -> tp.AsyncIterator[SqlAlchemyRepository]:
    engine: AsyncEngine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'eyvan.db'}")
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    # Production claims run at READ COMMITTED because SKIP LOCKED carries the
    # serialization on PostgreSQL. SQLite's dialect has no such level, and the
    # isolation is not what is under test here -- the claimable predicate is.
    repository._claim_sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield repository
    finally:
        await engine.dispose()


def _command(record_id: str, aggregate_id: str) -> OutboxRecord:
    return OutboxRecord(
        id=record_id,
        topic="workload.cancel",
        aggregate_type="workload",
        aggregate_id=aggregate_id,
        idempotency_key=f"cancel:{aggregate_id}:{record_id}",
        generation=1,
        payload={},
    )


async def test_a_live_claim_holds_back_the_aggregates_next_command(repository: SqlAlchemyRepository) -> None:
    async with repository.transaction() as transaction:
        for record_id, aggregate_id in (
            ("cmd-apply", "wl-1"),
            ("cmd-drain", "wl-1"),
            ("cmd-cancel", "wl-2"),
        ):
            await transaction.put_outbox(_command(record_id, aggregate_id))

    replica_a = await repository.claim_outbox(worker_id="replica-a", limit=1)
    replica_b = await repository.claim_outbox(worker_id="replica-b", limit=10)

    assert [command.id for command in replica_a] == ["cmd-apply"]
    # Replica B may take the unrelated aggregate, never wl-1's next command:
    # the drain must not overtake the apply replica A is still inside.
    assert [command.id for command in replica_b] == ["cmd-cancel"]
    assert await repository.snapshot_claimable_outbox_ids(limit=10) == ()


async def test_an_expired_claim_reclaims_the_head_before_the_aggregate_tail(repository: SqlAlchemyRepository) -> None:
    """A dead replica's head is retried without letting its tail overtake it."""

    async with repository.transaction() as transaction:
        for record_id, aggregate_id in (("cmd-apply", "wl-1"), ("cmd-drain", "wl-1")):
            await transaction.put_outbox(_command(record_id, aggregate_id))

    held = await repository.claim_outbox(worker_id="replica-a", limit=1)
    assert [command.id for command in held] == ["cmd-apply"]
    async with repository.transaction() as transaction:
        current = await transaction.get_outbox("cmd-apply")
        assert current is not None and current.claimed_until is not None
        await transaction.put_outbox(current.model_copy(update={"claimed_until": utcnow() - timedelta(seconds=1)}))

    recovered = await repository.claim_outbox(worker_id="replica-b", limit=10)

    assert [command.id for command in recovered] == ["cmd-apply"]
    assert recovered[0].claimed_by == "replica-b"
    assert await repository.claim_outbox(worker_id="replica-c", limit=10) == []
    async with repository.transaction() as transaction:
        current = await transaction.get_outbox("cmd-apply")
        assert current is not None
        await transaction.put_outbox(
            current.model_copy(update={"state": OutboxState.SUCCEEDED, "claimed_by": None, "claimed_until": None})
        )
    tail = await repository.claim_outbox(worker_id="replica-b", limit=10)
    assert [command.id for command in tail] == ["cmd-drain"]


async def test_a_settled_command_releases_its_hold_on_the_aggregate(repository: SqlAlchemyRepository) -> None:
    """Once the first command finishes, the next becomes deliverable again."""

    async with repository.transaction() as transaction:
        for record_id, aggregate_id in (("cmd-apply", "wl-1"), ("cmd-drain", "wl-1")):
            await transaction.put_outbox(_command(record_id, aggregate_id))

    held = await repository.claim_outbox(worker_id="replica-a", limit=1)
    assert [command.id for command in held] == ["cmd-apply"]
    async with repository.transaction() as transaction:
        current = await transaction.get_outbox("cmd-apply")
        assert current is not None
        await transaction.put_outbox(
            current.model_copy(update={"state": OutboxState.SUCCEEDED, "claimed_by": None, "claimed_until": None})
        )

    replica_b = await repository.claim_outbox(worker_id="replica-b", limit=10)

    assert [command.id for command in replica_b] == ["cmd-drain"]
