# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistence contracts: idempotency, the retention cutoff and environment precedence."""

from __future__ import annotations

import typing
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import AuditEvent, IdempotencyRecord, utcnow
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.migrate import build_alembic_config
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema


@pytest.fixture(params=("memory", "sqlite"))
async def repository(request: pytest.FixtureRequest, tmp_path) -> typing.AsyncIterator[object]:
    if request.param == "memory":
        yield InMemoryRepository()
        return
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'repository.db'}")
    await create_schema(engine)
    try:
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()


async def test_put_idempotency_keeps_the_first_binding(repository) -> None:
    """First binding wins on every backend, so a replay cannot replace it."""
    first = IdempotencyRecord(
        scope="s", operation="op", key="k", fingerprint="f1", resource_type="lease", resource_id="r1"
    )
    second = IdempotencyRecord(
        scope="s", operation="op", key="k", fingerprint="f2", resource_type="lease", resource_id="r2"
    )
    async with repository.transaction() as tx:
        await tx.put_idempotency(first)
        await tx.put_idempotency(second)
        stored = await tx.get_idempotency("s", "op", "k")
    assert stored == first


async def test_prune_events_pushes_the_cutoff_into_sql(repository) -> None:
    """Recent events at the sequence front must not consume the retention page."""
    now = utcnow()
    old = now - timedelta(days=2)
    async with repository.transaction() as tx:
        for index in range(3):
            await tx.append_event(
                AuditEvent(
                    id=f"evt-new-{index}",
                    actor="a",
                    org_id="acme",
                    project_id="research",
                    entity_type="x",
                    entity_id="e",
                    action="x.happened",
                    payload={},
                    timestamp=now,
                )
            )
        for index in range(2):
            await tx.append_event(
                AuditEvent(
                    id=f"evt-old-{index}",
                    actor="a",
                    org_id="acme",
                    project_id="research",
                    entity_type="x",
                    entity_id="e",
                    action="x.happened",
                    payload={},
                    timestamp=old,
                )
            )
    async with repository.transaction() as tx:
        removed = await tx.prune_events(before=now - timedelta(days=1), limit=2)
    assert removed == 2


def test_explicit_database_url_marks_the_config() -> None:
    """A per-invocation URL is flagged so env.py cannot override it with the env var."""
    explicit = build_alembic_config("postgresql+asyncpg://u:p@localhost/db2")
    ambient = build_alembic_config(None)
    assert explicit.attributes.get("eyvan_explicit_database_url") is True
    assert ambient.attributes.get("eyvan_explicit_database_url") is False
