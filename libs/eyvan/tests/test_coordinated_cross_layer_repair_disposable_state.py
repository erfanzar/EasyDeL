# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinated cross-layer repair contracts on disposable state."""

from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import OutboxRecord, OutboxState, utcnow
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_age_filtered_outbox_is_bounded_before_delivery_order_limit(tmp_path, backend) -> None:
    engine = None
    if backend == "sqlite":
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'outbox.db'}")
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
    else:
        repository = InMemoryRepository()
    now = utcnow()
    try:
        async with repository.transaction() as tx:
            for index in range(1001):
                await tx.put_outbox(
                    OutboxRecord(
                        id=f"fresh-{index:04}",
                        topic="capacity.apply",
                        aggregate_type="cell",
                        aggregate_id="c",
                        generation=1,
                        idempotency_key=f"fresh-{index}",
                        state=OutboxState.FAILED,
                        created_at=now,
                        available_at=now - timedelta(days=1),
                    )
                )
            await tx.put_outbox(
                OutboxRecord(
                    id="old",
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id="c",
                    generation=1,
                    idempotency_key="old",
                    state=OutboxState.FAILED,
                    created_at=now - timedelta(hours=2),
                    available_at=now + timedelta(hours=1),
                )
            )
        async with repository.transaction() as tx:
            rows = await tx.list_outbox(state=OutboxState.FAILED, created_before=now - timedelta(minutes=15), limit=1)
            assert [record.id for record in rows] == ["old"]
            default = await tx.list_outbox(state=OutboxState.FAILED, limit=1)
            assert default[0].id == "fresh-0000"
    finally:
        if engine is not None:
            await engine.dispose()
