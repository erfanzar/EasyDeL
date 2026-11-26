# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Commit-safe audit cursors on actual repository transactions and SSE bytes."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.config import EyvanSettings
from eyvan.records import AuditEvent, Principal, Role, RoleGrant
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.base import TransientRepositoryError
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SCHEMA_REVISION, SqlAlchemyRepository, create_schema


@pytest.fixture
async def pg_repository(postgres_url):
    schema = f"audit_{uuid.uuid4().hex}"
    admin = create_async_engine(postgres_url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(postgres_url, connect_args={"server_settings": {"search_path": schema}})
    try:
        await create_schema(engine)
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


def _event(identifier):
    return AuditEvent(
        id=identifier,
        actor="alice",
        org_id="acme",
        project_id="research",
        entity_type="workload",
        entity_id="wl-1",
        action="workload.submitted",
    )


async def _frames(repository, tmp_path, after):
    app = create_app(
        repository=repository,
        settings=EyvanSettings(
            dev_auth_enabled=True,
            controller_enabled=False,
            artifact_store_uri=str(tmp_path / "artifacts"),
            artifact_token_key="a" * 32,
        ),
    )
    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/v1/events/stream")
    response = await endpoint(
        raw_request=None,
        org_id="acme",
        project_id="research",
        principal=principal,
        after_sequence=after,
        last_event_id=None,
        follow=False,
    )
    frames = [frame async for frame in response.body_iterator]
    assert all("event: workload.submitted\n" in frame and "id: " in frame for frame in frames)
    assert all("audit.gap" not in frame for frame in frames)
    return [json.loads(frame.split("data: ", 1)[1]) for frame in frames]


async def test_delayed_lower_commit_cannot_be_lost_beyond_sse_lookback(pg_repository, tmp_path):
    """A later writer must abort/retry rather than publish ahead of an allocated low row."""
    repository = pg_repository
    allocated = asyncio.Event()
    release = asyncio.Event()

    async def lower():
        async with repository.transaction() as tx:
            stored = await tx.append_event(_event("low"))
            allocated.set()
            await release.wait()
        return stored

    task = asyncio.create_task(lower())
    await asyncio.wait_for(allocated.wait(), 5)
    # A real sequence jump models arbitrarily many rolled-back allocations;
    # cursors must not depend on a finite lookback or contiguous sequences.
    async with repository.engine.begin() as connection:
        await connection.execute(text("SELECT setval(pg_get_serial_sequence('audit_events', 'sequence'), 1000)"))
    high = None
    try:
        try:
            async with repository.transaction() as tx:
                high = await tx.append_event(_event("high"))
        except TransientRepositoryError:
            pass
        before = await _frames(repository, tmp_path, 0)
    finally:
        release.set()
        low = await asyncio.wait_for(task, 5)
    cursor = max((event["sequence"] for event in before), default=0)
    if high is None:
        async with repository.transaction() as tx:
            high = await tx.append_event(_event("high"))
    after = await _frames(repository, tmp_path, cursor)
    assert low.sequence < high.sequence
    assert {event["id"] for event in before + after} == {"low", "high"}, (before, after)
    # The repository's exclusive cursor contract (no compensating lookback).
    async with repository.transaction() as tx:
        rows = await tx.list_events(after_sequence=0)
    assert [event.id for event in rows] == ["low", "high"]


async def test_rollback_gap_does_not_stall_or_duplicate_pg_followers(pg_repository, tmp_path):
    repository = pg_repository
    with pytest.raises(ValueError, match="rollback"):
        async with repository.transaction() as tx:
            rolled_back = await tx.append_event(_event("rolled-back"))
            raise ValueError("rollback")
    async with repository.transaction() as tx:
        first = await tx.append_event(_event("first"))
        second = await tx.append_event(_event("second"))
    assert first.sequence > rolled_back.sequence
    events = await repository.wait_for_events(after_sequence=rolled_back.sequence, timeout=0.1)
    assert [event.id for event in events] == ["first", "second"]
    assert await repository.wait_for_events(after_sequence=second.sequence, timeout=0.01) == []
    frames = await _frames(repository, tmp_path, 0)
    assert [event["id"] for event in frames] == ["first", "second"]
    assert len({event["sequence"] for event in frames}) == 2


async def test_audit_contention_never_waits_under_an_inverted_domain_lock(pg_repository):
    repository = pg_repository
    audit_held = asyncio.Event()
    domain_held = asyncio.Event()

    async def audit_owner():
        async with repository.transaction() as tx:
            event = await tx.append_event(_event("owner"))
            audit_held.set()
            await domain_held.wait()
            revision = await tx.session.scalar(text("SELECT revision FROM schema_revisions FOR UPDATE"))
            assert revision == SCHEMA_REVISION
        return event

    async def domain_owner():
        await audit_held.wait()
        with pytest.raises(TransientRepositoryError, match="audit sequence allocation"):
            async with repository.transaction() as tx:
                await tx.session.execute(text("UPDATE schema_revisions SET revision = 'uncommitted'"))
                domain_held.set()
                await asyncio.sleep(0.05)
                await tx.append_event(_event("must-retry"))

    await asyncio.wait_for(asyncio.gather(audit_owner(), domain_owner()), timeout=5)
    async with repository.transaction() as tx:
        assert [event.id for event in await tx.list_events()] == ["owner"]
        assert await tx.session.scalar(text("SELECT revision FROM schema_revisions")) == SCHEMA_REVISION
        # A different transaction can acquire the released allocation lock.
        await tx.append_event(_event("retried"))
    assert [event.id for event in await repository.wait_for_events(after_sequence=0, timeout=0)] == ["owner", "retried"]


@pytest.mark.parametrize("backend", ["memory", "local"])
async def test_state_repositories_serialize_allocation_with_commit(backend, tmp_path):
    repository = InMemoryRepository() if backend == "memory" else LocalJsonRepository(tmp_path / "state.json")
    await repository.open()
    allocated = asyncio.Event()
    release = asyncio.Event()

    async def lower():
        async with repository.transaction() as tx:
            stored = await tx.append_event(_event("low"))
            allocated.set()
            await release.wait()
        return stored

    async def higher():
        async with repository.transaction() as tx:
            return await tx.append_event(_event("high"))

    low_task = asyncio.create_task(lower())
    await allocated.wait()
    high_task = asyncio.create_task(higher())
    await asyncio.sleep(0)
    assert not high_task.done()
    release.set()
    low, high = await asyncio.gather(low_task, high_task)
    assert low.sequence < high.sequence
    assert [event["id"] for event in await _frames(repository, tmp_path, 0)] == ["low", "high"]
    await repository.close()
    if backend == "local":
        restored = LocalJsonRepository(tmp_path / "state.json")
        await restored.open()
        try:
            assert [event.id for event in await restored.wait_for_events(after_sequence=0, timeout=0)] == ["low", "high"]
        finally:
            await restored.close()
