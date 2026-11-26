# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PostgreSQL execution of the merged artifact-lifecycle schema transition."""

from __future__ import annotations

import asyncio
import json

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import ManagedArtifactState
from eyvan.server.persistence.migrate import build_alembic_config
from eyvan.server.persistence.sqlalchemy import SCHEMA_REVISION, SqlAlchemyRepository

_BEFORE_BRANCHES = "0011_quota_grants"
_MERGE = "0013_merge_artifact_audit"
_URI = "eyvan://callables/" + "a" * 64


@pytest.fixture
async def database(postgres_url: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)
    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    try:
        yield engine
    finally:
        await engine.dispose()


def _config(engine):
    url = str(engine.url.render_as_string(hide_password=False))
    config = build_alembic_config(url)
    config.set_main_option("sqlalchemy.url", url)
    return config


async def _upgrade(engine, revision: str) -> None:
    await asyncio.to_thread(command.upgrade, _config(engine), revision)


async def _downgrade(engine, revision: str) -> None:
    await asyncio.to_thread(command.downgrade, _config(engine), revision)


async def _populate_pre_branch_state(engine) -> None:
    artifact = {
        "uri": _URI,
        "kind": "callables",
        "org_id": "acme",
        "project_id": "research",
        "size_bytes": 13,
        "sha256": "a" * 64,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    event = {
        "id": "evt-before-branches",
        "sequence": 0,
        "timestamp": "2026-01-02T03:04:05Z",
        "actor": "alice",
        "action": "artifact.indexed",
        "entity_type": "artifact",
        "entity_id": _URI,
        "org_id": "acme",
        "project_id": "research",
        "payload": {},
    }
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO managed_artifacts "
                "(uri, org_id, project_id, kind, size_bytes, created_at, data) "
                "VALUES (:uri, 'acme', 'research', 'callables', 13, "
                "TIMESTAMPTZ '2026-01-01 00:00:00+00', CAST(:data AS json))"
            ),
            {"uri": _URI, "data": json.dumps(artifact)},
        )
        await connection.execute(
            text(
                "INSERT INTO audit_events "
                "(id, org_id, project_id, entity_type, entity_id, data) "
                "VALUES ('evt-before-branches', 'acme', 'research', 'artifact', :uri, CAST(:data AS json))"
            ),
            {"uri": _URI, "data": json.dumps(event)},
        )


@pytest.mark.asyncio
async def test_populated_parallel_heads_merge_then_quarantine_legacy_artifacts(database) -> None:
    await _upgrade(database, _BEFORE_BRANCHES)
    await _populate_pre_branch_state(database)
    await _upgrade(database, "head")

    async with database.connect() as connection:
        versions = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalars().all()
        artifact = (
            await connection.execute(
                text(
                    "SELECT state, record_generation, object_version, size_bytes, "
                    "data ->> 'state', data ->> 'record_generation', data ->> 'quarantine_reason' "
                    "FROM managed_artifacts WHERE org_id='acme' AND project_id='research' AND uri=:uri"
                ),
                {"uri": _URI},
            )
        ).one()
        occurred_at = (
            await connection.execute(text("SELECT occurred_at FROM audit_events WHERE id='evt-before-branches'"))
        ).scalar_one()
    assert versions == [SCHEMA_REVISION]
    assert tuple(artifact) == (
        "quarantined",
        1,
        None,
        13,
        "quarantined",
        "1",
        "legacy object version has not been verified",
    )
    assert occurred_at.isoformat().startswith("2026-01-02T03:04:05")

    repository = SqlAlchemyRepository(database)
    await repository.open()
    async with repository.transaction() as transaction:
        record = await transaction.get_managed_artifact("acme", "research", _URI)
    assert record is not None
    assert record.state is ManagedArtifactState.QUARANTINED
    assert record.charged_bytes == 13


@pytest.mark.asyncio
async def test_untouched_quarantine_can_drop_only_the_lifecycle_revision(database) -> None:
    await _upgrade(database, _BEFORE_BRANCHES)
    await _populate_pre_branch_state(database)
    await _upgrade(database, "head")
    await _downgrade(database, _MERGE)

    async with database.connect() as connection:
        version = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
        columns = {
            row[0]
            for row in (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema='public' AND table_name='managed_artifacts'"
                    )
                )
            ).all()
        }
    assert version == _MERGE
    assert {"state", "record_generation", "object_version", "updated_at"}.isdisjoint(columns)


@pytest.mark.asyncio
async def test_downgrade_refuses_any_written_lifecycle_state_without_partial_ddl(database) -> None:
    await _upgrade(database, _BEFORE_BRANCHES)
    await _populate_pre_branch_state(database)
    await _upgrade(database, "head")
    async with database.begin() as connection:
        await connection.execute(
            text(
                "UPDATE managed_artifacts SET state='active', object_version='gcs:7', "
                "data=(data::jsonb || CAST(:patch AS jsonb))::json"
            ),
            {"patch": json.dumps({"state": "active", "object_version": "gcs:7"})},
        )

    with pytest.raises(DBAPIError, match="0014 downgrade refused"):
        await _downgrade(database, _MERGE)

    async with database.connect() as connection:
        version = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
        state = (await connection.execute(text("SELECT state FROM managed_artifacts"))).scalar_one()
        columns = {
            row[0]
            for row in (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema='public' AND table_name='managed_artifacts'"
                    )
                )
            ).all()
        }
    assert version == SCHEMA_REVISION
    assert state == "active"
    assert {"state", "record_generation", "object_version", "updated_at"} <= columns
