# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Populated 0011 <-> 0012 execution against PostgreSQL.

These tests deliberately do not have a SQLite substitute: table locks,
transactional DDL, jsonb updates, partial indexes, and named primary-key
constraints are the migration's production semantics. The shared fixture skips
when PostgreSQL is unavailable and turns that skip into a failure when
EYVAN_REQUIRE_POSTGRES is set.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import artifact_reference_id
from eyvan.server.persistence.migrate import build_alembic_config
from eyvan.server.persistence.sqlalchemy import create_schema

_N_MINUS_1 = "0011_quota_grants"
_N = "0012_scoped_artifact_identity"
_URI = "eyvan://callables/" + "a" * 64


@pytest.fixture
async def database(postgres_url: str, monkeypatch: pytest.MonkeyPatch):
    # Packaged env.py intentionally lets EYVAN_DATABASE_URL override Config.
    # A migration test must never inherit a deployment URL over its disposable
    # fixture database.
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


def _legacy_id(uri: str, workload_id: str) -> str:
    digest = hashlib.sha256(f"{uri}\n{workload_id}".encode()).hexdigest()
    return f"ref-{digest[:40]}"


async def _insert_workload(connection, *, workload_id: str, org_id: str, project_id: str) -> None:
    data = {
        "id": workload_id,
        "org_id": org_id,
        "project_id": project_id,
    }
    await connection.execute(
        text(
            "INSERT INTO workloads (id, org_id, project_id, state, data) "
            "VALUES (:id, :org_id, :project_id, 'pending', CAST(:data AS json))"
        ),
        {"id": workload_id, "org_id": org_id, "project_id": project_id, "data": json.dumps(data)},
    )


async def _insert_managed(
    connection,
    *,
    org_id: str | None,
    project_id: str | None,
    json_org_id: str | None = None,
    json_project_id: str | None = None,
    uri: str = _URI,
    size: int = 13,
    lifecycle: bool = False,
) -> None:
    data = {
        "uri": uri,
        "kind": "callables",
        "org_id": org_id if json_org_id is None else json_org_id,
        "project_id": project_id if json_project_id is None else json_project_id,
        "size_bytes": size,
    }
    if lifecycle:
        data.update(
            {
                "state": "quarantined",
                "record_generation": 1,
                "object_version": None,
                "quarantine_reason": "legacy object version has not been verified",
            }
        )
        statement = (
            "INSERT INTO managed_artifacts "
            "(uri, org_id, project_id, kind, state, record_generation, object_version, size_bytes, "
            "created_at, updated_at, data) "
            "VALUES (:uri, :org_id, :project_id, 'callables', 'quarantined', 1, NULL, :size, "
            "now(), now(), CAST(:data AS json))"
        )
    else:
        statement = (
            "INSERT INTO managed_artifacts (uri, org_id, project_id, kind, size_bytes, created_at, data) "
            "VALUES (:uri, :org_id, :project_id, 'callables', :size, now(), CAST(:data AS json))"
        )
    await connection.execute(
        text(statement),
        {
            "uri": uri,
            "org_id": org_id,
            "project_id": project_id,
            "size": size,
            "data": json.dumps(data),
        },
    )


async def _insert_reference(
    connection,
    *,
    workload_id: str,
    org_id: str | None,
    project_id: str | None,
    uri: str = _URI,
) -> str:
    reference_id = _legacy_id(uri, workload_id)
    data = {
        "id": reference_id,
        "uri": uri,
        "workload_id": workload_id,
        "org_id": org_id,
        "project_id": project_id,
    }
    await connection.execute(
        text(
            "INSERT INTO artifact_references (id, uri, workload_id, released_at, created_at, data) "
            "VALUES (:id, :uri, :workload_id, NULL, now(), CAST(:data AS json))"
        ),
        {"id": reference_id, "uri": uri, "workload_id": workload_id, "data": json.dumps(data)},
    )
    return reference_id


@pytest.mark.asyncio
async def test_populated_upgrade_and_downgrade_move_typed_and_json_reference_ids_together(database) -> None:
    await _upgrade(database, _N_MINUS_1)
    async with database.begin() as connection:
        await _insert_workload(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        await _insert_managed(connection, org_id="acme", project_id="alpha")
        legacy_id = await _insert_reference(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")

    await _upgrade(database, _N)
    scoped_id = artifact_reference_id("acme", "alpha", _URI, "wl-alpha")
    async with database.connect() as connection:
        upgraded = (
            await connection.execute(text("SELECT id, data ->> 'id', org_id, project_id FROM artifact_references"))
        ).one()
        version = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
    assert tuple(upgraded) == (scoped_id, scoped_id, "acme", "alpha")
    assert upgraded[0] != legacy_id
    assert version == _N

    await _downgrade(database, _N_MINUS_1)
    async with database.connect() as connection:
        downgraded = (await connection.execute(text("SELECT id, data ->> 'id' FROM artifact_references"))).one()
        columns = {
            row[0]
            for row in (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema='public' AND table_name='artifact_references'"
                    )
                )
            ).all()
        }
    assert tuple(downgraded) == (legacy_id, legacy_id)
    assert {"org_id", "project_id"}.isdisjoint(columns)


@pytest.mark.parametrize(
    "case",
    ("managed_missing", "managed_mismatched", "reference_missing", "reference_workload_mismatched"),
)
@pytest.mark.asyncio
async def test_upgrade_rejects_missing_or_mismatched_scope_without_partial_schema_change(database, case: str) -> None:
    await _upgrade(database, _N_MINUS_1)
    async with database.begin() as connection:
        await _insert_workload(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        if case == "managed_missing":
            await _insert_managed(connection, org_id=None, project_id="alpha")
            await _insert_reference(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        elif case == "managed_mismatched":
            await _insert_managed(
                connection,
                org_id="acme",
                project_id="alpha",
                json_project_id="different",
            )
            await _insert_reference(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        elif case == "reference_missing":
            await _insert_managed(connection, org_id="acme", project_id="alpha")
            await _insert_reference(connection, workload_id="wl-alpha", org_id=None, project_id="alpha")
        else:
            await _insert_managed(connection, org_id="acme", project_id="alpha")
            await _insert_reference(connection, workload_id="wl-alpha", org_id="acme", project_id="beta")

    with pytest.raises(DBAPIError, match="0012 preflight"):
        await _upgrade(database, _N)

    async with database.connect() as connection:
        version = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
        columns = {
            row[0]
            for row in (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema='public' AND table_name='artifact_references'"
                    )
                )
            ).all()
        }
    assert version == _N_MINUS_1
    assert {"org_id", "project_id"}.isdisjoint(columns), "transactional DDL was only partly rolled back"


@pytest.mark.asyncio
async def test_cross_scope_references_are_preserved_without_fabricating_a_managed_row(database) -> None:
    await _upgrade(database, _N_MINUS_1)
    async with database.begin() as connection:
        await _insert_workload(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        await _insert_workload(connection, workload_id="wl-beta", org_id="acme", project_id="beta")
        await _insert_managed(connection, org_id="acme", project_id="alpha")
        await _insert_reference(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        await _insert_reference(connection, workload_id="wl-beta", org_id="acme", project_id="beta")

    await _upgrade(database, _N)

    async with database.connect() as connection:
        references = (
            await connection.execute(
                text(
                    "SELECT org_id, project_id, workload_id, id, data ->> 'id' "
                    "FROM artifact_references ORDER BY project_id"
                )
            )
        ).all()
        managed = (await connection.execute(text("SELECT org_id, project_id, uri FROM managed_artifacts"))).all()
    assert [(row.org_id, row.project_id, row.workload_id) for row in references] == [
        ("acme", "alpha", "wl-alpha"),
        ("acme", "beta", "wl-beta"),
    ]
    assert all(row.id == row[4] for row in references)
    assert [tuple(row) for row in managed] == [("acme", "alpha", _URI)]


@pytest.mark.asyncio
async def test_downgrade_refuses_duplicate_scoped_uris_before_changing_schema(database) -> None:
    await _upgrade(database, _N_MINUS_1)
    async with database.begin() as connection:
        await _insert_workload(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        await _insert_managed(connection, org_id="acme", project_id="alpha")
        await _insert_reference(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
    await _upgrade(database, _N)
    async with database.begin() as connection:
        await _insert_managed(connection, org_id="acme", project_id="beta", size=21)

    with pytest.raises(DBAPIError, match="0012 downgrade refused"):
        await _downgrade(database, _N_MINUS_1)

    async with database.connect() as connection:
        version = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
        count = (await connection.execute(text("SELECT count(*) FROM managed_artifacts"))).scalar_one()
        reference_scope_columns = (
            await connection.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name='artifact_references' "
                    "AND column_name IN ('org_id', 'project_id')"
                )
            )
        ).scalar_one()
    assert version == _N
    assert count == 2
    assert reference_scope_columns == 2


@pytest.mark.asyncio
async def test_head_bootstrap_uses_migration_constraint_names_and_can_conditionally_downgrade(database) -> None:
    await create_schema(database)
    async with database.connect() as connection:
        primary_keys = {
            row[0]
            for row in (
                await connection.execute(
                    text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE contype='p' AND conrelid IN "
                        "('managed_artifacts'::regclass, 'artifact_references'::regclass)"
                    )
                )
            ).all()
        }
    assert primary_keys == {"pk_managed_artifacts_scoped", "pk_artifact_references_scoped"}

    async with database.begin() as connection:
        await _insert_workload(connection, workload_id="wl-alpha", org_id="acme", project_id="alpha")
        # Head tables use the same identity plus an explicitly downgrade-safe
        # quarantined lifecycle row.
        await _insert_managed(connection, org_id="acme", project_id="alpha", lifecycle=True)
        scoped_id = artifact_reference_id("acme", "alpha", _URI, "wl-alpha")
        data = {
            "id": scoped_id,
            "uri": _URI,
            "workload_id": "wl-alpha",
            "org_id": "acme",
            "project_id": "alpha",
        }
        await connection.execute(
            text(
                "INSERT INTO artifact_references "
                "(org_id, project_id, uri, workload_id, id, released_at, created_at, data) "
                "VALUES ('acme', 'alpha', :uri, 'wl-alpha', :id, NULL, now(), CAST(:data AS json))"
            ),
            {"uri": _URI, "id": scoped_id, "data": json.dumps(data)},
        )

    await _downgrade(database, _N_MINUS_1)

    legacy_id = _legacy_id(_URI, "wl-alpha")
    async with database.connect() as connection:
        reference = (await connection.execute(text("SELECT id, data ->> 'id' FROM artifact_references"))).one()
        managed_pk = (
            await connection.execute(
                text("SELECT conname FROM pg_constraint WHERE contype='p' AND conrelid='managed_artifacts'::regclass")
            )
        ).scalar_one()
    assert tuple(reference) == (legacy_id, legacy_id)
    assert managed_pk == "managed_artifacts_pkey"
