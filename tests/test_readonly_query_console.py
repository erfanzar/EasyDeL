# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The read-only guarantee, asserted where it is actually made.

A console over the control plane's database is only safe if it cannot write to
it, and there are two ways to try to arrange that. Inspecting the statement is a
filter that has to be right about every way SQL can write -- ``WITH ... DELETE``,
a writable CTE, a volatile function -- and being wrong once is a write to
production state during an incident. Asking PostgreSQL to refuse writes is a
guarantee, because it is the thing that would otherwise perform them.

So these run real writes against a real PostgreSQL. A test that asserted a
parser rejected the string ``DELETE`` would pass on a build with no protection
at all, which is the reason none of them do that.
"""

from __future__ import annotations

import typing as tp

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.base import ReadOnlyQueryUnsupported
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema


@pytest.fixture
async def repository(request, tmp_path) -> tp.AsyncIterator[SqlAlchemyRepository]:
    """A real PostgreSQL, because the guarantee is PostgreSQL's to make."""

    url = request.getfixturevalue("postgres_url")
    engine = create_async_engine(url)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await create_schema(engine)
    try:
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()


@pytest.fixture
async def sqlite_repository(tmp_path) -> tp.AsyncIterator[SqlAlchemyRepository]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'eyvan.db'}")
    await create_schema(engine)
    try:
        yield SqlAlchemyRepository(engine)
    finally:
        await engine.dispose()


async def test_a_select_is_answered_with_its_columns(repository: SqlAlchemyRepository) -> None:
    result = await repository.readonly_query("SELECT 1 AS one, 2 AS two", limit=10, timeout_ms=5000)

    assert result.columns == ("one", "two")
    assert result.rows == ((1, 2),)
    assert result.truncated is False


async def test_a_result_that_exactly_fills_the_limit_is_not_reported_as_truncated(
    repository: SqlAlchemyRepository,
) -> None:
    """The off-by-one that would make every full page look cut short.

    Truncation is decided by fetching one row past the cap, so a result of
    exactly ``limit`` rows is known to be complete rather than guessed at.
    """

    result = await repository.readonly_query("SELECT 1 UNION ALL SELECT 2", limit=2, timeout_ms=5000)

    assert len(result.rows) == 2
    assert result.truncated is False


async def test_more_rows_than_the_limit_are_reported_as_truncated(repository: SqlAlchemyRepository) -> None:
    result = await repository.readonly_query("SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3", limit=2, timeout_ms=5000)

    assert len(result.rows) == 2
    assert result.truncated is True


async def test_a_write_is_refused_and_leaves_nothing_behind(repository: SqlAlchemyRepository) -> None:
    """The claim, run rather than parsed."""

    async with repository.engine.begin() as connection:
        await connection.execute(text("CREATE TABLE query_probe (id INTEGER PRIMARY KEY)"))

    with pytest.raises(DBAPIError):
        await repository.readonly_query("INSERT INTO query_probe (id) VALUES (1)", limit=10, timeout_ms=5000)

    async with repository.engine.connect() as connection:
        remaining = (await connection.execute(text("SELECT COUNT(*) FROM query_probe"))).scalar_one()
    assert remaining == 0


async def test_a_write_hidden_in_a_cte_is_refused_too(repository: SqlAlchemyRepository) -> None:
    """The statement that reads as a SELECT and deletes.

    PostgreSQL's writable CTEs are why read-ness is not decided by looking at
    the first keyword: this begins with ``WITH`` and ends with rows gone.
    """

    async with repository.engine.begin() as connection:
        await connection.execute(text("CREATE TABLE cte_probe (id INTEGER PRIMARY KEY)"))
        await connection.execute(text("INSERT INTO cte_probe (id) VALUES (1), (2)"))

    with pytest.raises(DBAPIError):
        await repository.readonly_query(
            "WITH removed AS (DELETE FROM cte_probe RETURNING id) SELECT COUNT(*) FROM removed",
            limit=10,
            timeout_ms=5000,
        )

    async with repository.engine.connect() as connection:
        remaining = (await connection.execute(text("SELECT COUNT(*) FROM cte_probe"))).scalar_one()
    assert remaining == 2


async def test_a_database_that_cannot_promise_read_only_is_refused(
    sqlite_repository: SqlAlchemyRepository,
) -> None:
    """SQLite backs development planes and has no read-only transaction mode.

    Served anyway, the only thing between an arbitrary statement and the schema
    would be a missing ``COMMIT`` -- which is the unguarded console this route
    exists to replace, not a weaker version of it.
    """

    with pytest.raises(ReadOnlyQueryUnsupported, match="read-only transaction"):
        await sqlite_repository.readonly_query("SELECT 1", limit=1, timeout_ms=1000)


async def test_an_in_memory_plane_refuses_rather_than_answering_no_rows() -> None:
    """ "No rows" from a store that was never queried reads as "the fleet is empty".

    That is the most alarming possible wrong answer, and an operator opens this
    console precisely when an alarming answer would be believed.
    """

    with pytest.raises(ReadOnlyQueryUnsupported):
        await InMemoryRepository().readonly_query("SELECT 1", limit=1, timeout_ms=1000)
