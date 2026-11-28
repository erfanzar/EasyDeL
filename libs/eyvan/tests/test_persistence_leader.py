# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Behavioral coverage for PostgresAdvisoryLeader over a fake async engine."""

from __future__ import annotations

import typing as tp

import pytest

from eyvan.server.persistence.leader import PostgresAdvisoryLeader


class FakeAsyncConnection:
    """Fake AsyncConnection answering only the advisory-lock SQL the leader issues."""

    def __init__(self, engine: FakeAsyncEngine) -> None:
        self.engine = engine
        self.closed = False
        self.broken = False
        self.isolation_levels: list[str] = []
        self.liveness_probes = 0
        self.unlocked: list[int] = []
        self.close_calls = 0
        self.invalidate_calls = 0

    async def execution_options(self, *, isolation_level: str) -> FakeAsyncConnection:
        """Record the requested isolation level like the real connection would."""

        self.isolation_levels.append(isolation_level)
        return self

    async def scalar(self, statement: object, params: dict[str, tp.Any] | None = None) -> tp.Any:
        """Answer ownership probes and ``pg_try_advisory_lock`` attempts."""

        sql = str(statement)
        if "pg_try_advisory_lock" in sql:
            if self.engine.try_lock_failure is not None:
                raise self.engine.try_lock_failure
            assert params is not None
            return self.engine.try_advisory_lock(self, params["lock_id"])
        assert "pg_locks" in sql, f"unexpected statement {sql!r}"
        self.liveness_probes += 1
        if self.closed or self.broken:
            raise RuntimeError("connection lost")
        # Model the real query: it reports whether THIS session still holds the
        # advisory lock, not merely that the connection answered.
        assert params is not None
        return self.engine.lock_holder is self and self.engine.lock_id == params["lock_id"]

    async def execute(self, statement: object, params: dict[str, tp.Any] | None = None) -> None:
        """Apply ``pg_advisory_unlock`` against the fake server-side lock state."""

        sql = str(statement)
        assert "pg_advisory_unlock" in sql, f"unexpected statement {sql!r}"
        assert params is not None
        self.unlocked.append(params["lock_id"])
        self.engine.advisory_unlock(self, params["lock_id"])

    async def invalidate(self) -> None:
        """Discard the physical session rather than returning it to the pool."""

        self.invalidate_calls += 1
        self.broken = True
        self.engine.session_closed(self)

    async def close(self) -> None:
        """Close the session, which drops any server-side session advisory lock."""

        self.close_calls += 1
        self.closed = True
        self.engine.session_closed(self)


class FakeAsyncEngine:
    """Fake AsyncEngine modeling one session-scoped Postgres advisory lock."""

    def __init__(self) -> None:
        self.connections: list[FakeAsyncConnection] = []
        self.lock_holder: FakeAsyncConnection | None = None
        self.lock_id: int | None = None
        self.try_lock_attempts = 0
        self.try_lock_failure: Exception | None = None

    async def connect(self) -> FakeAsyncConnection:
        """Open a fresh session with no advisory locks held."""

        connection = FakeAsyncConnection(self)
        self.connections.append(connection)
        return connection

    def try_advisory_lock(self, connection: FakeAsyncConnection, lock_id: int) -> bool:
        """Grant the lock when free or already held by this exact session."""

        self.try_lock_attempts += 1
        if self.lock_holder is None or self.lock_holder is connection:
            self.lock_holder = connection
            self.lock_id = lock_id
            return True
        return False

    def advisory_unlock(self, connection: FakeAsyncConnection, lock_id: int) -> None:
        """Release the lock only when the calling session actually holds it."""

        if self.lock_holder is connection and self.lock_id == lock_id:
            self.lock_holder = None

    def session_closed(self, connection: FakeAsyncConnection) -> None:
        """Model Postgres dropping session-scoped locks when the session ends."""

        if self.lock_holder is connection:
            self.lock_holder = None


async def test_acquire_holds_one_dedicated_autocommit_connection() -> None:
    engine = FakeAsyncEngine()
    leader = PostgresAdvisoryLeader(engine, lock_id=42)

    assert await leader.acquire() is True

    assert len(engine.connections) == 1
    connection = engine.connections[0]
    assert connection.isolation_levels == ["AUTOCOMMIT"]
    assert engine.lock_holder is connection
    assert engine.lock_id == 42
    assert not connection.closed


async def test_renew_rechecks_liveness_without_reacquiring_the_lock() -> None:
    engine = FakeAsyncEngine()
    leader = PostgresAdvisoryLeader(engine)

    assert await leader.acquire() is True
    assert await leader.acquire() is True
    assert await leader.acquire() is True

    assert len(engine.connections) == 1
    assert engine.connections[0].liveness_probes == 2
    assert engine.try_lock_attempts == 1


async def test_acquire_reports_not_leader_while_another_session_holds_the_lock() -> None:
    engine = FakeAsyncEngine()
    first = PostgresAdvisoryLeader(engine)
    second = PostgresAdvisoryLeader(engine)

    assert await first.acquire() is True
    assert await second.acquire() is False

    assert engine.connections[1].closed
    assert engine.lock_holder is engine.connections[0]
    assert await second.acquire() is False

    await first.release()
    assert await second.acquire() is True
    assert engine.lock_holder is engine.connections[-1]


async def test_release_unlocks_closes_and_is_idempotent() -> None:
    engine = FakeAsyncEngine()
    leader = PostgresAdvisoryLeader(engine, lock_id=7)

    await leader.release()
    assert engine.connections == []

    assert await leader.acquire() is True
    connection = engine.connections[0]

    await leader.release()

    assert connection.unlocked == [7]
    assert engine.lock_holder is None
    assert connection.closed
    assert connection.close_calls == 1

    await leader.release()
    assert connection.close_calls == 1


async def test_lost_connection_reacquires_the_lock_on_a_fresh_session() -> None:
    engine = FakeAsyncEngine()
    leader = PostgresAdvisoryLeader(engine)

    assert await leader.acquire() is True
    dead = engine.connections[0]
    dead.broken = True

    assert await leader.acquire() is True

    assert dead.invalidate_calls == dead.close_calls == 1
    assert len(engine.connections) == 2
    assert engine.lock_holder is engine.connections[1]


async def test_lost_advisory_lock_makes_the_leader_step_down() -> None:
    engine = FakeAsyncEngine()
    leader = PostgresAdvisoryLeader(engine)
    contender = PostgresAdvisoryLeader(engine)

    assert await leader.acquire() is True
    engine.connections[0].broken = True
    engine.session_closed(engine.connections[0])
    assert await contender.acquire() is True

    assert await leader.acquire() is False

    assert engine.lock_holder is engine.connections[1]
    assert engine.connections[2].closed


async def test_a_live_session_that_silently_lost_the_lock_steps_down() -> None:
    """The hazard a ``SELECT 1`` probe cannot see.

    Behind a transaction-pooling proxy the connection keeps answering while the
    advisory lock lives in a backend session this replica no longer owns.
    Treating a successful round trip as proof of leadership would then run two
    leaders at once, which is exactly what the autoscaler's ``len(managed) + 1``
    cell-id minting cannot survive.
    """

    engine = FakeAsyncEngine()
    leader = PostgresAdvisoryLeader(engine)
    contender = PostgresAdvisoryLeader(engine)

    assert await leader.acquire() is True
    held = engine.connections[0]

    # The lock moves away, but the connection stays perfectly healthy.
    engine.lock_holder = None
    assert await contender.acquire() is True
    assert held.broken is False and held.closed is False

    assert await leader.acquire() is False
    assert held.closed, "a session that no longer owns the lock must be dropped"


async def test_ownership_probe_is_issued_with_the_configured_lock_id() -> None:
    engine = FakeAsyncEngine()
    leader = PostgresAdvisoryLeader(engine, lock_id=1234)

    assert await leader.acquire() is True
    assert await leader.acquire() is True

    assert engine.connections[0].liveness_probes == 1
    assert engine.lock_id == 1234


async def test_try_lock_failure_closes_the_session_and_propagates() -> None:
    engine = FakeAsyncEngine()
    engine.try_lock_failure = RuntimeError("database is restarting")
    leader = PostgresAdvisoryLeader(engine)

    with pytest.raises(RuntimeError, match="database is restarting"):
        await leader.acquire()

    assert engine.connections[0].closed
    assert engine.connections[0].invalidate_calls == 1
    assert engine.lock_holder is None

    engine.try_lock_failure = None
    assert await leader.acquire() is True
    assert engine.lock_holder is engine.connections[1]
