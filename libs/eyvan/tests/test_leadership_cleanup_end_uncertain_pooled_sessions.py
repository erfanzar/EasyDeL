# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Leadership cleanup must end uncertain pooled sessions, not check them in."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.server.persistence.leader import PostgresAdvisoryLeader


@dataclass
class Backend:
    """Physical server session surviving a pooled checkout's close."""

    lock_count: int = 0
    terminated: bool = False


class Server:
    """One advisory lock shared by competing physical sessions."""

    def __init__(self) -> None:
        self.backends: list[Backend] = []

    def try_lock(self, backend: Backend) -> bool:
        if any(other.lock_count for other in self.backends if other is not backend):
            return False
        backend.lock_count += 1
        return True


class PooledConnection:
    """Checkout wrapper whose close does not release session advisory locks."""

    def __init__(self, engine: PooledEngine, backend: Backend) -> None:
        self.engine = engine
        self.backend = backend
        self.closed = False

    def fail(self, phase: str) -> None:
        if self.engine.failure_phase == phase:
            self.engine.failure_phase = None
            raise self.engine.failure

    async def execution_options(self, *, isolation_level: str) -> PooledConnection:
        assert isolation_level == "AUTOCOMMIT"
        self.fail("options")
        return self

    async def scalar(self, statement: object, params: dict[str, int]) -> bool:
        assert params == {"lock_id": 42}
        if "pg_try_advisory_lock" in str(statement):
            acquired = self.engine.server.try_lock(self.backend)
            # A lost result can occur after PostgreSQL already granted the lock.
            self.fail("acquire")
            return acquired
        assert "pg_locks" in str(statement)
        self.fail("probe")
        return bool(self.backend.lock_count)

    async def execute(self, statement: object, params: dict[str, int]) -> None:
        assert "pg_advisory_unlock" in str(statement)
        assert params == {"lock_id": 42}
        self.fail("unlock")
        if self.backend.lock_count:
            self.backend.lock_count -= 1

    async def invalidate(self) -> None:
        self.backend.terminated = True
        self.backend.lock_count = 0

    async def close(self) -> None:
        self.closed = True
        # Normal pool reset/rollback does NOT drop session-scoped advisory locks.


class PooledEngine:
    """Reuse physical sessions through distinct checkout wrappers."""

    def __init__(self, server: Server) -> None:
        self.server = server
        self.backend: Backend | None = None
        self.connections: list[PooledConnection] = []
        self.failure_phase: str | None = None
        self.failure: BaseException = RuntimeError("recoverable statement timeout")

    async def connect(self) -> PooledConnection:
        if self.backend is None or self.backend.terminated:
            self.backend = Backend()
            self.server.backends.append(self.backend)
        connection = PooledConnection(self, self.backend)
        self.connections.append(connection)
        return connection


async def test_recoverable_probe_error_does_not_leave_a_reentrant_pooled_lock() -> None:
    server = Server()
    engine = PooledEngine(server)
    leader = PostgresAdvisoryLeader(engine, lock_id=42)
    assert await leader.acquire()
    original = engine.backend
    engine.failure_phase = "probe"

    assert await leader.acquire()
    await leader.release()

    assert sum(backend.lock_count for backend in server.backends) == 0
    assert original is not None and original.terminated
    contender = PostgresAdvisoryLeader(PooledEngine(server), lock_id=42)
    assert await contender.acquire()
    await contender.release()


@pytest.mark.parametrize("phase", ["acquire", "probe", "unlock", "options"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_uncertain_operation_cleanup_does_not_strand_session_locks(phase: str, cancelled: bool) -> None:
    server = Server()
    engine = PooledEngine(server)
    leader = PostgresAdvisoryLeader(engine, lock_id=42)
    if phase in {"probe", "unlock"}:
        assert await leader.acquire()
    engine.failure_phase = phase
    if cancelled:
        engine.failure = asyncio.CancelledError()
    operation = leader.release if phase == "unlock" else leader.acquire
    if phase == "probe" and not cancelled:
        assert await operation()
    else:
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            await operation()
    await leader.release()

    assert sum(backend.lock_count for backend in server.backends) == 0
    assert all(connection.closed for connection in engine.connections)
    contender = PostgresAdvisoryLeader(PooledEngine(server), lock_id=42)
    assert await contender.acquire()
    await contender.release()


async def test_repeated_shutdown_cancellation_waits_for_physical_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    server = Server()
    engine = PooledEngine(server)
    leader = PostgresAdvisoryLeader(engine, lock_id=42)
    assert await leader.acquire()
    engine.failure_phase = "unlock"
    invalidation_started = asyncio.Event()
    finish_invalidation = asyncio.Event()
    original_invalidate = PooledConnection.invalidate

    async def delayed_invalidate(connection: PooledConnection) -> None:
        invalidation_started.set()
        await finish_invalidation.wait()
        await original_invalidate(connection)

    monkeypatch.setattr(PooledConnection, "invalidate", delayed_invalidate)
    release = asyncio.create_task(leader.release())
    try:
        await asyncio.wait_for(invalidation_started.wait(), timeout=1)
        release.cancel()
        await asyncio.sleep(0)
        release.cancel()
        await asyncio.sleep(0)
        assert not release.done()
        assert engine.backend is not None and engine.backend.lock_count == 1
        assert not engine.connections[0].closed
    finally:
        finish_invalidation.set()
        with pytest.raises(asyncio.CancelledError):
            await release
    assert engine.backend.terminated
    assert engine.backend.lock_count == 0
    assert engine.connections[0].closed


async def test_real_postgres_pooled_probe_statement_timeout_releases_leadership(postgres_url: str) -> None:
    """Parent PG gate: timeout a real ownership query without disconnecting first."""

    lock_id = 0x083083083083
    engine = create_async_engine(postgres_url, pool_size=1, max_overflow=0)
    competitor_engine = create_async_engine(postgres_url, pool_size=1, max_overflow=0)
    leader = PostgresAdvisoryLeader(engine, lock_id=lock_id)
    contender = PostgresAdvisoryLeader(competitor_engine, lock_id=lock_id)
    force_timeout = False

    @event.listens_for(engine.sync_engine, "before_cursor_execute", retval=True)
    def timeout_probe(connection, cursor, statement, parameters, context, executemany):
        nonlocal force_timeout
        if force_timeout and "pg_locks" in statement:
            force_timeout = False
            return "SELECT pg_sleep(0.1)", ()
        return statement, parameters

    try:
        assert await leader.acquire()
        held = leader._connection
        assert held is not None
        await held.execute(text("SET statement_timeout = '10ms'"))
        force_timeout = True
        assert await leader.acquire()
        assert not force_timeout, "the real ownership statement must have been replaced and timed out"
        await leader.release()
        async with engine.connect() as observer:
            assert not await observer.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND granted "
                    "AND objsubid = 1 AND ((classid::bigint << 32) | objid::bigint) = :lock_id)"
                ),
                {"lock_id": lock_id},
            )
        assert await contender.acquire()
    finally:
        await contender.release()
        await leader.release()
        await competitor_engine.dispose()
        await engine.dispose()
