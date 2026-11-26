# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PostgreSQL advisory-lock leadership for the global scheduler."""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

#: Verify this backend session still *holds* the advisory lock, rather than
#: merely that its connection is alive. ``pg_try_advisory_lock`` is re-entrant,
#: so re-calling it would succeed and silently increment the lock counter; and a
#: bare ``SELECT 1`` proves only that some connection answered.
#:
#: The bigint form of an advisory lock is stored split across ``classid`` (high
#: 32 bits) and ``objid`` (low 32 bits) with ``objsubid = 1``.
_LOCK_OWNERSHIP_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM pg_locks
    WHERE locktype = 'advisory'
      AND granted
      AND pid = pg_backend_pid()
      AND objsubid = 1
      AND ((classid::bigint << 32) | objid::bigint) = :lock_id
)
"""


class PostgresAdvisoryLeader:
    """Hold a session-scoped advisory lock on one dedicated connection.

    **Deployment constraint.** Session-scoped advisory locks require a stable
    backend session. Behind a transaction-pooling proxy (PgBouncer in
    ``transaction`` mode) a connection can be handed to a different backend
    between statements, so the lock this object believes it holds may live in a
    session it no longer owns. ``acquire`` re-verifies ownership on every renewal
    precisely so that situation demotes this replica instead of producing two
    leaders -- but the supported deployment is a direct connection or a
    session-pooling proxy.
    """

    def __init__(self, engine: AsyncEngine, *, lock_id: int = 0x4D4552494449414E):
        self.engine = engine
        self.lock_id = lock_id
        self._connection: AsyncConnection | None = None
        self._guard = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._guard:
            if self._connection is not None:
                try:
                    still_owned = bool(
                        await self._connection.scalar(text(_LOCK_OWNERSHIP_SQL), {"lock_id": self.lock_id})
                    )
                except BaseException as error:
                    connection, self._connection = self._connection, None
                    await self._discard_uncertain_connection(connection)
                    if not isinstance(error, Exception):
                        raise
                else:
                    if still_owned:
                        return True
                    # The session is alive but no longer holds the lock. Drop it
                    # and fall through to a clean re-acquisition attempt: another
                    # replica may legitimately be leader now.
                    connection, self._connection = self._connection, None
                    await connection.close()
            connection = await self.engine.connect()
            try:
                connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
                acquired = bool(
                    await connection.scalar(text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": self.lock_id})
                )
            except BaseException:
                await self._discard_uncertain_connection(connection)
                raise
            if not acquired:
                await connection.close()
                return False
            self._connection = connection
            return True

    async def release(self) -> None:
        async with self._guard:
            if self._connection is None:
                return
            connection, self._connection = self._connection, None
            try:
                await connection.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": self.lock_id})
            except BaseException:
                await self._discard_uncertain_connection(connection)
                raise
            else:
                await connection.close()

    @staticmethod
    async def _discard_uncertain_connection(connection: AsyncConnection) -> None:
        """End the physical session before returning an uncertain checkout.

        Pool rollback/close does not release session advisory locks. A failed
        statement may leave a healthy backend owning the lock, including when
        acquisition succeeded on PostgreSQL but its result was lost.
        """

        async def discard() -> None:
            await connection.invalidate()
            await connection.close()

        cleanup = asyncio.create_task(discard())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Finish invalidation even if shutdown cancels cleanup again.
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError
