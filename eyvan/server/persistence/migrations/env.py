# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Alembic environment for the async Eyvan PostgreSQL engine."""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from eyvan.server.persistence.sqlalchemy import Base

config = context.config
# An explicitly per-invocation URL (``eyvan-migrate --database-url``) must win
# over the ambient environment variable: the safety classification in
# migrate.py is evaluated against the config URL, so an unconditional env
# override here would migrate a database other than the one that was
# classified. The env var therefore only fills in a URL that nothing else
# provided -- the plain source-tree ``alembic`` CLI flow, where alembic.ini's
# placeholder is the only prior value.
if not config.attributes.get("eyvan_explicit_database_url"):
    database_url = os.getenv("EYVAN_DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    # Escape only at the ConfigParser boundary, not in the engine's URL.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    def run(connection) -> None:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    async with engine.connect() as connection:
        await connection.run_sync(run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
