# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline Alembic migrations: environment override and generated DDL shape."""

from __future__ import annotations

import io
import typing as tp
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from alembic import command, context
from alembic.config import Config

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"
#: The loopback-development default is password-less on purpose: a shipped
#: static credential would be a credential every installation shares, and the
#: local default relies on the host's trust/peer auth instead. Production
#: URLs always come from EYVAN_DATABASE_URL or a secret reference.
INI_DATABASE_URL = "postgresql+asyncpg://eyvan@127.0.0.1:5432/eyvan"

EXPECTED_TABLES = (
    "schema_revisions",
    "workloads",
    "cells",
    "leases",
    "policies",
    "budgets",
    "quotas",
    "idempotency_keys",
    "outbox",
    "audit_events",
    "capacity_pools",
    "workload_topologies",
)


def _offline_upgrade(monkeypatch: pytest.MonkeyPatch, database_url: str | None) -> tuple[str, str]:
    """Run ``alembic upgrade head --sql`` programmatically and capture SQL plus the resolved URL."""

    if database_url is None:
        monkeypatch.delenv("EYVAN_DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("EYVAN_DATABASE_URL", database_url)
    resolved: dict[str, tp.Any] = {}
    original_configure = context.configure

    def recording_configure(**kwargs: tp.Any) -> tp.Any:
        resolved.update(kwargs)
        return original_configure(**kwargs)

    monkeypatch.setattr(context, "configure", recording_configure)
    output = io.StringIO()
    with redirect_stdout(output):
        command.upgrade(Config(str(ALEMBIC_INI)), "head", sql=True)
    return output.getvalue(), resolved["url"]


def test_database_url_environment_variable_overrides_the_ini_default(monkeypatch: pytest.MonkeyPatch) -> None:
    override = "postgresql+asyncpg://eyvan:eyvan@postgres.internal:5433/eyvan_override"

    _, resolved_url = _offline_upgrade(monkeypatch, override)

    assert resolved_url == override


def test_database_url_falls_back_to_the_ini_default_without_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, resolved_url = _offline_upgrade(monkeypatch, None)

    assert resolved_url == INI_DATABASE_URL


def test_offline_upgrade_emits_the_full_control_plane_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    sql, _ = _offline_upgrade(monkeypatch, "postgresql+asyncpg://eyvan:eyvan@db.internal:5432/eyvan")

    for table in EXPECTED_TABLES:
        assert f"CREATE TABLE {table} (" in sql, f"missing CREATE TABLE for {table}"
    assert "uq_idempotency_scope_operation_key" in sql
    assert "CREATE INDEX ix_outbox_state" in sql
    assert "CREATE INDEX ix_audit_events_org_id" in sql

    first_revision = sql.index("Running upgrade  -> 0002_transactional_outbox")
    second_revision = sql.index("Running upgrade 0002_transactional_outbox -> 0003_capacity_pools")
    assert first_revision < second_revision
    assert sql.index("CREATE TABLE audit_events (") < sql.index("CREATE TABLE capacity_pools (")

    # Retention needs to page on the column it filters, which means an indexed
    # typed copy of each cutoff timestamp. Without these the prune could return
    # 0 forever while expired rows sat past its LIMIT.
    assert "ALTER TABLE outbox ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE" in sql
    assert "ALTER TABLE idempotency_keys ADD COLUMN created_at TIMESTAMP WITH TIME ZONE" in sql
    assert "CREATE INDEX ix_outbox_updated_at" in sql
    assert "CREATE INDEX ix_idempotency_keys_created_at" in sql
    # Backfilled from the payload, which stays the source of truth, so rows
    # written by an older schema version are prunable after the upgrade.
    assert "UPDATE outbox SET updated_at = CAST(data ->> 'updated_at' AS timestamptz)" in sql


def test_offline_upgrade_uses_postgres_ddl_not_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    sql, _ = _offline_upgrade(monkeypatch, "postgresql+asyncpg://eyvan:eyvan@db.internal:5432/eyvan")

    assert "BIGSERIAL" in sql
    assert "TIMESTAMP WITH TIME ZONE" in sql
