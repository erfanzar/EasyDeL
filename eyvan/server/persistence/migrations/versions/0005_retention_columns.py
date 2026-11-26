# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add indexed retention columns so a prune can page on what it filters.

``prune_outbox`` paged by ``available_at`` and ``prune_idempotency`` by the
surrogate ``id``, while both applied their cutoff to a timestamp *inside* the
JSON payload after the LIMIT had already chosen the page. Neither paging column
is the filter column, so a full window of ineligible rows starved the genuinely
expired rows behind it and the prune returned 0 on every pass -- an idempotency
key stayed honored forever, which is the bug retention was added to fix.

Revision ID: 0005_retention_columns
Revises: 0004_query_indexes
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_retention_columns"
down_revision = "0004_query_indexes"
branch_labels = None
depends_on = None

#: Table -> (new column, JSON payload key it copies). Both are timestamps the
#: retention windows compare against.
_RETENTION_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("outbox", "updated_at", "updated_at"),
    ("idempotency_keys", "created_at", "created_at"),
)


def upgrade() -> None:
    for table, column, _key in _RETENTION_COLUMNS:
        op.add_column(table, sa.Column(column, sa.DateTime(timezone=True), nullable=True))
    # The JSON payload stays the source of truth. A row written by an older
    # schema version backfills here; one written by a newer one populates the
    # column directly. Rows that somehow have neither stay NULL and are excluded
    # by the prune's explicit IS NOT NULL guard rather than being deleted blind.
    for table, column, key in _RETENTION_COLUMNS:
        op.execute(f"UPDATE {table} SET {column} = CAST(data ->> '{key}' AS timestamptz)")
    for table, column, _key in _RETENTION_COLUMNS:
        op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade() -> None:
    for table, column, _key in reversed(_RETENTION_COLUMNS):
        op.drop_index(f"ix_{table}_{column}", table_name=table)
        op.drop_column(table, column)
