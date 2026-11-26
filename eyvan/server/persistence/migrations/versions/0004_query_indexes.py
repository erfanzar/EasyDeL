# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add typed query-filter columns with indexes, backfilled from JSON payloads.

Revision ID: 0004_query_indexes
Revises: 0003_capacity_pools
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_query_indexes"
down_revision = "0003_capacity_pools"
branch_labels = None
depends_on = None

# Aggregate table -> typed copies of the hot filter keys stored in each JSON payload.
_FILTER_COLUMNS: dict[str, tuple[tuple[str, int], ...]] = {
    "workloads": (("org_id", 128), ("project_id", 128), ("state", 32)),
    "cells": (("state", 32), ("pool_id", 192)),
    "capacity_pools": (("fleet_name", 96),),
    "leases": (("org_id", 128), ("project_id", 128), ("workload_id", 96), ("state", 32)),
    "policies": (("org_id", 128),),
    "quotas": (("org_id", 128), ("project_id", 128), ("principal_id", 128)),
    "budgets": (("org_id", 128),),
}


def upgrade() -> None:
    for table, columns in _FILTER_COLUMNS.items():
        for column, length in columns:
            op.add_column(table, sa.Column(column, sa.String(length), nullable=True))
    # The JSON payload stays the source of truth; ``data ->> 'key'`` yields SQL NULL for
    # missing keys and JSON nulls, so payloads written by any schema version backfill safely.
    for table, columns in _FILTER_COLUMNS.items():
        assignments = ", ".join(f"{column} = data ->> '{column}'" for column, _ in columns)
        op.execute(f"UPDATE {table} SET {assignments}")
    for table, columns in _FILTER_COLUMNS.items():
        for column, _ in columns:
            op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade() -> None:
    for table, columns in reversed(_FILTER_COLUMNS.items()):
        for column, _ in columns:
            op.drop_index(f"ix_{table}_{column}", table_name=table)
            op.drop_column(table, column)
