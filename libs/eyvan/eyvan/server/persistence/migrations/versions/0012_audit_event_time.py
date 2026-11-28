# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Index audit event time for bounded recent-window queries.

Queue history is a polling query over the audit ledger.  Ordering the first
5,000 sequences returned the oldest history forever on a mature control plane;
filtering on a timestamp hidden inside JSON would replace that correctness bug
with an unindexed full-table scan.  The durable column is backfilled from the
existing immutable event payload before it becomes non-null.

Revision ID: 0012_audit_event_time
Revises: 0011_quota_grants
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_audit_event_time"
down_revision = "0011_quota_grants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "audit_events",
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE audit_events "
        "SET occurred_at = CAST(data ->> 'timestamp' AS TIMESTAMP WITH TIME ZONE) "
        "WHERE occurred_at IS NULL"
    )
    op.alter_column("audit_events", "occurred_at", nullable=False)
    op.create_index(
        "ix_audit_events_scope_type_occurred_at",
        "audit_events",
        ["org_id", "project_id", "entity_type", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_events_scope_type_occurred_at", table_name="audit_events")
    op.drop_column("audit_events", "occurred_at")
