# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add plane identity and budget grants.

``BudgetRecord.spent`` is a single durable counter in one database, so a tenant
with projects on three planes has three independent budgets -- each enforcing its
own limit and each reporting compliance while the tenant spends three times what
its operator set. Grants bound that: the home plane issues slices of the limit to
peers, and refuses to issue more than the limit holds.

``audit_events.plane_id`` is **nullable**, deliberately. The Postgres backend
re-validates every stored row on read, so a required field would make every event
written before this migration unparseable -- turning a schema addition into an
outage of the ledger it was meant to extend.

``budget_grants`` indexes ``state`` *and* ``expires_at`` because the expiry sweep
filters and pages on both. That is migration 0005's lesson: paging on a column
other than the one being filtered lets a full window of ineligible rows starve
the eligible ones behind it, and the sweep silently does nothing forever.

Revision ID: 0007_federation_identity
Revises: 0006_artifact_index
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_federation_identity"
down_revision = "0006_artifact_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("audit_events", sa.Column("plane_id", sa.String(length=128), nullable=True))
    op.create_index("ix_audit_events_plane_id", "audit_events", ["plane_id"])

    op.create_table(
        "plane_peers",
        sa.Column("plane_id", sa.String(length=128), primary_key=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )

    op.create_table(
        "budget_grants",
        sa.Column("id", sa.String(length=96), primary_key=True),
        sa.Column("org_id", sa.String(length=128), nullable=True),
        sa.Column("project_id", sa.String(length=128), nullable=True),
        sa.Column("grantee_plane_id", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_budget_grants_org_id", "budget_grants", ["org_id"])
    op.create_index("ix_budget_grants_project_id", "budget_grants", ["project_id"])
    op.create_index("ix_budget_grants_grantee_plane_id", "budget_grants", ["grantee_plane_id"])
    op.create_index("ix_budget_grants_state", "budget_grants", ["state"])
    op.create_index("ix_budget_grants_expires_at", "budget_grants", ["expires_at"])
    op.create_index("ix_budget_grants_issued_at", "budget_grants", ["issued_at"])
    # The expiry sweep filters on both and pages on both.
    op.create_index("ix_budget_grants_state_expiry", "budget_grants", ["state", "expires_at"])


def downgrade() -> None:
    op.drop_index("ix_budget_grants_state_expiry", table_name="budget_grants")
    op.drop_index("ix_budget_grants_issued_at", table_name="budget_grants")
    op.drop_index("ix_budget_grants_expires_at", table_name="budget_grants")
    op.drop_index("ix_budget_grants_state", table_name="budget_grants")
    op.drop_index("ix_budget_grants_grantee_plane_id", table_name="budget_grants")
    op.drop_index("ix_budget_grants_project_id", table_name="budget_grants")
    op.drop_index("ix_budget_grants_org_id", table_name="budget_grants")
    op.drop_table("budget_grants")
    op.drop_table("plane_peers")
    op.drop_index("ix_audit_events_plane_id", table_name="audit_events")
    op.drop_column("audit_events", "plane_id")
