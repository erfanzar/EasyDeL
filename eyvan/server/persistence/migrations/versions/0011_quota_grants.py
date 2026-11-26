# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Store the quota slices this control plane issued to its peers.

Budget grants closed the cross-plane hole for money in 0007. This closes the same
hole for capacity: ``QuotaRecord.limits`` is one durable vector in one database,
so a tenant with projects on three planes could run three times its TPU ceiling
with every plane locally compliant and every operator's dashboard green.

The two limits are independent and a fleet can hold one while breaching the
other. A tenant granted a small budget on cheap Spot capacity can still exhaust
an accelerator pool everyone else is queued behind, because the thing in short
supply is chips rather than dollars.

Indexes mirror ``budget_grants`` deliberately, including the ``state`` /
``expires_at`` pair the expiry sweep both filters and pages on. That pairing is
migration 0005's lesson: paging on a column other than the one being filtered
lets a full window of ineligible rows starve the eligible ones behind it, and the
sweep silently does nothing forever.

Revision ID: 0011_quota_grants
Revises: 0010_role_grants
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_quota_grants"
down_revision = "0010_role_grants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quota_grants",
        sa.Column("id", sa.String(length=96), primary_key=True),
        sa.Column("org_id", sa.String(length=128), nullable=True),
        sa.Column("project_id", sa.String(length=128), nullable=True),
        sa.Column("grantee_plane_id", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_quota_grants_org_id", "quota_grants", ["org_id"])
    op.create_index("ix_quota_grants_project_id", "quota_grants", ["project_id"])
    op.create_index("ix_quota_grants_grantee_plane_id", "quota_grants", ["grantee_plane_id"])
    op.create_index("ix_quota_grants_state", "quota_grants", ["state"])
    op.create_index("ix_quota_grants_expires_at", "quota_grants", ["expires_at"])
    op.create_index("ix_quota_grants_issued_at", "quota_grants", ["issued_at"])


def downgrade() -> None:
    for index in (
        "ix_quota_grants_issued_at",
        "ix_quota_grants_expires_at",
        "ix_quota_grants_state",
        "ix_quota_grants_grantee_plane_id",
        "ix_quota_grants_project_id",
        "ix_quota_grants_org_id",
    ):
        op.drop_index(index, table_name="quota_grants")
    op.drop_table("quota_grants")
