# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add users' budgeted research plans.

One new table, nothing altered: an older build rolled back over this reads every
existing record unchanged and simply runs no research plans.

Revision ID: 0024_research_plans
Revises: 0023_assistant_conversations
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0024_research_plans"
down_revision = "0023_assistant_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_plans",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_research_plans_subject", "research_plans", ["subject"])
    op.create_index("ix_research_plans_state", "research_plans", ["state"])


def downgrade() -> None:
    op.drop_index("ix_research_plans_state", table_name="research_plans")
    op.drop_index("ix_research_plans_subject", table_name="research_plans")
    op.drop_table("research_plans")
