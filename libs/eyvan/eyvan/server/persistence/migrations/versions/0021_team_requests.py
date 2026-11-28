# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add team capacity/approval requests.

One new table, nothing altered: an older build rolled back over this reads every
existing record unchanged and simply never sees the request queue.

Revision ID: 0021_team_requests
Revises: 0020_team_accounts
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021_team_requests"
down_revision = "0020_team_accounts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "team_requests",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_team_requests_subject", "team_requests", ["subject"])
    op.create_index("ix_team_requests_state", "team_requests", ["state"])


def downgrade() -> None:
    op.drop_index("ix_team_requests_state", table_name="team_requests")
    op.drop_index("ix_team_requests_subject", table_name="team_requests")
    op.drop_table("team_requests")
