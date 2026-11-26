# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add assistant providers configured by fleet admins.

One new table, nothing altered: an older build rolled back over this reads every
existing record unchanged and simply has no assistant providers. Provider keys
are stored inside ``data`` as ciphertext only.

Revision ID: 0022_assistant_providers
Revises: 0021_team_requests
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0022_assistant_providers"
down_revision = "0021_team_requests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_providers",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("assistant_providers")
