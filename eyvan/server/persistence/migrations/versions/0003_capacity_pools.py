# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add declarative capacity pools for demand autoscaling.

Revision ID: 0003_capacity_pools
Revises: 0002_transactional_outbox
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_capacity_pools"
down_revision = "0002_transactional_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "capacity_pools",
        sa.Column("id", sa.String(192), primary_key=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("capacity_pools")
