# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add users' saved assistant conversations.

One new table, nothing altered: an older build rolled back over this reads every
existing record unchanged and simply shows no chat history.

Revision ID: 0023_assistant_conversations
Revises: 0022_assistant_providers
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0023_assistant_conversations"
down_revision = "0022_assistant_providers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_conversations",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_assistant_conversations_subject", "assistant_conversations", ["subject"])


def downgrade() -> None:
    op.drop_index("ix_assistant_conversations_subject", table_name="assistant_conversations")
    op.drop_table("assistant_conversations")
