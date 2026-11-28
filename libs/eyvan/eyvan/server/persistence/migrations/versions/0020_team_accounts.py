# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add team-mode users, invites and API tokens.

Three new tables, nothing altered, so an older build rolled back over this
reads every existing record unchanged; it simply never selects from tables it
does not know, and team sign-in stops working until the newer build returns.

Only hashes of invite and token secrets are stored. ``token_hash`` and
``email`` are real unique columns because sign-in and every bearer request look
them up directly, and that read must stay an index lookup.

Revision ID: 0020_team_accounts
Revises: 0019_webhooks
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0020_team_accounts"
down_revision = "0019_webhooks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("subject", sa.String(length=64), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table(
        "invites",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_invites_token_hash", "invites", ["token_hash"], unique=True)
    op.create_table(
        "api_tokens",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("subject", sa.String(length=64), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_api_tokens_subject", "api_tokens", ["subject"])
    op.create_index("ix_api_tokens_token_hash", "api_tokens", ["token_hash"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_api_tokens_token_hash", table_name="api_tokens")
    op.drop_index("ix_api_tokens_subject", table_name="api_tokens")
    op.drop_table("api_tokens")
    op.drop_index("ix_invites_token_hash", table_name="invites")
    op.drop_table("invites")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
