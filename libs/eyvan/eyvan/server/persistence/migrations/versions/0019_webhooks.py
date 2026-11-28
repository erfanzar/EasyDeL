# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the durable webhook-subscription table.

One new table, nothing altered, so an older build rolled back over this reads
every existing record unchanged -- it simply never selects from a table it does
not know about, and the fleet stops delivering notifications it did not
previously send.

Nothing is backfilled and nothing could be. A subscription is a statement about
where somebody wants events sent, and no existing record carries that.

The delivery cursor lives in the row's document rather than in a separate table:
it is per subscription, advances only on a successful delivery, and is read and
written in the same transaction that reads the subscription. A second table
would buy nothing except the chance for the two to disagree.

Revision ID: 0019_webhooks
Revises: 0018_service_operations
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019_webhooks"
down_revision = "0018_service_operations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "webhooks",
        sa.Column("id", sa.String(length=96), primary_key=True),
        # Real columns rather than JSON extraction: delivery lists every
        # subscription for one scope on every cycle, and that read must stay an
        # index lookup as the table grows.
        sa.Column("org_id", sa.String(length=128), nullable=False),
        sa.Column("project_id", sa.String(length=128), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_webhooks_org_id", "webhooks", ["org_id"])
    op.create_index("ix_webhooks_project_id", "webhooks", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_webhooks_project_id", table_name="webhooks")
    op.drop_index("ix_webhooks_org_id", table_name="webhooks")
    op.drop_table("webhooks")
