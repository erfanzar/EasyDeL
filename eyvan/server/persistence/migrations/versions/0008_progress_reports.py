# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the latest-per-rank workload progress table.

A training loop's own notion of progress ("step 4200 of 50000") is neither a
result nor a log line, and had nowhere to go. This is that channel.

The composite primary key ``(workload_id, rank)`` is the design, not an
implementation detail: a report *replaces* that rank's previous one, so the table
is bounded by the number of live ranks rather than by how chatty user code is.
Unbounded user-metric history inside the control-plane database is the growth
this refuses -- it would be the fastest-growing table in the system, written by
code Eyvan does not control.

No retention window is needed for the same reason. There is nothing to prune: a
row is overwritten in place for as long as its workload runs, and collected with
the workload when it is.

Revision ID: 0008_progress_reports
Revises: 0007_federation_identity
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_progress_reports"
down_revision = "0007_federation_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "progress_reports",
        sa.Column("workload_id", sa.String(length=96), primary_key=True),
        sa.Column("rank", sa.Integer(), primary_key=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("progress_reports")
