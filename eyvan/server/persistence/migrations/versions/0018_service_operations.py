# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the durable service replica-health and decision-audit tables.

Two new tables, nothing altered. The alternative was fields on the persisted
workload document, which every replica decodes with ``extra='forbid'``: the
first health probe would have made every workload unreadable to a rolled-back
build, and a control plane that cannot read its workloads schedules nothing. New
tables are invisible to older code -- it never selects from them -- so a rollback
loses the probe counters and the decision log and nothing else.

Nothing is backfilled, and nothing could be. A row in ``service_replica_health``
is a claim that a specific probe against a specific replica generation completed
and established something; today's workload inventory cannot support that claim,
because nobody probed anything before this schema existed. Replicas that predate
these tables therefore report health as *unknown*, which is the honest answer and
is deliberately not an answer that gets anything torn down.

Revision ID: 0018_service_operations
Revises: 0017_merge_stage_rot
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0018_service_operations"
down_revision = "0017_merge_stage_rot"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "service_replica_health",
        # Keyed by the replica, not the service: one probe result upserts one
        # row, and forgetting a retired replica is a delete of exactly its own
        # row rather than a rewrite of the whole service's history.
        sa.Column("service_id", sa.String(length=96), primary_key=True),
        sa.Column("replica_id", sa.String(length=96), primary_key=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_table(
        "service_decisions",
        sa.Column("service_id", sa.String(length=96), primary_key=True),
        # Per-service and monotonic rather than a global sequence. The newest
        # row for one service is that service's cooldown clock, so reading it
        # has to be a primary-key range scan and not a sort over every service.
        sa.Column("sequence", sa.Integer(), primary_key=True),
        # A real column so "when did this service last scale out" is answerable
        # in SQL without extracting JSON.
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_check_constraint(
        "ck_service_decisions_sequence_positive",
        "service_decisions",
        "sequence >= 1",
    )


def downgrade() -> None:
    op.drop_constraint("ck_service_decisions_sequence_positive", "service_decisions", type_="check")
    op.drop_table("service_decisions")
    op.drop_table("service_replica_health")
