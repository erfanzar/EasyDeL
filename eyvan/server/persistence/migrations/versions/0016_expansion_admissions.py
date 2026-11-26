# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the durable fan-out admission ledger.

Purely additive. The plan one expansion was admitted under -- its ceiling, its
wave size, and how many children have been released so far -- has to survive a
leader restart, or a control plane that dies between two waves either forgets
the remaining children or releases them all at once. Neither is acceptable: the
first loses work the submitter was told was owed, and the second is the queue
flood waves exist to prevent.

A new table rather than columns on ``workloads``: the persisted workload
document is decoded with ``extra='forbid'``, so the first write of a new field
would make every workload unreadable to a rolled-back replica. Nothing selects
from this table on an older build, so a rollback loses the wave plan and nothing
else -- and an expansion with no ledger is read as one admitted immediately,
which is what every expansion submitted before this migration in fact was.

Nothing is backfilled, for that same reason: an already-admitted expansion had
no wave plan, and inventing one now would re-gate children that are already
running.

Revision ID: 0016_expansion_admissions
Revises: 0015_attempt_topology
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016_expansion_admissions"
down_revision = "0015_attempt_topology"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "expansion_admissions",
        # Derived from the coordinator id, so a replayed submission rebinds the
        # row it already wrote instead of opening a second plan.
        sa.Column("id", sa.String(length=32), primary_key=True),
        # Unique, not merely indexed: one coordinator has exactly one plan, and
        # two rows would mean two ledgers releasing waves over the same children.
        sa.Column("workload_id", sa.String(length=96), nullable=False, unique=True),
        # A real column so the leader's wave phase scans only the expansions that
        # still owe children, rather than decoding every row to find them.
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("total_children", sa.Integer(), nullable=False),
        sa.Column("wave_size", sa.Integer(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_expansion_admissions_state", "expansion_admissions", ["state"])
    op.create_check_constraint(
        "ck_expansion_admissions_total_positive",
        "expansion_admissions",
        "total_children >= 1",
    )
    op.create_check_constraint(
        "ck_expansion_admissions_wave_positive",
        "expansion_admissions",
        "wave_size >= 1",
    )


def downgrade() -> None:
    op.drop_constraint("ck_expansion_admissions_wave_positive", "expansion_admissions", type_="check")
    op.drop_constraint("ck_expansion_admissions_total_positive", "expansion_admissions", type_="check")
    op.drop_index("ix_expansion_admissions_state", table_name="expansion_admissions")
    op.drop_table("expansion_admissions")
