# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the frozen per-attempt global rank topology table.

Purely additive, and deliberately so. The alternative was a field on the
persisted workload document, which every replica decodes with ``extra='forbid'``:
the first write would have made every workload unreadable to a rolled-back
build, and a control plane that cannot read its workloads schedules nothing. A
new table is invisible to older code instead -- it never selects from it -- so a
rollback loses the proof of a rank map and nothing else.

Nothing is backfilled. A row here is a claim that these exact cells, at these
exact generations and widths, are the processes an already-running attempt was
dispatched onto; today's cell inventory cannot support that claim, because a
reconciliation is free to change a cell's observed process width at any moment.
Attempts that predate this table therefore report an unknown topology and are
served with explicitly unproven rank identity, which is the honest answer.

Revision ID: 0015_attempt_topology
Revises: 0014_artifact_lifecycle
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015_attempt_topology"
down_revision = "0014_artifact_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workload_topologies",
        # Keyed by the attempt, not the workload: a retry is a different rank
        # space, so overwriting would destroy the only record of which host was
        # rank N during the attempt whose logs are being read.
        sa.Column("workload_id", sa.String(length=96), primary_key=True),
        sa.Column("attempt", sa.Integer(), primary_key=True),
        # ``sha256:`` plus 64 hex characters. A column rather than a JSON
        # extraction so a mismatch is greppable and comparable in SQL.
        sa.Column("digest", sa.String(length=71), nullable=False),
        sa.Column("total_rank_count", sa.Integer(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_check_constraint(
        "ck_workload_topologies_attempt_positive",
        "workload_topologies",
        "attempt >= 1",
    )
    op.create_check_constraint(
        "ck_workload_topologies_rank_count_positive",
        "workload_topologies",
        "total_rank_count >= 1",
    )


def downgrade() -> None:
    op.drop_constraint("ck_workload_topologies_rank_count_positive", "workload_topologies", type_="check")
    op.drop_constraint("ck_workload_topologies_attempt_positive", "workload_topologies", type_="check")
    op.drop_table("workload_topologies")
