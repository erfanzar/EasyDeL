# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the managed per-cell stage ledger.

Purely additive, and deliberately so. The alternative was a field on the
persisted cell document, which every replica decodes with ``extra='forbid'``:
the first write would have made every cell unreadable to a rolled-back build,
and a control plane that cannot read its cells schedules nothing. A new table is
invisible to older code instead -- it never selects from it -- so a rollback
loses the record of what is staged and nothing else.

Nothing is backfilled, and nothing could be. A row here is a claim about bytes
on a specific host at a specific generation, and the only existing signal --
``CellRecord.materialized_assets``, an agent's free-form report -- carries no
size, no digest and no generation. Inventing rows from it would make the byte
budget bound a number nobody measured. Cells that predate this table therefore
hold no stages, which reads as "nothing is proven staged here yet" and is the
honest answer.

Revision ID: 0016_cell_stages
Revises: 0015_attempt_topology
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016_cell_stages"
down_revision = "0015_attempt_topology"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cell_stages",
        # Keyed by (cell, asset): one cell holds one copy of one named asset, and
        # a second row for the same pair would let the byte budget count the same
        # disk space twice.
        sa.Column("cell_id", sa.String(length=96), primary_key=True),
        sa.Column("asset", sa.String(length=512), primary_key=True),
        # Real columns rather than JSON extractions because both are filtered on:
        # the staging phase selects unproven rows per cell, and placement selects
        # staged rows by asset.
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("cell_generation", sa.Integer(), nullable=False),
        sa.Column("reserved_bytes", sa.BigInteger(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_cell_stages_cell_id", "cell_stages", ["cell_id"])
    op.create_index("ix_cell_stages_asset_state", "cell_stages", ["asset", "state"])
    op.create_check_constraint(
        "ck_cell_stages_generation_positive",
        "cell_stages",
        "cell_generation >= 1",
    )
    op.create_check_constraint(
        "ck_cell_stages_reserved_bytes_non_negative",
        "cell_stages",
        "reserved_bytes >= 0",
    )


def downgrade() -> None:
    op.drop_constraint("ck_cell_stages_reserved_bytes_non_negative", "cell_stages", type_="check")
    op.drop_constraint("ck_cell_stages_generation_positive", "cell_stages", type_="check")
    op.drop_index("ix_cell_stages_asset_state", table_name="cell_stages")
    op.drop_index("ix_cell_stages_cell_id", table_name="cell_stages")
    op.drop_table("cell_stages")
