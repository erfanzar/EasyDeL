# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Join the credential rotation and cell stage heads.

Three additive tables were introduced independently against ``0015``, so the graph
has two heads and Alembic refuses to upgrade without being told they compose.
They do: ``credential_rotations``, ``cell_stages``, and ``expansion_admissions``
share no table, column, or constraint, so this revision only records that fact. It creates and drops
nothing, which is what makes it safe to apply to a database already carrying
any of them.

Revision ID: 0017_merge_stage_rot
Revises: 0016_cell_stages, 0016_credential_rotation, 0016_expansion_admissions
"""

from __future__ import annotations

revision = "0017_merge_stage_rot"
down_revision = ("0016_cell_stages", "0016_credential_rotation", "0016_expansion_admissions")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Record that the additive heads compose; there is nothing to apply."""


def downgrade() -> None:
    """Split back into three heads; no table is touched."""
