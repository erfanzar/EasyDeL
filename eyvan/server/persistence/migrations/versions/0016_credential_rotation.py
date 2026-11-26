# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the durable credential rotation ledger.

Additive, like ``0015_attempt_topology`` and for the same reason. The state a
rotation needs -- which generation is live, which fence holds it, and the exact
instant the superseded generation stops verifying -- could have been columns on
an existing persisted record, and the first rotation would then have made that
record unreadable to a rolled-back replica decoding it with ``extra='forbid'``.
A replica that predates this table never selects from it and simply never
rotates, which is the correct behaviour for a build that does not know how.

Keyed by ``(credential, generation)`` rather than by the credential. A rotation
whose outcome could not be observed is recorded as ``inconclusive`` and is the
one row an operator must still act on; a key on the credential alone would let
the next attempt overwrite it.

No material and nothing derived from material is stored here, deliberately --
not even a digest, which would be an offline cracking target carried by every
backup and support bundle. ``generation`` is the only reference to a key.

Revision ID: 0016_credential_rotation
Revises: 0015_attempt_topology
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016_credential_rotation"
down_revision = "0015_attempt_topology"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "credential_rotations",
        sa.Column("credential", sa.String(length=96), primary_key=True),
        sa.Column("generation", sa.Integer(), primary_key=True),
        # Real columns, not JSON extractions: "is a rotation in flight" has to be
        # answerable from psql by an operator whose control plane is the thing
        # that is misbehaving.
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("fence", sa.String(length=64), nullable=False),
        sa.Column("overlap_s", sa.Float(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_check_constraint(
        # Generation 1 is the material a deployment started with, which nothing
        # rotated into, so a row here is always at least the second generation.
        "ck_credential_rotations_generation_positive",
        "credential_rotations",
        "generation >= 2",
    )
    op.create_check_constraint(
        "ck_credential_rotations_overlap_non_negative",
        "credential_rotations",
        "overlap_s >= 0",
    )


def downgrade() -> None:
    op.drop_constraint("ck_credential_rotations_overlap_non_negative", "credential_rotations", type_="check")
    op.drop_constraint("ck_credential_rotations_generation_positive", "credential_rotations", type_="check")
    op.drop_table("credential_rotations")
