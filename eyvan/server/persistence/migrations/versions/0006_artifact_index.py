# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the managed-artifact index and the reverse reference index.

The control plane could always serve artifacts and never knew how many existed
or how many bytes they occupied. Two new tables, no changes to existing ones:

``managed_artifacts`` is the byte ledger, keyed by the URI the store already
addresses objects with. ``size_bytes`` is a real column rather than a JSON field
because the quota check SUMs it on the upload path, and a JSON extraction cannot
use an index.

``artifact_references`` is the reverse index -- which workloads claim which
artifact -- so "unreferenced" becomes a computable property. Nothing in this
revision or the code that writes to it deletes an object; the index has to exist
and be trusted before any collector may be written against it.

The partial index on unreleased references is the shape that matters: the only
interesting query is "does this URI still have a live claim", and released
history would otherwise grow without bound inside the index that answers it.

Revision ID: 0006_artifact_index
Revises: 0005_retention_columns
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_artifact_index"
down_revision = "0005_retention_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "managed_artifacts",
        sa.Column("uri", sa.String(length=512), primary_key=True),
        sa.Column("org_id", sa.String(length=128), nullable=True),
        sa.Column("project_id", sa.String(length=128), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_managed_artifacts_org_id", "managed_artifacts", ["org_id"])
    op.create_index("ix_managed_artifacts_project_id", "managed_artifacts", ["project_id"])
    op.create_index("ix_managed_artifacts_kind", "managed_artifacts", ["kind"])
    op.create_index("ix_managed_artifacts_created_at", "managed_artifacts", ["created_at"])
    # The quota check filters on (org_id, project_id) and sums size_bytes, so a
    # covering composite lets Postgres answer it index-only.
    op.create_index(
        "ix_managed_artifacts_scope_bytes",
        "managed_artifacts",
        ["org_id", "project_id", "size_bytes"],
    )

    op.create_table(
        "artifact_references",
        sa.Column("id", sa.String(length=200), primary_key=True),
        sa.Column("uri", sa.String(length=512), nullable=False),
        sa.Column("workload_id", sa.String(length=96), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_artifact_references_uri", "artifact_references", ["uri"])
    op.create_index("ix_artifact_references_workload_id", "artifact_references", ["workload_id"])
    op.create_index("ix_artifact_references_released_at", "artifact_references", ["released_at"])
    op.create_index("ix_artifact_references_created_at", "artifact_references", ["created_at"])
    op.create_index(
        "ix_artifact_references_live",
        "artifact_references",
        ["uri"],
        postgresql_where=sa.text("released_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_artifact_references_live", table_name="artifact_references")
    op.drop_index("ix_artifact_references_created_at", table_name="artifact_references")
    op.drop_index("ix_artifact_references_released_at", table_name="artifact_references")
    op.drop_index("ix_artifact_references_workload_id", table_name="artifact_references")
    op.drop_index("ix_artifact_references_uri", table_name="artifact_references")
    op.drop_table("artifact_references")
    op.drop_index("ix_managed_artifacts_scope_bytes", table_name="managed_artifacts")
    op.drop_index("ix_managed_artifacts_created_at", table_name="managed_artifacts")
    op.drop_index("ix_managed_artifacts_kind", table_name="managed_artifacts")
    op.drop_index("ix_managed_artifacts_project_id", table_name="managed_artifacts")
    op.drop_index("ix_managed_artifacts_org_id", table_name="managed_artifacts")
    op.drop_table("managed_artifacts")
