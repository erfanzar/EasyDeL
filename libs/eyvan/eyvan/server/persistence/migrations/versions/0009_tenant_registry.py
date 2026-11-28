# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Register the organizations and projects every request already names.

``org_id`` and ``project_id`` were free-form strings on every route, validated
for length and nothing else. Nothing could answer "which organizations exist",
so nothing could list them, and a typo in ``PUT /quotas`` silently created a
quota for a tenant that would never submit anything.

Two tables rather than one with a nullable ``project_id``: an organization has an
owner and a lifecycle of its own, and folding projects into the same table would
make "disable this organization" and "disable this project" the same row shape
with different meanings.

Projects are keyed ``(org_id, id)`` because project names are unique only within
a tenant. A globally unique project id would make the second organization's
``research`` project impossible to create for a reason no user could see.

``enabled`` rather than deletion: workloads, leases and audit events reference
these ids for as long as history is retained, so removing a row would orphan all
of them. Disabling refuses new work and leaves the past readable.

Revision ID: 0009_tenant_registry
Revises: 0008_progress_reports
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_tenant_registry"
down_revision = "0008_progress_reports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(length=128), primary_key=True),
        # A real column rather than a JSON field: listing a tenant's projects and
        # filtering enabled organizations are both index scans, and a JSON
        # extraction cannot use an index -- the lesson migration 0004 records.
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_table(
        "projects",
        sa.Column("org_id", sa.String(length=128), primary_key=True),
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_projects_org_id", "projects", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_projects_org_id", table_name="projects")
    op.drop_table("projects")
    op.drop_table("organizations")
