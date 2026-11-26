# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Create the Eyvan control-plane and transactional-outbox schema.

Revision ID: 0002_transactional_outbox
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_transactional_outbox"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "schema_revisions",
        sa.Column("revision", sa.String(64), primary_key=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=False),
    )
    table_key_lengths = {
        "workloads": 96,
        "cells": 96,
        "leases": 96,
        "policies": 256,
        "quotas": 384,
        "budgets": 256,
    }
    for table, key_length in table_key_lengths.items():
        op.create_table(
            table,
            sa.Column("id", sa.String(key_length), primary_key=True),
            sa.Column("data", sa.JSON(), nullable=False),
        )
    op.create_table(
        "idempotency_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("scope", sa.String(256), nullable=False),
        sa.Column("operation", sa.String(96), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.UniqueConstraint("scope", "operation", "key", name="uq_idempotency_scope_operation_key"),
    )
    op.create_index("ix_idempotency_keys_scope", "idempotency_keys", ["scope"])
    op.create_table(
        "outbox",
        sa.Column("id", sa.String(96), primary_key=True),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_until", sa.DateTime(timezone=True)),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_outbox_state", "outbox", ["state"])
    op.create_index("ix_outbox_available_at", "outbox", ["available_at"])
    op.create_index("ix_outbox_claimed_until", "outbox", ["claimed_until"])
    op.create_table(
        "audit_events",
        sa.Column("sequence", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(96), nullable=False, unique=True),
        sa.Column("org_id", sa.String(128)),
        sa.Column("project_id", sa.String(128)),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(256), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    for column in ("org_id", "project_id", "entity_type", "entity_id"):
        op.create_index(f"ix_audit_events_{column}", "audit_events", [column])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("outbox")
    op.drop_table("idempotency_keys")
    for table in ("budgets", "quotas", "policies", "leases", "cells", "workloads"):
        op.drop_table(table)
    op.drop_table("schema_revisions")
