# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the fenced managed-artifact lifecycle.

Existing rows are deliberately quarantined. A SQL row cannot prove which live
provider generation it names; the bounded inventory reconciler must observe an
exact scoped object before an operator can activate it. Old writers must remain
quiesced throughout this maintenance-only migration.

Revision ID: 0014_artifact_lifecycle
Revises: 0013_merge_artifact_audit
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014_artifact_lifecycle"
down_revision = "0013_merge_artifact_audit"
branch_labels = None
depends_on = None

_LOCK = "LOCK TABLE artifact_references, managed_artifacts IN ACCESS EXCLUSIVE MODE"

_PREFLIGHT = r"""
DO $eyvan$
BEGIN
    IF EXISTS (
        SELECT 1 FROM managed_artifacts
        WHERE jsonb_typeof(data::jsonb) <> 'object'
           OR NULLIF(btrim(org_id), '') IS NULL
           OR NULLIF(btrim(project_id), '') IS NULL
           OR NULLIF(btrim(uri), '') IS NULL
           OR org_id <> data ->> 'org_id'
           OR project_id <> data ->> 'project_id'
           OR uri <> data ->> 'uri'
           OR size_bytes < 0
           OR (data ->> 'size_bytes') IS NULL
           OR size_bytes <> (data ->> 'size_bytes')::bigint
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0014 preflight: artifact identity/accounting is incomplete; lifecycle ownership will not be guessed';
    END IF;
END
$eyvan$
"""

_DOWNGRADE_PREFLIGHT = r"""
DO $eyvan$
BEGIN
    IF EXISTS (
        SELECT 1 FROM managed_artifacts
        WHERE state <> 'quarantined'
           OR state IS DISTINCT FROM data ->> 'state'
           OR record_generation <> 1
           OR record_generation::text IS DISTINCT FROM data ->> 'record_generation'
           OR object_version IS NOT NULL
           OR object_version IS DISTINCT FROM data ->> 'object_version'
           OR data::jsonb ?| ARRAY[
               'upload_started_at', 'activated_at', 'deletion_claim_id',
               'deletion_started_at', 'deleted_at'
           ]
           OR COALESCE(data ->> 'quarantine_reason', '') <>
              'legacy object version has not been verified'
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0014 downgrade refused: lifecycle state was written; restore the snapshot or repair forward';
    END IF;
END
$eyvan$
"""

_LIFECYCLE_JSON_KEYS = """
ARRAY[
    'state', 'record_generation', 'object_version', 'upload_started_at',
    'activated_at', 'deletion_claim_id', 'deletion_started_at',
    'deleted_at', 'quarantine_reason'
]
"""


def upgrade() -> None:
    op.execute(_LOCK)
    op.execute(_PREFLIGHT)

    op.add_column("managed_artifacts", sa.Column("state", sa.String(length=32), nullable=True))
    op.add_column("managed_artifacts", sa.Column("record_generation", sa.BigInteger(), nullable=True))
    op.add_column("managed_artifacts", sa.Column("object_version", sa.String(length=256), nullable=True))
    op.add_column("managed_artifacts", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))

    op.execute(
        """
        UPDATE managed_artifacts
        SET state = 'quarantined',
            record_generation = 1,
            object_version = NULL,
            updated_at = COALESCE(created_at, CURRENT_TIMESTAMP),
            data = jsonb_set(
                jsonb_set(
                    jsonb_set(
                        jsonb_set(data::jsonb, '{state}', '"quarantined"'::jsonb, true),
                        '{record_generation}', '1'::jsonb, true
                    ),
                    '{object_version}', 'null'::jsonb, true
                ),
                '{quarantine_reason}',
                to_jsonb('legacy object version has not been verified'::text),
                true
            )::json
        """
    )
    op.alter_column("managed_artifacts", "state", existing_type=sa.String(length=32), nullable=False)
    op.alter_column("managed_artifacts", "record_generation", existing_type=sa.BigInteger(), nullable=False)
    op.alter_column("managed_artifacts", "updated_at", existing_type=sa.DateTime(timezone=True), nullable=False)
    op.create_check_constraint(
        "ck_managed_artifacts_record_generation_positive",
        "managed_artifacts",
        "record_generation >= 1",
    )
    op.create_check_constraint(
        "ck_managed_artifacts_lifecycle_state",
        "managed_artifacts",
        "state IN ('pending_upload', 'active', 'deleting', 'deleted', 'quarantined')",
    )
    op.create_index("ix_managed_artifacts_state", "managed_artifacts", ["state"])
    op.create_index("ix_managed_artifacts_updated_at", "managed_artifacts", ["updated_at"])
    op.create_index(
        "ix_managed_artifacts_scope_state_updated",
        "managed_artifacts",
        ["org_id", "project_id", "state", "updated_at"],
    )


def downgrade() -> None:
    op.execute(_LOCK)
    op.execute(_DOWNGRADE_PREFLIGHT)

    op.execute(f"UPDATE managed_artifacts SET data = (data::jsonb - {_LIFECYCLE_JSON_KEYS})::json")
    op.drop_index("ix_managed_artifacts_scope_state_updated", table_name="managed_artifacts")
    op.drop_index("ix_managed_artifacts_updated_at", table_name="managed_artifacts")
    op.drop_index("ix_managed_artifacts_state", table_name="managed_artifacts")
    op.drop_constraint("ck_managed_artifacts_lifecycle_state", "managed_artifacts", type_="check")
    op.drop_constraint("ck_managed_artifacts_record_generation_positive", "managed_artifacts", type_="check")
    op.drop_column("managed_artifacts", "updated_at")
    op.drop_column("managed_artifacts", "object_version")
    op.drop_column("managed_artifacts", "record_generation")
    op.drop_column("managed_artifacts", "state")
