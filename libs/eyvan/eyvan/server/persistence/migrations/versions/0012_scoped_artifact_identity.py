# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Make managed artifact identity tenant-scoped.

This is a maintenance-only migration. All 0011 writers must be quiesced before
it runs: their URI-only reads and writes cannot safely coexist with the scoped
primary keys introduced here. The locks below freeze the relevant ledgers while
the preflight and key switch execute, but they are not a substitute for stopping
old replicas; a still-running 0011 process could resume after the transaction.

The managed-artifact preflight is intentionally fail-closed. It preserves the
single existing ledger row in exactly the typed scope already recorded on that
row and verifies the JSON record agrees. It never derives missing ownership or
size, and it never clones a URI into scopes found only in references. References
receive typed scope columns only from their own recorded JSON scope, which must
match their workload's durable typed scope.

Downgrade is conditional, not a general rollback. Once two scoped rows share a
URI, collapsing them to the 0011 URI primary key would discard one tenant's
ledger. The downgrade refuses before changing schema in that case. Production
recovery after scoped writes is forward repair or restoration of the matching
pre-migration database and object-store snapshot.

Revision ID: 0012_scoped_artifact_identity
Revises: 0011_quota_grants
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_scoped_artifact_identity"
down_revision = "0011_quota_grants"
branch_labels = None
depends_on = None

_LOCK_TABLES = "LOCK TABLE artifact_references, managed_artifacts, workloads IN ACCESS EXCLUSIVE MODE"

_MANAGED_SCOPE_PREFLIGHT = r"""
DO $eyvan$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM managed_artifacts
        WHERE org_id IS NULL
           OR btrim(org_id) = ''
           OR project_id IS NULL
           OR btrim(project_id) = ''
           OR jsonb_typeof(data::jsonb) <> 'object'
           OR NULLIF(btrim(data ->> 'org_id'), '') IS NULL
           OR NULLIF(btrim(data ->> 'project_id'), '') IS NULL
           OR NULLIF(btrim(data ->> 'uri'), '') IS NULL
           OR org_id <> data ->> 'org_id'
           OR project_id <> data ->> 'project_id'
           OR uri <> data ->> 'uri'
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0012 preflight: managed artifact scope missing/mismatched; ownership will not be synthesized';
    END IF;
END
$eyvan$
"""

_REFERENCE_RECORD_PREFLIGHT = r"""
DO $eyvan$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM artifact_references
        WHERE jsonb_typeof(data::jsonb) <> 'object'
           OR NULLIF(btrim(data ->> 'org_id'), '') IS NULL
           OR NULLIF(btrim(data ->> 'project_id'), '') IS NULL
           OR NULLIF(btrim(data ->> 'uri'), '') IS NULL
           OR NULLIF(btrim(data ->> 'workload_id'), '') IS NULL
           OR uri <> data ->> 'uri'
           OR workload_id <> data ->> 'workload_id'
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0012 preflight: artifact reference has no complete recorded scope or disagrees with its typed identity';
    END IF;
END
$eyvan$
"""

_REFERENCE_WORKLOAD_PREFLIGHT = r"""
DO $eyvan$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM artifact_references AS reference
        LEFT JOIN workloads AS workload ON workload.id = reference.workload_id
        WHERE workload.id IS NULL
           OR workload.org_id IS NULL
           OR btrim(workload.org_id) = ''
           OR workload.project_id IS NULL
           OR btrim(workload.project_id) = ''
           OR jsonb_typeof(workload.data::jsonb) <> 'object'
           OR workload.org_id <> workload.data ->> 'org_id'
           OR workload.project_id <> workload.data ->> 'project_id'
           OR reference.data ->> 'org_id' <> workload.org_id
           OR reference.data ->> 'project_id' <> workload.project_id
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0012 preflight: artifact reference scope does not match its workload durable scope';
    END IF;
END
$eyvan$
"""

_REPORT_CROSS_SCOPE_URIS = r"""
DO $eyvan$
DECLARE
    shared_uri text;
BEGIN
    FOR shared_uri IN
        SELECT uri
        FROM artifact_references
        GROUP BY uri
        HAVING count(DISTINCT (org_id, project_id)) > 1
        ORDER BY uri
    LOOP
        RAISE WARNING
            '0012 inventory required: URI % has references in multiple scopes; no managed row was cloned',
            shared_uri;
    END LOOP;
END
$eyvan$
"""

_DOWNGRADE_PREFLIGHT = r"""
DO $eyvan$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM managed_artifacts
        GROUP BY uri
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0012 downgrade refused: scoped managed artifacts share a URI; restore the pre-migration snapshot';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM artifact_references
        GROUP BY uri, workload_id
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0012 downgrade refused: multiple scoped references collapse to one legacy URI/workload identity';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM (
            SELECT
                'ref-' || substring(
                    encode(sha256(convert_to(uri || chr(10) || workload_id, 'UTF8')), 'hex')
                    FROM 1 FOR 40
                ) AS legacy_id
            FROM artifact_references
        ) AS target_ids
        GROUP BY legacy_id
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION USING MESSAGE =
            '0012 downgrade refused: legacy artifact reference ids would collide';
    END IF;
END
$eyvan$
"""


def upgrade() -> None:
    # These statements run in Alembic's migration transaction. The deployment
    # must already have stopped every 0011 writer; the lock prevents a forgotten
    # in-flight transaction from changing evidence during the preflight.
    op.execute(_LOCK_TABLES)

    op.add_column("artifact_references", sa.Column("org_id", sa.String(length=128), nullable=True))
    op.add_column("artifact_references", sa.Column("project_id", sa.String(length=128), nullable=True))

    op.execute(_MANAGED_SCOPE_PREFLIGHT)
    op.execute(_REFERENCE_RECORD_PREFLIGHT)
    op.execute(_REFERENCE_WORKLOAD_PREFLIGHT)

    # Reference scope comes only from the reference's own durable record. A
    # workload is used to validate it, never as a source from which to invent it.
    op.execute("UPDATE artifact_references SET org_id = data ->> 'org_id', project_id = data ->> 'project_id'")
    op.alter_column("managed_artifacts", "org_id", existing_type=sa.String(length=128), nullable=False)
    op.alter_column("managed_artifacts", "project_id", existing_type=sa.String(length=128), nullable=False)
    op.alter_column("artifact_references", "org_id", existing_type=sa.String(length=128), nullable=False)
    op.alter_column("artifact_references", "project_id", existing_type=sa.String(length=128), nullable=False)

    # Report evidence of URI-only history without cloning the surviving ledger
    # row into those scopes. Inventory is the only safe source for missing rows.
    op.execute(_REPORT_CROSS_SCOPE_URIS)

    op.drop_constraint("managed_artifacts_pkey", "managed_artifacts", type_="primary")
    op.create_primary_key(
        "pk_managed_artifacts_scoped",
        "managed_artifacts",
        ["org_id", "project_id", "uri"],
    )
    op.create_index(
        "ix_managed_artifacts_scope_created",
        "managed_artifacts",
        ["org_id", "project_id", "created_at", "uri"],
    )

    op.drop_index("ix_artifact_references_live", table_name="artifact_references")
    op.drop_index("ix_artifact_references_uri", table_name="artifact_references")
    op.drop_index("ix_artifact_references_workload_id", table_name="artifact_references")
    op.drop_constraint("artifact_references_pkey", "artifact_references", type_="primary")
    op.create_primary_key(
        "pk_artifact_references_scoped",
        "artifact_references",
        ["org_id", "project_id", "uri", "workload_id"],
    )
    # The id is display/audit metadata. Recompute it over the full scope only
    # after the composite key owns correctness, so even a hash collision cannot
    # merge or reject two otherwise distinct references.
    scoped_id = (
        "'ref-' || substring(encode(sha256(convert_to("
        "org_id || chr(10) || project_id || chr(10) || uri || chr(10) || workload_id, "
        "'UTF8')), 'hex') FROM 1 FOR 40)"
    )
    # The typed display id and the JSON source-of-truth move together in this
    # one statement. Leaving data.id at its 0011 value makes the first ORM read
    # disagree with the typed row even though the primary-key migration passed.
    op.execute(
        f"UPDATE artifact_references SET id = {scoped_id}, "
        f"data = jsonb_set(data::jsonb, '{{id}}', to_jsonb({scoped_id}), true)::json"
    )
    op.create_index(
        "ix_artifact_references_scope_uri",
        "artifact_references",
        ["org_id", "project_id", "uri"],
    )
    op.create_index(
        "ix_artifact_references_scope_workload",
        "artifact_references",
        ["org_id", "project_id", "workload_id"],
    )
    op.create_index(
        "ix_artifact_references_live",
        "artifact_references",
        ["org_id", "project_id", "uri"],
        postgresql_where=sa.text("released_at IS NULL"),
    )


def downgrade() -> None:
    # Run every lossy-collapse check before the first schema change. This makes a
    # failed downgrade an explicit refusal rather than a half-applied rollback.
    op.execute(_LOCK_TABLES)
    op.execute(_DOWNGRADE_PREFLIGHT)

    op.drop_index("ix_artifact_references_live", table_name="artifact_references")
    op.drop_index("ix_artifact_references_scope_workload", table_name="artifact_references")
    op.drop_index("ix_artifact_references_scope_uri", table_name="artifact_references")
    op.drop_constraint("pk_artifact_references_scoped", "artifact_references", type_="primary")
    legacy_id = (
        "'ref-' || substring(encode(sha256(convert_to(uri || chr(10) || workload_id, 'UTF8')), 'hex') FROM 1 FOR 40)"
    )
    # Downgrade restores the complete N-1 row representation, not merely the
    # typed column. This is safe only after the collapse preflight above.
    op.execute(
        f"UPDATE artifact_references SET id = {legacy_id}, "
        f"data = jsonb_set(data::jsonb, '{{id}}', to_jsonb({legacy_id}), true)::json"
    )
    op.create_primary_key("artifact_references_pkey", "artifact_references", ["id"])
    op.drop_column("artifact_references", "project_id")
    op.drop_column("artifact_references", "org_id")
    op.create_index("ix_artifact_references_uri", "artifact_references", ["uri"])
    op.create_index("ix_artifact_references_workload_id", "artifact_references", ["workload_id"])
    op.create_index(
        "ix_artifact_references_live",
        "artifact_references",
        ["uri"],
        postgresql_where=sa.text("released_at IS NULL"),
    )

    op.drop_index("ix_managed_artifacts_scope_created", table_name="managed_artifacts")
    op.drop_constraint("pk_managed_artifacts_scoped", "managed_artifacts", type_="primary")
    op.create_primary_key("managed_artifacts_pkey", "managed_artifacts", ["uri"])
    op.alter_column("managed_artifacts", "project_id", existing_type=sa.String(length=128), nullable=True)
    op.alter_column("managed_artifacts", "org_id", existing_type=sa.String(length=128), nullable=True)
