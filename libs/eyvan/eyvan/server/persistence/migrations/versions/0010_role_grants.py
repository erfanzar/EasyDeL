# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Store role grants this control plane issued and can revoke.

``eyvan_grants`` in an OIDC token was the only source of authority, so onboarding
a tenant meant teaching an identity provider to mint a bespoke claim -- one
mentioned nowhere in the README -- and revoking access meant an IdP change rather
than an API call. Deployments whose IdP could not be taught a custom claim could
not use Eyvan's authorization at all.

The claim is still honored. This is a second source, unioned with it, so an
existing deployment keeps working unchanged.

Keyed by (subject, role, org_id, project_id): re-issuing the same grant
overwrites rather than accumulating, because holding a role twice means nothing.
The empty string stands in for an absent scope so the key stays a primary key --
NULLs are not equal to each other in SQL, and a nullable component would let the
same fleet-admin grant be inserted an unbounded number of times.

``expires_at`` is indexed because the only query that filters on it is the sweep
looking for grants to drop, which pages on exactly that column.

Revision ID: 0010_role_grants
Revises: 0009_tenant_registry
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_role_grants"
down_revision = "0009_tenant_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "role_grants",
        sa.Column("subject", sa.String(length=256), primary_key=True),
        sa.Column("role", sa.String(length=32), primary_key=True),
        sa.Column("org_id", sa.String(length=128), primary_key=True),
        sa.Column("project_id", sa.String(length=128), primary_key=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_role_grants_subject", "role_grants", ["subject"])
    op.create_index("ix_role_grants_expires_at", "role_grants", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_role_grants_expires_at", table_name="role_grants")
    op.drop_index("ix_role_grants_subject", table_name="role_grants")
    op.drop_table("role_grants")
