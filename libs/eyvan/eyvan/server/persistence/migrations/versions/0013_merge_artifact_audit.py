# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Merge the scoped-artifact and bounded-audit-history schema branches.

Revision ID: 0013_merge_artifact_audit
Revises: 0012_scoped_artifact_identity, 0012_audit_event_time
"""

from __future__ import annotations

revision = "0013_merge_artifact_audit"
down_revision = ("0012_scoped_artifact_identity", "0012_audit_event_time")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Merge two already-applied, independent schema branches."""


def downgrade() -> None:
    """Return the graph to its two explicit branch heads."""
