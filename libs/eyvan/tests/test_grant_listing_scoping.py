# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A project admin's grant listing stops at their own projects.

``list_role_grants`` must not expose sibling projects' subjects, roles or
reasons to a PROJECT_ADMIN who does not administer those projects.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    Principal,
    Role,
    RoleGrant,
    RoleGrantRecord,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


@pytest.mark.asyncio
async def test_a_project_admin_does_not_see_sibling_project_grants() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    for subject, role, project in (
        ("bob", Role.ORG_ADMIN, None),
        ("carol", Role.PROJECT_ADMIN, "p1"),
        ("dave", Role.PROJECT_ADMIN, "p2"),
    ):
        await service.issue_role_grant(
            fleet_admin,
            RoleGrantRecord(
                subject=subject,
                grant=RoleGrant(role=role, org_id="acme", project_id=project),
                granted_by="system",
            ),
            idempotency_key=f"grant-{subject}",
        )
    p1_admin = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="p1"),),
    )

    visible = {item.subject for item in await service.list_role_grants(p1_admin)}

    assert visible == {"bob", "carol"}, "sibling-project grants must not be visible"


@pytest.mark.asyncio
async def test_an_org_admin_still_sees_the_whole_organization() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    for subject, role, project in (
        ("bob", Role.ORG_ADMIN, None),
        ("carol", Role.PROJECT_ADMIN, "p1"),
        ("dave", Role.PROJECT_ADMIN, "p2"),
    ):
        await service.issue_role_grant(
            fleet_admin,
            RoleGrantRecord(
                subject=subject,
                grant=RoleGrant(role=role, org_id="acme", project_id=project),
                granted_by="system",
            ),
            idempotency_key=f"grant-{subject}",
        )
    org_admin = Principal(subject="org-admin", grants=(RoleGrant(role=Role.ORG_ADMIN, org_id="acme"),))

    visible = {item.subject for item in await service.list_role_grants(org_admin)}

    assert visible == {"bob", "carol", "dave"}
