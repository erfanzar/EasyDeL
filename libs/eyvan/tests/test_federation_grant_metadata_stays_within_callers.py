# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Federation grant metadata stays within the caller's administrative scope."""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import Permission, Principal, ResourceLimits, Role, RoleGrant, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.federation import BudgetGrantRecord, GrantState, QuotaGrantRecord


async def _seed() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository, plane_id="home")
    now = utcnow()
    async with repository.transaction() as tx:
        for org, project in (("acme", "a"), ("acme", "b"), ("other", "a")):
            for state in (GrantState.ACTIVE, GrantState.REVOKED):
                values = dict(
                    org_id=org,
                    project_id=project,
                    home_plane_id="home",
                    grantee_plane_id="peer",
                    issued_at=now,
                    expires_at=now + timedelta(days=1),
                    state=state,
                )
                await tx.put_budget_grant(
                    BudgetGrantRecord(id=f"budget-{org}-{project}-{state.value}", amount=50, **values)
                )
                await tx.put_quota_grant(
                    QuotaGrantRecord(id=f"quota-{org}-{project}-{state.value}", limits=ResourceLimits(cpu=8), **values)
                )
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("include_inactive", [False, True])
@pytest.mark.parametrize("filters", [{}, {"org_id": "acme"}, {"org_id": "acme", "project_id": "b"}])
async def test_project_admin_cannot_list_sibling_grants(kind, include_inactive, filters) -> None:
    service = await _seed()
    principal = Principal(
        subject="project-admin", grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="a"),)
    )
    assert not service.authorizer.allows(principal, Permission.READ, org_id="acme", project_id="b")
    try:
        listing = service.list_budget_grants if kind == "budget" else service.list_quota_grants
        actual = await listing(principal, include_inactive=include_inactive, **filters)
        expected_states = ("active", "revoked") if include_inactive else ("active",)
        expected_ids = (
            set() if filters.get("project_id") == "b" else {f"{kind}-acme-a-{state}" for state in expected_states}
        )
        assert {grant.id for grant in actual} == expected_ids
    finally:
        await service.repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("include_inactive", [False, True])
@pytest.mark.parametrize("role", [Role.ORG_ADMIN, Role.FLEET_ADMIN])
async def test_org_and_fleet_admin_keep_their_grant_visibility(kind, include_inactive, role) -> None:
    service = await _seed()
    role_grant = RoleGrant(role=role, org_id="acme") if role is Role.ORG_ADMIN else RoleGrant(role=role)
    principal = Principal(subject="admin", grants=(role_grant,))
    try:
        listing = service.list_budget_grants if kind == "budget" else service.list_quota_grants
        actual = await listing(principal, include_inactive=include_inactive)
        scopes = (("acme", "a"), ("acme", "b"))
        if role is Role.FLEET_ADMIN:
            scopes += (("other", "a"),)
        states = ("active", "revoked") if include_inactive else ("active",)
        assert {grant.id for grant in actual} == {
            f"{kind}-{org}-{project}-{state}" for org, project in scopes for state in states
        }
        filtered = await listing(principal, org_id="acme", project_id="b", include_inactive=include_inactive)
        assert {grant.id for grant in filtered} == {f"{kind}-acme-b-{state}" for state in states}
    finally:
        await service.repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("role", [Role.PROJECT_VIEWER, Role.PROJECT_OPERATOR, Role.PROJECT_SUBMITTER])
async def test_non_admin_project_roles_do_not_gain_financial_listing_access(kind, role) -> None:
    service = await _seed()
    principal = Principal(subject="non-admin", grants=(RoleGrant(role=role, org_id="acme", project_id="a"),))
    try:
        listing = service.list_budget_grants if kind == "budget" else service.list_quota_grants
        assert await listing(principal, include_inactive=True) == []
    finally:
        await service.repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_project_admin_grants_combine_exact_scopes_without_org_inheritance(kind) -> None:
    service = await _seed()
    principal = Principal(
        subject="two-project-admin",
        grants=(
            RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="a"),
            RoleGrant(role=Role.PROJECT_ADMIN, org_id="other", project_id="a"),
        ),
    )
    try:
        listing = service.list_budget_grants if kind == "budget" else service.list_quota_grants
        actual = await listing(principal)
        assert {grant.id for grant in actual} == {f"{kind}-acme-a-active", f"{kind}-other-a-active"}
    finally:
        await service.repository.close()
