# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retiring peer authority retains the full lifetime slice, not only reported spend."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from eyvan.records import BudgetRecord, Principal, Role, RoleGrant, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import InputError
from eyvan.server.service.federation import GrantState, issuable_amount


async def _retire_spent_grant(mode: str) -> tuple[ControlPlaneService, Principal, datetime]:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository, plane_id="home")
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    moment = utcnow()
    async with repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(org_id="acme", project_id="research", limit=100.0, period="lifetime", updated_by="root")
        )
    for peer in ("peer-a", "peer-b"):
        await service.put_plane_peer(admin, plane_id=peer, endpoint=f"https://{peer}.example.test")
    grant = await service.issue_budget_grant(
        admin,
        org_id="acme",
        project_id="research",
        grantee_plane_id="peer-a",
        amount=100.0,
        expires_at=moment + timedelta(days=1),
        idempotency_key="first-grant",
        now=moment,
    )
    await service.record_grant_spend(grant_id=grant.id, reported_spend=80.0, now=moment)
    before = await service.federated_budget_status(admin, org_id="acme", project_id="research", now=moment)
    assert before.reported_peer_spend == before.total_spent == 80.0
    if mode == "revoke":
        await service.revoke_budget_grant(admin, grant_id=grant.id)
    elif mode == "expire":
        moment += timedelta(days=2)
        assert await service.expire_budget_grants_once(now=moment) == 1
    else:
        assert mode == "replace"
        await service.issue_budget_grant(
            admin,
            org_id="acme",
            project_id="research",
            grantee_plane_id="peer-a",
            amount=0.0,
            expires_at=moment + timedelta(days=1),
            idempotency_key="replacement-grant",
            now=moment,
        )
    async with repository.transaction() as tx:
        retained = await tx.get_budget_grant(grant.id)
        budget = await tx.get_budget("acme", "research")
    assert retained is not None
    assert retained.state is (GrantState.EXPIRED if mode == "expire" else GrantState.REVOKED)
    assert retained.reported_spend == 80.0
    assert budget is not None and budget.period == "lifetime" and budget.spent == 0.0
    return service, admin, moment


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["revoke", "expire", "replace"])
async def test_retired_grant_spend_remains_in_federated_status(mode: str) -> None:
    service, admin, moment = await _retire_spent_grant(mode)
    try:
        status = await service.federated_budget_status(admin, org_id="acme", project_id="research", now=moment)
        assert status.reported_peer_spend == 80.0
        assert status.total_spent == 80.0
        assert status.granted_away == 100.0
        assert status.active_grants == (1 if mode == "replace" else 0)
        async with service.repository.transaction() as tx:
            grants = await tx.list_budget_grants(org_id="acme", project_id="research")
        assert issuable_amount(100.0, grants, now=moment) == 0.0
    finally:
        await service.repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["revoke", "expire", "replace"])
async def test_retired_grant_authority_cannot_be_reissued_to_another_peer(mode: str) -> None:
    service, admin, moment = await _retire_spent_grant(mode)
    try:
        with pytest.raises(InputError, match="exceeds"):
            await service.issue_budget_grant(
                admin,
                org_id="acme",
                project_id="research",
                grantee_plane_id="peer-b",
                amount=1.0,
                expires_at=moment + timedelta(days=1),
                idempotency_key="fresh-peer-grant",
                now=moment,
            )
    finally:
        await service.repository.close()
