# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tiny retained capacity rights must bind issuance and quota-only admission."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal, localcontext

import pytest
from test_federation_retained_authority import ADMIN, _home, _issue, _lease, _submit
from test_federation_retained_authority import repository as _repository_fixture

from eyvan.records import InfeasibilityCode, ResourceGang, ResourceLimits, ResourceVector, utcnow
from eyvan.server.service.errors import ConflictError, InputError
from eyvan.server.service.federation import GrantState, QuotaGrantRecord, outstanding_quota, quota_headroom

repository = _repository_fixture


async def _tiny_right(service, state):
    moment = utcnow()
    grant = await _issue(service, amount=1e-30, now=moment, kind="quota")
    if state is GrantState.REVOKED:
        await service.revoke_quota_grant(ADMIN, grant_id=grant.id)
    elif state is GrantState.EXPIRED:
        assert await service.expire_budget_grants_once(now=moment + timedelta(hours=2)) == 1
    async with service.repository.transaction() as tx:
        assert (await tx.get_quota_grant(grant.id)).state is state
        assert await tx.list_budget_grants() == [], "exercise quota precision without a budget grant context"
    return grant


@pytest.mark.parametrize("state", list(GrantState))
async def test_tiny_retained_quota_right_refuses_full_new_grant(repository, state):
    service = await _home(repository)
    grant = await _tiny_right(service, state)
    with pytest.raises(InputError, match="headroom"):
        await _issue(service, amount=1, peer="other", key="full", kind="quota")
    async with repository.transaction() as tx:
        assert [stored.id for stored in await tx.list_quota_grants()] == [grant.id]


@pytest.mark.parametrize("state", list(GrantState))
async def test_tiny_retained_quota_right_refuses_full_local_admission(repository, state):
    service = await _home(repository)
    await _tiny_right(service, state)
    await _submit(service)
    assert not (await service.schedule_once(ADMIN)).placements, "retired and active tiny rights remain reserved"


async def test_stale_full_cpu_proposal_is_refused_after_tiny_quota_issuance(repository):
    service = await _home(repository)
    workload = await _submit(service)
    plan = await service.schedule_once(ADMIN)
    assert len(plan.placements) == 1
    proposal = _lease(workload, plan.placements[0])
    await _tiny_right(service, GrantState.ACTIVE)
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="stale")
    async with repository.transaction() as tx:
        assert await tx.get_lease(proposal.id) is None


async def _reserve_tiny_cpu(service, *, required_labels=None):
    workload = await _submit(
        service,
        key="tiny",
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1e-30, memory_gib=0)),
        required_labels=required_labels or {},
        intra_band_priority=1000,
    )
    plan = await service.schedule_once(ADMIN)
    tiny_placements = [item for item in plan.placements if item.workload_id == workload.id]
    assert len(tiny_placements) == 1
    lease, _ = await service.create_lease(ADMIN, _lease(workload, tiny_placements[0]), idempotency_key="tiny-lease")
    async with service.repository.transaction() as tx:
        stored = await tx.get_lease(lease.id)
        assert stored.resources.per_replica.cpu == 1e-30
        assert await tx.list_budget_grants() == []
    return lease


async def test_tiny_active_local_cpu_reservation_refuses_full_quota_grant(repository):
    service = await _home(repository, limit=10)
    lease = await _reserve_tiny_cpu(service)
    with pytest.raises(InputError, match="headroom"):
        await _issue(service, amount=1, kind="quota")
    async with repository.transaction() as tx:
        assert await tx.get_lease(lease.id) is not None
        assert await tx.list_quota_grants() == []


async def _add_spare_cell(service):
    async with service.repository.transaction() as tx:
        cell = await tx.get_cell("cell")
        assert cell is not None
        await tx.put_cell(cell.model_copy(update={"labels": {"slot": "full"}}))
        await tx.put_cell(cell.model_copy(update={"id": "spare", "labels": {"slot": "tiny"}}))


async def test_tiny_active_local_cpu_reservation_refuses_full_local_admission(repository):
    service = await _home(repository, limit=10)
    await _add_spare_cell(service)
    await _reserve_tiny_cpu(service, required_labels={"slot": "tiny"})
    full = await _submit(service, key="full", required_labels={"slot": "full"})
    result = await service.schedule_once(ADMIN)
    assert not result.placements, "the unused cell cannot bypass retained quota usage"
    assert any(
        item.workload_id == full.id and item.code is InfeasibilityCode.QUOTA_EXCEEDED for item in result.infeasibilities
    )


async def test_stale_full_cpu_proposal_is_refused_after_tiny_local_reservation(repository):
    service = await _home(repository, limit=10)
    await _add_spare_cell(service)
    full = await _submit(service, key="full", required_labels={"slot": "full"})
    before = await service.schedule_once(ADMIN)
    assert len(before.placements) == 1
    proposal = _lease(full, before.placements[0])
    assert proposal.cell_id == "cell"
    tiny = await _reserve_tiny_cpu(service, required_labels={"slot": "tiny"})
    assert tiny.cell_id == "spare", "leave the stale proposal's cell unoccupied"
    with pytest.raises(ConflictError, match="quota"):
        await service.create_lease(ADMIN, proposal, idempotency_key="stale-local")
    async with repository.transaction() as tx:
        assert await tx.get_lease(proposal.id) is None
        assert await tx.get_lease(tiny.id) is not None
        assert await tx.list_budget_grants() == []
        assert await tx.list_quota_grants() == []


def _grant(cpu):
    now = utcnow()
    return QuotaGrantRecord(
        id=f"quota-{cpu}",
        org_id="acme",
        project_id="research",
        home_plane_id="home",
        grantee_plane_id="peer",
        limits=ResourceLimits(cpu=cpu),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )


def test_public_quota_headroom_preserves_float_interface_with_conservative_projection():
    now = utcnow()
    headroom = quota_headroom(ResourceLimits(cpu=1), [_grant(1e-30)], now=now)
    assert all(isinstance(quantity, float) for quantity in headroom.values())
    with localcontext(prec=2048):
        assert Decimal(str(headroom["cpu"])) <= Decimal(1) - Decimal("1e-30")
    assert headroom["cpu"] < 1


def test_public_outstanding_quota_preserves_float_interface_with_conservative_projection():
    now = utcnow()
    totals = outstanding_quota([_grant(1), _grant(1e-30)], now=now)
    assert all(isinstance(quantity, float) for quantity in totals.values())
    with localcontext(prec=2048):
        assert Decimal(str(totals["cpu"])) >= Decimal(1) + Decimal("1e-30")
    assert totals["cpu"] > 1
