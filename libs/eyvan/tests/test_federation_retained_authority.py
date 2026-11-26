# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Home issuance and admission partition authority, including retired rights.

No final-consumption receipt exists. Retirement therefore cannot refund either
unreported monetary authority or capacity that a peer may still be using.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from test_admission_region_policy_cross_plane_authority import ADMIN, _lease, _submit

from eyvan.records import BudgetRecord, CellRecord, CellState, QuotaRecord, ResourceLimits, ResourceVector, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.base import TransientRepositoryError
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError, InputError
from eyvan.server.service.federation import GrantState
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.state import ACTIVE_LEASE_STATES


@pytest.fixture(params=["memory", "local", "postgres"])
async def repository(request, tmp_path):
    engine = admin = None
    schema = "federation_" + uuid4().hex
    if request.param == "memory":
        repo = InMemoryRepository()
    elif request.param == "local":
        repo = LocalJsonRepository(tmp_path / "state.json")
    else:
        url = request.getfixturevalue("postgres_url")
        admin = create_async_engine(url)
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
        await create_schema(engine)
        repo = SqlAlchemyRepository(engine)
    await repo.open()
    try:
        yield repo
    finally:
        await repo.close()
        if engine is not None:
            await engine.dispose()
        if admin is not None:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await admin.dispose()


async def _home(repository, *, limit=1.0, spent=0.0, price=1.0, period="lifetime", started=None, soft=False):
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id="home")
    async with repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(
                org_id="acme",
                project_id="research",
                limit=limit,
                spent=spent,
                period=period,
                period_started_at=started,
                hard_limit=not soft,
                updated_by="root",
            )
        )
        await tx.put_quota(
            QuotaRecord(org_id="acme", project_id="research", limits=ResourceLimits(cpu=1), updated_by="root")
        )
        await tx.put_cell(
            CellRecord(
                id="cell",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=16, memory_gib=64),
                allocatable=ResourceVector(cpu=16, memory_gib=64),
                unit_prices={"cpu": price},
            )
        )
    for peer in ("peer", "other"):
        await service.put_plane_peer(ADMIN, plane_id=peer, endpoint=f"https://{peer}.example.test")
    return service


async def _issue(service, *, amount=1.0, peer="peer", key="grant", now=None, kind="budget"):
    moment = now or utcnow()
    fields = dict(
        org_id="acme",
        project_id="research",
        grantee_plane_id=peer,
        expires_at=moment + timedelta(hours=1),
        idempotency_key=key,
        now=moment,
    )
    if kind == "budget":
        return await service.issue_budget_grant(ADMIN, amount=amount, **fields)
    return await service.issue_quota_grant(ADMIN, limits=ResourceLimits(cpu=amount), **fields)


@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_home_issuance_and_admission_share_authority(repository, kind):
    service = await _home(repository)
    workload = await _submit(service)
    before = await service.schedule_once(ADMIN)
    proposal = _lease(workload, before.placements[0])
    await _issue(service, kind=kind)
    assert not (await service.schedule_once(ADMIN)).placements
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="lease")
    async with repository.transaction() as tx:
        assert await tx.get_lease(proposal.id) is None


@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_issuance_counts_existing_local_reservations(repository, kind):
    service = await _home(repository)
    workload = await _submit(service)
    plan = await service.schedule_once(ADMIN)
    lease, _ = await service.create_lease(ADMIN, _lease(workload, plan.placements[0]), idempotency_key="lease")
    with pytest.raises(InputError):
        await _issue(service, kind=kind)
    async with repository.transaction() as tx:
        assert (await tx.get_lease(lease.id)).reserved_cost == 1
        assert await tx.list_budget_grants() == []
        assert await tx.list_quota_grants() == []


@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("first", ["issuance", "admission"])
async def test_concurrent_issuance_and_admission_cannot_both_succeed(repository, kind, first):
    service = await _home(repository)
    workload = await _submit(service)
    plan = await service.schedule_once(ADMIN)
    proposal = _lease(workload, plan.placements[0])
    other_service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id="home")
    operations = [
        _issue(service, kind=kind),
        other_service.create_lease(ADMIN, proposal, idempotency_key="lease"),
    ]
    if first == "admission":
        operations.reverse()
    results = await asyncio.gather(*operations, return_exceptions=True)
    assert sum(not isinstance(result, Exception) for result in results) == 1, results
    for index, result in enumerate(results):
        if isinstance(result, TransientRepositoryError):
            # PostgreSQL SERIALIZABLE may abort the losing snapshot. A fresh
            # public-service retry must now reject, not issue/admit a second right.
            with pytest.raises((InputError, ConflictError)):
                if index == (0 if first == "issuance" else 1):
                    await _issue(service, kind=kind)
                else:
                    await other_service.create_lease(ADMIN, proposal, idempotency_key="lease")
        elif isinstance(result, Exception):
            assert isinstance(result, (InputError, ConflictError)), results
    async with repository.transaction() as tx:
        grants = await (tx.list_budget_grants() if kind == "budget" else tx.list_quota_grants())
        leases = [lease for lease in await tx.list_leases() if lease.state in ACTIVE_LEASE_STATES]
    assert len(grants) + len(leases) == 1


@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("mode", ["revoke", "expire", "replace"])
async def test_retirement_retains_full_authority_after_restart(repository, kind, mode):
    service = await _home(repository)
    moment = utcnow()
    grant = await _issue(service, now=moment, kind=kind)
    if kind == "budget":
        await service.record_grant_spend(grant_id=grant.id, reported_spend=0.8, now=moment)
    if mode == "revoke":
        revoke = service.revoke_budget_grant if kind == "budget" else service.revoke_quota_grant
        await revoke(ADMIN, grant_id=grant.id)
    elif mode == "expire":
        assert await service.expire_budget_grants_once(now=moment + timedelta(hours=2)) == 1
    elif kind == "budget":
        await _issue(service, amount=0, key="replace", now=moment)
    else:
        # Quota grants must be positive. Refusing this replacement must leave
        # the original rights intact, not revoke them before checking headroom.
        with pytest.raises(InputError):
            await _issue(service, amount=1, key="replace", now=moment, kind=kind)
    await repository.close()
    await repository.open()
    restarted = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id="home")
    await _submit(restarted)
    assert not (await restarted.schedule_once(ADMIN)).placements
    with pytest.raises(InputError):
        await _issue(restarted, amount=0.01, peer="other", key="fresh", kind=kind)
    async with repository.transaction() as tx:
        stored = await (tx.get_budget_grant(grant.id) if kind == "budget" else tx.get_quota_grant(grant.id))
    assert stored is not None
    if mode != "replace" or kind == "budget":
        assert stored.state is (GrantState.EXPIRED if mode == "expire" else GrantState.REVOKED)
    if kind == "budget":
        status = await restarted.federated_budget_status(ADMIN, org_id="acme", project_id="research")
        assert status.reported_peer_spend == status.total_spent == 0.8
        assert status.granted_away == 1, "retirement is not final-consumption proof"
        await restarted.record_grant_spend(grant_id=grant.id, reported_spend=0.9)
        assert (await restarted.federated_budget_status(ADMIN, org_id="acme", project_id="research")).total_spent == 0.9


async def test_reported_spend_is_not_subtracted_twice_and_decimal_boundary_is_exact(repository):
    service = await _home(repository, limit=0.3, price=0.2)
    grant = await _issue(service, amount=0.1)
    await service.record_grant_spend(grant_id=grant.id, reported_spend=0.08)
    workload = await _submit(service)
    plan = await service.schedule_once(ADMIN)
    assert len(plan.placements) == 1, "0.1 issued includes its 0.08 report; 0.2 remains locally"
    assert plan.placements[0].lease.reserved_cost == Decimal("0.2")
    await service.create_lease(ADMIN, _lease(workload, plan.placements[0]), idempotency_key="lease")
    with pytest.raises(InputError):
        await _issue(service, amount=1e-18, peer="other", key="tiny")


async def test_replacement_keeps_old_rights_reports_and_idempotent_replay(repository):
    service = await _home(repository, limit=1)
    now = utcnow()
    first = await _issue(service, amount=0.4, now=now)
    await service.record_grant_spend(grant_id=first.id, reported_spend=0.3)
    with pytest.raises(InputError):
        await _issue(service, amount=0.7, key="overdraw", now=now)
    second = await _issue(service, amount=0.6, key="replacement", now=now)
    replay = await _issue(service, amount=0.6, key="replacement", now=now)
    assert replay.id == second.id
    status = await service.federated_budget_status(ADMIN, org_id="acme", project_id="research")
    assert status.granted_away == 1 and status.total_spent == 0.3 and status.active_grants == 1
    async with repository.transaction() as tx:
        assert (await tx.get_budget_grant(first.id)).state is GrantState.REVOKED
        assert len(await tx.list_budget_grants()) == 2


async def test_current_period_rolls_local_spend_but_not_unproven_issued_authority(repository):
    now = utcnow()
    service = await _home(repository, limit=1, spent=1, price=0.6, period="daily", started=now - timedelta(days=2))
    grant = await _issue(service, amount=0.4, now=now)
    await service.record_grant_spend(grant_id=grant.id, reported_spend=0.3)
    workload = await _submit(service)
    plan = await service.schedule_once(ADMIN)
    assert len(plan.placements) == 1, "home admission and issuance must use the same current local period"
    await service.create_lease(ADMIN, _lease(workload, plan.placements[0]), idempotency_key="lease")
    rolled = await service.roll_budget_period(ADMIN, org_id="acme", project_id="research", now=now)
    assert rolled.spent == 0
    with pytest.raises(InputError):
        await _issue(service, amount=0.01, peer="other", key="fresh", now=now + timedelta(days=3))
    status = await service.federated_budget_status(
        ADMIN, org_id="acme", project_id="research", now=now + timedelta(days=3)
    )
    assert status.granted_away == 0.4 and status.reported_peer_spend == 0.3


async def test_outgoing_authority_is_a_hard_boundary_even_with_soft_local_budget(repository):
    service = await _home(repository, soft=True)
    await _issue(service)
    await _submit(service)
    assert not (await service.schedule_once(ADMIN)).placements


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
async def test_nonfinite_reports_cannot_corrupt_retained_authority(repository, value):
    service = await _home(repository)
    grant = await _issue(service)
    await service.revoke_budget_grant(ADMIN, grant_id=grant.id)
    with pytest.raises(InputError, match="finite"):
        await service.record_grant_spend(grant_id=grant.id, reported_spend=value)
    async with repository.transaction() as tx:
        assert (await tx.get_budget_grant(grant.id)).reported_spend == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
async def test_nonfinite_issuance_is_a_clean_public_service_refusal(repository, value):
    service = await _home(repository)
    with pytest.raises(InputError, match="finite"):
        await _issue(service, amount=value)
    async with repository.transaction() as tx:
        assert await tx.list_budget_grants() == []


async def test_outgoing_quota_partition_also_bounds_sparse_local_dimensions(repository):
    service = await _home(repository)
    async with repository.transaction() as tx:
        await tx.put_quota(
            QuotaRecord(
                org_id="acme",
                project_id="research",
                limits=ResourceLimits(cpu=1, tpu=100),
                constrained_resources=("tpu",),
                updated_by="root",
            )
        )
    workload = await _submit(service)
    plan = await service.schedule_once(ADMIN)
    proposal = _lease(workload, plan.placements[0])
    await _issue(service, kind="quota")
    assert not (await service.schedule_once(ADMIN)).placements
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="lease")


@pytest.mark.parametrize("held", ["grant", "reservation", "spent"])
async def test_sub_decimal_context_authority_cannot_disappear(repository, held):
    service = await _home(repository, price=1e-30 if held == "reservation" else 1, spent=1e-30 if held == "spent" else 0)
    if held == "grant":
        await _issue(service, amount=1e-30)
        await _submit(service)
        assert not (await service.schedule_once(ADMIN)).placements, "tiny rights must also bind home admission"
    elif held == "reservation":
        workload = await _submit(service)
        plan = await service.schedule_once(ADMIN)
        await service.create_lease(ADMIN, _lease(workload, plan.placements[0]), idempotency_key="lease")
    with pytest.raises(InputError):
        await _issue(service, amount=1, peer="other", key="full")


@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("state", [GrantState.ACTIVE, GrantState.REVOKED])
async def test_an_inbound_scope_cannot_relabel_its_local_limit_as_home_authority(repository, kind, state):
    from eyvan.server.service.federation import BudgetGrantRecord, QuotaGrantRecord

    service = await _home(repository)
    now = utcnow()
    fields = dict(
        id="inbound",
        org_id="acme",
        project_id="research",
        home_plane_id="upstream",
        grantee_plane_id="home",
        state=state,
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )
    async with repository.transaction() as tx:
        if kind == "budget":
            await tx.put_budget_grant(BudgetGrantRecord(**fields, amount=0.1))
        else:
            await tx.put_quota_grant(QuotaGrantRecord(**fields, limits=ResourceLimits(cpu=0.1)))
    with pytest.raises(InputError):
        await _issue(service, kind=kind)
    async with repository.transaction() as tx:
        retained = await (tx.list_budget_grants() if kind == "budget" else tx.list_quota_grants())
    assert [grant.id for grant in retained] == ["inbound"]


@pytest.mark.parametrize("issued", [0.0, 0.1, 1e-30])
def test_exported_headroom_keeps_float_interface_without_rounding_away_rights(issued):
    from decimal import localcontext

    from eyvan.server.service.federation import BudgetGrantRecord, issuable_amount

    now = utcnow()
    grant = BudgetGrantRecord(
        id="public-helper",
        org_id="acme",
        project_id="research",
        home_plane_id="home",
        grantee_plane_id="peer",
        amount=issued,
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )
    headroom = issuable_amount(1.0, [grant], now=now)
    assert isinstance(headroom, float), "the exported helper must retain its existing float interface"
    with localcontext(prec=2048):
        assert Decimal(str(headroom)) <= Decimal(1) - Decimal(str(issued))
    if issued > 0:
        assert headroom < 1.0
