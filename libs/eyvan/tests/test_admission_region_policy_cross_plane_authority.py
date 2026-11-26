# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Admission regressions for region policy and cross-plane authority."""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    POOL_ALTERNATE_TEMPLATES_LABEL,
    POOL_ALTERNATES_LABEL,
    POOL_LABEL,
    POOL_TEMPLATE_LABEL,
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    PolicyRecord,
    Principal,
    QuotaRecord,
    ResourceGang,
    ResourceHours,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError, InputError
from eyvan.server.service.federation import BudgetGrantRecord, GrantState, PlanePeerRecord, QuotaGrantRecord
from eyvan.server.service.scheduling import CoreSchedulingService


def _principal(role: Role) -> Principal:
    grant = (
        RoleGrant(role=role) if role is Role.FLEET_ADMIN else RoleGrant(role=role, org_id="acme", project_id="research")
    )
    return Principal(subject="operator", grants=(grant,))


ADMIN = _principal(Role.FLEET_ADMIN)
OPERATOR = _principal(Role.PROJECT_OPERATOR)


async def _service(*, plane_id: str = "") -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id=plane_id)
    async with repository.transaction() as tx:
        await tx.put_cell(
            CellRecord(
                id="cell",
                provider="local",
                region="forbidden",
                state=CellState.READY,
                capacity=ResourceVector(cpu=16, memory_gib=64, gpu=8, custom={"widgets": 8}),
                allocatable=ResourceVector(cpu=16, memory_gib=64, gpu=8, custom={"widgets": 8}),
                unit_prices={"cpu": 1.0},
            )
        )
    return service


async def _submit(service: ControlPlaneService, *, key: str = "work", **spec_fields):
    workload, _ = await service.submit_workload(
        OPERATOR,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name=key,
            execution=CommandExecution(command="true"),
            gang=spec_fields.pop("gang", ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=0))),
            estimated_duration_seconds=3600,
            **spec_fields,
        ),
        idempotency_key=key,
    )
    return workload


def _lease(workload, placement) -> LeaseRecord:
    core = placement.lease
    quantities = {name: float(quantity) for name, quantity in core.resource_hours_reserved.items()}
    builtins = {"cpu", "memory_gib", "gpu", "tpu"}
    return LeaseRecord(
        id=core.lease_id,
        workload_id=workload.id,
        cell_id=core.cell_id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        resources=workload.spec.gang,
        hourly_cost=float(core.hourly_cost),
        reserved_cost=float(core.reserved_cost),
        resource_hours_reserved=ResourceHours(
            **{name: quantity for name, quantity in quantities.items() if name in builtins},
            custom={name: quantity for name, quantity in quantities.items() if name not in builtins},
        ),
        generation=workload.generation,
        fencing_token="owned-test-fence",
        expires_at=utcnow() + timedelta(hours=1),
    )


@pytest.mark.parametrize("regions", [("forbidden",), ("approved", "forbidden")])
async def test_region_policy_cannot_be_replaced_by_a_workload_list(regions) -> None:
    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_policy(
            PolicyRecord(org_id="acme", project_id="research", allowed_regions=("approved",), updated_by="root")
        )
    await _submit(service, allowed_regions=regions)
    result = await service.schedule_once(ADMIN)
    assert not result.placements, "workload region choices must not replace project authority"


async def test_region_policy_change_rejects_a_previously_admissible_lease() -> None:
    service = await _service()
    workload = await _submit(service, allowed_regions=("forbidden",))
    result = await service.schedule_once(ADMIN)
    assert len(result.placements) == 1
    proposal = _lease(workload, result.placements[0])
    async with service.repository.transaction() as tx:
        await tx.put_policy(
            PolicyRecord(org_id="acme", project_id="research", allowed_regions=("approved",), updated_by="root")
        )
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="commit")
    async with service.repository.transaction() as tx:
        assert await tx.get_lease(proposal.id) is None


async def test_region_policy_keeps_an_approved_workload_choice_admissible() -> None:
    service = await _service()
    async with service.repository.transaction() as tx:
        cell = await tx.get_cell("cell")
        await tx.put_cell(cell.model_copy(update={"region": "approved"}))
        await tx.put_policy(
            PolicyRecord(org_id="acme", project_id="research", allowed_regions=("approved",), updated_by="root")
        )
    workload = await _submit(service, allowed_regions=("approved", "forbidden"))
    result = await service.schedule_once(ADMIN)
    assert len(result.placements) == 1
    stored, replay = await service.create_lease(ADMIN, _lease(workload, result.placements[0]), idempotency_key="commit")
    assert stored.cell_id == "cell" and not replay


@pytest.mark.parametrize("region,admitted", [("approved", True), ("forbidden", False)])
async def test_cross_pool_region_choices_remain_bounded_by_policy(region, admitted) -> None:
    service = await _service()
    workload = await _submit(service, allowed_regions=("approved", "forbidden"))
    workload = workload.model_copy(
        update={
            "spec": workload.spec.model_copy(
                update={
                    "required_labels": {
                        POOL_LABEL: "primary",
                        POOL_TEMPLATE_LABEL: "primary-template",
                        POOL_ALTERNATES_LABEL: "alternate",
                        POOL_ALTERNATE_TEMPLATES_LABEL: "alternate-template",
                    }
                }
            )
        }
    )
    async with service.repository.transaction() as tx:
        await tx.put_workload(workload)
        cell = await tx.get_cell("cell")
        await tx.put_cell(
            cell.model_copy(
                update={
                    "region": region,
                    "labels": {
                        POOL_LABEL: "alternate",
                        POOL_TEMPLATE_LABEL: "alternate-template",
                    },
                }
            )
        )
        await tx.put_policy(
            PolicyRecord(org_id="acme", project_id="research", allowed_regions=("approved",), updated_by="root")
        )
    result = await service.schedule_once(ADMIN)
    assert bool(result.placements) is admitted
    if admitted:
        stored, _ = await service.create_lease(ADMIN, _lease(workload, result.placements[0]), idempotency_key="commit")
        assert stored.cell_id == "cell"


async def _inbound(service, kind, *, allowance=1, local=False):
    now = utcnow()
    fields = dict(
        id="inbound",
        org_id="acme",
        project_id="research",
        home_plane_id="home",
        grantee_plane_id="peer",
        issued_at=now - timedelta(hours=1),
        expires_at=now + timedelta(hours=2),
    )
    async with service.repository.transaction() as tx:
        if kind == "budget":
            grant = BudgetGrantRecord(**fields, amount=allowance)
            await tx.put_budget_grant(grant)
            if local:
                await tx.put_budget(BudgetRecord(org_id="acme", project_id="research", limit=100, updated_by="root"))
        else:
            grant = QuotaGrantRecord(**fields, limits=ResourceLimits(cpu=allowance))
            await tx.put_quota_grant(grant)
            if local:
                await tx.put_quota(
                    QuotaRecord(org_id="acme", project_id="research", limits=ResourceLimits(cpu=100), updated_by="root")
                )
    return grant


async def _lapse(service, grant, kind, state):
    if state is GrantState.REVOKED:
        if kind == "budget":
            await service.revoke_budget_grant(ADMIN, grant_id=grant.id)
        else:
            await service.revoke_quota_grant(ADMIN, grant_id=grant.id)
    else:
        expired = grant.model_copy(update={"expires_at": utcnow() - timedelta(seconds=1)})
        async with service.repository.transaction() as tx:
            if kind == "budget":
                await tx.put_budget_grant(expired)
            else:
                await tx.put_quota_grant(expired)
        assert await service.expire_budget_grants_once() == 1


@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("state", [GrantState.EXPIRED, GrantState.REVOKED])
@pytest.mark.parametrize("local", [False, True])
async def test_inactive_inbound_authority_keeps_admission_closed(kind, state, local) -> None:
    service = await _service(plane_id="peer")
    grant = await _inbound(service, kind, local=local)
    await _submit(service, gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=2, memory_gib=0)))
    assert not (await service.schedule_once(ADMIN)).placements, "active authority refusal control"
    await _lapse(service, grant, kind, state)
    assert not (await service.schedule_once(ADMIN)).placements, "inactive authority must not disappear from admission"


@pytest.mark.parametrize("kind", ["budget", "quota"])
@pytest.mark.parametrize("state", [GrantState.EXPIRED, GrantState.REVOKED])
async def test_inbound_authority_lapsing_after_plan_rejects_commit(kind, state) -> None:
    service = await _service(plane_id="peer")
    grant = await _inbound(service, kind, allowance=2, local=True)
    workload = await _submit(service)
    result = await service.schedule_once(ADMIN)
    assert len(result.placements) == 1
    proposal = _lease(workload, result.placements[0])
    await _lapse(service, grant, kind, state)
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="commit")
    async with service.repository.transaction() as tx:
        assert await tx.get_lease(proposal.id) is None


@pytest.mark.parametrize("resource", ["cpu", "gpu", "widgets"])
@pytest.mark.parametrize("local", [False, True])
async def test_quota_authority_constrains_grant_only_and_omitted_resources(resource, local) -> None:
    service = await _service(plane_id="peer")
    demand = (
        ResourceVector(cpu=8, memory_gib=0)
        if resource == "cpu"
        else ResourceVector(
            cpu=1,
            memory_gib=0,
            gpu=8 if resource == "gpu" else 0,
            custom={"widgets": 8} if resource == "widgets" else {},
        )
    )
    workload = await _submit(service, gang=ResourceGang(replicas=1, per_replica=demand))
    before = await service.schedule_once(ADMIN)
    assert len(before.placements) == 1, "never-federated sparse quota control"
    proposal = _lease(workload, before.placements[0])
    await _inbound(service, "quota", allowance=1)
    if local:
        async with service.repository.transaction() as tx:
            limits = ResourceLimits(gpu=100) if resource == "cpu" else ResourceLimits(cpu=100)
            await tx.put_quota(QuotaRecord(org_id="acme", project_id="research", limits=limits, updated_by="root"))
    assert not (await service.schedule_once(ADMIN)).placements, "grant limits apply independently of local sparsity"
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="commit")


@pytest.mark.parametrize("grant_gpu,demand_gpu,admitted", [(0, 1, False), (2, 4, False), (2, 2, True)])
async def test_sparse_local_quota_preserves_finite_gpu_grant(grant_gpu, demand_gpu, admitted) -> None:
    service = await _service(plane_id="peer")
    grant = await _inbound(service, "quota", allowance=8, local=True)
    async with service.repository.transaction() as tx:
        await tx.put_quota_grant(grant.model_copy(update={"limits": ResourceLimits(cpu=8, gpu=grant_gpu)}))
    workload = await _submit(
        service, gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=0, gpu=demand_gpu))
    )
    result = await service.schedule_once(ADMIN)
    assert bool(result.placements) is admitted
    if admitted:
        stored, replay = await service.create_lease(
            ADMIN, _lease(workload, result.placements[0]), idempotency_key="commit"
        )
        assert stored.resources.per_replica.gpu == demand_gpu and not replay


@pytest.mark.parametrize("resource", ["gpu", "widgets"])
async def test_inactive_sparse_quota_also_closes_unmentioned_dimensions(resource) -> None:
    service = await _service(plane_id="peer")
    grant = await _inbound(service, "quota", allowance=8, local=True)
    await _lapse(service, grant, "quota", GrantState.REVOKED)
    demand = ResourceVector(
        cpu=0,
        memory_gib=0,
        gpu=1 if resource == "gpu" else 0,
        custom={"widgets": 1} if resource == "widgets" else {},
    )
    await _submit(service, gang=ResourceGang(replicas=1, per_replica=demand))
    assert not (await service.schedule_once(ADMIN)).placements


async def test_grant_only_settled_spend_survives_service_restart(monkeypatch) -> None:
    # Build legacy settled records before enabling federated admission. A repair
    # may safely refuse unaccounted grant-only budgets rather than reconstruct
    # their spend from an incomplete, windowless snapshot.
    service = await _service()
    grant = await _inbound(service, "budget", allowance=1)
    first = await _submit(service, key="first")
    plan = await service.schedule_once(ADMIN)
    assert len(plan.placements) == 1
    stored, _ = await service.create_lease(ADMIN, _lease(first, plan.placements[0]), idempotency_key="first-lease")
    await service.transition_lease(ADMIN, stored.id, LeaseState.ACTIVE)
    await service.transition_workload(ADMIN, first.id, WorkloadState.DISPATCHING)
    await service.transition_workload(ADMIN, first.id, WorkloadState.RUNNING)
    now = utcnow()
    async with service.repository.transaction() as tx:
        active = await tx.get_lease(stored.id)
        await tx.put_lease(
            active.model_copy(
                update={
                    "created_at": now - timedelta(hours=1),
                    "run_started_at": now - timedelta(hours=1),
                }
            )
        )
    with monkeypatch.context() as clock:
        clock.setattr("eyvan.server.service._control_plane.lease_settlement.utcnow", lambda: now)
        _, settled, _ = await service.settle_terminal_workload(ADMIN, first.id, WorkloadState.SUCCEEDED)
        await service.settle_terminal_workload(ADMIN, first.id, WorkloadState.SUCCEEDED)
    assert settled.settled_cost == 1 and grant.active_at(now)
    async with service.repository.transaction() as tx:
        leases = await tx.list_leases()
        assert sum(lease.settled_cost or 0 for lease in leases) == 1, "settlement replay must not double-charge"
    restarted = ControlPlaneService(service.repository, scheduler=CoreSchedulingService(), plane_id="peer")
    await _submit(restarted, key="second")
    result = await restarted.schedule_once(ADMIN)
    assert not result.placements, "settled grant consumption cannot disappear when the reservation is released"


@pytest.mark.parametrize("local", [False, True])
async def test_budget_accounting_is_required_before_using_inbound_money(local) -> None:
    service = await _service(plane_id="peer")
    await _inbound(service, "budget", allowance=1, local=local)
    workload = await _submit(service)
    result = await service.schedule_once(ADMIN)
    assert bool(result.placements) is local, "a grant supplies authority, not a durable consumption ledger"
    if local:
        stored, _ = await service.create_lease(ADMIN, _lease(workload, result.placements[0]), idempotency_key="commit")
        assert stored.reserved_cost == 1
    else:
        restarted = ControlPlaneService(service.repository, scheduler=CoreSchedulingService(), plane_id="peer")
        assert not (await restarted.schedule_once(ADMIN)).placements


async def test_budget_accounting_gate_rejects_an_unaccounted_stale_lease() -> None:
    service = await _service(plane_id="peer")
    workload = await _submit(service)
    plan = await service.schedule_once(ADMIN)
    assert len(plan.placements) == 1, "never-federated scope preserves default admission"
    proposal = _lease(workload, plan.placements[0])
    await _inbound(service, "budget", allowance=1)
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="commit")
    async with service.repository.transaction() as tx:
        assert await tx.get_lease(proposal.id) is None


async def _home(kind):
    service = await _service(plane_id="home")
    async with service.repository.transaction() as tx:
        await tx.put_plane_peer(PlanePeerRecord(plane_id="peer", endpoint="https://peer.example.test"))
        if kind == "budget":
            await tx.put_budget(BudgetRecord(org_id="acme", project_id="research", limit=1, updated_by="root"))
        else:
            await tx.put_quota(
                QuotaRecord(org_id="acme", project_id="research", limits=ResourceLimits(cpu=1), updated_by="root")
            )
    return service


async def _delegate(service, kind):
    fields = dict(
        org_id="acme",
        project_id="research",
        grantee_plane_id="peer",
        expires_at=utcnow() + timedelta(hours=1),
        idempotency_key="delegate",
    )
    if kind == "budget":
        return await service.issue_budget_grant(ADMIN, amount=1, **fields)
    return await service.issue_quota_grant(ADMIN, limits=ResourceLimits(cpu=1), **fields)


@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_delegated_home_authority_cannot_also_admit_a_home_lease(kind) -> None:
    service = await _home(kind)
    workload = await _submit(service)
    before = await service.schedule_once(ADMIN)
    assert len(before.placements) == 1, "no outgoing authority control"
    proposal = _lease(workload, before.placements[0])
    await _delegate(service, kind)
    assert not (await service.schedule_once(ADMIN)).placements, "home must reserve the slice it granted away"
    with pytest.raises(ConflictError):
        await service.create_lease(ADMIN, proposal, idempotency_key="commit")


@pytest.mark.parametrize("kind", ["budget", "quota"])
async def test_home_grant_issuance_cannot_overcommit_existing_reservations(kind) -> None:
    service = await _home(kind)
    workload = await _submit(service)
    before = await service.schedule_once(ADMIN)
    assert len(before.placements) == 1
    await service.create_lease(ADMIN, _lease(workload, before.placements[0]), idempotency_key="commit")
    with pytest.raises(InputError):
        await _delegate(service, kind)
