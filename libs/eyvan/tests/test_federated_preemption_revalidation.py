# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Federation bounds are revalidated before victim RPCs, not just at plan commit."""

from datetime import timedelta

import pytest
from test_priority_preemption_properties import PerCellAgent
from test_priority_preemption_properties import repository as _repository_fixture

from eyvan.records import (
    BudgetRecord,
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxState,
    PriorityBand,
    QuotaRecord,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.federation import BudgetGrantRecord, QuotaGrantRecord
from eyvan.server.service.scheduling import CoreSchedulingService

repository = _repository_fixture


@pytest.mark.parametrize("changed_bound", ("budget", "quota", "unchanged"))
async def test_dispatch_revalidates_post_victim_federation_bounds(repository, changed_bound):
    scheduler = CoreSchedulingService()
    service = ControlPlaneService(repository, scheduler=scheduler, plane_id="plane-b")
    agent = PerCellAgent("RUNNING")
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)
    resources = ResourceGang(replicas=1, per_replica=ResourceVector(cpu=8, memory_gib=8))
    spec = WorkloadDefinition(
        name="victim",
        execution=CommandExecution(command="true"),
        gang=resources,
        capacity_class=CapacityClass.SPOT,
        priority_band=PriorityBand.BATCH,
        checkpointable=True,
        checkpoint_hook="python checkpoint.py",
        checkpoint_output_uri="gs://checkpoints/victim",
        estimated_duration_seconds=3600,
    )
    victim = WorkloadRecord(
        id="victim",
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=spec,
        state=WorkloadState.RUNNING,
        lease_id="victim-lease",
        remote_id="remote-victim",
    )
    target = WorkloadRecord(
        id="target",
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=spec.model_copy(update={"name": "target", "priority_band": PriorityBand.PRODUCTION}),
    )
    cell = CellRecord(
        id="cell",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=8),
        allocatable=ResourceVector(cpu=8, memory_gib=8),
        capacity_class=CapacityClass.SPOT,
        unit_prices={"cpu": 1.0},
    )
    lease = LeaseRecord(
        id="victim-lease",
        workload_id=victim.id,
        cell_id=cell.id,
        org_id="acme",
        project_id="research",
        state=LeaseState.ACTIVE,
        resources=resources,
        fencing_token="victim-fence",
        reserved_cost=8.0,
        expires_at=utcnow() + timedelta(minutes=5),
    )
    budget = BudgetRecord(org_id="acme", project_id="research", limit=1000, updated_by="root")
    now = utcnow()
    grant = BudgetGrantRecord(
        id="budget-grant",
        org_id="acme",
        project_id="research",
        home_plane_id="plane-a",
        grantee_plane_id="plane-b",
        amount=20,
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )
    quota = QuotaRecord(
        org_id="acme",
        project_id="research",
        limits=ResourceLimits(cpu=32, memory_gib=32),
        updated_by="root",
    )
    quota_grant = QuotaGrantRecord(
        id="quota-grant",
        org_id="acme",
        project_id="research",
        home_plane_id="plane-a",
        grantee_plane_id="plane-b",
        limits=ResourceLimits(cpu=32, memory_gib=32),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )
    async with repository.transaction() as tx:
        await tx.put_workload(victim)
        await tx.put_workload(target)
        await tx.put_cell(cell)
        await tx.put_lease(lease)
        await tx.put_budget(budget)
        await tx.put_budget_grant(grant)
        await tx.put_quota(quota)
        await tx.put_quota_grant(quota_grant)

    result = scheduler.schedule(
        [victim, target],
        [cell],
        leases=[lease],
        budgets=[budget],
        quotas=[quota],
        grants=[grant],
        quota_grants=[quota_grant],
        plane_id=service.plane_id,
    )
    assert len(result.preemptions) == 1
    assert result.preemptions[0].replacement_lease.reserved_cost == 8.0
    await controller._commit_preemption(result.preemptions[0])
    async with repository.transaction() as tx:
        commands = [command for command in await tx.list_outbox() if command.topic == "workload.preempt"]
        assert len(commands) == 1
        command = commands[0]
        committed_victim = await tx.get_workload(victim.id)
        assert committed_victim.state is WorkloadState.CHECKPOINTING
        assert committed_victim.preemption_id == command.id
        if changed_bound == "budget":
            # 15 spent + 8 target is over the grant even after removing the
            # victim's reservation; merely counting that released reservation
            # would not prove the target is inadmissible.
            await tx.put_budget(budget.model_copy(update={"spent": 15}))
        elif changed_bound == "quota":
            await tx.put_quota_grant(quota_grant.model_copy(update={"limits": ResourceLimits(cpu=4, memory_gib=32)}))

    assert await controller.dispatch_outbox_once() == 1
    async with repository.transaction() as tx:
        restored_victim = await tx.get_workload(victim.id)
        restored_target = await tx.get_workload(target.id)
        restored_lease = await tx.get_lease(lease.id)
        delivered = await tx.get_outbox(command.id)
        target_leases = [candidate for candidate in await tx.list_leases() if candidate.workload_id == target.id]
        events = await tx.list_events()
        restored_budget = await tx.get_budget("acme", "research")
    assert delivered.state is OutboxState.SUCCEEDED
    assert restored_target.state is WorkloadState.PENDING
    assert restored_target.preemption_id is None
    assert target_leases == []
    if changed_bound == "unchanged":
        assert len(agent.checkpoints) == len(agent.stops) == 1
        assert restored_victim.state is WorkloadState.PENDING
        assert restored_lease.state is LeaseState.RELEASED
        assert delivered.payload["execution_completed"] is True
    else:
        assert agent.observations == agent.checkpoints == agent.stops == agent.submissions == []
        assert restored_victim.state is WorkloadState.RUNNING
        assert restored_victim.preemption_id is None
        assert restored_victim.generation == restored_victim.attempt == 1
        assert restored_lease.state is LeaseState.ACTIVE
        assert restored_lease.revocation_reason is None and restored_lease.settled_at is None
        assert not delivered.payload.get("execution_started")
        assert not any(event.action == "lease.settled" for event in events)
        assert restored_budget.spent == (15 if changed_bound == "budget" else 0)
