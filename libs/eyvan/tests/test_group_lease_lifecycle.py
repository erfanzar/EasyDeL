# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A lease group remains a group through accounting, preemption and recovery."""

from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxState,
    PlacementMode,
    PriorityBand,
    ResourceGang,
    ResourceVector,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.controller._support import _PreemptionVictimOutcome
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def repository(request, tmp_path):
    engine = None
    admin = None
    schema = "probe_" + uuid4().hex
    if request.param == "memory":
        repo = InMemoryRepository()
    else:
        if request.param == "postgres":
            url = request.getfixturevalue("postgres_url")
            admin = create_async_engine(url)
            async with admin.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
        else:
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
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


def _workload(identifier, spec, state=WorkloadState.PENDING, **kwargs):
    return WorkloadRecord(
        id=identifier, org_id="acme", project_id="research", created_by="alice", spec=spec, state=state, **kwargs
    )


async def _preemption(repository, *, correlation_enabled=False, checkpointable=False, remote_id=None):
    scheduler = CoreSchedulingService()
    controller = EyvanController(
        repository,
        ControlPlaneService(
            repository,
            scheduler=scheduler,
            correlation_enabled=correlation_enabled,
        ),
    )
    spec = WorkloadDefinition(
        name="gang",
        execution=CommandExecution(command="true"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=8, memory_gib=8)),
        capacity_class=CapacityClass.SPOT,
        priority_band=PriorityBand.BATCH,
        cell_group_count=2,
        checkpointable=checkpointable,
        checkpoint_hook="python checkpoint.py" if checkpointable else None,
        checkpoint_output_uri="gs://checkpoints/gang" if checkpointable else None,
    )
    victim = _workload("victim", spec, WorkloadState.RUNNING, lease_id="lease-a", lease_ids=("lease-a", "lease-b"))
    if remote_id:
        victim = victim.model_copy(update={"remote_id": remote_id})
    target = _workload(
        "priority", spec.model_copy(update={"cell_group_count": 1, "priority_band": PriorityBand.PRODUCTION})
    )
    cells = [
        CellRecord(
            id="cell-" + s,
            provider="local",
            region="local",
            state=CellState.READY,
            capacity=ResourceVector(cpu=8, memory_gib=8),
            allocatable=ResourceVector(cpu=8, memory_gib=8),
            capacity_class=CapacityClass.SPOT,
        )
        for s in ("a", "b")
    ]
    leases = [
        LeaseRecord(
            id="lease-" + s,
            workload_id=victim.id,
            cell_id="cell-" + s,
            org_id="acme",
            project_id="research",
            state=LeaseState.ACTIVE,
            resources=spec.gang,
            fencing_token="fence-" + s,
            group_id="gang",
            group_index=i,
            expires_at=utcnow() + timedelta(minutes=5),
        )
        for i, s in enumerate(("a", "b"))
    ]
    async with repository.transaction() as tx:
        for workload in (victim, target):
            await tx.put_workload(workload)
        for cell in cells:
            await tx.put_cell(cell)
        for lease in leases:
            await tx.put_lease(lease)
    plan = scheduler.schedule([victim, target], cells, leases=leases).preemptions[0]
    assert len(plan.victims) == 2
    await controller._commit_preemption(plan)
    async with repository.transaction() as tx:
        command = next(c for c in await tx.list_outbox() if c.topic == "workload.preempt")
        command = command.model_copy(
            update={
                "state": OutboxState.PROCESSING,
                "claimed_by": controller.worker_id,
                "claimed_until": utcnow() + timedelta(minutes=5),
            }
        )
        await tx.put_outbox(command)
    return controller, command


async def test_an_invalidated_gang_preemption_restores_every_member(repository):
    controller, command = await _preemption(repository)
    async with repository.transaction() as tx:
        target = await tx.get_workload("priority")
        await tx.put_workload(target.model_copy(update={"state": WorkloadState.CANCELLED}))
    assert await controller._prepare_preemption_plan(command) is None
    async with repository.transaction() as tx:
        victim = await tx.get_workload("victim")
        assert victim.state is WorkloadState.RUNNING
        assert victim.preemption_id is None
        leases = await tx.list_leases()
        assert all(lease.state is LeaseState.ACTIVE and lease.revocation_reason is None for lease in leases)


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (("preempted", "preempted"), WorkloadState.PENDING),
        (("succeeded", "succeeded"), WorkloadState.SUCCEEDED),
        (("succeeded", "preempted"), WorkloadState.PENDING),
        (("succeeded", "failed"), WorkloadState.FAILED),
    ],
)
async def test_a_gang_settles_every_member_but_transitions_the_workload_once(repository, outcomes, expected):
    controller, command = await _preemption(repository)
    prepared = await controller._prepare_preemption_plan(command)
    assert prepared is not None
    victims = controller._preemption_victims(prepared)
    for victim in victims:
        assert await controller._load_preemption_victim(prepared, victim) is not None
    stopping = await controller._advance_preemption_to_stopping(prepared, {}, {})
    for victim, outcome in zip(victims, outcomes, strict=True):
        stopping = await controller._record_preemption_victim_stopped(
            stopping.id, victim.lease_id, _PreemptionVictimOutcome(outcome)
        )
    await controller._finalize_preemption_plan(stopping, {})
    await controller._finalize_preemption_plan(stopping, {})
    async with repository.transaction() as tx:
        workload = await tx.get_workload("victim")
        assert workload.state is expected
        assert workload.generation == (2 if expected is WorkloadState.PENDING else 1)
        assert workload.attempt == (2 if expected is WorkloadState.PENDING else 1)
        assert workload.preemption_id is None
        assert all(
            lease.state is LeaseState.RELEASED and lease.settled_at is not None for lease in await tx.list_leases()
        )
        if expected is WorkloadState.PENDING:
            assert workload.lease_id is None and workload.lease_ids == ()
        events = await tx.list_events()
        transitions = [
            e
            for e in events
            if e.action
            in {"workload.preempted", "workload.succeeded_during_preemption", "workload.failed_during_preemption"}
        ]
        assert len(transitions) == 1


async def test_elastic_resource_usage_is_identical_before_and_after_committing_leases(repository):
    scheduler = CoreSchedulingService()
    controller = EyvanController(repository, ControlPlaneService(repository, scheduler=scheduler))
    spec = WorkloadDefinition(
        name="elastic",
        provider="gcp",
        required_executor="eyvan-native",
        execution=CommandExecution(command="true"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(tpu=8)),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
        capacity_class=CapacityClass.SPOT,
    )
    workload = _workload("elastic", spec)
    cells = [
        CellRecord(
            id="cell-" + s,
            provider="gcp",
            provider_project="test-project",
            zone="us-central1-a",
            capacity_backend="gcp-tpu",
            accelerator_type="v6e-8",
            provider_options={"node_count": 2, "node_prefix": "probe-" + s},
            executor="eyvan-native",
            region="us-central1",
            state=CellState.READY,
            capacity=ResourceVector(cpu=16, memory_gib=16, tpu=16),
            allocatable=ResourceVector(cpu=16, memory_gib=16, tpu=16),
            capacity_class=CapacityClass.SPOT,
            worker_count=2,
        )
        for s in ("a", "b")
    ]
    async with repository.transaction() as tx:
        await tx.put_workload(workload)
        for cell in cells:
            await tx.put_cell(cell)
    placement = scheduler.schedule([workload], cells).placements[0]
    await controller._commit_placement(placement)
    # The real adapter passes its reconstructed leases to the pure scheduler.
    from eyvan.domain import Scheduler

    class RecordingScheduler(Scheduler):
        def schedule(self, workloads, cells, *, leases=(), **kwargs):
            leases = tuple(leases)
            self.resources = {lease.cell_id: lease.resources for lease in leases}
            return super().schedule(workloads, cells, leases=leases, **kwargs)

    core = RecordingScheduler()
    async with repository.transaction() as tx:
        CoreSchedulingService(core).schedule(
            await tx.list_workloads(), await tx.list_cells(), leases=await tx.list_leases()
        )
    assert core.resources == {lease.cell_id: lease.resources for lease in placement.all_leases}


async def test_an_unchanged_cell_can_be_updated_while_holding_an_elastic_member(repository):
    await test_elastic_resource_usage_is_identical_before_and_after_committing_leases(repository)
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    async with repository.transaction() as tx:
        cell = await tx.get_cell("cell-a")
    updated, replayed = await service.upsert_cell(controller._principal, cell, idempotency_key="unchanged-elastic")
    assert not replayed
    assert updated.allocatable == cell.allocatable

    from eyvan.server.service.errors import ConflictError

    smaller = cell.model_copy(update={"allocatable": ResourceVector(cpu=16, memory_gib=16, tpu=8)})
    with pytest.raises(ConflictError, match="cannot contain active lease"):
        await service.upsert_cell(controller._principal, smaller, idempotency_key="too-small-elastic")
