# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Endpoint readiness belongs to one execution generation, not one workload id."""

from datetime import timedelta

import pytest

from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxState,
    PriorityBand,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.controller._support import _PreemptionVictimOutcome
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import ServicePolicy, probe_from_response


async def _admit_on_fresh_cells(controller, replica, prefix):
    """Use actual group admission, isolating readiness from scheduler rankings."""
    cells = [
        CellRecord(
            id=f"{prefix}-{suffix}",
            provider="local",
            region="local",
            state=CellState.READY,
            capacity=ResourceVector(cpu=8, memory_gib=8),
            allocatable=ResourceVector(cpu=8, memory_gib=8),
            capacity_class=CapacityClass.SPOT,
        )
        for suffix in "ab"
    ]
    async with controller.repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)
    leases = tuple(
        LeaseRecord(
            id=f"lease-{cell.id}",
            workload_id=replica.id,
            cell_id=cell.id,
            org_id=replica.org_id,
            project_id=replica.project_id,
            generation=replica.generation,
            state=LeaseState.ACTIVE,
            resources=replica.spec.gang,
            fencing_token=f"fence-{cell.id}",
            group_id=prefix,
            group_index=index,
            expires_at=utcnow() + timedelta(minutes=5),
        )
        for index, cell in enumerate(cells)
    )
    await controller.service.create_lease_group(controller._principal, leases, idempotency_key=prefix)
    await controller.service.transition_workload(controller._principal, replica.id, WorkloadState.DISPATCHING)
    return await controller.service.transition_workload(controller._principal, replica.id, WorkloadState.RUNNING)


async def _resumed_service(mode):
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=None)
    controller = EyvanController(repository, service)
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    parent, _ = await service.submit_service(
        controller._principal,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="api",
            execution=CommandExecution(command="python serve.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=8, memory_gib=8)),
            capacity_class=CapacityClass.SPOT,
            priority_band=PriorityBand.BATCH,
            cell_group_count=2,
            can_submit_children=True,
        ),
        policy=policy,
        idempotency_key="service",
    )
    assert await controller.reconcile_services_once() == 1
    async with repository.transaction() as tx:
        (replica,) = await tx.list_child_workloads(parent.id)
    running = await _admit_on_fresh_cells(controller, replica, "previous")
    registered = await service.register_service_endpoint(workload=running, endpoint="http://previous-host:8080")
    assert [entry["endpoint"] for entry in await service.service_endpoints(controller._principal, parent.id)] == [
        "http://previous-host:8080"
    ]
    if mode == "recycle":
        resumed, _, _ = await service.settle_preempted_workload(controller._principal, replica.id, recycle=True)
    else:
        target = WorkloadRecord(
            id="urgent",
            org_id="acme",
            project_id="research",
            created_by="alice",
            spec=replica.spec.model_copy(update={"cell_group_count": 1, "priority_band": PriorityBand.PRODUCTION}),
        )
        async with repository.transaction() as tx:
            await tx.put_workload(target)
            plan = (
                CoreSchedulingService()
                .schedule([registered, target], await tx.list_cells(), leases=await tx.list_leases())
                .preemptions[0]
            )
        service.scheduler = CoreSchedulingService()
        await controller._commit_preemption(plan)
        async with repository.transaction() as tx:
            command = next(record for record in await tx.list_outbox() if record.topic == "workload.preempt")
            command = command.model_copy(
                update={
                    "state": OutboxState.PROCESSING,
                    "claimed_by": controller.worker_id,
                    "claimed_until": utcnow() + timedelta(minutes=5),
                }
            )
            await tx.put_outbox(command)
        prepared = await controller._prepare_preemption_plan(command)
        assert prepared is not None
        stopping = await controller._advance_preemption_to_stopping(prepared, {}, {})
        for victim in controller._preemption_victims(stopping):
            stopping = await controller._record_preemption_victim_stopped(
                stopping.id, victim.lease_id, _PreemptionVictimOutcome.PREEMPTED
            )
        await controller._finalize_preemption_plan(stopping, {})
        resumed = await service.get_workload(controller._principal, replica.id)
        service.scheduler = None
    assert resumed.generation == 2 and resumed.state is WorkloadState.PENDING
    assert await service.service_endpoints(controller._principal, parent.id) == []
    replacement = await _admit_on_fresh_cells(controller, resumed, "replacement")
    assert replacement.generation == 2
    return controller, parent, registered, replacement


@pytest.mark.parametrize("mode", ["recycle", "priority-preemption"])
@pytest.mark.parametrize("consumer", ["discovery", "health-probe"])
async def test_requeued_service_withholds_previous_endpoint_until_registration(mode, consumer):
    controller, parent, _previous, replacement = await _resumed_service(mode)
    service = controller.service
    if consumer == "discovery":
        assert await service.service_endpoints(controller._principal, parent.id) == []
    else:

        async def probe(*, endpoint, path, timeout_s):
            return probe_from_response(200, "")

        controller.service_health_probe = probe
        recorded = await controller.probe_service_health_once()
        async with controller.repository.transaction() as tx:
            assert await tx.list_service_replica_health(parent.id) == []
        assert recorded == 0
    await service.register_service_endpoint(workload=replacement, endpoint="http://replacement-host:8080")
    assert [entry["endpoint"] for entry in await service.service_endpoints(controller._principal, parent.id)] == [
        "http://replacement-host:8080"
    ]


@pytest.mark.parametrize("mode", ["recycle", "priority-preemption"])
async def test_previous_generation_cannot_register_after_service_requeue(mode):
    controller, parent, previous, replacement = await _resumed_service(mode)
    with pytest.raises(ConflictError, match="generation 2"):
        await controller.service.register_service_endpoint(workload=previous, endpoint="http://stale-host:8080")
    await controller.service.register_service_endpoint(workload=replacement, endpoint="http://replacement-host:8080")
    endpoints = await controller.service.service_endpoints(controller._principal, parent.id)
    assert [entry["endpoint"] for entry in endpoints] == ["http://replacement-host:8080"]
