# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Replacement must still be warranted by durable evidence when it commits."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    ServiceProbeOutcome,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import ServicePolicy, plan_replica_lifecycle, probe_from_response, service_metadata


async def _running_service():
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings())
    operator = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=3)
    parent, _ = await service.submit_service(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="api", execution=CommandExecution(command="python serve.py")),
        policy=policy,
        idempotency_key="service",
    )
    await controller.reconcile_services_once()
    async with repository.transaction() as tx:
        (replica,) = await tx.list_child_workloads(parent.id)
        await tx.put_cell(
            CellRecord(
                id="cell",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=4),
                allocatable=ResourceVector(cpu=4),
            )
        )
        await tx.put_lease(
            LeaseRecord(
                id="lease",
                workload_id=replica.id,
                cell_id="cell",
                cell_generation=1,
                org_id="acme",
                project_id="research",
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1)),
                state=LeaseState.ACTIVE,
                fencing_token="cell:1",
                expires_at=utcnow() + timedelta(hours=1),
            )
        )
        replica = replica.model_copy(
            update={
                "state": WorkloadState.RUNNING,
                "service_endpoint": "http://replica.invalid:8080",
                "lease_id": "lease",
                "lease_ids": ("lease",),
            }
        )
        await tx.put_workload(replica)

    async def failure(**_kwargs: object):
        return probe_from_response(503, "")

    controller.service_health_probe = failure
    for _ in range(3):
        assert await controller.probe_service_health_once() == 1
    return repository, controller, service, parent, replica


@pytest.mark.parametrize(
    "recovery",
    ["success", "threshold", "no-health-path", "revision", "lease", "cell", "generation", "preemption"],
)
async def test_replacement_revalidates_evidence_between_plan_and_commit(recovery: str, monkeypatch) -> None:
    repository, controller, service, parent, replica = await _running_service()
    if recovery in {"lease", "cell"}:
        # These cases start with no failed-health reason, isolating placement recovery.
        async def success(**_kwargs: object):
            return probe_from_response(200, "")

        controller.service_health_probe = success
        assert await controller.probe_service_health_once() == 1
        async with repository.transaction() as tx:
            if recovery == "lease":
                lease = await tx.get_lease("lease")
                assert lease is not None
                await tx.put_lease(lease.model_copy(update={"state": LeaseState.RELEASED}))
            else:
                cell = await tx.get_cell("cell")
                assert cell is not None
                await tx.put_cell(cell.model_copy(update={"state": CellState.DELETED}))

    original = service.replace_service_replica
    planned = asyncio.Event()
    resume = asyncio.Event()

    async def blocked_replace(*args, **kwargs):
        planned.set()
        await resume.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, "replace_service_replica", blocked_replace)
    task = asyncio.create_task(controller.reconcile_service_replicas_once())
    try:
        await asyncio.wait_for(planned.wait(), timeout=5)
        if recovery == "success":

            async def success(**_kwargs: object):
                return probe_from_response(200, "")

            controller.service_health_probe = success
            assert await controller.probe_service_health_once() == 1
        else:
            async with repository.transaction() as tx:
                if recovery in {"threshold", "no-health-path", "revision"}:
                    current = await tx.get_workload(parent.id)
                    assert current is not None
                    policy = ServicePolicy(
                        replicas=1,
                        revision="v2" if recovery == "revision" else "v1",
                        health_path="" if recovery == "no-health-path" else "/healthz",
                        health_failure_threshold=4 if recovery == "threshold" else 3,
                    )
                    await tx.put_workload(
                        current.model_copy(
                            update={"spec": current.spec.model_copy(update={"metadata": service_metadata(policy)})}
                        )
                    )
                elif recovery == "lease":
                    lease = await tx.get_lease("lease")
                    assert lease is not None
                    await tx.put_lease(lease.model_copy(update={"state": LeaseState.ACTIVE}))
                elif recovery == "cell":
                    cell = await tx.get_cell("cell")
                    assert cell is not None
                    await tx.put_cell(cell.model_copy(update={"state": CellState.READY}))
                else:
                    current = await tx.get_workload(replica.id)
                    assert current is not None
                    if recovery == "generation":
                        await tx.put_workload(current.model_copy(update={"generation": current.generation + 1}))
                    else:
                        await tx.put_outbox(
                            OutboxRecord(
                                id="preempt",
                                idempotency_key="preempt",
                                topic="workload.preempt",
                                aggregate_type="workload",
                                aggregate_id=replica.id,
                                generation=replica.generation,
                                payload={"execution_started": True},
                            )
                        )
                        await tx.put_workload(current.model_copy(update={"preemption_id": "preempt"}))
        if recovery == "success":
            async with repository.transaction() as tx:
                (health,) = await tx.list_service_replica_health(parent.id)
            assert health.replica_generation == replica.generation
            assert health.last_outcome is ServiceProbeOutcome.SUCCESS
            assert health.consecutive_failures == 0
        resume.set()
        assert await asyncio.wait_for(task, timeout=5) == 0
    finally:
        resume.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async with repository.transaction() as tx:
        current = await tx.get_workload(replica.id)
        decisions = await tx.list_service_decisions(parent.id)
        events = await tx.list_events(org_id="acme", project_id="research", limit=100)
    assert current is not None
    assert current.state is WorkloadState.RUNNING
    assert decisions == []
    assert not any(event.action == "service.replica_replaced" for event in events)


async def test_current_failure_evidence_still_retires_replica() -> None:
    repository, controller, _service, parent, replica = await _running_service()
    observations = await controller._observe_replicas(parent.id)
    assert plan_replica_lifecycle(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=3), observations
    )[0].replace
    assert await controller.reconcile_service_replicas_once() == 1
    async with repository.transaction() as tx:
        current = await tx.get_workload(replica.id)
        (decision,) = await tx.list_service_decisions(parent.id)
    assert current is not None and current.state is WorkloadState.CANCELLING
    assert decision.replica_id == replica.id
    assert "3 definite health probes" in decision.reason
    assert decision.observed.unhealthy_replicas == 1
