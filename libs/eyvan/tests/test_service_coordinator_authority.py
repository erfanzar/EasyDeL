# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal service reconciliation does not grant replica job capabilities."""

from __future__ import annotations

import pytest

from eyvan.records import CommandExecution, Principal, Role, RoleGrant, WorkloadDefinition, WorkloadState
from eyvan.server.controller import EyvanController
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import ServicePolicy
from tests.test_coordinated_admission_preserves_unresolved_asset_eviction import repository as _repository

repository = _repository


def _operator() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _service(repository, *, can_submit_children: bool = False):
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    parent, _ = await service.submit_service(
        _operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="api",
            execution=CommandExecution(command="python serve.py"),
            can_submit_children=can_submit_children,
        ),
        policy=ServicePolicy(replicas=1, revision="v1"),
        idempotency_key="default-service",
    )
    return service, controller, parent


async def test_default_service_creates_replica_without_granting_job_child_authority(repository) -> None:
    service, controller, parent = await _service(repository)
    assert not parent.spec.can_submit_children
    assert await controller.reconcile_services_once() == 1
    async with repository.transaction() as tx:
        children = await tx.list_child_workloads(parent.id)
        assert len(children) == 1
        replica = children[0]
        assert not replica.spec.can_submit_children
        assert replica.created_by == parent.created_by
        assert replica.org_id == parent.org_id and replica.project_id == parent.project_id
        assert await tx.get_workload(parent.id) == parent
    with pytest.raises(AuthorizationError, match="can_submit_children"):
        await service.submit_child_workload(
            parent=replica,
            spec=WorkloadDefinition(name="grandchild", execution=CommandExecution(command="true")),
            idempotency_key="untrusted-grandchild",
        )
    await controller.reconcile_services_once()
    async with repository.transaction() as tx:
        assert await tx.list_child_workloads(parent.id) == children
        assert await tx.list_child_workloads(replica.id) == []


async def test_service_snapshot_cannot_forge_public_job_authority(repository) -> None:
    service, _controller, parent = await _service(repository)
    forged = parent.model_copy(update={"spec": parent.spec.model_copy(update={"can_submit_children": True})})
    with pytest.raises(AuthorizationError, match="can_submit_children"):
        await service.submit_child_workload(
            parent=forged,
            spec=WorkloadDefinition(name="forged", execution=CommandExecution(command="true")),
            idempotency_key="untrusted-service-child",
        )
    async with repository.transaction() as tx:
        assert await tx.list_child_workloads(parent.id) == []


@pytest.mark.parametrize("change", ["generation", "cancel", "template", "schedulable"])
async def test_stale_service_plan_cannot_admit_a_replica(repository, monkeypatch, change: str) -> None:
    _service_api, controller, parent = await _service(repository, can_submit_children=True)
    original = controller._start_service_replica

    async def superseded(snapshot, policy, *, slot):
        async with repository.transaction() as tx:
            current = await tx.get_workload(parent.id)
            assert current is not None
            if change == "generation":
                update = {"generation": current.generation + 1}
            elif change == "cancel":
                update = {"state": WorkloadState.CANCELLED}
            elif change == "schedulable":
                update = {"schedulable": True}
            else:
                update = {
                    "spec": current.spec.model_copy(update={"execution": CommandExecution(command="python v2.py")})
                }
            await tx.put_workload(current.model_copy(update=update))
        await original(snapshot, policy, slot=slot)

    monkeypatch.setattr(controller, "_start_service_replica", superseded)
    await controller.reconcile_services_once()
    async with repository.transaction() as tx:
        assert await tx.list_child_workloads(parent.id) == []


@pytest.mark.parametrize("change", ["generation", "cancel", "template", "schedulable"])
async def test_service_replica_replay_rechecks_durable_internal_authority(repository, monkeypatch, change: str) -> None:
    service, controller, parent = await _service(repository)
    original = service._submit_service_replica
    submitted = {}

    async def capture(**kwargs):
        submitted.update(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(service, "_submit_service_replica", capture)
    await controller.reconcile_services_once()
    async with repository.transaction() as tx:
        children = await tx.list_child_workloads(parent.id)
        assert len(children) == 1
        current = await tx.get_workload(parent.id)
        assert current is not None
        if change == "generation":
            update = {"generation": current.generation + 1}
        elif change == "cancel":
            update = {"state": WorkloadState.CANCELLED}
        elif change == "schedulable":
            update = {"schedulable": True}
        else:
            update = {"spec": current.spec.model_copy(update={"execution": CommandExecution(command="python v2.py")})}
        await tx.put_workload(current.model_copy(update=update))
        events = await tx.list_events()
    with pytest.raises((AuthorizationError, ConflictError)):
        await original(**submitted)
    async with repository.transaction() as tx:
        assert await tx.list_child_workloads(parent.id) == children
        assert await tx.list_events() == events
