# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every execution-generation reset invalidates endpoint registration."""

from datetime import timedelta

import pytest

from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceVector,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError


@pytest.fixture(params=["memory", "local-json"])
async def endpoint_repository(request, tmp_path):
    repository = (
        InMemoryRepository() if request.param == "memory" else LocalJsonRepository(tmp_path / "endpoint-state.json")
    )
    await repository.open()
    try:
        yield repository
    finally:
        await repository.close()


@pytest.mark.parametrize("mode", ["failure", "manual", "unstarted", "capacity-loss", "provider-preemption"])
async def test_generation_reset_clears_persisted_endpoint(endpoint_repository, mode):
    repository = endpoint_repository
    service = ControlPlaneService(repository, scheduler=None)
    admin = EyvanController(repository, service)._principal
    workload, _ = await service.submit_workload(
        admin,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="endpoint-owner",
            execution=CommandExecution(command="python serve.py"),
            max_failure_retries=1,
            max_retries=1,
        ),
        idempotency_key="submit",
    )
    cell = CellRecord(
        id="previous-cell",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=8),
        allocatable=ResourceVector(cpu=8, memory_gib=8),
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
    lease = LeaseRecord(
        id="previous-lease",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        generation=workload.generation,
        resources=workload.spec.gang,
        fencing_token="previous-fence",
        expires_at=utcnow() + timedelta(minutes=5),
    )
    await service.create_lease(admin, lease, idempotency_key="admit")
    if mode != "unstarted":
        await service.transition_lease(admin, lease.id, LeaseState.ACTIVE)
        await service.transition_workload(admin, workload.id, WorkloadState.DISPATCHING)
        await service.transition_workload(admin, workload.id, WorkloadState.RUNNING)
    current = await service.get_workload(admin, workload.id)
    # Registration currently permits a leased workload, so unstarted requeue
    # must clear registration as well as the running/terminal retry paths.
    registered = await service.register_service_endpoint(workload=current, endpoint="http://previous-host:8080")
    if mode == "failure":
        resumed, _, _ = await service.settle_failed_workload(admin, workload.id)
    elif mode == "manual":
        await service.settle_terminal_workload(admin, workload.id, WorkloadState.FAILED)
        resumed, duplicate = await service.retry_workload(admin, workload.id, idempotency_key="retry")
        assert duplicate is False
    elif mode == "unstarted":
        resumed, _ = await service.requeue_unstarted_workload(
            admin,
            workload.id,
            scheduling_reason="replace undispatched cell",
            expected_generation=registered.generation,
            expected_lease_id=lease.id,
            expected_cell_generation=cell.generation,
            expected_cell_state=cell.state,
            expected_unavailable_reason=cell.unavailable_reason,
            expected_cell_id=cell.id,
        )
    else:
        resumed, _, _ = await service.settle_preempted_workload(
            admin, workload.id, provider_preempted=mode == "provider-preemption"
        )
    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
        released = await tx.get_lease(lease.id)
    assert stored == resumed
    assert stored.generation == 2 and stored.state is WorkloadState.PENDING
    assert stored.service_endpoint is None
    assert released.state is LeaseState.RELEASED
    assert registered.service_endpoint == "http://previous-host:8080"
    with pytest.raises(ConflictError, match="generation 2"):
        await service.register_service_endpoint(workload=registered, endpoint="http://stale-host:8080")
    registered_again = await service.register_service_endpoint(workload=resumed, endpoint="http://replacement-host:8080")
    if mode == "manual":
        replayed, duplicate = await service.retry_workload(admin, workload.id, idempotency_key="retry")
        assert duplicate is True and replayed == registered_again
    async with repository.transaction() as tx:
        assert await tx.get_workload(workload.id) == registered_again
