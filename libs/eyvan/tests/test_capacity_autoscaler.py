# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.capacity.drivers.base import CapacityStatus
from eyvan.records import (
    AutoscaleBlockReason,
    CallableExecution,
    CapacityPoolRecord,
    CellState,
    CommandExecution,
    NativeRuntimeConfig,
    Principal,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.models import AgentObservation


def fleet_admin() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def operator() -> Principal:
    return Principal(
        subject="operator",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


class CapacityAgent:
    def __init__(self, state: CapacityStatus = CapacityStatus.WAITING):
        self.state = state

    def apply_capacity(self, cell_id, request):
        return AgentObservation(cell_id=cell_id, state=self.state, generation=request.mutation.generation)

    def reconcile(self, cell_id, request):
        return AgentObservation(cell_id=cell_id, state=self.state, generation=request.mutation.generation)


def cpu_pool(
    pool_id: str = "research/cpu",
    *,
    min_cells: int = 0,
    initial_cells: int | None = None,
    max_cells: int = 4,
    idle_timeout_s: float = 300,
    queue_timeout_s: float | None = None,
    provisioning_timeout_s: float = 1800,
    hourly_cpu_price: float = 0,
    shrink_stabilization_s: float = 0,
) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type=pool_id.rsplit("/", 1)[-1],
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 4, "memory_gib": 16},
        allocatable={"cpu": 4, "memory_gib": 16},
        unit_prices={"cpu": hourly_cpu_price},
        executor="native-local",
        agent_endpoint="http://agent:8090",
        provider_options={"kind": "cpu", "machine_type": "n2-standard-4"},
        min_cells=min_cells,
        initial_cells=initial_cells,
        max_cells=max_cells,
        idle_timeout_s=idle_timeout_s,
        queue_timeout_s=queue_timeout_s,
        provisioning_timeout_s=provisioning_timeout_s,
        shrink_stabilization_s=shrink_stabilization_s,
    )


def tpu_pool(
    pool_id: str = "research/v5p",
    *,
    accelerator_type: str = "v5p-512",
    max_cells: int = 4,
    shrink_stabilization_s: float = 0,
) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type=pool_id.rsplit("/", 1)[-1],
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity={"tpu": 512},
        allocatable={"tpu": 512},
        accelerator_type=accelerator_type,
        capacity_class="spot",
        executor="native-local",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=0,
        max_cells=max_cells,
        shrink_stabilization_s=shrink_stabilization_s,
    )


def multislice_tpu_pool(
    *,
    min_workers: int = 0,
    max_workers: int = 10_000,
    idle_timeout_s: float = 300,
    shrink_stabilization_s: float = 0,
) -> CapacityPoolRecord:
    """Return the autoscaler template for one v5p-16 TPU slice."""

    return tpu_pool("research/v5p-16", accelerator_type="v5p-16", max_cells=max_workers).model_copy(
        update={
            "capacity": ResourceLimits(cpu=32, tpu=8),
            "allocatable": ResourceLimits(cpu=32, tpu=8),
            "executor": "eyvan-native",
            "launch_batch": 8,
            "min_cells": min_workers,
            "idle_timeout_s": idle_timeout_s,
            "shrink_stabilization_s": shrink_stabilization_s,
        }
    )


def shared_quota_pool(
    pool: CapacityPoolRecord,
    *,
    group: str,
    limit: int,
    units_per_cell: int,
) -> CapacityPoolRecord:
    """Attach the controller-owned shared provider quota declaration to a pool."""

    return pool.model_copy(
        update={
            "quota_group": group,
            "quota_limit": limit,
            "quota_units_per_cell": units_per_cell,
        }
    )


@pytest.mark.asyncio
async def test_initial_capacity_is_created_once_and_disabled_fleet_drains_to_zero() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(initial_cells=2),),
        idempotency_key="fleet-v1",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 2, "drained": 0, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        pool = await transaction.get_capacity_pool("research/cpu")
    assert [cell.state for cell in cells] == [CellState.REGISTERING, CellState.REGISTERING]
    assert pool is not None and pool.initialized_at is not None

    await service.disable_fleet(fleet_admin(), "research", idempotency_key="fleet-down")
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 2, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 2}


@pytest.mark.asyncio
async def test_shared_quota_group_caps_aggregate_minimum_workers_without_affecting_other_pools() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    first = shared_quota_pool(
        tpu_pool("research/shared-a", max_cells=4).model_copy(update={"min_cells": 1}),
        group="gcp-project-a-us-central1-a-v5p-spot",
        limit=4,
        units_per_cell=2,
    )
    second = shared_quota_pool(
        tpu_pool("research/shared-b", max_cells=4).model_copy(update={"min_cells": 1}),
        group="gcp-project-a-us-central1-a-v5p-spot",
        limit=4,
        units_per_cell=2,
    )
    unrelated = tpu_pool("research/unrelated", max_cells=4).model_copy(update={"min_cells": 1})
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (first, second, unrelated),
        idempotency_key="fleet-v1",
    )

    first_pass = await controller.reconcile_capacity_pools_once()
    second_pass = await controller.reconcile_capacity_pools_once()

    assert first_pass == {"launched": 3, "drained": 0, "deleted": 0}
    assert second_pass == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    by_pool = {
        pool_id: [cell for cell in cells if cell.pool_id == pool_id and cell.state is not CellState.DELETED]
        for pool_id in (first.id, second.id, unrelated.id)
    }
    assert len(by_pool[first.id]) == 1
    assert len(by_pool[second.id]) == 1
    assert sum(len(by_pool[pool.id]) * 2 for pool in (first, second)) == 4
    assert len(by_pool[unrelated.id]) == 1


@pytest.mark.asyncio
async def test_shared_quota_group_caps_demand_scale_up_while_unrelated_demand_scales_normally() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    first = shared_quota_pool(
        tpu_pool("research/shared-a", max_cells=3),
        group="gcp-project-a-us-central1-a-v5p-spot",
        limit=4,
        units_per_cell=2,
    )
    second = shared_quota_pool(
        tpu_pool("research/shared-b", max_cells=3),
        group="gcp-project-a-us-central1-a-v5p-spot",
        limit=4,
        units_per_cell=2,
    )
    unrelated = tpu_pool("research/unrelated", max_cells=3)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (first, second, unrelated),
        idempotency_key="fleet-v1",
    )
    for pool in (first, second, unrelated):
        for index in range(3):
            await service.submit_pool_workload(
                operator(),
                fleet_name="research",
                node_type=pool.node_type,
                org_id="acme",
                project_id="research",
                command=f"python train.py --pool {pool.node_type} --worker {index}",
                idempotency_key=f"{pool.node_type}-{index}",
            )

    first_pass = await controller.reconcile_capacity_pools_once()
    second_pass = await controller.reconcile_capacity_pools_once()

    assert first_pass == {"launched": 5, "drained": 0, "deleted": 0}
    assert second_pass == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    shared_cells = [
        cell for cell in cells if cell.pool_id in {first.id, second.id} and cell.state is not CellState.DELETED
    ]
    assert len([cell for cell in shared_cells if cell.pool_id == first.id]) == 1
    assert len([cell for cell in shared_cells if cell.pool_id == second.id]) == 1
    assert len(shared_cells) * 2 == 4
    assert len([cell for cell in cells if cell.pool_id == unrelated.id and cell.state is not CellState.DELETED]) == 3


@pytest.mark.asyncio
async def test_pending_workload_scales_pool_and_registering_capacity_prevents_launch_storm() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(fleet_admin(), "research", (cpu_pool(),), idempotency_key="fleet-v1")
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )

    first = await controller.reconcile_capacity_pools_once()
    second = await controller.reconcile_capacity_pools_once()

    assert first == {"launched": 1, "drained": 0, "deleted": 0}
    assert second == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 1
    assert cells[0].pool_id == "research/cpu"
    assert cells[0].provider_options["machine_type"] == "n2-standard-4"


@pytest.mark.asyncio
async def test_execution_connecting_cell_is_retained_for_its_queued_workload() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = cpu_pool(max_cells=1, idle_timeout_s=0).model_copy(update={"executor": "eyvan-native"})
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
            required_executor="eyvan-native",
        ),
        idempotency_key="workload-1",
    )
    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "provider_phase": CapacityStatus.EXECUTION_CONNECTING,
                    "provider_phase_started_at": cell.created_at,
                }
            )
        )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        retained = (await transaction.list_cells())[0]
    assert retained.state is CellState.REGISTERING
    assert retained.idle_since is None


@pytest.mark.asyncio
async def test_global_planner_launches_only_the_cheapest_compatible_pool() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    cheap = cpu_pool("research/cheap", hourly_cpu_price=0.1)
    expensive = cpu_pool("research/expensive", hourly_cpu_price=1.0)
    await service.apply_fleet(fleet_admin(), "research", (cheap, expensive), idempotency_key="fleet-v1")
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": ResourceVector(cpu=2, memory_gib=4)},
            estimated_duration_seconds=3600,
        ),
        idempotency_key="workload-1",
    )

    result = await controller.reconcile_capacity_pools_once()

    assert result["launched"] == 1
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert [(cell.pool_id, cell.unit_prices["cpu"]) for cell in cells] == [("research/cheap", 0.1)]


@pytest.mark.asyncio
async def test_pool_run_scales_only_the_selected_zero_worker_pool() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    cheap = cpu_pool("research/cheap", hourly_cpu_price=0.1)
    selected = cpu_pool("opaque-selected-pool", hourly_cpu_price=1.0).model_copy(update={"node_type": "selected"})
    await service.apply_fleet(fleet_admin(), "research", (cheap, selected), idempotency_key="fleet-v1")
    await service.submit_pool_workload(
        operator(),
        fleet_name="research",
        node_type="selected",
        org_id="acme",
        project_id="research",
        command="python train.py",
        idempotency_key="run-1",
    )

    first = await controller.reconcile_capacity_pools_once()
    second = await controller.reconcile_capacity_pools_once()

    assert first == {"launched": 1, "drained": 0, "deleted": 0}
    assert second == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        workloads = await transaction.list_workloads()
    assert [cell.pool_id for cell in cells] == ["opaque-selected-pool"]
    assert workloads[0].spec.required_labels == {
        "eyvan_pool": "opaque-selected-pool",
        "eyvan_pool_template": selected.template_fingerprint,
        "eyvan_worker_count": "1",
    }


@pytest.mark.asyncio
async def test_pool_callable_scales_matching_tpu_from_zero_without_launch_storm() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    selected = tpu_pool("opaque-v5p-pool")
    other = tpu_pool("research/v6e", accelerator_type="v6e-8")
    await service.apply_fleet(fleet_admin(), "research", (selected, other), idempotency_key="fleet-v1")
    await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-512",
        pods=1,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
            result_uri="gs://artifacts/results/train.result",
        ),
        idempotency_key="callable-1",
    )

    first = await controller.reconcile_capacity_pools_once()
    second = await controller.reconcile_capacity_pools_once()

    assert first == {"launched": 1, "drained": 0, "deleted": 0}
    assert second == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        workloads = await transaction.list_workloads()
    assert [cell.pool_id for cell in cells] == [selected.id]
    assert workloads[0].spec.required_labels == {
        "eyvan_pool": selected.id,
        "eyvan_pool_template": selected.template_fingerprint,
        "eyvan_worker_count": "1",
    }


@pytest.mark.asyncio
async def test_tpu_multislice_workload_scales_one_atomic_queued_resource() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    selected = multislice_tpu_pool()
    await service.apply_fleet(fleet_admin(), "research", (selected,), idempotency_key="fleet-v1")
    workload, _ = await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=6,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key="six-v5p-16-pods",
    )

    assert workload.spec.cell_group_count == 1
    assert workload.spec.worker_count == 6
    assert workload.spec.worker_resource_scope is WorkerResourceScope.PER_WORKER
    assert workload.spec.gang.per_replica.cpu == 32
    assert workload.spec.gang.per_replica.tpu == 8
    assert workload.spec.resources_per_cell == ResourceLimits(cpu=192, tpu=48)
    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 1
    assert cells[0].pool_id == selected.id
    assert cells[0].accelerator_type == "v5p-16"
    assert cells[0].worker_count == 6
    assert cells[0].autoscale_request_workload_id == workload.id
    assert cells[0].allocatable == ResourceLimits(cpu=192, tpu=48)
    assert cells[0].provider_options["node_count"] == 6
    assert cells[0].provider_options["node_prefix"].startswith(f"{cells[0].id}-node")


@pytest.mark.asyncio
async def test_provisioning_atomic_multislice_does_not_receive_a_stale_hard_constraint_reason() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool()
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    workload, _ = await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key="provisioning-v5p-16-x2",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    result = await controller.reconcile_once()

    assert result.infeasibilities[0].code.value == "hard_constraint"
    async with repository.transaction() as transaction:
        current = await transaction.get_workload(workload.id)
        cell = next(iter(await transaction.list_cells()))
    assert current is not None
    assert current.scheduling_reason is None
    assert cell.worker_count == 2
    assert cell.state is CellState.REGISTERING


@pytest.mark.asyncio
async def test_cancelled_autoscale_request_releases_its_cell_reservation() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(max_workers=4)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    workload, _ = await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/cancelled-train.pkl",
            sha256="b" * 64,
            python="3.13",
        ),
        idempotency_key="cancelled-v5p-16",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    await service.cancel_workload(operator(), workload.id, idempotency_key="cancel-workload")
    await controller.reconcile_capacity_pools_once()

    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 1
    assert cells[0].autoscale_request_workload_id is None


@pytest.mark.asyncio
async def test_tpu_min_workers_is_a_persistent_independent_worker_floor() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(min_workers=4)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")

    assert await controller.reconcile_capacity_pools_once() == {"launched": 4, "drained": 0, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 4
    assert all(cell.state is CellState.REGISTERING for cell in cells)
    assert all(cell.worker_count == 1 for cell in cells)
    assert all("node_count" not in cell.provider_options for cell in cells)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [CellState.READY, CellState.REGISTERING])
async def test_existing_exact_multislice_capacity_prevents_duplicate_launch(state: CellState) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(max_workers=4)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            pool.make_cell("existing-v5p-group", worker_count=2).model_copy(update={"state": state})
        )
    await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key=f"two-v5p-16-{state.value}",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert [(cell.id, cell.worker_count, cell.state) for cell in cells] == [("existing-v5p-group", 2, state)]


@pytest.mark.asyncio
async def test_multislice_demand_and_warm_floor_share_worker_unit_target() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(min_workers=4, max_workers=8)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key="two-v5p-16",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 3, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert sorted(cell.worker_count for cell in cells) == [1, 1, 2]
    assert sum(cell.worker_count for cell in cells) == 4


@pytest.mark.asyncio
async def test_multislice_demand_drains_warm_singletons_before_atomic_replacement() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(min_workers=4, max_workers=4)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    assert await controller.reconcile_capacity_pools_once() == {"launched": 4, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        for cell in cells:
            await transaction.put_cell(cell.model_copy(update={"state": CellState.READY}))
    await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        # This suite pins the exact-shape (atomic) drain and reclaim paths;
        # the elastic default would pack the warm cells instead.
        atomic=True,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key="fragmented-two-v5p-16",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 2, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 2}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        for cell in cells:
            if cell.state is CellState.DELETING:
                await transaction.put_cell(cell.model_copy(update={"state": CellState.DELETED}))

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    live = [cell for cell in cells if cell.state is not CellState.DELETED]
    assert sorted(cell.worker_count for cell in live) == [1, 1, 2]
    assert sum(cell.worker_count for cell in live) == pool.max_cells


@pytest.mark.asyncio
async def test_elastic_resize_reuses_warm_slices_and_provisions_only_the_deficit() -> None:
    """The pods 2 -> 4 resize, end to end: keep the two, ask for two more."""

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(max_workers=8)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            pool.make_cell("warm-v5p-group", worker_count=2).model_copy(update={"state": CellState.READY})
        )
    workload, _ = await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=4,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key="grow-two-to-four",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert sorted(cell.worker_count for cell in cells) == [2, 2]
    assert any(cell.id == "warm-v5p-group" and cell.state is CellState.READY for cell in cells)

    # All-or-nothing: while the deficit is still provisioning, the warm cell
    # is not leased hostage to an allocation that does not exist yet.
    await controller.reconcile_once()
    async with repository.transaction() as transaction:
        record = await transaction.get_workload(workload.id)
        assert record is not None and record.state is WorkloadState.PENDING

    async with repository.transaction() as transaction:
        for cell in await transaction.list_cells():
            if cell.state is not CellState.READY:
                await transaction.put_cell(cell.model_copy(update={"state": CellState.READY}))

    await controller.reconcile_once()
    async with repository.transaction() as transaction:
        record = await transaction.get_workload(workload.id)
        assert record is not None
        assert record.state is WorkloadState.LEASED
        leases = [lease for lease in await transaction.list_leases() if lease.workload_id == workload.id]
    assert len(leases) == 2
    assert {lease.cell_id for lease in leases} == {cell.id for cell in cells}


@pytest.mark.asyncio
async def test_an_atomic_launch_drains_a_leftover_smaller_cell_nothing_else_demands() -> None:
    """The other half of the resize: atomic mode rebuilds, so the old shape leaves now."""

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(max_workers=8)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            pool.make_cell("leftover-v5p-group", worker_count=2).model_copy(update={"state": CellState.READY})
        )
    await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=4,
        atomic=True,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key="atomic-four",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 1, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    leftover = next(cell for cell in cells if cell.id == "leftover-v5p-group")
    assert leftover.state is CellState.DRAINING
    launched = next(cell for cell in cells if cell.id != "leftover-v5p-group")
    assert launched.worker_count == 4


@pytest.mark.asyncio
async def test_shared_quota_never_splits_a_multislice_launch_to_fill_fragment() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = shared_quota_pool(
        multislice_tpu_pool(max_workers=8),
        group="gcp-project-a-us-east5-a-v5p-spot",
        limit=24,
        units_per_cell=8,
    )
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    for index in range(2):
        await service.submit_pool_callable(
            operator(),
            fleet_name="research",
            org_id="acme",
            project_id="research",
            accelerator_type="v5p-16",
            pods=2,
            execution=CallableExecution(
                artifact_uri=f"gs://artifacts/callables/train-{index}.pkl",
                sha256="a" * 64,
                python="3.13",
            ),
            idempotency_key=f"quota-two-v5p-16-{index}",
        )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert [cell.worker_count for cell in cells] == [2]
    assert sum(cell.worker_count * 8 for cell in cells) == 16


@pytest.mark.asyncio
async def test_shared_quota_marks_only_rejected_workload_and_clears_after_capacity_expands() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = shared_quota_pool(
        multislice_tpu_pool(max_workers=8),
        group="gcp-project-a-us-east5-a-v5p-spot",
        limit=24,
        units_per_cell=8,
    )
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    submitted_ids: list[str] = []
    for index in range(2):
        workload, _ = await service.submit_pool_callable(
            operator(),
            fleet_name="research",
            org_id="acme",
            project_id="research",
            accelerator_type="v5p-16",
            pods=2,
            execution=CallableExecution(
                artifact_uri=f"gs://artifacts/callables/quota-block-{index}.pkl",
                sha256="a" * 64,
                python="3.13",
            ),
            idempotency_key=f"quota-block-v5p-16-{index}",
        )
        submitted_ids.append(workload.id)

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    scheduling = await controller.reconcile_once()
    async with repository.transaction() as transaction:
        pending = [await transaction.get_workload(workload_id) for workload_id in submitted_ids]
    blocked = [
        workload
        for workload in pending
        if workload is not None and workload.autoscale_block_reason is AutoscaleBlockReason.SHARED_TPU_QUOTA
    ]
    assert len(blocked) == 1
    blocked_workload = blocked[0]
    assert blocked_workload.scheduling_reason == (
        "shared Spot TPU quota cannot fit this job's requested complete worker capacity"
    )
    assert any(failure.workload_id == blocked_workload.id for failure in scheduling.infeasibilities)
    assert all(
        workload is not None and (workload.id == blocked_workload.id or workload.autoscale_block_reason is None)
        for workload in pending
    )

    expanded_pool = pool.model_copy(update={"quota_limit": 32})
    await service.apply_fleet(fleet_admin(), "research", (expanded_pool,), idempotency_key="fleet-v2")
    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        unblocked = await transaction.get_workload(blocked_workload.id)
    assert unblocked is not None
    assert unblocked.autoscale_block_reason is None
    assert unblocked.scheduling_reason is None


@pytest.mark.asyncio
async def test_quota_full_warm_floor_is_reclaimed_for_exact_multislice_shape() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    controller = EyvanController(repository, service)
    pool = shared_quota_pool(
        multislice_tpu_pool(min_workers=4, max_workers=8),
        group="gcp-project-a-us-east5-a-v5p-spot",
        limit=32,
        units_per_cell=8,
    )
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    assert await controller.reconcile_capacity_pools_once() == {"launched": 4, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        for cell in cells:
            await transaction.put_cell(cell.model_copy(update={"state": CellState.READY}))
    await service.submit_pool_callable(
        operator(),
        fleet_name="research",
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-16",
        pods=2,
        # Pins the exact-shape reclaim path this test exists for.
        atomic=True,
        execution=CallableExecution(
            artifact_uri="gs://artifacts/callables/train.pkl",
            sha256="a" * 64,
            python="3.13",
        ),
        idempotency_key="quota-full-two-v5p-16",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 2, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 2}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        for cell in cells:
            if cell.state is CellState.DELETING:
                await transaction.put_cell(cell.model_copy(update={"state": CellState.DELETED}))

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    live = [cell for cell in cells if cell.state is not CellState.DELETED]
    assert sorted(cell.worker_count for cell in live) == [1, 1, 2]
    assert sum(cell.worker_count * 8 for cell in live) == 32


@pytest.mark.asyncio
async def test_idle_multislice_cell_drains_and_deletes_as_one_atomic_cell() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool(max_workers=4, idle_timeout_s=0)
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            pool.make_cell("idle-v5p-group", worker_count=3).model_copy(update={"state": CellState.READY})
        )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 1, "deleted": 0}
    async with repository.transaction() as transaction:
        draining = await transaction.get_cell("idle-v5p-group")
    assert draining is not None
    assert draining.state is CellState.DRAINING
    assert draining.worker_count == 3

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 1}
    async with repository.transaction() as transaction:
        deleting = await transaction.get_cell("idle-v5p-group")
    assert deleting is not None
    assert deleting.state is CellState.DELETING
    assert deleting.worker_count == 3


@pytest.mark.asyncio
async def test_pool_runs_scale_from_zero_up_to_max_workers() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = cpu_pool("opaque-dynamic-pool", max_cells=2).model_copy(update={"node_type": "dynamic"})
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    for index in range(3):
        await service.submit_pool_workload(
            operator(),
            fleet_name="research",
            node_type="dynamic",
            org_id="acme",
            project_id="research",
            command=f"python train.py --run {index}",
            idempotency_key=f"run-{index}",
        )

    first = await controller.reconcile_capacity_pools_once()
    second = await controller.reconcile_capacity_pools_once()

    assert first == {"launched": 2, "drained": 0, "deleted": 0}
    assert second == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 2
    assert {cell.pool_id for cell in cells} == {"opaque-dynamic-pool"}


@pytest.mark.asyncio
async def test_pool_run_submitted_during_rollout_targets_only_the_new_template() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    original = cpu_pool("research/rollout", min_cells=1, max_cells=2)
    await service.apply_fleet(fleet_admin(), "research", (original,), idempotency_key="fleet-v1")
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        old_cell = (await transaction.list_cells())[0]
        await transaction.put_cell(old_cell.model_copy(update={"state": CellState.READY}))

    replacement = original.model_copy(update={"provider_options": {"machine_type": "c3-standard-4"}})
    await service.apply_fleet(fleet_admin(), "research", (replacement,), idempotency_key="fleet-v2")
    workload, _ = await service.submit_pool_workload(
        operator(),
        fleet_name="research",
        node_type="rollout",
        org_id="acme",
        project_id="research",
        command="python train.py",
        idempotency_key="run-1",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        new_cell = cells[1]
        await transaction.put_cell(new_cell.model_copy(update={"state": CellState.READY}))

    assert old_cell.labels["eyvan_pool_template"] == original.template_fingerprint
    assert new_cell.labels["eyvan_pool_template"] == replacement.template_fingerprint
    assert workload.spec.required_labels["eyvan_pool_template"] == replacement.template_fingerprint
    result = await service.schedule_once(fleet_admin())
    assert [placement.lease.cell_id for placement in result.placements] == [new_cell.id]


@pytest.mark.asyncio
async def test_pool_template_update_launches_replacement_and_drains_old_cell() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    original = cpu_pool(min_cells=1)
    await service.apply_fleet(fleet_admin(), "research", (original,), idempotency_key="fleet-v1")
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        old_cell = (await transaction.list_cells())[0]
        await transaction.put_cell(old_cell.model_copy(update={"state": CellState.READY}))

    replacement = original.model_copy(update={"provider_options": {"machine_type": "c3-standard-4"}})
    await service.apply_fleet(fleet_admin(), "research", (replacement,), idempotency_key="fleet-v2")

    launch = await controller.reconcile_capacity_pools_once()

    assert launch == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        new_cell = cells[1]
        await transaction.put_cell(new_cell.model_copy(update={"state": CellState.READY}))
    assert [cell.state for cell in cells] == [CellState.READY, CellState.REGISTERING]
    assert cells[1].provider_options == {"machine_type": "c3-standard-4"}

    rollout = await controller.reconcile_capacity_pools_once()

    assert rollout == {"launched": 0, "drained": 1, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert [cell.state for cell in cells] == [CellState.DRAINING, CellState.READY]


@pytest.mark.asyncio
async def test_runtime_template_update_launches_replacement_and_drains_old_cell() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    original = cpu_pool(min_cells=1).model_copy(update={"executor": "eyvan-native"})
    await service.apply_fleet(fleet_admin(), "research", (original,), idempotency_key="fleet-v1")
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        old_cell = (await transaction.list_cells())[0]
        await transaction.put_cell(old_cell.model_copy(update={"state": CellState.READY}))

    replacement = original.model_copy(
        update={"runtime": NativeRuntimeConfig(env={"HF_HOME": "/var/lib/eyvan/workload/hf"})}
    )
    await service.apply_fleet(fleet_admin(), "research", (replacement,), idempotency_key="fleet-v2")

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        replacement_cell = cells[1]
        await transaction.put_cell(replacement_cell.model_copy(update={"state": CellState.READY}))

    assert replacement_cell.runtime.env == {"HF_HOME": "/var/lib/eyvan/workload/hf"}
    assert old_cell.labels["eyvan_pool_template"] == original.template_fingerprint
    assert replacement_cell.labels["eyvan_pool_template"] == replacement.template_fingerprint
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 1, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert [cell.state for cell in cells] == [CellState.DRAINING, CellState.READY]


@pytest.mark.asyncio
async def test_pool_template_update_converges_without_exceeding_max_workers() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    original = cpu_pool(min_cells=1, max_cells=1)
    await service.apply_fleet(fleet_admin(), "research", (original,), idempotency_key="fleet-v1")
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        old_cell = (await transaction.list_cells())[0]
        await transaction.put_cell(old_cell.model_copy(update={"state": CellState.READY}))

    replacement = original.model_copy(update={"provider_options": {"machine_type": "c3-standard-4"}})
    await service.apply_fleet(fleet_admin(), "research", (replacement,), idempotency_key="fleet-v2")

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 1, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 1}
    async with repository.transaction() as transaction:
        deleting = await transaction.get_cell(old_cell.id)
        assert deleting is not None
        await transaction.put_cell(deleting.model_copy(update={"state": CellState.DELETED}))

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    live = [cell for cell in cells if cell.state is not CellState.DELETED]
    assert len(live) == 1
    assert live[0].provider_options == {"machine_type": "c3-standard-4"}


@pytest.mark.asyncio
async def test_pool_metadata_update_is_applied_without_replacing_capacity() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    original = cpu_pool(min_cells=1)
    await service.apply_fleet(fleet_admin(), "research", (original,), idempotency_key="fleet-v1")
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "state": CellState.READY,
                    "provider_phase": CapacityStatus.HEALTHY,
                    "provider_phase_started_at": cell.created_at,
                }
            )
        )
    updated = original.model_copy(
        update={
            "unit_prices": {"cpu": 0.25},
            "labels": {"team": "research"},
            "topology": {"fabric": "dcn-1"},
        }
    )
    await service.apply_fleet(fleet_admin(), "research", (updated,), idempotency_key="fleet-v2")

    result = await controller.reconcile_capacity_pools_once()

    assert result == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 1
    assert cells[0].unit_prices == {"cpu": 0.25}
    assert cells[0].labels["team"] == "research"
    assert cells[0].topology == {"fabric": "dcn-1"}
    assert cells[0].provider_phase is CapacityStatus.HEALTHY
    assert cells[0].provider_phase_started_at == cell.created_at


@pytest.mark.asyncio
async def test_failed_or_deleting_cell_holds_max_worker_slot_until_deleted() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(fleet_admin(), "research", (cpu_pool(max_cells=1),), idempotency_key="fleet-v1")
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "state": CellState.UNAVAILABLE,
                    "provider_phase": CapacityStatus.PROVIDER_ERROR,
                    "provider_phase_started_at": cell.updated_at,
                }
            )
        )

    unavailable = await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        current = await transaction.get_cell(cell.id)
        assert current is not None
        await transaction.put_cell(current.model_copy(update={"state": CellState.DELETING}))
    deleting = await controller.reconcile_capacity_pools_once()

    assert unavailable == {"launched": 0, "drained": 0, "deleted": 0}
    assert deleting == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        current = await transaction.get_cell(cell.id)
        assert current is not None
        await transaction.put_cell(current.model_copy(update={"state": CellState.DELETED}))
    assert len([candidate for candidate in cells if candidate.state is not CellState.DELETED]) == 1

    replacement = await controller.reconcile_capacity_pools_once()

    assert replacement == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len([candidate for candidate in cells if candidate.state is not CellState.DELETED]) == 1
    assert any(candidate.state is CellState.REGISTERING and candidate.id != cell.id for candidate in cells)


@pytest.mark.asyncio
async def test_definitively_absent_cell_is_drained_immediately() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(fleet_admin(), "research", (cpu_pool(max_cells=1),), idempotency_key="fleet-v1")
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        await transaction.put_cell(
            cell.model_copy(update={"state": CellState.UNAVAILABLE, "provider_phase": CapacityStatus.ABSENT})
        )

    result = await controller.reconcile_capacity_pools_once()

    assert result == {"launched": 0, "drained": 1, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert [(candidate.id, candidate.state) for candidate in cells] == [(cell.id, CellState.DRAINING)]


@pytest.mark.asyncio
async def test_registering_cell_is_replaced_after_provisioning_timeout() -> None:
    """The timeout applies to a cell the provider was actually asked about.

    This used to age ``created_at`` on a cell whose ``provider_phase`` was still
    None -- a cell nothing had been asked about yet -- and assert it was drained.
    That is the loop: the reaped cell's queued ``capacity.apply`` is discarded as
    stale, a replacement launches, its apply joins the back of the same queue,
    and the next timeout repeats it while the zone stays empty.

    Provisioning is now timed from when provisioning starts, so the phase is set
    here. See ``test_an_undispatched_cell_is_not_drained_by_the_provisioning_timeout``
    for the case this test used to cover by accident.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(max_cells=1, provisioning_timeout_s=1),),
        idempotency_key="fleet-v1",
    )
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "provider_phase": CapacityStatus.PROVISIONING,
                    "provider_phase_started_at": cell.created_at - timedelta(seconds=2),
                    "created_at": cell.created_at - timedelta(seconds=2),
                }
            )
        )

    result = await controller.reconcile_capacity_pools_once()

    assert result == {"launched": 0, "drained": 1, "deleted": 0}


@pytest.mark.asyncio
async def test_an_undispatched_cell_is_not_drained_by_the_provisioning_timeout() -> None:
    """A cell nobody has asked the provider about cannot be late.

    ``capacity.apply`` is durable in the outbox; until it runs there is no
    provider request, so charging the cell for a slow queue destroys the very
    cell the pending apply belongs to and launches another behind it. The
    provisioning timeout must start only after the provider request is dispatched.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(max_cells=1, provisioning_timeout_s=1),),
        idempotency_key="fleet-v1",
    )
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        # Far past the budget, and still undispatched: no phase, no phase start.
        await transaction.put_cell(cell.model_copy(update={"created_at": cell.created_at - timedelta(seconds=3600)}))

    result = await controller.reconcile_capacity_pools_once()

    assert result["drained"] == 0, "a cell with no provider request was reaped for failing to provision"
    async with repository.transaction() as transaction:
        assert (await transaction.list_cells())[0].state is CellState.REGISTERING


@pytest.mark.asyncio
async def test_queued_cell_is_not_replaced_by_provisioning_timeout() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(max_cells=1, provisioning_timeout_s=1),),
        idempotency_key="fleet-v1",
    )
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        phase_started_at = cell.created_at - timedelta(hours=1)
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "provider_phase": CapacityStatus.WAITING,
                    "provider_phase_started_at": phase_started_at,
                }
            )
        )

    result = await controller.reconcile_capacity_pools_once()

    assert result == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert [(candidate.id, candidate.state) for candidate in cells] == [(cell.id, CellState.REGISTERING)]


@pytest.mark.asyncio
async def test_queued_cell_can_expire_under_explicit_queue_timeout() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(max_cells=1, queue_timeout_s=1),),
        idempotency_key="fleet-v1",
    )
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "provider_phase": CapacityStatus.WAITING,
                    "provider_phase_started_at": cell.created_at - timedelta(seconds=2),
                }
            )
        )

    result = await controller.reconcile_capacity_pools_once()

    assert result == {"launched": 0, "drained": 1, "deleted": 0}


@pytest.mark.asyncio
async def test_provider_phase_timer_resets_between_queue_and_provisioning_only() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = CapacityAgent()
    controller = EyvanController(repository, service, agent_factory=lambda _cell: agent)  # type: ignore[arg-type]
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(max_cells=1, provisioning_timeout_s=1),),
        idempotency_key="fleet-v1",
    )
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        idempotency_key="workload-1",
    )
    await controller.reconcile_capacity_pools_once()
    await controller.dispatch_outbox_once()
    async with repository.transaction() as transaction:
        waiting = (await transaction.list_cells())[0]
        assert waiting.provider_phase is CapacityStatus.WAITING
        assert waiting.provider_phase_started_at is not None
        old_queue_start = waiting.provider_phase_started_at - timedelta(seconds=10)
        await transaction.put_cell(waiting.model_copy(update={"provider_phase_started_at": old_queue_start}))

    agent.state = CapacityStatus.ACTIVE
    assert await controller.observe_cells_once() == 1
    async with repository.transaction() as transaction:
        active = (await transaction.list_cells())[0]
        assert active.provider_phase is CapacityStatus.ACTIVE
        assert active.provider_phase_started_at is not None
        assert active.provider_phase_started_at > old_queue_start
        old_provisioning_start = active.provider_phase_started_at - timedelta(seconds=2)
        await transaction.put_cell(active.model_copy(update={"provider_phase_started_at": old_provisioning_start}))

    agent.state = CapacityStatus.PROVISIONING
    assert await controller.observe_cells_once() == 1
    async with repository.transaction() as transaction:
        provisioning = (await transaction.list_cells())[0]
    assert provisioning.provider_phase is CapacityStatus.PROVISIONING
    assert provisioning.provider_phase_started_at == old_provisioning_start
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 1, "deleted": 0}


@pytest.mark.asyncio
async def test_a_cell_reclaimed_before_it_ever_ran_records_a_stockout_belief() -> None:
    """Otherwise the autoscaler never learns which zone is empty.

    A cell parked in ``WAITING`` past ``queue_timeout_s`` keeps reporting a *pending*
    provider status, and both belief-recording sites key off that status — pending
    leaves ``availability`` untouched. So the pool that never delivers keeps its
    pristine success record and keeps being chosen first, forever.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(initial_cells=1).model_copy(update={"provisioning_timeout_s": 60, "idle_timeout_s": 0}),),
        idempotency_key="fleet-v1",
    )
    assert (await controller.reconcile_capacity_pools_once())["launched"] == 1

    # Age the cell past its provisioning timeout without it ever becoming READY.
    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        stale = cell.created_at - timedelta(seconds=3600)
        await transaction.put_cell(cell.model_copy(update={"created_at": stale, "provider_phase_started_at": stale}))
        before = cell.availability

    await controller.reconcile_capacity_pools_once()

    async with repository.transaction() as transaction:
        reclaimed = (await transaction.list_cells())[0]

    # `before` is the Beta(1, 1) prior a fresh cell starts with, not zero evidence;
    # what matters is that reclaiming a cell that never ran moves it toward failure.
    assert reclaimed.availability.failure_evidence > before.failure_evidence
    assert reclaimed.availability.consecutive_failures >= 1


@pytest.mark.asyncio
async def test_a_healthy_cell_drained_for_idleness_records_no_failure() -> None:
    """Idle-draining a cell that served is not evidence the zone is empty.

    This is the other half of the contract: only a cell reclaimed *without ever
    becoming READY* is a failed acquisition. Recording a failure here would teach the
    autoscaler to avoid a zone precisely because it worked.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(initial_cells=1).model_copy(update={"idle_timeout_s": 0, "min_cells": 0}),),
        idempotency_key="fleet-v1",
    )
    await controller.reconcile_capacity_pools_once()

    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "state": CellState.READY,
                    "provider_phase": CapacityStatus.HEALTHY,
                    "idle_since": cell.created_at - timedelta(seconds=3600),
                }
            )
        )
        served = (await transaction.list_cells())[0]

    await controller.reconcile_capacity_pools_once()

    async with repository.transaction() as transaction:
        drained = (await transaction.list_cells())[0]

    assert drained.state is CellState.DRAINING
    assert drained.availability == served.availability
    assert drained.availability.consecutive_failures == 0


@pytest.mark.asyncio
async def test_the_idle_sweep_does_not_clobber_capacity_that_arrived_concurrently(monkeypatch) -> None:
    """Provisioned capacity was being put back to REGISTERING and never used.

    The idle sweep read its candidates in one transaction and wrote whole-cell
    snapshots back in another, guarded only on the generation. A `capacity.apply`
    result moves a cell REGISTERING -> READY *without* bumping the generation, so
    a result landing between those two transactions passed the guard and was
    overwritten by the stale snapshot: the capacity arrived, the sweep un-arrived
    it, and the fleet held a cell it would never schedule onto.

    The race is modelled where it actually happens -- the apply lands just before
    the sweep re-reads the cell -- because a test that changes the state before
    the candidates are read proves nothing: the snapshot is fresh then, and both
    the correct and the broken implementation agree.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    admin = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    pool = multislice_tpu_pool()
    await service.apply_fleet(admin, "research", (pool,), idempotency_key="fleet-v1")

    cell = pool.make_cell("research-v5p-16-aaa-0001", worker_count=1).model_copy(
        update={"state": CellState.REGISTERING, "idle_since": None}
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)

    original_transaction = repository.transaction
    landed = {"done": False}

    class _ApplyLandsMidSweep:
        """Wrap one transaction so a capacity result lands on the sweep's re-read.

        A class rather than assigning ``__aenter__`` on the context object:
        Python resolves special methods on the type, so an instance attribute is
        simply ignored -- which is why the first attempt at this test silently
        never fired its hook.
        """

        def __init__(self, context):
            self._context = context

        async def __aenter__(self):
            transaction = await self._context.__aenter__()
            original_get_cell = transaction.get_cell

            async def _get_cell(cell_id: str):
                record = await original_get_cell(cell_id)
                # The apply result lands the first time the sweep re-reads this
                # cell: state advances, generation does not.
                if not landed["done"] and record is not None and record.id == cell.id:
                    landed["done"] = True
                    record = record.model_copy(update={"state": CellState.READY})
                    await transaction.put_cell(record)
                return record

            transaction.get_cell = _get_cell  # type: ignore[method-assign]
            return transaction

        async def __aexit__(self, *exc_info):
            return await self._context.__aexit__(*exc_info)

    def _transaction_with_a_concurrent_apply(*args, **kwargs):
        return _ApplyLandsMidSweep(original_transaction(*args, **kwargs))

    monkeypatch.setattr(repository, "transaction", _transaction_with_a_concurrent_apply)

    await controller.reconcile_capacity_pools_once()

    async with original_transaction() as transaction:
        settled = await transaction.get_cell(cell.id)

    assert landed["done"], "the test needs the sweep to have re-read the cell"
    assert settled.state is CellState.READY, "the idle sweep discarded a concurrent capacity result"


@pytest.mark.asyncio
async def test_a_cell_reclaimed_after_it_served_records_no_stockout() -> None:
    """Do not record stockout for a reclaimed cell that previously delivered capacity.

    Current state alone cannot distinguish that cell from one that never became
    ready; ``provision_latencies_s`` records its successful READY transition.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(initial_cells=1).model_copy(update={"provisioning_timeout_s": 60, "idle_timeout_s": 0}),),
        idempotency_key="fleet-v1",
    )
    assert (await controller.reconcile_capacity_pools_once())["launched"] == 1

    async with repository.transaction() as transaction:
        cell = (await transaction.list_cells())[0]
        stale = cell.created_at - timedelta(seconds=3600)
        # Delivered capacity once, then lost it: exactly a spot reclamation.
        delivered = cell.availability.model_copy(update={"provision_latencies_s": (1427.8,)})
        await transaction.put_cell(
            cell.model_copy(
                update={
                    "created_at": stale,
                    "provider_phase_started_at": stale,
                    "availability": delivered,
                }
            )
        )
        before = delivered

    await controller.reconcile_capacity_pools_once()

    async with repository.transaction() as transaction:
        reclaimed = (await transaction.list_cells())[0]

    assert reclaimed.availability.failure_evidence == before.failure_evidence
    assert reclaimed.availability.cooldown_until is None


@pytest.mark.asyncio
async def test_an_idle_cell_is_kept_for_reuse_until_its_window_expires() -> None:
    """A freed cell stays usable, so the next job reuses it instead of re-queuing.

    This is the reason ``idle_timeout_s`` defaults high rather than to zero. A
    real TPU cell recorded 1427.8 seconds to reach READY and a contended zone can
    hold a request in WAITING_FOR_RESOURCES indefinitely, so a sweep that
    reclaims capacity the moment a lease ends trades a twenty-four minute
    reacquisition for a few idle minutes of already-granted capacity.

    Both halves are asserted together because only the pair is the contract:
    holding forever is a leak, and releasing immediately is the trade above.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(initial_cells=1).model_copy(update={"idle_timeout_s": 1800, "min_cells": 0}),),
        idempotency_key="fleet-v1",
    )
    await controller.reconcile_capacity_pools_once()

    async def set_idle_for(seconds: float) -> None:
        async with repository.transaction() as transaction:
            cell = (await transaction.list_cells())[0]
            await transaction.put_cell(
                cell.model_copy(
                    update={
                        "state": CellState.READY,
                        "provider_phase": CapacityStatus.HEALTHY,
                        "idle_since": cell.created_at - timedelta(seconds=seconds),
                    }
                )
            )

    # Inside the window: the slice is held for whatever lands next.
    await set_idle_for(1500)
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        held = (await transaction.list_cells())[0]
    assert held.state is CellState.READY

    # Past it: nobody came, and holding a granted slice indefinitely is a leak.
    await set_idle_for(1900)
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        released = (await transaction.list_cells())[0]
    assert released.state is CellState.DRAINING


async def test_a_demand_shortfall_must_hold_before_the_pool_begins_idling_out_supply() -> None:
    """Shrink hysteresis: one quiet pass does not tear down what cold starts buy.

    With a stabilization window configured, demand that drops below supply
    holds the pool's shape until the shortfall has persisted for the window;
    only then does the idle clock get its turn. The fall also has to stick:
    the held passes must leave the idle clocks running, and the pass that acts
    on an elapsed window must leave the pool clock standing -- clearing either
    would re-arm the hold on the next pass and re-raise the target forever.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    await service.apply_fleet(
        fleet_admin(),
        "research",
        (cpu_pool(initial_cells=2).model_copy(update={"idle_timeout_s": 60, "shrink_stabilization_s": 3600}),),
        idempotency_key="fleet-v1",
    )
    await controller.reconcile_capacity_pools_once()

    async def set_idle_for(cell_id: str, seconds: float) -> None:
        async with repository.transaction() as transaction:
            cell = next(c for c in await transaction.list_cells() if c.id == cell_id)
            await transaction.put_cell(
                cell.model_copy(
                    update={
                        "state": CellState.READY,
                        "provider_phase": CapacityStatus.HEALTHY,
                        "idle_since": cell.created_at - timedelta(seconds=seconds),
                    }
                )
            )

    async with repository.transaction() as transaction:
        cell_ids = sorted(cell.id for cell in await transaction.list_cells())
    assert len(cell_ids) == 2
    surplus, survivor = cell_ids
    # Both cells are READY; the surplus one is long past the idle timeout, the
    # survivor has barely started idling.
    await set_idle_for(surplus, 10_000)
    await set_idle_for(survivor, 5)

    # The shortfall was first seen this pass: the shape holds, the clock
    # starts, and -- the part the held pass decides -- the idle clocks it is
    # holding alive are left running, not cleared.
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cells = {cell.id: cell for cell in await transaction.list_cells()}
        assert all(cell.state is CellState.READY for cell in cells.values())
        assert cells[surplus].idle_since is not None, "a held pass must not clear the idle clock"
        pool = await transaction.get_capacity_pool("research/cpu")
        assert pool is not None
    stamped = pool.demand_below_supply_since
    assert stamped is not None

    # The shortfall holds past the window.
    async with repository.transaction() as transaction:
        current = await transaction.get_capacity_pool("research/cpu")
        assert current is not None and current.demand_below_supply_since is not None
        aged = current.demand_below_supply_since - timedelta(seconds=7200)
        await transaction.put_capacity_pool(current.model_copy(update={"demand_below_supply_since": aged}))

    # The window has elapsed: the target falls, and the idle clock that ran
    # through the hold drains the surplus. The clock is left standing -- this
    # pass must not clear what it just acted on.
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cells = {cell.id: cell for cell in await transaction.list_cells()}
        assert cells[surplus].state is CellState.DRAINING
        assert cells[survivor].state is CellState.READY
        pool = await transaction.get_capacity_pool("research/cpu")
        assert pool is not None
        assert pool.demand_below_supply_since == aged

    # The next pass finds the shortfall still standing and holds the fall: no
    # replacement is launched, the survivor is not re-protected, and neither
    # the pool clock nor the survivor's idle clock was disturbed.
    await controller.reconcile_capacity_pools_once()
    async with repository.transaction() as transaction:
        cells = {cell.id: cell for cell in await transaction.list_cells()}
        assert len(cells) == 2, "a persistent shortfall never buys the capacity back"
        assert cells[survivor].state is CellState.READY
        assert cells[survivor].idle_since is not None
        pool = await transaction.get_capacity_pool("research/cpu")
        assert pool is not None
        assert pool.demand_below_supply_since == aged
