# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regressions for unsupported and permanently infeasible pool demand."""

from __future__ import annotations

import pytest

from eyvan.capacity.planning import CapacityPlanner, CapacitySimulation, PoolDemand
from eyvan.domain.models import PlacementMode
from eyvan.records import (
    CapacityPoolRecord,
    CellState,
    CommandExecution,
    PolicyRecord,
    Principal,
    Role,
    RoleGrant,
    WorkerResourceScope,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


def _admin() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _operator() -> Principal:
    return Principal(
        subject="operator", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
    )


def _pool(backend: str = "gcp-tpu", *, max_cells: int = 512, region: str = "region-a") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=f"research/{backend}",
        fleet_name="research",
        node_type=backend,
        provider="gcp" if backend == "gcp-tpu" else backend,
        capacity_backend=backend,
        provider_project="test-project",
        region=region,
        zone=f"{region}-a",
        capacity={"cpu": 32, "tpu": 8},
        allocatable={"cpu": 32, "tpu": 8},
        accelerator_type="v5p-16",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=max_cells,
        launch_batch=8,
        idle_timeout_s=99999,
    )


def _spec(pool: CapacityPoolRecord, width: int, *, named: bool = True, elastic: bool = False) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="training",
        execution=CommandExecution(command="python train.py"),
        gang={"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
        provider=pool.provider,
        required_executor="eyvan-native",
        required_labels={"eyvan_pool": pool.id} if named else {},
        worker_count=width,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC if elastic else PlacementMode.ATOMIC,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,width", [("manual", 2), ("kubernetes", 2), ("onprem-ssh", 2), ("gcp-tpu", 257)])
@pytest.mark.parametrize("shape", ["named", "generic", "elastic"])
async def test_unsupported_pool_shape_does_not_abort_unrelated_acquisition(backend: str, width: int, shape: str) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    unsupported = _pool(backend)
    unrelated = _pool().model_copy(update={"id": "research/unrelated", "node_type": "unrelated"})
    assert unsupported.hosts_multi_worker_cells
    await service.apply_fleet(_admin(), "research", (unsupported, unrelated), idempotency_key="fleet")
    workload, _ = await service.submit_workload(
        _operator(),
        org_id="acme",
        project_id="research",
        spec=_spec(unsupported, width, named=shape != "generic", elastic=shape == "elastic"),
        idempotency_key="unsupported",
    )
    other, _ = await service.submit_workload(
        _operator(), org_id="acme", project_id="research", spec=_spec(unrelated, 1), idempotency_key="other"
    )
    assert workload.state is WorkloadState.PENDING
    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        persisted = await transaction.get_workload(workload.id)
    assert [(cell.pool_id, cell.worker_count, cell.autoscale_request_workload_id) for cell in cells] == [
        (unrelated.id, 1, other.id)
    ]
    assert persisted is not None and persisted.state is WorkloadState.PENDING


@pytest.mark.parametrize("backend,width", [("manual", 2), ("kubernetes", 2), ("onprem-ssh", 2), ("gcp-tpu", 257)])
def test_race_or_direct_demand_cannot_bypass_factory_shape_refusal(backend: str, width: int) -> None:
    # Race expansion adds exact demand after simulated scheduling. A planner
    # must not turn such a refused shape into a real failing launch either.
    pool = _pool(backend)
    demand = PoolDemand(real_cell_ids=frozenset(), virtual_worker_counts=(width,), virtual_workload_ids=("race",))
    plan = CapacityPlanner((pool,), {pool.id: []}, utcnow()).plan({pool.id: demand})[pool.id]
    assert plan.launch_worker_counts == ()
    assert plan.target_worker_units == 0
    assert plan.forced_drain_cell_ids == frozenset()


def test_backend_width_refusal_has_no_virtual_cell_bookkeeping() -> None:
    pool = _pool()
    workload = WorkloadRecord(
        id="oversized", org_id="acme", project_id="research", created_by="operator", spec=_spec(pool, 257)
    )
    simulation = CapacitySimulation.build((pool,), (), (workload,), (), utcnow())
    assert simulation.cells == ()
    assert simulation.pool_by_cell_id == {}
    assert simulation.virtual_worker_count_by_cell_id == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("lower_after_admission", [False, True])
async def test_permanently_oversized_atomic_demand_does_not_launch_or_force_drain(
    lower_after_admission: bool,
) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = _pool(max_cells=4 if lower_after_admission else 2)
    await service.apply_fleet(_admin(), "research", (pool,), idempotency_key="fleet")
    workload, _ = await service.submit_workload(
        _operator(), org_id="acme", project_id="research", spec=_spec(pool, 4), idempotency_key="oversized"
    )
    if lower_after_admission:
        pool = CapacityPoolRecord.model_validate({**pool.model_dump(), "max_cells": 2})
        await service.upsert_capacity_pool(_admin(), pool, idempotency_key="lower-ceiling")
    assert workload.state is WorkloadState.PENDING
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        assert await transaction.list_cells() == []
        # Preexisting idle capacity must not be forcibly reclaimed for a shape
        # that cannot fit, even after every cell has been deleted.
        for index in range(2):
            await transaction.put_cell(pool.make_cell(f"small-{index}").model_copy(update={"state": CellState.READY}))
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        assert all(cell.state is CellState.READY for cell in cells)
        for cell in cells:
            await transaction.put_cell(cell.model_copy(update={"state": CellState.DELETED}))
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}


@pytest.mark.parametrize("warm_floor", [0, 2])
def test_planner_discards_impossible_exact_demand_without_losing_explicit_warm_floor(warm_floor: int) -> None:
    pool = CapacityPoolRecord.model_validate({**_pool(max_cells=2).model_dump(), "min_cells": warm_floor})
    demand = PoolDemand(real_cell_ids=frozenset(), virtual_worker_counts=(4,), virtual_workload_ids=("oversized",))
    plan = CapacityPlanner((pool,), {pool.id: []}, utcnow()).plan({pool.id: demand})[pool.id]
    assert plan.launch_worker_counts == (1,) * warm_floor
    assert plan.launch_workload_ids == (None,) * warm_floor
    assert plan.target_worker_units == warm_floor
    assert plan.forced_drain_cell_ids == frozenset()
    cells = [pool.make_cell(f"small-{index}").model_copy(update={"state": CellState.READY}) for index in range(2)]
    plan = CapacityPlanner((pool,), {pool.id: cells}, utcnow()).plan({pool.id: demand})[pool.id]
    assert plan.launch_worker_counts == ()
    assert plan.forced_drain_cell_ids == frozenset()


def test_finite_fragmentation_reclaims_exact_shape_without_launching_placeholder_singletons() -> None:
    pool = _pool(max_cells=4)
    demand = PoolDemand(real_cell_ids=frozenset(), virtual_worker_counts=(4,), virtual_workload_ids=("feasible",))
    cells = [pool.make_cell(f"small-{index}").model_copy(update={"state": CellState.READY}) for index in range(3)]
    plan = CapacityPlanner((pool,), {pool.id: cells}, utcnow()).plan({pool.id: demand})[pool.id]
    assert plan.launch_worker_counts == ()
    assert plan.forced_drain_cell_ids == frozenset(cell.id for cell in cells)
    deleted = [cell.model_copy(update={"state": CellState.DELETED}) for cell in cells]
    plan = CapacityPlanner((pool,), {pool.id: deleted}, utcnow()).plan({pool.id: demand})[pool.id]
    assert plan.launch_worker_counts == (4,)
    assert plan.launch_workload_ids == ("feasible",)


@pytest.mark.asyncio
@pytest.mark.parametrize("tighten_after_admission", [False, True])
async def test_policy_ineligible_warm_supply_must_not_suppress_eligible_alternate(
    tighten_after_admission: bool,
) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool_a = _pool(region="region-a")
    pool_b = _pool(region="region-b").model_copy(update={"id": "research/alternate", "node_type": "alternate"})
    await service.apply_fleet(_admin(), "research", (pool_a, pool_b), idempotency_key="fleet")
    project_admin = Principal(
        subject="project-admin", grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),)
    )
    policy = PolicyRecord(
        org_id="acme", project_id="research", allowed_regions=("region-b",), updated_by="project-admin"
    )
    await service.put_policy(
        project_admin,
        policy.model_copy(update={"allowed_regions": ()}) if tighten_after_admission else policy,
        idempotency_key="initial-policy",
    )
    spec = WorkloadDefinition.model_validate(
        {
            **_spec(pool_a, 4, elastic=True).model_dump(),
            "required_labels": {"eyvan_pool": pool_a.id, "eyvan_pool_alternates": pool_b.id},
        }
    )
    workload, _ = await service.submit_workload(
        _operator(), org_id="acme", project_id="research", spec=spec, idempotency_key="elastic"
    )
    if tighten_after_admission:
        await service.put_policy(project_admin, policy, idempotency_key="tighten-policy")
    warm_a = pool_a.make_cell("warm-a", worker_count=4).model_copy(update={"state": CellState.READY})
    ready_b = pool_b.make_cell("control-b", worker_count=4).model_copy(update={"state": CellState.READY})
    # Real scheduling proves the workload has an eligible alternate and that
    # A is indeed rejected by the authoritative policy, not by a test double.
    assert service.scheduler.schedule((workload,), (warm_a,), policies=(policy,)).placements == ()
    assert len(service.scheduler.schedule((workload,), (ready_b,), policies=(policy,)).placements) == 1
    async with repository.transaction() as transaction:
        await transaction.put_cell(warm_a)
    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    acquired = [cell for cell in cells if cell.id != warm_a.id]
    assert [(cell.pool_id, cell.worker_count, cell.autoscale_request_workload_id) for cell in acquired] == [
        (pool_b.id, 4, workload.id)
    ]
