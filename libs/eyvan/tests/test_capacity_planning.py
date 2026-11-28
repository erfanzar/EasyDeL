# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Behavioral tests for pure capacity planning over simulated pool demand."""

from __future__ import annotations

import re
from collections import Counter
from datetime import timedelta

import pytest

from eyvan.capacity.planning import (
    MAX_CONSECUTIVE_ACQUISITION_FAILURES,
    POOL_ALTERNATES_LABEL,
    CapacityPlanner,
    CapacitySimulation,
    PoolDemand,
    _candidate_pool_ids,
    _preferred_candidate,
    cell_matches_pool,
    cell_provides_pool_supply,
    cell_requires_replacement,
)
from eyvan.domain.models import PlacementMode, PriorityBand
from eyvan.records import (
    CapacityPoolRecord,
    CapacityStatus,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    Principal,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
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
from eyvan.server.service.errors import InputError
from eyvan.server.service.scheduling import CoreSchedulingService


@pytest.mark.parametrize("timestamp_source", ["observed", "housekeeping", "created"])
def test_recent_acquisition_failures_use_outcome_time(timestamp_source: str) -> None:
    from eyvan.capacity.planning import ACQUISITION_FAILURE_MEMORY_S, _recent_failed_acquisition_streak
    from eyvan.records import CapacityBeliefState

    now = utcnow()
    old = now - timedelta(seconds=ACQUISITION_FAILURE_MEMORY_S + 60)
    recent = now - timedelta(seconds=30)
    cells = [
        CellRecord.model_construct(
            id=f"failed-{index}",
            state=CellState.DELETED,
            created_at=recent if timestamp_source == "created" else old,
            updated_at=None if timestamp_source == "created" else recent,
            availability=CapacityBeliefState(
                consecutive_failures=1, last_observed_at=recent if timestamp_source == "observed" else None
            ),
        )
        for index in range(5)
    ]
    # Ownership cleanup may refresh updated_at on an old tombstone, but it is
    # not a new acquisition outcome and must not revive expired evidence.
    expected = 0 if timestamp_source == "housekeeping" else 5
    assert _recent_failed_acquisition_streak(cells, now) == expected
    expired = [
        cell.model_copy(update={"availability": CapacityBeliefState(consecutive_failures=1, last_observed_at=old)})
        for cell in cells
    ]
    assert _recent_failed_acquisition_streak(expired, now) == 0
    success = cells[0].model_copy(
        update={
            "id": "success",
            "created_at": old - timedelta(seconds=60),
            "availability": CapacityBeliefState(provision_latencies_s=(120.0,), last_observed_at=now),
        }
    )
    assert _recent_failed_acquisition_streak([*cells, success], now) == 0


def fleet_admin() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def operator() -> Principal:
    return Principal(
        subject="operator",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def cpu_pool(pool_id: str = "research/cpu", *, max_cells: int = 4) -> CapacityPoolRecord:
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
        executor="native-local",
        agent_endpoint="http://agent:8090",
        provider_options={"kind": "cpu", "machine_type": "n2-standard-4"},
        max_cells=max_cells,
    )


def multislice_tpu_pool(pool_id: str = "research/v5p-16", *, max_cells: int = 16) -> CapacityPoolRecord:
    """Return the autoscaler template for one v5p-16 TPU worker."""

    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type=pool_id.rsplit("/", 1)[-1],
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity={"cpu": 32, "tpu": 8},
        allocatable={"cpu": 32, "tpu": 8},
        accelerator_type="v5p-16",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=max_cells,
        launch_batch=8,
    )


def shared_quota_pool(
    pool: CapacityPoolRecord,
    *,
    group: str,
    limit: int | None,
    units_per_cell: int | None,
) -> CapacityPoolRecord:
    """Attach a shared quota declaration without re-running pool validators."""

    return pool.model_copy(update={"quota_group": group, "quota_limit": limit, "quota_units_per_cell": units_per_cell})


def generic_multislice_workload(workload_id: str = "workload-multislice", worker_count: int = 2) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=WorkloadDefinition(
            name="multislice-train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
            provider="gcp",
            required_executor="eyvan-native",
            worker_count=worker_count,
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
        ),
    )


def elastic_multislice_workload(
    workload_id: str = "workload-elastic",
    worker_count: int = 4,
    *,
    pool_id: str = "research/v5p-16",
) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=WorkloadDefinition(
            name="elastic-train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
            provider="gcp",
            required_executor="eyvan-native",
            required_labels={"eyvan_pool": pool_id},
            worker_count=worker_count,
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
            placement=PlacementMode.ELASTIC,
        ),
    )


def generic_cpu_workload(workload_id: str = "workload-cpu") -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=WorkloadDefinition(
            name="cpu-train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
    )


def demand_for(pool: CapacityPoolRecord, worker_counts: tuple[int, ...], workload_ids: tuple[str, ...]) -> PoolDemand:
    return PoolDemand(
        real_cell_ids=frozenset(),
        virtual_worker_counts=worker_counts,
        virtual_workload_ids=workload_ids,
    )


def test_generic_multislice_workload_gets_shape_exact_virtual_capacity_and_scale_up_intent() -> None:
    pool = multislice_tpu_pool()
    workload = generic_multislice_workload(worker_count=2)
    now = utcnow()

    simulation = CapacitySimulation.build((pool,), (), (workload,), (), now)

    assert len(simulation.cells) == 1
    virtual_cell = simulation.cells[0]
    assert virtual_cell.state is CellState.READY
    assert virtual_cell.worker_count == 2
    assert virtual_cell.allocatable == ResourceLimits(cpu=64, tpu=16)
    assert simulation.pool_by_cell_id == {virtual_cell.id: pool.id}
    assert simulation.virtual_worker_count_by_cell_id == {virtual_cell.id: 2}

    result = CoreSchedulingService().schedule((workload,), simulation.cells)
    assert [placement.workload_id for placement in result.placements] == [workload.id]

    demand = simulation.demand((pool,), (), result)
    assert demand[pool.id].virtual_worker_counts == (2,)
    assert demand[pool.id].virtual_workload_ids == (workload.id,)

    planner = CapacityPlanner((pool,), {pool.id: []}, now)
    plan = planner.plan(demand)[pool.id]
    assert plan.launch_worker_counts == (2,)
    assert plan.launch_workload_ids == (workload.id,)
    assert plan.target_worker_units == 2


def test_elastic_workload_provisions_only_the_deficit_beyond_warm_cells() -> None:
    """The pods 2 -> 4 resize: two warm slices are reused, two are asked for."""

    pool = multislice_tpu_pool()
    warm = _idle_cell(pool, 1, worker_count=2)
    workload = elastic_multislice_workload("workload-grow", worker_count=4)
    now = utcnow()

    simulation = CapacitySimulation.build((pool,), (warm,), (workload,), (), now)

    virtual_counts = sorted(simulation.virtual_worker_count_by_cell_id.values())
    assert virtual_counts == [2]  # the deficit, not the whole allocation
    warm_in_snapshot = any(cell.id == warm.id for cell in simulation.cells)
    assert warm_in_snapshot

    result = CoreSchedulingService().schedule((workload,), simulation.cells)
    assert [placement.workload_id for placement in result.placements] == [workload.id]
    placement = result.placements[0]
    assert len(placement.all_leases) == 2
    leased_ids = {lease.cell_id for lease in placement.all_leases}
    assert warm.id in leased_ids

    demand = simulation.demand((pool,), (), result)
    assert demand[pool.id].virtual_worker_counts == (2,)
    assert demand[pool.id].real_cell_ids == {warm.id}

    planner = CapacityPlanner((pool,), {pool.id: [warm]}, now)
    plan = planner.plan(demand)[pool.id]
    assert plan.launch_worker_counts == (2,)


def test_elastic_workload_covered_entirely_by_warm_cells_launches_nothing() -> None:
    pool = multislice_tpu_pool()
    warm = (
        _idle_cell(pool, 1, worker_count=2),
        _idle_cell(pool, 2, worker_count=2),
    )
    workload = elastic_multislice_workload("workload-warm", worker_count=4)

    simulation = CapacitySimulation.build((pool,), warm, (workload,), (), utcnow())

    assert simulation.virtual_worker_count_by_cell_id == {}
    result = CoreSchedulingService().schedule((workload,), simulation.cells)
    assert [placement.workload_id for placement in result.placements] == [workload.id]
    assert len(result.placements[0].all_leases) == 2
    demand = simulation.demand((pool,), (), result)
    assert demand[pool.id].virtual_worker_counts == ()


def test_elastic_workload_with_no_warm_supply_provisions_the_whole_shape() -> None:
    pool = multislice_tpu_pool()
    workload = elastic_multislice_workload("workload-cold", worker_count=4)

    simulation = CapacitySimulation.build((pool,), (), (workload,), (), utcnow())

    assert sorted(simulation.virtual_worker_count_by_cell_id.values()) == [4]


def test_generic_multislice_workload_skips_pools_without_a_multiworker_backend() -> None:
    pool = cpu_pool()
    workload = generic_multislice_workload(worker_count=2)

    simulation = CapacitySimulation.build((pool,), (), (workload,), (), utcnow())

    assert simulation.cells == ()
    assert simulation.virtual_worker_count_by_cell_id == {}
    result = CoreSchedulingService().schedule((workload,), simulation.cells)
    assert result.placements == ()
    assert [failure.workload_id for failure in result.infeasibilities] == [workload.id]


def test_generic_single_worker_workload_still_triggers_scale_up() -> None:
    pool = cpu_pool()
    workload = generic_cpu_workload()
    now = utcnow()

    simulation = CapacitySimulation.build((pool,), (), (workload,), (), now)

    assert len(simulation.cells) == 1
    virtual_cell = simulation.cells[0]
    assert virtual_cell.worker_count == 1
    assert virtual_cell.allocatable == ResourceLimits(cpu=4, memory_gib=16)
    assert simulation.virtual_worker_count_by_cell_id == {virtual_cell.id: 1}

    result = CoreSchedulingService().schedule((workload,), simulation.cells)
    assert [placement.workload_id for placement in result.placements] == [workload.id]

    demand = simulation.demand((pool,), (), result)
    assert demand[pool.id].virtual_worker_counts == (1,)
    assert demand[pool.id].virtual_workload_ids == (workload.id,)

    planner = CapacityPlanner((pool,), {pool.id: []}, now)
    plan = planner.plan(demand)[pool.id]
    assert plan.launch_worker_counts == (1,)
    assert plan.launch_workload_ids == (workload.id,)


def test_shared_quota_group_allocates_launches_round_robin() -> None:
    group = "gcp-project-a-us-east5-a-v5p"
    first = shared_quota_pool(multislice_tpu_pool("research/shared-a"), group=group, limit=3, units_per_cell=1)
    second = shared_quota_pool(multislice_tpu_pool("research/shared-b"), group=group, limit=3, units_per_cell=1)
    demand = {
        first.id: demand_for(first, (1, 1), ("workload-a1", "workload-a2")),
        second.id: demand_for(second, (1, 1), ("workload-b1", "workload-b2")),
    }

    planner = CapacityPlanner((first, second), {first.id: [], second.id: []}, utcnow())
    plans = planner.plan(demand)

    assert plans[first.id].launch_worker_counts == (1, 1)
    assert plans[first.id].launch_workload_ids == ("workload-a1", "workload-a2")
    assert plans[second.id].launch_worker_counts == (1,)
    assert plans[second.id].launch_workload_ids == ("workload-b1",)
    assert planner.quota_blocked_workload_ids == frozenset({"workload-b2"})


def test_quota_group_pool_missing_limit_fails_with_value_error() -> None:
    pool = shared_quota_pool(multislice_tpu_pool("research/shared"), group="group", limit=None, units_per_cell=1)
    planner = CapacityPlanner((pool,), {pool.id: []}, utcnow())
    demand = {pool.id: demand_for(pool, (1,), ("workload-1",))}

    with pytest.raises(ValueError, match="quota group pools must configure quota_limit"):
        planner.plan(demand)


def test_quota_group_pool_missing_units_fails_with_value_error_during_allocation() -> None:
    pool = shared_quota_pool(multislice_tpu_pool("research/shared"), group="group", limit=4, units_per_cell=None)
    planner = CapacityPlanner((pool,), {pool.id: []}, utcnow())
    demand = {pool.id: demand_for(pool, (1,), ("workload-1",))}

    with pytest.raises(ValueError, match="quota group pools must configure quota_units_per_cell"):
        planner.plan(demand)


def test_quota_group_pool_missing_units_fails_with_value_error_during_reclaim() -> None:
    pool = shared_quota_pool(multislice_tpu_pool("research/shared"), group="group", limit=0, units_per_cell=None)
    planner = CapacityPlanner((pool,), {pool.id: []}, utcnow())
    demand = {pool.id: demand_for(pool, (1,), ("workload-1",))}

    with pytest.raises(ValueError, match="quota group pools must configure quota_units_per_cell"):
        planner.plan(demand)


# --------------------------------------------------------------- defragmentation
#
# The fleet shape these describe: a quota group whose units are all spent on one
# pool's *idle* small cells, and a peer that cannot launch a larger shape because
# of it. Nothing the blocked pool owns can be given up -- it owns nothing -- so
# before this it waited for as long as the idle cells lived, with the
# accelerators it needed sitting in slices too small to use.


def _idle_cell(pool: CapacityPoolRecord, index: int, *, worker_count: int = 1) -> CellRecord:
    from eyvan.capacity.planning import pool_cell_id

    cell_id = pool_cell_id(pool, index)
    return pool.make_cell(cell_id, worker_count=worker_count).model_copy(update={"state": CellState.READY})


def _defrag_group() -> tuple[CapacityPoolRecord, CapacityPoolRecord]:
    group = "gcp-project-a-us-east5-a-v5p"
    small = shared_quota_pool(multislice_tpu_pool("research/v5p-small"), group=group, limit=4, units_per_cell=1)
    large = shared_quota_pool(multislice_tpu_pool("research/v5p-large"), group=group, limit=4, units_per_cell=1)
    return small, large


def _blocked_demand(*, worker_count: int, waited_seconds: float) -> PoolDemand:
    return PoolDemand(
        real_cell_ids=frozenset(),
        virtual_worker_counts=(worker_count,),
        virtual_workload_ids=("workload-large",),
        virtual_priorities=(PriorityBand.PRODUCTION,),
        virtual_pending_seconds=(waited_seconds,),
    )


def test_idle_cells_are_reclaimed_so_a_peer_pool_can_launch_a_larger_shape() -> None:
    """The case that leaves a fleet stuck, and the reason this exists.

    Four idle single-worker cells hold the whole group quota. A four-worker
    shape is pending in a peer pool and fits nowhere: the blocked pool owns no
    cells to reclaim, and the small pool has no reason of its own to scale down.
    """

    small, large = _defrag_group()
    idle = [_idle_cell(small, index) for index in (1, 2, 3, 4)]
    demand = {
        small.id: demand_for(small, (), ()),
        large.id: _blocked_demand(worker_count=4, waited_seconds=600),
    }

    plans = CapacityPlanner((small, large), {small.id: idle, large.id: []}, utcnow()).plan(demand)

    assert plans[small.id].forced_drain_cell_ids == frozenset(cell.id for cell in idle)
    # The drain lands in the *owning* pool's plan, so the executor that already
    # drains cells needs no knowledge that another pool asked for it.
    assert plans[large.id].forced_drain_cell_ids == frozenset()


def test_a_busy_peer_keeps_its_cells() -> None:
    """Defragmentation reclaims idle capacity, never running work.

    A cell serving demand is protected from draining, so counting its units here
    would report a shortage as covered while nothing was actually freed. Taking
    only idle cells keeps the promise and the drain consistent -- and means this
    can never stop a running job.
    """

    small, large = _defrag_group()
    busy = [_idle_cell(small, index) for index in (1, 2, 3, 4)]
    demand = {
        small.id: PoolDemand(
            real_cell_ids=frozenset(cell.id for cell in busy),
            virtual_worker_counts=(),
            virtual_workload_ids=(),
        ),
        large.id: _blocked_demand(worker_count=4, waited_seconds=600),
    }

    plans = CapacityPlanner((small, large), {small.id: busy, large.id: []}, utcnow()).plan(demand)

    assert plans[small.id].forced_drain_cell_ids == frozenset()


def test_a_demand_that_has_barely_waited_does_not_reclaim_a_peer_s_capacity() -> None:
    """Contention and fragmentation are indistinguishable in the first passes.

    Both read as "the group's quota is spent and this shape does not fit". Only
    time separates them, so reclaiming immediately would destroy warm capacity
    to serve a demand that was about to be satisfied anyway, and pay a whole
    provisioning cycle to do it.
    """

    small, large = _defrag_group()
    idle = [_idle_cell(small, index) for index in (1, 2, 3, 4)]
    demand = {
        small.id: demand_for(small, (), ()),
        large.id: _blocked_demand(worker_count=4, waited_seconds=5),
    }

    plans = CapacityPlanner((small, large), {small.id: idle, large.id: []}, utcnow()).plan(demand)

    assert plans[small.id].forced_drain_cell_ids == frozenset()


def test_a_partial_donation_is_not_taken() -> None:
    """All or nothing: the idle half frees capacity and still does not fit.

    Two of the four cells are busy and cannot be taken, so at most two units are
    available against a shortage of four. Draining them would leave the
    four-worker shape exactly as unplaceable as before, having destroyed warm
    capacity to achieve it.
    """

    small, large = _defrag_group()
    cells = [_idle_cell(small, index) for index in (1, 2, 3, 4)]
    busy_ids = frozenset(cell.id for cell in cells[:2])
    demand = {
        small.id: PoolDemand(
            real_cell_ids=busy_ids,
            virtual_worker_counts=(),
            virtual_workload_ids=(),
        ),
        large.id: _blocked_demand(worker_count=4, waited_seconds=600),
    }

    plans = CapacityPlanner((small, large), {small.id: cells, large.id: []}, utcnow()).plan(demand)

    assert plans[small.id].forced_drain_cell_ids == frozenset()


def test_a_pool_that_is_itself_blocked_is_not_asked_to_donate() -> None:
    """Otherwise one stuck demand is traded for another.

    Both pools want capacity the group cannot give. The small pool still
    reclaims *for itself* -- that is the pre-existing same-pool behaviour, and it
    gives up exactly the one cell its own pending shape needs. What must not
    happen is the larger pool taking all four on top of that, which would leave
    the small pool with nothing and its own demand no closer to placed.
    """

    small, large = _defrag_group()
    idle = [_idle_cell(small, index) for index in (1, 2, 3, 4)]
    demand = {
        small.id: PoolDemand(
            real_cell_ids=frozenset(),
            virtual_worker_counts=(1,),
            virtual_workload_ids=("workload-small",),
            virtual_priorities=(PriorityBand.PRODUCTION,),
            virtual_pending_seconds=(600.0,),
        ),
        large.id: _blocked_demand(worker_count=4, waited_seconds=600),
    }

    plans = CapacityPlanner((small, large), {small.id: idle, large.id: []}, utcnow()).plan(demand)

    assert len(plans[small.id].forced_drain_cell_ids) == 1


def test_reclaiming_does_not_repeat_while_the_capacity_is_already_coming_back() -> None:
    """Planning must converge rather than alternate.

    A donation that reappeared every pass would drain the peer, watch it refill,
    and drain it again -- paying a provisioning cycle each time and never
    placing the shape it was for.
    """

    small, large = _defrag_group()
    idle = [_idle_cell(small, index) for index in (1, 2, 3, 4)]
    demand = {
        small.id: demand_for(small, (), ()),
        large.id: _blocked_demand(worker_count=4, waited_seconds=600),
    }
    now = utcnow()

    first = CapacityPlanner((small, large), {small.id: idle, large.id: []}, now).plan(demand)
    draining = [cell.model_copy(update={"state": CellState.DRAINING}) for cell in idle]
    second = CapacityPlanner((small, large), {small.id: draining, large.id: []}, now).plan(demand)

    assert first[small.id].forced_drain_cell_ids == frozenset(cell.id for cell in idle)
    assert second[small.id].forced_drain_cell_ids == frozenset()


@pytest.mark.asyncio
async def test_generic_multislice_workload_scales_tpu_pool_end_to_end() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool()
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    workload, _ = await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="generic-multislice",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
            provider="gcp",
            required_executor="eyvan-native",
            worker_count=2,
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
        ),
        idempotency_key="generic-multislice-1",
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 1
    assert cells[0].pool_id == pool.id
    assert cells[0].worker_count == 2
    assert cells[0].autoscale_request_workload_id == workload.id
    assert cells[0].allocatable == ResourceLimits(cpu=64, tpu=16)
    assert cells[0].provider_options["node_count"] == 2


@pytest.mark.asyncio
async def test_a_multi_worker_workload_pinned_to_a_non_tpu_pool_cannot_stop_the_leader() -> None:
    """One unplaceable workload used to halt scheduling for the whole plane.

    ``capacity-pools`` is the second leader phase, and when this was written a
    raise there aborted every later phase of the cycle -- observation,
    scheduling, settlement, billing -- with the loop retrying the identical
    failure on every tick. Failures are contained per phase now, but the
    behaviour asserted below is the one that made the containment unnecessary
    here: the phase completes. The pinned branch of
    ``CapacitySimulation.build`` called ``make_cell(worker_count=2)`` with no
    backend guard, and ``make_cell`` refuses a multi-worker cell on a non-TPU
    backend by raising. So a workload pinned to such a pool was a durable poison
    pill: it stayed PENDING, so nothing cleared it, and the plane never scheduled
    anything again.

    A workload that can never be placed is skipped and left pending. That is what
    an unplaceable workload is; it must not also be an outage.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    # A TPU pool on the compute backend -- what the catalog says a v5p-8 is, and
    # a shape `eyvan-native` legitimately supports at one worker.
    pool = multislice_tpu_pool(pool_id="research/v5p-8").model_copy(
        update={"capacity_backend": "gcp-compute", "accelerator_type": "v5p-8"}
    )
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    await service.submit_workload(
        operator(),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="pinned-multislice-on-compute",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
            provider="gcp",
            required_executor="eyvan-native",
            required_labels={"eyvan_pool": pool.id},
            worker_count=2,
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
        ),
        idempotency_key="pinned-multislice-1",
    )

    # The phase completes rather than raising, and mints nothing for a shape that
    # cannot exist.
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    # Still true on the next tick: the workload is durable, so a fix that merely
    # survived once would still stop the plane forever.
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        assert await transaction.list_cells() == []


def test_cell_ids_do_not_repeat_across_control_plane_incarnations() -> None:
    """Give new control-plane incarnations distinct cell IDs to avoid durable agent tombstones."""

    from eyvan.capacity.planning import pool_cell_id

    first = multislice_tpu_pool()
    # A control plane on an empty database re-creates the pool, which stamps a
    # fresh created_at. Same id, same prefix, same starting index.
    reborn = first.model_copy(update={"created_at": utcnow()})

    assert pool_cell_id(first, 1) != pool_cell_id(reborn, 1)


def test_re_applying_a_live_fleet_keeps_minting_into_one_namespace() -> None:
    """`eyvan up` bumps generation; that must not orphan the counter."""

    from eyvan.capacity.planning import pool_cell_id

    pool = multislice_tpu_pool()
    # What upsert_capacity_pool does on a re-apply: bump generation, clear
    # initialized_at, carry created_at forward.
    reapplied = pool.model_copy(
        update={"generation": pool.generation + 1, "initialized_at": None, "updated_at": utcnow()}
    )

    assert pool_cell_id(pool, 7) == pool_cell_id(reapplied, 7)


def test_two_pools_sharing_a_prefix_get_different_namespaces() -> None:
    """Otherwise two fleets with the same node type collide on cell ids."""

    from eyvan.capacity.planning import pool_cell_id

    first = multislice_tpu_pool()
    other = first.model_copy(update={"id": f"{first.id}-other"})

    assert pool_cell_id(first, 1) != pool_cell_id(other, 1)


def test_a_minted_cell_id_fits_the_record_and_stays_provider_safe() -> None:
    """Cell ids are `max_length=96` and become GCP resource names."""

    from eyvan.capacity.planning import pool_cell_id

    pool = multislice_tpu_pool().model_copy(update={"fleet_name": "f" * 60, "node_type": "n" * 60})

    for index in (1, 9999, 10_000):
        cell_id = pool_cell_id(pool, index)
        assert len(cell_id) <= 96, cell_id
        assert re.fullmatch(r"[a-z0-9-]+", cell_id), cell_id
        assert not cell_id.startswith("-") and not cell_id.endswith("-")


def test_the_counter_is_never_truncated_away() -> None:
    """Trimming the tail to fit would alias two distinct cells onto one id.

    The interesting index is the one where the id *would* overflow: at a maximal
    prefix, index 10000 needs one character more than index 1000, so a naive
    ``[:96]`` on the assembled string turns "10000" back into "1000" -- a cell
    silently sharing an id, a lease, and a fence with a different one. The prefix
    is the only part it is safe to shorten.
    """

    from eyvan.capacity.planning import pool_cell_id

    pool = multislice_tpu_pool().model_copy(update={"fleet_name": "f" * 60, "node_type": "n" * 60})
    # max_cells is capped at 10_000, so this is the whole reachable range.
    indices = (1, 2, 999, 1000, 1001, 9999, 10_000)

    minted = [pool_cell_id(pool, index) for index in indices]

    assert len(set(minted)) == len(indices), minted
    assert minted[0].endswith("-0001")
    assert minted[-1].endswith("-10000")


def test_a_deleted_cell_never_gives_its_index_back() -> None:
    """Never reuse a deleted cell's index, which could collide with a retained provider resource."""

    from eyvan.capacity.planning import next_pool_cell_index, pool_cell_id

    pool = multislice_tpu_pool()
    first = CellRecord(
        id=pool_cell_id(pool, 1),
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.DELETED,
        capacity=pool.capacity,
        allocatable=pool.allocatable,
        pool_id=pool.id,
    )

    assert next_pool_cell_index(pool, [first]) == 2, "a spent index must never be reissued"


def test_the_index_advances_past_the_highest_ever_used() -> None:
    """Gaps are fine; going backwards is not."""

    from eyvan.capacity.planning import next_pool_cell_index, pool_cell_id

    pool = multislice_tpu_pool()
    cells = [
        CellRecord(
            id=pool_cell_id(pool, index),
            provider="gcp",
            provider_project="project-a",
            region="us-east5",
            zone="us-east5-a",
            state=CellState.DELETED if index < 5 else CellState.READY,
            capacity=pool.capacity,
            allocatable=pool.allocatable,
            pool_id=pool.id,
        )
        for index in (1, 5)
    ]

    assert next_pool_cell_index(pool, cells) == 6


def test_a_pool_with_no_history_starts_at_one() -> None:
    from eyvan.capacity.planning import next_pool_cell_index

    assert next_pool_cell_index(multislice_tpu_pool(), []) == 1


def test_another_pools_cells_do_not_advance_this_counter() -> None:
    """Two pools sharing a prefix must not consume each other's indices."""

    from eyvan.capacity.planning import next_pool_cell_index, pool_cell_id

    pool = multislice_tpu_pool()
    other = pool.model_copy(update={"id": f"{pool.id}-other"})
    foreign = CellRecord(
        id=pool_cell_id(other, 9),
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        capacity=pool.capacity,
        allocatable=pool.allocatable,
        pool_id=other.id,
    )

    assert next_pool_cell_index(pool, [foreign]) == 1


def test_a_pool_edit_that_orphans_a_cell_shape_does_not_stop_the_leader() -> None:
    """The poison pill again, reached by editing a pool instead of submitting.

    `cell_matches_pool` and `cell_requires_replacement` both mint a comparison
    cell from the pool, and `make_cell` refuses a multi-worker cell on a non-TPU
    backend by raising. Narrowing a pool that already has a multi-worker cell
    therefore raised inside the `capacity-pools` leader phase, which carries no
    failure isolation at the time -- so scheduling, settlement and billing stopped every
    cycle, over a configuration change.

    A pool that can no longer mint its own cell has answered both questions: the
    cell does not match, and it requires replacement.
    """

    pool = multislice_tpu_pool()
    wide = pool.make_cell("cell-wide", worker_count=2)
    narrowed = pool.model_copy(update={"capacity_backend": "gcp-compute"})

    assert cell_matches_pool(wide, narrowed) is False
    assert cell_requires_replacement(wide, narrowed) is True


# --------------------------------------------------------------------------- a leased cell in a stopped pool


def _live_workload() -> WorkloadRecord:
    """Return the workload whose lease holds the cell below."""

    return WorkloadRecord(
        id="wl-live",
        # Running, as anything holding a live lease is. A pending workload with
        # an active lease is itself a snapshot the scheduler refuses, and for a
        # different reason than the one under test here.
        state=WorkloadState.RUNNING,
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=WorkloadDefinition(
            name="live",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
            provider="gcp",
            required_executor="eyvan-native",
        ),
    )


def _leased_cell_and_lease(
    pool: CapacityPoolRecord, workload: WorkloadRecord | None = None
) -> tuple[CellRecord, LeaseRecord]:
    """Return one ready cell of ``pool`` and the live lease holding it."""

    holder = workload if workload is not None else _live_workload()
    cell = pool.make_cell(f"{pool.id}-0001", worker_count=1).model_copy(update={"state": CellState.READY})
    lease = LeaseRecord(
        id="lease-live",
        workload_id=holder.id,
        cell_id=cell.id,
        org_id="acme",
        project_id="research",
        # Memory named explicitly at zero: this pool template declares none, and
        # the gang's own default would put a gigabyte on a cell that has no
        # memory to give -- an overcommit the scheduler refuses before it ever
        # reaches the property under test.
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0, memory_gib=0.0)),
        fencing_token=f"{cell.id}:1",
        hourly_cost=1.0,
        expires_at=utcnow() + timedelta(hours=1),
    )
    return cell, lease


def test_a_disabled_pool_does_not_take_its_leased_cell_out_of_the_snapshot() -> None:
    """The scheduler is handed the cells and the live leases in one call.

    It rejects a snapshot whose lease names a cell the list does not hold, and it
    is right to: there is no capacity to check that lease against. So a cell that
    is no longer supply cannot simply be dropped while something still runs on
    it.

    Disabling a pool must therefore retain leased cells in the snapshot so
    observation, settlement and release can continue.
    """

    pool = multislice_tpu_pool()
    cell, lease = _leased_cell_and_lease(pool)
    disabled = pool.model_copy(update={"enabled": False})

    simulation = CapacitySimulation.build((disabled,), (cell,), (), (lease,), utcnow())

    assert [held.id for held in simulation.cells] == [cell.id]


def test_a_leased_cell_in_a_disabled_pool_attracts_no_new_work() -> None:
    """Keeping it in the snapshot must not put it back into supply.

    The cell is reported as draining, which is what it is: it carries work and
    must accept none. ``to_core_cell`` reads that as ``enabled=False``.
    """

    pool = multislice_tpu_pool()
    cell, lease = _leased_cell_and_lease(pool)
    disabled = pool.model_copy(update={"enabled": False})

    simulation = CapacitySimulation.build((disabled,), (cell,), (), (lease,), utcnow())

    assert simulation.cells[0].state is CellState.DRAINING


def test_an_unleased_cell_in_a_disabled_pool_stays_out_of_the_snapshot() -> None:
    """The fix must not resurrect a disabled pool's free capacity."""

    pool = multislice_tpu_pool()
    cell, _lease = _leased_cell_and_lease(pool)
    disabled = pool.model_copy(update={"enabled": False})

    simulation = CapacitySimulation.build((disabled,), (cell,), (), (), utcnow())

    assert simulation.cells == ()


def test_the_scheduler_accepts_the_snapshot_a_disabled_pool_produces() -> None:
    """The property the crash was: this call must not raise.

    Asserted against the real adapter rather than by inspecting the list, because
    what broke was the pair -- cells and leases -- reaching the scheduler
    together.
    """

    pool = multislice_tpu_pool()
    workload = _live_workload()
    cell, lease = _leased_cell_and_lease(pool, workload)
    disabled = pool.model_copy(update={"enabled": False})

    simulation = CapacitySimulation.build((disabled,), (cell,), (workload,), (lease,), utcnow())

    CoreSchedulingService().schedule([workload], simulation.cells, leases=[lease])


@pytest.mark.asyncio
async def test_disabling_a_pool_under_a_live_lease_does_not_stop_the_leader_cycle() -> None:
    """What `eyvan down` does, with something still running.

    The phase must complete. Everything that releases capacity and settles
    billing runs after it, so a raise here is not one phase lost -- it is an
    accelerator nobody stops paying for.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    pool = multislice_tpu_pool()
    await service.apply_fleet(fleet_admin(), "research", (pool,), idempotency_key="fleet-v1")
    workload = _live_workload()
    cell, lease = _leased_cell_and_lease(pool, workload)
    async with repository.transaction() as transaction:
        await transaction.put_workload(workload)
        await transaction.put_cell(cell)
        await transaction.put_lease(lease)
    # The route `eyvan down` calls, rather than a hand-disabled pool: it is what
    # sets enabled=False with min and max at zero, and that combination is the
    # one the live plane wedged on.
    await service.disable_fleet(fleet_admin(), "research", idempotency_key="fleet-down")

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}


# --------------------------------------------------------------------- cross-zone alternates


def _shape_pool(pool_id: str) -> CapacityPoolRecord:
    """One pool of a shape; several of these differ only by zone."""

    return multislice_tpu_pool(pool_id)


def _pending_workload(*, alternates: str = "") -> WorkloadRecord:
    labels = {"eyvan_pool": "research/zone-a"}
    if alternates:
        labels[POOL_ALTERNATES_LABEL] = alternates
    return WorkloadRecord(
        id="wl-pending",
        state=WorkloadState.PENDING,
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=WorkloadDefinition(
            name="pending",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 32, "memory_gib": 0, "tpu": 8}},
            provider="gcp",
            required_executor="eyvan-native",
            required_labels=labels,
        ),
    )


def test_candidate_pools_keep_the_primary_first_and_drop_duplicates() -> None:
    workload = _pending_workload(alternates="research/zone-b, research/zone-a ,research/zone-c")

    assert _candidate_pool_ids(workload) == ("research/zone-a", "research/zone-b", "research/zone-c")


def test_a_workload_without_alternates_names_exactly_the_pool_it_always_did() -> None:
    assert _candidate_pool_ids(_pending_workload()) == ("research/zone-a",)


def test_placement_leaves_a_cooling_zone_for_a_healthy_one() -> None:
    """Prefer a healthy alternate instead of repeatedly asking a cooling zone for capacity."""

    cold, warm = _shape_pool("research/zone-a"), _shape_pool("research/zone-b")
    workload = _pending_workload(alternates=warm.id)

    chosen = _preferred_candidate(
        [cold, warm],
        workload=workload,
        free_owned_exact_supply=Counter(),
        free_shared_exact_supply=Counter(),
        pool_cooling=Counter({cold.id: 2}),
        pool_consecutive_failures=Counter({cold.id: 3}),
    )

    assert chosen.id == warm.id


def test_capacity_that_already_exists_beats_a_healthier_but_empty_zone() -> None:
    """A free cell is certain; a healthy zone is only a better bet."""

    cold, warm = _shape_pool("research/zone-a"), _shape_pool("research/zone-b")
    workload = _pending_workload(alternates=warm.id)

    chosen = _preferred_candidate(
        [cold, warm],
        workload=workload,
        free_owned_exact_supply=Counter(),
        free_shared_exact_supply=Counter({(cold.id, workload.spec.worker_count): 1}),
        pool_cooling=Counter({cold.id: 2}),
        pool_consecutive_failures=Counter({cold.id: 3}),
    )

    assert chosen.id == cold.id


def test_equally_healthy_empty_zones_keep_the_primary() -> None:
    """Ties go to the pool the caller was pinned to, so nothing moves at random."""

    first, second = _shape_pool("research/zone-a"), _shape_pool("research/zone-b")

    chosen = _preferred_candidate(
        [first, second],
        workload=_pending_workload(alternates=second.id),
        free_owned_exact_supply=Counter(),
        free_shared_exact_supply=Counter(),
        pool_cooling=Counter(),
        pool_consecutive_failures=Counter(),
    )

    assert chosen.id == first.id


def _failing_cell(pool: CapacityPoolRecord, cell_id: str, *, consecutive_failures: int) -> CellRecord:
    """One reclaimed cell carrying the belief a failed acquisition writes."""

    cell = pool.make_cell(cell_id, worker_count=2)
    return cell.model_copy(
        update={
            "state": CellState.DELETING,
            "availability": cell.availability.model_copy(update={"consecutive_failures": consecutive_failures}),
        }
    )


def test_a_pool_that_keeps_failing_stops_being_asked_for_new_capacity() -> None:
    """The twenty-cell loop: retrying forever is not a plan.

    Belief already ranks a cold pool last, which decides *where* to ask and never
    whether to stop. A workload whose only candidate keeps failing therefore asks
    it again every time the cooldown expires, and ``max_backoff_s`` caps that at
    an hour -- so one probe produced twenty cells in a single zone across
    nineteen hours, each granted and released without ever becoming READY, while
    the workload sat PENDING with no reason given.

    Below the bound the pool is still asked, because a handful of retries is how
    spot capacity is normally won.
    """

    pool = _shape_pool("research/zone-a")
    workload = _pending_workload()
    now = utcnow()

    tolerated = CapacitySimulation.build(
        (pool,),
        (_failing_cell(pool, "cell-1", consecutive_failures=MAX_CONSECUTIVE_ACQUISITION_FAILURES - 1),),
        (workload,),
        (),
        now,
    )
    assert len(tolerated.cells) == 1, "a pool below the bound must still be asked"
    assert tolerated.acquisition_exhausted_workload_ids == frozenset()

    exhausted = CapacitySimulation.build(
        (pool,),
        (_failing_cell(pool, "cell-1", consecutive_failures=MAX_CONSECUTIVE_ACQUISITION_FAILURES),),
        (workload,),
        (),
        now,
    )
    # No virtual candidate means nothing is launched -- that is the loop broken.
    assert exhausted.cells == ()
    # And the workload is named, so the caller can say why it is not progressing
    # rather than leaving it pending in silence.
    assert exhausted.acquisition_exhausted_workload_ids == frozenset({workload.id})


def test_a_deleted_cell_frozen_belief_never_exhausts_its_pool() -> None:
    """A tombstone's failures must not park the pool's future workloads.

    Deleted cells are retained forever and their belief is frozen -- only the
    cell's own SUCCESS resets it, and a deleted cell has none. Counting them
    in the acquisition bound meant one cell that crossed the bound during an
    outage parked every later workload for that pool permanently, and the gate
    blocked the very acquisitions that could have cured the belief.
    """

    pool = _shape_pool("research/zone-a")
    workload = _pending_workload()

    tombstone = _failing_cell(pool, "cell-1", consecutive_failures=MAX_CONSECUTIVE_ACQUISITION_FAILURES).model_copy(
        update={"state": CellState.DELETED}
    )
    simulation = CapacitySimulation.build((pool,), (tombstone,), (workload,), (), utcnow())

    assert simulation.acquisition_exhausted_workload_ids == frozenset()
    assert len(simulation.cells) == 1, "the pool is still asked for new capacity"


def test_one_healthy_alternate_keeps_the_workload_launching() -> None:
    """The bound is per pool, not per workload: another zone is still tried."""

    cold = _shape_pool("research/zone-a")
    warm = _shape_pool("research/zone-b")
    workload = _pending_workload(alternates="research/zone-b")

    simulation = CapacitySimulation.build(
        (cold, warm),
        (_failing_cell(cold, "cell-1", consecutive_failures=MAX_CONSECUTIVE_ACQUISITION_FAILURES),),
        (workload,),
        (),
        utcnow(),
    )

    assert simulation.acquisition_exhausted_workload_ids == frozenset()
    assert len(simulation.cells) == 1
    assert simulation.pool_by_cell_id[simulation.cells[0].id] == warm.id


def test_a_cell_whose_provider_request_has_not_started_is_never_timed_out() -> None:
    """The churn loop: a provisioning timeout charged for provisioning nobody began.

    ``capacity.apply`` is durable in the outbox. Until it runs the cell has
    ``provider_phase`` None and no request exists at the provider -- so measuring
    ``provisioning_timeout_s`` from ``created_at`` times out something that never
    started.

    The cost is not one wasted cell. The reaped cell's queued apply is discarded
    as stale, a replacement is launched, its apply joins the back of the same
    queue, and the next timeout repeats it while the zone stays empty.

    Once provisioning genuinely starts, the budget applies exactly as before.
    """

    pool = _shape_pool("research/zone-a")
    now = utcnow()
    undispatched = pool.make_cell("cell-undispatched", worker_count=2).model_copy(
        update={
            "state": CellState.REGISTERING,
            "provider_phase": None,
            "provider_phase_started_at": None,
            "created_at": now - timedelta(seconds=pool.provisioning_timeout_s * 4),
        }
    )

    assert cell_provides_pool_supply(undispatched, pool, now), (
        "a cell whose provider request never started was reaped for failing to provision"
    )

    # The budget still bites once the provider has actually been asked.
    dispatched = undispatched.model_copy(
        update={
            "provider_phase": CapacityStatus.PROVISIONING,
            "provider_phase_started_at": now - timedelta(seconds=pool.provisioning_timeout_s + 1),
        }
    )
    assert not cell_provides_pool_supply(dispatched, pool, now)

    fresh = dispatched.model_copy(update={"provider_phase_started_at": now - timedelta(seconds=1)})
    assert cell_provides_pool_supply(fresh, pool, now)


async def test_a_pool_upsert_that_disagrees_with_its_quota_group_is_rejected_loudly() -> None:
    """One misconfigured member must not silently tighten the whole group.

    The planner spends the minimum member limit, so the write path has to be
    where disagreement dies: a pool joining an established group with its own
    idea of ``quota_limit`` is refused instead of shrinking everyone's budget.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    admin = fleet_admin()
    group = "project-us-east5-a-v5p-spot"

    def quota_pool(pool_id: str, *, limit: int) -> CapacityPoolRecord:
        return shared_quota_pool(
            multislice_tpu_pool(pool_id),
            group=group,
            limit=limit,
            units_per_cell=1,
        )

    await service.upsert_capacity_pool(
        admin,
        quota_pool("research/shared-a", limit=8),
        idempotency_key="pool-a",
    )
    stored, _ = await service.upsert_capacity_pool(
        admin,
        quota_pool("research/shared-b", limit=8),
        idempotency_key="pool-b",
    )
    assert stored.quota_limit == 8

    with pytest.raises(InputError, match="one quota_limit"):
        await service.upsert_capacity_pool(
            admin,
            quota_pool("research/shared-c", limit=4),
            idempotency_key="pool-c",
        )

    # And editing an existing member into disagreement dies the same way.
    with pytest.raises(InputError, match="one quota_limit"):
        await service.upsert_capacity_pool(
            admin,
            quota_pool("research/shared-b", limit=6),
            idempotency_key="pool-b-2",
        )


def test_mixed_naive_and_aware_outcomes_are_both_read_as_utc() -> None:
    """A naive persisted timestamp must neither crash the sort nor shift by the host's zone."""

    from datetime import timedelta

    from eyvan.capacity.planning import _recent_failed_acquisition_streak
    from eyvan.records import CapacityBeliefState

    now = utcnow()
    outcomes = (now - timedelta(seconds=30), (now - timedelta(seconds=60)).replace(tzinfo=None))
    cells = [
        CellRecord.model_construct(
            id=f"failed-{index}",
            state=CellState.DELETED,
            created_at=now - timedelta(hours=3),
            availability=CapacityBeliefState(consecutive_failures=1, last_observed_at=outcome),
        )
        for index, outcome in enumerate(outcomes)
    ]

    assert _recent_failed_acquisition_streak(cells, now) == 2


def _replaced_incarnations(pool, count: int, *, age_s: float, delivered_after: int | None = None):
    """``count`` deleted cells, oldest first, each failed once and replaced by the next."""

    from datetime import timedelta

    now = utcnow()
    cells = []
    for index in range(count):
        cell = _failing_cell(pool, f"cell-{index + 1}", consecutive_failures=2).model_copy(
            update={
                "state": CellState.DELETED,
                "created_at": now - timedelta(seconds=age_s - index),
                "updated_at": now - timedelta(seconds=age_s - index),
            }
        )
        if delivered_after is not None and index == delivered_after:
            cell = cell.model_copy(
                update={"availability": cell.availability.model_copy(update={"provision_latencies_s": (120.0,)})}
            )
        cells.append(cell)
    return tuple(cells), now


def test_a_replacement_loop_trips_the_bound_even_though_each_cell_is_new() -> None:
    """Bound repeated acquisition failures across replacement cells rather than resetting per incarnation."""

    pool = _shape_pool("research/zone-a")
    workload = _pending_workload()
    cells, now = _replaced_incarnations(pool, MAX_CONSECUTIVE_ACQUISITION_FAILURES, age_s=600)

    simulation = CapacitySimulation.build((pool,), cells, (workload,), (), now)

    assert simulation.acquisition_exhausted_workload_ids == frozenset({workload.id})
    assert simulation.cells == ()


def test_a_replacement_loop_below_the_bound_is_still_asked() -> None:
    pool = _shape_pool("research/zone-a")
    workload = _pending_workload()
    cells, now = _replaced_incarnations(pool, MAX_CONSECUTIVE_ACQUISITION_FAILURES - 1, age_s=600)

    simulation = CapacitySimulation.build((pool,), cells, (workload,), (), now)

    assert simulation.acquisition_exhausted_workload_ids == frozenset()
    assert len(simulation.cells) == 1


def test_old_failed_incarnations_never_park_a_pool() -> None:
    """Yesterday's outage is not today's belief: past the memory window the pool is asked again."""

    from eyvan.capacity.planning import ACQUISITION_FAILURE_MEMORY_S

    pool = _shape_pool("research/zone-a")
    workload = _pending_workload()
    cells, now = _replaced_incarnations(
        pool, MAX_CONSECUTIVE_ACQUISITION_FAILURES + 3, age_s=ACQUISITION_FAILURE_MEMORY_S + 60
    )

    simulation = CapacitySimulation.build((pool,), cells, (workload,), (), now)

    assert simulation.acquisition_exhausted_workload_ids == frozenset()
    assert len(simulation.cells) == 1


def test_one_delivered_incarnation_ends_the_failure_streak() -> None:
    """A pool that granted since its failures is not failing; one success ends a run."""

    pool = _shape_pool("research/zone-a")
    workload = _pending_workload()
    cells, now = _replaced_incarnations(
        pool,
        MAX_CONSECUTIVE_ACQUISITION_FAILURES + 2,
        age_s=600,
        delivered_after=MAX_CONSECUTIVE_ACQUISITION_FAILURES,
    )

    simulation = CapacitySimulation.build((pool,), cells, (workload,), (), now)

    assert simulation.acquisition_exhausted_workload_ids == frozenset()
