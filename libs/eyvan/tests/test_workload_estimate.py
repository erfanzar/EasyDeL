# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-submission cost estimates: what will this cost, before it exists.

The only previous way to learn a workload's price was to submit it and read the
resulting lease's ``reserved_cost`` -- so answering "can I afford this" required
spending some of the budget to find out.

The estimate is deliberately *not* a scheduling dry run. It consults declared
pool attributes and prices and never live cell availability, for two reasons: a
prediction about placement is wrong the moment the fleet moves, and an estimate
that skipped a full pool would tell an operator their workload is unaffordable
when it is merely queued.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    BudgetRecord,
    CapacityClass,
    CapacityPoolRecord,
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


def _pool(
    pool_id: str = "research/v5p-16",
    *,
    prices: dict[str, float] | None = None,
    accelerator_type: str | None = "v5p-16",
    capacity_class: CapacityClass = CapacityClass.ON_DEMAND,
    region: str = "us-east5",
    enabled: bool = True,
    labels: dict[str, str] | None = None,
) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="v5p-16",
        provider="manual",
        capacity_backend="manual",
        region=region,
        agent_endpoint="http://agent:8080",
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        unit_prices=prices if prices is not None else {"tpu": 2.0},
        accelerator_type=accelerator_type,
        capacity_class=capacity_class,
        enabled=enabled,
        labels=labels or {},
        max_cells=10,
    )


def _spec(
    *,
    accelerator_type: str | None = "v5p-16",
    capacity_class: CapacityClass = CapacityClass.ON_DEMAND,
    duration_s: int = 3600,
    required_labels: dict[str, str] | None = None,
    region: str | None = None,
) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="train",
        execution=CommandExecution(kind="command", command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0, tpu=8)),
        accelerator_type=accelerator_type,
        capacity_class=capacity_class,
        estimated_duration_seconds=duration_s,
        required_labels=required_labels or {},
        region=region,
    )


async def _seed(service: ControlPlaneService, *pools: CapacityPoolRecord, budget: BudgetRecord | None = None) -> None:
    async with service.repository.transaction() as tx:
        for pool in pools:
            await tx.put_capacity_pool(pool)
        if budget is not None:
            await tx.put_budget(budget)


@pytest.mark.asyncio
async def test_an_estimate_prices_one_hour_of_a_matching_pool() -> None:
    service = await _service()
    await _seed(service, _pool(prices={"tpu": 2.0}))

    estimate = await service.estimate_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(duration_s=3600)
    )

    assert len(estimate.candidates) == 1
    candidate = estimate.candidates[0]
    assert candidate.hourly_cost == pytest.approx(16.0), "8 tpu at 2/unit"
    assert candidate.estimated_cost == pytest.approx(16.0)
    assert candidate.price_source == "pool:research/v5p-16"


@pytest.mark.asyncio
async def test_the_estimate_scales_with_the_declared_duration() -> None:
    service = await _service()
    await _seed(service, _pool(prices={"tpu": 2.0}))

    estimate = await service.estimate_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(duration_s=7200)
    )

    assert estimate.candidates[0].estimated_cost == pytest.approx(32.0)
    assert estimate.estimated_duration_seconds == 7200


@pytest.mark.asyncio
async def test_candidates_are_ordered_cheapest_first() -> None:
    service = await _service()
    await _seed(
        service,
        _pool("research/expensive", prices={"tpu": 5.0}),
        _pool("research/cheap", prices={"tpu": 1.0}),
    )

    estimate = await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=_spec())

    assert [candidate.pool_id for candidate in estimate.candidates] == ["research/cheap", "research/expensive"]
    assert estimate.cheapest is not None
    assert estimate.cheapest.pool_id == "research/cheap"


@pytest.mark.asyncio
async def test_a_pool_with_the_wrong_accelerator_is_not_a_candidate() -> None:
    service = await _service()
    await _seed(service, _pool(accelerator_type="v6e-16"))

    estimate = await service.estimate_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(accelerator_type="v5p-16")
    )

    assert estimate.candidates == ()
    assert estimate.cheapest is None, "no match is a different answer from costing zero"


@pytest.mark.asyncio
async def test_spot_and_on_demand_do_not_estimate_for_each_other() -> None:
    service = await _service()
    await _seed(service, _pool("research/spot", capacity_class=CapacityClass.SPOT, prices={"tpu": 1.0}))

    on_demand = await service.estimate_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(capacity_class=CapacityClass.ON_DEMAND)
    )
    spot = await service.estimate_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(capacity_class=CapacityClass.SPOT)
    )

    assert on_demand.candidates == ()
    assert len(spot.candidates) == 1


@pytest.mark.asyncio
async def test_a_disabled_pool_is_not_a_candidate() -> None:
    service = await _service()
    await _seed(service, _pool(enabled=False))

    estimate = await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=_spec())

    assert estimate.candidates == ()


@pytest.mark.asyncio
async def test_required_labels_narrow_the_candidates() -> None:
    service = await _service()
    await _seed(
        service,
        _pool("research/ml", labels={"team": "ml"}),
        _pool("research/other", labels={"team": "infra"}),
    )

    estimate = await service.estimate_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(required_labels={"team": "ml"}),
    )

    assert [candidate.pool_id for candidate in estimate.candidates] == ["research/ml"]


@pytest.mark.asyncio
async def test_an_unpriced_pool_is_named_rather_than_estimated_as_free() -> None:
    """D7 wearing a friendly face: a zero estimate reads as a bargain."""

    service = await _service()
    await _seed(service, _pool(prices={}))

    estimate = await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=_spec())

    assert estimate.candidates[0].estimated_cost == 0.0
    assert estimate.candidates[0].price_source is None
    assert estimate.unpriced_pool_ids == ("research/v5p-16",)


@pytest.mark.asyncio
async def test_a_hard_budget_names_the_pools_it_would_refuse() -> None:
    service = await _service()
    await _seed(
        service,
        _pool("research/cheap", prices={"tpu": 1.0}),
        _pool("research/expensive", prices={"tpu": 10.0}),
        budget=BudgetRecord(
            org_id="acme", project_id="research", limit=100.0, spent=70.0, hard_limit=True, updated_by="root"
        ),
    )

    estimate = await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=_spec())

    assert estimate.budget_limit == 100.0
    assert estimate.budget_spent == 70.0
    assert estimate.budget_blocked_pool_ids == ("research/expensive",), "80/hour exceeds the 30 remaining"
    assert [candidate.pool_id for candidate in estimate.candidates] == ["research/cheap", "research/expensive"], (
        "a blocked pool stays listed; dropping it would look like missing capacity"
    )


@pytest.mark.asyncio
async def test_a_soft_budget_blocks_nothing() -> None:
    """A soft limit reports overspend; it never refuses admission, so it must not here."""

    service = await _service()
    await _seed(
        service,
        _pool(prices={"tpu": 10.0}),
        budget=BudgetRecord(
            org_id="acme", project_id="research", limit=10.0, spent=9.0, hard_limit=False, updated_by="root"
        ),
    )

    estimate = await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=_spec())

    assert estimate.budget_blocked_pool_ids == ()


@pytest.mark.asyncio
async def test_a_project_with_no_budget_reports_none_rather_than_zero() -> None:
    service = await _service()
    await _seed(service, _pool())

    estimate = await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=_spec())

    assert estimate.budget_limit is None
    assert estimate.budget_spent is None
    assert estimate.budget_blocked_pool_ids == ()


@pytest.mark.asyncio
async def test_the_estimate_writes_nothing() -> None:
    """It is a read: no workload, no lease, no idempotency binding, no audit."""

    service = await _service()
    await _seed(service, _pool())

    await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=_spec())

    async with service.repository.transaction() as tx:
        assert await tx.list_workloads(org_id="acme", project_id="research") == []
        assert await tx.list_leases(org_id="acme", project_id="research") == []
        assert await tx.list_events(org_id="acme", project_id="research") == []


def _multislice_pool() -> CapacityPoolRecord:
    resources = ResourceLimits(tpu=8)
    return CapacityPoolRecord(
        id="research/v5p-8",
        fleet_name="research",
        node_type="v5p-8",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=resources,
        allocatable=resources,
        unit_prices={"tpu": 2.0},
        accelerator_type="v5p-8",
        capacity_class=CapacityClass.ON_DEMAND,
        executor="eyvan-native",
        agent_endpoint="http://agent:8080",
        max_cells=10,
    )


@pytest.mark.asyncio
async def test_a_multislice_estimate_counts_cells_not_cells_times_workers() -> None:
    """``make_cell(worker_count=N)`` already builds and prices an N-wide cell.

    The estimate multiplied that worker-scaled hourly by
    ``cell_group_count * worker_count`` again, so a pods=8 job was priced eight
    times its real hourly -- and a hard budget then refused affordable multislice
    work as unaffordable. Placement is one physical cell of width N here, so
    ``cells`` is the cell group count and nothing else.
    """

    service = await _service()
    await _seed(service, _multislice_pool())
    spec = WorkloadDefinition(
        name="train",
        execution=CommandExecution(kind="command", command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(tpu=8)),
        accelerator_type="v5p-8",
        capacity_class=CapacityClass.ON_DEMAND,
        provider="gcp",
        region="us-east5",
        required_executor="eyvan-native",
        worker_count=2,
    )

    estimate = await service.estimate_workload(_principal(), org_id="acme", project_id="research", spec=spec)

    assert len(estimate.candidates) == 1
    candidate = estimate.candidates[0]
    # One 2-worker cell: 16 tpu at 2/unit -- not two of it.
    assert candidate.cells == 1
    assert candidate.hourly_cost == pytest.approx(32.0)
    assert candidate.estimated_cost == pytest.approx(32.0)
