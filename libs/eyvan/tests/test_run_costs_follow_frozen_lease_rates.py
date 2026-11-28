# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run costs follow frozen lease rates and observed placements, not requests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

PRINCIPAL = Principal(
    subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
)
RESOURCES = ResourceVector(cpu=8, memory_gib=32, tpu=4)


def _pool(pool_id: str, price: float) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="v5p-8",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="a-project",
        region="us-east5",
        zone="us-east5-a",
        capacity=RESOURCES,
        allocatable=RESOURCES,
        accelerator_type="v5p-8",
        capacity_class=CapacityClass.SPOT,
        unit_prices={"tpu": price} if price else {},
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=4,
    )


async def _plane(
    primary_price: float,
    rates: tuple[float, ...],
    *,
    same_cell: bool = False,
) -> tuple[ControlPlaneService, InMemoryRepository]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    spec = WorkloadDefinition(
        name="run",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=RESOURCES),
        required_labels={"eyvan_pool": "primary", "eyvan_pool_alternates": "alternate"},
        metadata={"eyvan_pool": "primary", "eyvan_pool_alternates": "alternate"},
    )
    async with repository.transaction() as tx:
        await tx.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id="acme",
                project_id="research",
                spec=spec,
                state=WorkloadState.SUCCEEDED,
                created_by="alice",
            )
        )
        await tx.put_capacity_pool(_pool("primary", primary_price))
        await tx.put_capacity_pool(_pool("alternate", rates[0] / 4))
        for index, rate in enumerate(rates):
            cell_id = f"cell-{0 if same_cell else index}"
            await tx.put_cell(
                CellRecord(
                    id=cell_id,
                    provider="gcp",
                    provider_project="a-project",
                    region="us-east5",
                    zone="us-east5-b",
                    state=CellState.READY,
                    capacity=RESOURCES,
                    allocatable=RESOURCES,
                    pool_id="alternate",
                    accelerator_type="v5p-8",
                    capacity_class=CapacityClass.SPOT,
                    unit_prices={"tpu": rate / 4} if rate else {},
                )
            )
            await tx.put_lease(
                LeaseRecord(
                    id=f"lease-{index}",
                    workload_id="wl-1",
                    cell_id=cell_id,
                    org_id="acme",
                    project_id="research",
                    state=LeaseState.RELEASED,
                    resources=spec.gang,
                    hourly_cost=rate,
                    settled_cost=rate,
                    accelerator_hours_settled={"tpu": 4.0},
                    settled_at=utcnow(),
                    settlement_reason="released",
                    expires_at=utcnow() + timedelta(minutes=5),
                    fencing_token=f"fence-{index}",
                )
            )
    return service, repository


@pytest.mark.asyncio
@pytest.mark.parametrize(("primary_price", "rate", "priced"), [(2.0, 0.0, False), (0.0, 8.0, True)])
async def test_actual_alternate_lease_controls_price_trust(primary_price: float, rate: float, priced: bool) -> None:
    service, _ = await _plane(primary_price, (rate,))

    cost = await service.workload_cost(PRINCIPAL, "wl-1")

    assert cost.priced is priced
    assert cost.settled == rate
    assert cost.accelerator_hours == {"tpu": 4.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [0.0, 8.0])
async def test_historical_price_trust_survives_pool_and_cell_price_changes(rate: float) -> None:
    service, repository = await _plane(2.0, (rate,))
    before = await service.workload_cost(PRINCIPAL, "wl-1")
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(_pool("primary", 0.0))
        await tx.put_capacity_pool(_pool("alternate", 20.0))
        cell = await tx.get_cell("cell-0")
        assert cell is not None
        await tx.put_cell(cell.model_copy(update={"unit_prices": {"tpu": 20.0}}))
    after = await service.workload_cost(PRINCIPAL, "wl-1")

    assert before.priced is (rate > 0)
    assert after.priced is before.priced
    assert after.settled == before.settled == rate


@pytest.mark.asyncio
@pytest.mark.parametrize("same_cell", [False, True])
async def test_every_lease_needs_a_frozen_positive_rate_even_when_cell_reused(same_cell: bool) -> None:
    service, _ = await _plane(2.0, (8.0, 0.0), same_cell=same_cell)

    cost = await service.workload_cost(PRINCIPAL, "wl-1")

    assert not cost.priced
    assert cost.leases == 2
    assert cost.settled == 8.0
    assert cost.accelerator_hours == {"tpu": 8.0}


@pytest.mark.asyncio
async def test_request_pool_shape_and_class_are_not_observed_placement() -> None:
    service, repository = await _plane(2.0, (8.0,))
    async with repository.transaction() as tx:
        primary = await tx.get_capacity_pool("primary")
        assert primary is not None
        await tx.put_capacity_pool(
            primary.model_copy(update={"accelerator_type": "v6e-8", "capacity_class": CapacityClass.ON_DEMAND})
        )
        on_demand = _pool("on-demand", 4.0).model_copy(update={"capacity_class": CapacityClass.ON_DEMAND})
        await tx.put_capacity_pool(on_demand)

    cost = await service.workload_cost(PRINCIPAL, "wl-1")

    assert cost.shape == "v5p-8"
    assert cost.capacity_class == CapacityClass.SPOT.value
    assert cost.on_demand_hourly == 4.0
    assert cost.priced


@pytest.mark.asyncio
async def test_mixed_observed_shape_and_class_remain_unknown() -> None:
    service, repository = await _plane(2.0, (8.0, 8.0))
    async with repository.transaction() as tx:
        cell = await tx.get_cell("cell-1")
        assert cell is not None
        await tx.put_cell(
            cell.model_copy(update={"accelerator_type": "v6e-8", "capacity_class": CapacityClass.ON_DEMAND})
        )

    cost = await service.workload_cost(PRINCIPAL, "wl-1")

    assert cost.shape is None
    assert cost.capacity_class is None
    assert cost.on_demand_hourly is None
    assert cost.priced


@pytest.mark.asyncio
async def test_a_replaced_cell_cannot_describe_the_historical_lease() -> None:
    service, repository = await _plane(2.0, (8.0,))
    async with repository.transaction() as tx:
        cell = await tx.get_cell("cell-0")
        assert cell is not None
        await tx.put_cell(cell.model_copy(update={"generation": 2, "accelerator_type": "v6e-8"}))

    cost = await service.workload_cost(PRINCIPAL, "wl-1")

    assert cost.shape is None
    assert cost.capacity_class is None
    assert cost.priced


@pytest.mark.asyncio
async def test_positive_settlement_does_not_invent_a_missing_legacy_rate() -> None:
    service, repository = await _plane(2.0, (0.0,))
    async with repository.transaction() as tx:
        lease = await tx.get_lease("lease-0")
        assert lease is not None
        await tx.put_lease(lease.model_copy(update={"settled_cost": 8.0}))

    cost = await service.workload_cost(PRINCIPAL, "wl-1")

    assert not cost.priced
    assert cost.settled == 8.0
