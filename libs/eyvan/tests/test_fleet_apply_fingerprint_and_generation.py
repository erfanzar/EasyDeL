# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.capacity.planning import CapacityPlanner, PoolDemand
from eyvan.records import CapacityPoolRecord, CellState, Principal, Role, RoleGrant, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError


def _admin() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _pool(pool_id: str = "shared", fleet_name: str = "one") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name=fleet_name,
        node_type="cpu",
        provider="local",
        region="local",
        capacity={"cpu": 4, "memory_gib": 16},
        allocatable={"cpu": 4, "memory_gib": 16},
        executor="native-local",
        agent_endpoint="http://agent:8090",
        min_cells=0,
        initial_cells=1,
        max_cells=4,
        shrink_stabilization_s=300,
    )


@pytest.mark.asyncio
async def test_unchanged_apply_preserves_controller_shrink_clock_and_initialization() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    declaration = _pool()
    applied, _ = await controller.apply_fleet(_admin(), "one", [declaration], idempotency_key="apply-1")
    now = utcnow()
    initialized = applied[0].model_copy(update={"initialized_at": now - timedelta(seconds=30)})
    cell = initialized.make_cell("managed", worker_count=1).model_copy(update={"state": CellState.READY})
    async with repository.transaction() as transaction:
        await transaction.put_capacity_pool(initialized)
        await transaction.put_cell(cell)

    demand = PoolDemand(real_cell_ids=frozenset(), virtual_worker_counts=(), virtual_workload_ids=())
    plans = CapacityPlanner((initialized,), {initialized.id: [cell]}, now).plan({initialized.id: demand})
    assert plans[initialized.id].shrink_since_changed
    assert plans[initialized.id].shrink_since_update == now
    await controller._persist_shrink_clocks((initialized,), plans, now)
    before = await service.get_capacity_pool(_admin(), pool_id=initialized.id)
    assert before.demand_below_supply_since == now

    reapplied, replayed = await controller.apply_fleet(_admin(), "one", [declaration], idempotency_key="apply-2")
    assert not replayed
    assert reapplied[0] == before
    after_plans = CapacityPlanner((reapplied[0],), {initialized.id: [cell]}, now).plan({initialized.id: demand})
    assert not after_plans[initialized.id].shrink_since_changed
    assert after_plans[initialized.id].launch_worker_counts == ()
    async with repository.transaction() as transaction:
        assert await transaction.get_cell(cell.id) == cell
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_field", ["initialized_at", "demand_below_supply_since"])
async def test_fleet_apply_fingerprint_excludes_controller_clocks(runtime_field: str) -> None:
    service = ControlPlaneService(InMemoryRepository())
    declaration = _pool()
    applied, _ = await service.apply_fleet(_admin(), "one", [declaration], idempotency_key="same-key")
    with_clock = declaration.model_copy(update={runtime_field: utcnow()})
    replayed, is_replay = await service.apply_fleet(_admin(), "one", [with_clock], idempotency_key="same-key")
    assert is_replay
    assert replayed == applied


@pytest.mark.asyncio
async def test_real_configuration_change_and_drained_reactivation_still_bump_generation() -> None:
    service = ControlPlaneService(InMemoryRepository())
    declaration = _pool()
    first, _ = await service.apply_fleet(_admin(), "one", [declaration], idempotency_key="first")
    changed_declaration = declaration.model_copy(update={"max_cells": 5})
    changed, _ = await service.apply_fleet(_admin(), "one", [changed_declaration], idempotency_key="changed")
    assert changed[0].generation == first[0].generation + 1
    assert changed[0].max_cells == 5
    drained, _ = await service.disable_fleet(_admin(), "one", idempotency_key="drain")
    restored, _ = await service.apply_fleet(_admin(), "one", [changed_declaration], idempotency_key="restore")
    assert restored[0].enabled
    assert restored[0].max_cells == 5
    assert restored[0].initial_cells == 1
    assert restored[0].generation == drained[0].generation + 1


@pytest.mark.asyncio
async def test_cross_fleet_pool_collision_rejects_entire_apply_without_resetting_owner_or_generation() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    declaration = _pool()
    await controller.apply_fleet(_admin(), "one", [declaration], idempotency_key="first")
    generation_two, _ = await controller.apply_fleet(
        _admin(), "one", [declaration.model_copy(update={"max_cells": 5})], idempotency_key="second"
    )
    before = generation_two[0]
    assert before.generation == 2
    cell = before.make_cell("managed", worker_count=1).model_copy(update={"state": CellState.READY})
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
    existing_two, _ = await controller.apply_fleet(
        _admin(), "two", [_pool("two-existing", "two")], idempotency_key="two-existing"
    )
    requests = [_pool("two-new", "two"), _pool("shared", "two")]
    with pytest.raises(ConflictError, match=r"shared.*one"):
        await controller.apply_fleet(_admin(), "two", requests, idempotency_key="collision")

    assert await service.list_capacity_pools(_admin(), fleet_name="one") == [before]
    assert await service.list_capacity_pools(_admin(), fleet_name="two") == existing_two
    async with repository.transaction() as transaction:
        assert await transaction.get_capacity_pool("shared") == before
        assert await transaction.get_capacity_pool("two-new") is None
        assert await transaction.get_cell(cell.id) == cell
        assert await transaction.get_idempotency("fleet", "fleet.apply", "collision") is None
    await controller.close()
