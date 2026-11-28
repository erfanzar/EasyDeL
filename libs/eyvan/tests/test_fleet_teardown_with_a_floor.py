# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tearing down a fleet whose pool declares a floor.

Found on hardware, and reproducible without it, which makes it a missing test
rather than a hardware-only defect.

``disable_fleet`` drains a pool by zeroing ``min_cells`` and ``max_cells``. A
pool that declared a floor also carries ``initial_cells``, and the record
requires ``min_cells <= initial_cells <= max_cells`` -- so the drained copy is
invalid, the transaction raises, and the API answers 422 naming a field the
operator never set.

The consequence is the one that matters: ``eyvan down`` reports an error, the
fleet stays enabled, and the cells keep billing. Teardown is the operation that
stops money, so it is the last one that should fail on a validation rule about
its own bookkeeping.
"""

from __future__ import annotations

import pytest

from eyvan.records import CapacityPoolRecord, Principal, ResourceVector, Role, RoleGrant
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _pool(**overrides: object) -> CapacityPoolRecord:
    defaults: dict[str, object] = {
        "id": "research/cpu",
        "fleet_name": "research",
        "node_type": "cpu",
        "provider": "local",
        "region": "local",
        "capacity": ResourceVector(cpu=4, memory_gib=16),
        "allocatable": ResourceVector(cpu=4, memory_gib=16),
        "agent_endpoint": "http://agent:8090",
        "min_cells": 1,
        "max_cells": 1,
        "initial_cells": 1,
    }
    return CapacityPoolRecord(**{**defaults, **overrides})


async def _service(pool: CapacityPoolRecord) -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(pool)
    return ControlPlaneService(repository=repository)


@pytest.mark.asyncio
async def test_a_fleet_with_a_floor_can_be_torn_down() -> None:
    """The defect, stated as the thing an operator needs to be true.

    A pool declaring `min_workers: 1` in fleet YAML gets `initial_cells: 1`, and
    every such fleet was untearable: `eyvan down` returned 422 and left the
    capacity running.
    """

    service = await _service(_pool())

    await service.disable_fleet(_admin(), fleet_name="research", idempotency_key="down-1")

    async with service.repository.transaction() as tx:
        stored = await tx.get_capacity_pool("research/cpu")
    assert stored is not None
    assert stored.enabled is False
    assert (stored.min_cells, stored.max_cells) == (0, 0)
    # The field that made the drained copy invalid. Zero rather than left alone:
    # a floor is a statement about a pool that is accepting work, and this pool
    # is not.
    assert stored.initial_cells == 0


@pytest.mark.asyncio
async def test_a_fleet_without_a_floor_still_tears_down() -> None:
    """The path that already worked must keep working.

    Scale-to-zero pools carry `initial_cells` of 0 or None and were unaffected,
    which is why this went unnoticed: the proof fleet that first hit it was the
    first one to declare a floor.
    """

    service = await _service(_pool(min_cells=0, initial_cells=None))

    await service.disable_fleet(_admin(), fleet_name="research", idempotency_key="down-1")

    async with service.repository.transaction() as tx:
        stored = await tx.get_capacity_pool("research/cpu")
    assert stored is not None and stored.enabled is False


@pytest.mark.asyncio
async def test_tearing_down_twice_is_not_an_error() -> None:
    """Down is what an operator retries when they are unsure, so it must be safe."""

    service = await _service(_pool())

    await service.disable_fleet(_admin(), fleet_name="research", idempotency_key="down-1")
    await service.disable_fleet(_admin(), fleet_name="research", idempotency_key="down-2")

    async with service.repository.transaction() as tx:
        stored = await tx.get_capacity_pool("research/cpu")
    assert stored is not None and stored.enabled is False


@pytest.mark.asyncio
async def test_a_pool_dropped_from_a_fleet_is_pruned_cleanly() -> None:
    """The same defect on the declarative path, which had no test at all.

    ``apply_fleet`` disables pools absent from the request, using the same
    zeroing as ``disable_fleet`` and carrying the same omission. Removing a
    node type from fleet YAML therefore wrote an invalid record for it -- and
    unlike ``down``, nothing in that flow tells the operator, because the apply
    itself succeeds.
    """

    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)
    keep = _pool(id="research/cpu")
    drop = _pool(id="research/gpu", node_type="gpu")

    await service.apply_fleet(_admin(), "research", [keep, drop], idempotency_key="apply-1")
    await service.apply_fleet(_admin(), "research", [keep], idempotency_key="apply-2")

    async with repository.transaction() as tx:
        pruned = await tx.get_capacity_pool("research/gpu")
    assert pruned is not None
    assert pruned.enabled is False
    assert (pruned.min_cells, pruned.max_cells, pruned.initial_cells) == (0, 0, 0)
