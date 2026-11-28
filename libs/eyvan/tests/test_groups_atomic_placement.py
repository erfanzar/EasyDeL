# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Heterogeneous groups are placed whole or not at all.

The feasibility gate answers "could all of these be placed" and then hands the
members to the ordinary scheduler, which places them one at a time. Between the
answer and the leases, another workload can take a cell -- so the TPU members
lease, the GPU member does not, and half a job holds accelerators for hours.

This asserts the commit that closes that gap: the decision and every member's
lease live in one transaction, so either every member holds a fenced lease or
nothing was written at all. The partial case is the interesting one, and it is
asserted on persisted rows -- no lease, no cell claimed, every member still
exactly as it was.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.base import TransientRepositoryError
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.groups import GroupRole
from eyvan.server.service.scheduling import CoreSchedulingService


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(
            RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),
            RoleGrant(role=Role.FLEET_ADMIN),
        ),
    )


def _template(*, cpu: float = 1.0, tpu: int = 0, gpu: int = 0, name: str = "member") -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(kind="command", command="python run.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=cpu, memory_gib=1.0, tpu=tpu, gpu=gpu)),
        capacity_class=CapacityClass.ON_DEMAND,
    )


def _cell(cell_id: str, *, cpu: float = 8.0, tpu: int = 0, gpu: int = 0) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="manual",
        capacity_backend="manual",
        region="us-east5",
        state=CellState.READY,
        capacity=ResourceLimits(cpu=cpu, memory_gib=64.0, tpu=tpu, gpu=gpu),
        allocatable=ResourceLimits(cpu=cpu, memory_gib=64.0, tpu=tpu, gpu=gpu),
        unit_prices={"cpu": 0.1},
        capacity_class=CapacityClass.ON_DEMAND,
    )


_ROLES = (
    GroupRole(name="trainer", template=_template(tpu=4, name="trainer"), count=2),
    GroupRole(name="reward", template=_template(gpu=2, name="reward"), count=1),
)


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository, scheduler=CoreSchedulingService())


async def _put_cells(service: ControlPlaneService, *cells: CellRecord) -> None:
    async with service.repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)


def _controller(service: ControlPlaneService, *, atomic: bool = True) -> EyvanController:
    return EyvanController(
        service.repository,
        service,
        settings=EyvanSettings(atomic_group_placement=atomic),
    )


async def _submit(service: ControlPlaneService):
    return await service.submit_group(
        _principal(),
        org_id="acme",
        project_id="research",
        roles=_ROLES,
        idempotency_key="group-1",
    )


async def test_a_partially_placeable_group_places_nothing_and_holds_nothing() -> None:
    """Two trainers fit; the reward member does not. A per-member loop would lease two."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8))
    _parent, members = await _submit(service)

    placed = await _controller(service).place_admitted_groups_once()

    assert placed == 0
    async with service.repository.transaction() as tx:
        assert await tx.list_leases() == [], "not one accelerator was stranded holding half a job"
        stored = [await tx.get_workload(member.id) for member in members]
        outbox = await tx.list_outbox(limit=100)
    assert all(item is not None and item.state is WorkloadState.PENDING for item in stored)
    assert all(item is not None and not item.schedulable for item in stored)
    assert [command for command in outbox if command.topic == "workload.dispatch"] == []


async def test_a_placeable_group_is_leased_whole_with_a_fence_and_a_dispatch_each() -> None:
    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    _parent, members = await _submit(service)

    placed = await _controller(service).place_admitted_groups_once()

    assert placed == 3
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
        leases = await tx.list_leases()
        dispatches = [command for command in await tx.list_outbox(limit=100) if command.topic == "workload.dispatch"]
    assert all(item is not None and item.state is WorkloadState.LEASED for item in stored)
    assert {lease.workload_id for lease in leases} == {member.id for member in members}
    assert len({lease.cell_id for lease in leases}) == 3, "members compete for the same cells in one pass"
    assert all(lease.fencing_token for lease in leases)
    assert all(lease.cell_generation >= 1 and lease.generation == 1 for lease in leases), (
        "each lease names the cell generation it was decided against"
    )
    assert {command.aggregate_id for command in dispatches} == {member.id for member in members}


async def test_a_leased_group_is_never_a_scheduling_candidate_as_well() -> None:
    """Members stay gated, so no second path can double-place what this leased."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    _parent, members = await _submit(service)
    controller = _controller(service)
    await controller.place_admitted_groups_once()

    result = await controller.reconcile_once()

    assert {member.id for member in members}.isdisjoint(set(result.considered_order))
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
        leases = await tx.list_leases()
    assert all(item is not None and not item.schedulable for item in stored)
    assert len(leases) == 3, "no member acquired a second lease from the ordinary pass"


async def test_the_feasibility_gate_does_not_re_release_an_already_placed_group() -> None:
    """Two admission paths must not both act on one group."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    _parent, _members = await _submit(service)
    controller = _controller(service)
    await controller.place_admitted_groups_once()

    assert await controller.release_admitted_groups_once() == 0


async def test_placing_twice_leases_nothing_the_second_time() -> None:
    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    await _submit(service)
    controller = _controller(service)
    await controller.place_admitted_groups_once()

    assert await controller.place_admitted_groups_once() == 0
    async with service.repository.transaction() as tx:
        assert len(await tx.list_leases()) == 3


async def test_atomic_placement_is_off_until_an_operator_turns_it_on() -> None:
    """The shipped behaviour is the feasibility gate; this changes who places a group."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    _parent, members = await _submit(service)

    placed = await _controller(service, atomic=False).place_admitted_groups_once()

    assert placed == 0
    async with service.repository.transaction() as tx:
        assert await tx.list_leases() == []
        stored = [await tx.get_workload(member.id) for member in members]
    assert all(item is not None and item.state is WorkloadState.PENDING for item in stored)


async def test_a_group_becomes_placeable_when_the_missing_capacity_appears() -> None:
    """Refusing to place a group partially must not make it unplaceable forever."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8))
    _parent, members = await _submit(service)
    controller = _controller(service)
    assert await controller.place_admitted_groups_once() == 0

    await _put_cells(service, _cell("cell-gpu", gpu=4))

    assert await controller.place_admitted_groups_once() == 3
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
    assert all(item is not None and item.state is WorkloadState.LEASED for item in stored)


async def test_a_group_with_a_member_already_leased_is_left_to_the_path_that_leased_it() -> None:
    """Half-placed by something else is not a group this call may commit as a whole."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    parent, members = await _submit(service)
    async with service.repository.transaction() as tx:
        stored = await tx.get_workload(members[0].id)
        assert stored is not None
        await tx.put_workload(stored.model_copy(update={"schedulable": True}))

    placed = await service.place_group_atomically(_principal(), coordinator_id=parent.id, lease_ttl_s=120)

    assert placed == ()
    async with service.repository.transaction() as tx:
        assert await tx.list_leases() == []


async def test_a_cell_reserved_for_another_workload_is_not_taken_by_a_group() -> None:
    """Capacity provisioned for one job is not what makes another job whole."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    _parent, members = await _submit(service)
    async with service.repository.transaction() as tx:
        reserved = await tx.get_cell("cell-tpu-b")
        assert reserved is not None
        await tx.put_cell(reserved.model_copy(update={"autoscale_request_workload_id": "wl-someone-else"}))

    placed = await _controller(service).place_admitted_groups_once()

    assert placed == 0
    async with service.repository.transaction() as tx:
        assert await tx.list_leases() == []
        stored = [await tx.get_workload(member.id) for member in members]
        still_reserved = await tx.get_cell("cell-tpu-b")
    assert all(item is not None and item.state is WorkloadState.PENDING for item in stored)
    assert still_reserved is not None
    assert still_reserved.autoscale_request_workload_id == "wl-someone-else"


async def test_an_interrupted_commit_leaves_no_member_leased() -> None:
    """All or nothing is a property of the transaction, not of the happy path.

    The write is interrupted after the first member's lease, which is precisely
    the moment a group could be left half-holding accelerators. What must survive
    is the durable state: no lease, no dispatch, every member still pending.
    """

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _cell("cell-gpu", gpu=4))
    parent, members = await _submit(service)

    class _FailsAfterTheFirstLease:
        """One transaction that stops persisting leases partway through."""

        def __init__(self, inner) -> None:
            self._inner = inner
            self._written = 0

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        async def put_lease(self, lease):
            self._written += 1
            if self._written > 1:
                raise TransientRepositoryError("connection lost while committing the group")
            return await self._inner.put_lease(lease)

    class _InterruptingRepository:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def transaction(self):
            @asynccontextmanager
            async def _wrapped():
                async with self._inner.transaction() as session:
                    yield _FailsAfterTheFirstLease(session)

            return _wrapped()

    interrupted = ControlPlaneService(
        repository=_InterruptingRepository(service.repository),  # type: ignore[arg-type]
        scheduler=CoreSchedulingService(),
    )

    with pytest.raises(TransientRepositoryError):
        await interrupted.place_group_atomically(_principal(), coordinator_id=parent.id, lease_ttl_s=120)

    async with service.repository.transaction() as tx:
        assert await tx.list_leases() == []
        stored = [await tx.get_workload(member.id) for member in members]
        dispatches = [command for command in await tx.list_outbox(limit=100) if command.topic == "workload.dispatch"]
    assert all(item is not None and item.state is WorkloadState.PENDING for item in stored)
    assert all(item is not None and not item.schedulable for item in stored)
    assert dispatches == []
