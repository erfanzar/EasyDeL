# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Heterogeneous groups: differently-shaped workloads that admit together.

A training run is often one CPU coordinator, four TPU slices, and two GPU cells
serving a reward model. Submitted as independent workloads, the TPU slices lease
immediately and burn accelerator hours for as long as the GPU cells stay
unplaceable. Atomic admission is the one thing a group buys over siblings, and it
is what this module asserts.

The gate is deliberately a *feasibility* check, not a reservation -- capacity can
still be taken between the check and the members' leases, because the core
scheduler has no reservation primitive. What it prevents is half a group holding
accelerators while the rest is structurally infeasible. The test for a partial
release failing is therefore about the *transaction*, not about the race.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    CoordinationKind,
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
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.groups import GroupError, GroupRole, member_names, validate
from eyvan.server.service.scheduling import CoreSchedulingService


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository, scheduler=CoreSchedulingService())


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


async def _put_cells(service: ControlPlaneService, *cells: CellRecord) -> None:
    async with service.repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)


async def _finish(service: ControlPlaneService, *outcomes: tuple[str, WorkloadState]) -> None:
    """Force members terminal.

    Written directly rather than walked through admit/lease/run: the settler's
    contract is about children *being* terminal, and staging a full lifecycle for
    each member would make the test depend on placement succeeding, which is a
    different property with its own tests.
    """

    async with service.repository.transaction() as tx:
        for workload_id, state in outcomes:
            stored = await tx.get_workload(workload_id)
            assert stored is not None
            await tx.put_workload(stored.model_copy(update={"state": state}))


def _controller(service: ControlPlaneService) -> EyvanController:
    return EyvanController(repository=service.repository, service=service)


_ROLES = (
    GroupRole(name="trainer", template=_template(tpu=4, name="trainer"), count=2),
    GroupRole(name="reward", template=_template(gpu=2, name="reward"), count=1),
)


def test_a_group_must_declare_more_than_one_shape() -> None:
    """One shape is a batch, which admits independently and needs no gate."""

    with pytest.raises(GroupError, match="batch"):
        validate((GroupRole(name="only", template=_template()),))


def test_role_names_must_be_unique_and_counts_positive() -> None:
    with pytest.raises(GroupError, match="unique"):
        validate(
            (
                GroupRole(name="same", template=_template()),
                GroupRole(name="same", template=_template()),
            )
        )
    with pytest.raises(GroupError, match="positive"):
        validate(
            (
                GroupRole(name="a", template=_template(), count=0),
                GroupRole(name="b", template=_template()),
            )
        )


def test_an_oversized_group_is_refused_by_total_members_not_role_count() -> None:
    """Two roles of 5000 each is 10000 members, however few roles it declares."""

    with pytest.raises(GroupError, match="exceeds"):
        validate(
            (
                GroupRole(name="a", template=_template(), count=6),
                GroupRole(name="b", template=_template(), count=6),
            ),
            max_members=10,
        )


def test_member_names_are_deterministic_in_declaration_order() -> None:
    """A replay must derive the same keys, or it creates a second group."""

    names = member_names(_ROLES, prefix="run")

    assert names == ("run-trainer-0", "run-trainer-1", "run-reward-0")
    assert member_names(_ROLES, prefix="run") == names


@pytest.mark.asyncio
async def test_every_member_is_created_gated_and_the_coordinator_is_marked() -> None:
    service = await _service()

    parent, members = await service.submit_group(
        _principal(),
        org_id="acme",
        project_id="research",
        roles=_ROLES,
        idempotency_key="group-1",
        name_prefix="run",
    )

    assert parent.coordination_kind is CoordinationKind.GROUP
    assert parent.schedulable is False
    assert len(members) == 3
    assert all(member.schedulable is False for member in members), "no member may start before all can"
    assert [member.spec.name for member in members] == ["run-trainer-0", "run-trainer-1", "run-reward-0"]
    assert all(member.parent_workload_id == parent.id for member in members)


@pytest.mark.asyncio
async def test_a_gated_member_is_invisible_to_the_scheduler() -> None:
    service = await _service()
    await _put_cells(service, _cell("cell-tpu", tpu=8), _cell("cell-gpu", gpu=4))
    _parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )

    result = await _controller(service).reconcile_once()

    placed = {placement.workload_id for placement in result.placements}
    assert placed.isdisjoint({member.id for member in members})


@pytest.mark.asyncio
async def test_a_group_is_released_when_every_member_can_be_placed() -> None:
    service = await _service()
    await _put_cells(
        service,
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _cell("cell-gpu", gpu=4),
    )
    _parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )

    released = await _controller(service).release_admitted_groups_once()

    assert released == 3
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
    assert all(item is not None and item.schedulable for item in stored)


@pytest.mark.asyncio
async def test_a_group_stays_gated_while_one_role_has_nowhere_to_run() -> None:
    """The whole point: the TPU members must not lease while the GPU member cannot."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8))
    _parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )

    released = await _controller(service).release_admitted_groups_once()

    assert released == 0
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
    assert all(item is not None and not item.schedulable for item in stored), "not even the feasible ones"


@pytest.mark.asyncio
async def test_members_compete_for_the_same_cells_in_one_pass() -> None:
    """Evaluated separately, two members would each 'fit' the one free TPU cell."""

    service = await _service()
    await _put_cells(service, _cell("cell-tpu-a", tpu=8), _cell("cell-gpu", gpu=4))
    _parent, _members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )

    released = await _controller(service).release_admitted_groups_once()

    assert released == 0, "two trainers need two TPU cells; one cell is not enough"


@pytest.mark.asyncio
async def test_a_released_group_is_then_placed_normally() -> None:
    service = await _service()
    await _put_cells(
        service,
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _cell("cell-gpu", gpu=4),
    )
    _parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )
    await _controller(service).release_admitted_groups_once()

    result = await _controller(service).reconcile_once()

    placed = {placement.workload_id for placement in result.placements}
    assert {member.id for member in members} <= placed


@pytest.mark.asyncio
async def test_releasing_twice_releases_nothing_the_second_time() -> None:
    service = await _service()
    await _put_cells(
        service,
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _cell("cell-gpu", gpu=4),
    )
    await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )
    controller = _controller(service)
    await controller.release_admitted_groups_once()

    assert await controller.release_admitted_groups_once() == 0


@pytest.mark.asyncio
async def test_a_replayed_submission_returns_the_same_members() -> None:
    service = await _service()
    first_parent, first_members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )

    second_parent, second_members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )

    assert second_parent.id == first_parent.id
    assert sorted(item.id for item in second_members) == sorted(item.id for item in first_members)


@pytest.mark.asyncio
async def test_placeability_of_an_empty_member_set_is_false_not_vacuously_true() -> None:
    """``all()`` over nothing is True, which would release a group with no members."""

    service = await _service()

    assert await service.group_is_placeable(_principal(), member_ids=()) is False


@pytest.mark.asyncio
async def test_placeability_of_a_missing_member_is_false() -> None:
    """A pruned or lost member must not be silently treated as satisfied."""

    service = await _service()
    await _put_cells(service, _cell("cell-a"))

    assert await service.group_is_placeable(_principal(), member_ids=("wl-does-not-exist",)) is False


@pytest.mark.asyncio
async def test_a_group_coordinator_settles_from_its_members() -> None:
    """A coordinator holds no lease, so nothing but this phase can ever finish it."""

    service = await _service()
    parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )
    await _finish(service, *((member.id, WorkloadState.SUCCEEDED) for member in members))

    settled = await _controller(service).settle_coordinator_workloads_once()

    assert settled == 1
    async with service.repository.transaction() as tx:
        stored = await tx.get_workload(parent.id)
    assert stored is not None
    assert stored.state is WorkloadState.SUCCEEDED
    assert "group of 3 members" in (stored.scheduling_reason or ""), "a group is not a batch"


@pytest.mark.asyncio
async def test_one_failed_member_fails_the_whole_group() -> None:
    """A partial group is not a result, so the coordinator must not report success."""

    service = await _service()
    parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )
    await _finish(
        service,
        (members[0].id, WorkloadState.FAILED),
        *((member.id, WorkloadState.SUCCEEDED) for member in members[1:]),
    )

    await _controller(service).settle_coordinator_workloads_once()

    async with service.repository.transaction() as tx:
        stored = await tx.get_workload(parent.id)
    assert stored is not None
    assert stored.state is WorkloadState.FAILED


@pytest.mark.asyncio
async def test_cancelling_the_coordinator_cancels_every_member() -> None:
    """A group's members have no independent meaning; the existing cascade covers it."""

    service = await _service()
    parent, members = await service.submit_group(
        _principal(), org_id="acme", project_id="research", roles=_ROLES, idempotency_key="group-1"
    )

    await service.cancel_workload(_principal(), parent.id, idempotency_key="cancel-1")

    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
    assert all(item is not None and item.state in {WorkloadState.CANCELLED, WorkloadState.CANCELLING} for item in stored)
