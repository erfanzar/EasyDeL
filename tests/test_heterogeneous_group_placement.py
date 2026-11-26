# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One job spanning three kinds of hardware admits whole or not at all.

A heterogeneous run is one coordinator on a CPU cell, TPU slices doing the
training, and GPU cells serving a reward model. Submitted as independent
workloads the TPU members lease immediately and burn accelerator hours for as
long as the GPU member stays unplaceable -- and "unplaceable" here is not exotic,
it is a GPU cell that is one device short.

The atomic group commit already exists and is deliberately reused rather than
duplicated for GPUs: a second placement path for one kind of accelerator is how
two components come to believe they own the same cell. What was never covered is
that the commit holds when the members do not share a resource dimension, which
is the case where a per-member loop looks correct right up until the last member
fails.

**No claim here is hardware-qualified.** The GPU members are cells with GPU
resource vectors; nothing in this file executes on a GPU, and nothing asserts
that anything would.
"""

from __future__ import annotations

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


#: A coordinator, a trainer, and a reward server: one role per resource
#: dimension, which is the arrangement no single-kind gang can describe.
_ROLES = (
    GroupRole(name="coordinator", template=_template(cpu=2.0, name="coordinator"), count=1),
    GroupRole(name="trainer", template=_template(tpu=4, name="trainer"), count=2),
    GroupRole(name="reward", template=_template(gpu=8, name="reward"), count=1),
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
    return EyvanController(service.repository, service, settings=EyvanSettings(atomic_group_placement=atomic))


async def _submit(service: ControlPlaneService):
    return await service.submit_group(
        _principal(),
        org_id="acme",
        project_id="research",
        roles=_ROLES,
        idempotency_key="hetero-1",
    )


def _multi_gpu_cell() -> CellRecord:
    """A cell shaped like the catalog's eight-device GPU host."""

    return _cell("cell-gpu-8", cpu=208.0, gpu=8)


def _single_gpu_cell() -> CellRecord:
    """A cell shaped like the catalog's one-device GPU host."""

    return _cell("cell-gpu-1", cpu=12.0, gpu=1)


async def test_a_three_kind_group_leases_every_member_or_none_of_them() -> None:
    """CPU, TPU and GPU in one job, committed together."""

    service = await _service()
    await _put_cells(
        service,
        _cell("cell-cpu"),
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _multi_gpu_cell(),
    )
    _parent, members = await _submit(service)

    placed = await _controller(service).place_admitted_groups_once()

    assert placed == 4
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
        leases = await tx.list_leases()
        dispatches = [command for command in await tx.list_outbox(limit=100) if command.topic == "workload.dispatch"]
    assert all(item is not None and item.state is WorkloadState.LEASED for item in stored)
    assert {lease.workload_id for lease in leases} == {member.id for member in members}
    assert all(lease.fencing_token for lease in leases)
    assert {command.aggregate_id for command in dispatches} == {member.id for member in members}
    # Every member landed somewhere that can actually serve its dimension. A
    # group whose GPU member leased a TPU cell would satisfy the count above and
    # still be a run that cannot start.
    placement = {lease.workload_id: lease.cell_id for lease in leases}
    by_name = {member.spec.name: member for member in members}
    assert placement[by_name["group-reward-0"].id] == "cell-gpu-8"
    assert placement[by_name["group-trainer-0"].id].startswith("cell-tpu-")
    assert placement[by_name["group-trainer-1"].id].startswith("cell-tpu-")


async def test_a_gpu_cell_one_device_short_strands_no_tpu_slice() -> None:
    """The failure the atomic commit exists for, in its cross-kind form.

    The CPU and both TPU members are individually placeable. A loop that leased
    as it went would commit three members and then discover the GPU member does
    not fit -- leaving two TPU slices holding accelerators for a run that cannot
    start.
    """

    service = await _service()
    await _put_cells(
        service,
        _cell("cell-cpu"),
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _cell("cell-gpu-short", cpu=208.0, gpu=7),
    )
    _parent, members = await _submit(service)

    placed = await _controller(service).place_admitted_groups_once()

    assert placed == 0
    async with service.repository.transaction() as tx:
        leases = await tx.list_leases()
        stored = [await tx.get_workload(member.id) for member in members]
        dispatches = [command for command in await tx.list_outbox(limit=100) if command.topic == "workload.dispatch"]
    assert leases == [], "not one TPU slice was stranded holding a job that cannot run"
    assert all(item is not None and item.state is WorkloadState.PENDING for item in stored)
    assert all(item is not None and not item.schedulable for item in stored)
    assert dispatches == []


async def test_the_missing_kind_can_be_any_of_the_three() -> None:
    """Whichever dimension is short, the answer is the same: nothing is held."""

    for absent in ("cpu", "tpu", "gpu"):
        service = await _service()
        cells = [_cell("cell-cpu"), _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8), _multi_gpu_cell()]
        if absent == "cpu":
            cells = [cell for cell in cells if cell.id != "cell-cpu"]
        elif absent == "tpu":
            cells = [cell for cell in cells if cell.id != "cell-tpu-b"]
        else:
            cells = [cell for cell in cells if cell.id != "cell-gpu-8"]
        await _put_cells(service, *cells)
        await _submit(service)

        placed = await _controller(service).place_admitted_groups_once()

        assert placed == 0, absent
        async with service.repository.transaction() as tx:
            assert await tx.list_leases() == [], absent


async def test_a_group_stranded_on_gpu_capacity_admits_when_that_capacity_appears() -> None:
    """Refusing a partial placement must not make a group unplaceable forever."""

    service = await _service()
    await _put_cells(service, _cell("cell-cpu"), _cell("cell-tpu-a", tpu=8), _cell("cell-tpu-b", tpu=8))
    _parent, members = await _submit(service)
    controller = _controller(service)
    assert await controller.place_admitted_groups_once() == 0

    await _put_cells(service, _multi_gpu_cell())

    assert await controller.place_admitted_groups_once() == 4
    async with service.repository.transaction() as tx:
        stored = [await tx.get_workload(member.id) for member in members]
    assert all(item is not None and item.state is WorkloadState.LEASED for item in stored)


async def test_one_multi_gpu_host_admits_a_group_that_eight_single_gpu_hosts_cannot() -> None:
    """Single-GPU and multi-GPU hosts differ exactly where a placement should differ.

    A member asking for eight devices needs them on one cell: Eyvan leases a
    member to a cell, and eight one-device cells are eight cells. This is the
    difference an operator has to be able to predict, so it is asserted rather
    than left to a scheduler detail.
    """

    scattered = await _service()
    await _put_cells(
        scattered,
        _cell("cell-cpu"),
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        *[_cell(f"cell-gpu-{index}", cpu=12.0, gpu=1) for index in range(8)],
    )
    await _submit(scattered)

    assert await _controller(scattered).place_admitted_groups_once() == 0

    consolidated = await _service()
    await _put_cells(
        consolidated,
        _cell("cell-cpu"),
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _multi_gpu_cell(),
    )
    await _submit(consolidated)

    assert await _controller(consolidated).place_admitted_groups_once() == 4


async def test_a_single_device_gpu_member_is_placed_on_a_single_device_host() -> None:
    """The one-GPU host is a first-class host, not a shape that only nearly works."""

    service = await _service()
    roles = (
        GroupRole(name="coordinator", template=_template(cpu=2.0, name="coordinator"), count=1),
        GroupRole(name="reward", template=_template(cpu=2.0, gpu=1, name="reward"), count=1),
    )
    await _put_cells(service, _cell("cell-cpu"), _single_gpu_cell())
    _parent, members = await service.submit_group(
        _principal(),
        org_id="acme",
        project_id="research",
        roles=roles,
        idempotency_key="hetero-single",
    )

    assert await _controller(service).place_admitted_groups_once() == 2
    async with service.repository.transaction() as tx:
        leases = {lease.workload_id: lease.cell_id for lease in await tx.list_leases()}
        stored = {member.id: await tx.get_workload(member.id) for member in members}
    reward = next(member for member in members if member.spec.name.endswith("reward-0"))
    assert leases[reward.id] == "cell-gpu-1"
    assert all(item is not None and item.state is WorkloadState.LEASED for item in stored.values())


async def test_atomic_cross_kind_placement_stays_off_until_an_operator_turns_it_on() -> None:
    """Adding GPU coverage must not turn the feature on for deployments that never asked."""

    service = await _service()
    await _put_cells(
        service,
        _cell("cell-cpu"),
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _multi_gpu_cell(),
    )
    _parent, members = await _submit(service)

    assert await _controller(service, atomic=False).place_admitted_groups_once() == 0
    async with service.repository.transaction() as tx:
        assert await tx.list_leases() == []
        stored = [await tx.get_workload(member.id) for member in members]
    assert all(item is not None and item.state is WorkloadState.PENDING for item in stored)


async def test_a_placed_cross_kind_group_is_not_placed_a_second_time() -> None:
    """One group, one commit: a second pass must find nothing left to lease."""

    service = await _service()
    await _put_cells(
        service,
        _cell("cell-cpu"),
        _cell("cell-tpu-a", tpu=8),
        _cell("cell-tpu-b", tpu=8),
        _multi_gpu_cell(),
    )
    await _submit(service)
    controller = _controller(service)
    assert await controller.place_admitted_groups_once() == 4

    assert await controller.place_admitted_groups_once() == 0
    async with service.repository.transaction() as tx:
        assert len(await tx.list_leases()) == 4


def test_a_single_kind_run_is_still_refused_as_a_group() -> None:
    """The gate costs something every cycle; a batch must not be made to pay it."""

    from eyvan.server.service.groups import GroupError, validate

    with pytest.raises(GroupError, match="a single-shape run is a batch"):
        validate((GroupRole(name="reward", template=_template(gpu=8, name="reward"), count=4),))
