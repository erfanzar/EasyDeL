# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capacity requested for one workload must not outlive its need for it.

The prerequisite for racing several zones at once. A cell requested on a
workload's behalf is reserved for it, and the reservation is what stops another
workload taking capacity somebody is still waiting on. The same reservation, if
it outlives the wait, is a spot slice that is granted, billed, and reclaimable
by nothing -- and racing turns that from an edge case into the normal outcome,
because every race but one produces a loser.

Two directions, and both have to hold before anything speculative may ship:
a reservation must survive while its workload is still waiting, and must not
survive once it is not.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService

ORG = "acme"
PROJECT = "research"


def _cell(cell_id: str, *, zone: str, reserved_for: str | None) -> CellRecord:
    resources = ResourceVector(cpu=0, memory_gib=0, tpu=8)
    return CellRecord(
        id=cell_id,
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region=zone.rsplit("-", 1)[0],
        zone=zone,
        state=CellState.READY,
        capacity=resources,
        allocatable=resources,
        accelerator_type="v6e-8",
        endpoint="http://agent:8090",
        pool_id=f"pool-{zone}",
        autoscale_request_workload_id=reserved_for,
    )


async def _controller(*, workload_state: WorkloadState, leased_cell: str | None = None) -> EyvanController:
    repository = InMemoryRepository()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
    )
    async with repository.transaction() as transaction:
        for zone in ("us-east5-a", "us-east5-b"):
            await transaction.put_cell(_cell(f"cell-{zone}", zone=zone, reserved_for="wl-1"))
        lease_ids: list[str] = []
        if leased_cell is not None:
            lease_ids.append("lease-1")
            await transaction.put_lease(
                LeaseRecord(
                    id="lease-1",
                    workload_id="wl-1",
                    cell_id=leased_cell,
                    org_id=ORG,
                    project_id=PROJECT,
                    state=LeaseState.ACTIVE,
                    resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                    generation=1,
                    fencing_token="token-1",
                    expires_at=utcnow() + timedelta(hours=24),
                )
            )
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id=ORG,
                project_id=PROJECT,
                spec=WorkloadDefinition(name="racer", execution=CommandExecution(command="python train.py")),
                state=workload_state,
                lease_id=lease_ids[0] if lease_ids else None,
                lease_ids=tuple(lease_ids),
                created_by="alice",
            )
        )
    return controller


async def _reservations(controller: EyvanController) -> dict[str, str | None]:
    async with controller.repository.transaction() as transaction:
        cells = await transaction.list_cells()
    return {cell.id: cell.autoscale_request_workload_id for cell in cells}


async def _sweep(controller: EyvanController) -> dict[str, str | None]:
    async with controller.repository.transaction() as transaction:
        cells = await transaction.list_cells()
    await controller._release_inactive_autoscale_request_ownership(cells)
    return await _reservations(controller)


# --------------------------------------------------------------------------- kept while waiting


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [WorkloadState.PENDING, WorkloadState.ADMITTED])
async def test_capacity_stays_reserved_while_its_workload_is_still_waiting(waiting: WorkloadState) -> None:
    """Releasing it early gives away the capacity somebody is queued for.

    Both states are waiting. PENDING has not been admitted yet; ADMITTED has,
    and is waiting for a lease -- which is exactly the window a speculative
    request lives in, so treating it as "no longer waiting" would hand every
    raced slice to another workload before the race finished.
    """

    controller = await _controller(workload_state=waiting)

    assert await _sweep(controller) == {"cell-us-east5-a": "wl-1", "cell-us-east5-b": "wl-1"}


# --------------------------------------------------------------------------- released once placed


@pytest.mark.asyncio
async def test_the_losing_zone_is_released_once_the_workload_is_leased_elsewhere() -> None:
    """The whole prerequisite. A race produces one winner and N-1 losers.

    A loser that keeps its reservation is a spot slice that is granted, billed,
    and reclaimable by nothing -- so racing must not ship until this holds.
    """

    controller = await _controller(workload_state=WorkloadState.LEASED, leased_cell="cell-us-east5-a")

    remaining = await _sweep(controller)

    assert remaining["cell-us-east5-b"] is None, "the losing zone must return to shared supply"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finished",
    [WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.CANCELLED],
)
async def test_capacity_is_released_when_its_workload_will_never_need_it(finished: WorkloadState) -> None:
    """A cancelled or finished job leaves nothing to hold capacity for."""

    controller = await _controller(workload_state=finished)

    assert set((await _sweep(controller)).values()) == {None}


@pytest.mark.asyncio
async def test_capacity_reserved_for_a_workload_that_no_longer_exists_is_released() -> None:
    """A record that is gone would otherwise hold its slices for the fleet's lifetime."""

    controller = await _controller(workload_state=WorkloadState.PENDING)
    async with controller.repository.transaction() as transaction:
        for cell in await transaction.list_cells():
            await transaction.put_cell(cell.model_copy(update={"autoscale_request_workload_id": "wl-vanished"}))

    assert set((await _sweep(controller)).values()) == {None}


# --------------------------------------------------------------------------- and then drainable


@pytest.mark.asyncio
async def test_a_released_cell_is_no_longer_protected_from_reclamation() -> None:
    """Releasing the reservation is only half of it; the planner has to act.

    The planner protects every cell carrying a reservation, unconditionally. A
    loser therefore becomes reclaimable only because the reservation is gone,
    and this asserts the two halves actually meet.
    """

    from eyvan.capacity.planning import PoolDemand, PoolScaleIntent

    controller = await _controller(workload_state=WorkloadState.LEASED, leased_cell="cell-us-east5-a")
    await _sweep(controller)
    async with controller.repository.transaction() as transaction:
        cells = await transaction.list_cells()
    loser = next(cell for cell in cells if cell.id == "cell-us-east5-b")

    intent = PoolScaleIntent(
        pool=None,  # type: ignore[arg-type]
        live=(loser,),
        matching_supply=(loser,),
        demand=PoolDemand(real_cell_ids=frozenset(), virtual_worker_counts=(), virtual_workload_ids=()),
        target_worker_units=0,
        requested_worker_counts=(),
        requested_exact_count=0,
        demanded_virtual_worker_counts=(),
    )
    protected = {cell.id for cell in intent.matching_supply if cell.autoscale_request_workload_id is not None}

    assert protected == set(), "a released cell must not still read as reserved"
