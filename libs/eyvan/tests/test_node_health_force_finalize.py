# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Force-finalize when something *did* confirm the process stopped.

``force_finalize_stuck_workload`` gives the cell up. That was the only safe
answer while nobody could say whether the workload was still running on it:
returning the host would put new work beside a process nothing could kill.

Node health changes what is knowable. A cell quarantined for a matched host
fault carries the fault it matched, and those are the faults where the
accelerator runtime is dead -- so the process that needed it is not still
running. That cell is reclaimed instead of surrendered.

The decision is per cell and the record says which happened to each one. A gang
where one rank's TPU died and the others simply stopped answering is exactly the
case an operator must be able to read: two hosts given up because nobody could
account for them, one left on the replacement path because something could.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
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
from eyvan.server.service.control_plane import STUCK_CANCEL_AFTER_S
from eyvan.worker.runtime.node_health import (
    HOST_FAULT_REASON,
    HostFault,
    HostFaultCode,
    format_host_fault_marker,
)

NOW = utcnow()
FAULT = HostFault(
    code=HostFaultCode.TPU_SYSTEM_INIT_FAILED,
    evidence="RuntimeError: Failed to initialize TPU system: 4 of 8 chips did not respond",
)


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _cell(cell_id: str, *, fault_for: tuple[str, int] | None = None) -> CellRecord:
    """One cell, optionally carrying the host fault node health recorded for it."""

    quarantine: dict = {}
    if fault_for is not None:
        workload_id, generation = fault_for
        quarantine = {
            "state": CellState.UNAVAILABLE,
            "provider_phase": CapacityStatus.NEEDS_RECREATE,
            "unavailable_reason": HOST_FAULT_REASON,
            "unavailable_detail": format_host_fault_marker(
                FAULT,
                workload_id=workload_id,
                generation=generation,
                observed_at=NOW - timedelta(minutes=5),
            ),
        }
    return CellRecord(
        id=cell_id,
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=32, tpu=8),
        allocatable=ResourceVector(cpu=8, memory_gib=32, tpu=8),
        accelerator_type="v5p-16",
        endpoint="http://agent:8090",
    ).model_copy(update=quarantine)


def _workload(cell_ids: tuple[str, ...]) -> WorkloadRecord:
    lease_ids = tuple(f"lease-{cell_id}" for cell_id in cell_ids)
    return WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        state=WorkloadState.CANCELLING,
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        ),
        lease_id=lease_ids[0],
        lease_ids=lease_ids if len(lease_ids) > 1 else (),
        created_by="alice",
    )


def _lease(cell_id: str) -> LeaseRecord:
    return LeaseRecord(
        id=f"lease-{cell_id}",
        workload_id="wl-1",
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        state=LeaseState.ACTIVE,
        fencing_token=f"{cell_id}:1",
        expires_at=NOW + timedelta(hours=1),
    )


def _cancel() -> OutboxRecord:
    return OutboxRecord(
        id="cmd-cancel",
        topic="workload.cancel",
        aggregate_type="workload",
        aggregate_id="wl-1",
        idempotency_key="cancel-1",
        generation=1,
        state=OutboxState.FAILED,
        attempts=24,
        created_at=NOW - timedelta(seconds=STUCK_CANCEL_AFTER_S + 60),
        updated_at=NOW,
        available_at=NOW + timedelta(seconds=60),
    )


async def _service(cells: list[CellRecord]) -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    async with repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)
            await tx.put_lease(_lease(cell.id))
        await tx.put_workload(_workload(tuple(cell.id for cell in cells)))
        await tx.put_outbox(_cancel())
    return ControlPlaneService(repository=repository)


async def _cells(service: ControlPlaneService) -> dict[str, CellRecord]:
    async with service.repository.transaction() as tx:
        return {cell.id: cell for cell in await tx.list_cells()}


async def _record(service: ControlPlaneService) -> JsonObject:
    async with service.repository.transaction() as tx:
        events = await tx.list_events(org_id="acme", project_id="research", limit=50)
    forced = [event for event in events if event.action == "workload.force_finalized"]
    assert len(forced) == 1
    return forced[0].payload


def _outcomes(payload: JsonObject) -> dict[str, JsonObject]:
    entries = payload["capacity_outcomes"]
    assert isinstance(entries, list)
    return {str(entry["cell_id"]): entry for entry in entries if is_json_object(entry)}


@pytest.mark.asyncio
async def test_a_host_fault_for_this_execution_reclaims_the_cell_instead_of_draining_it() -> None:
    """The evidence answers the question the drain existed to be safe about."""

    service = await _service([_cell("cell-a", fault_for=("wl-1", 1))])

    settled, _replayed = await service.force_finalize_stuck_workload(
        _admin(), "wl-1", reason="worker 500s on stop", idempotency_key="ff-1"
    )

    assert settled.state is WorkloadState.CANCELLED
    cell = (await _cells(service))["cell-a"]
    assert cell.state is not CellState.DRAINING
    assert cell.state is CellState.UNAVAILABLE, "the cell stays on the replacement path node health put it on"
    assert cell.provider_phase is CapacityStatus.NEEDS_RECREATE


@pytest.mark.asyncio
async def test_without_that_evidence_the_cell_is_still_drained() -> None:
    """Unchanged, and it has to stay unchanged: this is the common case."""

    service = await _service([_cell("cell-a")])

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    assert (await _cells(service))["cell-a"].state is CellState.DRAINING


@pytest.mark.asyncio
async def test_the_record_says_which_of_the_two_happened_to_each_cell_and_why() -> None:
    """ "We gave up a v5p" and "we knew we did not have to" are different sentences."""

    service = await _service([_cell("cell-a", fault_for=("wl-1", 1)), _cell("cell-b")])

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="worker 500s", idempotency_key="ff-1")

    payload = await _record(service)
    assert payload["reclaimed_cells"] == ["cell-a"]
    assert payload["drained_cells"] == ["cell-b"]
    outcomes = _outcomes(payload)
    assert outcomes["cell-a"]["outcome"] == "reclaimed"
    assert outcomes["cell-a"]["host_fault"] == HostFaultCode.TPU_SYSTEM_INIT_FAILED.value
    assert outcomes["cell-a"]["evidence"] == FAULT.evidence
    assert "process that needed it is gone" in str(outcomes["cell-a"]["reason"])
    assert outcomes["cell-b"]["outcome"] == "drained"
    assert "nothing confirmed" in str(outcomes["cell-b"]["reason"])
    assert "host_fault" not in outcomes["cell-b"]


@pytest.mark.asyncio
async def test_one_ranks_proof_does_not_reclaim_a_host_nobody_could_account_for() -> None:
    """Fate sharing is a good guess about a gang and it is not evidence about a host."""

    service = await _service([_cell("cell-a", fault_for=("wl-1", 1)), _cell("cell-b")])

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    cells = await _cells(service)
    assert cells["cell-b"].state is CellState.DRAINING
    assert cells["cell-a"].state is CellState.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_fault_recorded_for_a_different_execution_is_not_proof_about_this_one() -> None:
    """A previous tenant's dead machine has since been replaced; this one has not."""

    service = await _service([_cell("cell-a", fault_for=("wl-earlier", 1))])

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    assert (await _cells(service))["cell-a"].state is CellState.DRAINING
    assert (await _record(service))["drained_cells"] == ["cell-a"]


@pytest.mark.asyncio
async def test_a_fault_recorded_for_an_earlier_generation_of_this_workload_is_not_proof() -> None:
    """A retry is a different execution on a cell that has been through recreation."""

    service = await _service([_cell("cell-a", fault_for=("wl-1", 7))])

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    assert (await _cells(service))["cell-a"].state is CellState.DRAINING


@pytest.mark.asyncio
async def test_an_unreadable_fault_record_gives_the_host_up() -> None:
    """Every doubt resolves toward the behaviour that existed before node health."""

    truncated = _cell("cell-a", fault_for=("wl-1", 1)).model_copy(
        update={"unavailable_detail": "code=tpu_system_init_failed workload=wl-1 gener"}
    )
    service = await _service([truncated])

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    assert (await _cells(service))["cell-a"].state is CellState.DRAINING


@pytest.mark.asyncio
async def test_the_lease_is_released_either_way_because_that_is_what_was_stuck() -> None:
    """The deadlock is the lease, not the cell: ``cell-delete`` refuses while it is held."""

    from eyvan.server.service.state import ACTIVE_LEASE_STATES

    for fault_for in (("wl-1", 1), None):
        service = await _service([_cell("cell-a", fault_for=fault_for)])

        await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

        async with service.repository.transaction() as tx:
            lease = await tx.get_lease("lease-cell-a")
        assert lease is not None and lease.state not in ACTIVE_LEASE_STATES


@pytest.mark.asyncio
async def test_a_reclaimed_finalize_says_so_in_the_workloads_own_failure_reason() -> None:
    """The workload record is where an operator looks first, and it used to claim
    the stop was unconfirmed even when it was not."""

    proven = await _service([_cell("cell-a", fault_for=("wl-1", 1))])
    unproven = await _service([_cell("cell-a")])

    settled_proven, _ = await proven.force_finalize_stuck_workload(
        _admin(), "wl-1", reason="stuck", idempotency_key="ff-1"
    )
    settled_unproven, _ = await unproven.force_finalize_stuck_workload(
        _admin(), "wl-1", reason="stuck", idempotency_key="ff-1"
    )

    assert settled_proven.failure_reason == (
        "force-finalized after a host fault accounted for every cell it held: stuck"
    )
    assert settled_unproven.failure_reason == "force-finalized after an unconfirmed stop: stuck"


@pytest.mark.asyncio
async def test_a_partly_proven_gang_still_reports_the_unconfirmed_stop() -> None:
    """One accounted-for host does not make the whole finalize accounted for."""

    service = await _service([_cell("cell-a", fault_for=("wl-1", 1)), _cell("cell-b")])

    settled, _replayed = await service.force_finalize_stuck_workload(
        _admin(), "wl-1", reason="stuck", idempotency_key="ff-1"
    )

    assert settled.failure_reason == "force-finalized after an unconfirmed stop: stuck"


@pytest.mark.asyncio
async def test_the_timestamp_in_a_marker_is_read_as_an_aware_instant() -> None:
    """A naive parse would make every proof unreadable in a non-UTC deployment."""

    marker = format_host_fault_marker(
        FAULT, workload_id="wl-1", generation=1, observed_at=NOW.astimezone(UTC) - timedelta(days=1)
    )
    service = await _service([_cell("cell-a", fault_for=("wl-1", 1))])
    async with service.repository.transaction() as tx:
        cell = await tx.get_cell("cell-a")
        assert cell is not None
        await tx.put_cell(cell.model_copy(update={"unavailable_detail": marker}))

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    assert (await _cells(service))["cell-a"].state is CellState.UNAVAILABLE


@pytest.mark.asyncio
async def test_the_proof_this_reads_is_the_one_the_controller_actually_writes() -> None:
    """End to end, because a marker only a test writes proves nothing about the pair.

    Node health quarantines the host from the attempt's own output; the operator
    then reaches for the break-glass while the cell is still in that state, which
    is exactly when people reach for it. What it finds is the record the
    controller wrote, and it reclaims on that.
    """

    from test_node_health_replacement import TPU_INIT, FailingAgent, _attempt
    from test_node_health_replacement import _cell as _pool_cell

    from eyvan.worker.runtime.node_health import HostReplacementBudget

    controller = await _attempt(
        FailingAgent(TPU_INIT),
        [_pool_cell("cell-a")],
        budget=HostReplacementBudget(max_replacements=1),
    )
    await controller.observe_workloads_once()
    async with controller.repository.transaction() as tx:
        workload = await tx.get_workload("wl-1")
        assert workload is not None
        await tx.put_workload(workload.model_copy(update={"state": WorkloadState.CANCELLING}))
        await tx.put_outbox(_cancel())

    settled, _replayed = await controller.service.force_finalize_stuck_workload(
        _admin(), "wl-1", reason="worker 500s on stop", idempotency_key="ff-1"
    )

    assert settled.state is WorkloadState.CANCELLED
    async with controller.repository.transaction() as tx:
        cell = await tx.get_cell("cell-a")
        events = await tx.list_events(org_id="acme", project_id="research", limit=50)
    assert cell is not None and cell.state is not CellState.DRAINING
    forced = next(event for event in events if event.action == "workload.force_finalized")
    assert forced.payload["reclaimed_cells"] == ["cell-a"]
    assert forced.payload["drained_cells"] == []
