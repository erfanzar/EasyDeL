# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The escape from a deadlock that had none.

Observed on hardware: a native worker answered 500 to every ``/stop``, the agent
retried indefinitely, and the workload stayed CANCELLING. Its lease therefore
stayed active, so ``cell-delete`` refused -- "still owns 1 active lease(s)" --
so the cell never left DRAINING and ``eyvan down`` never completed. A v5p-16 was
pinned until the queued resource was force-deleted through gcloud, and no
``eyvan`` command resolved the control-plane state at all.
``force_finalize_preemption`` attests *preemption* victims and does not apply.

The design decision worth arguing with is that this **drains** the cell instead
of returning it. Nobody confirmed the process stopped, so it may still be
running; scheduling other work onto that host would be worse than the deadlock.
The capacity is given up deliberately.

And it is refused on anything that is not genuinely stuck, at the same threshold
I3 uses to report it. A break-glass that also works on healthy work is one that
gets used on healthy work.
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
from eyvan.server.service.errors import ConflictError

NOW = utcnow()


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _cell() -> CellRecord:
    return CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=32),
        allocatable=ResourceVector(cpu=8, memory_gib=32),
        endpoint="http://agent:8090",
    )


def _workload(state: WorkloadState = WorkloadState.CANCELLING) -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        state=state,
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        ),
        created_by="alice",
    )


def _lease() -> LeaseRecord:
    return LeaseRecord(
        id="lease-1",
        workload_id="wl-1",
        cell_id="cell-a",
        org_id="acme",
        project_id="research",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        state=LeaseState.ACTIVE,
        fencing_token="cell-a:1",
        expires_at=NOW + timedelta(hours=1),
    )


def _cancel(age_s: float) -> OutboxRecord:
    created = NOW - timedelta(seconds=age_s)
    return OutboxRecord(
        id="cmd-cancel",
        topic="workload.cancel",
        aggregate_type="workload",
        aggregate_id="wl-1",
        idempotency_key="cancel-1",
        generation=1,
        state=OutboxState.FAILED,
        attempts=24,
        created_at=created,
        updated_at=NOW,
        available_at=NOW + timedelta(seconds=60),
    )


async def _service(workload: WorkloadRecord, *, cancel_age_s: float = STUCK_CANCEL_AFTER_S + 60):
    repository = InMemoryRepository()
    await repository.open()
    async with repository.transaction() as tx:
        await tx.put_cell(_cell())
        await tx.put_workload(workload)
        await tx.put_lease(_lease())
        await tx.put_outbox(_cancel(cancel_age_s))
    return ControlPlaneService(repository=repository)


@pytest.mark.asyncio
async def test_a_stuck_cancel_is_settled_and_its_lease_released() -> None:
    """The deadlock's exit: the workload terminates and the capacity comes back."""

    service = await _service(_workload())

    settled, replayed = await service.force_finalize_stuck_workload(
        _admin(), "wl-1", reason="worker 500s on stop", idempotency_key="ff-1"
    )

    assert replayed is False
    assert settled.state is WorkloadState.CANCELLED
    async with service.repository.transaction() as tx:
        lease = await tx.get_lease("lease-1")
    from eyvan.server.service.state import ACTIVE_LEASE_STATES

    assert lease is not None and lease.state not in ACTIVE_LEASE_STATES


@pytest.mark.asyncio
async def test_the_cell_is_drained_rather_than_returned_to_service() -> None:
    """The decision that makes this safe, and the one worth arguing with.

    Nobody confirmed the process stopped. Returning the host to the scheduler
    would put new work beside a process that may still be running and that
    nothing could kill.
    """

    service = await _service(_workload())

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    async with service.repository.transaction() as tx:
        cell = await tx.get_cell("cell-a")
    assert cell is not None and cell.state is CellState.DRAINING


@pytest.mark.asyncio
async def test_the_stuck_command_stops_being_retried() -> None:
    """Left PENDING it would keep failing forever and re-report under I3."""

    service = await _service(_workload())

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    async with service.repository.transaction() as tx:
        command = await tx.get_outbox("cmd-cancel")
    assert command is not None and command.state is OutboxState.DISCARDED


@pytest.mark.asyncio
async def test_a_cancellation_still_in_flight_is_refused() -> None:
    """A break-glass that works on healthy work is one that gets used on it."""

    service = await _service(_workload(), cancel_age_s=30)

    with pytest.raises(ConflictError, match="not stuck"):
        await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="impatient", idempotency_key="ff-1")


@pytest.mark.asyncio
async def test_a_running_workload_cannot_be_force_finalized() -> None:
    """This is not a second cancel. It gives up the cell, so it needs a cancel first."""

    service = await _service(_workload(WorkloadState.RUNNING))

    with pytest.raises(ConflictError, match="only a stuck cancellation"):
        await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")


@pytest.mark.asyncio
async def test_an_already_terminal_workload_is_refused() -> None:
    service = await _service(_workload(WorkloadState.SUCCEEDED))

    with pytest.raises(ConflictError, match="already"):
        await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")


@pytest.mark.asyncio
async def test_an_empty_reason_is_refused() -> None:
    """The ledger entry is the only record of why capacity was given up."""

    from eyvan.server.service.errors import InputError

    service = await _service(_workload())

    with pytest.raises(InputError, match="reason"):
        await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="   ", idempotency_key="ff-1")


@pytest.mark.asyncio
async def test_replaying_the_same_key_does_not_settle_twice() -> None:
    """Break-glass is retried by people who are unsure it worked."""

    service = await _service(_workload())
    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="stuck", idempotency_key="ff-1")

    _settled, replayed = await service.force_finalize_stuck_workload(
        _admin(), "wl-1", reason="stuck", idempotency_key="ff-1"
    )

    assert replayed is True


@pytest.mark.asyncio
async def test_it_is_audited_with_the_reason_and_what_it_gave_up() -> None:
    """Capacity was surrendered; the ledger has to say so and why."""

    service = await _service(_workload())

    await service.force_finalize_stuck_workload(_admin(), "wl-1", reason="worker 500s", idempotency_key="ff-1")

    async with service.repository.transaction() as tx:
        events = await tx.list_events(org_id="acme", project_id="research", limit=50)
    forced = [e for e in events if e.action == "workload.force_finalized"]
    assert len(forced) == 1
    assert forced[0].payload["reason"] == "worker 500s"
    assert forced[0].payload["drained_cells"] == ["cell-a"]


@pytest.mark.asyncio
async def test_an_operator_without_fleet_scope_cannot_use_it() -> None:
    from eyvan.server.service.errors import AuthorizationError

    service = await _service(_workload())
    tenant = Principal(
        subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
    )

    with pytest.raises(AuthorizationError):
        await service.force_finalize_stuck_workload(tenant, "wl-1", reason="stuck", idempotency_key="ff-1")
