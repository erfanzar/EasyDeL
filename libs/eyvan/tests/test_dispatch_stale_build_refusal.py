# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A permanent dispatch refusal the plane cannot fix settles the workload.

``_refuse_stale_cell_builds`` now raises a refusal the plane answers by
recycling the cell (see ``test_stale_build_cells_recycle_themselves.py``);
this file pins the other half of the family contract. A refusal raised as the
base class -- the shape of the rank-width and rendezvous refusions, whose
causes retrying and recycling both leave in place -- must settle rather than
retry: raised outside the handler that settles the family, it falls to the
outbox's bare retry handler, the command fails and is retried forever, the
workload stays LEASED, and its PENDING lease holds the cell -- which is how a
permanent error blocks ``eyvan down``.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_controller import FakeAgent

from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.controller.workload_commands import PermanentDispatchRefusal
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service import ControlPlaneService

WORKLOAD_ID = "wl-1"
LEASE_ID = "lease-1"
COMMAND_ID = "cmd-1"


class _UnfixablyRefusing(EyvanController):
    """A controller whose stale-build hook refuses without a fix attached."""

    async def _refuse_stale_cell_builds(self, cells) -> None:
        raise PermanentDispatchRefusal("cell 'cell-a' runs a build this plane cannot recycle")

    async def _prepare_single_dispatch(self, command, *, recovered_cells):
        async with self.repository.transaction() as tx:
            workload = await tx.get_workload(WORKLOAD_ID)
            lease = await tx.get_lease(LEASE_ID)
            cell = await tx.get_cell("cell-a")
        return workload, lease, cell, object(), object()


async def _seeded_command(controller: EyvanController) -> OutboxRecord:
    command = OutboxRecord(
        id=COMMAND_ID,
        topic="workload.dispatch",
        aggregate_type="workload",
        aggregate_id=WORKLOAD_ID,
        idempotency_key="dispatch-1",
        generation=1,
        payload={"workload_id": WORKLOAD_ID, "lease_id": LEASE_ID, "cell_id": "cell-a"},
        state=OutboxState.PROCESSING,
        claimed_by=controller.worker_id,
        claimed_until=utcnow() + timedelta(seconds=60),
    )
    async with controller.repository.transaction() as tx:
        await tx.put_workload(
            WorkloadRecord(
                id=WORKLOAD_ID,
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
                state=WorkloadState.LEASED,
                lease_id=LEASE_ID,
                lease_ids=(LEASE_ID,),
                created_by="alice",
            )
        )
        await tx.put_lease(
            LeaseRecord(
                id=LEASE_ID,
                workload_id=WORKLOAD_ID,
                cell_id="cell-a",
                org_id="acme",
                project_id="research",
                state=LeaseState.PENDING,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
                fencing_token=f"lease:{LEASE_ID}:1",
                expires_at=utcnow() + timedelta(hours=1),
            )
        )
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8, memory_gib=64),
                allocatable=ResourceVector(cpu=8, memory_gib=64),
            )
        )
        await tx.put_outbox(command)
    return command


@pytest.mark.asyncio
async def test_an_unfixable_refusal_settles_the_workload_instead_of_retrying() -> None:
    repository = InMemoryRepository()
    await repository.open()
    agent = FakeAgent()
    controller = _UnfixablyRefusing(
        repository,
        ControlPlaneService(repository),
        settings=EyvanSettings(),
        agent_factory=lambda _cell: agent,
    )
    command = await _seeded_command(controller)

    await controller._dispatch_workload(command)

    async with repository.transaction() as tx:
        workload = await tx.get_workload(WORKLOAD_ID)
        lease = await tx.get_lease(LEASE_ID)
    assert workload is not None
    assert workload.state is WorkloadState.FAILED, "a permanent refusal must not be retried"
    assert workload.failure_reason is not None and "cannot recycle" in workload.failure_reason
    assert lease is not None and lease.state is LeaseState.RELEASED, "the settled attempt frees its cell"
    assert agent.submissions == []
