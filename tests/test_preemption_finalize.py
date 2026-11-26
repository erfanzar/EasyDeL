# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Finalizing a preemption plan, and the two duties it repeats from settlement.

Every ordinary terminal path settles through the same funnel, which archives
before releasing -- releasing a cell destroys its journal -- and requeues a
workload with no lease from the generation it lost. The preemption finalize
cannot use that funnel: it settles inside one transaction across every victim.
These tests pin the two duties as the funnel would.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
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
from eyvan.server.controller import EyvanController
from eyvan.server.controller._support import _PreemptionVictim
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

WORKLOAD_ID = "wl-1"
LEASE_ID = "lease-1"
COMMAND_ID = "pre-1"


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.FLEET_ADMIN),),
    )


def _workload() -> WorkloadRecord:
    return WorkloadRecord(
        id=WORKLOAD_ID,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        state=WorkloadState.CHECKPOINTING,
        lease_id=LEASE_ID,
        # The pod-group shape: a leader lease and the group it led. This is the
        # stored shape a stopping victim carries into finalize.
        lease_ids=(LEASE_ID,),
        preemption_id=COMMAND_ID,
        created_by="alice",
    )


def _lease(*, state: LeaseState) -> LeaseRecord:
    return LeaseRecord(
        id=LEASE_ID,
        workload_id=WORKLOAD_ID,
        cell_id="cell-a",
        org_id="acme",
        project_id="research",
        state=state,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        fencing_token=f"lease:{LEASE_ID}:1",
        revocation_reason="preemption",
        expires_at=utcnow() + timedelta(hours=1),
    )


def _victim() -> _PreemptionVictim:
    return _PreemptionVictim(
        workload_id=WORKLOAD_ID,
        lease_id=LEASE_ID,
        cell_id="cell-a",
        generation=1,
        checkpoint=False,
        grace_period_seconds=0,
        hook=None,
        output_uri=None,
        workload_state=WorkloadState.RUNNING,
        lease_state=LeaseState.ACTIVE,
    )


async def _plane(controller: EyvanController, *, outcome: str) -> OutboxRecord:
    """One stopping victim and its claimed preemption command, ready to finalize."""

    async with controller.repository.transaction() as tx:
        await tx.put_workload(_workload())
        await tx.put_lease(_lease(state=LeaseState.REVOKING))
    command = OutboxRecord(
        id=COMMAND_ID,
        topic="workload.preempt",
        aggregate_type="workload",
        aggregate_id=WORKLOAD_ID,
        idempotency_key="preempt-1",
        generation=1,
        payload={
            "target_workload_id": "wl-priority",
            "victims": [_victim().model_dump(mode="json")],
            "phase": "stopping",
            "victim_outcomes": {LEASE_ID: outcome},
        },
        state=OutboxState.PROCESSING,
        claimed_by=controller.worker_id,
        claimed_until=utcnow() + timedelta(seconds=60),
    )
    async with controller.repository.transaction() as tx:
        await tx.put_outbox(command)
    return command


def _controller(repository: InMemoryRepository) -> EyvanController:
    return EyvanController(repository, ControlPlaneService(repository), settings=EyvanSettings())


@pytest.mark.asyncio
async def test_a_preempted_group_workload_requeues_without_its_old_lease_group() -> None:
    """The released generation's leases belong to that generation.

    The sibling resume path resets ``lease_ids`` alongside ``lease_id``; a
    resume that kept the group with no leader in it would violate the
    leader-first shape the settlement paths read the leases through.
    """

    repository = InMemoryRepository()
    await repository.open()
    controller = _controller(repository)
    command = await _plane(controller, outcome="preempted")

    await controller._finalize_preemption_plan(command, {})

    async with repository.transaction() as tx:
        workload = await tx.get_workload(WORKLOAD_ID)
    assert workload is not None
    assert workload.state is WorkloadState.PENDING
    assert workload.lease_id is None
    assert workload.lease_ids == (), "a requeued attempt inherits no lease from the generation it lost"


class _ArchiveSpy(EyvanController):
    """A controller that records when the terminal archive runs, and against what."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.archived: list[str] = []
        self.lease_state_at_archive: LeaseState | None = None

    async def _archive_terminal_logs(self, workload: WorkloadRecord) -> None:
        self.archived.append(workload.id)
        async with self.repository.transaction() as tx:
            lease = await tx.get_lease(LEASE_ID)
        self.lease_state_at_archive = None if lease is None else lease.state


@pytest.mark.asyncio
async def test_a_terminal_victim_archives_before_its_lease_is_released() -> None:
    """A victim observed SUCCEEDED still dies through the settlement funnel.

    Settling releases the lease, which frees the cell for deletion and destroys
    the journal -- so the tail is copied first, while the lease still names the
    cell the journal lives on.
    """

    repository = InMemoryRepository()
    await repository.open()
    controller = _ArchiveSpy(repository, ControlPlaneService(repository), settings=EyvanSettings())
    command = await _plane(controller, outcome="succeeded")

    await controller._finalize_preemption_plan(command, {})

    assert controller.archived == [WORKLOAD_ID]
    assert controller.lease_state_at_archive is LeaseState.REVOKING, "the journal is copied before the release"
    async with repository.transaction() as tx:
        workload = await tx.get_workload(WORKLOAD_ID)
        lease = await tx.get_lease(LEASE_ID)
    assert workload is not None and workload.state is WorkloadState.SUCCEEDED
    assert lease is not None and lease.state is LeaseState.RELEASED
