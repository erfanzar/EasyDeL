# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which budget an operator's kick is charged to.

A kick is what an operator wants when one attempt is wedged on a host that is
not coming back and the job itself is fine: stop the attempt, requeue, place it
somewhere else. Cancelling ends the job, and waiting for a timeout keeps the
accelerator busy doing nothing until it fires.

The whole question is which budget pays for it, and getting it wrong is not a
cosmetic accounting error. A capacity loss is charged against ``max_retries``,
which a spot job sets to **0** precisely because it expects to be interrupted --
so a kick charged there ends the job the operator was trying to move.
"""

from __future__ import annotations

import pytest
from test_budget_settlement import _principal, _running_workload

from eyvan.records import BudgetRecord, Role, WorkloadState
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


async def _kickable() -> tuple[ControlPlaneService, InMemoryRepository, object, str]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    await service.put_budget(
        _principal(Role.PROJECT_ADMIN),
        BudgetRecord(org_id="acme", project_id="research", limit=100, updated_by="operator"),
        idempotency_key="budget",
    )
    fleet_admin, workload_id, _lease_id = await _running_workload(repository, service, key="kick")
    return service, repository, fleet_admin, workload_id


@pytest.mark.asyncio
async def test_a_kick_is_charged_to_the_preemption_budget() -> None:
    """The claim, stated where it is decided.

    ``preemption_count`` moving and ``capacity_loss_count`` staying is what lets
    a spot job survive being moved. The reverse would end it on the first kick.
    """

    service, _repository, fleet_admin, workload_id = await _kickable()

    resumed, _lease, _budget = await service.settle_preempted_workload(
        fleet_admin,
        workload_id,
        scheduling_reason="kicked by operator; waiting to resume",
        action="workload.kicked",
        provider_preempted=True,
    )

    assert resumed.state is WorkloadState.PENDING
    assert resumed.preemption_count == 1
    assert resumed.capacity_loss_count == 0


@pytest.mark.asyncio
async def test_a_kicked_workload_is_requeued_rather_than_ended() -> None:
    """``cancel`` is the other command. This one has to leave the job runnable."""

    service, _repository, fleet_admin, workload_id = await _kickable()

    resumed, _lease, _budget = await service.settle_preempted_workload(
        fleet_admin,
        workload_id,
        scheduling_reason="kicked by operator; waiting to resume",
        action="workload.kicked",
        provider_preempted=True,
    )

    assert resumed.state is WorkloadState.PENDING
    assert resumed.lease_id is None
    assert resumed.attempt == 2


@pytest.mark.asyncio
async def test_a_kick_is_distinguishable_from_a_provider_reclamation_in_the_ledger() -> None:
    """Both charge the preemption budget; only the ledger says which happened.

    "Why did this move at 04:12" has different answers -- the cloud took the
    slice, or a person took it -- and only one of them means something about the
    zone. Sharing a budget must not mean sharing an audit action.
    """

    service, repository, fleet_admin, workload_id = await _kickable()

    await service.settle_preempted_workload(
        fleet_admin,
        workload_id,
        scheduling_reason="kicked by operator; waiting to resume",
        action="workload.kicked",
        provider_preempted=True,
    )

    async with repository.transaction() as tx:
        events = await tx.list_events(org_id="acme", project_id="research", limit=100)
    actions = [event.action for event in events]
    assert actions.count("workload.kicked") == 1
    assert "workload.preempted" not in actions
