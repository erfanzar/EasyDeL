# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What happens when an execution's status can never be read.

An INDETERMINATE observation is a statement about the observation, not about the
workload, so the controller keeps polling rather than guessing. Persistently
unreadable observations must eventually escalate instead of holding capacity
indefinitely.

The escalation drains the *cell* rather than settling the workload. Settling it
would release the lease while the workload might still be running, letting a
second workload onto the same accelerator -- the exact hazard the unreadable
status is warning about. The host is the only thing that can be reclaimed with
certainty, and replacing it destroys anything that survived; the existing
cell-loss path then recovers the workload.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_controller import FakeAgent, principal, snapshot

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    ResourceVector,
    Role,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


class UnreadableAgent(FakeAgent):
    """An agent whose workload status never resolves."""

    def __init__(self) -> None:
        super().__init__()
        self.remote_status = ExecutionWorkloadStatus.INDETERMINATE
        self.remote_message = "cannot observe native OCI container: docker daemon unreachable"


async def _running_workload(controller: EyvanController, service: ControlPlaneService) -> str:
    operator = principal(Role.PROJECT_OPERATOR)
    async with controller.repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="gcp",
                provider_project="gcp-project-a",
                region="us-central2",
                zone="us-central2-b",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8, memory_gib=16, tpu=8),
                allocatable=ResourceVector(cpu=8, memory_gib=16, tpu=8),
                accelerator_type="v5p-8",
                endpoint="http://agent:8090",
            )
        )
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 2, "tpu": 8}},
        ),
        idempotency_key="submit",
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    return workload.id


def _controller(settings: EyvanSettings, agent: FakeAgent) -> tuple[EyvanController, ControlPlaneService]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=settings, agent_factory=lambda _cell: agent)
    return controller, service


@pytest.mark.asyncio
async def test_an_unreadable_execution_does_not_escalate_inside_the_grace_period() -> None:
    """Escalation ends in draining a host, so it must not fire on a blip."""

    agent = UnreadableAgent()
    controller, service = _controller(EyvanSettings(indeterminate_execution_grace_s=900), agent)
    await _running_workload(controller, service)

    for _ in range(5):
        await controller.observe_workloads_once()

    _workloads, cells, _leases, _commands = await snapshot(controller.repository)
    assert cells[0].state is CellState.READY, "a transient unreadable status must not destroy a host"
    assert not agent.drains


@pytest.mark.asyncio
async def test_an_execution_unreadable_past_the_grace_period_replaces_its_host() -> None:
    """The forty-minute hang: something has to give, and it is the cell."""

    agent = UnreadableAgent()
    controller, service = _controller(EyvanSettings(indeterminate_execution_grace_s=900), agent)
    workload_id = await _running_workload(controller, service)

    await controller.observe_workloads_once()
    # Rewind the first-unreadable stamp rather than sleeping fifteen minutes.
    key = next(iter(controller._indeterminate_since))
    controller._indeterminate_since[key] = utcnow() - timedelta(seconds=901)

    await controller.observe_workloads_once()

    _workloads, cells, _leases, _commands = await snapshot(controller.repository)
    assert cells[0].state is CellState.DRAINING
    workloads, _cells, _leases, _commands = await snapshot(controller.repository)
    assert next(w for w in workloads if w.id == workload_id).state is not WorkloadState.SUCCEEDED


@pytest.mark.asyncio
async def test_a_status_that_becomes_readable_again_clears_the_escalation_clock() -> None:
    """A workload that recovers must not carry a countdown from an old blip."""

    agent = UnreadableAgent()
    controller, service = _controller(EyvanSettings(indeterminate_execution_grace_s=900), agent)
    await _running_workload(controller, service)
    await controller.observe_workloads_once()
    assert controller._indeterminate_since

    agent.remote_status = ExecutionWorkloadStatus.RUNNING
    await controller.observe_workloads_once()

    assert not controller._indeterminate_since


@pytest.mark.asyncio
async def test_escalation_does_not_re_drain_a_cell_already_leaving() -> None:
    """A cell that is already draining is owned by the cell-loss path."""

    agent = UnreadableAgent()
    controller, service = _controller(EyvanSettings(indeterminate_execution_grace_s=0), agent)
    await _running_workload(controller, service)
    admin = principal(Role.FLEET_ADMIN)
    await service.drain_cell(admin, "cell-a", idempotency_key="operator-drain")
    agent.drains.clear()

    await controller.observe_workloads_once()
    await controller.observe_workloads_once()

    assert not agent.drains, "escalation issued a redundant drain for a cell already leaving"
