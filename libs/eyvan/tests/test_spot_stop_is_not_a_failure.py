# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A Spot VM seen stopping on its way to a preemption is not a provider failure.

For a Spot instance moving from RUNNING through STOPPING to TERMINATED, the
intermediate DRAINING observation must not add provider failures or a cooldown
to a zone that delivered capacity.
"""

from __future__ import annotations

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.records import CellRecord
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.models import AgentObservation
from tests.test_capacity_lifecycle_logging import LifecycleAgent, _principal, _tpu_pool


class _ReasonedAgent(LifecycleAgent):
    def __init__(self) -> None:
        super().__init__()
        self.failure_code: CapacityProviderFailureCode | None = None

    def _observation(self, cell_id: str, generation: int) -> AgentObservation:
        return super()._observation(cell_id, generation).model_copy(update={"failure_code": self.failure_code})


@pytest.mark.asyncio
async def test_a_spot_instance_stopping_then_preempted_sets_no_cooldown() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _ReasonedAgent()

    def agent_factory(_cell: CellRecord) -> CellAgentClient:
        return agent

    controller = EyvanController(repository, service, agent_factory=agent_factory)
    await service.apply_fleet(_principal(), "research", (_tpu_pool(),), idempotency_key="fleet-v1")
    await controller.reconcile_capacity_pools_once()
    await controller.dispatch_outbox_once()

    async def cell() -> CellRecord:
        async with repository.transaction() as transaction:
            return (await transaction.list_cells())[0]

    agent.state, agent.detail = CapacityStatus.HEALTHY, "Compute instance is running"
    await controller.observe_cells_once()
    delivered = await cell()
    assert delivered.availability.provision_latencies_s

    agent.state, agent.detail = CapacityStatus.DRAINING, "Compute instance is stopping"
    await controller.observe_cells_once()
    stopping = await cell()
    assert stopping.availability.consecutive_failures == delivered.availability.consecutive_failures
    assert stopping.availability.cooldown_until is None

    agent.state, agent.detail = CapacityStatus.NEEDS_RECREATE, "Compute instance is terminal (TERMINATED)"
    agent.failure_code = CapacityProviderFailureCode.PREEMPTED
    await controller.observe_cells_once()
    reclaimed = await cell()
    assert reclaimed.provider_failure_code is CapacityProviderFailureCode.PREEMPTED
    assert reclaimed.availability.interruptions > 0
    assert reclaimed.availability.cooldown_until is None
    assert reclaimed.availability.consecutive_failures == delivered.availability.consecutive_failures
