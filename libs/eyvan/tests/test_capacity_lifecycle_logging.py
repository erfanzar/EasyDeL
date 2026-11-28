# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.records import CapacityPoolRecord, CellRecord, Principal, Role, RoleGrant
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.models import AgentObservation, CapacityRequest, WorkloadMutationRequest


class LifecycleAgent(CellAgentClient):
    """Typed in-process cell agent whose provider state is test-controlled."""

    def __init__(self, state: CapacityStatus = CapacityStatus.WAITING):
        super().__init__("http://agent:8090")
        self.state = state
        self.detail = "queued for Spot TPU capacity"

    def _observation(self, cell_id: str, generation: int) -> AgentObservation:
        return AgentObservation(
            cell_id=cell_id,
            state=self.state,
            generation=generation,
            detail=self.detail,
        )

    def apply_capacity(self, cell_id: str, request: CapacityRequest) -> AgentObservation:
        return self._observation(cell_id, request.mutation.generation)

    def reconcile(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        return self._observation(cell_id, request.mutation.generation)

    def drain(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        return self._observation(cell_id, request.mutation.generation)

    def release(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        return AgentObservation(
            cell_id=cell_id,
            state=CapacityStatus.RELEASED,
            generation=request.mutation.generation,
            detail="provider capacity released",
        )


def _principal() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _tpu_pool() -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id="research-v5p-16",
        fleet_name="research",
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity={"cpu": 32, "tpu": 8},
        allocatable={"cpu": 32, "tpu": 8},
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=1,
        max_cells=1,
    )


def _lifecycle_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("capacity lifecycle ")]


@pytest.mark.asyncio
async def test_capacity_lifecycle_logs_only_authoritative_tpu_transitions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = LifecycleAgent()

    def agent_factory(_cell: CellRecord) -> CellAgentClient:
        return agent

    controller = EyvanController(repository, service, agent_factory=agent_factory)
    caplog.set_level(logging.INFO, logger="eyvan.server.controller")
    try:
        await service.apply_fleet(_principal(), "research", (_tpu_pool(),), idempotency_key="fleet-v1")

        assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
        assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
        messages = _lifecycle_messages(caplog)
        assert len([message for message in messages if "event=planned" in message]) == 1
        planned = messages[0]
        assert 'fleet="research"' in planned
        assert 'pool="research-v5p-16"' in planned
        assert 'provider="gcp"' in planned
        assert 'project="project-a"' in planned
        assert 'location="us-east5-a"' in planned
        assert "resource_kind=tpu" in planned
        assert 'resource="v5p-16"' in planned
        assert "workers=1" in planned
        assert "capacity_class=spot" in planned

        await controller.dispatch_outbox_once()
        messages = _lifecycle_messages(caplog)
        assert len([message for message in messages if "event=requesting" in message]) == 1
        requesting = messages[-2]
        assert "previous_provider_state=none" in requesting
        assert "provider_state=REQUESTING" in requesting
        waiting = messages[-1]
        assert "event=provider_transition" in waiting
        assert "previous_provider_state=REQUESTING" in waiting
        assert "provider_state=WAITING" in waiting

        caplog.clear()
        assert await controller.observe_cells_once() == 1
        assert await controller.observe_cells_once() == 1
        assert _lifecycle_messages(caplog) == []

        agent.state = CapacityStatus.PROVISIONING
        agent.detail = "queued resource is provisioning"
        assert await controller.observe_cells_once() == 1
        provisioning = _lifecycle_messages(caplog)[-1]
        assert "event=provisioning" in provisioning
        assert "previous_provider_state=WAITING" in provisioning
        assert "provider_state=PROVISIONING" in provisioning
        provisioning_count = len(_lifecycle_messages(caplog))
        assert await controller.observe_cells_once() == 1
        assert len(_lifecycle_messages(caplog)) == provisioning_count

        agent.state = CapacityStatus.ACTIVE
        agent.detail = "queued resource active; TPU VM starting"
        assert await controller.observe_cells_once() == 1
        assert "event=provider_transition" in _lifecycle_messages(caplog)[-1]
        assert "previous_provider_state=PROVISIONING" in _lifecycle_messages(caplog)[-1]
        assert "provider_state=ACTIVE" in _lifecycle_messages(caplog)[-1]
        provider_transition_count = len(_lifecycle_messages(caplog))
        assert await controller.observe_cells_once() == 1
        assert len(_lifecycle_messages(caplog)) == provider_transition_count

        agent.state = CapacityStatus.HEALTHY
        agent.detail = "TPU VM hosts and native workers are connected"
        assert await controller.observe_cells_once() == 1
        ready = _lifecycle_messages(caplog)[-1]
        assert "event=ready" in ready
        assert "cell_state=ready" in ready
        assert "provider_state=HEALTHY" in ready
        ready_count = len(_lifecycle_messages(caplog))
        assert await controller.observe_cells_once() == 1
        assert len(_lifecycle_messages(caplog)) == ready_count

        agent.state = CapacityStatus.PROVISIONING
        agent.detail = "Spot TPU capacity was preempted and is provisioning again"
        assert await controller.observe_cells_once() == 1
        assert "event=preempted" in _lifecycle_messages(caplog)[-1]

        agent.state = CapacityStatus.UNAVAILABLE
        agent.detail = "TPU provider health is indeterminate"
        assert await controller.observe_cells_once() == 1
        assert "event=unhealthy" in _lifecycle_messages(caplog)[-1]
        async with repository.transaction() as transaction:
            unhealthy = (await transaction.list_cells())[0]
            await transaction.put_cell(
                unhealthy.model_copy(
                    update={"availability": unhealthy.availability.model_copy(update={"cooldown_until": None})}
                )
            )

        agent.state = CapacityStatus.WAITING
        agent.detail = "retry queued for replacement Spot capacity"
        assert await controller.observe_cells_once() == 1
        assert "event=retry" in _lifecycle_messages(caplog)[-1]

        agent.state = CapacityStatus.NEEDS_RECREATE
        agent.detail = "TPU queued resource was lost"
        assert await controller.observe_cells_once() == 1
        assert "event=lost" in _lifecycle_messages(caplog)[-1]

        assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 1, "deleted": 0}
        replacement = _lifecycle_messages(caplog)[-1]
        assert "event=replacement" in replacement
        assert 'reason="provider_lost"' in replacement

        assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 1}
        await controller.dispatch_outbox_once()
        removed = _lifecycle_messages(caplog)[-1]
        assert "event=removed" in removed
        assert "cell_state=deleted" in removed

        assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
        assert "event=planned" in _lifecycle_messages(caplog)[-1]
    finally:
        agent.close()
