# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small, schema-valid decay must not stop capacity observations or scoring."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from eyvan._internal.compat import UTC
from eyvan.capacity import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.domain.availability import CapacityBelief, CapacityPoolKey, ProvisionOutcome
from eyvan.records import (
    AvailabilityPolicy,
    CapacityBeliefState,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    ResourceVector,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.models import AgentObservation, WorkloadMutationRequest

KEY = CapacityPoolKey("local", "", "local", "unknown", "on_demand")
NOW = datetime(2026, 10, 1, tzinfo=UTC)


class LocalObservationAgent:
    """Return validated observations without provider or network contact."""

    state = CapacityStatus.HEALTHY

    def reconcile(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        return AgentObservation(cell_id=cell_id, state=self.state, generation=request.mutation.generation)


def make_pool(decay: float) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id="availability-pool",
        fleet_name="availability",
        node_type="cpu",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=1),
        allocatable=ResourceVector(cpu=1),
        agent_endpoint="http://127.0.0.1:8090",
        availability_policy=AvailabilityPolicy(evidence_decay=decay),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("decay", [1e-200, 0.98])
async def test_success_then_pending_persists_capacity_transition(decay: float) -> None:
    pool = make_pool(decay)
    template = pool.make_cell("availability-cell")
    cell = CellRecord.model_validate({**template.model_dump(), "provider_phase": CapacityStatus.PROVISIONING})
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = LocalObservationAgent()
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _cell: agent)
    async with repository.transaction() as transaction:
        await transaction.put_capacity_pool(pool)
        await transaction.put_cell(cell)

    assert await controller.observe_cells_once() == 1
    async with repository.transaction() as transaction:
        ready = await transaction.get_cell(cell.id)
    assert ready is not None
    assert ready.state is CellState.READY
    assert ready.availability.success_evidence == 1 + decay
    assert ready.availability.failure_evidence == decay

    agent.state = CapacityStatus.PROVISIONING
    for _ in range(2):
        assert await controller.observe_cells_once() == 1
        async with repository.transaction() as transaction:
            pending = await transaction.get_cell(cell.id)
        assert pending is not None
        assert pending.state is CellState.REGISTERING
        assert pending.provider_phase is CapacityStatus.PROVISIONING
        assert pending.availability.success_evidence > 0
        assert pending.availability.failure_evidence > 0
        assert pending.availability.interruptions == 1
        assert CapacityBeliefState.model_validate_json(pending.availability.model_dump_json()) == pending.availability
    # Updating the repository must not mutate the committed READY snapshot.
    assert ready.state is CellState.READY
    assert ready.availability.interruptions == 0
    if decay == 0.98:
        assert pending.availability.success_evidence == (1 + decay) * decay
        assert pending.availability.failure_evidence == decay * decay
    await controller.close()


@pytest.mark.parametrize("outcome", [ProvisionOutcome.PROVIDER_ERROR, ProvisionOutcome.SUCCESS])
def test_repeated_decay_remains_valid_through_controller_adapter(outcome: ProvisionOutcome) -> None:
    cell = make_pool(0.1).make_cell("availability-cell")
    original = cell.availability
    for _ in range(400):
        observed = EyvanController._observe_capacity_belief(cell, outcome)
        assert observed.success_evidence > 0
        assert observed.failure_evidence > 0
        assert CapacityBeliefState.model_validate_json(observed.model_dump_json()) == observed
        cell = cell.model_copy(update={"availability": observed})
    assert original == CapacityBeliefState()
    belief = CapacityBelief(KEY, success_evidence=observed.success_evidence, failure_evidence=observed.failure_evidence)
    assert math.isfinite(belief.success_probability)
    assert 0 <= belief.success_probability <= 1
    assert 0 < belief.uncertainty <= 0.5


@pytest.mark.parametrize("decay", [0.1, 1e-200])
def test_interruption_only_decay_keeps_scores_defined(decay: float) -> None:
    original = CapacityBelief(KEY)
    belief = original
    for _ in range(400):
        belief = belief.observe(ProvisionOutcome.INTERRUPTED, now=NOW, evidence_decay=decay)
    assert belief.success_evidence > 0
    assert belief.failure_evidence > 0
    assert belief.success_probability == 0.5
    assert belief.uncertainty == pytest.approx(0.5)
    assert belief.should_probe(blocked_value_usd=10, probe_cost_usd=1, probe_budget_remaining_usd=2, now=NOW)
    assert original == CapacityBelief(KEY)
