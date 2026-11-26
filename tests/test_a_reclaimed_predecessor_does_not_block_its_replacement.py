# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Allow replacement placement when the predecessor's capacity is definitively lost.

Retirement fences the predecessor so two generations cannot run at once. A host
whose capacity is definitively lost cannot still run the workload, so that fence
is already satisfied. An unreachable host that still exists may still be
running and must continue to block replacement.
"""

from __future__ import annotations

import pytest
from test_agent import FakeCapacity, NoopExecutor, StaticResolver

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CellObservation
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.worker.agent.models import MutationModel, WorkloadRequest
from eyvan.worker.agent.service import CellAgentService

RECLAIMED = "research-v5p-16-b59987-0001"
REPLACEMENT = "research-v5p-16-b59987-0002"
WORKLOAD = "wl_11b69c6f08bc4eaab504579e6b7b00de"


class Capacity(FakeCapacity):
    """Two cells, each answering with the state the live plane reported.

    The reclaimed cell publishes no ``agent_endpoint`` because its host is gone;
    that absence is the whole trigger, so it is modelled rather than stubbed out.
    """

    def __init__(self, reclaimed_state: CapacityStatus) -> None:
        super().__init__()
        self.reclaimed_state = reclaimed_state

    def observe(self, cell_id: str) -> CellObservation:
        if cell_id == RECLAIMED:
            return CellObservation(
                cell_id=cell_id,
                state=self.reclaimed_state,
                generation=2,
                detail="TPU VM is terminal (PREEMPTED)",
            )
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.HEALTHY,
            generation=2,
            execution_endpoint="local",
            metadata={"agent_endpoint": "http://replacement:8090"},
        )

    def reconcile(self, cell_id: str, _context) -> CellObservation:
        return self.observe(cell_id)

    def release(self, cell_id: str, _context) -> CellObservation:
        return self.observe(cell_id)


def _request(generation: int, submission: str) -> WorkloadRequest:
    return WorkloadRequest(
        spec={
            "submission_id": submission,
            "metadata": {"eyvan_workload_id": WORKLOAD},
        },
        mutation=MutationModel(
            idempotency_key=f"submit-{generation}",
            generation=generation,
            fencing_token=f"cell:{generation}",
        ),
    )


def _service(tmp_path, reclaimed_state: CapacityStatus) -> CellAgentService:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    state._owner = "agent"
    return CellAgentService(
        capacity=Capacity(reclaimed_state),  # type: ignore[arg-type]
        state=state,
        executors=StaticResolver(NoopExecutor()),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "reclaimed_state",
    [
        CapacityStatus.NEEDS_RECREATE,  # what the live preempted v5p-16 reported
        CapacityStatus.ABSENT,
        CapacityStatus.RELEASED,
    ],
)
def test_a_workload_is_replaced_onto_healthy_capacity_after_its_host_is_reclaimed(
    tmp_path, reclaimed_state: CapacityStatus
) -> None:
    """The live failure, as the successor's own submit sees it.

    The assertion is on the *replacement* succeeding, because that is what the
    reclaimed cell was blocking and what an operator loses when it does not: the
    preemption budget is spent, the capacity is back, and the job still does not
    run.
    """

    service = _service(tmp_path, reclaimed_state)
    service.submit(RECLAIMED, _request(generation=1, submission="run-1"))

    observation = service.submit(REPLACEMENT, _request(generation=2, submission="run-2"))

    assert observation.remote_id == "run-2"


def test_a_predecessor_that_may_still_be_running_is_not_waved_through(tmp_path) -> None:
    """The half that must keep failing, and the reason the check is narrow.

    A reachable-but-unanswering host is not a reclaimed one. Treating every
    missing endpoint as a dead host would let a second generation start while
    the first still holds the accelerators -- the exact split-brain the fence
    exists to prevent, and a far worse outcome than a stalled workload.
    """

    service = _service(tmp_path, CapacityStatus.HEALTHY)
    service.submit(RECLAIMED, _request(generation=1, submission="run-1"))

    with pytest.raises(RuntimeError, match="no agent endpoint for remote retirement"):
        service.submit(REPLACEMENT, _request(generation=2, submission="run-2"))
