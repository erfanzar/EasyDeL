# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The commit guard of a cell observation, and the quarantine it must not erase.

An observation snapshots a cell and then spends the length of an agent RPC
before it writes. A fence-conflict or host-fault quarantine can land inside
that window; both markers keep the state and the generation the snapshot was
validated against, so only a re-read of the markers stops the write from
silently re-admitting a cell the controller has slated for replacement.
"""

from __future__ import annotations

import pytest
from test_controller import FakeAgent

from eyvan.capacity import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    ResourceVector,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.worker.runtime.node_health import HOST_FAULT_REASON


def _cell(*, unavailable_reason: str | None) -> CellRecord:
    return CellRecord(
        id="cell-a",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        state=CellState.UNAVAILABLE,
        capacity=ResourceVector(cpu=32, memory_gib=64, tpu=8),
        allocatable=ResourceVector(cpu=32, memory_gib=64, tpu=8),
        accelerator_type="v5p-16",
        endpoint="http://agent:8090",
        unavailable_reason=unavailable_reason,
    )


async def _plane() -> tuple[EyvanController, InMemoryRepository]:
    repository = InMemoryRepository()
    await repository.open()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository),
        settings=EyvanSettings(),
        agent_factory=lambda _cell: FakeAgent(),
    )
    return controller, repository


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["agent_fence_conflict", HOST_FAULT_REASON])
async def test_an_observation_refuses_to_overwrite_a_quarantine(reason) -> None:
    """The commit re-reads the markers, not only the generation and state."""

    controller, repository = await _plane()
    snapshot = _cell(unavailable_reason="agent_unreachable")
    async with repository.transaction() as tx:
        await tx.put_cell(snapshot)
        # What a concurrent loop wrote while the agent RPC was in flight: the
        # same generation and state, an entirely different verdict.
        await tx.put_cell(
            snapshot.model_copy(
                update={
                    "provider_phase": CapacityStatus.NEEDS_RECREATE,
                    "unavailable_reason": reason,
                    "unavailable_detail": "the cell was slated for replacement",
                }
            )
        )

    assert await controller._observe_cell_once_locked(snapshot, utcnow()) is False

    async with repository.transaction() as tx:
        stored = await tx.get_cell("cell-a")
    assert stored is not None
    assert stored.provider_phase is CapacityStatus.NEEDS_RECREATE
    assert stored.unavailable_reason == reason
    assert stored.unavailable_detail == "the cell was slated for replacement"


@pytest.mark.asyncio
async def test_an_unmarked_stale_cell_is_still_observed() -> None:
    """The guard narrows to the markers; an ordinary observation still lands."""

    controller, repository = await _plane()
    snapshot = _cell(unavailable_reason="agent_unreachable")
    async with repository.transaction() as tx:
        await tx.put_cell(snapshot)

    assert await controller._observe_cell_once_locked(snapshot, utcnow()) is True

    async with repository.transaction() as tx:
        stored = await tx.get_cell("cell-a")
    assert stored is not None
    assert stored.state is CellState.READY
