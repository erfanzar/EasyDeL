# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unstructured gateway failures must not become provider capacity evidence."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from eyvan.capacity.drivers.base import CapacityProviderFailureCode, CapacityStatus
from eyvan.records import CapacityBeliefState, CellRecord, CellState, ResourceVector, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.worker.agent.client import CellAgentCapacityProviderError, CellAgentClient, CellAgentUnavailableError
from eyvan.worker.agent.models import MutationModel, WorkloadMutationRequest

_UNSTRUCTURED_RESPONSES = [
    pytest.param(500, b"Internal Server Error", id="plain-500"),
    pytest.param(503, b"Service Unavailable", id="plain-503"),
    pytest.param(502, b"Bad Gateway", id="plain-502"),
    pytest.param(502, b'{"detail":', id="malformed-json-502"),
    pytest.param(502, b'{"detail":"Bad Gateway"}', id="missing-code-502"),
    pytest.param(502, b'{"detail":"Bad Gateway","code":"proxy_error"}', id="invalid-code-502"),
    pytest.param(502, b"", id="empty-502"),
]
_PROVIDER_FAILURE = b'{"detail":"provider stockout","code":"provider_error"}'


def _request() -> WorkloadMutationRequest:
    return WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="reconcile:cell-a:1", generation=1, fencing_token="cell:cell-a:1")
    )


def _ready_cell() -> CellRecord:
    observed_at = utcnow() - timedelta(minutes=5)
    return CellRecord(
        id="cell-a",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=1, memory_gib=1),
        allocatable=ResourceVector(cpu=1, memory_gib=1),
        endpoint="http://agent",
        provider_phase=CapacityStatus.HEALTHY,
        provider_phase_started_at=observed_at,
        last_heartbeat_at=observed_at,
        availability=CapacityBeliefState(success_evidence=4, failure_evidence=2, last_observed_at=observed_at),
    )


@pytest.mark.parametrize(("status", "content"), _UNSTRUCTURED_RESPONSES)
def test_unstructured_5xx_is_agent_unavailability(status: int, content: bytes) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content)

    with httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)) as transport:
        client = CellAgentClient("http://agent", client=transport)
        with pytest.raises(CellAgentUnavailableError):
            client.reconcile("cell-a", _request())


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "content"), _UNSTRUCTURED_RESPONSES)
async def test_unstructured_5xx_preserves_persisted_capacity(status: int, content: bytes) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    cell = _ready_cell()
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content)

    with httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)) as transport:
        client = CellAgentClient("http://agent", client=transport)
        controller = EyvanController(repository, service, agent_factory=lambda _cell: client)
        changed = await controller.observe_cells_once()

    async with repository.transaction() as transaction:
        persisted = await transaction.get_cell(cell.id)
    assert persisted is not None
    # Compare the complete record, including the provider phase clock, belief,
    # heartbeat, runtime width and update timestamp: a proxy taught us nothing.
    assert persisted.model_dump() == cell.model_dump()
    assert changed == 0


@pytest.mark.asyncio
async def test_structured_502_still_persists_provider_failure() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    cell = _ready_cell()
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=_PROVIDER_FAILURE)

    with httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler)) as transport:
        client = CellAgentClient("http://agent", client=transport)
        with pytest.raises(CellAgentCapacityProviderError, match="provider stockout") as error:
            client.reconcile(cell.id, _request())
        assert error.value.code is CapacityProviderFailureCode.PROVIDER_ERROR
        controller = EyvanController(repository, service, agent_factory=lambda _cell: client)
        assert await controller.observe_cells_once() == 1

    async with repository.transaction() as transaction:
        persisted = await transaction.get_cell(cell.id)
    assert persisted is not None
    assert persisted.state is CellState.UNAVAILABLE
    assert persisted.provider_phase is CapacityStatus.PROVIDER_ERROR
    assert persisted.provider_failure_code is CapacityProviderFailureCode.PROVIDER_ERROR
    assert persisted.unavailable_detail == "provider stockout"
    assert persisted.availability.failure_evidence > cell.availability.failure_evidence
    assert persisted.provider_phase_started_at != cell.provider_phase_started_at
    assert persisted.last_heartbeat_at == cell.last_heartbeat_at
