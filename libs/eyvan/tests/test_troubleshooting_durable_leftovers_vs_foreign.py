# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Troubleshooting must distinguish durable leftovers from foreign inventory."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import CapacityPoolRecord, CellRecord, CellState
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


def test_troubleshooting_does_not_promise_automatic_provider_orphan_cleanup() -> None:
    text = (Path(__file__).resolve().parents[1] / "docs" / "troubleshoot.md").read_text()
    capacity_section = text.split("## Capacity and cells", 1)[1].split("## The plane and the agent", 1)[0]

    assert "the controller reaps on reconcile passes" not in capacity_section
    assert "known control-plane" in capacity_section
    assert "agent/provider" in capacity_section
    assert "eyvan capacity orphans --json" in capacity_section
    assert "read-only" in capacity_section
    assert "scanned_endpoints" in capacity_section
    assert "unreachable" in capacity_section
    assert "empty database" in capacity_section
    assert "ownership" in capacity_section
    assert "explicitly authorize cleanup" in capacity_section


class _InventoryAgent:
    def __init__(self) -> None:
        self.records = [{"name": "provider-orphan-0001", "desired_state": "up", "generation": 1}]
        self.reads = 0

    def inventory(self) -> list[dict[str, object]]:
        self.reads += 1
        return deepcopy(self.records)


@pytest.mark.asyncio
async def test_pool_reconciliation_does_not_adopt_or_release_unknown_inventory() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _InventoryAgent()
    before = deepcopy(agent.records)
    cell = CellRecord(
        id="known-cell",
        state=CellState.READY,
        provider="local",
        region="local",
        capacity={"cpu": 1},
        allocatable={"cpu": 1},
        endpoint="http://inventory-agent.invalid:8090",
    )
    pool = CapacityPoolRecord(
        id="idle/cpu",
        fleet_name="idle",
        node_type="cpu",
        provider="local",
        region="local",
        capacity={"cpu": 1},
        allocatable={"cpu": 1},
        agent_endpoint=cell.endpoint,
        max_cells=0,
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        await transaction.put_capacity_pool(pool)
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda _: agent,  # type: ignore[arg-type,return-value]
    )

    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    assert await controller.reconcile_unmanaged_capacity_recoveries_once() == 0
    assert agent.reads == 0
    report = await controller.capacity_orphans()

    assert report == {
        "orphans": [{**before[0], "endpoint": cell.endpoint}],
        "scanned_endpoints": [cell.endpoint],
        "unreachable": [],
    }
    assert agent.reads == 1
    assert agent.records == before
    async with repository.transaction() as transaction:
        assert await transaction.list_cells() == [cell]
        assert await transaction.list_outbox() == []


@pytest.mark.asyncio
async def test_empty_database_orphan_report_scans_no_agents() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _InventoryAgent()
    before = deepcopy(agent.records)
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda _: agent,  # type: ignore[arg-type,return-value]
    )

    assert await controller.capacity_orphans() == {"orphans": [], "scanned_endpoints": [], "unreachable": []}
    assert agent.reads == 0
    assert agent.records == before


@pytest.mark.asyncio
async def test_unreachable_agent_is_not_a_clean_orphan_scan() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    cell = CellRecord(
        id="known-cell",
        provider="local",
        region="local",
        capacity={"cpu": 1},
        allocatable={"cpu": 1},
        endpoint="http://inventory-agent.invalid:8090",
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)

    def unreachable(_: CellRecord) -> None:
        raise RuntimeError("inventory unavailable")

    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=unreachable)
    assert await controller.capacity_orphans() == {
        "orphans": [],
        "scanned_endpoints": [cell.endpoint],
        "unreachable": [{"endpoint": cell.endpoint, "detail": "inventory unavailable"}],
    }
    async with repository.transaction() as transaction:
        assert await transaction.list_cells() == [cell]
