# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from eyvan.capacity.drivers.base import CapacityStatus, CellCapabilities, CellObservation
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.records import (
    CapacityPoolRecord,
    CellRecord,
    CellState,
    OutboxState,
    Principal,
    ResourceVector,
    Role,
    RoleGrant,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient, CellAgentUnavailableError
from eyvan.worker.agent.service import CellAgentService


class _RecordingCapacity:
    def __init__(self) -> None:
        self.applied: list[tuple[str, dict]] = []

    def capabilities(self) -> CellCapabilities:
        return CellCapabilities(
            provider="test",
            executor="native-local",
            accelerator_types=("tpu",),
            capacity_classes=("spot",),
        )

    def observe(self, cell_id: str) -> CellObservation:
        return CellObservation(cell_id=cell_id, state=CapacityStatus.HEALTHY, generation=1)

    def reconcile(self, cell_id: str, context) -> CellObservation:
        return CellObservation(cell_id=cell_id, state=CapacityStatus.HEALTHY, generation=context.generation)

    def apply_capacity(self, cell_id: str, desired, context) -> CellObservation:
        self.applied.append((cell_id, dict(desired)))
        return CellObservation(cell_id=cell_id, state=CapacityStatus.HEALTHY, generation=context.generation)

    def drain(self, cell_id: str, context) -> CellObservation:
        return CellObservation(cell_id=cell_id, state=CapacityStatus.DRAINING, generation=context.generation)

    def release(self, cell_id: str, context) -> CellObservation:
        return CellObservation(cell_id=cell_id, state=CapacityStatus.RELEASED, generation=context.generation)


class _UnusedExecutors:
    def resolve(self, _cell_id, _observation):
        raise AssertionError("this test does not dispatch a workload")


class _BlockingSchedulerController(EyvanController):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reconciliation_entered = asyncio.Event()
        self.release_reconciliation = asyncio.Event()

    async def reconcile_capacity_pools_once(self) -> dict[str, int]:
        self.reconciliation_entered.set()
        await self.release_reconciliation.wait()
        return {"launched": 0, "drained": 0, "deleted": 0}


@pytest.mark.asyncio
async def test_outbox_delivery_continues_while_leader_reconciliation_is_blocked(tmp_path) -> None:
    """A slow leader cycle must not starve durable agent mutations."""

    secret = "test-secret"
    capacity = _RecordingCapacity()
    fleet_state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    agent_settings = EyvanSettings(agent_shared_secret=secret)
    agent_service = CellAgentService(
        capacity=capacity,  # type: ignore[arg-type]
        state=fleet_state,
        executors=_UnusedExecutors(),  # type: ignore[arg-type]
        settings=agent_settings,
    )
    transport = TestClient(
        create_agent_app(agent_service, agent_settings),
        headers={"Authorization": f"Bearer {secret}"},
    )
    agent = CellAgentClient("http://agent", client=transport)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = _BlockingSchedulerController(
        repository,
        service,
        settings=EyvanSettings(scheduler_interval_s=0.01, agent_shared_secret=secret),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type]
    )
    admin = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-live",
            provider="gcp",
            provider_project="project-a",
            region="us-east5",
            zone="us-east5-a",
            capacity=ResourceVector(cpu=0, memory_gib=0, tpu=4),
            allocatable=ResourceVector(cpu=0, memory_gib=0, tpu=4),
            accelerator_type="v5p-8",
            capacity_class="spot",
            endpoint="http://agent",
        ),
        idempotency_key="cell-live",
    )

    stop = asyncio.Event()
    task = asyncio.create_task(controller.run(stop))
    try:
        await asyncio.wait_for(controller.reconciliation_entered.wait(), timeout=1)

        async def wait_for_capacity_update() -> None:
            while True:
                async with repository.transaction() as transaction:
                    cell = await transaction.get_cell("cell-live")
                    commands = await transaction.list_outbox()
                if cell is not None and cell.state is CellState.READY and commands[0].state is OutboxState.SUCCEEDED:
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_capacity_update(), timeout=1)
    finally:
        controller.release_reconciliation.set()
        stop.set()
        await task
        await controller.close()
        agent.close()

    assert capacity.applied == [
        (
            "cell-live",
            {
                "cell_id": "cell-live",
                "provider": "gcp",
                "project": "project-a",
                "region": "us-east5",
                "zone": "us-east5-a",
                "accelerator_type": "v5p-8",
                "capacity_class": "spot",
                "executor": "eyvan-native",
                "agent_endpoint": "http://agent",
                "provider_options": {},
                "runtime": {
                    "env": {},
                    "init_commands": [],
                    "artifacts": [],
                    "vm_image": None,
                    "container": None,
                    # Declared so a callable pinned to another interpreter is
                    # refused at submit rather than on the accelerator.
                    "python": None,
                },
            },
        )
    ]


@pytest.mark.asyncio
async def test_an_unreachable_agent_is_not_recorded_as_a_provider_failure(tmp_path) -> None:
    """Unreachable tells you nothing about the provider.

    `CellAgentUnavailableError` -- the agent could not be reached at all -- fell
    into the outbox's bare exception handler, which spends an attempt and, for
    `capacity.apply`, records a PROVIDER_ERROR against the cell. A restarting
    agent or a momentary network fault therefore became durable evidence that GCP
    had failed, which is how a healthy pool acquires a failure it never had.

    Asserted on the durable record rather than on which handler ran: what matters
    is that no provider verdict is invented and the attempt is not spent.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())

    class _UnreachableAgent:
        def apply_capacity(self, *_args, **_kwargs):
            raise CellAgentUnavailableError("the cell agent could not be reached: connection refused")

        def __getattr__(self, _name):
            def _unreachable(*_args, **_kwargs):
                raise CellAgentUnavailableError("the cell agent could not be reached: connection refused")

            return _unreachable

    controller = EyvanController(
        repository,
        service,
        agent_factory=lambda _cell: _UnreachableAgent(),  # type: ignore[arg-type]
    )
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    pool = CapacityPoolRecord(
        id="research/cpu",
        fleet_name="research",
        node_type="cpu",
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 4, "memory_gib": 16},
        allocatable={"cpu": 4, "memory_gib": 16},
        executor="native-local",
        agent_endpoint="http://agent:8090",
        provider_options={"kind": "cpu", "machine_type": "n2-standard-4"},
        min_cells=1,
        initial_cells=1,
        max_cells=1,
    )
    await service.apply_fleet(fleet_admin, "research", (pool,), idempotency_key="fleet-v1")

    await controller.reconcile_capacity_pools_once()
    await controller.dispatch_outbox_once()

    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
        outbox = await transaction.list_outbox()

    applies = [record for record in outbox if record.topic == "capacity.apply"]
    assert applies, "the test needs a capacity.apply command to have been created"
    # Retried without spending an attempt, exactly like a busy agent.
    assert applies[0].attempts == 0
    # And no provider verdict was invented from a transport failure.
    assert all(cell.provider_failure_code is None for cell in cells)
