# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Correlation uses durable ownership, not caller payload hints."""

import pytest
from test_group_lease_lifecycle import repository as _repository_fixture

from eyvan.records import CapacityPoolRecord, CellRecord, ResourceLimits
from eyvan.server.controller import EyvanController
from eyvan.server.service import ControlPlaneService

repository = _repository_fixture


@pytest.mark.parametrize("enabled", [False, True])
async def test_workload_driven_capacity_keeps_its_origin_in_the_outbox(repository, enabled):
    service = ControlPlaneService(repository, correlation_enabled=enabled)
    pool = CapacityPoolRecord(
        id="pool",
        fleet_name="fleet",
        node_type="cpu",
        provider="local",
        region="local",
        agent_endpoint="http://agent.invalid",
        capacity=ResourceLimits(cpu=1),
        allocatable=ResourceLimits(cpu=1),
    )
    cell = CellRecord(
        id="cell",
        provider="local",
        region="local",
        capacity=ResourceLimits(cpu=1),
        pool_id=pool.id,
        allocatable=ResourceLimits(cpu=1),
        autoscale_request_workload_id="wl_origin",
    )
    async with repository.transaction() as transaction:
        await transaction.put_capacity_pool(pool)
        await transaction.put_cell(cell)
        command = await service._outbox(
            transaction,
            topic="capacity.apply",
            aggregate_type="cell",
            aggregate_id=cell.id,
            idempotency_key="apply",
            generation=1,
            payload={"correlation_id": "spoofed"},
        )
    async with repository.transaction() as transaction:
        stored = await transaction.get_outbox(command.id)
    assert stored.correlation_id == ("wl_origin" if enabled else None)
    wire = stored.model_dump(mode="json")
    assert ("correlation_id" in wire) is enabled
    mutation = EyvanController._mutation(stored, "fence")
    assert mutation.correlation_id == stored.correlation_id


async def test_background_capacity_gets_an_operation_id_not_an_invented_workload(repository):
    service = ControlPlaneService(repository, correlation_enabled=True)
    async with repository.transaction() as transaction:
        command = await service._outbox(
            transaction,
            topic="capacity.apply",
            aggregate_type="cell",
            aggregate_id="warm",
            idempotency_key="warm",
            generation=1,
            payload={"correlation_id": "spoofed"},
        )
    assert command.correlation_id == command.id


async def test_workload_commands_use_the_server_workload_id(repository):
    service = ControlPlaneService(repository, correlation_enabled=True)
    async with repository.transaction() as transaction:
        command = await service._outbox(
            transaction,
            topic="workload.dispatch",
            aggregate_type="workload",
            aggregate_id="wl_server",
            idempotency_key="dispatch",
            generation=1,
            payload={"correlation_id": "spoofed"},
        )
    assert command.correlation_id == "wl_server"


async def test_legacy_identifiers_cannot_break_delivery_when_correlation_is_enabled(repository):
    from eyvan.worker.agent.models import MutationModel

    service = ControlPlaneService(repository, correlation_enabled=True)
    async with repository.transaction() as transaction:
        command = await service._outbox(
            transaction,
            topic="workload.dispatch",
            aggregate_type="workload",
            aggregate_id="legacy identifier",
            idempotency_key="legacy",
            generation=1,
            payload={},
        )
    async with repository.transaction() as transaction:
        restored = await transaction.get_outbox(command.id)
    mutation = EyvanController._mutation(restored, "fence")
    assert isinstance(mutation, MutationModel)
    assert mutation.correlation_id and " " not in mutation.correlation_id


@pytest.mark.parametrize("enabled", [False, True])
async def test_preemption_commands_keep_the_triggering_workloads_correlation(repository, enabled):
    from test_group_lease_lifecycle import _preemption

    _, command = await _preemption(repository, correlation_enabled=enabled)
    assert command.correlation_id == ("priority" if enabled else None)
