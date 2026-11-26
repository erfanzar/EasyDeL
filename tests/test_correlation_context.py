# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Concurrent operation correlation must be scoped and reset on failure."""

import asyncio

import pytest


def test_operation_correlation_is_reset_after_a_failure(caplog):
    from eyvan._internal.correlation import current_correlation_id, traced_operation

    caplog.set_level("INFO", logger="eyvan.correlation")
    with pytest.raises(RuntimeError):
        with traced_operation("provider.apply", "wl_test"):
            assert current_correlation_id() == "wl_test"
            raise RuntimeError("sensitive-provider-detail")
    assert current_correlation_id() is None
    assert "correlation_id=wl_test" in caplog.text
    assert "sensitive-provider-detail" not in caplog.text


@pytest.mark.parametrize("identifier", [None, "bad id"])
def test_uncorrelated_nested_operations_do_not_inherit_the_outer_identity(identifier):
    from eyvan._internal.correlation import current_correlation_id, traced_operation

    with traced_operation("outer", "wl_outer"):
        with traced_operation("inner", identifier):
            assert current_correlation_id() is None
        assert current_correlation_id() == "wl_outer"
    assert current_correlation_id() is None


async def test_concurrent_operations_do_not_exchange_correlation_ids():
    from eyvan._internal.correlation import current_correlation_id, traced_operation

    async def operation(identifier):
        with traced_operation("provider.apply", identifier):
            await asyncio.sleep(0)
            return await asyncio.to_thread(current_correlation_id)

    assert await asyncio.gather(operation("wl_a"), operation("wl_b")) == ["wl_a", "wl_b"]
    assert current_correlation_id() is None


def test_the_provider_boundary_runs_with_the_mutation_correlation(tmp_path, caplog):
    from dataclasses import replace

    from eyvan._internal.correlation import current_correlation_id
    from eyvan.capacity.drivers.local import LocalCapacityDriver
    from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
    from eyvan.config import EyvanSettings
    from eyvan.worker.agent.models import CapacityRequest, MutationModel
    from eyvan.worker.agent.service import CellAgentService

    class ObservingDriver(LocalCapacityDriver):
        def apply_capacity(self, cell_id, desired, context):
            observed = super().apply_capacity(cell_id, desired, context)
            return replace(observed, detail=current_correlation_id() or "missing")

    state = FleetStateStore(LocalBackend(tmp_path / "state.json"))
    service = CellAgentService(
        capacity=ObservingDriver(state), state=state, settings=EyvanSettings(agent_state_dir=str(tmp_path / "agent"))
    )
    request = CapacityRequest(
        mutation=MutationModel(
            idempotency_key="apply", generation=1, fencing_token="fence", correlation_id="wl_provider"
        ),
        desired={
            "provider": "local",
            "capacity_backend": "local",
            "executor": "native-local",
            "accelerator_type": "cpu",
            "capacity_class": "spot",
            "region": "local",
        },
    )
    caplog.set_level("INFO", logger="eyvan.correlation")
    try:
        assert service.apply_capacity("cell", request).detail == "wl_provider"
    finally:
        service.close()
    assert "correlation_id=wl_provider" in caplog.text
    assert current_correlation_id() is None


async def test_outbox_delivery_runs_inside_the_persisted_correlation(caplog):
    from eyvan._internal.correlation import current_correlation_id
    from eyvan.records import OutboxRecord, OutboxState
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    class Delivery(EyvanController):
        async def _dispatch(self, command):
            assert current_correlation_id() == command.correlation_id

    repository = InMemoryRepository()
    await repository.open()
    controller = Delivery(repository, ControlPlaneService(repository))
    command = OutboxRecord(
        id="cmd_test",
        topic="workload.dispatch",
        aggregate_type="workload",
        aggregate_id="wl_test",
        idempotency_key="dispatch",
        generation=1,
        correlation_id="wl_test",
    )
    async with repository.transaction() as tx:
        await tx.put_outbox(command)
    caplog.set_level("INFO", logger="eyvan.correlation")
    try:
        await controller.dispatch_outbox_once()
        async with repository.transaction() as tx:
            completed = await tx.get_outbox(command.id)
        assert completed.state is OutboxState.SUCCEEDED
        assert "correlation_id=wl_test" in caplog.text
        assert current_correlation_id() is None
    finally:
        await controller.close()
        await repository.close()
