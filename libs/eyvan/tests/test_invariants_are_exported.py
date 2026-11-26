# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish durability invariant violations on each controller cycle without requiring a manual HTTP query."""

from __future__ import annotations

from datetime import timedelta

import pytest
from prometheus_client import CollectorRegistry

from eyvan._internal.observability import EyvanMetrics
from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller import EyvanController
from eyvan.server.invariants import Invariant
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def _controller() -> tuple[EyvanController, InMemoryRepository, EyvanMetrics]:
    repository = InMemoryRepository()
    metrics = EyvanMetrics(registry=CollectorRegistry())
    service = ControlPlaneService(repository)
    return EyvanController(repository, service, metrics=metrics), repository, metrics


def _sample(metrics: EyvanMetrics, invariant: Invariant) -> float:
    return metrics.invariant_violations.labels(invariant=invariant.value)._value.get()


def _pool() -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id="research/cpu",
        fleet_name="research",
        node_type="cpu",
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 8, "memory_gib": 32},
        allocatable={"cpu": 8, "memory_gib": 32},
        capacity_class=CapacityClass.SPOT,
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=4,
    )


def _workload() -> WorkloadRecord:
    return WorkloadRecord(
        id="wl-live",
        state=WorkloadState.RUNNING,
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=WorkloadDefinition(
            name="live",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 1, "memory_gib": 0}},
            provider="gcp",
            required_executor="eyvan-native",
        ),
    )


def _lease(cell_id: str) -> LeaseRecord:
    return LeaseRecord(
        id="lease-live",
        workload_id="wl-live",
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0, memory_gib=0.0)),
        fencing_token=f"{cell_id}:1",
        hourly_cost=1.0,
        expires_at=utcnow() + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_a_healthy_fleet_publishes_a_zero_for_every_invariant() -> None:
    """A series that appears only once something breaks cannot be alerted on.

    It also cannot be distinguished from a scrape that missed. Publishing zero
    is what makes "this fleet is fine" an observation rather than an absence.
    """

    controller, _repository, metrics = _controller()

    await controller.export_metrics_once()

    for invariant in Invariant:
        assert _sample(metrics, invariant) == 0.0, invariant


@pytest.mark.asyncio
async def test_the_lease_on_a_missing_cell_from_2026_08_12_is_published() -> None:
    """I9, the condition that wedged a live plane behind a dropped cell.

    The scheduler refused the snapshot for exactly this reason and said so only
    in a traceback. The invariant naming it was never evaluated.
    """

    controller, repository, metrics = _controller()
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())
        await transaction.put_lease(_lease("cell-that-is-gone"))

    await controller.export_metrics_once()

    assert _sample(metrics, Invariant.NO_LEASE_ON_A_MISSING_CELL) == 1.0


@pytest.mark.asyncio
async def test_a_violation_that_clears_returns_to_zero() -> None:
    """A gauge left at its last value is a false alarm that never ends."""

    controller, repository, metrics = _controller()
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())
        await transaction.put_lease(_lease("cell-that-is-gone"))
    await controller.export_metrics_once()
    assert _sample(metrics, Invariant.NO_LEASE_ON_A_MISSING_CELL) == 1.0

    async with repository.transaction() as transaction:
        pool = _pool()
        await transaction.put_capacity_pool(pool)
        await transaction.put_cell(
            pool.make_cell("cell-that-is-gone", worker_count=1).model_copy(update={"state": CellState.READY})
        )

    await controller.export_metrics_once()

    assert _sample(metrics, Invariant.NO_LEASE_ON_A_MISSING_CELL) == 0.0


@pytest.mark.asyncio
async def test_violations_are_said_out_loud_as_one_line(caplog) -> None:
    """They arrive in correlated groups, and the shape is what an operator reads."""

    import logging

    controller, repository, metrics = _controller()
    del metrics
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())
        await transaction.put_lease(_lease("cell-that-is-gone"))

    with caplog.at_level(logging.WARNING):
        await controller.export_metrics_once()

    assert "I9" in "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_failing_invariant_check_does_not_lose_the_other_metrics() -> None:
    """It is the newest and least trusted part of this phase.

    Fleet state, capacity and spend are what an operator dashboards; they must
    not stop being published because a self-check raised.
    """

    controller, repository, metrics = _controller()
    async with repository.transaction() as transaction:
        await transaction.put_workload(_workload())

    async def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("invariant read failed")

    import eyvan.server.controller.settlement as settlement

    original = settlement.check_fleet
    settlement.check_fleet = boom  # type: ignore[assignment]
    try:
        exported = await controller.export_metrics_once()
    finally:
        settlement.check_fleet = original  # type: ignore[assignment]

    assert exported == 1
    assert metrics.workloads.labels(state=WorkloadState.RUNNING.value)._value.get() == 1.0
