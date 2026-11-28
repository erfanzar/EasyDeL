# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Metrics the control plane declared, exported, and never wrote.

Seven Prometheus families are defined and served at ``/metrics``. Three were
written by the request middleware and the scheduler. The other four --
``eyvan_workloads``, ``eyvan_capacity_accelerators``, ``eyvan_project_spend_usd``
and ``eyvan_reconcile_seconds`` -- had no production writer at all, so fleet
state, capacity inventory and cost always read zero.

A gauge stuck at zero is worse than an absent one: an absent metric is a
question, a zero is an answer. An operator dashboarding "how much capacity do I
have" got a flat line and no reason to doubt it.

``trace_span`` was the same shape: a complete, correct implementation with
OpenTelemetry as a hard dependency and not one call site anywhere.
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry
from test_controller import FakeAgent, principal

from eyvan._internal.observability import EyvanMetrics, trace_span
from eyvan.config import EyvanSettings
from eyvan.records import (
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    ResourceVector,
    Role,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


class _FailingAgent(FakeAgent):
    """An agent whose provider call fails, to time the unhappy path."""

    def apply_capacity(self, cell_id, request):  # type: ignore[no-untyped-def]
        del cell_id, request
        raise RuntimeError("provider is on fire")


def _controller(**settings: object) -> tuple[EyvanController, ControlPlaneService, EyvanMetrics]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    metrics = EyvanMetrics(registry=CollectorRegistry())
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(**settings),  # type: ignore[arg-type]
        metrics=metrics,
    )
    return controller, service, metrics


def test_the_metrics_phase_is_registered() -> None:
    """The whole defect was that nothing wrote these families."""

    controller, _service, _metrics = _controller()

    assert "export-metrics" in {phase.name for phase in controller._leader_phases()}


@pytest.mark.asyncio
async def test_workload_state_is_published() -> None:
    """``eyvan_workloads`` is the first thing an operator graphs."""

    controller, service, metrics = _controller()
    await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 1}},
        ),
        idempotency_key="submit",
    )

    await controller.export_metrics_once()

    assert metrics.workloads.labels(state=WorkloadState.PENDING.value)._value.get() == 1


@pytest.mark.asyncio
async def test_accelerator_capacity_is_published() -> None:
    """ "How much capacity do I have" must not be a flat zero."""

    controller, _service, metrics = _controller()
    async with controller.repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="gcp",
                provider_project="p",
                region="us-east5",
                zone="us-east5-a",
                state=CellState.READY,
                capacity=ResourceVector(cpu=32, tpu=8),
                allocatable=ResourceVector(cpu=32, tpu=8),
                accelerator_type="v5p-16",
                capacity_class="spot",
                endpoint="http://agent:8090",
            )
        )

    await controller.export_metrics_once()

    published = metrics.capacity.labels(
        state=CellState.READY.value,
        capacity_class="spot",
        accelerator_type="v5p-16",
    )._value.get()
    assert published == 8


@pytest.mark.asyncio
async def test_a_deleted_cell_is_not_counted_as_capacity() -> None:
    """Capacity that no longer exists must not inflate the inventory."""

    controller, _service, metrics = _controller()
    async with controller.repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-gone",
                provider="gcp",
                provider_project="p",
                region="us-east5",
                zone="us-east5-a",
                state=CellState.DELETED,
                capacity=ResourceVector(cpu=32, tpu=8),
                allocatable=ResourceVector(cpu=32, tpu=8),
                accelerator_type="v5p-16",
                capacity_class="spot",
                endpoint="http://agent:8090",
            )
        )

    await controller.export_metrics_once()

    families = {family.name for family in metrics.capacity.collect()}
    assert families  # the family exists
    assert not [
        sample
        for family in metrics.capacity.collect()
        for sample in family.samples
        if sample.labels.get("state") == CellState.DELETED.value
    ]


@pytest.mark.asyncio
async def test_project_spend_is_published() -> None:
    """Cost is the third thing operators dashboard and it read zero."""

    controller, _service, metrics = _controller()
    async with controller.repository.transaction() as transaction:
        await transaction.put_budget(
            BudgetRecord(org_id="acme", project_id="research", limit=100, spent=42, updated_by="ops")
        )

    await controller.export_metrics_once()

    assert metrics.project_spend.labels(org="acme", project="research", kind="spent")._value.get() == 42.0


@pytest.mark.asyncio
async def test_stale_labels_do_not_persist_across_exports() -> None:
    """A gauge is a level: a state that emptied must stop being reported.

    Without clearing, a workload state that no longer has members keeps its last
    value forever and the dashboard shows work that finished hours ago.
    """

    controller, service, metrics = _controller()
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 1}},
        ),
        idempotency_key="submit",
    )
    await controller.export_metrics_once()

    async with controller.repository.transaction() as transaction:
        stored = await transaction.get_workload(workload.id)
        assert stored is not None
        await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))
    await controller.export_metrics_once()

    pending = [
        sample
        for family in metrics.workloads.collect()
        for sample in family.samples
        if sample.labels.get("state") == WorkloadState.PENDING.value
    ]
    assert not pending, "a state with no members kept reporting its old count"


@pytest.mark.asyncio
async def test_delivery_latency_is_recorded_per_topic() -> None:
    """``eyvan_reconcile_seconds`` is where a slow provider becomes visible.

    Outbox delivery is the one place a provider or a cell agent can block for
    minutes, and the histogram declared to measure exactly that had no call
    site, so a hung ``capacity.apply`` produced no signal at all.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    metrics = EyvanMetrics(registry=CollectorRegistry())
    fake_agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(lease_ttl_s=120),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
        metrics=metrics,
    )
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="p",
            region="us-central2",
            zone="us-central2-b",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            allocatable=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            accelerator_type="v5p-8",
            capacity_class="spot",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    assert await controller.dispatch_outbox_once() == 1

    counted = {
        sample.labels["operation"]: sample.value
        for family in metrics.reconcile_duration.collect()
        for sample in family.samples
        if sample.name.endswith("_count")
    }
    assert counted.get("capacity.apply") == 1


@pytest.mark.asyncio
async def test_delivery_latency_is_recorded_even_when_delivery_fails() -> None:
    """A failing operation is the one whose latency an operator most needs.

    Timing only the success path would hide precisely the outage being
    diagnosed: a provider that takes two minutes to reject every request.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    metrics = EyvanMetrics(registry=CollectorRegistry())
    fake_agent = _FailingAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(lease_ttl_s=120),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
        metrics=metrics,
    )
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="p",
            region="us-central2",
            zone="us-central2-b",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            allocatable=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            accelerator_type="v5p-8",
            capacity_class="spot",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    await controller.dispatch_outbox_once()

    counted = {
        sample.labels["operation"]: sample.value
        for family in metrics.reconcile_duration.collect()
        for sample in family.samples
        if sample.name.endswith("_count")
    }
    assert counted.get("capacity.apply") == 1


@pytest.mark.asyncio
async def test_a_controller_without_metrics_still_dispatches() -> None:
    """Metrics are optional everywhere else; delivery must not start requiring them."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    fake_agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(lease_ttl_s=120),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
    )
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="gcp",
            provider_project="p",
            region="us-central2",
            zone="us-central2-b",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            allocatable=ResourceVector(cpu=8, memory_gib=64, tpu=8),
            accelerator_type="v5p-8",
            capacity_class="spot",
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell-create",
    )

    assert await controller.dispatch_outbox_once() == 1


def test_trace_span_is_usable_without_a_tracer_provider() -> None:
    """Tracing must cost nothing when nobody is collecting.

    It is entered once per leader phase, so a version that raised or allocated
    without an exporter configured would tax every cycle.
    """

    with trace_span("eyvan.test", phase="unit"):
        pass
