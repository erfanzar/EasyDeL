# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A committed command retains its producer parent across delayed redelivery."""

import pytest
from opentelemetry import baggage, context, trace
from test_group_lease_lifecycle import repository as _repository_fixture
from test_trace_context_transport import telemetry as _telemetry_fixture

from eyvan.config import EyvanSettings
from eyvan.records import OutboxState, utcnow
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.service import ControlPlaneService

repository = _repository_fixture
telemetry = _telemetry_fixture


class DeliveryProbe(EyvanController):
    async def _dispatch(self, command):
        with trace.get_tracer("test").start_as_current_span("delivery.probe") as span:
            self.observed.append((span.get_span_context(), dict(baggage.get_all())))


async def _history(repository, enabled, telemetry):
    service = ControlPlaneService(repository, correlation_enabled=enabled)
    token = context.attach(baggage.set_baggage("private", "not-for-storage"))
    try:
        with trace.get_tracer("test").start_as_current_span("producer") as producer:
            producer_context = producer.get_span_context()
            async with repository.transaction() as tx:
                command = await service._outbox(
                    tx,
                    topic="test.probe",
                    aggregate_type="workload",
                    aggregate_id="wl_trace",
                    idempotency_key="trace",
                    generation=1,
                    payload={"traceparent": "untrusted"},
                )
    finally:
        context.detach(token)
    async with repository.transaction() as tx:
        stored = await tx.get_outbox(command.id)
    wire = stored.model_dump(mode="json")
    parent = wire.get("traceparent")
    assert (parent is not None) is enabled
    if enabled:
        assert parent.split("-")[1] == f"{producer_context.trace_id:032x}"
    assert "baggage" not in wire and "tracestate" not in wire
    assert "not-for-storage" not in stored.model_dump_json()

    for _ in range(2):
        # New controller objects, real durable claiming, and a different ambient
        # trace: delivery must not accidentally inherit the polling loop's trace.
        controller = DeliveryProbe(
            repository, ControlPlaneService(repository), settings=EyvanSettings(correlation_enabled=enabled)
        )
        controller.observed = []
        with trace.get_tracer("test").start_as_current_span("unrelated") as unrelated:
            before = trace.get_current_span().get_span_context()
            assert await controller.dispatch_outbox_once() == 1
            assert trace.get_current_span().get_span_context() == before
        observed, observed_baggage = controller.observed[0]
        if enabled:
            assert observed.trace_id == producer_context.trace_id
            assert observed.trace_id != unrelated.get_span_context().trace_id
            assert parent.split("-")[2] == f"{producer_context.span_id:016x}"
            spans = {span.context.span_id: span for span in telemetry.get_finished_spans()}
            dispatch = spans[spans[observed.span_id].parent.span_id]
            assert dispatch.name == "eyvan.outbox_dispatch"
            operation = spans[dispatch.parent.span_id]
            assert operation.name == "outbox.test.probe"
            assert operation.parent.span_id == producer_context.span_id
        assert observed_baggage == {}
        async with repository.transaction() as tx:
            finished = await tx.get_outbox(command.id)
            assert finished.state is OutboxState.SUCCEEDED
            assert finished.model_dump(mode="json").get("traceparent") == parent
            await tx.put_outbox(finished.model_copy(update={"state": OutboxState.PENDING, "available_at": utcnow()}))


@pytest.mark.parametrize("repository", ["memory", "postgres"], indirect=True)
@pytest.mark.parametrize("enabled", [False, True])
async def test_durable_trace_parent_on_repository(repository, telemetry, enabled):
    await _history(repository, enabled, telemetry)


@pytest.mark.parametrize("enabled", [False, True])
async def test_durable_trace_parent_on_local_repository(tmp_path, telemetry, enabled):
    repository = LocalJsonRepository(tmp_path / "state.json")
    await repository.open()
    try:
        await _history(repository, enabled, telemetry)
    finally:
        await repository.close()


@pytest.mark.parametrize("repository", ["memory"], indirect=True)
async def test_enqueue_does_not_finish_exportable_spans_inside_transaction(repository, telemetry):
    service = ControlPlaneService(repository, correlation_enabled=True)
    with trace.get_tracer("test").start_as_current_span("producer"):
        async with repository.transaction() as tx:
            before = len(telemetry.get_finished_spans())
            await service._outbox(
                tx,
                topic="test.probe",
                aggregate_type="workload",
                aggregate_id="wl_trace",
                idempotency_key="trace",
                generation=1,
                payload={},
            )
            assert len(telemetry.get_finished_spans()) == before


@pytest.mark.parametrize("repository", ["memory"], indirect=True)
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_delivery_restores_context_and_joins_detached_task(repository, telemetry, cancel):
    import asyncio

    started, release = asyncio.Event(), asyncio.Event()
    service = ControlPlaneService(repository, correlation_enabled=True)
    with trace.get_tracer("test").start_as_current_span("producer") as producer:
        async with repository.transaction() as tx:
            command = await service._outbox(
                tx,
                topic="test.probe",
                aggregate_type="workload",
                aggregate_id="wl_trace",
                idempotency_key="trace",
                generation=1,
                payload={},
            )

    class InterruptedDelivery(EyvanController):
        async def _dispatch(self, command):
            self.inner = asyncio.current_task()
            assert trace.get_current_span().get_span_context().trace_id == producer.get_span_context().trace_id
            assert dict(baggage.get_all()) == {}
            started.set()
            await release.wait()
            if not cancel:
                raise RuntimeError("delivery failed")

    controller = InterruptedDelivery(repository, service, settings=EyvanSettings(correlation_enabled=True))
    command = await repository.claim_outbox_id(worker_id=controller.worker_id, command_id=command.id, lease_seconds=90)

    async def deliver():
        token = context.attach(baggage.set_baggage("ambient", "must-stay-local"))
        try:
            with trace.get_tracer("test").start_as_current_span("unrelated") as ambient:
                try:
                    await controller._dispatch_claimed_outbox(command)
                except asyncio.CancelledError:
                    assert cancel
                assert trace.get_current_span().get_span_context() == ambient.get_span_context()
                assert dict(baggage.get_all()) == {"ambient": "must-stay-local"}
        finally:
            context.detach(token)

    outer = asyncio.create_task(deliver())
    try:
        await asyncio.wait_for(started.wait(), 5)
        if cancel:
            outer.cancel()
            await asyncio.wait_for(outer, 5)
            assert not controller.inner.done()
        release.set()
        await asyncio.wait_for(outer, 5)
    finally:
        release.set()
        await asyncio.wait_for(outer, 5)
        if hasattr(controller, "inner"):
            await asyncio.wait_for(asyncio.gather(controller.inner, return_exceptions=True), 5)


@pytest.mark.parametrize("repository", ["memory"], indirect=True)
@pytest.mark.parametrize("parent", [None, "00-" + "0" * 32 + "-" + "0" * 16 + "-01"])
async def test_legacy_or_zero_parent_delivers_without_adopting_ambient_trace(repository, telemetry, parent):
    from eyvan.records import OutboxRecord

    wire = dict(
        id="cmd_legacy",
        topic="test.probe",
        aggregate_type="workload",
        aggregate_id="wl_trace",
        idempotency_key="legacy",
        generation=1,
        payload={},
        correlation_id="wl_trace",
    )
    if parent is not None:
        wire["traceparent"] = parent
    record = OutboxRecord.model_validate(wire)
    assert ("traceparent" in record.model_dump(mode="json")) is (parent is not None)
    async with repository.transaction() as tx:
        await tx.put_outbox(record)
    controller = DeliveryProbe(
        repository, ControlPlaneService(repository), settings=EyvanSettings(correlation_enabled=True)
    )
    controller.observed = []
    with trace.get_tracer("test").start_as_current_span("unrelated") as unrelated:
        assert await controller.dispatch_outbox_once() == 1
    assert controller.observed[0][0].trace_id != unrelated.get_span_context().trace_id
    async with repository.transaction() as tx:
        finished = await tx.get_outbox(record.id)
        assert finished.state is OutboxState.SUCCEEDED
        assert finished.payload == {} and finished.attempts == 1


@pytest.mark.parametrize("repository", ["memory", "postgres"], indirect=True)
@pytest.mark.parametrize("state", ["draining", "deleting"])
@pytest.mark.parametrize("ambient_span", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_stale_lifecycle_replacement_retains_producer_trace(repository, telemetry, state, ambient_span, enabled):
    from contextlib import nullcontext

    from eyvan.records import CellRecord, CellState, ResourceVector
    from eyvan.worker.agent.client import CellAgentStaleMutationError

    class StaleDelivery(DeliveryProbe):
        async def _dispatch(self, command):
            if command.generation == 1:
                raise CellAgentStaleMutationError("stale lifecycle fence")
            await super()._dispatch(command)

    service = ControlPlaneService(repository, correlation_enabled=enabled)
    cell = CellRecord(
        id="cell_trace_recovery",
        provider="manual",
        region="local",
        capacity=ResourceVector(cpu=8),
        allocatable=ResourceVector(cpu=8),
        state=CellState(state),
        generation=1,
    )
    topic = "capacity.drain" if cell.state is CellState.DRAINING else "capacity.release"
    with trace.get_tracer("test").start_as_current_span("producer") as producer:
        producer_context = producer.get_span_context()
        async with repository.transaction() as tx:
            await tx.put_cell(cell)
            command = await service._outbox(
                tx,
                topic=topic,
                aggregate_type="cell",
                aggregate_id=cell.id,
                idempotency_key="trace-recovery",
                generation=1,
                payload={"cell_id": cell.id},
            )

    controller = StaleDelivery(repository, service, settings=EyvanSettings(correlation_enabled=enabled))
    controller.observed = []
    token = context.attach(baggage.set_baggage("ambient", "must-stay-local"))
    try:
        ambient = trace.get_tracer("test").start_as_current_span("unrelated") if ambient_span else nullcontext()
        with ambient:
            before = trace.get_current_span().get_span_context()
            assert await controller.dispatch_outbox_once() == 1
            assert trace.get_current_span().get_span_context() == before
            assert dict(baggage.get_all()) == {"ambient": "must-stay-local"}
    finally:
        context.detach(token)

    async with repository.transaction() as tx:
        original = await tx.get_outbox(command.id)
        assert original.state is OutboxState.DISCARDED
        rows = await tx.list_outbox(limit=100)
        replacements = [row for row in rows if row.id != command.id]
        assert len(replacements) == 1
        replacement = replacements[0]
        assert replacement.topic == topic and replacement.generation == 2
        assert replacement.state is OutboxState.PENDING
        assert (await tx.get_cell(cell.id)).generation == 2
    if enabled:
        assert replacement.traceparent is not None
        assert replacement.traceparent.split("-")[1] == f"{producer_context.trace_id:032x}"
    else:
        assert "traceparent" not in replacement.model_dump(mode="json")
    assert "must-stay-local" not in replacement.model_dump_json()

    # Reclaim the replacement through normal dispatch, not a direct helper call.
    assert await controller.dispatch_outbox_once() == 1
    observed, observed_baggage = controller.observed[0]
    if enabled:
        assert observed.trace_id == producer_context.trace_id
    assert observed_baggage == {}
    async with repository.transaction() as tx:
        finished = await tx.get_outbox(replacement.id)
        assert finished.state is OutboxState.SUCCEEDED
        assert finished.traceparent == replacement.traceparent
