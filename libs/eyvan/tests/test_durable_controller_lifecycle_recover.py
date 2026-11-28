# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Controller lifecycle recovery: bounded endpoint probes, endpoint fencing and replacement revalidation."""

from __future__ import annotations

import asyncio
import socket
from dataclasses import replace

import httpx
import pytest
from test_services_health import _health, _running_replica, _service_plane
from test_services_lifecycle import _cell, _lease

from eyvan.records import ServiceProbeOutcome, ServiceReplicaHealthState, WorkloadState, utcnow
from eyvan.server.controller.services import HttpServiceHealthProbe
from eyvan.server.service.services import (
    ReplicaObservation,
    ServicePolicy,
    apply_probe,
    plan_replica_lifecycle,
    probe_from_response,
    replica_health_state,
    service_metadata,
    service_observation,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network connections forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


class Trickle(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        for _ in range(40):
            await asyncio.sleep(0.005)
            yield b" "

    async def aclose(self) -> None:
        self.closed = True


async def test_total_http_deadline_includes_the_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    body = Trickle()
    original = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    def client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", client)
    result = await HttpServiceHealthProbe()(endpoint="http://10.0.0.1", path="/healthz", timeout_s=0.02)
    assert result.outcome is ServiceProbeOutcome.UNKNOWN
    assert "Timeout" in result.detail
    assert body.closed


async def test_custom_probe_cannot_hold_the_serial_leader_phase() -> None:
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_timeout_s=0.02)
    controller, _service, record, _operator = await _service_plane(policy)
    await _running_replica(controller, record.id, "http://10.0.0.1")

    async def slow(**kwargs: object):
        await asyncio.sleep(0.2)
        return probe_from_response(200, "")

    controller.service_health_probe = slow
    assert await controller.probe_service_health_once() == 1
    (health,) = await _health(controller, record.id)
    assert health.last_outcome is ServiceProbeOutcome.UNKNOWN
    assert health.consecutive_failures == 0


async def test_a_superseded_endpoint_probe_is_not_committed() -> None:
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=1)
    controller, service, record, _operator = await _service_plane(policy)
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.1")

    async def changed(**kwargs: object):
        async with controller.repository.transaction() as tx:
            replica = await tx.get_workload(replica_id)
        await service.register_service_endpoint(workload=replica, endpoint="http://10.0.0.2")
        return probe_from_response(503, "")

    controller.service_health_probe = changed
    assert await controller.probe_service_health_once() == 0
    assert await _health(controller, record.id) == []


async def test_same_generation_endpoint_aba_also_fences_the_old_probe() -> None:
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    controller, service, record, _operator = await _service_plane(policy)
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.1")

    async def changed(**kwargs: object):
        async with controller.repository.transaction() as tx:
            replica = await tx.get_workload(replica_id)
        replica = await service.register_service_endpoint(workload=replica, endpoint="http://10.0.0.2")
        await service.register_service_endpoint(workload=replica, endpoint="http://10.0.0.1")
        return probe_from_response(503, "")

    controller.service_health_probe = changed
    assert await controller.probe_service_health_once() == 0
    assert await _health(controller, record.id) == []


async def test_registration_resets_counters_and_load_but_same_endpoint_retry_does_not() -> None:
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=1)
    controller, service, record, operator = await _service_plane(policy)
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.1")
    async with controller.repository.transaction() as tx:
        replica = await tx.get_workload(replica_id)
        healthy = apply_probe(
            None,
            probe_from_response(200, '{"load": 0.9}'),
            service_id=record.id,
            replica_id=replica_id,
            replica_generation=replica.generation,
            endpoint=replica.service_endpoint,
            now=utcnow(),
        )
        failed = apply_probe(
            healthy,
            probe_from_response(503, ""),
            service_id=record.id,
            replica_id=replica_id,
            replica_generation=replica.generation,
            endpoint=replica.service_endpoint,
            now=utcnow(),
        )
        await tx.put_service_replica_health(failed)
    same = await service.register_service_endpoint(workload=replica, endpoint=replica.service_endpoint)
    assert await _health(controller, record.id) == [failed]
    await service.register_service_endpoint(workload=same, endpoint="http://10.0.0.2")
    assert await _health(controller, record.id) == []
    (reported,) = await service.service_health(operator, record.id)
    assert reported["health"] == "unknown"
    assert reported["consecutive_failures"] == 0
    assert reported["observed_load"] is None


async def test_stale_endpoint_health_is_neither_classified_nor_folded_nor_counted_as_load() -> None:
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=1)
    controller, service, record, operator = await _service_plane(policy)
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.2")
    async with controller.repository.transaction() as tx:
        replica = await tx.get_workload(replica_id)
    stale = apply_probe(
        None,
        probe_from_response(200, '{"load": 0.9}'),
        service_id=record.id,
        replica_id=replica_id,
        replica_generation=replica.generation,
        endpoint="http://10.0.0.1",
        now=utcnow(),
    )
    stale = apply_probe(
        stale,
        probe_from_response(503, ""),
        service_id=record.id,
        replica_id=replica_id,
        replica_generation=replica.generation,
        endpoint="http://10.0.0.1",
        now=utcnow(),
    )
    assert replica_health_state(replica, stale, policy) is ServiceReplicaHealthState.UNKNOWN
    observed = service_observation(policy, [ReplicaObservation(replica=replica, health=stale)])
    assert observed.loaded_replicas == 0
    assert observed.average_load is None
    folded = apply_probe(
        stale,
        probe_from_response(503, ""),
        service_id=record.id,
        replica_id=replica_id,
        replica_generation=replica.generation,
        endpoint=replica.service_endpoint,
        now=utcnow(),
    )
    assert folded.consecutive_failures == 1
    assert folded.observed_load is None
    async with controller.repository.transaction() as tx:
        await tx.put_service_replica_health(stale)
    (reported,) = await service.service_health(operator, record.id)
    assert reported["health"] == "unknown"
    assert reported["consecutive_failures"] == 0
    assert reported["last_outcome"] is None
    assert reported["observed_load"] is None


@pytest.mark.parametrize("recovery", ["health", "lease", "cell", "policy", "revision"])
async def test_replacement_revalidates_current_durable_evidence(recovery: str) -> None:
    policy = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=1)
    controller, service, record, operator = await _service_plane(policy)
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.1")
    async with controller.repository.transaction() as tx:
        replica = await tx.get_workload(replica_id)
        replica = replica.model_copy(update={"lease_id": "lease-1"})
        lease = _lease(workload_id=replica_id)
        cell = _cell()
        health = apply_probe(
            None,
            probe_from_response(503, ""),
            service_id=record.id,
            replica_id=replica_id,
            replica_generation=replica.generation,
            endpoint=replica.service_endpoint,
            now=utcnow(),
        )
        if recovery == "lease":
            from eyvan.records import LeaseState

            lease = lease.model_copy(update={"state": LeaseState.EXPIRED})
            health = None
        elif recovery == "cell":
            cell = cell.model_copy(update={"generation": 2})
            health = None
        await tx.put_workload(replica)
        await tx.put_lease(lease)
        await tx.put_cell(cell)
        if health is not None:
            await tx.put_service_replica_health(health)
    (planned,) = plan_replica_lifecycle(
        policy, [ReplicaObservation(replica=replica, lease=lease, cell=cell, health=health)]
    )
    assert planned.replace
    async with controller.repository.transaction() as tx:
        if recovery == "health":
            await tx.put_service_replica_health(
                apply_probe(
                    health,
                    probe_from_response(200, ""),
                    service_id=record.id,
                    replica_id=replica_id,
                    replica_generation=replica.generation,
                    endpoint=replica.service_endpoint,
                    now=utcnow(),
                )
            )
        elif recovery == "lease":
            await tx.put_lease(_lease(workload_id=replica_id))
        elif recovery == "cell":
            await tx.put_cell(_cell())
        else:
            current = await tx.get_workload(record.id)
            fresh_policy = replace(policy, health_path="") if recovery == "policy" else replace(policy, revision="v2")
            spec = current.spec.model_copy(update={"metadata": service_metadata(fresh_policy)})
            await tx.put_workload(current.model_copy(update={"spec": spec}))
    result = await service.replace_service_replica(
        operator,
        service_id=record.id,
        replica_id=replica_id,
        expected_generation=replica.generation,
        reason=planned.reason,
        observed=service_observation(
            policy, [ReplicaObservation(replica=replica, lease=lease, cell=cell, health=health)]
        ),
    )
    assert result is None
    async with controller.repository.transaction() as tx:
        current = await tx.get_workload(replica_id)
        assert current.state is WorkloadState.RUNNING
        assert await tx.list_service_decisions(record.id) == []
