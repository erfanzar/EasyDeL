# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Health checking with a third answer, and why two answers is the bug.

A service was a registered address. Nothing proved the address answered, and the
tempting fix -- "probe it, and replace it when the probe fails" -- collapses two
different facts into one. A probe that returns 503 is the replica saying it is
not serving. A probe that times out is *this control plane* saying it could not
reach the replica, which is a statement about a network and not about a process.

Acting on the second as if it were the first means a DNS blip, a saturated
egress link or a partition deletes healthy accelerators -- and does so fleet-wide
and simultaneously, because whatever broke reachability broke it for every
replica at once. So the outcome has three values and only one of them can ever
end in a teardown.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import (
    CommandExecution,
    Principal,
    Role,
    RoleGrant,
    ServiceProbeOutcome,
    ServiceReplicaHealthRecord,
    ServiceReplicaHealthState,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.controller.services import HttpServiceHealthProbe
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import (
    CHILD_REVISION,
    ServicePolicy,
    ServiceProbeResult,
    apply_probe,
    probe_from_response,
    probe_unreachable,
    replica_health_state,
    reported_load,
)

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
HEALTHY = ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_failure_threshold=3)


def _replica(identifier: str = "replica-1", *, generation: int = 1) -> WorkloadRecord:
    return WorkloadRecord(
        id=identifier,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name=identifier,
            execution=CommandExecution(command="python serve.py"),
            metadata={CHILD_REVISION: "v1"},
        ),
        state=WorkloadState.RUNNING,
        generation=generation,
        service_endpoint="http://10.0.0.1:8080",
        created_by="alice",
    )


def _fold(*results: ServiceProbeResult, generation: int = 1) -> ServiceReplicaHealthRecord:
    """Fold a sequence of probe results into one replica's durable counters."""

    record: ServiceReplicaHealthRecord | None = None
    for index, result in enumerate(results):
        record = apply_probe(
            record,
            result,
            service_id="svc-1",
            replica_id="replica-1",
            replica_generation=generation,
            endpoint="http://10.0.0.1:8080",
            now=NOW + timedelta(seconds=index),
        )
    assert record is not None
    return record


# --------------------------------------------------------------------------- classifying


def test_a_completed_error_response_is_a_definite_failure() -> None:
    """The replica was reached, asked, and answered that it is not serving."""

    result = probe_from_response(503, "")

    assert result.outcome is ServiceProbeOutcome.FAILURE
    assert "503" in result.detail


def test_an_unreachable_replica_is_unknown_rather_than_failed() -> None:
    """The whole point. A timeout says nothing about the process."""

    result = probe_unreachable("ReadTimeout: timed out after 2.0s")

    assert result.outcome is ServiceProbeOutcome.UNKNOWN
    assert result.outcome is not ServiceProbeOutcome.FAILURE


def test_a_two_hundred_is_a_success() -> None:
    assert probe_from_response(200, "").outcome is ServiceProbeOutcome.SUCCESS


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        ('{"load": 0.5}', 0.5),
        ('{"load": 0}', 0.0),
        ('{"load": 1}', 1.0),
        ("", None),
        ("not json", None),
        ("[]", None),
        ("{}", None),
        ('{"load": "0.5"}', None),
        ('{"load": true}', None),
        ('{"load": 1.5}', None),
        ('{"load": -0.1}', None),
    ),
)
def test_a_load_that_was_not_reported_is_absent_rather_than_zero(body: str, expected: float | None) -> None:
    """Zero is a load. "Did not say" is not, and defaulting it to zero would
    hand the autoscaler a reason to scale in exactly when it cannot see."""

    assert reported_load(body) == expected


# --------------------------------------------------------------------------- folding


def test_a_definite_failure_run_is_counted() -> None:
    record = _fold(probe_from_response(503, ""), probe_from_response(500, ""))

    assert record.consecutive_failures == 2
    assert record.last_outcome is ServiceProbeOutcome.FAILURE
    assert record.last_failure_at == NOW + timedelta(seconds=1)


def test_a_success_clears_the_failure_run_and_records_when() -> None:
    record = _fold(probe_from_response(503, ""), probe_from_response(503, ""), probe_from_response(200, ""))

    assert record.consecutive_failures == 0
    assert record.last_success_at == NOW + timedelta(seconds=2)
    # The last definite failure is still on the record: an operator asking "has
    # this ever misbehaved" gets a real answer.
    assert record.last_failure_at == NOW + timedelta(seconds=1)


def test_an_unknown_probe_neither_raises_nor_clears_the_failure_run() -> None:
    """Raising it records "I could not reach this" as "this said no". Clearing
    it grants a recovery the endpoint never demonstrated. Both are inventions."""

    record = _fold(probe_from_response(503, ""), probe_unreachable("ReadTimeout"))

    assert record.consecutive_failures == 1
    assert record.consecutive_unknowns == 1
    assert record.last_outcome is ServiceProbeOutcome.UNKNOWN
    assert record.last_success_at is None


def test_counters_do_not_carry_across_a_replica_generation() -> None:
    """A requeued replica is a different process on a different host.

    Carrying its predecessor's failures forward would let a machine that has
    never answered a probe be declared unhealthy on evidence about a machine
    that no longer exists.
    """

    failing = _fold(*(probe_from_response(503, "") for _ in range(5)))

    fresh = apply_probe(
        failing,
        probe_unreachable("ConnectError"),
        service_id="svc-1",
        replica_id="replica-1",
        replica_generation=failing.replica_generation + 1,
        endpoint="http://10.0.0.2:8080",
        now=NOW,
    )

    assert fresh.consecutive_failures == 0
    assert fresh.replica_generation == failing.replica_generation + 1


def test_a_success_without_a_reported_load_keeps_the_last_one_and_its_age() -> None:
    """Dropping it would make every silent success look like a fresh unknown;
    replacing its timestamp would make a stale number look current."""

    first = _fold(probe_from_response(200, '{"load": 0.9}'))
    second = apply_probe(
        first,
        probe_from_response(200, ""),
        service_id="svc-1",
        replica_id="replica-1",
        replica_generation=1,
        endpoint="http://10.0.0.1:8080",
        now=NOW + timedelta(seconds=30),
    )

    assert second.observed_load == 0.9
    assert second.load_observed_at == NOW


# --------------------------------------------------------------------------- the verdict


def test_a_bounded_run_of_definite_failures_is_unhealthy_and_a_shorter_one_is_not() -> None:
    replica = _replica()
    two = _fold(*(probe_from_response(503, "") for _ in range(2)))
    three = _fold(*(probe_from_response(503, "") for _ in range(3)))

    assert replica_health_state(replica, two, HEALTHY) is ServiceReplicaHealthState.UNKNOWN
    assert replica_health_state(replica, three, HEALTHY) is ServiceReplicaHealthState.UNHEALTHY


def test_an_unreachable_replica_never_becomes_unhealthy_however_long_it_lasts() -> None:
    """Twenty unknowns in a row is twenty times nothing."""

    replica = _replica()
    record = _fold(*(probe_unreachable("ReadTimeout") for _ in range(20)))

    assert record.consecutive_unknowns == 20
    assert replica_health_state(replica, record, HEALTHY) is ServiceReplicaHealthState.UNKNOWN


def test_a_flapping_probe_never_reaches_the_threshold() -> None:
    """Failure, success, failure, success is a service with a problem, not one
    to tear down: every success proves the process is still there."""

    replica = _replica()
    flapping: list[ServiceProbeResult] = []
    for _ in range(10):
        flapping.append(probe_from_response(503, ""))
        flapping.append(probe_from_response(200, ""))
    record = _fold(*flapping)

    assert record.consecutive_failures == 0
    assert replica_health_state(replica, record, HEALTHY) is ServiceReplicaHealthState.HEALTHY


def test_observations_from_an_earlier_generation_read_as_unknown() -> None:
    replica = _replica(generation=4)
    stale = _fold(*(probe_from_response(503, "") for _ in range(9)), generation=3)

    assert replica_health_state(replica, stale, HEALTHY) is ServiceReplicaHealthState.UNKNOWN


def test_a_service_nobody_probes_has_no_health_verdict() -> None:
    """A record left over from when a path was configured must not outlive it."""

    replica = _replica()
    record = _fold(*(probe_from_response(503, "") for _ in range(9)))
    unprobed = ServicePolicy(replicas=1, revision="v1")

    assert replica_health_state(replica, record, unprobed) is ServiceReplicaHealthState.UNKNOWN


def test_a_replica_nobody_has_probed_yet_is_unknown() -> None:
    assert replica_health_state(_replica(), None, HEALTHY) is ServiceReplicaHealthState.UNKNOWN


# --------------------------------------------------------------------------- the phase


class _ScriptedProbe:
    """A probe that answers from a script and records what it was asked."""

    def __init__(self, *results: ServiceProbeResult) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str, float]] = []

    async def __call__(self, *, endpoint: str, path: str, timeout_s: float) -> ServiceProbeResult:
        self.calls.append((endpoint, path, timeout_s))
        return self._results[min(len(self.calls) - 1, len(self._results) - 1)]


async def _service_plane(policy: ServicePolicy, **settings: object):
    """A control plane holding exactly one service."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings(**settings))  # type: ignore[arg-type]
    operator = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    record, _ = await service.submit_service(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="api", execution=CommandExecution(command="python serve.py")),
        policy=policy,
        idempotency_key="svc",
    )
    return controller, service, record, operator


async def _running_replica(controller: EyvanController, service_id: str, endpoint: str) -> str:
    """Start one replica and advance it to RUNNING with a registered endpoint."""

    await controller.reconcile_services_once()
    async with controller.repository.transaction() as transaction:
        children = await transaction.list_child_workloads(service_id)
        replica = next(child for child in children if child.state is WorkloadState.PENDING)
        await transaction.put_workload(
            replica.model_copy(update={"state": WorkloadState.RUNNING, "service_endpoint": endpoint})
        )
    return replica.id


async def _health(controller: EyvanController, service_id: str) -> list[ServiceReplicaHealthRecord]:
    async with controller.repository.transaction() as transaction:
        return await transaction.list_service_replica_health(service_id)


async def test_health_checking_is_off_unless_an_operator_turns_it_on() -> None:
    """Probing makes a replica replaceable for being unhealthy, so switching it
    on by default would change what an upgrade does to a running fleet."""

    defaults = EyvanSettings()
    controller, _service, _record, _operator = await _service_plane(ServicePolicy(replicas=1, revision="v1"))
    phase = next(entry for entry in controller._leader_phases() if entry.name == "probe-service-health")

    assert defaults.service_health_interval_s == 0.0
    assert (phase.zero_interval_disables, phase.requires) == (True, ())
    assert phase.interval_s == 0.0


async def test_a_service_with_no_health_path_is_never_probed() -> None:
    """The per-service gate, independent of the phase interval."""

    controller, _service, record, _operator = await _service_plane(ServicePolicy(replicas=1, revision="v1"))
    await _running_replica(controller, record.id, "http://10.0.0.1:8080")
    probe = _ScriptedProbe(probe_from_response(200, ""))
    controller.service_health_probe = probe

    assert await controller.probe_service_health_once() == 0
    assert probe.calls == []
    assert await _health(controller, record.id) == []


async def test_a_probe_result_is_recorded_durably_against_the_replica_it_answered() -> None:
    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz", health_timeout_s=1.5)
    )
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.1:8080")
    controller.service_health_probe = _ScriptedProbe(probe_from_response(200, '{"load": 0.42}'))

    assert await controller.probe_service_health_once() == 1

    (stored,) = await _health(controller, record.id)
    assert (stored.service_id, stored.replica_id) == (record.id, replica_id)
    assert stored.last_outcome is ServiceProbeOutcome.SUCCESS
    assert stored.observed_load == 0.42
    assert stored.endpoint == "http://10.0.0.1:8080"
    assert stored.last_success_at is not None


async def test_the_probe_is_asked_for_the_configured_path_and_timeout() -> None:
    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/ready", health_timeout_s=0.25)
    )
    await _running_replica(controller, record.id, "http://10.0.0.1:8080")
    probe = _ScriptedProbe(probe_from_response(200, ""))
    controller.service_health_probe = probe

    await controller.probe_service_health_once()

    assert probe.calls == [("http://10.0.0.1:8080", "/ready", 0.25)]


async def test_an_unreachable_replica_is_recorded_as_unknown_rather_than_failed() -> None:
    """Recorded, but with no effect on the failure run the teardown reads."""

    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    await _running_replica(controller, record.id, "http://10.0.0.1:8080")
    controller.service_health_probe = _ScriptedProbe(probe_unreachable("ReadTimeout"))

    await controller.probe_service_health_once()

    (stored,) = await _health(controller, record.id)
    assert stored.last_outcome is ServiceProbeOutcome.UNKNOWN
    assert (stored.consecutive_failures, stored.consecutive_unknowns) == (0, 1)


async def test_a_probe_that_raises_is_an_unknown_and_not_a_lost_pass() -> None:
    """A transport that throws is still only a statement about reachability."""

    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    await _running_replica(controller, record.id, "http://10.0.0.1:8080")

    async def exploding(**_kwargs: object) -> ServiceProbeResult:
        raise RuntimeError("the probe itself is broken")

    controller.service_health_probe = exploding  # type: ignore[assignment]

    assert await controller.probe_service_health_once() == 1

    (stored,) = await _health(controller, record.id)
    assert stored.last_outcome is ServiceProbeOutcome.UNKNOWN
    assert stored.consecutive_failures == 0


async def test_an_answer_about_a_superseded_generation_is_discarded() -> None:
    """The replica was requeued while the probe was in flight.

    Attributing the old process's answer to the new one is how a machine that
    has never been probed is declared unhealthy on somebody else's evidence.
    """

    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.1:8080")

    async def requeue_then_answer(**_kwargs: object) -> ServiceProbeResult:
        async with controller.repository.transaction() as transaction:
            replica = await transaction.get_workload(replica_id)
            assert replica is not None
            await transaction.put_workload(replica.model_copy(update={"generation": replica.generation + 1}))
        return probe_from_response(503, "")

    controller.service_health_probe = requeue_then_answer  # type: ignore[assignment]

    assert await controller.probe_service_health_once() == 0
    assert await _health(controller, record.id) == []


async def test_the_probe_runs_outside_every_database_transaction() -> None:
    """A probe held inside a transaction holds a row lock for its whole timeout,
    against a machine that may be exactly the unreachable one -- so the leader
    cycle stalls for the timeout and every phase behind it stops."""

    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    await _running_replica(controller, record.id, "http://10.0.0.1:8080")
    reentered = False

    async def probe_that_reads_the_database(**_kwargs: object) -> ServiceProbeResult:
        nonlocal reentered

        async def read() -> None:
            async with controller.repository.transaction() as transaction:
                await transaction.list_workloads()

        await asyncio.wait_for(read(), timeout=2.0)
        reentered = True
        return probe_from_response(200, "")

    controller.service_health_probe = probe_that_reads_the_database  # type: ignore[assignment]

    await controller.probe_service_health_once()

    assert reentered, "the probe could not open a transaction, so one was held across it"


async def test_health_is_forgotten_when_its_replica_is() -> None:
    """A service is the one workload shape designed to restart, so keeping a row
    per dead replica is unbounded growth by construction."""

    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    replica_id = await _running_replica(controller, record.id, "http://10.0.0.1:8080")
    controller.service_health_probe = _ScriptedProbe(probe_from_response(200, ""))
    await controller.probe_service_health_once()
    assert len(await _health(controller, record.id)) == 1

    async with controller.repository.transaction() as transaction:
        replica = await transaction.get_workload(replica_id)
        assert replica is not None
        await transaction.put_workload(replica.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    await controller.probe_service_health_once()

    assert await _health(controller, record.id) == []


async def test_a_replica_that_has_not_registered_is_not_probed() -> None:
    """RUNNING says a process started; registration says it answered."""

    controller, _service, record, _operator = await _service_plane(
        ServicePolicy(replicas=1, revision="v1", health_path="/healthz")
    )
    await controller.reconcile_services_once()
    async with controller.repository.transaction() as transaction:
        (replica,) = await transaction.list_child_workloads(record.id)
        await transaction.put_workload(replica.model_copy(update={"state": WorkloadState.RUNNING}))
    probe = _ScriptedProbe(probe_from_response(200, ""))
    controller.service_health_probe = probe

    assert await controller.probe_service_health_once() == 0
    assert probe.calls == []


async def test_the_reported_health_keeps_unknown_distinguishable_from_unhealthy() -> None:
    """The operator-facing read must not collapse what the record separates."""

    controller, service, record, operator = await _service_plane(
        ServicePolicy(replicas=2, revision="v1", health_path="/healthz", health_failure_threshold=2)
    )
    first = await _running_replica(controller, record.id, "http://10.0.0.1:8080")
    second = await _running_replica(controller, record.id, "http://10.0.0.2:8080")

    async def by_endpoint(*, endpoint: str, path: str, timeout_s: float) -> ServiceProbeResult:
        del path, timeout_s
        return probe_from_response(503, "") if endpoint.endswith("1:8080") else probe_unreachable("ReadTimeout")

    controller.service_health_probe = by_endpoint  # type: ignore[assignment]
    await controller.probe_service_health_once()
    await controller.probe_service_health_once()

    reported = {entry["workload_id"]: entry for entry in await service.service_health(operator, record.id)}
    assert reported[first]["health"] == "unhealthy"
    assert reported[first]["consecutive_failures"] == 2
    assert reported[second]["health"] == "unknown"
    assert reported[second]["consecutive_unknowns"] == 2
    assert reported[second]["consecutive_failures"] == 0


async def test_the_http_probe_turns_every_transport_failure_into_an_unknown() -> None:
    """Nothing is listening on the port, which is still not the replica saying no."""

    probe = HttpServiceHealthProbe()

    result = await probe(endpoint="http://127.0.0.1:1", path="/healthz", timeout_s=0.25)

    assert result.outcome is ServiceProbeOutcome.UNKNOWN
    assert result.detail


async def test_the_http_probe_classifies_a_real_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed exchange is the only thing that can be a definite verdict."""

    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://10.0.0.1:8080/healthz"
        return httpx.Response(503, json={"load": 0.9})

    original = httpx.AsyncClient

    def transported(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", transported)

    result = await HttpServiceHealthProbe()(endpoint="http://10.0.0.1:8080/", path="/healthz", timeout_s=1.0)

    assert result.outcome is ServiceProbeOutcome.FAILURE
    # A failing replica's own load is not a load signal: it is a number from a
    # process that just said it is not serving.
    assert result.load is None
