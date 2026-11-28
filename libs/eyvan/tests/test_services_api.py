# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A managed service an operator can actually read, over HTTP and the CLI.

Health probing, generation-fenced replacement and bounded autoscaling all
decide durably and write down why. Until these routes existed the only readers
were a SQL client and the raw event stream -- so the third health answer, the
one the whole design exists to keep, was invisible to everybody who was not
already inside the database.

The hazard the tests below aim at is the last hop, not the first. ``UNKNOWN``
survives the probe, survives the record and survives the service read; a
transport model with two states, or a summary with one count, would collapse it
at the point where an operator finally sees it -- and an operator who reads
"unhealthy" for a replica nobody could reach will go and delete an accelerator
that was serving fine.
"""

from __future__ import annotations

import asyncio
import json
import os
import typing as tp
from dataclasses import dataclass

import httpx
import pytest
from click.testing import CliRunner
from fastapi import FastAPI

import eyvan.sdk.cli as cli
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_value
from eyvan.config import EyvanSettings
from eyvan.records import (
    CommandExecution,
    Principal,
    Role,
    RoleGrant,
    ServiceObservation,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.api.app import create_app
from eyvan.server.api.schemas import ServiceHealthResponse, ServiceReplicaHealth
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.services import (
    ServicePolicy,
    ServiceProbeResult,
    probe_from_response,
    probe_unreachable,
)

HEALTH_PATH = "/api/v1/workloads/{service_id}/service-health"
DECISIONS_PATH = "/api/v1/workloads/{service_id}/service-decisions"


@pytest.fixture(autouse=True)
def _owned_artifact_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All application constructors in this module use test-owned artifact storage."""

    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "managed-artifacts"))


def _operator(project: str = "research") -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id=project),),
    )


def _headers(project: str = "research", *, role: str = "operator") -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": project,
        "X-Eyvan-Roles": role,
    }


@dataclass(slots=True)
class Plane:
    """One control plane holding exactly one managed service."""

    repository: InMemoryRepository
    service: ControlPlaneService
    controller: EyvanController
    record: WorkloadRecord
    operator: Principal
    artifact_store_uri: str
    _app: FastAPI | None = None

    def app(self) -> FastAPI:
        """Return the one HTTP application over this plane's durable state.

        Built once: ``create_app`` binds the managed-artifact store onto the
        injected service, so a second application over the same service is
        refused as a split store rather than being a second view of one plane.
        """

        if self._app is None:
            self._app = create_app(
                settings=EyvanSettings(dev_auth_enabled=True, artifact_store_uri=self.artifact_store_uri),
                repository=self.repository,
                service=self.service,
            )
        return self._app


async def _plane(policy: ServicePolicy, *, name: str = "api") -> Plane:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings())  # type: ignore[arg-type]
    operator = _operator()
    record, _replayed = await service.submit_service(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name=name, execution=CommandExecution(command="python serve.py")),
        policy=policy,
        idempotency_key=f"svc-{name}",
    )
    return Plane(
        repository=repository,
        service=service,
        controller=controller,
        record=record,
        operator=operator,
        artifact_store_uri=os.environ["EYVAN_ARTIFACT_STORE_URI"],
    )


async def _running_replica(plane: Plane, endpoint: str) -> str:
    """Start one replica and advance it to RUNNING with a registered endpoint."""

    await plane.controller.reconcile_services_once()
    async with plane.repository.transaction() as transaction:
        children = await transaction.list_child_workloads(plane.record.id)
        replica = next(child for child in children if child.state is WorkloadState.PENDING)
        await transaction.put_workload(
            replica.model_copy(update={"state": WorkloadState.RUNNING, "service_endpoint": endpoint})
        )
    return replica.id


async def _get(plane: Plane, path: str, *, headers: dict[str, str] | None = None, **params: object) -> httpx.Response:
    """Issue one authenticated read against this plane's HTTP application."""

    transport = httpx.ASGITransport(app=plane.app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, headers=headers or _headers(), params=params or None)


class _ByEndpoint:
    """A probe that answers differently per replica address."""

    def __init__(self, answers: dict[str, ServiceProbeResult]) -> None:
        self._answers = answers

    async def __call__(self, *, endpoint: str, path: str, timeout_s: float) -> ServiceProbeResult:
        del path, timeout_s
        return self._answers[endpoint]


# --------------------------------------------------------------------------- health over HTTP


async def test_an_unknown_replica_reaches_the_operator_as_unknown_and_not_as_unhealthy() -> None:
    """The route must not collapse the third answer into the negative one.

    One replica answers 503 twice, which is the process saying it is not
    serving. The other times out twice, which is this control plane saying it
    could not reach the process. They must arrive as different words and in
    different counts, because they justify different actions -- one is a reason
    to replace an accelerator and the other is a reason to wait.
    """

    plane = await _plane(ServicePolicy(replicas=2, revision="v1", health_path="/healthz", health_failure_threshold=2))
    refusing = await _running_replica(plane, "http://10.0.0.1:8080")
    unreachable = await _running_replica(plane, "http://10.0.0.2:8080")
    plane.controller.service_health_probe = _ByEndpoint(  # type: ignore[assignment]
        {
            "http://10.0.0.1:8080": probe_from_response(503, ""),
            "http://10.0.0.2:8080": probe_unreachable("ReadTimeout: timed out after 2.0s"),
        }
    )
    await plane.controller.probe_service_health_once()
    await plane.controller.probe_service_health_once()

    response = await _get(plane, HEALTH_PATH.format(service_id=plane.record.id))

    assert response.status_code == 200, response.text
    body = response.json()
    reported = {entry["workload_id"]: entry for entry in body["replicas"]}
    assert reported[refusing]["health"] == "unhealthy"
    assert reported[unreachable]["health"] == "unknown"
    assert reported[unreachable]["health"] != reported[refusing]["health"]
    assert reported[unreachable]["last_outcome"] == "unknown"
    assert reported[unreachable]["consecutive_unknowns"] == 2
    # The counter the teardown reads. An unknown probe never raises it, and the
    # wire must not invent a failure the record refused to record.
    assert reported[unreachable]["consecutive_failures"] == 0
    assert (body["healthy"], body["unhealthy"], body["unknown"]) == (0, 1, 1)


async def test_an_unknown_replica_is_never_summarized_away_to_zero() -> None:
    """Every replica is counted exactly once, so nobody disappears.

    A summary that reported only healthy and unhealthy would show ``0 / 0`` for
    a fleet nobody could reach -- which reads as an empty service rather than as
    a control plane that has lost sight of three running ones.
    """

    plane = await _plane(ServicePolicy(replicas=3, revision="v1", health_path="/healthz"))
    endpoints = [f"http://10.0.0.{index}:8080" for index in (1, 2, 3)]
    replicas = [await _running_replica(plane, endpoint) for endpoint in endpoints]
    plane.controller.service_health_probe = _ByEndpoint(  # type: ignore[assignment]
        {endpoint: probe_unreachable("ConnectTimeout") for endpoint in endpoints}
    )
    await plane.controller.probe_service_health_once()

    body = (await _get(plane, HEALTH_PATH.format(service_id=plane.record.id))).json()

    assert body["unknown"] == 3
    assert (body["healthy"], body["unhealthy"]) == (0, 0)
    assert sorted(entry["workload_id"] for entry in body["replicas"]) == sorted(replicas)
    assert body["healthy"] + body["unhealthy"] + body["unknown"] == len(body["replicas"])
    # A load nobody reported is absent, never 0.0: a reader cannot tell an
    # invented zero from a measured one, and an autoscaler multiplies it.
    assert [entry["observed_load"] for entry in body["replicas"]] == [None, None, None]


async def test_a_service_with_no_health_path_reads_as_not_probed_rather_than_failing() -> None:
    """No probe was configured, so nothing was asked and nothing failed.

    Every service registered before health checking existed is in exactly this
    state. The route answers 200 and says so, rather than erroring or reporting
    replicas that never had a chance to pass.
    """

    plane = await _plane(ServicePolicy(replicas=1, revision="v1"))
    replica = await _running_replica(plane, "http://10.0.0.1:8080")
    assert await plane.controller.probe_service_health_once() == 0

    response = await _get(plane, HEALTH_PATH.format(service_id=plane.record.id))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["probing_enabled"] is False
    assert body["health_path"] == ""
    (entry,) = body["replicas"]
    assert entry["workload_id"] == replica
    assert entry["probed"] is False
    assert entry["health"] == "unknown"
    assert entry["last_outcome"] is None
    assert (body["unhealthy"], body["unknown"]) == (0, 1)


async def test_the_reported_declared_count_is_the_policy_and_not_the_replica_list() -> None:
    """A service mid-reconcile has fewer live replicas than it declares.

    Reporting ``len(replicas)`` as the desired size would make a service that
    lost two hosts look correctly sized and healthy.
    """

    plane = await _plane(ServicePolicy(replicas=3, revision="v7", health_path="/healthz"))
    await _running_replica(plane, "http://10.0.0.1:8080")

    body = (await _get(plane, HEALTH_PATH.format(service_id=plane.record.id))).json()

    assert body["declared_replicas"] == 3
    assert body["revision"] == "v7"
    assert body["health_path"] == "/healthz"
    assert len(body["replicas"]) == 1


async def test_the_wire_view_carries_every_field_the_durable_read_reports() -> None:
    """A field added to the service read and not to the schema is a silent loss.

    The transport model forbids unknown keys, so this pins both directions: a
    new durable fact fails loudly here rather than becoming one more thing only
    SQL can see, which is the defect this whole surface exists to remove.
    """

    plane = await _plane(ServicePolicy(replicas=1, revision="v1", health_path="/healthz"))
    await _running_replica(plane, "http://10.0.0.1:8080")
    plane.controller.service_health_probe = _ByEndpoint(  # type: ignore[assignment]
        {"http://10.0.0.1:8080": probe_from_response(200, '{"load": 0.42}')}
    )
    await plane.controller.probe_service_health_once()

    (entry,) = await plane.service.service_health(plane.operator, plane.record.id)
    body = (await _get(plane, HEALTH_PATH.format(service_id=plane.record.id))).json()

    assert set(ServiceReplicaHealth.model_fields) == set(entry) | {"probed"}
    (reported,) = body["replicas"]
    assert reported["health"] == "healthy"
    assert reported["probed"] is True
    assert reported["observed_load"] == 0.42
    assert (body["healthy"], body["unhealthy"], body["unknown"]) == (1, 0, 0)


async def test_a_workload_that_is_not_a_service_is_refused_rather_than_reported_empty() -> None:
    """ "No replicas" and "not the kind of thing that has replicas" differ.

    Returning an empty document for an ordinary workload would make a mistyped
    identifier read as a perfectly healthy service with nothing in it.
    """

    plane = await _plane(ServicePolicy(replicas=1, revision="v1"))
    ordinary, _replayed = await plane.service.submit_workload(
        plane.operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        idempotency_key="ordinary",
    )

    health = await _get(plane, HEALTH_PATH.format(service_id=ordinary.id))
    decisions = await _get(plane, DECISIONS_PATH.format(service_id=ordinary.id))

    assert health.status_code == 404, health.text
    assert "not a managed service" in health.text
    assert decisions.status_code == 404, decisions.text


# --------------------------------------------------------------------------- decisions over HTTP


async def _scale(plane: Plane, *, to_replicas: int, reason: str, observed: ServiceObservation) -> None:
    """Commit one explained scaling decision against the current service."""

    async with plane.repository.transaction() as transaction:
        current = await transaction.get_workload(plane.record.id)
    assert current is not None
    decision = await plane.service.scale_service(
        plane.operator,
        service_id=plane.record.id,
        expected_generation=current.generation,
        from_replicas=int(current.spec.metadata["eyvan.service/replicas"]),
        to_replicas=to_replicas,
        reason=reason,
        observed=observed,
    )
    assert decision is not None


def _observation(**overrides: object) -> ServiceObservation:
    values: dict[str, object] = {
        "live_replicas": 3,
        "healthy_replicas": 1,
        "unhealthy_replicas": 0,
        "unknown_replicas": 2,
        "loaded_replicas": 0,
    }
    values.update(overrides)
    return ServiceObservation.model_validate(values)


async def test_a_decision_carries_what_was_observed_and_why_over_http() -> None:
    """The audit is only useful if it reconstructs the call after the fact."""

    plane = await _plane(ServicePolicy(replicas=2, revision="v1", health_path="/healthz"))
    await _scale(
        plane,
        to_replicas=3,
        reason="mean load 0.91 over 3 reporting replicas exceeded scale_out_load 0.8",
        observed=_observation(
            live_replicas=2,
            healthy_replicas=2,
            unknown_replicas=0,
            loaded_replicas=2,
            average_load=0.91,
            oldest_load_observed_at="2026-08-04T12:00:00Z",
        ),
    )

    response = await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id))

    assert response.status_code == 200, response.text
    (decision,) = response.json()
    assert decision["kind"] == "scale_out"
    assert (decision["from_replicas"], decision["to_replicas"]) == (2, 3)
    assert decision["reason"] == "mean load 0.91 over 3 reporting replicas exceeded scale_out_load 0.8"
    assert decision["observed"]["average_load"] == 0.91
    assert decision["observed"]["healthy_replicas"] == 2


async def test_the_decision_history_keeps_unknown_replicas_out_of_the_unhealthy_count() -> None:
    """The observation a decision was taken from separates them too."""

    plane = await _plane(ServicePolicy(replicas=2, revision="v1", health_path="/healthz"))
    await _scale(
        plane,
        to_replicas=1,
        reason="two replicas unreachable; declining to count them as failed",
        observed=_observation(live_replicas=3, healthy_replicas=1, unhealthy_replicas=0, unknown_replicas=2),
    )

    (decision,) = (await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id))).json()

    assert decision["observed"]["unknown_replicas"] == 2
    assert decision["observed"]["unhealthy_replicas"] == 0


async def test_the_decision_history_is_newest_first_and_bounded_by_the_limit() -> None:
    """An operator asking for the last decision must not get the first one."""

    plane = await _plane(ServicePolicy(replicas=1, revision="v1"))
    for index, size in enumerate((2, 3, 4), start=1):
        await _scale(plane, to_replicas=size, reason=f"step {index}", observed=_observation())

    everything = (await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id))).json()
    newest = (await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id), limit=1)).json()

    assert [entry["sequence"] for entry in everything] == [3, 2, 1]
    assert [entry["reason"] for entry in newest] == ["step 3"]


async def test_a_limit_outside_the_bounds_is_refused_rather_than_clamped() -> None:
    """Silently returning 50 for a request for 100000 is an answer to a
    different question than the one asked."""

    plane = await _plane(ServicePolicy(replicas=1, revision="v1"))

    assert (await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id), limit=0)).status_code == 422
    assert (await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id), limit=100000)).status_code == 422


# --------------------------------------------------------------------------- tenancy


async def test_another_tenant_is_refused_both_reads() -> None:
    """A service's health names its replicas, its addresses and its size.

    Read authorization is on the workload's own project, so a principal holding
    a grant on a different project of the same organization is refused before
    any of that is disclosed.
    """

    plane = await _plane(ServicePolicy(replicas=1, revision="v1", health_path="/healthz"))
    replica = await _running_replica(plane, "http://10.0.0.1:8080")
    await _scale(plane, to_replicas=2, reason="secret capacity plan", observed=_observation())
    intruder = _headers(project="marketing")

    health = await _get(plane, HEALTH_PATH.format(service_id=plane.record.id), headers=intruder)
    decisions = await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id), headers=intruder)

    assert health.status_code == 403, health.text
    assert decisions.status_code == 403, decisions.text
    assert replica not in health.text
    assert "10.0.0.1" not in health.text
    assert "secret capacity plan" not in decisions.text


async def test_a_stranger_is_refused_before_the_workload_shape_is_disclosed() -> None:
    """The refusal must not double as an oracle for what a workload is.

    If the "not a managed service" answer came first, a principal with no read
    on the project could sweep identifiers and learn which of another tenant's
    workloads are services -- the shape of their deployment, from a route that
    never returned any of their data.
    """

    plane = await _plane(ServicePolicy(replicas=1, revision="v1"))
    ordinary, _replayed = await plane.service.submit_workload(
        plane.operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        idempotency_key="ordinary-oracle",
    )
    intruder = _headers(project="marketing")

    ordinary_probe = await _get(plane, HEALTH_PATH.format(service_id=ordinary.id), headers=intruder)
    service_probe = await _get(plane, HEALTH_PATH.format(service_id=plane.record.id), headers=intruder)

    assert ordinary_probe.status_code == 403, ordinary_probe.text
    assert service_probe.status_code == 403, service_probe.text
    assert ordinary_probe.json() == service_probe.json()


async def test_the_owning_tenant_is_admitted_to_both_reads() -> None:
    """The refusal above must be about the tenant and not about the routes."""

    plane = await _plane(ServicePolicy(replicas=1, revision="v1", health_path="/healthz"))
    await _running_replica(plane, "http://10.0.0.1:8080")

    health = await _get(plane, HEALTH_PATH.format(service_id=plane.record.id))
    decisions = await _get(plane, DECISIONS_PATH.format(service_id=plane.record.id))

    assert health.status_code == 200, health.text
    assert decisions.status_code == 200, decisions.text
    assert decisions.json() == []


# --------------------------------------------------------------------------- the CLI


class _AsgiCliClient:
    """A CLI transport that speaks to one real application in process.

    Real routing, real dependency resolution, real authorization: the only thing
    replaced is the socket. A fake returning canned documents would pass whether
    or not the routes existed, which is the failure mode this file is written
    against.
    """

    def __init__(self, app: FastAPI, headers: dict[str, str]) -> None:
        self._app = app
        self.headers = headers

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, object] | None = None,
        json: JsonObject | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        async def call() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers={**self.headers, **dict(headers or {})},
                )

        response = asyncio.run(call())
        if response.is_error:
            raise cli.EyvanApiError(response.status_code, response.text)
        payload = response.json()
        if not is_json_value(payload):
            raise AssertionError("the control plane returned a non-JSON value")
        return payload

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        self.headers = dict(headers)


def _invoke(app: FastAPI, *arguments: str) -> tuple[int, str]:
    """Run one CLI command against a real application and return its output."""

    runner = CliRunner()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, "EyvanClient", lambda _url, *, headers: _AsgiCliClient(app, dict(headers)))
        result = runner.invoke(
            cli.main,
            ["--user", "alice", "--org", "acme", "--project", "research", "--role", "operator", *arguments],
        )
    return result.exit_code, result.output


@pytest.fixture(autouse=True)
def _isolated_cli_context(monkeypatch, tmp_path) -> None:
    """Keep the CLI off the developer's real context file."""

    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "eyvan-config.json"))


def test_the_cli_reports_unknown_replicas_as_unknown() -> None:
    """The whole route to an operator, end to end, with nothing collapsed."""

    async def build() -> tuple[FastAPI, str, str]:
        plane = await _plane(
            ServicePolicy(replicas=2, revision="v1", health_path="/healthz", health_failure_threshold=2)
        )
        refusing = await _running_replica(plane, "http://10.0.0.1:8080")
        unreachable = await _running_replica(plane, "http://10.0.0.2:8080")
        plane.controller.service_health_probe = _ByEndpoint(  # type: ignore[assignment]
            {
                "http://10.0.0.1:8080": probe_from_response(503, ""),
                "http://10.0.0.2:8080": probe_unreachable("ReadTimeout"),
            }
        )
        await plane.controller.probe_service_health_once()
        await plane.controller.probe_service_health_once()
        return plane.app(), plane.record.id, f"{refusing}|{unreachable}"

    app, service_id, identifiers = asyncio.run(build())
    refusing, unreachable = identifiers.split("|")

    exit_code, output = _invoke(app, "service", "health", service_id)

    assert exit_code == 0, output
    document = json.loads(output)
    reported = {entry["workload_id"]: entry for entry in document["replicas"]}
    assert reported[unreachable]["health"] == "unknown"
    assert reported[refusing]["health"] == "unhealthy"
    assert (document["healthy"], document["unhealthy"], document["unknown"]) == (0, 1, 1)


def test_the_cli_says_not_probed_for_a_service_with_no_health_path() -> None:
    """The operator-facing answer for every service that predates probing."""

    async def build() -> tuple[FastAPI, str]:
        plane = await _plane(ServicePolicy(replicas=1, revision="v1"))
        await _running_replica(plane, "http://10.0.0.1:8080")
        return plane.app(), plane.record.id

    app, service_id = asyncio.run(build())

    exit_code, output = _invoke(app, "service", "health", service_id)

    assert exit_code == 0, output
    document = json.loads(output)
    assert document["probing_enabled"] is False
    assert [entry["probed"] for entry in document["replicas"]] == [False]
    assert document["unknown"] == 1


def test_a_decision_reason_survives_to_the_cli_output() -> None:
    """What the operator came for: not that it scaled, but why."""

    reason = "mean load 0.12 below scale_in_load 0.3 for 2 consecutive windows"

    async def build() -> tuple[FastAPI, str]:
        plane = await _plane(ServicePolicy(replicas=3, revision="v1", health_path="/healthz"))
        await _scale(
            plane,
            to_replicas=2,
            reason=reason,
            observed=_observation(
                live_replicas=3,
                healthy_replicas=3,
                unknown_replicas=0,
                loaded_replicas=3,
                average_load=0.12,
                oldest_load_observed_at="2026-08-04T12:00:00Z",
            ),
        )
        return plane.app(), plane.record.id

    app, service_id = asyncio.run(build())

    exit_code, output = _invoke(app, "service", "decisions", service_id)

    assert exit_code == 0, output
    assert reason in output
    (decision,) = json.loads(output)
    assert decision["reason"] == reason
    assert decision["kind"] == "scale_in"
    assert decision["observed"]["average_load"] == 0.12


def test_the_cli_limit_reaches_the_route() -> None:
    """The flag has to change the request, not just the help text."""

    async def build() -> tuple[FastAPI, str]:
        plane = await _plane(ServicePolicy(replicas=1, revision="v1"))
        for index, size in enumerate((2, 3), start=1):
            await _scale(plane, to_replicas=size, reason=f"step {index}", observed=_observation())
        return plane.app(), plane.record.id

    app, service_id = asyncio.run(build())

    exit_code, output = _invoke(app, "service", "decisions", service_id, "--limit", "1")

    assert exit_code == 0, output
    assert [entry["reason"] for entry in json.loads(output)] == ["step 2"]


def test_the_cli_refuses_a_limit_the_route_would_reject() -> None:
    """Refused locally, so the user reads a flag error rather than a 422."""

    async def build() -> tuple[FastAPI, str]:
        plane = await _plane(ServicePolicy(replicas=1, revision="v1"))
        return plane.app(), plane.record.id

    app, service_id = asyncio.run(build())

    exit_code, output = _invoke(app, "service", "decisions", service_id, "--limit", "0")

    assert exit_code != 0
    assert "0" in output


def test_the_cli_surfaces_a_refusal_rather_than_printing_an_empty_document() -> None:
    """A tenant who may not read this must not get an empty success."""

    async def build() -> tuple[FastAPI, str]:
        plane = await _plane(ServicePolicy(replicas=1, revision="v1", health_path="/healthz"))
        await _running_replica(plane, "http://10.0.0.1:8080")
        ordinary, _replayed = await plane.service.submit_workload(
            plane.operator,
            org_id="acme",
            project_id="research",
            spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
            idempotency_key="ordinary-cli",
        )
        return plane.app(), ordinary.id

    app, workload_id = asyncio.run(build())

    exit_code, output = _invoke(app, "service", "health", workload_id)

    assert exit_code != 0
    assert "not a managed service" in output


# --------------------------------------------------------------------------- the schema itself


def test_the_health_summary_must_account_for_every_replica() -> None:
    """The counts are the summary, so a wrong one is a wrong answer.

    Building a response whose buckets do not add up is refused rather than
    served: a report claiming one healthy replica out of three, with no word
    about the other two, is worse than no report.
    """

    with pytest.raises(ValueError, match="exactly one health bucket"):
        ServiceHealthResponse(
            service_id="svc-1",
            revision="v1",
            declared_replicas=1,
            healthy=1,
            unhealthy=0,
            unknown=0,
            replicas=[],
        )
