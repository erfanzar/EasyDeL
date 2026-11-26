# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The half of the progress channel that was missing.

``POST /workloads/{id}/progress-reports``, ``ProgressReportRequest``, the
``UserProgressReport`` record, migration ``0008_progress_reports`` and
``WorkloadProgress.user_progress`` all existed, and the route had zero callers
anywhere: not in the worker, not in the SDK, not in the tests, which drove the
service method directly. Server-complete, client-absent -- so no job could put
anything into the channel and every read of it returned an empty tuple.

Three things make the producer usable rather than merely present: a job learns
the address from its environment rather than assembling a control-plane URL; the
call is a no-op off the fleet so the same callable runs on a laptop; and it
never raises for a transport failure, because a monitoring channel that can
abort the job it monitors is worse than no channel.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import sys
import urllib.error
from unittest import mock
from urllib.parse import urlsplit

import pytest

from eyvan.artifacts.managed import ArtifactTokenSigner
from eyvan.artifacts.transfers import ManagedArtifactAccess
from eyvan.worker.runtime import progress
from eyvan.worker.runtime.progress import PROGRESS_TOKEN_ENV, PROGRESS_URL_ENV, report_progress


def _env(**overrides: str) -> dict[str, str]:
    base = {
        PROGRESS_URL_ENV: "https://plane.example/api/v1/workloads/wl-1/progress-reports?org_id=a&project_id=b",
        PROGRESS_TOKEN_ENV: "token-value",
    }
    base.update(overrides)
    return base


class _Recorder:
    """Capture what the reporter would have sent, without a network."""

    def __init__(self, *, fail: list[Exception] | None = None) -> None:
        self.requests: list[urllib.request.Request] = []  # type: ignore[name-defined]
        self._fail = list(fail or [])

    def __call__(self, request, timeout=None):  # type: ignore[no-untyped-def]
        del timeout
        self.requests.append(request)
        if self._fail:
            raise self._fail.pop(0)

        class _Response:
            def read(self) -> bytes:
                return b"{}"

            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

        return _Response()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the retry backoff from making the suite wait for real seconds."""

    monkeypatch.setattr(progress.time, "sleep", lambda _seconds: None)


def _http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://plane.example",
        code,
        "boom",
        http.client.HTTPMessage() if headers is None else _message(headers),
        None,
    )


def _message(headers: dict[str, str]) -> http.client.HTTPMessage:
    message = http.client.HTTPMessage()
    for name, value in headers.items():
        message[name] = value
    return message


def test_it_is_a_no_op_when_nothing_is_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same callable has to run unchanged on a laptop.

    Off by absence rather than by a flag: with no address in the environment
    there is nothing to report to, and requiring the caller to guard every call
    site would mean the reporting code differs between local and fleet runs --
    which is exactly the code that then only breaks on the fleet.
    """

    recorder = _Recorder()
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    assert report_progress(step=1, total=10, environment={}) is False
    assert recorder.requests == []


def test_a_url_without_a_token_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Half a configuration must not produce an unauthenticated request."""

    recorder = _Recorder()
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    assert report_progress(step=1, environment={PROGRESS_URL_ENV: "https://plane.example/x"}) is False
    assert recorder.requests == []


def test_a_report_carries_the_attempt_token_and_the_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wire shape has to match what ``ProgressReportRequest`` accepts."""

    recorder = _Recorder()
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    assert report_progress(step=3, total=10, message="warming up", environment=_env()) is True

    (request,) = recorder.requests
    assert request.get_method() == "POST"
    assert request.get_header("X-eyvan-artifact-token") == "token-value"
    assert json.loads(request.data) == {
        "rank": 0,
        "attempt": 1,
        "step": 3,
        "total": 10,
        "message": "warming up",
        "metrics": {},
    }


def test_the_rank_comes_from_the_slice_not_the_replica_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EYVAN_REPLICA_INDEX`` is 0 on every single-host job and would shadow a real rank.

    Reporting every rank of a multi-host job as rank 0 would collapse the whole
    slice onto one upserted row: the record is keyed on ``(workload, rank)``, so
    the ranks would silently overwrite each other and the progress view would
    show one host.
    """

    environment = _env(EYVAN_REPLICA_INDEX="0", EYVAN_DISTRIBUTED_PROCESS_ID="3")
    recorder = _Recorder()
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    report_progress(step=1, environment=environment)

    assert json.loads(recorder.requests[0].data)["rank"] == 3


def test_the_attempt_is_reported_so_a_late_report_can_be_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report in flight when a retry starts belongs to the attempt that sent it.

    The read path filters on ``attempt >= workload.attempt``. If the reporter
    omitted its attempt and the server stamped the current one, a slow report
    from the dead attempt would overwrite the new attempt's progress with a
    higher step number that no longer happened.
    """

    recorder = _Recorder()
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    report_progress(step=1, environment=_env(EYVAN_ATTEMPT="4"))

    assert json.loads(recorder.requests[0].data)["attempt"] == 4


def test_a_transport_failure_does_not_reach_the_training_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """This is called between training steps; it must not be able to end the job."""

    recorder = _Recorder(fail=[urllib.error.URLError("no route to host")] * 3)
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    assert report_progress(step=1, environment=_env()) is False
    assert len(recorder.requests) == 3


def test_a_retryable_status_is_retried_and_a_permanent_one_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying a 403 only delays reporting a problem a human has to fix."""

    transient = _Recorder(fail=[_http_error(503)])
    monkeypatch.setattr(progress.urllib.request, "urlopen", transient)
    assert report_progress(step=1, environment=_env()) is True
    assert len(transient.requests) == 2

    permanent = _Recorder(fail=[_http_error(403)] * 3)
    monkeypatch.setattr(progress.urllib.request, "urlopen", permanent)
    assert report_progress(step=1, environment=_env()) is False
    assert len(permanent.requests) == 1


def test_retry_after_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A control plane shedding load says how long to wait; one policy reads it."""

    slept: list[float] = []
    monkeypatch.setattr(progress.time, "sleep", slept.append)
    recorder = _Recorder(fail=[_http_error(429, {"Retry-After": "5"})])
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    assert report_progress(step=1, environment=_env()) is True
    assert slept == [5.0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step": -1}, "must not be negative"),
        ({"total": -1}, "must not be negative"),
        ({"step": 11, "total": 10}, "must not exceed"),
        ({"metrics": {"loss": float("nan")}}, "finite"),
        ({"metrics": {"loss": float("inf")}}, "finite"),
    ],
)
def test_a_caller_bug_raises_rather_than_being_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict,
    match: str,
) -> None:
    """Swallowing a bad argument would hide a bug that fails on every call.

    A NaN in particular survives ``json.dumps`` as the literal ``NaN``, which is
    not JSON: the server would reject the body with a parse error naming no
    field, and the caller would see a silent ``False`` forever.
    """

    recorder = _Recorder()
    monkeypatch.setattr(progress.urllib.request, "urlopen", recorder)

    with pytest.raises(ValueError, match=match):
        report_progress(environment=_env(), **kwargs)
    assert recorder.requests == []


def test_importing_the_reporter_does_not_drag_in_the_client_or_cloudpickle() -> None:
    """The package root is lazy so the in-job path stays light; this is on it.

    ``report_progress`` runs inside a training loop on a TPU worker. Reaching it
    through the SDK -- or through ``callable_protocol`` for a retry constant --
    would import the client and ``cloudpickle`` on a path whose whole reason for
    being lazy is to avoid exactly that.
    """

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, eyvan; eyvan.report_progress;print('eyvan.sdk' in sys.modules, 'cloudpickle' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


def test_the_control_plane_mints_the_address_the_job_reads() -> None:
    """A job must never assemble a control-plane URL from parts.

    The producer is only usable if the address arrives fully formed, the same
    way every managed transfer URL does -- otherwise the job needs a base URL,
    the org and the project, and the route's query contract, all of which would
    then be two places that must agree.
    """

    access = ManagedArtifactAccess(
        store=None,  # type: ignore[arg-type]
        signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=60),
        public_url="https://plane.example",
        token_ttl_seconds=60,
    )

    url = access.progress_report_url(workload_id="wl-1", org_id="acme", project_id="research")

    assert url == ("https://plane.example/api/v1/workloads/wl-1/progress-reports?org_id=acme&project_id=research")


def test_a_workload_id_needing_escaping_does_not_break_the_path() -> None:
    """A path segment built by string interpolation is a path-traversal shape."""

    access = ManagedArtifactAccess(
        store=None,  # type: ignore[arg-type]
        signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=60),
        public_url="https://plane.example",
        token_ttl_seconds=60,
    )

    url = access.progress_report_url(workload_id="../../admin", org_id="a", project_id="b")

    assert "/api/v1/workloads/..%2F..%2Fadmin/progress-reports" in url


@pytest.mark.asyncio
async def test_a_dispatched_job_is_told_where_to_report(tmp_path) -> None:
    """Without this the producer would exist and still never be reachable.

    The address is injected beside the result transfer because it is
    authenticated by the result-write token: the two travel together or the job
    has half of a credential.
    """

    from test_controller import FakeAgent, principal

    from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri, result_uri
    from eyvan.config import EyvanSettings
    from eyvan.records import CallableExecution, CellRecord, CellState, ResourceVector, Role, WorkloadDefinition
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path / "artifacts"),
        signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=3600, max_ttl_seconds=3600),
        public_url="http://127.0.0.1:8080",
        token_ttl_seconds=3600,
    )
    fake_agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(dev_auth_enabled=True),
        agent_factory=lambda _cell: fake_agent,  # type: ignore[arg-type]
        artifact_access=access,
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-managed",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
            )
        )
    digest = "a" * 64
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="managed-callable",
            execution=CallableExecution(
                artifact_uri=callable_uri(digest),
                sha256=digest,
                python="3.13",
                result_uri=result_uri("call-managed"),
            ),
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
        ),
        idempotency_key="managed-callable",
    )
    await controller.reconcile_once()
    assert await controller.dispatch_outbox_once() == 1

    environment = fake_agent.submissions[0][1].spec["environment"]
    assert environment[PROGRESS_URL_ENV] == (
        f"http://127.0.0.1:8080/api/v1/workloads/{workload.id}/progress-reports?org_id=acme&project_id=research"
    )
    # The credential is the one already in the environment, not a second one:
    # a separate token kind would be a second thing to rotate and revoke.
    assert PROGRESS_TOKEN_ENV in environment
    assert report_progress(step=1, total=2, environment={**environment}) is not None


@pytest.mark.asyncio
async def test_the_gauge_drops_a_finished_workload_and_keeps_a_live_one() -> None:
    """``workload`` is the one unbounded label in the whole metric set.

    Progress series are written as reports arrive rather than rebuilt from a
    snapshot -- reading them would be one query per live workload -- so the
    export phase prunes instead of clearing. Without the prune, every workload
    the process ever saw would keep a series for the process's lifetime, which
    is the same unbounded-cardinality failure the record's own docstring refuses
    for user-named metrics.
    """

    from prometheus_client import CollectorRegistry
    from test_controller import principal

    from eyvan._internal.observability import EyvanMetrics
    from eyvan.config import EyvanSettings
    from eyvan.records import CommandExecution, Role, WorkloadDefinition, WorkloadState
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    metrics = EyvanMetrics(registry=CollectorRegistry())
    controller = EyvanController(repository, service, settings=EyvanSettings(), metrics=metrics)

    live, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="live", execution=CommandExecution(command="python train.py")),
        idempotency_key="live",
    )
    done, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="done", execution=CommandExecution(command="python train.py")),
        idempotency_key="done",
    )
    metrics.workload_progress.labels(workload=live.id, rank="0").set(0.5)
    metrics.workload_progress.labels(workload=done.id, rank="0").set(1.0)
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(done.id)
        assert stored is not None
        await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    await controller.export_metrics_once()

    published = {
        sample.labels["workload"] for family in metrics.workload_progress.collect() for sample in family.samples
    }
    assert published == {live.id}


def test_a_real_job_report_reaches_the_read_path_and_the_gauge(tmp_path) -> None:
    """The whole channel, end to end: reporter -> route -> record -> read path.

    Each half was correct alone. This is the test that would have failed while
    the producer was missing, because there was nothing to drive the route with.
    """

    from fastapi.testclient import TestClient

    from eyvan.artifacts.managed import result_uri
    from eyvan.config import EyvanSettings
    from eyvan.records import CallableExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
    from eyvan.server.api.app import create_app

    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            trusted_callable_projects=("acme/research",),
            public_url="https://eyvan.test",
            artifact_store_uri=str(tmp_path / "managed"),
            artifact_token_key="k" * 32,
            artifact_token_ttl_s=3600,
        )
    )
    uri = result_uri("wl-1")

    async def store() -> None:
        async with app.state.service.repository.transaction() as transaction:
            await transaction.put_workload(
                WorkloadRecord(
                    id="wl-1",
                    org_id="acme",
                    project_id="research",
                    spec=WorkloadDefinition(
                        name="managed-callable",
                        execution=CallableExecution(
                            artifact_uri="eyvan://callables/" + "0" * 64,
                            sha256="0" * 64,
                            python="3.13",
                            result_uri=uri,
                        ),
                    ),
                    state=WorkloadState.RUNNING,
                    created_by="alice",
                )
            )

    asyncio.run(store())
    access = app.state.artifact_access
    transfer = access.result_upload_transfer(uri, org_id="acme", project_id="research", workload_id="wl-1")
    url = access.progress_report_url(workload_id="wl-1", org_id="acme", project_id="research")

    client = TestClient(app)
    captured: dict[str, object] = {}

    def _urlopen(request, timeout=None):
        del timeout
        response = client.post(
            _route(request.full_url),
            content=request.data,
            headers={name: value for name, value in request.header_items()},
        )
        captured["status"] = response.status_code
        captured["body"] = response.json()
        if response.status_code >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status_code, "", None, None)

        class _Response:
            def read(self) -> bytes:
                return response.content

            def __enter__(self):
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

        return _Response()

    with mock.patch.object(progress.urllib.request, "urlopen", _urlopen):
        delivered = report_progress(
            step=25,
            total=100,
            message="epoch 1",
            metrics={"loss": 0.5},
            environment={PROGRESS_URL_ENV: url, PROGRESS_TOKEN_ENV: transfer.token, "EYVAN_ATTEMPT": "1"},
        )

    assert delivered is True
    assert captured["status"] == 200

    read = client.get(
        "/api/v1/workloads/wl-1/progress",
        headers={
            "X-Eyvan-User": "alice",
            "X-Eyvan-Roles": "operator",
            "X-Eyvan-Org": "acme",
            "X-Eyvan-Project": "research",
        },
    )
    assert read.status_code == 200
    (reported,) = read.json()["user_progress"]
    assert (reported["rank"], reported["step"], reported["total"]) == (0, 25, 100)
    assert reported["metrics"] == {"loss": 0.5}

    exported = {
        (sample.labels["workload"], sample.labels["rank"]): sample.value
        for family in app.state.metrics.workload_progress.collect()
        for sample in family.samples
    }
    assert exported == {("wl-1", "0"): 0.25}
    # The free-form metric is recorded and displayed and is not a series: label
    # names chosen by user code are unbounded cardinality, so one mistyped name
    # in a training loop would be a self-inflicted outage.
    scraped = client.get("/metrics").text
    assert 'eyvan_workload_progress_fraction{rank="0",workload="wl-1"} 0.25' in scraped
    assert "loss" not in scraped


def _route(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.path}?{parsed.query}"


def test_a_stale_attempt_report_is_rejected_at_the_read_path() -> None:
    """A leftover attempt-1 reporter must not overwrite attempt 2's progress."""

    # Covered by the read path's attempt filter; asserted here so the producer's
    # decision to report its own attempt stays load-bearing rather than cosmetic.
    assert progress._attempt({"EYVAN_ATTEMPT": "2"}) == 2
    assert progress._attempt({"EYVAN_ATTEMPT": "not-a-number"}) == 1
    assert progress._attempt({}) == 1
