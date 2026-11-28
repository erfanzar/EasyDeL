# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import eyvan
import eyvan.sdk.client.identity as identity_module
import eyvan.sdk.client.sync as sync_module
import eyvan.sdk.client.transport as transport_module
import eyvan.sdk.execution as execution_module
from eyvan._internal.client_version import CLIENT_VERSION_HEADER, client_version
from eyvan.artifacts import GCSArtifactStore, LocalArtifactStore
from eyvan.capacity.autoscale import AutoscaleConfigCache
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfileStore
from eyvan.capacity.providers.gcp.runtime.supervisor import GcpRuntimeResult, LocalAgentResult, LocalAgentState
from eyvan.records import (
    AutoscaleBlockReason,
    CallableExecution,
    CapacityClass,
    GcpQueueAdmissionDenialTarget,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk.client.callables import EyvanCallableSubmitter
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.contexts import EyvanContext, set_active_context
from eyvan.server.local.daemon import LocalDaemonResult, LocalDaemonState
from eyvan.worker.agent.models import AgentCapabilities
from eyvan.worker.runtime.results import dump_result


@pytest.fixture(autouse=True)
def _disable_real_loopback_daemon(monkeypatch, tmp_path) -> None:
    """Keep callable serialization tests independent from a real local API process."""

    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def unavailable_daemon(endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.SKIPPED,
            pid=None,
            log_path=None,
        )

    monkeypatch.setattr(execution_module, "ensure_loopback_control_plane", unavailable_daemon)


def _native_gcp_capabilities() -> AgentCapabilities:
    """Return the typed capability contract needed to construct a runtime result fixture."""

    return AgentCapabilities(
        provider="gcp",
        executor="eyvan-native",
        accelerator_types=("tpu",),
        capacity_classes=("spot",),
        supports_checkpoint=True,
        supports_callables=True,
        supports_exec=True,
        max_gang_replicas=1,
        resource_enforcement="none",
    )


def test_init_ensures_a_literal_loopback_control_plane_before_constructing_the_default_submitter(
    tmp_path, monkeypatch
) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19084")
    events: list[str] = []
    ensured: list[ControlPlaneEndpoint] = []

    class FakeClient:
        def __init__(self, *, base_url: str, identity: ClientIdentity) -> None:
            assert base_url == endpoint.url
            assert identity.dev_user is not None
            events.append("client")

        def workload_log_page(self, workload_id: str, *, after: str, limit: int):
            return {"records": [], "next_cursor": after}

    def ensure(configured_endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        parsed = ControlPlaneEndpoint.parse(configured_endpoint)
        events.append("daemon")
        ensured.append(parsed)
        return LocalDaemonResult(
            endpoint=parsed,
            state=LocalDaemonState.STARTED,
            pid=1234,
            log_path=tmp_path / "control-plane.log",
        )

    monkeypatch.setattr(execution_module, "_context", None)
    monkeypatch.setattr(execution_module, "ensure_loopback_control_plane", ensure)
    monkeypatch.setattr(sync_module, "EyvanClient", FakeClient)

    context = eyvan.init(
        fleet="research",
        org_id="acme",
        project_id="models",
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        base_url=endpoint.url,
    )

    assert events == ["daemon", "client"]
    assert ensured == [endpoint]
    assert context.fleet == "research"
    assert context.org_id == "acme"
    assert context.project_id == "models"


def test_init_uses_direct_gcs_artifacts_after_native_gcp_runtime_setup(tmp_path, monkeypatch) -> None:
    endpoint = ControlPlaneEndpoint.parse("http://127.0.0.1:19085")
    store = GcpNativeRuntimeProfileStore(tmp_path / "home" / ".eyvan" / "gcp-native-runtime.json")
    profile = store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'a' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )
    ensured: list[ControlPlaneEndpoint] = []

    def ensure(configured_endpoint: str | ControlPlaneEndpoint, *, profile_path: Path | None = None) -> GcpRuntimeResult:
        parsed = ControlPlaneEndpoint.parse(configured_endpoint)
        ensured.append(parsed)
        return GcpRuntimeResult(
            profile=profile,
            control_plane=LocalDaemonResult(
                endpoint=parsed,
                state=LocalDaemonState.STARTED,
                pid=1234,
                log_path=tmp_path / "control-plane.log",
            ),
            agent=LocalAgentResult(
                endpoint=ControlPlaneEndpoint.parse("http://127.0.0.1:8090"),
                state=LocalAgentState.STARTED,
                pid=5678,
                log_path=tmp_path / "cell-agent.log",
                capabilities=_native_gcp_capabilities(),
            ),
        )

    class FakeClient:
        def __init__(self, *, base_url: str, identity: ClientIdentity) -> None:
            assert base_url == endpoint.url
            assert identity.dev_user is not None

        def workload_log_page(self, workload_id: str, *, after: str, limit: int):
            return {"records": [], "next_cursor": after}

    monkeypatch.setattr(execution_module, "_context", None)
    monkeypatch.setattr(execution_module, "ensure_gcp_native_runtime", ensure)
    monkeypatch.setattr(sync_module, "EyvanClient", FakeClient)

    context = eyvan.init(
        fleet="research",
        org_id="acme",
        project_id="models",
        base_url=endpoint.url,
    )

    assert ensured == [endpoint]
    assert isinstance(context.artifact_store, GCSArtifactStore)
    assert context.artifact_store.root_uri == profile.artifact_root_uri


def test_native_gcp_profile_rejects_a_local_artifact_uri_before_submitting_work(tmp_path, monkeypatch) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "home" / ".eyvan" / "gcp-native-runtime.json")
    store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'b' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )
    monkeypatch.setattr(execution_module, "_context", None)

    with pytest.raises(ValueError, match="GCS-backed callable artifacts"):
        eyvan.init(
            fleet="research",
            org_id="acme",
            project_id="models",
            artifact_uri=str(tmp_path / "local-artifacts"),
            base_url="http://127.0.0.1:19086",
        )


def test_native_gcp_profile_rejects_a_local_artifact_store_before_submitting_work(tmp_path, monkeypatch) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "home" / ".eyvan" / "gcp-native-runtime.json")
    store.create(
        project_id="eyvan-project",
        artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_wheel_uri=(f"gs://eyvan-artifacts/eyvan/runtime/wheels/{'c' * 64}/eyvan-0.1.0-py3-none-any.whl"),
    )
    monkeypatch.setattr(execution_module, "_context", None)

    with pytest.raises(ValueError, match="requires a GCSArtifactStore"):
        eyvan.init(
            fleet="research",
            org_id="acme",
            project_id="models",
            artifact_store=LocalArtifactStore(tmp_path / "local-artifacts"),
            base_url="http://127.0.0.1:19087",
        )


def test_decorator_lazily_uses_active_context_and_managed_artifacts(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    config_path = tmp_path / "eyvan-context.json"
    set_active_context(
        EyvanContext(
            fleet="research",
            org_id="acme",
            project_id="models",
            base_url="http://127.0.0.1:8080",
        ),
        path=config_path,
    )
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("EYVAN_DEV_USER", raising=False)
    monkeypatch.setattr(identity_module.getpass, "getuser", lambda: "alice")
    monkeypatch.setattr(execution_module, "_context", None)
    requests: list[httpx.Request] = []
    client_kwargs: list[dict[str, object]] = []
    encoded_result = dump_result(9)
    stored: WorkloadRecord | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        nonlocal stored
        requests.append(request)
        if request.method == "PUT" and "/artifacts/callables/" in request.url.path:
            digest = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"uri": f"eyvan://callables/{digest}", "sha256": digest})
        if request.method == "POST" and request.url.path == "/api/v1/fleets/research/callables":
            payload = json.loads(request.content)
            execution = CallableExecution.model_validate(payload["execution"])
            execution = execution.model_copy(update={"result_uri": "eyvan://results/wl-managed-sdk-g1"})
            stored = WorkloadRecord(
                id="wl-managed-sdk",
                org_id="acme",
                project_id="models",
                spec=WorkloadDefinition(name="train", execution=execution),
                created_by="alice",
            )
            return httpx.Response(201, json=stored.model_dump(mode="json"))
        if request.method == "GET" and request.url.path == "/api/v1/workloads/wl-managed-sdk":
            assert stored is not None
            return httpx.Response(
                200,
                json=stored.model_copy(update={"state": WorkloadState.SUCCEEDED}).model_dump(mode="json"),
            )
        if request.method == "GET" and request.url.path.startswith("/api/v1/fleets/"):
            # `init` checks whether the fleet it was given is serving, so a job
            # can be submitted without `eyvan up` first. Answering "already
            # serving" leaves these tests exercising submission, which is what
            # they are about.
            return httpx.Response(200, json={"name": "research", "pools": [{"id": "p", "enabled": True}]})
        if request.method == "GET" and request.url.path.startswith("/api/v1/artifacts/results/"):
            return httpx.Response(200, content=encoded_result)
        raise AssertionError((request.method, request.url))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")

    def client(**kwargs: object) -> httpx.Client:
        client_kwargs.append(kwargs)
        return raw

    monkeypatch.setattr(transport_module.httpx, "Client", client)

    @eyvan.execute(eyvan.TpuConfig(type="v5p-512"), workspace=False)
    def train(value: int) -> int:
        return value + 1

    job = train.remote(8)

    assert job.id == "wl-managed-sdk"
    assert capsys.readouterr().err.splitlines() == [
        "Eyvan: job wl-managed-sdk submitted to fleet research.",
        "  request  v5p-512 x 1 worker",
        "  allocation  elastic",
        "  this job keeps running if this client exits.",
        "  status  eyvan workload status wl-managed-sdk",
        "  logs    eyvan workload logs wl-managed-sdk",
        "  cancel  eyvan workload cancel wl-managed-sdk",
    ]
    assert job.result() == 9
    # Found by method rather than by position: `init` now reads the fleet
    # first, to bring it up if it is not serving, so the upload is no longer
    # request zero. The assertion is that the callable is uploaded at all, and
    # as a PUT -- which is what the positional check was really saying.
    uploads = [item for item in requests if item.method == "PUT"]
    assert len(uploads) == 1, "the callable artifact should be uploaded exactly once"
    assert uploads[0].url.path.startswith("/api/v1/artifacts/")
    assert client_kwargs[0]["headers"] == {
        CLIENT_VERSION_HEADER: client_version(),
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "models",
        # fleet_admin, not operator: no token, no explicit user, no explicit
        # role, and a loopback plane -- the same rule `eyvan` itself applies,
        # and what lets `init` bring up a fleet that is not serving instead of
        # sending the operator to `eyvan up`. Naming any of those three keeps
        # the identity named; a remote plane is never elevated.
        "X-Eyvan-Roles": "fleet_admin",
        "X-Eyvan-User": "alice",
    }
    # By path, not by position: `init` reads the fleet before anything is
    # uploaded or submitted, so the submission is no longer request one.
    submitted = json.loads(next(item for item in requests if item.url.path.endswith("/callables")).content)
    assert submitted["accelerator_type"] == "v5p-512"
    assert submitted["execution"]["artifact_uri"].startswith("eyvan://callables/")
    assert submitted["execution"]["result_uri"].startswith("eyvan://results/")
    assert requests[-1].url.path == "/api/v1/artifacts/results/wl-managed-sdk-g1"
    raw.close()


def test_decorator_uses_the_active_context_autoscale_cache_for_queue_denials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "eyvan-context.json"
    cache_dir = tmp_path / "selected-autoscale"
    set_active_context(
        EyvanContext(
            fleet="research",
            org_id="acme",
            project_id="models",
            base_url="http://127.0.0.1:8080",
            autoscale_cache_dir=str(cache_dir),
        ),
        path=config_path,
    )
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(config_path))
    monkeypatch.setattr(execution_module, "_context", None)
    submitted: WorkloadRecord | None = None
    denial_occurred_at = utcnow()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        nonlocal submitted
        if request.method == "POST" and request.url.path == "/api/v1/fleets/research/callables":
            payload = json.loads(request.content)
            execution = CallableExecution.model_validate(payload["execution"])
            submitted = WorkloadRecord(
                id="wl-queue-denied",
                org_id="acme",
                project_id="models",
                spec=WorkloadDefinition(name="train", execution=execution),
                created_by="alice",
            )
            return httpx.Response(201, json=submitted.model_dump(mode="json"))
        if request.method == "GET" and request.url.path == "/api/v1/workloads/wl-queue-denied":
            assert submitted is not None
            denied = submitted.model_copy(
                update={
                    "autoscale_block_reason": AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
                    "failure_reason": "GCP rejected queue admission",
                    "queue_admission_denial": GcpQueueAdmissionDenialTarget(
                        provider_project="gcp-project",
                        zone="us-east5-a",
                        accelerator_type="v6e-16",
                        capacity_class=CapacityClass.SPOT,
                        occurred_at=denial_occurred_at,
                        event_id="queue-denial-current-occurrence",
                    ),
                    "state": WorkloadState.FAILED,
                }
            )
            return httpx.Response(200, json=denied.model_dump(mode="json"))
        if request.method == "GET" and request.url.path.startswith("/api/v1/fleets/"):
            # `init` reads the fleet so it can bring one up that is not serving,
            # which is what makes submitting the whole workflow. Answering
            # "already serving" keeps these tests about submission.
            return httpx.Response(200, json={"name": "research", "pools": [{"id": "p", "enabled": True}]})
        raise AssertionError((request.method, request.url))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8080")
    monkeypatch.setattr(transport_module.httpx, "Client", lambda **_kwargs: raw)
    eyvan.init(artifact_store=LocalArtifactStore(tmp_path / "artifacts"))

    @eyvan.execute(eyvan.TpuConfig(type="v6e-16"), workspace=False)
    def train() -> None:
        return None

    job = train.remote()

    assert job.status() is WorkloadState.FAILED
    cache = AutoscaleConfigCache(cache_dir)
    assert cache.blocks_queue_admission_denial(
        project="gcp-project",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    assert cache.list_queue_admission_denials()[0].observed_at == denial_occurred_at
    assert not cache.blocks_queue_admission_denial(
        project="gcp-project",
        zone="us-east5-b",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )
    raw.close()


def test_decorator_job_result_reports_control_plane_unavailability_without_httpx_chain(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "eyvan-context.json"
    set_active_context(
        EyvanContext(
            fleet="research",
            org_id="acme",
            project_id="models",
            base_url="http://127.0.0.1:8080",
        ),
        path=config_path,
    )
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(config_path))
    monkeypatch.setattr(execution_module, "_context", None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "PUT" and "/artifacts/callables/" in request.url.path:
            digest = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"uri": f"eyvan://callables/{digest}", "sha256": digest})
        if request.method == "POST" and request.url.path == "/api/v1/fleets/research/callables":
            payload = json.loads(request.content)
            execution = CallableExecution.model_validate(payload["execution"])
            workload = WorkloadRecord(
                id="wl-lost-control-plane",
                org_id="acme",
                project_id="models",
                spec=WorkloadDefinition(name="train", execution=execution),
                created_by="alice",
            )
            return httpx.Response(201, json=workload.model_dump(mode="json"))
        if request.method == "GET" and request.url.path == "/api/v1/workloads/wl-lost-control-plane":
            raise httpx.ConnectError("connection refused", request=request)
        if request.method == "GET" and request.url.path.startswith("/api/v1/fleets/"):
            # `init` reads the fleet so it can bring one up that is not serving,
            # which is what makes submitting the whole workflow. Answering
            # "already serving" keeps these tests about submission.
            return httpx.Response(200, json={"name": "research", "pools": [{"id": "p", "enabled": True}]})
        raise AssertionError((request.method, request.url))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8080")
    monkeypatch.setattr(transport_module.httpx, "Client", lambda **_kwargs: raw)

    @eyvan.execute(eyvan.TpuConfig(type="v6e-16", pods=2), workspace=False)
    def train() -> None:
        pass

    job = train.remote()

    with pytest.raises(eyvan.EyvanUnavailableError) as raised:
        job.result()

    assert str(raised.value) == (
        "Eyvan control plane is unavailable at http://127.0.0.1:8080. "
        "Start it with 'eyvan-server --log-level info' and keep it running while jobs are active."
    )
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    raw.close()


def test_decorator_reports_actionable_error_when_selected_fleet_has_no_matching_tpu(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "eyvan-context.json"
    set_active_context(
        EyvanContext(
            fleet="research-us-central1-c",
            org_id="default",
            project_id="default",
            base_url="http://127.0.0.1:8080",
        ),
        path=config_path,
    )
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(config_path))
    monkeypatch.setattr(execution_module, "_context", None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        if request.method == "PUT" and "/artifacts/callables/" in request.url.path:
            digest = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"uri": f"eyvan://callables/{digest}", "sha256": digest})
        if request.method == "POST" and request.url.path.endswith("/callables"):
            return httpx.Response(
                404,
                json={
                    "detail": (
                        "no enabled capacity pool in fleet 'research-us-central1-c' matches accelerator type 'v5p-16'"
                    ),
                    "code": "no_matching_capacity_pool",
                    "context": {
                        "fleet": "research-us-central1-c",
                        "requested_accelerator_type": "v5p-16",
                        "available_enabled_accelerator_types": ["v6e-16", "v6e-8"],
                        "future_server_field": "ignored by older clients",
                    },
                    "future_top_level_field": True,
                },
            )
        if request.method == "GET" and request.url.path.startswith("/api/v1/fleets/"):
            # `init` reads the fleet so it can bring one up that is not serving,
            # which is what makes submitting the whole workflow. Answering
            # "already serving" keeps these tests about submission.
            return httpx.Response(200, json={"name": "research", "pools": [{"id": "p", "enabled": True}]})
        raise AssertionError((request.method, request.url))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eyvan.test")
    monkeypatch.setattr(transport_module.httpx, "Client", lambda **_kwargs: raw)

    @eyvan.execute(eyvan.TpuConfig(type="v5p-16"), workspace=False)
    def train() -> None:
        pass

    with pytest.raises(eyvan.EyvanPlacementError) as raised:
        train.remote()

    assert raised.value.fleet == "research-us-central1-c"
    assert raised.value.requested_accelerator_type == "v5p-16"
    assert raised.value.available_accelerator_types == ("v6e-8", "v6e-16")
    assert str(raised.value) == (
        "Eyvan cannot run TPU 'v5p-16' in selected fleet 'research-us-central1-c'. "
        "Compatible TPU types for this request: v6e-8, v6e-16. Change TpuConfig(type=...) to one of those. "
        "Select a fleet with `eyvan use FLEET`, list them with `eyvan capacity fleets`, "
        "or apply one with `eyvan up CONFIG`."
    )
    assert raised.value.__cause__ is None
    raw.close()


def test_executable_auto_resolves_fleet_when_context_fleet_is_none(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    invocation_id = "call-auto-fleet"
    result_uri = store.allocate_result_uri(invocation_id)
    pending = WorkloadRecord(
        id="wl-auto",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="smoke",
            execution=CallableExecution(
                artifact_uri="gs://a/b.pkl",
                sha256="a" * 64,
                python="3.13",
                result_uri=result_uri,
            ),
        ),
        created_by="alice",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "pools": [
                        {
                            "fleet": "auto-fleet",
                            "pool": "v5p-pool",
                            "accelerator_type": "v5p-16",
                            "capacity_class": "spot",
                            "region": "us-east5",
                            "zone": "us-east5-a",
                            "min_cells": 0,
                            "max_cells": 4,
                            "enabled": True,
                        }
                    ]
                },
            )
        return httpx.Response(201, json=pending.model_dump(mode="json"))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = sync_module.EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter(client, store)

    context = execution_module.ExecutionContext(
        fleet=None,
        org_id="acme",
        project_id="research",
        artifact_store=store,
        submitter=submitter,
    )

    @execution_module.execute(execution_module.TpuConfig(type="v5p-16", pods=2), context=context)
    def smoke() -> dict:
        return {"ok": True}

    job = smoke.remote()

    assert job.id == pending.id
    assert len(requests) == 2
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v1/capacity"
    assert requests[1].method == "POST"
    assert requests[1].url.path == "/api/v1/fleets/auto-fleet/callables"
    payload = json.loads(requests[1].content)
    assert payload["accelerator_type"] == "v5p-16"
    assert payload["pods"] == 2
    raw.close()


def test_auto_resolve_without_a_matching_pool_raises_placement_error_not_type_error(tmp_path: Path) -> None:
    """An unresolvable TpuConfig must surface the actionable placement error.

    The auto-resolve branch previously constructed ``EyvanPlacementError``
    positionally against its keyword-only signature, so this path raised
    ``TypeError`` instead of telling the caller what to do about it.
    """

    store = LocalArtifactStore(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/log-page"):
            return httpx.Response(200, json={"records": [], "next_cursor": ""})
        requests.append(request)
        return httpx.Response(200, json={"pools": []})

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = sync_module.EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter(client, store)

    context = execution_module.ExecutionContext(
        fleet=None,
        org_id="acme",
        project_id="research",
        artifact_store=store,
        submitter=submitter,
    )

    @execution_module.execute(execution_module.TpuConfig(type="v5p-16"), context=context)
    def smoke() -> dict:
        return {"ok": True}

    with pytest.raises(eyvan.EyvanPlacementError) as raised:
        smoke.remote()

    assert raised.value.fleet is None
    assert raised.value.requested_accelerator_type == "v5p-16"
    assert raised.value.available_accelerator_types == ()
    assert str(raised.value) == (
        "Eyvan cannot run TPU 'v5p-16' in any enabled fleet for acme/research "
        "with capacity class 'spot'. "
        "No enabled fleet exposes a compatible TPU pool for this request. "
        "Select a fleet with `eyvan use FLEET`, list them with `eyvan capacity fleets`, "
        "or apply one with `eyvan up CONFIG`."
    )
    assert [request.url.path for request in requests] == ["/api/v1/capacity"]
    raw.close()


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
