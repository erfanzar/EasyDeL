# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json

import httpx
import pytest

from eyvan.artifacts import LocalArtifactStore
from eyvan.config import EyvanSettings
from eyvan.records import (
    CallableExecution,
    CapacityClass,
    CommandExecution,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.sdk import AsyncEyvanClient, ClientIdentity, EyvanCallableSubmitter, EyvanClient
from eyvan.sdk.execution import CallableSubmission, TpuConfig
from eyvan.server.api.app import create_app
from eyvan.server.api.schemas import WorkloadSubmitRequest
from eyvan.worker.runtime.callables import dump_callable
from eyvan.worker.runtime.results import dump_result


def test_sync_sdk_runs_on_configured_node_type() -> None:
    stored = WorkloadRecord(
        id="wl-run",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="daily-train", execution=CommandExecution(command="python train.py")),
        created_by="alice",
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json=stored.model_dump(mode="json"))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)

    submitted = client.run(
        "east5-training",
        "v5p",
        "python train.py",
        org_id="acme",
        project_id="research",
        name="daily-train",
        environment={"RUN_ID": "42"},
        metadata={"team": "alignment"},
        max_retries=2,
        idempotency_key="run-daily-train",
    )

    assert submitted == stored
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/v1/fleets/east5-training/node-types/v5p/workloads"
    assert seen[0].headers["Idempotency-Key"] == "run-daily-train"
    assert json.loads(seen[0].content) == {
        "org_id": "acme",
        "project_id": "research",
        "command": "python train.py",
        # Shape and placement a caller may state, at their defaults here: one
        # slice, and no zone pinned, so the fleet still chooses.
        "pods": 1,
        "zone": None,
        "name": "daily-train",
        "environment": {"RUN_ID": "42"},
        "secret_mounts": {},
        "metadata": {"team": "alignment"},
        "max_retries": 2,
        # Separate from ``max_retries``: that budget pays for capacity the fleet
        # lost, this one for the job's own non-zero exit, which usually repeats.
        "max_failure_retries": 0,
        "max_preemption_retries": 10,
        # Explicit opt-in: the SDK sends command intent only, so a submission
        # never carries a working tree the caller did not ask to ship.
        "workspace": None,
        # The job's own checkpoint contract. Off unless asked for, because a
        # hook that does not exist is worse than none: the job would be stopped
        # for a checkpoint it never wrote.
        "checkpointable": False,
        "checkpoint_hook": None,
        "checkpoint_output_uri": None,
        "can_submit_children": False,
        # Scheduling intent the pool cannot supply: without these a long
        # training run submitted here sat in the interactive band.
        "priority_band": "interactive",
        "intra_band_priority": 0,
        "checkpoint_grace_seconds": 0,
        "estimated_duration_seconds": 3600,
        "named_assets": [],
        # Job intent too: what a job needs installed is a fact about the job,
        # and empty is what one that asked for nothing sends.
        "dependencies": [],
    }
    client.close()


def test_sync_sdk_waits_for_terminal_workload() -> None:
    states = iter((WorkloadState.PENDING, WorkloadState.RUNNING, WorkloadState.SUCCEEDED))
    base = WorkloadRecord(
        id="wl-wait",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
        created_by="alice",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=base.model_copy(update={"state": next(states)}).model_dump(mode="json"))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)

    observed_states: list[WorkloadState] = []
    terminal = client.wait_workload(
        "wl-wait",
        timeout=1,
        poll_interval=0.001,
        on_update=lambda workload: observed_states.append(workload.state),
    )

    assert terminal.state is WorkloadState.SUCCEEDED
    assert observed_states == [WorkloadState.PENDING, WorkloadState.RUNNING, WorkloadState.SUCCEEDED]
    client.close()


def test_sync_sdk_submits_callable_by_accelerator_type() -> None:
    execution = CallableExecution(
        artifact_uri="gs://artifacts/callables/train.pkl",
        sha256="a" * 64,
        python="3.13",
        result_uri="gs://artifacts/results/run-1.result",
    )
    stored = WorkloadRecord(
        id="wl-callable",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=execution),
        created_by="alice",
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json=stored.model_dump(mode="json"))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)

    submitted = client.submit_pool_callable(
        "research",
        "v5p-512",
        execution,
        org_id="acme",
        project_id="research",
        idempotency_key="callable-run-1",
    )

    assert submitted == stored
    assert seen[0].url.path == "/api/v1/fleets/research/callables"
    assert seen[0].headers["Idempotency-Key"] == "callable-run-1"
    assert json.loads(seen[0].content) == {
        "org_id": "acme",
        "project_id": "research",
        "accelerator_type": "v5p-512",
        "pods": 1,
        "atomic": False,
        "capacity_class": "spot",
        "execution": execution.model_dump(mode="json"),
        "name": None,
        "environment": {},
        "secret_mounts": {},
        "metadata": {},
        "max_retries": 3,
        "max_failure_retries": 0,
        "max_preemption_retries": 10,
        "checkpointable": False,
        "checkpoint_hook": None,
        "checkpoint_output_uri": None,
        "can_submit_children": False,
        # Scheduling intent the pool cannot supply: without these a long
        # training run submitted here sat in the interactive band.
        "priority_band": "interactive",
        "intra_band_priority": 0,
        "checkpoint_grace_seconds": 0,
        "estimated_duration_seconds": 3600,
        "named_assets": [],
        # Job intent too: what a job needs installed is a fact about the job,
        # and empty is what one that asked for nothing sends.
        "dependencies": [],
    }
    client.close()


def test_sync_sdk_serializes_each_requested_pod_without_legacy_capacity_or_ray_fields() -> None:
    execution = CallableExecution(
        artifact_uri="gs://artifacts/callables/train.pkl",
        sha256="b" * 64,
        python="3.13",
        result_uri="gs://artifacts/results/run-6.result",
    )
    stored = WorkloadRecord(
        id="wl-six-pods",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=execution),
        created_by="alice",
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json=stored.model_dump(mode="json"))

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)

    submitted = client.submit_pool_callable(
        "research",
        "v5p-16",
        execution,
        org_id="acme",
        project_id="research",
        pods=6,
        idempotency_key="callable-six-pods",
    )

    assert submitted == stored
    assert seen[0].url.path == "/api/v1/fleets/research/callables"
    payload = json.loads(seen[0].content)
    assert payload["accelerator_type"] == "v5p-16"
    assert payload["pods"] == 6
    assert {"slices", "node_count", "ray"}.isdisjoint(payload)
    client.close()


def test_callable_submitter_returns_durable_result_handle(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    invocation_id = "call-sdk-result"
    result_uri = store.allocate_result_uri(invocation_id)
    result_path = tmp_path / "results" / f"{invocation_id}.result"
    result_path.write_bytes(dump_result({"loss": 0.125}))
    stored_artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=stored_artifact.uri,
        sha256=stored_artifact.sha256,
        python=stored_artifact.python,
        result_uri=result_uri,
    )
    pending = WorkloadRecord(
        id="wl-callable-result",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=execution),
        created_by="alice",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(201, json=pending.model_dump(mode="json"))
        return httpx.Response(
            200,
            json=pending.model_copy(update={"state": WorkloadState.SUCCEEDED}).model_dump(mode="json"),
        )

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter(client, store)

    job = submitter.submit_callable(
        CallableSubmission(
            invocation_id=invocation_id,
            fleet="research",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v5p-512", max_retries=7),
            artifact=stored_artifact,
            result_uri=result_uri,
        )
    )

    assert job.id == pending.id
    assert job.result(timeout=1) == {"loss": 0.125}
    submitted = json.loads(requests[0].content)
    assert submitted["accelerator_type"] == "v5p-512"
    assert submitted["max_retries"] == 7
    assert submitted["execution"]["result_uri"] == result_uri
    assert submitted["metadata"] == {"eyvan_invocation_id": invocation_id}
    client.close()


def test_callable_submitter_rejects_caller_signing_keys(tmp_path) -> None:
    raw = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)), base_url="http://test")
    client = EyvanClient(client=raw)

    with pytest.raises(ValueError, match="remote callable signing keys are not supported"):
        EyvanCallableSubmitter(client, LocalArtifactStore(tmp_path), signing_key=b"caller-secret")

    client.close()


def test_callable_submitter_rejects_signed_artifacts_before_any_http_request(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    artifact = store.put(dump_callable(lambda: None, signing_key=b"caller-secret"))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)
    submitter = EyvanCallableSubmitter(client, store)

    with pytest.raises(ValueError, match="remote callable signatures are not supported"):
        submitter.submit_callable(
            CallableSubmission(
                invocation_id="call-signed",
                fleet="research",
                org_id="acme",
                project_id="research",
                callable_name="tests.train",
                tpu=TpuConfig(type="v5p-16"),
                artifact=artifact,
                result_uri=store.allocate_result_uri("call-signed"),
            )
        )

    assert requests == []
    client.close()


def test_sync_sdk_rejects_callable_signing_environment_before_any_http_request() -> None:
    execution = CallableExecution(
        artifact_uri="gs://trusted/callable.pkl",
        sha256="a" * 64,
        python="3.13",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)

    with pytest.raises(ValueError, match="remote callable signing keys are not supported"):
        client.submit_pool_callable(
            "research",
            "v5p-16",
            execution,
            org_id="acme",
            project_id="research",
            environment={"EYVAN_CALLABLE_SIGNING_KEY": "caller-secret"},
        )

    assert requests == []
    client.close()


@pytest.mark.asyncio
async def test_async_sdk_run_retries_with_one_idempotency_key() -> None:
    stored = WorkloadRecord(
        id="wl-async-run",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="v5p", execution=CommandExecution(command="python train.py")),
        created_by="alice",
    )
    keys: list[str] = []
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers["Idempotency-Key"])
        paths.append(request.url.path)
        if len(keys) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, json={"detail": "retry transaction"})
        return httpx.Response(201, json=stored.model_dump(mode="json"))

    raw = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    client = AsyncEyvanClient(client=raw)

    submitted = await client.run(
        "east5-training",
        "v5p",
        "python train.py",
        org_id="acme",
        project_id="research",
    )

    assert submitted == stored
    assert paths == [
        "/api/v1/fleets/east5-training/node-types/v5p/workloads",
        "/api/v1/fleets/east5-training/node-types/v5p/workloads",
    ]
    assert len(keys) == 2
    assert keys[0] == keys[1]
    assert keys[0].startswith("sdk-")
    await client.aclose()


@pytest.mark.asyncio
async def test_async_sdk_submits_and_lists_typed_workloads() -> None:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True))
    transport = httpx.ASGITransport(app=app)
    raw = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = AsyncEyvanClient(
        client=raw,
        identity=ClientIdentity(dev_user="alice", dev_org="acme", dev_project="research", dev_roles=("org_admin",)),
    )
    request = WorkloadSubmitRequest(
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
    )

    submitted = await client.submit_workload(request, idempotency_key="sdk-submit")
    listed = await client.list_workloads(org_id="acme", project_id="research")
    leases = await client.list_leases(org_id="acme", project_id="research")
    events = await client.list_events(org_id="acme", project_id="research")
    budgets = await client.list_budgets(org_id="acme")
    policies = await client.list_policies(org_id="acme")

    assert listed == [submitted]
    assert leases == []
    assert [event.action for event in events] == ["workload.submitted"]
    assert budgets == []
    assert policies == []
    assert submitted.spec.execution.command == "python train.py"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_sdk_retries_transient_mutation_with_one_idempotency_key() -> None:
    request = WorkloadSubmitRequest(
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
    )
    stored = WorkloadRecord(
        id="wl-retry",
        org_id="acme",
        project_id="research",
        spec=request.spec,
        created_by="alice",
    )
    keys: list[str] = []

    async def handler(raw_request: httpx.Request) -> httpx.Response:
        keys.append(str(raw_request.headers["Idempotency-Key"]))
        if len(keys) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, json={"detail": "retry transaction"})
        return httpx.Response(201, json=stored.model_dump(mode="json"))

    raw = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    client = AsyncEyvanClient(client=raw)

    submitted = await client.submit_workload(request)

    assert submitted == stored
    assert len(keys) == 2
    assert keys[0] == keys[1]
    assert keys[0].startswith("sdk-")
    await client.aclose()


def test_sync_sdk_lists_capacity_and_parses_pool_response() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "pools": [
                    {
                        "fleet": "research",
                        "pool": "v5p-pool",
                        "accelerator_type": "v5p-8",
                        "capacity_class": "spot",
                        "region": "us-east5",
                        "zone": "us-east5-a",
                        "min_cells": 0,
                        "max_cells": 4,
                        "enabled": True,
                    },
                    {
                        "fleet": "research",
                        "pool": "v6e-pool",
                        "accelerator_type": "v6e-16",
                        "capacity_class": "on_demand",
                        "region": "us-central1",
                        "zone": "us-central1-c",
                        "min_cells": 0,
                        "max_cells": 2,
                        "enabled": True,
                    },
                ]
            },
        )

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client = EyvanClient(client=raw)

    pools = client.list_capacity(
        org_id="acme",
        project_id="research",
        accelerator_type="v5p-8",
        capacity_class=CapacityClass.SPOT,
    )

    assert len(pools) == 2
    assert pools[0].fleet == "research"
    assert pools[0].accelerator_type == "v5p-8"
    assert pools[1].accelerator_type == "v6e-16"
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/api/v1/capacity"
    assert seen[0].url.params["accelerator_type"] == "v5p-8"
    assert seen[0].url.params["capacity_class"] == "spot"
    assert seen[0].url.params["org_id"] == "acme"
    assert seen[0].url.params["project_id"] == "research"
    client.close()
