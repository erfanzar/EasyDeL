# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kubernetes lifecycle regressions using only local API and fleet state doubles."""

from __future__ import annotations

import copy
import typing as tp

import httpx
import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderError, CellObservation
from eyvan.capacity.providers.kubernetes.api import KUEUE_API, QUEUE_NAME_LABEL
from eyvan.capacity.providers.kubernetes.workers import FENCE_ANNOTATION, GENERATION_LABEL
from eyvan.config import EyvanSettings
from eyvan.worker.agent.service import CellAgentService
from tests.test_kubernetes_capacity import _context, _desired, _driver, _FakeApi


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("provider_state", ["ready", "provisioning", "suspended"])
def test_fresh_drain_remains_draining_without_worker_endpoints(tmp_path, generation, provider_state) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    assert driver.apply_capacity("cell-a", _desired(), _context()).state is CapacityStatus.HEALTHY
    job = api.jobs["eyvan/eyvan-cell-a"]
    if provider_state == "provisioning":
        job["status"]["ready"] = 1
    elif provider_state == "suspended":
        job["spec"]["suspend"] = True
    drained = driver.drain("cell-a", _context(generation=generation, key="drain-fresh"))
    record = driver.state.get("cell-a")
    assert record is not None
    assert record.desired_state == "down"
    assert record.state is CapacityStatus.DRAINING
    assert drained.state is CapacityStatus.DRAINING
    assert api.deleted == []
    assert api.jobs["eyvan/eyvan-cell-a"] is job

    observed = driver.observe("cell-a")

    assert observed.state is CapacityStatus.DRAINING
    assert observed.generation == generation
    assert observed.metadata["desired_state"] == "down"
    assert observed.metadata["native_worker_endpoints"] == []
    assert observed.execution_endpoint is None
    assert observed.runtime_process_count is None


@pytest.mark.parametrize(
    ("provider_state", "expected"),
    [
        ("absent", CapacityStatus.ABSENT),
        ("failed", CapacityStatus.NEEDS_RECREATE),
        ("foreign", CapacityStatus.NEEDS_RECREATE),
        ("evicted", CapacityStatus.UNAVAILABLE),
    ],
)
def test_drain_does_not_hide_provider_loss_or_foreign_ownership(tmp_path, provider_state, expected) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    driver.drain("cell-a", _context(key="drain-fresh"))
    job = api.jobs["eyvan/eyvan-cell-a"]
    if provider_state == "absent":
        api.objects.pop("/apis/batch/v1/namespaces/eyvan/jobs/eyvan-cell-a")
    elif provider_state == "failed":
        job["status"]["failed"] = 1
    elif provider_state == "foreign":
        job["metadata"]["annotations"] = {}
    else:
        job["status"]["conditions"] = [{"status": "True", "reason": "Evicted"}]

    observed = driver.observe("cell-a")

    assert observed.state is expected
    assert observed.metadata["native_worker_endpoints"] == []
    assert observed.execution_endpoint is None
    assert api.deleted == []


class _ReadyResolver:
    def __init__(self) -> None:
        self.probes: list[str] = []

    def readiness(self, cell_id: str, observation: CellObservation) -> tuple[bool, str]:
        self.probes.append(cell_id)
        assert observation.state is CapacityStatus.HEALTHY
        return True, "ready"

    def resolve(self, cell_id: str, observation: CellObservation) -> tp.NoReturn:
        raise AssertionError("observation must not resolve an execution driver")


def test_public_agent_observation_preserves_fresh_drain_without_readiness_probe(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    resolver = _ReadyResolver()
    agent = CellAgentService(
        capacity=driver,
        state=driver.state,
        executors=resolver,
        settings=EyvanSettings(capacity_backend="kubernetes"),
    )
    assert agent.observe("cell-a").state is CapacityStatus.HEALTHY
    assert resolver.probes == ["cell-a"]
    driver.drain("cell-a", _context(key="drain-fresh"))

    observed = agent.observe("cell-a")

    assert observed.state is CapacityStatus.DRAINING
    assert observed.metadata["native_worker_endpoints"] == []
    assert observed.execution_endpoint is None
    assert resolver.probes == ["cell-a"]


class _SafeQueueApi(_FakeApi):
    def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        assert method == "GET"
        assert path == f"{KUEUE_API}/clusterqueues/research-cq"
        return httpx.Response(
            200,
            json={"spec": {"preemption": {"withinClusterQueue": "Never", "reclaimWithinCohort": "Never"}}},
            request=httpx.Request(method, f"https://local.invalid{path}"),
        )


def _queued_desired():
    return _desired(provider_options={"worker_count": 2, "resources": {"google.com/tpu": "4"}, "queue_name": "research"})


def test_exact_apply_replay_accepts_owned_kueue_unsuspension_without_resource_rotation(tmp_path) -> None:
    api = _SafeQueueApi()
    api.publish_queue()
    driver = _driver(tmp_path, api)
    assert driver.apply_capacity("cell-a", _queued_desired(), _context()).state is CapacityStatus.WAITING
    job = api.jobs["eyvan/eyvan-cell-a"]
    assert job["spec"]["suspend"] is True
    job["spec"]["suspend"] = False
    before = copy.deepcopy(api.objects)
    record_before = driver.state.get("cell-a")
    assert record_before is not None

    replayed = driver.apply_capacity("cell-a", _queued_desired(), _context())

    assert replayed.state is CapacityStatus.HEALTHY
    assert job["spec"]["suspend"] is False
    assert api.objects == before
    assert api.deleted == []
    record_after = driver.state.get("cell-a")
    assert record_after is not None
    assert record_after.provider_generation == record_before.provider_generation
    assert record_after.idempotency_keys == record_before.idempotency_keys
    assert record_after.metadata == record_before.metadata


@pytest.mark.parametrize("tamper", ["fence", "generation", "queue", "image", "credential", "node-selector"])
def test_kueue_unsuspension_does_not_allow_security_or_flavor_mutations(tmp_path, tamper) -> None:
    api = _SafeQueueApi()
    api.publish_queue()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _queued_desired(), _context())
    job = api.jobs["eyvan/eyvan-cell-a"]
    job["spec"]["suspend"] = False
    pod = job["spec"]["template"]["spec"]
    if tamper == "fence":
        job["metadata"]["annotations"][FENCE_ANNOTATION] = "f" * 32
    elif tamper == "generation":
        job["metadata"]["labels"][GENERATION_LABEL] = "2"
    elif tamper == "queue":
        job["metadata"]["labels"][QUEUE_NAME_LABEL] = "foreign-queue"
    elif tamper == "image":
        pod["containers"][0]["image"] = "registry.test/foreign@sha256:" + "b" * 64
    elif tamper == "credential":
        pod["volumes"][0]["secret"]["secretName"] = "foreign-secret"
    else:
        pod["nodeSelector"] = {"foreign-flavor": "true"}
    before = copy.deepcopy(api.objects)

    with pytest.raises(CapacityProviderError, match="Job conflict") as raised:
        driver.apply_capacity("cell-a", _queued_desired(), _context())

    assert raised.value.retryable is False
    assert api.objects == before
    assert api.deleted == []


@pytest.mark.parametrize("suspend", [None, "false", 0])
def test_kueue_replay_requires_a_boolean_unsuspension(tmp_path, suspend) -> None:
    api = _SafeQueueApi()
    api.publish_queue()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _queued_desired(), _context())
    api.jobs["eyvan/eyvan-cell-a"]["spec"]["suspend"] = suspend

    with pytest.raises(CapacityProviderError, match="Job conflict"):
        driver.apply_capacity("cell-a", _queued_desired(), _context())

    assert api.deleted == []
