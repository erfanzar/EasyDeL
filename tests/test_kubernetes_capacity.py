# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kubernetes-backed capacity: one cell is one Indexed Job.

The manifest builders are the primary test surface because they are where the
design decisions live, and every one of them is a decision that fails silently if
it is wrong: a non-deterministic name breaks idempotent create, a wrong
completion mode breaks rank ordering, a token in the wrong field leaks a
credential to anyone with ``get`` in the namespace.

The driver tests use a fake API rather than a cluster. That is a real limit and
it is stated: this proves the *control-plane* logic -- fencing, generation
bumping, leak detection, endpoint publication -- and proves nothing about whether
a real kube-scheduler places the pods.
"""

from __future__ import annotations

import base64
import copy
import json
import re
import threading
import typing as tp

import httpx
import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderError, MutationContext, StaleMutationError
from eyvan.capacity.drivers.kubernetes import MAX_WORKERS, KubernetesCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.capacity.providers.kubernetes.api import (
    KUEUE_API,
    QUEUE_NAME_LABEL,
    ApiObservation,
    JobStatus,
    capacity_state,
    interpret_job,
    local_queue_cluster_queue,
)
from eyvan.capacity.providers.kubernetes.kueue import foreign_workloads, inspect_cluster_queue
from eyvan.capacity.providers.kubernetes.workers import (
    CELL_ID_ANNOTATION,
    CELL_LABEL,
    FENCE_ANNOTATION,
    GENERATION_LABEL,
    MAX_INDEXED_JOB_NAME,
    MAX_OBJECT_NAME,
    cell_label_value,
    fence_fingerprint,
    headless_service_manifest,
    indexed_job_name,
    job_manifest,
    object_name,
    secret_manifest,
    worker_endpoints,
)
from eyvan.server.api.schemas import CellUpsertRequest

_TOKEN = "fencing-token-abcdef"
_WORKER_IMAGE = "registry.test/worker@sha256:" + "a" * 64
_DNS_1123_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$", re.ASCII)
_MAX_LENGTH_CELL_IDS = (
    pytest.param("é" * 96, id="unicode"),
    pytest.param("ResearchCell" * 8, id="mixed-case"),
    pytest.param("cell_/.:+" * 10 + "cell_/", id="punctuation"),
    pytest.param("a" * 96, id="long-ascii"),
)


def _job(**overrides: object) -> JsonObject:
    arguments: dict[str, tp.Any] = {
        "cell_id": "cell-a",
        "namespace": "eyvan",
        "image": _WORKER_IMAGE,
        "worker_count": 2,
        "provider_generation": 3,
        "fencing_token": _TOKEN,
        "resources": {"google.com/tpu": "4"},
    }
    arguments.update(overrides)
    return job_manifest(**arguments)


# --- manifest builders ------------------------------------------------------


def test_object_names_are_deterministic_which_is_what_makes_create_idempotent() -> None:
    assert object_name("eyvan", "cell-a") == object_name("eyvan", "cell-a")
    assert object_name("eyvan", "cell-a") != object_name("eyvan", "cell-b")


def test_a_long_cell_id_is_truncated_to_a_valid_dns_label() -> None:
    """Cell IDs reach 96 characters; a DNS-1123 label may not exceed 63."""

    name = object_name("eyvan", "c" * 120)

    assert _DNS_1123_LABEL.fullmatch(name) is not None
    assert len(name) <= MAX_OBJECT_NAME


def test_two_long_cell_ids_sharing_a_prefix_do_not_collide() -> None:
    """Truncation alone would map every long id with the same prefix to one name."""

    first_id = "c" * 100 + "-alpha"
    second_id = "c" * 100 + "-beta"
    first = object_name("eyvan", first_id)
    second = object_name("eyvan", second_id)

    assert first != second
    assert indexed_job_name(first_id) != indexed_job_name(second_id)


def test_normalized_cell_ids_do_not_collapse_onto_one_object_or_selector() -> None:
    """Replacement and case-folding need a digest just as truncation does."""

    cell_ids = ("cell-a", "Cell-A", "cell_a", "cell.a", "cell/a")
    names = {object_name("eyvan", cell_id) for cell_id in cell_ids}
    tokens = {cell_label_value(cell_id) for cell_id in cell_ids}

    assert len(names) == len(cell_ids)
    assert len(tokens) == len(cell_ids)
    assert all(_DNS_1123_LABEL.fullmatch(name) is not None for name in names)
    assert all(_DNS_1123_LABEL.fullmatch(token) is not None for token in tokens)


def test_a_schema_valid_unicode_cell_id_maps_to_admissible_ascii_names_and_labels() -> None:
    cell_id = "é" * 96
    request = CellUpsertRequest(
        id=cell_id,
        provider="kubernetes",
        region="us-east5",
        capacity={},
        allocatable={},
    )

    name = object_name("eyvan", request.id)
    token = cell_label_value(request.id)
    job = _job(cell_id=request.id)
    service = headless_service_manifest(request.id, "eyvan")
    secret = secret_manifest(request.id, "eyvan", "secret-token")

    assert request.id == cell_id
    assert name == object_name("eyvan", cell_id)
    assert token == cell_label_value(cell_id)
    assert _DNS_1123_LABEL.fullmatch(name) is not None
    assert _DNS_1123_LABEL.fullmatch(token) is not None
    assert all(ord(character) < 128 for character in name + token)
    assert job["metadata"]["name"] == indexed_job_name(cell_id)
    assert service["metadata"]["name"] == name
    assert job["spec"]["template"]["spec"]["subdomain"] == name
    assert _DNS_1123_LABEL.fullmatch(str(secret["metadata"]["name"])) is not None
    assert job["metadata"]["labels"][CELL_LABEL] == token
    assert job["spec"]["template"]["metadata"]["labels"][CELL_LABEL] == token
    assert service["spec"]["selector"][CELL_LABEL] == token
    assert secret["metadata"]["labels"][CELL_LABEL] == token


def test_long_cell_id_uses_one_safe_token_and_preserves_the_full_annotation() -> None:
    cell_id = "research-cell-" + "x" * 82
    token = cell_label_value(cell_id)
    job = _job(cell_id=cell_id)
    service = headless_service_manifest(cell_id, "eyvan")
    secret = secret_manifest(cell_id, "eyvan", "secret-token")

    assert len(cell_id) == 96
    assert _DNS_1123_LABEL.fullmatch(token) is not None
    assert job["metadata"]["labels"][CELL_LABEL] == token
    assert job["spec"]["template"]["metadata"]["labels"][CELL_LABEL] == token
    assert service["metadata"]["labels"][CELL_LABEL] == token
    assert service["spec"]["selector"][CELL_LABEL] == token
    assert secret["metadata"]["labels"][CELL_LABEL] == token
    for manifest in (job, service, secret):
        assert manifest["metadata"]["annotations"][CELL_ID_ANNOTATION] == cell_id


@pytest.mark.parametrize("cell_id", _MAX_LENGTH_CELL_IDS)
@pytest.mark.parametrize("worker_count", (1, 2, 10, 100, MAX_WORKERS))
def test_indexed_job_names_reserve_every_supported_completion_suffix(cell_id: str, worker_count: int) -> None:
    """The Job controller appends ``-<index>`` and validates the result as one label."""

    request = CellUpsertRequest(
        id=cell_id,
        provider="kubernetes",
        region="us-east5",
        capacity={},
        allocatable={},
    )
    job = _job(cell_id=request.id, worker_count=worker_count)
    job_name = str(job["metadata"]["name"])
    service_name = str(headless_service_manifest(cell_id, "eyvan")["metadata"]["name"])
    last_hostname = f"{job_name}-{worker_count - 1}"
    endpoints = worker_endpoints(cell_id, "eyvan", worker_count)

    assert job_name == indexed_job_name(cell_id)
    assert len(job_name) <= MAX_INDEXED_JOB_NAME
    assert len(last_hostname) <= MAX_OBJECT_NAME
    assert _DNS_1123_LABEL.fullmatch(last_hostname) is not None
    assert endpoints[-1].startswith(f"http://{last_hostname}.{service_name}.")
    assert job["spec"]["template"]["spec"]["subdomain"] == service_name
    assert job["metadata"]["annotations"][CELL_ID_ANNOTATION] == cell_id


def test_hashed_cell_label_tokens_resist_long_common_prefix_collisions() -> None:
    first = cell_label_value("c" * 95 + "a")
    second = cell_label_value("c" * 95 + "b")

    assert first != second
    assert first == cell_label_value("c" * 95 + "a")


def test_the_job_refuses_a_mutable_worker_image() -> None:
    with pytest.raises(ValueError, match="repository@sha256"):
        _job(image="registry.test/worker:latest")


def test_the_job_uses_the_dedicated_worker_account_without_an_api_token() -> None:
    pod_spec = _job()["spec"]["template"]["spec"]

    assert pod_spec["serviceAccountName"] == "eyvan-native-worker"
    assert pod_spec["automountServiceAccountToken"] is False


def test_the_job_is_indexed_with_parallelism_matching_the_worker_count() -> None:
    """Indexed completion is what gives each pod a stable rank and DNS name."""

    manifest = _job(worker_count=4)

    spec = manifest["spec"]
    assert spec["completionMode"] == "Indexed"
    assert spec["parallelism"] == 4
    assert spec["completions"] == 4


def test_the_job_never_retries_itself() -> None:
    """A cell is a capacity unit, not a retry unit: Eyvan's budgets decide reruns."""

    assert _job()["spec"]["backoffLimit"] == 0


def test_the_fencing_token_appears_nowhere_in_the_job_manifest() -> None:
    """Annotations are readable by anyone with ``get`` in the namespace."""

    serialized = json.dumps(_job())

    assert _TOKEN not in serialized
    assert fence_fingerprint(_TOKEN) in serialized


def test_the_fence_fingerprint_is_not_reversible_to_the_token() -> None:
    fingerprint = fence_fingerprint(_TOKEN)

    assert _TOKEN not in fingerprint
    assert fingerprint != fence_fingerprint("a-different-token")


def test_the_job_carries_its_cell_and_generation_as_labels() -> None:
    manifest = _job(provider_generation=7)

    labels = manifest["metadata"]["labels"]
    assert labels[CELL_LABEL] == "cell-a"
    assert labels[GENERATION_LABEL] == "7"
    assert manifest["metadata"]["annotations"][FENCE_ANNOTATION] == fence_fingerprint(_TOKEN)


def test_pods_carry_the_same_labels_so_the_headless_service_selects_them() -> None:
    manifest = _job()

    assert manifest["spec"]["template"]["metadata"]["labels"][CELL_LABEL] == "cell-a"
    assert headless_service_manifest("cell-a", "eyvan")["spec"]["selector"][CELL_LABEL] == "cell-a"


def test_the_pod_subdomain_matches_the_service_name() -> None:
    """Stable DNS is `<job>-<index>.<subdomain>.<ns>.svc`; a mismatch resolves nothing."""

    manifest = _job()
    service = headless_service_manifest("cell-a", "eyvan")

    assert manifest["spec"]["template"]["spec"]["subdomain"] == service["metadata"]["name"]


def test_a_provider_owned_service_carries_the_exact_generation_fence() -> None:
    fingerprint = fence_fingerprint(_TOKEN)
    service = headless_service_manifest("cell-a", "eyvan", provider_generation=3, fence=fingerprint)

    assert service["metadata"]["labels"][GENERATION_LABEL] == "3"
    assert service["metadata"]["annotations"][FENCE_ANNOTATION] == fingerprint


def test_the_service_is_headless_and_publishes_before_readiness() -> None:
    spec = headless_service_manifest("cell-a", "eyvan")["spec"]

    # The literal string "None", not a JSON null: that is Kubernetes' own spelling
    # for a headless Service, and a null here would allocate a cluster IP and give
    # every pod the same address.
    assert spec["clusterIP"] == "None"
    assert spec["publishNotReadyAddresses"] is True


def test_worker_endpoints_are_index_ordered_because_index_is_rank() -> None:
    """NativeSliceExecutor derives per-rank checkpoint URIs from position."""

    endpoints = worker_endpoints("cell-a", "eyvan", 3)

    assert len(endpoints) == 3
    assert endpoints[0].startswith("http://eyvan-cell-a-0.")
    assert endpoints[2].startswith("http://eyvan-cell-a-2.")
    assert endpoints == worker_endpoints("cell-a", "eyvan", 3)


def test_a_zero_worker_cell_is_refused() -> None:
    with pytest.raises(ValueError):
        worker_endpoints("cell-a", "eyvan", 0)
    with pytest.raises(ValueError):
        _job(worker_count=0)


def test_the_worker_index_comes_from_the_completion_index_annotation() -> None:
    """That annotation is what Kubernetes sets per Indexed pod; nothing else is the rank."""

    container = _job()["spec"]["template"]["spec"]["containers"][0]
    index_env = next(item for item in container["env"] if item["name"] == "EYVAN_WORKER_INDEX")

    assert "job-completion-index" in index_env["valueFrom"]["fieldRef"]["fieldPath"]


def test_a_secret_backed_token_is_mounted_never_passed_as_an_env_value() -> None:
    manifest = _job(secret_name="eyvan-token-cell-a")

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["volumeMounts"][0]["readOnly"] is True
    assert manifest["spec"]["template"]["spec"]["volumes"][0]["secret"]["secretName"] == "eyvan-token-cell-a"
    assert all("token" not in str(item.get("value", "")).lower() for item in container["env"])


def test_the_secret_manifest_holds_the_token_and_nothing_else_does() -> None:
    secret = secret_manifest("cell-a", "eyvan", token="derived-token")

    assert secret["stringData"]["token"] == "derived-token"
    assert "derived-token" not in json.dumps(_job())


def test_kueue_is_opt_in_and_only_then_is_the_job_suspended() -> None:
    """Kueue is not installed in most clusters; requiring it would make this untestable."""

    assert "suspend" not in _job()["spec"]
    assert _job(queue_name="research")["spec"]["suspend"] is True
    assert _job(queue_name="research")["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "research"


# --- status interpretation --------------------------------------------------


def test_an_absent_job_is_a_legitimate_observation() -> None:
    assert interpret_job(None).exists is False
    assert capacity_state(interpret_job(None))[0] is CapacityStatus.ABSENT


def test_a_suspended_job_is_waiting_so_the_queue_timeout_applies() -> None:
    """It is queued behind admission; nothing is being built yet."""

    status = interpret_job({"spec": {"suspend": True, "parallelism": 2}, "status": {}})

    assert capacity_state(status)[0] is CapacityStatus.WAITING


def test_a_partially_ready_job_is_provisioning() -> None:
    status = interpret_job({"spec": {"parallelism": 4}, "status": {"ready": 2}})

    assert capacity_state(status)[0] is CapacityStatus.PROVISIONING


def test_a_fully_ready_job_is_healthy() -> None:
    status = interpret_job({"spec": {"parallelism": 4}, "status": {"ready": 4}})

    assert capacity_state(status)[0] is CapacityStatus.HEALTHY


def test_a_failed_job_needs_recreation_because_nothing_will_retry_it() -> None:
    status = interpret_job({"spec": {"parallelism": 2}, "status": {"failed": 1}})

    assert capacity_state(status)[0] is CapacityStatus.NEEDS_RECREATE


def test_kueue_inadmissible_is_waiting_and_eviction_is_unavailable() -> None:
    """Different states because only one of them destroyed something."""

    inadmissible = interpret_job(
        {"spec": {"parallelism": 2}, "status": {"conditions": [{"reason": "Inadmissible", "status": "True"}]}}
    )
    evicted = interpret_job(
        {"spec": {"parallelism": 2}, "status": {"conditions": [{"reason": "Evicted", "status": "True"}]}}
    )

    assert capacity_state(inadmissible)[0] is CapacityStatus.WAITING
    assert capacity_state(evicted)[0] is CapacityStatus.UNAVAILABLE
    assert "kueue" in capacity_state(evicted)[1]


def test_a_false_kueue_condition_is_not_treated_as_active() -> None:
    status = interpret_job(
        {"spec": {"parallelism": 1}, "status": {"ready": 1, "conditions": [{"reason": "Evicted", "status": "False"}]}}
    )

    assert capacity_state(status)[0] is CapacityStatus.HEALTHY


def test_malformed_status_counts_do_not_crash_interpretation() -> None:
    """A field the API server never sends must not take down reconciliation."""

    status = interpret_job({"spec": {"parallelism": "two"}, "status": {"ready": None, "failed": True}})

    assert status.parallelism == 0
    assert status.ready == 0
    assert status.failed == 0


# --- what a queued cell carries about its own queue and shape ---------------


def test_the_queue_label_the_manifest_writes_is_the_one_interpretation_reads() -> None:
    """One spelling, proven by round trip rather than by two matching literals.

    ``job_manifest`` writes the label and ``interpret_job`` reads it back; a
    silent drift between the two would leave every queued cell reporting no
    queue at all, which is indistinguishable from a cluster without Kueue.
    """

    status = interpret_job(_job(queue_name="research-lq"))

    assert status.queue_label == "research-lq"
    assert _job(queue_name="research-lq")["metadata"]["labels"][QUEUE_NAME_LABEL] == "research-lq"


def test_a_job_created_without_kueue_reports_no_queue() -> None:
    """Kueue is opt-in, and an absent label is not an empty queue name to guess at."""

    assert interpret_job(_job()).queue_label == ""


def test_the_cells_request_is_read_from_the_job_the_driver_already_fetched() -> None:
    """No second GET: the shape a cell asks the cluster for is in its own Job."""

    status = interpret_job(_job(resources={"google.com/tpu": "4", "cpu": "8"}))

    assert status.resource_requests == (("cpu", "8"), ("google.com/tpu", "4"))


def test_a_job_declaring_only_limits_reports_them_as_the_request() -> None:
    """Kubernetes defaults requests to limits, so reading nothing would be wrong."""

    payload = copy.deepcopy(_job())
    container = payload["spec"]["template"]["spec"]["containers"][0]
    container["resources"] = {"limits": {"google.com/tpu": 4}}

    assert interpret_job(payload).resource_requests == (("google.com/tpu", "4"),)


def test_an_unquantifiable_resource_value_is_dropped_rather_than_stringified() -> None:
    """A dict is not a quantity, and ``str()`` of one is a quantity nothing can parse."""

    payload = copy.deepcopy(_job())
    container = payload["spec"]["template"]["spec"]["containers"][0]
    container["resources"] = {"requests": {"google.com/tpu": {"amount": 4}, "cpu": "8"}}

    assert interpret_job(payload).resource_requests == (("cpu", "8"),)


def test_a_job_with_no_readable_container_reports_no_request() -> None:
    """Unknown shape, never a request for nothing."""

    payload = copy.deepcopy(_job())
    payload["spec"]["template"]["spec"]["containers"] = []

    assert interpret_job(payload).resource_requests == ()


def test_a_local_queue_names_the_cluster_queue_that_holds_the_quota() -> None:
    """The label on a Job is a LocalQueue; the quota is on the ClusterQueue."""

    observation = ApiObservation(payload={"spec": {"clusterQueue": "research-cq"}}, readable=True)

    assert local_queue_cluster_queue(observation) == "research-cq"


@pytest.mark.parametrize(
    ("observation", "why"),
    [
        (ApiObservation(payload=None, readable=False, reason="HTTP 403"), "unreadable"),
        (ApiObservation(payload=None, readable=True), "absent"),
        (ApiObservation(payload={"spec": {}}, readable=True), "no clusterQueue"),
        (ApiObservation(payload={"spec": {"clusterQueue": "   "}}, readable=True), "blank clusterQueue"),
        (ApiObservation(payload={"spec": "research"}, readable=True), "malformed spec"),
    ],
)
def test_a_local_queue_that_names_no_cluster_queue_resolves_to_nothing(observation: ApiObservation, why: str) -> None:
    """Never the LocalQueue's own name: a reading taken against the wrong object
    answers with numbers, which is worse than not answering."""

    assert local_queue_cluster_queue(observation) is None, why


# --- Kueue preconditions -----------------------------------------------------


def test_a_missing_cluster_queue_is_a_finding_not_a_clean_bill() -> None:
    """A pool pointed at a nonexistent queue suspends every Job forever."""

    findings = inspect_cluster_queue(None, "research")

    assert len(findings) == 1
    assert "does not exist" in findings[0].problem


def test_a_queue_with_preemption_disabled_is_safe() -> None:
    payload = {"spec": {"preemption": {"withinClusterQueue": "Never", "reclaimWithinCohort": "Never"}}}

    assert inspect_cluster_queue(payload, "research") == ()


def test_within_queue_preemption_is_reported() -> None:
    """Eviction destroys a pod with no checkpoint hook, bypassing PreemptionPlan."""

    payload = {"spec": {"preemption": {"withinClusterQueue": "LowerPriority"}}}

    findings = inspect_cluster_queue(payload, "research")

    assert len(findings) == 1
    assert "withinClusterQueue" in findings[0].problem
    assert "Never" in findings[0].remedy


def test_cohort_reclaim_is_reported_separately() -> None:
    payload = {"spec": {"preemption": {"reclaimWithinCohort": "Any"}}}

    findings = inspect_cluster_queue(payload, "research")

    assert len(findings) == 1
    assert "reclaimWithinCohort" in findings[0].problem


def test_a_workload_without_the_cell_label_is_foreign() -> None:
    """Absence of the label is the only test that cannot be spoofed."""

    workloads = [
        {"metadata": {"name": "eyvan-cell-a", "labels": {CELL_LABEL: "cell-a"}}},
        {"metadata": {"name": "someone-elses-job", "labels": {"app": "batch"}}},
        {"metadata": {"name": "unlabelled"}},
    ]

    assert foreign_workloads(workloads) == ("someone-elses-job", "unlabelled")


# --- driver lifecycle --------------------------------------------------------


class _FakeApi:
    """Record calls and model exact named-object conflicts and GETs."""

    def __init__(self) -> None:
        self.objects: dict[str, JsonObject] = {}
        self.jobs: dict[str, JsonObject] = {}
        self.created: list[tuple[str, JsonObject]] = []
        self.deleted: list[str] = []
        self.delete_preconditions: list[tuple[str, str | None, str | None]] = []
        self.deletion_pending: set[str] = set()
        self.got: list[str] = []
        self.get_overrides: dict[str, object] = {}
        #: Every describing read, in order. Separate from ``got`` because these
        #: never mutate and never raise, and their *count* is the property the
        #: capacity cache exists to bound.
        self.observed: list[str] = []
        self.observations: dict[str, ApiObservation] = {}
        self._next_uid = 1

    # -- describing reads: never raise, and a 404 is a readable answer --------

    def observe(self, path: str, *, params: tp.Mapping[str, str] | None = None) -> ApiObservation:
        del params
        self.observed.append(path)
        return self.observations.get(path, ApiObservation(payload=None, readable=True))

    def observe_local_queue(self, namespace: str, name: str) -> ApiObservation:
        return self.observe(f"{KUEUE_API}/namespaces/{namespace}/localqueues/{name}")

    def observe_cluster_queue(self, name: str) -> ApiObservation:
        return self.observe(f"{KUEUE_API}/clusterqueues/{name}")

    def observe_resource_flavor(self, name: str) -> ApiObservation:
        return self.observe(f"{KUEUE_API}/resourceflavors/{name}")

    def observe_nodes(self, *, label_selector: str = "", limit: int = 500) -> ApiObservation:
        del limit
        return self.observe(f"/api/v1/nodes?{label_selector}")

    def publish_queue(
        self,
        *,
        namespace: str = "eyvan",
        local_queue: str = "research",
        cluster_queue: str = "research-cq",
        payload: JsonObject | None = None,
    ) -> None:
        """Install a LocalQueue that routes into a ClusterQueue with a reading."""

        self.observations[f"{KUEUE_API}/namespaces/{namespace}/localqueues/{local_queue}"] = ApiObservation(
            payload={"spec": {"clusterQueue": cluster_queue}}, readable=True
        )
        if payload is not None:
            self.observations[f"{KUEUE_API}/clusterqueues/{cluster_queue}"] = ApiObservation(
                payload=payload, readable=True
            )

    def _store(self, path: str, manifest: JsonObject) -> JsonObject:
        stored = copy.deepcopy(manifest)
        metadata = stored["metadata"]
        assert isinstance(metadata, dict)
        metadata.setdefault("uid", f"uid-{self._next_uid}")
        metadata.setdefault("resourceVersion", str(self._next_uid))
        self._next_uid += 1
        if stored.get("kind") == "Secret":
            string_data = stored.pop("stringData", {})
            assert isinstance(string_data, dict)
            stored["data"] = {key: base64.b64encode(str(value).encode()).decode() for key, value in string_data.items()}
        if stored.get("kind") == "Job":
            spec = stored["spec"]
            assert isinstance(spec, dict)
            stored["status"] = {"ready": spec["parallelism"]}
            self.jobs[f"{metadata['namespace']}/{metadata['name']}"] = stored
        self.objects[path] = stored
        return stored

    def seed(self, path: str, manifest: JsonObject) -> JsonObject:
        """Install an already-existing API object with realistic fencing metadata."""

        return self._store(path, manifest)

    def get(self, path: str) -> JsonObject | None:
        self.got.append(path)
        if path in self.get_overrides:
            return tp.cast(JsonObject | None, self.get_overrides[path])
        return self.objects.get(path)

    def get_job(self, namespace: str, name: str) -> JsonObject | None:
        return self.get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}")

    def create(self, path: str, manifest: JsonObject) -> bool:
        self.created.append((path, copy.deepcopy(manifest)))
        metadata = manifest["metadata"]
        assert isinstance(metadata, dict)
        object_path = f"{path}/{metadata['name']}"
        if object_path in self.objects:
            return False
        self._store(object_path, manifest)
        return True

    def delete(
        self,
        path: str,
        *,
        uid: str | None = None,
        resource_version: str | None = None,
    ) -> None:
        existing = self.objects.get(path)
        if existing is not None and (uid is not None or resource_version is not None):
            metadata = existing["metadata"]
            assert isinstance(metadata, dict)
            assert metadata.get("uid") == uid
            assert metadata.get("resourceVersion") == resource_version
        self.deleted.append(path)
        self.delete_preconditions.append((path, uid, resource_version))
        if path in self.deletion_pending:
            return
        removed = self.objects.pop(path, None)
        if removed is not None and removed.get("kind") == "Job":
            metadata = removed["metadata"]
            assert isinstance(metadata, dict)
            self.jobs.pop(f"{metadata['namespace']}/{metadata['name']}", None)


def _driver(tmp_path, api: _FakeApi) -> KubernetesCapacityDriver:
    return KubernetesCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        api=api,
        namespace="eyvan",
        image=_WORKER_IMAGE,
        shared_secret="agent-secret",
    )


def _desired(**overrides: object) -> dict[str, tp.Any]:
    payload: dict[str, tp.Any] = {
        "provider": "kubernetes",
        "capacity_backend": "kubernetes",
        "executor": "eyvan-native",
        "accelerator_type": "tpu",
        "capacity_class": "on-demand",
        "region": "us-east5",
        "provider_options": {"worker_count": 2, "resources": {"google.com/tpu": "4"}},
    }
    payload.update(overrides)
    return payload


def _context(generation: int = 1, key: str = "apply-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=_TOKEN)


def test_apply_creates_the_service_before_the_job(tmp_path) -> None:
    """Pods whose DNS name does not resolve at start have nothing to connect to."""

    api = _FakeApi()
    _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())

    paths = [path for path, _manifest in api.created]
    assert paths.index("/api/v1/namespaces/eyvan/services") < paths.index("/apis/batch/v1/namespaces/eyvan/jobs")


def test_apply_creates_the_secret_before_the_job(tmp_path) -> None:
    """A pod that starts and fails to mount burns the backoffLimit-0 budget."""

    api = _FakeApi()
    _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())

    paths = [path for path, _manifest in api.created]
    assert paths.index("/api/v1/namespaces/eyvan/secrets") < paths.index("/apis/batch/v1/namespaces/eyvan/jobs")


def test_newer_release_cannot_pass_an_apply_paused_before_service_create(tmp_path) -> None:
    """The durable tombstone and deterministic provider names are one mutation window."""

    class ServiceCreateBarrierApi(_FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.before_service_create = threading.Barrier(2)
            self.resume_service_create = threading.Event()
            self.paused = False

        def create(self, path: str, manifest: JsonObject) -> bool:
            if path.endswith("/services") and not self.paused:
                self.paused = True
                self.before_service_create.wait(timeout=5)
                assert self.resume_service_create.wait(timeout=5)
            return super().create(path, manifest)

    api = ServiceCreateBarrierApi()
    driver = _driver(tmp_path, api)
    observations: dict[str, tp.Any] = {}
    errors: list[BaseException] = []
    release_returned = threading.Event()

    def apply_generation_one() -> None:
        try:
            observations["apply"] = driver.apply_capacity(
                "cell-a",
                _desired(),
                _context(generation=1, key="apply-1"),
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def release_generation_two() -> None:
        try:
            observations["release"] = driver.release(
                "cell-a",
                MutationContext(
                    idempotency_key="release-2",
                    generation=2,
                    fencing_token="generation-two-token",
                ),
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            release_returned.set()

    apply_thread = threading.Thread(target=apply_generation_one)
    apply_thread.start()
    api.before_service_create.wait(timeout=5)

    contender = FleetStateStore(LocalBackend(tmp_path / "fleet.json"), owner="competing-agent")
    lease_covers_service_create = not contender.acquire_lease()
    if not lease_covers_service_create:  # keep a broken implementation from stranding the test
        contender.release_lease()

    release_thread = threading.Thread(target=release_generation_two)
    release_thread.start()
    returned_before_apply_resumed = release_returned.wait(timeout=0.25)
    api.resume_service_create.set()
    apply_thread.join(timeout=5)
    release_thread.join(timeout=5)

    assert lease_covers_service_create is True
    assert returned_before_apply_resumed is False
    assert not apply_thread.is_alive() and not release_thread.is_alive()
    assert errors == []
    assert observations["apply"].state is CapacityStatus.HEALTHY
    assert observations["release"].state is CapacityStatus.RELEASED
    assert driver.state.get("cell-a") is None
    assert api.objects == {}

    with pytest.raises(StaleMutationError):
        driver.apply_capacity("cell-a", _desired(), _context(generation=1, key="apply-1-replayed"))
    assert api.objects == {}


def test_apply_reports_lease_loss_and_a_newer_release_cleans_partial_resources(tmp_path) -> None:
    class LeaseLosingApi(_FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.lose_lease: tp.Callable[[], None] | None = None

        def create(self, path: str, manifest: JsonObject) -> bool:
            if path.endswith("/services") and self.lose_lease is not None:
                lose_lease, self.lose_lease = self.lose_lease, None
                lose_lease()
            return super().create(path, manifest)

    api = LeaseLosingApi()
    driver = _driver(tmp_path, api)
    api.lose_lease = driver.state.release_lease

    with pytest.raises(StaleMutationError, match="fleet mutation lease"):
        driver.apply_capacity("cell-a", _desired(), _context(generation=1, key="apply-1"))

    assert driver.state.lease_holder() is None
    assert api.objects == {}, "the settled stale POST must clean its exact UID/resourceVersion"
    record = driver.state.get("cell-a")
    assert record is not None and "kubernetes_pending_creates" not in record.metadata

    observation = driver.release(
        "cell-a",
        MutationContext(
            idempotency_key="release-2",
            generation=2,
            fencing_token="generation-two-token",
        ),
    )

    assert observation.state is CapacityStatus.RELEASED
    assert driver.state.get("cell-a") is None
    assert api.objects == {}
    assert driver.state.lease_holder() is None


def test_release_retries_an_inflight_post_then_tombstone_replay_cleans_only_the_predecessor(tmp_path) -> None:
    """Inject lease loss after intent persistence but immediately before API storage."""

    class PauseBeforeServiceStoreApi(_FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.lose_lease: tp.Callable[[], None] | None = None
            self.before_store = threading.Event()
            self.resume_store = threading.Event()

        def create(self, path: str, manifest: JsonObject) -> bool:
            if path.endswith("/services") and self.lose_lease is not None:
                lose_lease, self.lose_lease = self.lose_lease, None
                lose_lease()
                self.before_store.set()
                assert self.resume_store.wait(timeout=5)
            return super().create(path, manifest)

    path = tmp_path / "fleet.json"
    api = PauseBeforeServiceStoreApi()
    generation_one = KubernetesCapacityDriver(
        FleetStateStore(LocalBackend(path), owner="generation-one-agent"),
        api=api,
        namespace="eyvan",
        image=_WORKER_IMAGE,
        shared_secret="agent-secret",
    )
    generation_two = KubernetesCapacityDriver(
        FleetStateStore(LocalBackend(path), owner="generation-two-agent"),
        api=api,
        namespace="eyvan",
        image=_WORKER_IMAGE,
        shared_secret="agent-secret",
    )
    api.lose_lease = generation_one.state.release_lease
    apply_errors: list[BaseException] = []

    def apply_generation_one() -> None:
        try:
            generation_one.apply_capacity("cell-a", _desired(), _context(generation=1, key="apply-1"))
        except BaseException as exc:  # pragma: no cover - asserted below
            apply_errors.append(exc)

    apply_thread = threading.Thread(target=apply_generation_one)
    apply_thread.start()
    assert api.before_store.wait(timeout=5)
    release_context = MutationContext(
        idempotency_key="release-2",
        generation=2,
        fencing_token="generation-two-token",
    )

    with pytest.raises(CapacityProviderError, match="POST is still inflight") as raised:
        generation_two.release("cell-a", release_context)

    assert raised.value.retryable is True
    assert api.objects == {}
    assert api.got == [] and api.deleted == []
    assert generation_two.state.get("cell-a") is not None

    api.resume_store.set()
    apply_thread.join(timeout=5)
    assert not apply_thread.is_alive()
    assert len(apply_errors) == 1 and isinstance(apply_errors[0], StaleMutationError)
    assert api.objects == {}, "the returned stale POST must delete only its exact object"

    released = generation_two.release("cell-a", release_context)
    assert released.state is CapacityStatus.RELEASED
    assert generation_two.state.get("cell-a") is None
    tombstone = generation_two.state.cell_tombstone("cell-a")
    assert tombstone is not None and "kubernetes_cleanup" in tombstone

    # Reproduce a process dying after the API stored the old response but before
    # its post-check could clean it. A record-absent replay must still GET and
    # UID/resourceVersion-fenced delete that exact predecessor.
    stale_service = headless_service_manifest(
        "cell-a",
        "eyvan",
        provider_generation=1,
        fence=fence_fingerprint(_TOKEN),
    )
    service_path = "/api/v1/namespaces/eyvan/services/eyvan-cell-a"
    stored_stale_service = api.seed(service_path, stale_service)
    stored_stale_metadata = stored_stale_service["metadata"]
    assert isinstance(stored_stale_metadata, dict)
    stale_uid = str(stored_stale_metadata["uid"])
    stale_resource_version = str(stored_stale_metadata["resourceVersion"])
    api.got.clear()
    api.deleted.clear()
    api.delete_preconditions.clear()

    replayed = generation_two.release("cell-a", release_context)

    assert replayed.state is CapacityStatus.RELEASED
    assert service_path in api.got
    assert api.objects == {}
    assert api.delete_preconditions == [(service_path, stale_uid, stale_resource_version)]

    # The same deterministic name with a newer generation/fence is a collision,
    # not cleanup authority. The all-object preflight prevents any partial delete.
    newer_service = headless_service_manifest(
        "cell-a",
        "eyvan",
        provider_generation=2,
        fence=fence_fingerprint("newer-provider-owner"),
    )
    api.seed(service_path, newer_service)
    api.deleted.clear()
    api.delete_preconditions.clear()

    with pytest.raises(CapacityProviderError, match="Service conflict"):
        generation_two.release("cell-a", release_context)

    assert api.objects[service_path]["metadata"]["labels"][GENERATION_LABEL] == "2"
    assert api.deleted == [] and api.delete_preconditions == []


def test_provider_failure_releases_the_fleet_lease(tmp_path) -> None:
    class FailingApi(_FakeApi):
        def create(self, path: str, manifest: JsonObject) -> bool:
            if path.endswith("/services"):
                raise RuntimeError("injected Service create failure")
            return super().create(path, manifest)

    driver = _driver(tmp_path, FailingApi())

    with pytest.raises(RuntimeError, match="injected Service create failure"):
        driver.apply_capacity("cell-a", _desired(), _context())

    assert driver.state.lease_holder() is None


def test_a_ready_cell_publishes_one_endpoint_per_worker(tmp_path) -> None:
    api = _FakeApi()
    observation = _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())

    assert observation.state is CapacityStatus.HEALTHY
    assert len(observation.metadata["native_worker_endpoints"]) == 2
    assert observation.runtime_process_count == 2


@pytest.mark.parametrize("cell_id", _MAX_LENGTH_CELL_IDS)
def test_driver_apply_replay_and_observe_use_the_indexed_job_name_at_maximum_scale(tmp_path, cell_id: str) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    desired = _desired(
        provider_options={"worker_count": MAX_WORKERS, "resources": {"google.com/tpu": "4"}},
    )
    job_name = indexed_job_name(cell_id)
    service_name = object_name("eyvan", cell_id)
    job_path = f"/apis/batch/v1/namespaces/eyvan/jobs/{job_name}"
    legacy_job_path = f"/apis/batch/v1/namespaces/eyvan/jobs/{service_name}"
    service_path = f"/api/v1/namespaces/eyvan/services/{service_name}"

    applied = driver.apply_capacity(cell_id, desired, _context())
    api.got.clear()
    replayed = driver.apply_capacity(cell_id, desired, _context())
    observed = driver.observe(cell_id)

    assert len(cell_id) == 96
    assert applied.state is CapacityStatus.HEALTHY
    assert replayed.state is CapacityStatus.HEALTHY
    assert observed.state is CapacityStatus.HEALTHY
    assert observed.metadata["resource_id"] == f"kubernetes:eyvan/{job_name}"
    assert len(observed.metadata["native_worker_endpoints"]) == MAX_WORKERS
    assert observed.metadata["native_worker_endpoints"][-1] == (
        f"http://{job_name}-{MAX_WORKERS - 1}.{service_name}.eyvan.svc:8100"
    )
    assert job_path in api.objects
    assert service_path in api.objects
    assert job_path in api.got
    if legacy_job_path != job_path:
        assert legacy_job_path not in api.got


@pytest.mark.parametrize("cell_id", _MAX_LENGTH_CELL_IDS)
def test_driver_release_and_its_replay_use_the_indexed_job_name_at_maximum_scale(tmp_path, cell_id: str) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    desired = _desired(
        provider_options={"worker_count": MAX_WORKERS, "resources": {"google.com/tpu": "4"}},
    )
    driver.apply_capacity(cell_id, desired, _context())
    job_path = f"/apis/batch/v1/namespaces/eyvan/jobs/{indexed_job_name(cell_id)}"
    legacy_job_path = f"/apis/batch/v1/namespaces/eyvan/jobs/{object_name('eyvan', cell_id)}"
    service_path = f"/api/v1/namespaces/eyvan/services/{object_name('eyvan', cell_id)}"
    api.got.clear()
    api.deleted.clear()

    released = driver.release(cell_id, _context())

    assert released.state is CapacityStatus.RELEASED
    assert job_path in api.got
    assert job_path in api.deleted
    assert service_path in api.got
    assert service_path in api.deleted
    if legacy_job_path != job_path:
        assert legacy_job_path not in api.got
        assert legacy_job_path not in api.deleted
    assert api.objects == {}

    api.got.clear()
    replayed = driver.release(cell_id, _context())

    assert replayed.state is CapacityStatus.RELEASED
    assert job_path in api.got
    assert service_path in api.got
    if legacy_job_path != job_path:
        assert legacy_job_path not in api.got


def test_endpoints_are_withheld_until_every_worker_is_ready(tmp_path) -> None:
    """Publishing early makes the readiness probe resolve a nonexistent DNS name."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    for job in api.jobs.values():
        job["status"] = {"ready": 1}

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.metadata["native_worker_endpoints"] == []
    assert observation.runtime_process_count is None


def test_the_management_uid_is_derived_and_includes_the_namespace(tmp_path) -> None:
    """One cell_id in two clusters must not collide on a derived worker token."""

    api = _FakeApi()
    first = _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())
    other = KubernetesCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "other.json")),
        api=_FakeApi(),
        namespace="other-namespace",
        image=_WORKER_IMAGE,
        shared_secret="agent-secret",
    ).apply_capacity("cell-a", _desired(), _context())

    assert first.metadata["management_uid"] != other.metadata["management_uid"]


def test_reapplying_the_same_request_does_not_bump_the_generation(tmp_path) -> None:
    """A replayed outbox command must not invalidate every running worker's token."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    first = driver.apply_capacity("cell-a", _desired(), _context())

    second = driver.apply_capacity("cell-a", _desired(), _context())

    assert second.metadata["provider_generation"] == first.metadata["provider_generation"]


def test_an_exact_create_retry_gets_and_verifies_every_conflict(tmp_path) -> None:
    """A deterministic name routes the GET; equality, not 409, proves replay."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    first = driver.apply_capacity("cell-a", _desired(), _context())
    api.got.clear()
    api.deleted.clear()

    second = driver.apply_capacity("cell-a", _desired(), _context())

    assert second.metadata["provider_generation"] == first.metadata["provider_generation"]
    assert "/api/v1/namespaces/eyvan/services/eyvan-cell-a" in api.got
    assert "/api/v1/namespaces/eyvan/secrets/eyvan-token-cell-a" in api.got
    assert "/apis/batch/v1/namespaces/eyvan/jobs/eyvan-cell-a" in api.got
    assert api.deleted == []


def test_a_stale_secret_token_is_rotated_only_after_the_old_job_is_gone(tmp_path) -> None:
    """The old pod must never observe the next generation's mounted credential."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    secret_path = "/api/v1/namespaces/eyvan/secrets/eyvan-token-cell-a"
    first_secret = copy.deepcopy(api.objects[secret_path])
    api.deleted.clear()

    observation = driver.apply_capacity("cell-a", _desired(), _context(generation=2, key="apply-2"))

    job_delete = "/apis/batch/v1/namespaces/eyvan/jobs/eyvan-cell-a"
    assert api.deleted.index(job_delete) < api.deleted.index(secret_path)
    assert observation.metadata["provider_generation"] == 2
    replacement = api.objects[secret_path]
    assert replacement["data"] != first_secret["data"]
    assert replacement["metadata"]["labels"][GENERATION_LABEL] == "2"
    assert first_secret["data"]["token"] not in json.dumps(api.objects)


def test_a_colliding_service_is_not_adopted_or_deleted(tmp_path) -> None:
    api = _FakeApi()
    service_path = "/api/v1/namespaces/eyvan/services/eyvan-cell-a"
    collision = headless_service_manifest("cell-a", "eyvan")
    collision["spec"]["selector"] = {CELL_LABEL: "another-cell"}
    api.seed(service_path, collision)

    with pytest.raises(CapacityProviderError, match="Service conflict") as raised:
        _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())

    assert raised.value.retryable is False
    assert api.deleted == []
    assert all(not path.endswith(("/secrets", "/jobs")) for path, _manifest in api.created)


def test_a_same_name_job_with_different_security_critical_spec_fails_closed(tmp_path) -> None:
    api = _FakeApi()
    job_path = "/apis/batch/v1/namespaces/eyvan/jobs/eyvan-cell-a"
    collision = _job(
        provider_generation=1,
        secret_name="eyvan-token-cell-a",
        image="registry.test/other@sha256:" + "b" * 64,
    )
    api.seed(job_path, collision)

    with pytest.raises(CapacityProviderError, match="Job conflict") as raised:
        _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())

    assert raised.value.retryable is False
    assert api.deleted == []
    assert api.objects[job_path]["spec"]["template"]["spec"]["containers"][0]["image"].endswith("b" * 64)


def test_a_malformed_conflict_get_is_never_accepted_as_replay(tmp_path) -> None:
    api = _FakeApi()
    service_path = "/api/v1/namespaces/eyvan/services/eyvan-cell-a"
    api.seed(service_path, headless_service_manifest("cell-a", "eyvan"))
    api.get_overrides[service_path] = []

    with pytest.raises(CapacityProviderError, match="non-object") as raised:
        _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())

    assert raised.value.retryable is False
    assert api.deleted == []


def test_a_stale_generation_is_refused(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context(generation=5, key="apply-5"))

    with pytest.raises(StaleMutationError):
        driver.apply_capacity("cell-a", _desired(), _context(generation=2, key="apply-2"))


def test_a_job_from_a_previous_incarnation_is_not_adopted(tmp_path) -> None:
    """Its workers hold a token derived from a generation this record moved past."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    for job in api.jobs.values():
        job["metadata"]["annotations"][FENCE_ANNOTATION] = fence_fingerprint("someone-elses-token")

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert "ownership metadata" in observation.detail


def test_observe_fails_closed_when_job_ownership_metadata_is_missing(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    job = next(iter(api.jobs.values()))
    job["metadata"]["labels"] = {}
    job["metadata"]["annotations"] = {}

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.execution_endpoint is None
    assert observation.metadata["native_worker_endpoints"] == []
    assert api.deleted == []


def test_release_deletes_the_job_first(tmp_path) -> None:
    """Deleting the Secret first strands running pods on a missing volume."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())

    driver.release("cell-a", _context())

    assert "/jobs/" in api.deleted[0]
    assert any("/secrets/" in path for path in api.deleted)
    assert any("/services/" in path for path in api.deleted)
    assert len(api.delete_preconditions) == 3
    assert all(uid and resource_version for _path, uid, resource_version in api.delete_preconditions)


def test_release_provider_failure_keeps_the_record_and_releases_the_fleet_lease(tmp_path) -> None:
    class FailingDeleteApi(_FakeApi):
        fail_delete = False

        def delete(
            self,
            path: str,
            *,
            uid: str | None = None,
            resource_version: str | None = None,
        ) -> None:
            if self.fail_delete:
                raise RuntimeError("injected Job delete failure")
            super().delete(path, uid=uid, resource_version=resource_version)

    api = FailingDeleteApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    api.fail_delete = True

    with pytest.raises(RuntimeError, match="injected Job delete failure"):
        driver.release("cell-a", _context(key="release-fails"))

    assert driver.state.get("cell-a") is not None
    assert driver.state.lease_holder() is None

    api.fail_delete = False
    observation = driver.release("cell-a", _context(key="release-retry"))
    assert observation.state is CapacityStatus.RELEASED
    assert api.objects == {}


def test_newer_generation_release_deletes_resources_owned_by_the_durable_predecessor(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context(generation=1, key="apply-1"))

    observation = driver.release(
        "cell-a",
        MutationContext(idempotency_key="release-2", generation=2, fencing_token="new-owner-token"),
    )

    assert observation.state is CapacityStatus.RELEASED
    assert api.objects == {}
    assert all(uid and resource_version for _path, uid, resource_version in api.delete_preconditions)


def test_stale_release_deletes_nothing(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context(generation=1, key="apply-1"))
    driver.apply_capacity(
        "cell-a",
        _desired(),
        MutationContext(idempotency_key="apply-2", generation=2, fencing_token="new-owner-token"),
    )
    api.deleted.clear()
    api.delete_preconditions.clear()
    before = copy.deepcopy(api.objects)

    with pytest.raises(StaleMutationError):
        driver.release("cell-a", _context(generation=1, key="stale-release"))

    assert api.deleted == []
    assert api.delete_preconditions == []
    assert api.objects == before


def test_release_collision_preflight_deletes_nothing(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    service_path = "/api/v1/namespaces/eyvan/services/eyvan-cell-a"
    api.objects[service_path]["metadata"]["annotations"].pop(CELL_ID_ANNOTATION)
    api.deleted.clear()
    api.delete_preconditions.clear()
    before = copy.deepcopy(api.objects)

    with pytest.raises(CapacityProviderError, match="Service conflict") as raised:
        driver.release("cell-a", _context(key="release-collision"))

    assert raised.value.retryable is False
    assert api.deleted == []
    assert api.delete_preconditions == []
    assert api.objects == before


def test_release_waits_for_foreground_job_absence_before_secret_and_service(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())
    job_path = "/apis/batch/v1/namespaces/eyvan/jobs/eyvan-cell-a"
    secret_path = "/api/v1/namespaces/eyvan/secrets/eyvan-token-cell-a"
    service_path = "/api/v1/namespaces/eyvan/services/eyvan-cell-a"
    api.deleted.clear()
    api.delete_preconditions.clear()
    api.deletion_pending.add(job_path)

    with pytest.raises(CapacityProviderError, match="foreground deletion") as raised:
        driver.release("cell-a", _context(key="release-pending"))

    assert raised.value.retryable is True
    assert api.deleted == [job_path]
    assert secret_path in api.objects
    assert service_path in api.objects
    assert driver.state.get("cell-a") is not None


def test_release_tombstones_the_record(tmp_path) -> None:
    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())

    observation = driver.release("cell-a", _context())

    assert observation.state is CapacityStatus.RELEASED
    assert driver.observe("cell-a").state is CapacityStatus.ABSENT


def test_drain_does_not_delete_the_job(tmp_path) -> None:
    """Leased workloads keep their pods; that is what distinguishes drain from release."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())

    driver.drain("cell-a", _context())

    assert api.deleted == []
    assert api.jobs


def test_an_unsupported_capacity_class_is_refused(tmp_path) -> None:
    api = _FakeApi()

    with pytest.raises(ValueError, match="capacity class"):
        _driver(tmp_path, api).apply_capacity("cell-a", _desired(capacity_class="preemptible-ish"), _context())


def test_an_oversized_worker_count_is_refused(tmp_path) -> None:
    api = _FakeApi()
    desired = _desired(provider_options={"worker_count": MAX_WORKERS + 1})

    with pytest.raises(ValueError, match="at most"):
        _driver(tmp_path, api).apply_capacity("cell-a", desired, _context())


def test_the_k8s_alias_is_accepted_as_the_same_backend(tmp_path) -> None:
    api = _FakeApi()

    observation = _driver(tmp_path, api).apply_capacity("cell-a", _desired(capacity_backend="k8s"), _context())

    assert observation.state is CapacityStatus.HEALTHY


def test_a_provider_error_carries_its_retryability() -> None:
    """An unreachable API server is the ordinary state of a cluster during upgrade."""

    from eyvan.capacity.drivers.base import CapacityProviderFailureCode

    unavailable = CapacityProviderError("down", code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE)
    misconfigured = CapacityProviderError("no namespace", code=CapacityProviderFailureCode.CONFIGURATION)

    assert unavailable.retryable is True
    assert misconfigured.retryable is False


def test_job_status_defaults_are_not_silently_healthy() -> None:
    """A default-constructed status must never read as a ready cell."""

    assert capacity_state(JobStatus(exists=True))[0] is not CapacityStatus.HEALTHY


def test_a_create_carrying_a_queue_resolves_the_local_queue_before_judging_it(tmp_path) -> None:
    """The safety check read a LocalQueue name as a ClusterQueue name.

    ``job_manifest`` writes ``queue_name`` to ``kueue.x-k8s.io/queue-name``,
    which is the only queue name a Job may carry -- a LocalQueue. Reading it as a
    ClusterQueue meant that unless a deployment happened to give both objects one
    name, the GET 404'd and every create carrying a queue was refused: the check
    that exists to keep Kueue from evicting Eyvan's work made the supported
    configuration unusable.
    """

    class _QueueReadingApi(_FakeApi):
        """Answers the ClusterQueue safety GET, recording the path it was asked for."""

        def __init__(self) -> None:
            super().__init__()
            self.requested: list[str] = []

        def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
            del kwargs
            self.requested.append(path)
            return httpx.Response(
                200,
                json={"spec": {"preemption": {"reclaimWithinCohort": "Never", "withinClusterQueue": "Never"}}},
                request=httpx.Request(method, f"https://cluster{path}"),
            )

    api = _QueueReadingApi()
    api.publish_queue(namespace="eyvan", local_queue="research", cluster_queue="research-cq")
    driver = _driver(tmp_path, api)

    driver.apply_capacity(
        "cell-a",
        _desired(provider_options={"worker_count": 2, "resources": {"google.com/tpu": "4"}, "queue_name": "research"}),
        _context(),
    )

    assert f"{KUEUE_API}/namespaces/eyvan/localqueues/research" in api.observed
    assert api.requested == [f"{KUEUE_API}/clusterqueues/research-cq"], (
        "the safety check must read the ClusterQueue the LocalQueue routes into"
    )


def test_a_local_queue_that_cannot_be_resolved_refuses_the_create(tmp_path) -> None:
    """Nothing established means nothing may be assumed.

    An unresolvable LocalQueue is the same class of not-knowing as an unreadable
    ClusterQueue: the guarantee that Kueue cannot evict Eyvan's work has not been
    checked, so the cell is not created.
    """

    api = _FakeApi()
    driver = _driver(tmp_path, api)

    with pytest.raises(CapacityProviderError, match="could not resolve LocalQueue"):
        driver.apply_capacity(
            "cell-a",
            _desired(
                provider_options={"worker_count": 2, "resources": {"google.com/tpu": "4"}, "queue_name": "missing"}
            ),
            _context(),
        )


def test_drain_refuses_a_superseded_fence(tmp_path) -> None:
    """Draining takes capacity out of service, so an old controller cannot."""

    api = _FakeApi()
    driver = _driver(tmp_path, api)
    driver.apply_capacity("cell-a", _desired(), _context())

    with pytest.raises(StaleMutationError):
        driver.drain("cell-a", _context(generation=0))

    record = driver.observe("cell-a")
    assert record.state is not CapacityStatus.DRAINING
