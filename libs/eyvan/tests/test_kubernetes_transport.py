# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Kubernetes transport itself, which nothing had ever instantiated.

The driver is real and heavily tested -- 456 lines, forty-seven scenarios -- and
every one of them substitutes a fake API object. So the layer that actually
talks to a cluster (bearer token re-read per call, CA verification, status-code
interpretation, 409-as-replay, foreground deletion) had never been exercised by
anything, including by the tests that most look like they cover it.

``httpx.MockTransport`` closes that without a cluster. It cannot prove what a
real API server does -- that is the GKE run, and this does not claim to replace
it -- but it does prove that the request Eyvan builds is the request it meant to
build, and that each status code produces the classification the driver acts on.

The status codes are the part worth the most here. Every one of them routes to a
``CapacityProviderFailureCode`` that decides whether capacity is retried or
discarded, and getting that wrong is invisible until a cluster upgrade turns a
transient 503 into dropped capacity the control plane still believes it asked
for.
"""

from __future__ import annotations

import httpx
import pytest

from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode
from eyvan.capacity.providers.kubernetes.api import KubernetesApi, local_queue_cluster_queue, read_token


def _api(handler, tmp_path, *, token: str = "sa-token-value") -> KubernetesApi:
    """Build the real transport over a mock, with a real token file on disk."""

    token_path = tmp_path / "token"
    token_path.write_text(token, encoding="utf-8")
    return KubernetesApi(
        "https://kubernetes.default.svc",
        token_path=str(token_path),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://kubernetes.default.svc"),
    )


# --------------------------------------------------------------------------- the request


def test_every_request_carries_the_service_account_token(tmp_path) -> None:
    """Read per call, not cached at construction.

    A projected service-account token is rotated by the kubelet while the pod
    runs. An API object that read it once would authenticate for an hour and
    then start failing with a 401 that looks like an RBAC problem.
    """

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    api = _api(handler, tmp_path)
    api.request("GET", "/apis/batch/v1/namespaces/eyvan/jobs")
    (tmp_path / "token").write_text("rotated-token", encoding="utf-8")
    api.request("GET", "/apis/batch/v1/namespaces/eyvan/jobs")

    assert seen[0].headers["Authorization"] == "Bearer sa-token-value"
    assert seen[1].headers["Authorization"] == "Bearer rotated-token"


def test_the_request_is_built_against_the_configured_api_server(tmp_path) -> None:
    """A path joined wrong reaches a different cluster, or none."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    _api(handler, tmp_path).request("GET", "/apis/batch/v1/namespaces/eyvan/jobs/cell-a")

    assert str(seen[0].url) == "https://kubernetes.default.svc/apis/batch/v1/namespaces/eyvan/jobs/cell-a"


def test_an_unreadable_token_is_a_configuration_failure_not_a_transient_one(tmp_path) -> None:
    """A pod without a mounted token will never grow one.

    Classifying it retryable would make the driver wait forever on something no
    amount of waiting fixes.
    """

    with pytest.raises(CapacityProviderError) as raised:
        read_token(str(tmp_path / "absent"))

    assert raised.value.code is CapacityProviderFailureCode.CONFIGURATION
    assert raised.value.retryable is False


def test_a_transport_failure_is_retryable(tmp_path) -> None:
    """An API server briefly unreachable is the ordinary state of an upgrading cluster.

    Discarding the command would drop capacity the control plane believes it
    asked for -- and it would never ask again, because it thinks it did.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(CapacityProviderError) as raised:
        _api(handler, tmp_path).request("GET", "/apis/batch/v1/namespaces/eyvan/jobs")

    assert raised.value.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
    assert raised.value.retryable is True


# --------------------------------------------------------------------------- status codes


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (500, CapacityProviderFailureCode.PROVIDER_UNAVAILABLE, True),
        (503, CapacityProviderFailureCode.PROVIDER_UNAVAILABLE, True),
        (429, CapacityProviderFailureCode.PROVIDER_UNAVAILABLE, True),
        (401, CapacityProviderFailureCode.CONFIGURATION, False),
        (403, CapacityProviderFailureCode.CONFIGURATION, False),
        (422, CapacityProviderFailureCode.CONFIGURATION, False),
    ],
)
def test_each_status_routes_to_the_classification_the_driver_acts_on(
    tmp_path, status: int, code: CapacityProviderFailureCode, retryable: bool
) -> None:
    """The split that decides whether capacity is retried or discarded.

    A 429 is the API server asking for patience; a 403 is RBAC that will refuse
    identically forever. Treating either as the other is invisible until it
    matters.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": "denied"})

    with pytest.raises(CapacityProviderError) as raised:
        _api(handler, tmp_path).create("/apis/batch/v1/namespaces/eyvan/jobs", {"kind": "Job"})

    assert raised.value.code is code
    assert raised.value.retryable is retryable


def test_an_error_body_is_truncated_rather_than_pasted_whole(tmp_path) -> None:
    """An API server error body can be a full manifest with its events attached.

    A diagnostic that carries all of it lands in a lease record, an audit row and
    a log line.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="x" * 5_000)

    with pytest.raises(CapacityProviderError) as raised:
        _api(handler, tmp_path).create("/apis/batch/v1/namespaces/eyvan/jobs", {"kind": "Job"})

    assert len(str(raised.value)) < 1_000


# --------------------------------------------------------------------------- semantics


def test_a_409_on_create_is_a_replay_and_not_a_failure(tmp_path) -> None:
    """Names are deterministic, so an existing object is this cell's own create.

    Which is exactly what an at-least-once outbox produces, and treating it as an
    error would fail every redelivery.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"reason": "AlreadyExists"})

    assert _api(handler, tmp_path).create("/apis/batch/v1/namespaces/eyvan/jobs", {"kind": "Job"}) is False


def test_a_201_on_create_reports_that_it_was_new(tmp_path) -> None:
    """The caller distinguishes "I made this" from "it was already here"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"metadata": {"name": "cell-a"}})

    assert _api(handler, tmp_path).create("/apis/batch/v1/namespaces/eyvan/jobs", {"kind": "Job"}) is True


def test_a_missing_job_reads_as_absent_rather_than_as_an_error(tmp_path) -> None:
    """A 404 is the answer to "does this exist", not a failure to ask."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"reason": "NotFound"})

    assert _api(handler, tmp_path).get_job("eyvan", "cell-a") is None


def test_a_job_is_returned_decoded(tmp_path) -> None:
    """The happy path, which nothing had executed against a real response object."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"metadata": {"name": "cell-a"}, "status": {"ready": 2}})

    job = _api(handler, tmp_path).get_job("eyvan", "cell-a")

    assert job is not None
    assert job["metadata"]["name"] == "cell-a"


def test_delete_asks_for_foreground_propagation(tmp_path) -> None:
    """A background delete returns while pods are still running.

    The control plane would believe the capacity was freed and place onto a node
    that is still occupied by the workload it just released.
    """

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    _api(handler, tmp_path).delete("/apis/batch/v1/namespaces/eyvan/jobs/cell-a")

    assert seen[0].method == "DELETE"
    assert b"Foreground" in seen[0].content


def test_deleting_something_already_gone_is_not_an_error(tmp_path) -> None:
    """Release is retried, and the second attempt finds nothing.

    Raising there would make a successful release look like a stuck one forever.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"reason": "NotFound"})

    _api(handler, tmp_path).delete("/apis/batch/v1/namespaces/eyvan/jobs/cell-a")


def test_a_success_with_an_empty_body_decodes_rather_than_raising(tmp_path) -> None:
    """Some API server responses carry no JSON at all.

    A decoder that insisted would turn a successful delete into a provider
    error, and the driver would retry a deletion that already happened.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    _api(handler, tmp_path).delete("/apis/batch/v1/namespaces/eyvan/jobs/cell-a")


# --------------------------------------------------------------------------- the LocalQueue


def test_a_local_queue_is_read_from_its_own_namespaced_path(tmp_path) -> None:
    """A LocalQueue is namespaced and a ClusterQueue is not.

    The Job carries only the LocalQueue name, so this is the read that turns the
    one name a queued cell has into the one name the quota is published under.
    """

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"spec": {"clusterQueue": "research-cq"}})

    observation = _api(handler, tmp_path).observe_local_queue("eyvan", "research")

    assert seen == ["/apis/kueue.x-k8s.io/v1beta1/namespaces/eyvan/localqueues/research"]
    assert local_queue_cluster_queue(observation) == "research-cq"


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_an_unreadable_local_queue_establishes_nothing_and_does_not_raise(tmp_path, status: int) -> None:
    """Same never-raises contract as every other describing read.

    A ServiceAccount without ``get localqueues`` is the ordinary state of a
    cluster where Eyvan was installed before Kueue, and it must degrade the
    capacity signal rather than fail an unrelated reconciliation.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": "denied"})

    observation = _api(handler, tmp_path).observe_local_queue("eyvan", "research")

    assert observation.unreadable is True
    assert str(status) in observation.reason
    assert local_queue_cluster_queue(observation) is None


def test_a_local_queue_that_is_not_there_is_readable_and_still_names_nothing(tmp_path) -> None:
    """A 404 answers "does this exist"; it does not answer "is the cluster full"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"reason": "NotFound"})

    observation = _api(handler, tmp_path).observe_local_queue("eyvan", "research")

    assert observation.readable is True
    assert local_queue_cluster_queue(observation) is None


def test_a_severed_connection_reading_a_local_queue_is_unreadable_not_an_exception(tmp_path) -> None:
    """A transport failure here must not escape into a cell reconciliation."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    observation = _api(handler, tmp_path).observe_local_queue("eyvan", "research")

    assert observation.unreadable is True
    assert local_queue_cluster_queue(observation) is None
