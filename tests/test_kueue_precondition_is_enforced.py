# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a ClusterQueue that could evict Eyvan's work.

Kueue eviction destroys a pod with no checkpoint hook, bypassing the entire
``PreemptionPlan`` / ``CHECKPOINT_AND_STOP`` / ``grace_period_seconds``
protocol. The only trace is a training job that lost hours with no ledger entry
explaining why -- which is exactly why the hazard has to be refused rather than
observed after the fact.

``inspect_cluster_queue`` and ``foreign_workloads`` were written, tested, and
called from nowhere in production. Two docstrings meanwhile asserted that
``eyvan doctor`` "refuses a configuration that could evict Eyvan's work"; doctor
warns, and cannot read the cluster at all -- it runs on an operator's machine.

The check now runs in the driver, before the Job is created, because that is
where cluster access exists: the driver runs in-cluster with a ServiceAccount.
"""

from __future__ import annotations

import httpx
import pytest
from test_kubernetes_capacity import _context, _desired, _driver, _FakeApi

from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode


class _QueueApi(_FakeApi):
    """A fake that answers ClusterQueue reads, behind a resolvable LocalQueue.

    The pool's ``queue_name`` is a LocalQueue -- it is what a Job may carry --
    so the driver resolves it to the ClusterQueue it routes into before judging
    that queue's preemption. A fake that answered ClusterQueue reads directly
    under the LocalQueue's name would test a lookup Kueue does not have.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        payload: object | None = None,
        local_queue_resolves: bool = True,
    ) -> None:
        super().__init__()
        self._status = status
        self._payload = payload
        self.queue_reads: list[str] = []
        if local_queue_resolves:
            self.publish_queue(namespace="eyvan", local_queue="research", cluster_queue="research-cq")

    def request(self, method: str, path: str, **_kwargs: object) -> httpx.Response:
        self.queue_reads.append(path)
        return httpx.Response(self._status, json=self._payload, request=httpx.Request(method, f"https://k8s{path}"))


def _queue_desired() -> dict:
    desired = dict(_desired())
    options = dict(desired.get("provider_options") or {})
    options["queue_name"] = "research"
    desired["provider_options"] = options
    return desired


def _safe_queue() -> dict:
    return {"spec": {"preemption": {"withinClusterQueue": "Never", "reclaimWithinCohort": "Never"}}}


def test_a_pool_without_a_queue_never_reads_the_cluster(tmp_path) -> None:
    """Kueue is opt-in; a pool that does not use it must not require it."""

    api = _QueueApi(payload=_safe_queue())

    _driver(tmp_path, api).apply_capacity("cell-a", _desired(), _context())

    assert api.queue_reads == []


def test_a_safe_queue_is_accepted_and_the_job_is_created(tmp_path) -> None:
    """The precondition must not block the configuration it exists to permit."""

    api = _QueueApi(payload=_safe_queue())

    _driver(tmp_path, api).apply_capacity("cell-a", _queue_desired(), _context())

    assert any("clusterqueues/research-cq" in path for path in api.queue_reads)
    assert any(path.endswith("/jobs") for path, _manifest in api.created)


@pytest.mark.parametrize(
    "preemption",
    [
        {"withinClusterQueue": "LowerPriority", "reclaimWithinCohort": "Never"},
        {"withinClusterQueue": "Never", "reclaimWithinCohort": "Any"},
        {"withinClusterQueue": "LowerPriority", "reclaimWithinCohort": "Any"},
    ],
    ids=["within-queue", "cross-cohort", "both"],
)
def test_an_evicting_queue_is_refused_before_any_job_exists(tmp_path, preemption: dict) -> None:
    """Refusal must precede creation, or the hazard is already live.

    A Job admitted into an evicting queue can be destroyed before the refusal
    would ever be noticed.
    """

    api = _QueueApi(payload={"spec": {"preemption": preemption}})

    with pytest.raises(CapacityProviderError) as raised:
        _driver(tmp_path, api).apply_capacity("cell-a", _queue_desired(), _context())

    assert raised.value.code is CapacityProviderFailureCode.CONFIGURATION
    assert not [path for path, _manifest in api.created if path.endswith("/jobs")]


def test_a_missing_queue_is_refused(tmp_path) -> None:
    """A pool pointed at a queue that does not exist suspends every Job forever.

    Reporting "no problems" for that would be actively misleading, so absence is
    a finding rather than an empty result.
    """

    api = _QueueApi(status=404, payload=None)

    with pytest.raises(CapacityProviderError, match="does not exist"):
        _driver(tmp_path, api).apply_capacity("cell-a", _queue_desired(), _context())


def test_an_unreadable_queue_is_not_assumed_safe(tmp_path) -> None:
    """The one case where optimism silently discards the guarantee.

    If Eyvan cannot read the ClusterQueue it does not know whether preemption is
    disabled, and proceeding would mean claiming a safety property it never
    verified.
    """

    api = _QueueApi(status=403, payload={"message": "forbidden"})

    with pytest.raises(CapacityProviderError, match="could not read ClusterQueue"):
        _driver(tmp_path, api).apply_capacity("cell-a", _queue_desired(), _context())


def test_the_refusal_is_not_retryable(tmp_path) -> None:
    """An unsafe queue fails identically forever; retrying only hides it.

    CONFIGURATION is the code that says a human must change something, which is
    what stops the command churning in the outbox until someone reads a log.
    """

    api = _QueueApi(payload={"spec": {"preemption": {"withinClusterQueue": "LowerPriority"}}})

    with pytest.raises(CapacityProviderError) as raised:
        _driver(tmp_path, api).apply_capacity("cell-a", _queue_desired(), _context())

    assert raised.value.retryable is False
