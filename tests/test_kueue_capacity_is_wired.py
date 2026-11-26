# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The cluster capacity signal, wired to the cell observation that shows it.

W3 shipped ``observe_cluster_capacity``, ``explain_cluster_capacity`` and
``cell_capacity_state`` fully tested, and nothing called them. A queued Eyvan
cell therefore said "job is suspended pending kueue admission" -- Eyvan's own
state, restated -- while the cause sat one GET away.

Everything here holds one line, which is the same line W3 was organised around
and which matters more now that a *reconciliation* depends on it:

    **Any failure anywhere in the chain returns the detail exactly as it was.**

A diagnostic that can break an observation is worse than no diagnostic. So the
first test is not the feature; it is the feature failing and costing nothing.

The second property is cost. One reconcile pass observes every cell in a pool
and they all share one queue, so the resolution and the reading are cached for a
few seconds. Uncached, a forty-cell pool would issue eighty describing GETs per
pass against an API server Eyvan does not own -- which is how an operator
decides a diagnostic is not worth having.
"""

from __future__ import annotations

import time
import typing as tp

import httpx
from test_kubernetes_capacity import _context, _desired, _driver, _FakeApi

from eyvan._internal.json_types import JsonObject
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.kubernetes import CLUSTER_CAPACITY_DETAIL_ENV, KubernetesCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.capacity.providers.kubernetes.api import KUEUE_API, ApiObservation
from eyvan.capacity.providers.kubernetes.workers import indexed_job_name

_TODAYS_WAITING_DETAIL = "job is suspended pending kueue admission"
_TPU = "google.com/tpu"
_WORKER_IMAGE = "registry.test/worker@sha256:" + "a" * 64


class _KueueApi(_FakeApi):
    """A fake that also answers the create-time ClusterQueue safety read."""

    def __init__(self) -> None:
        super().__init__()
        self.queue_reads: list[str] = []

    def request(self, method: str, path: str, **_kwargs: object) -> httpx.Response:
        self.queue_reads.append(path)
        return httpx.Response(
            200, json={"spec": {"preemption": {"withinClusterQueue": "Never", "reclaimWithinCohort": "Never"}}}
        )


def _cluster_queue(
    *,
    nominal: str = "32",
    reserved: str | None = "8",
    flavor: str = "tpu-v5p",
    cohort: str = "",
    pending: int = 0,
    admitted: int = 0,
    resource: str = _TPU,
) -> JsonObject:
    """Build a ClusterQueue shaped exactly as Kueue v1beta1 publishes one."""

    spec: JsonObject = {
        "resourceGroups": [
            {
                "coveredResources": [resource],
                "flavors": [{"name": flavor, "resources": [{"name": resource, "nominalQuota": nominal}]}],
            }
        ]
    }
    if cohort:
        spec["cohort"] = cohort
    status: JsonObject = {"pendingWorkloads": pending, "admittedWorkloads": admitted}
    if reserved is not None:
        status["flavorsReservation"] = [{"name": flavor, "resources": [{"name": resource, "total": reserved}]}]
    return {"apiVersion": "kueue.x-k8s.io/v1beta1", "kind": "ClusterQueue", "spec": spec, "status": status}


def _queued_desired(**options: object) -> dict[str, tp.Any]:
    """Return a cell template that Kueue admits, which is what suspends its Job."""

    desired = dict(_desired())
    merged = dict(desired["provider_options"])
    merged["queue_name"] = "research"
    merged.update(options)
    desired["provider_options"] = merged
    return desired


def _queued_cell(tmp_path, api: _FakeApi, cell_id: str = "cell-a", **options: object) -> KubernetesCapacityDriver:
    """Create one cell whose Job is suspended behind Kueue admission.

    The create-time safety check resolves the LocalQueue to its ClusterQueue and
    refuses when it cannot, so the cell is created against a queue that resolves
    and the test then breaks the reading it wants to exercise. That is also the
    real sequence: a cell outlives the RBAC grant or the object that let it be
    created, and the observation has to degrade rather than the cell vanish.
    """

    driver = _driver(tmp_path, api)
    observations = dict(api.observations)
    api.publish_queue(namespace="eyvan", local_queue="research", cluster_queue="research-cq")
    driver.apply_capacity(cell_id, _queued_desired(**options), _context(key=f"apply-{cell_id}"))
    api.observations = observations
    # Every test below counts what *observing* reads. The create-time safety
    # check resolves the same LocalQueue, and leaving its read in the record
    # would make the observation look like it had asked for something it did not.
    api.observed.clear()
    driver._local_queue_cache.clear()
    driver._reading_cache.clear()
    return driver


def _unsuspend(api: _FakeApi, cell_id: str = "cell-a") -> None:
    """Admit the Job, which is what makes the cell stop being WAITING."""

    job = api.objects[f"/apis/batch/v1/namespaces/eyvan/jobs/{indexed_job_name(cell_id)}"]
    spec = job["spec"]
    assert isinstance(spec, dict)
    spec["suspend"] = False


# --------------------------------------------------------------- failing costs nothing


def _create_cells(api: _FakeApi, cold: KubernetesCapacityDriver, count: int) -> None:
    """Create N queued cells against a queue that resolves, then forget the reads.

    The create-time safety check resolves the LocalQueue to the ClusterQueue it
    routes into and refuses when it cannot, so a cell cannot be brought into
    existence against an unresolvable queue at all. Every test here is about what
    *observing* an existing cell costs and says, so the cell is created against a
    queue that resolves and the reading the test wants is armed afterwards --
    which is also the real sequence: a cell outlives the grant or the object that
    let it be created.
    """

    observations = dict(api.observations)
    api.publish_queue(namespace="eyvan", local_queue="research", cluster_queue="research-cq")
    for index in range(count):
        cold.apply_capacity(f"cell-{index}", _queued_desired(), _context(key=f"apply-{index}"))
    api.observations = observations
    api.observed.clear()
    cold._local_queue_cache.clear()
    cold._reading_cache.clear()


def test_an_unreadable_local_queue_leaves_todays_detail_byte_identical(tmp_path) -> None:
    """The governing rule, stated as the first test rather than the last.

    A ServiceAccount without ``get localqueues`` is the ordinary state of a
    cluster where Eyvan was installed before Kueue. It must cost the observation
    nothing at all.
    """

    api = _KueueApi()
    api.observations[f"{KUEUE_API}/namespaces/eyvan/localqueues/research"] = ApiObservation(
        payload=None, readable=False, reason="HTTP 403: localqueues is forbidden"
    )
    driver = _queued_cell(tmp_path, api)

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.WAITING
    assert observation.detail == _TODAYS_WAITING_DETAIL


def test_a_local_queue_that_names_no_cluster_queue_leaves_the_detail_alone(tmp_path) -> None:
    """Falling back to the LocalQueue's own name would read a different object."""

    api = _KueueApi()
    api.observations[f"{KUEUE_API}/namespaces/eyvan/localqueues/research"] = ApiObservation(
        payload={"spec": {}}, readable=True
    )
    driver = _queued_cell(tmp_path, api)

    observation = driver.observe("cell-a")

    assert observation.detail == _TODAYS_WAITING_DETAIL
    assert not [path for path in api.observed if "clusterqueues" in path]


def test_an_unreadable_cluster_queue_leaves_the_detail_alone(tmp_path) -> None:
    """UNKNOWN is never full: an RBAC refusal must not read as no room."""

    api = _KueueApi()
    api.publish_queue()
    api.observations[f"{KUEUE_API}/clusterqueues/research-cq"] = ApiObservation(
        payload=None, readable=False, reason="HTTP 403: clusterqueues is forbidden"
    )
    driver = _queued_cell(tmp_path, api)

    assert driver.observe("cell-a").detail == _TODAYS_WAITING_DETAIL


def test_a_cluster_queue_with_no_published_reservation_leaves_the_detail_alone(tmp_path) -> None:
    """The moment Eyvan knows least about the cluster is not the moment to claim.

    Reading a missing ``flavorsReservation`` as "nothing is reserved" would
    report a saturated cluster as entirely free, which here would append a
    confident sentence saying there is room.
    """

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(reserved=None))
    driver = _queued_cell(tmp_path, api)

    assert driver.observe("cell-a").detail == _TODAYS_WAITING_DETAIL


def test_a_cell_declaring_no_readable_request_leaves_the_detail_alone(tmp_path) -> None:
    """No shape to fit is unknown capacity, not a request for nothing."""

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32"))
    driver = _queued_cell(tmp_path, api, resources={})

    assert driver.observe("cell-a").detail == _TODAYS_WAITING_DETAIL


def test_an_api_that_raises_leaves_the_detail_alone_rather_than_failing_the_observation(tmp_path) -> None:
    """The last line of defence, because the alternative is a lost reconciliation.

    Every read in the chain is contractually never-raises. If one ever breaks
    that contract the correct answer is still the cell's real state with today's
    wording -- not an exception escaping into the capacity reconciler.
    """

    class _BrokenApi(_KueueApi):
        """Breaks the never-raises contract, but only once the cell exists."""

        def __init__(self) -> None:
            super().__init__()
            self.broken = False

        def observe(self, path: str, *, params: tp.Mapping[str, str] | None = None) -> ApiObservation:
            if self.broken:
                raise RuntimeError("the describing read broke its own contract")
            return super().observe(path, params=params)

    api = _BrokenApi()
    driver = _queued_cell(tmp_path, api)
    api.broken = True

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.WAITING
    assert observation.detail == _TODAYS_WAITING_DETAIL


# --------------------------------------------------------------- naming the cause


def test_a_waiting_cell_behind_a_proven_full_cluster_explains_itself_with_numbers(tmp_path) -> None:
    """The whole point of W3 for a human, now reachable from a cell observation."""

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32", admitted=8, pending=3))
    driver = _queued_cell(tmp_path, api)

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.WAITING
    assert observation.detail.startswith(_TODAYS_WAITING_DETAIL)
    assert "ClusterQueue 'research-cq' admits no further cell of this shape" in observation.detail
    assert f"{_TPU} free 0" in observation.detail
    assert "3 workload(s) already pending" in observation.detail
    assert "raise nominalQuota" in observation.detail
    assert "wait for 8 admitted workload(s) to finish" in observation.detail


def test_a_waiting_cell_with_room_in_the_cluster_keeps_todays_wording(tmp_path) -> None:
    """Room is not a cause. A cell queued behind admission with capacity free is
    waiting on Kueue's own scheduling, and saying so would be noise."""

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="0"))
    driver = _queued_cell(tmp_path, api)

    assert driver.observe("cell-a").detail == _TODAYS_WAITING_DETAIL


def test_a_lower_bound_inside_a_cohort_never_becomes_a_shortage(tmp_path) -> None:
    """Borrowing depends on queues Eyvan cannot see, so free capacity is a lower
    bound: enough to prove a cell fits, never enough to prove one does not."""

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32", cohort="shared"))
    driver = _queued_cell(tmp_path, api)

    assert driver.observe("cell-a").detail == _TODAYS_WAITING_DETAIL


def test_the_explained_detail_is_bounded(tmp_path) -> None:
    """It lands in a lifecycle reason, a log line, and a dashboard panel."""

    resource = "example.com/" + "a" * 240
    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="4", reserved="4", resource=resource))
    driver = _queued_cell(tmp_path, api, resources={resource: "4"})

    detail = driver.observe("cell-a").detail

    assert detail.startswith(_TODAYS_WAITING_DETAIL)
    assert len(detail) <= 512


# --------------------------------------------------------------- what it costs


def test_a_cell_that_is_not_waiting_issues_no_capacity_read_at_all(tmp_path) -> None:
    """WAITING is the one state whose detail restates Eyvan's own state.

    Every other state already names something an operator can act on, so paying
    describing GETs to qualify a healthy cell buys nothing.
    """

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32"))
    driver = _queued_cell(tmp_path, api)
    _unsuspend(api)
    api.observed.clear()

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.HEALTHY
    assert api.observed == []


def test_every_cell_in_one_pool_shares_one_reading(tmp_path) -> None:
    """One reconcile pass observes every cell in a pool and they share one queue.

    Uncached this is two describing GETs per cell; the whole pass must cost the
    LocalQueue resolution once and the ClusterQueue reading once.
    """

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32"))
    cold = KubernetesCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        api=api,
        namespace="eyvan",
        image=_WORKER_IMAGE,
        shared_secret="agent-secret",
        cluster_capacity_detail=False,
    )
    _create_cells(api, cold, 4)
    assert api.observed == []

    driver = _driver(tmp_path, api)
    details = [driver.observe(f"cell-{index}").detail for index in range(4)]

    assert api.observed == [
        f"{KUEUE_API}/namespaces/eyvan/localqueues/research",
        f"{KUEUE_API}/clusterqueues/research-cq",
    ]
    assert all("admits no further cell of this shape" in detail for detail in details)


def test_the_shared_reading_expires_so_a_cluster_that_freed_up_is_seen(tmp_path) -> None:
    """A cache that never expired would explain today's cell with last hour's cluster."""

    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32"))
    driver = _driver(tmp_path, api)
    driver.cluster_capacity_ttl_s = 0.05
    driver.apply_capacity("cell-a", _queued_desired(), _context())
    assert "admits no further cell" in driver.observe("cell-a").detail

    api.observations[f"{KUEUE_API}/clusterqueues/research-cq"] = ApiObservation(
        payload=_cluster_queue(nominal="32", reserved="0"), readable=True
    )
    assert "admits no further cell" in driver.observe("cell-a").detail
    time.sleep(0.06)

    assert driver.observe("cell-a").detail == _TODAYS_WAITING_DETAIL


def test_an_unresolvable_queue_is_remembered_for_the_pass_rather_than_re_asked_per_cell(tmp_path) -> None:
    """A refusal answers identically for every cell, and re-asking spends exactly
    the cost the cache exists to avoid on the answer that establishes nothing."""

    api = _KueueApi()
    api.observations[f"{KUEUE_API}/namespaces/eyvan/localqueues/research"] = ApiObservation(
        payload=None, readable=False, reason="HTTP 403: forbidden"
    )
    cold = KubernetesCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        api=api,
        namespace="eyvan",
        image=_WORKER_IMAGE,
        shared_secret="agent-secret",
        cluster_capacity_detail=False,
    )
    _create_cells(api, cold, 3)

    driver = _driver(tmp_path, api)
    details = [driver.observe(f"cell-{index}").detail for index in range(3)]

    assert api.observed == [f"{KUEUE_API}/namespaces/eyvan/localqueues/research"]
    assert details == [_TODAYS_WAITING_DETAIL] * 3


# --------------------------------------------------------------- the way back


def test_the_diagnostic_can_be_switched_off_to_exactly_todays_bytes(tmp_path, monkeypatch) -> None:
    """An operator who wants the shipped wording back must not need a rebuild.

    Off is not "read and discard": no describing GET is issued at all.
    """

    monkeypatch.setenv(CLUSTER_CAPACITY_DETAIL_ENV, "false")
    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32"))
    driver = _queued_cell(tmp_path, api)

    observation = driver.observe("cell-a")

    assert driver.cluster_capacity_detail is False
    assert observation.detail == _TODAYS_WAITING_DETAIL
    assert api.observed == []


def test_the_diagnostic_is_on_when_the_environment_says_nothing(tmp_path, monkeypatch) -> None:
    """A cause an operator can act on is worth having by default."""

    monkeypatch.delenv(CLUSTER_CAPACITY_DETAIL_ENV, raising=False)
    api = _KueueApi()
    api.publish_queue(payload=_cluster_queue(nominal="32", reserved="32"))
    driver = _queued_cell(tmp_path, api)

    assert driver.cluster_capacity_detail is True
    assert "admits no further cell of this shape" in driver.observe("cell-a").detail
