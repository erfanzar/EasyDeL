# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The autoscale pass that takes a cluster reading, and the pool it takes it for.

``advise_launch`` was complete and had no caller: the autoscaler sized every
Kubernetes pool from demand alone and launched into whatever the cluster had
left, which on a shared cluster means launching cells that sit WAITING until
``queue_timeout_s`` reclaims them, repeatedly.

Three properties are load-bearing and each is a way this goes wrong quietly:

* **Off means no request.** Not "reads and ignores". A control plane that
  upgrades into this code issues exactly the traffic it issued before.
* **The reading is per *cell*, not per pod.** A cell is an Indexed Job with
  ``parallelism == worker_count``, so passing the pod request would count pods
  as cells and let a pool launch several times what fits.
* **UNKNOWN returns the caller's own number.** Anything else makes an
  unreadable cluster into a fleet-sizing decision.
"""

from __future__ import annotations

import asyncio
import typing as tp
from datetime import datetime, timedelta

import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject
from eyvan.capacity.autoscale import ClusterCapacityPolicy
from eyvan.capacity.autoscale.pool_capacity import (
    CLUSTER_CAPACITY_MAX_AGE_ENV,
    CLUSTER_CAPACITY_PLANNING_ENV,
    DEFAULT_NAMESPACE,
    KUBERNETES_NAMESPACE_ENV,
    ClusterCapacityPass,
    cluster_capacity_policy_from_env,
    kubernetes_capacity_target,
)
from eyvan.capacity.drivers.kubernetes import KubernetesCapacityDriver
from eyvan.capacity.providers.kubernetes.api import KUEUE_API, ApiObservation
from eyvan.records import CapacityPoolRecord, Principal, Role, RoleGrant
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService

_NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_TPU = "google.com/tpu"
_ON = ClusterCapacityPolicy(enabled=True)


def _cluster_queue(*, nominal: str = "32", reserved: str | None = "0", cohort: str = "") -> JsonObject:
    spec: JsonObject = {
        "resourceGroups": [
            {
                "coveredResources": [_TPU],
                "flavors": [{"name": "tpu-v5p", "resources": [{"name": _TPU, "nominalQuota": nominal}]}],
            }
        ]
    }
    if cohort:
        spec["cohort"] = cohort
    status: JsonObject = {"admittedWorkloads": 0, "pendingWorkloads": 0}
    if reserved is not None:
        status["flavorsReservation"] = [{"name": "tpu-v5p", "resources": [{"name": _TPU, "total": reserved}]}]
    return {"spec": spec, "status": status}


class _ClusterApi:
    """Answer describing reads and count every one of them."""

    def __init__(self, *, cluster_queue: JsonObject | None = None, local_queue: str = "research-cq") -> None:
        self.observed: list[str] = []
        self.observations: dict[str, ApiObservation] = {}
        if local_queue:
            self.observations[f"{KUEUE_API}/namespaces/eyvan/localqueues/research"] = ApiObservation(
                payload={"spec": {"clusterQueue": local_queue}}, readable=True
            )
        if cluster_queue is not None:
            self.observations[f"{KUEUE_API}/clusterqueues/{local_queue}"] = ApiObservation(
                payload=cluster_queue, readable=True
            )

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


def _pool(
    pool_id: str = "research/k8s",
    *,
    queue_name: str | None = "research",
    resources: dict[str, object] | None = None,
    namespace: str | None = None,
    backend: str | None = "kubernetes",
    initial_cells: int | None = None,
    flavor: str | None = None,
) -> CapacityPoolRecord:
    options: dict[str, object] = {"worker_count": 1}
    if queue_name is not None:
        options["queue_name"] = queue_name
    if resources is not None:
        options["resources"] = resources
    elif resources is None:
        options["resources"] = {_TPU: "4"}
    if namespace is not None:
        options["namespace"] = namespace
    if flavor is not None:
        options["resource_flavor"] = flavor
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type=pool_id.rsplit("/", 1)[-1],
        provider="kubernetes",
        capacity_backend=backend,
        region="us-east5",
        capacity={"tpu": 4},
        allocatable={"tpu": 4},
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        provider_options=options,
        initial_cells=initial_cells,
        max_cells=8,
    )


def _pass(api: _ClusterApi | None, *, policy: ClusterCapacityPolicy | None = None) -> ClusterCapacityPass:
    return ClusterCapacityPass(policy=policy or _ON, api_factory=lambda: api)


# ------------------------------------------------------------ what is read at all


def test_a_kubernetes_pool_with_a_queue_and_a_request_is_a_target() -> None:
    target = kubernetes_capacity_target(_pool())

    assert target is not None
    assert (target.pool_id, target.namespace, target.local_queue) == ("research/k8s", DEFAULT_NAMESPACE, "research")
    assert target.request == ((_TPU, 4.0),)


@pytest.mark.parametrize(
    ("label", "pool"),
    [
        ("not-kubernetes", _pool(backend="onprem-ssh")),
        ("no-queue", _pool(queue_name=None)),
        ("no-resources", _pool(resources={})),
        ("unreadable-quantity", _pool(resources={_TPU: "four"})),
        ("only-zero-resources", _pool(resources={_TPU: "0"})),
    ],
)
def test_a_pool_with_nothing_to_read_produces_no_target(label: str, pool: CapacityPoolRecord) -> None:
    """Settled before a GET is spent on it. An unreadable declared quantity is
    unknown capacity, and reading around it would answer about a different shape."""

    assert kubernetes_capacity_target(pool) is None, label


def test_a_pool_may_name_the_namespace_its_queue_lives_in() -> None:
    target = kubernetes_capacity_target(_pool(namespace="team-research"))

    assert target is not None
    assert target.namespace == "team-research"


def test_the_default_namespace_is_the_one_the_driver_creates_jobs_in() -> None:
    """Two spellings of one namespace would send every read to a 404."""

    assert DEFAULT_NAMESPACE == KubernetesCapacityDriver().namespace


def test_a_cells_request_is_its_pod_request_times_its_pods() -> None:
    """A cell is an Indexed Job with ``parallelism == worker_count``.

    Passing the per-pod request would count pods as cells, and a pool of
    four-worker cells would launch four times what the queue can hold.
    """

    target = kubernetes_capacity_target(_pool())

    assert target is not None
    assert target.cell_request(1) == {_TPU: 4.0}
    assert target.cell_request(4) == {_TPU: 16.0}


# ------------------------------------------------------------ off means silent


def test_a_disabled_policy_issues_no_request_at_all() -> None:
    """Not "reads and ignores": an upgrade must be byte-identical on the wire."""

    api = _ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="32"))

    advice = _pass(api, policy=ClusterCapacityPolicy()).advise_launches([(_pool(), (1, 1, 1))], now=_NOW)

    assert advice == {}
    assert api.observed == []


def test_the_policy_is_disabled_when_the_environment_says_nothing() -> None:
    assert cluster_capacity_policy_from_env({}).enabled is False
    assert ClusterCapacityPass.from_env({}).enabled is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_an_operator_turns_it_on_by_name(value: str) -> None:
    assert cluster_capacity_policy_from_env({CLUSTER_CAPACITY_PLANNING_ENV: value}).enabled is True


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "maybe"])
def test_anything_that_is_not_an_enablement_leaves_it_off(value: str) -> None:
    """A typo reads as off, which is the direction that changes nothing."""

    assert cluster_capacity_policy_from_env({CLUSTER_CAPACITY_PLANNING_ENV: value}).enabled is False


@pytest.mark.parametrize("value", ["not-a-number", "0", "-5", "100000"])
def test_an_unusable_freshness_bound_falls_back_rather_than_failing_a_pass(value: str) -> None:
    """This is read inside a reconciliation; a typo must not stop a fleet scaling."""

    policy = cluster_capacity_policy_from_env({CLUSTER_CAPACITY_PLANNING_ENV: "1", CLUSTER_CAPACITY_MAX_AGE_ENV: value})

    assert policy.enabled is True
    assert policy.max_observation_age_s == 60.0


def test_the_pass_reads_the_namespace_the_agent_was_configured_with() -> None:
    assert ClusterCapacityPass.from_env({KUBERNETES_NAMESPACE_ENV: "team-a"}).default_namespace == "team-a"


# ------------------------------------------------------------ the decision moves


def test_the_decision_moves_with_real_capacity() -> None:
    """The whole point: what is launched follows what the cluster actually has."""

    roomy = _pass(_ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="0")))
    tight = _pass(_ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="24")))
    full = _pass(_ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="32")))
    planned = [(_pool(), (1, 1, 1, 1))]

    assert roomy.advise_launches(planned, now=_NOW)["research/k8s"].allowed_cells == 4
    assert tight.advise_launches(planned, now=_NOW)["research/k8s"].allowed_cells == 2
    assert full.advise_launches(planned, now=_NOW)["research/k8s"].allowed_cells == 0


def test_a_multi_worker_cell_is_measured_whole() -> None:
    """Eight free accelerators are two four-accelerator cells, not eight."""

    reading = _pass(_ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="24")))

    advice = reading.advise_launches([(_pool(), (2, 2, 2))], now=_NOW)

    assert advice["research/k8s"].allowed_cells == 1
    assert advice["research/k8s"].withheld_cells == 2


@pytest.mark.parametrize(
    ("label", "api"),
    [
        ("no-api-access", None),
        ("no-local-queue", _ClusterApi(cluster_queue=_cluster_queue(reserved="32"), local_queue="")),
        ("unpopulated-status", _ClusterApi(cluster_queue=_cluster_queue(reserved=None))),
        ("cluster-queue-absent", _ClusterApi()),
        ("cohort-lower-bound", _ClusterApi(cluster_queue=_cluster_queue(reserved="32", cohort="shared"))),
    ],
)
def test_an_unknown_cluster_leaves_the_launch_count_untouched(label: str, api: _ClusterApi | None) -> None:
    """Every way of establishing nothing produces the caller's own number."""

    advice = _pass(api).advise_launches([(_pool(), (1, 1, 1))], now=_NOW)["research/k8s"]

    assert advice.allowed_cells == 3, label
    assert advice.applied is False
    assert advice.withheld_cells == 0


def test_an_api_that_raises_leaves_the_launch_count_untouched() -> None:
    """A plan input that can fail a reconciliation is not a plan input."""

    class _BrokenApi(_ClusterApi):
        def observe(self, path: str, *, params: tp.Mapping[str, str] | None = None) -> ApiObservation:
            raise RuntimeError("the describing read broke its own contract")

    advice = _pass(_BrokenApi()).advise_launches([(_pool(), (1, 1))], now=_NOW)["research/k8s"]

    assert advice.allowed_cells == 2
    assert advice.applied is False


def test_a_factory_that_cannot_build_a_client_is_unknown_rather_than_an_error() -> None:
    """A control plane outside the cluster has no ServiceAccount, and says so by
    changing nothing."""

    def _factory():
        raise RuntimeError("no in-cluster credentials")

    capacity = ClusterCapacityPass(policy=_ON, api_factory=_factory)

    assert capacity.advise_launches([(_pool(), (1,))], now=_NOW)["research/k8s"].allowed_cells == 1


def test_a_reading_this_pass_took_is_fresh_enough_to_use() -> None:
    """The pass stamps its own clock on the reading it just took.

    Two clocks -- a fixed planning time against a reading stamped by the wall
    clock -- would leave every reading either expired or stamped in the future,
    and a signal that can never be used is a signal that silently is not one.
    """

    advice = _pass(_ClusterApi(cluster_queue=_cluster_queue(reserved="32"))).advise_launches([(_pool(), (1, 1))])[
        "research/k8s"
    ]

    assert advice.allowed_cells == 0
    assert advice.applied is True


def test_a_reading_kept_past_the_freshness_bound_stops_being_evidence() -> None:
    """A cluster observed minutes ago has since admitted other people's work."""

    capacity = _pass(_ClusterApi(cluster_queue=_cluster_queue(reserved="32")))
    target = kubernetes_capacity_target(_pool())
    assert target is not None
    readings = capacity.read([target], now=_NOW)

    fresh = capacity.advise(target, (1, 1), readings["research/k8s"], now=_NOW + timedelta(seconds=1))
    stale = capacity.advise(target, (1, 1), readings["research/k8s"], now=_NOW + timedelta(seconds=600))

    assert fresh.allowed_cells == 0
    assert stale.allowed_cells == 2
    assert stale.applied is False


def test_a_batch_of_more_than_one_shape_is_not_bounded_by_one_fit() -> None:
    """Capping a mixed batch by the largest cell's fit refuses smaller cells that
    do fit, which is inventing a shortage out of an unanswered question."""

    advice = _pass(_ClusterApi(cluster_queue=_cluster_queue(reserved="32"))).advise_launches(
        [(_pool(), (1, 4))], now=_NOW
    )["research/k8s"]

    assert advice.allowed_cells == 2
    assert advice.applied is False
    assert "more than one shape" in advice.reason


# ------------------------------------------------------------ what it costs


def test_a_pool_planning_no_launch_is_not_read_for() -> None:
    api = _ClusterApi(cluster_queue=_cluster_queue(reserved="32"))

    assert _pass(api).advise_launches([(_pool(), ())], now=_NOW) == {}
    assert api.observed == []


def test_two_pools_behind_one_queue_are_read_once() -> None:
    """One reading per Kubernetes pool is the ceiling, not the requirement."""

    api = _ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="32"))

    advice = _pass(api).advise_launches(
        [(_pool("research/a"), (1,)), (_pool("research/b"), (1,))],
        now=_NOW,
    )

    assert api.observed == [
        f"{KUEUE_API}/namespaces/eyvan/localqueues/research",
        f"{KUEUE_API}/clusterqueues/research-cq",
    ]
    assert advice["research/a"].allowed_cells == 0
    assert advice["research/b"].allowed_cells == 0


def test_two_pools_behind_one_queue_and_one_flavor_are_still_read_once() -> None:
    """Naming the same flavor twice does not turn one reading into two."""

    api = _ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="32"))
    api.observations[f"{KUEUE_API}/resourceflavors/tpu-v5p"] = ApiObservation(
        payload={"spec": {"nodeLabels": {"eyvan.io/flavor": "tpu-v5p"}}}, readable=True
    )

    advice = _pass(api).advise_launches(
        [(_pool("research/a", flavor="tpu-v5p"), (1,)), (_pool("research/b", flavor="tpu-v5p"), (1,))],
        now=_NOW,
    )

    assert [path for path in api.observed if "resourceflavors" in path] == [
        f"{KUEUE_API}/resourceflavors/tpu-v5p",
    ]
    assert [path for path in api.observed if path.startswith("/api/v1/nodes")] == [
        "/api/v1/nodes?eyvan.io/flavor=tpu-v5p",
    ]
    assert advice["research/a"].allowed_cells == 0
    assert advice["research/b"].allowed_cells == 0


def test_two_flavors_behind_one_queue_are_not_answered_with_one_reading() -> None:
    """A reading is scoped to one ResourceFlavor; a shared queue is not enough.

    Two pools on one ClusterQueue with different flavors would otherwise fit
    pool B's request against pool A's flavor, so an exact shortage in A's
    flavor caps B's launches to zero on quota B never asked for.
    """

    api = _ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="32"))
    for flavor in ("tpu-v5p", "tpu-v6e"):
        api.observations[f"{KUEUE_API}/resourceflavors/{flavor}"] = ApiObservation(
            payload={"spec": {"nodeLabels": {"eyvan.io/flavor": flavor}}}, readable=True
        )

    advice = _pass(api).advise_launches(
        [(_pool("research/a", flavor="tpu-v5p"), (1,)), (_pool("research/b", flavor="tpu-v6e"), (1,))],
        now=_NOW,
    )

    assert [path for path in api.observed if "resourceflavors" in path] == [
        f"{KUEUE_API}/resourceflavors/tpu-v5p",
        f"{KUEUE_API}/resourceflavors/tpu-v6e",
    ]
    assert [path for path in api.observed if path.startswith("/api/v1/nodes")] == [
        "/api/v1/nodes?eyvan.io/flavor=tpu-v5p",
        "/api/v1/nodes?eyvan.io/flavor=tpu-v6e",
    ]
    assert advice["research/a"].allowed_cells == 0
    assert advice["research/b"].allowed_cells == 0


def test_no_node_list_is_read_unless_a_pool_names_a_flavor() -> None:
    """The node reading only ever produces a remedy sentence; a third GET per
    pool is not a trade a scaling pass should make by default."""

    api = _ClusterApi(cluster_queue=_cluster_queue(reserved="32"))

    _pass(api).advise_launches([(_pool(), (1,))], now=_NOW)

    assert not [path for path in api.observed if path.startswith("/api/v1/nodes")]


# ------------------------------------------------------------ in the real pass


def _fleet_admin() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _controller(capacity: ClusterCapacityPass) -> tuple[EyvanController, InMemoryRepository]:
    """Build a control plane whose autoscaler reads the cluster through ``capacity``."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service)
    controller._cluster_capacity_pass = capacity
    await service.apply_fleet(
        _fleet_admin(),
        "research",
        (_pool(initial_cells=4),),
        idempotency_key="fleet-v1",
    )
    return controller, repository


@pytest.mark.asyncio
async def test_the_autoscaler_launches_only_what_the_cluster_is_proven_to_admit() -> None:
    """The wiring, end to end: eight free accelerators are two four-accelerator cells."""

    api = _ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="24"))
    controller, repository = await _controller(_pass(api))

    assert await controller.reconcile_capacity_pools_once() == {"launched": 2, "drained": 0, "deleted": 0}
    async with repository.transaction() as transaction:
        cells = await transaction.list_cells()
    assert len(cells) == 2
    assert api.observed == [
        f"{KUEUE_API}/namespaces/eyvan/localqueues/research",
        f"{KUEUE_API}/clusterqueues/research-cq",
    ]


@pytest.mark.asyncio
async def test_the_autoscaler_is_untouched_when_the_cluster_is_unknown() -> None:
    """An RBAC refusal must not strand a pool that scaled fine yesterday."""

    api = _ClusterApi(cluster_queue=None)
    api.observations[f"{KUEUE_API}/clusterqueues/research-cq"] = ApiObservation(
        payload=None, readable=False, reason="HTTP 403: clusterqueues is forbidden"
    )
    controller, _repository = await _controller(_pass(api))

    assert await controller.reconcile_capacity_pools_once() == {"launched": 4, "drained": 0, "deleted": 0}


@pytest.mark.asyncio
async def test_a_control_plane_that_did_not_enable_this_scales_exactly_as_before() -> None:
    """Default-off, proven where it matters: no request, and the same fleet."""

    api = _ClusterApi(cluster_queue=_cluster_queue(nominal="32", reserved="32"))
    controller, _repository = await _controller(ClusterCapacityPass(api_factory=lambda: api))

    assert await controller.reconcile_capacity_pools_once() == {"launched": 4, "drained": 0, "deleted": 0}
    assert api.observed == []


@pytest.mark.asyncio
async def test_the_cluster_is_read_before_any_cell_exists_and_outside_every_transaction() -> None:
    """It describes someone else's API server.

    Holding a lease while waiting on one turns a slow read into a stuck cell, so
    the pass must own no transaction while it reads. Proven by opening an
    independent transaction from inside the read: the repository serializes
    transactions, so one already held would make this time out rather than
    return.
    """

    loop = asyncio.get_running_loop()
    observed_cell_counts: list[int | None] = []

    class _ProbingApi(_ClusterApi):
        def observe_local_queue(self, namespace: str, name: str) -> ApiObservation:
            observed_cell_counts.append(asyncio.run_coroutine_threadsafe(_count_cells(), loop).result(timeout=10))
            return super().observe_local_queue(namespace, name)

    async def _count_cells() -> int | None:
        try:
            async with asyncio.timeout(1.0):
                async with repository.transaction() as transaction:
                    return len(await transaction.list_cells())
        except TimeoutError:
            return None

    api = _ProbingApi(cluster_queue=_cluster_queue(nominal="32", reserved="24"))
    controller, repository = await _controller(_pass(api))

    assert await controller.reconcile_capacity_pools_once() == {"launched": 2, "drained": 0, "deleted": 0}
    assert observed_cell_counts == [0]
