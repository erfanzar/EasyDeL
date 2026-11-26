# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cluster capacity awareness on Kubernetes (W3), read as an observation.

Everything here exists to hold one line: **an unreadable cluster is UNKNOWN and
changes nothing -- never full, never empty.** The failure this guards against is
not a crash, it is a plausible number. A ClusterQueue whose ``status`` has not
been populated yet, read carelessly, reports an entire saturated cluster as
free; an RBAC refusal, read carelessly, reports it as full and strands the pool.
The reading needs an explicit state rather than only a number to distinguish
those cases from known capacity.

The fake API server is ``httpx.MockTransport``, the same substitute
``test_kubernetes_transport.py`` uses. It proves what Eyvan does with a given
response body; it proves nothing about what a real Kueue installation puts in
that body, which needs a cluster and is stated as such in the W3 gate.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject
from eyvan.capacity import CapacityStatus
from eyvan.capacity.providers.kubernetes.api import ApiObservation, KubernetesApi, interpret_job
from eyvan.capacity.providers.kubernetes.kueue import (
    ClusterCapacityReading,
    ClusterCapacityState,
    capacity_fit,
    cell_capacity_state,
    explain_cluster_capacity,
    node_label_selector,
    observe_cluster_capacity,
    parse_quantity,
    read_cluster_queue_capacity,
    read_node_allocatable,
    resource_flavor_node_labels,
)

_NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_TPU = "google.com/tpu"


def _cluster_queue(
    *,
    nominal: str = "32",
    reserved: str | None = "8",
    flavor: str = "tpu-v5p",
    cohort: str = "",
    borrowing_limit: str | None = None,
    pending: int = 0,
    admitted: int = 0,
    active: str | None = "True",
    resource: str = _TPU,
) -> JsonObject:
    """Build a ClusterQueue shaped exactly as Kueue v1beta1 publishes one."""

    quota: JsonObject = {"name": resource, "nominalQuota": nominal}
    if borrowing_limit is not None:
        quota["borrowingLimit"] = borrowing_limit
    spec: JsonObject = {
        "resourceGroups": [{"coveredResources": [resource], "flavors": [{"name": flavor, "resources": [quota]}]}]
    }
    if cohort:
        spec["cohort"] = cohort
    status: JsonObject = {"pendingWorkloads": pending, "admittedWorkloads": admitted}
    if reserved is not None:
        status["flavorsReservation"] = [{"name": flavor, "resources": [{"name": resource, "total": reserved}]}]
    if active is not None:
        status["conditions"] = [{"type": "Active", "status": active}]
    return {"apiVersion": "kueue.x-k8s.io/v1beta1", "kind": "ClusterQueue", "spec": spec, "status": status}


def _readable(payload: JsonObject | None) -> ApiObservation:
    return ApiObservation(payload=payload, readable=True)


def _reading(payload: JsonObject | None, *, name: str = "research", flavor: str = "") -> ClusterCapacityReading:
    return ClusterCapacityReading(
        queue=read_cluster_queue_capacity(_readable(payload), name),
        observed_at=_NOW,
        flavor=flavor,
    )


def _api(handler, tmp_path) -> KubernetesApi:
    """Build the real transport over a mock, with a real token file on disk."""

    token_path = tmp_path / "token"
    token_path.write_text("sa-token-value", encoding="utf-8")
    return KubernetesApi(
        "https://kubernetes.default.svc",
        token_path=str(token_path),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://kubernetes.default.svc"),
    )


# --------------------------------------------------------------- quantities


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4", 4.0),
        (4, 4.0),
        ("250m", 0.25),
        ("32Gi", 32 * 1024**3),
        ("2G", 2e9),
        ("1e3", 1000.0),
        ("1E", 1e18),
        ("0", 0.0),
    ],
)
def test_kubernetes_quantities_are_read_in_base_units(raw: object, expected: float) -> None:
    """A TPU count and a memory quantity share one field type and one parser."""

    assert parse_quantity(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "  ", "abc", "4x", None, True, "-4", "NaN", [], {}])
def test_an_unreadable_quantity_is_none_and_never_zero(raw: object) -> None:
    """Zero is a capacity claim. ``None`` is the absence of one."""

    assert parse_quantity(raw) is None


# --------------------------------------------------------------- the reading


def test_free_capacity_is_nominal_quota_minus_what_kueue_reserved() -> None:
    """Reservation, not usage: reservation is the number Kueue admits against."""

    capacity = read_cluster_queue_capacity(_readable(_cluster_queue(nominal="32", reserved="8")), "research")

    assert capacity.state is ClusterCapacityState.OBSERVED
    assert capacity.free(_TPU) == (24.0, True)


def test_a_cluster_queue_without_a_populated_status_is_unknown_not_empty() -> None:
    """The defect this whole package is organised around.

    A queue that has not reported ``flavorsReservation`` yet is the moment Eyvan
    knows *least* about the cluster. Reading a missing reservation as "nothing is
    reserved" would report a saturated cluster as entirely free, and the pool
    would launch straight into it.
    """

    capacity = read_cluster_queue_capacity(_readable(_cluster_queue(reserved=None)), "research")

    assert capacity.state is ClusterCapacityState.UNKNOWN
    assert capacity.free(_TPU) is None
    assert "flavorsReservation" in capacity.reason


def test_an_unreadable_cluster_queue_is_unknown_not_full() -> None:
    """RBAC, a 500, or a severed connection all establish the same nothing."""

    capacity = read_cluster_queue_capacity(
        ApiObservation(payload=None, readable=False, reason="HTTP 403: forbidden"), "research"
    )

    assert capacity.state is ClusterCapacityState.UNKNOWN
    assert capacity.free(_TPU) is None
    assert "403" in capacity.reason


def test_a_missing_cluster_queue_is_unknown_capacity_rather_than_no_capacity() -> None:
    """Absent is a readable answer about existence, not about free capacity."""

    capacity = read_cluster_queue_capacity(_readable(None), "research")

    assert capacity.state is ClusterCapacityState.UNKNOWN
    assert "does not exist" in capacity.reason


def test_an_unparseable_reservation_total_makes_the_whole_reading_unknown() -> None:
    """One malformed total cannot be dropped: the rest would understate usage."""

    payload = _cluster_queue()
    payload["status"]["flavorsReservation"][0]["resources"][0]["total"] = "eight"

    assert read_cluster_queue_capacity(_readable(payload), "research").state is ClusterCapacityState.UNKNOWN


def test_a_resource_with_no_declared_quota_is_absent_rather_than_free() -> None:
    """Silence about ``cpu`` is not a promise of unlimited ``cpu``."""

    capacity = read_cluster_queue_capacity(_readable(_cluster_queue()), "research")

    assert capacity.free("cpu") is None


def test_cohort_borrowing_makes_free_capacity_a_lower_bound() -> None:
    """Borrowing depends on queues Eyvan cannot see, so the number is a floor."""

    capacity = read_cluster_queue_capacity(_readable(_cluster_queue(cohort="shared")), "research")

    assert capacity.free(_TPU) == (24.0, False)


def test_an_explicit_zero_borrowing_limit_restores_an_exact_reading() -> None:
    """A cohort member that may not borrow is as exact as one with no cohort."""

    payload = _cluster_queue(cohort="shared", borrowing_limit="0")

    assert read_cluster_queue_capacity(_readable(payload), "research").free(_TPU) == (24.0, True)


def test_kueue_pending_and_admitted_counts_are_retained_for_the_explanation() -> None:
    capacity = read_cluster_queue_capacity(_readable(_cluster_queue(pending=7, admitted=3)), "research")

    assert capacity.pending_workloads == 7
    assert capacity.admitted_workloads == 3
    assert capacity.active is True


def test_an_absent_active_condition_is_unknown_rather_than_inactive() -> None:
    """A condition Kueue has not written is not a condition Kueue denied."""

    assert read_cluster_queue_capacity(_readable(_cluster_queue(active=None)), "research").active is None


# --------------------------------------------------------------- the fit


def test_an_observed_queue_says_how_many_cells_it_admits() -> None:
    capacity = read_cluster_queue_capacity(_readable(_cluster_queue(nominal="32", reserved="8")), "research")

    fit = capacity_fit(capacity, {_TPU: "4"})

    assert fit.state is ClusterCapacityState.OBSERVED
    assert fit.admits == 6
    assert fit.exact is True
    assert fit.blocked is False


def test_a_full_cluster_is_blocked_only_when_the_reading_is_exact() -> None:
    """``admits == 0`` from a lower bound is not proof of a full cluster."""

    exact = read_cluster_queue_capacity(_readable(_cluster_queue(nominal="32", reserved="32")), "research")
    bounded = read_cluster_queue_capacity(
        _readable(_cluster_queue(nominal="32", reserved="32", cohort="shared")), "research"
    )

    assert capacity_fit(exact, {_TPU: "4"}).blocked is True
    assert capacity_fit(bounded, {_TPU: "4"}).admits == 0
    assert capacity_fit(bounded, {_TPU: "4"}).blocked is False


def test_an_unknown_queue_never_produces_a_numeric_fit() -> None:
    """``admits`` is ``None``, not ``0``: nothing can read it as "no room"."""

    fit = capacity_fit(read_cluster_queue_capacity(_readable(None), "research"), {_TPU: "4"})

    assert fit.state is ClusterCapacityState.UNKNOWN
    assert fit.admits is None
    assert fit.blocked is False


def test_a_request_naming_a_resource_the_queue_does_not_cover_is_unknown() -> None:
    capacity = read_cluster_queue_capacity(_readable(_cluster_queue()), "research")

    fit = capacity_fit(capacity, {_TPU: "4", "nvidia.com/gpu": "8"})

    assert fit.state is ClusterCapacityState.UNKNOWN
    assert "nvidia.com/gpu" in fit.reason


def test_the_scarcest_resource_is_the_one_that_limits_the_fit() -> None:
    payload = _cluster_queue(nominal="32", reserved="0")
    payload["spec"]["resourceGroups"][0]["flavors"][0]["resources"].append({"name": "cpu", "nominalQuota": "16"})
    payload["status"]["flavorsReservation"][0]["resources"].append({"name": "cpu", "total": "8"})

    fit = capacity_fit(read_cluster_queue_capacity(_readable(payload), "research"), {_TPU: "4", "cpu": "8"})

    assert fit.admits == 1
    assert fit.limiting_resource == "cpu"


def test_two_flavors_do_not_pool_their_free_capacity_into_one_cell() -> None:
    """One resource in one Workload is admitted against one ResourceFlavor.

    Summing them would report room for a four-accelerator cell that fits in
    neither half of it, and the cell would sit WAITING exactly as it does today.
    """

    payload = _cluster_queue(nominal="2", reserved="0", flavor="tpu-a")
    payload["spec"]["resourceGroups"][0]["flavors"].append(
        {"name": "tpu-b", "resources": [{"name": _TPU, "nominalQuota": "2"}]}
    )
    payload["status"]["flavorsReservation"].append({"name": "tpu-b", "resources": [{"name": _TPU, "total": "0"}]})

    capacity = read_cluster_queue_capacity(_readable(payload), "research")

    assert capacity.free(_TPU) == (2.0, True)
    assert capacity_fit(capacity, {_TPU: "4"}).admits == 0
    assert capacity_fit(capacity, {_TPU: "2"}).admits == 1


def test_resources_in_different_groups_may_use_different_flavors() -> None:
    """Kueue assigns a flavor per resource group, which the fit must respect."""

    payload = _cluster_queue(nominal="32", reserved="0", flavor="tpu-v5p")
    payload["spec"]["resourceGroups"].append(
        {
            "coveredResources": ["cpu"],
            "flavors": [{"name": "default", "resources": [{"name": "cpu", "nominalQuota": "64"}]}],
        }
    )
    payload["status"]["flavorsReservation"].append({"name": "default", "resources": [{"name": "cpu", "total": "24"}]})

    fit = capacity_fit(read_cluster_queue_capacity(_readable(payload), "research"), {_TPU: "4", "cpu": "8"})

    assert fit.admits == 5
    assert fit.limiting_resource == "cpu"


def test_a_named_flavor_scopes_its_own_group_and_not_every_other_one() -> None:
    """A TPU flavor says nothing about the group that covers ``cpu``.

    Treating the name as a global filter would report an unknown fit for a
    ClusterQueue that had just been read perfectly well.
    """

    payload = _cluster_queue(nominal="32", reserved="0", flavor="tpu-v5p")
    payload["spec"]["resourceGroups"][0]["flavors"].append(
        {"name": "tpu-v5e", "resources": [{"name": _TPU, "nominalQuota": "128"}]}
    )
    payload["status"]["flavorsReservation"].append({"name": "tpu-v5e", "resources": [{"name": _TPU, "total": "0"}]})
    payload["spec"]["resourceGroups"].append(
        {
            "coveredResources": ["cpu"],
            "flavors": [{"name": "default", "resources": [{"name": "cpu", "nominalQuota": "640"}]}],
        }
    )
    payload["status"]["flavorsReservation"].append({"name": "default", "resources": [{"name": "cpu", "total": "0"}]})
    capacity = read_cluster_queue_capacity(_readable(payload), "research")

    pinned = capacity_fit(capacity, {_TPU: "4", "cpu": "8"}, flavor="tpu-v5p")
    unpinned = capacity_fit(capacity, {_TPU: "4", "cpu": "8"})

    assert pinned.state is ClusterCapacityState.OBSERVED
    assert pinned.admits == 8
    assert unpinned.admits == 32


def test_a_flavor_that_covers_only_part_of_its_group_is_not_a_flavor_with_room() -> None:
    """A flavor missing one covered resource cannot host the request at all."""

    payload = _cluster_queue(nominal="32", reserved="0", flavor="tpu-v5p")
    payload["spec"]["resourceGroups"][0]["coveredResources"].append("cpu")
    payload["spec"]["resourceGroups"][0]["flavors"].append(
        {"name": "cpu-only", "resources": [{"name": "cpu", "nominalQuota": "64"}]}
    )
    payload["status"]["flavorsReservation"].append({"name": "cpu-only", "resources": [{"name": "cpu", "total": "0"}]})

    fit = capacity_fit(read_cluster_queue_capacity(_readable(payload), "research"), {_TPU: "4", "cpu": "8"})

    assert fit.state is ClusterCapacityState.UNKNOWN
    assert "no ResourceFlavor covers every requested resource" in fit.reason


def test_sub_unit_quantities_do_not_floor_a_cell_away() -> None:
    """1500m of free CPU is six 250m cells, not five.

    Understating free capacity is the direction that reads a cluster as fuller
    than it was observed to be.
    """

    payload = _cluster_queue(nominal="1500m", reserved="0", resource="cpu")

    assert capacity_fit(read_cluster_queue_capacity(_readable(payload), "research"), {"cpu": "250m"}).admits == 6


# --------------------------------------------------------------- nodes


def _node(*, ready: str = "True", tpu: str = "4", unschedulable: bool = False) -> JsonObject:
    return {
        "spec": {"unschedulable": unschedulable},
        "status": {
            "allocatable": {_TPU: tpu, "cpu": "8"},
            "conditions": [{"type": "Ready", "status": ready}],
        },
    }


def test_node_allocatable_sums_only_schedulable_ready_nodes() -> None:
    payload: JsonObject = {"items": [_node(), _node(), _node(ready="False"), _node(unschedulable=True)]}

    nodes = read_node_allocatable(_readable(payload))

    assert nodes.state is ClusterCapacityState.OBSERVED
    assert nodes.node_count == 2
    assert nodes.allocatable[_TPU] == 8.0
    assert nodes.complete is True


def test_a_node_whose_readiness_is_unknown_makes_the_total_a_lower_bound() -> None:
    """``Ready=Unknown`` is the kubelet having stopped reporting.

    Counting it would overstate capacity; declaring the node gone would
    understate it. Neither is known, so only the confidence changes.
    """

    nodes = read_node_allocatable(_readable({"items": [_node(), _node(ready="Unknown")]}))

    assert nodes.state is ClusterCapacityState.OBSERVED
    assert nodes.node_count == 1
    assert nodes.complete is False


def test_a_truncated_node_list_is_incomplete_rather_than_the_whole_cluster() -> None:
    """``metadata.continue`` means there are nodes Eyvan did not see."""

    payload: JsonObject = {"metadata": {"continue": "next-page-token"}, "items": [_node()]}

    assert read_node_allocatable(_readable(payload)).complete is False


def test_an_unreadable_node_list_is_unknown_and_carries_no_totals() -> None:
    nodes = read_node_allocatable(ApiObservation(payload=None, readable=False, reason="HTTP 403: forbidden"))

    assert nodes.state is ClusterCapacityState.UNKNOWN
    assert nodes.allocatable == {}
    assert "403" in nodes.reason


def test_a_resource_flavor_selecting_every_node_differs_from_an_unreadable_one() -> None:
    """``{}`` selects everything; ``None`` establishes nothing."""

    assert resource_flavor_node_labels(_readable({"spec": {}})) == {}
    assert resource_flavor_node_labels(_readable({"spec": {"nodeLabels": {"a": "b"}}})) == {"a": "b"}
    assert resource_flavor_node_labels(ApiObservation(payload=None, readable=False, reason="403")) is None
    assert resource_flavor_node_labels(_readable(None)) is None


def test_node_labels_render_as_a_stable_label_selector() -> None:
    assert node_label_selector({"b": "2", "a": "1"}) == "a=1,b=2"


# --------------------------------------------------------------- explain


def test_a_full_cluster_explains_itself_with_numbers_and_a_remedy() -> None:
    """The W3 deliverable for a human.

    Today an Eyvan pool behind a saturated ClusterQueue says "job is suspended
    pending kueue admission" -- which restates Eyvan's own state and names no
    cause. This is the sentence that replaces it.
    """

    reading = _reading(_cluster_queue(nominal="32", reserved="32", pending=7, admitted=3))

    explanation = explain_cluster_capacity(reading, {_TPU: "4"})

    assert explanation.state is ClusterCapacityState.OBSERVED
    assert explanation.blocked is True
    assert "research" in explanation.summary
    assert _TPU in explanation.summary
    assert "7 workload(s) already pending" in explanation.summary
    assert "raise nominalQuota" in explanation.remedy
    assert "3 admitted workload(s)" in explanation.remedy


def test_an_unknown_cluster_explains_that_it_is_unknown_and_is_not_planning() -> None:
    """ "I could not find out" stays "I could not find out"."""

    explanation = explain_cluster_capacity(_reading(None), {_TPU: "4"})

    assert explanation.state is ClusterCapacityState.UNKNOWN
    assert explanation.blocked is False
    assert "unknown" in explanation.summary
    assert "not being used as one" in explanation.remedy


def test_a_cluster_with_room_reports_how_much_room_rather_than_nothing() -> None:
    explanation = explain_cluster_capacity(_reading(_cluster_queue(nominal="32", reserved="8")), {_TPU: "4"})

    assert explanation.blocked is False
    assert "admits 6 cell(s)" in explanation.summary


def test_a_lower_bound_says_so_instead_of_claiming_an_exact_count() -> None:
    reading = _reading(_cluster_queue(nominal="32", reserved="8", cohort="shared"))

    assert "or more" in explain_cluster_capacity(reading, {_TPU: "4"}).summary


def test_quota_that_no_node_can_honour_is_named_as_a_misconfiguration() -> None:
    """A pool queues forever on a nominal quota with no hardware behind it."""

    reading = ClusterCapacityReading(
        queue=read_cluster_queue_capacity(_readable(_cluster_queue(nominal="32", reserved="0")), "research"),
        nodes=read_node_allocatable(_readable({"items": [_node(tpu="4"), _node(tpu="4")]})),
        flavor="tpu-v5p",
        observed_at=_NOW,
    )

    explanation = explain_cluster_capacity(reading, {_TPU: "4"})

    assert "not fully backed by hardware" in explanation.remedy


def test_a_flavor_whose_labels_match_no_node_is_named_as_unbacked_quota() -> None:
    """The strongest form of the same misconfiguration: no hardware at all."""

    reading = ClusterCapacityReading(
        queue=read_cluster_queue_capacity(_readable(_cluster_queue(nominal="32", reserved="0")), "research"),
        nodes=read_node_allocatable(_readable({"items": []})),
        flavor="tpu-v5p",
        observed_at=_NOW,
    )

    explanation = explain_cluster_capacity(reading, {_TPU: "4"})

    assert explanation.blocked is False
    assert "allocate only 0" in explanation.remedy


def test_an_incomplete_node_reading_never_claims_a_hardware_shortage() -> None:
    """A lower bound on allocatable can always be made to look like a shortage."""

    reading = ClusterCapacityReading(
        queue=read_cluster_queue_capacity(_readable(_cluster_queue(nominal="32", reserved="0")), "research"),
        nodes=read_node_allocatable(_readable({"metadata": {"continue": "more"}, "items": [_node(tpu="4")]})),
        flavor="tpu-v5p",
        observed_at=_NOW,
    )

    assert explain_cluster_capacity(reading, {_TPU: "4"}).remedy == ""


def test_a_reading_must_carry_a_timezone_aware_observation_time() -> None:
    """A reading with no time cannot be told from a stale one."""

    with pytest.raises(ValueError):
        ClusterCapacityReading(
            queue=read_cluster_queue_capacity(_readable(_cluster_queue()), "research"),
            observed_at=datetime(2026, 8, 9, 12, 0),
        )


# --------------------------------------------------------------- cell state


def _suspended():
    return interpret_job({"spec": {"suspend": True, "parallelism": 2}, "status": {}})


def test_a_waiting_cell_keeps_todays_wording_when_capacity_is_unknown() -> None:
    """The default path must be byte-identical to the shipped behaviour."""

    unknown = explain_cluster_capacity(_reading(None), {_TPU: "4"})

    assert cell_capacity_state(_suspended()) == (CapacityStatus.WAITING, "job is suspended pending kueue admission")
    assert cell_capacity_state(_suspended(), unknown) == (
        CapacityStatus.WAITING,
        "job is suspended pending kueue admission",
    )


def test_a_waiting_cell_behind_a_proven_full_cluster_says_why() -> None:
    """WAITING with a cause, rather than WAITING with Eyvan's own state."""

    explanation = explain_cluster_capacity(_reading(_cluster_queue(nominal="32", reserved="32")), {_TPU: "4"})

    state, detail = cell_capacity_state(_suspended(), explanation)

    assert state is CapacityStatus.WAITING
    assert detail.startswith("job is suspended pending kueue admission")
    assert "admits no further cell of this shape" in detail
    assert "raise nominalQuota" in detail


def test_a_healthy_cell_is_never_annotated_with_cluster_capacity() -> None:
    """A running cell's state is not a question about free capacity."""

    healthy = interpret_job({"spec": {"parallelism": 2}, "status": {"ready": 2}})
    explanation = explain_cluster_capacity(_reading(_cluster_queue(nominal="32", reserved="32")), {_TPU: "4"})

    assert cell_capacity_state(healthy, explanation) == (CapacityStatus.HEALTHY, "2/2 workers ready")


# --------------------------------------------------------------- transport


def test_an_rbac_refusal_degrades_to_unknown_instead_of_raising(tmp_path) -> None:
    """The required W3 behaviour, end to end through the real transport.

    A ServiceAccount without ``get clusterqueues`` is the ordinary state of a
    cluster where Eyvan was installed before Kueue. It must not raise -- the
    reconciliation that would have failed has nothing to do with cluster
    capacity -- and it must not be mistaken for an empty or a full queue.
    """

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(403, json={"reason": "Forbidden", "message": "clusterqueues is forbidden"})

    reading = observe_cluster_capacity(_api(handler, tmp_path), "research", now=_NOW)

    assert seen == ["/apis/kueue.x-k8s.io/v1beta1/clusterqueues/research"]
    assert reading.state is ClusterCapacityState.UNKNOWN
    assert "403" in reading.queue.reason
    assert capacity_fit(reading.queue, {_TPU: "4"}).admits is None


def test_a_transport_failure_degrades_to_unknown_rather_than_propagating(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    reading = observe_cluster_capacity(_api(handler, tmp_path), "research", now=_NOW)

    assert reading.state is ClusterCapacityState.UNKNOWN


def test_an_observed_cluster_is_read_through_queue_flavor_and_nodes(tmp_path) -> None:
    """The whole read path: ClusterQueue, then ResourceFlavor, then its nodes."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/clusterqueues/research"):
            return httpx.Response(200, json=_cluster_queue(nominal="32", reserved="8"))
        if request.url.path.endswith("/resourceflavors/tpu-v5p"):
            return httpx.Response(200, json={"spec": {"nodeLabels": {"cloud.google.com/gke-tpu": "v5p"}}})
        return httpx.Response(200, json={"items": [_node(tpu="4"), _node(tpu="4")]})

    reading = observe_cluster_capacity(_api(handler, tmp_path), "research", flavor="tpu-v5p", now=_NOW)

    assert reading.state is ClusterCapacityState.OBSERVED
    assert reading.queue.free(_TPU) == (24.0, True)
    assert reading.nodes.allocatable[_TPU] == 8.0
    assert seen[-1].url.params["labelSelector"] == "cloud.google.com/gke-tpu=v5p"
    assert reading.observed_at == _NOW


def test_an_unreadable_node_list_does_not_degrade_a_readable_queue(tmp_path) -> None:
    """Each read fails on its own; one 403 does not erase the other answer."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/clusterqueues/research"):
            return httpx.Response(200, json=_cluster_queue(nominal="32", reserved="8"))
        if request.url.path.endswith("/resourceflavors/tpu-v5p"):
            return httpx.Response(200, json={"spec": {"nodeLabels": {"a": "b"}}})
        return httpx.Response(403, json={"reason": "Forbidden"})

    reading = observe_cluster_capacity(_api(handler, tmp_path), "research", flavor="tpu-v5p", now=_NOW)

    assert reading.state is ClusterCapacityState.OBSERVED
    assert reading.queue.free(_TPU) == (24.0, True)
    assert reading.nodes.state is ClusterCapacityState.UNKNOWN


def test_a_flavor_that_cannot_be_read_does_not_select_every_node(tmp_path) -> None:
    """An unreadable flavor would otherwise attribute the whole cluster to it."""

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/clusterqueues/research"):
            return httpx.Response(200, json=_cluster_queue())
        return httpx.Response(403, json={"reason": "Forbidden"})

    reading = observe_cluster_capacity(_api(handler, tmp_path), "research", flavor="tpu-v5p", now=_NOW)

    assert "/api/v1/nodes" not in seen
    assert reading.nodes.state is ClusterCapacityState.UNKNOWN


def test_the_node_read_is_bounded_so_one_observation_is_one_round_trip(tmp_path) -> None:
    """An observation that costs the API server twenty GETs gets turned off."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"items": []})

    _api(handler, tmp_path).observe_nodes(label_selector="a=b", limit=10_000)

    assert seen[0].url.params["limit"] == "500"


def test_a_missing_object_reads_as_readable_and_absent(tmp_path) -> None:
    """404 answers "does it exist", which is not the same as "I could not ask"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"reason": "NotFound"})

    observation = _api(handler, tmp_path).observe_cluster_queue("research")

    assert observation.readable is True
    assert observation.payload is None


def test_a_non_json_body_is_unreadable_rather_than_an_empty_object(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error</html>")

    observation = _api(handler, tmp_path).observe_cluster_queue("research")

    assert observation.unreadable is True
    assert observation.payload is None


def test_an_error_body_is_truncated_on_an_unreadable_observation(tmp_path) -> None:
    """The reason lands in a cell detail, a log line and a dashboard panel."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="x" * 5_000)

    observation = _api(handler, tmp_path).observe_cluster_queue("research")

    assert observation.unreadable is True
    assert len(observation.reason) < 400


def test_an_observation_defaults_its_own_time_when_none_is_given(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_cluster_queue())

    reading = observe_cluster_capacity(_api(handler, tmp_path), "research")

    assert reading.observed_at.tzinfo is not None
    assert abs(datetime.now(UTC) - reading.observed_at) < timedelta(seconds=30)
