# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ClusterQueue headroom belongs to the whole launch pass, not each pool."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from eyvan._internal.compat import UTC
from eyvan.capacity.autoscale.cluster_capacity import ClusterCapacityPolicy
from eyvan.capacity.autoscale.pool_capacity import ClusterCapacityPass
from eyvan.capacity.providers.kubernetes.api import ApiObservation
from eyvan.capacity.providers.kubernetes.kueue import ClusterCapacityState
from eyvan.records import CapacityPoolRecord

_NOW = datetime(2026, 10, 1, tzinfo=UTC)
_TPU = "google.com/tpu"


class _Api:
    def __init__(self, groups, *, cohort="", readable=True):
        self.calls = []
        self.groups = groups
        self.cohort = cohort
        self.readable = readable

    def observe_local_queue(self, namespace, name):
        self.calls.append(("local", namespace, name))
        return ApiObservation(payload={"spec": {"clusterQueue": "shared"}}, readable=True)

    def observe_cluster_queue(self, name):
        self.calls.append(("cluster", name))
        flavors = {}
        for group in self.groups:
            for flavor in group["flavors"]:
                flavors.setdefault(flavor["name"], []).extend(
                    {"name": resource["name"], "total": "0"} for resource in flavor["resources"]
                )
        return ApiObservation(
            payload={
                "spec": {"resourceGroups": self.groups, "cohort": self.cohort},
                "status": {
                    "flavorsReservation": [{"name": name, "resources": resources} for name, resources in flavors.items()]
                },
            },
            readable=self.readable,
        )

    def observe_resource_flavor(self, name):
        self.calls.append(("flavor", name))
        return ApiObservation(payload=None, readable=True)


def _group(resources_by_flavor):
    return {
        "coveredResources": sorted({resource for resources in resources_by_flavor.values() for resource in resources}),
        "flavors": [
            {
                "name": flavor,
                "resources": [{"name": resource, "nominalQuota": str(amount)} for resource, amount in resources.items()],
            }
            for flavor, resources in resources_by_flavor.items()
        ],
    }


def _pool(name, resources, *, flavor="", queue="research"):
    return CapacityPoolRecord(
        id=f"research/{name}",
        fleet_name="research",
        node_type=name,
        provider="kubernetes",
        capacity_backend="kubernetes",
        region="local",
        capacity={"tpu": 4},
        allocatable={"tpu": 4},
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        provider_options={"queue_name": queue, "resources": resources, "resource_flavor": flavor},
        max_cells=8,
    )


def _advise(api, planned, *, enabled=True):
    return ClusterCapacityPass(policy=ClusterCapacityPolicy(enabled=enabled), api_factory=lambda: api).advise_launches(
        planned, now=_NOW
    )


@pytest.mark.parametrize("flavor", ["", "v5p"])
def test_shared_queue_reserves_headroom_once(flavor):
    api = _Api([_group({"v5p": {_TPU: 8}})])
    advice = _advise(
        api,
        [(_pool("a", {_TPU: "4"}, flavor=flavor), (1, 1)), (_pool("b", {_TPU: "4"}, flavor=flavor), (1, 1))],
    )
    assert {name: value.allowed_cells for name, value in advice.items()} == {"research/a": 2, "research/b": 0}
    assert all(value.applied for value in advice.values())
    assert api.calls.count(("cluster", "shared")) == 1


def test_different_shapes_and_local_queues_share_one_resource_balance():
    api = _Api([_group({"v5p": {_TPU: 12}})])
    advice = _advise(
        api,
        [(_pool("a", {_TPU: "4"}), (2,)), (_pool("b", {_TPU: "2"}, queue="other"), (1, 1, 1, 1))],
    )
    assert advice["research/a"].allowed_cells == 1
    assert advice["research/b"].allowed_cells == 2


def test_distinct_accelerator_flavors_share_the_cpu_resource_group():
    api = _Api([_group({"v5p": {_TPU: 8}, "v6e": {_TPU: 8}}), _group({"default": {"cpu": 8}})])
    advice = _advise(
        api,
        [
            (_pool("a", {_TPU: "4", "cpu": "4"}, flavor="v5p"), (1, 1)),
            (_pool("b", {_TPU: "4", "cpu": "4"}, flavor="v6e"), (1, 1)),
        ],
    )
    assert advice["research/a"].allowed_cells == 2
    assert advice["research/b"].allowed_cells == 0
    assert advice["research/b"].applied


def test_distinct_flavor_headroom_is_not_subtracted_from_the_other_flavor():
    api = _Api([_group({"v5p": {_TPU: 8}, "v6e": {_TPU: 8}})])
    advice = _advise(
        api,
        [(_pool("a", {_TPU: "4"}, flavor="v5p"), (1, 1)), (_pool("b", {_TPU: "4"}, flavor="v6e"), (1, 1))],
    )
    assert [value.allowed_cells for value in advice.values()] == [2, 2]


@pytest.mark.parametrize("scoped", [False, True])
def test_multiple_flavors_reserve_only_when_the_request_has_an_exact_scope(scoped):
    api = _Api([_group({"v5p": {_TPU: 8}, "v6e": {_TPU: 8}})])
    advice = _advise(
        api,
        [
            (_pool(name, {_TPU: "4"}, flavor=flavor if scoped else ""), (1, 1))
            for name, flavor in (("a", "v5p"), ("b", "v6e"), ("c", "v5p"))
        ],
    )
    assert [value.allowed_cells for value in advice.values()] == ([2, 2, 0] if scoped else [2, 2, 2])
    for value in advice.values():
        assert value.applied is scoped
        assert value.state is (ClusterCapacityState.OBSERVED if scoped else ClusterCapacityState.UNKNOWN)
        assert value.explanation is not None
        assert value.explanation.fit.exact is scoped
        if not scoped:
            assert "lower bound" in value.reason


@pytest.mark.parametrize("mode", ["disabled", "unknown", "lower-bound", "mixed"])
def test_uncertain_or_disabled_planning_still_leaves_counts_unchanged(mode):
    api = _Api(
        [_group({"v5p": {_TPU: 8}})], cohort="shared" if mode == "lower-bound" else "", readable=mode != "unknown"
    )
    counts = (1, 2) if mode == "mixed" else (1, 1)
    advice = _advise(api, [(_pool(name, {_TPU: "4"}), counts) for name in ("a", "b")], enabled=mode != "disabled")
    if mode == "disabled":
        assert advice == {}
        assert api.calls == []
    else:
        assert all(value.allowed_cells == 2 and not value.applied for value in advice.values())


def test_distinct_cluster_queues_keep_independent_balances():
    class _SeparateQueues(_Api):
        def observe_local_queue(self, namespace, name):
            return ApiObservation(payload={"spec": {"clusterQueue": name}}, readable=True)

    api = _SeparateQueues([_group({"v5p": {_TPU: 8}})])
    advice = _advise(
        api,
        [(_pool("a", {_TPU: "4"}, queue="one"), (1, 1)), (_pool("b", {_TPU: "4"}, queue="two"), (1, 1))],
    )
    assert [value.allowed_cells for value in advice.values()] == [2, 2]


def test_reservations_follow_stable_pool_order_not_input_order():
    api = _Api([_group({"v5p": {_TPU: 8}})])
    advice = _advise(api, [(_pool("b", {_TPU: "4"}), (1, 1)), (_pool("a", {_TPU: "4"}), (1, 1))])
    assert advice["research/a"].allowed_cells == 2
    assert advice["research/b"].allowed_cells == 0


@pytest.mark.parametrize("scoped", [False, True])
def test_a_reservation_keeps_coupled_resources_in_the_same_flavor(scoped):
    api = _Api([_group({"v5p": {_TPU: 8, "cpu": 4}, "v6e": {_TPU: 4, "cpu": 8}})])
    advice = _advise(
        api,
        [
            (_pool(name, {_TPU: "4", "cpu": "4"}, flavor=flavor if scoped else ""), (1,))
            for name, flavor in (("a", "v5p"), ("b", "v6e"), ("c", "v5p"))
        ],
    )
    assert [value.allowed_cells for value in advice.values()] == ([1, 1, 0] if scoped else [1, 1, 1])
    for value in advice.values():
        assert value.applied is scoped
        assert value.state is (ClusterCapacityState.OBSERVED if scoped else ClusterCapacityState.UNKNOWN)
        assert value.explanation is not None
        assert value.explanation.fit.exact is scoped
        if not scoped:
            assert "lower bound" in value.reason
    if scoped:
        assert advice["research/c"].explanation.fit.limiting_resource == "cpu"


@pytest.mark.parametrize("stale", [False, True])
def test_original_observations_remain_immutable_and_stale_readings_do_not_reserve(stale):
    class _CapturedPass(ClusterCapacityPass):
        def read(self, targets, *, now=None):
            self.readings = super().read(targets, now=now)
            if stale:
                self.readings = {
                    key: replace(reading, observed_at=now - timedelta(seconds=600))
                    for key, reading in self.readings.items()
                }
            return self.readings

    api = _Api([_group({"v5p": {_TPU: 8}})])
    capacity = _CapturedPass(policy=ClusterCapacityPolicy(enabled=True), api_factory=lambda: api)
    planned = [(_pool(name, {_TPU: "4"}), (1, 1)) for name in ("a", "b")]
    advice = capacity.advise_launches(planned, now=_NOW)
    assert [value.allowed_cells for value in advice.values()] == ([2, 2] if stale else [2, 0])
    assert all(value.applied is not stale for value in advice.values())
    assert all(entry.reserved == 0 for reading in capacity.readings.values() for entry in reading.queue.headroom)
    # A second pass starts from a new observation, not this pass's reservations.
    replay = capacity.advise_launches(planned, now=_NOW)
    assert replay == advice
