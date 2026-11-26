# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A best-single-flavor fit cannot cap independent multi-flavor cells."""

from datetime import datetime

import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject
from eyvan.capacity.autoscale.cluster_capacity import ClusterCapacityPolicy
from eyvan.capacity.autoscale.pool_capacity import ClusterCapacityPass, PoolCapacityTarget
from eyvan.capacity.providers.kubernetes.api import ApiObservation
from eyvan.capacity.providers.kubernetes.kueue import (
    ClusterCapacityReading,
    capacity_fit,
    read_cluster_queue_capacity,
)

_NOW = datetime(2026, 10, 1, tzinfo=UTC)
_GPU = "nvidia.com/gpu"


def _reading(quotas: tuple[int, ...], *, cohort: str = "") -> ClusterCapacityReading:
    flavors: list[JsonObject] = []
    reservations: list[JsonObject] = []
    for index, quota in enumerate(quotas):
        name = f"gpu-{index}"
        flavors.append({"name": name, "resources": [{"name": _GPU, "nominalQuota": str(quota)}]})
        reservations.append({"name": name, "resources": [{"name": _GPU, "total": "0"}]})
    payload: JsonObject = {
        "spec": {"cohort": cohort, "resourceGroups": [{"coveredResources": [_GPU], "flavors": flavors}]},
        "status": {"flavorsReservation": reservations},
    }
    return ClusterCapacityReading(
        queue=read_cluster_queue_capacity(ApiObservation(payload=payload, readable=True), "research"),
        observed_at=_NOW,
    )


def test_independent_cells_are_not_capped_by_the_best_single_flavor() -> None:
    """Two four-GPU quotas can admit two separate four-GPU cells."""

    reading = _reading((4, 4))
    target = PoolCapacityTarget("pool", "eyvan", "research", ((_GPU, 4.0),))
    planner = ClusterCapacityPass(policy=ClusterCapacityPolicy(enabled=True))

    advice = planner.advise(target, [1, 1], reading, now=_NOW)

    assert advice.allowed_cells == 2
    assert advice.withheld_cells == 0
    assert advice.applied is False
    assert "lower bound" in advice.reason
    fit = capacity_fit(reading.queue, {_GPU: "4"})
    assert fit.admits == 1
    assert fit.exact is False


@pytest.mark.parametrize("quotas", [(4, 4), (4, 8), (4, 4, 4)])
def test_unscoped_multiple_fitting_flavors_produce_only_a_lower_bound(quotas: tuple[int, ...]) -> None:
    fit = capacity_fit(_reading(quotas).queue, {_GPU: "4"})

    assert fit.admits == max(quotas) // 4
    assert fit.exact is False
    assert fit.blocked is False


def test_a_single_cell_still_cannot_split_across_flavors() -> None:
    reading = _reading((2, 2))
    fit = capacity_fit(reading.queue, {_GPU: "4"})
    target = PoolCapacityTarget("pool", "eyvan", "research", ((_GPU, 4.0),))
    advice = ClusterCapacityPass(policy=ClusterCapacityPolicy(enabled=True)).advise(target, [1], reading, now=_NOW)

    assert fit.admits == 0
    assert fit.exact is True
    assert fit.blocked is True
    assert advice.allowed_cells == 0
    assert advice.applied is True


@pytest.mark.parametrize("quotas", [(4,), (4, 2)])
def test_one_fitting_flavor_retains_an_exact_capacity_ceiling(quotas: tuple[int, ...]) -> None:
    fit = capacity_fit(_reading(quotas).queue, {_GPU: "4"})

    assert fit.admits == 1
    assert fit.exact is True


def test_a_named_flavor_still_caps_to_its_own_quota() -> None:
    reading = _reading((4, 4))
    target = PoolCapacityTarget("pool", "eyvan", "research", ((_GPU, 4.0),), flavor="gpu-0")
    pinned = ClusterCapacityReading(queue=reading.queue, observed_at=_NOW, flavor="gpu-0")
    advice = ClusterCapacityPass(policy=ClusterCapacityPolicy(enabled=True)).advise(target, [1, 1], pinned, now=_NOW)

    assert advice.allowed_cells == 1
    assert advice.applied is True
    assert capacity_fit(reading.queue, {_GPU: "4"}, flavor="gpu-0").exact is True


def test_default_disabled_planning_is_unchanged() -> None:
    target = PoolCapacityTarget("pool", "eyvan", "research", ((_GPU, 4.0),))
    advice = ClusterCapacityPass().advise(target, [1, 1], _reading((4, 4)), now=_NOW)

    assert advice.allowed_cells == 2
    assert advice.applied is False


def test_cohort_borrowing_remains_inexact_with_one_fitting_flavor() -> None:
    fit = capacity_fit(_reading((4,), cohort="shared").queue, {_GPU: "4"})

    assert fit.admits == 1
    assert fit.exact is False
