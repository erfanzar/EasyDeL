# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observed cluster capacity as an autoscale plan input -- and when it is not one.

The autoscaler's failure mode here is not a wrong number, it is a *confident*
one. Every test below fixes one of the three ways a cluster signal turns into a
plan input it has not earned:

* it was never established (RBAC, a 500, an unpopulated ClusterQueue status);
* it was established once and then kept past the point where it described
  anything;
* it is a lower bound, which proves room and never proves its absence.

In all three the required behaviour is identical and is the only behaviour that
is safe: the autoscaler's own decision is returned unchanged. "Launch nothing"
would strand a pool the first time RBAC is wrong, and "launch everything" is
today's behaviour restated -- so neither is a defensible default for a signal
Eyvan could not read.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject
from eyvan.capacity.autoscale import ClusterCapacityAdvice, ClusterCapacityPolicy, advise_launch, observation_age
from eyvan.capacity.providers.kubernetes.api import ApiObservation
from eyvan.capacity.providers.kubernetes.kueue import (
    ClusterCapacityReading,
    ClusterCapacityState,
    read_cluster_queue_capacity,
)

_NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_TPU = "google.com/tpu"
_CELL = {_TPU: "4"}
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


def _reading(payload: JsonObject | None, *, observed_at: datetime = _NOW) -> ClusterCapacityReading:
    return ClusterCapacityReading(
        queue=read_cluster_queue_capacity(ApiObservation(payload=payload, readable=True), "research"),
        observed_at=observed_at,
    )


def _unreadable(reason: str = "HTTP 403: forbidden") -> ClusterCapacityReading:
    return ClusterCapacityReading(
        queue=read_cluster_queue_capacity(ApiObservation(payload=None, readable=False, reason=reason), "research"),
        observed_at=_NOW,
    )


# ------------------------------------------------- unknown is never an input


@pytest.mark.parametrize(
    ("label", "reading"),
    [
        ("rbac-refusal", _unreadable()),
        ("api-server-500", _unreadable("HTTP 500: internal error")),
        ("queue-absent", _reading(None)),
        ("status-not-populated", _reading(_cluster_queue(reserved=None))),
    ],
)
def test_an_unknown_cluster_leaves_the_autoscalers_own_decision_untouched(
    label: str, reading: ClusterCapacityReading
) -> None:
    """The rule the whole package is organised around, at the plan boundary."""

    advice = advise_launch(4, reading, _CELL, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 4, label
    assert advice.applied is False
    assert advice.state is ClusterCapacityState.UNKNOWN
    assert advice.withheld_cells == 0


def test_no_reading_at_all_is_the_same_answer_as_an_unreadable_one() -> None:
    """A pool on a provider that publishes no cluster signal is not blocked."""

    advice = advise_launch(3, None, _CELL, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 3
    assert advice.applied is False


def test_a_request_the_queue_does_not_cover_is_unknown_rather_than_blocked() -> None:
    """Silence about a resource is not a statement that there is none."""

    advice = advise_launch(2, _reading(_cluster_queue()), {"nvidia.com/gpu": "8"}, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 2
    assert advice.applied is False
    assert "nvidia.com/gpu" in advice.reason


def test_an_empty_resource_request_cannot_conclude_anything() -> None:
    """A cell shape Eyvan did not state is not a cell shape that fits anywhere."""

    advice = advise_launch(2, _reading(_cluster_queue(reserved="32")), {}, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 2
    assert advice.applied is False


# ------------------------------------------------- observed capacity plans


def test_a_capacity_change_moves_the_autoscaler_decision() -> None:
    """Same pool, same request, same clock: only the cluster moved.

    This is what W3 buys. Without it both passes launch four cells and the two
    that do not fit sit WAITING until ``queue_timeout_s`` reclaims them.
    """

    roomy = advise_launch(4, _reading(_cluster_queue(nominal="32", reserved="0")), _CELL, policy=_ON, now=_NOW)
    tight = advise_launch(4, _reading(_cluster_queue(nominal="32", reserved="24")), _CELL, policy=_ON, now=_NOW)
    full = advise_launch(4, _reading(_cluster_queue(nominal="32", reserved="32")), _CELL, policy=_ON, now=_NOW)

    assert (roomy.allowed_cells, tight.allowed_cells, full.allowed_cells) == (4, 2, 0)
    assert (roomy.applied, tight.applied, full.applied) == (True, True, True)
    assert full.withheld_cells == 4
    assert full.state is ClusterCapacityState.OBSERVED


def test_a_cluster_with_room_to_spare_does_not_inflate_the_request() -> None:
    """The signal caps a launch; it never proposes one."""

    advice = advise_launch(2, _reading(_cluster_queue(nominal="64", reserved="0")), _CELL, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 2
    assert advice.applied is True
    assert "room for all 2 cell(s)" in advice.reason


def test_a_blocked_launch_carries_the_actionable_explanation_with_it() -> None:
    """The autoscaler's refusal and the operator's explanation are one object."""

    advice = advise_launch(1, _reading(_cluster_queue(nominal="32", reserved="32")), _CELL, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 0
    assert advice.explanation is not None
    assert advice.explanation.blocked is True
    assert "raise nominalQuota" in advice.explanation.remedy


def test_a_lower_bound_never_caps_a_launch() -> None:
    """Cohort borrowing depends on queues Eyvan cannot see.

    ``free == 0`` there means "at least zero", which is every possible cluster.
    Capping on it would read an unobservable cluster as a full one.
    """

    advice = advise_launch(4, _reading(_cluster_queue(reserved="32", cohort="shared")), _CELL, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 4
    assert advice.applied is False
    assert "lower bound" in advice.reason


# ------------------------------------------------- freshness


def test_a_stale_reading_is_not_reused_as_a_fresh_one() -> None:
    """Same reading, two clocks. A cluster observed once stays observed once."""

    reading = _reading(_cluster_queue(nominal="32", reserved="32"))

    fresh = advise_launch(4, reading, _CELL, policy=_ON, now=_NOW + timedelta(seconds=30))
    stale = advise_launch(4, reading, _CELL, policy=_ON, now=_NOW + timedelta(seconds=90))

    assert fresh.allowed_cells == 0
    assert stale.allowed_cells == 4
    assert stale.applied is False
    assert "past the 60s bound" in stale.reason


def test_the_tolerated_age_is_the_operators_and_is_honoured() -> None:
    reading = _reading(_cluster_queue(nominal="32", reserved="32"))
    policy = ClusterCapacityPolicy(enabled=True, max_observation_age_s=600)

    assert advise_launch(4, reading, _CELL, policy=policy, now=_NOW + timedelta(seconds=90)).allowed_cells == 0
    assert advise_launch(4, reading, _CELL, policy=policy, now=_NOW + timedelta(seconds=601)).allowed_cells == 4


def test_a_reading_stamped_in_the_future_is_refused_rather_than_immortal() -> None:
    """An unsynchronized clock would otherwise produce a reading that never ages."""

    reading = _reading(_cluster_queue(nominal="32", reserved="32"), observed_at=_NOW + timedelta(hours=1))

    advice = advise_launch(4, reading, _CELL, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 4
    assert advice.applied is False
    assert "future" in advice.reason


def test_a_small_clock_skew_still_counts_as_fresh() -> None:
    """Two hosts a second apart are not evidence of anything being wrong."""

    reading = _reading(_cluster_queue(nominal="32", reserved="32"), observed_at=_NOW + timedelta(seconds=2))

    assert advise_launch(4, reading, _CELL, policy=_ON, now=_NOW).allowed_cells == 0


def test_observation_age_is_measured_against_a_timezone_aware_clock() -> None:
    reading = _reading(_cluster_queue())

    assert observation_age(reading, _NOW + timedelta(seconds=45)) == timedelta(seconds=45)
    with pytest.raises(ValueError):
        observation_age(reading, datetime(2026, 8, 9, 12, 0))


# ------------------------------------------------- rollout safety


def test_the_signal_is_off_by_default_so_an_upgrade_plans_as_it_did_before() -> None:
    """A control plane that upgrades into this code must not resize its fleet."""

    reading = _reading(_cluster_queue(nominal="32", reserved="32"))

    advice = advise_launch(4, reading, _CELL, now=_NOW)

    assert ClusterCapacityPolicy().enabled is False
    assert advice.allowed_cells == 4
    assert advice.applied is False
    assert "not enabled" in advice.reason


def test_a_disabled_policy_is_not_consulted_even_with_a_proven_full_cluster() -> None:
    reading = _reading(_cluster_queue(nominal="32", reserved="32"))

    advice = advise_launch(4, reading, _CELL, policy=ClusterCapacityPolicy(enabled=False), now=_NOW)

    assert advice == ClusterCapacityAdvice(
        requested_cells=4,
        allowed_cells=4,
        state=ClusterCapacityState.UNKNOWN,
        applied=False,
        reason="cluster capacity is not enabled as a plan input",
    )


@pytest.mark.parametrize("age", [0, -1, 3601, True, "60"])
def test_a_freshness_bound_that_could_never_expire_anything_is_refused(age: object) -> None:
    with pytest.raises(ValueError):
        ClusterCapacityPolicy(enabled=True, max_observation_age_s=age)


@pytest.mark.parametrize("requested", [-1, True, 1.5, "2"])
def test_a_nonsensical_launch_count_is_refused_rather_than_coerced(requested: object) -> None:
    with pytest.raises(ValueError):
        advise_launch(requested, _reading(_cluster_queue()), _CELL, policy=_ON, now=_NOW)


def test_zero_requested_cells_stays_zero_without_consulting_the_cluster() -> None:
    advice = advise_launch(0, _reading(_cluster_queue(reserved="32")), _CELL, policy=_ON, now=_NOW)

    assert advice.allowed_cells == 0
    assert advice.withheld_cells == 0
