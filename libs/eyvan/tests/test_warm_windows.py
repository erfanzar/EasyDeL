# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scheduled warm minimums.

Scale-to-zero is correct overnight and wrong at 09:00, when every user pays the
full provisioning latency simultaneously. A recurring window raises the floor for
the hours that matter without predicting anything -- which is the line this feature
deliberately does not cross.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eyvan.records import CapacityPoolRecord, ResourceLimits, WarmWindow

UTC = UTC


def _pool(**updates) -> CapacityPoolRecord:
    base = CapacityPoolRecord(
        id="research/cpu",
        fleet_name="research",
        node_type="cpu",
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceLimits(cpu=8),
        allocatable=ResourceLimits(cpu=8),
        agent_endpoint="http://agent:8090",
        min_cells=0,
        max_cells=10,
    )
    return base.model_copy(update=updates)


def _at(hour: int, *, day: int = 1) -> datetime:
    """Return a UTC moment on a chosen ISO weekday (2026-01-05 is a Monday)."""

    return datetime(2026, 1, 4 + day, hour, 0, tzinfo=UTC)


def test_a_window_raises_the_floor_only_while_it_applies() -> None:
    pool = _pool(warm_windows=(WarmWindow(min_cells=4, start_hour=9, end_hour=17),))

    assert pool.warm_floor(_at(8)) == 0
    assert pool.warm_floor(_at(9)) == 4
    assert pool.warm_floor(_at(16)) == 4
    # Half-open: 17:00 is outside, so adjacent windows never double-count an hour.
    assert pool.warm_floor(_at(17)) == 0


def test_a_pool_without_windows_keeps_its_plain_minimum() -> None:
    assert _pool(min_cells=2).warm_floor(_at(3)) == 2


def test_a_window_never_lowers_the_floor_below_min_cells() -> None:
    """A window is a *raise*; an operator's baseline minimum still holds."""

    pool = _pool(min_cells=3, warm_windows=(WarmWindow(min_cells=1, start_hour=9, end_hour=17),))

    assert pool.warm_floor(_at(12)) == 3
    assert pool.warm_floor(_at(2)) == 3


def test_overlapping_windows_compose_to_the_largest_regardless_of_order() -> None:
    """The highest match wins, not the first, so declaration order cannot matter."""

    business = WarmWindow(min_cells=2, start_hour=9, end_hour=17)
    conference = WarmWindow(min_cells=8, start_hour=9, end_hour=12)

    assert _pool(warm_windows=(business, conference)).warm_floor(_at(10)) == 8
    assert _pool(warm_windows=(conference, business)).warm_floor(_at(10)) == 8
    assert _pool(warm_windows=(business, conference)).warm_floor(_at(14)) == 2


def test_a_window_can_be_limited_to_chosen_weekdays() -> None:
    pool = _pool(warm_windows=(WarmWindow(min_cells=5, start_hour=9, end_hour=17, days=(1, 2, 3, 4, 5)),))

    assert pool.warm_floor(_at(12, day=1)) == 5  # Monday
    assert pool.warm_floor(_at(12, day=5)) == 5  # Friday
    assert pool.warm_floor(_at(12, day=6)) == 0  # Saturday


def test_a_window_can_wrap_past_midnight() -> None:
    """An overnight batch window is a real shape and must not need two entries."""

    pool = _pool(warm_windows=(WarmWindow(min_cells=6, start_hour=22, end_hour=4),))

    assert pool.warm_floor(_at(23)) == 6
    assert pool.warm_floor(_at(2)) == 6
    assert pool.warm_floor(_at(12)) == 0


def test_a_utc_offset_shifts_the_window_without_a_timezone_database() -> None:
    """Fixed offset, stated as such: Eyvan ships no tz database.

    A DST-aware window it could not honour would be worse than one whose limits are
    documented -- this shifts by an hour twice a year, visibly and adjustably.
    """

    pool = _pool(warm_windows=(WarmWindow(min_cells=4, start_hour=9, end_hour=17, utc_offset_minutes=-480),))

    # 17:00 UTC is 09:00 in UTC-8.
    assert pool.warm_floor(_at(17)) == 4
    assert pool.warm_floor(_at(9)) == 0


def test_a_zero_length_window_is_rejected_rather_than_silently_inert() -> None:
    with pytest.raises(ValueError, match="must differ"):
        WarmWindow(min_cells=4, start_hour=9, end_hour=9)


@pytest.mark.parametrize("days", [(0,), (8,), (1, 1)])
def test_malformed_weekdays_are_rejected(days: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        WarmWindow(min_cells=1, start_hour=9, end_hour=17, days=days)


def test_the_planner_uses_the_window_floor_when_it_scales(monkeypatch) -> None:
    """The floor has to reach the planner, not merely exist on the record."""

    from eyvan.capacity.planning import CapacityPlanner, PoolDemand

    pool = _pool(
        initialized_at=datetime(2026, 1, 1, tzinfo=UTC),
        warm_windows=(WarmWindow(min_cells=3, start_hour=9, end_hour=17),),
    )

    idle = {pool.id: PoolDemand(real_cell_ids=frozenset(), virtual_worker_counts=(), virtual_workload_ids=())}
    inside = CapacityPlanner(pools=(pool,), cells_by_pool={pool.id: []}, now=_at(12))
    outside = CapacityPlanner(pools=(pool,), cells_by_pool={pool.id: []}, now=_at(3))

    # No pending demand at all: the only thing that can raise the target is the window.
    assert inside.plan(idle)[pool.id].target_worker_units == 3
    assert outside.plan(idle)[pool.id].target_worker_units == 0
