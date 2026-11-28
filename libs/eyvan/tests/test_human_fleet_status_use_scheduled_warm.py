# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Human fleet status must use the scheduled warm floor, not only the static one."""

from datetime import UTC, datetime
from io import StringIO

import pytest
from rich.console import Console

from eyvan.records import CapacityPoolRecord, CellState, ResourceLimits, WarmWindow
from eyvan.sdk.commands import fleet_status
from eyvan.server.api.schemas import FleetStatusResponse


def _pool(**updates: object) -> CapacityPoolRecord:
    values = {
        "id": "research-cpu",
        "fleet_name": "research",
        "node_type": "cpu",
        "provider": "gcp",
        "provider_project": "project-a",
        "region": "us-east5",
        "zone": "us-east5-a",
        "capacity": ResourceLimits(cpu=8),
        "allocatable": ResourceLimits(cpu=8),
        "agent_endpoint": "http://agent:8090",
        "min_cells": 0,
        "max_cells": 3,
        "initialized_at": datetime(2026, 10, 1, tzinfo=UTC),
        "warm_windows": (WarmWindow(min_cells=3, start_hour=9, end_hour=17),),
    }
    values.update(updates)
    return CapacityPoolRecord.model_validate(values)


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, hour: int) -> None:
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 1, hour, tzinfo=UTC)

    # The pre-repair renderer does not read a clock at all.
    monkeypatch.setattr(fleet_status, "datetime", FixedDatetime, raising=False)


def _status(pool: CapacityPoolRecord, ready_workers: int = 0) -> FleetStatusResponse:
    cells = [
        pool.make_cell(f"research-cpu-{index}").model_copy(update={"state": CellState.READY})
        for index in range(ready_workers)
    ]
    return FleetStatusResponse(name="research", pools=[pool], cells=cells)


def _render(status: FleetStatusResponse) -> str:
    stream = StringIO()
    fleet_status.FleetStatusRenderer(Console(file=stream, width=140, color_system=None)).emit(status)
    return stream.getvalue()


@pytest.mark.parametrize("ready_workers", [0, 1])
def test_scheduled_warm_shortfall_is_visible_in_human_output(monkeypatch, ready_workers: int) -> None:
    _freeze_clock(monkeypatch, 12)
    pool = _pool()
    status = _status(pool, ready_workers)
    assert pool.warm_floor(datetime(2026, 10, 1, 12, tzinfo=UTC)) == 3

    output = _render(status)

    assert "under-provisioned" in output
    assert f"{ready_workers} ready / {ready_workers} active / 3 target" in output
    view = fleet_status.FleetStatusView.build(status)
    assert view.pool_target(pool) == 3
    assert view.fleet_state[0] == "under-provisioned"


@pytest.mark.parametrize(
    ("hour", "updates", "target", "state"),
    [
        (18, {}, 0, "idle"),
        (18, {"min_cells": 1}, 1, "under-provisioned"),
        (12, {"enabled": False}, 0, "idle"),
        (12, {"initialized_at": None, "initial_cells": 2}, 2, "under-provisioned"),
    ],
)
def test_scheduled_floor_preserves_static_disabled_and_first_apply_controls(
    monkeypatch, hour: int, updates: dict[str, object], target: int, state: str
) -> None:
    _freeze_clock(monkeypatch, hour)
    pool = _pool(**updates)
    status = _status(pool)
    view = fleet_status.FleetStatusView.build(status)

    assert view.pool_target(pool) == target
    assert view.fleet_state[0] == state
    output = _render(status)
    assert state in output
    if target:
        assert f"0 ready / 0 active / {target} target" in output
    else:
        assert "target" not in output
    if not pool.enabled:
        assert "disabled" in output


def test_one_render_uses_one_timestamp_even_when_the_warm_window_ends(monkeypatch) -> None:
    times = iter([datetime(2026, 10, 1, 16, 59, tzinfo=UTC), datetime(2026, 10, 1, 17, tzinfo=UTC)])
    calls = []

    class MovingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            return next(times)

    monkeypatch.setattr(fleet_status, "datetime", MovingDatetime, raising=False)
    output = _render(_status(_pool()))

    assert "under-provisioned" in output
    assert "0 ready / 0 active / 3 target" in output
    assert calls == [UTC]
