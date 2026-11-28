# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A soak cannot claim invariants held over hours it did not watch.

The duration is measured on the monotonic clock, which is immune to NTP steps --
and on macOS also stops while the machine sleeps. A laptop that suspends for
thirty-one hours still completes every observation slot, so the run reported a
full-length soak over a period it spent a third of not observing anything. The
gap was invisible: the receipt carried no wall-clock evidence at all.
"""

from __future__ import annotations

import typing as tp

from eyvan.sdk.commands.soak import SoakState, run_soak_for_duration


def _clock(offsets: list[float]) -> tp.Callable[[], float]:
    """A monotonic clock that returns each offset in turn, then holds the last."""

    remaining = list(offsets)
    held = [0.0]

    def monotonic() -> float:
        if remaining:
            held[0] = remaining.pop(0)
        return held[0]

    return monotonic


def _healthy() -> dict[str, object]:
    return {"violations": [], "artifact_bytes": 0}


def test_an_uninterrupted_run_reports_no_gap() -> None:
    """The control: ordinary jitter must not read as an unobserved stretch."""

    state = run_soak_for_duration(
        _healthy,
        duration_s=5,
        interval_s=1,
        sleep=lambda _seconds: None,
        monotonic=_clock([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]),
    )

    assert state.longest_observation_gap_s == 0.0
    assert state.unobserved_s == 0.0


def test_a_suspended_machine_is_recorded_rather_than_averaged_away() -> None:
    """The clock stopping is exactly the case a slot count cannot detect."""

    # Second observation lands 40 intervals later: the machine slept.
    state = run_soak_for_duration(
        _healthy,
        duration_s=100,
        interval_s=1,
        sleep=lambda _seconds: None,
        monotonic=_clock([0.0, 0.0, 1.0, 40.0, 40.0, 41.0, 41.0, 42.0, 100.0]),
    )

    assert state.longest_observation_gap_s >= 39.0
    assert state.unobserved_s >= 38.0


def test_an_observation_gap_makes_the_verdict_inconclusive_not_a_pass() -> None:
    """Nothing was seen to be wrong -- and nothing was seen to be right."""

    from eyvan.sdk.commands.soak import REASON_CODES

    assert "observation_gap" in REASON_CODES


def test_the_report_carries_the_gap_so_a_reader_can_see_it() -> None:
    """A receipt that cannot express the hole cannot be audited for it."""

    state = SoakState()
    state.longest_observation_gap_s = 113_400.0
    state.unobserved_s = 113_340.0

    report = state.report(None)

    assert report["longest_observation_gap_s"] == 113_400.0
    assert report["unobserved_s"] == 113_340.0
