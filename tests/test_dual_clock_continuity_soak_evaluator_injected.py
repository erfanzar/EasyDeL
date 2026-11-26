# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dual-clock continuity regressions for the soak evaluator (injected clocks)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from test_soak_runner import _inputs, _qualifying_payloads

from eyvan.sdk.commands.soak import execute_soak
from eyvan.soak_evidence import ExternalEvidenceSample, SoakEvidenceStore


class _IndependentClocks:
    def __init__(self, wall_step_s: float = 0.0, *, wall_step_sleep: int = 1) -> None:
        self.elapsed = 0.0
        self.wall_elapsed = 0.0
        self.wall_step_s = wall_step_s
        self.wall_step_sleep = wall_step_sleep
        self.sleeps = 0

    def monotonic(self) -> float:
        return self.elapsed

    def wall(self) -> float:
        return self.wall_elapsed

    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.wall_elapsed)

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds
        self.wall_elapsed += seconds
        self.sleeps += 1
        if self.sleeps == self.wall_step_sleep:
            self.wall_elapsed += self.wall_step_s


def _run_with_fresh_evidence(
    clock: _IndependentClocks,
    *,
    stall_at: str | None = None,
    stall_s: float = 0.0,
    wall_only_stall: bool = False,
) -> dict[str, object]:
    payloads = _qualifying_payloads()
    store = SoakEvidenceStore(monotonic=clock.monotonic, wall=clock.wall)

    def stall() -> None:
        if not wall_only_stall:
            clock.elapsed += stall_s
        clock.wall_elapsed += stall_s

    def checkpoint(tick: int, _state: object) -> None:
        if tick == 2 and stall_at == "checkpoint":
            stall()

    def observe() -> dict[str, object]:
        payload = payloads.pop(0)
        sample = ExternalEvidenceSample.from_dict(payload["snapshot"]["p8_evidence"])
        # A resumed monitor has replaced the stale sample. Its freshness does
        # not prove that the evaluator observed the preceding interval.
        store.record(sample)
        assert store.age_s() == 0.0
        latest = store.latest(proof_id=sample.proof_id)
        assert latest is not None
        payload["snapshot"]["p8_evidence"] = latest.as_dict()
        if not payloads and stall_at == "observe":
            stall()
        return payload

    receipt = execute_soak(
        observe,
        inputs=_inputs(),
        output=None,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=clock.now,
        on_tick=checkpoint,
    )
    assert not payloads
    assert receipt["observation_counts"]["successful"] == 3
    assert receipt["status"] == "completed"
    return receipt


@pytest.mark.parametrize("wall_step_s", [3600.0, 113400.0])
def test_wall_only_suspend_gap_cannot_pass_with_fresh_resumed_evidence(wall_step_s: float) -> None:
    clock = _IndependentClocks(wall_step_s)
    receipt = _run_with_fresh_evidence(clock)

    assert clock.elapsed == 3.0  # The scheduling clock did not include suspension.
    assert receipt["verdict"] == "inconclusive"
    assert receipt["inconclusive_reason_codes"] == ["observation_gap"]
    assert receipt["longest_observation_gap_s"] == wall_step_s + 1.0
    assert receipt["unobserved_s"] == wall_step_s
    assert receipt["actual_duration_s"] == wall_step_s + 3.0


@pytest.mark.parametrize("wall_step_s", [0.0, -3600.0])
def test_uninterrupted_and_backward_wall_step_preserve_monotonic_schedule(wall_step_s: float) -> None:
    clock = _IndependentClocks(wall_step_s)
    receipt = _run_with_fresh_evidence(clock)

    assert clock.elapsed == 3.0
    assert receipt["verdict"] == "pass"
    assert receipt["reasons"] == []
    assert receipt["longest_observation_gap_s"] == 0.0
    assert receipt["unobserved_s"] == 0.0
    assert receipt["actual_duration_s"] == 3.0


@pytest.mark.parametrize("stall_at", ["observe", "checkpoint"])
@pytest.mark.parametrize("wall_only_stall", [False, True])
def test_final_poll_and_checkpoint_stalls_are_observation_gaps(stall_at: str, wall_only_stall: bool) -> None:
    receipt = _run_with_fresh_evidence(
        _IndependentClocks(),
        stall_at=stall_at,
        stall_s=100.0,
        wall_only_stall=wall_only_stall,
    )

    assert receipt["verdict"] == "inconclusive"
    assert receipt["inconclusive_reason_codes"] == ["observation_gap"]
    assert receipt["longest_observation_gap_s"] == (101.0 if wall_only_stall else 100.0)
    assert receipt["unobserved_s"] == (100.0 if wall_only_stall else 99.0)
    assert receipt["actual_duration_s"] == (103.0 if wall_only_stall else 102.0)


@pytest.mark.parametrize("stall_at", ["observe", "checkpoint"])
@pytest.mark.parametrize("stall_s", [0.25, 1.5, 2.0])
def test_bounded_final_poll_and_checkpoint_latency_preserves_pass(stall_at: str, stall_s: float) -> None:
    receipt = _run_with_fresh_evidence(_IndependentClocks(), stall_at=stall_at, stall_s=stall_s)

    assert receipt["verdict"] == "pass"
    assert receipt["longest_observation_gap_s"] == 0.0
    assert receipt["unobserved_s"] == 0.0
    assert receipt["actual_duration_s"] == max(3.0, 2.0 + stall_s)


def test_suspend_during_final_deadline_wait_is_not_hidden() -> None:
    clock = _IndependentClocks(3600.0, wall_step_sleep=3)
    receipt = _run_with_fresh_evidence(clock)

    assert clock.elapsed == 3.0
    assert receipt["verdict"] == "inconclusive"
    assert receipt["inconclusive_reason_codes"] == ["observation_gap"]
    assert receipt["longest_observation_gap_s"] == 3601.0
    assert receipt["unobserved_s"] == 3600.0
    assert receipt["actual_duration_s"] == 3603.0


def test_between_poll_and_terminal_gaps_are_each_counted_once() -> None:
    receipt = _run_with_fresh_evidence(
        _IndependentClocks(3600.0),
        stall_at="checkpoint",
        stall_s=100.0,
    )

    assert receipt["verdict"] == "inconclusive"
    assert receipt["inconclusive_reason_codes"] == ["observation_gap"]
    assert receipt["longest_observation_gap_s"] == 3601.0
    assert receipt["unobserved_s"] == 3600.0 + 99.0
    assert receipt["actual_duration_s"] == 3600.0 + 102.0
