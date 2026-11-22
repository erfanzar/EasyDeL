# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deadline admission and exactly-once credit accounting for async rollouts (threads only)."""

from __future__ import annotations

import threading

import pytest
from easydel.trainers._online_rl.async_rollout import (
    AsyncRolloutAdmissionDeferred,
    AsyncRolloutCancelled,
    AsyncRolloutWorkerError,
    BoundedAsyncRolloutCoordinator,
    PolicySnapshot,
    admission_is_feasible,
    min_admissible_version,
)
from easydel.trainers.async_rollout_mixin import AsyncRolloutMixin, with_legacy_async_grpo_aliases

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "deadlines,head,groups,expected",
    [
        ([], 0, 1, True),
        ([0], 0, 1, True),
        ([0, 0], 0, 1, False),  # second rollout would be consumed at version 1 > deadline 0
        ([0, 1], 0, 1, True),
        ([1, 0], 0, 1, True),  # order of submission does not matter
        ([-1], 0, 1, False),  # already past its deadline
        ([1, 1], 0, 1, True),
        ([1, 1, 1], 0, 1, False),
        ([0, 0], 0, 2, True),
        ([0, 0, 0], 0, 2, False),
        ([0, 0, 1, 1], 0, 2, True),
        ([5, 5, 6], 5, 1, False),
        ([5, 6, 7], 5, 1, True),
    ],
)
def test_admission_feasibility_hand_cases(deadlines, head, groups, expected):
    assert admission_is_feasible(deadlines, head_version=head, groups_per_update=groups) is expected


def test_admission_rejects_non_positive_groups():
    with pytest.raises(ValueError, match="groups_per_update"):
        admission_is_feasible([0], head_version=0, groups_per_update=0)


@pytest.mark.parametrize(
    "outstanding,head,staleness,groups,expected",
    [
        ([], 0, 0, 1, 0),
        ([1, 1], 0, 1, 1, 1),  # v0 would need three slots by v1; at v1 one is consumed and [1, 2] fits
        ([0], 0, 0, 1, 1),
        ([0, 0], 0, 0, 2, 1),
        ([3, 3, 3], 0, 3, 1, 0),  # [3, 3, 3, 3] fits four slots 0..3
    ],
)
def test_min_admissible_version_hand_cases(outstanding, head, staleness, groups, expected):
    assert (
        min_admissible_version(outstanding, head_version=head, max_staleness=staleness, groups_per_update=groups)
        == expected
    )


def _snapshot(version):
    return PolicySnapshot(version=version, state=None)


def test_infeasible_submission_is_not_started_and_reports_retry_version():
    started = []

    def worker(_snapshot, payload):
        started.append(payload)
        return payload

    with BoundedAsyncRolloutCoordinator(worker, max_workers=1, max_outstanding=4, max_staleness=0) as coordinator:
        first = coordinator.submit("a", snapshot=_snapshot(0))
        assert first.deadline_version == 0
        with pytest.raises(AsyncRolloutAdmissionDeferred) as info:
            coordinator.submit("b", snapshot=_snapshot(0))
        assert info.value.retry_at_version == 1
        assert info.value.deadline_version == 0
        assert coordinator.admission_deferred_count == 1
        assert coordinator.credits_in_use == 1
        assert coordinator.next_completed(timeout=2.0).value == "a"

        coordinator.advance_head(1)
        coordinator.advance_head(0)  # never moves backwards
        assert coordinator.head_version == 1
        assert coordinator.submit("b", snapshot=_snapshot(1)).deadline_version == 1
        assert coordinator.next_completed(timeout=2.0).value == "b"
    assert started == ["a", "b"]


def test_without_max_staleness_admission_is_disabled():
    with BoundedAsyncRolloutCoordinator(lambda _s, p: p, max_workers=1, max_outstanding=3) as coordinator:
        tickets = [coordinator.submit(i, snapshot=_snapshot(0)) for i in range(3)]
        assert all(ticket.deadline_version is None for ticket in tickets)
        assert sorted(coordinator.next_completed(timeout=2.0).value for _ in range(3)) == [0, 1, 2]


def _drain_capacity(coordinator, count):
    """Full capacity is available again iff ``count`` immediate submissions succeed."""
    for index in range(count):
        coordinator.submit(index, snapshot=_snapshot(0), timeout=0.5)
    with pytest.raises(TimeoutError):
        coordinator.submit("over", snapshot=_snapshot(0), timeout=0.02)


def test_credit_released_once_for_cancelled_and_failed_rollouts():
    gate = threading.Event()
    started = threading.Event()

    def worker(_snapshot, payload):
        if payload == "running":
            started.set()
            assert gate.wait(2.0)
        if payload == "boom":
            raise ValueError("bad")
        return payload

    coordinator = BoundedAsyncRolloutCoordinator(worker, max_workers=1, max_outstanding=3)
    try:
        coordinator.submit("running", snapshot=_snapshot(0))
        assert started.wait(2.0)
        cancelled = coordinator.submit("queued", snapshot=_snapshot(0))
        coordinator.submit("boom", snapshot=_snapshot(0))
        assert coordinator.cancel(cancelled)
        assert coordinator.credits_in_use == 3

        with pytest.raises(AsyncRolloutCancelled):
            coordinator.next_completed(timeout=2.0)
        assert coordinator.credits_in_use == 2
        assert coordinator.cancel(cancelled) is False  # settled; cancelling again is a no-op
        gate.set()
        outcomes = []
        for _ in range(2):
            try:
                outcomes.append(coordinator.next_completed(timeout=2.0).value)
            except AsyncRolloutWorkerError as error:
                outcomes.append(type(error.__cause__).__name__)
        assert sorted(outcomes) == ["ValueError", "running"]
        assert coordinator.credits_in_use == 0
        assert coordinator.outstanding_count == 0
        # BoundedSemaphore raises on over-release, so a double release would already have failed.
        _drain_capacity(coordinator, 3)
    finally:
        gate.set()
        coordinator.close(wait=True)


def test_close_settles_unconsumed_rollouts_once_and_is_idempotent():
    gate = threading.Event()
    started = threading.Event()

    def worker(_snapshot, payload):
        if payload == "running":
            started.set()
            assert gate.wait(2.0)
        return payload

    coordinator = BoundedAsyncRolloutCoordinator(worker, max_workers=1, max_outstanding=3)
    coordinator.submit("running", snapshot=_snapshot(0))
    assert started.wait(2.0)
    coordinator.submit("queued-1", snapshot=_snapshot(0))
    coordinator.submit("queued-2", snapshot=_snapshot(0))
    idle = threading.Event()
    coordinator.close(wait=False, cancel_pending=True, on_idle=idle.set)
    assert coordinator.credits_in_use == 0
    assert coordinator.outstanding_count == 0
    gate.set()
    assert idle.wait(2.0)  # the running worker finishing after close must not re-publish or re-release
    coordinator.close(wait=True)
    assert coordinator.credits_in_use == 0
    with pytest.raises(RuntimeError, match="no outstanding"):
        coordinator.next_completed(timeout=0.1)


def test_close_wakes_blocked_consumer():
    gate = threading.Event()
    coordinator = BoundedAsyncRolloutCoordinator(lambda _s, p: gate.wait(2.0) and p, max_workers=1)
    coordinator.submit("slow", snapshot=_snapshot(0))
    errors = []

    def consume():
        try:
            coordinator.next_completed(timeout=5.0)
        except RuntimeError as exc:
            errors.append(str(exc))

    consumer = threading.Thread(target=consume)
    consumer.start()
    closer = threading.Timer(0.1, lambda: coordinator.close(wait=False))
    closer.start()
    consumer.join(3.0)
    gate.set()
    closer.join()
    assert not consumer.is_alive()
    assert errors and "closed" in errors[0]
    assert coordinator.credits_in_use == 0


def test_legacy_async_grpo_aliases_mirror_async_rollout_keys():
    metrics = {"async_rollout/stale_dropped": 2, "async_rollout/admission_deferred": 1, "loss": 0.5}
    aliased = with_legacy_async_grpo_aliases(metrics)
    assert aliased == {
        **metrics,
        "async_grpo/stale_dropped": 2,
        "async_grpo/admission_deferred": 1,
    }
    assert AsyncRolloutMixin.emit_legacy_async_grpo_metrics is False


def test_force_old_logps_flag_is_thread_local():
    trainer = object.__new__(AsyncRolloutMixin)
    seen_in_worker = []
    worker_set = threading.Event()
    main_checked = threading.Event()

    def worker():
        trainer._async_grpo_force_old_logps = True
        seen_in_worker.append(trainer._async_grpo_force_old_logps)
        worker_set.set()
        assert main_checked.wait(2.0)

    thread = threading.Thread(target=worker)
    thread.start()
    assert worker_set.wait(2.0)
    assert trainer._async_grpo_force_old_logps is False
    main_checked.set()
    thread.join(2.0)
    assert seen_in_worker == [True]
