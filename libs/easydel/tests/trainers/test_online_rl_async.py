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

import threading

import pytest
from easydel.trainers._online_rl.async_rollout import (
    AsyncRolloutCancelled,
    AsyncRolloutWorkerError,
    BoundedAsyncRolloutCoordinator,
    PolicySnapshot,
)


def test_coordinator_returns_completion_order_with_snapshot_version_metadata():
    gates = {"slow": threading.Event(), "fast": threading.Event()}

    def worker(snapshot, payload):
        assert gates[payload].wait(timeout=2.0)
        return f"{snapshot.state}:{payload}"

    snapshot = PolicySnapshot(version=12, state="weights", metadata={"sync_step": 12})
    with BoundedAsyncRolloutCoordinator(worker, max_workers=2, max_outstanding=2) as coordinator:
        slow_ticket = coordinator.submit("slow", snapshot=snapshot, metadata={"prompt_id": 1})
        fast_ticket = coordinator.submit("fast", snapshot=snapshot, metadata={"prompt_id": 2})
        gates["fast"].set()

        first = coordinator.next_completed(timeout=2.0)
        assert first.ticket == fast_ticket
        assert first.value == "weights:fast"
        assert first.policy_version == 12
        assert first.snapshot_metadata["sync_step"] == 12
        assert first.ticket.metadata["prompt_id"] == 2
        assert first.staleness(15) == 3
        assert first.queue_time >= 0
        assert first.execution_time >= 0

        gates["slow"].set()
        second = coordinator.next_completed(timeout=2.0)
        assert second.ticket == slow_ticket
        assert second.value == "weights:slow"
        assert coordinator.outstanding_count == 0


def test_completed_but_unconsumed_result_applies_backpressure():
    finished = threading.Event()

    def worker(_snapshot, payload):
        finished.set()
        return payload

    snapshot = PolicySnapshot(version=0, state=None)
    with BoundedAsyncRolloutCoordinator(worker, max_workers=1, max_outstanding=1) as coordinator:
        coordinator.submit("first", snapshot=snapshot)
        assert finished.wait(timeout=2.0)

        with pytest.raises(TimeoutError, match="capacity"):
            coordinator.submit("second", snapshot=snapshot, timeout=0.02)

        assert coordinator.next_completed(timeout=2.0).value == "first"
        coordinator.submit("second", snapshot=snapshot, timeout=0.1)
        assert coordinator.next_completed(timeout=2.0).value == "second"


def test_completion_timeout_does_not_discard_running_work():
    gate = threading.Event()

    def worker(_snapshot, payload):
        assert gate.wait(timeout=2.0)
        return payload

    snapshot = PolicySnapshot(version=1, state=None)
    with BoundedAsyncRolloutCoordinator(worker) as coordinator:
        coordinator.submit(4, snapshot=snapshot)
        with pytest.raises(TimeoutError, match="waiting"):
            coordinator.next_completed(timeout=0.02)
        assert coordinator.outstanding_count == 1
        gate.set()
        assert coordinator.next_completed(timeout=2.0).value == 4


def test_worker_exception_is_propagated_with_ticket_and_original_cause():
    def worker(_snapshot, _payload):
        raise ValueError("bad rollout")

    snapshot = PolicySnapshot(version=5, state=None)
    with BoundedAsyncRolloutCoordinator(worker) as coordinator:
        ticket = coordinator.submit("request", snapshot=snapshot)
        with pytest.raises(AsyncRolloutWorkerError) as error_info:
            coordinator.next_completed(timeout=2.0)

    assert error_info.value.ticket == ticket
    assert isinstance(error_info.value.__cause__, ValueError)
    assert str(error_info.value.__cause__) == "bad rollout"


def test_queued_rollout_can_be_cancelled_and_consumed_as_completion():
    running_gate = threading.Event()
    running_started = threading.Event()

    def worker(_snapshot, payload):
        if payload == "running":
            running_started.set()
            assert running_gate.wait(timeout=2.0)
        return payload

    snapshot = PolicySnapshot(version=2, state=None)
    with BoundedAsyncRolloutCoordinator(worker, max_workers=1, max_outstanding=2) as coordinator:
        coordinator.submit("running", snapshot=snapshot)
        assert running_started.wait(timeout=2.0)
        cancelled_ticket = coordinator.submit("queued", snapshot=snapshot)
        assert coordinator.cancel(cancelled_ticket)

        with pytest.raises(AsyncRolloutCancelled) as error_info:
            coordinator.next_completed(timeout=2.0)
        assert error_info.value.ticket == cancelled_ticket

        running_gate.set()
        assert coordinator.next_completed(timeout=2.0).value == "running"


def test_close_cancels_pending_work_and_rejects_new_submissions():
    gate = threading.Event()
    started = threading.Event()

    def worker(_snapshot, payload):
        if payload == "running":
            started.set()
            assert gate.wait(timeout=2.0)
        return payload

    snapshot = PolicySnapshot(version=0, state=None)
    coordinator = BoundedAsyncRolloutCoordinator(worker, max_workers=1, max_outstanding=2)
    coordinator.submit("running", snapshot=snapshot)
    assert started.wait(timeout=2.0)
    queued = coordinator.submit("queued", snapshot=snapshot)
    assert coordinator.cancel(queued)
    gate.set()
    coordinator.close(wait=True)

    assert coordinator.closed
    assert coordinator.outstanding_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.submit("late", snapshot=snapshot)
