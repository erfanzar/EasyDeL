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

"""Real-ZMQ loopback tests for the multi-host step-coordination plane.

A leader and a worker coordinator talk over ``tcp://127.0.0.1`` inside one
process, with mock runners recording their invocation sequences. These lock
the plane's core contracts: the worker replays the leader's exact call
stream (the correctness primitive), failures propagate as
StepCoordinationError instead of hangs, divergence is detected via
num_reqs/digest cross-checks, and unauthenticated peers are rejected before
any payload is unpickled.
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import pytest

from easydel.inference.esurge.distributed.coordinator import (
    LocalCoordinator,
    StepCoordinationError,
    StepHandle,
)
from easydel.inference.esurge.distributed.zmq_coordinator import (
    ZmqLeaderCoordinator,
    ZmqWorkerCoordinator,
)

AUTH = "test-token"
FINGERPRINT = "test-fingerprint"
DEADLINE_S = 20.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sched_output(tag: str):
    return SimpleNamespace(tag=tag, total_num_scheduled_tokens=1, num_scheduled_tokens={f"{tag}-r0": 1})


class _MockRunner:
    """Records the exact invocation stream; emits configurable outputs."""

    def __init__(self, *, sampled_token: int = 42, fail_on: str | None = None):
        self.calls: list[str] = []
        self.sampled_token = sampled_token
        self.fail_on = fail_on
        self._lock = threading.Lock()

    def _record(self, entry: str) -> None:
        with self._lock:
            self.calls.append(entry)

    def _output(self, scheduler_output):
        return SimpleNamespace(
            tag=scheduler_output.tag,
            req_ids=list(scheduler_output.num_scheduled_tokens),
            sampled_token_ids=[[self.sampled_token] for _ in scheduler_output.num_scheduled_tokens],
        )

    def execute_model(self, scheduler_output):
        self._record(f"sync:{scheduler_output.tag}")
        if self.fail_on == scheduler_output.tag:
            raise RuntimeError(f"scripted failure at {scheduler_output.tag}")
        return self._output(scheduler_output)

    def execute_model_async(self, scheduler_output):
        self._record(f"dispatch:{scheduler_output.tag}")
        return SimpleNamespace(scheduler_output=scheduler_output)

    def wait_for_execution(self, handle):
        self._record(f"drain:{handle.scheduler_output.tag}")
        return self._output(handle.scheduler_output)


def _wait_until(predicate, timeout: float = DEADLINE_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached in time")


class _Plane:
    """One leader + one worker over loopback, torn down deterministically."""

    def __init__(self, *, worker_runner=None, leader_runner=None, verify_digest_interval: int = 1):
        port = _free_port()
        self.leader_runner = leader_runner or _MockRunner()
        self.worker_runner = worker_runner or _MockRunner()
        self.leader = ZmqLeaderCoordinator(
            self.leader_runner,
            world_size=2,
            bind_host="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            ready_timeout_s=DEADLINE_S,
            step_timeout_s=5.0,
            heartbeat_interval_s=0.2,
            heartbeat_timeout_s=DEADLINE_S,
            verify_digest_interval=verify_digest_interval,
            max_inflight_steps=4,
        )
        self.worker = ZmqWorkerCoordinator(
            self.worker_runner,
            rank=1,
            world_size=2,
            leader_addr="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            connect_timeout_s=DEADLINE_S,
            heartbeat_interval_s=0.2,
            heartbeat_timeout_s=DEADLINE_S,
        )
        self.stop_event = threading.Event()
        self.worker_error: list[BaseException] = []

        def _worker_body():
            try:
                self.worker.start()
                self.worker.run_worker_loop(self.stop_event)
            except BaseException as exc:
                self.worker_error.append(exc)

        self.worker_thread = threading.Thread(target=_worker_body, daemon=True)

    def __enter__(self):
        self.worker_thread.start()
        self.leader.start()
        return self

    def __exit__(self, *exc_info):
        self.leader.shutdown("test teardown")
        self.stop_event.set()
        self.worker_thread.join(timeout=DEADLINE_S)
        self.worker.shutdown()
        assert not self.worker_thread.is_alive(), "worker replay thread failed to exit"


def test_worker_replays_leader_stream_in_order():
    with _Plane() as plane:
        s1, s2, s3 = _sched_output("S1"), _sched_output("S2"), _sched_output("S3")
        out = plane.leader.execute_sync(s1)
        assert out.tag == "S1"
        handle = plane.leader.execute_async(s2)
        assert isinstance(handle, StepHandle)
        plane.leader.execute_sync(s3)
        drained = plane.leader.drain(handle)
        assert drained.tag == "S2"

        _wait_until(lambda: len(plane.worker_runner.calls) >= 4)
        assert plane.worker_runner.calls == ["sync:S1", "dispatch:S2", "sync:S3", "drain:S2"]

        # deferred acks + matching outputs keep the plane healthy
        _wait_until(lambda: plane.leader._ledger.min_acked() >= 2)
        plane.leader.check_health()
        assert not plane.worker_error


def test_worker_failure_propagates_as_coordination_error():
    with _Plane(worker_runner=_MockRunner(fail_on="BAD")) as plane:
        plane.leader.execute_sync(_sched_output("S1"))

        def _sees_failure():
            try:
                plane.leader.check_health()
            except StepCoordinationError:
                return True
            return False

        plane.leader.execute_sync(_sched_output("BAD"))
        _wait_until(_sees_failure)
        with pytest.raises(StepCoordinationError, match="scripted failure at BAD"):
            plane.leader.check_health()
        _wait_until(lambda: bool(plane.worker_error))
        assert isinstance(plane.worker_error[0], RuntimeError)


def test_sampled_token_divergence_is_detected():
    with _Plane(worker_runner=_MockRunner(sampled_token=7)) as plane:
        plane.leader.execute_sync(_sched_output("S1"))

        def _sees_failure():
            try:
                plane.leader.check_health()
            except StepCoordinationError:
                return True
            return False

        _wait_until(_sees_failure)
        with pytest.raises(StepCoordinationError, match="digest mismatch"):
            plane.leader.check_health()


def test_bad_auth_worker_never_becomes_ready():
    port = _free_port()
    leader_runner = _MockRunner()
    leader = ZmqLeaderCoordinator(
        leader_runner,
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=1.5,
        step_timeout_s=2.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    intruder = ZmqWorkerCoordinator(
        _MockRunner(),
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token="wrong-token",
        config_fingerprint=FINGERPRINT,
        connect_timeout_s=1.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )

    def _intruder_body():
        try:
            intruder.start()
        except StepCoordinationError:
            pass

    thread = threading.Thread(target=_intruder_body, daemon=True)
    thread.start()
    with pytest.raises(StepCoordinationError, match="timed out"):
        leader.start()
    leader.shutdown()
    thread.join(timeout=DEADLINE_S)
    intruder.shutdown()


def test_config_fingerprint_mismatch_fails_the_handshake():
    port = _free_port()
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint="leader-fp",
        ready_timeout_s=DEADLINE_S,
        step_timeout_s=2.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    worker = ZmqWorkerCoordinator(
        _MockRunner(),
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint="worker-fp",
        connect_timeout_s=DEADLINE_S,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    thread = threading.Thread(target=lambda: worker.start(), daemon=True)
    thread.start()
    with pytest.raises(StepCoordinationError, match="config mismatch"):
        leader.start()
    leader.shutdown()
    thread.join(timeout=DEADLINE_S)
    worker.shutdown()


def test_local_coordinator_is_identity_passthrough():
    runner = _MockRunner()
    local = LocalCoordinator(runner)
    local.start()
    out = local.execute_sync(_sched_output("A"))
    assert out.tag == "A"
    handle = local.execute_async(_sched_output("B"))
    assert local.drain(handle).tag == "B"
    local.check_health()
    assert runner.calls == ["sync:A", "dispatch:B", "drain:B"]
    with pytest.raises(StepCoordinationError):
        local.run_worker_loop(threading.Event())
