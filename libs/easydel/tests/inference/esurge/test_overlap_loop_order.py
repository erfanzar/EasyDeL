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

"""Characterization: the async-overlap scheduler loop's dispatch/drain ordering.

The overlap loop's throughput contract is that step ``N+1`` is *dispatched*
(``execute_model_async``) before step ``N`` is *drained*
(``wait_for_execution``) whenever the runner reports deferred dispatch is
safe — otherwise every decode token is serialized on a host round trip. Its
correctness contract is that a drain failure after the scheduler has already
advanced (a next step was scheduled and possibly dispatched) must abort the
loop, never retry: retrying with an unexecuted prefetched batch corrupts
scheduler/runner state.

These tests run the real ``EngineLifecycleMixin`` loop against a scripted
scheduler and runner recording call order — no model, no device work. They
lock the loop-shape invariants through the EngineLoop extraction refactor.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from easydel.inference.esurge.mixins.lifecycle import EngineLifecycleMixin

JOIN_TIMEOUT_S = 20.0
POLL_INTERVAL_S = 0.002


def _sched_output(tag: str, *, num_tokens: int, finished: set[str] | None = None, defer_safe: bool = True):
    """A SchedulerOutput-shaped namespace with only the fields the loop reads."""
    return SimpleNamespace(
        tag=tag,
        total_num_scheduled_tokens=num_tokens,
        num_scheduled_tokens={f"{tag}-r{i}": 1 for i in range(num_tokens)},
        finished_req_ids=finished or set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(num_reqs=0),
        async_scheduling=True,
        defer_safe=defer_safe,
    )


def _idle_output():
    return _sched_output("idle", num_tokens=0)


class _FakeRunner:
    def __init__(self, events: list[str], *, fail_drain_tags: set[str] | None = None):
        self.events = events
        self.fail_drain_tags = fail_drain_tags or set()
        self.executor_manager = SimpleNamespace(kv_pages=object())

    def execute_model(self, scheduler_output):
        self.events.append(f"execute_sync:{scheduler_output.tag}")
        return SimpleNamespace(tag=scheduler_output.tag)

    def execute_model_async(self, scheduler_output):
        self.events.append(f"dispatch:{scheduler_output.tag}")
        return SimpleNamespace(tag=scheduler_output.tag)

    def wait_for_execution(self, handle):
        self.events.append(f"drain:{handle.tag}")
        if handle.tag in self.fail_drain_tags:
            raise RuntimeError(f"scripted drain failure for {handle.tag}")
        return SimpleNamespace(tag=handle.tag)

    def can_dispatch_next_before_async_drain(self, scheduler_output):
        return bool(scheduler_output.defer_safe)

    def log_it(self, *args, **kwargs):
        pass


class _FakeScheduler:
    def __init__(self, events: list[str], script: list):
        self.events = events
        self._script = list(script)
        self.script_drained = threading.Event()
        self.running: list = []
        self.waiting: list = []

    def schedule(self):
        if self._script:
            out = self._script.pop(0)
            if not self._script:
                self.script_drained.set()
        else:
            out = _idle_output()
        self.events.append(f"schedule:{out.tag}")
        return out

    def update_from_output(self, scheduler_output, model_output):
        assert scheduler_output.tag == model_output.tag, "update paired with the wrong step's output"
        self.events.append(f"update:{scheduler_output.tag}")
        return {}


class _LoopHarness(EngineLifecycleMixin):
    """Real lifecycle loop over scripted collaborators; periphery stubbed out."""

    def __init__(self, script: list, *, fail_drain_tags: set[str] | None = None):
        self.events: list[str] = []
        self.runner = _FakeRunner(self.events, fail_drain_tags=fail_drain_tags)
        self.scheduler = _FakeScheduler(self.events, script)
        self.runtime_config = SimpleNamespace(
            overlap_execution=True,
            async_scheduling=True,
            runner_verbose=False,
        )
        self._scheduler_lock = threading.RLock()
        self._scheduler_running = False
        self._scheduler_thread = None
        self._profiling_active = False
        self.aborts: list[BaseException] = []

    # --- stubbed periphery (not under test) ---
    def _info(self, *args, **kwargs):
        pass

    def _start_engine_output_worker(self):
        pass

    def _install_signal_diagnostics(self):
        pass

    def _touch_activity(self):
        pass

    def _start_idle_monitor(self):
        pass

    def _update_scheduler_heartbeat(self):
        pass

    def _drain_parser_stop_requests_locked(self):
        pass

    def _enqueue_engine_outputs(self, engine_outputs):
        pass

    def _abort_scheduler_due_to_error(self, exc):
        self.aborts.append(exc)
        self._scheduler_running = False

    # --- test driver ---
    def run_until(self, predicate, timeout: float = JOIN_TIMEOUT_S) -> None:
        self.initiate()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self):
                break
            time.sleep(POLL_INTERVAL_S)
        else:
            self._stop()
            raise AssertionError(f"loop never reached expected state; events={self.events}")
        self._stop()

    def _stop(self) -> None:
        self._scheduler_running = False
        thread = self._scheduler_thread
        if thread is not None:
            thread.join(timeout=JOIN_TIMEOUT_S)
            assert not thread.is_alive(), "scheduler loop thread failed to exit"


def test_overlap_dispatches_next_step_before_draining_previous():
    s1 = _sched_output("S1", num_tokens=2)
    s2 = _sched_output("S2", num_tokens=2)
    harness = _LoopHarness([s1, s2])

    harness.run_until(lambda h: "update:S2" in h.events)

    ev = harness.events
    assert ev.index("dispatch:S2") < ev.index("drain:S1"), (
        f"overlap regressed to drain-before-dispatch (serialized decode): {ev}"
    )
    # CPU bookkeeping still happens in scheduler order, each after its drain.
    assert ev.index("drain:S1") < ev.index("update:S1") < ev.index("update:S2")
    assert ev.index("drain:S2") < ev.index("update:S2")
    assert not harness.aborts


def test_drain_failure_after_scheduler_advanced_aborts_instead_of_retrying():
    s1 = _sched_output("S1", num_tokens=2)
    s2 = _sched_output("S2", num_tokens=2)
    harness = _LoopHarness([s1, s2], fail_drain_tags={"S1"})

    harness.run_until(lambda h: h.aborts)

    ev = harness.events
    # The abort must come from the advanced-state path: S2 was already dispatched.
    assert "dispatch:S2" in ev
    assert len(harness.aborts) == 1
    # Neither step's bookkeeping may run after the failed drain: S1's output
    # never materialized and S2 must not be applied out of order.
    assert "update:S1" not in ev
    assert "update:S2" not in ev


def test_zero_token_bookkeeping_schedule_executes_synchronously_after_drain():
    s1 = _sched_output("S1", num_tokens=2)
    z = _sched_output("Z", num_tokens=0, finished={"S1-r0"})
    harness = _LoopHarness([s1, z])

    harness.run_until(lambda h: "update:Z" in h.events)

    ev = harness.events
    # Zero-token housekeeping still flows through the runner synchronously,
    # and only after the in-flight step has been drained and applied.
    assert ev.index("drain:S1") < ev.index("execute_sync:Z") < ev.index("update:Z")
    assert ev.index("update:S1") < ev.index("update:Z")
    assert not harness.aborts


def test_non_defer_safe_schedule_falls_back_to_synchronous_execution():
    s1 = _sched_output("S1", num_tokens=8, defer_safe=False)  # prefill-shaped
    harness = _LoopHarness([s1])

    harness.run_until(lambda h: "update:S1" in h.events)

    ev = harness.events
    assert "execute_sync:S1" in ev, f"non-defer-safe step must run synchronously: {ev}"
    assert "dispatch:S1" not in ev
    assert not harness.aborts
