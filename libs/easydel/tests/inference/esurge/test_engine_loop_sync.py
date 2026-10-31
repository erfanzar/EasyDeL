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

"""Behavior of the synchronous scheduler loop: profiler ticking and clean DP shutdown."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.engine.loop import EngineLoop
from easydel.inference.esurge.scheduler.dp_scheduler import DPSchedulerClosed

pytestmark = pytest.mark.cpu_ok


class _Coordinator:
    def check_health(self) -> None:
        return None

    def execute_sync(self, scheduler_output):
        return SimpleNamespace()


def _make_loop(scheduler, *, handle_profiling_step, on_fatal):
    return EngineLoop(
        scheduler=scheduler,
        runner=SimpleNamespace(),
        coordinator=_Coordinator(),
        scheduler_lock=threading.RLock(),
        emit_outputs=lambda outputs: None,
        drain_parser_stops=lambda: None,
        on_fatal=on_fatal,
        heartbeat=lambda: None,
        handle_profiling_step=handle_profiling_step,
        is_nonrecoverable=lambda exc: False,
        overlap_execution=False,
        async_scheduling=False,
        runner_verbose=False,
        info=lambda *args, **kwargs: None,
    )


class _Scheduler:
    def __init__(self, raise_exc: BaseException | None = None):
        self.raise_exc = raise_exc
        self.schedule_calls = 0
        self.running: list = []
        self.waiting: list = []

    def schedule(self):
        self.schedule_calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"r": 1})

    def update_from_output(self, scheduler_output, model_output):
        return {}


def test_sync_loop_ticks_profiler_step_budget_each_executed_step():
    ticks: list[int] = []
    holder: dict[str, EngineLoop] = {}

    def handle_profiling_step() -> None:
        ticks.append(1)
        if len(ticks) >= 3:
            holder["loop"].request_stop()

    loop = _make_loop(_Scheduler(), handle_profiling_step=handle_profiling_step, on_fatal=lambda exc: None)
    holder["loop"] = loop
    loop._running = True
    loop._run_sync()
    assert len(ticks) == 3


def test_sync_loop_stops_quietly_on_dp_scheduler_closed():
    fatal: list[BaseException] = []
    scheduler = _Scheduler(raise_exc=DPSchedulerClosed("shut down"))
    loop = _make_loop(scheduler, handle_profiling_step=lambda: None, on_fatal=fatal.append)
    loop._running = True
    loop._run_sync()
    assert scheduler.schedule_calls == 1
    assert fatal == []
