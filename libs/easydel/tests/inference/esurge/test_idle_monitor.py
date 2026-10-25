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

"""Regression test: an idle reset (stop -> start from the watchdog thread) must not leak watchdog threads."""

from __future__ import annotations

import threading
import time

import pytest
from easydel.inference.esurge.engine.idle import IdleMonitor

pytestmark = pytest.mark.cpu_ok


def test_idle_reset_does_not_leak_watchdog_threads():
    baseline = set(threading.enumerate())
    live_at_reset: list[int] = []
    holder: dict[str, IdleMonitor] = {}

    def on_idle_reset() -> None:
        live_at_reset.append(len([t for t in threading.enumerate() if t not in baseline and t.is_alive()]))
        # What eSurge does on an idle reset: pause() (terminate -> stop) then resume() (start).
        holder["monitor"].stop()
        if len(live_at_reset) < 3:
            holder["monitor"].start()

    monitor = IdleMonitor(
        on_idle_reset=on_idle_reset,
        idle_reset_seconds=0.2,
        min_interval=0.0,
        is_running=lambda: True,
        is_busy=lambda: False,
        info=lambda *args, **kwargs: None,
    )
    holder["monitor"] = monitor
    monitor.start()
    deadline = time.time() + 15.0
    while len(live_at_reset) < 3 and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.6)  # let the last callback finish and every stopped thread exit
    leaked = [t for t in threading.enumerate() if t not in baseline and t.is_alive()]
    monitor.stop()

    assert len(live_at_reset) == 3, "idle reset should fire exactly three times"
    # Exactly one watchdog thread is alive whenever a reset begins.
    assert live_at_reset == [1, 1, 1]
    assert leaked == []
