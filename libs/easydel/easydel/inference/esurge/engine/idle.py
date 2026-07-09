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

"""Idle-reset watchdog for the eSurge engine.

:class:`IdleMonitor` owns the daemon thread that frees KV pages and clears
runner buffers after a configurable inactivity window, so long-lived idle
engines don't keep HBM occupied. The engine injects its reset action
(``pause`` → ``resume``), the "is the scheduler running" gate, and the
"is there active work" guard.
"""

from __future__ import annotations

import threading
import time
import typing

from ..logger import logger


class IdleMonitor:
    """Watchdog thread that triggers an engine reset after long idle periods.

    Args:
        on_idle_reset: The engine's reset action (``pause`` followed by
            ``resume``); exceptions raised inside are logged, not raised.
        idle_reset_seconds: Inactivity window in seconds; ``None`` disables
            the watchdog entirely (``touch``/``start`` become no-ops).
        min_interval: Minimum spacing in seconds between two idle resets.
        is_running: Guard returning whether the scheduler loop is running;
            resets never fire while it reports ``False``.
        is_busy: Guard returning whether the engine has active work
            (running, pending, or tracked requests); busy polls re-arm the
            activity timestamp instead of firing a reset.
        info: Engine-level info logger callable.
    """

    def __init__(
        self,
        *,
        on_idle_reset: typing.Callable[[], None],
        idle_reset_seconds: float | None,
        min_interval: float,
        is_running: typing.Callable[[], bool],
        is_busy: typing.Callable[[], bool],
        info: typing.Callable[..., None],
    ) -> None:
        self._on_idle_reset = on_idle_reset
        self._idle_reset_seconds = idle_reset_seconds
        self._min_interval = min_interval
        self._is_running = is_running
        self._is_busy = is_busy
        self._info = info

        self._idle_reset_last_activity = time.time()
        self._idle_reset_last_reset = 0.0
        self._idle_monitor_event = threading.Event()
        self._idle_monitor_thread: threading.Thread | None = None

    def touch(self) -> None:
        """Mark the engine as active so the idle-reset watchdog re-arms.

        Call sites cover every code path that admits work to the
        scheduler (request-add, lifecycle initiate, multimodal chat
        start). When ``idle_reset_seconds`` is ``None`` the watchdog is
        disabled and this method is a no-op; otherwise
        ``_idle_reset_last_activity`` is bumped to ``time.time()`` so
        the daemon thread in :meth:`start` defers its next check.
        """
        if self._idle_reset_seconds is None:
            return
        self._idle_reset_last_activity = time.time()

    def start(self) -> None:
        """Spawn the idle-reset daemon (or no-op when disabled / already running).

        Idle reset frees KV pages and clears runner buffers after a
        configurable inactivity window so long-lived idle engines don't
        keep HBM occupied. The daemon polls
        ``_idle_reset_last_activity`` at ``idle_reset_seconds / 4`` (capped
        to 1s and floored at 0.1s) and triggers the injected
        ``on_idle_reset`` action when:

        * the scheduler is currently running,
        * idle time exceeds ``idle_reset_seconds``,
        * at least ``min_interval`` seconds have passed since the
          previous reset, and
        * no requests are running, pending, or active.

        Resets observed mid-poll re-arm the activity timestamp without
        firing a reset. Idempotent — if the watchdog thread is already
        alive, nothing is created.
        """
        if self._idle_reset_seconds is None:
            return
        if self._idle_monitor_thread and self._idle_monitor_thread.is_alive():
            return
        self._idle_monitor_event.clear()

        check_interval = min(1.0, max(self._idle_reset_seconds / 4.0, 0.1))

        def _idle_loop() -> None:
            """Background daemon that resets the engine after long idle periods.

            Wakes on a short interval, exiting when ``_idle_monitor_event``
            is set. When the engine has been quiet for at least
            ``idle_reset_seconds`` *and* the cooldown gate ``min_interval``
            has passed *and* no requests are in flight, invokes the
            injected ``on_idle_reset`` action to drain runner / cache /
            pipeline state. Resumed activity (running, pending, or active
            requests) bumps the activity stamp and the gate restarts.
            Exceptions during the reset are logged without killing the
            daemon.
            """
            while not self._idle_monitor_event.wait(check_interval):
                if not self._is_running():
                    continue
                now = time.time()
                if self._idle_reset_seconds is None:
                    continue
                idle_for = now - self._idle_reset_last_activity
                if idle_for < self._idle_reset_seconds:
                    continue
                if now - self._idle_reset_last_reset < self._min_interval:
                    continue
                if self._is_busy():
                    # Activity resumed while waiting.
                    self._idle_reset_last_activity = now
                    continue
                self._idle_reset_last_reset = now
                self._idle_reset_last_activity = now
                self._info("Idle reset triggered after %.1fs of inactivity", idle_for)
                try:
                    self._on_idle_reset()
                except Exception:
                    logger.exception("Idle reset failed")

        self._idle_monitor_thread = threading.Thread(target=_idle_loop, daemon=True)
        self._idle_monitor_thread.start()

    def stop(self) -> None:
        """Signal the idle-reset daemon to exit and join with a short timeout.

        Sets the wake event so the monitor's ``wait`` returns and its
        next iteration exits. When called from the daemon itself
        (defensive re-entrant guard) the thread reference is cleared
        without a self-join. Otherwise joins for up to two seconds and
        logs a DEBUG line if the thread refused to exit. Idempotent —
        invoked from :meth:`eSurge.terminate` even when no monitor was
        started.
        """
        if not self._idle_monitor_thread:
            return
        self._idle_monitor_event.set()
        if threading.current_thread() is self._idle_monitor_thread:
            # Avoid joining the current thread; it will exit on the next wake-up.
            self._idle_monitor_thread = None
            return
        self._idle_monitor_thread.join(timeout=2.0)
        if self._idle_monitor_thread.is_alive():
            logger.debug("Idle monitor thread did not stop gracefully")
        self._idle_monitor_thread = None
