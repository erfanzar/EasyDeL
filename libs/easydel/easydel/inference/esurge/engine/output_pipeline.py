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

"""The off-hot-path output worker for the eSurge engine.

The scheduler loop is the hot generation path: it applies model tokens to
scheduler state and must immediately return to launching device work.
Detokenization, reasoning/tool parsing, stream-delta assembly, and
request-event wakeups happen on this single FIFO worker thread instead, so
output processing overlaps the next device step while preserving per-token
ordering.
"""

from __future__ import annotations

import queue
import threading
import traceback
import typing

from ..logger import logger

_STOP_SENTINEL = None


class OutputPipeline:
    """FIFO worker thread decoupling output processing from the scheduler loop.

    Bundles submitted via :meth:`submit` are processed strictly in order by
    the ``process`` callback on a dedicated daemon thread. On a processing
    failure ``on_fatal`` receives the exception (so the engine can record it
    and wake every blocked caller); the worker keeps draining subsequent
    bundles so producers never block on a full pipeline.

    Args:
        process: Callback consuming one engine-output bundle (detokenize,
            parse, update outputs, wake waiters).
        on_fatal: Callback invoked with ``(exc, formatted_traceback)`` when
            ``process`` raises; runs on the worker thread.
    """

    def __init__(
        self,
        *,
        process: typing.Callable[[typing.Any], None],
        on_fatal: typing.Callable[[BaseException, str], None],
    ) -> None:
        self.queue: queue.Queue = queue.Queue()
        self._process = process
        self._on_fatal = on_fatal
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the worker thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the worker thread (no-op if it is already running)."""
        if self.running:
            return
        self._thread = threading.Thread(target=self._loop, name="eSurgeOutputWorker", daemon=True)
        self._thread.start()

    def submit(self, engine_outputs) -> None:
        """Queue a bundle for ordered processing; process inline if stopped.

        The inline fallback keeps lightweight harnesses (and the window
        between worker death and engine teardown) functional at the cost of
        running on the caller's thread.

        Args:
            engine_outputs: Scheduler-produced output bundle from
                ``update_from_output``. Falsy bundles are ignored.
        """
        if not engine_outputs:
            return
        if not self.running:
            self._process(engine_outputs)
            return
        self.queue.put(engine_outputs)

    def stop(self, timeout: float = 5.0) -> None:
        """Drain the queue and stop the worker via the ``None`` sentinel."""
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            self.queue.put(_STOP_SENTINEL)
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Engine output worker did not stop gracefully")
                return
        if self._thread is thread:
            self._thread = None

    def _loop(self) -> None:
        """Drain bundles in FIFO order until the sentinel or a fatal error."""
        while True:
            engine_outputs = self.queue.get()
            try:
                if engine_outputs is _STOP_SENTINEL:
                    return
                self._process(engine_outputs)
            except Exception as exc:
                traceback.print_exc()
                logger.error("Engine output worker failed: %s", exc)
                self._on_fatal(exc, traceback.format_exc())
            finally:
                self.queue.task_done()
