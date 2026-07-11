# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""Performance monitoring and failure reporting utilities."""

from __future__ import annotations

import contextlib
import glob
import logging
import logging as pylogging
import os
import threading
import time

import ray

from .exceptions import ExceptionInfo


def current_actor_handle() -> ray.actor.ActorHandle:
    """Get the handle of the currently executing Ray actor.

    Returns:
        The ActorHandle for the current actor context.

    Raises:
        RuntimeError: If called outside of an actor context (e.g., in a regular task
                     or driver process).

    Example:
        Inside a Ray actor:

        >>> @ray.remote
        ... class MyActor:
        ...     def get_self_handle(self):
        ...         return current_actor_handle()
        >>>
        >>> actor = MyActor.remote()
        >>> handle = ray.get(actor.get_self_handle.remote())
    """
    return ray.runtime_context.get_runtime_context().current_actor


class SnitchRecipient:
    """Base class for actors that can receive and handle failure reports from child actors.

    This class provides a standardized interface for parent actors to receive
    error notifications from their child actors or tasks. It implements a "snitch"
    pattern where children report their failures up the hierarchy for centralized
    error handling and logging.

    Attributes:
        logger: Logger instance for recording child failure events.

    Example:
        >>> @ray.remote
        ... class ParentActor(SnitchRecipient):
        ...     def __init__(self):
        ...         self.logger = logging.getLogger("ParentActor")
        ...
        ...     def spawn_child(self):
        ...         child = ChildActor.remote()
        ...
        ...         return child
    """

    logger: logging.Logger

    def _child_failed(self, child: ray.actor.ActorHandle | str | None, exception: ExceptionInfo) -> None:
        """Handle failure notification from a child actor or task.

        This method is called when a child reports a failure. It logs the error
        and re-raises the exception to propagate the failure up the call stack.

        Args:
            child: Handle or identifier of the failed child actor/task.
            exception: Serialized exception information from the child failure.

        Raises:
            The original exception that caused the child to fail.

        Note:
            This method is typically called remotely by child actors using
            the log_failures_to context manager.
        """
        info = exception.restore()
        self.logger.error(f"Child {child} failed with exception {info[1]}", exc_info=info)
        exception.reraise()


@contextlib.contextmanager
def log_failures_to(parent, suppress: bool = False):
    """Context manager that reports exceptions to a parent actor.

    This context manager wraps code execution and automatically reports any
    exceptions to a designated parent actor. It's useful for implementing
    hierarchical error reporting in distributed Ray applications.

    Args:
        parent: Parent actor that implements the SnitchRecipient interface
               and has a _child_failed method.
        suppress: If True, suppresses the exception after reporting it.
                 If False, re-raises the exception after reporting.

    Yields:
        None

    Raises:
        Any exception that occurs in the wrapped code (unless suppress=True).

    Example:
        In a child actor:

        >>> @ray.remote
        ... class ChildActor:
        ...     def __init__(self, parent):
        ...         self.parent = parent
        ...
        ...     def risky_operation(self):
        ...         with log_failures_to(self.parent):
        ...
        ...             dangerous_computation()

        Suppressing exceptions:

        >>> with log_failures_to(parent_actor, suppress=True):
        ...     might_fail()
    """
    try:
        yield
    except Exception as e:
        try:
            handle = current_actor_handle()
        except RuntimeError:
            handle = ray.runtime_context.get_runtime_context().get_task_id()

        parent._child_failed.remote(handle, ExceptionInfo.ser_exc_info(e))
        if not suppress:
            raise e


DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"


@ray.remote
class StopwatchActor:
    """Ray actor for collecting and aggregating performance timing measurements.

    This actor provides centralized performance monitoring for distributed
    computations. It accumulates timing measurements from multiple sources
    and periodically logs performance statistics. Useful for profiling
    distributed training or inference workloads.

    The actor maintains running totals and counts for different measurement
    categories, allowing calculation of average execution times.

    Example:
        >>> stopwatch = StopwatchActor.remote()
        >>>
        >>>
        >>> start_time = time.time()
        >>> expensive_computation()
        >>> duration = time.time() - start_time
        >>> stopwatch.measure.remote("computation", duration)
        >>>
        >>>
        >>> total_time, count = ray.get(stopwatch.get.remote("computation"))
        >>> avg_time = ray.get(stopwatch.average.remote("computation"))
    """

    def __init__(self):
        """Initialize the StopwatchActor with logging and timing storage.

        Sets up logging configuration and initializes internal data structures
        for storing timing measurements and counts.
        """
        pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        self._logger = pylogging.getLogger("StopwatchActor")
        self._times_per = {}
        self._counts_per = {}
        self._total = 0

    def measure(self, name: str, time: float) -> None:
        """Record a timing measurement for a named operation.

        Args:
            name: Identifier for the type of operation being measured.
            time: Duration of the operation in seconds (or other consistent unit).

        Note:
            After every 1000 measurements, the actor automatically logs
            average times for all tracked operations.

        Example:
            >>>
            >>> start = time.time()
            >>> execute_query(sql)
            >>> duration = time.time() - start
            >>> stopwatch.measure.remote("db_query", duration)
        """
        self._times_per[name] = self._times_per.get(name, 0) + time
        self._counts_per[name] = self._counts_per.get(name, 0) + 1
        self._total += 1

        if self._total % 1000 == 0:
            for name, time in self._times_per.items():
                self._logger.info(f"{name}: {time / self._counts_per[name]}")

    def get(self, name: str) -> tuple[float, int]:
        """Get total time and count for a named operation.

        Args:
            name: Identifier for the operation to query.

        Returns:
            A tuple containing (total_time, count) for the operation.
            Returns (0, 0) if the operation name hasn't been measured.

        Example:
            >>> total_time, count = ray.get(stopwatch.get.remote("training_step"))
            >>> print(f"Total: {total_time:.2f}s over {count} steps")
        """
        return self._times_per.get(name, 0), self._counts_per.get(name, 0)

    def average(self, name: str) -> float:
        """Calculate average time for a named operation.

        Args:
            name: Identifier for the operation to query.

        Returns:
            Average time per operation. Returns 0.0 if the operation
            name hasn't been measured (uses 1 as denominator to avoid
            division by zero).

        Example:
            >>> avg_time = ray.get(stopwatch.average.remote("inference"))
            >>> print(f"Average inference time: {avg_time:.3f}s")
        """
        return self._times_per.get(name, 0) / self._counts_per.get(name, 1)


# Raylet / GCS log-spam guard
# Ray (<= 2.54) does not rotate the raylet/GCS component logs: a raylet stuck
# in a retry loop (e.g. "Caller of RequestWorkerLease is dead" after killed
# drivers) appends to ``raylet.out`` at tens of GB per hour until the node's
# disk fills and object spilling fails, taking live jobs down with it.
# Truncating the file in place is safe on the live daemon (its logger appends;
# observed repeatedly in production) and is the only mitigation short of a
# raylet restart, which kills running jobs.

RAYLET_LOG_GUARD_ENV = "ERAY_RAYLET_LOG_GUARD"
RAYLET_LOG_GUARD_MAX_GB_ENV = "ERAY_RAYLET_LOG_GUARD_MAX_GB"
RAYLET_LOG_GUARD_INTERVAL_ENV = "ERAY_RAYLET_LOG_GUARD_INTERVAL_S"

_RAYLET_LOG_GUARD_FILES = ("raylet.out", "raylet.err", "gcs_server.out", "gcs_server.err")
_DEFAULT_RAYLET_LOG_MAX_BYTES = 5 * 1024**3
_DEFAULT_RAYLET_LOG_INTERVAL_S = 300.0

_raylet_guard_lock = threading.Lock()
_raylet_guard_thread: threading.Thread | None = None

_raylet_guard_logger = pylogging.getLogger("eray.raylet-log-guard")


def _ray_session_log_dirs() -> list[str]:
    """Best-effort discovery of Ray session ``logs`` directories on this host.

    Prefers the live Ray node's session directory when Ray is initialized in
    this process; otherwise globs the usual temp roots (``RAY_TMPDIR``,
    ``TMPDIR``, ``/tmp/ray``), which covers guard processes that run beside
    Ray rather than inside it.
    """
    with contextlib.suppress(Exception):
        node = ray._private.worker._global_node
        if node is not None:
            logs = os.path.join(node.get_session_dir_path(), "logs")
            if os.path.isdir(logs):
                return [logs]

    found: list[str] = []
    seen: set[str] = set()
    roots = [os.environ.get("RAY_TMPDIR"), os.environ.get("TMPDIR"), "/tmp/ray"]
    for root in roots:
        if not root:
            continue
        for pattern in (
            os.path.join(root, "session_*", "logs"),
            os.path.join(root, "ray", "session_*", "logs"),
            os.path.join(root, "ray", "ray", "session_*", "logs"),
        ):
            for logs in glob.glob(pattern):
                real = os.path.realpath(logs)
                if real not in seen and os.path.isdir(real):
                    seen.add(real)
                    found.append(real)
    return found


def sweep_raylet_logs(
    log_dirs: list[str] | None = None,
    max_bytes: int = _DEFAULT_RAYLET_LOG_MAX_BYTES,
) -> list[tuple[str, int]]:
    """Truncate oversized Ray component logs in place; returns what was cut.

    Args:
        log_dirs: Session ``logs`` directories to sweep. ``None`` auto-discovers
            via the live Ray node or the usual temp-root globs.
        max_bytes: Size threshold above which a component log is truncated.

    Returns:
        ``[(path, size_before_bytes), ...]`` for every file truncated.
    """
    truncated: list[tuple[str, int]] = []
    for logs_dir in log_dirs if log_dirs is not None else _ray_session_log_dirs():
        for name in _RAYLET_LOG_GUARD_FILES:
            path = os.path.join(logs_dir, name)
            with contextlib.suppress(OSError):
                size = os.path.getsize(path)
                if size > max_bytes:
                    with open(path, "r+") as f:
                        f.truncate(0)
                    truncated.append((path, size))
                    _raylet_guard_logger.warning(
                        "Truncated %s (%.1f GB): Ray does not rotate component logs and a spamming "
                        "raylet can fill the disk under live jobs.",
                        path,
                        size / 1024**3,
                    )
    return truncated


def start_raylet_log_guard(
    interval_s: float | None = None,
    max_bytes: int | None = None,
    log_dirs: list[str] | None = None,
) -> threading.Thread | None:
    """Start the per-process daemon that keeps Ray component logs bounded.

    Idempotent (one guard per process) and gated by ``ERAY_RAYLET_LOG_GUARD``
    (set ``0`` to disable). Threshold and cadence come from
    ``ERAY_RAYLET_LOG_GUARD_MAX_GB`` (default 5) and
    ``ERAY_RAYLET_LOG_GUARD_INTERVAL_S`` (default 300) unless given explicitly.

    Returns:
        The guard thread, or ``None`` when disabled by env.
    """
    if os.environ.get(RAYLET_LOG_GUARD_ENV, "1").strip().lower() in ("0", "false", "off"):
        return None

    global _raylet_guard_thread
    with _raylet_guard_lock:
        if _raylet_guard_thread is not None and _raylet_guard_thread.is_alive():
            return _raylet_guard_thread

        if max_bytes is None:
            max_bytes = int(float(os.environ.get(RAYLET_LOG_GUARD_MAX_GB_ENV, "5")) * 1024**3)
        if interval_s is None:
            interval_s = float(os.environ.get(RAYLET_LOG_GUARD_INTERVAL_ENV, str(_DEFAULT_RAYLET_LOG_INTERVAL_S)))

        def _loop():
            while True:
                with contextlib.suppress(Exception):
                    sweep_raylet_logs(log_dirs=log_dirs, max_bytes=max_bytes)
                time.sleep(interval_s)

        _raylet_guard_thread = threading.Thread(target=_loop, name="eray-raylet-log-guard", daemon=True)
        _raylet_guard_thread.start()
        return _raylet_guard_thread
