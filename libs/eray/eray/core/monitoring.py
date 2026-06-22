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
import logging
import logging as pylogging

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
