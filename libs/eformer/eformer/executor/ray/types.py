# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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


"""Type definitions and error handling utilities for Ray-based distributed execution.

This module provides a comprehensive type system and error handling framework for
managing distributed computation jobs using Ray. It includes:

1. Error handling utilities for Ray-specific exceptions
2. Serializable exception information for cross-process error propagation
3. Job status tracking with success/failure/preemption states
4. Reference management and communication primitives
5. Performance monitoring utilities
6. Multi-slice coordination data structures for TPU/GPU clusters

The module is designed to provide robust error handling and status tracking
for distributed training and inference workloads, with special attention to
preemption scenarios common in cloud TPU environments.

Example:
    Basic job status tracking:

    >>> job_info = JobInfo(name="training_job", state="running", kind="training")
    >>>
    >>> status = JobSucceeded(job_info, result={"loss": 0.1})
    >>>
    >>> status = JobFailed(job_info, error=ValueError("Invalid input"))

    Exception serialization:

    >>> try:
    ...     risky_operation()
    ... except Exception:
    ...     exc_info = ExceptionInfo.ser_exc_info()
    ...
    ...     exc_info.reraise()
"""

from __future__ import annotations

import contextlib
import logging
import logging as pylogging
import sys
import traceback
from dataclasses import dataclass

import ray
import tblib
from ray.exceptions import (
    ActorDiedError,
    ActorUnavailableError,
    NodeDiedError,
    OwnerDiedError,
    RayError,
    RaySystemError,
    RayTaskError,
    WorkerCrashedError,
)
from tblib import Traceback

logger = logging.getLogger("ray")


def handle_ray_error(job_info: JobInfo, e: RayError) -> JobStatus:
    """Classify Ray errors and convert them to appropriate JobStatus objects.

    This function analyzes Ray-specific exceptions and categorizes them into
    different types of job failures (preemption, system error, task error, etc.).
    It provides consistent error handling across the distributed execution system.

    Args:
        job_info: Metadata about the job that encountered the error.
        e: The Ray exception that was raised during job execution.

    Returns:
        A JobStatus subclass indicating the type of failure:
        - JobPreempted: For infrastructure failures (node/actor death, worker crashes)
        - JobError: For system errors, task errors, or unknown exceptions

    Example:
        >>> job_info = JobInfo(name="training", state="running", kind="ml_job")
        >>> try:
        ...     ray.get(some_remote_task.remote())
        ... except NodeDiedError as e:
        ...     status = handle_ray_error(job_info, e)
        ...     assert isinstance(status, JobPreempted)
    """
    if isinstance(e, NodeDiedError | OwnerDiedError | ActorDiedError | ActorUnavailableError | WorkerCrashedError):
        logger.exception("Infra/preemption-related error", exc_info=e)
        return JobPreempted(job_info, e)
    elif isinstance(e, RaySystemError):
        logger.exception("System error", exc_info=e)
        return JobError(job_info, e)
    elif isinstance(e, RayTaskError):
        logger.exception("Task error", exc_info=e)
        return JobError(job_info, e)
    else:
        logger.exception("Unknown error", exc_info=e)
        return JobError(job_info, e)


@dataclass
class ExceptionInfo:
    """Serializable container for exception information across process boundaries.

    This class captures exception details and tracebacks in a format that can be
    serialized and transmitted between Ray actors/tasks. It uses tblib to preserve
    traceback information, enabling proper error reporting in distributed systems.

    Attributes:
        ex: The original exception instance, or None if no exception was captured.
        tb: Serialized traceback information using tblib.Traceback.

    Example:
        Capturing and re-raising an exception in a different process:

        >>> try:
        ...     raise ValueError("Something went wrong")
        ... except Exception:
        ...     exc_info = ExceptionInfo.ser_exc_info()
        ...
        >>>
        >>> exc_info.reraise()
    """

    ex: BaseException | None
    tb: tblib.Traceback

    def restore(self) -> tuple[type[BaseException], BaseException, traceback.TracebackType]:
        """Restore the exception information to standard Python exc_info format.

        Returns:
            A tuple containing (exception_type, exception_value, traceback)
            compatible with sys.exc_info() format. If no exception was captured,
            returns a generic Exception with an appropriate message.

        Example:
            >>> exc_info = ExceptionInfo.ser_exc_info()
            >>> exc_type, exc_value, exc_tb = exc_info.restore()
            >>>
        """
        if self.ex is not None:
            exc_value = self.ex.with_traceback(self.tb.as_traceback())
            return (self.ex.__class__, exc_value, self.tb.as_traceback())
        else:
            return (
                Exception,
                Exception("Process failed with no exception"),
                self.tb.as_traceback(),
            )

    def reraise(self) -> None:
        """Re-raise the captured exception with its original traceback.

        Raises:
            The original exception that was captured, or a generic Exception
            if no specific exception was available.

        Example:
            >>> try:
            ...     dangerous_operation()
            ... except Exception:
            ...     exc_info = ExceptionInfo.ser_exc_info()
            >>>
            >>> exc_info.reraise()
        """
        if self.ex is not None:
            raise self.ex.with_traceback(self.tb.as_traceback())
        else:
            raise Exception("Process failed with no exception").with_traceback(self.tb.as_traceback())

    @classmethod
    def ser_exc_info(cls, exception: BaseException | None = None) -> ExceptionInfo:
        """Create an ExceptionInfo from current exception context or provided exception.

        Args:
            exception: Specific exception to serialize. If None, uses sys.exc_info()
                      to capture the current exception being handled.

        Returns:
            ExceptionInfo containing the serialized exception and traceback.

        Example:
            Capture current exception:

            >>> try:
            ...     risky_function()
            ... except ValueError as e:
            ...     exc_info = ExceptionInfo.ser_exc_info()

            Capture specific exception:

            >>> try:
            ...     risky_function()
            ... except ValueError as e:
            ...     exc_info = ExceptionInfo.ser_exc_info(e)
        """
        if exception is None:
            _, exc_value, exc_traceback = sys.exc_info()
            tb = tblib.Traceback(exc_traceback)
            return ExceptionInfo(exc_value, tb)
        else:
            tb = exception.__traceback__
            tb = tblib.Traceback(tb)
            return ExceptionInfo(exception, tb)


@dataclass
class JobInfo:
    """
    Metadata describing a TPU/GPU/CPU job managed via Ray.

    Attributes:
        name (str): A human-readable identifier for the job.
        state (str): The current state of the job (e.g., "pending", "running", "succeeded", "failed").
        kind (str): The type or classification of the job (e.g., "training", "inference").
    """

    name: str
    state: str
    kind: str


@dataclass
class JobStatus:
    """
    Base class representing the final status of a job after a Ray call.

    This class wraps job metadata and serves as a common interface for
    distinguishing between successful and failed executions.

    Attributes:
        info (JobInfo): Metadata about the job.
    """

    info: JobInfo


@dataclass
class JobSucceeded(JobStatus):
    """
    Indicates that the job completed successfully and returned a result.

    Attributes:
        result (object): The output produced by the job.
    """

    result: object


@dataclass
class JobPreempted(JobStatus):
    """
    Indicates that the job was interrupted or preempted, likely by external factors
    such as TPU quota eviction or infrastructure scaling events.

    Attributes:
        error (Exception): The exception raised due to preemption.
    """

    error: Exception


@dataclass
class JobFailed(JobStatus):
    """
    Indicates that the job ran to completion but failed due to an expected runtime issue.

    This could include errors such as invalid input, failed assertions, or handled exceptions.

    Attributes:
        error (Exception): The exception describing why the job failed.
    """

    error: Exception


@dataclass
class JobError(JobStatus):
    """
    Indicates that the job encountered an internal or unexpected error.

    This is typically reserved for unexpected exceptions, infrastructure issues,
    or serialization problems in the Ray runtime.

    Attributes:
        error (Exception): The exception or error message from the failure.
    """

    error: Exception


def print_remote_raise(ray_error) -> None:
    """Print the traceback from a Ray remote task error.

    This utility function extracts and prints the traceback from a Ray task
    error, which contains serialized exception information. Useful for debugging
    failures in distributed Ray computations.

    Args:
        ray_error: The .error attribute from a Ray task output,
                   containing a pickled exception with tblib.Traceback.

    Example:
        >>> future = some_remote_task.remote()
        >>> try:
        ...     result = ray.get(future)
        ... except Exception as e:
        ...     print_remote_raise(e)
    """
    tb: Traceback = ray_error.cause.args[0].tb
    traceback.print_tb(tb.as_traceback())


@dataclass
class RefBox:
    """Wrapper to prevent automatic ObjectRef dereferencing in Ray.

    Ray automatically dereferences ObjectRefs when they are passed as arguments
    to remote functions, but this doesn't happen when they're nested inside other
    objects. RefBox takes advantage of this behavior to control when dereferencing
    occurs, which can be useful for lazy evaluation or passing references between
    actors without triggering computation.

    Attributes:
        ref: The Ray ObjectRef to be wrapped.

    Example:
        >>>
        >>> result_ref = expensive_computation.remote()
        >>> boxed = RefBox(result_ref)
        >>> another_task.remote(boxed)
        >>>
        >>>
        >>> actual_result = boxed.get()

    See Also:
        Ray documentation on object passing:
        https://docs.ray.io/en/latest/ray-core/objects.html
    """

    ref: ray.ObjectRef

    def get(self):
        """Dereference the wrapped ObjectRef and return its value.

        Returns:
            The actual value stored in the ObjectRef.

        Raises:
            Any exception that occurred during the computation of the ObjectRef.

        Example:
            >>> computation_ref = expensive_task.remote()
            >>> box = RefBox(computation_ref)
            >>> result = box.get()
        """
        return ray.get(self.ref)


class DoneSentinel:
    """Sentinel class to indicate completion or termination state.

    This class serves as a unique marker object to signal that a process,
    computation, or data stream has reached its end. Using a sentinel class
    instead of None or other values prevents ambiguity when None might be
    a valid result.

    Example:
        >>> def process_items(items):
        ...     for item in items:
        ...         if item is DONE:
        ...             break
        ...         yield process_item(item)
    """

    pass


DONE = DoneSentinel()
"""Global instance of DoneSentinel for signaling completion.

This singleton instance should be used throughout the codebase to maintain
consistency when checking for completion states.

Example:
    >>> queue.put(result)
    >>> queue.put(DONE)
    >>>
    >>> while True:
    ...     item = queue.get()
    ...     if item is DONE:
    ...         break
    ...     process(item)
"""


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


@dataclass
class MultisliceInfo:
    """Information about a multi-slice configuration for distributed execution.

    This class stores configuration data for multi-slice TPU/GPU clusters where
    computation is distributed across multiple slices that coordinate through
    a central coordinator node.

    Attributes:
        coordinator_ip: IP address of the coordinator node.
        slice_id: Unique identifier for this slice within the multi-slice setup.
        num_slices: Total number of slices in the multi-slice configuration.
        port: Port number for multi-slice coordination communication.

    Example:
        >>> multi_slice_config = MultisliceInfo(
        ...     coordinator_ip="10.0.0.1",
        ...     slice_id=0,
        ...     num_slices=4,
        ...     port=8081
        ... )
        >>> print(f"Slice {multi_slice_config.slice_id} of {multi_slice_config.num_slices}")
    """

    coordinator_ip: str
    slice_id: int
    num_slices: int
    port: int = 8081


@dataclass
class SliceInfo:
    """Information about a single compute slice in a distributed cluster.

    This class represents the configuration and metadata for a single compute
    slice, which typically consists of multiple hosts with accelerators (TPUs/GPUs).
    Used in multi-slice configurations for large-scale distributed training.

    Attributes:
        slice_name: Unique name identifier for this slice.
        num_hosts: Number of host machines in this slice.
        ip_address: IP address of the slice head node.
        num_accelerators_per_host: Number of accelerators (TPUs/GPUs) per host machine.

    Example:
        >>> slice_config = SliceInfo(
        ...     slice_name="slice-0",
        ...     num_hosts=8,
        ...     ip_address="10.0.1.10",
        ...     num_accelerators_per_host=8
        ... )
        >>> total_accelerators = slice_config.num_hosts * slice_config.num_accelerators_per_host
        >>> print(f"Slice has {total_accelerators} total accelerators")
    """

    slice_name: str
    num_hosts: int
    ip_address: str
    num_accelerators_per_host: int
    node_ids: list[str] | None = None
    host_infos: list[dict] | None = None


@dataclass(frozen=True)
class HostInfo:
    """Information about a TPU host within a slice.

    Attributes:
        host_id: Unique identifier for the host within its slice.
        slice_name: Name of the TPU slice this host belongs to.
        num_devices: Number of TPU devices available on this host.
        healthy: Whether the host is currently healthy and operational.
        failed: Whether the host has encountered a failure.
    """

    host_id: int
    slice_name: str
    num_devices: int | None
    healthy: bool
    failed: bool
    node_id: str | None = None
