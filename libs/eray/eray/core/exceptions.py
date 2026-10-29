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


"""Error handling and exception serialization for Ray execution."""

from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass

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

from .status import JobError, JobInfo, JobPreempted, JobStatus

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


def print_remote_raise(ray_error) -> None:
    """Print the traceback from a Ray remote task error.

    This utility function extracts and prints the traceback from a Ray task
    error, which contains serialized exception information. Useful for debugging
    failures in distributed Ray computations.

    Args:
        ray_error: A Ray exception object (e.g., RayTaskError) whose
            ``.cause.args[0]`` is an ExceptionInfo containing a serialized
            traceback.

    Example:
        >>> future = some_remote_task.remote()
        >>> try:
        ...     result = ray.get(future)
        ... except Exception as e:
        ...     print_remote_raise(e)
    """
    tb: Traceback = ray_error.cause.args[0].tb
    traceback.print_tb(tb.as_traceback())
