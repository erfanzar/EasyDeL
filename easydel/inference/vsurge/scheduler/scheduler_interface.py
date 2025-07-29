# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..request_type import EngineRequest, EngineRequestStatus
    from ..utils import EngineCoreOutputs, ModelRunnerOutput
    from .output import SchedulerOutput


class SchedulerInterface(ABC):
    """Abstract base class defining the interface for scheduling requests in a model execution engine.

    This interface provides methods for managing the lifecycle of requests, from scheduling
    to completion, in a model execution pipeline. Implementations of this interface are
    responsible for handling request prioritization, token allocation, and state management
    for efficient model inference.

    The scheduler operates in a loop, where each iteration corresponds to a single forward
    pass of the model. It manages both new requests (e.g., prompt processing) and ongoing
    requests (e.g., token generation), supporting features like chunked prefills, prefix
    caching, and speculative decoding.
    """

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """Schedule requests for processing in the current iteration.

        This method determines which requests to process and how many tokens to allocate
        to each in a single forward pass of the model. For example:
        - New requests may process multiple prompt tokens at once.
        - Ongoing requests may generate one token at a time (auto-regressive generation).
        - Advanced techniques like chunked prefills or speculative decoding may process
          variable token counts.

        The scheduler returns a `SchedulerOutput` object containing the scheduling
        decisions (e.g., {request_id: num_tokens}) and additional metadata for the
        model runner to prepare model inputs.

        Returns:
            SchedulerOutput: Contains scheduling decisions and metadata for the current
            iteration.

        Example:
            >>> scheduler = MyScheduler()
            >>> output = scheduler.schedule()
        """
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update scheduler state based on model runner output.

        Called after the model processes the scheduled requests, this method updates
        the scheduler's internal state using the model runner's output (e.g., generated
        tokens, draft tokens for speculative decoding). It checks for completed requests
        and prepares outputs for each client.

        Args:
            scheduler_output: The output from the previous scheduling step.
            model_runner_output: The output from the model runner, including generated tokens.

        Returns:
            dict[int, EngineCoreOutputs]: A dictionary mapping client indices to their
            respective request outputs.

        Example:
            >>> scheduler_output = scheduler.schedule()
            >>> model_output = model_runner.process(scheduler_output)
            >>> client_outputs = scheduler.update_from_output(scheduler_output, model_output)
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "EngineRequest") -> None:
        """Add a new request to the scheduler's internal queue.

        This method queues a new request for scheduling, typically a new inference task
        with a prompt or other input data.

        Args:
            request: The EngineRequest object containing the request details.

        Example:
            >>> request = EngineRequest(id=1, prompt="Hello, world!", max_tokens=50)
            >>> scheduler.add_request(request)
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: "EngineRequestStatus",
    ) -> None:
        """Mark requests as finished in the scheduler's internal queue.

        This method is invoked when requests are completed (e.g., due to client abortion
        or detection of a stop string after token generation). If a request ID is not
        in the queue, the method silently ignores it.

        Args:
            request_ids: A single request ID or an iterable of request IDs to finish.
            finished_status: The status indicating why the requests are finished (e.g., completed, aborted).

        Example:
            >>> scheduler.finish_requests("req_1", EngineRequestStatus.ABORTED)
            >>> scheduler.finish_requests(["req_2", "req_3"], EngineRequestStatus.COMPLETED)
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Return the number of unfinished requests in the scheduler's queue.

        This method provides the count of requests that are still active (i.e., not yet finished).

        Returns:
            int: The number of unfinished requests.

        Example:
            >>> num_unfinished = scheduler.get_num_unfinished_requests()
            >>> print(num_unfinished)
            5
        """
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests in the scheduler's queue.

        Returns:
            bool: True if there are unfinished requests, False otherwise.

        Example:
            >>> if scheduler.has_unfinished_requests():
            ...     print("There are still requests to process!")
        """
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Check if there are finished requests that need to be cleared.

        Unlike `has_unfinished_requests`, this method checks for requests that were
        marked as finished in the previous scheduling step but have not yet been
        cleared from the scheduler's state. These requests may need to be sent to
        the model runner to clear cached states (e.g., for KV cache management).

        Returns:
            bool: True if there are finished requests to clear, False otherwise.

        Example:
            >>> if scheduler.has_finished_requests():
            ...     print("Clearing finished requests in next schedule step.")
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """Check if there are any requests (unfinished or finished) in the scheduler.

        This method returns True if there are either unfinished requests or finished
        requests that have not yet been cleared.

        Returns:
            bool: True if there are any requests, False otherwise.

        Example:
            >>> if scheduler.has_requests():
            ...     print("Scheduler has requests to process or clear.")
        """
        return self.has_unfinished_requests() or self.has_finished_requests()

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache for the KV cache.

        This method is used to clear the prefix cache, typically required when model
        weights are updated or when the cache needs to be invalidated.

        Returns:
            bool: True if the cache was successfully reset, False otherwise.

        Example:
            >>> if scheduler.reset_prefix_cache():
            ...     print("Prefix cache reset successfully.")
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Return the number of running and waiting requests.

        This method provides a tuple containing the count of requests currently being
        processed (running) and those waiting to be scheduled.

        Returns:
            tuple[int, int]: A tuple of (num_running_requests, num_waiting_requests).

        Example:
            >>> running, waiting = scheduler.get_request_counts()
            >>> print(f"Running: {running}, Waiting: {waiting}")
            Running: 2, Waiting: 3
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the scheduler and release associated resources.

        This method is called to gracefully terminate the scheduler, ensuring all
        resources (e.g., queues, caches) are properly cleaned up.

        Example:
            >>> scheduler.shutdown()
            >>> print("Scheduler has been shut down.")
        """
        raise NotImplementedError
