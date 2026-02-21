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

"""Abstract scheduler interface for the eSurge inference engine.

This module defines the abstract base class that all scheduler implementations
must inherit from. It provides a consistent interface for scheduling operations
used by the engine core.

The scheduler interface defines the contract for:
    - Scheduling requests into batches
    - Updating state based on model outputs
    - Managing request lifecycle (add, finish)
    - Querying scheduler state

Classes:
    SchedulerInterface: Abstract base class for scheduler implementations.

Example:
    Creating a custom scheduler::

        >>> from easydel.inference.esurge.scheduler.interface import SchedulerInterface
        >>>
        >>> class CustomScheduler(SchedulerInterface):
        ...     def schedule(self):
        ...         # Implementation
        ...         pass
        ...     # ... implement other abstract methods
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine_types import EngineCoreOutputs
    from ..outputs import ModelRunnerOutput
    from ..request import EngineRequest, EngineRequestStatus
    from .output import SchedulerOutput


class SchedulerInterface(ABC):
    """Abstract base class defining the scheduler interface.

    This class defines the contract that all scheduler implementations must
    follow. It provides abstract methods for the core scheduling operations
    needed by the eSurge inference engine.

    The scheduler is responsible for:
        - Deciding which requests to process in each forward pass
        - Managing KV cache allocation
        - Handling request preemption and resumption
        - Tracking request completion

    Implementations must handle:
        - Token budget management
        - Request prioritization
        - Memory constraints
        - Batching efficiency

    Example:
        >>> class MyScheduler(SchedulerInterface):
        ...     def schedule(self):
        ...         # Return SchedulerOutput with batch decisions
        ...         pass
        ...     def update_from_output(self, scheduler_output, model_output):
        ...         # Process model outputs and update state
        ...         pass
        ...     # ... implement other required methods
    """

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """Schedule requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. This method is
        called repeatedly by the engine's main loop.

        The scheduler produces a dictionary of {req_id: num_tokens} that
        specifies how many tokens to process for each request in this step.
        For example:
            - num_tokens can be as large as the prompt length for new requests
            - num_tokens is typically 1 for autoregressive generation
            - num_tokens can vary for chunked prefills, prefix caching, etc.

        Additionally, the scheduler returns useful metadata about each request
        and the batch as a whole, which the model runner uses to prepare inputs.

        Returns:
            SchedulerOutput: Object containing:
                - scheduled_new_reqs: New requests being scheduled
                - scheduled_cached_reqs: Continuing requests
                - num_scheduled_tokens: Token counts per request
                - total_num_scheduled_tokens: Total tokens in batch
                - And other scheduling metadata

        Note:
            This method should not block. If no requests can be scheduled
            (e.g., all are waiting for resources), it should return an
            empty batch.
        """
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update scheduler state based on model runner output.

        This method is called after the model runner has processed the
        scheduled batch. It processes the generated tokens, updates request
        states, and determines which requests have finished.

        The model runner output includes:
            - Generated token IDs
            - Draft tokens for speculative decoding
            - Logprobs (if requested)
            - Other generation metadata

        The scheduler uses this information to:
            - Append generated tokens to requests
            - Check stop conditions
            - Update KV cache states
            - Mark finished requests

        Args:
            scheduler_output: The scheduling decision that was executed.
            model_runner_output: Output from the model runner containing
                generated tokens and metadata.

        Returns:
            dict[int, EngineCoreOutputs]: Mapping from client index to
                EngineCoreOutputs containing the outputs for requests
                from that client. Each EngineCoreOutputs contains
                per-request outputs and finished request IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "EngineRequest") -> None:
        """Add a new request to the scheduler's queue.

        The request is added to the waiting queue and will be considered
        for scheduling in subsequent calls to schedule().

        Args:
            request: The engine request to add. Must have all required
                fields populated including request_id, prompt_token_ids,
                and sampling_params.

        Note:
            The request is not immediately scheduled. It will be picked up
            by the next schedule() call based on the scheduling policy and
            available resources.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: "EngineRequestStatus",
    ) -> None:
        """Mark requests as finished from external signal.

        This method handles finish signals from outside the scheduler,
        such as:
            1. Client-initiated abort (e.g., connection closed)
            2. Stop string detection by the frontend after detokenization

        If a request ID is not found in the scheduler's queues, this
        method silently ignores it.

        Args:
            request_ids: A single request ID string, or an iterable of
                request ID strings to finish.
            finished_status: The finished status to assign to the requests.
                Must be a finished status from EngineRequestStatus (e.g.,
                FINISHED_ABORTED, FINISHED_STOPPED).

        Note:
            This is different from natural completion (hitting EOS or
            length limit), which is detected internally during
            update_from_output().
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests in the scheduler.

        Returns the total count of requests that are either waiting in
        the queue or currently running (being processed).

        Returns:
            int: Number of unfinished requests (waiting + running).
        """
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests in the scheduler.

        Returns:
            bool: True if there are waiting or running requests,
                False otherwise.

        Note:
            This is a convenience method that checks if
            get_num_unfinished_requests() > 0.
        """
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Check if there are finished requests pending notification.

        The scheduler maintains an internal list of requests that finished
        in the previous step. This list is included in the next schedule()
        output so the model runner can clear cached states for these
        requests.

        This method checks if this internal finished list is non-empty.
        This is different from checking if there are no unfinished requests
        - a scheduler can have both unfinished and pending finished requests.

        Returns:
            bool: True if there are finished requests that haven't been
                reported in a SchedulerOutput yet.

        Note:
            This information is particularly useful for data-parallel
            attention implementations.
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """Check if there are any requests to process.

        Returns True if there are either:
            - Unfinished requests (waiting or running)
            - Finished requests not yet returned in SchedulerOutput

        Returns:
            bool: True if the scheduler has any requests to handle.
        """
        return self.has_unfinished_requests() or self.has_finished_requests()

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache for KV cache.

        Clears all cached prefix entries from the KV cache. This is
        required when model weights are updated (e.g., during live
        model updates) to ensure cached values are invalidated.

        Returns:
            bool: True if the cache was successfully reset, False if
                the operation failed or caching is not enabled.

        Warning:
            This should only be called when necessary, as it invalidates
            all prefix cache entries and may significantly impact
            performance until the cache is rebuilt.
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Get the counts of running and waiting requests.

        Returns:
            tuple[int, int]: A tuple of (num_running_reqs, num_waiting_reqs)
                where:
                - num_running_reqs: Requests currently being processed
                - num_waiting_reqs: Requests waiting to be scheduled
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scheduler and release resources.

        Performs cleanup operations when the scheduler is being shut down.
        This may include:
            - Releasing KV cache memory
            - Clearing request queues
            - Stopping background threads (if any)

        After calling this method, the scheduler should not be used.
        """
        raise NotImplementedError
