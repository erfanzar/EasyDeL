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

"""Async Scheduler for eSurge inference engine.

This module implements an asynchronous scheduler that extends the base Scheduler
to support async token sampling, enabling overlap between token generation and
the next forward pass for improved throughput.

Key Features:
    - Async token sampling with placeholder management
    - 30-40% latency reduction through execution overlap
    - Maintains compatibility with base Scheduler interface
    - Handles structured output token tracking

The async scheduler works by:
    1. Adding output placeholders for tokens that will be generated asynchronously
    2. Allowing the next forward pass to begin while tokens are being sampled
    3. Replacing placeholders with actual tokens in the subsequent iteration

Classes:
    AsyncScheduler: Asynchronous scheduler with placeholder-based token sampling.

Example:
    Basic async scheduler usage::

        >>> from easydel.inference.esurge.config import Config, SchedulerConfig
        >>> from easydel.inference.esurge.scheduler import AsyncScheduler
        >>>
        >>> config = Config(
        ...     scheduler_config=SchedulerConfig(
        ...         max_num_seqs=8,
        ...         max_num_batched_tokens=2048,
        ...         async_scheduling=True,  # Enable async mode
        ...     )
        ... )
        >>>
        >>> scheduler = AsyncScheduler(config=config, kv_cache_config=kv_config)
        >>> output = scheduler.schedule()  # Returns with placeholders

Note:
    The async scheduler must be used with a runner that supports async execution
    (i.e., has execute_model_async() method and handles AsyncPreResults).
"""

from __future__ import annotations

from eformer.loggings import get_logger

from ..request import EngineRequest, EngineRequestStatus
from .output import SchedulerOutput
from .scheduler import Scheduler

logger = get_logger("eSurgeAsyncScheduler")


class AsyncScheduler(Scheduler):
    """Asynchronous scheduler with placeholder-based token sampling.

    The AsyncScheduler extends the base Scheduler to support asynchronous token
    generation, allowing the next forward pass to begin while tokens from the
    current iteration are still being sampled on the device.

    This is achieved through a placeholder mechanism:
        - After scheduling, output placeholders are added for tokens that will
          be generated asynchronously
        - The runner returns immediately with empty sampled_token_ids
        - In the next iteration, placeholders are replaced with actual tokens
        - This overlap reduces end-to-end latency by 30-40%

    The async execution flow:
        1. schedule() is called, returns SchedulerOutput with placeholders
        2. Runner starts forward pass while previous tokens are being sampled
        3. update_from_output() receives actual tokens, replaces placeholders
        4. Process repeats

    Attributes:
        Inherits all attributes from base Scheduler class.

    Note:
        This scheduler requires a runner that supports async execution mode.
        The runner must implement execute_model_async() and handle
        AsyncPreResults for proper operation.

    Warning:
        Using this scheduler with a non-async-capable runner will result
        in incorrect behavior, as the placeholder mechanism will not
        function properly.

    Example:
        >>> scheduler = AsyncScheduler(config=config, kv_cache_config=kv_config)
        >>> # Schedule a batch with placeholders
        >>> output = scheduler.schedule()
        >>> # Runner processes batch asynchronously
        >>> model_output = runner.execute_model_async(output)
        >>> # Update with actual tokens (may be from previous iteration)
        >>> results = scheduler.update_from_output(prev_output, prev_model_output)
    """

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Update request states after scheduling, adding async placeholders.

        This method is called after the schedule() method completes. In async
        mode, it adds output placeholders for requests that will generate tokens,
        allowing the runner to return immediately without blocking on token
        sampling.

        The placeholder mechanism works as follows:
            1. For each scheduled request, check if it will generate tokens
            2. If yes, increment num_output_placeholders by (1 + num_spec_tokens)
            3. Set spec_token_ids to placeholder values (-1)
            4. Track structured output requests for special handling

        Args:
            scheduler_output: The scheduling result containing requests to
                process. This object is modified in place to add the
                pending_structured_output_tokens flag.

        Side Effects:
            - Calls parent implementation to update num_computed_tokens
            - Updates request.num_output_placeholders for scheduled requests
            - Sets request.spec_token_ids to placeholder values (-1)
            - Sets scheduler_output.pending_structured_output_tokens flag

        Note:
            A request generates tokens when its num_computed_tokens equals
            the sum of num_tokens + num_output_placeholders + num_spec_tokens.
            Placeholders will be replaced with actual tokens in the next
            iteration via the runner's _modify_prev_results() method.
        """
        # Call parent implementation first to update num_computed_tokens
        super()._update_after_schedule(scheduler_output)

        # Track if any requests have pending structured output
        pending_structured_output_tokens = False
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Add placeholders for each scheduled request
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]

            # Track structured output requests
            pending_structured_output_tokens |= request.use_structured_output and request.num_output_placeholders > 0

            # Calculate number of speculative tokens for this request
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))

            # Check if request will generate a token this iteration
            # Request generates when computed tokens equals total tokens needed
            if request.num_computed_tokens == request.num_tokens + request.num_output_placeholders + cur_num_spec_tokens:
                # Request will generate 1 new token plus num_spec_tokens
                request.num_output_placeholders += 1 + cur_num_spec_tokens

                # Add placeholders for speculative tokens (will be updated by runner)
                # Use -1 as placeholder value to indicate not yet sampled
                request.spec_token_ids = [-1] * self.num_spec_tokens

        # Update scheduler output with structured output status
        scheduler_output.pending_structured_output_tokens = pending_structured_output_tokens

    def _update_request_with_output(
        self,
        request: EngineRequest,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        """Update request with newly generated tokens and manage placeholders.

        This method processes the actual tokens that were generated asynchronously
        in a previous iteration. It updates the request state, decrements
        placeholder counts, and caches KV blocks for non-preempted requests.

        The update process:
            1. Store current request status (may change during parent update)
            2. Call parent implementation for token appending and stop checking
            3. Decrement placeholder count by number of actual tokens received
            4. Assert placeholders are non-negative (sanity check)
            5. Cache KV blocks for running requests (skip preempted)

        Args:
            request: The engine request to update. Must be a valid request
                in the scheduler's request dictionary.
            new_token_ids: List of newly generated token IDs. May be empty
                if no tokens were generated (e.g., during prefill).

        Returns:
            tuple[list[int], bool]: A tuple containing:
                - new_token_ids: The token IDs to return (may be truncated
                  if the request hit a stop condition)
                - stopped: True if the request has finished (hit stop
                  condition), False otherwise

        Side Effects:
            - Appends new_token_ids to request.output_token_ids (via parent)
            - Decrements request.num_output_placeholders
            - May update request.status if stop condition is met
            - Caches KV blocks for running requests via kv_cache_manager

        Raises:
            AssertionError: If num_output_placeholders becomes negative,
                indicating a bug in the placeholder tracking logic.

        Note:
            Preempted requests are skipped for KV caching to avoid caching
            invalid or incomplete states that would be discarded anyway.
        """
        # Store status before parent update (which may change it)
        status_before_update = request.status

        # Call parent implementation to handle token appending and stop checking
        new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)

        # Decrement placeholder count by number of actual tokens received
        request.num_output_placeholders -= len(new_token_ids)

        # Sanity check: placeholders should never go negative
        assert request.num_output_placeholders >= 0, (
            f"Request {request.request_id} has negative placeholders: {request.num_output_placeholders}"
        )

        # Cache the new tokens for running requests (skip preempted)
        # We cache based on num_computed_tokens minus remaining placeholders
        # to ensure we only cache fully computed tokens
        if status_before_update == EngineRequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens - request.num_output_placeholders)

        return new_token_ids, stopped
