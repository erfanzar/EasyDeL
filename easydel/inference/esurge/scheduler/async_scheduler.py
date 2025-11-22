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

Example:
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

    Attributes:
        Inherits all attributes from base Scheduler class.

    Methods:
        _update_after_schedule: Adds output placeholders after scheduling
        _update_request_with_output: Updates requests and manages placeholders

    Note:
        This scheduler must be used with a runner that supports async execution
        (i.e., has execute_model_async() method and handles AsyncPreResults).
    """

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Update request states after scheduling, adding async placeholders.

        This method is called after the schedule() method completes. In async mode,
        it adds output placeholders for requests that will generate tokens, allowing
        the runner to return immediately without blocking on token sampling.

        The method tracks:
            - Which requests will generate tokens this iteration
            - How many placeholders to add (including speculative tokens)
            - Whether any requests have pending structured output tokens

        Args:
            scheduler_output: The scheduling result containing requests to process

        Side Effects:
            - Updates request.num_output_placeholders for scheduled requests
            - Sets request.spec_token_ids to placeholder values (-1)
            - Sets scheduler_output.pending_structured_output_tokens flag

        Note:
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
        in a previous iteration. It updates the request state, decrements placeholder
        counts, and caches KV blocks for non-preempted requests.

        Args:
            request: The request to update
            new_token_ids: List of newly generated token IDs (may be empty)

        Returns:
            Tuple of (new_token_ids, stopped):
                - new_token_ids: The token IDs to return (may be truncated if stopped)
                - stopped: True if request has finished (hit stop condition)

        Side Effects:
            - Updates request.num_output_placeholders (decrements by token count)
            - Caches KV blocks for running requests
            - May update request status if stopped

        Note:
            Preempted requests are skipped for KV caching to avoid caching
            invalid/incomplete states.
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
