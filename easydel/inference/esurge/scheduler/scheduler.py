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

"""Main scheduler implementation for the eSurge inference engine.

This module provides the core Scheduler class that manages request batching,
KV cache allocation, and scheduling decisions for the inference engine.

The scheduler is responsible for:
    - Managing waiting and running request queues
    - Allocating KV cache pages to requests
    - Deciding which requests to include in each batch
    - Handling request preemption when resources are constrained
    - Processing model outputs and updating request states

Classes:
    Scheduler: Main request scheduler implementation.

Example:
    Creating a scheduler from configuration::

        >>> from easydel.inference.esurge.config import Config, SchedulerConfig, CacheConfig
        >>> from easydel.inference.esurge.scheduler import Scheduler
        >>>
        >>> config = Config(
        ...     scheduler_config=SchedulerConfig(
        ...         max_num_seqs=16,
        ...         max_num_batched_tokens=2048,
        ...         max_model_len=8192
        ...     ),
        ...     cache_config=CacheConfig(
        ...         num_pages=1000,
        ...         page_size=16
        ...     )
        ... )
        >>> scheduler = Scheduler(config=config, kv_cache_config=kv_cache_config)

    Creating from an eSurgeRunner (recommended)::

        >>> scheduler = Scheduler.from_runner(
        ...     runner=runner,
        ...     enable_prefix_caching=True
        ... )
"""

from __future__ import annotations

import itertools
import time
import typing
from collections import defaultdict
from collections.abc import Iterable

from eformer.loggings import get_logger

from ..config import Config
from ..core.dp_sharding import dp_shard_for_page_id, pages_per_dp_shard
from ..core.interface import CacheGroupsConfig
from ..core.manager import CacheManager
from ..engine_types import EngineCoreOutput, EngineCoreOutputs
from ..metrics import get_metrics_collector
from ..outputs import ModelRunnerOutput
from ..request import EngineRequest, EngineRequestStatus
from .interface import SchedulerInterface
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request_queue import SchedulingPolicy, create_request_queue
from .token_budget import TokenBudgetManager
from .utils import check_stop

if typing.TYPE_CHECKING:
    from ..runners.model_runner import eSurgeRunner

logger = get_logger("eSurgeScheduler")


class Scheduler(SchedulerInterface):
    """Main request scheduler for the eSurge inference engine.

    The Scheduler manages the lifecycle of inference requests, from receiving
    them to completion. It handles batching, KV cache allocation, preemption,
    and coordinates with the model runner.

    Key responsibilities:
        - Managing waiting and running request queues
        - Allocating and freeing KV cache pages
        - Making scheduling decisions respecting token budgets
        - Handling request preemption under memory pressure
        - Processing model outputs and detecting completion

    The scheduling algorithm:
        1. Process running requests first (decode phase)
        2. Allocate tokens for each running request up to budget
        3. Preempt requests if memory is insufficient
        4. Schedule waiting requests (prefill phase) if budget remains
        5. Return batch information for model runner

    Attributes:
        config: The complete engine configuration.
        scheduler_config: Scheduler-specific configuration.
        cache_config: KV cache configuration.
        kv_cache_config: KV cache groups configuration.
        max_num_running_reqs: Maximum concurrent requests.
        max_num_scheduled_tokens: Maximum tokens per batch.
        max_model_len: Maximum sequence length.
        page_size: Tokens per KV cache page.
        requests: Dictionary mapping request_id to EngineRequest.
        policy: The scheduling policy (FCFS or PRIORITY).
        waiting: Queue of waiting requests.
        running: List of currently running requests.
        finished_req_ids: Set of recently finished request IDs.
        kv_cache_manager: Manager for KV cache allocation.
        use_eagle: Whether EAGLE speculative decoding is enabled.
        num_spec_tokens: Number of speculative tokens.
        max_num_seq_buckets: Bucket sizes for batch optimization.

    Example:
        >>> scheduler = Scheduler(config=config, kv_cache_config=kv_cache_config)
        >>> scheduler.add_request(request)
        >>> output = scheduler.schedule()
        >>> # ... run model ...
        >>> results = scheduler.update_from_output(output, model_output)
    """

    def __init__(
        self,
        config: Config,
        kv_cache_config: CacheGroupsConfig,
        include_finished_set: bool = False,
        max_num_seq_buckets: list[int] | None = None,
    ) -> None:
        """Initialize the Scheduler with configuration.

        Sets up the scheduler with the provided configuration, initializing
        request queues, KV cache manager, and scheduling parameters.

        Args:
            config: Complete engine configuration containing scheduler_config,
                cache_config, and optionally speculative_config.
            kv_cache_config: Configuration for KV cache groups including
                num_pages and kv_cache_groups specifications.
            include_finished_set: If True, track finished request IDs per
                client for multi-client scenarios. Defaults to False.
            max_num_seq_buckets: Optional list of bucket sizes for batch
                optimization. If None, uses values from scheduler_config
                or defaults to [max_num_seqs].

        Raises:
            ValueError: If an unknown scheduling policy is specified.
            AssertionError: If num_pages is not positive.

        Example:
            >>> scheduler = Scheduler(
            ...     config=config,
            ...     kv_cache_config=kv_cache_config,
            ...     include_finished_set=True  # For multi-client
            ... )
        """
        self.config = config
        self.scheduler_config = config.scheduler_config
        self.cache_config = config.cache_config
        self.kv_cache_config = kv_cache_config

        self.finished_req_ids_dict: dict[int, set[str]] | None = defaultdict(set) if include_finished_set else None

        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        if self.max_num_scheduled_tokens is None:
            # Keep runtime behavior aligned with engine docs: unset budget
            # falls back to model context length.
            self.max_num_scheduled_tokens = self.max_model_len
        self.data_parallel_size = 1
        num_pages = self.cache_config.num_pages
        assert num_pages is not None and num_pages > 0

        self.page_size = self.cache_config.page_size
        safety_margin = self.scheduler_config.token_safety_margin
        if safety_margin is None:
            self._token_budget_manager = None
        else:
            self._token_budget_manager = TokenBudgetManager(
                max_batch_tokens=self.max_num_scheduled_tokens,
                page_size=self.page_size,
                safety_margin_tokens=safety_margin,
            )

        self.requests: dict[str, EngineRequest] = {}

        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(f"Unknown scheduling policy: {self.scheduler_config.policy}")

        self.waiting = create_request_queue(self.policy)
        self.running: list[EngineRequest] = []
        # Maintained from model-runner outputs; used to keep DP-shard hints
        # aligned with actual sequence-buffer row placement.
        self.req_id_to_row_index: dict[str, int] = {}

        self.finished_req_ids: set[str] = set()

        self.finished_recving_kv_req_ids: set[str] = set()

        speculative_config = config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        self.kv_cache_manager = CacheManager(
            num_pages=num_pages,
            kv_cache_groups=kv_cache_config.kv_cache_groups,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
        )

        buckets = max_num_seq_buckets or list(self.scheduler_config.max_num_seq_buckets or ())
        if not buckets:
            buckets = [self.max_num_running_reqs]
        buckets = sorted({int(b) for b in buckets if b > 0})
        if not buckets:
            buckets = [self.max_num_running_reqs]
        if buckets[-1] != self.max_num_running_reqs:
            buckets.append(self.max_num_running_reqs)
        self.max_num_seq_buckets = buckets
        self._current_seq_bucket = self._select_seq_bucket(0)

    def _dp_shard_hint_for_row(self, row_index: int) -> int | None:
        """Compute a DP shard hint for a logical request row index."""
        dp_size = int(getattr(self, "data_parallel_size", 1))
        if dp_size <= 1:
            return None
        rows_cap = max(1, int(self.max_num_seq_buckets[-1]))
        rows_per_shard = max(1, rows_cap // dp_size)
        return min(max(int(row_index), 0) // rows_per_shard, dp_size - 1)

    @classmethod
    def from_runner(
        cls,
        runner: eSurgeRunner,
        max_num_batched_tokens: int | None = None,
        enable_prefix_caching: bool = True,
    ) -> Scheduler:
        """Create a Scheduler instance from an eSurgeRunner.

        This factory method automatically detects the model's attention types
        (full, sliding window, chunked) from the model config and creates
        appropriate cache specifications. This is the recommended way to
        create a Scheduler for most use cases.

        Args:
            runner: The eSurgeRunner instance containing model and metadata.
                Must have metadata with page_size, num_kv_heads, head_dim,
                num_pages, and kvdtype attributes.
            max_num_batched_tokens: Maximum tokens per batch. If None,
                defaults to runner.max_model_len.
            enable_prefix_caching: Whether to enable prefix caching for
                faster inference on repeated prefixes. Defaults to True.

        Returns:
            Scheduler: A configured Scheduler instance ready for use.

        Example:
            >>> scheduler = Scheduler.from_runner(
            ...     runner=runner,
            ...     max_num_batched_tokens=4096,
            ...     enable_prefix_caching=True
            ... )
        """
        from ..config import CacheConfig, SchedulerConfig
        from ..core.interface import create_kv_cache_specs_from_config

        metadata = runner.metadata
        model_config = runner.model.config

        if max_num_batched_tokens is None:
            max_num_batched_tokens = runner.max_model_len

        kv_cache_groups = create_kv_cache_specs_from_config(
            config=model_config,
            page_size=metadata.page_size,
            num_kv_heads=metadata.num_kv_heads,
            head_size=getattr(metadata, "k_headdim", None) or getattr(metadata, "head_dim", None),
            dtype=metadata.kvdtype,
            use_mla=False,
        )

        scheduler = Scheduler(
            config=Config(
                scheduler_config=SchedulerConfig(
                    max_num_seqs=runner.max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_model_len=runner.max_model_len,
                    max_num_seq_buckets=tuple(runner.max_num_seq_buckets),
                ),
                cache_config=CacheConfig(
                    num_pages=metadata.num_pages,
                    page_size=metadata.page_size,
                    enable_prefix_caching=enable_prefix_caching,
                ),
            ),
            kv_cache_config=CacheGroupsConfig(num_pages=metadata.num_pages, kv_cache_groups=kv_cache_groups),
        )
        scheduler.data_parallel_size = int(getattr(metadata, "data_parallel_size", 1) or 1)
        return scheduler

    def _select_seq_bucket(self, num_reqs: int) -> int:
        """Select the appropriate sequence bucket for the given request count.

        Buckets allow for more efficient memory allocation by pre-allocating
        buffers for common batch sizes.

        Args:
            num_reqs: The number of requests to find a bucket for.

        Returns:
            int: The smallest bucket size that can accommodate num_reqs,
                or the largest bucket if num_reqs exceeds all buckets.
        """
        if num_reqs <= 0:
            return self.max_num_seq_buckets[0]
        for bucket in self.max_num_seq_buckets:
            if num_reqs <= bucket:
                return bucket
        return self.max_num_seq_buckets[-1]

    def _ensure_capacity(self, desired_running: int) -> bool:
        """Ensure capacity for the desired number of running requests.

        Updates the current sequence bucket and checks if the desired
        number of running requests can be accommodated.

        Args:
            desired_running: The desired number of running requests.

        Returns:
            bool: True if the desired number fits within the selected bucket,
                False otherwise.
        """
        bucket = self._select_seq_bucket(desired_running)
        self._current_seq_bucket = bucket
        return desired_running <= bucket

    def schedule(self) -> SchedulerOutput:
        """Schedule requests for the next forward pass.

        This is the main scheduling method that decides which requests to
        process and how many tokens for each. It handles both running
        requests (decode phase) and waiting requests (prefill phase).

        The scheduling algorithm:
            1. Initialize token budget from cache manager
            2. Process running requests, allocating tokens up to budget
            3. Preempt requests if memory allocation fails
            4. If no preemptions, schedule waiting requests with remaining budget
            5. Build and return SchedulerOutput with batch information

        Returns:
            SchedulerOutput: Contains all information needed by the model
                runner to execute the batch, including:
                - scheduled_new_reqs: New requests being scheduled
                - scheduled_cached_reqs: Continuing requests
                - num_scheduled_tokens: Token counts per request
                - total_num_scheduled_tokens: Total batch tokens
                - suggested_bucket: Optimal batch size hint

        Note:
            This method also records metrics for monitoring scheduler
            performance and cache utilization.
        """
        schedule_start_time = time.time()
        scheduled_new_reqs: list[EngineRequest] = []
        scheduled_resumed_reqs: list[EngineRequest] = []
        scheduled_running_reqs: list[EngineRequest] = []
        preempted_reqs: list[EngineRequest] = []

        req_to_new_page_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        if self._token_budget_manager:
            token_budget = self._token_budget_manager.begin_cycle(self.kv_cache_manager, len(self.running))
        else:
            assert self.max_num_scheduled_tokens is not None
            token_budget = self.max_num_scheduled_tokens

        dp_size = max(1, int(getattr(self, "data_parallel_size", 1) or 1))
        num_pages = int(getattr(self.cache_config, "num_pages", 0) or 0)
        pages_per_shard = pages_per_dp_shard(num_pages, dp_size)
        use_dp_local_shard_hints = (
            dp_size > 1
            and int(self.max_num_running_reqs) > 0
            and int(self.max_num_running_reqs) % dp_size == 0
            and pages_per_shard is not None
        )
        rows_per_shard = int(self.max_num_running_reqs) // dp_size if use_dp_local_shard_hints else 0
        pages_per_shard = int(pages_per_shard) if use_dp_local_shard_hints else 0

        _shard_occupancy: list[int] = [0] * dp_size if use_dp_local_shard_hints else []
        if use_dp_local_shard_hints:
            for _req in self.running:
                _row = self.req_id_to_row_index.get(_req.request_id)
                if _row is not None:
                    _s = min(max(int(_row), 0) // rows_per_shard, dp_size - 1)
                    _shard_occupancy[_s] += 1

        def _row_to_dp_shard(row_index: int | None) -> int | None:
            if not use_dp_local_shard_hints or row_index is None:
                return None
            return min(max(int(row_index), 0) // rows_per_shard, dp_size - 1)

        def _infer_dp_shard_from_pages(request: EngineRequest) -> int | None:
            if not use_dp_local_shard_hints:
                return None
            inferred: int | None = None
            for group_page_ids in self.kv_cache_manager.get_page_ids(request.request_id):
                for page_id in group_page_ids:
                    pid = int(page_id)
                    if pid <= 0:
                        continue
                    shard = dp_shard_for_page_id(pid, pages_per_shard, dp_size)
                    if shard is None:
                        continue
                    if inferred is None:
                        inferred = shard
                    elif inferred != shard:
                        return None
            return inferred

        def _pick_running_shard(request: EngineRequest) -> int | None:
            """Shard hint for a RUNNING request — always its existing shard."""
            if not use_dp_local_shard_hints:
                return None
            shard = _infer_dp_shard_from_pages(request)
            if shard is not None:
                return shard
            shard = _row_to_dp_shard(self.req_id_to_row_index.get(request.request_id))
            if shard is not None:
                return shard
            return None

        def _pick_new_shard(request: EngineRequest) -> int | None:
            """Shard hint for a NEW/WAITING request — balanced distribution."""
            if not use_dp_local_shard_hints:
                return None
            shard = _infer_dp_shard_from_pages(request)
            if shard is not None:
                return shard
            shard = _row_to_dp_shard(self.req_id_to_row_index.get(request.request_id))
            if shard is not None:
                return shard
            candidates = [sid for sid in range(dp_size) if _shard_occupancy[sid] < rows_per_shard]
            if not candidates:
                return None
            return min(candidates, key=lambda sid: (_shard_occupancy[sid], sid))

        def _reserve_new_shard(shard_hint: int | None) -> None:
            """Increment shard occupancy when a new request is assigned."""
            if not use_dp_local_shard_hints or shard_hint is None:
                return
            _shard_occupancy[shard_hint] += 1

        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        req_index = 0
        self._ensure_capacity(len(self.running))
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            if (
                self.scheduler_config.long_prefill_token_threshold is not None
                and 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens
            ):
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)

            if num_new_tokens == 0:
                req_index += 1
                continue

            preemption_attempts = 0
            max_preemption_attempts = len(self.running) + 1  # Allow one full cycle plus one
            new_pages = None

            while True:
                row_shard_hint = _pick_running_shard(request)
                new_pages = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    dp_shard_hint=row_shard_hint,
                    data_parallel_size=self.data_parallel_size,
                )
                if new_pages is None:
                    preemption_attempts += 1
                    if preemption_attempts >= max_preemption_attempts:
                        # Cannot allocate even after preempting all requests
                        logger.warning(
                            f"Cannot allocate {num_new_tokens} tokens for request {request.request_id} "
                            f"after {preemption_attempts} preemption attempts. Skipping."
                        )
                        can_schedule = False
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = EngineRequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0

                    # Decrement shard occupancy so new requests can use freed rows
                    if use_dp_local_shard_hints:
                        _preempt_row = self.req_id_to_row_index.get(preempted_req.request_id)
                        if _preempt_row is not None:
                            _preempt_shard = min(max(int(_preempt_row), 0) // rows_per_shard, dp_size - 1)
                            _shard_occupancy[_preempt_shard] = max(0, _shard_occupancy[_preempt_shard] - 1)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                req_index += 1
                continue
            assert new_pages is not None

            scheduled_running_reqs.append(request)
            req_to_new_page_ids[request.request_id] = new_pages.get_page_ids()
            num_scheduled_tokens[request.request_id] = num_new_tokens
            if self._token_budget_manager:
                self._token_budget_manager.consume(num_new_tokens)
                token_budget = self._token_budget_manager.remaining
            else:
                token_budget -= num_new_tokens
            if token_budget <= 0:
                req_index += 1
                break
            req_index += 1

            if request.spec_token_ids:
                num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = request.spec_token_ids
        skipped_waiting_requests = create_request_queue(self.policy)
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if not self._ensure_capacity(len(self.running) + 1):
                    break

                request = self.waiting.peek_request()

                if request.status == EngineRequestStatus.WAITING_FOR_REMOTE_KVS:
                    request.status = EngineRequestStatus.WAITING

                if request.status == EngineRequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = EngineRequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                num_external_computed_tokens = 0
                load_kv_async = False
                row_shard_hint = _pick_new_shard(request)
                if use_dp_local_shard_hints and row_shard_hint is None:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                if request.num_computed_tokens == 0:
                    new_computed_pages, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_pages(
                        request,
                        dp_shard_hint=row_shard_hint,
                        data_parallel_size=self.data_parallel_size,
                    )

                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens

                else:
                    new_computed_pages = self.kv_cache_manager.create_empty_page_list()
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0

                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (
                        self.scheduler_config.long_prefill_token_threshold is not None
                        and 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens
                    ):
                        num_new_tokens = self.scheduler_config.long_prefill_token_threshold

                    if not self.scheduler_config.chunked_prefill_enabled and num_new_tokens > token_budget:
                        # If the request is larger than the max batch size, we MUST allow it if the batch is empty,
                        # otherwise it will never run.
                        # If it's larger than available memory (reflected in token_budget via capacity),
                        # allocate_slots will fail anyway.
                        is_inherently_too_large = (
                            self.max_num_scheduled_tokens is not None and num_new_tokens > self.max_num_scheduled_tokens
                        )
                        is_batch_empty = (
                            len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) == 0
                        )

                        if is_inherently_too_large and is_batch_empty:
                            # Allow it to proceed to allocation
                            pass
                        else:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    if num_new_tokens <= 0:
                        # Avoid starvation: a waiting request with no schedulable
                        # tokens cannot make progress in this queue state.
                        request = self.waiting.pop_request()
                        logger.warning(
                            "Dropping request %s: zero schedulable prompt tokens "
                            "(num_tokens=%d, num_computed_tokens=%d).",
                            request.request_id,
                            request.num_tokens,
                            num_computed_tokens,
                        )
                        request.status = EngineRequestStatus.FINISHED_ABORTED
                        self._free_request(request)
                        continue
                new_pages = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_pages,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    delay_cache_pages=load_kv_async,
                    dp_shard_hint=row_shard_hint,
                    data_parallel_size=self.data_parallel_size,
                )
                if new_pages is None:
                    break

                request = self.waiting.pop_request()
                if load_kv_async:
                    skipped_waiting_requests.prepend_request(request)
                    request.status = EngineRequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if request.status == EngineRequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == EngineRequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")
                _reserve_new_shard(row_shard_hint)

                req_to_new_page_ids[request.request_id] = self.kv_cache_manager.get_page_ids(request.request_id)
                if self._token_budget_manager:
                    self._token_budget_manager.consume(num_new_tokens)
                    token_budget = self._token_budget_manager.remaining
                else:
                    token_budget -= num_new_tokens
                num_scheduled_tokens[request.request_id] = num_new_tokens
                request.status = EngineRequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

                if token_budget <= 0:
                    break

        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Keep row-index affinity for requests that were preempted and resumed in
        # the same schedule step; drop stale mapping only for preempted requests
        # that are not scheduled this step.
        if preempted_reqs:
            for preempted_req in preempted_reqs:
                if preempted_req.request_id not in num_scheduled_tokens:
                    self.req_id_to_row_index.pop(preempted_req.request_id, None)

        self._ensure_capacity(len(self.running))
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert self.max_num_scheduled_tokens is None or total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self._current_seq_bucket

        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)

        num_common_prefix_pages = [0] * len(self.kv_cache_config.kv_cache_groups)
        scheduled_req_count = len(num_scheduled_tokens)

        if scheduled_req_count > 0:
            if scheduled_running_reqs:
                representative_req = scheduled_running_reqs[0]
            elif scheduled_resumed_reqs:
                representative_req = scheduled_resumed_reqs[0]
            elif scheduled_new_reqs:
                representative_req = scheduled_new_reqs[0]
            else:
                representative_req = None

            if representative_req is not None:
                num_common_prefix_pages = self.kv_cache_manager.get_num_common_prefix_pages(
                    representative_req, scheduled_req_count
                )

        new_reqs_data = [
            NewRequestData.from_request(req, req_to_new_page_ids[req.request_id]) for req in scheduled_new_reqs
        ]

        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_page_ids,
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_common_prefix_pages=num_common_prefix_pages,
            finished_req_ids=self.finished_req_ids,
            preempted_req_ids={r.request_id for r in preempted_reqs},
            suggested_bucket=self._current_seq_bucket,  # Hint for runner's buffer selection
            async_scheduling=self.scheduler_config.async_scheduling,  # Pass async config to runner
        )
        self._update_after_schedule(scheduler_output)
        # Log scheduler metrics
        schedule_time = time.time() - schedule_start_time
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_scheduler_metrics(
                num_waiting=len(self.waiting),
                num_running=len(self.running),
                num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
                num_preempted=len(preempted_reqs),
                batch_size=len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs),
                schedule_time=schedule_time,
            )

            # Log cache metrics
            cache_manager = self.kv_cache_manager
            total_pages = cache_manager.num_pages
            used_pages = total_pages - cache_manager.page_pool.get_num_free_pages()

            num_cached_pages = len(cache_manager.page_pool.cached_page_hash_to_page)
            cache_hit_rate = num_cached_pages / max(total_pages, 1)

            metrics_collector.record_cache_metrics(
                total_pages=total_pages,
                used_pages=used_pages,
                cache_hit_rate=cache_hit_rate,
            )

        return scheduler_output

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Update internal state after scheduling completes.

        This method is called at the end of schedule() to update request
        states based on the scheduling decisions made. It updates the
        num_computed_tokens for each scheduled request.

        Args:
            scheduler_output: The scheduling output containing information
                about scheduled requests and token counts.

        Side Effects:
            - Updates num_computed_tokens for each scheduled request
            - Clears finished_req_ids set for next iteration
        """
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[EngineRequest],
        resumed_reqs: list[EngineRequest],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_page_ids: dict[str, tuple[list[int], ...]],
    ) -> CachedRequestData:
        """Build CachedRequestData from running and resumed requests.

        Constructs the batched data structure for requests that are
        continuing execution (not new).

        Args:
            running_reqs: List of requests that were already running.
            resumed_reqs: List of requests resumed from preemption.
            num_scheduled_tokens: Dict mapping request ID to scheduled tokens.
            spec_decode_tokens: Dict mapping request ID to speculative tokens.
            req_to_new_page_ids: Dict mapping request ID to new page IDs.

        Returns:
            CachedRequestData: Batched data for cached requests containing
                request IDs, token IDs, page IDs, and computed token counts.
        """
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_page_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []

        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = num_scheduled_tokens[req_id] - len(spec_decode_tokens.get(req_id, ()))
            token_ids = req.all_token_ids[req.num_computed_tokens : req.num_computed_tokens + num_tokens]
            new_token_ids.append(token_ids)

            new_page_ids.append(req_to_new_page_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_page_ids=new_page_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update scheduler state based on model runner output.

        Processes the tokens generated by the model, updates request states,
        checks for completion, and prepares outputs for clients.

        Args:
            scheduler_output: The scheduling decision that was executed.
            model_runner_output: Output from the model runner containing
                sampled tokens, logprobs, and other metadata.

        Returns:
            dict[int, EngineCoreOutputs]: Mapping from client index to
                EngineCoreOutputs containing per-request outputs. Each
                output includes new tokens, finish reason, and metadata.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        stopped_running_reqs: set[EngineRequest] = set()
        stopped_preempted_reqs: set[EngineRequest] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                continue

            out_index = model_runner_output.req_id_to_index.get(req_id)
            if out_index is None:
                logger.warning(
                    "Skipping scheduler output for request %s: missing req_id_to_index entry in model output.",
                    req_id,
                )
                continue
            row_index_map = model_runner_output.req_id_to_row_index
            row_index = row_index_map.get(req_id) if row_index_map is not None else out_index
            if row_index is not None:
                self.req_id_to_row_index[req_id] = int(row_index)

            if sampled_token_ids and (out_index < 0 or out_index >= len(sampled_token_ids)):
                logger.warning(
                    "Skipping scheduler output for request %s: req_index=%s out of sampled_token_ids range=%s.",
                    req_id,
                    out_index,
                    len(sampled_token_ids),
                )
                continue
            generated_token_ids = sampled_token_ids[out_index] if sampled_token_ids else []
            stopped = False
            new_token_ids = generated_token_ids
            status_before_stop = request.status

            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
            if stopped:
                self._free_request(request)
                if status_before_stop == EngineRequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        if stopped_running_reqs:
            self.running = [req for req in self.running if req not in stopped_running_reqs]
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)

        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            for client_index, finished_set in finished_req_ids.items():
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: EngineRequest,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        """Update a request with newly generated tokens.

        Appends generated tokens to the request and checks for stop conditions.

        Args:
            request: The request to update.
            new_token_ids: List of newly generated token IDs.

        Returns:
            tuple[list[int], bool]: A tuple containing:
                - new_token_ids: Tokens to return (may be truncated if stopped)
                - stopped: True if request hit a stop condition
        """
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]
                break
        return new_token_ids, stopped

    def get_request_counts(self) -> tuple[int, int]:
        """Get the counts of running and waiting requests.

        Returns:
            tuple[int, int]: A tuple of (num_running_reqs, num_waiting_reqs).
        """
        return len(self.running), len(self.waiting)

    def add_request(self, request: EngineRequest) -> None:
        """Add a new request to the scheduler.

        The request is added to the waiting queue and registered in the
        requests dictionary.

        Args:
            request: The engine request to add. Must have a unique request_id.
        """
        self.waiting.add_request(request)
        self.requests[request.request_id] = request

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: EngineRequestStatus,
    ) -> None:
        """Mark requests as finished from external signal.

        Handles finish signals from outside the scheduler, such as client
        disconnection or stop string detection.

        Args:
            request_ids: A single request ID or iterable of request IDs
                to finish.
            finished_status: The finished status to assign. Must be a
                finished status (e.g., FINISHED_ABORTED).
        """
        assert EngineRequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                continue

            valid_requests.append(request)
            if request.status == EngineRequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)

        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: EngineRequest) -> None:
        """Free resources associated with a finished request.

        Records the request as finished and frees its KV cache pages.

        Args:
            request: The finished request to free. Must have is_finished()
                return True.
        """
        assert request.is_finished()

        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        self._free_pages(request)

    def _free_pages(self, request: EngineRequest) -> None:
        """Free KV cache pages for a finished request.

        Releases all KV cache resources and removes the request from
        the requests dictionary.

        Args:
            request: The finished request whose pages should be freed.
        """
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_page_hashes(request)
        self.req_id_to_row_index.pop(request.request_id, None)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        """Get the total number of unfinished requests.

        Returns:
            int: Count of waiting plus running requests.
        """
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        """Check if there are finished requests pending notification.

        Returns:
            bool: True if finished_req_ids is non-empty.
        """
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache.

        Clears all cached prefix entries from the KV cache.

        Returns:
            bool: True if reset was successful.
        """
        return self.kv_cache_manager.reset_prefix_cache()

    def shutdown(self) -> None:
        """Shutdown the scheduler.

        Performs cleanup operations. Currently a no-op but may be extended
        for resource cleanup in future versions.
        """
        ...
