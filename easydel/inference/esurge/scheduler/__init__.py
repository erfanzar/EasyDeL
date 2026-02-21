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

"""Request scheduling components for the eSurge inference engine.

This module provides the scheduling infrastructure for managing and batching
inference requests. It includes request queues, scheduling algorithms, and
output structures.

The scheduler module is responsible for:
    - Managing request queues with different scheduling policies (FCFS, Priority)
    - Allocating KV cache pages to requests
    - Batching requests for efficient GPU utilization
    - Handling request preemption when resources are constrained
    - Managing async token generation for improved throughput

Classes:
    Scheduler: Main request scheduler that handles batching and resource allocation.
    AsyncScheduler: Async scheduler with placeholder-based token sampling for
        overlapping execution.
    SchedulerInterface: Abstract base class defining the scheduler interface.
    RequestQueue: Abstract base class for request queue implementations.
    FCFSRequestQueue: First-come-first-served queue implementation.
    PriorityRequestQueue: Priority-based queue with heap-based ordering.
    SchedulerOutput: Data class containing the output of a scheduling decision.
    NewRequestData: Data class for new requests being scheduled.
    CachedRequestData: Data class for cached/running requests being scheduled.
    TokenBudgetManager: Utility for managing token budgets based on KV cache capacity.

Example:
    Basic scheduler usage::

        >>> from easydel.inference.esurge.scheduler import Scheduler
        >>> from easydel.inference.esurge.config import SchedulerConfig
        >>>
        >>> config = SchedulerConfig(
        ...     max_num_seqs=16,
        ...     max_num_batched_tokens=2048,
        ...     max_model_len=8192
        ... )
        >>> scheduler = Scheduler(config)
        >>> output = scheduler.schedule()

    For async scheduling with improved throughput::

        >>> config_async = SchedulerConfig(
        ...     max_num_seqs=16,
        ...     max_num_batched_tokens=2048,
        ...     max_model_len=8192,
        ...     async_scheduling=True
        ... )
        >>> async_scheduler = AsyncScheduler(config_async)
        >>> output = async_scheduler.schedule()

Note:
    The scheduler should be created using the appropriate configuration and
    KV cache configuration. For most use cases, creating a Scheduler from
    an eSurgeRunner using ``Scheduler.from_runner()`` is recommended.
"""

from .async_scheduler import AsyncScheduler
from .interface import SchedulerInterface
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request_queue import FCFSRequestQueue, PriorityRequestQueue, RequestQueue
from .scheduler import Scheduler
from .token_budget import TokenBudgetManager

__all__ = (
    "AsyncScheduler",
    "CachedRequestData",
    "FCFSRequestQueue",
    "NewRequestData",
    "PriorityRequestQueue",
    "RequestQueue",
    "Scheduler",
    "SchedulerInterface",
    "SchedulerOutput",
    "TokenBudgetManager",
)
