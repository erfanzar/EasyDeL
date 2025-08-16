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

"""Request scheduling components for the eSurge engine.

This module provides the scheduling infrastructure for managing and batching
inference requests. It includes request queues, scheduling algorithms, and
output structures.

Classes:
    Scheduler: Main request scheduler
    SchedulerInterface: Abstract scheduler interface
    RequestQueue: Abstract request queue
    FCFSRequestQueue: First-come-first-served queue
    PriorityRequestQueue: Priority-based queue
    SchedulerOutput: Output from scheduling decisions
    NewRequestData: Data for new requests
    CachedRequestData: Data for cached requests

Example:
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
"""

from .interface import SchedulerInterface
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request_queue import FCFSRequestQueue, PriorityRequestQueue, RequestQueue
from .scheduler import Scheduler

__all__ = (
    "CachedRequestData",
    "FCFSRequestQueue",
    "NewRequestData",
    "PriorityRequestQueue",
    "RequestQueue",
    "Scheduler",
    "SchedulerInterface",
    "SchedulerOutput",
)
