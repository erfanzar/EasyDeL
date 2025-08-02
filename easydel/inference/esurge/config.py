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

from dataclasses import dataclass
from typing import Literal


@dataclass
class SchedulerConfig:
    """
    Configuration for the scheduler.
    """

    max_num_seqs: int
    """The maximum number of sequences running at the same time."""

    max_num_batched_tokens: int
    """The maximum number of tokens to be processed in a single batch."""

    max_model_len: int
    """The maximum length of the model's input."""

    policy: Literal["priority", "fcfs"] = "fcfs"
    """The scheduling policy to use, such as 'priority' or 'fcfs'."""

    long_prefill_token_threshold: int = 256
    """A token threshold for handling long prefill requests."""

    chunked_prefill_enabled: bool = False
    """A flag to enable or disable chunked prefilling."""


@dataclass
class CacheConfig:
    """
    Configuration for the KV cache.
    """

    num_pages: int | None
    """The number of GPU pages allocated for the cache."""

    page_size: int
    """The size of each cache page."""

    enable_prefix_caching: bool
    """A flag to enable or disable prefix caching."""


@dataclass
class Config:
    """
    A unified configuration class.
    """

    scheduler_config: SchedulerConfig
    """Nested configuration for the scheduler."""

    cache_config: CacheConfig
    """Nested configuration for the cache."""
