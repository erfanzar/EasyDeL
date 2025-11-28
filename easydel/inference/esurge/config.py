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
    """Configuration for the request scheduler.

    Controls how requests are scheduled and batched for processing.

    Attributes:
        max_num_seqs: Maximum number of sequences running simultaneously.
        max_num_batched_tokens: Maximum tokens processed in a single batch.
        max_model_len: Maximum input length the model can handle.
        policy: Scheduling policy ('fcfs' for first-come-first-served, 'priority' for priority-based).
        long_prefill_token_threshold: Token count threshold for identifying long prefill requests.
        chunked_prefill_enabled: Enable chunked processing of long prefill requests.

    Example:
        >>> config = SchedulerConfig(
        ...     max_num_seqs=16,
        ...     max_num_batched_tokens=2048,
        ...     max_model_len=8192,
        ...     policy="priority"
        ... )
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

    token_safety_margin: int | None = None
    """Reserved tokens per running request to prevent over-allocation."""

    max_num_seq_buckets: tuple[int, ...] | None = None
    """Optional explicit request-capacity buckets (e.g., (8, 16, 32, 64))."""

    async_scheduling: bool = False
    """Enable async token sampling to overlap with next forward pass (30-40% latency reduction)."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")

        if self.max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens must be positive, got {self.max_num_batched_tokens}")

        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")

        if self.max_num_batched_tokens > self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) cannot exceed "
                f"max_model_len ({self.max_model_len})"
            )

        if self.long_prefill_token_threshold < 0:
            raise ValueError(
                f"long_prefill_token_threshold must be non-negative, got {self.long_prefill_token_threshold}"
            )

        if self.token_safety_margin is not None and self.token_safety_margin < 0:
            raise ValueError(f"token_safety_margin must be non-negative, got {self.token_safety_margin}")

        if self.max_num_seq_buckets is not None:
            if not self.max_num_seq_buckets:
                raise ValueError("max_num_seq_buckets cannot be empty")
            if any(b <= 0 for b in self.max_num_seq_buckets):
                raise ValueError(f"All bucket sizes must be positive, got {self.max_num_seq_buckets}")


@dataclass
class CacheConfig:
    """Configuration for the KV (key-value) cache.

    Manages memory allocation and caching strategies for attention mechanisms.

    Attributes:
        num_pages: Number of GPU pages allocated for cache (None for automatic).
        page_size: Size of each cache page in tokens.
        enable_prefix_caching: Enable caching of common prefixes across requests.

    Example:
        >>> config = CacheConfig(
        ...     num_pages=1000,
        ...     page_size=16,
        ...     enable_prefix_caching=True
        ... )

    Note:
        Page-based allocation allows efficient memory management and
        sharing of cache blocks between sequences.
    """

    num_pages: int | None
    """The number of GPU pages allocated for the cache."""

    page_size: int
    """The size of each cache page."""

    enable_prefix_caching: bool
    """A flag to enable or disable prefix caching."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")

        if self.num_pages is not None and self.num_pages <= 0:
            raise ValueError(f"num_pages must be positive when specified, got {self.num_pages}")


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Attributes:
        num_speculative_tokens: Number of speculative tokens to generate.
        speculative_model: Path to the speculative model (e.g., Eagle model).
    """

    num_speculative_tokens: int = 0
    speculative_model: str | None = None

    def use_eagle(self) -> bool:
        """Check if Eagle speculative decoding is enabled."""
        return self.num_speculative_tokens > 0 and self.speculative_model is not None


@dataclass
class Config:
    """Unified configuration for the eSurge engine.

    Combines scheduler and cache configurations into a single object.

    Attributes:
        scheduler_config: Configuration for request scheduling.
        cache_config: Configuration for KV cache management.
        speculative_config: Configuration for speculative decoding.

    Example:
        >>> config = Config(
        ...     scheduler_config=SchedulerConfig(...),
        ...     cache_config=CacheConfig(...),
        ...     speculative_config=SpeculativeConfig(num_speculative_tokens=5)
        ... )
    """

    scheduler_config: SchedulerConfig
    """Nested configuration for the scheduler."""

    cache_config: CacheConfig
    """Nested configuration for the cache."""

    speculative_config: SpeculativeConfig | None = None
    """Nested configuration for speculative decoding."""
