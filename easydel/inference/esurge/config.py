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

"""Configuration classes for the eSurge inference engine.

This module provides configuration dataclasses for controlling the behavior
of the eSurge engine's scheduler, KV cache, and speculative decoding.

Classes:
    SchedulerConfig: Configuration for request scheduling and batching.
    CacheConfig: Configuration for KV cache memory management.
    SpeculativeConfig: Configuration for speculative decoding (Eagle).
    Config: Unified configuration combining all subsystems.

Constants:
    LONG_PREFILL_TRS: Default threshold for long prefill detection (2048 tokens).

Example:
    >>> from easydel.inference.esurge.config import (
    ...     Config,
    ...     SchedulerConfig,
    ...     CacheConfig
    ... )
    >>>
    >>> # Create scheduler config
    >>> scheduler_config = SchedulerConfig(
    ...     max_num_seqs=16,
    ...     max_num_batched_tokens=2048,
    ...     max_model_len=8192,
    ...     policy="fcfs"
    ... )
    >>>
    >>> # Create cache config
    >>> cache_config = CacheConfig(
    ...     num_pages=1000,
    ...     page_size=128,
    ...     enable_prefix_caching=True
    ... )
    >>>
    >>> # Combine into unified config
    >>> config = Config(
    ...     scheduler_config=scheduler_config,
    ...     cache_config=cache_config
    ... )
"""

from dataclasses import dataclass
from typing import Literal

LONG_PREFILL_TRS: int = 2048


@dataclass
class SchedulerConfig:
    """Configuration for the request scheduler.

    Controls how requests are scheduled and batched for processing,
    including capacity limits, scheduling policy, and advanced features
    like chunked prefill and async scheduling.

    Attributes:
        max_num_seqs: Maximum number of sequences running simultaneously.
            This limits the batch size in terms of requests.
        max_num_batched_tokens: Maximum tokens processed in a single batch.
            This limits the total compute per forward pass.
        max_model_len: Maximum input length the model can handle.
            Requests exceeding this will be rejected or truncated.
        policy: Scheduling policy ('fcfs' for first-come-first-served,
            'priority' for priority-based). Defaults to 'fcfs'.
        long_prefill_token_threshold: Token count threshold for identifying
            long prefill requests. Requests above this threshold may be
            chunked. Defaults to max_num_batched_tokens.
        chunked_prefill_enabled: Enable chunked processing of long prefill
            requests to prevent head-of-line blocking.
        token_safety_margin: Reserved tokens per running request to prevent
            over-allocation and OOM errors. Defaults to None.
        max_num_seq_buckets: Optional explicit request-capacity buckets for
            compilation (e.g., (8, 16, 32, 64)). Helps reduce JIT recompilation.
        async_scheduling: Enable async token sampling to overlap with next
            forward pass, providing 30-40% latency reduction.

    Example:
        >>> config = SchedulerConfig(
        ...     max_num_seqs=16,
        ...     max_num_batched_tokens=2048,
        ...     max_model_len=8192,
        ...     policy="priority",
        ...     chunked_prefill_enabled=True
        ... )

    Raises:
        ValueError: If configuration parameters are invalid (negative values,
            incompatible settings, etc.).
    """

    max_num_seqs: int
    """The maximum number of sequences running at the same time."""

    max_num_batched_tokens: int
    """The maximum number of tokens to be processed in a single batch."""

    max_model_len: int
    """The maximum length of the model's input."""

    policy: Literal["priority", "fcfs"] = "fcfs"
    """The scheduling policy to use, such as 'priority' or 'fcfs'."""

    long_prefill_token_threshold: int | None = None
    """A token threshold for handling long prefill requests (this can overwrite the max_num_batched_tokens)."""

    chunked_prefill_enabled: bool = False
    """A flag to enable or disable chunked prefilling."""

    token_safety_margin: int | None = None
    """Reserved tokens per running request to prevent over-allocation."""

    max_num_seq_buckets: tuple[int, ...] | None = None
    """Optional explicit request-capacity buckets (e.g., (8, 16, 32, 64))."""

    async_scheduling: bool = False
    """Enable async token sampling to overlap with next forward pass (30-40% latency reduction)."""

    def __post_init__(self):
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
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
        if self.long_prefill_token_threshold is None:
            if self.max_num_batched_tokens is not None:
                self.long_prefill_token_threshold = self.max_num_batched_tokens
            else:
                self.long_prefill_token_threshold = LONG_PREFILL_TRS
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
    The cache uses page-based allocation for efficient memory management and
    optional prefix caching for sharing common prefixes between sequences.

    Attributes:
        num_pages: Number of GPU pages allocated for cache. Set to None for
            automatic calculation based on available memory.
        page_size: Size of each cache page in tokens. Recommended >=256 for
            GPUs to ensure efficient memory access patterns.
        enable_prefix_caching: Enable caching of common prefixes across requests.
            This can significantly improve throughput for similar prompts.

    Example:
        >>> config = CacheConfig(
        ...     num_pages=1000,
        ...     page_size=128,
        ...     enable_prefix_caching=True
        ... )

    Note:
        Page-based allocation allows efficient memory management and
        sharing of cache blocks between sequences. Larger page sizes
        reduce metadata overhead but may waste memory for short sequences.

    Raises:
        ValueError: If page_size is not positive or num_pages is invalid.
    """

    num_pages: int | None
    """The number of GPU pages allocated for the cache."""

    page_size: int
    """The size of each cache page."""

    enable_prefix_caching: bool
    """A flag to enable or disable prefix caching."""

    def __post_init__(self):
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")

        if self.num_pages is not None and self.num_pages <= 0:
            raise ValueError(f"num_pages must be positive when specified, got {self.num_pages}")


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Speculative decoding uses a smaller draft model to predict multiple
    tokens ahead, which are then verified by the main model in parallel.
    This can significantly improve throughput for autoregressive generation.

    Attributes:
        num_speculative_tokens: Number of speculative tokens to generate
            per verification step. Higher values may improve throughput
            but reduce acceptance rate. Defaults to 0 (disabled).
        speculative_model: Path to the speculative draft model (e.g., an
            Eagle model trained for the base model). Required when
            num_speculative_tokens > 0.

    Example:
        >>> config = SpeculativeConfig(
        ...     num_speculative_tokens=5,
        ...     speculative_model="path/to/eagle-model"
        ... )

    Note:
        Currently supports Eagle-style speculative decoding. The draft
        model must be compatible with the base model's vocabulary.
    """

    num_speculative_tokens: int = 0
    """Number of speculative tokens to generate per step."""

    speculative_model: str | None = None
    """Path to the speculative/draft model."""

    def use_eagle(self) -> bool:
        """Check if Eagle speculative decoding is enabled.

        Returns:
            True if both num_speculative_tokens > 0 and a speculative_model
            is specified, False otherwise.
        """
        return self.num_speculative_tokens > 0 and self.speculative_model is not None


@dataclass
class Config:
    """Unified configuration for the eSurge engine.

    Combines scheduler, cache, and speculative decoding configurations
    into a single object for easy management and passing to the engine.

    Attributes:
        scheduler_config: Configuration for request scheduling and batching.
        cache_config: Configuration for KV cache memory management.
        speculative_config: Optional configuration for speculative decoding.
            Defaults to None (no speculative decoding).

    Example:
        >>> from easydel.inference.esurge.config import (
        ...     Config, SchedulerConfig, CacheConfig, SpeculativeConfig
        ... )
        >>>
        >>> config = Config(
        ...     scheduler_config=SchedulerConfig(
        ...         max_num_seqs=16,
        ...         max_num_batched_tokens=2048,
        ...         max_model_len=8192
        ...     ),
        ...     cache_config=CacheConfig(
        ...         num_pages=1000,
        ...         page_size=128,
        ...         enable_prefix_caching=True
        ...     ),
        ...     speculative_config=SpeculativeConfig(
        ...         num_speculative_tokens=5,
        ...         speculative_model="eagle-model"
        ...     )
        ... )
    """

    scheduler_config: SchedulerConfig
    """Nested configuration for the scheduler."""

    cache_config: CacheConfig
    """Nested configuration for the cache."""

    speculative_config: SpeculativeConfig | None = None
    """Nested configuration for speculative decoding."""
