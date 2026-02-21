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

"""Token budget management for the scheduler.

This module provides utilities for managing token budgets based on KV cache
capacity. The TokenBudgetManager ensures that scheduling decisions respect
both the configured batch size limits and the actual available memory.

Classes:
    TokenBudgetManager: Manages token budgets for batch scheduling.

Example:
    >>> from easydel.inference.esurge.scheduler.token_budget import TokenBudgetManager
    >>> budget_manager = TokenBudgetManager(
    ...     max_batch_tokens=2048,
    ...     page_size=16,
    ...     safety_margin_tokens=64
    ... )
    >>> available = budget_manager.begin_cycle(cache_manager, num_running=5)
    >>> granted = budget_manager.consume(256)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.manager import CacheManager


@dataclass
class TokenBudgetManager:
    """Utility that keeps batch-level token usage in sync with KV cache capacity.

    The TokenBudgetManager calculates and tracks the available token budget
    for each scheduling cycle. It considers both the configured maximum batch
    size and the actual KV cache availability to prevent over-scheduling.

    The budget calculation accounts for:
        - Free pages in the KV cache pool
        - Safety margin tokens reserved per running request
        - Configured maximum batch tokens limit

    Attributes:
        max_batch_tokens: Maximum number of tokens allowed per batch.
        page_size: Number of tokens per KV cache page.
        safety_margin_tokens: Tokens to reserve per running request as a
            safety buffer to prevent memory pressure.

    Example:
        >>> manager = TokenBudgetManager(
        ...     max_batch_tokens=2048,
        ...     page_size=16,
        ...     safety_margin_tokens=64
        ... )
        >>> # At the start of each scheduling cycle
        >>> budget = manager.begin_cycle(cache_manager, num_running=5)
        >>> print(f"Available budget: {budget}")
        >>> # Consume tokens as requests are scheduled
        >>> granted = manager.consume(256)
        >>> print(f"Remaining: {manager.remaining}")
    """

    max_batch_tokens: int
    """Maximum number of tokens allowed per batch."""

    page_size: int
    """Number of tokens per KV cache page."""

    safety_margin_tokens: int
    """Tokens to reserve per running request as a safety buffer."""

    def __post_init__(self) -> None:
        """Initialize the remaining budget to max_batch_tokens.

        This is called automatically after dataclass initialization.
        """
        self._remaining = self.max_batch_tokens

    @property
    def remaining(self) -> int:
        """Get the remaining token budget for the current cycle.

        Returns:
            int: Number of tokens still available in the budget.
        """
        return self._remaining

    def begin_cycle(self, cache_manager: CacheManager, num_running_requests: int) -> int:
        """Refresh the budget using latest KV cache statistics.

        This method should be called at the beginning of each scheduling cycle
        to recalculate the available token budget based on current KV cache
        state.

        The budget is calculated as the minimum of:
            - Configured max_batch_tokens
            - Available KV cache capacity minus safety margin

        Args:
            cache_manager: The cache manager instance to query for page availability.
            num_running_requests: Number of currently running requests (used to
                calculate total safety margin).

        Returns:
            int: The available token budget for this scheduling cycle.

        Example:
            >>> budget = manager.begin_cycle(cache_manager, num_running=5)
            >>> # Now schedule requests up to 'budget' tokens
        """
        free_pages = cache_manager.page_pool.get_num_free_pages()
        available_tokens = free_pages * self.page_size
        reserved_tokens = num_running_requests * self.safety_margin_tokens
        capacity = max(0, available_tokens - reserved_tokens)
        if self.max_batch_tokens is None:
            self._remaining = capacity
        else:
            self._remaining = min(self.max_batch_tokens, capacity)
        return self._remaining

    def consume(self, requested_tokens: int) -> int:
        """Consume tokens from the budget, clamping to the remaining capacity.

        This method attempts to consume the requested number of tokens from the
        budget. If the request exceeds the remaining budget, only the available
        amount is consumed.

        Args:
            requested_tokens: Number of tokens to consume from the budget.

        Returns:
            int: Number of tokens actually consumed (may be less than requested
                if budget is insufficient, or 0 if budget is exhausted).

        Example:
            >>> granted = manager.consume(256)
            >>> if granted < 256:
            ...     print("Budget partially or fully exhausted")
        """
        if requested_tokens <= 0 or self._remaining <= 0:
            return 0
        grant = min(requested_tokens, self._remaining)
        self._remaining -= grant
        return grant
