from __future__ import annotations

from dataclasses import dataclass

from ..core.manager import CacheManager


@dataclass
class TokenBudgetManager:
    """Utility that keeps batch-level token usage in sync with KV cache capacity."""

    max_batch_tokens: int
    page_size: int
    safety_margin_tokens: int

    def __post_init__(self) -> None:
        self._remaining = self.max_batch_tokens

    @property
    def remaining(self) -> int:
        return self._remaining

    def begin_cycle(self, cache_manager: CacheManager, num_running_requests: int) -> int:
        """Refresh the budget using latest KV cache statistics."""
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
        """Consume tokens from the budget, clamping to the remaining capacity."""
        if requested_tokens <= 0 or self._remaining <= 0:
            return 0
        grant = min(requested_tokens, self._remaining)
        self._remaining -= grant
        return grant
