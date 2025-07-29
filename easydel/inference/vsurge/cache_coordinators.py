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

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Generic, TypeVar

import jax

from easydel.layers.caching import FullAttentionSpec, KVCacheSpec, PagesCacheMetaData

from .cache_manager import FullAttentionManager, SingleTypeKVCacheManager
from .cache_utils import KVCachePage, PageHash
from .page_table import PagePool
from .request_type import EngineRequest

T = TypeVar("T", bound=SingleTypeKVCacheManager)


class KVCacheCoordinator(ABC, Generic[T]):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        metadata: PagesCacheMetaData,
        use_eagle: bool,
        enable_caching: bool,
        caching_hash_fn: Callable,
    ):
        self.metadata = metadata
        self.max_model_len = metadata.max_model_length
        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.caching_hash_fn = caching_hash_fn

        # Initialize page pool
        self.page_pool = PagePool(metadata.page_size, enable_caching)

        # Initialize managers - to be overridden by subclasses
        self._single_type_managers: tuple[T, ...] = self._create_managers()

        # Cache frequently accessed values
        self._num_managers = len(self._single_type_managers)

    @abstractmethod
    def _create_managers(self) -> tuple[T, ...]:
        """Create and return the single type managers."""
        raise NotImplementedError

    @property
    def single_type_managers(self) -> tuple[T, ...]:
        """Get the single type managers."""
        return self._single_type_managers

    def get_num_pages_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_pages: tuple[list[KVCachePage], ...],
    ) -> int:
        """
        Get the number of pages needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot.
            new_computed_pages: The new computed pages just hitting the prefix caching.

        Returns:
            The total number of pages to allocate.
        """
        # Use sum with generator for efficiency
        return sum(
            manager.get_num_pages_to_allocate(request_id, num_tokens, pages)
            for manager, pages in zip(self._single_type_managers, new_computed_pages, strict=False)
        )

    def save_new_computed_pages(
        self,
        request_id: str,
        new_computed_pages: tuple[list[KVCachePage], ...],
    ) -> None:
        """
        Add the new computed pages to the request.

        Args:
            request_id: The request ID.
            new_computed_pages: The new computed pages just hitting the prefix cache.
        """
        for manager, pages in zip(self._single_type_managers, new_computed_pages, strict=False):
            manager.save_new_computed_pages(request_id, pages)

    def allocate_new_pages(self, request_id: str, num_tokens: int) -> tuple[list[KVCachePage], ...]:
        """
        Allocate new pages for the request to give it at least `num_tokens` token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot.

        Returns:
            Tuple of newly allocated pages for each manager.
        """
        return tuple(manager.allocate_new_pages(request_id, num_tokens) for manager in self._single_type_managers)

    def cache_pages(
        self,
        request: EngineRequest,
        page_hashes: list[PageHash],
        num_computed_tokens: int,
    ) -> None:
        """
        Cache the pages for the request.

        Args:
            request: The request.
            page_hashes: The page hashes of the request.
            num_computed_tokens: The total number of tokens to be cached.
        """
        # Cache pages in all managers
        for manager in self._single_type_managers:
            print("Coordiantors -----------> ", page_hashes)
            manager.cache_pages(request, page_hashes, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the pages for the request.

        Args:
            request_id: The request ID.
        """
        # Free pages from all managers
        for manager in self._single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> list[int]:
        """
        Get the number of common prefix pages for a request.

        Args:
            request_id: The request ID.
            num_running_requests: Number of running requests.

        Returns:
            List of common prefix pages per manager.
        """
        return [
            manager.get_num_common_prefix_pages(request_id, num_running_requests)
            for manager in self._single_type_managers
        ]

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        """
        Remove pages that are no longer needed.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        for manager in self._single_type_managers:
            manager.remove_skipped_pages(request_id, num_computed_tokens)

    def get_pages(self, request_id: str) -> tuple[list[KVCachePage], ...]:
        """
        Get the pages for the request.

        Args:
            request_id: The request ID.

        Returns:
            Tuple of pages for each manager.
        """
        return tuple(manager.req_to_pages.get(request_id, []) for manager in self._single_type_managers)

    def has_request(self, request_id: str) -> bool:
        """Check if a request exists in any manager."""
        return any(request_id in manager.req_to_pages for manager in self._single_type_managers)

    def get_total_pages_for_request(self, request_id: str) -> int:
        """Get total number of pages allocated for a request across all managers."""
        return sum(len(manager.req_to_pages.get(request_id, [])) for manager in self._single_type_managers)

    @abstractmethod
    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCachePage], ...], int]:
        """Find the longest cache hit prefix."""
        raise NotImplementedError


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator[FullAttentionManager]):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    Supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        metadata: PagesCacheMetaData,
        use_eagle: bool,
        caching_hash_fn: Callable,
    ):
        # Force disable caching
        super().__init__(metadata, use_eagle, False, caching_hash_fn)

    def _create_managers(self) -> tuple[FullAttentionManager, ...]:
        """Create a single FullAttentionManager with caching disabled."""
        return (
            FullAttentionManager(
                kv_cache_spec=KVCacheSpec(self.metadata.page_size),
                page_pool=self.page_pool,
                kv_cache_group_id=0,
                caching_hash_fn=self.caching_hash_fn,
            ),
        )

    @cached_property
    def num_single_type_manager(self) -> int:
        """Get the number of single type managers."""
        return self._num_managers

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> list[int]:
        """No prefix caching, so always return zeros."""
        return [0] * self._num_managers

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCachePage], ...], int]:
        """No prefix caching, so always return empty pages."""
        empty_pages = tuple([] for _ in range(self._num_managers))
        return empty_pages, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator[FullAttentionManager]):
    """
    KV cache coordinator for models with only one KV cache group.
    Optimized for models where all attention layers use the same attention type.
    """

    def __init__(
        self,
        metadata: PagesCacheMetaData,
        use_eagle: bool,
        enable_caching: bool,
        caching_hash_fn: Callable,
    ):
        # Cache page size before calling super().__init__
        self._page_size = metadata.page_size
        super().__init__(metadata, use_eagle, enable_caching, caching_hash_fn)

    def _create_managers(self) -> tuple[FullAttentionManager, ...]:
        """Create a single FullAttentionManager."""
        return (
            FullAttentionManager(
                kv_cache_spec=self._create_kv_cache_spec(),
                page_pool=self.page_pool,
                kv_cache_group_id=0,
                caching_hash_fn=self.caching_hash_fn,
            ),
        )

    @cached_property
    def page_size(self) -> int:
        """Get the page size."""
        return self._page_size

    @cached_property
    def kv_cache_spec(self) -> FullAttentionSpec:
        """Get the KV cache spec."""
        return self._create_kv_cache_spec()

    def _create_kv_cache_spec(self) -> FullAttentionSpec:
        """Create the KV cache spec."""
        return FullAttentionSpec(
            page_size=self._page_size,
            num_kv_heads=self.metadata.num_kv_heads,
            head_size=self.metadata.k_headdim,
            dtype=jax.numpy.bfloat16,
            use_mla=False,
        )

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCachePage], ...], int]:
        """
        Find the longest cache hit for a unitary cache.

        Args:
            page_hashes: The page hashes to check.
            max_cache_hit_length: Maximum length to check.

        Returns:
            Tuple of (hit_pages, hit_length_in_tokens).
        """
        # Get the single manager
        manager = self._single_type_managers[0]

        # Find cache hit
        hit_pages = manager.find_longest_cache_hit(
            page_hashes=page_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            page_pool=self.page_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
        )

        # Calculate hit length in tokens
        hit_length = len(hit_pages[0]) * self.page_size if hit_pages else 0

        return hit_pages, hit_length

    def get_single_manager(self) -> FullAttentionManager:
        """Get the single manager (convenience method for unitary coordinator)."""
        return self._single_type_managers[0]


class HybridKVCacheCoordinator(KVCacheCoordinator[SingleTypeKVCacheManager]):
    """
    KV cache coordinator for models with multiple KV cache groups.
    Supports models with mixed attention types (e.g., some layers use full attention,
    others use sliding window).
    """

    def __init__(
        self,
        metadata: PagesCacheMetaData,
        kv_cache_specs: list[KVCacheSpec],
        use_eagle: bool,
        enable_caching: bool,
        caching_hash_fn: Callable,
    ):
        self.kv_cache_specs = kv_cache_specs
        super().__init__(metadata, use_eagle, enable_caching, caching_hash_fn)

    def _create_managers(self) -> tuple[SingleTypeKVCacheManager, ...]:
        """Create managers based on KV cache specs."""
        from .cache_manager import ChunkedLocalAttentionManager, FullAttentionManager, SlidingWindowManager

        managers = []
        for i, spec in enumerate(self.kv_cache_specs):
            # Select appropriate manager based on spec type
            if isinstance(spec, FullAttentionSpec):
                manager_class = FullAttentionManager
            elif hasattr(spec, "sliding_window"):  # SlidingWindowSpec
                manager_class = SlidingWindowManager
            elif hasattr(spec, "attention_chunk_size"):  # ChunkedLocalAttentionSpec
                manager_class = ChunkedLocalAttentionManager
            else:
                # Default to FullAttentionManager
                manager_class = FullAttentionManager

            manager = manager_class(
                kv_cache_spec=spec,
                page_pool=self.page_pool,
                kv_cache_group_id=i,
                caching_hash_fn=self.caching_hash_fn,
            )
            managers.append(manager)

        return tuple(managers)

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCachePage], ...], int]:
        """
        Find the longest cache hit across all cache groups.

        Returns the minimum hit length across all groups to ensure consistency.
        """
        all_hit_pages = []
        min_hit_length = max_cache_hit_length

        for i, (manager, spec) in enumerate(zip(self._single_type_managers, self.kv_cache_specs, strict=False)):
            hit_pages = manager.find_longest_cache_hit(
                page_hashes=page_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=[i],
                page_pool=self.page_pool,
                kv_cache_spec=spec,
                use_eagle=self.use_eagle,
            )

            all_hit_pages.append(hit_pages[0])  # Extract the single group's pages

            # Calculate hit length for this group
            group_hit_length = len(hit_pages[0]) * spec.page_size if hit_pages else 0
            min_hit_length = min(min_hit_length, group_hit_length)

        # Ensure all groups have consistent hit length
        consistent_hit_pages = []
        pages_per_group = min_hit_length // self.kv_cache_specs[0].page_size

        for hit_pages in all_hit_pages:
            # Truncate to consistent length
            consistent_pages = hit_pages[:pages_per_group] if hit_pages else []
            consistent_hit_pages.append(consistent_pages)

        return tuple(consistent_hit_pages), min_hit_length

    def get_manager_by_group_id(self, group_id: int) -> SingleTypeKVCacheManager:
        """Get manager by group ID."""
        if 0 <= group_id < len(self._single_type_managers):
            return self._single_type_managers[group_id]
        raise IndexError(f"Invalid group_id {group_id}, valid range: 0-{len(self._single_type_managers) - 1}")


class KVCacheCoordinatorStats:
    """Statistics tracking for KV cache coordinator."""

    def __init__(self, coordinator: KVCacheCoordinator):
        self.coordinator = coordinator
        self.cache_hits = 0
        self.cache_misses = 0
        self.pages_allocated = 0
        self.pages_freed = 0
        self.requests_processed = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def total_pages_in_use(self) -> int:
        """Get total pages currently in use."""
        return sum(len(manager.req_to_pages) for manager in self.coordinator.single_type_managers)

    def record_cache_lookup(self, hit: bool) -> None:
        """Record a cache lookup result."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_allocation(self, num_pages: int) -> None:
        """Record page allocation."""
        self.pages_allocated += num_pages

    def record_free(self, num_pages: int) -> None:
        """Record page deallocation."""
        self.pages_freed += num_pages

    def get_memory_usage_summary(self) -> dict[str, int]:
        """Get memory usage summary."""
        return {
            "total_pages_allocated": self.pages_allocated,
            "total_pages_freed": self.pages_freed,
            "pages_in_use": self.pages_allocated - self.pages_freed,
            "active_requests": len(
                set().union(*(manager.req_to_pages.keys() for manager in self.coordinator.single_type_managers))
            ),
        }


def calculate_cache_efficiency(coordinator: KVCacheCoordinator, request_ids: list[str]) -> dict[str, float]:
    """
    Calculate cache efficiency metrics for given requests.

    Args:
        coordinator: The KV cache coordinator.
        request_ids: List of request IDs to analyze.

    Returns:
        Dictionary with efficiency metrics.
    """
    total_pages = 0
    shared_pages = 0

    for request_id in request_ids:
        pages_per_group = coordinator.get_pages(request_id)
        for pages in pages_per_group:
            total_pages += len(pages)
            shared_pages += sum(1 for page in pages if page.ref_cnt > 1)

    sharing_ratio = shared_pages / total_pages if total_pages > 0 else 0.0

    return {
        "total_pages": total_pages,
        "shared_pages": shared_pages,
        "sharing_ratio": sharing_ratio,
        "average_pages_per_request": total_pages / len(request_ids) if request_ids else 0,
    }


def optimize_cache_allocation(
    coordinator: KVCacheCoordinator,
    pending_requests: list[tuple[str, int]],  # (request_id, num_tokens)
    max_pages_available: int,
) -> list[str]:
    """
    Optimize cache allocation for pending requests based on available pages.

    Args:
        coordinator: The KV cache coordinator.
        pending_requests: List of (request_id, num_tokens) tuples.
        max_pages_available: Maximum pages that can be allocated.

    Returns:
        List of request IDs that can be allocated within the page limit.
    """
    # Calculate pages needed for each request
    request_costs = []
    for request_id, num_tokens in pending_requests:
        # Estimate pages needed (simplified calculation)
        pages_needed = sum(
            coordinator.get_num_pages_to_allocate(
                request_id, num_tokens, tuple([] for _ in coordinator.single_type_managers)
            )
            for _ in coordinator.single_type_managers
        )
        request_costs.append((request_id, pages_needed))

    # Sort by cost (greedy allocation of cheapest requests first)
    request_costs.sort(key=lambda x: x[1])

    # Allocate requests within budget
    allocated_requests = []
    total_cost = 0

    for request_id, cost in request_costs:
        if total_cost + cost <= max_pages_available:
            allocated_requests.append(request_id)
            total_cost += cost
        else:
            break

    return allocated_requests


# Context manager for batch operations
class BatchKVCacheOperation:
    """Context manager for batching KV cache operations."""

    def __init__(self, coordinator: KVCacheCoordinator):
        self.coordinator = coordinator
        self.pending_operations: list[tuple[str, str, tuple]] = []  # (operation, request_id, args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._execute_batch()

    def add_allocation(self, request_id: str, num_tokens: int):
        """Add an allocation operation to the batch."""
        self.pending_operations.append(("allocate", request_id, (num_tokens,)))

    def add_free(self, request_id: str):
        """Add a free operation to the batch."""
        self.pending_operations.append(("free", request_id, ()))

    def _execute_batch(self):
        """Execute all pending operations in batch."""
        # Group operations by type for efficiency
        allocations = []
        frees = []

        for op_type, request_id, args in self.pending_operations:
            if op_type == "allocate":
                allocations.append((request_id, args[0]))
            elif op_type == "free":
                frees.append(request_id)

        # Execute frees first to release memory
        for request_id in frees:
            self.coordinator.free(request_id)

        # Then execute allocations
        for request_id, num_tokens in allocations:
            self.coordinator.allocate_new_pages(request_id, num_tokens)


# Example usage and testing utilities
def validate_coordinator_consistency(coordinator: KVCacheCoordinator) -> list[str]:
    """
    Validate that the coordinator is in a consistent state.

    Returns:
        List of validation errors (empty if consistent).
    """
    errors = []

    # Check that all managers have consistent request sets
    all_request_ids = set()
    for i, manager in enumerate(coordinator.single_type_managers):
        manager_requests = set(manager.req_to_pages.keys())
        if i == 0:
            all_request_ids = manager_requests
        elif manager_requests != all_request_ids:
            errors.append(f"Manager {i} has inconsistent request set")

    # Check page reference counts
    for manager in coordinator.single_type_managers:
        for request_id, pages in manager.req_to_pages.items():
            for page in pages:
                if hasattr(page, "ref_cnt") and page.ref_cnt < 0:
                    errors.append(f"Negative ref_cnt for page in request {request_id}")

    return errors
