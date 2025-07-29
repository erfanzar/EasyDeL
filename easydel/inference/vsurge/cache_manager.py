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
import hashlib
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from easydel.layers.caching import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    PagesCacheMetaData,
    SlidingWindowSpec,
)
from easydel.utils.helpers import get_logger

from .cache_utils import KVCachePage, PageHash, hash_request_tokens
from .page_table import PagePool
from .request_type import EngineRequest, EngineRequestStatus

logger = get_logger(__name__)


T = TypeVar("T")


def cdiv(a, b):
    return (a + b - 1) // b


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handles the kv cache management
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        page_pool: PagePool,
        kv_cache_group_id: int,
        caching_hash_fn: Callable,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.

        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            page_pool: The page pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            caching_hash_fn: The caching hash function.
        """
        self.page_size = kv_cache_spec.page_size
        self.kv_cache_spec = kv_cache_spec
        self.page_pool = page_pool
        self.kv_cache_group_id = kv_cache_group_id
        self.caching_hash_fn = caching_hash_fn
        self._null_page = page_pool.null_page

        # Request tracking
        self.req_to_pages: defaultdict[str, list[KVCachePage]] = defaultdict(list)
        self.num_cached_page: dict[str, int] = {}

    def get_num_pages_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_pages: list[KVCachePage],
    ) -> int:
        """
        Get the number of pages needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot.
            new_computed_pages: The new computed pages just hitting the prefix caching.

        Returns:
            The number of pages to allocate.
        """
        num_required_pages = cdiv(num_tokens, self.page_size)
        current_pages = len(self.req_to_pages[request_id])
        num_new_pages = num_required_pages - len(new_computed_pages) - current_pages

        # Count evictable pages more efficiently
        num_evictable = sum(1 for page in new_computed_pages if page.ref_cnt == 0 and not page.is_null)

        return num_new_pages + num_evictable

    def save_new_computed_pages(self, request_id: str, new_computed_pages: list[KVCachePage]) -> None:
        """
        Add the new computed pages to the request.

        Args:
            request_id: The request ID.
            new_computed_pages: The new computed pages just hitting the prefix cache.
        """
        if request_id not in self.num_cached_page:
            # First time saving pages for this request
            req_pages = self.req_to_pages[request_id]
            assert not req_pages, f"Expected empty pages for new request {request_id}"
            req_pages.extend(new_computed_pages)
            self.num_cached_page[request_id] = len(new_computed_pages)
        else:
            # Request already has cached pages
            assert not new_computed_pages, f"Unexpected new pages for existing request {request_id}"

    def allocate_new_pages(self, request_id: str, num_tokens: int) -> list[KVCachePage]:
        """
        Allocate new pages for the request to give it at least `num_tokens` token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot.

        Returns:
            The newly allocated pages.
        """
        req_pages = self.req_to_pages[request_id]
        num_required_pages = cdiv(num_tokens, self.page_size)
        num_new_pages = num_required_pages - len(req_pages)

        if num_new_pages <= 0:
            return []

        new_pages = self.page_pool.get_new_pages(num_new_pages)
        req_pages.extend(new_pages)
        return new_pages

    def cache_pages(self, request: EngineRequest, page_hashes: list[PageHash], num_tokens: int) -> None:
        """
        Cache the pages for the request.

        Args:
            request: The request.
            page_hashes: The page hashes of the request.
            num_tokens: The total number of tokens to be cached.
        """
        request_id = request.request_id
        num_cached_pages = self.num_cached_page.get(request_id, 0)
        num_full_pages = num_tokens // self.page_size
        self.page_pool.cache_full_pages(
            request=request,
            pages=self.req_to_pages[request_id],
            page_hashes=page_hashes,
            num_cached_pages=num_cached_pages,
            num_full_pages=num_full_pages,
            page_size=self.page_size,
            kv_cache_group_id=self.kv_cache_group_id,
            hash_fn=self.caching_hash_fn,
        )

        self.num_cached_page[request_id] = num_full_pages

    def free(self, request_id: str) -> None:
        """
        Free the pages for the request.

        Args:
            request_id: The request ID.
        """
        req_pages = self.req_to_pages.pop(request_id, [])
        if req_pages:
            # Free pages in reverse order for better memory locality
            self.page_pool.free_pages(reversed(req_pages))

        self.num_cached_page.pop(request_id, None)

    def get_request_pages(self, request_id: str) -> list[KVCachePage]:
        """Get pages for a request, returning empty list if not found."""
        return self.req_to_pages.get(request_id, [])

    @abstractmethod
    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        """Get the number of common prefix pages for a request."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
    ) -> tuple[list[KVCachePage], ...]:
        """Find the longest cache hit prefix."""
        raise NotImplementedError

    @abstractmethod
    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        """Remove pages that are no longer needed."""
        raise NotImplementedError


class FullAttentionManager(SingleTypeKVCacheManager):
    """Manager for full attention and chunked local attention."""

    @classmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
    ) -> tuple[list[KVCachePage], ...]:
        assert isinstance(kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec), (
            "Invalid kv_cache_spec type for FullAttentionManager"
        )

        num_groups = len(kv_cache_group_ids)
        computed_pages: tuple[list[KVCachePage], ...] = tuple([] for _ in range(num_groups))
        max_num_pages = max_length // kv_cache_spec.page_size

        # Find longest prefix of cached pages
        for _, page_hash in enumerate(page_hashes[:max_num_pages]):
            cached_page = page_pool.get_cached_page(page_hash, kv_cache_group_ids)
            if cached_page:
                for computed, cached in zip(computed_pages, cached_page, strict=False):
                    computed.append(cached)
            else:
                break

        # Handle eagle mode - drop last page if present
        if use_eagle and computed_pages[0]:
            for computed in computed_pages:
                computed.pop()

        return computed_pages

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        """Full attention doesn't skip pages."""
        pass

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        """Count pages shared by all running requests."""
        pages = self.req_to_pages.get(request_id, [])

        # Count consecutive pages with ref_cnt == num_running_requests
        for i, page in enumerate(pages):
            if page.ref_cnt != num_running_requests:
                return i

        return len(pages)


class SlidingWindowManager(SingleTypeKVCacheManager):
    """Manager for sliding window attention."""

    def __init__(self, kv_cache_spec: SlidingWindowSpec, page_pool: PagePool, **kwargs) -> None:
        super().__init__(kv_cache_spec, page_pool, **kwargs)
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        self.sliding_window = kv_cache_spec.sliding_window
        self._sliding_window_pages = cdiv(self.sliding_window - 1, self.page_size)

    @classmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
    ) -> tuple[list[KVCachePage], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), "Invalid spec for SlidingWindowManager"

        sliding_window_pages = cdiv(kv_cache_spec.sliding_window - 1, kv_cache_spec.page_size)
        if use_eagle:
            sliding_window_pages += 1

        max_num_pages = max_length // kv_cache_spec.page_size
        num_groups = len(kv_cache_group_ids)

        # Initialize with null pages
        computed_pages = tuple([page_pool.null_page] * max_num_pages for _ in range(num_groups))

        # Search backwards for contiguous cached pages
        num_contiguous = 0
        match_found = False

        for i in range(max_num_pages - 1, -1, -1):
            cached_page = page_pool.get_cached_page(page_hashes[i], kv_cache_group_ids)

            if cached_page:
                # Update all groups
                for computed, cached in zip(computed_pages, cached_page, strict=False):
                    computed[i] = cached

                num_contiguous += 1

                # Check if we have enough contiguous pages
                if num_contiguous >= sliding_window_pages:
                    # Trim to only include the found pages
                    for computed in computed_pages:
                        del computed[i + num_contiguous :]
                    match_found = True
                    break
            else:
                num_contiguous = 0

        # If no full match, keep only contiguous pages found
        if not match_found and num_contiguous > 0:
            for computed in computed_pages:
                del computed[num_contiguous:]

        # Handle eagle mode
        if use_eagle and computed_pages[0]:
            for computed in computed_pages:
                computed.pop()

        return computed_pages

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        """Remove pages outside the sliding window."""
        pages = self.req_to_pages.get(request_id, [])
        if not pages:
            return

        # Calculate the last useful page
        last_useful_token = num_computed_tokens - self.sliding_window + 1
        if last_useful_token <= 0:
            return

        last_useful_page = last_useful_token // self.page_size

        # Collect pages to remove
        removed_pages: list[KVCachePage] = []

        for i in range(min(last_useful_page, len(pages) - 1), -1, -1):
            if pages[i] == self._null_page:
                break
            removed_pages.append(pages[i])
            pages[i] = self._null_page

        # Free removed pages
        if removed_pages:
            self.page_pool.free_pages(removed_pages)

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        """
        Sliding window uses null pages for prefix, so common prefix is always 0.
        TODO: Support cascade attention + sliding window in the future.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    """Manager for chunked local attention."""

    def __init__(
        self,
        kv_cache_spec: ChunkedLocalAttentionSpec,
        page_pool: PagePool,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, page_pool, **kwargs)
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
    ) -> tuple[list[KVCachePage], ...]:
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), "Invalid spec"
        assert not use_eagle, "Eagle not supported with chunked local attention"

        max_num_pages = max_length // kv_cache_spec.page_size
        num_groups = len(kv_cache_group_ids)

        # Calculate local attention window start
        if max_length > 0:
            chunks = max_length // kv_cache_spec.attention_chunk_size
            local_attention_start_idx = chunks * kv_cache_spec.attention_chunk_size
        else:
            local_attention_start_idx = 0

        local_attention_start_page = local_attention_start_idx // kv_cache_spec.page_size

        # Initialize with null pages for out-of-window pages
        # Initialize with null pages for out-of-window pages
        computed_pages: tuple[list[KVCachePage], ...] = tuple(
            [page_pool.null_page] * local_attention_start_page for _ in range(num_groups)
        )

        # Check for cached pages within the attention window
        for i in range(local_attention_start_page, max_num_pages):
            cached_page = page_pool.get_cached_page(page_hashes[i], kv_cache_group_ids)
            if cached_page:
                for computed, cached in zip(computed_pages, cached_page, strict=False):
                    computed.append(cached)
            else:
                break

        return computed_pages

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        """Remove pages outside the local attention window."""
        pages = self.req_to_pages.get(request_id, [])
        if not pages:
            return

        # Calculate the first useful page
        chunks = num_computed_tokens // self.attention_chunk_size
        local_attention_start_idx = chunks * self.attention_chunk_size
        first_useful_page_idx = local_attention_start_idx // self.page_size

        # Limit by cached pages to avoid removing pages we haven't cached yet
        num_cached = self.num_cached_page.get(request_id, 0)
        if num_cached > 0:
            first_useful_page_idx = min(first_useful_page_idx, num_cached - 1)

        # Remove pages before the useful range
        removed_pages: list[KVCachePage] = []

        for i in range(min(first_useful_page_idx, len(pages)) - 1, -1, -1):
            if pages[i] == self._null_page:
                break
            removed_pages.append(pages[i])
            pages[i] = self._null_page

        # Free removed pages
        if removed_pages:
            self.page_pool.free_pages(removed_pages)

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        """Cascade attention is not supported by chunked local attention."""
        return 0


# Utility class for managing multiple attention types
class MultiTypeKVCacheManager:
    """Manages multiple SingleTypeKVCacheManager instances."""

    def __init__(self):
        self.managers: dict[int, SingleTypeKVCacheManager] = {}
        self._manager_types: dict[int, type[SingleTypeKVCacheManager]] = {}

    def register_manager(self, kv_cache_group_id: int, manager: SingleTypeKVCacheManager) -> None:
        """Register a manager for a specific KV cache group."""
        self.managers[kv_cache_group_id] = manager
        self._manager_types[kv_cache_group_id] = type(manager)

    def get_manager(self, kv_cache_group_id: int) -> SingleTypeKVCacheManager:
        """Get the manager for a specific KV cache group."""
        return self.managers[kv_cache_group_id]

    def allocate_pages_for_request(
        self, request_id: str, num_tokens: int, kv_cache_group_ids: list[int] | None = None
    ) -> dict[int, list[KVCachePage]]:
        """Allocate pages for a request across all relevant managers."""
        if kv_cache_group_ids is None:
            kv_cache_group_ids = list(self.managers.keys())

        allocated_pages = {}
        for group_id in kv_cache_group_ids:
            if group_id in self.managers:
                pages = self.managers[group_id].allocate_new_pages(request_id, num_tokens)
                allocated_pages[group_id] = pages

        return allocated_pages

    def free_request(self, request_id: str) -> None:
        """Free a request from all managers."""
        for manager in self.managers.values():
            manager.free(request_id)

    def get_total_pages_allocated(self, request_id: str) -> int:
        """Get total number of pages allocated for a request across all managers."""
        total = 0
        for manager in self.managers.values():
            total += len(manager.req_to_pages.get(request_id, []))
        return total


# Helper functions for efficient operations
def batch_get_cached_pages(
    page_hashes: list[PageHash],
    kv_cache_group_ids: list[int],
    page_pool: PagePool,
    max_pages: int | None = None,
) -> list[tuple[KVCachePage, ...] | None]:
    """
    Batch retrieve cached pages for multiple hashes.

    Returns:
        List where each element is either None (cache miss) or tuple of cached pages.
    """
    if max_pages is not None:
        page_hashes = page_hashes[:max_pages]

    results = []
    for page_hash in page_hashes:
        cached = page_pool.get_cached_page(page_hash, kv_cache_group_ids)
        results.append(cached)

    return results


def calculate_pages_needed(num_tokens: int, page_size: int, current_pages: int = 0) -> int:
    """Calculate number of additional pages needed."""
    total_pages_needed = cdiv(num_tokens, page_size)
    return max(0, total_pages_needed - current_pages)


def find_contiguous_cached_suffix(
    page_hashes: list[PageHash], kv_cache_group_ids: list[int], page_pool: PagePool, min_contiguous: int = 1
) -> tuple[int, list[tuple[KVCachePage, ...]]]:
    """
    Find the longest contiguous suffix of cached pages.

    Returns:
        (start_index, cached_pages) where cached_pages contains the contiguous suffix.
    """
    cached_pages = []

    # Search backwards
    for i in range(len(page_hashes) - 1, -1, -1):
        cached = page_pool.get_cached_page(page_hashes[i], kv_cache_group_ids)
        if cached:
            cached_pages.insert(0, cached)
            if len(cached_pages) >= min_contiguous:
                return i, cached_pages
        else:
            cached_pages.clear()

    return len(page_hashes), []


# Performance monitoring utilities
class KVCacheStats:
    """Track KV cache performance statistics."""

    def __init__(self):
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

    def record_lookup(self, hit: bool) -> None:
        """Record a cache lookup."""
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

    def reset(self) -> None:
        """Reset all statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.pages_allocated = 0
        self.pages_freed = 0
        self.requests_processed = 0


def sha256(input) -> int:  # noqa
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        An integer representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return int.from_bytes(hashlib.sha256(input_bytes).digest(), byteorder="big")


@dataclass
class KVCachePages:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    pages: tuple[list[KVCachePage], ...]
    """
    pages[i][j] refers to the i-th kv_cache_group and the j-th page of tokens.
    We don't use page of tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of pages, which is true for now but
    will be broken if we want to give different page_size to different
    kv_cache_groups in the future.
    """

    def __add__(self, other: "KVCachePages") -> "KVCachePages":
        """Adds two KVCachePages instances."""
        return KVCachePages(tuple(blk1 + blk2 for blk1, blk2 in zip(self.pages, other.pages, strict=False)))

    def get_page_ids(self) -> tuple[list[int], ...]:
        """
        Converts the KVCachePages instance to page_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where
            * the outer tuple corresponds to KV cache groups
            * each inner list contains the page_ids of the pages in that group
        """
        return tuple([blk.page_id for blk in group] for group in self.pages)

    def get_unhashed_page_ids(self) -> list[int]:
        """Get page_ids of unhashed pages from KVCachePages instance."""
        assert len(self.pages) == 1, "Only one group is supported"
        return [page.page_id for page in self.pages[0] if page.page_hash is None]

    def new_empty(self) -> "KVCachePages":
        """Creates a new KVCachePages instance with no pages."""
        return KVCachePages(tuple([] for _ in range(len(self.pages))))


class KVCacheManager:
    def __init__(
        self,
        metadata: PagesCacheMetaData,
        enable_caching: bool = True,
        use_eagle: bool = False,
        num_kv_cache_groups: int = 1,
    ) -> None:
        self.max_model_len = metadata.max_model_length
        self.enable_caching = enable_caching
        self.use_eagle = use_eagle

        self.page_size: int | None = None
        if self.enable_caching:
            self.page_size = metadata.page_size

        from .cache_coordinators import KVCacheCoordinatorNoPrefixCache, UnitaryKVCacheCoordinator

        if self.enable_caching:
            self.coordinator = UnitaryKVCacheCoordinator(
                metadata=metadata,
                use_eagle=self.use_eagle,
                enable_caching=self.enable_caching,
                caching_hash_fn=sha256,
            )
        else:
            self.coordinator = KVCacheCoordinatorNoPrefixCache(
                metadata=metadata,
                use_eagle=self.use_eagle,
                caching_hash_fn=sha256,
            )
        self.num_kv_cache_groups = num_kv_cache_groups
        self.page_pool = self.coordinator.page_pool
        self.metadata = metadata

        self.req_to_page_hashes: defaultdict[str, list[PageHash]] = defaultdict(list)

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.page_pool.get_usage()

    def get_computed_pages(self, request: EngineRequest) -> tuple[KVCachePages, int]:
        """Get the computed (cached) pages for the request.
        Note that the computed pages must be full.

        Args:
            request: The request to get the computed pages.

        Returns:
            A tuple containing:
                - A list of pages that are computed for the request.
                - The number of computed tokens.
        """

        if not self.enable_caching or (
            request.sampling_params is not None and request.sampling_params.prompt_logprobs is not None
        ):
            return self.create_empty_page_list(), 0

        page_hashes = self.req_to_page_hashes[request.request_id]
        if not page_hashes:
            assert self.page_size is not None
            page_hashes = hash_request_tokens(sha256, self.page_size, request)
            self.req_to_page_hashes[request.request_id] = page_hashes

        max_cache_hit_length = request.num_tokens - 1
        computed_pages, num_new_computed_tokens = self.coordinator.find_longest_cache_hit(
            page_hashes, max_cache_hit_length
        )

        return KVCachePages(computed_pages), num_new_computed_tokens

    def allocate_slots(
        self,
        request: EngineRequest,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_pages: KVCachePages | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_pages: bool = False,
    ) -> KVCachePages | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_pages).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_pages: The cached pages for the above new computed
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            delay_cache_pages: Whether to skip caching the pages. This is
                used by P/D when allocating pages used in a KV transfer
                which will complete in a future step.
        The following *_pages are illustrated in this layout.

        Returns:
            A list of new allocated pages.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_pages is not None:
            new_computed_page_list = new_computed_pages.pages
        else:
            new_computed_page_list = tuple([] for _ in range(self.num_kv_cache_groups))

        self.coordinator.remove_skipped_pages(request.request_id, request.num_computed_tokens)

        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(num_computed_tokens + num_new_tokens + num_lookahead_tokens, self.max_model_len)

        num_pages_to_allocate = self.coordinator.get_num_pages_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_pages=new_computed_page_list,
        )

        if num_pages_to_allocate > self.page_pool.get_num_free_pages():
            return None

        if self.enable_caching:
            self.page_pool.touch(new_computed_page_list)
        else:
            assert not any(new_computed_page_list), "Computed pages should be empty when prefix caching is disabled"

        print("New Computed Blocks", new_computed_page_list)
        self.coordinator.save_new_computed_pages(request.request_id, new_computed_page_list)

        new_pages = self.coordinator.allocate_new_pages(request.request_id, num_tokens_need_slot)

        if not self.enable_caching or delay_cache_pages:
            return KVCachePages(new_pages)

        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens, request.num_tokens)
        print("self.req_to_page_hashes[request.request_id]", self.req_to_page_hashes[request.request_id])
        self.coordinator.cache_pages(
            request,
            self.req_to_page_hashes[request.request_id],
            num_tokens_to_cache,
        )

        return KVCachePages(new_pages)

    def free(self, request: EngineRequest) -> None:
        """Free the pages allocated for the request.
        We free the pages in reverse order so that he tail pages are evicted
        first when caching is enabled.

        Args:
            request: The request to free the pages.
        """
        self.coordinator.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.page_pool.reset_prefix_cache():
            return False
        return True

    def get_num_common_prefix_pages(
        self,
        request: EngineRequest,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix pages shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its pages.  A page is considered a common prefix page if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its pages unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix pages
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix pages.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix pages for each kv cache
            group.
        """
        assert request.status == EngineRequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_pages(request.request_id, num_running_requests)

    def free_page_hashes(self, request: EngineRequest) -> None:
        """Discard the page hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_page_hashes.pop(request.request_id, None)

    def get_page_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the page ids of a request."""
        return KVCachePages(self.coordinator.get_pages(request_id)).get_page_ids()

    def cache_pages(self, request: EngineRequest, num_computed_tokens: int) -> None:
        """Cache the pages for the request, if enabled."""
        if self.enable_caching:
            page_hashes = self.req_to_page_hashes[request.request_id]
            self.coordinator.cache_pages(request, page_hashes, num_computed_tokens)

    def create_empty_page_list(self) -> KVCachePages:
        """Creates a new KVCachePages instance with no pages."""
        return KVCachePages(tuple([] for _ in range(self.num_kv_cache_groups)))
