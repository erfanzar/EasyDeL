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

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional
from weakref import WeakSet

from easydel.utils.helpers import get_logger

from .request_type import EngineRequest
from .utils import PageHash, PageHashWithGroupId

logger = get_logger(__name__)


@dataclass
class KVCachePage:
    """
    KV-cache page metadata with optimized reference counting and hash management.
    """

    page_id: int
    ref_cnt: int = 0
    is_null: bool = False

    # Private fields for internal state management
    _page_hash: PageHashWithGroupId | None = field(default=None, init=False)
    _prev_free_page: Optional["KVCachePage"] = field(default=None, init=False, repr=False)
    _next_free_page: Optional["KVCachePage"] = field(default=None, init=False, repr=False)

    # Track which queues this page belongs to for debugging
    _in_free_queue: bool = field(default=False, init=False, repr=False)

    def incr_ref(self) -> int:
        """Increment reference count and return new value."""
        self.ref_cnt += 1
        return self.ref_cnt

    def decr_ref(self) -> int:
        """Decrement reference count and return new value."""
        if self.ref_cnt <= 0:
            logger.warning(f"Attempting to decrement ref_cnt of page {self.page_id} below 0")
            return self.ref_cnt

        self.ref_cnt -= 1
        return self.ref_cnt

    @property
    def page_hash(self) -> PageHashWithGroupId | None:
        """Get the page hash."""
        return self._page_hash

    @page_hash.setter
    def page_hash(self, page_hash: PageHashWithGroupId) -> None:
        """Set the page hash, ensuring it's only set once."""
        if self._page_hash is not None:
            raise ValueError(f"Page {self.page_id} already has a hash. Cannot reassign.")
        self._page_hash = page_hash

    def reset_hash(self) -> None:
        """Reset the page hash when the page is evicted."""
        self._page_hash = None

    @property
    def prev_free_page(self) -> Optional["KVCachePage"]:
        """Get the previous free page in the linked list."""
        return self._prev_free_page

    @prev_free_page.setter
    def prev_free_page(self, page: Optional["KVCachePage"]) -> None:
        """Set the previous free page."""
        self._prev_free_page = page

    @property
    def next_free_page(self) -> Optional["KVCachePage"]:
        """Get the next free page in the linked list."""
        return self._next_free_page

    @next_free_page.setter
    def next_free_page(self, page: Optional["KVCachePage"]) -> None:
        """Set the next free page."""
        self._next_free_page = page

    @property
    def is_free(self) -> bool:
        """Check if the page is currently in a free queue."""
        return self._in_free_queue

    def mark_as_free(self) -> None:
        """Mark the page as being in a free queue."""
        self._in_free_queue = True

    def mark_as_allocated(self) -> None:
        """Mark the page as allocated (not in free queue)."""
        self._in_free_queue = False
        self._prev_free_page = None
        self._next_free_page = None

    def is_referenced(self) -> bool:
        """Check if the page has any references."""
        return self.ref_cnt > 0

    def __repr__(self) -> str:
        prev_id = self._prev_free_page.page_id if self._prev_free_page else None
        next_id = self._next_free_page.page_id if self._next_free_page else None
        return (
            f"KVCachePage(page_id={self.page_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"hash={self._page_hash is not None}, "
            f"free={self._in_free_queue}, "
            f"prev={prev_id}, next={next_id})"
        )

    def __hash__(self) -> int:
        """Make pages hashable by page_id."""
        return hash(self.page_id)

    def __eq__(self, other: object) -> bool:
        """Compare pages by page_id."""
        if not isinstance(other, KVCachePage):
            return NotImplemented
        return self.page_id == other.page_id


class FreeKVCachePageQueue:
    """
    Optimized doubly linked list of free KV cache pages.

    This implementation provides O(1) operations for:
    - Adding/removing pages from front or back
    - Removing arbitrary pages from the middle

    The queue maintains LRU order where:
    1. Least recently used pages are at the front
    2. For pages with same access time, those with more hash tokens are prioritized
    """

    def __init__(self, pages: list[KVCachePage] | None = None) -> None:
        """
        Initialize the free page queue.

        Args:
            pages: Initial list of pages to add to the queue.
        """
        # Sentinel nodes for easier linked list manipulation
        self._head = KVCachePage(page_id=-1, is_null=True)
        self._tail = KVCachePage(page_id=-1, is_null=True)
        self._head._next_free_page = self._tail
        self._tail._prev_free_page = self._head

        self._num_free_pages = 0
        self._page_set: WeakSet[KVCachePage] = WeakSet()

        # Add initial pages if provided
        if pages:
            self._initialize_with_pages(pages)

    def _initialize_with_pages(self, pages: list[KVCachePage]) -> None:
        """Initialize the queue with a list of pages."""
        if not pages:
            return

        # Link pages together
        prev_page = self._head
        for page in pages:
            page._prev_free_page = prev_page
            prev_page._next_free_page = page
            page.mark_as_free()
            self._page_set.add(page)
            prev_page = page

        # Connect last page to tail
        prev_page._next_free_page = self._tail
        self._tail._prev_free_page = prev_page

        self._num_free_pages = len(pages)

    @property
    def num_free_pages(self) -> int:
        """Get the number of free pages."""
        return self._num_free_pages

    @property
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self._num_free_pages == 0

    def popleft(self) -> KVCachePage:
        """
        Pop the first free page (LRU).

        Returns:
            The least recently used free page.

        Raises:
            ValueError: If no free pages are available.
        """
        if self.is_empty:
            raise ValueError("No free pages available")

        first_page = self._head._next_free_page
        if first_page is None or first_page is self._tail:
            raise RuntimeError("Queue corruption: invalid first page")

        self._remove_page_from_list(first_page)
        first_page.mark_as_allocated()
        self._page_set.discard(first_page)

        return first_page

    def popleft_n(self, n: int) -> list[KVCachePage]:
        """
        Pop the first n free pages.

        Args:
            n: Number of pages to pop.

        Returns:
            List of n free pages.

        Raises:
            ValueError: If not enough free pages are available.
        """
        if n == 0:
            return []

        if n > self._num_free_pages:
            raise ValueError(f"Requested {n} pages but only {self._num_free_pages} available")

        pages = []
        for _ in range(n):
            page = self.popleft()
            pages.append(page)

        return pages

    def remove(self, page: KVCachePage) -> None:
        """
        Remove a specific page from the queue.

        Args:
            page: The page to remove.

        Raises:
            ValueError: If the page is not in the queue.
        """
        if not page.is_free or page not in self._page_set:
            raise ValueError(f"Page {page.page_id} is not in the free queue")

        self._remove_page_from_list(page)
        page.mark_as_allocated()
        self._page_set.discard(page)

    def append(self, page: KVCachePage) -> None:
        """
        Add a page to the end of the queue (most recently used).

        Args:
            page: The page to add.
        """
        if page.is_free:
            logger.warning(f"Page {page.page_id} is already in a free queue")
            return

        self._insert_before_tail(page)
        page.mark_as_free()
        self._page_set.add(page)

    def append_n(self, pages: list[KVCachePage]) -> None:
        """
        Add multiple pages to the end of the queue.

        Args:
            pages: List of pages to add.
        """
        if not pages:
            return

        for page in pages:
            if page.is_free:
                logger.warning(f"Page {page.page_id} is already in a free queue")
                continue
            self.append(page)

    def _remove_page_from_list(self, page: KVCachePage) -> None:
        """Remove a page from the linked list structure."""
        if page._prev_free_page is None or page._next_free_page is None:
            raise RuntimeError(f"Cannot remove page {page.page_id}: invalid links")

        page._prev_free_page._next_free_page = page._next_free_page
        page._next_free_page._prev_free_page = page._prev_free_page

        self._num_free_pages -= 1

    def _insert_before_tail(self, page: KVCachePage) -> None:
        """Insert a page before the tail sentinel."""
        last_page = self._tail._prev_free_page
        if last_page is None:
            raise RuntimeError("Queue corruption: tail has no previous page")

        page._prev_free_page = last_page
        page._next_free_page = self._tail
        last_page._next_free_page = page
        self._tail._prev_free_page = page

        self._num_free_pages += 1

    def get_all_free_pages(self) -> list[KVCachePage]:
        """
        Get all free pages in order (for testing/debugging).

        Returns:
            List of all free pages from front to back.
        """
        pages = []
        current = self._head._next_free_page

        while current is not None and current is not self._tail:
            pages.append(current)
            current = current._next_free_page

        return pages

    def peek_front(self) -> KVCachePage | None:
        """Peek at the front page without removing it."""
        if self.is_empty:
            return None
        return self._head._next_free_page

    def peek_back(self) -> KVCachePage | None:
        """Peek at the back page without removing it."""
        if self.is_empty:
            return None
        return self._tail._prev_free_page

    def __iter__(self) -> Iterator[KVCachePage]:
        """Iterate over all free pages from front to back."""
        current = self._head._next_free_page
        while current is not None and current is not self._tail:
            yield current
            current = current._next_free_page

    def __len__(self) -> int:
        """Get the number of free pages."""
        return self._num_free_pages

    def __bool__(self) -> bool:
        """Check if the queue has any pages."""
        return not self.is_empty

    def validate_integrity(self) -> list[str]:
        """
        Validate the integrity of the linked list structure.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Count pages by traversing forward
        forward_count = 0
        current = self._head._next_free_page
        visited = set()

        while current is not None and current is not self._tail:
            if current.page_id in visited:
                errors.append(f"Cycle detected at page {current.page_id}")
                break

            visited.add(current.page_id)
            forward_count += 1

            # Check backward link
            if current._next_free_page and current._next_free_page._prev_free_page is not current:
                errors.append(f"Broken backward link at page {current.page_id}")

            current = current._next_free_page

        # Check count consistency
        if forward_count != self._num_free_pages:
            errors.append(f"Count mismatch: traversed {forward_count}, stored {self._num_free_pages}")

        return errors


def hash_page_tokens(
    hash_function: Callable,
    parent_page_hash: int | None,
    curr_page_token_ids: tuple[int, ...],  # Use tuple for hashability
    extra_keys: tuple[Any, ...] | None = None,
) -> PageHash:
    """
    Compute hash value for a page with LRU caching for performance.

    Args:
        hash_function: Function to compute the hash.
        parent_page_hash: Hash of the parent page (None for first page).
        curr_page_token_ids: Tuple of token IDs in the current page.
        extra_keys: Additional keys for the hash.

    Returns:
        PageHash object containing the computed hash and metadata.
    """
    curr_page_token_ids = tuple(curr_page_token_ids)  # Ensure it's a tuple for hashing
    return PageHash(
        hash_value=hash_function((parent_page_hash, curr_page_token_ids, extra_keys)),
        token_ids=curr_page_token_ids,
        extra_keys=extra_keys,
    )


def hash_page_tokens_uncached(
    hash_function: Callable,
    parent_page_hash: int | None,
    curr_page_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> PageHash:
    """
    Compute hash value for a page without caching (for dynamic content).

    Args:
        hash_function: Function to compute the hash.
        parent_page_hash: Hash of the parent page (None for first page).
        curr_page_token_ids: Sequence of token IDs in the current page.
        extra_keys: Additional keys for the hash.

    Returns:
        PageHash object containing the computed hash and metadata.
    """
    token_ids_tuple = tuple(curr_page_token_ids)
    return hash_page_tokens(hash_function, parent_page_hash, token_ids_tuple, extra_keys)


def hash_request_tokens(
    hash_function: Callable,
    page_size: int,
    request: EngineRequest,
    use_caching: bool = True,
) -> list[PageHash]:
    """
    Compute hash values for a chain of pages from a request's token sequence.

    Args:
        hash_function: Function to compute hashes.
        page_size: Size of each page in tokens.
        request: The request object containing token IDs.
        use_caching: Whether to use LRU caching for hash computation.

    Returns:
        List of PageHash objects for each complete page.
    """
    token_ids = request.all_token_ids

    if not token_ids:
        return []

    # Extract request-specific extra keys if available
    req_extra_keys = getattr(request, "extra_hash_keys", None)

    # Choose hash function based on caching preference
    hash_func = hash_page_tokens if use_caching else hash_page_tokens_uncached

    page_hashes = []
    parent_hash_value = None

    # Process complete pages only
    num_complete_pages = len(token_ids) // page_size

    for page_idx in range(num_complete_pages):
        start_idx = page_idx * page_size
        end_idx = start_idx + page_size
        page_token_ids = token_ids[start_idx:end_idx]

        # Compute hash for this page
        page_hash = hash_func(
            hash_function=hash_function,
            parent_page_hash=parent_hash_value,
            curr_page_token_ids=page_token_ids,
            extra_keys=req_extra_keys,
        )

        page_hashes.append(page_hash)
        parent_hash_value = page_hash.hash_value

    return page_hashes


def batch_hash_requests(
    hash_function: Callable,
    page_size: int,
    requests: list[EngineRequest],
    use_caching: bool = True,
) -> dict[str, list[PageHash]]:
    """
    Efficiently compute hashes for multiple requests in batch.

    Args:
        hash_function: Function to compute hashes.
        page_size: Size of each page in tokens.
        requests: List of requests to process.
        use_caching: Whether to use LRU caching.

    Returns:
        Dictionary mapping request IDs to their page hashes.
    """
    result = {}

    for request in requests:
        try:
            hashes = hash_request_tokens(
                hash_function=hash_function,
                page_size=page_size,
                request=request,
                use_caching=use_caching,
            )
            result[request.request_id] = hashes
        except Exception as e:
            logger.error(f"Failed to hash request {request.request_id}: {e}")
            result[request.request_id] = []

    return result


class HashCache:
    """
    Specialized cache for page hashes with size limits and eviction policies.
    """

    def __init__(self, max_size: int = 50000):
        """
        Initialize the hash cache.

        Args:
            max_size: Maximum number of entries to cache.
        """
        self.max_size = max_size
        self._cache: dict[tuple, PageHash] = {}
        self._access_order: list[tuple] = []
        self._hits = 0
        self._misses = 0

    def get(
        self,
        hash_function: Callable,
        parent_hash: int | None,
        token_ids: tuple[int, ...],
        extra_keys: tuple[Any, ...] | None = None,
    ) -> PageHash:
        """
        Get or compute a page hash.

        Args:
            hash_function: Hash function to use.
            parent_hash: Parent page hash.
            token_ids: Token IDs for the page.
            extra_keys: Extra keys for hashing.

        Returns:
            PageHash object.
        """
        cache_key = (parent_hash, token_ids, extra_keys)

        if cache_key in self._cache:
            self._hits += 1
            # Move to end for LRU
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]

        # Cache miss - compute hash
        self._misses += 1
        page_hash = PageHash(
            hash_value=hash_function((parent_hash, token_ids, extra_keys)),
            token_ids=token_ids,
            extra_keys=extra_keys,
        )

        # Add to cache
        self._cache[cache_key] = page_hash
        self._access_order.append(cache_key)

        # Evict if necessary
        if len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        return page_hash

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0


# Global hash cache instance
_global_hash_cache = HashCache()


def hash_page_tokens_cached(
    hash_function: Callable,
    parent_page_hash: int | None,
    curr_page_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> PageHash:
    """
    Hash page tokens using the global cache.

    Args:
        hash_function: Hash function to use.
        parent_page_hash: Parent page hash.
        curr_page_token_ids: Token IDs for the page.
        extra_keys: Extra keys for hashing.

    Returns:
        PageHash object.
    """
    token_ids_tuple = tuple(curr_page_token_ids)
    return _global_hash_cache.get(hash_function, parent_page_hash, token_ids_tuple, extra_keys)


def get_hash_cache_stats() -> dict[str, int | float]:
    """Get statistics about the global hash cache."""
    return {
        "cache_size": len(_global_hash_cache._cache),
        "max_size": _global_hash_cache.max_size,
        "hits": _global_hash_cache._hits,
        "misses": _global_hash_cache._misses,
        "hit_rate": _global_hash_cache.hit_rate,
    }


def clear_hash_cache() -> None:
    """Clear the global hash cache."""
    _global_hash_cache.clear()


# Validation and debugging utilities
def validate_page_integrity(page: KVCachePage) -> list[str]:
    """
    Validate the integrity of a single page.

    Args:
        page: The page to validate.

    Returns:
        List of validation errors.
    """
    errors = []

    if page.ref_cnt < 0:
        errors.append(f"Page {page.page_id} has negative ref_cnt: {page.ref_cnt}")

    if page.is_free and page.ref_cnt > 0:
        errors.append(f"Page {page.page_id} is marked free but has ref_cnt > 0")

    if page.is_free and (page._prev_free_page is None or page._next_free_page is None):
        errors.append(f"Page {page.page_id} is marked free but has invalid links")

    return errors


def analyze_page_usage(pages: list[KVCachePage]) -> dict[str, Any]:
    """
    Analyze usage patterns of a list of pages.

    Args:
        pages: List of pages to analyze.

    Returns:
        Dictionary with usage statistics.
    """
    if not pages:
        return {"total_pages": 0}

    ref_counts = [page.ref_cnt for page in pages]
    free_pages = sum(1 for page in pages if page.is_free)
    hashed_pages = sum(1 for page in pages if page.page_hash is not None)

    return {
        "total_pages": len(pages),
        "free_pages": free_pages,
        "allocated_pages": len(pages) - free_pages,
        "hashed_pages": hashed_pages,
        "min_ref_cnt": min(ref_counts),
        "max_ref_cnt": max(ref_counts),
        "avg_ref_cnt": sum(ref_counts) / len(ref_counts),
        "zero_ref_pages": sum(1 for rc in ref_counts if rc == 0),
    }
