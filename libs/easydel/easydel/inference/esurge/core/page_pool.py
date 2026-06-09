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

"""Page pool management for KV-cache allocation.

Manages a pool of cache pages that can be allocated, freed, and cached
for efficient memory management during inference.

Classes:
    PagePool: Main page pool manager

Example:
    >>> pool = PagePool(num_pages=1000, enable_caching=True)
    >>> pages = pool.allocate(num_pages=10)
    >>> pool.free(pages)
"""

from collections import defaultdict
from collections.abc import Iterable

from ..logger import logger
from ..request import EngineRequest
from .dp_sharding import dp_shard_for_page_id, dp_shard_page_bounds, pages_per_dp_shard
from .utils import CachePage, FreeCachePageQueue, PageHash, PageHashWithGroupId, hash_page_tokens


class PagePool:
    """Pool manager for KV-cache pages with prefix caching support.

    This class manages a pool of CachePage objects, providing allocation,
    deallocation, and prefix caching operations. It uses a free page queue
    ordered by eviction priority (LRU) and maintains a hash-to-page mapping
    for efficient prefix cache lookups.

    The pool supports:
    - O(1) page allocation from the free queue
    - O(1) page deallocation back to the free queue
    - Prefix caching with hash-based page lookup
    - Reference counting for shared page management
    - LRU eviction when the pool is exhausted

    Attributes:
        num_pages: Total number of pages in the pool.
        enable_caching: Whether prefix caching is enabled.
        pages: List of all CachePage objects in the pool.
        free_page_queue: Queue of free pages ordered by eviction priority.
        cached_page_hash_to_page: Mapping from page hash to cached pages.
        null_page: Special placeholder page that is never freed.

    Example:
        >>> pool = PagePool(num_pages=1000, enable_caching=True)
        >>> pages = pool.get_new_pages(num_pages=10)
        >>> pool.free_pages(reversed(pages))  # Free in reverse for LRU order
    """

    def __init__(self, num_pages: int, enable_caching: bool):
        """Initialize the page pool.

        Args:
            num_pages: Total number of pages to allocate. Must be a positive integer.
            enable_caching: Whether to enable prefix caching functionality.

        Raises:
            ValueError: If num_pages is not a positive integer.
        """
        if not isinstance(num_pages, int) or num_pages <= 0:
            raise ValueError("num_pages must be a positive integer")
        self.num_pages = num_pages
        self.enable_caching = enable_caching

        self.pages: list[CachePage] = [CachePage(idx) for idx in range(num_pages)]

        self.free_page_queue = FreeCachePageQueue(self.pages)

        self.cached_page_hash_to_page: dict[PageHashWithGroupId, dict[int, CachePage]] = defaultdict(dict)
        self.null_page = self.free_page_queue.popleft()
        self.null_page.is_null = True

    def get_cached_page(
        self,
        page_hash: PageHash,
        kv_cache_group_ids: list[int],
        *,
        dp_shard_hint: int | None = None,
        data_parallel_size: int | None = None,
    ) -> list[CachePage] | None:
        """Look up cached pages by hash for every KV cache group.

        Returns the cached page corresponding to ``page_hash`` for each
        group in ``kv_cache_group_ids``, or ``None`` if any group misses.
        If multiple cached pages share the same hash (typically duplicates
        across requests), the first one in the cache is returned.

        Args:
            page_hash: The hash value of the page.
            kv_cache_group_ids: The ids of the KV cache groups.
            dp_shard_hint: Optional DP shard hint. When provided with
                ``data_parallel_size > 1``, only cached pages in the shard's
                page-ID range are considered cache hits.
            data_parallel_size: Total number of DP shards.

        Returns:
            The list of cached pages (one per group) if all groups hit,
            ``None`` on any miss.
        """
        use_shard_hint = (
            dp_shard_hint is not None
            and data_parallel_size is not None
            and pages_per_dp_shard(self.num_pages, int(data_parallel_size)) is not None
        )
        page_lo = 0
        page_hi = 0
        if use_shard_hint:
            dp_size = int(data_parallel_size)
            shard_idx = int(dp_shard_hint) % dp_size
            pages_per_shard = pages_per_dp_shard(self.num_pages, dp_size)
            if pages_per_shard is None:
                raise RuntimeError("pages_per_dp_shard returned None despite prior check")
            page_lo, page_hi = dp_shard_page_bounds(shard_idx, pages_per_shard)

        cached_pages = []
        for group_id in kv_cache_group_ids:
            cached_pages_one_group = self.cached_page_hash_to_page.get(PageHashWithGroupId(page_hash, group_id))
            if not cached_pages_one_group:
                return None
            if use_shard_hint:
                shard_page = next(
                    (page for page in cached_pages_one_group.values() if page_lo <= int(page.page_id) < page_hi),
                    None,
                )
                if shard_page is None:
                    return None
                cached_pages.append(shard_page)
            else:
                first_page = next(iter(cached_pages_one_group.values()))
                cached_pages.append(first_page)
        return cached_pages

    def cache_full_pages(
        self,
        request: EngineRequest,
        pages: list[CachePage],
        page_hashes: list[PageHash],
        num_cached_pages: int,
        num_full_pages: int,
        page_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a contiguous range of newly-full pages for prefix sharing.

        Updates page-hash metadata on pages between ``num_cached_pages`` and
        ``num_full_pages``, computing any missing hashes from the request's
        token IDs, and inserts them into ``cached_page_hash_to_page`` so
        later requests can re-use the prefix.

        Args:
            request: The request to cache the pages.
            pages: All pages currently held by the request.
            page_hashes: Page hashes of the pages in the request. This list
                may be shorter than ``pages``; missing hashes are computed
                in-place and appended.
            num_cached_pages: The number of pages already cached.
            num_full_pages: The number of pages that are full and should be
                cached after this call.
            page_size: Number of tokens in each page.
            kv_cache_group_id: The id of the KV cache group these pages
                belong to.

        Raises:
            ValueError: If ``page_hashes`` is shorter than ``num_cached_pages``.
            RuntimeError: If a page already has a hash assigned, or if a
                page's token slice does not contain ``page_size`` tokens.
        """
        if num_cached_pages == num_full_pages:
            return
        new_full_pages = pages[num_cached_pages:num_full_pages]
        if len(page_hashes) < num_cached_pages:
            raise ValueError(f"page_hashes length ({len(page_hashes)}) must be >= num_cached_pages ({num_cached_pages})")
        new_page_hashes = page_hashes[num_cached_pages:]

        if num_cached_pages == 0:
            prev_page_hash_value = None
        else:
            # Find the last page with a hash (skip null pages which don't have hashes)
            # This is important for sliding window where early pages may be null
            prev_page_hash_value = None
            for idx in range(num_cached_pages - 1, -1, -1):
                prev_page = pages[idx]
                if prev_page.page_hash is not None:
                    prev_page_hash_value = prev_page.page_hash.get_hash_value()
                    break

        for i, blk in enumerate(new_full_pages):
            if blk.page_hash is not None:
                raise RuntimeError(
                    f"Expected page_hash to be None for new full page at index {i}, but got {blk.page_hash}"
                )

            if i < len(new_page_hashes):
                page_hash = new_page_hashes[i]
            else:
                blk_idx = num_cached_pages + i
                start_token_idx = blk_idx * page_size
                end_token_idx = (blk_idx + 1) * page_size
                page_tokens = request.all_token_ids[start_token_idx:end_token_idx]
                if len(page_tokens) != page_size:
                    raise RuntimeError(
                        f"Expected {page_size} tokens, got "
                        f"{len(page_tokens)} at {blk_idx}th page for request "
                        f"{request.request_id}({request})"
                    )
                page_hash = hash_page_tokens(hash, prev_page_hash_value, page_tokens, None)
                page_hashes.append(page_hash)

            page_hash_with_group_id = PageHashWithGroupId(page_hash, kv_cache_group_id)
            blk.page_hash = page_hash_with_group_id
            self.cached_page_hash_to_page[page_hash_with_group_id][blk.page_id] = blk
            prev_page_hash_value = page_hash.hash_value

    def get_new_pages(
        self,
        num_pages: int,
        *,
        dp_shard_hint: int | None = None,
        data_parallel_size: int | None = None,
    ) -> list[CachePage]:
        """Allocate fresh pages from the free pool, ignoring the page cache.

        Pages are bumped to ``ref_cnt = 1`` and (when caching is enabled)
        any prior cache hash is evicted before they are returned.

        Args:
            num_pages: The number of pages to allocate.
            dp_shard_hint: Optional data-parallel shard hint. When provided
                with ``data_parallel_size > 1``, allocation is restricted to
                the page-ID range owned by the hinted shard.
            data_parallel_size: Total data-parallel shard count for page
                partitioning.

        Returns:
            A list of ``num_pages`` newly allocated pages.

        Raises:
            ValueError: If the requested number exceeds free pages, or if a
                DP-shard-restricted allocation cannot find enough free pages
                in the hinted shard's range.
            RuntimeError: If shard partitioning becomes inconsistent or a
                page is observed with non-zero ``ref_cnt`` at allocation.
        """
        if num_pages > self.get_num_free_pages():
            raise ValueError(f"Cannot get {num_pages} free pages from the pool")

        use_shard_hint = (
            dp_shard_hint is not None
            and data_parallel_size is not None
            and pages_per_dp_shard(self.num_pages, int(data_parallel_size)) is not None
        )
        if use_shard_hint:
            dp_size = int(data_parallel_size)
            shard_idx = int(dp_shard_hint) % dp_size
            pages_per_shard = pages_per_dp_shard(self.num_pages, dp_size)
            if pages_per_shard is None:
                raise RuntimeError("pages_per_dp_shard returned None despite prior check")
            page_lo, page_hi = dp_shard_page_bounds(shard_idx, pages_per_shard)

            selected: list[CachePage] = []
            for page in self.free_page_queue.get_all_free_pages():
                if dp_shard_for_page_id(int(page.page_id), pages_per_shard, dp_size) == shard_idx:
                    selected.append(page)
                    if len(selected) >= num_pages:
                        break
            if len(selected) >= num_pages:
                for page in selected:
                    self.free_page_queue.remove(page)
                ret = selected
            else:
                raise ValueError(
                    "Insufficient free pages in requested DP shard range: "
                    f"shard={shard_idx} range=[{page_lo}, {page_hi}) requested={num_pages} found={len(selected)}"
                )
        else:
            ret = self.free_page_queue.popleft_n(num_pages)

        for page in ret:
            if self.enable_caching:
                self._maybe_evict_cached_page(page)
            if page.ref_cnt != 0:
                raise RuntimeError(f"Expected ref_cnt to be 0 for free page {page.page_id}, but got {page.ref_cnt}")
            page.ref_cnt += 1
        return ret

    def _maybe_evict_cached_page(self, page: CachePage) -> bool:
        """Evict a page from the prefix cache if it is cached.

        Removes the page's hash association and evicts it from the
        cached_page_hash_to_page mapping. This is called when a cached
        page is being reallocated for new content.

        Args:
            page: The page to potentially evict from the cache.

        Returns:
            True if the page was cached and has been evicted,
            False if the page was not in the cache.

        Note:
            This method handles the case where multiple pages may share
            the same hash (e.g., when a prefix is duplicated across requests).
        """
        page_hash = page.page_hash
        if page_hash is None:
            return False
        pages_by_id = self.cached_page_hash_to_page.get(page_hash)
        if pages_by_id is None:
            return False
        page.reset_hash()
        pages_by_id.pop(page.page_id, None)
        if len(pages_by_id) == 0:
            del self.cached_page_hash_to_page[page_hash]
        return True

    def touch(self, pages: tuple[list[CachePage], ...]) -> None:
        """Increment reference counts on pages re-hit by a new request.

        For each non-null page with ``ref_cnt == 0``, the page is first
        removed from the free queue (it was eligible for eviction); then
        every page's reference count is bumped by 1.

        Args:
            pages: Per-group page lists to touch.
        """
        for pages_per_group in pages:
            for page in pages_per_group:
                if page.ref_cnt == 0 and not page.is_null:
                    self.free_page_queue.remove(page)
                page.incr_ref()

    def free_pages(self, ordered_pages: Iterable[CachePage]) -> None:
        """Decrement refs and re-queue pages whose count drops to zero.

        Pages must be ordered by eviction priority — the first page is
        evicted first. Non-null pages with ``ref_cnt == 0`` after the
        decrement are appended to the free queue.

        Args:
            ordered_pages: Pages to free, ordered by eviction priority.
        """

        pages_list = list(ordered_pages)
        for page in pages_list:
            page.ref_cnt -= 1
        self.free_page_queue.append_n([page for page in pages_list if page.ref_cnt == 0 and not page.is_null])

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache, clearing every cached page hash.

        Used in RLHF flows to invalidate prefix caching after weights are
        updated, or for resetting prefix caching state during benchmarking.
        The reset fails (returning ``False``) if any pages besides the null
        page are still in use.

        Returns:
            True if the prefix cache is successfully reset, False if there
            are still pages in use that prevent the reset.
        """
        num_used_pages = self.num_pages - self.get_num_free_pages()
        if num_used_pages != 1:
            logger.warning("Failed to reset prefix cache because some pages (%d) are not freed yet", num_used_pages - 1)
            return False

        self.cached_page_hash_to_page.clear()
        for page in self.pages:
            page.reset_hash()

        logger.info("Successfully reset prefix cache")
        return True

    def get_num_free_pages(self) -> int:
        """Get the number of free pages in the pool.

        Returns:
            The number of free pages.
        """
        return self.free_page_queue.num_free_pages

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return 1.0 - (self.get_num_free_pages() / self.num_pages)
