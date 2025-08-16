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

from ..request import EngineRequest
from .utils import CachePage, FreeCachePageQueue, PageHash, PageHashWithGroupId, hash_page_tokens


class PagePool:
    """PagePool that manages CachePages.
    It provides methods to allocate, free and cache the kv cache pages. The
    free_page_queue stores the free pages in eviction order to enable
    allocation, free, and cache eviction. The cached_page_hash_to_page
    maps between page hash and cached page to support finding cached pages
    by their page hash.

    Args:
        num_pages: The number of pages in the pool.
        enable_caching: Whether to enable prefix caching.
    """

    def __init__(self, num_pages: int, enable_caching: bool):
        assert isinstance(num_pages, int) and num_pages > 0
        self.num_pages = num_pages
        self.enable_caching = enable_caching

        self.pages: list[CachePage] = [CachePage(idx) for idx in range(num_pages)]

        self.free_page_queue = FreeCachePageQueue(self.pages)

        self.cached_page_hash_to_page: dict[PageHashWithGroupId, dict[int, CachePage]] = defaultdict(dict)
        self.null_page = self.free_page_queue.popleft()
        self.null_page.is_null = True

    def get_cached_page(self, page_hash: PageHash, kv_cache_group_ids: list[int]) -> list[CachePage] | None:
        """Get the cached page by the page hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated pages, we return the first page in the cache.

        Args:
            page_hash: The hash value of the page.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached pages if exists, or None.
        """
        cached_pages = []
        for group_id in kv_cache_group_ids:
            cached_pages_one_group = self.cached_page_hash_to_page.get(PageHashWithGroupId(page_hash, group_id))
            if not cached_pages_one_group:
                return None
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
        """Cache a list of full pages for prefix caching.
        This function takes a list of pages that will have their page hash
        metadata to be updated and cached. Given a request, it computes the
        page hashes for the pages starting from `num_cached_pages` to
        `num_full_pages`, updating the metadata for each page
        and caching them in the `cached_page_hash_to_page`.

        Args:
            request: The request to cache the pages.
            pages: All pages in the request.
            page_hashes: Page hashes of the pages in the request. Note that
            this list may be shorter than the pages list. In this case the
            missed page hash will be computed in this function.
            num_cached_pages: The number of pages that are already cached.
            num_full_pages: The number of pages that are full and should
                be cached after this function.
            page_size: Number of tokens in each page.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_pages == num_full_pages:
            return
        new_full_pages = pages[num_cached_pages:num_full_pages]
        assert len(page_hashes) >= num_cached_pages
        new_page_hashes = page_hashes[num_cached_pages:]

        if num_cached_pages == 0:
            prev_page_hash_value = None
        else:
            prev_page = pages[num_cached_pages - 1]
            assert prev_page.page_hash is not None
            prev_page_hash_value = prev_page.page_hash.get_hash_value()

        new_hashes = None
        for i, blk in enumerate(new_full_pages):
            assert blk.page_hash is None

            if i < len(new_page_hashes):
                page_hash = new_page_hashes[i]
            else:
                blk_idx = num_cached_pages + i
                start_token_idx = blk_idx * page_size
                end_token_idx = (blk_idx + 1) * page_size
                page_tokens = request.all_token_ids[start_token_idx:end_token_idx]
                assert len(page_tokens) == page_size, (
                    f"Expected {page_size} tokens, got "
                    f"{len(page_tokens)} at {blk_idx}th page for request "
                    f"{request.request_id}({request})"
                )

                page_hash = hash_page_tokens(hash, prev_page_hash_value, page_tokens, None)
                page_hashes.append(page_hash)

            page_hash_with_group_id = PageHashWithGroupId(page_hash, kv_cache_group_id)
            blk.page_hash = page_hash_with_group_id
            self.cached_page_hash_to_page[page_hash_with_group_id][blk.page_id] = blk
            if new_hashes is not None:
                new_hashes.append(page_hash.hash_value)
            prev_page_hash_value = page_hash.hash_value

    def get_new_pages(self, num_pages: int) -> list[CachePage]:
        """Get new pages from the free page pool.

        Note that we do not check page cache in this function.

        Args:
            num_pages: The number of pages to allocate.

        Returns:
            A list of new page.
        """
        if num_pages > self.get_num_free_pages():
            raise ValueError(f"Cannot get {num_pages} free pages from the pool")

        ret: list[CachePage] = self.free_page_queue.popleft_n(num_pages)

        if self.enable_caching:
            for page in ret:
                self._maybe_evict_cached_page(page)
                assert page.ref_cnt == 0
                page.ref_cnt += 1
        else:
            for page in ret:
                assert page.ref_cnt == 0
                page.ref_cnt += 1
        return ret

    def _maybe_evict_cached_page(self, page: CachePage) -> bool:
        """
        If a page is cached in `cached_page_hash_to_page`, we reset its hash
        metadata and evict it from the cache.

        Args:
            page: The page to evict.

        Returns:
            True if the page is evicted, False otherwise.
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
        """Touch a page increases its reference count by 1, and may remove
        the page from the free queue. This is used when a page is hit by
        another request with the same prefix.

        Args:
            pages: A list of pages to touch.
        """
        for pages_per_group in pages:
            for page in pages_per_group:
                if page.ref_cnt == 0 and not page.is_null:
                    self.free_page_queue.remove(page)
                page.incr_ref()

    def free_pages(self, ordered_pages: Iterable[CachePage]) -> None:
        """Free a list of pages. The pages should be ordered by their
        eviction priority, where the first page will be evicted first.

        Args:
            ordered_pages: A list of pages to free ordered by their eviction
                priority.
        """

        pages_list = list(ordered_pages)
        for page in pages_list:
            page.ref_cnt -= 1
        self.free_page_queue.append_n([page for page in pages_list if page.ref_cnt == 0 and not page.is_null])

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_pages = self.num_pages - self.get_num_free_pages()
        if num_used_pages != 1:
            print("Failed to reset prefix cache because some pages (%d) are not freed yet", num_used_pages - 1)
            return False

        self.cached_page_hash_to_page = defaultdict(dict)

        for page in self.pages:
            page.reset_hash()

        print("Successfully reset prefix cache")

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
