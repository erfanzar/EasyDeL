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
from collections import defaultdict

from ..request import EngineRequest
from ..utils import cdiv
from .interface import CacheSpec, ChunkedLocalAttentionSpec, FullAttentionSpec, MambaSpec, SlidingWindowSpec
from .page_pool import PagePool
from .utils import CachePage, PageHash


class SingleTypeCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: CacheSpec,
        page_pool: PagePool,
        kv_cache_group_id: int,
    ) -> None:
        """
        Initializes the SpecializedManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            page_pool: The page pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
        """

        self.page_size = kv_cache_spec.page_size
        self.kv_cache_spec = kv_cache_spec
        self.page_pool = page_pool

        self.req_to_pages: defaultdict[str, list[CachePage]] = defaultdict(list)

        self.num_cached_page: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_page = page_pool.null_page

    def get_num_pages_to_allocate(self, request_id: str, num_tokens: int, new_computed_pages: list[CachePage]) -> int:
        """
        Get the number of pages needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_pages: The new computed pages just hitting the
                prefix caching.

        Returns:
            The number of pages.
        """

        num_required_pages = cdiv(num_tokens, self.page_size)
        num_new_pages = num_required_pages - len(new_computed_pages) - len(self.req_to_pages[request_id])

        num_evictable_computed_pages = sum(blk.ref_cnt == 0 and not blk.is_null for blk in new_computed_pages)
        return num_new_pages + num_evictable_computed_pages

    def save_new_computed_pages(self, request_id: str, new_computed_pages: list[CachePage]) -> None:
        """
        Add the new computed pages to the request.

        Args:
            request_id: The request ID.
            new_computed_pages: The new computed pages just hitting the
                prefix cache.
        """
        if request_id not in self.num_cached_page:
            req_pages = self.req_to_pages[request_id]
            assert len(req_pages) == 0
            req_pages.extend(new_computed_pages)
            self.num_cached_page[request_id] = len(new_computed_pages)
        else:
            assert len(new_computed_pages) == 0

    def allocate_new_pages(self, request_id: str, num_tokens: int) -> list[CachePage]:
        """
        Allocate new pages for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            The new allocated pages.
        """
        req_pages = self.req_to_pages[request_id]
        num_required_pages = cdiv(num_tokens, self.page_size)
        num_new_pages = num_required_pages - len(req_pages)
        if num_new_pages <= 0:
            return []
        else:
            new_pages = self.page_pool.get_new_pages(num_new_pages)
            req_pages.extend(new_pages)
            return new_pages

    def cache_pages(self, request: EngineRequest, page_hashes: list[PageHash], num_tokens: int) -> None:
        """
        Cache the pages for the request.

        Args:
            request: The request.
            page_hashes: The page hashes of the request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_cached_pages = self.num_cached_page[request.request_id]
        num_full_pages = num_tokens // self.page_size

        self.page_pool.cache_full_pages(
            request=request,
            pages=self.req_to_pages[request.request_id],
            page_hashes=page_hashes,
            num_cached_pages=num_cached_pages,
            num_full_pages=num_full_pages,
            page_size=self.page_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_page[request.request_id] = num_full_pages

    def free(self, request_id: str) -> None:
        """
        Free the pages for the request.

        Args:
            request_id: The request ID.
        """

        req_pages = self.req_to_pages.pop(request_id, [])

        ordered_pages = reversed(req_pages)

        self.page_pool.free_pages(ordered_pages)
        self.num_cached_page.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        """
        Get the number of common prefix pages for a request.

        Args:
            request_id: The request ID.
            page_hashes: The page hashes of the request.

        Returns:
            The number of common prefix pages.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: CacheSpec,
        use_eagle: bool,
    ) -> tuple[list[CachePage], ...]:
        """
        Get the longest cache hit prefix of the pages that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.
        If eagle is enabled, drop the last matched page to force recompute the
        last page to get the required hidden states for eagle drafting head.
        Need to be customized for each attention type.

        Args:
            page_hashes: The page hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            page_pool: The page pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached pages with skipped pages replaced by null page
            for each kv cache group in `kv_cache_group_ids`.
            Return a list of length `len(kv_cache_group_ids)`, where the i-th
            element is a list of cached pages for the i-th kv cache group
            in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, CachePage(7), CachePage(8)]) for page size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """

        raise NotImplementedError

    @abstractmethod
    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        """
        Remove the pages that are no longer needed from `pages` and free the
        pages. The removed pages should be replaced by null_page.
        Need to be customized for each attention type.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        raise NotImplementedError


class FullAttentionManager(SingleTypeCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: CacheSpec,
        use_eagle: bool,
    ) -> tuple[list[CachePage], ...]:
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), "FullAttentionManager can only be used for full attention and chunked local attention groups"
        computed_pages: tuple[list[CachePage], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        max_num_pages = max_length // kv_cache_spec.page_size
        for _, page_hash in zip(range(max_num_pages), page_hashes, strict=False):
            if cached_page := page_pool.get_cached_page(page_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_pages, cached_page, strict=False):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_pages[0]:
            for computed in computed_pages:
                computed.pop()
        return computed_pages

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        pass

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        pages = self.req_to_pages[request_id]
        num_common_pages = 0
        for page in pages:
            if page.ref_cnt == num_running_requests:
                num_common_pages += 1
            else:
                break
        return num_common_pages


class SlidingWindowManager(SingleTypeCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, page_pool: PagePool, **kwargs) -> None:
        super().__init__(kv_cache_spec, page_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_page = page_pool.null_page

    @classmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: CacheSpec,
        use_eagle: bool,
    ) -> tuple[list[CachePage], ...]:
        assert isinstance(
            kv_cache_spec, SlidingWindowSpec
        ), "SlidingWindowManager can only be used for sliding window groups"

        sliding_window_contiguous_pages = cdiv(kv_cache_spec.sliding_window - 1, kv_cache_spec.page_size)
        if use_eagle:
            sliding_window_contiguous_pages += 1

        max_num_pages = max_length // kv_cache_spec.page_size
        computed_pages = tuple([page_pool.null_page] * max_num_pages for _ in range(len(kv_cache_group_ids)))
        num_contiguous_pages = 0
        match_found = False

        for i in range(max_num_pages - 1, -1, -1):
            if cached_page := page_pool.get_cached_page(page_hashes[i], kv_cache_group_ids):
                for computed, cached in zip(computed_pages, cached_page, strict=False):
                    computed[i] = cached
                num_contiguous_pages += 1
                if num_contiguous_pages >= sliding_window_contiguous_pages:
                    for computed in computed_pages:
                        del computed[i + num_contiguous_pages :]
                    match_found = True
                    break
            else:
                num_contiguous_pages = 0
        if not match_found:
            for computed in computed_pages:
                del computed[num_contiguous_pages:]
        if use_eagle and computed_pages[0]:
            for computed in computed_pages:
                computed.pop()
        return computed_pages

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        last_useful_token = num_computed_tokens - self.sliding_window + 1
        last_useful_page = last_useful_token // self.page_size
        pages = self.req_to_pages[request_id]
        removed_pages: list[CachePage] = []
        for i in range(last_useful_page - 1, -1, -1):
            if pages[i] == self._null_page:
                break
            removed_pages.append(pages[i])
            pages[i] = self._null_page
        self.page_pool.free_pages(removed_pages)

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        """
        NOTE(Chen): The prefix pages are null pages for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeCacheManager):
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, page_pool: PagePool, **kwargs) -> None:
        super().__init__(kv_cache_spec, page_pool, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size
        self._null_page = page_pool.null_page

    @classmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: CacheSpec,
        use_eagle: bool,
    ) -> tuple[list[CachePage], ...]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the pages that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole page is outside of the local
        window, and set the page as null. Examples:

        1. Attention chunk size of 8, page size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        page3 (8th - 11th tokens), Assume page 3 is hit, we will return
        [null, null, page 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, page size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 pages[null, null, null, null]

        Args:
            page_hashes: The page hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            page_pool: The page pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached pages
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for " + "chunked local attention groups"
        )
        assert use_eagle is False, "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        max_num_pages = max_length // kv_cache_spec.page_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length // kv_cache_spec.attention_chunk_size * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0

        local_attention_start_page_idx = local_attention_start_idx // kv_cache_spec.page_size
        computed_pages: tuple[list[CachePage], ...] = tuple(
            [page_pool.null_page] * local_attention_start_page_idx for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_page_idx, max_num_pages):
            page_hash = page_hashes[i]
            if cached_page := page_pool.get_cached_page(page_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_pages, cached_page, strict=False):
                    computed.append(cached)
            else:
                break
        return computed_pages

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        num_cached_page = self.num_cached_page.get(request_id, 0)
        local_attention_start_idx = (num_computed_tokens) // self.attention_chunk_size * self.attention_chunk_size
        first_useful_page_idx = local_attention_start_idx // self.page_size
        if num_cached_page > 0:
            first_useful_page_idx = min(first_useful_page_idx, num_cached_page - 1)

        pages = self.req_to_pages[request_id]
        removed_pages: list[CachePage] = []

        for i in range(first_useful_page_idx - 1, -1, -1):
            if pages[i] == self._null_page:
                break
            removed_pages.append(pages[i])
            pages[i] = self._null_page
        self.page_pool.free_pages(removed_pages)

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


class MambaManager(SingleTypeCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        page_hashes: list[PageHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        page_pool: PagePool,
        kv_cache_spec: CacheSpec,
        use_eagle: bool,
    ) -> tuple[list[CachePage], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), "MambaManager can only be used for mamba groups"

        computed_pages: tuple[list[CachePage], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        return computed_pages

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        pass

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> int:
        return 0

    def allocate_new_pages(self, request_id: str, num_tokens: int) -> list[CachePage]:
        new_pages = super().allocate_new_pages(request_id, num_tokens)
        assert len(self.req_to_pages[request_id]) == 1, "MambaManager should only allocate 1 page for each request."
        return new_pages


spec_manager_map: dict[type[CacheSpec], type[SingleTypeCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
}


def get_manager_for_kv_cache_spec(kv_cache_spec: CacheSpec, **kwargs) -> SingleTypeCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
