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

from collections import defaultdict
from dataclasses import dataclass

from ..request import EngineRequest, EngineRequestStatus
from .coordinator import get_kv_cache_coordinator
from .interface import CacheGroupSpec
from .utils import CachePage, PageHash, hash_request_tokens, init_none_hash


@dataclass
class CachePages:
    """
    The allocation result of CacheManager, work as the interface between
    Scheduler and CacheManager, to hide CacheManager's internal data
    structure from the Scheduler.
    """

    pages: tuple[list[CachePage], ...]

    def __add__(self, other: "CachePages") -> "CachePages":
        """Adds two CachePages instances."""
        return CachePages(tuple(blk1 + blk2 for blk1, blk2 in zip(self.pages, other.pages, strict=False)))

    def get_page_ids(self) -> tuple[list[int], ...]:
        """
        Converts the CachePages instance to page_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where
            * the outer tuple corresponds to KV cache groups
            * each inner list contains the page_ids of the pages in that group
        """
        return tuple([blk.page_id for blk in group] for group in self.pages)

    def get_unhashed_page_ids(self) -> list[int]:
        """Get page_ids of unhashed pages from CachePages instance."""
        assert len(self.pages) == 1, "Only one group is supported"
        return [page.page_id for page in self.pages[0] if page.page_hash is None]

    def new_empty(self) -> "CachePages":
        """Creates a new CachePages instance with no pages."""
        return CachePages(tuple([] for _ in range(len(self.pages))))


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
    ) -> None:
        self.num_pages = num_pages
        self.kv_cache_groups = kv_cache_groups
        self.max_model_len = max_model_len

        if len(kv_cache_groups) == 0:
            enable_caching = False
        self.enable_caching = enable_caching
        init_none_hash()
        self.use_eagle = use_eagle

        self.page_size: int | None = None
        if self.enable_caching:
            self.page_size = kv_cache_groups[0].kv_cache_spec.page_size

        self.coordinator = get_kv_cache_coordinator(
            num_pages=self.num_pages,
            kv_cache_groups=self.kv_cache_groups,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
        )
        self.num_kv_cache_groups = len(kv_cache_groups)
        self.page_pool = self.coordinator.page_pool

        self.req_to_page_hashes: defaultdict[str, list[PageHash]] = defaultdict(list)

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.page_pool.get_usage()

    def get_computed_pages(self, request: EngineRequest) -> tuple[CachePages, int]:
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
            page_hashes = hash_request_tokens(hash, self.page_size, request)
            self.req_to_page_hashes[request.request_id] = page_hashes

        max_cache_hit_length = request.num_tokens - 1
        computed_pages, num_new_computed_tokens = self.coordinator.find_longest_cache_hit(
            page_hashes, max_cache_hit_length
        )

        return CachePages(computed_pages), num_new_computed_tokens

    def allocate_slots(
        self,
        request: EngineRequest,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_pages: CachePages | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_pages: bool = False,
    ) -> CachePages | None:
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

        Pages layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_pages are illustrated in this layout.

        Returns:
            A list of new allocated pages.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_pages is not None:
            new_computed_page_list = new_computed_pages.pages
        else:
            new_computed_page_list = tuple([] for _ in range(len(self.kv_cache_groups)))

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

        self.coordinator.save_new_computed_pages(request.request_id, new_computed_page_list)

        new_pages = self.coordinator.allocate_new_pages(request.request_id, num_tokens_need_slot)

        if not self.enable_caching or delay_cache_pages:
            return CachePages(new_pages)

        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens, request.num_tokens)
        self.coordinator.cache_pages(
            request,
            self.req_to_page_hashes[request.request_id],
            num_tokens_to_cache,
        )

        return CachePages(new_pages)

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
        return CachePages(self.coordinator.get_pages(request_id)).get_page_ids()

    def cache_pages(self, request: EngineRequest, num_computed_tokens: int) -> None:
        """Cache the pages for the request, if enabled."""
        if self.enable_caching:
            page_hashes = self.req_to_page_hashes[request.request_id]
            self.coordinator.cache_pages(request, page_hashes, num_computed_tokens)

    def create_empty_page_list(self) -> CachePages:
        """Creates a new CachePages instance with no pages."""
        return CachePages(tuple([] for _ in range(self.num_kv_cache_groups)))
