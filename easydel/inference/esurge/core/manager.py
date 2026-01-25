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

"""Cache management for KV-cache allocation and prefix caching.

This module provides the main CacheManager class that handles KV-cache
allocation, deallocation, and prefix caching for efficient inference.
It acts as the high-level interface between the scheduler and the underlying
cache coordinator.

Classes:
    CachePages: Allocation result wrapper for cache pages.
    CacheManager: Main cache management interface.

Example:
    >>> manager = CacheManager(
    ...     num_pages=1000,
    ...     kv_cache_groups=cache_groups,
    ...     max_model_len=4096,
    ...     enable_caching=True
    ... )
    >>> pages = manager.allocate_slots(request, num_new_tokens=10)
    >>> manager.free(request)
"""

from collections import defaultdict
from dataclasses import dataclass

from ..request import EngineRequest, EngineRequestStatus
from .coordinator import get_kv_cache_coordinator
from .interface import CacheGroupSpec
from .utils import CachePage, PageHash, hash_request_tokens, init_none_hash


@dataclass
class CachePages:
    """Allocation result container for cache pages.

    This dataclass wraps the allocation result from CacheManager, providing
    an interface between the Scheduler and CacheManager to hide internal
    data structures from the Scheduler.

    Attributes:
        pages: A tuple of lists where each list contains CachePage objects
            for a specific KV cache group. The outer tuple corresponds to
            different KV cache groups (e.g., full attention vs sliding window).

    Example:
        >>> cache_pages = manager.allocate_slots(request, num_new_tokens=10)
        >>> page_ids = cache_pages.get_page_ids()
        >>> print(page_ids)  # ((1, 2, 3), (4, 5, 6))
    """

    pages: tuple[list[CachePage], ...]

    def __add__(self, other: "CachePages") -> "CachePages":
        """Concatenate two CachePages instances.

        Combines the pages from two CachePages instances by concatenating
        the page lists for each corresponding KV cache group.

        Args:
            other: Another CachePages instance to add.

        Returns:
            A new CachePages instance with concatenated page lists.

        Example:
            >>> combined = cache_pages1 + cache_pages2
        """
        return CachePages(tuple(blk1 + blk2 for blk1, blk2 in zip(self.pages, other.pages, strict=False)))

    def get_page_ids(self) -> tuple[list[int], ...]:
        """Convert CachePages to page IDs.

        Extracts the page IDs from all CachePage objects, organized by
        KV cache group.

        Returns:
            A tuple of lists where the outer tuple corresponds to KV cache
            groups and each inner list contains the page_ids of the pages
            in that group.

        Example:
            >>> page_ids = cache_pages.get_page_ids()
            >>> print(page_ids)  # ((1, 2, 3), (4, 5, 6))
        """
        return tuple([blk.page_id for blk in group] for group in self.pages)

    def get_unhashed_page_ids(self) -> list[int]:
        """Get page IDs of pages without computed hashes.

        Returns the IDs of pages that haven't been hashed yet, which are
        typically newly allocated pages that haven't been filled with tokens.

        Returns:
            A list of page IDs for unhashed pages.

        Raises:
            AssertionError: If there is more than one KV cache group.

        Note:
            This method only supports single-group configurations.
        """
        assert len(self.pages) == 1, "Only one group is supported"
        return [page.page_id for page in self.pages[0] if page.page_hash is None]

    def new_empty(self) -> "CachePages":
        """Create an empty CachePages instance with the same structure.

        Creates a new CachePages instance with empty page lists for each
        KV cache group, preserving the number of groups.

        Returns:
            A new CachePages instance with no pages but the same group count.
        """
        return CachePages(tuple([] for _ in range(len(self.pages))))


class CacheManager:
    """High-level KV-cache manager for inference requests.

    This class provides the main interface for managing KV-cache allocation,
    deallocation, and prefix caching. It coordinates between the scheduler
    and the underlying cache coordinator to efficiently manage memory for
    multiple concurrent requests.

    The CacheManager supports:
    - Prefix caching for reusing computed KV states across requests
    - Multiple KV cache groups for hybrid attention patterns
    - EAGLE speculative decoding support
    - Automatic page eviction using LRU policy

    Attributes:
        num_pages: Total number of pages in the cache pool.
        kv_cache_groups: List of cache group specifications.
        max_model_len: Maximum sequence length supported.
        enable_caching: Whether prefix caching is enabled.
        use_eagle: Whether EAGLE speculative decoding is enabled.
        page_size: Number of tokens per page (if caching enabled).
        coordinator: The underlying cache coordinator.
        page_pool: The page pool managed by the coordinator.
        req_to_page_hashes: Mapping from request IDs to their page hashes.

    Example:
        >>> manager = CacheManager(
        ...     num_pages=1000,
        ...     kv_cache_groups=cache_groups,
        ...     max_model_len=4096,
        ...     enable_caching=True,
        ...     use_eagle=False
        ... )
        >>> # Get cached pages for a request
        >>> computed_pages, num_computed = manager.get_computed_pages(request)
        >>> # Allocate new pages
        >>> new_pages = manager.allocate_slots(request, num_new_tokens=10)
        >>> # Free pages when done
        >>> manager.free(request)
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
    ) -> None:
        """Initialize the CacheManager.

        Args:
            num_pages: Total number of pages to allocate in the cache pool.
            kv_cache_groups: List of CacheGroupSpec defining the cache
                configuration for each attention type.
            max_model_len: Maximum sequence length the model supports.
            enable_caching: Whether to enable prefix caching. Defaults to True.
                If no cache groups are provided, this is automatically disabled.
            use_eagle: Whether EAGLE speculative decoding is enabled.
                Defaults to False.
        """
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
        """Reset the prefix cache to invalidate all cached prefixes.

        This method clears all cached page hashes, forcing future requests
        to recompute their KV states. Useful in scenarios like:
        - RLHF training where model weights are updated between rounds
        - Benchmarking to ensure cold-start performance measurement
        - Cache invalidation after model parameter changes

        Returns:
            True if the prefix cache was successfully reset, False if
            there are still pages in use that couldn't be freed.

        Note:
            This operation will fail if any pages are still allocated
            to active requests. Ensure all requests are freed first.
        """
        if not self.page_pool.reset_prefix_cache():
            return False

        return True

    def get_num_common_prefix_pages(
        self,
        request: EngineRequest,
        num_scheduled_requests: int,
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
        return self.coordinator.get_num_common_prefix_pages(request.request_id, num_scheduled_requests)

    def free_page_hashes(self, request: EngineRequest) -> None:
        """Discard the page hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_page_hashes.pop(request.request_id, None)

    def get_page_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the page IDs allocated to a request.

        Args:
            request_id: The unique identifier of the request.

        Returns:
            A tuple of lists containing page IDs for each KV cache group.
        """
        return CachePages(self.coordinator.get_pages(request_id)).get_page_ids()

    def cache_pages(self, request: EngineRequest, num_computed_tokens: int) -> None:
        """Cache the pages for a request to enable prefix reuse.

        Updates the page hash metadata for the request's pages, making them
        available for prefix cache hits by future requests with matching
        token prefixes.

        Args:
            request: The request whose pages should be cached.
            num_computed_tokens: The number of tokens that have been computed
                and should be cached.

        Note:
            This is a no-op if prefix caching is disabled.
        """
        if self.enable_caching:
            page_hashes = self.req_to_page_hashes[request.request_id]
            self.coordinator.cache_pages(request, page_hashes, num_computed_tokens)

    def create_empty_page_list(self) -> CachePages:
        """Create an empty CachePages instance for the current configuration.

        Returns:
            A CachePages instance with empty page lists for each KV cache group.
        """
        return CachePages(tuple([] for _ in range(self.num_kv_cache_groups)))
