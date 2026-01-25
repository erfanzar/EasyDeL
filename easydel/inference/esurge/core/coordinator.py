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

"""Cache coordinator for managing multiple KV cache groups.

This module provides coordinator classes that manage KV-cache operations
across multiple cache groups. The coordinators handle the complexity of
hybrid models that use different attention patterns (e.g., full attention
and sliding window) in different layers.

Classes:
    CacheCoordinator: Abstract base class for cache coordination.
    CacheCoordinatorNoPrefixCache: Coordinator when prefix caching is disabled.
    UnitaryCacheCoordinator: Coordinator for single cache group models.
    HybridCacheCoordinator: Coordinator for multi-group hybrid models.

Functions:
    get_kv_cache_coordinator: Factory function to create appropriate coordinator.

Example:
    >>> coordinator = get_kv_cache_coordinator(
    ...     num_pages=1000,
    ...     kv_cache_groups=cache_groups,
    ...     max_model_len=4096,
    ...     use_eagle=False,
    ...     enable_caching=True
    ... )
"""

from abc import ABC, abstractmethod

from ..request import EngineRequest
from .interface import CacheGroupSpec, FullAttentionSpec
from .page_pool import PagePool
from .single_type_cache_manager import FullAttentionManager, get_manager_for_kv_cache_spec
from .utils import CachePage, PageHash


class CacheCoordinator(ABC):
    """Abstract base class for coordinating KV-cache across multiple groups.

    This class coordinates KV-cache management operations across multiple
    cache groups, each potentially using different attention patterns.
    It delegates attention-type-specific operations to SingleTypeCacheManager
    instances while providing a unified interface for the CacheManager.

    Attributes:
        num_pages: Total number of pages in the shared page pool.
        kv_cache_groups: List of cache group specifications.
        max_model_len: Maximum sequence length supported by the model.
        enable_caching: Whether prefix caching is enabled.
        page_pool: The shared page pool for all cache groups.
        use_eagle: Whether EAGLE speculative decoding is enabled.
        single_type_managers: Tuple of managers for each cache group.

    Note:
        Subclasses must implement `find_longest_cache_hit` to define
        the cache lookup strategy for their specific configuration.
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
    ):
        """Initialize the cache coordinator.

        Args:
            num_pages: Total number of pages to allocate in the page pool.
            kv_cache_groups: List of CacheGroupSpec defining each cache group.
            max_model_len: Maximum sequence length the model supports.
            use_eagle: Whether EAGLE speculative decoding is enabled.
            enable_caching: Whether to enable prefix caching.
        """
        self.num_pages = num_pages
        self.kv_cache_groups = kv_cache_groups
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.page_pool = PagePool(self.num_pages, enable_caching)

        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                page_pool=self.page_pool,
                kv_cache_group_id=i,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_groups)
        )

    def get_num_pages_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_pages: tuple[list[CachePage], ...],
    ) -> int:
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
        num_pages_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_pages_to_allocate += manager.get_num_pages_to_allocate(request_id, num_tokens, new_computed_pages[i])
        return num_pages_to_allocate

    def save_new_computed_pages(self, request_id: str, new_computed_pages: tuple[list[CachePage], ...]) -> None:
        """
        Add the new computed pages to the request.

        Args:
            request_id: The request ID.
            new_computed_pages: The new computed pages just hitting the
                prefix cache.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_pages(request_id, new_computed_pages[i])

    def allocate_new_pages(self, request_id: str, num_tokens: int) -> tuple[list[CachePage], ...]:
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
        return tuple(manager.allocate_new_pages(request_id, num_tokens) for manager in self.single_type_managers)

    def cache_pages(self, request: EngineRequest, page_hashes: list[PageHash], num_computed_tokens: int) -> None:
        """
        Cache the pages for the request.

        Args:
            request: The request.
            page_hashes: The page hashes of the request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_pages(request, page_hashes, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the pages for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int) -> list[int]:
        """
        Get the number of common prefix pages for a request.

        Args:
            request_id: The request ID.
            page_hashes: The page hashes of the request.

        Returns:
            The number of common prefix pages.
        """
        num_pages_per_group = [
            manager.get_num_common_prefix_pages(request_id, num_scheduled_requests)
            for manager in self.single_type_managers
        ]
        return num_pages_per_group

    def remove_skipped_pages(self, request_id: str, num_computed_tokens: int) -> None:
        """
        Remove the pages that are no longer needed from `pages` and replace
        the removed pages with null_page.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_pages(request_id, num_computed_tokens)

    def get_pages(self, request_id: str) -> tuple[list[CachePage], ...]:
        """Get all pages allocated to a request across all cache groups.

        Args:
            request_id: The unique identifier of the request.

        Returns:
            Tuple of page lists, one for each cache group.
        """
        return tuple(manager.req_to_pages.get(request_id) or [] for manager in self.single_type_managers)

    @abstractmethod
    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[CachePage], ...], int]:
        """Find the longest prefix cache hit for a sequence of page hashes.

        Args:
            page_hashes: List of page hashes to look up in the cache.
            max_cache_hit_length: Maximum number of tokens to consider.

        Returns:
            A tuple containing:
            - Tuple of page lists, one per cache group, with cached pages
            - Number of tokens in the cache hit
        """
        pass


class CacheCoordinatorNoPrefixCache(CacheCoordinator):
    """Cache coordinator for configurations without prefix caching.

    This coordinator is used when prefix caching is disabled or unsupported.
    It provides a minimal implementation that allocates pages without any
    caching behavior. Unlike other coordinators, it supports arbitrary
    numbers of KV cache groups including zero groups.

    Example:
        >>> coordinator = CacheCoordinatorNoPrefixCache(
        ...     num_pages=1000,
        ...     kv_cache_groups=[],
        ...     max_model_len=4096,
        ...     use_eagle=False
        ... )
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
    ):
        """Initialize the no-prefix-cache coordinator.

        Args:
            num_pages: Total number of pages in the page pool.
            kv_cache_groups: List of cache group specifications (can be empty).
            max_model_len: Maximum sequence length supported.
            use_eagle: Whether EAGLE speculative decoding is enabled.
        """
        super().__init__(num_pages, kv_cache_groups, max_model_len, use_eagle, False)
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int) -> list[int]:
        """Get common prefix page counts (always 0 without caching).

        Args:
            request_id: The request ID.
            num_scheduled_requests: Number of scheduled requests.

        Returns:
            List of zeros, one per cache group.
        """
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[CachePage], ...], int]:
        """Find cache hit (always returns empty without caching).

        Args:
            page_hashes: List of page hashes (ignored).
            max_cache_hit_length: Maximum hit length (ignored).

        Returns:
            Empty page lists and zero hit length.
        """
        pages: tuple[list[CachePage], ...] = tuple([] for _ in range(self.num_single_type_manager))
        return pages, 0


class UnitaryCacheCoordinator(CacheCoordinator):
    """Cache coordinator for models with a single KV cache group.

    This coordinator handles the common case where all attention layers
    use the same attention pattern (e.g., all full attention or all
    sliding window). It provides optimized cache lookup for this
    simpler configuration.

    Attributes:
        kv_cache_spec: The cache specification for the single group.
        page_size: Number of tokens per page.

    Example:
        >>> coordinator = UnitaryCacheCoordinator(
        ...     num_pages=1000,
        ...     kv_cache_groups=[full_attention_group],
        ...     max_model_len=4096,
        ...     use_eagle=False,
        ...     enable_caching=True
        ... )
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
    ):
        """Initialize the unitary cache coordinator.

        Args:
            num_pages: Total number of pages in the page pool.
            kv_cache_groups: List containing exactly one cache group.
            max_model_len: Maximum sequence length supported.
            use_eagle: Whether EAGLE speculative decoding is enabled.
            enable_caching: Whether prefix caching is enabled.

        Raises:
            AssertionError: If more than one cache group is provided.
        """
        super().__init__(num_pages, kv_cache_groups, max_model_len, use_eagle, enable_caching)
        self.kv_cache_spec = self.kv_cache_groups[0].kv_cache_spec
        self.page_size = self.kv_cache_spec.page_size
        assert len(self.kv_cache_groups) == 1, "UnitaryCacheCoordinator assumes only one kv cache group"

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[CachePage], ...], int]:
        """Find the longest cache hit for the single cache group.

        Args:
            page_hashes: List of page hashes to look up.
            max_cache_hit_length: Maximum number of tokens to consider.

        Returns:
            A tuple containing:
            - Single-element tuple with the list of cached pages
            - Number of tokens in the cache hit
        """
        hit_pages = self.single_type_managers[0].find_longest_cache_hit(
            page_hashes=page_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            page_pool=self.page_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
        )
        return hit_pages, len(hit_pages[0]) * self.page_size


class HybridCacheCoordinator(CacheCoordinator):
    """Cache coordinator for hybrid models with multiple attention types.

    This coordinator handles models that use different attention patterns
    in different layers (e.g., some layers with full attention and others
    with sliding window). It coordinates cache lookup across both types
    to find a consistent prefix cache hit.

    Currently supports exactly two attention types, with one being full
    attention. Future versions may support more general configurations.

    Attributes:
        full_attention_group_ids: Indices of full attention cache groups.
        other_group_ids: Indices of non-full-attention cache groups.
        full_attention_spec: Cache spec for full attention groups.
        other_spec: Cache spec for other attention groups.
        full_attn_first: Whether full attention groups precede others.

    Example:
        >>> coordinator = HybridCacheCoordinator(
        ...     num_pages=1000,
        ...     kv_cache_groups=[full_attn_group, sliding_window_group],
        ...     max_model_len=4096,
        ...     use_eagle=False,
        ...     enable_caching=True
        ... )
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
    ):
        """Initialize the hybrid cache coordinator.

        Args:
            num_pages: Total number of pages in the page pool.
            kv_cache_groups: List of cache groups with exactly two types.
            max_model_len: Maximum sequence length supported.
            use_eagle: Whether EAGLE speculative decoding is enabled.
            enable_caching: Whether prefix caching is enabled.

        Raises:
            AssertionError: If cache groups don't meet the hybrid requirements.
        """
        super().__init__(num_pages, kv_cache_groups, max_model_len, use_eagle, enable_caching)
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """Verify cache group configuration and split by attention type.

        Validates that the model has exactly two types of KV cache groups,
        with one being full attention. Splits groups into full attention
        and other attention for coordinated cache lookup.

        Raises:
            AssertionError: If configuration doesn't meet requirements:
                - Must have exactly one full attention type
                - Must have exactly one other attention type
                - Groups must not interleave by type
        """
        full_attention_type_id: str | None = None
        other_type_id: str | None = None
        self.full_attention_group_ids: list[int] = []
        self.other_group_ids: list[int] = []
        for i, g in enumerate(self.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                if full_attention_type_id is None:
                    full_attention_type_id = g.kv_cache_spec.type_id
                else:
                    assert full_attention_type_id == g.kv_cache_spec.type_id, (
                        "HybridCacheCoordinator assumes exactly one type of full attention groups now."
                    )
                self.full_attention_group_ids.append(i)
            else:
                if other_type_id is None:
                    other_type_id = g.kv_cache_spec.type_id
                else:
                    assert other_type_id == g.kv_cache_spec.type_id, (
                        "HybridCacheCoordinator assumes exactly one other type of groups now."
                    )
                self.other_group_ids.append(i)

        assert full_attention_type_id is not None, (
            "HybridCacheCoordinator assumes exactly one type of full attention groups now."
        )
        assert other_type_id is not None, "HybridCacheCoordinator assumes exactly one type of other groups now."

        self.full_attention_manager_cls = FullAttentionManager
        self.other_attention_cls = self.single_type_managers[self.other_group_ids[0]].__class__

        self.full_attention_spec = self.kv_cache_groups[self.full_attention_group_ids[0]].kv_cache_spec
        self.other_spec = self.kv_cache_groups[self.other_group_ids[0]].kv_cache_spec

        self.full_attention_page_size = self.full_attention_spec.page_size
        self.other_page_size = self.other_spec.page_size

        if self.enable_caching:
            divisible = self.other_page_size % self.full_attention_page_size
            assert divisible == 0, (
                "CacheCoordinator assumes the page_size of full attention layers is divisible by other layers now."
            )

        if max(self.full_attention_group_ids) < min(self.other_group_ids):
            self.full_attn_first = True
        elif max(self.other_group_ids) < min(self.full_attention_group_ids):
            self.full_attn_first = False
        else:
            raise ValueError(
                "HybridCacheCoordinator assumes the full "
                "attention group ids and other attention group ids "
                "do not interleave, either full attention group ids "
                "are before other attention group ids or vice versa."
                "This is for simplifying merging hit_pages_full_attn and "
                "hit_pages_other_attn to hit_pages."
            )

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[CachePage], ...], int]:
        """Find the longest consistent cache hit across all cache groups.

        Performs coordinated cache lookup across full attention and other
        attention groups to find the longest prefix that hits in both.
        The full attention groups are searched first, then the other groups
        are searched up to the hit length from full attention.

        Args:
            page_hashes: List of page hashes to look up in the cache.
            max_cache_hit_length: Maximum number of tokens to consider.

        Returns:
            A tuple containing:
            - Tuple of page lists, ordered by cache group index
            - Number of tokens in the consistent cache hit

        Note:
            The hit length is constrained by both attention types - if one
            type has a shorter hit, the other is truncated to match.
        """
        hit_pages_full_attn = self.full_attention_manager_cls.find_longest_cache_hit(
            page_hashes=page_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=self.full_attention_group_ids,
            page_pool=self.page_pool,
            kv_cache_spec=self.full_attention_spec,
            use_eagle=self.use_eagle,
        )
        hit_length = len(hit_pages_full_attn[0]) * self.full_attention_page_size

        hit_pages_other_attn = self.other_attention_cls.find_longest_cache_hit(
            page_hashes=page_hashes,
            max_length=hit_length,
            kv_cache_group_ids=self.other_group_ids,
            page_pool=self.page_pool,
            kv_cache_spec=self.other_spec,
            use_eagle=self.use_eagle,
        )
        hit_length = len(hit_pages_other_attn[0]) * self.other_page_size

        assert hit_length % self.full_attention_page_size == 0

        for group_hit_pages in hit_pages_full_attn:
            del group_hit_pages[hit_length // self.full_attention_page_size :]

        if self.full_attn_first:
            hit_pages = hit_pages_full_attn + hit_pages_other_attn
        else:
            hit_pages = hit_pages_other_attn + hit_pages_full_attn
        return hit_pages, hit_length


def get_kv_cache_coordinator(
    num_pages: int,
    kv_cache_groups: list[CacheGroupSpec],
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
) -> CacheCoordinator:
    """Factory function to create the appropriate cache coordinator.

    Selects and instantiates the correct CacheCoordinator subclass based
    on the caching configuration and number of cache groups.

    Args:
        num_pages: Total number of pages to allocate in the page pool.
        kv_cache_groups: List of cache group specifications.
        max_model_len: Maximum sequence length the model supports.
        use_eagle: Whether EAGLE speculative decoding is enabled.
        enable_caching: Whether prefix caching should be enabled.

    Returns:
        An appropriate CacheCoordinator instance:
        - CacheCoordinatorNoPrefixCache: When caching is disabled
        - UnitaryCacheCoordinator: When there's exactly one cache group
        - HybridCacheCoordinator: When there are multiple cache groups

    Example:
        >>> coordinator = get_kv_cache_coordinator(
        ...     num_pages=1000,
        ...     kv_cache_groups=groups,
        ...     max_model_len=4096,
        ...     use_eagle=False,
        ...     enable_caching=True
        ... )
    """
    if not enable_caching:
        return CacheCoordinatorNoPrefixCache(
            num_pages,
            kv_cache_groups,
            max_model_len,
            use_eagle,
        )
    if len(kv_cache_groups) == 1:
        return UnitaryCacheCoordinator(
            num_pages,
            kv_cache_groups,
            max_model_len,
            use_eagle,
            enable_caching,
        )
    return HybridCacheCoordinator(
        num_pages,
        kv_cache_groups,
        max_model_len,
        use_eagle,
        enable_caching,
    )
