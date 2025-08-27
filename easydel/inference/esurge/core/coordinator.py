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

from ..request import EngineRequest
from .interface import CacheGroupSpec, FullAttentionSpec
from .page_pool import PagePool
from .single_type_cache_manager import FullAttentionManager, get_manager_for_kv_cache_spec
from .utils import CachePage, PageHash


class CacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
    ):
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

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> list[int]:
        """
        Get the number of common prefix pages for a request.

        Args:
            request_id: The request ID.
            page_hashes: The page hashes of the request.

        Returns:
            The number of common prefix pages.
        """
        num_pages_per_group = [
            manager.get_num_common_prefix_pages(request_id, num_running_requests)
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
        """
        Get the pages for the request.
        """
        return tuple(manager.req_to_pages.get(request_id) or [] for manager in self.single_type_managers)

    @abstractmethod
    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[CachePage], ...], int]:
        pass


class CacheCoordinatorNoPrefixCache(CacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryCacheCoordinator and HybridCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
    ):
        super().__init__(num_pages, kv_cache_groups, max_model_len, use_eagle, False)
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_pages(self, request_id: str, num_running_requests: int) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[CachePage], ...], int]:
        pages: tuple[list[CachePage], ...] = tuple([] for _ in range(self.num_single_type_manager))
        return pages, 0


class UnitaryCacheCoordinator(CacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
    ):
        super().__init__(num_pages, kv_cache_groups, max_model_len, use_eagle, enable_caching)
        self.kv_cache_spec = self.kv_cache_groups[0].kv_cache_spec
        self.page_size = self.kv_cache_spec.page_size
        assert len(self.kv_cache_groups) == 1, "UnitaryCacheCoordinator assumes only one kv cache group"

    def find_longest_cache_hit(
        self,
        page_hashes: list[PageHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[CachePage], ...], int]:
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
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
    ):
        super().__init__(num_pages, kv_cache_groups, max_model_len, use_eagle, enable_caching)
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Verifies that the model has exactly two types of KV cache groups, and
        one of them is full attention. Then, split the kv cache groups into full
        attention groups and other groups.
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
                    assert (
                        full_attention_type_id == g.kv_cache_spec.type_id
                    ), "HybridCacheCoordinator assumes exactly one type of full attention groups now."
                self.full_attention_group_ids.append(i)
            else:
                if other_type_id is None:
                    other_type_id = g.kv_cache_spec.type_id
                else:
                    assert (
                        other_type_id == g.kv_cache_spec.type_id
                    ), "HybridCacheCoordinator assumes exactly one other type of groups now."
                self.other_group_ids.append(i)

        assert (
            full_attention_type_id is not None
        ), "HybridCacheCoordinator assumes exactly one type of full attention groups now."
        assert other_type_id is not None, "HybridCacheCoordinator assumes exactly one type of other groups now."

        self.full_attention_manager_cls = FullAttentionManager
        self.other_attention_cls = self.single_type_managers[self.other_group_ids[0]].__class__

        self.full_attention_spec = self.kv_cache_groups[self.full_attention_group_ids[0]].kv_cache_spec
        self.other_spec = self.kv_cache_groups[self.other_group_ids[0]].kv_cache_spec

        self.full_attention_page_size = self.full_attention_spec.page_size
        self.other_page_size = self.other_spec.page_size

        if self.enable_caching:
            divisible = self.other_page_size % self.full_attention_page_size
            assert (
                divisible == 0
            ), "CacheCoordinator assumes the page_size of full attention layers is divisible by other layers now."

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
        """
        Find the longest cache hit for the request.

        Args:
            page_hashes: The page hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A list of the cache hit pages for each single type manager.
                - The number of tokens of the longest cache hit.
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
