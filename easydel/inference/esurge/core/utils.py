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

"""Utility classes and functions for KV-cache management.

This module provides the foundational data structures and utility functions
for managing KV-cache pages in the eSurge inference engine. It includes
page metadata classes, hash computation utilities, and a doubly-linked list
implementation for efficient free page management.

Classes:
    PageHash: Named tuple representing a page's hash value and token contents.
    PageHashWithGroupId: Named tuple combining PageHash with KV cache group ID.
    CachePage: Dataclass representing a single cache page's metadata.
    FreeCachePageQueue: Doubly-linked list for O(1) free page operations.

Functions:
    init_none_hash: Initialize the global hash seed for prefix caching.
    hash_page_tokens: Compute hash for a single page's token contents.
    hash_request_tokens: Compute hashes for all pages in a request.

Example:
    >>> init_none_hash()
    >>> page_hash = hash_page_tokens(hash, None, (1, 2, 3, 4))
    >>> print(page_hash.hash_value)
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

from ..request import EngineRequest


class PageHash(NamedTuple):
    """Hash representation of a cache page for prefix caching.

    This named tuple stores the hash value of a page along with the original
    token IDs and any extra keys (e.g., multimodal content hashes). The full
    tuple is used as the cache key to minimize hash collision probability.

    Attributes:
        hash_value: Integer hash computed from parent hash, tokens, and extra keys.
        token_ids: Tuple of token IDs contained in this page.
        extra_keys: Optional additional keys for the hash (e.g., image hashes
            for vision-language models). Defaults to None.

    Note:
        While the hash_value alone could identify pages, we keep the full
        token_ids and extra_keys to verify matches and handle the extremely
        rare case of hash collisions. Using SHA256-based hashing makes
        collisions practically impossible.
    """

    hash_value: int
    token_ids: tuple[int, ...]
    extra_keys: Any | None = None


class PageHashWithGroupId(NamedTuple):
    """Page hash combined with its KV cache group identifier.

    This named tuple associates a PageHash with the KV cache group it belongs to,
    enabling correct page lookup in hybrid models with multiple attention types.

    Attributes:
        page_hash: The PageHash containing the hash value and token IDs.
        group_id: The index of the KV cache group this page belongs to.
    """

    page_hash: PageHash
    group_id: int

    def get_hash_value(self) -> int:
        """Get the underlying hash value.

        Returns:
            The integer hash value from the contained PageHash.
        """
        return self.page_hash.hash_value


@dataclass
class CachePage:
    """Metadata container for a single KV-cache page.

    This dataclass tracks all metadata associated with a cache page, including
    its reference count for garbage collection, hash for prefix caching, and
    linked list pointers for efficient free queue management.

    Attributes:
        page_id: Unique identifier for this page in the page pool.
        ref_cnt: Number of active references to this page. When 0, the page
            can be evicted or reused.
        _page_hash: Private storage for the page's hash (use page_hash property).
        prev_free_page: Pointer to previous page in free list (if free).
        next_free_page: Pointer to next page in free list (if free).
        is_null: Whether this is a special null/placeholder page that should
            never be freed or evicted.

    Example:
        >>> page = CachePage(page_id=0)
        >>> page.incr_ref()
        >>> print(page.ref_cnt)  # 1
        >>> page.decr_ref()
        >>> print(page.ref_cnt)  # 0
    """

    page_id: int
    ref_cnt: int = 0
    _page_hash: PageHashWithGroupId | None = None
    prev_free_page: Optional["CachePage"] = None
    next_free_page: Optional["CachePage"] = None
    is_null: bool = False

    def incr_ref(self) -> None:
        """Increment the reference count by 1.

        Called when a new request starts using this page, either through
        allocation or prefix cache hit.
        """
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        """Decrement the reference count by 1.

        Called when a request releases this page. When ref_cnt reaches 0,
        the page becomes eligible for eviction or reuse.
        """
        self.ref_cnt -= 1

    @property
    def page_hash(self) -> PageHashWithGroupId | None:
        """Get the page's hash for prefix caching.

        Returns:
            The PageHashWithGroupId if the page has been hashed, None otherwise.
        """
        return self._page_hash

    @page_hash.setter
    def page_hash(self, page_hash: PageHashWithGroupId) -> None:
        """Set the page's hash for prefix caching.

        Args:
            page_hash: The hash to associate with this page.

        Raises:
            AssertionError: If the page already has a hash assigned.
        """
        assert self.page_hash is None, "The page already has a hash. This should not happen."
        self._page_hash = page_hash

    def reset_hash(self) -> None:
        """Clear the page hash when the page is evicted.

        Called during page eviction to remove the hash association,
        allowing the page to be reused for new content.
        """
        self._page_hash = None

    def __repr__(self) -> str:
        """Return a detailed string representation of the page.

        Returns:
            String showing page_id, ref_cnt, hash, and linked list neighbors.
        """
        prev_page_id = self.prev_free_page.page_id if self.prev_free_page else None
        next_page_id = self.next_free_page.page_id if self.next_free_page else None
        return (
            f"CachePage(page_id={self.page_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_page_hash={self._page_hash}, "
            f"prev_free_page={prev_page_id}, "
            f"next_free_page={next_page_id})"
        )


class FreeCachePageQueue:
    """Doubly-linked list for O(1) free page queue operations.

    This class manages a doubly-linked list of free CachePage objects,
    enabling O(1) time complexity for all queue operations including
    removal from the middle. Unlike Python's builtin deque, this
    implementation manipulates the prev_free_page and next_free_page
    attributes directly on CachePage objects, avoiding allocation overhead.

    The queue maintains LRU (Least Recently Used) eviction order:
    1. Least recently used pages are at the front (evicted first).
    2. For pages freed simultaneously (same request), pages with more
       hash tokens (chain tails) are ordered earlier.

    Note:
        The LRU order is maintained by the caller reversing pages before
        calling append_n() when freeing a request's pages.

    Attributes:
        num_free_pages: Current count of free pages in the queue.
        fake_free_list_head: Sentinel node marking the start of the list.
        fake_free_list_tail: Sentinel node marking the end of the list.

    Example:
        >>> pages = [CachePage(i) for i in range(10)]
        >>> queue = FreeCachePageQueue(pages)
        >>> page = queue.popleft()  # Get first free page
        >>> queue.append(page)  # Return it to the end
    """

    def __init__(self, pages: list[CachePage]) -> None:
        """Initialize the free page queue with a list of pages.

        Creates the doubly-linked list structure from the given pages,
        initially ordered by their position in the input list.

        Args:
            pages: List of CachePage objects to initialize the queue with.
                All pages will be linked in the order provided.
        """
        self.num_free_pages = len(pages)

        for i in range(self.num_free_pages):
            if i > 0:
                pages[i].prev_free_page = pages[i - 1]
            if i < self.num_free_pages - 1:
                pages[i].next_free_page = pages[i + 1]

        self.fake_free_list_head = CachePage(page_id=-1)
        self.fake_free_list_tail = CachePage(page_id=-1)
        if self.num_free_pages > 0:
            self.fake_free_list_head.next_free_page = pages[0]
            pages[0].prev_free_page = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_page = pages[-1]
            pages[-1].next_free_page = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_page = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_page = self.fake_free_list_head

    def popleft(self) -> CachePage:
        """Pop the first free page and reduce num_free_pages by 1.

        Returns:
            The first free page.
        """
        if (
            self.fake_free_list_head.next_free_page is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_page is None
        ):
            assert self.num_free_pages == 0, f"num_free_pages ({self.num_free_pages}) is out of sync with the free list."
            raise ValueError("No free pages available")

        first_page: CachePage = self.fake_free_list_head.next_free_page

        if first_page.next_free_page is None:
            raise RuntimeError("Invalid page found in popleft() which doesn't have a valid next_free_page")

        self.fake_free_list_head.next_free_page = first_page.next_free_page
        first_page.next_free_page.prev_free_page = self.fake_free_list_head

        first_page.prev_free_page = first_page.next_free_page = None

        self.num_free_pages -= 1
        return first_page

    def popleft_n(self, n: int) -> list[CachePage]:
        """Pop the first n free pages and reduce num_free_pages by n.

        Args:
            n: The number of pages to pop.

        Returns:
            A list of n free pages.
        """
        if n == 0:
            return []
        assert self.num_free_pages >= n
        self.num_free_pages -= n

        curr_page = self.fake_free_list_head.next_free_page

        ret = []
        for _ in range(n):
            assert curr_page is not None
            ret.append(curr_page)
            last_page = curr_page
            curr_page = curr_page.next_free_page

            last_page.prev_free_page = None
            last_page.next_free_page = None

        if curr_page is not None:
            self.fake_free_list_head.next_free_page = curr_page
            curr_page.prev_free_page = self.fake_free_list_head
        return ret

    def remove(self, page: CachePage) -> None:
        """Remove a page in the free list and reduce num_free_pages by 1.

        Args:
            page: The page to remove.
        """
        if page.prev_free_page is None or page.next_free_page is None:
            raise RuntimeError(f"remove() called on an invalid page: {page}")

        page.prev_free_page.next_free_page = page.next_free_page

        page.next_free_page.prev_free_page = page.prev_free_page

        page.prev_free_page = page.next_free_page = None
        self.num_free_pages -= 1

    def append(self, page: CachePage) -> None:
        """Put a page back into the free list and increase
        num_free_pages by 1.

        Args:
            page: The page to append.
        """
        if self.fake_free_list_tail.prev_free_page is None:
            raise RuntimeError("prev_free_page of fake_free_list_tail should always exist")
        last_page: CachePage = self.fake_free_list_tail.prev_free_page

        last_page.next_free_page = page
        page.prev_free_page = last_page

        page.next_free_page = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_page = page

        self.num_free_pages += 1

    def append_n(self, pages: list[CachePage]) -> None:
        """Put a list of pages back into the free list

        Args:
            pages: The pages to append.
        """
        if len(pages) == 0:
            return
        self.num_free_pages += len(pages)

        last_page = self.fake_free_list_tail.prev_free_page
        assert last_page is not None, "prev_free_page of fake_free_list_tail should always exist"

        for page in pages:
            page.prev_free_page = last_page
            last_page.next_free_page = page
            last_page = page

        last_page.next_free_page = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_page = last_page

    def get_all_free_pages(self) -> list[CachePage]:
        """Get all free pages in the free list. Mainly used for testing.

        Returns:
            A list of free pages.
        """
        ret = []
        if self.fake_free_list_head.next_free_page is None:
            raise RuntimeError("next_free_page of fake_free_list_head should always exist")

        curr_page: CachePage = self.fake_free_list_head.next_free_page

        while curr_page.next_free_page is not None:
            ret.append(curr_page)
            curr_page = curr_page.next_free_page
        return ret


NONE_HASH: int
"""Global hash value used as the parent hash for the first page in a sequence."""


def init_none_hash() -> None:
    """Initialize the global NONE_HASH used for prefix caching.

    This function sets up the base hash value used as the "parent" hash for
    the first page in any token sequence. The hash is deterministic when
    PYTHONHASHSEED environment variable is set, enabling reproducible
    prefix cache behavior across runs.

    The NONE_HASH is used in hash_page_tokens when computing the hash for
    the first page of a request (where there is no parent page).

    Note:
        This function must be called once during initialization before
        any prefix caching operations. The CacheManager calls this
        automatically during construction.

    Environment Variables:
        PYTHONHASHSEED: If set, uses this as the seed for deterministic
            hashing. If not set or set to None, generates a random hash
            from 32 bytes of os.urandom().
    """
    global NONE_HASH

    hash_seed = int(os.getenv("PYTHONHASHSEED", "1524618910112"))
    NONE_HASH = int.from_bytes(os.urandom(32), byteorder="big") if hash_seed is None else hash(hash_seed)


def hash_page_tokens(
    hash_function: Callable,
    parent_page_hash: int | None,
    curr_page_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> PageHash:
    """Computes a hash value corresponding to the contents of a page and
    the contents of the preceding page(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same page contents.

    Args:
        parent_page_hash: The hash of the parent page. None
            if this is the first page.
        curr_page_token_ids: A list of token ids in the current
            page. The current page is assumed to be full.
        extra_keys: Extra keys for the page.

    Returns:
        The hash value of the page and the token ids in the page.
        The entire tuple is used as the hash key of the page.
    """
    if not parent_page_hash:
        parent_page_hash = NONE_HASH

    curr_page_token_ids_tuple = tuple(curr_page_token_ids)
    return PageHash(
        hash_function((parent_page_hash, curr_page_token_ids_tuple, extra_keys)),
        curr_page_token_ids_tuple,
        extra_keys,
    )


def hash_request_tokens(hash_function: Any, page_size: int, request: EngineRequest) -> list[PageHash]:
    """Compute hash values for all complete pages in a request's token sequence.

    This function computes a chain of PageHash values for prefix caching,
    where each page's hash depends on its parent page's hash, creating
    a content-addressable chain. Only complete pages (with exactly
    page_size tokens) are hashed.

    For vision-language models, multimodal content is included in the
    hash computation to prevent incorrect cache hits when different
    images produce the same token IDs.

    Args:
        hash_function: The hash function to use (typically Python's builtin hash).
        page_size: Number of tokens per page.
        request: The EngineRequest containing token IDs and optional
            multimodal features.

    Returns:
        List of PageHash objects, one for each complete page in the request.
        Partial pages at the end are not included.

    Example:
        >>> # For a request with 100 tokens and page_size=16
        >>> hashes = hash_request_tokens(hash, 16, request)
        >>> len(hashes)  # 6 complete pages (96 tokens)
        6
    """
    token_ids = request.all_token_ids

    req_extra_keys = None
    # Vision-language models: KV prefix caching must include multimodal content.
    #
    # For many VLMs the prompt `token_ids` can be identical across different
    # images/videos (e.g. `<image>` placeholders expand to a fixed count based
    # on resolution bucket), while the actual KV state depends on the vision
    # inputs. If we don't include a multimodal key here, prefix caching may
    # incorrectly reuse a KV prefix computed for a different image, causing the
    # model to "see" the wrong picture.
    mm_features = getattr(request, "mm_features", None)
    if getattr(request, "has_vision", False) and mm_features:
        try:
            req_extra_keys = tuple(
                (str(getattr(feat, "modality", "mm")), str(getattr(feat, "mm_hash", ""))) for feat in mm_features
            )
        except Exception:
            req_extra_keys = ("multimodal",)

    ret = []
    parent_page_hash_value = None
    for start in range(0, len(token_ids), page_size):
        end = start + page_size
        page_token_ids = token_ids[start:end]

        if len(page_token_ids) < page_size:
            break

        page_hash = hash_page_tokens(hash_function, parent_page_hash_value, page_token_ids, req_extra_keys)
        ret.append(page_hash)
        parent_page_hash_value = page_hash.hash_value
    return ret
