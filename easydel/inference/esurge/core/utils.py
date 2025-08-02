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
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

from ..request import EngineRequest


class PageHash(NamedTuple):
    """Hash value of a page (int), the token IDs in the page, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """

    hash_value: int
    token_ids: tuple[int, ...]
    extra_keys: Any | None = None


class PageHashWithGroupId(NamedTuple):
    page_hash: PageHash
    group_id: int

    def get_hash_value(self) -> int:
        return self.page_hash.hash_value


@dataclass
class CachePage:
    """KV-cache page metadata."""

    page_id: int
    ref_cnt: int = 0
    _page_hash: PageHashWithGroupId | None = None
    prev_free_page: Optional["CachePage"] = None
    next_free_page: Optional["CachePage"] = None
    is_null: bool = False

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    @property
    def page_hash(self) -> PageHashWithGroupId | None:
        return self._page_hash

    @page_hash.setter
    def page_hash(self, page_hash: PageHashWithGroupId):
        assert self.page_hash is None, "The page already has a hash. This should not happen."
        self._page_hash = page_hash

    def reset_hash(self):
        """Reset the page hash when the page is evicted."""
        self._page_hash = None

    def __repr__(self) -> str:
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
    """This class organizes a list of CachePage objects to a doubly linked
    list of free pages. We implement this class instead of using Python
    builtin deque to support removing a page in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_page and next_free_page attributes of the given pages.

    The queue is ordered by page ID in the beginning. When a page is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used page is at the front (LRU).
    2. If two pages have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a page
       chain) is at the front.
    Note that we maintain this order by reversing the page order when free
    pages of a request. This operation is outside of this class.

    Args:
        pages: A list of CachePage objects.
    """

    def __init__(self, pages: list[CachePage]) -> None:
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


def init_none_hash():
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
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
    """Computes hash values of a chain of pages given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        page_size: The size of each page.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids

    req_extra_keys = None

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
