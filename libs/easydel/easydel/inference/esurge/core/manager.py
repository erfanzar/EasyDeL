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

"""KV-cache allocation and prefix caching for the eSurge scheduler.

:class:`CacheManager` binds surgecore's Rust KV core (``surgecore._native.KvCore``): the page pool with its LRU free
list and null page 0, the chained page keys and prefix map, the full / sliding-window / chunked-local group rules, the
transactional allocation with rollback and publication after compute. This module only maps request ids to the core's
request slots and reads the few request fields the core needs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..logger import logger
from .interface import CacheGroupSpec, ChunkedLocalAttentionSpec, FullAttentionSpec, SlidingWindowSpec

_KIND = {FullAttentionSpec: 0, SlidingWindowSpec: 1, ChunkedLocalAttentionSpec: 2}


EngineRequest = Any
"""Any object with the request fields the KV core reads (id, tokens, computed count, sampling params, vision data)."""


def request_page_hash_extra_keys(request: EngineRequest) -> tuple[Any, ...] | None:
    """Return the same multimodal identity for initial and incremental page hashes."""
    mm_features = getattr(request, "mm_features", None)
    if getattr(request, "has_vision", False) and mm_features:
        try:
            return tuple(
                (str(getattr(feat, "modality", "mm")), str(getattr(feat, "mm_hash", ""))) for feat in mm_features
            )
        except Exception:
            return ("multimodal",)
    return None


def page_hash_extra_key(request: EngineRequest) -> int:
    """The request's multimodal identity as a 128-bit int (0 for text), mixed into every page key."""
    keys = request_page_hash_extra_keys(request)
    if keys is None:
        return 0
    return int.from_bytes(hashlib.blake2b(repr(keys).encode(), digest_size=16).digest(), "little") or 1


@dataclass
class CachePages:
    """Page ids per KV cache group (null page = 0), as returned by allocation and prefix lookups."""

    pages: tuple[list[int], ...]

    def get_page_ids(self) -> tuple[list[int], ...]:
        """The page ids, one list per KV cache group."""
        return tuple(list(group) for group in self.pages)

    def new_empty(self) -> CachePages:
        """Empty page lists with the same group count."""
        return CachePages(tuple([] for _ in self.pages))


class _PagePoolView:
    """Read-only pool counters (``page_pool.get_num_free_pages()`` and friends)."""

    __slots__ = ("_core",)

    def __init__(self, core: Any) -> None:
        self._core = core

    @property
    def num_pages(self) -> int:
        return self._core.num_pages

    def get_num_free_pages(self) -> int:
        return self._core.num_free

    def get_usage(self) -> float:
        return self._core.usage


class CacheManager:
    """KV page allocation and prefix caching over one page pool (the Rust KV core).

    Args:
        num_pages: Pages in the pool, null page 0 included.
        kv_cache_groups: Cache group specs; no groups turns prefix caching off.
        max_model_len: Token slots are capped at this length.
        enable_caching: Prefix caching on.
        use_eagle: Reserve one prefix-hit page for EAGLE hidden-state recomputation.
    """

    def __init__(
        self,
        num_pages: int,
        kv_cache_groups: list[CacheGroupSpec],
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
    ) -> None:
        from surgecore import KvCore

        self.num_pages = num_pages
        self.kv_cache_groups = kv_cache_groups
        self.max_model_len = max_model_len
        self.enable_caching = bool(enable_caching) and len(kv_cache_groups) > 0
        self.use_eagle = use_eagle
        self.page_size: int | None = kv_cache_groups[0].kv_cache_spec.page_size if self.enable_caching else None
        self.num_kv_cache_groups = len(kv_cache_groups)
        types: dict[str, int] = {}
        groups = []
        for group in kv_cache_groups:
            spec = group.kv_cache_spec
            kind = _KIND[type(spec)]
            window = (0, getattr(spec, "sliding_window", 0), getattr(spec, "attention_chunk_size", 0))[kind]
            groups.append((kind, int(spec.page_size), int(window), types.setdefault(spec.type_id, len(types))))
        if not isinstance(num_pages, int) or num_pages <= 0:
            raise ValueError("num_pages must be a positive integer")
        self._core = KvCore(
            num_pages=num_pages,
            groups=groups,
            max_model_len=int(max_model_len),
            caching=self.enable_caching,
            eagle=bool(use_eagle),
        )
        self._slots: dict[str, int] = {}
        self.page_pool = _PagePoolView(self._core)

    @property
    def usage(self) -> float:
        """KV cache usage in [0, 1]."""
        return self._core.usage

    @property
    def num_cached_pages(self) -> int:
        """Distinct published (page key, cache group) entries."""
        return self._core.num_cached

    @property
    def tracked_request_ids(self) -> frozenset[str]:
        """Ids of the requests the core holds pages or page keys for."""
        return frozenset(self._slots)

    def _slot(self, request: EngineRequest) -> int:
        slot = self._slots.get(request.request_id)
        if slot is None:
            slot = self._slots[request.request_id] = self._core.open(page_hash_extra_key(request))
        return slot

    def _tokens_if_unkeyed(self, slot: int | None, request: EngineRequest) -> list[int] | None:
        return request.all_token_ids[:] if slot is None or self._core.num_keys(slot) == 0 else None

    @staticmethod
    def _no_hit(request: EngineRequest) -> bool:
        return request.sampling_params is not None and request.sampling_params.prompt_logprobs is not None

    def get_computed_pages(
        self,
        request: EngineRequest,
        *,
        dp_shard_hint: int | None = None,
        data_parallel_size: int | None = None,
        max_cache_hit_tokens: int | None = None,
    ) -> tuple[CachePages, int]:
        """The longest prefix hit (none when prompt logprobs are requested); records the request's page keys."""
        if not self.enable_caching:
            return self.create_empty_page_list(), 0
        slot = self._slot(request)
        max_len = request.num_tokens - 1
        if max_cache_hit_tokens is not None:
            max_len = min(max_len, int(max_cache_hit_tokens))
        hint = None if dp_shard_hint is None else int(dp_shard_hint)
        dp = None if data_parallel_size is None else int(data_parallel_size)
        pages, tokens = self._core.lookup(
            slot, self._tokens_if_unkeyed(slot, request), self._no_hit(request), max_len, hint, dp
        )
        return CachePages(tuple(pages)), tokens

    def request_page_hashes(self, request: EngineRequest) -> list[int]:
        """The chained full-page keys (128-bit ints) of the request's tokens, computed on first use."""
        if not self.enable_caching:
            raise AssertionError("page keys need prefix caching")
        slot = self._slot(request)
        return self._core.keys(slot, self._tokens_if_unkeyed(slot, request))

    def probe_num_computed_tokens(self, request: EngineRequest) -> int:
        """Read-only prefix probe: the cached-token count, keeping no per-request state."""
        if not self.enable_caching or self._no_hit(request):
            return 0
        slot = self._slots.get(request.request_id)
        tokens = self._tokens_if_unkeyed(slot, request)
        return int(self._core.probe(slot, tokens, page_hash_extra_key(request), request.num_tokens - 1))

    def allocate_slots(
        self,
        request: EngineRequest,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_pages: CachePages | None = None,
        num_lookahead_tokens: int = 0,
        dp_shard_hint: int | None = None,
        data_parallel_size: int | None = None,
    ) -> CachePages | None:
        """Transactionally attach the prefix hit and allocate slots for ``num_new_tokens`` more tokens.

        Returns the newly allocated pages per group, or ``None`` (nothing changed) when the pool cannot hold them.
        New pages are published as prefix hits only by :meth:`cache_pages`.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")
        hits = None if new_computed_pages is None else list(new_computed_pages.pages)
        pages = self._core.allocate(
            self._slot(request),
            int(request.num_computed_tokens),
            int(num_new_tokens),
            int(num_new_computed_tokens),
            hits,
            int(num_lookahead_tokens),
            None if dp_shard_hint is None else int(dp_shard_hint),
            None if data_parallel_size is None else int(data_parallel_size),
        )
        return None if pages is None else CachePages(tuple(pages))

    def get_num_pages_needed(
        self,
        request: EngineRequest,
        num_tokens: int,
        new_computed_pages: CachePages | None = None,
        *,
        max_chunk_tokens: int | None = None,
    ) -> int:
        """Free pages the request still draws before it holds ``num_tokens`` slots (read-only).

        Held and prefix-hit pages count as present, a hit page with no reference costs one free page, and windowed
        groups count at most the pages they hold at once when the rest arrives in chunks of ``max_chunk_tokens``.
        """
        hits = None if new_computed_pages is None else list(new_computed_pages.pages)
        return self._core.needed(
            self._slots.get(request.request_id),
            min(int(num_tokens), self.max_model_len),
            hits,
            None if max_chunk_tokens is None else int(max_chunk_tokens),
        )

    def free(self, request: EngineRequest) -> None:
        """Free every page of the request, tail first (preemption and finish)."""
        slot = self._slots.get(request.request_id)
        if slot is not None and self._core.free(slot):
            del self._slots[request.request_id]

    def free_page_hashes(self, request: EngineRequest) -> None:
        """Forget the request's page keys (finish only; preemption keeps them)."""
        slot = self._slots.get(request.request_id)
        if slot is not None and self._core.forget_keys(slot):
            del self._slots[request.request_id]

    def release_tail(self, request: EngineRequest, keep_tokens: int) -> int:
        """Release the request's unpublished pages past those ``keep_tokens`` slots need; returns how many.

        The caller guarantees that no dispatched, uncommitted step of the request still reserves them.
        """
        slot = self._slots.get(request.request_id)
        return 0 if slot is None else self._core.release_tail(slot, int(keep_tokens))

    def reset_prefix_cache(self) -> bool:
        """Forget every published page; ``False`` while any non-null page is in use."""
        if not self._core.reset():
            logger.warning("Failed to reset prefix cache because some pages are not freed yet")
            return False
        logger.info("Successfully reset prefix cache")
        return True

    def get_num_common_prefix_pages(self, request: EngineRequest, num_scheduled_requests: int) -> list[int]:
        """Per group, the leading pages of ``request`` that at least ``num_scheduled_requests`` requests hold."""
        return list(self._core.common_prefix(self._slots.get(request.request_id), int(num_scheduled_requests)))

    def get_page_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Page ids per group of a request (null page = 0)."""
        slot = self._slots.get(request_id)
        if slot is None:
            return tuple([] for _ in range(self.num_kv_cache_groups))
        return tuple(self._core.page_ids(slot))

    def cache_pages(self, request: EngineRequest, num_computed_tokens: int) -> None:
        """Publish the request's pages made full by its first ``num_computed_tokens`` (after their KV is computed)."""
        if not self.enable_caching:
            return
        slot = self._slots.get(request.request_id)
        if slot is None:
            raise KeyError(request.request_id)
        n = int(num_computed_tokens)
        needed = self._core.publish(slot, n)
        if needed is not None:
            lo, hi = needed
            self._core.publish(slot, n, request.all_token_ids[lo:hi], lo)

    def create_empty_page_list(self) -> CachePages:
        """Empty page lists, one per KV cache group."""
        return CachePages(tuple([] for _ in range(self.num_kv_cache_groups)))
