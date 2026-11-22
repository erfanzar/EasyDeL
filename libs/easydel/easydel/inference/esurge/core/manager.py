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

import numpy as np

from ..logger import logger
from ..multimodal.types import compute_content_hash
from .interface import CacheGroupSpec, ChunkedLocalAttentionSpec, FullAttentionSpec, SlidingWindowSpec

_KIND = {FullAttentionSpec: 0, SlidingWindowSpec: 1, ChunkedLocalAttentionSpec: 2}


EngineRequest = Any
"""Any object with the request fields the KV core reads (id, tokens, computed count, sampling params, vision data)."""


def _hash_media_payload(value: Any) -> Any:
    """Compute a deterministic content hash for a raw media array or sequence of arrays."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return compute_content_hash(np.asarray(value))
        except (TypeError, ValueError):
            return tuple(_hash_media_payload(item) for item in value)
    return compute_content_hash(value)


def _fallback_multimodal_nonce(request: EngineRequest) -> tuple[Any, ...]:
    """Return a per-request non-colliding key when multimodal identity extraction fails."""
    return ("multimodal-nonce", str(getattr(request, "request_id", "")), id(request))


def request_page_hash_extra_keys(request: EngineRequest) -> tuple[Any, ...] | None:
    """Return the same multimodal identity for initial and incremental page hashes."""
    if not getattr(request, "has_vision", False):
        return None
    mm_features = getattr(request, "mm_features", None)
    try:
        if mm_features:
            feature_keys: list[tuple[Any, ...]] = []
            for feat in mm_features:
                modality = str(getattr(feat, "modality", "mm"))
                mm_hash = str(getattr(feat, "mm_hash", ""))
                if not mm_hash:
                    pix = getattr(feat, "pixel_values", None)
                    grid = getattr(feat, "grid_thw", None)
                    if pix is not None or grid is not None:
                        mm_hash = repr((_hash_media_payload(pix), _hash_media_payload(grid)))
                    else:
                        return _fallback_multimodal_nonce(request)
                feature_keys.append((modality, mm_hash))
            return tuple(feature_keys)

        raw_fields = (
            ("pixel_values", getattr(request, "pixel_values", None)),
            ("image_grid_thw", getattr(request, "image_grid_thw", None)),
            ("pixel_values_videos", getattr(request, "pixel_values_videos", None)),
            ("video_grid_thw", getattr(request, "video_grid_thw", None)),
        )
        raw_keys = tuple((name, _hash_media_payload(val)) for name, val in raw_fields if val is not None)
        if raw_keys:
            return raw_keys
        return _fallback_multimodal_nonce(request)
    except Exception:
        return _fallback_multimodal_nonce(request)


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
        # Detect the 2-type (Full, Sliding) HitSearch::Hybrid layout when use_eagle=True.
        # Older compiled surgecore wheels double-subtract the EAGLE reserve page on this layout;
        # we perform the single reserve step explicitly in Python while keeping the Rust fix in sync.
        full_indices = [i for i, g in enumerate(groups) if g[0] == 0]
        other_indices = [i for i, g in enumerate(groups) if g[0] != 0]
        self._hybrid_sliding_eagle = (
            bool(use_eagle)
            and self.enable_caching
            and len(groups) > 1
            and bool(full_indices)
            and bool(other_indices)
            and len({groups[i][3] for i in full_indices}) == 1
            and len({groups[i][3] for i in other_indices}) == 1
            and (max(full_indices) < min(other_indices) or max(other_indices) < min(full_indices))
            and groups[other_indices[0]][0] == 1
        )
        self._core = KvCore(
            num_pages=num_pages,
            groups=groups,
            max_model_len=int(max_model_len),
            caching=self.enable_caching,
            eagle=False if self._hybrid_sliding_eagle else bool(use_eagle),
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

    def _expected_num_keys(self, request: EngineRequest) -> int:
        if not self.page_size:
            return 0
        return len(request.all_token_ids) // self.page_size

    def _tokens_if_unkeyed(self, slot: int | None, request: EngineRequest) -> list[int] | None:
        if slot is None:
            return request.all_token_ids[:]
        known = self._core.num_keys(slot)
        if known == 0 or known < self._expected_num_keys(request):
            return request.all_token_ids[:]
        return None

    def _prepare_slot_and_tokens(self, request: EngineRequest) -> tuple[int, list[int] | None]:
        """Ensure a slot's cached keys are refreshed when tokens grew past previously hashed pages."""
        slot = self._slot(request)
        tokens = self._tokens_if_unkeyed(slot, request)
        if tokens is not None and self._core.num_keys(slot) > 0:
            # Older compiled surgecore wheels skip ensure_keys when req.keys is non-empty.
            # Forgetting stale keys preserves held pages/cached counts and forces full re-chaining.
            if self._core.forget_keys(slot):
                self._slots.pop(request.request_id, None)
                slot = self._slot(request)
        return slot, tokens

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
        slot, tokens_arg = self._prepare_slot_and_tokens(request)
        max_len = request.num_tokens - 1
        if max_cache_hit_tokens is not None:
            max_len = min(max_len, int(max_cache_hit_tokens))
        hint = None if dp_shard_hint is None else int(dp_shard_hint)
        dp = None if data_parallel_size is None else int(data_parallel_size)
        no_hit = self._no_hit(request)
        pages, tokens = self._core.lookup(slot, tokens_arg, no_hit, max_len, hint, dp)
        if self._hybrid_sliding_eagle and not no_hit and tokens > 0:
            ps = int(self.page_size or 0)
            reserved_max = tokens - ps
            if reserved_max <= 0:
                return self.create_empty_page_list(), 0
            pages, tokens = self._core.lookup(slot, None, False, reserved_max, hint, dp)
        return CachePages(tuple(pages)), tokens

    def request_page_hashes(self, request: EngineRequest) -> list[int]:
        """The chained full-page keys (128-bit ints) of the request's tokens, computed on first use."""
        if not self.enable_caching:
            raise AssertionError("page keys need prefix caching")
        slot, tokens_arg = self._prepare_slot_and_tokens(request)
        return self._core.keys(slot, tokens_arg)

    def probe_num_computed_tokens(self, request: EngineRequest) -> int:
        """Read-only prefix probe: the cached-token count, keeping no per-request state."""
        if not self.enable_caching or self._no_hit(request):
            return 0
        slot = self._slots.get(request.request_id)
        tokens = self._tokens_if_unkeyed(slot, request)
        # If the existing slot has stale keys (tokens grew past num_keys), probe with slot=None
        # so read-only probing hashes fresh keys into scratch without mutating the slot.
        probe_slot = None if (slot is not None and tokens is not None and self._core.num_keys(slot) > 0) else slot
        extra = page_hash_extra_key(request)
        max_len = request.num_tokens - 1
        hit_tokens = int(self._core.probe(probe_slot, tokens, extra, max_len))
        if self._hybrid_sliding_eagle and hit_tokens > 0:
            ps = int(self.page_size or 0)
            reserved_max = hit_tokens - ps
            if reserved_max <= 0:
                return 0
            hit_tokens = int(self._core.probe(probe_slot, tokens, extra, reserved_max))
        return hit_tokens

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
        # Clamp to allocated page capacity so older compiled surgecore wheels never overshoot req.cached[g].
        page_ids = self._core.page_ids(slot)
        if page_ids and self.kv_cache_groups:
            max_publishable = min(
                len(group_pages) * int(group.kv_cache_spec.page_size)
                for group_pages, group in zip(page_ids, self.kv_cache_groups, strict=True)
            )
            n = min(n, max_publishable)
        needed = self._core.publish(slot, n)
        if needed is not None:
            lo, hi = needed
            self._core.publish(slot, n, request.all_token_ids[lo:hi], lo)

    def create_empty_page_list(self) -> CachePages:
        """Empty page lists, one per KV cache group."""
        return CachePages(tuple([] for _ in range(self.num_kv_cache_groups)))
