from __future__ import annotations

from types import SimpleNamespace

from easydel.inference.esurge.core.manager import CacheManager, CachePages


class _DummyPagePool:
    def __init__(self):
        self.touch_calls = []
        self.free_calls = []

    def get_num_free_pages(self):
        return 8

    def touch(self, pages):
        self.touch_calls.append(tuple(tuple(group) for group in pages))

    def free_pages(self, pages):
        self.free_calls.append(list(pages))


class _DummyCoordinator:
    def __init__(self):
        self.saved = []
        self.rolled_back = []

    def remove_skipped_pages(self, request_id, num_computed_tokens):
        del request_id, num_computed_tokens

    def get_num_pages_to_allocate(self, request_id, num_tokens, new_computed_pages):
        del request_id, num_tokens, new_computed_pages
        return 1

    def save_new_computed_pages(self, request_id, new_computed_pages):
        self.saved.append((request_id, new_computed_pages))

    def rollback_new_computed_pages(self, request_id, new_computed_pages):
        self.rolled_back.append((request_id, new_computed_pages))

    def allocate_new_pages(self, request_id, num_tokens, *, dp_shard_hint=None, data_parallel_size=None):
        del request_id, num_tokens, dp_shard_hint, data_parallel_size
        raise ValueError("Insufficient free pages in requested DP shard range")


def test_allocate_slots_rolls_back_saved_prefix_pages_on_shard_allocation_error():
    manager = object.__new__(CacheManager)
    manager.enable_caching = True
    manager.max_model_len = 64
    manager.kv_cache_groups = [object()]
    manager.page_pool = _DummyPagePool()
    manager.coordinator = _DummyCoordinator()
    manager.req_to_page_hashes = {"req-1": []}

    request = SimpleNamespace(request_id="req-1", num_computed_tokens=0, num_tokens=12)
    cached_prefix = CachePages((["page-a", "page-b"],))

    result = manager.allocate_slots(
        request,
        num_new_tokens=4,
        new_computed_pages=cached_prefix,
    )

    assert result is None
    assert manager.coordinator.saved == [("req-1", (["page-a", "page-b"],))]
    assert manager.coordinator.rolled_back == [("req-1", (["page-a", "page-b"],))]
    assert manager.page_pool.touch_calls == [(("page-a", "page-b"),)]
    assert manager.page_pool.free_calls == [["page-a", "page-b"]]
