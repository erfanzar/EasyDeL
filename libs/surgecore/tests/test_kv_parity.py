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

"""Parity of surgecore's KV core with eSurge's former Python KV core.

``data/kv_goldens.npz`` holds randomized operation sequences (lookup, probe, page-key reads, page counts, allocation
with and without prefix hits, failed and rolled-back allocations, publication, preemption, finish and reset) captured
from the former Python ``CacheManager`` and ``RecurrentSnapshotIndex``; its header names the source module and commit.
Every recorded result is compared exactly: page ids per group, hit lengths, errors, free and cached-page counts after
every operation, the free list's eviction order and every page's reference count. The core's own invariants
(``KvCore.check``) are asserted after every operation as well.

:func:`replay_case` drives any object with ``CacheManager``'s methods; this file drives ``KvCore`` through the small
:class:`KvDriver` below, the binding protocol eSurge's ``CacheManager`` follows. The goldens are not replayed through
``CacheManager`` itself.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
from surgecore import KvCore, SnapshotIndex

pytestmark = pytest.mark.cpu_ok

FIXTURE = pathlib.Path(__file__).parent / "data" / "kv_goldens.npz"


def load_goldens() -> tuple[dict, dict, np.ndarray]:
    """``(header, body, tokens)`` of the fixture."""
    with np.load(FIXTURE) as npz:
        header = json.loads(str(npz["header"]))
        body = json.loads(npz["cases"].tobytes())
        tokens = npz["tokens"].copy()
    return header, body, tokens


_HEADER, _BODY, _TOKENS = load_goldens()


class FakeRequest:
    """The request fields the KV core reads."""

    def __init__(self, rid: str, tokens: list[int], extra, plog: bool, running_status=None):
        self.request_id = rid
        self.all_token_ids = tokens
        self.num_computed_tokens = 0
        self.sampling_params = SimpleNamespace(prompt_logprobs=0 if plog else None)
        self.mm_features = [SimpleNamespace(modality=m, mm_hash=h) for m, h in (extra or [])]
        self.has_vision = bool(extra)
        self.status = running_status

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)


def _extra(request) -> int:
    if not (request.has_vision and request.mm_features):
        return 0
    keys = tuple((str(f.modality), str(f.mm_hash)) for f in request.mm_features)
    return int.from_bytes(hashlib.blake2b(repr(keys).encode(), digest_size=16).digest(), "little") or 1


class KvDriver:
    """``KvCore`` behind ``CacheManager``'s method names (request ids map to core slots)."""

    def __init__(self, groups, num_pages, max_model_len, caching, eagle):
        self.max_model_len = max_model_len
        self.caching = bool(caching) and bool(groups)
        self.ng = len(groups)
        self._core = KvCore(
            num_pages=num_pages,
            groups=[tuple(g) for g in groups],
            max_model_len=max_model_len,
            caching=self.caching,
            eagle=eagle,
        )
        self.slots: dict[str, int] = {}
        self.page_pool = SimpleNamespace(get_num_free_pages=lambda: self._core.num_free)

    @property
    def num_cached_pages(self) -> int:
        return self._core.num_cached

    def _slot(self, r) -> int:
        if r.request_id not in self.slots:
            self.slots[r.request_id] = self._core.open(_extra(r))
        return self.slots[r.request_id]

    def _toks(self, slot, r):
        return list(r.all_token_ids) if slot is None or self._core.num_keys(slot) == 0 else None

    def get_computed_pages(self, r, *, dp_shard_hint=None, data_parallel_size=None, max_cache_hit_tokens=None):
        if not self.caching:
            return SimpleNamespace(pages=tuple([] for _ in range(self.ng))), 0
        slot = self._slot(r)
        mx = r.num_tokens - 1 if max_cache_hit_tokens is None else min(r.num_tokens - 1, max_cache_hit_tokens)
        plog = r.sampling_params.prompt_logprobs is not None
        pages, n = self._core.lookup(slot, self._toks(slot, r), plog, mx, dp_shard_hint, data_parallel_size)
        return SimpleNamespace(pages=tuple(pages)), n

    def request_page_hashes(self, r):
        slot = self._slot(r)
        return self._core.keys(slot, self._toks(slot, r))

    def probe_num_computed_tokens(self, r):
        if not self.caching or r.sampling_params.prompt_logprobs is not None:
            return 0
        slot = self.slots.get(r.request_id)
        return self._core.probe(slot, self._toks(slot, r), _extra(r), r.num_tokens - 1)

    def allocate_slots(self, r, n, nc=0, hit=None, la=0, h=None, d=None):
        hits = None if hit is None else list(hit.pages)
        pages = self._core.allocate(self._slot(r), r.num_computed_tokens, n, nc, hits, la, h, d)
        return None if pages is None else SimpleNamespace(pages=tuple(pages))

    def get_num_pages_needed(self, r, n, hit=None, *, max_chunk_tokens=None):
        hits = None if hit is None else list(hit.pages)
        return self._core.needed(self.slots.get(r.request_id), min(n, self.max_model_len), hits, max_chunk_tokens)

    def free(self, r):
        slot = self.slots.get(r.request_id)
        if slot is not None and self._core.free(slot):
            del self.slots[r.request_id]

    def free_page_hashes(self, r):
        slot = self.slots.get(r.request_id)
        if slot is not None and self._core.forget_keys(slot):
            del self.slots[r.request_id]

    def cache_pages(self, r, n):
        if not self.caching:
            return
        slot = self.slots[r.request_id]
        needed = self._core.publish(slot, n)
        if needed is not None:
            lo, hi = needed
            self._core.publish(slot, n, list(r.all_token_ids[lo:hi]), lo)

    def get_num_common_prefix_pages(self, r, k):
        return self._core.common_prefix(self.slots.get(r.request_id), k)

    def get_page_ids(self, rid):
        slot = self.slots.get(rid)
        return tuple([] for _ in range(self.ng)) if slot is None else tuple(self._core.page_ids(slot))

    def reset_prefix_cache(self):
        return self._core.reset()


def _ids(pages) -> list[list[int]]:
    return [[int(p) for p in g] for g in pages.pages]


def _crc(core) -> int:
    return zlib.crc32(np.asarray([*core.free_order(), -1, *core.refs()], np.int32).tobytes())


def replay_case(case: dict, tokens: np.ndarray, make_manager, running_status=None) -> None:
    """Replay one golden case through ``make_manager(groups, num_pages, max_model_len, caching, eagle)``.

    The manager must expose ``CacheManager``'s methods plus ``_core`` (the ``KvCore``). Raises ``AssertionError`` at
    the first operation whose result or state differs from the golden.
    """
    name = case["name"]
    try:
        m = make_manager(case["groups"], case["num_pages"], case["max_model_len"], case["caching"], case["eagle"])
    except Exception as exc:
        assert case["init_error"] == type(exc).__name__, (name, exc)
        return
    assert case["init_error"] is None, name
    reqs = [
        FakeRequest(f"r{i}", tokens[q["off"] : q["off"] + q["len"]].tolist(), q["extra"], q["plog"], running_status)
        for i, q in enumerate(case["requests"])
    ]
    hits: dict[int, object] = {}
    key_classes: dict[int, int] = {}

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:
            return None, type(exc).__name__

    for step, op in enumerate(case["ops"]):
        kind, i, a = op["o"], op["r"], op["a"]
        r = reqs[i] if i is not None else None
        where = (name, step, kind, i, a)
        if kind == "append":
            r.all_token_ids.append(a["t"])
            continue
        if kind == "probe":
            res, err = call(m.probe_num_computed_tokens, r)
            got = err or int(res)
        elif kind == "keys":
            res, err = call(m.request_page_hashes, r)
            got = err or [key_classes.setdefault(int(k), len(key_classes)) for k in res]
        elif kind == "lookup":
            res, err = call(
                m.get_computed_pages, r, dp_shard_hint=a["h"], data_parallel_size=a["d"], max_cache_hit_tokens=a["mx"]
            )
            hits[i] = None if err else res[0]
            got = err or [_ids(res[0]), int(res[1])]
        elif kind == "needed":
            hit = hits.get(i) if a["hit"] else None
            res, err = call(m.get_num_pages_needed, r, a["n"], hit, max_chunk_tokens=a["mc"])
            got = err or int(res)
        elif kind == "alloc":
            r.num_computed_tokens = a["c"]
            hit = hits.get(i) if a.get("hit") else None
            res, err = call(m.allocate_slots, r, a["n"], a["nc"], hit, a["la"], a["h"], a["d"])
            got = err or (None if res is None else _ids(res))
            if err is None and res is not None:
                r.num_computed_tokens = a["c"] + a["nc"] + a["n"]
        elif kind == "publish":
            _, got = call(m.cache_pages, r, a["n"])
        elif kind == "common":
            res, err = call(m.get_num_common_prefix_pages, r, a["k"])
            got = err or [int(x) for x in res]
        elif kind == "preempt":
            _, got = call(m.free, r)
            r.num_computed_tokens = 0
        elif kind == "finish":
            _, err = call(m.free, r)
            _, err2 = call(m.free_page_hashes, r)
            r.num_computed_tokens = 0
            got = err or err2
        elif kind == "reset":
            res, err = call(m.reset_prefix_cache)
            got = err or bool(res)
        else:
            raise AssertionError(f"unknown golden op {kind}")
        assert got == op["res"], (*where, got, op["res"])
        state = [int(m.page_pool.get_num_free_pages()), int(m.num_cached_pages)]
        assert state == op["st"], (*where, state, op["st"])
        if "ids" in op:
            assert [list(g) for g in m.get_page_ids(r.request_id)] == op["ids"], where
        if "q" in op:
            assert _crc(m._core) == op["q"], where
        m._core.check()
    assert [m._core.free_order(), m._core.refs()] == case["final"], name


_LEGACY_HYBRID_SLIDING_EAGLE_CASES: dict[int, tuple[str, int]] = {
    # Captured from the former Python CacheManager (commit f7856e6b18f0), whose HybridCacheCoordinator
    # double-subtracted the EAGLE reserve page on 2-type (Full, Sliding) layouts (once in Full, once in Sliding).
    # surgecore now reserves a single page; these cases verify exact bitwise parity up to the first EAGLE hit,
    # the exact +1 page single-reserve delta at that hit, and full KV invariants across the entire op stream.
    13: ("13_full_sw8_p1_c_e", 18),
    14: ("14_sw8_full_p16_c_e", 15),
    21: ("21_full_full_sw64_p1_c_e_mm", 36),
}


def replay_corrected_hybrid_sliding_eagle_case(
    case: dict,
    tokens: np.ndarray,
    make_manager,
    first_hit_step: int,
    running_status=None,
) -> None:
    """Replay a legacy (Full, Sliding) + eagle golden case with single-page EAGLE invariant checks."""
    name = case["name"]
    assert case["init_error"] is None and case["caching"] and case["eagle"], name
    kinds = {g[0] for g in case["groups"]}
    assert kinds == {0, 1}, (name, kinds)
    ps = int(case["groups"][0][1])
    m = make_manager(case["groups"], case["num_pages"], case["max_model_len"], case["caching"], case["eagle"])
    reqs = [
        FakeRequest(f"r{i}", tokens[q["off"] : q["off"] + q["len"]].tolist(), q["extra"], q["plog"], running_status)
        for i, q in enumerate(case["requests"])
    ]
    hits: dict[int, object] = {}
    key_classes: dict[int, int] = {}

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:
            return None, type(exc).__name__

    for step, op in enumerate(case["ops"]):
        kind, i, a = op["o"], op["r"], op["a"]
        r = reqs[i] if i is not None else None
        where = (name, step, kind, i, a)
        if kind == "append":
            r.all_token_ids.append(a["t"])
            continue
        before_probe = None
        if kind == "probe":
            before_probe = (
                m.page_pool.get_num_free_pages(),
                m.num_cached_pages,
                m._core.free_order(),
                m._core.refs(),
            )
            res, err = call(m.probe_num_computed_tokens, r)
            got = err or int(res)
        elif kind == "keys":
            res, err = call(m.request_page_hashes, r)
            got = err or [key_classes.setdefault(int(k), len(key_classes)) for k in res]
        elif kind == "lookup":
            res, err = call(
                m.get_computed_pages, r, dp_shard_hint=a["h"], data_parallel_size=a["d"], max_cache_hit_tokens=a["mx"]
            )
            hits[i] = None if err else res[0]
            got = err or [_ids(res[0]), int(res[1])]
        elif kind == "needed":
            hit = hits.get(i) if a["hit"] else None
            res, err = call(m.get_num_pages_needed, r, a["n"], hit, max_chunk_tokens=a["mc"])
            got = err or int(res)
        elif kind == "alloc":
            r.num_computed_tokens = a["c"]
            hit = hits.get(i) if a.get("hit") else None
            res, err = call(m.allocate_slots, r, a["n"], a["nc"], hit, a["la"], a["h"], a["d"])
            got = err or (None if res is None else _ids(res))
            if err is None and res is not None:
                r.num_computed_tokens = a["c"] + a["nc"] + a["n"]
        elif kind == "publish":
            _, got = call(m.cache_pages, r, a["n"])
        elif kind == "common":
            res, err = call(m.get_num_common_prefix_pages, r, a["k"])
            got = err or [int(x) for x in res]
        elif kind == "preempt":
            _, got = call(m.free, r)
            r.num_computed_tokens = 0
        elif kind == "finish":
            _, err = call(m.free, r)
            _, err2 = call(m.free_page_hashes, r)
            r.num_computed_tokens = 0
            got = err or err2
        elif kind == "reset":
            res, err = call(m.reset_prefix_cache)
            got = err or bool(res)
        else:
            raise AssertionError(f"unknown golden op {kind}")

        if step < first_hit_step:
            assert got == op["res"], (*where, got, op["res"])
            state = [int(m.page_pool.get_num_free_pages()), int(m.num_cached_pages)]
            assert state == op["st"], (*where, state, op["st"])
            if "ids" in op:
                assert [list(g) for g in m.get_page_ids(r.request_id)] == op["ids"], where
            if "q" in op:
                assert _crc(m._core) == op["q"], where
        elif step == first_hit_step:
            if kind == "lookup":
                assert not isinstance(got, str), (*where, got)
                assert got[1] == op["res"][1] + ps, (*where, got[1], op["res"][1], ps)
                assert all(len(g) == got[1] // ps for g in got[0]), (*where, got)
            else:
                assert kind == "probe" and isinstance(got, int), (*where, got)
                assert got == op["res"] + ps, (*where, got, op["res"], ps)
        else:
            if kind == "probe" and isinstance(got, int):
                after_probe = (
                    m.page_pool.get_num_free_pages(),
                    m.num_cached_pages,
                    m._core.free_order(),
                    m._core.refs(),
                )
                assert after_probe == before_probe, where
                assert got >= 0 and got % ps == 0, (*where, got)
                if got > 0:
                    assert got <= (r.num_tokens - 1) - ps, (*where, got, r.num_tokens, ps)
            elif kind == "lookup" and not isinstance(got, str):
                hit_pages, hit_tokens = got
                assert hit_tokens >= 0 and hit_tokens % ps == 0, (*where, got)
                assert all(len(g) == hit_tokens // ps for g in hit_pages), (*where, got)
                if hit_tokens > 0:
                    max_len = r.num_tokens - 1 if a["mx"] is None else min(r.num_tokens - 1, a["mx"])
                    assert hit_tokens <= max_len - ps, (*where, hit_tokens, max_len, ps)

        assert m.page_pool.get_num_free_pages() + sum(1 for x in m._core.refs() if x) == case["num_pages"] - 1, where
        m._core.check()


def test_fixture_header_names_source_and_commit():
    assert _HEADER == {"commit": "f7856e6b18f0", "source": "easydel.inference.esurge.core.manager"}
    assert len(_BODY["cases"]) == 45 and len(_BODY["snapshots"]) == 24


@pytest.mark.parametrize("index", range(45))
def test_kv_core_matches_python_goldens(index):
    case = _BODY["cases"][index]
    if index in _LEGACY_HYBRID_SLIDING_EAGLE_CASES:
        expected_name, first_hit_step = _LEGACY_HYBRID_SLIDING_EAGLE_CASES[index]
        assert case["name"] == expected_name
        replay_corrected_hybrid_sliding_eagle_case(case, _TOKENS, KvDriver, first_hit_step)
        return
    replay_case(case, _TOKENS, KvDriver)


def test_hybrid_full_sliding_eagle_reserves_only_one_page():
    for groups, full_idx, sw_idx in [
        ([(0, 4, 0, 0), (1, 4, 4, 1)], 0, 1),
        ([(1, 4, 4, 0), (0, 4, 0, 1)], 1, 0),
    ]:
        core = KvCore(num_pages=32, groups=groups, max_model_len=64, caching=True, eagle=True)
        toks = list(range(1, 10))
        donor = core.open(0)
        hits, n_hit = core.lookup(donor, toks, False, len(toks) - 1)
        assert n_hit == 0 and hits == [[], []]
        assert core.allocate(donor, 0, len(toks), 0, hits, 0) is not None
        assert core.publish(donor, 8) is None
        core.free(donor)

        # 2 cached pages (8 tokens) with EAGLE reserving 1 page must hit 4 tokens (1 page), never 0.
        assert core.probe(None, toks, 0, len(toks) - 1) == 4
        slot = core.open(0)
        pages, hit_tokens = core.lookup(slot, toks, False, len(toks) - 1)
        assert hit_tokens == 4
        assert len(pages[full_idx]) == 1 and pages[full_idx][0] != 0
        assert len(pages[sw_idx]) == 1 and pages[sw_idx][0] != 0
        core.check()

    # Window=8, page_size=4, 4 cached pages (16 tokens): EAGLE reserves 1 page -> 12 tokens with null prefix on page 0.
    core = KvCore(num_pages=32, groups=[(0, 4, 0, 0), (1, 4, 8, 1)], max_model_len=64, caching=True, eagle=True)
    toks17 = list(range(100, 117))
    donor = core.open(0)
    hits, _ = core.lookup(donor, toks17, False, len(toks17) - 1)
    assert core.allocate(donor, 0, len(toks17), 0, hits, 0) is not None
    assert core.publish(donor, 16) is None
    core.free(donor)

    assert core.probe(None, toks17, 0, len(toks17) - 1) == 12
    slot = core.open(0)
    pages, hit_tokens = core.lookup(slot, toks17, False, len(toks17) - 1)
    assert hit_tokens == 12
    assert len(pages[0]) == 3 and all(p != 0 for p in pages[0])
    assert len(pages[1]) == 3 and pages[1][0] == 0 and pages[1][1] != 0 and pages[1][2] != 0
    core.check()


def test_hybrid_full_chunked_eagle_reserves_once_at_chunk_boundary():
    for groups, full_idx, ch_idx in [
        ([(0, 4, 0, 0), (2, 4, 8, 1)], 0, 1),
        ([(2, 4, 8, 0), (0, 4, 0, 1)], 1, 0),
    ]:
        # Pool size=8 (7 usable pages): donor allocates 3 pages in full (3) + 1 page in chunked after dropping chunk 0
        # behind boundary 8, then an evictor claims the freed chunk-0 pages so chunked has only page 2 cached (pages 0,1
        # are evicted in chunked while full still holds pages 0,1,2).
        core = KvCore(num_pages=10, groups=groups, max_model_len=64, caching=True, eagle=True)
        toks = list(range(50, 63))
        donor = core.open(0)
        hits, _ = core.lookup(donor, toks, False, len(toks) - 1)
        assert core.allocate(donor, 0, 8, 0, hits, 0) is not None
        assert core.publish(donor, 8) is None
        # Advancing past chunk boundary 8 releases chunk-0 pages in the chunked group into the free list.
        assert core.allocate(donor, 8, 4, 0, None, 0) is not None
        assert core.publish(donor, 12) is None
        # Evict the unreferenced chunk-0 pages from the chunked group while donor keeps full pages 0,1,2 pinned.
        evictor = core.open(1)
        ev_toks = list(range(200, 213))
        ev_hits, _ = core.lookup(evictor, ev_toks, False, len(ev_toks) - 1)
        assert core.allocate(evictor, 0, 8, 0, ev_hits, 0) is not None
        core.free(evictor)
        core.free(donor)

        # Full still has 3 pages (12 tokens) -> EAGLE drops 1 page to chunk boundary 8 -> chunked hits [0, 0] (8 tokens)
        # without subtracting a second page into the evicted chunk 0.
        assert core.probe(None, toks, 0, len(toks) - 1) == 8
        slot = core.open(0)
        pages, hit_tokens = core.lookup(slot, toks, False, len(toks) - 1)
        assert hit_tokens == 8
        assert len(pages[full_idx]) == 2 and all(p != 0 for p in pages[full_idx])
        assert pages[ch_idx] == [0, 0]
        core.check()


@pytest.mark.parametrize("index", range(24))
def test_snapshot_index_matches_python_goldens(index):
    case = _BODY["snapshots"][index]
    idx = SnapshotIndex(case["num_slots"], case["slot_offset"])
    for step, (op, a, b, res, state) in enumerate(case["ops"]):
        if op == "reserve":
            got = idx.reserve(a, exclude=set(b))
        elif op == "commit":
            got = idx.commit(a)
        elif op == "cancel":
            got = idx.cancel(a)
        elif op == "lookup":
            got = idx.lookup(a)
        elif op == "contains":
            got = idx.contains(a)
        elif op == "is_ready":
            got = idx.is_ready(a)
        else:
            got = idx.reset()
        assert got == res, (index, step, op)
        ready, pending, free = idx.state()
        assert [[list(x) for x in ready], [list(x) for x in pending], list(free)] == state, (index, step, op)


def test_snapshot_index_rejects_bad_sizes():
    with pytest.raises(ValueError):
        SnapshotIndex(0)
    with pytest.raises(ValueError):
        SnapshotIndex(2, -1)
    idx = SnapshotIndex(1, 7)
    assert (idx.num_slots, idx.slot_offset) == (1, 7)
    assert idx.reserve(2**127 + 5) == 7 and idx.reserve(3) is None
