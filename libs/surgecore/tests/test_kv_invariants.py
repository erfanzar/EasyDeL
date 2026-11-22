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

"""Reference-free invariants of surgecore's KV core under random operation sequences.

After every operation: K1 (reference counts equal page-list entries, the free list holds exactly the unreferenced
pages, every prefix entry points to a page with that key, the null page is never allocated), and K5 (a prefix hit only
ever returns pages that were published, and lookups and probes change no state). ``release_tail`` never releases a
published page.
"""

from __future__ import annotations

import numpy as np
import pytest
from surgecore import KvCore

pytestmark = pytest.mark.cpu_ok

LAYOUTS = {
    "full": [(0, 4, 0, 0)],
    "sliding": [(1, 4, 8, 0)],
    "chunked": [(2, 4, 8, 0)],
    "hybrid": [(0, 4, 0, 0), (1, 4, 8, 1)],
    "grouped": [(0, 4, 0, 0), (2, 4, 8, 1), (2, 4, 16, 2)],
}


def _snapshot(core: KvCore, slots) -> tuple:
    return (core.num_free, core.num_cached, core.free_order(), core.refs(), [core.page_ids(s) for s in slots])


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("seed", range(6))
def test_random_sequences_keep_k1_and_k5(layout, seed):
    rng = np.random.default_rng(seed)
    core = KvCore(num_pages=40, groups=LAYOUTS[layout], max_model_len=200, caching=True, eagle=bool(seed % 2))
    family = rng.integers(0, 50, 120).tolist()
    published: set[int] = set()  # pages that were published at some point (a hit must come from these)
    live: dict[int, dict] = {}
    for _ in range(300):
        u = rng.random()
        if not live or u < 0.1:
            p = int(rng.integers(0, 80))
            toks = family[:p] + rng.integers(50, 99, int(rng.integers(1, 20))).tolist()
            slot = core.open(int(rng.integers(0, 2)))
            live[slot] = {"tokens": toks, "computed": 0, "started": False}
            continue
        slot = int(rng.choice(list(live)))
        r = live[slot]
        toks = r["tokens"]
        if not r["started"]:
            before = _snapshot(core, live)
            probe = core.probe(None, toks, 0, len(toks) - 1)
            assert _snapshot(core, live) == before  # probes keep nothing
            assert probe >= 0
            hits, n_hit = core.lookup(slot, toks, False, len(toks) - 1)
            assert {p for g in hits for p in g if p} <= published  # K5
            n_new = min(len(toks) - n_hit, int(rng.integers(1, 30)))
            got = core.allocate(slot, 0, n_new, n_hit, hits, 0)
            if got is not None:
                r["computed"], r["started"] = n_hit + n_new, True
        elif u < 0.6:
            if r["computed"] >= len(toks):
                toks.append(int(rng.integers(0, 99)))
            n_new = min(len(toks) - r["computed"], int(rng.integers(1, 20)), 199 - r["computed"])
            if n_new > 0 and core.allocate(slot, r["computed"], n_new, 0, None, 0) is not None:
                r["computed"] += n_new
                n = min(r["computed"], len(toks))
                need = core.publish(slot, n)
                if need is not None:
                    core.publish(slot, n, toks[need[0] : need[1]], need[0])
                for group in core.page_ids(slot):
                    published.update(p for p in group[: n // 4] if p)
        elif u < 0.7:
            before = core.page_ids(slot)
            core.release_tail(slot, r["computed"])
            after = core.page_ids(slot)
            assert all(a == b[: len(a)] for a, b in zip(after, before, strict=True))
        elif u < 0.85:  # preemption: pages go, keys stay; a slot left holding nothing is closed and reopened
            r["computed"], r["started"] = 0, False
            if core.free(slot):
                del live[slot]
                live[core.open(0)] = r
        else:  # finish
            if not core.free(slot):
                assert core.forget_keys(slot)
            del live[slot]
        core.check()  # K1
        assert core.num_free + sum(1 for x in core.refs() if x) == 39  # null page never allocated or freed
    for slot in list(live):
        if not core.free(slot):
            assert core.forget_keys(slot)
    core.check()
    assert core.num_free == 39 and core.num_open == 0
    assert core.reset() and core.num_cached == 0


def test_reset_refused_while_pages_in_use_and_errors_map_to_python_types():
    core = KvCore(num_pages=8, groups=[(0, 2, 0, 0)], max_model_len=32, caching=True, eagle=False)
    slot = core.open(0)
    assert core.allocate(slot, 0, 3, 0, None, 0) == [[1, 2]]
    assert not core.reset()
    with pytest.raises(ValueError):
        core.allocate(slot, 0, 0, 0, None, 0)
    with pytest.raises(ValueError):
        KvCore(num_pages=0, groups=[], max_model_len=8, caching=False, eagle=False)
    with pytest.raises(ValueError):
        KvCore(num_pages=8, groups=[(0, 4, 0, 0), (0, 8, 0, 1)], max_model_len=8, caching=True, eagle=False)
    core.free(slot)
    assert core.reset()
