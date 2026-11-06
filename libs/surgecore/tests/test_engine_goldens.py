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

"""Parity of surgecore's EngineCore with eSurge's former Python scheduler, through the Python binding.

``data/engine_goldens.npz`` holds one call log per case, captured from the Python scheduler (single-rank, async and
data-parallel) with a fake runner; its header names the source module and the commit. Each log is a zigzag-LEB128
varint stream: a config block, then ``ADD`` (a request and the rank it was routed to), ``FINISH`` (an abort or a
stop-string finish), ``SCHEDULE`` (the expected plan: per entry the request, scheduled and computed tokens, whether
it samples, its kind and its page ids per group; then the preempted and finished sets and the snapshot pairs) and
``COMMIT`` (the sampled tokens of the oldest uncommitted plan and the expected events). Every case must be
reproduced exactly, and the core's invariant check must pass after every call.
"""

from __future__ import annotations

import json
import pathlib
from collections import deque

import numpy as np
import pytest
from surgecore._native import CoreClosed, EngineCore

pytestmark = pytest.mark.cpu_ok

_FIXTURE = pathlib.Path(__file__).parent / "data" / "engine_goldens.npz"
with np.load(_FIXTURE) as _npz:
    _HEADER = json.loads(_npz["header"].item())
    _NAMES = json.loads(_npz["names"].item())
    _OFFSETS = _npz["offsets"].astype(np.int64)
    _LOG = _npz["log"].tobytes()

OP_ADD, OP_FINISH, OP_SCHEDULE, OP_COMMIT = 1, 2, 3, 4
FLAG_NO_PREFIX_HIT = 2


class _Reader:
    """Zigzag LEB128 varints."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def done(self) -> bool:
        return self.pos >= len(self.data)

    def next(self) -> int:
        z, shift = 0, 0
        while True:
            b = self.data[self.pos]
            self.pos += 1
            z |= (b & 0x7F) << shift
            if not b & 0x80:
                return (z >> 1) ^ -(z & 1)
            shift += 7

    def vec(self, n: int) -> list[int]:
        return [self.next() for _ in range(n)]

    def lst(self) -> list[int]:
        return self.vec(self.next())


def _make_core(c: list[int]) -> EngineCore:
    ng = c[12]
    groups = np.asarray(c[13 : 13 + 4 * ng], np.int32).reshape(ng, 4)
    return EngineCore(
        max_rows=c[0],
        max_model_len=c[1],
        num_pages=c[2],
        dp_size=c[3],
        groups=groups,
        token_budget=c[4],
        long_prefill_threshold=c[5],
        chunked_prefill=bool(c[6]),
        policy=c[7],
        prefix_caching=bool(c[8]),
        snapshot_slots=c[9],
        async_scheduling=bool(c[10]),
        token_safety_margin=c[11],
    )


def _one_token_each(tokens: list[int]) -> dict[str, np.ndarray]:
    """Commit arguments for at most one sampled token per entry (-1: the entry samples nothing)."""
    flat = [t for t in tokens if t >= 0]
    offsets = np.cumsum([0] + [int(t >= 0) for t in tokens]).astype(np.int32)
    return {"sampled_tokens": np.asarray(flat, np.int32), "sampled_offsets": offsets}


def _replay(log: bytes) -> int:
    r = _Reader(log)
    c = r.lst()
    ng = c[12]
    core = _make_core(c)
    handle_of: dict[int, int] = {}
    id_of: dict[int, int] = {}
    pending: deque = deque()
    steps = 0
    while not r.done():
        op = r.next()
        where = f"step {steps}"
        if op == OP_ADD:
            idx = r.next()
            prompt = r.lst()
            max_tokens, eos, ignore_eos, priority, arrival, no_hit = r.vec(6)
            stops = r.lst()
            nsp = r.next()
            spans = r.vec(2 * nsp)
            rank = r.next()
            (h,) = core.add_requests(
                prompt_tokens=np.asarray(prompt, np.int32),
                prompt_offsets=np.asarray([0, len(prompt)], np.int32),
                max_tokens=np.asarray([max_tokens], np.int64),
                eos_token=np.asarray([eos], np.int32),
                ignore_eos=np.asarray([bool(ignore_eos)]),
                priority=np.asarray([priority], np.int64),
                arrival=np.asarray([float(arrival)], np.float64),
                flags=np.asarray([FLAG_NO_PREFIX_HIT if no_hit else 0], np.uint32),
                stop_tokens=np.asarray(stops, np.int32),
                stop_offsets=np.asarray([0, len(stops)], np.int32),
                image_spans=np.asarray(spans, np.int64).reshape(nsp, 2) if nsp else None,
                span_offsets=np.asarray([0, nsp], np.int32),
            ).tolist()
            handle_of[idx], id_of[h] = h, idx
            assert core.rank_of(h) == rank, f"{where}: request {idx} routed to rank {core.rank_of(h)}, expected {rank}"
        elif op == OP_FINISH:
            victims = [handle_of[i] for i in r.lst()]
            core.finish(np.asarray(victims, np.int64), r.next())
        elif op == OP_SCHEDULE:
            plan = core.schedule()
            n = r.next()
            assert plan.n == n, f"{where}: {plan.n} entries, expected {n}"
            handles, sched, comp = plan.handles.tolist(), plan.scheduled.tolist(), plan.computed.tolist()
            emits, kinds = plan.emits.tolist(), plan.kinds.tolist()
            pages, offs = plan.new_pages.tolist(), plan.new_page_offsets.tolist()
            for i in range(n):
                want = r.vec(5)
                got = [id_of[handles[i]], sched[i], comp[i], emits[i], kinds[i]]
                assert got == want, f"{where} entry {i}: {got} != {want}"
                for g in range(ng):
                    k = i * ng + g
                    assert pages[offs[k] : offs[k + 1]] == r.lst(), f"{where} entry {i} group {g}: pages differ"
            assert sorted(id_of[h] for h in plan.preempted.tolist()) == r.lst(), f"{where}: preempted"
            assert sorted(id_of[h] for h in plan.finished.tolist()) == r.lst(), f"{where}: finished"
            for what, arr in (("restores", plan.snapshot_restores), ("saves", plan.snapshot_saves)):
                have = sorted((id_of[handles[e]], s) for e, s in arr.tolist())
                assert [x for p in have for x in p] == r.vec(2 * r.next()), f"{where}: snapshot {what}"
            assert isinstance(core.record(plan), bytes)
            if plan.n or plan.preempted.size or plan.finished.size:
                pending.append(plan)
            steps += 1
        elif op == OP_COMMIT:
            tokens = r.lst()
            plan = pending.popleft()
            ev = core.commit(plan, **_one_token_each(tokens))
            ne = r.next()
            assert ev.n == ne, f"{where}: {ev.n} events, expected {ne}"
            eh, eo, et = ev.handles.tolist(), ev.token_offsets.tolist(), ev.tokens.tolist()
            fin, stop, cached = ev.finish.tolist(), ev.stop_token.tolist(), ev.cached_tokens.tolist()
            for i in range(ne):
                idx = r.next()
                want_tokens = r.lst()
                want_tail = r.vec(3)
                got = (id_of[eh[i]], et[eo[i] : eo[i + 1]], [fin[i], stop[i], cached[i]])
                assert got == (idx, want_tokens, want_tail), f"{where} event {i}: {got}"
        else:
            raise AssertionError(f"unknown op {op}")
        core.check()
    assert not pending and core.num_inflight() == 0
    return steps


def test_fixture_header_names_only_source_and_commit():
    assert _HEADER == {"commit": "c499246765d4", "source": "easydel.inference.esurge.scheduler"}
    assert len(_NAMES) == 108


@pytest.mark.parametrize("index", range(len(_NAMES)), ids=_NAMES)
def test_golden_case_reproduces(index: int):
    assert _replay(_LOG[_OFFSETS[index] : _OFFSETS[index + 1]]) > 0


def _small_core(**kw) -> EngineCore:
    args = dict(max_rows=4, max_model_len=64, num_pages=32, dp_size=1, groups=np.asarray([[0, 4, 0]], np.int32))
    args.update(token_budget=16, async_scheduling=False)
    args.update(kw)
    return EngineCore(**args)


def _add(core: EngineCore, prompts: list[list[int]], max_tokens: int = 4):
    flat = [t for p in prompts for t in p]
    offs = np.cumsum([0] + [len(p) for p in prompts]).astype(np.int32)
    return core.add_requests(
        prompt_tokens=np.asarray(flat, np.int32),
        prompt_offsets=offs,
        max_tokens=np.full(len(prompts), max_tokens, np.int64),
    )


def test_admission_is_atomic():
    core = _small_core()
    with pytest.raises(ValueError):
        _add(core, [[7, 8, 9], list(range(64))])
    assert core.counts().tolist()[:2] == [0, 0]
    assert _add(core, [[7, 8, 9]]).size == 1


def test_plans_commit_in_dispatch_order_and_shutdown_closes():
    core = _small_core(async_scheduling=True)
    _add(core, [[7, 8, 9], [10, 11]])
    first = core.schedule()
    second = core.schedule()
    assert first.n == 2 and second.n == 2
    with pytest.raises(ValueError):
        core.commit(second, **_one_token_each([1, 1]))
    ev = core.commit(first, **_one_token_each([20, 21]))
    assert ev.tokens.tolist() == [20, 21]
    core.commit(second, **_one_token_each([22, 23]))
    core.check()
    core.shutdown()
    with pytest.raises(CoreClosed):
        core.schedule()


def test_finish_frees_pages_and_reports_once():
    core = _small_core()
    (h,) = _add(core, [[7, 8, 9, 10, 11]]).tolist()
    plan = core.schedule()
    free_before = core.counts().tolist()[2]
    assert core.finish(np.asarray([h, h], np.int64), 3).tolist() == [h]
    assert core.counts().tolist()[2] > free_before
    core.commit(plan, **_one_token_each([30]))
    assert core.schedule().finished.tolist() == [h]
    assert core.schedule().finished.size == 0
    core.check()
