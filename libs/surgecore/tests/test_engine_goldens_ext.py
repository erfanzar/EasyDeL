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

"""Parity of surgecore's EngineCore with eSurge's former Python scheduler on the extended cases, through the binding.

``data/engine_goldens_ext.npz`` holds call logs captured from the Python scheduler with a fake runner that covers
speculative decoding (synchronous drafts, seeded acceptance, EAGLE lookahead, budget trims, stops and length limits
inside accepted drafts, drafts returned as ``None``), entries absent from the runner output or listed without a
sampled row, multimodal page keys (``hash_extra`` with image spans), request-count buckets, ``include_finished_set``,
the structured-output pending flag, and the scheduler determinism scenario (sync and async). The encoding is the one
of ``test_engine_goldens.py`` with the ops ``ADD2``, ``SCHEDULE2`` and ``COMMIT2``, which add per entry the
scheduled drafts and the fed tokens, per plan the structured-pending flag, the request bucket and the common prefix
pages, and per commit the runner output's presence codes, accepted counts and new drafts, each event's logprob start
and the finished set. Every case must be reproduced exactly, and the invariant check must pass after every call.
"""

from __future__ import annotations

import json
import pathlib
from collections import deque

import numpy as np
import pytest
from surgecore._native import PLAN_STRUCTURED_PENDING, EngineCore

pytestmark = pytest.mark.cpu_ok

_FIXTURE = pathlib.Path(__file__).parent / "data" / "engine_goldens_ext.npz"
with np.load(_FIXTURE) as _npz:
    _HEADER = json.loads(_npz["header"].item())
    _NAMES = json.loads(_npz["names"].item())
    _OFFSETS = _npz["offsets"].astype(np.int64)
    _LOG = _npz["log"].tobytes()

OP_FINISH, OP_ADD2, OP_SCHEDULE2, OP_COMMIT2 = 2, 5, 6, 7


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
    at = 13 + 4 * ng
    k, eagle, fin, nb = c[at : at + 4]
    buckets = c[at + 4 : at + 4 + nb]
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
        req_buckets=np.asarray(buckets, np.int64) if buckets else None,
        num_draft_tokens=k,
        use_eagle=bool(eagle),
        include_finished_set=bool(fin),
    )


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
        if op == OP_ADD2:
            idx = r.next()
            prompt = r.lst()
            max_tokens, eos, ignore_eos, priority, arrival, flags = r.vec(6)
            stops = r.lst()
            nsp = r.next()
            spans = r.vec(2 * nsp)
            h0, h1, l0, l1 = r.vec(4)
            rank = r.next()
            (h,) = core.add_requests(
                prompt_tokens=np.asarray(prompt, np.int32),
                prompt_offsets=np.asarray([0, len(prompt)], np.int32),
                max_tokens=np.asarray([max_tokens], np.int64),
                eos_token=np.asarray([eos], np.int32),
                ignore_eos=np.asarray([bool(ignore_eos)]),
                priority=np.asarray([priority], np.int64),
                arrival=np.asarray([float(arrival)], np.float64),
                flags=np.asarray([flags], np.uint32),
                stop_tokens=np.asarray(stops, np.int32),
                stop_offsets=np.asarray([0, len(stops)], np.int32),
                hash_extra=np.asarray([[(h0 << 32) | h1, (l0 << 32) | l1]], np.uint64),
                image_spans=np.asarray(spans, np.int64).reshape(nsp, 2) if nsp else None,
                span_offsets=np.asarray([0, nsp], np.int32),
                # Pinned to the recorded rank: the cases check scheduling after admission; routing (which also weighs
                # rank load) is tested in src/engine/dp.rs.
                dp_rank=np.asarray([rank], np.int64),
            ).tolist()
            handle_of[idx], id_of[h] = h, idx
            assert core.rank_of(h) == rank, f"{where}: request {idx} routed to rank {core.rank_of(h)}, expected {rank}"
        elif op == OP_FINISH:
            victims = [handle_of[i] for i in r.lst()]
            core.finish(np.asarray(victims, np.int64), r.next())
        elif op == OP_SCHEDULE2:
            plan = core.schedule()
            n = r.next()
            assert plan.n == n, f"{where}: {plan.n} entries, expected {n}"
            handles, sched, comp = plan.handles.tolist(), plan.scheduled.tolist(), plan.computed.tolist()
            emits, kinds = plan.emits.tolist(), plan.kinds.tolist()
            pages, offs = plan.new_pages.tolist(), plan.new_page_offsets.tolist()
            dt, do = plan.draft_tokens.tolist(), plan.draft_offsets.tolist()
            nt, no = plan.new_tokens.tolist(), plan.new_token_offsets.tolist()
            for i in range(n):
                want = r.vec(5)
                got = [id_of[handles[i]], sched[i], comp[i], emits[i], kinds[i]]
                assert got == want, f"{where} entry {i}: {got} != {want}"
                for g in range(ng):
                    k = i * ng + g
                    assert pages[offs[k] : offs[k + 1]] == r.lst(), f"{where} entry {i} group {g}: pages differ"
                assert dt[do[i] : do[i + 1]] == r.lst(), f"{where} entry {i}: drafts differ"
                assert nt[no[i] : no[i + 1]] == r.lst(), f"{where} entry {i}: new tokens differ"
            assert sorted(id_of[h] for h in plan.preempted.tolist()) == r.lst(), f"{where}: preempted"
            assert sorted(id_of[h] for h in plan.finished.tolist()) == r.lst(), f"{where}: finished"
            for what, arr in (("restores", plan.snapshot_restores), ("saves", plan.snapshot_saves)):
                have = sorted((id_of[handles[e]], s) for e, s in arr.tolist())
                assert [x for p in have for x in p] == r.vec(2 * r.next()), f"{where}: snapshot {what}"
            structured, bucket = r.vec(2)
            assert int(bool(plan.flags & PLAN_STRUCTURED_PENDING)) == structured, f"{where}: structured pending"
            assert plan.req_bucket == bucket, f"{where}: req_bucket {plan.req_bucket} != {bucket}"
            assert plan.common_prefix_pages.tolist() == r.lst(), f"{where}: common prefix pages"
            if plan.n or plan.preempted.size or plan.finished.size:
                pending.append(plan)
            steps += 1
        elif op == OP_COMMIT2:
            n = r.next()
            present, sampled, soff, accepted = [], [], [0], []
            for _ in range(n):
                present.append(r.next())
                sampled += r.lst()
                soff.append(len(sampled))
                accepted.append(r.next())
            drafts_none = r.next()
            kwargs = {}
            if not drafts_none:
                drafts, doff = [], [0]
                for _ in range(n):
                    drafts += r.lst()
                    doff.append(len(drafts))
                kwargs = dict(new_drafts=np.asarray(drafts, np.int32), new_draft_offsets=np.asarray(doff, np.int32))
            plan = pending.popleft()
            ev = core.commit(
                plan,
                sampled_tokens=np.asarray(sampled, np.int32),
                sampled_offsets=np.asarray(soff, np.int32),
                accepted=np.asarray(accepted, np.int32),
                present=np.asarray(present, np.uint8),
                **kwargs,
            )
            ne = r.next()
            assert ev.n == ne, f"{where}: {ev.n} events, expected {ne}"
            eh, eo, et = ev.handles.tolist(), ev.token_offsets.tolist(), ev.tokens.tolist()
            fin, stop, cached = ev.finish.tolist(), ev.stop_token.tolist(), ev.cached_tokens.tolist()
            lps = ev.logprob_starts.tolist()
            for i in range(ne):
                idx = r.next()
                want_tokens = r.lst()
                want_tail = r.vec(3)
                lp = r.next()
                got = (id_of[eh[i]], et[eo[i] : eo[i + 1]], [fin[i], stop[i], cached[i]])
                assert got == (idx, want_tokens, want_tail), f"{where} event {i}: {got}"
                assert lp < 0 or lps[i] == lp, f"{where} event {i}: logprob start {lps[i]} != {lp}"
            assert sorted(id_of[h] for h in ev.finished_all.tolist()) == r.lst(), f"{where}: finished_all"
        else:
            raise AssertionError(f"unknown op {op}")
        core.check()
    assert not pending and core.num_inflight() == 0
    return steps


def test_extended_fixture_header_names_only_source_and_commit():
    assert _HEADER == {"commit": "c499246765d4", "source": "easydel.inference.esurge.scheduler"}
    assert len(_NAMES) == 53


@pytest.mark.parametrize("index", range(len(_NAMES)), ids=_NAMES)
def test_extended_golden_case_reproduces(index: int):
    assert _replay(_LOG[_OFFSETS[index] : _OFFSETS[index + 1]]) > 0
