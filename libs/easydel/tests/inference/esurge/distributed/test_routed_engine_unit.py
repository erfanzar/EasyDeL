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

"""Lightweight (no-engine) unit tests for :class:`RoutedEngine` and :class:`ReplicaAttacher`.

These duck-type the member surface so the router's public request-lifecycle
bookkeeping (`generate`, `stream`, `chat`, `replace_member`, `close`, and
cross-front shared inflight accounting) can be verified on CPU without building
real eSurge replicas.
"""

from __future__ import annotations

import gc
import mmap
import os
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import routed_engine as routed_engine_mod
from easydel.inference.esurge.distributed.request_plane import RequestPlaneUnavailable
from easydel.inference.esurge.distributed.routed_engine import (
    _SHARED_MAX_MEMBERS,
    _SHARED_ROW_BYTES,
    RoutedEngine,
    _allocate_shared_file,
    _is_pid_alive,
    clear_shared_inflight_row,
)
from easydel.inference.esurge.serving.app import ReplicaAttacher

pytestmark = pytest.mark.cpu_ok


class _FakeMember:
    """Minimal engine-shaped member supporting ``generate``, ``stream``, ``chat``, and ``close``."""

    def __init__(
        self,
        *,
        name: str = "fake",
        generate_error: Exception | None = None,
        barrier: threading.Barrier | None = None,
        on_enter=None,
    ) -> None:
        self.esurge_name = name
        self.max_model_len = 128
        self.max_num_seqs = 4
        self.page_size = 16
        self.alive = True
        self.closed = False
        self.tokenizer = SimpleNamespace(
            apply_chat_template=lambda msgs, **_kw: "|".join(f"{m['role']}:{m['content']}" for m in msgs)
        )
        self.calls: list[tuple[list[str], dict]] = []
        self.stream_calls: list[str] = []
        self.chat_calls: list[list[dict]] = []
        self.active_streams = 0
        self.closed_streams = 0
        self._lock = threading.Lock()
        self._generate_error = generate_error
        self._barrier = barrier
        self._on_enter = on_enter

    def generate(self, prompts, sampling_params=None, use_tqdm=False, tool_parser_request=None, **kwargs):
        with self._lock:
            self.calls.append((list(prompts), dict(kwargs)))
        if self._on_enter is not None:
            self._on_enter()
        if self._barrier is not None:
            self._barrier.wait(timeout=5.0)
        if self._generate_error is not None:
            raise self._generate_error
        return [SimpleNamespace(prompt=p, member=self.esurge_name) for p in prompts]

    def stream(self, prompt, sampling_params=None, request_id=None, tool_parser_request=None):
        with self._lock:
            self.stream_calls.append(prompt)
            self.active_streams += 1

        def _gen():
            try:
                yield SimpleNamespace(prompt=prompt, member=self.esurge_name, chunk=0)
                yield SimpleNamespace(prompt=prompt, member=self.esurge_name, chunk=1)
            finally:
                with self._lock:
                    self.active_streams -= 1
                    self.closed_streams += 1

        return _gen()

    def chat(self, messages, *, stream: bool = False, **_kwargs):
        with self._lock:
            self.chat_calls.append(list(messages))
        if not stream:
            return SimpleNamespace(messages=list(messages), member=self.esurge_name)
        with self._lock:
            self.active_streams += 1

        def _gen():
            try:
                yield SimpleNamespace(messages=list(messages), member=self.esurge_name, chunk=0)
                yield SimpleNamespace(messages=list(messages), member=self.esurge_name, chunk=1)
            finally:
                with self._lock:
                    self.active_streams -= 1
                    self.closed_streams += 1

        return _gen()

    def close(self) -> None:
        self.closed = True


def _open_stream(router: RoutedEngine, prompt: str):
    """Start a public ``router.stream`` call and advance to its first chunk so its slot is held."""
    it = router.stream(prompt)
    first = next(it)
    return it, first


def test_routed_generate_rejects_mixed_ids_without_leaking_slots():
    # Two fresh members: the batch would fan prompt 0 -> member 0 and
    # prompt 1 -> member 1 (JSQ by index), so any per-member check would miss
    # the mix. Up-front validation must reject it on the caller's thread before
    # any slot is acquired or member is invoked.
    member_a, member_b = _FakeMember(name="a"), _FakeMember(name="b")
    router = RoutedEngine([member_a, member_b])

    with pytest.raises(ValueError, match="Mix of explicit and auto"):
        router.generate(["p0", "p1"], request_id=["explicit-0", None])

    assert member_a.calls == [] and member_b.calls == [], "no member.generate should fire for the invalid batch"

    # A subsequent clean batch still balances cleanly across both members.
    out = router.generate(["p0", "p1"])
    assert [o.member for o in out] == ["a", "b"]


def test_routed_generate_all_explicit_ids_is_accepted():
    member = _FakeMember()
    router = RoutedEngine([member])
    out = router.generate(["p0", "p1"], request_id=["id-0", "id-1"])
    assert len(out) == 2
    assert member.calls[0][1]["request_id"] == ["id-0", "id-1"]


def test_shared_prefix_burst_spreads_once_the_pinned_member_is_busy():
    # One prompt shared by a burst: affinity pins it to member 0 while member 0
    # holds at most `affinity_max_imbalance` more in-flight requests than the
    # idle member; once exceeded, new streams spill to member 1.
    members = [_FakeMember(name="a"), _FakeMember(name="b")]
    router = RoutedEngine(members, affinity_max_imbalance=2)
    prompt = "shared system prompt " * 64
    open_streams = []
    chosen: list[str] = []
    try:
        for _ in range(8):
            it, first = _open_stream(router, prompt)
            open_streams.append(it)
            chosen.append(first.member)
        assert chosen.count("a") >= 1 and chosen.count("b") >= 1
        assert abs(members[0].active_streams - members[1].active_streams) <= 3
        # Once equally loaded (both at 4 active), the prefix stays on the member that last served it.
        expected_pinned = chosen[-1]
        assert router.members[router.select_member(prompt)].esurge_name == expected_pinned
    finally:
        for it in open_streams:
            it.close()
    assert members[0].active_streams == 0 and members[1].active_streams == 0
    with pytest.raises(ValueError, match="affinity_max_imbalance"):
        RoutedEngine(members, affinity_max_imbalance=-1)


def test_routed_generate_surfaces_member_errors_and_releases_slots():
    failing = _FakeMember(name="fail", generate_error=RuntimeError("member exploded"))
    healthy = _FakeMember(name="ok")
    router = RoutedEngine([failing, healthy])

    with pytest.raises(RuntimeError, match="member exploded"):
        router.generate(["p0"])

    # Because the failed call released its slot, failing (member 0) is back at
    # load 0 and stays eligible rather than leaking load onto healthy.
    failing._generate_error = None
    out = router.generate(["p1", "p2"])
    assert [o.member for o in out] == ["fail", "ok"]


def test_shared_inflight_table_balances_uneven_multi_front_burst(tmp_path):
    # 8 independent fronts attach 4 replicas in different orders via ReplicaAttacher
    # and receive an uneven 128-request burst through public `stream()` + `chat(stream=True)`.
    # Shared mmap accounting + canonical member_id mapping must route exactly 32/32/32/32.
    orders = [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]] * 2
    per_front_counts = [21, 13, 19, 11, 17, 15, 18, 14]
    assert sum(per_front_counts) == 128

    fronts: list[RoutedEngine] = []
    try:
        for f_idx, order in enumerate(orders):
            up: set[int] = set()

            def _make_connect(allowed: set[int]):
                def _connect(addr: str):
                    idx = int(addr)
                    if idx not in allowed:
                        raise RequestPlaneUnavailable(f"r{idx} not up yet")
                    return _FakeMember(name=f"r{idx}")

                return _connect

            def _make_factory(front_index: int):
                def _factory(members, *, planned_members=None, member_ids=None):
                    return RoutedEngine(
                        members,
                        planned_members=planned_members,
                        member_ids=member_ids,
                        tie_breaker_offset=front_index,
                        shared_inflight_dir=tmp_path,
                        front_index=front_index,
                        front_count=8,
                    )

                return _factory

            attacher = ReplicaAttacher(
                [str(i) for i in range(4)],
                connect=_make_connect(up),
                log=lambda _m: None,
                router_factory=_make_factory(f_idx),
            )
            # Bring replicas up in this front's distinct order.
            up.add(order[0])
            router = attacher.wait_for_first()
            for replica_idx in order[1:]:
                up.add(replica_idx)
                assert attacher.attach_round() == 1
            fronts.append(router)

        replica_totals = [0, 0, 0, 0]
        open_streams = []
        try:
            for f_idx, count in enumerate(per_front_counts):
                router = fronts[f_idx]
                for k in range(count):
                    if k % 2 == 0:
                        it, first = _open_stream(router, f"unique-prompt-{f_idx}-{k}")
                    else:
                        it = router.chat([{"role": "user", "content": f"unique-chat-{f_idx}-{k}"}], stream=True)
                        first = next(it)
                    replica_totals[int(first.member[1:])] += 1
                    open_streams.append(it)
            assert replica_totals == [32, 32, 32, 32], f"expected exact 32/32/32/32 balance, got {replica_totals}"
        finally:
            for it in open_streams:
                it.close()

        # Concurrent multi-front generate() wave after streams close also balances cleanly.
        # Synchronize acquisition order via an entry event so each front observes prior fronts'
        # mmap writes deterministically while all 8 generate() calls still overlap at the barrier.
        barrier = threading.Barrier(8)
        entered = threading.Event()
        shared_replicas = [_FakeMember(name=f"g{i}", barrier=barrier, on_enter=entered.set) for i in range(4)]
        for router in fronts:
            for slot, member in enumerate(router.members):
                router.replace_member(slot, shared_replicas[int(member.esurge_name[1:])])
        thread_errors: list[BaseException] = []

        def _run_front(idx: int) -> None:
            try:
                fronts[idx].generate(f"gen-wave-{idx}")
            except BaseException as exc:
                thread_errors.append(exc)
                barrier.abort()
                entered.set()

        threads: list[threading.Thread] = []
        for i in range(8):
            entered.clear()
            t = threading.Thread(target=_run_front, args=(i,))
            t.start()
            threads.append(t)
            assert entered.wait(timeout=5.0), f"front {i} did not reach member.generate()"
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "concurrent generate worker thread timed out"
        assert not thread_errors, f"concurrent generate wave failed: {thread_errors!r}"
        assert [len(m.calls) for m in shared_replicas] == [2, 2, 2, 2]
    finally:
        for router in fronts:
            router.close()


def test_replace_member_unwinding_streams_do_not_clobber_shared_inflight_table(tmp_path):
    # Regression for #9 item 1: unwinding streams from an old replaced member must
    # NOT overwrite the replacement member's live count in router_inflight.bin.
    old_r0 = _FakeMember(name="r0-old")
    new_r0 = _FakeMember(name="r0-new")
    f0 = RoutedEngine(
        [old_r0, _FakeMember(name="r1")],
        planned_members=2,
        tie_breaker_offset=0,
        shared_inflight_dir=tmp_path,
        front_index=0,
        front_count=2,
    )
    f1 = RoutedEngine(
        [_FakeMember(name="r0"), _FakeMember(name="r1")],
        planned_members=2,
        tie_breaker_offset=0,
        shared_inflight_dir=tmp_path,
        front_index=1,
        front_count=2,
    )
    old_streams = []
    new_streams = []
    f1_streams = []
    try:
        # Hold 4 active requests on old_r0 (via chat stream & stream) while r1 stays idle
        # by temporarily marking f0's r1 down so all 4 land on old_r0.
        f0.members[1].alive = False
        for i in range(4):
            if i % 2 == 0:
                it = f0.chat([{"role": "user", "content": f"old-chat-{i}"}], stream=True)
                first = next(it)
            else:
                it, first = _open_stream(f0, f"old-stream-{i}")
            assert first.member == "r0-old"
            old_streams.append(it)

        # Replace replica 0 with a fresh handle and open 2 live streams on new_r0.
        f0.replace_member(0, new_r0)
        for i in range(2):
            it = f0.chat([{"role": "user", "content": f"new-chat-{i}"}], stream=True)
            first = next(it)
            assert first.member == "r0-new"
            new_streams.append(it)
        f0.members[1].alive = True

        # Unwind ALL 4 old streams: must not overwrite mmap[0, 0] (= 2) with 3..0!
        for it in old_streams:
            it.close()
        old_streams.clear()

        # Front 1 must still observe replica 0 at load 2 (replica 1 at load 0),
        # so the next 2 requests on Front 1 must route to replica 1.
        for i in range(2):
            it, first = _open_stream(f1, f"peer-check-{i}")
            f1_streams.append(it)
            assert first.member == "r1", "stale replaced counter clobbered live replacement's shared count"
    finally:
        for it in (*old_streams, *new_streams, *f1_streams):
            it.close()
        f0.close()
        f1.close()


def test_replica_attacher_out_of_order_attach_under_load_preserves_shared_counts(tmp_path):
    # Regression for #9 item 6: attaching replica 2 after replica 1 is already serving
    # in slot 0 must not momentarily collide on member_id=1 and zero replica 1's shared load.
    up_f0 = {1}

    def connect_f0(addr: str):
        idx = int(addr)
        if idx not in up_f0:
            raise RequestPlaneUnavailable(f"r{idx} down")
        return _FakeMember(name=f"r{idx}")

    def make_factory(front_index: int):
        def _factory(members, *, planned_members=None, member_ids=None):
            return RoutedEngine(
                members,
                planned_members=planned_members,
                member_ids=member_ids,
                tie_breaker_offset=0,
                shared_inflight_dir=tmp_path,
                front_index=front_index,
                front_count=2,
            )

        return _factory

    attacher_f0 = ReplicaAttacher(
        ["0", "1", "2"],
        connect=connect_f0,
        log=lambda _m: None,
        router_factory=make_factory(0),
    )
    f0 = attacher_f0.wait_for_first()
    f1 = RoutedEngine(
        [_FakeMember(name="r1"), _FakeMember(name="r2")],
        planned_members=3,
        member_ids=[1, 2],
        tie_breaker_offset=0,
        shared_inflight_dir=tmp_path,
        front_index=1,
        front_count=2,
    )
    f0_streams = []
    f1_streams = []
    try:
        # Hold 4 active streams on replica 1 (slot 0 on f0).
        for i in range(4):
            it, first = _open_stream(f0, f"r1-live-{i}")
            assert first.member == "r1"
            f0_streams.append(it)

        # Late-attach replica 2 into slot 1 while replica 1 is actively serving.
        up_f0.add(2)
        assert attacher_f0.attach_round() == 1

        # Front 1 must still see replica 1 carrying load 4 and send its next 4 requests to replica 2.
        for i in range(4):
            it, first = _open_stream(f1, f"f1-check-{i}")
            f1_streams.append(it)
            assert first.member == "r2", "out-of-order attach wiped replica 1's shared inflight count"

        # Legacy 2-step caller (`add_member(handle)` without `member_id` then `set_member_id`)
        # must also preserve replica 1's shared count when `slot == 1` would otherwise collide
        # with slot 0's `member_id == 1` while slot 0 holds 4 active streams.
        legacy_dir = tmp_path / "legacy-collision"
        f0_legacy = RoutedEngine(
            [_FakeMember(name="r1")],
            planned_members=3,
            member_ids=[1],
            tie_breaker_offset=0,
            shared_inflight_dir=legacy_dir,
            front_index=0,
            front_count=2,
        )
        f1_legacy = RoutedEngine(
            [_FakeMember(name="r1"), _FakeMember(name="r2")],
            planned_members=3,
            member_ids=[1, 2],
            tie_breaker_offset=0,
            shared_inflight_dir=legacy_dir,
            front_index=1,
            front_count=2,
        )
        legacy_f0_streams = []
        legacy_f1_streams = []
        try:
            for i in range(4):
                it, first = _open_stream(f0_legacy, f"legacy-r1-live-{i}")
                assert first.member == "r1"
                legacy_f0_streams.append(it)

            # Adding a second member gives slot == 1, which matches r1's member_id == 1.
            legacy_slot = f0_legacy.add_member(_FakeMember(name="r2"))
            assert legacy_slot == 1
            # Before set_member_id runs, r1's shared count (4) must remain intact.
            for i in range(4):
                it, first = _open_stream(f1_legacy, f"f1-legacy-pre-set-{i}")
                legacy_f1_streams.append(it)
                assert first.member == "r2", "legacy add_member wiped replica 1's shared count on slot=1 collision"
            for it in legacy_f1_streams:
                it.close()
            legacy_f1_streams.clear()

            f0_legacy.set_member_id(legacy_slot, 2)
            for i in range(4):
                it, first = _open_stream(f1_legacy, f"f1-legacy-post-set-{i}")
                legacy_f1_streams.append(it)
                assert first.member == "r2", "legacy add_member + set_member_id wiped replica 1's shared count"
        finally:
            for it in (*legacy_f0_streams, *legacy_f1_streams):
                it.close()
            f0_legacy.close()
            f1_legacy.close()
    finally:
        for it in (*f0_streams, *f1_streams):
            it.close()
        f0.close()
        f1.close()


def test_dead_front_pid_row_is_ignored_and_cleared_for_surviving_fronts(tmp_path):
    # Regression for #9 item 7: a crashed front (SIGKILL / os._exit without close())
    # must not poison surviving fronts — neither while unreaped (zombie) nor after reaping.
    child_code = (
        "import os, sys\n"
        "from types import SimpleNamespace\n"
        "from easydel.inference.esurge.distributed.routed_engine import RoutedEngine\n"
        "class _M:\n"
        "    esurge_name = 'r0'; max_model_len = 128; max_num_seqs = 4; page_size = 16; alive = True\n"
        "    def stream(self, p, **_kw):\n"
        "        yield SimpleNamespace(prompt=p)\n"
        "        yield SimpleNamespace(prompt=p)\n"
        "r = RoutedEngine([_M()], planned_members=2, shared_inflight_dir=sys.argv[1], front_index=0, front_count=2)\n"
        "held = [r.stream(f'p-{i}') for i in range(6)]\n"
        "for it in held:\n"
        "    next(it)\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "os._exit(9)\n"
    )
    env = {**os.environ, "ENABLE_DISTRIBUTED_INIT": "0", "JAX_PLATFORMS": "cpu"}
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    f1 = None
    try:
        assert proc.stdout is not None
        out = proc.stdout.read()
        assert "ready" in out, f"child did not populate shared row: stderr={proc.stderr.read() if proc.stderr else ''}"

        # Wait until the child has transitioned to zombie (before reaping with proc.wait()).
        stat_path = f"/proc/{proc.pid}/stat"
        deadline = time.monotonic() + 5.0
        while os.path.exists(stat_path) and time.monotonic() < deadline:
            try:
                with open(stat_path, "rb") as fh:
                    raw = fh.read()
                rparen = raw.rfind(b")")
                if rparen != -1 and raw[rparen + 1 :].lstrip()[:1] in (b"Z", b"X", b"x"):
                    break
            except FileNotFoundError:
                break
            time.sleep(0.01)

        f1 = RoutedEngine(
            [_FakeMember(name="r0"), _FakeMember(name="r1")],
            planned_members=2,
            tie_breaker_offset=0,
            shared_inflight_dir=tmp_path,
            front_index=1,
            front_count=2,
        )

        # 1. While crashed front is still an unreaped zombie, f1 ignores its stale count of 6 on r0.
        zombie_streams = []
        try:
            picked = []
            for i in range(4):
                it, first = _open_stream(f1, f"zombie-check-{i}")
                zombie_streams.append(it)
                picked.append(first.member)
            assert picked.count("r0") == 2 and picked.count("r1") == 2
        finally:
            for it in zombie_streams:
                it.close()

        # 2. After reaping the dead front, f1 still balances evenly.
        assert proc.wait(timeout=5.0) == 9
        reaped_streams = []
        try:
            picked = []
            for i in range(4):
                it, first = _open_stream(f1, f"reaped-check-{i}")
                reaped_streams.append(it)
                picked.append(first.member)
            assert picked.count("r0") == 2 and picked.count("r1") == 2
        finally:
            for it in reaped_streams:
                it.close()

        # 3. Supervisor helper clears the dead row only when expected_pid matches.
        assert not clear_shared_inflight_row(tmp_path, 0, 2, expected_pid=proc.pid + 1)
        assert clear_shared_inflight_row(tmp_path, 0, 2, expected_pid=proc.pid)
        assert not clear_shared_inflight_row(tmp_path, 0, 2, expected_pid=proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        if f1 is not None:
            f1.close()


def test_close_during_unwinding_streams_is_safe_and_clears_shared_row(tmp_path):
    m0 = _FakeMember(name="r0")
    m1 = _FakeMember(name="r1")
    f0 = RoutedEngine(
        [m0, m1],
        planned_members=2,
        tie_breaker_offset=0,
        shared_inflight_dir=tmp_path,
        front_index=0,
        front_count=2,
    )
    f1 = RoutedEngine(
        [_FakeMember(name="r0"), _FakeMember(name="r1")],
        planned_members=2,
        tie_breaker_offset=0,
        shared_inflight_dir=tmp_path,
        front_index=1,
        front_count=2,
    )
    try:
        # Hold open stream() and chat(stream=True) iterators on f0's r0.
        f0.members[1].alive = False
        s_it, _ = _open_stream(f0, "stream-open")
        c_it = f0.chat([{"role": "user", "content": "chat-open"}], stream=True)
        next(c_it)
        gc_it = f0.chat([{"role": "user", "content": "chat-gc"}], stream=True)
        next(gc_it)

        # Close f0 while all three iterators are still live.
        f0.close()
        assert m0.closed and m1.closed

        # Unwinding via close(), exhaustion, and GC must not raise ValueError("mmap closed").
        s_it.close()
        assert list(c_it)  # drains remaining chunk and releases counter
        del gc_it
        gc.collect()

        # Surviving peer f1 sees f0's row cleared and balances cleanly 2/2.
        f1_streams = []
        try:
            picked = []
            for i in range(4):
                it, first = _open_stream(f1, f"after-close-{i}")
                f1_streams.append(it)
                picked.append(first.member)
            assert picked.count("r0") == 2 and picked.count("r1") == 2
        finally:
            for it in f1_streams:
                it.close()
    finally:
        f0.close()
        f1.close()


def test_shared_inflight_edge_cases_empty_dir_front_index_and_out_of_range_member_id(tmp_path, monkeypatch, capsys):
    # 1. Empty string shared_inflight_dir="" does not touch CWD or fail.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    r_empty = RoutedEngine(
        [_FakeMember(name="a"), _FakeMember(name="b")],
        shared_inflight_dir="",
        front_index=0,
        front_count=4,
    )
    try:
        out = r_empty.generate(["p0", "p1"])
        assert [o.member for o in out] == ["a", "b"]
        assert not (cwd / "router_inflight.bin").exists()
    finally:
        r_empty.close()

    # 2. Out-of-range front_index logs a warning and falls back safely to local routing.
    capsys.readouterr()
    r_bad_front = RoutedEngine(
        [_FakeMember(name="a"), _FakeMember(name="b")],
        shared_inflight_dir=tmp_path / "bad-front",
        front_index=8,
        front_count=8,
    )
    try:
        assert "outside [0, 8)" in capsys.readouterr().err
        assert not (tmp_path / "bad-front" / "router_inflight.bin").exists()
        out = r_bad_front.generate(["p0", "p1"])
        assert [o.member for o in out] == ["a", "b"]
    finally:
        r_bad_front.close()

    # 3. Negative member_id is rejected across __init__, add_member, and set_member_id.
    with pytest.raises(ValueError, match="member_id must be >= 0"):
        RoutedEngine([_FakeMember()], member_ids=[-1])
    r_neg = RoutedEngine([_FakeMember()])
    with pytest.raises(ValueError, match="member_id must be >= 0"):
        r_neg.add_member(_FakeMember(), member_id=-2)
    with pytest.raises(ValueError, match="member_id must be >= 0"):
        r_neg.set_member_id(0, -3)
    r_neg.close()

    # 4. member_id >= _SHARED_MAX_MEMBERS warns and falls back consistently to local
    # counts instead of crashing or skewing shared-vs-local loads across members.
    overflow_dir = tmp_path / "overflow"
    overflow_id = _SHARED_MAX_MEMBERS + 6
    capsys.readouterr()
    f0 = RoutedEngine(
        [_FakeMember(name="low"), _FakeMember(name="high")],
        member_ids=[0, overflow_id],
        tie_breaker_offset=0,
        shared_inflight_dir=overflow_dir,
        front_index=0,
        front_count=2,
    )
    f1 = RoutedEngine(
        [_FakeMember(name="low"), _FakeMember(name="high")],
        member_ids=[0, overflow_id],
        tie_breaker_offset=0,
        shared_inflight_dir=overflow_dir,
        front_index=1,
        front_count=2,
    )
    f0_streams = []
    f1_streams = []
    try:
        assert "exceeds shared inflight table capacity" in capsys.readouterr().err
        # Put 4 active streams on both members on f0 (2 on low, 2 on high).
        for i in range(4):
            it, _ = _open_stream(f0, f"f0-overflow-{i}")
            f0_streams.append(it)
        # f1 must compare both members on the same local scale (2/2 split), NOT
        # mix shared count on member 0 against local 0 on overflow_id (which would
        # dump all 4 onto "high").
        picked = []
        for i in range(4):
            it, first = _open_stream(f1, f"f1-overflow-{i}")
            f1_streams.append(it)
            picked.append(first.member)
        assert picked.count("low") == 2 and picked.count("high") == 2
    finally:
        for it in (*f0_streams, *f1_streams):
            it.close()
        f0.close()
        f1.close()

    # 5. Sparse member_ids with planned_members=None break ties canonically,
    # independent of attach order (no modulo len(members) collision).
    r_order_a = RoutedEngine([_FakeMember(name="r0"), _FakeMember(name="r2")], member_ids=[0, 2], tie_breaker_offset=0)
    r_order_b = RoutedEngine([_FakeMember(name="r2"), _FakeMember(name="r0")], member_ids=[2, 0], tie_breaker_offset=0)
    try:
        assert r_order_a.generate(["p0"])[0].member == "r0"
        assert r_order_b.generate(["p0"])[0].member == "r0"
    finally:
        r_order_a.close()
        r_order_b.close()


def test_pid_liveness_permission_error_non_procfs_and_live_peer_process(tmp_path, monkeypatch):
    # 1. PermissionError from os.kill(pid, 0) means the PID belongs to another user (recycled),
    # so _is_pid_alive must return False rather than resurrecting stale counts.
    fake_pid = os.getpid() + 99999

    def _raise_eperm(_pid: int, _sig: int) -> None:
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(os, "kill", _raise_eperm)
    assert _is_pid_alive(fake_pid) is False

    # 2. Without procfs (_HAS_PROC_SELF_STAT=False), os.kill succeeding must return True
    # without failing on missing /proc/<pid>/stat.
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(routed_engine_mod, "_HAS_PROC_SELF_STAT", False)
    assert _is_pid_alive(fake_pid) is True

    # Restore os.kill and _HAS_PROC_SELF_STAT for a real cross-process live peer check.
    monkeypatch.undo()

    child_code = (
        "import sys\n"
        "from types import SimpleNamespace\n"
        "from easydel.inference.esurge.distributed.routed_engine import RoutedEngine\n"
        "class _M:\n"
        "    esurge_name = 'r0'; max_model_len = 128; max_num_seqs = 4; page_size = 16; alive = True\n"
        "    def stream(self, p, **_kw):\n"
        "        yield SimpleNamespace(prompt=p)\n"
        "        yield SimpleNamespace(prompt=p)\n"
        "r = RoutedEngine([_M()], planned_members=2, shared_inflight_dir=sys.argv[1], front_index=0, front_count=2)\n"
        "held = [r.stream(f'p-{i}') for i in range(4)]\n"
        "for it in held:\n"
        "    next(it)\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stdin.readline()\n"
        "for it in held:\n"
        "    it.close()\n"
        "r.close()\n"
    )
    env = {
        **os.environ,
        "ENABLE_DISTRIBUTED_INIT": "0",
        "JAX_PLATFORMS": "cpu",
        "PJRT_DEVICE": "CPU",
        "TPU_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "",
    }
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    f1 = None
    f1_streams = []
    try:
        assert proc.stdout is not None and proc.stdin is not None
        line = proc.stdout.readline()
        assert "ready" in line, f"child failed: {proc.stderr.read() if proc.stderr else ''}"
        assert _is_pid_alive(proc.pid) is True

        f1 = RoutedEngine(
            [_FakeMember(name="r0"), _FakeMember(name="r1")],
            planned_members=2,
            tie_breaker_offset=0,
            shared_inflight_dir=tmp_path,
            front_index=1,
            front_count=2,
        )
        # Live peer child holds 4 inflight on r0 -> f1 must route its next 4 streams to r1.
        for i in range(4):
            it, first = _open_stream(f1, f"live-peer-{i}")
            f1_streams.append(it)
            assert first.member == "r1"
        proc.stdin.write("done\n")
        proc.stdin.flush()
        assert proc.wait(timeout=5.0) == 0
    finally:
        for it in f1_streams:
            it.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        if f1 is not None:
            f1.close()


def test_pid_liveness_ttl_cache_reuses_within_ttl_and_refreshes_on_expiry_or_pid_change(tmp_path, monkeypatch):
    f0 = RoutedEngine(
        [_FakeMember(name="r0"), _FakeMember(name="r1")],
        planned_members=2,
        tie_breaker_offset=0,
        shared_inflight_dir=tmp_path,
        front_index=0,
        front_count=2,
    )
    f1 = RoutedEngine(
        [_FakeMember(name="r0"), _FakeMember(name="r1")],
        planned_members=2,
        tie_breaker_offset=0,
        shared_inflight_dir=tmp_path,
        front_index=1,
        front_count=2,
    )
    held_streams = []
    try:
        # Hold 2 active streams on f0's r0, then stamp synthetic peer PIDs into row 0.
        f0.members[1].alive = False
        for i in range(2):
            it, _ = _open_stream(f0, f"ttl-seed-{i}")
            held_streams.append(it)

        assert f0._shared_buf is not None
        pid_a = os.getpid() + 50001
        pid_b = os.getpid() + 50002
        struct.pack_into("<q", f0._shared_buf, 0, pid_a)

        calls: list[int] = []
        alive_state = {pid_a: True, pid_b: True}

        def _fake_is_pid_alive(pid: int) -> bool:
            calls.append(pid)
            return alive_state.get(pid, False)

        now_val = [100.0]
        monkeypatch.setattr(routed_engine_mod, "_is_pid_alive", _fake_is_pid_alive)
        monkeypatch.setattr(routed_engine_mod.time, "monotonic", lambda: now_val[0])

        # Multiple selections within the TTL only probe pid_a once.
        for _ in range(5):
            assert f1.select_member("ttl-probe") == 1
        assert calls == [pid_a]

        # Advancing past the TTL triggers a refresh; if pid_a died, f1 immediately stops counting it.
        alive_state[pid_a] = False
        now_val[0] += routed_engine_mod._PID_ALIVE_TTL_S + 0.05
        assert f1.select_member("ttl-probe") == 0
        assert calls == [pid_a, pid_a]

        # A restarted front writing a new PID bypasses pid_a's cache immediately (even without TTL expiry).
        struct.pack_into("<q", f0._shared_buf, 0, pid_b)
        assert f1.select_member("ttl-probe") == 1
        assert calls == [pid_a, pid_a, pid_b]
    finally:
        for it in held_streams:
            it.close()
        f0.close()
        f1.close()


def test_duplicate_member_id_rejected_and_implicit_add_member_picks_unused_id(tmp_path):
    with pytest.raises(ValueError, match="duplicate member_id"):
        RoutedEngine([_FakeMember(name="a"), _FakeMember(name="b")], member_ids=[0, 0])

    r = RoutedEngine(
        [_FakeMember(name="r1")],
        planned_members=3,
        member_ids=[1],
        shared_inflight_dir=tmp_path,
        front_index=0,
        front_count=2,
    )
    try:
        with pytest.raises(ValueError, match="duplicate member_id"):
            r.add_member(_FakeMember(name="dup"), member_id=1)

        # Implicit add_member when slot == 1 collides with existing member_id == 1 picks unused id 0.
        slot = r.add_member(_FakeMember(name="r0"))
        assert slot == 1
        assert r._slot_counters[slot].member_id == 0

        with pytest.raises(ValueError, match="duplicate member_id"):
            r.set_member_id(slot, 1)

        # Rebinding to an unused id succeeds cleanly.
        r.set_member_id(slot, 2)
        assert r._slot_counters[slot].member_id == 2
    finally:
        r.close()


def test_allocate_shared_file_skips_fallocate_when_sized_and_mmap_value_error_falls_back(tmp_path, monkeypatch):
    # 1. Already-sized file must not invoke posix_fallocate again.
    path = tmp_path / "already_sized.bin"
    size = 2 * _SHARED_ROW_BYTES
    path.write_bytes(b"\x00" * size)

    fallocate_calls = []

    def _fail_fallocate(*args):
        fallocate_calls.append(args)
        raise AssertionError("posix_fallocate should not be called when st_size >= size")

    monkeypatch.setattr(os, "posix_fallocate", _fail_fallocate, raising=False)
    fd = os.open(path, os.O_RDWR)
    try:
        _allocate_shared_file(fd, size)
    finally:
        os.close(fd)
    assert fallocate_calls == []
    monkeypatch.undo()

    # 2. ValueError from mmap.mmap falls back safely in both RoutedEngine.__init__ and clear_shared_inflight_row.
    shared_dir = tmp_path / "mmap-valerr"
    shared_dir.mkdir()
    (shared_dir / "router_inflight.bin").write_bytes(b"\x00" * size)

    def _raise_value_error(*_args, **_kwargs):
        raise ValueError("mmap length is greater than file size")

    monkeypatch.setattr(mmap, "mmap", _raise_value_error)
    r = RoutedEngine(
        [_FakeMember(name="a"), _FakeMember(name="b")],
        shared_inflight_dir=shared_dir,
        front_index=0,
        front_count=2,
    )
    try:
        assert r._shared_buf is None
        assert [o.member for o in r.generate(["p0", "p1"])] == ["a", "b"]
    finally:
        r.close()
    assert clear_shared_inflight_row(shared_dir, 0, 2) is False


def test_replica_attacher_legacy_fallback_cleans_up_on_set_member_id_failure():
    # 1. Legacy router_factory (no member_ids kwarg): if set_member_id fails, router.close() runs
    #    (releasing router resources + member handle) and attacher.router stays None.
    class _LegacyRouterInitFail:
        def __init__(self, members, *, planned_members=None):
            self.members = list(members)
            self.closed = False

        def set_member_id(self, _slot: int, _mid: int) -> None:
            raise RuntimeError("boom on init set_member_id")

        def close(self) -> None:
            self.closed = True
            for m in self.members:
                m.close()

    created_routers: list[_LegacyRouterInitFail] = []
    handles_init: list[_FakeMember] = []

    def _connect_init(addr: str):
        h = _FakeMember(name=addr)
        handles_init.append(h)
        return h

    def _factory_init(members, *, planned_members=None):
        router = _LegacyRouterInitFail(members, planned_members=planned_members)
        created_routers.append(router)
        return router

    attacher_init = ReplicaAttacher(
        ["r0"],
        connect=_connect_init,
        log=lambda _m: None,
        router_factory=_factory_init,
    )
    with pytest.raises(RuntimeError, match="boom on init set_member_id"):
        attacher_init.attach_round()
    assert attacher_init.router is None
    assert len(created_routers) == 1 and created_routers[0].closed
    assert len(handles_init) == 1 and handles_init[0].closed

    # 2. Legacy add_member (no member_id kwarg): if set_member_id fails after add_member succeeded,
    #    handle stays open inside router.members and _slots records it so retry does not duplicate it.
    class _LegacyRouterAddFail:
        def __init__(self, members, *, planned_members=None):
            self.members = list(members)
            self.fail_next_set = False

        def add_member(self, member) -> int:
            self.members.append(member)
            return len(self.members) - 1

        def set_member_id(self, _slot: int, _mid: int) -> None:
            if self.fail_next_set:
                self.fail_next_set = False
                raise RuntimeError("boom on add set_member_id")

        def close(self) -> None:
            for m in self.members:
                m.close()

    up = {"r0": True, "r1": False}
    handles_add: dict[str, _FakeMember] = {}

    def _connect_add(addr: str):
        if not up[addr]:
            raise RequestPlaneUnavailable(addr)
        h = _FakeMember(name=addr)
        handles_add[addr] = h
        return h

    attacher_add = ReplicaAttacher(
        ["r0", "r1"],
        connect=_connect_add,
        log=lambda _m: None,
        router_factory=lambda members, *, planned_members=None: _LegacyRouterAddFail(
            members, planned_members=planned_members
        ),
    )
    router_add = attacher_add.wait_for_first()
    up["r1"] = True
    router_add.fail_next_set = True
    with pytest.raises(RuntimeError, match="boom on add set_member_id"):
        attacher_add.attach_round()
    # Handle r1 was already added to router.members, so it must NOT be closed and must be tracked in _slots.
    assert len(router_add.members) == 2
    assert handles_add["r1"].closed is False
    assert attacher_add._slots[1] == 1
    # Subsequent attach_round sees r1 alive in its slot and does NOT duplicate-add it.
    assert attacher_add.attach_round() == 0
    assert len(router_add.members) == 2
