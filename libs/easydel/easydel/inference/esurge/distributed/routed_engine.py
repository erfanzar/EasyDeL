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

"""A data-parallel front over independent eSurge replicas.

:class:`RoutedEngine` presents the engine-adapter surface while fanning
requests across N members — local ``eSurge`` instances and/or
:class:`RemoteEngineHandle` clients — with the routing policy lifted from
the in-core DP routing: **prefix affinity first, then join-shortest-queue**.
Affinity is tracked router-side in a :class:`PrefixAffinityMap` (LRU of
block-aligned prompt-prefix hashes → member) instead of synchronously
probing every replica's prefix cache; the JSQ signal is
``pending_prefill_tokens``, read directly from local members and from the
heartbeat-piggybacked ``QueueStats`` for remote ones.

Health is fail-fast per member: a dead replica (heartbeat loss, shutdown)
is excluded from routing and its in-flight requests surface as aborted /
raised errors on their callers; there is no migration or retry. The other
members keep serving.
"""

from __future__ import annotations

import errno
import hashlib
import inspect
import mmap
import os
import struct
import threading
import time
import typing
import weakref
from collections import OrderedDict

from ..logger import logger
from .request_plane import RequestPlaneError

if typing.TYPE_CHECKING:
    from easydel.inference.sampling_params import SamplingParams

    from ..esurge_engine import RequestOutput

#: Fallback affinity block width (characters) when a member reports no page size.
_DEFAULT_BLOCK_CHARS = 512
#: Longest prefix (in blocks) tracked per prompt.
_MAX_BLOCKS_PER_PROMPT = 32
#: Maximum canonical replica slots tracked per front in the shared mmap table.
_SHARED_MAX_MEMBERS = 64
#: Leading bytes per front row holding the owning front PID (int64, 0 when unowned/closed).
_SHARED_HEADER_BYTES = 8
#: Bytes of per-member int32 inflight counts in one front row.
_SHARED_COUNTS_BYTES = _SHARED_MAX_MEMBERS * 4
#: Total bytes per front row in ``router_inflight.bin`` (8-byte aligned).
_SHARED_ROW_BYTES = _SHARED_HEADER_BYTES + _SHARED_COUNTS_BYTES
#: Whether Linux procfs ``/proc/<pid>/stat`` is available for zombie detection.
_HAS_PROC_SELF_STAT = os.path.exists("/proc/self/stat")
#: How long a peer front PID's liveness result is cached per router before re-probing.
_PID_ALIVE_TTL_S = 0.25


def _allocate_shared_file(fd: int, size: int) -> None:
    """Ensure ``fd`` has at least ``size`` backing bytes (avoiding sparse-file SIGBUS when supported)."""
    if os.fstat(fd).st_size >= size:
        return
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is not None:
        try:
            fallocate(fd, 0, size)
            return
        except OSError as exc:
            if exc.errno not in (errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS):
                raise
    if os.fstat(fd).st_size < size:
        os.ftruncate(fd, size)


def _clear_shared_row_buf(buf: mmap.mmap, front_index: int, *, expected_pid: int | None = None) -> bool:
    """Clear owner PID first, then zero every member counter for ``front_index``."""
    row_off = front_index * _SHARED_ROW_BYTES
    if expected_pid is not None:
        owner_pid = struct.unpack_from("<q", buf, row_off)[0]
        if owner_pid != int(expected_pid):
            return False
    struct.pack_into("<q", buf, row_off, 0)
    counts_off = row_off + _SHARED_HEADER_BYTES
    buf[counts_off : counts_off + _SHARED_COUNTS_BYTES] = b"\x00" * _SHARED_COUNTS_BYTES
    return True


def _close_shared_buf(buf: mmap.mmap, front_index: int, owner_pid: int | None = None) -> None:
    """Best-effort clear + close for a front's shared inflight mmap buffer."""
    try:
        if not buf.closed:
            _clear_shared_row_buf(buf, front_index, expected_pid=owner_pid)
            buf.close()
    except Exception:
        logger.debug("Shared inflight table cleanup failed", exc_info=True)


def _is_pid_alive(pid: int) -> bool:
    """Return whether ``pid`` is a live, non-zombie same-user process."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # router_inflight.bin is 0o600 in a 0o700 dir; peer fronts always share our UID.
        # PermissionError means the original front exited and its PID was recycled by another user.
        return False
    except OSError:
        return False
    # On Linux, an unreaped crashed child (zombie) still responds to kill(pid, 0).
    if _HAS_PROC_SELF_STAT:
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read()
            rparen = stat.rfind(b")")
            if rparen != -1:
                rest = stat[rparen + 1 :].lstrip()
                if rest and rest[:1] in (b"Z", b"X", b"x"):
                    return False
        except FileNotFoundError:
            return False
        except OSError:
            pass
    return True


def clear_shared_inflight_row(
    shared_inflight_dir: str | os.PathLike[str] | None,
    front_index: int,
    front_count: int,
    *,
    expected_pid: int | None = None,
) -> bool:
    """Zero one front's row in ``router_inflight.bin`` (e.g. when a supervisor reaps a crashed front).

    When ``expected_pid`` is supplied, the row is only cleared if its recorded
    owner PID matches, avoiding any race with a newly restarted front on the
    same ``front_index``.
    """
    shared_dir = "" if shared_inflight_dir is None else os.fspath(shared_inflight_dir).strip()
    idx = int(front_index)
    count = max(1, int(front_count))
    if not shared_dir or count <= 1 or not (0 <= idx < count):
        return False
    path = os.path.join(shared_dir, "router_inflight.bin")
    size = count * _SHARED_ROW_BYTES
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        if os.fstat(fd).st_size < size:
            return False
        with mmap.mmap(fd, size) as buf:
            return _clear_shared_row_buf(buf, idx, expected_pid=expected_pid)
    except (OSError, ValueError):
        return False
    finally:
        os.close(fd)


class PrefixAffinityMap:
    """LRU map from block-aligned prompt-prefix hashes to member index.

    Routing wants "the replica that most recently saw this prefix" without
    a synchronous cache probe per replica. Prompts are cut into fixed-width
    character blocks (a router-side proxy for page-aligned token blocks);
    each cumulative prefix hash maps to the member that last served it, and
    lookups return the member holding the longest matching prefix.

    Args:
        block_chars: Prefix block width in characters.
        capacity: Maximum retained prefix hashes (LRU-evicted).
    """

    def __init__(self, *, block_chars: int = _DEFAULT_BLOCK_CHARS, capacity: int = 16384) -> None:
        self._block_chars = max(1, int(block_chars))
        self._capacity = int(capacity)
        self._lock = threading.Lock()
        self._map: OrderedDict[str, int] = OrderedDict()

    def _prefix_hashes(self, prompt: str) -> list[str]:
        hashes = []
        hasher = hashlib.sha256()
        for start in range(0, min(len(prompt), self._block_chars * _MAX_BLOCKS_PER_PROMPT), self._block_chars):
            block = prompt[start : start + self._block_chars]
            if len(block) < self._block_chars and start > 0:
                break  # only full blocks participate after the first
            hasher.update(block.encode())
            hashes.append(hasher.hexdigest()[:24])
        return hashes

    def record(self, prompt: str, member_index: int) -> None:
        """Remember that ``member_index`` served this prompt's prefixes."""
        with self._lock:
            for prefix_hash in self._prefix_hashes(prompt):
                self._map.pop(prefix_hash, None)
                self._map[prefix_hash] = member_index
            while len(self._map) > self._capacity:
                self._map.popitem(last=False)

    def lookup(self, prompt: str) -> tuple[int | None, int]:
        """Find the member holding this prompt's longest known prefix.

        Args:
            prompt: The incoming prompt text.

        Returns:
            ``(member_index | None, matched_blocks)``.
        """
        best: int | None = None
        matched = 0
        with self._lock:
            for depth, prefix_hash in enumerate(self._prefix_hashes(prompt), start=1):
                member = self._map.get(prefix_hash)
                if member is None:
                    break
                best = member
                matched = depth
        return best, matched

    def drop_member(self, member_index: int) -> None:
        """Forget every prefix pinned to a (dead) member."""
        with self._lock:
            for key in [key for key, value in self._map.items() if value == member_index]:
                self._map.pop(key, None)


class _SlotCounter:
    """Per-member-instance in-flight request counter.

    Pinned to the member instance rather than its slot index so that
    :meth:`RoutedEngine.replace_member` gives the replacement a fresh counter
    and unwinding requests from the old member never decrement the replacement
    or clobber its shared mmap count.
    """

    __slots__ = ("active", "count", "member_id")

    def __init__(self, member_id: int = 0) -> None:
        self.count = 0
        self.member_id = int(member_id)
        self.active = True


class _ReleasedIterator:
    """Iterator wrapper releasing a router in-flight slot on exhaustion, close, or GC.

    Unlike a generator function, closing or discarding this wrapper before the
    first ``next()`` still closes the wrapped iterator and releases the slot.
    """

    def __init__(self, iterator: typing.Iterator, release: typing.Callable[[], None]) -> None:
        self._iterator = iterator
        self._release = release
        self._closed = False

    def __iter__(self) -> _ReleasedIterator:
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            inner_close = getattr(self._iterator, "close", None)
            if inner_close is not None:
                inner_close()
        finally:
            self._release()

    def __enter__(self) -> _ReleasedIterator:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _member_alive(member) -> bool:
    """Read public liveness metadata, assuming unknown engine-shaped members are alive."""
    alive = getattr(member, "alive", None)
    return True if alive is None else bool(alive)


def _member_pending_prefill_tokens(member) -> int:
    """JSQ cost signal for a member (remote stats or a local engine-core count)."""
    remote = getattr(member, "pending_prefill_tokens", None)
    if remote is not None:
        return int(remote)
    core = getattr(member, "core", None)
    if core is None:
        return 0
    try:
        return int(core.pending_prefill_tokens)
    except Exception:
        return 0


class RoutedEngine:
    """Prefix-affinity + JSQ router over engine-shaped members.

    Args:
        members: Engines to route across — any mix of local ``eSurge``
            instances and :class:`RemoteEngineHandle` clients (anything
            exposing the adapter surface).
        affinity_block_chars: Prefix block width for the affinity map;
            defaults to a page-size-derived estimate when a member reports
            one (≈4 characters per token), else ``512`` characters.
        affinity_max_imbalance: How many more in-flight requests (router-local,
            or summed across peer fronts when ``shared_inflight_dir`` is active)
            the prefix-pinned member may hold than the least-loaded live member
            before affinity yields to join-shortest-queue. A prefix shared by a
            burst (one system prompt, one benchmark prompt) otherwise lands
            every request on one member while the others idle.
        planned_members: Optional total replica count expected for the
            deployment (used to size ``max_num_seqs`` and tie-breaking while
            late replicas are still attaching).
        tie_breaker_offset: Per-front rotation offset applied to canonical
            ``member_id`` when breaking equal-load ties.
        shared_inflight_dir: Optional directory for ``router_inflight.bin``
            used to sum live in-flight counts across ``front_count`` host fronts.
        front_index: Zero-based index of this front in ``[0, front_count)``.
        front_count: Total number of peer fronts sharing ``shared_inflight_dir``.
        member_ids: Optional canonical replica index per initial member
            (defaults to ``0..len(members)-1``).
    """

    def __init__(
        self,
        members: list,
        *,
        affinity_block_chars: int | None = None,
        affinity_max_imbalance: int = 4,
        planned_members: int | None = None,
        tie_breaker_offset: int = 0,
        shared_inflight_dir: str | os.PathLike[str] | None = None,
        front_index: int = 0,
        front_count: int = 1,
        member_ids: typing.Sequence[int] | None = None,
    ) -> None:
        if not members:
            raise ValueError("RoutedEngine needs at least one member")
        if int(affinity_max_imbalance) < 0:
            raise ValueError(f"affinity_max_imbalance must be >= 0, got {affinity_max_imbalance}")
        if planned_members is not None and int(planned_members) < len(members):
            raise ValueError(f"planned_members={planned_members} is below the {len(members)} members given")
        self.members = list(members)
        if member_ids is None:
            resolved_member_ids = list(range(len(self.members)))
        else:
            resolved_member_ids = [int(mid) for mid in member_ids]
            if len(resolved_member_ids) != len(self.members):
                raise ValueError(
                    f"member_ids length ({len(resolved_member_ids)}) must match members length ({len(self.members)})"
                )
            seen_ids: set[int] = set()
            for mid in resolved_member_ids:
                if mid < 0:
                    raise ValueError(f"member_id must be >= 0, got {mid}")
                if mid in seen_ids:
                    raise ValueError(f"duplicate member_id: {mid}")
                seen_ids.add(mid)
        self.planned_members = None if planned_members is None else int(planned_members)
        self.affinity_max_imbalance = int(affinity_max_imbalance)
        self._tie_breaker_offset = int(tie_breaker_offset)
        self._front_index = int(front_index)
        self._front_count = max(1, int(front_count))
        self._shared_buf: mmap.mmap | None = None
        self._shared_owner_pid: int | None = None
        self._shared_finalizer: weakref.finalize | None = None
        self._pid_alive_cache: dict[int, tuple[bool, float]] = {}
        self._warned_shared_overflow = False
        shared_dir = "" if shared_inflight_dir is None else os.fspath(shared_inflight_dir).strip()
        if shared_dir and self._front_count > 1:
            if not (0 <= self._front_index < self._front_count):
                logger.warning(
                    "RoutedEngine front_index=%d is outside [0, %d); disabling shared inflight table",
                    self._front_index,
                    self._front_count,
                )
            else:
                try:
                    os.makedirs(shared_dir, exist_ok=True)
                    path = os.path.join(shared_dir, "router_inflight.bin")
                    size = self._front_count * _SHARED_ROW_BYTES
                    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                    try:
                        _allocate_shared_file(fd, size)
                        self._shared_buf = mmap.mmap(fd, size)
                    finally:
                        os.close(fd)
                    owner_pid = os.getpid()
                    _clear_shared_row_buf(self._shared_buf, self._front_index)
                    struct.pack_into("<q", self._shared_buf, self._front_index * _SHARED_ROW_BYTES, owner_pid)
                    self._shared_owner_pid = owner_pid
                    self._shared_finalizer = weakref.finalize(
                        self, _close_shared_buf, self._shared_buf, self._front_index, owner_pid
                    )
                except (OSError, ValueError):
                    logger.debug(
                        "Could not open shared router inflight table; falling back to local counts", exc_info=True
                    )
                    if self._shared_buf is not None:
                        _close_shared_buf(self._shared_buf, self._front_index, self._shared_owner_pid)
                    self._shared_buf = None
                    self._shared_owner_pid = None
                    self._shared_finalizer = None
        if affinity_block_chars is None:
            page_size = next((int(getattr(m, "page_size", 0) or 0) for m in self.members), 0)
            affinity_block_chars = page_size * 4 if page_size > 0 else _DEFAULT_BLOCK_CHARS
        self.affinity = PrefixAffinityMap(block_chars=affinity_block_chars)
        self._route_lock = threading.Lock()
        # Per-member-instance in-flight counts (one _SlotCounter per member):
        # the JSQ primary signal. Remote QueueStats ride 1s heartbeats, so a
        # burst arriving between beats would otherwise dump entirely on the
        # lowest-indexed member. When multiple host fronts share a mmap table,
        # each front writes its own row and sums all live fronts lock-free.
        self._slot_counters: list[_SlotCounter] = [_SlotCounter(member_id=mid) for mid in resolved_member_ids]
        for mid in resolved_member_ids:
            self._warn_shared_overflow_locked(mid)

    def _warn_shared_overflow_locked(self, member_id: int) -> None:
        if self._shared_buf is not None and member_id >= _SHARED_MAX_MEMBERS and not self._warned_shared_overflow:
            self._warned_shared_overflow = True
            logger.warning(
                "RoutedEngine member_id=%d exceeds shared inflight table capacity (%d); "
                "falling back to router-local counts while out-of-range members are live",
                member_id,
                _SHARED_MAX_MEMBERS,
            )

    def _write_shared_member_locked(self, member_id: int) -> None:
        buf = self._shared_buf
        if buf is None or not (0 <= member_id < _SHARED_MAX_MEMBERS):
            return
        local_total = sum(c.count for c in self._slot_counters if c.member_id == member_id)
        offset = self._front_index * _SHARED_ROW_BYTES + _SHARED_HEADER_BYTES + member_id * 4
        struct.pack_into("<i", buf, offset, local_total)

    def _sync_shared_locked(self, counter: _SlotCounter) -> None:
        if not counter.active:
            return
        self._write_shared_member_locked(counter.member_id)

    def _is_peer_pid_alive_locked(self, owner_pid: int, *, now: float) -> bool:
        cached = self._pid_alive_cache.get(owner_pid)
        if cached is not None:
            alive, checked_at = cached
            if now - checked_at < _PID_ALIVE_TTL_S:
                return alive
        alive = _is_pid_alive(owner_pid)
        if len(self._pid_alive_cache) >= max(32, self._front_count * 4):
            self._pid_alive_cache.clear()
        self._pid_alive_cache[owner_pid] = (alive, now)
        return alive

    def _total_inflight_locked(self, slot: int, *, now: float | None = None) -> int:
        counter = self._slot_counters[slot]
        buf = self._shared_buf
        mid = counter.member_id
        if buf is None or not (0 <= mid < _SHARED_MAX_MEMBERS):
            return counter.count
        total = sum(c.count for c in self._slot_counters if c.member_id == mid)
        check_now = time.monotonic() if now is None else now
        for f in range(self._front_count):
            if f == self._front_index:
                continue
            row_off = f * _SHARED_ROW_BYTES
            owner_pid = struct.unpack_from("<q", buf, row_off)[0]
            if owner_pid <= 0:
                continue
            val = struct.unpack_from("<i", buf, row_off + _SHARED_HEADER_BYTES + mid * 4)[0]
            if val <= 0:
                continue
            if self._is_peer_pid_alive_locked(owner_pid, now=check_now):
                total += val
        return total

    def set_member_id(self, slot: int, member_id: int) -> None:
        """Bind a member slot to its canonical replica index for cross-front accounting."""
        mid = int(member_id)
        if mid < 0:
            raise ValueError(f"member_id must be >= 0, got {member_id}")
        with self._route_lock:
            counter = self._slot_counters[slot]
            old_id = counter.member_id
            if old_id == mid:
                return
            if any(i != slot and c.member_id == mid for i, c in enumerate(self._slot_counters)):
                raise ValueError(f"duplicate member_id: {mid}")
            counter.member_id = mid
            self._warn_shared_overflow_locked(mid)
            self._write_shared_member_locked(old_id)
            self._write_shared_member_locked(mid)

    @property
    def _inflight(self) -> list[int]:
        """Snapshot of router-local in-flight request counts per slot."""
        with self._route_lock:
            return [counter.count for counter in self._slot_counters]

    def add_member(self, member, *, member_id: int | None = None) -> int:
        """Attach a member that came up after the router started serving (a late replica).

        Returns:
            The new member's index.
        """
        with self._route_lock:
            slot = len(self.members)
            used_ids = {c.member_id for c in self._slot_counters}
            if member_id is None:
                if slot not in used_ids:
                    mid = slot
                else:
                    mid = 0
                    while mid in used_ids:
                        mid += 1
            else:
                mid = int(member_id)
                if mid < 0:
                    raise ValueError(f"member_id must be >= 0, got {member_id}")
                if mid in used_ids:
                    raise ValueError(f"duplicate member_id: {mid}")
            # In-flight first: a reader that sees the member under the lock also sees its counter.
            counter = _SlotCounter(member_id=mid)
            self._slot_counters.append(counter)
            self.members.append(member)
            self._warn_shared_overflow_locked(mid)
            self._sync_shared_locked(counter)
            return slot

    def replace_member(self, index: int, member) -> None:
        """Put a fresh member in a dead member's slot (a replica that came back on the same address)."""
        with self._route_lock:
            old_counter = self._slot_counters[index]
            old_counter.active = False
            mid = old_counter.member_id
            self.members[index] = member
            counter = _SlotCounter(member_id=mid)
            self._slot_counters[index] = counter
            self._sync_shared_locked(counter)
        self.affinity.drop_member(index)

    def _pick_member_locked(self, pinned: int | None) -> tuple[int | None, list[int]]:
        """Pick the best live member index under ``self._route_lock``.

        Returns:
            ``(chosen_index | None, dead_indices)``; ``chosen_index`` is ``None``
            when no member is alive.
        """
        alive: list[int] = []
        dead: list[int] = []
        for index, member in enumerate(self.members):
            if _member_alive(member):
                alive.append(index)
            else:
                dead.append(index)
        if not alive:
            return None, dead
        use_shared = self._shared_buf is not None and all(
            0 <= self._slot_counters[index].member_id < _SHARED_MAX_MEMBERS for index in alive
        )
        if use_shared:
            now = time.monotonic()
            loads = {index: self._total_inflight_locked(index, now=now) for index in alive}
        else:
            loads = {index: self._slot_counters[index].count for index in alive}
        if pinned is not None and pinned in alive:
            least = min(loads[index] for index in alive)
            if loads[pinned] - least <= self.affinity_max_imbalance:
                return pinned, dead
        max_mid = max(self._slot_counters[index].member_id for index in alive)
        span = max(1, self.planned_members or 0, len(self.members), max_mid + 1)
        chosen = min(
            alive,
            key=lambda index: (
                loads[index],
                _member_pending_prefill_tokens(self.members[index]),
                (self._slot_counters[index].member_id - self._tie_breaker_offset) % span,
                self._slot_counters[index].member_id,
            ),
        )
        return chosen, dead

    def _alive_indices(self) -> list[int]:
        with self._route_lock:
            alive: list[int] = []
            dead: list[int] = []
            for index, member in enumerate(self.members):
                if _member_alive(member):
                    alive.append(index)
                else:
                    dead.append(index)
        for index in dead:
            self.affinity.drop_member(index)
        if not alive:
            raise RequestPlaneError("no live replica: every routed engine member is down")
        return alive

    def _acquire(self, member_index: int) -> None:
        with self._route_lock:
            counter = self._slot_counters[member_index]
            counter.count += 1
            self._sync_shared_locked(counter)

    def _release(self, member_index: int) -> None:
        with self._route_lock:
            counter = self._slot_counters[member_index]
            counter.count = max(0, counter.count - 1)
            self._sync_shared_locked(counter)

    def _release_counter(self, counter: _SlotCounter) -> None:
        with self._route_lock:
            counter.count = max(0, counter.count - 1)
            self._sync_shared_locked(counter)

    def _select_and_acquire(self, prompt: str) -> tuple[int, typing.Any, _SlotCounter]:
        """Atomically select a live member, increment its counter, and record prefix affinity."""
        pinned, _depth = self.affinity.lookup(prompt)
        with self._route_lock:
            member_index, dead = self._pick_member_locked(pinned)
            if member_index is not None:
                member = self.members[member_index]
                counter = self._slot_counters[member_index]
                counter.count += 1
                self._sync_shared_locked(counter)
        for index in dead:
            self.affinity.drop_member(index)
        if member_index is None:
            raise RequestPlaneError("no live replica: every routed engine member is down")
        self.affinity.record(prompt, member_index)
        return member_index, member, counter

    def select_member(self, prompt: str) -> int:
        """Pick the member for one prompt: bounded prefix affinity, then JSQ.

        The prefix-pinned member wins while its in-flight count (router-local,
        or summed across live peer fronts when ``shared_inflight_dir`` is
        active) is at most ``affinity_max_imbalance`` above the least-loaded
        live member's; beyond that the request joins the shortest queue (the
        caller then re-pins the prefix there). The queue key is (in-flight
        count, owner-reported pending prefill tokens, rotated canonical
        ``member_id``, ``member_id``): the in-flight count is instantaneous,
        and the remote signal covers load from other clients.

        Args:
            prompt: The incoming prompt text.

        Returns:
            Index into :attr:`members`.
        """
        pinned, _depth = self.affinity.lookup(prompt)
        with self._route_lock:
            member_index, dead = self._pick_member_locked(pinned)
        for index in dead:
            self.affinity.drop_member(index)
        if member_index is None:
            raise RequestPlaneError("no live replica: every routed engine member is down")
        return member_index

    @property
    def esurge_name(self) -> str:
        """The replica group's display name (first member's)."""
        return str(getattr(self.members[0], "esurge_name", "routed-esurge"))

    @property
    def max_model_len(self) -> int:
        """Group context limit (min across members)."""
        return min(int(member.max_model_len) for member in self.members)

    @property
    def max_num_seqs(self) -> int:
        """Aggregate concurrency limit: the sum across members, or the planned total while members attach late.

        A server sizes its generation slots from this once, at start-up; a router that starts with the first ready
        replica of ``planned_members`` reports the full deployment so late replicas are not capped away.
        """
        attached = sum(int(member.max_num_seqs) for member in self.members)
        if self.planned_members is None or len(self.members) >= self.planned_members:
            return attached
        return max(attached, max(int(member.max_num_seqs) for member in self.members) * self.planned_members)

    @property
    def tokenizer(self):
        """A tokenizer for the served model (first member's)."""
        return self.members[0].tokenizer

    @property
    def num_pending_requests(self) -> int:
        """Aggregate waiting-queue depth across live members."""
        with self._route_lock:
            members = list(self.members)
        return sum(int(getattr(member, "num_pending_requests", 0)) for member in members if _member_alive(member))

    @property
    def num_running_requests(self) -> int:
        """Aggregate running-set size across live members."""
        with self._route_lock:
            members = list(self.members)
        return sum(int(getattr(member, "num_running_requests", 0)) for member in members if _member_alive(member))

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = False,
        tool_parser_request: typing.Any | None = None,
    ) -> list[RequestOutput]:
        """Route each prompt to its member and gather the finished outputs.

        Prompts of one call may land on different members; outputs return
        in prompt order.

        Args:
            prompts: One prompt or a batch.
            sampling_params: Generation parameters, shared by the batch.
            request_id: Optional explicit id(s), forwarded to the member.
            use_tqdm: Forwarded to local members.
            tool_parser_request: Forwarded to members that support it.

        Returns:
            One finished ``RequestOutput`` per prompt, in order.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        elif len(prompts) == 0:
            raise ValueError("Empty prompt list provided")
        if request_id is None:
            request_ids: list[str | None] = [None] * len(prompts)
        elif isinstance(request_id, str):
            if len(prompts) != 1:
                raise ValueError("request_id must be a list when providing multiple prompts.")
            request_ids = [request_id]
        else:
            request_ids = list(request_id)
            if len(request_ids) != len(prompts):
                raise ValueError("Length of request_id list must match number of prompts.")

        # Validate the explicit/auto id mix up front, on the caller's thread.
        # A mixed batch used to raise inside a daemon `_run` thread before its
        # try/finally, so the error was swallowed (caller saw `None` outputs)
        # and the acquired JSQ slots leaked. Fail cleanly before acquiring any.
        provided = [rid for rid in request_ids if rid is not None]
        if provided and len(provided) != len(request_ids):
            raise ValueError("Mix of explicit and auto request ids in one routed batch is unsupported")

        by_counter: dict[int, tuple[typing.Any, _SlotCounter, list[int]]] = {}
        acquired: list[_SlotCounter] = []
        try:
            for position, prompt in enumerate(prompts):
                _member_index, member, counter = self._select_and_acquire(prompt)
                acquired.append(counter)
                key = id(counter)
                if key not in by_counter:
                    by_counter[key] = (member, counter, [position])
                else:
                    by_counter[key][2].append(position)
        except BaseException:
            for counter in acquired:
                self._release_counter(counter)
            raise

        results: list[RequestOutput | None] = [None] * len(prompts)
        errors: list[Exception] = []
        threads: list[threading.Thread] = []
        groups = list(by_counter.values())

        def _run(member, counter: _SlotCounter, positions: list[int]) -> None:
            # The whole body runs under the release-protecting try/finally so
            # the acquired in-flight slots free on every exit path.
            try:
                member_prompts = [prompts[p] for p in positions]
                member_ids = [request_ids[p] for p in positions]
                explicit = [rid for rid in member_ids if rid is not None]
                kwargs: dict[str, typing.Any] = {}
                if explicit:
                    kwargs["request_id"] = explicit if len(explicit) > 1 else explicit[0]
                outputs = member.generate(
                    member_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                    tool_parser_request=tool_parser_request,
                    **kwargs,
                )
                for position, output in zip(positions, outputs, strict=True):
                    results[position] = output
            except Exception as exc:  # surfaced after the join below
                errors.append(exc)
            finally:
                for _ in positions:
                    self._release_counter(counter)

        if len(groups) == 1:
            member, counter, positions = groups[0]
            _run(member, counter, positions)
        else:
            started_count = 0
            try:
                for member, counter, positions in groups:
                    thread = threading.Thread(target=_run, args=(member, counter, positions), daemon=True)
                    thread.start()
                    threads.append(thread)
                    started_count += 1
            except BaseException:
                for _member, counter, positions in groups[started_count:]:
                    for _ in positions:
                        self._release_counter(counter)
                for thread in threads:
                    thread.join()
                raise
            for thread in threads:
                thread.join()
        if errors:
            raise errors[0]
        return typing.cast("list[RequestOutput]", results)

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        tool_parser_request: typing.Any | None = None,
    ) -> typing.Iterator[RequestOutput]:
        """Route one streaming request to its member and relay its updates."""
        if isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("Empty prompt list provided")
            prompt = prompts[0]
        else:
            prompt = prompts
        _member_index, member, counter = self._select_and_acquire(prompt)
        try:
            iterator = member.stream(
                prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                tool_parser_request=tool_parser_request,
            )
            yield from iterator
        finally:
            self._release_counter(counter)

    def _chat_routing_key(self, messages: list[dict]) -> str:
        """A stable prompt-shaped key for affinity routing of chat calls."""
        try:
            return str(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            return repr(messages)

    def chat(self, messages: list[dict], *args, **kwargs):
        """Route a chat call to one member (affinity by templated prompt)."""
        _member_index, member, counter = self._select_and_acquire(self._chat_routing_key(messages))
        try:
            result = member.chat(messages, *args, **kwargs)
        except BaseException:
            self._release_counter(counter)
            raise
        if hasattr(result, "__next__"):
            # Streaming chat: the slot frees when the iterator ends, closes, or is GC'd.
            return _ReleasedIterator(result, lambda: self._release_counter(counter))
        self._release_counter(counter)
        return result

    @staticmethod
    def _stream_usage_kwargs(stream_fn, usage_callback) -> dict[str, typing.Any]:
        """Forward optional progress to supporting members, preserving legacy signatures."""
        if usage_callback is None:
            return {}
        try:
            parameters = inspect.signature(stream_fn).parameters
        except (TypeError, ValueError):
            return {}
        # As on the API server, generic **kwargs is not an advertisement:
        # a legacy adapter may pass it through to a member without this API.
        parameter = parameters.get("usage_callback")
        if parameter is not None and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return {"usage_callback": usage_callback}
        return {}

    def iter_chat_completion_stream(
        self,
        *,
        messages: list[dict],
        usage_callback: typing.Callable[[int, int], None] | None = None,
        **kwargs,
    ):
        """Route chat chunks and optional cumulative usage progress to one member."""
        _member_index, member, counter = self._select_and_acquire(self._chat_routing_key(messages))
        try:
            stream_fn = member.iter_chat_completion_stream
            yield from stream_fn(messages=messages, **kwargs, **self._stream_usage_kwargs(stream_fn, usage_callback))
        finally:
            self._release_counter(counter)

    def iter_responses_stream(
        self,
        *,
        messages: list[dict],
        usage_callback: typing.Callable[[int, int], None] | None = None,
        **kwargs,
    ):
        """Route Responses frames and optional cumulative usage progress to one member."""
        _member_index, member, counter = self._select_and_acquire(self._chat_routing_key(messages))
        try:
            stream_fn = member.iter_responses_stream
            yield from stream_fn(messages=messages, **kwargs, **self._stream_usage_kwargs(stream_fn, usage_callback))
        finally:
            self._release_counter(counter)

    def abort_request(self, request_id: str) -> None:
        """Cancel a request wherever it landed (no-op on non-owners)."""
        with self._route_lock:
            members = list(self.members)
        for member in members:
            try:
                member.abort_request(request_id)
            except Exception:
                logger.debug("Routed abort failed on one member", exc_info=True)

    def close(self) -> None:
        """Close remote members (local engines are left to their owners)."""
        with self._route_lock:
            finalizer = self._shared_finalizer
            self._shared_finalizer = None
            if finalizer is not None:
                finalizer.detach()
            buf = self._shared_buf
            self._shared_buf = None
            owner_pid = self._shared_owner_pid
            self._shared_owner_pid = None
            members = list(self.members)
            if buf is not None:
                _close_shared_buf(buf, self._front_index, owner_pid)
        for member in members:
            close = getattr(member, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.debug("Routed member close failed", exc_info=True)
