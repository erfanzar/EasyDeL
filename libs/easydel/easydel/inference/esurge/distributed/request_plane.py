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

"""The eSurge unified request plane: owner-schedules, origins-render.

One process per group — the step-plane leader — runs the scheduler. Every
other rank forwards admissions upstream over its existing DEALER socket and
receives raw token deltas back; the *origin* rank synthesizes
``EngineCoreOutputs`` and feeds its own unchanged ``OutputPipeline``, so
detokenization, reasoning/tool parsing, stop-string policy, and stream
event wakeups run through identical code on the rank that owns the caller.
The owner does zero text work for remote-only requests: it has no registry
record for them, so its pipeline skips them by construction.

Two halves:

* :class:`OwnerRequestPlane` — leader side. Receives ``Admit`` messages
  (handed off from the coordinator IO thread), renames the origin-built
  ``EngineRequest`` objects into the owner id space, enqueues them on the
  scheduler, and tees each step's engine outputs back to the origins as
  ``OutputUpdate`` deltas. All engine-touching work runs on a dedicated
  ingest thread — never on the socket IO thread.
* :class:`OriginRequestPlane` — non-leader side. Forwards locally-admitted
  requests to the owner (blocking on the ``AdmitAck``), applies inbound
  deltas to the local output pipeline, and forwards aborts and
  parser-detected stop-string hits upstream.

**Coalescing** makes replicated trainer traffic free: every rank of a
lockstep trainer calls ``generate()`` with identical prompts and
auto-generated ids, so the plane keys each *coalescable* admission by
``(per-rank coalesce counter, content hash)`` and schedules one physical
request per logical group; every other rank attaches as a subscriber and
the tee fans the same token deltas to all of them. A late attach — a rank
whose admission arrives after tokens already flowed — receives a catch-up
replay from the group's token log; groups are retained ``retention_s``
past finish (or until every group rank attached) for stragglers. Explicit
request ids never coalesce, and per-rank counters strictly increase, so
same-rank duplicates and server traffic keep their own scheduled rows.

The owner's *own* coalescable admissions stay on the fast path: when the
owner admits first, the group is scheduled under the owner's local ids and
its caller renders directly from the local pipeline (no synthesis, no
rename) — the entry merely logs outputs for later subscribers. Only when a
remote admission arrives first is the group scheduled under a neutral
``q{counter}-{hash8}`` id and the late local caller rendered from replayed
deltas like any origin.

Id spaces stay disjoint by construction: non-coalescable remote parents
schedule as ``r{rank}c{counter}-{hash8}``, remote-first coalesced groups as
``q{counter}-{hash8}``, and owner-local ids keep their native ``req-``
forms, so nothing collides and every delta is translated into the
subscriber's own id space before it crosses the wire.
"""

from __future__ import annotations

import itertools
import queue
import re
import threading
import time
import typing

from ..engine_types import EngineCoreOutput, EngineCoreOutputs
from ..logger import logger
from . import wire

if typing.TYPE_CHECKING:
    from ..request import EngineRequest
    from .zmq_coordinator import ZmqLeaderCoordinator, ZmqWorkerCoordinator

_STOP = object()

#: Auto-generated multi-host request ids (``RequestAdmission.generate_request_id``
#: under ``jax.process_count() > 1``). Only these coalesce: an explicit caller
#: id marks intentionally distinct traffic and salts the content hash.
_AUTO_ID_PATTERN = re.compile(r"^req-\d{10}$")

#: Subscriber-identity sentinel for the owner-local caller.
_LOCAL = None


class RequestPlaneError(RuntimeError):
    """A remote admission or plane transport failed.

    Raised on the origin when the owner NACKs an admission, the ack deadline
    lapses, or the plane is shut down while requests are pending — and on
    the owner when lockstep admissions diverge (same coalesce counter,
    different content). Like step coordination, the plane is fail-fast:
    there is no admission retry.
    """


class RequestPlaneUnavailable(RequestPlaneError):
    """The owner's control endpoint is not reachable (yet).

    The retryable subset of plane failures: the engine may simply still be
    compiling before it binds its control socket. Launchers poll on this;
    every other :class:`RequestPlaneError` is permanent.
    """


def _is_coalescable_id(request_id: str) -> bool:
    """Whether ``request_id`` is an auto-generated (coalescable) parent id."""
    return _AUTO_ID_PATTERN.match(request_id) is not None


def _group_by_parent(scheduler_requests: list[EngineRequest]) -> list[tuple[str, list[EngineRequest]]]:
    """Group admission-built requests by parent id, preserving order.

    Args:
        scheduler_requests: Requests produced by ``RequestAdmission`` —
            one per sample, children carrying ``parent_request_id``.

    Returns:
        ``[(parent_id, [requests...]), ...]`` in first-seen order.
    """
    groups: dict[str, list[EngineRequest]] = {}
    for request in scheduler_requests:
        parent_id = request.parent_request_id or request.request_id
        groups.setdefault(parent_id, []).append(request)
    return list(groups.items())


def _rename_output(engine_output: EngineCoreOutput, request_id: str) -> EngineCoreOutput:
    """Copy one engine output into a subscriber's id space."""
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=engine_output.new_token_ids,
        new_logprobs=engine_output.new_logprobs,
        new_prompt_logprobs_tensors=engine_output.new_prompt_logprobs_tensors,
        finish_reason=engine_output.finish_reason,
        stop_reason=engine_output.stop_reason,
        num_cached_tokens=engine_output.num_cached_tokens,
    )


class _CoalesceEntry:
    """One logical (cross-rank shared) request group on the owner.

    Attributes:
        key: ``(coalesce_counter, content_hash)`` identity of the group.
        scheduled_parent: Parent id the group is scheduled under — the
            owner's native local id when the owner admitted first
            (``local_direct``), else a neutral ``q...`` id.
        suffixes: Per-sample child suffixes (``[""]`` for ``n=1``).
        local_direct: The owner-local caller renders straight from its own
            registry (scheduled ids == its local ids); the tee only logs
            for later subscribers and fans out to remote ones.
        subscribers: ``(identity | _LOCAL, subscriber parent id) -> direct``
            — attached parties; ``direct`` subscribers receive no fan-out.
        log: Per-suffix ordered engine outputs (scheduled-id space) for
            catch-up replay of late attaches.
        finished_suffixes: Suffixes that saw a terminal output.
        scheduler_finished: Suffixes finished via a scheduler-side signal
            (no terminal output; replayed as ``finished_requests``).
        finished_at: Monotonic time the whole group finished, for the
            retention sweep.
    """

    __slots__ = (
        "finished_at",
        "finished_suffixes",
        "key",
        "local_direct",
        "log",
        "scheduled_parent",
        "scheduler_finished",
        "subscribers",
        "suffixes",
    )

    def __init__(self, key: tuple[int, str], scheduled_parent: str, suffixes: list[str], local_direct: bool) -> None:
        self.key = key
        self.scheduled_parent = scheduled_parent
        self.suffixes = suffixes
        self.local_direct = local_direct
        self.subscribers: dict[tuple[bytes | None, str], bool] = {}
        self.log: dict[str, list[EngineCoreOutput]] = {suffix: [] for suffix in suffixes}
        self.finished_suffixes: set[str] = set()
        self.scheduler_finished: set[str] = set()
        self.finished_at: float | None = None

    @property
    def all_finished(self) -> bool:
        """Whether every sample of the group reached a terminal state."""
        return len(self.finished_suffixes | self.scheduler_finished) >= len(self.suffixes)

    def build_replay(self, parent_id: str) -> tuple[list[EngineCoreOutput], set[str] | None]:
        """Replay the group's full log in a subscriber's id space."""
        outputs: list[EngineCoreOutput] = []
        finished: set[str] = set()
        for suffix in self.suffixes:
            subscriber_id = parent_id + suffix
            for engine_output in self.log[suffix]:
                outputs.append(_rename_output(engine_output, subscriber_id))
            if suffix in self.scheduler_finished:
                finished.add(subscriber_id)
        return outputs, (finished or None)


class OwnerRequestPlane:
    """Leader-side request plane: admit remote requests, tee outputs back.

    Tables are guarded by one reentrant lock: the ingest thread (wire
    messages, output fan-out), engine caller threads (local coalescable
    admissions, stop/abort translation), and the retention sweep all touch
    them. ``needs_tee`` is a plain bool published for the engine-loop hot
    path; the engine tees a step's outputs only while remote requests or
    coalesce groups are live.

    Args:
        coordinator: The leader step coordinator (targeted sends).
        scheduler_submit: Engine callable enqueueing a batch of
            ``EngineRequest`` directly under the scheduler lock.
        abort_request: Engine callable cancelling one request id.
        apply_stop_strings: Engine callable routing
            ``{request_id: stop_string}`` to the scheduler-safe stop queue.
        submit_local_outputs: Engine callable feeding a synthesized bundle
            to the owner's own output pipeline (late local attaches).
        next_arrival_stamp: Engine-admission callable producing the next
            monotonic arrival stamp, so remote admissions interleave FCFS
            with owner-local ones.
        retention_s: How long a finished coalesce group is retained for
            late attaches when not every group rank attached yet.
        info: Engine-level info logger callable.
    """

    owns_scheduling = True

    def __init__(
        self,
        *,
        coordinator: ZmqLeaderCoordinator,
        scheduler_submit: typing.Callable[[list[EngineRequest]], None],
        abort_request: typing.Callable[[str], None],
        apply_stop_strings: typing.Callable[[dict[str, str]], None],
        submit_local_outputs: typing.Callable[[dict[int, EngineCoreOutputs]], None],
        next_arrival_stamp: typing.Callable[[], float],
        retention_s: float = 120.0,
        info: typing.Callable[..., None],
    ) -> None:
        self._coordinator = coordinator
        self._scheduler_submit = scheduler_submit
        self._abort_request = abort_request
        self._apply_stop_strings = apply_stop_strings
        self._submit_local_outputs = submit_local_outputs
        self._next_arrival_stamp = next_arrival_stamp
        self._retention_s = float(retention_s)
        self._info = info
        self._world_size = int(getattr(coordinator, "world_size", 1))

        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._out_seq = 0
        self._lock = threading.RLock()

        # non-coalescable remote requests
        # Owner scheduler id (per-sample child) -> (identity, origin id).
        self._owner_to_origin: dict[str, tuple[bytes, str]] = {}
        # (identity, origin id) -> owner id, for AbortReq/StopHit translation.
        self._origin_to_owner: dict[tuple[bytes, str], str] = {}
        # Owner parent id -> live owner child ids (drives map retirement).
        self._children: dict[str, set[str]] = {}
        # Owner child id -> owner parent id; parent id -> (identity, origin parent).
        self._owner_child_to_parent: dict[str, str] = {}
        self._parents: dict[str, tuple[bytes, str]] = {}

        self._entries: dict[tuple[int, str], _CoalesceEntry] = {}
        # coalesce counter -> content hash, for loud lockstep-divergence NACKs.
        self._by_counter: dict[int, str] = {}
        # Scheduled child id -> (entry, suffix).
        self._scheduled_children: dict[str, tuple[_CoalesceEntry, str]] = {}
        # (identity | _LOCAL, subscriber child id) -> (entry, suffix, subscriber parent id).
        self._subscriber_children: dict[tuple[bytes | None, str], tuple[_CoalesceEntry, str, str]] = {}
        self._local_coalesce_counter = itertools.count(1)

        #: Single-bool hot-path gate for the engine loop's output tee.
        self.needs_tee = False
        #: Plane-behavior counters (tests and postmortems).
        self.stats: dict[str, int] = {
            "remote_admits": 0,
            "scheduled_groups": 0,
            "coalesced_attaches": 0,
            "replayed_attaches": 0,
        }

        coordinator.set_plane_handler(self.handle_inbound)

    def ensure_started(self) -> None:
        """Start the ingest thread (idempotent; safe across engine restarts)."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="eSurgePlaneIngest", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the ingest thread; id maps survive for a later restart."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        self._queue.put(_STOP)
        thread.join(timeout=5.0)
        if thread.is_alive():
            logger.warning("Request-plane ingest thread did not stop gracefully")
            return
        if self._thread is thread:
            self._thread = None

    def reset(self) -> None:
        """Drop every remote mapping and coalesce group (weight swaps clear all requests)."""
        with self._lock:
            self._owner_to_origin.clear()
            self._origin_to_owner.clear()
            self._children.clear()
            self._owner_child_to_parent.clear()
            self._parents.clear()
            self._entries.clear()
            self._by_counter.clear()
            self._scheduled_children.clear()
            self._subscriber_children.clear()
            self.needs_tee = False

    def handle_inbound(self, identity: bytes, message: wire.WireMessage, payload: bytes | None) -> None:
        """Coordinator IO-thread handoff for plane wire messages."""
        self._queue.put(("wire", identity, message, payload))

    def tee_outputs(self, engine_outputs) -> None:
        """Engine-loop handoff: fan this step's outputs back to subscribers.

        Called once per step when :attr:`needs_tee` is set; the bundle walk
        happens on the ingest thread, keeping the scheduler hot path at one
        bool read plus one queue put.
        """
        self._queue.put(("outputs", None, engine_outputs, None))

    def admit_local(self, scheduler_requests: list[EngineRequest]) -> None:
        """Route the owner's own admissions through the coalescing table.

        Non-coalescable parents (explicit caller ids) schedule directly,
        untracked — the single-ingress serving hot path is unchanged.
        Coalescable parents create or attach to their group:

        * First arrival creates the entry. When the owner is first, the
          group schedules under the owner's own ids and the local caller
          renders directly (``local_direct``); the entry only logs.
        * A later arrival attaches as a subscriber and receives a catch-up
          replay of the group log through the owner's own pipeline.

        Called on the admitting caller's thread.

        Args:
            scheduler_requests: Requests produced by ``RequestAdmission``.

        Raises:
            RequestPlaneError: When lockstep admissions diverge (same
                coalesce counter, different content hash).
        """
        from ..engine.admission import compute_admission_key

        to_schedule: list[EngineRequest] = []
        created_entries: list[_CoalesceEntry] = []
        with self._lock:
            for parent_id, requests in _group_by_parent(scheduler_requests):
                if not _is_coalescable_id(parent_id):
                    to_schedule.extend(requests)
                    continue
                counter = next(self._local_coalesce_counter)
                content_hash = compute_admission_key(
                    requests[0].prompt_token_ids, requests[0].sampling_params, len(requests)
                )
                entry = self._lookup_entry(counter, content_hash, origin="owner-local")
                if entry is None:
                    suffixes = [request.request_id[len(parent_id) :] for request in requests]
                    entry = self._create_entry((counter, content_hash), parent_id, suffixes, local_direct=True)
                    self._attach(entry, _LOCAL, parent_id, direct=True)
                    for request in requests:
                        self._scheduled_children[request.request_id] = (
                            entry,
                            request.request_id[len(parent_id) :],
                        )
                    to_schedule.extend(requests)
                    created_entries.append(entry)
                    self.stats["scheduled_groups"] += 1
                else:
                    self._attach(entry, _LOCAL, parent_id, direct=False)
                    outputs, finished = entry.build_replay(parent_id)
                    # Submitted while holding the lock: a concurrent tee
                    # fanning newer tokens to this subscriber serializes
                    # behind the attach, so the pipeline sees the catch-up
                    # replay strictly before any later delta.
                    if outputs or finished:
                        self._submit_local_outputs(
                            {
                                0: EngineCoreOutputs(
                                    outputs=outputs,
                                    timestamp=time.perf_counter(),
                                    finished_requests=finished,
                                )
                            }
                        )
                        self.stats["replayed_attaches"] += 1
                    self.stats["coalesced_attaches"] += 1
        if to_schedule:
            try:
                self._scheduler_submit(to_schedule)
            except Exception:
                with self._lock:
                    for entry in created_entries:
                        self._retire_entry(entry)
                raise

    def translate_local_stop_hits(self, stop_string_finishes: dict[str, str]) -> dict[str, str]:
        """Map owner-local sample ids onto scheduled ids where a group renamed them.

        ``local_direct`` groups and untracked requests pass through
        unchanged (their scheduler ids are the local ids).

        Args:
            stop_string_finishes: ``{sample_request_id: matched_stop}`` from
                the owner's output pipeline.

        Returns:
            The same hits keyed by scheduler-resolvable ids.
        """
        with self._lock:
            if not self._subscriber_children:
                return stop_string_finishes
            translated: dict[str, str] = {}
            for request_id, stop_string in stop_string_finishes.items():
                mapped = self._subscriber_children.get((_LOCAL, request_id))
                if mapped is not None:
                    entry, suffix, _parent = mapped
                    translated[entry.scheduled_parent + suffix] = stop_string
                else:
                    translated[request_id] = stop_string
            return translated

    def notify_local_abort(self, request_id: str) -> None:
        """Owner-local abort hook: detach the local subscriber from its group.

        For a ``local_direct`` group the scheduled rows are the caller's own
        ids and the engine's normal abort tears them down (remote
        subscribers observe the abort via scheduler-finish signals on the
        tee). For a renamed (remote-first) group this detaches the local
        subscriber only and aborts the scheduled rows when it was the last
        one.

        Args:
            request_id: Owner-local parent (or sample child) id.
        """
        scheduled_abort: str | None = None
        with self._lock:
            keys = [key for key in self._subscriber_children if key[0] is _LOCAL]
            for key in keys:
                mapped = self._subscriber_children.get(key)
                if mapped is None:
                    continue
                entry, _suffix, parent_id = mapped
                if key[1] != request_id and parent_id != request_id:
                    continue
                if entry.local_direct:
                    # The engine abort will kill the scheduled rows itself;
                    # scheduler-finish signals propagate to remote subscribers.
                    continue
                self._subscriber_children.pop(key, None)
                # A sample-level abort keeps the subscriber attached for its
                # remaining samples; a parent-level (or n=1) abort detaches it.
                if parent_id == request_id or key[1] == parent_id:
                    entry.subscribers.pop((_LOCAL, parent_id), None)
                if not entry.subscribers and not entry.all_finished:
                    scheduled_abort = entry.scheduled_parent
                    self._retire_entry(entry)
        if scheduled_abort is not None:
            self._abort_request(scheduled_abort)

    def _loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._sweep_retention()
                continue
            try:
                if item is _STOP:
                    return
                kind, identity, message, payload = item
                if kind == "wire":
                    self._dispatch_wire(identity, message, payload)
                elif kind == "outputs":
                    self._fan_out(message)
            except Exception:
                logger.exception("Request-plane ingest failed on %r", item[0] if isinstance(item, tuple) else item)
            finally:
                self._queue.task_done()

    def _dispatch_wire(self, identity: bytes, message: wire.WireMessage, payload: bytes | None) -> None:
        if isinstance(message, wire.Admit):
            self._handle_admit(identity, message, payload)
        elif isinstance(message, wire.AbortReq):
            self._handle_abort(identity, message)
        elif isinstance(message, wire.StopHit):
            self._handle_stop_hit(identity, message)
        else:
            logger.warning("Request plane: ignoring unexpected %s", type(message).__name__)

    def _lookup_entry(self, counter: int, content_hash: str, *, origin: str) -> _CoalesceEntry | None:
        """Find the group for a coalescing key, failing loudly on divergence."""
        known_hash = self._by_counter.get(counter)
        if known_hash is not None and known_hash != content_hash:
            raise RequestPlaneError(
                f"lockstep admission divergence at coalesce counter {counter} ({origin}): "
                f"content hash {content_hash[:12]}... != existing {known_hash[:12]}...; "
                "ranks are admitting different requests in the same slot"
            )
        return self._entries.get((counter, content_hash))

    def _create_entry(
        self, key: tuple[int, str], scheduled_parent: str, suffixes: list[str], *, local_direct: bool
    ) -> _CoalesceEntry:
        entry = _CoalesceEntry(key, scheduled_parent, suffixes, local_direct)
        self._entries[key] = entry
        self._by_counter[key[0]] = key[1]
        self.needs_tee = True
        return entry

    def _attach(self, entry: _CoalesceEntry, identity: bytes | None, parent_id: str, *, direct: bool) -> None:
        entry.subscribers[(identity, parent_id)] = direct
        for suffix in entry.suffixes:
            self._subscriber_children[(identity, parent_id + suffix)] = (entry, suffix, parent_id)

    def _detach(self, entry: _CoalesceEntry, identity: bytes | None, parent_id: str) -> None:
        entry.subscribers.pop((identity, parent_id), None)
        for suffix in entry.suffixes:
            self._subscriber_children.pop((identity, parent_id + suffix), None)

    def _retire_entry(self, entry: _CoalesceEntry) -> None:
        self._entries.pop(entry.key, None)
        if self._by_counter.get(entry.key[0]) == entry.key[1]:
            self._by_counter.pop(entry.key[0], None)
        for suffix in entry.suffixes:
            self._scheduled_children.pop(entry.scheduled_parent + suffix, None)
        for key in [key for key, value in self._subscriber_children.items() if value[0] is entry]:
            self._subscriber_children.pop(key, None)
        self._refresh_needs_tee()

    def _refresh_needs_tee(self) -> None:
        self.needs_tee = bool(self._owner_to_origin or self._entries)

    def _sweep_retention(self) -> None:
        """Retire finished groups past retention (or fully attached)."""
        now = time.monotonic()
        with self._lock:
            if not self._entries:
                return
            for entry in list(self._entries.values()):
                if entry.finished_at is None:
                    continue
                fully_attached = len(entry.subscribers) >= self._world_size
                if fully_attached or now - entry.finished_at > self._retention_s:
                    self._retire_entry(entry)

    def _handle_admit(self, identity: bytes, admit: wire.Admit, payload: bytes | None) -> None:
        """Schedule (or coalesce-attach) a remote admission and ack it."""
        try:
            if payload is None:
                raise RequestPlaneError("Admit carried no request payload")
            requests: list[EngineRequest] = wire.decode_payload(payload)
            origin_parent = admit.request_id
            if admit.n_samples != len(requests):
                raise RequestPlaneError(
                    f"Admit for {origin_parent!r} declared n_samples={admit.n_samples} "
                    f"but carried {len(requests)} request(s)"
                )
            for request in requests:
                if not request.request_id.startswith(origin_parent):
                    raise RequestPlaneError(
                        f"request id {request.request_id!r} does not extend parent {origin_parent!r}"
                    )
            self.stats["remote_admits"] += 1
            if admit.coalescable:
                ack, ack_payload = self._admit_remote_coalescable(identity, admit, requests)
            else:
                self._admit_remote_distinct(identity, admit, requests)
                ack, ack_payload = wire.AdmitAck(request_id=origin_parent, ok=True), None
        except Exception as exc:
            logger.exception("Request plane: remote admission failed for %r", admit.request_id)
            self._coordinator.send_to_peer(
                identity,
                wire.AdmitAck(request_id=admit.request_id, ok=False, error=str(exc)),
            )
            return
        self._coordinator.send_to_peer(identity, ack, ack_payload)

    def _admit_remote_distinct(self, identity: bytes, admit: wire.Admit, requests: list[EngineRequest]) -> None:
        """Distinct path: schedule a non-coalescable remote group under ``r...`` ids.

        The owner id embeds a digest of the ZMQ identity: request-plane
        clients all report ``origin_rank=-1``, so rank+counter alone would
        collide across two clients.
        """
        import zlib

        origin_parent = admit.request_id
        hash8 = (admit.content_hash or "00000000")[:8]
        identity_tag = f"{zlib.crc32(identity) & 0xFFFF:04x}"
        owner_parent = f"r{admit.origin_rank}i{identity_tag}c{admit.origin_counter}-{hash8}"
        for request in requests:
            suffix = request.request_id[len(origin_parent) :]
            request.request_id = owner_parent + suffix
            if request.parent_request_id is not None:
                request.parent_request_id = owner_parent
            request.arrival_time = self._next_arrival_stamp()

        with self._lock:
            child_ids = set()
            for request in requests:
                origin_id = origin_parent + request.request_id[len(owner_parent) :]
                self._owner_to_origin[request.request_id] = (identity, origin_id)
                self._origin_to_owner[(identity, origin_id)] = request.request_id
                self._owner_child_to_parent[request.request_id] = owner_parent
                child_ids.add(request.request_id)
            self._origin_to_owner[(identity, origin_parent)] = owner_parent
            self._parents[owner_parent] = (identity, origin_parent)
            self._children[owner_parent] = child_ids
            self.needs_tee = True
        try:
            self._scheduler_submit(requests)
        except Exception:
            with self._lock:
                for child_id in list(child_ids):
                    self._retire(child_id)
                self._retire(owner_parent)
            raise

    def _admit_remote_coalescable(
        self, identity: bytes, admit: wire.Admit, requests: list[EngineRequest]
    ) -> tuple[wire.AdmitAck, bytes | None]:
        """Coalescing path: create the group or attach as a subscriber."""
        origin_parent = admit.request_id
        key = (admit.coalesce_counter, admit.content_hash)
        to_schedule: list[EngineRequest] | None = None
        ack_payload: bytes | None = None
        already_finished = False
        with self._lock:
            entry = self._lookup_entry(key[0], key[1], origin=f"rank identity {identity!r}")
            if entry is None:
                hash8 = (admit.content_hash or "00000000")[:8]
                scheduled_parent = f"q{admit.coalesce_counter:010d}-{hash8}"
                suffixes = []
                for request in requests:
                    suffix = request.request_id[len(origin_parent) :]
                    suffixes.append(suffix)
                    request.request_id = scheduled_parent + suffix
                    if request.parent_request_id is not None:
                        request.parent_request_id = scheduled_parent
                    request.arrival_time = self._next_arrival_stamp()
                entry = self._create_entry(key, scheduled_parent, suffixes, local_direct=False)
                self._attach(entry, identity, origin_parent, direct=False)
                for suffix in suffixes:
                    self._scheduled_children[scheduled_parent + suffix] = (entry, suffix)
                to_schedule = requests
                self.stats["scheduled_groups"] += 1
            else:
                self._attach(entry, identity, origin_parent, direct=False)
                outputs, finished = entry.build_replay(origin_parent)
                if outputs or finished:
                    ack_payload = wire.encode_payload((outputs, finished))
                    self.stats["replayed_attaches"] += 1
                already_finished = entry.all_finished
                self.stats["coalesced_attaches"] += 1
        if to_schedule is not None:
            try:
                self._scheduler_submit(to_schedule)
            except Exception:
                with self._lock:
                    self._retire_entry(entry)
                raise
        return (
            wire.AdmitAck(request_id=origin_parent, ok=True, already_finished=already_finished),
            ack_payload,
        )

    def _handle_abort(self, identity: bytes, message: wire.AbortReq) -> None:
        scheduled_aborts: list[str] = []
        with self._lock:
            owner_id = self._origin_to_owner.get((identity, message.request_id))
            if owner_id is not None:
                # Non-coalescable: the rows belong to this origin alone.
                for child_id in list(self._children.get(owner_id, ())):
                    self._retire(child_id)
                self._retire(owner_id)
                scheduled_aborts.append(owner_id)
            else:
                # Coalescable abort. Subscriber children are keyed by
                # ``parent + suffix``, so a bare parent id never matches a
                # ``get`` on the raw id (``AUTO`` vs ``AUTO-0``). Mirror
                # ``notify_local_abort``: the wire id is either a subscriber
                # parent id (``AbortReq`` contract: aborts every ``n>1``
                # child) or a subscriber sample-child id (aborts one sample).
                request_id = message.request_id
                matched_keys = [
                    key
                    for key, (_entry, _suffix, parent_id) in self._subscriber_children.items()
                    if key[0] == identity and (key[1] == request_id or parent_id == request_id)
                ]
                for key in matched_keys:
                    mapped = self._subscriber_children.get(key)
                    if mapped is None:
                        continue  # already retired while handling a sibling suffix
                    entry, suffix, parent_id = mapped
                    self._subscriber_children.pop(key, None)
                    # A parent-level (or n=1) abort detaches the subscriber
                    # entirely; a sample-level abort keeps it attached for its
                    # remaining samples.
                    if parent_id == request_id or key[1] == parent_id:
                        entry.subscribers.pop((identity, parent_id), None)
                    if not entry.subscribers and not entry.all_finished:
                        # Last subscriber gone: tear the whole group down.
                        scheduled_aborts.append(entry.scheduled_parent)
                        self._retire_entry(entry)
                    elif (
                        suffix not in entry.finished_suffixes
                        and suffix not in entry.scheduler_finished
                        and not any(
                            value[0] is entry and value[1] == suffix
                            for value in self._subscriber_children.values()
                        )
                    ):
                        # Subscribers remain, but none still wants this sample:
                        # abort only its scheduled row so the survivors stream
                        # on (the AbortReq contract is "a child id aborts one
                        # sample"). Mark it finished so the group retires once
                        # the rest terminate.
                        entry.scheduler_finished.add(suffix)
                        if entry.all_finished and entry.finished_at is None:
                            entry.finished_at = time.monotonic()
                        scheduled_aborts.append(entry.scheduled_parent + suffix)
        for scheduled_abort in scheduled_aborts:
            try:
                self._abort_request(scheduled_abort)
            except Exception:
                logger.exception("Request plane: remote abort failed for %r", scheduled_abort)

    def _handle_stop_hit(self, identity: bytes, message: wire.StopHit) -> None:
        mapped: dict[str, str] = {}
        with self._lock:
            for origin_id, stop_string in message.hits.items():
                owner_id = self._origin_to_owner.get((identity, origin_id))
                if owner_id is not None:
                    mapped[owner_id] = stop_string
                    continue
                subscribed = self._subscriber_children.get((identity, origin_id))
                if subscribed is not None:
                    entry, suffix, _parent = subscribed
                    mapped[entry.scheduled_parent + suffix] = stop_string
        if mapped:
            self._apply_stop_strings(mapped)

    def _retire(self, owner_id: str) -> None:
        """Retire one owner child id; retire the group when its last child drains."""
        entry = self._owner_to_origin.pop(owner_id, None)
        if entry is not None:
            self._origin_to_owner.pop((entry[0], entry[1]), None)
        owner_parent = self._owner_child_to_parent.pop(owner_id, owner_id)
        children = self._children.get(owner_parent)
        if children is not None:
            children.discard(owner_id)
            if not children:
                self._children.pop(owner_parent, None)
                parent_entry = self._parents.pop(owner_parent, None)
                if parent_entry is not None:
                    self._origin_to_owner.pop((parent_entry[0], parent_entry[1]), None)
        self._refresh_needs_tee()

    def _fan_out(self, engine_outputs) -> None:
        """Log and translate one step's outputs to every subscriber."""
        if not engine_outputs:
            return
        per_origin: dict[bytes, tuple[list[EngineCoreOutput], set[str]]] = {}
        local_outputs: list[EngineCoreOutput] = []
        local_finished: set[str] = set()
        with self._lock:
            finished_owner_ids: list[str] = []
            for client_outputs in engine_outputs.values():
                for engine_output in client_outputs.outputs:
                    self._fan_one(engine_output, per_origin, local_outputs, finished_owner_ids)
                for finished_id in client_outputs.finished_requests or ():
                    self._fan_one_finished(finished_id, per_origin, local_finished, finished_owner_ids)
            for owner_id in finished_owner_ids:
                self._retire(owner_id)
            # Local fan-out submits under the lock so a late local attach's
            # catch-up replay (also lock-serialized) can never be reordered
            # behind a newer delta in the pipeline queue.
            if local_outputs or local_finished:
                self._submit_local_outputs(
                    {
                        0: EngineCoreOutputs(
                            outputs=local_outputs,
                            timestamp=time.perf_counter(),
                            finished_requests=local_finished or None,
                        )
                    }
                )

        for identity, (outputs, finished) in per_origin.items():
            self._out_seq += 1
            self._coordinator.send_to_peer(
                identity,
                wire.OutputUpdate(seq=self._out_seq),
                wire.encode_payload((outputs, finished or None)),
            )

    def _fan_one(
        self,
        engine_output: EngineCoreOutput,
        per_origin: dict[bytes, tuple[list[EngineCoreOutput], set[str]]],
        local_outputs: list[EngineCoreOutput],
        finished_owner_ids: list[str],
    ) -> None:
        scheduled_id = engine_output.request_id
        mapped = self._owner_to_origin.get(scheduled_id)
        if mapped is not None:
            identity, origin_id = mapped
            outputs, _finished = per_origin.setdefault(identity, ([], set()))
            outputs.append(_rename_output(engine_output, origin_id))
            if engine_output.finish_reason is not None:
                finished_owner_ids.append(scheduled_id)
            return
        coalesced = self._scheduled_children.get(scheduled_id)
        if coalesced is None:
            return
        entry, suffix = coalesced
        entry.log[suffix].append(engine_output)
        if engine_output.finish_reason is not None:
            entry.finished_suffixes.add(suffix)
            if entry.all_finished and entry.finished_at is None:
                entry.finished_at = time.monotonic()
        for (identity, parent_id), direct in entry.subscribers.items():
            if direct:
                continue  # local_direct caller renders straight from its registry
            subscriber_id = parent_id + suffix
            if identity is _LOCAL:
                local_outputs.append(_rename_output(engine_output, subscriber_id))
            else:
                outputs, _finished = per_origin.setdefault(identity, ([], set()))
                outputs.append(_rename_output(engine_output, subscriber_id))

    def _fan_one_finished(
        self,
        scheduled_id: str,
        per_origin: dict[bytes, tuple[list[EngineCoreOutput], set[str]]],
        local_finished: set[str],
        finished_owner_ids: list[str],
    ) -> None:
        mapped = self._owner_to_origin.get(scheduled_id)
        if mapped is not None:
            identity, origin_id = mapped
            _outputs, finished = per_origin.setdefault(identity, ([], set()))
            finished.add(origin_id)
            finished_owner_ids.append(scheduled_id)
            return
        coalesced = self._scheduled_children.get(scheduled_id)
        if coalesced is None:
            return
        entry, suffix = coalesced
        entry.scheduler_finished.add(suffix)
        if entry.all_finished and entry.finished_at is None:
            entry.finished_at = time.monotonic()
        for (identity, parent_id), direct in entry.subscribers.items():
            if direct:
                continue
            subscriber_id = parent_id + suffix
            if identity is _LOCAL:
                local_finished.add(subscriber_id)
            else:
                _outputs, finished = per_origin.setdefault(identity, ([], set()))
                finished.add(subscriber_id)


class _PendingAdmit:
    """Origin-side slot tracking one in-flight ``Admit`` round trip."""

    __slots__ = ("ack", "event")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.ack: wire.AdmitAck | None = None


class OriginRequestPlane:
    """Non-leader request plane: forward admissions, render inbound deltas.

    Args:
        coordinator: The worker step coordinator (leader DEALER).
        submit_outputs: Callable feeding a synthesized engine-outputs bundle
            into the local ``OutputPipeline`` (ordered FIFO worker).
        alive: Callable reporting whether the local generation backend (the
            replay thread) is still running — used to fail admissions fast
            instead of waiting out the ack deadline.
        rank: This process's rank (stamped into ``Admit``).
        admit_timeout_s: How long a caller blocks on the owner's ack. The
            first admissions overlap owner-side warmup compilation, so this
            defaults generously.
        info: Engine-level info logger callable.
    """

    owns_scheduling = False

    def __init__(
        self,
        *,
        coordinator: ZmqWorkerCoordinator,
        submit_outputs: typing.Callable[[dict[int, EngineCoreOutputs]], None],
        alive: typing.Callable[[], bool],
        rank: int,
        admit_timeout_s: float = 120.0,
        info: typing.Callable[..., None],
    ) -> None:
        self._coordinator = coordinator
        self._submit_outputs = submit_outputs
        self._alive = alive
        self._rank = int(rank)
        self._admit_timeout_s = float(admit_timeout_s)
        self._info = info

        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._coalesce_counter = itertools.count(1)
        self._pending: dict[str, _PendingAdmit] = {}
        # Origin-side ids (parents and sample children) currently owned remotely.
        self._remote_ids: set[str] = set()
        # Origin parent id -> live child count (drives id retirement).
        self._live_children: dict[str, int] = {}
        self._child_to_parent: dict[str, str] = {}

        coordinator.set_plane_handler(self._handle_inbound)

    def submit_remote(self, scheduler_requests: list[EngineRequest]) -> None:
        """Forward admission-built requests to the owner and await its acks.

        Called on the admitting caller's thread with no engine locks held.
        Registers the request ids as remotely-owned *before* sending so
        abort/stop forwarding sees them from the first token. Auto-id
        parents are marked coalescable and numbered on the per-rank
        coalesce sequence; explicit ids salt the content hash and never
        coalesce.

        Args:
            scheduler_requests: Requests produced by ``RequestAdmission`` —
                possibly several parents when ``generate()`` batch-defers.

        Raises:
            RequestPlaneError: If the plane transport is down, the owner
                NACKs an admission, or the ack deadline lapses.
        """
        from ..engine.admission import compute_admission_key

        if not self._alive():
            raise RequestPlaneError("request plane is down: the worker replay loop is not running")

        pending: list[tuple[str, _PendingAdmit]] = []
        for parent_id, requests in _group_by_parent(scheduler_requests):
            first = requests[0]
            coalescable = _is_coalescable_id(parent_id)
            content_hash = compute_admission_key(
                first.prompt_token_ids,
                first.sampling_params,
                len(requests),
                salt="" if coalescable else parent_id,
            )
            slot = _PendingAdmit()
            with self._lock:
                self._pending[parent_id] = slot
                self._remote_ids.add(parent_id)
                self._live_children[parent_id] = len(requests)
                for request in requests:
                    self._remote_ids.add(request.request_id)
                    self._child_to_parent[request.request_id] = parent_id
                coalesce_counter = next(self._coalesce_counter) if coalescable else 0
            self._coordinator.send_to_leader(
                wire.Admit(
                    request_id=parent_id,
                    origin_rank=self._rank,
                    origin_counter=next(self._counter),
                    content_hash=content_hash,
                    coalescable=coalescable,
                    coalesce_counter=coalesce_counter,
                    n_samples=len(requests),
                ),
                wire.encode_payload(requests),
            )
            pending.append((parent_id, slot))

        for parent_id, slot in pending:
            if not slot.event.wait(self._admit_timeout_s):
                self._forget(parent_id)
                raise RequestPlaneError(
                    f"owner did not acknowledge admission of {parent_id!r} within {self._admit_timeout_s}s"
                )
            ack = slot.ack
            if ack is None or not ack.ok:
                self._forget(parent_id)
                raise RequestPlaneError(
                    f"owner rejected admission of {parent_id!r}: {ack.error if ack else 'no ack recorded'}"
                )

    def _forget(self, parent_id: str) -> None:
        """Drop all plane state for one origin parent id."""
        with self._lock:
            self._pending.pop(parent_id, None)
            self._live_children.pop(parent_id, None)
            self._remote_ids.discard(parent_id)
            for child_id, parent in list(self._child_to_parent.items()):
                if parent == parent_id:
                    self._child_to_parent.pop(child_id, None)
                    self._remote_ids.discard(child_id)

    def is_remote(self, request_id: str) -> bool:
        """Whether ``request_id`` (parent or sample child) is remotely owned."""
        with self._lock:
            return request_id in self._remote_ids

    def notify_abort(self, request_id: str) -> bool:
        """Forward an abort upstream; local registry cleanup stays with the caller.

        Args:
            request_id: Origin-side id being aborted. A parent id retires the
                whole group; a sample-child id retires only that sample (the
                other samples keep streaming).

        Returns:
            ``True`` when the id was remotely owned and an ``AbortReq`` was
            sent (the caller's local abort then renders the terminal state);
            ``False`` when the id is unknown to the plane.
        """
        with self._lock:
            if request_id not in self._remote_ids:
                return False
            is_parent = request_id in self._live_children
        self._coordinator.send_to_leader(wire.AbortReq(request_id=request_id))
        if is_parent:
            self._forget(request_id)
        else:
            with self._lock:
                self._remote_ids.discard(request_id)
                parent_id = self._child_to_parent.pop(request_id, request_id)
                live = self._live_children.get(parent_id)
                if live is not None:
                    if live <= 1:
                        self._live_children.pop(parent_id, None)
                        self._remote_ids.discard(parent_id)
                    else:
                        self._live_children[parent_id] = live - 1
        return True

    def forward_stop_hits(self, stop_string_finishes: dict[str, str]) -> dict[str, str]:
        """Send remotely-owned stop-string hits upstream; return the local rest.

        Args:
            stop_string_finishes: ``{sample_request_id: matched_stop}`` from
                the local output pipeline.

        Returns:
            The subset of hits the plane does not own (locally-scheduled
            requests, if any), for the caller's normal stop path.
        """
        with self._lock:
            remote = {rid: s for rid, s in stop_string_finishes.items() if rid in self._remote_ids}
        if remote:
            self._coordinator.send_to_leader(wire.StopHit(hits=remote))
        return {rid: s for rid, s in stop_string_finishes.items() if rid not in remote}

    def fail_pending(self, reason: str) -> None:
        """Wake every blocked admission with a NACK (engine teardown path)."""
        with self._lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for parent_id, slot in pending:
            slot.ack = wire.AdmitAck(request_id=parent_id, ok=False, error=reason)
            slot.event.set()

    def _handle_inbound(self, message: wire.WireMessage, payload: bytes | None) -> None:
        """Worker replay-thread handoff for plane wire messages."""
        if isinstance(message, wire.AdmitAck):
            if message.ok and payload is not None:
                # Coalesced catch-up replay: apply before releasing the
                # admitting caller so it never observes a pre-replay gap.
                outputs, finished = wire.decode_payload(payload)
                self._track_finishes(outputs, finished)
                self._submit_outputs(
                    {
                        0: EngineCoreOutputs(
                            outputs=outputs,
                            timestamp=time.perf_counter(),
                            finished_requests=finished,
                        )
                    }
                )
            with self._lock:
                slot = self._pending.pop(message.request_id, None)
            if slot is not None:
                slot.ack = message
                slot.event.set()
            return
        if isinstance(message, wire.OutputUpdate):
            if payload is None:
                return
            outputs, finished = wire.decode_payload(payload)
            self._track_finishes(outputs, finished)
            # Re-stamp with the local clock: perf_counter is process-local,
            # and the origin's records were stamped at local admission.
            bundle = EngineCoreOutputs(
                outputs=outputs,
                timestamp=time.perf_counter(),
                finished_requests=finished,
            )
            self._submit_outputs({0: bundle})
            return
        logger.warning("Request plane: origin ignoring unexpected %s", type(message).__name__)

    def _track_finishes(self, outputs: list[EngineCoreOutput], finished: set[str] | None) -> None:
        """Retire origin-side ids as their terminal deltas arrive."""
        done_ids = [out.request_id for out in outputs if out.finish_reason is not None]
        if finished:
            done_ids.extend(finished)
        if not done_ids:
            return
        with self._lock:
            for request_id in done_ids:
                if request_id not in self._remote_ids:
                    continue
                self._remote_ids.discard(request_id)
                parent_id = self._child_to_parent.pop(request_id, request_id)
                live = self._live_children.get(parent_id)
                if live is not None:
                    live -= 1
                    if live <= 0:
                        self._live_children.pop(parent_id, None)
                        self._remote_ids.discard(parent_id)
                    else:
                        self._live_children[parent_id] = live


def create_request_plane(
    coordinator,
    *,
    submit_outputs,
    scheduler_submit,
    abort_request,
    apply_stop_strings,
    next_arrival_stamp,
    alive,
    admit_timeout_s: float,
    info,
) -> OwnerRequestPlane | OriginRequestPlane | None:
    """Build the request-plane half matching this rank's coordinator.

    Single-host (or replicated-mode) engines get ``None`` — no plane, no
    hot-path cost. ZMQ leaders get the owner half; ZMQ workers the origin
    half.

    Args:
        coordinator: The engine's step coordinator.
        submit_outputs: Feed a synthesized bundle to the local pipeline
            (origin rendering; owner-side late-attach replays).
        scheduler_submit: Owner: enqueue requests directly under the
            scheduler lock (no plane re-entry).
        abort_request: Owner: cancel one owner-side request id.
        apply_stop_strings: Owner: route mapped stop hits to the stop queue.
        next_arrival_stamp: Owner: next monotonic admission arrival stamp.
        alive: Origin: whether the local generation backend is running.
        admit_timeout_s: Origin: ack deadline for remote admissions.
        info: Engine-level info logger callable.

    Returns:
        The plane instance for this rank, or ``None`` when no plane applies.
    """
    from .zmq_coordinator import ZmqLeaderCoordinator, ZmqWorkerCoordinator

    if isinstance(coordinator, ZmqLeaderCoordinator):
        return OwnerRequestPlane(
            coordinator=coordinator,
            scheduler_submit=scheduler_submit,
            abort_request=abort_request,
            apply_stop_strings=apply_stop_strings,
            submit_local_outputs=submit_outputs,
            next_arrival_stamp=next_arrival_stamp,
            info=info,
        )
    if isinstance(coordinator, ZmqWorkerCoordinator):
        return OriginRequestPlane(
            coordinator=coordinator,
            submit_outputs=submit_outputs,
            alive=alive,
            rank=coordinator.rank,
            admit_timeout_s=admit_timeout_s,
            info=info,
        )
    return None
