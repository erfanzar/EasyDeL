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

Id spaces are disjoint by construction: the owner schedules a remote parent
as ``r{rank}c{counter}-{hash8}`` (its ``n>1`` children keep the shared
``-{sample_idx}`` suffix scheme), so remote admissions can never collide
with owner-local ids, and every delta is translated back to the origin's
ids before it crosses the wire.
"""

from __future__ import annotations

import itertools
import queue
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


class RequestPlaneError(RuntimeError):
    """A remote admission or plane transport failed.

    Raised on the origin when the owner NACKs an admission, the ack deadline
    lapses, or the plane is shut down while requests are pending. Like step
    coordination, the plane is fail-fast: there is no admission retry.
    """


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


class OwnerRequestPlane:
    """Leader-side request plane: admit remote requests, tee outputs back.

    All state below is mutated exclusively on the ingest thread (socket
    handoffs and engine tees are queue puts), so the id maps need no lock;
    ``has_remote`` is a plain bool published for the engine-loop hot path.

    Args:
        coordinator: The leader step coordinator (targeted sends).
        scheduler_submit: Engine callable enqueueing a batch of
            ``EngineRequest`` under the scheduler lock.
        abort_request: Engine callable cancelling one request id.
        apply_stop_strings: Engine callable routing
            ``{request_id: stop_string}`` to the scheduler-safe stop queue.
        next_arrival_stamp: Engine-admission callable producing the next
            monotonic arrival stamp, so remote admissions interleave FCFS
            with owner-local ones.
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
        next_arrival_stamp: typing.Callable[[], float],
        info: typing.Callable[..., None],
    ) -> None:
        self._coordinator = coordinator
        self._scheduler_submit = scheduler_submit
        self._abort_request = abort_request
        self._apply_stop_strings = apply_stop_strings
        self._next_arrival_stamp = next_arrival_stamp
        self._info = info

        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._out_seq = 0

        # Owner scheduler id (per-sample child) -> (identity, origin id).
        self._owner_to_origin: dict[str, tuple[bytes, str]] = {}
        # (identity, origin id) -> owner id, for AbortReq/StopHit translation.
        self._origin_to_owner: dict[tuple[bytes, str], str] = {}
        # Owner parent id -> live owner child ids (drives map retirement).
        self._children: dict[str, set[str]] = {}
        # Owner child id -> owner parent id; parent id -> (identity, origin parent).
        self._owner_child_to_parent: dict[str, str] = {}
        self._parents: dict[str, tuple[bytes, str]] = {}

        #: Single-bool hot-path gate: the engine loop tees outputs only when
        #: at least one remotely-admitted request is live.
        self.has_remote = False

        coordinator.set_plane_handler(self.handle_inbound)

    # ------------------------------------------------------------- lifecycle
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
        """Drop every remote mapping (model-weight swaps clear all requests)."""
        self._queue.put(("reset", None, None, None))

    # -------------------------------------------------------------- handoffs
    def handle_inbound(self, identity: bytes, message: wire.WireMessage, payload: bytes | None) -> None:
        """Coordinator IO-thread handoff for plane wire messages."""
        self._queue.put(("wire", identity, message, payload))

    def tee_outputs(self, engine_outputs) -> None:
        """Engine-loop handoff: fan this step's outputs back to origins.

        Called once per step when :attr:`has_remote` is set; the bundle walk
        happens on the ingest thread, keeping the scheduler hot path at one
        bool read plus one queue put.
        """
        self._queue.put(("outputs", None, engine_outputs, None))

    # ---------------------------------------------------------- ingest thread
    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                kind, identity, message, payload = item
                if kind == "wire":
                    self._dispatch_wire(identity, message, payload)
                elif kind == "outputs":
                    self._fan_out(message)
                elif kind == "reset":
                    self._owner_to_origin.clear()
                    self._origin_to_owner.clear()
                    self._children.clear()
                    self._owner_child_to_parent.clear()
                    self._parents.clear()
                    self.has_remote = False
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

    def _handle_admit(self, identity: bytes, admit: wire.Admit, payload: bytes | None) -> None:
        """Rename the origin's requests into the owner id space and schedule them."""
        try:
            if payload is None:
                raise RequestPlaneError("Admit carried no request payload")
            requests: list[EngineRequest] = wire.decode_payload(payload)
            origin_parent = admit.request_id
            hash8 = (admit.content_hash or "00000000")[:8]
            owner_parent = f"r{admit.origin_rank}c{admit.origin_counter}-{hash8}"
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
                suffix = request.request_id[len(origin_parent) :]
                request.request_id = owner_parent + suffix
                if request.parent_request_id is not None:
                    request.parent_request_id = owner_parent
                request.arrival_time = self._next_arrival_stamp()

            self._scheduler_submit(requests)

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
            self.has_remote = True
        except Exception as exc:
            logger.exception("Request plane: remote admission failed for %r", admit.request_id)
            self._coordinator.send_to_peer(
                identity,
                wire.AdmitAck(request_id=admit.request_id, ok=False, error=str(exc)),
            )
            return
        self._coordinator.send_to_peer(identity, wire.AdmitAck(request_id=admit.request_id, ok=True))

    def _handle_abort(self, identity: bytes, message: wire.AbortReq) -> None:
        owner_id = self._origin_to_owner.get((identity, message.request_id))
        if owner_id is None:
            return
        try:
            self._abort_request(owner_id)
        except Exception:
            logger.exception("Request plane: remote abort failed for %r", owner_id)
        # The origin already rendered its local abort; retire the mappings now
        # so no further deltas are built for the group. A parent-level abort
        # covers every sample child.
        for child_id in list(self._children.get(owner_id, ())):
            self._retire(child_id)
        self._retire(owner_id)

    def _handle_stop_hit(self, identity: bytes, message: wire.StopHit) -> None:
        mapped = {}
        for origin_id, stop_string in message.hits.items():
            owner_id = self._origin_to_owner.get((identity, origin_id))
            if owner_id is not None:
                mapped[owner_id] = stop_string
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
        if not self._owner_to_origin:
            self.has_remote = False

    def _fan_out(self, engine_outputs) -> None:
        """Translate one step's outputs into per-origin ``OutputUpdate`` deltas."""
        if not engine_outputs:
            return
        per_origin: dict[bytes, tuple[list[EngineCoreOutput], set[str]]] = {}
        finished_owner_ids: list[str] = []
        for client_outputs in engine_outputs.values():
            for engine_output in client_outputs.outputs:
                entry = self._owner_to_origin.get(engine_output.request_id)
                if entry is None:
                    continue
                identity, origin_id = entry
                outs, _fin = per_origin.setdefault(identity, ([], set()))
                outs.append(
                    EngineCoreOutput(
                        request_id=origin_id,
                        new_token_ids=engine_output.new_token_ids,
                        new_logprobs=engine_output.new_logprobs,
                        new_prompt_logprobs_tensors=engine_output.new_prompt_logprobs_tensors,
                        finish_reason=engine_output.finish_reason,
                        stop_reason=engine_output.stop_reason,
                        num_cached_tokens=engine_output.num_cached_tokens,
                    )
                )
                if engine_output.finish_reason is not None:
                    finished_owner_ids.append(engine_output.request_id)
            for finished_id in client_outputs.finished_requests or ():
                entry = self._owner_to_origin.get(finished_id)
                if entry is None:
                    continue
                identity, origin_id = entry
                _outs, fin = per_origin.setdefault(identity, ([], set()))
                fin.add(origin_id)
                finished_owner_ids.append(finished_id)

        for identity, (outs, fin) in per_origin.items():
            self._out_seq += 1
            self._coordinator.send_to_peer(
                identity,
                wire.OutputUpdate(seq=self._out_seq),
                wire.encode_payload((outs, fin or None)),
            )
        for owner_id in finished_owner_ids:
            self._retire(owner_id)


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
        self._pending: dict[str, _PendingAdmit] = {}
        # Origin-side ids (parents and sample children) currently owned remotely.
        self._remote_ids: set[str] = set()
        # Origin parent id -> live child count (drives id retirement).
        self._live_children: dict[str, int] = {}
        self._child_to_parent: dict[str, str] = {}

        coordinator.set_plane_handler(self._handle_inbound)

    # -------------------------------------------------------------- admission
    def submit_remote(self, scheduler_requests: list[EngineRequest]) -> None:
        """Forward admission-built requests to the owner and await its acks.

        Called on the admitting caller's thread with no engine locks held.
        Registers the request ids as remotely-owned *before* sending so
        abort/stop forwarding sees them from the first token.

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
            content_hash = compute_admission_key(
                first.prompt_token_ids,
                first.sampling_params,
                len(requests),
            )
            slot = _PendingAdmit()
            with self._lock:
                self._pending[parent_id] = slot
                self._remote_ids.add(parent_id)
                self._live_children[parent_id] = len(requests)
                for request in requests:
                    self._remote_ids.add(request.request_id)
                    self._child_to_parent[request.request_id] = parent_id
            self._coordinator.send_to_leader(
                wire.Admit(
                    request_id=parent_id,
                    origin_rank=self._rank,
                    origin_counter=next(self._counter),
                    content_hash=content_hash,
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

    # ------------------------------------------------------------ engine forks
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

    # ------------------------------------------------------------ inbound path
    def _handle_inbound(self, message: wire.WireMessage, payload: bytes | None) -> None:
        """Worker replay-thread handoff for plane wire messages."""
        if isinstance(message, wire.AdmitAck):
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
        submit_outputs: Origin: feed a synthesized bundle to the pipeline.
        scheduler_submit: Owner: enqueue requests under the scheduler lock.
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
