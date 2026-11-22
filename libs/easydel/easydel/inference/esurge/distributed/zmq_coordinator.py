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

"""ZeroMQ implementation of the multi-host step-coordination plane.

Topology: the leader (rank 0) binds one ROUTER socket; every worker
connects a DEALER with identity ``rank-<i>``. Per-connection FIFO ordering
is the correctness primitive — the leader's command stream *is* the
replicated program, and each worker replays it verbatim: ``Step(ASYNC)``
dispatches, ``Step(SYNC)`` executes-and-acks, ``Drain`` materializes, in
exactly the order the leader's engine loop produced them.

The leader never waits a network round trip on the step path: commands are
handed to a dedicated IO thread (ZMQ sockets are single-threaded) and
worker acknowledgements are consumed asynchronously into a ledger. Flow
control bounds leader/worker skew to ``max_inflight_steps``; ack deadlines
and heartbeats turn silent worker death into a
:class:`StepCoordinationError` within one timeout instead of a hang.

Failure semantics are fail-fast: a JAX collective cannot survive a lost
participant, so any NACK, deadline, heartbeat loss, or digest mismatch
aborts the whole pod cleanly. There is no step retry and no request
migration.
"""

from __future__ import annotations

import hmac
import itertools
import queue
import threading
import time
import typing

from ..logger import logger
from . import wire
from .coordinator import StepCoordinationError, StepHandle, format_tcp_endpoint
from .protocol import compute_sampled_digest

if typing.TYPE_CHECKING:
    from ..outputs import ModelRunnerOutput
    from ..scheduler.output import SchedulerOutput

_POLL_MS = 50

# Distinct inproc endpoints per IO loop, so a restarted coordinator never rebinds a name a closing socket still holds.
_LEADER_WAKE_IDS = itertools.count()
_WORKER_WAKE_IDS = itertools.count()

# Outbox destination sentinels: the step stream broadcasts to worker ranks
# only; _DEST_ALL additionally covers request-plane clients (shutdown).
_DEST_WORKERS = object()
_DEST_ALL = object()


def _configure_zmq_socket(sock, zmq) -> None:
    """Apply common transport options: no silent HWM drops, dual-stack IPv6 support, bounded linger."""
    sock.setsockopt(zmq.LINGER, 500)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    try:
        sock.setsockopt(zmq.IPV6, 1)
    except zmq.ZMQError:
        pass


def _worker_identity(rank: int) -> bytes:
    return f"rank-{rank}".encode()


class _StepLedger:
    """Leader-side record of expected and received worker acknowledgements."""

    def __init__(self, worker_ranks: list[int], *, step_timeout_s: float, heartbeat_timeout_s: float) -> None:
        self.cond = threading.Condition()
        self.step_timeout_s = float(step_timeout_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.last_acked: dict[int, int] = {rank: 0 for rank in worker_ranks}
        self.deadlines: dict[tuple[int, int], float] = {}
        self.last_heartbeat: dict[int, float] = {rank: time.monotonic() for rank in worker_ranks}
        self.failure: str | None = None
        # step_id -> {"leader": (num_reqs, digest) | None, "workers": {rank: (num_reqs, digest)}}
        self.pending_checks: dict[int, dict] = {}

    def expect(self, step_id: int, ranks: typing.Iterable[int]) -> None:
        with self.cond:
            deadline = time.monotonic() + self.step_timeout_s
            for rank in ranks:
                self.deadlines[(step_id, rank)] = deadline

    def record_leader_result(self, step_id: int, num_reqs: int, digest: str | None) -> None:
        with self.cond:
            entry = self.pending_checks.setdefault(step_id, {"leader": None, "workers": {}})
            entry["leader"] = (num_reqs, digest)
            self._compare_locked(step_id)

    def record_ack(self, ack: wire.StepAck) -> None:
        with self.cond:
            self.deadlines.pop((ack.step_id, ack.rank), None)
            # Completion is also proof of liveness after blocking runner work.
            self.last_heartbeat[ack.rank] = time.monotonic()
            if not ack.ok:
                self.failure = f"worker rank={ack.rank} failed step {ack.step_id}: {ack.error}\n{ack.traceback or ''}"
                self.cond.notify_all()
                return
            self.last_acked[ack.rank] = max(self.last_acked.get(ack.rank, 0), ack.step_id)
            if ack.num_reqs >= 0 or ack.digest is not None:
                entry = self.pending_checks.setdefault(ack.step_id, {"leader": None, "workers": {}})
                entry["workers"][ack.rank] = (ack.num_reqs, ack.digest)
                self._compare_locked(ack.step_id)
            self.cond.notify_all()

    def _compare_locked(self, step_id: int) -> None:
        entry = self.pending_checks.get(step_id)
        if entry is None or entry["leader"] is None:
            return
        leader_num, leader_digest = entry["leader"]
        for rank, (num_reqs, digest) in list(entry["workers"].items()):
            if num_reqs >= 0 and leader_num >= 0 and num_reqs != leader_num:
                self.failure = (
                    f"lockstep divergence at step {step_id}: leader num_reqs={leader_num}, "
                    f"worker rank={rank} num_reqs={num_reqs}"
                )
            if digest is not None and leader_digest is not None and digest != leader_digest:
                self.failure = (
                    f"sampled-token digest mismatch at step {step_id}: leader={leader_digest}, "
                    f"worker rank={rank} digest={digest}"
                )
        if all(rank in entry["workers"] for rank in self.last_acked):
            self.pending_checks.pop(step_id, None)

    def record_heartbeat(self, rank: int) -> None:
        with self.cond:
            self.last_heartbeat[rank] = time.monotonic()

    def min_acked(self) -> int:
        with self.cond:
            return min(self.last_acked.values()) if self.last_acked else 0

    def wait_for_window(self, step_id: int, max_inflight: int) -> None:
        deadline = time.monotonic() + self.step_timeout_s
        with self.cond:
            while self.failure is None:
                lowest = min(self.last_acked.values()) if self.last_acked else step_id
                if step_id - lowest <= max_inflight:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failure = (
                        f"flow-control window stalled before step {step_id}: slowest worker acked {lowest}, "
                        f"window={max_inflight}, timeout={self.step_timeout_s}s"
                    )
                    break
                self.cond.wait(timeout=min(remaining, 0.25))
            raise StepCoordinationError(self.failure)

    def sweep(self) -> None:
        now = time.monotonic()
        with self.cond:
            if self.failure is not None:
                return
            for (step_id, rank), deadline in list(self.deadlines.items()):
                if now > deadline:
                    self.failure = (
                        f"worker rank={rank} missed the ack deadline for step {step_id} "
                        f"({self.step_timeout_s}s); assuming it is wedged or dead"
                    )
                    self.cond.notify_all()
                    return
            busy_ranks = {rank for _, rank in self.deadlines}
            for rank, beat in self.last_heartbeat.items():
                # The replay thread cannot heartbeat during synchronous device
                # work or drains. Outstanding work has its own step deadline.
                if rank not in busy_ranks and now - beat > self.heartbeat_timeout_s:
                    self.failure = (
                        f"worker rank={rank} stopped heartbeating ({self.heartbeat_timeout_s}s); "
                        "assuming the process died"
                    )
                    self.cond.notify_all()
                    return

    def raise_if_failed(self) -> None:
        with self.cond:
            if self.failure is not None:
                raise StepCoordinationError(self.failure)


class ZmqLeaderCoordinator:
    """Leader-side coordinator: replicate the step stream, verify off-path.

    Args:
        runner: The engine's runner (local execution).
        world_size: Total process count (workers = ``world_size - 1``).
        bind_host: Interface the ROUTER binds on.
        control_port: TCP port of the control plane.
        auth_token: Shared secret validated in every worker's Hello.
        config_fingerprint: Engine-config fingerprint workers must match.
        ready_timeout_s: How long ``start()`` waits for all workers.
        step_timeout_s: Ack deadline per dispatched step.
        heartbeat_interval_s: Leader beacon cadence.
        heartbeat_timeout_s: Worker silence tolerated before aborting.
        verify_digest_interval: Sampled-digest check every K steps (0 = off).
        max_inflight_steps: Leader/worker skew bound.
    """

    is_leader = True
    rank = 0

    def __init__(
        self,
        runner,
        *,
        world_size: int,
        bind_host: str = "0.0.0.0",
        control_port: int = 19666,
        auth_token: str,
        config_fingerprint: str,
        ready_timeout_s: float = 600.0,
        step_timeout_s: float = 30.0,
        heartbeat_interval_s: float = 1.0,
        heartbeat_timeout_s: float = 5.0,
        verify_digest_interval: int = 64,
        max_inflight_steps: int = 4,
        engine_spec: wire.EngineSpec | None = None,
        plane_handler: typing.Callable[[bytes, wire.WireMessage, bytes | None], None] | None = None,
    ) -> None:
        self._runner = runner
        self.world_size = int(world_size)
        self._bind_endpoint = format_tcp_endpoint(bind_host, int(control_port))
        self._auth_token = str(auth_token)
        self._config_fingerprint = str(config_fingerprint)
        self._ready_timeout_s = float(ready_timeout_s)
        self._step_timeout_s = float(step_timeout_s)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._verify_digest_interval = int(verify_digest_interval)
        self._max_inflight = max(1, int(max_inflight_steps))
        self._engine_spec = engine_spec
        self._plane_handler = plane_handler
        self._client_evict_handler: typing.Callable[[bytes], None] | None = None
        self._stats_provider: typing.Callable[[], wire.QueueStats] | None = None

        self._outbox: queue.Queue = queue.Queue()
        self._io_thread: threading.Thread | None = None
        self._stop_io = threading.Event()
        # Caller threads hand frames to the IO thread (the ROUTER socket's only user) through the outbox and poke an
        # inproc wakeup socket that interrupts its poll, so a queued Step or Drain leaves at once instead of after the
        # poll interval: workers start a step only when its command arrives, and the leader's devices wait for them
        # in the step's first collective. The IO loop owns the wakeup sockets; _wake_push is None while it is down.
        self._wake_push = None
        self._wake_lock = threading.Lock()
        self._client_identities: dict[str, bytes] = {}
        self._next_step_id = 0
        self._undrained_async: set[int] = set()
        self._reset_control_state()

    @property
    def supports_overlap(self) -> bool:
        """Whether overlap execution is safe (multi-host overlap dispatches step N+1 before draining step N)."""
        return self.world_size <= 1 or self._max_inflight >= 2

    def _reset_control_state(self) -> None:
        """Fresh handshake/ledger state so a leader can start() again after shutdown()."""
        worker_ranks = list(range(1, self.world_size))
        self._ledger = _StepLedger(
            worker_ranks,
            step_timeout_s=self._step_timeout_s,
            heartbeat_timeout_s=self._heartbeat_timeout_s,
        )
        self._next_step_id = 0
        self._undrained_async = set()
        self._helloed: set[int] = set()
        self._ready: set[int] = set()
        self._bound_event = threading.Event()
        self._bind_error: BaseException | None = None
        self._ready_event = threading.Event()
        self._hello_error: str | None = None
        self._client_identities = {}
        self._stop_io.clear()
        while True:
            try:
                self._outbox.get_nowait()
            except queue.Empty:
                break

    def start(self) -> None:
        """Bind, then block until every worker has helloed and readied.

        Restartable: after :meth:`shutdown` a subsequent ``start()`` resets
        the handshake/ledger state and waits for a fresh worker handshake
        (needed by engine weight hot-swaps, which terminate and re-initiate).

        A ``world_size == 1`` leader has no workers to wait for — it exists
        purely to serve request-plane clients (DP replica engines) and is
        ready as soon as its control socket is bound.
        """
        if self._io_thread is not None and self._io_thread.is_alive():
            return
        self._io_thread = None
        self._reset_control_state()
        self._io_thread = threading.Thread(target=self._io_loop, name="eSurgeCoordLeaderIO", daemon=True)
        self._io_thread.start()
        if not self._bound_event.wait(self._ready_timeout_s):
            self.shutdown("bind timeout")
            raise StepCoordinationError(
                f"timed out after {self._ready_timeout_s}s binding control plane at {self._bind_endpoint}"
            )
        if self._bind_error is not None:
            bind_error = self._bind_error
            self.shutdown("bind failed")
            raise StepCoordinationError(
                f"failed to bind control plane at {self._bind_endpoint}: {bind_error}"
            ) from bind_error
        if not self._ready_event.wait(self._ready_timeout_s):
            helloed = sorted(self._helloed)
            ready = sorted(self._ready)
            self.shutdown("ready timeout")
            raise StepCoordinationError(
                f"timed out after {self._ready_timeout_s}s waiting for {self.world_size - 1} worker(s); "
                f"helloed={helloed}, ready={ready}"
            )
        if self._hello_error is not None:
            hello_error = self._hello_error
            self.shutdown("handshake error")
            raise StepCoordinationError(hello_error)
        logger.info("eSurge coordinator: %d worker(s) ready.", self.world_size - 1)

    def set_plane_handler(self, handler: typing.Callable[[bytes, wire.WireMessage, bytes | None], None] | None) -> None:
        """Install the request-plane inbound handler (Admit/AbortReq/StopHit/...)."""
        self._plane_handler = handler

    def set_client_evict_handler(self, handler: typing.Callable[[bytes], None] | None) -> None:
        """Install the callback notified when a silent or replaced request-plane client is evicted."""
        self._client_evict_handler = handler

    def _notify_client_evicted(self, identity: bytes) -> None:
        handler = self._client_evict_handler
        if handler is None:
            return
        try:
            handler(identity)
        except Exception:
            logger.exception("eSurge coordinator: client-evict handler failed for %r", identity)

    def set_stats_provider(self, provider: typing.Callable[[], wire.QueueStats] | None) -> None:
        """Install the queue-stats snapshot provider.

        When set, every client-directed heartbeat is followed by a
        :class:`wire.QueueStats` so routers can join-shortest-queue without
        polling. The provider runs on the IO thread and must be lock-free
        (best-effort reads of live scheduler state).
        """
        self._stats_provider = provider

    def send_to_peer(self, identity: bytes, header: wire.WireMessage, payload: bytes | None = None) -> None:
        """Queue a message for one specific peer (worker or client) identity."""
        self._outbox.put((identity, wire.encode_message(header), payload))
        self._wake_io()

    def _wake_io(self) -> None:
        """Interrupt the IO loop's poll so the frames just queued in the outbox leave now (callable from any thread)."""
        with self._wake_lock:
            if self._wake_push is not None:
                try:
                    self._wake_push.send(b"", flags=1)  # zmq.DONTWAIT
                except Exception:
                    pass

    def client_identity(self, client_id: str) -> bytes | None:
        """Resolve a request-plane client's ZMQ identity, if connected."""
        return self._client_identities.get(client_id)

    def _want_digest(self, step_id: int) -> bool:
        interval = self._verify_digest_interval
        return interval > 0 and step_id % interval == 0

    def _broadcast_step(self, step_id: int, mode: int, scheduler_output, want_digest: bool) -> None:
        if self.world_size <= 1:
            # Client-serving leader with no step-replay workers: nothing to
            # mirror, so skip the payload pickling entirely.
            return
        self._ledger.raise_if_failed()
        if len(self._undrained_async) >= self._max_inflight:
            raise StepCoordinationError(
                f"cannot dispatch step {step_id}: {len(self._undrained_async)} undrained async step(s) "
                f"already fill max_inflight_steps={self._max_inflight}; workers cannot ack async steps "
                "until Drain is sent (set distributed_max_inflight_steps >= 2 for overlap execution)"
            )
        self._ledger.wait_for_window(step_id, self._max_inflight)
        header = wire.Step(step_id=step_id, mode=mode, want_digest=want_digest)
        payload = wire.encode_payload(scheduler_output)
        self._ledger.expect(step_id, self._ledger.last_acked.keys())
        self._outbox.put((_DEST_WORKERS, wire.encode_message(header), payload))
        self._wake_io()

    def execute_sync(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Mirror STEP(SYNC) to all workers, then execute locally."""
        step_id = self._next_step_id + 1
        want_digest = self._want_digest(step_id)
        self._broadcast_step(step_id, wire.STEP_MODE_SYNC, scheduler_output, want_digest)
        self._next_step_id = step_id
        model_output = self._runner.execute_model(scheduler_output)
        self._record_local(step_id, model_output, want_digest)
        return model_output

    def execute_async(self, scheduler_output: SchedulerOutput) -> StepHandle:
        """Mirror STEP(ASYNC) to all workers, then dispatch locally."""
        step_id = self._next_step_id + 1
        self._broadcast_step(step_id, wire.STEP_MODE_ASYNC, scheduler_output, want_digest=False)
        self._next_step_id = step_id
        if self.world_size > 1:
            self._undrained_async.add(step_id)
        runner_handle = self._runner.execute_model_async(scheduler_output)
        return StepHandle(step_id=step_id, runner_handle=runner_handle, scheduler_output=scheduler_output)

    def drain(self, handle: StepHandle) -> ModelRunnerOutput:
        """Mirror DRAIN to all workers, then materialize locally."""
        if self.world_size > 1:
            self._ledger.raise_if_failed()
            self._undrained_async.discard(handle.step_id)
            want_digest = self._want_digest(handle.step_id)
            header = wire.Drain(step_id=handle.step_id, want_digest=want_digest)
            self._ledger.expect(handle.step_id, self._ledger.last_acked.keys())
            self._outbox.put((_DEST_WORKERS, wire.encode_message(header), None))
            self._wake_io()
        else:
            want_digest = False
        model_output = self._runner.wait_for_execution(handle.runner_handle)
        self._record_local(handle.step_id, model_output, want_digest)
        return model_output

    def _record_local(self, step_id: int, model_output, want_digest: bool) -> None:
        if self.world_size <= 1:
            return
        num_reqs = len(getattr(model_output, "req_ids", []) or [])
        digest = None
        if want_digest:
            digest = compute_sampled_digest(model_output.req_ids, model_output.sampled_token_ids)
        self._ledger.record_leader_result(step_id, num_reqs, digest)

    def check_health(self) -> None:
        """Raise if any worker NACKed, timed out, diverged, or died."""
        self._ledger.raise_if_failed()

    def run_worker_loop(self, stop_event) -> None:
        """Leaders do not replay a step stream."""
        raise StepCoordinationError("run_worker_loop called on the leader coordinator.")

    def shutdown(self, reason: str = "") -> None:
        """Broadcast Shutdown, then stop the IO thread and close the socket."""
        thread = self._io_thread
        if thread is None:
            return
        if thread.is_alive():
            self._outbox.put((_DEST_ALL, wire.encode_message(wire.Shutdown(reason=reason)), None))
            self._wake_io()
            time.sleep(0.05)
            self._stop_io.set()
            self._wake_io()
            thread.join(timeout=5.0)
        if thread.is_alive():
            logger.warning("eSurge coordinator: leader IO thread did not exit within 5.0s")
            return
        if self._io_thread is thread:
            self._io_thread = None

    def _handle_hello(
        self,
        sock,
        identity: bytes,
        message: wire.Hello,
        worker_identities: dict[int, bytes],
        worker_authed: dict[bytes, int],
        client_authed: dict[bytes, str],
        client_last_seen: dict[bytes, float],
    ) -> None:
        """Validate a Hello and register the peer (worker or plane client)."""
        if not hmac.compare_digest(str(message.auth), self._auth_token):
            logger.warning("eSurge coordinator: rejecting peer with bad auth token")
            return
        if message.role == wire.ROLE_CLIENT:
            client_id = str(message.client_id or "").strip()
            if not client_id:
                logger.warning("eSurge coordinator: rejecting ROLE_CLIENT peer with empty client_id")
                return
            failure = self._ledger.failure
            if failure is not None:
                # This IO thread outlives a failed world until the process aborts; answering HelloOk would let a
                # router re-attach a dead engine. Refuse with Shutdown so the client treats it as down.
                sock.send_multipart(
                    [identity, wire.encode_message(wire.Shutdown(reason=f"engine world failed: {failure}"))]
                )
                return
            # Request-plane clients attach to whatever engine the owner runs:
            # no world/fingerprint requirements, never counted toward Ready.
            prev_identity = self._client_identities.get(client_id)
            if prev_identity is not None and prev_identity != identity:
                client_authed.pop(prev_identity, None)
                client_last_seen.pop(prev_identity, None)
                self._notify_client_evicted(prev_identity)
            self._client_identities[client_id] = identity
            client_authed[identity] = client_id
            client_last_seen[identity] = time.monotonic()
            sock.send_multipart([identity, wire.encode_message(wire.HelloOk(rank=-1, engine_spec=self._engine_spec))])
            return
        if message.role != wire.ROLE_WORKER:
            logger.warning("eSurge coordinator: rejecting peer with unknown role %r", message.role)
            return
        if message.world_size != self.world_size:
            self._hello_error = (
                f"worker rank={message.rank} reports world_size={message.world_size}, leader expects {self.world_size}"
            )
            self._ready_event.set()
            return
        if not (1 <= message.rank < self.world_size):
            logger.warning(
                "eSurge coordinator: rejecting worker Hello with invalid rank=%r for world_size=%d",
                message.rank,
                self.world_size,
            )
            return
        if message.config_fingerprint != self._config_fingerprint:
            self._hello_error = (
                f"worker rank={message.rank} config mismatch: {message.config_fingerprint} != {self._config_fingerprint}"
            )
            self._ready_event.set()
            return
        prev_worker = worker_identities.get(message.rank)
        if prev_worker is not None and prev_worker != identity:
            worker_authed.pop(prev_worker, None)
        worker_identities[message.rank] = identity
        worker_authed[identity] = message.rank
        self._helloed.add(message.rank)
        sock.send_multipart(
            [identity, wire.encode_message(wire.HelloOk(rank=message.rank, engine_spec=self._engine_spec))]
        )

    def _io_loop(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = None
        wake_pull = None
        wake_push = None
        try:
            sock = ctx.socket(zmq.ROUTER)
            _configure_zmq_socket(sock, zmq)
            sock.bind(self._bind_endpoint)
            wake_endpoint = f"inproc://esurge-leader-wake-{next(_LEADER_WAKE_IDS)}"
            wake_pull = ctx.socket(zmq.PULL)
            wake_pull.bind(wake_endpoint)
            wake_push = ctx.socket(zmq.PUSH)
            wake_push.connect(wake_endpoint)
        except BaseException as exc:
            self._bind_error = exc
            if wake_push is not None:
                wake_push.close(linger=0)
            if wake_pull is not None:
                wake_pull.close(linger=0)
            if sock is not None:
                sock.close(linger=0)
            self._bound_event.set()
            self._ready_event.set()
            return

        with self._wake_lock:
            self._wake_push = wake_push
        self._bound_event.set()
        if self.world_size <= 1:
            self._ready_event.set()

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        poller.register(wake_pull, zmq.POLLIN)
        worker_identities: dict[int, bytes] = {}
        worker_authed: dict[bytes, int] = {}
        client_authed: dict[bytes, str] = {}
        client_last_seen: dict[bytes, float] = {}
        expected_worker_ranks = set(range(1, self.world_size))
        last_beat = 0.0
        last_sweep = 0.0
        failure_announced = False

        def _send_to(identity: bytes, header: bytes, payload: bytes | None) -> None:
            frames = [identity, header]
            if payload is not None:
                frames.append(payload)
            sock.send_multipart(frames)

        def _drain_outbox() -> None:
            # Outbound in FIFO order. Destinations: _DEST_WORKERS broadcasts
            # the step stream to worker ranks, _DEST_ALL additionally covers
            # plane clients (shutdown), an identity targets one peer.
            try:
                while True:
                    dest, header, payload = self._outbox.get_nowait()
                    if dest is _DEST_WORKERS or dest is _DEST_ALL:
                        for identity in worker_identities.values():
                            _send_to(identity, header, payload)
                        if dest is _DEST_ALL:
                            for identity in self._client_identities.values():
                                _send_to(identity, header, payload)
                    else:
                        _send_to(dest, header, payload)
            except queue.Empty:
                pass

        try:
            while not self._stop_io.is_set():
                _drain_outbox()

                now = time.monotonic()
                if now - last_beat >= self._heartbeat_interval_s and (worker_identities or self._client_identities):
                    beat = wire.encode_message(wire.Heartbeat(rank=0, ts=time.time()))
                    for identity in worker_identities.values():
                        sock.send_multipart([identity, beat])
                    stats_frame = None
                    if self._client_identities and self._stats_provider is not None:
                        try:
                            stats_frame = wire.encode_message(self._stats_provider())
                        except Exception:
                            logger.debug("eSurge coordinator: stats provider failed", exc_info=True)
                    for identity in self._client_identities.values():
                        sock.send_multipart([identity, beat])
                        if stats_frame is not None:
                            sock.send_multipart([identity, stats_frame])
                    last_beat = now
                if now - last_sweep >= 0.5:
                    # Evict disconnected request-plane clients whose heartbeats stopped.
                    for ident, seen_at in list(client_last_seen.items()):
                        if now - seen_at > self._heartbeat_timeout_s:
                            client_last_seen.pop(ident, None)
                            cid = client_authed.pop(ident, None)
                            if cid is not None and self._client_identities.get(cid) == ident:
                                self._client_identities.pop(cid, None)
                            self._notify_client_evicted(ident)
                            logger.info("eSurge coordinator: evicted silent request-plane client %r", cid)
                    if self._ready_event.is_set():
                        self._ledger.sweep()
                        failure = self._ledger.failure
                        if failure is not None and not failure_announced and self._client_identities:
                            # The step thread may be blocked inside a device collective with a dead worker and
                            # never return to raise; this IO thread keeps heartbeating meanwhile. Tell request-plane
                            # clients now, so a router stops routing here and fails in-flight requests at once.
                            notice = wire.encode_message(wire.Shutdown(reason=f"engine world failed: {failure}"))
                            for identity in self._client_identities.values():
                                sock.send_multipart([identity, notice])
                            failure_announced = True
                            logger.error("eSurge coordinator: told request-plane clients the world failed: %s", failure)
                    last_sweep = now

                events = dict(poller.poll(_POLL_MS))
                if wake_pull in events:
                    while wake_pull.poll(0):
                        wake_pull.recv()
                if sock not in events:
                    continue
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.warning("eSurge coordinator: dropping malformed %d-frame ROUTER message", len(frames))
                    continue
                identity, header_bytes = frames[0], frames[1]
                try:
                    message = wire.decode_message(header_bytes)
                except Exception:
                    logger.warning("eSurge coordinator: dropping undecodable frame from %r", identity)
                    continue

                if isinstance(message, wire.Hello):
                    self._handle_hello(
                        sock,
                        identity,
                        message,
                        worker_identities,
                        worker_authed,
                        client_authed,
                        client_last_seen,
                    )
                    continue

                worker_rank = worker_authed.get(identity)
                is_client = identity in client_authed
                if worker_rank is None and not is_client:
                    logger.warning("eSurge coordinator: dropping frame from unauthenticated peer")
                    continue

                if isinstance(message, wire.Ready):
                    if worker_rank is None or message.rank != worker_rank or worker_identities.get(worker_rank) != identity:
                        logger.warning(
                            "eSurge coordinator: rejecting forged Ready(rank=%r) from %r",
                            message.rank,
                            identity,
                        )
                        continue
                    self._ready.add(worker_rank)
                    self._ledger.record_heartbeat(worker_rank)
                    if self._ready == expected_worker_ranks:
                        self._ready_event.set()
                elif isinstance(message, wire.StepAck):
                    if worker_rank is None or message.rank != worker_rank or worker_identities.get(worker_rank) != identity:
                        logger.warning(
                            "eSurge coordinator: rejecting forged StepAck(rank=%r) from %r",
                            message.rank,
                            identity,
                        )
                        continue
                    self._ledger.record_ack(message)
                elif isinstance(message, wire.Heartbeat):
                    if worker_rank is not None:
                        if message.rank != worker_rank or worker_identities.get(worker_rank) != identity:
                            logger.warning(
                                "eSurge coordinator: rejecting forged worker Heartbeat(rank=%r) from %r",
                                message.rank,
                                identity,
                            )
                            continue
                        self._ledger.record_heartbeat(worker_rank)
                    elif is_client:
                        client_last_seen[identity] = time.monotonic()
                elif self._plane_handler is not None:
                    if is_client:
                        client_last_seen[identity] = time.monotonic()
                    # Request-plane traffic (Admit/AbortReq/StopHit/...): hand
                    # off untouched — engine work never runs on the IO thread.
                    payload = frames[2] if len(frames) > 2 else None
                    try:
                        self._plane_handler(identity, message, payload)
                    except Exception:
                        logger.exception("eSurge coordinator: plane handler failed")
        finally:
            # Drain once more before closing: shutdown() enqueues the
            # Shutdown broadcast and then sets _stop_io within one poll
            # interval, so the top-of-loop drain can miss it. LINGER on
            # close() flushes the queued frames to peers.
            try:
                _drain_outbox()
            except Exception:
                logger.debug("eSurge coordinator: final outbox drain failed during shutdown", exc_info=True)
            with self._wake_lock:
                if self._wake_push is wake_push:
                    self._wake_push = None
            for s, linger in ((wake_push, 0), (wake_pull, 0), (sock, 500)):
                try:
                    s.close(linger=linger)
                except Exception:
                    pass


class ZmqWorkerCoordinator:
    """Worker-side coordinator: replay the leader's step stream verbatim.

    Args:
        runner: The engine's runner (local execution).
        rank: This process's rank (>= 1).
        world_size: Total process count.
        leader_addr: Leader host/IP to connect to.
        control_port: TCP port of the control plane.
        auth_token: Shared secret sent in Hello.
        config_fingerprint: This engine's config fingerprint.
        connect_timeout_s: HelloOk wait budget.
        heartbeat_interval_s: Worker beacon cadence.
        heartbeat_timeout_s: Leader silence tolerated before exiting.
    """

    supports_overlap = True
    is_leader = False

    def __init__(
        self,
        runner,
        *,
        rank: int,
        world_size: int,
        leader_addr: str,
        control_port: int = 19666,
        auth_token: str,
        config_fingerprint: str,
        connect_timeout_s: float = 15.0,
        heartbeat_interval_s: float = 1.0,
        heartbeat_timeout_s: float = 5.0,
    ) -> None:
        self._runner = runner
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._endpoint = format_tcp_endpoint(leader_addr, int(control_port))
        self._auth_token = str(auth_token)
        self._config_fingerprint = str(config_fingerprint)
        self._connect_timeout_s = float(connect_timeout_s)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._sock = None
        self._ctx = None
        self.engine_spec: wire.EngineSpec | None = None
        self._plane_handler: typing.Callable[[wire.WireMessage, bytes | None], None] | None = None
        # Caller-thread → replay-thread send path: an outbox drained by the
        # replay loop plus an inproc wakeup socket that interrupts its poll.
        self._worker_outbox: queue.Queue = queue.Queue()
        self._wake_push = None
        self._wake_lock = threading.Lock()

    def set_plane_handler(self, handler: typing.Callable[[wire.WireMessage, bytes | None], None] | None) -> None:
        """Install the request-plane inbound handler (OutputUpdate/AdmitAck/...)."""
        self._plane_handler = handler

    def send_to_leader(self, header: wire.WireMessage, payload: bytes | None = None) -> None:
        """Queue a message for the leader from any thread.

        The replay thread owns the DEALER socket, so this enqueues and pokes
        the inproc wakeup socket to interrupt the replay loop's poll.
        """
        self._worker_outbox.put((wire.encode_message(header), payload))
        with self._wake_lock:
            if self._wake_push is not None:
                try:
                    self._wake_push.send(b"", flags=1)  # zmq.DONTWAIT
                except Exception:
                    pass

    def start(self) -> None:
        """Connect, Hello, and wait for the leader's HelloOk.

        Restartable: the sole caller (engine ``initiate``) reaches here only
        after any prior replay loop has exited, so an existing socket is
        stale — a weight hot-swap tears the worker down in lockstep with the
        leader, and the restarted leader resets its authed set and ledger. A
        stale DEALER would auto-reconnect at the transport layer but never
        re-Hello, so the leader would drop its frames as unauthenticated.
        Close it first and re-handshake.
        """
        import zmq

        if self._sock is not None:
            self.shutdown()
        while True:
            try:
                self._worker_outbox.get_nowait()
            except queue.Empty:
                break
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.DEALER)
        try:
            sock.setsockopt(zmq.IDENTITY, _worker_identity(self.rank))
            _configure_zmq_socket(sock, zmq)
            sock.connect(self._endpoint)
            sock.send_multipart(
                [
                    wire.encode_message(
                        wire.Hello(
                            rank=self.rank,
                            world_size=self.world_size,
                            config_fingerprint=self._config_fingerprint,
                            auth=self._auth_token,
                        )
                    )
                ]
            )
            if not sock.poll(int(self._connect_timeout_s * 1000)):
                raise StepCoordinationError(
                    f"worker rank={self.rank}: no HelloOk from leader at {self._endpoint} within {self._connect_timeout_s}s"
                )
            frames = sock.recv_multipart()
            if not frames:
                raise StepCoordinationError(f"worker rank={self.rank}: empty handshake reply from {self._endpoint}")
            try:
                message = wire.decode_message(frames[0])
            except Exception as exc:
                raise StepCoordinationError(
                    f"worker rank={self.rank}: undecodable handshake reply from {self._endpoint}: {exc}"
                ) from exc
            if not isinstance(message, wire.HelloOk):
                raise StepCoordinationError(f"worker rank={self.rank}: unexpected handshake reply {type(message).__name__}")
            self.engine_spec = message.engine_spec
            self._sock = sock
        except BaseException:
            sock.close(linger=0)
            self._sock = None
            raise

    # Workers never originate steps; the leader's stream is the only program.
    def execute_sync(self, scheduler_output):
        raise StepCoordinationError("worker ranks execute the leader's step stream, not their own.")

    def execute_async(self, scheduler_output):
        raise StepCoordinationError("worker ranks execute the leader's step stream, not their own.")

    def drain(self, handle):
        raise StepCoordinationError("worker ranks execute the leader's step stream, not their own.")

    def check_health(self) -> None:
        """Health is enforced inside :meth:`run_worker_loop`."""

    def run_worker_loop(self, stop_event) -> None:
        """Block replaying Step/Drain commands until Shutdown or failure.

        Every command executes on this thread so device dispatch order is
        exactly the leader's call order. On an execution failure the worker
        NACKs with the traceback and raises — a worker that cannot follow
        the stream must die loudly, not silently desync.

        Args:
            stop_event: Cooperative stop signal from the engine.

        Raises:
            StepCoordinationError: On leader loss or a failed step.
        """

        import zmq

        if self._sock is None:
            raise StepCoordinationError("run_worker_loop called before start().")
        sock = self._sock
        wake_endpoint = f"inproc://esurge-worker-wake-{next(_WORKER_WAKE_IDS)}"
        wake_pull = None
        wake_push = None
        try:
            wake_pull = self._ctx.socket(zmq.PULL)
            wake_pull.bind(wake_endpoint)
            wake_push = self._ctx.socket(zmq.PUSH)
            wake_push.connect(wake_endpoint)
            with self._wake_lock:
                self._wake_push = wake_push
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            poller.register(wake_pull, zmq.POLLIN)

            sock.send_multipart([wire.encode_message(wire.Ready(rank=self.rank))])
            inflight: dict[int, typing.Any] = {}
            last_leader_beat = time.monotonic()
            last_beat_sent = 0.0
            logger.info("eSurge worker rank=%d: replaying the leader step stream.", self.rank)
            while not stop_event.is_set():
                # Drain caller-thread messages (plane admits, aborts, ...).
                try:
                    while True:
                        header, payload = self._worker_outbox.get_nowait()
                        frames = [header]
                        if payload is not None:
                            frames.append(payload)
                        sock.send_multipart(frames)
                except queue.Empty:
                    pass

                now = time.monotonic()
                if now - last_beat_sent >= self._heartbeat_interval_s:
                    sock.send_multipart([wire.encode_message(wire.Heartbeat(rank=self.rank, ts=time.time()))])
                    last_beat_sent = now
                if now - last_leader_beat > self._heartbeat_timeout_s and not sock.poll(0):
                    raise StepCoordinationError(
                        f"worker rank={self.rank}: leader silent for more than "
                        f"{self._heartbeat_timeout_s}s; assuming it died"
                    )
                events = dict(poller.poll(_POLL_MS))
                if wake_pull in events:
                    while wake_pull.poll(0):
                        wake_pull.recv()
                if sock not in events:
                    continue
                frames = sock.recv_multipart()
                if not frames:
                    logger.warning("eSurge worker rank=%d: dropping empty frame set", self.rank)
                    continue
                try:
                    message = wire.decode_message(frames[0])
                except Exception:
                    logger.warning("eSurge worker rank=%d: dropping undecodable frame", self.rank)
                    continue
                if isinstance(message, wire.Heartbeat):
                    last_leader_beat = time.monotonic()
                    continue
                if isinstance(message, wire.Shutdown):
                    logger.info("eSurge worker rank=%d: leader shutdown (%s).", self.rank, message.reason)
                    return
                if isinstance(message, wire.Step):
                    last_leader_beat = time.monotonic()
                    if len(frames) < 2:
                        exc = StepCoordinationError(f"worker rank={self.rank}: Step {message.step_id} missing payload frame")
                        self._nack(sock, message.step_id, wire.ACK_PHASE_SYNC_DONE, exc)
                        raise exc
                    try:
                        scheduler_output = wire.decode_payload(frames[1])
                        if message.mode == wire.STEP_MODE_ASYNC:
                            inflight[message.step_id] = self._runner.execute_model_async(scheduler_output)
                        else:
                            model_output = self._runner.execute_model(scheduler_output)
                            self._ack(sock, message.step_id, wire.ACK_PHASE_SYNC_DONE, model_output, message.want_digest)
                    except Exception as exc:
                        self._nack(sock, message.step_id, wire.ACK_PHASE_SYNC_DONE, exc)
                        raise
                    # Runner work may legitimately exceed the heartbeat budget.
                    # Resume the silence clock now; queued beacons are consumed
                    # before deciding the leader is dead on the next iteration.
                    last_leader_beat = time.monotonic()
                    continue
                if isinstance(message, wire.Drain):
                    last_leader_beat = time.monotonic()
                    handle = inflight.pop(message.step_id, None)
                    if handle is None:
                        exc = StepCoordinationError(f"drain for unknown step {message.step_id}")
                        self._nack(sock, message.step_id, wire.ACK_PHASE_DRAINED, exc)
                        raise exc
                    try:
                        model_output = self._runner.wait_for_execution(handle)
                        self._ack(sock, message.step_id, wire.ACK_PHASE_DRAINED, model_output, message.want_digest)
                    except Exception as exc:
                        self._nack(sock, message.step_id, wire.ACK_PHASE_DRAINED, exc)
                        raise
                    last_leader_beat = time.monotonic()
                    continue
                if self._plane_handler is not None:
                    last_leader_beat = time.monotonic()
                    payload = frames[1] if len(frames) > 1 else None
                    try:
                        self._plane_handler(message, payload)
                    except Exception:
                        logger.exception("eSurge worker rank=%d: plane handler failed", self.rank)
            logger.info("eSurge worker rank=%d: stop requested.", self.rank)
        finally:
            with self._wake_lock:
                if self._wake_push is wake_push:
                    self._wake_push = None
            if wake_push is not None:
                try:
                    wake_push.close(linger=0)
                except Exception:
                    pass
            if wake_pull is not None:
                try:
                    wake_pull.close(linger=0)
                except Exception:
                    pass

    def _ack(self, sock, step_id: int, phase: int, model_output, want_digest: bool) -> None:
        digest = None
        if want_digest:
            digest = compute_sampled_digest(model_output.req_ids, model_output.sampled_token_ids)
        num_reqs = len(getattr(model_output, "req_ids", []) or [])
        sock.send_multipart(
            [
                wire.encode_message(
                    wire.StepAck(
                        rank=self.rank,
                        step_id=step_id,
                        phase=phase,
                        ok=True,
                        num_reqs=num_reqs,
                        digest=digest,
                    )
                )
            ]
        )

    def _nack(self, sock, step_id: int, phase: int, exc: BaseException) -> None:
        import traceback as _tb

        sock.send_multipart(
            [
                wire.encode_message(
                    wire.StepAck(
                        rank=self.rank,
                        step_id=step_id,
                        phase=phase,
                        ok=False,
                        error=str(exc),
                        traceback=_tb.format_exc(),
                    )
                )
            ]
        )

    def shutdown(self, reason: str = "") -> None:
        """Close the DEALER socket."""
        if self._sock is not None:
            self._sock.close(linger=500)
            self._sock = None
