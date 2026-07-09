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

import queue
import threading
import time
import typing

from ..logger import logger
from . import wire
from .coordinator import StepCoordinationError, StepHandle
from .protocol import compute_sampled_digest

if typing.TYPE_CHECKING:
    from ..outputs import ModelRunnerOutput
    from ..scheduler.output import SchedulerOutput

_POLL_MS = 50


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
            if not ack.ok:
                self.failure = (
                    f"worker rank={ack.rank} failed step {ack.step_id}: {ack.error}\n{ack.traceback or ''}"
                )
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
            for rank, beat in self.last_heartbeat.items():
                if now - beat > self.heartbeat_timeout_s:
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

    supports_overlap = True
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
    ) -> None:
        self._runner = runner
        self.world_size = int(world_size)
        self._bind_endpoint = f"tcp://{bind_host}:{int(control_port)}"
        self._auth_token = str(auth_token)
        self._config_fingerprint = str(config_fingerprint)
        self._ready_timeout_s = float(ready_timeout_s)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._verify_digest_interval = int(verify_digest_interval)
        self._max_inflight = max(1, int(max_inflight_steps))

        worker_ranks = list(range(1, self.world_size))
        self._ledger = _StepLedger(
            worker_ranks,
            step_timeout_s=step_timeout_s,
            heartbeat_timeout_s=heartbeat_timeout_s,
        )
        self._outbox: queue.Queue = queue.Queue()
        self._io_thread: threading.Thread | None = None
        self._stop_io = threading.Event()
        self._helloed: set[int] = set()
        self._ready: set[int] = set()
        self._ready_event = threading.Event()
        self._hello_error: str | None = None
        self._next_step_id = 0

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        """Bind, then block until every worker has helloed and readied."""
        if self._io_thread is not None:
            return
        self._io_thread = threading.Thread(target=self._io_loop, name="eSurgeCoordLeaderIO", daemon=True)
        self._io_thread.start()
        if not self._ready_event.wait(self._ready_timeout_s):
            raise StepCoordinationError(
                f"timed out after {self._ready_timeout_s}s waiting for {self.world_size - 1} worker(s); "
                f"helloed={sorted(self._helloed)}, ready={sorted(self._ready)}"
            )
        if self._hello_error is not None:
            raise StepCoordinationError(self._hello_error)
        logger.info("eSurge coordinator: %d worker(s) ready.", self.world_size - 1)

    # ------------------------------------------------------------- step plane
    def _want_digest(self, step_id: int) -> bool:
        interval = self._verify_digest_interval
        return interval > 0 and step_id % interval == 0

    def _broadcast_step(self, step_id: int, mode: int, scheduler_output, want_digest: bool) -> None:
        self._ledger.raise_if_failed()
        self._ledger.wait_for_window(step_id, self._max_inflight)
        header = wire.Step(step_id=step_id, mode=mode, want_digest=want_digest)
        payload = wire.encode_payload(scheduler_output)
        self._ledger.expect(step_id, self._ledger.last_acked.keys())
        self._outbox.put((wire.encode_message(header), payload))

    def execute_sync(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Mirror STEP(SYNC) to all workers, then execute locally."""
        self._next_step_id += 1
        step_id = self._next_step_id
        want_digest = self._want_digest(step_id)
        self._broadcast_step(step_id, wire.STEP_MODE_SYNC, scheduler_output, want_digest)
        model_output = self._runner.execute_model(scheduler_output)
        self._record_local(step_id, model_output, want_digest)
        return model_output

    def execute_async(self, scheduler_output: SchedulerOutput) -> StepHandle:
        """Mirror STEP(ASYNC) to all workers, then dispatch locally."""
        self._next_step_id += 1
        step_id = self._next_step_id
        self._broadcast_step(step_id, wire.STEP_MODE_ASYNC, scheduler_output, want_digest=False)
        runner_handle = self._runner.execute_model_async(scheduler_output)
        return StepHandle(step_id=step_id, runner_handle=runner_handle, scheduler_output=scheduler_output)

    def drain(self, handle: StepHandle) -> ModelRunnerOutput:
        """Mirror DRAIN to all workers, then materialize locally."""
        self._ledger.raise_if_failed()
        want_digest = self._want_digest(handle.step_id)
        header = wire.Drain(step_id=handle.step_id, want_digest=want_digest)
        self._ledger.expect(handle.step_id, self._ledger.last_acked.keys())
        self._outbox.put((wire.encode_message(header), None))
        model_output = self._runner.wait_for_execution(handle.runner_handle)
        self._record_local(handle.step_id, model_output, want_digest)
        return model_output

    def _record_local(self, step_id: int, model_output, want_digest: bool) -> None:
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
        if self._io_thread is None:
            return
        self._outbox.put((wire.encode_message(wire.Shutdown(reason=reason)), None))
        time.sleep(0.05)
        self._stop_io.set()
        self._io_thread.join(timeout=5.0)
        self._io_thread = None

    # -------------------------------------------------------------- IO thread
    def _io_loop(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 500)
        sock.bind(self._bind_endpoint)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        identities: dict[int, bytes] = {}
        authed: set[bytes] = set()
        last_beat = 0.0
        last_sweep = 0.0
        try:
            while not self._stop_io.is_set():
                # outbound commands, broadcast in FIFO order
                try:
                    while True:
                        header, payload = self._outbox.get_nowait()
                        for identity in identities.values():
                            frames = [identity, header]
                            if payload is not None:
                                frames.append(payload)
                            sock.send_multipart(frames)
                except queue.Empty:
                    pass

                now = time.monotonic()
                if now - last_beat >= self._heartbeat_interval_s and identities:
                    beat = wire.encode_message(wire.Heartbeat(rank=0, ts=time.time()))
                    for identity in identities.values():
                        sock.send_multipart([identity, beat])
                    last_beat = now
                if now - last_sweep >= 0.5:
                    if self._ready_event.is_set():
                        self._ledger.sweep()
                    last_sweep = now

                events = dict(poller.poll(_POLL_MS))
                if sock not in events:
                    continue
                frames = sock.recv_multipart()
                identity, header_bytes = frames[0], frames[1]
                try:
                    message = wire.decode_message(header_bytes)
                except Exception:
                    logger.warning("eSurge coordinator: dropping undecodable frame from %r", identity)
                    continue

                if isinstance(message, wire.Hello):
                    if message.auth != self._auth_token:
                        logger.warning("eSurge coordinator: rejecting worker with bad auth token")
                        continue
                    if message.world_size != self.world_size:
                        self._hello_error = (
                            f"worker rank={message.rank} reports world_size={message.world_size}, "
                            f"leader expects {self.world_size}"
                        )
                        self._ready_event.set()
                        continue
                    if message.config_fingerprint != self._config_fingerprint:
                        self._hello_error = (
                            f"worker rank={message.rank} config mismatch: "
                            f"{message.config_fingerprint} != {self._config_fingerprint}"
                        )
                        self._ready_event.set()
                        continue
                    identities[message.rank] = identity
                    authed.add(identity)
                    self._helloed.add(message.rank)
                    sock.send_multipart([identity, wire.encode_message(wire.HelloOk(rank=message.rank))])
                    continue

                if identity not in authed:
                    logger.warning("eSurge coordinator: dropping frame from unauthenticated peer")
                    continue

                if isinstance(message, wire.Ready):
                    self._ready.add(message.rank)
                    self._ledger.record_heartbeat(message.rank)
                    if len(self._ready) >= self.world_size - 1:
                        self._ready_event.set()
                elif isinstance(message, wire.StepAck):
                    self._ledger.record_ack(message)
                elif isinstance(message, wire.Heartbeat):
                    self._ledger.record_heartbeat(message.rank)
        finally:
            sock.close(linger=500)


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
        self._endpoint = f"tcp://{leader_addr}:{int(control_port)}"
        self._auth_token = str(auth_token)
        self._config_fingerprint = str(config_fingerprint)
        self._connect_timeout_s = float(connect_timeout_s)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._sock = None
        self._ctx = None

    def start(self) -> None:
        """Connect, Hello, and wait for the leader's HelloOk."""
        import zmq

        if self._sock is not None:
            return
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, _worker_identity(self.rank))
        sock.setsockopt(zmq.LINGER, 500)
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
            sock.close(linger=0)
            self._sock = None
            raise StepCoordinationError(
                f"worker rank={self.rank}: no HelloOk from leader at {self._endpoint} "
                f"within {self._connect_timeout_s}s"
            )
        frames = sock.recv_multipart()
        message = wire.decode_message(frames[0])
        if not isinstance(message, wire.HelloOk):
            sock.close(linger=0)
            self._sock = None
            raise StepCoordinationError(f"worker rank={self.rank}: unexpected handshake reply {type(message).__name__}")
        self._sock = sock

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

        if self._sock is None:
            raise StepCoordinationError("run_worker_loop called before start().")
        sock = self._sock
        sock.send_multipart([wire.encode_message(wire.Ready(rank=self.rank))])
        inflight: dict[int, typing.Any] = {}
        last_leader_beat = time.monotonic()
        last_beat_sent = 0.0
        logger.info("eSurge worker rank=%d: replaying the leader step stream.", self.rank)
        while not stop_event.is_set():
            now = time.monotonic()
            if now - last_beat_sent >= self._heartbeat_interval_s:
                sock.send_multipart([wire.encode_message(wire.Heartbeat(rank=self.rank, ts=time.time()))])
                last_beat_sent = now
            if now - last_leader_beat > self._heartbeat_timeout_s:
                raise StepCoordinationError(
                    f"worker rank={self.rank}: leader silent for more than "
                    f"{self._heartbeat_timeout_s}s; assuming it died"
                )
            if not sock.poll(_POLL_MS):
                continue
            frames = sock.recv_multipart()
            message = wire.decode_message(frames[0])
            if isinstance(message, wire.Heartbeat):
                last_leader_beat = time.monotonic()
                continue
            if isinstance(message, wire.Shutdown):
                logger.info("eSurge worker rank=%d: leader shutdown (%s).", self.rank, message.reason)
                return
            if isinstance(message, wire.Step):
                last_leader_beat = time.monotonic()
                scheduler_output = wire.decode_payload(frames[1])
                try:
                    if message.mode == wire.STEP_MODE_ASYNC:
                        inflight[message.step_id] = self._runner.execute_model_async(scheduler_output)
                    else:
                        model_output = self._runner.execute_model(scheduler_output)
                        self._ack(sock, message.step_id, wire.ACK_PHASE_SYNC_DONE, model_output, message.want_digest)
                except Exception as exc:
                    self._nack(sock, message.step_id, wire.ACK_PHASE_SYNC_DONE, exc)
                    raise
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
                continue
        logger.info("eSurge worker rank=%d: stop requested.", self.rank)

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
