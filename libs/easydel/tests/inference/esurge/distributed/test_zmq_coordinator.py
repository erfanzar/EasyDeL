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

"""Real-ZMQ loopback tests for the multi-host step-coordination plane.

A leader and a worker coordinator talk over ``tcp://127.0.0.1`` inside one
process, with mock runners recording their invocation sequences. These lock
the plane's core contracts: the worker replays the leader's exact call
stream (the correctness primitive), failures propagate as
StepCoordinationError instead of hangs, divergence is detected via
num_reqs/digest cross-checks, and unauthenticated peers are rejected before
any payload is unpickled.
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed.coordinator import (
    LocalCoordinator,
    StepCoordinationError,
    StepHandle,
)
from easydel.inference.esurge.distributed.zmq_coordinator import (
    ZmqLeaderCoordinator,
    ZmqWorkerCoordinator,
)

AUTH = "test-token"
FINGERPRINT = "test-fingerprint"
DEADLINE_S = 20.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sched_output(tag: str):
    return SimpleNamespace(tag=tag, total_num_scheduled_tokens=1, num_scheduled_tokens={f"{tag}-r0": 1})


class _MockRunner:
    """Records the exact invocation stream; emits configurable outputs."""

    def __init__(self, *, sampled_token: int = 42, fail_on: str | None = None):
        self.calls: list[str] = []
        self.sampled_token = sampled_token
        self.fail_on = fail_on
        self._lock = threading.Lock()

    def _record(self, entry: str) -> None:
        with self._lock:
            self.calls.append(entry)

    def _output(self, scheduler_output):
        return SimpleNamespace(
            tag=scheduler_output.tag,
            req_ids=list(scheduler_output.num_scheduled_tokens),
            sampled_token_ids=[[self.sampled_token] for _ in scheduler_output.num_scheduled_tokens],
        )

    def execute_model(self, scheduler_output):
        self._record(f"sync:{scheduler_output.tag}")
        if self.fail_on == scheduler_output.tag:
            raise RuntimeError(f"scripted failure at {scheduler_output.tag}")
        return self._output(scheduler_output)

    def execute_model_async(self, scheduler_output):
        self._record(f"dispatch:{scheduler_output.tag}")
        return SimpleNamespace(scheduler_output=scheduler_output)

    def wait_for_execution(self, handle):
        self._record(f"drain:{handle.scheduler_output.tag}")
        return self._output(handle.scheduler_output)


def _wait_until(predicate, timeout: float = DEADLINE_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached in time")


class _Plane:
    """One leader + one worker over loopback, torn down deterministically."""

    def __init__(self, *, worker_runner=None, leader_runner=None, verify_digest_interval: int = 1):
        from easydel.inference.esurge.distributed import wire

        port = _free_port()
        self.leader_runner = leader_runner or _MockRunner()
        self.worker_runner = worker_runner or _MockRunner()
        self.leader = ZmqLeaderCoordinator(
            self.leader_runner,
            world_size=2,
            bind_host="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            ready_timeout_s=DEADLINE_S,
            step_timeout_s=5.0,
            heartbeat_interval_s=0.2,
            heartbeat_timeout_s=DEADLINE_S,
            verify_digest_interval=verify_digest_interval,
            max_inflight_steps=4,
            engine_spec=wire.EngineSpec(
                esurge_name="mock-engine",
                max_model_len=128,
                max_num_seqs=2,
                reserve_tokens=2,
                eos_token_ids=[0],
                tokenizer_source="mock-tokenizer",
            ),
        )
        self.worker = ZmqWorkerCoordinator(
            self.worker_runner,
            rank=1,
            world_size=2,
            leader_addr="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            connect_timeout_s=DEADLINE_S,
            heartbeat_interval_s=0.2,
            heartbeat_timeout_s=DEADLINE_S,
        )
        self.stop_event = threading.Event()
        self.worker_error: list[BaseException] = []

        def _worker_body():
            try:
                self.worker.start()
                self.worker.run_worker_loop(self.stop_event)
            except BaseException as exc:
                self.worker_error.append(exc)

        self.worker_thread = threading.Thread(target=_worker_body, daemon=True)

    def __enter__(self):
        self.worker_thread.start()
        self.leader.start()
        return self

    def __exit__(self, *exc_info):
        self.leader.shutdown("test teardown")
        self.stop_event.set()
        self.worker_thread.join(timeout=DEADLINE_S)
        self.worker.shutdown()
        assert not self.worker_thread.is_alive(), "worker replay thread failed to exit"


def test_worker_replays_leader_stream_in_order():
    with _Plane() as plane:
        s1, s2, s3 = _sched_output("S1"), _sched_output("S2"), _sched_output("S3")
        out = plane.leader.execute_sync(s1)
        assert out.tag == "S1"
        handle = plane.leader.execute_async(s2)
        assert isinstance(handle, StepHandle)
        plane.leader.execute_sync(s3)
        drained = plane.leader.drain(handle)
        assert drained.tag == "S2"

        _wait_until(lambda: len(plane.worker_runner.calls) >= 4)
        assert plane.worker_runner.calls == ["sync:S1", "dispatch:S2", "sync:S3", "drain:S2"]

        # deferred acks + matching outputs keep the plane healthy
        _wait_until(lambda: plane.leader._ledger.min_acked() >= 2)
        plane.leader.check_health()
        assert not plane.worker_error


def test_worker_failure_propagates_as_coordination_error():
    with _Plane(worker_runner=_MockRunner(fail_on="BAD")) as plane:
        plane.leader.execute_sync(_sched_output("S1"))

        def _sees_failure():
            try:
                plane.leader.check_health()
            except StepCoordinationError:
                return True
            return False

        plane.leader.execute_sync(_sched_output("BAD"))
        _wait_until(_sees_failure)
        with pytest.raises(StepCoordinationError, match="scripted failure at BAD"):
            plane.leader.check_health()
        _wait_until(lambda: bool(plane.worker_error))
        assert isinstance(plane.worker_error[0], RuntimeError)


def test_sampled_token_divergence_is_detected():
    with _Plane(worker_runner=_MockRunner(sampled_token=7)) as plane:
        plane.leader.execute_sync(_sched_output("S1"))

        def _sees_failure():
            try:
                plane.leader.check_health()
            except StepCoordinationError:
                return True
            return False

        _wait_until(_sees_failure)
        with pytest.raises(StepCoordinationError, match="digest mismatch"):
            plane.leader.check_health()


def test_bad_auth_worker_never_becomes_ready():
    port = _free_port()
    leader_runner = _MockRunner()
    leader = ZmqLeaderCoordinator(
        leader_runner,
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=1.5,
        step_timeout_s=2.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    intruder = ZmqWorkerCoordinator(
        _MockRunner(),
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token="wrong-token",
        config_fingerprint=FINGERPRINT,
        connect_timeout_s=1.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )

    def _intruder_body():
        try:
            intruder.start()
        except StepCoordinationError:
            pass

    thread = threading.Thread(target=_intruder_body, daemon=True)
    thread.start()
    with pytest.raises(StepCoordinationError, match="timed out"):
        leader.start()
    leader.shutdown()
    thread.join(timeout=DEADLINE_S)
    intruder.shutdown()


def test_config_fingerprint_mismatch_fails_the_handshake():
    port = _free_port()
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint="leader-fp",
        ready_timeout_s=DEADLINE_S,
        step_timeout_s=2.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    worker = ZmqWorkerCoordinator(
        _MockRunner(),
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint="worker-fp",
        connect_timeout_s=DEADLINE_S,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    thread = threading.Thread(target=lambda: worker.start(), daemon=True)
    thread.start()
    with pytest.raises(StepCoordinationError, match="config mismatch"):
        leader.start()
    leader.shutdown()
    thread.join(timeout=DEADLINE_S)
    worker.shutdown()


def test_local_coordinator_is_identity_passthrough():
    runner = _MockRunner()
    local = LocalCoordinator(runner)
    local.start()
    out = local.execute_sync(_sched_output("A"))
    assert out.tag == "A"
    handle = local.execute_async(_sched_output("B"))
    assert local.drain(handle).tag == "B"
    local.check_health()
    assert runner.calls == ["sync:A", "dispatch:B", "drain:B"]
    with pytest.raises(StepCoordinationError):
        local.run_worker_loop(threading.Event())


class _RawClient:
    """Minimal request-plane client speaking Hello(role=client) over a DEALER."""

    def __init__(self, port: int, *, auth: str = AUTH, client_id: str = "client-a"):
        import zmq

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, f"client-{client_id}".encode())
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://127.0.0.1:{port}")
        self.client_id = client_id
        self.auth = auth

    def hello(self, timeout_s: float = DEADLINE_S):
        from easydel.inference.esurge.distributed import wire

        self.sock.send_multipart(
            [
                wire.encode_message(
                    wire.Hello(
                        rank=-1,
                        world_size=0,
                        config_fingerprint="",
                        auth=self.auth,
                        role=wire.ROLE_CLIENT,
                        client_id=self.client_id,
                    )
                )
            ]
        )
        assert self.sock.poll(int(timeout_s * 1000)), "no HelloOk for client"
        frames = self.sock.recv_multipart()
        return wire.decode_message(frames[0])

    def recv_until(self, predicate, timeout_s: float = DEADLINE_S):
        from easydel.inference.esurge.distributed import wire

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.sock.poll(200):
                continue
            frames = self.sock.recv_multipart()
            message = wire.decode_message(frames[0])
            if predicate(message):
                return message
        raise AssertionError("expected client message not received")

    def close(self):
        self.sock.close(linger=0)


def test_client_role_handshake_and_targeted_send():
    from easydel.inference.esurge.distributed import wire

    with _Plane() as plane:
        port = int(plane.leader._bind_endpoint.rsplit(":", 1)[1])
        client = _RawClient(port)
        try:
            ok = client.hello()
            assert isinstance(ok, wire.HelloOk)
            assert ok.rank == -1
            assert ok.engine_spec is not None and ok.engine_spec.esurge_name == "mock-engine"

            _wait_until(lambda: plane.leader.client_identity("client-a") is not None)
            # Clients must not receive the step broadcast.
            plane.leader.execute_sync(_sched_output("S1"))
            # Targeted delivery reaches only the addressed client.
            identity = plane.leader.client_identity("client-a")
            plane.leader.send_to_peer(identity, wire.Control(kind="ping", args={"x": 1}))
            message = client.recv_until(lambda m: isinstance(m, wire.Control))
            assert message.kind == "ping" and message.args == {"x": 1}
            # The worker replayed the step; the client saw none.
            _wait_until(lambda: plane.worker_runner.calls == ["sync:S1"])
        finally:
            client.close()


def test_client_with_bad_auth_is_ignored_but_workers_proceed():
    from easydel.inference.esurge.distributed import wire

    with _Plane() as plane:
        port = int(plane.leader._bind_endpoint.rsplit(":", 1)[1])
        intruder = _RawClient(port, auth="wrong", client_id="intruder")
        try:
            intruder.sock.send_multipart(
                [
                    wire.encode_message(
                        wire.Hello(
                            rank=-1,
                            world_size=0,
                            config_fingerprint="",
                            auth="wrong",
                            role=wire.ROLE_CLIENT,
                            client_id="intruder",
                        )
                    )
                ]
            )
            time.sleep(0.3)
            assert plane.leader.client_identity("intruder") is None
            plane.leader.execute_sync(_sched_output("S1"))
            _wait_until(lambda: plane.worker_runner.calls == ["sync:S1"])
        finally:
            intruder.close()


def test_worker_send_to_leader_reaches_plane_handler():
    from easydel.inference.esurge.distributed import wire

    received = []

    with _Plane() as plane:
        plane.leader._plane_handler = lambda identity, message, payload: received.append((message, payload))
        plane.worker.send_to_leader(wire.Control(kind="admit-test", args={"k": "v"}), b"payload-bytes")
        _wait_until(lambda: bool(received))
        message, payload = received[0]
        assert isinstance(message, wire.Control)
        assert message.kind == "admit-test"
        assert payload == b"payload-bytes"


def test_leader_shutdown_delivers_shutdown_broadcast_to_client():
    """leader.shutdown() must flush the Shutdown broadcast to connected peers.

    The IO loop drains its outbox at the top of each iteration and polls with
    the same 50ms timeout that shutdown() waits before flipping _stop_io, so
    the top-of-loop drain can miss the just-enqueued Shutdown and the loop
    exits without sending it — peers then see the owner go silent instead of
    receiving a clean teardown. Draining once more in the loop's finally
    (flushed by the socket LINGER) closes that race.
    """
    from easydel.inference.esurge.distributed import wire

    with _Plane() as plane:
        port = int(plane.leader._bind_endpoint.rsplit(":", 1)[1])
        client = _RawClient(port)
        try:
            assert isinstance(client.hello(), wire.HelloOk)
            _wait_until(lambda: plane.leader.client_identity("client-a") is not None)

            plane.leader.shutdown("bye")
            message = client.recv_until(lambda m: isinstance(m, wire.Shutdown), timeout_s=5.0)
            assert message.reason == "bye"
        finally:
            client.close()


def test_leader_restarts_after_shutdown():
    port = _free_port()

    def _make_worker():
        return ZmqWorkerCoordinator(
            _MockRunner(),
            rank=1,
            world_size=2,
            leader_addr="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            connect_timeout_s=DEADLINE_S,
            heartbeat_interval_s=0.2,
            heartbeat_timeout_s=DEADLINE_S,
        )

    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=DEADLINE_S,
        step_timeout_s=5.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )

    for round_idx in range(2):
        worker = _make_worker()
        stop_event = threading.Event()
        worker_runner = worker._runner

        def _body(w=worker, ev=stop_event):
            try:
                w.start()
                w.run_worker_loop(ev)
            except StepCoordinationError:
                pass

        thread = threading.Thread(target=_body, daemon=True)
        thread.start()
        leader.start()
        out = leader.execute_sync(_sched_output(f"R{round_idx}"))
        assert out.tag == f"R{round_idx}"
        _wait_until(lambda calls=worker_runner.calls, r=round_idx: calls == [f"sync:R{r}"])
        leader.shutdown(f"round {round_idx} teardown")
        stop_event.set()
        thread.join(timeout=DEADLINE_S)
        worker.shutdown()
        assert not thread.is_alive()


def test_worker_coordinator_rejoins_across_leader_restart():
    """A REUSED worker coordinator must re-Hello after a leader restart.

    This is the weight-hot-swap lifecycle (``update_model_weights`` on a
    worker rank): leader and worker tear down and re-initiate in lockstep,
    but the worker *coordinator object* survives — only its replay thread is
    respawned. ``start()`` must therefore drop the stale DEALER and
    re-handshake; otherwise the restarted leader (which resets its authed
    set and ledger) drops the worker's frames as unauthenticated and it
    never rejoins, so the second ``leader.start()`` would time out.

    Unlike :func:`test_leader_restarts_after_shutdown`, which builds a fresh
    worker (and socket) per round, this reuses one worker across rounds so
    the stale-socket path is actually exercised.
    """
    port = _free_port()
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=DEADLINE_S,
        step_timeout_s=5.0,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    worker = ZmqWorkerCoordinator(
        _MockRunner(),
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        connect_timeout_s=DEADLINE_S,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=DEADLINE_S,
    )
    worker_runner = worker._runner

    try:
        for round_idx in range(2):
            stop_event = threading.Event()

            def _body(ev=stop_event):
                try:
                    # Reused coordinator: round 1+ hits the stale-socket
                    # re-handshake path.
                    worker.start()
                    worker.run_worker_loop(ev)
                except StepCoordinationError:
                    pass

            thread = threading.Thread(target=_body, daemon=True)
            thread.start()
            leader.start()  # would time out here in round 1 without the re-Hello
            out = leader.execute_sync(_sched_output(f"R{round_idx}"))
            assert out.tag == f"R{round_idx}"
            expected = [f"sync:R{r}" for r in range(round_idx + 1)]
            _wait_until(lambda exp=expected: worker_runner.calls == exp)
            assert worker._sock is not None, "worker must hold a live handshake after rejoining"

            # Leader broadcasts Shutdown; the worker replay loop returns but
            # the coordinator object (and its now-stale socket) survives.
            leader.shutdown(f"round {round_idx} teardown")
            stop_event.set()
            thread.join(timeout=DEADLINE_S)
            assert not thread.is_alive()
    finally:
        worker.shutdown()
