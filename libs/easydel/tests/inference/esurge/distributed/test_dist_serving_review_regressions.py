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

"""Regression tests for Distributed Serving & Replica Launcher review findings.

Covers S2-C1..C3, S2-H2..H8, S2-M7, S2-M12 (serving config/dashboard), and S2-L1..L4
using real ZMQ loopback sockets, real unix sockets, real subprocesses, and real
state-machine objects (no array computation).
"""

from __future__ import annotations

import asyncio
import dataclasses
import errno
import gc
import itertools
import json
import os
import pickle
import socket
import subprocess
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import discovery, remote_engine, wire
from easydel.inference.esurge.distributed.coordinator import (
    StepCoordinationError,
    _split_host_port,
)
from easydel.inference.esurge.distributed.request_plane import (
    OriginRequestPlane,
    OwnerRequestPlane,
    RequestPlaneError,
    RequestPlaneUnavailable,
)
from easydel.inference.esurge.distributed.routed_engine import RoutedEngine
from easydel.inference.esurge.distributed.zmq_coordinator import (
    ZmqLeaderCoordinator,
    ZmqWorkerCoordinator,
)
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.esurge.serving import (
    ApiKeyFileError,
    ServingConfig,
    ServingConfigError,
    app,
    dashboard,
    host_replicas,
    keys,
    ray_replicas,
)
from easydel.inference.esurge.serving.cluster import public_urls
from easydel.inference.esurge.serving.config import CoordinationSpec, ModelSpec, ServerSpec
from easydel.inference.esurge.serving.replicas import BoxLayout, ReplicaPlan, box_layout
from easydel.inference.sampling_params import SamplingParams

from .test_request_plane import AUTO_1, FakeEngineRequest, _OwnerEngine, _PlaneHarness, _Sink
from .test_zmq_coordinator import AUTH, DEADLINE_S, FINGERPRINT, _free_port, _MockRunner, _sched_output, _wait_until

pytestmark = pytest.mark.cpu_ok


# ---------------------------------------------------------------------------
# S2-C1: Single-host DP replicas emit `distributed` config & bind ZMQ plane
# ---------------------------------------------------------------------------


def test_s2_c1_single_host_dp_replica_emits_distributed_section_and_serves_client():
    cfg = ServingConfig.from_mapping({"model": {"checkpoint": "gs://b/m"}})
    no_plane = app.build_engine_sections(cfg, world_size=1, token=None, control_bind_host=None)
    assert "distributed" not in no_plane

    with pytest.raises(ValueError):
        app.build_engine_sections(cfg, world_size=1, token=None, control_bind_host="127.0.0.1", enable_request_plane=True)

    with_plane = app.build_engine_sections(
        cfg,
        world_size=1,
        token="secret-tok",
        control_bind_host="127.0.0.1",
        enable_request_plane=True,
    )
    assert with_plane["distributed"] == {
        "coordination": "zmq",
        "distributed_auth_token": "secret-tok",
        "distributed_control_port": cfg.coordination.control_port,
        "distributed_control_bind_host": "127.0.0.1",
        "distributed_heartbeat_timeout_s": cfg.coordination.heartbeat_timeout_s,
    }

    # Verify a world_size=1 ZmqLeaderCoordinator binds and accepts a real request-plane client.
    port = _free_port()
    spec = wire.EngineSpec(esurge_name="dp1", max_model_len=128, max_num_seqs=2, tokenizer_source="tok")
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=1,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token="secret-tok",
        config_fingerprint="",
        ready_timeout_s=5.0,
        engine_spec=spec,
    )
    dead: list[str] = []
    channel = remote_engine._ClientChannel(
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token="secret-tok",
        client_id="front-0",
        connect_timeout_s=5.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=5.0,
        on_dead=dead.append,
    )
    leader.start()
    try:
        channel.start()
        assert channel.alive
        assert channel.engine_spec is not None and channel.engine_spec.esurge_name == "dp1"
        assert leader.client_identity("front-0") is not None
    finally:
        channel.close()
        leader.shutdown("done")
    assert dead == []


# ---------------------------------------------------------------------------
# S2-C2: ZmqLeaderCoordinator restart after > max_inflight_steps steps
# ---------------------------------------------------------------------------


def test_s2_c2_leader_restart_after_exceeding_max_inflight_steps():
    port = _free_port()
    max_inflight = 2
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=DEADLINE_S,
        step_timeout_s=2.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=DEADLINE_S,
        max_inflight_steps=max_inflight,
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
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=DEADLINE_S,
    )
    expected_calls: list[str] = []
    try:
        for round_idx in range(2):
            stop_event = threading.Event()
            worker_errors: list[BaseException] = []

            def _body(ev=stop_event, errs=worker_errors):
                try:
                    worker.start()
                    worker.run_worker_loop(ev)
                except BaseException as exc:
                    errs.append(exc)

            thread = threading.Thread(target=_body, daemon=True)
            thread.start()
            leader.start()
            # Run strictly more steps than max_inflight_steps so an unreset _next_step_id would deadlock on round 1.
            for step_idx in range(max_inflight + 4):
                tag = f"R{round_idx}-S{step_idx}"
                leader.execute_sync(_sched_output(tag))
                expected_calls.append(f"sync:{tag}")
            _wait_until(lambda: worker._runner.calls == expected_calls)
            leader.check_health()
            leader.shutdown(f"round {round_idx}")
            stop_event.set()
            thread.join(timeout=DEADLINE_S)
            assert not thread.is_alive()
            assert worker_errors == []
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# S2-C3: BoxLayout.libtpu_env on non-v5p TPUs (v4, v5e/v5litepod, v6e, v3)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _TypedChip:
    id: int
    coords: tuple[int, int, int]
    process_index: int
    device_kind: str


def test_s2_c3_box_layout_libtpu_env_non_v5p_tpus(monkeypatch):
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)

    # TPU v4-64 slice (4x4x2 = 32 chips, 2x2x1 per host), split into 2 replicas of 16 chips -> v4-32
    v4_chips = [
        _TypedChip(i, (x, y, z), x // 2 + 2 * (y // 2) + 4 * z, "TPU v4")
        for i, (z, y, x) in enumerate(itertools.product(range(2), range(4), range(4)))
    ]
    v4_half = [c for c in v4_chips if c.coords[1] < 2]
    v4_local = [c for c in v4_chips if c.process_index == 0]
    v4_layout = box_layout(v4_half, v4_chips, v4_local)
    assert v4_layout is not None
    assert v4_layout.libtpu_env(["10.0.0.0", "10.0.0.1", "10.0.0.4", "10.0.0.5"], 0)["TPU_ACCELERATOR_TYPE"] == "v4-32"

    # TPU v5e / v5litepod-16 slice (4x4x1 = 16 chips, 2x2x1 per host), half = 8 chips -> v5litepod-8 (1 core/chip)
    v5e_chips = [
        _TypedChip(i, (x, y, 0), x // 2 + 2 * (y // 2), "TPU v5 lite")
        for i, (y, x) in enumerate(itertools.product(range(4), range(4)))
    ]
    v5e_half = [c for c in v5e_chips if c.coords[1] < 2]
    v5e_local = [c for c in v5e_chips if c.process_index == 0]
    v5e_layout = box_layout(v5e_half, v5e_chips, v5e_local)
    assert v5e_layout is not None
    assert v5e_layout.libtpu_env(["10.0.0.0", "10.0.0.1"], 0)["TPU_ACCELERATOR_TYPE"] == "v5litepod-8"

    # Inherited TPU_ACCELERATOR_TYPE="v5e-16" preserves the "v5e" prefix -> v5e-8
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5e-16")
    v5e_env_layout = box_layout(v5e_half, v5e_chips, v5e_local)
    assert v5e_env_layout is not None
    assert v5e_env_layout.libtpu_env(["10.0.0.0", "10.0.0.1"], 0)["TPU_ACCELERATOR_TYPE"] == "v5e-8"

    # TPU v6e (Trillium) slice (4x4x1 = 16 chips), half = 8 chips -> v6e-8 even if an unrelated env var is set
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-128")
    v6e_chips = [
        _TypedChip(i, (x, y, 0), x // 2 + 2 * (y // 2), "TPU v6 lite")
        for i, (y, x) in enumerate(itertools.product(range(4), range(4)))
    ]
    v6e_half = [c for c in v6e_chips if c.coords[1] < 2]
    v6e_local = [c for c in v6e_chips if c.process_index == 0]
    v6e_layout = box_layout(v6e_half, v6e_chips, v6e_local)
    assert v6e_layout is not None
    assert v6e_layout.libtpu_env(["10.0.0.0", "10.0.0.1"], 0)["TPU_ACCELERATOR_TYPE"] == "v6e-8"

    # Direct BoxLayout.libtpu_env derives prefix & cores/chip from env when not passed via box_layout()
    direct = BoxLayout((0, 0, 0), (4, 2, 1), (4, 4, 1), (2, 2, 1))
    assert direct.libtpu_env(["a", "b"], 0, env={"TPU_ACCELERATOR_TYPE": "v6e-16"})["TPU_ACCELERATOR_TYPE"] == "v6e-8"

    # TPU v3 (2 logical devices per chip sharing coordinates): box_layout accepts it and reports v3-{2*chips}
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    v3_devices = []
    dev_id = 0
    for y, x in itertools.product(range(4), range(4)):
        proc = x // 2 + 2 * (y // 2)
        for _core in range(2):
            v3_devices.append(_TypedChip(dev_id, (x, y, 0), proc, "TPU v3"))
            dev_id += 1
    v3_half = [d for d in v3_devices if d.coords[1] < 2]
    v3_local = [d for d in v3_devices if d.process_index == 0]
    v3_layout = box_layout(v3_half, v3_devices, v3_local)
    assert v3_layout is not None
    assert v3_layout.libtpu_env(["a", "b"], 0)["TPU_ACCELERATOR_TYPE"] == "v3-16"


# ---------------------------------------------------------------------------
# S2-H2: ROLE_CLIENT cannot forge worker Ready/StepAck/Heartbeat; rank/role checks
# ---------------------------------------------------------------------------


def test_s2_h2_role_client_forgery_and_invalid_worker_rank_rejected():
    import zmq

    port = _free_port()
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=5.0,
        step_timeout_s=5.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=0.8,
    )
    worker = ZmqWorkerCoordinator(
        _MockRunner(),
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        connect_timeout_s=5.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=5.0,
    )

    leader_started = threading.Event()
    leader_errors: list[BaseException] = []

    def _start_leader():
        try:
            leader.start()
            leader_started.set()
        except BaseException as exc:
            leader_errors.append(exc)

    leader_thread = threading.Thread(target=_start_leader, daemon=True)
    leader_thread.start()
    _wait_until(lambda: leader._bound_event.is_set())

    ctx = zmq.Context.instance()
    client_sock = ctx.socket(zmq.DEALER)
    client_sock.setsockopt(zmq.IDENTITY, b"client-forger")
    client_sock.setsockopt(zmq.LINGER, 0)
    client_sock.connect(f"tcp://127.0.0.1:{port}")

    bad_worker_sock = ctx.socket(zmq.DEALER)
    bad_worker_sock.setsockopt(zmq.IDENTITY, b"bad-worker")
    bad_worker_sock.setsockopt(zmq.LINGER, 0)
    bad_worker_sock.connect(f"tcp://127.0.0.1:{port}")

    worker_stop = threading.Event()
    worker_thread = None
    try:
        # Bad role and out-of-range worker ranks must be rejected.
        for bad_hello in (
            wire.Hello(rank=0, world_size=2, config_fingerprint=FINGERPRINT, auth=AUTH, role=wire.ROLE_WORKER),
            wire.Hello(rank=2, world_size=2, config_fingerprint=FINGERPRINT, auth=AUTH, role=wire.ROLE_WORKER),
            wire.Hello(rank=1, world_size=2, config_fingerprint=FINGERPRINT, auth=AUTH, role="invalid-role"),
        ):
            bad_worker_sock.send_multipart([wire.encode_message(bad_hello)])
        time.sleep(0.2)
        assert not bad_worker_sock.poll(0)
        assert leader._helloed == set()

        # Authenticate legitimately as ROLE_CLIENT, then try to forge worker Ready.
        client_sock.send_multipart(
            [
                wire.encode_message(
                    wire.Hello(
                        rank=-1,
                        world_size=0,
                        config_fingerprint="",
                        auth=AUTH,
                        role=wire.ROLE_CLIENT,
                        client_id="forger",
                    )
                )
            ]
        )
        assert client_sock.poll(2000)
        assert isinstance(wire.decode_message(client_sock.recv_multipart()[0]), wire.HelloOk)

        client_sock.send_multipart([wire.encode_message(wire.Ready(rank=1))])
        time.sleep(0.25)
        assert not leader_started.is_set(), "ROLE_CLIENT must not satisfy worker Ready barrier"
        assert leader._ready == set()

        # Now start the real worker rank=1 and let the leader become ready.
        def _run_worker():
            try:
                worker.start()
                worker.run_worker_loop(worker_stop)
            except StepCoordinationError:
                pass

        worker_thread = threading.Thread(target=_run_worker, daemon=True)
        worker_thread.start()
        _wait_until(leader_started.is_set)
        assert leader_errors == []

        # Forged StepAck(ok=False) from ROLE_CLIENT must be ignored.
        client_sock.send_multipart(
            [
                wire.encode_message(
                    wire.StepAck(rank=1, step_id=1, phase=wire.ACK_PHASE_SYNC_DONE, ok=False, error="forged nack")
                )
            ]
        )
        time.sleep(0.25)
        leader.check_health()

        # Stop real worker without shutting down leader, while ROLE_CLIENT floods forged Heartbeat(rank=1).
        worker_stop.set()
        worker_thread.join(timeout=5.0)
        worker.shutdown()
        worker_thread = None

        end = time.monotonic() + 1.5
        saw_failure = False
        while time.monotonic() < end:
            client_sock.send_multipart([wire.encode_message(wire.Heartbeat(rank=1, ts=time.time()))])
            try:
                leader.check_health()
            except StepCoordinationError:
                saw_failure = True
                break
            time.sleep(0.05)
        assert saw_failure, "ROLE_CLIENT Heartbeat(rank=1) must not mask real worker rank 1 death"
    finally:
        client_sock.close(linger=0)
        bad_worker_sock.close(linger=0)
        leader.shutdown("cleanup")
        leader_thread.join(timeout=5.0)
        if worker_thread is not None:
            worker_stop.set()
            worker_thread.join(timeout=5.0)
            worker.shutdown()


# ---------------------------------------------------------------------------
# S2-H3: ZmqLeaderCoordinator.start() fails fast on bind() error (ws=1 & ws=2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [1, 2])
def test_s2_h3_start_fails_fast_and_cleans_up_on_bind_error(world_size):
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        leader = ZmqLeaderCoordinator(
            _MockRunner(),
            world_size=world_size,
            bind_host="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            ready_timeout_s=30.0,
        )
        started = time.monotonic()
        with pytest.raises(StepCoordinationError, match="failed to bind control plane"):
            leader.start()
        elapsed = time.monotonic() - started
        assert elapsed < 3.0, f"bind failure must fail fast instead of blocking (took {elapsed:.2f}s)"
        assert leader._io_thread is None, "failed start() must clean up _io_thread"
    finally:
        blocker.close()


# ---------------------------------------------------------------------------
# S2-H4: max_inflight_steps=1 disables overlap and fails fast on 2x async
# ---------------------------------------------------------------------------


def test_s2_h4_max_inflight_one_overlap_guard_prevents_deadlock():
    leader_ws1 = ZmqLeaderCoordinator(
        _MockRunner(), world_size=1, auth_token=AUTH, config_fingerprint=FINGERPRINT, max_inflight_steps=1
    )
    leader_ws2_1 = ZmqLeaderCoordinator(
        _MockRunner(), world_size=2, auth_token=AUTH, config_fingerprint=FINGERPRINT, max_inflight_steps=1
    )
    leader_ws2_2 = ZmqLeaderCoordinator(
        _MockRunner(), world_size=2, auth_token=AUTH, config_fingerprint=FINGERPRINT, max_inflight_steps=2
    )
    assert leader_ws1.supports_overlap is True
    assert leader_ws2_1.supports_overlap is False
    assert leader_ws2_2.supports_overlap is True

    # Real loopback test: sequential async->drain works at max_inflight_steps=1,
    # while overlapping two async dispatches fails fast instead of hanging for step_timeout_s.
    port = _free_port()
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=DEADLINE_S,
        step_timeout_s=15.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=DEADLINE_S,
        max_inflight_steps=1,
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
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=DEADLINE_S,
    )
    stop_event = threading.Event()
    thread = threading.Thread(
        target=lambda: (worker.start(), worker.run_worker_loop(stop_event)),
        daemon=True,
    )
    thread.start()
    try:
        leader.start()
        h1 = leader.execute_async(_sched_output("S1"))
        t0 = time.monotonic()
        with pytest.raises(StepCoordinationError, match="undrained async"):
            leader.execute_async(_sched_output("S2"))
        assert time.monotonic() - t0 < 2.0, "must fail fast instead of deadlocking for step_timeout_s"

        # Draining h1 clears the slot so a subsequent async->drain pair succeeds cleanly.
        assert leader.drain(h1).tag == "S1"
        h2 = leader.execute_async(_sched_output("S2"))
        assert leader.drain(h2).tag == "S2"
        _wait_until(lambda: worker._runner.calls == ["dispatch:S1", "drain:S1", "dispatch:S2", "drain:S2"])
    finally:
        leader.shutdown("done")
        stop_event.set()
        thread.join(timeout=DEADLINE_S)
        worker.shutdown()


# ---------------------------------------------------------------------------
# S2-H5: Coalesced n > 1 sample-level aborts & fan-out gating
# ---------------------------------------------------------------------------


def test_s2_h5_coalesced_n2_local_remote_sample_aborts_and_fanout_gating():
    with _PlaneHarness() as plane:
        remote_reqs = [
            FakeEngineRequest(f"{AUTO_1}-0", parent_request_id=AUTO_1, sample_index=0),
            FakeEngineRequest(f"{AUTO_1}-1", parent_request_id=AUTO_1, sample_index=1),
        ]
        local_reqs = [
            FakeEngineRequest(f"{AUTO_1}-0", parent_request_id=AUTO_1, sample_index=0),
            FakeEngineRequest(f"{AUTO_1}-1", parent_request_id=AUTO_1, sample_index=1),
        ]
        # Remote-first coalesced group, then owner attaches locally (local_direct=False).
        plane.origin.submit_remote(remote_reqs)
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        scheduled_parent = plane.owner_engine.submitted[0][0].parent_request_id
        plane.owner.admit_local(local_reqs)
        assert len(plane.owner_engine.submitted) == 1

        # 1. Remote aborts sample -0 only; local still wants both -0 and -1.
        assert plane.origin.notify_abort(f"{AUTO_1}-0")
        _wait_until(
            lambda: not any(
                key[0] is not None and key[1] == f"{AUTO_1}-0" for key in plane.owner._subscriber_children
            )
        )
        assert plane.owner_engine.aborted == [], "local subscriber still wants -0"

        # Fan-out for -0 MUST reach local ONLY, never the remote subscriber that aborted -0.
        plane.tee(
            EngineCoreOutput(request_id=f"{scheduled_parent}-0", new_token_ids=[11]),
            EngineCoreOutput(request_id=f"{scheduled_parent}-1", new_token_ids=[22]),
        )
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 1)
        _wait_until(lambda: len(plane.local_sink.all_outputs()) == 2)
        assert [o.request_id for o in plane.origin_sink.all_outputs()] == [f"{AUTO_1}-1"]
        assert {o.request_id for o in plane.local_sink.all_outputs()} == {f"{AUTO_1}-0", f"{AUTO_1}-1"}

        # 2. Local now aborts sample -0: nobody wants -0 anymore -> scheduled -0 MUST abort.
        plane.owner.notify_local_abort(f"{AUTO_1}-0")
        assert plane.owner_engine.aborted == [f"{scheduled_parent}-0"]

        # 3. Remote aborts its last remaining child (-1): remote subscriber MUST detach from entry.subscribers.
        assert plane.origin.notify_abort(f"{AUTO_1}-1")
        entry = next(iter(plane.owner._entries.values()))
        _wait_until(lambda: all(ident is None for ident, _pid in entry.subscribers))
        assert plane.owner_engine.aborted == [f"{scheduled_parent}-0"], "local still wants -1"

        # 4. Local aborts its last remaining child (-1): entry has no subscribers -> whole group aborts & retires.
        plane.owner.notify_local_abort(f"{AUTO_1}-1")
        assert plane.owner_engine.aborted == [f"{scheduled_parent}-0", scheduled_parent]
        assert not plane.owner.needs_tee
        assert plane.owner._entries == {}


# ---------------------------------------------------------------------------
# S2-H6: RemoteEngineHandle bootstrap race/cleanup, owner death in generate, n>1 events
# ---------------------------------------------------------------------------


def test_s2_h6_early_owner_death_during_bootstrap_and_late_bootstrap_cleanup(monkeypatch):
    import zmq

    port = _free_port()
    spec = wire.EngineSpec(
        esurge_name="m", max_model_len=128, max_num_seqs=4, reserve_tokens=1, eos_token_ids=[0], tokenizer_source="tok"
    )
    entered_tokenizer = threading.Event()
    owner_sent_shutdown = threading.Event()

    def _owner():
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{port}")
        try:
            ident, _hello = sock.recv_multipart()
            sock.send_multipart([ident, wire.encode_message(wire.HelloOk(rank=-1, engine_spec=spec))])
            entered_tokenizer.wait(timeout=5.0)
            sock.send_multipart([ident, wire.encode_message(wire.Shutdown(reason="owner crashed early"))])
            owner_sent_shutdown.set()
            time.sleep(0.2)
        finally:
            sock.close(linger=0)

    shutdowns = []

    class FakeWorkers:
        def __init__(self, *_a):
            pass

        def start(self, **_kw):
            return SimpleNamespace(), SimpleNamespace()

        def shutdown(self):
            shutdowns.append("workers")

    def slow_tokenizer(*_a, **_kw):
        entered_tokenizer.set()
        owner_sent_shutdown.wait(timeout=5.0)
        time.sleep(0.15)  # allow client IO thread to deliver Shutdown -> _on_owner_dead while _plane is still None
        return SimpleNamespace(eos_token_id=0, get_vocab=lambda: {})

    monkeypatch.setattr(remote_engine, "WorkerManager", FakeWorkers)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", slow_tokenizer)

    t = threading.Thread(target=_owner, daemon=True)
    t.start()
    try:
        with pytest.raises(RequestPlaneError, match="owner died during remote handle bootstrap"):
            remote_engine.RemoteEngineHandle("127.0.0.1", port, auth_token=AUTH, connect_timeout_s=5.0)
    finally:
        t.join(timeout=5.0)
    assert shutdowns == ["workers"], "workers must be shut down on bootstrap failure"

    # Also verify late bootstrap failure (after OutputPipeline.start()) stops pipeline & workers & channel.
    stopped = []

    class FakeChannel:
        alive = True
        dead_reason = None
        client_id = "c1"
        engine_spec = spec

        def __init__(self, **_kw):
            pass

        def start(self):
            pass

        def set_plane_handler(self, _h):
            pass

        def close(self):
            stopped.append("channel")

    class FakePipeline:
        def __init__(self, **_kw):
            pass

        def start(self):
            pass

        def stop(self):
            stopped.append("pipeline")

    monkeypatch.setattr(remote_engine, "_ClientChannel", FakeChannel)
    monkeypatch.setattr(remote_engine, "OutputPipeline", FakePipeline)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: SimpleNamespace(get_vocab=lambda: {}))
    monkeypatch.setattr(remote_engine, "RequestAdmission", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    shutdowns.clear()
    with pytest.raises(RequestPlaneError, match="boom"):
        remote_engine.RemoteEngineHandle("127.0.0.1", port, auth_token=AUTH)
    assert "channel" in stopped and "pipeline" in stopped and shutdowns == ["workers"]


def test_s2_h6_owner_death_mid_generate_raises_instead_of_returning_truncated_abort(monkeypatch):
    import zmq

    port = _free_port()
    spec = wire.EngineSpec(
        esurge_name="m", max_model_len=128, max_num_seqs=4, reserve_tokens=1, eos_token_ids=[0], tokenizer_source="tok"
    )

    def _owner():
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{port}")
        try:
            while True:
                if not sock.poll(5000):
                    return
                frames = sock.recv_multipart()
                ident, msg = frames[0], wire.decode_message(frames[1])
                if isinstance(msg, wire.Hello):
                    sock.send_multipart([ident, wire.encode_message(wire.HelloOk(rank=-1, engine_spec=spec))])
                elif isinstance(msg, wire.Admit):
                    sock.send_multipart([ident, wire.encode_message(wire.AdmitAck(request_id=msg.request_id, ok=True))])
                    time.sleep(0.05)
                    sock.send_multipart([ident, wire.encode_message(wire.Shutdown(reason="boom"))])
                    return
        finally:
            sock.close(linger=0)

    class FakeTokenizer:
        eos_token_id = 0
        chat_template = None

        def __call__(self, text, **_kw):
            return {"input_ids": [1, 2]}

        def decode(self, ids, **_kw):
            return "x"

        def get_vocab(self):
            return {}

    class FakeWorkers:
        def __init__(self, *_a):
            pass

        def start(self, **_kw):
            return SimpleNamespace(tokenize=lambda rid, prompt: [1, 2]), SimpleNamespace(reset=lambda rid: None)

        def shutdown(self):
            pass

    class FakePipeline:
        def __init__(self, **_kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def submit(self, _bundle):
            pass

    monkeypatch.setattr(remote_engine, "WorkerManager", FakeWorkers)
    monkeypatch.setattr(remote_engine, "OutputPipeline", FakePipeline)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: FakeTokenizer())
    monkeypatch.setattr("easydel.inference.esurge.engine.admission.jax.process_count", lambda: 1)

    t = threading.Thread(target=_owner, daemon=True)
    t.start()
    handle = remote_engine.RemoteEngineHandle("127.0.0.1", port, auth_token=AUTH, connect_timeout_s=5.0)
    try:
        with pytest.raises(RequestPlaneError, match=r"unreachable|shut down"):
            handle.generate("hello", SamplingParams(max_tokens=16), request_id="r-die")
    finally:
        handle.close()
        t.join(timeout=5.0)


def test_s2_h6_generate_n_gt_1_events_cleaned_and_partial_batch_aborts_earlier(monkeypatch):
    from .test_remote_handle_front_load import _offline_handle

    handle = _offline_handle(monkeypatch)
    try:
        # n=3 request must clean up parent + all 3 child event keys from _registry.events.
        handle.generate("hi", SamplingParams(max_tokens=2, n=3, temperature=0.8), request_id="multi-n")
        assert not any(k == "multi-n" or k.startswith("multi-n-") for k in handle._registry.events)
        assert "multi-n" not in handle._registry.outputs

        # Partial batch failure: prompt 1 succeeds, prompt 2 raises -> prompt 1 is aborted and forgotten.
        aborted: list[str] = []
        monkeypatch.setattr(handle, "abort_request", aborted.append)
        real_add = handle._admission.add_request
        calls = {"n": 0}

        def failing_second(rid, prompt, params, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RequestPlaneError("second prompt rejected")
            # Keep prompt 1 unfinished so we verify finally aborts it.
            real_add(rid, prompt, params, **kw)
            handle._registry.outputs[rid].finished = False

        monkeypatch.setattr(handle._admission, "add_request", failing_second)
        with pytest.raises(RequestPlaneError, match="second prompt rejected"):
            handle.generate(["p1", "p2"], SamplingParams(max_tokens=2), request_id=["b1", "b2"])
        assert aborted == ["b1"]
        assert "b1" not in handle._registry.outputs and "b2" not in handle._registry.outputs
        assert "b1" not in handle._registry.events and "b2" not in handle._registry.events
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# S2-H7 & S2-L2: RoutedEngine slot leaks, atomic acquire, replace_member, dead counters
# ---------------------------------------------------------------------------


class _RouterMember:
    def __init__(self, name="m", *, alive=True, pending=0, running=0, barrier=None, sync_stream_err=None):
        self.esurge_name = name
        self.alive = alive
        self.max_model_len = 128
        self.max_num_seqs = 4
        self.page_size = 16
        self.num_pending_requests = pending
        self.num_running_requests = running
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda msgs, **kw: repr(msgs))
        self._barrier = barrier
        self._sync_stream_err = sync_stream_err
        self.generate_count = 0
        self.inner_closed = 0

    def generate(self, prompts, **_kw):
        self.generate_count += 1
        if self._barrier is not None:
            self._barrier.wait(timeout=5.0)
        return [SimpleNamespace(prompt=p) for p in prompts]

    def stream(self, prompt, **_kw):
        if self._sync_stream_err is not None:
            raise self._sync_stream_err
        yield SimpleNamespace(prompt=prompt)

    def chat(self, messages, *, stream=False, **_kw):
        if stream:

            def _gen():
                try:
                    yield "tok"
                finally:
                    self.inner_closed += 1

            g = _gen()
            # Track explicit close even if generator was never started.
            parent = self

            class _TrackedIter:
                def __iter__(self):
                    return self

                def __next__(self):
                    return next(g)

                def close(self):
                    parent.inner_closed += 1
                    g.close()

            return _TrackedIter()
        return SimpleNamespace(messages=messages)


def test_s2_h7_and_s2_l2_routed_engine_validation_slots_atomic_replace_and_counters():
    m0 = _RouterMember("m0", pending=3, running=2)
    m1 = _RouterMember("m1", alive=False, pending=99, running=88)
    router = RoutedEngine([m0, m1])

    # S2-L2: queue counters ignore dead members.
    assert router.num_pending_requests == 3
    assert router.num_running_requests == 2

    # S2-H7(b): request_id length validation & empty prompts.
    with pytest.raises(ValueError, match="Empty prompt list"):
        router.generate([])
    with pytest.raises(ValueError, match="must be a list"):
        router.generate(["p0", "p1"], request_id="only-one")
    with pytest.raises(ValueError, match="Length of request_id"):
        router.generate(["p0", "p1"], request_id=["only-one"])
    assert router._inflight == [0, 0]

    # S2-H7(b): mid-loop select_member failure rolls back already-acquired slots.
    calls = {"n": 0}
    orig_pick = router._pick_member_locked

    def die_on_second(pinned):
        calls["n"] += 1
        if calls["n"] == 2:
            m0.alive = False
        return orig_pick(pinned)

    router._pick_member_locked = die_on_second
    with pytest.raises(RequestPlaneError, match="no live replica"):
        router.generate(["p0", "p1"])
    assert router._inflight == [0, 0]
    m0.alive = True
    router._pick_member_locked = orig_pick

    # S2-H7(a): stream() empty prompts & synchronous stream error release slot.
    with pytest.raises(ValueError, match="Empty prompt list"):
        list(router.stream([]))
    m0._sync_stream_err = TypeError("bad stream kwargs")
    with pytest.raises(TypeError, match="bad stream kwargs"):
        list(router.stream("hi"))
    assert router._inflight == [0, 0]
    m0._sync_stream_err = None

    # S2-H7(c): unstarted chat(stream=True) releases slot & closes inner iterator on close() and GC.
    it1 = router.chat([{"role": "user", "content": "hi"}], stream=True)
    assert router._inflight == [1, 0]
    it1.close()
    assert router._inflight == [0, 0]
    assert m0.inner_closed == 1

    it2 = router.chat([{"role": "user", "content": "hi"}], stream=True)
    assert router._inflight == [1, 0]
    del it2
    gc.collect()
    assert router._inflight == [0, 0]
    assert m0.inner_closed == 2

    # S2-H7(d): atomic select+acquire prevents concurrent burst stampedes.
    barrier = threading.Barrier(8)
    members = [_RouterMember(f"r{i}", barrier=barrier) for i in range(4)]
    burst_router = RoutedEngine(members, affinity_max_imbalance=0)
    threads = [
        threading.Thread(target=burst_router.generate, args=(f"unique-prompt-{i}",)) for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert [m.generate_count for m in members] == [2, 2, 2, 2]
    assert burst_router._inflight == [0, 0, 0, 0]

    # S2-H7(e): replace_member gives replacement a fresh counter isolated from unwinding old requests.
    old_entered, release_old = threading.Event(), threading.Event()
    new_entered, release_new = threading.Event(), threading.Event()

    class _HeldMember(_RouterMember):
        def __init__(self, name, entered, release):
            super().__init__(name)
            self._entered = entered
            self._release = release

        def generate(self, prompts, **_kw):
            self._entered.set()
            self._release.wait(timeout=5.0)
            return [SimpleNamespace(prompt=p) for p in prompts]

    old_m = _HeldMember("old", old_entered, release_old)
    new_m = _HeldMember("new", new_entered, release_new)
    rep_router = RoutedEngine([old_m])

    t_old = threading.Thread(target=rep_router.generate, args=("p-old",))
    t_old.start()
    assert old_entered.wait(timeout=5.0)
    assert rep_router._inflight == [1]

    rep_router.replace_member(0, new_m)
    assert rep_router._inflight == [0]

    t_new = threading.Thread(target=rep_router.generate, args=("p-new",))
    t_new.start()
    assert new_entered.wait(timeout=5.0)
    assert rep_router._inflight == [1]

    release_old.set()
    t_old.join(timeout=5.0)
    assert rep_router._inflight == [1], "unwinding old request must not decrement replacement member's counter"

    release_new.set()
    t_new.join(timeout=5.0)
    assert rep_router._inflight == [0]


# ---------------------------------------------------------------------------
# S2-H8: Replica launcher lifecycle & parser forwarding (host, ray, app)
# ---------------------------------------------------------------------------


def test_s2_h8_leader_engine_main_closes_engine_on_sigterm(tmp_path):
    marker = tmp_path / "closed.txt"
    ready = tmp_path / "ready.txt"
    code = (
        "import os, sys, types, pathlib\n"
        "import easydel.infra.elarge.builders as builders\n"
        "import easydel.inference.esurge.serving.ray_replicas as rr\n"
        "import jax\n"
        "from easydel.inference.esurge.serving import host_replicas\n"
        "rr._tolerate_foreign_libtpu_lockfile = lambda: False\n"
        "rr._require_free_chips = lambda: None\n"
        "jax.distributed.initialize = lambda *a, **kw: None\n"
        "class FakeEngine:\n"
        "    alive = True\n"
        f"    def close(self): pathlib.Path({str(marker)!r}).write_text('closed')\n"
        "def fake_build(cfg):\n"
        f"    pathlib.Path({str(ready)!r}).write_text('ready')\n"
        "    return FakeEngine()\n"
        "builders.build_esurge = fake_build\n"
        "host_replicas.leader_engine_main()\n"
    )
    read_fd, write_fd = os.pipe()
    plan = ReplicaPlan(index=0, rank=0, num_processes=2, coordinator="127.0.0.1:19667", leaders=("127.0.0.1",))
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code, str(read_fd), str(os.getpid())],
            env=host_replicas.child_env(os.environ, sys.path, {"JAX_PLATFORMS": "cpu"}),
            pass_fds=(read_fd,),
        )
    finally:
        os.close(read_fd)
    with os.fdopen(write_fd, "wb") as sink:
        pickle.dump(({}, plan), sink)
    try:
        _wait_until(ready.exists, timeout=15.0)
        proc.terminate()
        assert proc.wait(timeout=10.0) == 0
        assert marker.read_text() == "closed"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_s2_h8_elarge_replica_connector_checks_port_and_forwards_parsers(monkeypatch):
    seen = {}

    def fake_handle(addr, port, **kwargs):
        seen["addr"] = addr
        seen["port"] = port
        seen.update(kwargs)
        return SimpleNamespace(alive=True)

    monkeypatch.setattr("easydel.inference.esurge.distributed.remote_engine.RemoteEngineHandle", fake_handle)
    cfg = {
        "model": {"name_or_path": "org/model", "tokenizer": "/tok", "model_type": "qwen3"},
        "esurge": {"parsing": {"tool_parser": "hermes", "reasoning_parser": "deepseek_r1"}},
    }
    closed_port = _free_port()
    connect_closed = app.elarge_replica_connector(cfg, control_port=closed_port, token="tok")
    with pytest.raises(RequestPlaneUnavailable):
        connect_closed("127.0.0.1")
    assert seen == {}, "closed port must short-circuit via port_open() before RemoteEngineHandle"

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        open_port = listener.getsockname()[1]
        connect_open = app.elarge_replica_connector(cfg, control_port=open_port, token="tok")
        connect_open("127.0.0.1")
    assert seen["port"] == open_port
    assert seen["tool_parser"] == "hermes"
    assert seen["reasoning_parser"] == "deepseek_r1"
    assert seen["model_type"] == "qwen3"
    assert seen["tokenizer_source"] == "/tok"


def test_s2_h8_ray_routed_engine_close_order_and_launch_error_cancellation(monkeypatch):
    events: list[str] = []

    class FakeTask:
        def __init__(self, idx):
            self.idx = idx

    class FakeRemoteFn:
        def __init__(self, fn, fail_on=None):
            self.fn = fn
            self.fail_on = fail_on
            self.count = 0

        def options(self, **_kw):
            return self

        def remote(self, *args, **_kw):
            self.count += 1
            if self.fail_on is not None and self.count == self.fail_on:
                raise RuntimeError("submit failed")
            return FakeTask(self.count)

    plan0 = ReplicaPlan(
        index=0, rank=0, num_processes=1, coordinator="127.0.0.1:19667", leaders=("127.0.0.1", "127.0.0.2")
    )
    plan1 = ReplicaPlan(
        index=1, rank=0, num_processes=1, coordinator="127.0.0.2:19667", leaders=("127.0.0.1", "127.0.0.2")
    )
    nodes = [
        {"NodeID": "n0", "NodeManagerAddress": "127.0.0.1", "Resources": {"TPU": 4}},
        {"NodeID": "n1", "NodeManagerAddress": "127.0.0.2", "Resources": {"TPU": 4}},
    ]
    cfg = {
        "model": {"name_or_path": "m", "model_type": "qwen3"},
        "sharding": {"axis_dims": [1, 2, 1, 1, 1, 1], "axis_names": ["pp", "dp", "fsdp", "ep", "tp", "sp"]},
    }

    fake_strat_mod = types.ModuleType("ray.util.scheduling_strategies")
    fake_strat_mod.NodeAffinitySchedulingStrategy = lambda node_id, soft=False: (node_id, soft)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", fake_strat_mod)
    monkeypatch.setattr(app, "elarge_replica_connector", lambda *_a, **_kw: lambda addr: SimpleNamespace(
        alive=True, esurge_name="m", max_model_len=128, max_num_seqs=4, page_size=16,
        close=lambda: events.append(f"member.close:{addr}"),
    ))

    # Case 1: task submission failure on node 2 cancels node 1's already-submitted task.
    cancelled: list[int] = []
    call_idx = {"n": 0}

    def fake_remote_fail(**_kw):
        def wrap(fn):
            call_idx["n"] += 1
            return FakeRemoteFn(fn, fail_on=2 if call_idx["n"] == 2 else None)

        return wrap

    fake_ray_fail = SimpleNamespace(
        remote=fake_remote_fail,
        get=lambda _refs: [(plan0, {}), (plan1, {})],
        cancel=lambda task, force=True: cancelled.append(task.idx),
    )
    with pytest.raises(RuntimeError, match="submit failed"):
        ray_replicas._launch(fake_ray_fail, cfg, nodes)
    assert cancelled == [1]

    # Case 2: RayRoutedEngine.close() stops attacher BEFORE closing members, then cancels Ray tasks.
    orig_stop = app.ReplicaAttacher.stop

    def tracked_stop(self):
        events.append("attacher.stop")
        orig_stop(self)

    monkeypatch.setattr(app.ReplicaAttacher, "stop", tracked_stop)
    fake_ray_ok = SimpleNamespace(
        remote=lambda **_kw: lambda fn: FakeRemoteFn(fn),
        get=lambda _refs: [(plan0, {}), (plan1, {})],
        wait=lambda tasks, **_kw: ([], tasks),
        cancel=lambda task, force=True: events.append(f"ray.cancel:{task.idx}"),
    )
    router = ray_replicas._launch(fake_ray_ok, cfg, nodes)
    router.close()
    assert events.index("attacher.stop") < events.index("member.close:127.0.0.1")
    assert events.index("member.close:127.0.0.1") < events.index("ray.cancel:1")


def test_s2_h8_app_front_teardown_on_sigterm_vs_decoupled_engine_death(monkeypatch, tmp_path):
    from easydel.inference.esurge.serving import replicas as replicas_mod

    def _run_scenario(mode: str):
        closed_engine = []
        front_procs = []
        created_dirs = []

        class FakeFront:
            def __init__(self, idx):
                self.pid = 1000 + idx
                self.terminated = False
                self.returncode = None

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.terminated = True

        class FakeEngine:
            is_coordination_leader = True
            alive = True

            def close(self):
                closed_engine.append(True)

        engine = FakeEngine()
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"tok")
        os.close(write_fd)
        plan = ReplicaPlan(
            index=0, rank=0, num_processes=1, coordinator="127.0.0.1:19667", leaders=("127.0.0.1",), token_fd=read_fd
        )
        cfg = ServingConfig.from_mapping({"model": {"checkpoint": "m"}, "server": {"dashboard": False}})
        sock_dir = tmp_path / f"fronts-{mode}"
        sock_dir.mkdir()
        created_dirs.append(sock_dir)

        import easydel as real_ed

        fake_jax = SimpleNamespace(
            distributed=SimpleNamespace(initialize=lambda *a, **kw: None),
            process_count=lambda: 1,
            process_index=lambda: 0,
        )
        monkeypatch.setitem(sys.modules, "jax", fake_jax)
        monkeypatch.setattr(real_ed, "eSurge", lambda **kw: engine)
        monkeypatch.setattr("easydel.utils.compilation_cache.cache_namespace", lambda *_a: "ns")
        monkeypatch.setattr("easydel.utils.compilation_cache.configure_compilation_cache", lambda *_a, **_kw: None)
        monkeypatch.setattr(app, "build_loading_kwargs", lambda *a, **kw: {})
        monkeypatch.setattr(app, "FRONT_PROCESSES", 2)
        monkeypatch.setattr(app, "resolve_bind_host", lambda *_a: "127.0.0.1")
        monkeypatch.setattr(dashboard, "create_front_socket_dir", lambda: str(sock_dir))
        monkeypatch.setattr(replicas_mod, "spawn_front", lambda _p, _t, index=0: (front_procs.append(FakeFront(index)) or front_procs[-1]))

        def fake_sleep(_s):
            if mode == "sigterm":
                raise app.ShutdownRequested("SIGTERM")
            engine.alive = False

        monkeypatch.setattr(app.time, "sleep", fake_sleep)
        if mode == "sigterm":
            with pytest.raises(app.ShutdownRequested):
                app._serve(cfg, plan)
        else:
            app._serve(cfg, plan)
        return closed_engine, front_procs, sock_dir

    # SIGTERM: fronts are terminated and front_socket_dir is removed cleanly.
    closed, fronts, sock_dir = _run_scenario("sigterm")
    assert closed == [True]
    assert len(fronts) == 2 and all(f.terminated for f in fronts)
    assert not sock_dir.exists()

    # Engine death: leader exits loop, keeps fronts alive (replicas.py:284-286), preserves front_socket_dir.
    closed, fronts, sock_dir = _run_scenario("engine_death")
    assert closed == [True]
    assert len(fronts) == 2 and not any(f.terminated for f in fronts)
    assert sock_dir.exists()


def test_s2_h8_ray_run_replica_exits_when_leader_engine_dies(monkeypatch):
    closed = []

    class FakeEngine:
        is_coordination_leader = True
        alive = False
        _scheduler_exception = RuntimeError("runner crashed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(ray_replicas, "_tolerate_foreign_libtpu_lockfile", lambda: False)
    monkeypatch.setattr(ray_replicas, "_require_free_chips", lambda: None)
    monkeypatch.setattr("jax.distributed.initialize", lambda *a, **kw: None)
    monkeypatch.setattr("easydel.infra.elarge.builders.build_esurge", lambda _cfg: FakeEngine())
    plan = ReplicaPlan(index=0, rank=0, num_processes=1, coordinator="127.0.0.1:19667", leaders=("127.0.0.1",))
    with pytest.raises(RuntimeError, match="leader engine failed"):
        ray_replicas._run_replica({}, plan, 5.0)
    assert closed == [True]


# ---------------------------------------------------------------------------
# S2-M7: ZMQ/plane reliability, HWM/seq, n_samples==0, reset, IPv6 & dual-stack
# ---------------------------------------------------------------------------


def test_s2_m7_empty_client_id_rejected_stale_client_evicted_and_worker_outbox_drained():
    import zmq

    port = _free_port()
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=1,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=5.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=0.4,
    )
    leader.start()
    ctx = zmq.Context.instance()
    empty_sock = ctx.socket(zmq.DEALER)
    empty_sock.setsockopt(zmq.IDENTITY, b"empty-client")
    empty_sock.setsockopt(zmq.LINGER, 0)
    empty_sock.connect(f"tcp://127.0.0.1:{port}")
    silent_sock = ctx.socket(zmq.DEALER)
    silent_sock.setsockopt(zmq.IDENTITY, b"silent-client")
    silent_sock.setsockopt(zmq.LINGER, 0)
    silent_sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        # Empty client_id must be rejected.
        empty_sock.send_multipart(
            [wire.encode_message(wire.Hello(rank=-1, world_size=0, config_fingerprint="", auth=AUTH, role=wire.ROLE_CLIENT, client_id=""))]
        )
        time.sleep(0.2)
        assert not empty_sock.poll(0)
        assert leader.client_identity("") is None

        # Valid client that goes silent must be evicted after heartbeat_timeout_s.
        silent_sock.send_multipart(
            [wire.encode_message(wire.Hello(rank=-1, world_size=0, config_fingerprint="", auth=AUTH, role=wire.ROLE_CLIENT, client_id="temp-client"))]
        )
        assert silent_sock.poll(2000)
        assert isinstance(wire.decode_message(silent_sock.recv_multipart()[0]), wire.HelloOk)
        assert leader.client_identity("temp-client") is not None
        _wait_until(lambda: leader.client_identity("temp-client") is None, timeout=3.0)
    finally:
        empty_sock.close(linger=0)
        silent_sock.close(linger=0)
        leader.shutdown("done")

    # ZmqWorkerCoordinator drains stale _worker_outbox entries on start().
    worker = ZmqWorkerCoordinator(
        _MockRunner(), rank=1, world_size=2, leader_addr="127.0.0.1", control_port=_free_port(),
        auth_token=AUTH, config_fingerprint=FINGERPRINT, connect_timeout_s=0.1,
    )
    worker.send_to_leader(wire.Control(kind="stale"))
    assert not worker._worker_outbox.empty()
    with pytest.raises(StepCoordinationError):
        worker.start()
    assert worker._worker_outbox.empty()


def test_s2_m7_per_peer_seq_gap_n_samples_zero_decode_fail_and_reset():
    # Per-peer OutputUpdate.seq and OriginRequestPlane seq-gap detection + AdmitAck decode error wakeup.
    sent: list[tuple[bytes, wire.WireMessage, bytes | None]] = []

    class FakeCoord:
        world_size = 2

        def set_plane_handler(self, h):
            self.handler = h

        def send_to_peer(self, ident, msg, payload=None):
            sent.append((ident, msg, payload))

        def send_to_leader(self, msg, payload=None):
            self.last_sent = (msg, payload)

    coord = FakeCoord()
    submitted: list[list] = []
    owner = OwnerRequestPlane(
        coordinator=coord,
        scheduler_submit=submitted.append,
        abort_request=lambda _r: None,
        apply_stop_strings=lambda _h: None,
        submit_local_outputs=lambda _b: None,
        next_arrival_stamp=itertools.count(1).__next__,
        info=lambda *a, **k: None,
    )

    # n_samples == 0 must be NACKed and must NOT set needs_tee.
    owner._handle_admit(
        b"peer-a",
        wire.Admit(request_id="zero", origin_rank=1, origin_counter=1, n_samples=0),
        wire.encode_payload([]),
    )
    assert sent[-1][1].ok is False
    assert not owner.needs_tee
    sent.clear()

    # Two distinct peers each receive monotonic per-peer seq 1, 2 (no cross-peer interleaving gaps).
    owner._handle_admit(
        b"peer-a",
        wire.Admit(request_id="req-a", origin_rank=-1, origin_counter=1, n_samples=1),
        wire.encode_payload([FakeEngineRequest("req-a")]),
    )
    owner._handle_admit(
        b"peer-b",
        wire.Admit(request_id="req-b", origin_rank=-1, origin_counter=1, n_samples=1),
        wire.encode_payload([FakeEngineRequest("req-b")]),
    )
    owner_a = submitted[0][0].request_id
    owner_b = submitted[1][0].request_id
    sent.clear()
    owner._fan_out({0: SimpleNamespace(outputs=[EngineCoreOutput(request_id=owner_a, new_token_ids=[1])], finished_requests=None)})
    owner._fan_out({0: SimpleNamespace(outputs=[EngineCoreOutput(request_id=owner_b, new_token_ids=[2])], finished_requests=None)})
    owner._fan_out({0: SimpleNamespace(outputs=[EngineCoreOutput(request_id=owner_a, new_token_ids=[3])], finished_requests=None)})
    a_seqs = [m.seq for ident, m, _p in sent if ident == b"peer-a" and isinstance(m, wire.OutputUpdate)]
    b_seqs = [m.seq for ident, m, _p in sent if ident == b"peer-b" and isinstance(m, wire.OutputUpdate)]
    assert a_seqs == [1, 2]
    assert b_seqs == [1]

    # OriginRequestPlane seq-gap detection, AdmitAck corrupt replay wakeup, and reset() counter sync.
    origin_coord = FakeCoord()
    origin = OriginRequestPlane(
        coordinator=origin_coord,
        submit_outputs=lambda _b: None,
        alive=lambda: True,
        rank=1,
        admit_timeout_s=5.0,
        info=lambda *a, **k: None,
    )
    payload_ok = wire.encode_payload(([EngineCoreOutput(request_id="r", new_token_ids=[1])], None))
    origin._handle_inbound(wire.OutputUpdate(seq=1), payload_ok)
    origin._handle_inbound(wire.OutputUpdate(seq=2), payload_ok)
    assert origin.seq_gaps == 0
    origin._handle_inbound(wire.OutputUpdate(seq=5), payload_ok)
    assert origin.seq_gaps == 1

    # Corrupt AdmitAck replay payload wakes waiting caller immediately with NACK instead of hanging.
    def _send_bad_replay(msg, _payload=None):
        if isinstance(msg, wire.Admit):
            threading.Thread(
                target=lambda: origin._handle_inbound(wire.AdmitAck(request_id=msg.request_id, ok=True), b"not-pickle"),
                daemon=True,
            ).start()

    origin_coord.send_to_leader = _send_bad_replay
    t0 = time.monotonic()
    with pytest.raises(RequestPlaneError, match="decode/apply AdmitAck"):
        origin.submit_remote([FakeEngineRequest("bad-replay")])
    assert time.monotonic() - t0 < 2.0

    # Reset resets coalesce counters on both OwnerRequestPlane and OriginRequestPlane.
    owner.admit_local([FakeEngineRequest(AUTO_1)])
    owner.reset()
    origin.reset()
    assert next(owner._local_coalesce_counter) == 1
    assert next(origin._coalesce_counter) == 1
    assert origin.seq_gaps == 0


def test_s2_m7_ipv6_split_host_port_dual_stack_discovery_and_zmq_ipv6(monkeypatch):
    # _split_host_port handles bare and bracketed IPv6 without misparsing "::1" as (":", 1).
    assert _split_host_port("::1") == ("::1", None)
    assert _split_host_port("2001:db8::1") == ("2001:db8::1", None)
    assert _split_host_port("[::1]:19666") == ("::1", 19666)
    assert _split_host_port("[::1]") == ("::1", None)
    assert _split_host_port("10.0.0.1:8476") == ("10.0.0.1", 8476)

    # Dual-stack headless service must not double-count pods.
    dual_entries = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::1", 0, 0, 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::2", 0, 0, 0)),
    ]
    def fake_getaddrinfo(host, port, family=0, **kwargs):
        del host, port, kwargs
        return [e for e in dual_entries if family in (0, e[0])]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    res_auto = discovery.resolve_service_hosts("headless.svc", world_size=2)
    assert res_auto.hosts == ["10.0.0.1", "10.0.0.2"]
    res_v6 = discovery.resolve_service_hosts("headless.svc", world_size=2, family=socket.AF_INET6)
    assert res_v6.hosts == ["fd00::1", "fd00::2"]

    # Real ZMQ IPv6 loopback handshake + step replay when host supports IPv6 loopback.
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
            v6_port = probe.getsockname()[1]
    except OSError:
        return
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=2,
        bind_host="::1",
        control_port=v6_port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        ready_timeout_s=5.0,
        step_timeout_s=5.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=5.0,
    )
    worker = ZmqWorkerCoordinator(
        _MockRunner(),
        rank=1,
        world_size=2,
        leader_addr="::1",
        control_port=v6_port,
        auth_token=AUTH,
        config_fingerprint=FINGERPRINT,
        connect_timeout_s=5.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=5.0,
    )
    stop_event = threading.Event()
    t = threading.Thread(target=lambda: (worker.start(), worker.run_worker_loop(stop_event)), daemon=True)
    t.start()
    try:
        leader.start()
        leader.execute_sync(_sched_output("V6"))
        _wait_until(lambda: worker._runner.calls == ["sync:V6"])
    finally:
        leader.shutdown("done")
        stop_event.set()
        t.join(timeout=5.0)
        worker.shutdown()


def test_s2_m7_remote_handle_chat_preserves_logit_bias_and_stream_empty_prompts(monkeypatch):
    from .test_remote_handle_front_load import _offline_handle
    from .test_remote_handle_parsers import TOOLS

    handle = _offline_handle(monkeypatch)
    try:
        seen_params = {}
        orig_gen = handle.generate

        def spy_gen(prompts, sampling_params=None, **kw):
            seen_params["params"] = sampling_params
            return orig_gen(prompts, sampling_params=sampling_params, **kw)

        monkeypatch.setattr(handle, "generate", spy_gen)
        monkeypatch.setattr("easydel.inference.esurge.engine.chat_templating.format_chat_prompt", lambda *a, **kw: "p")
        handle.chat(
            [{"role": "user", "content": "hi"}],
            tools=TOOLS,
            sampling_params=SamplingParams(max_tokens=2, logit_bias={42: 1.5}),
            request_id="bias-req",
        )
        assert seen_params["params"].logit_bias == {42: 1.5}
        assert seen_params["params"].skip_special_tokens is False

        with pytest.raises(ValueError, match="Empty prompt list"):
            list(handle.stream([]))

        # Failing add_request inside stream() must clean up registry entries.
        def broken_add(rid, prompt, params, **kw):
            handle._registry.events[rid] = threading.Event()
            handle._registry.records[rid] = SimpleNamespace()
            raise RequestPlaneError("admit failed")

        monkeypatch.setattr(handle._admission, "add_request", broken_add)
        with pytest.raises(RequestPlaneError, match="admit failed"):
            list(handle.stream("hi", request_id="stream-fail"))
        assert "stream-fail" not in handle._registry.events
        assert "stream-fail" not in handle._registry.records
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# S2-M12 (serving portion): serving/config.py and serving/dashboard.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"server": ServerSpec(port=19666), "coordination": CoordinationSpec(control_port=19666)}, "server.port"),
        ({"server": ServerSpec(port=19667), "coordination": CoordinationSpec(control_port=19666)}, "server.port"),
        ({"server": ServerSpec(profiler_port=19666), "coordination": CoordinationSpec(control_port=19666)}, "profiler_port"),
        ({"server": ServerSpec(profiler_port=19667), "coordination": CoordinationSpec(control_port=19666)}, "profiler_port"),
    ],
)
def test_s2_m12_serving_config_port_collisions(kwargs, match):
    with pytest.raises(ServingConfigError, match=match):
        ServingConfig(model=ModelSpec(checkpoint="c"), **kwargs)


def test_s2_m12_serving_config_overflow_and_numeric_checks():
    assert CoordinationSpec(control_port=65533).control_port == 65533
    for bad_port in (65534, 65535):
        with pytest.raises(ServingConfigError, match="65533"):
            CoordinationSpec(control_port=bad_port)
    for bad_timeout in (True, False, float("nan"), float("inf"), -1.0, 0.0):
        with pytest.raises(ServingConfigError):
            CoordinationSpec(heartbeat_timeout_s=bad_timeout)
        with pytest.raises(ServingConfigError):
            CoordinationSpec(token_timeout_s=bad_timeout)
    for bad_cap in (0, -5, True, False):
        with pytest.raises(ServingConfigError, match="max_concurrent_generations"):
            ServerSpec(max_concurrent_generations=bad_cap)
    assert ServerSpec(max_concurrent_generations=16).max_concurrent_generations == 16


def test_s2_m12_dashboard_ipv6_empty_host_empty_merge_and_peer_fetch(tmp_path):
    sup_v6 = dashboard.DashboardSupervisor(host="::", port=9090, endpoint_port=8000, log=lambda _m: None)
    assert sup_v6.listen == "[::]:9090"
    assert sup_v6.upstream == "http://[::1]:8000"

    sup_empty = dashboard.DashboardSupervisor(host="", port=9090, endpoint_port=8000, log=lambda _m: None)
    assert sup_empty.listen == "0.0.0.0:9090"
    assert sup_empty.upstream == "http://127.0.0.1:8000"

    # Empty payloads list must not raise IndexError.
    merged_empty = dashboard.merge_front_metrics([], window_seconds=300.0)
    assert merged_empty["fronts"] == 0 and merged_empty["status"] is None

    # FrontMetricsMiddleware with sockets=[] reports fronts_expected == 1 (not 0).
    async def dummy_app(scope, receive, send):
        body = json.dumps({"status": "ready", "total_requests": 1, "engine_metrics": {}}).encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    mw = dashboard.FrontMetricsMiddleware(
        dummy_app,
        own_socket="",
        sockets=[],
        local_metrics=lambda: {"status": "ready"},
        request_rows=lambda _n: [],
        window_seconds=300.0,
    )
    sent_msgs: list[dict] = []

    async def _send(msg):
        sent_msgs.append(msg)

    async def _run_mw():
        await mw({"type": "http", "method": "GET", "path": "/metrics", "client": ("127.0.0.1", 1)}, None, _send)

    asyncio.run(_run_mw())
    body = json.loads(b"".join(m.get("body", b"") for m in sent_msgs if m["type"] == "http.response.body"))
    assert body["fronts"] == 1 and body["fronts_expected"] == 1

    # _fetch_peer reads over a real unix socket and closes cleanly with wait_closed().
    sock_path = str(tmp_path / "peer.sock")
    srv = dashboard.bind_unix_socket(sock_path)

    def _serve_once():
        conn, _ = srv.accept()
        conn.recv(4096)
        payload = b'{"status":"ready","total_requests":7}'
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + payload
        )
        conn.close()

    t = threading.Thread(target=_serve_once, daemon=True)
    t.start()
    try:
        result = asyncio.run(dashboard._fetch_peer(sock_path))
        assert result == {"status": "ready", "total_requests": 7}
    finally:
        t.join(timeout=5.0)
        srv.close()


# ---------------------------------------------------------------------------
# S2-L1: start() socket cleanup on decode error & worker Step missing payload
# ---------------------------------------------------------------------------


def test_s2_l1_bad_handshake_closes_socket_and_missing_step_payload_nacks():
    import zmq

    ctx = zmq.Context.instance()
    port = _free_port()

    def _corrupt_server():
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{port}")
        try:
            for _ in range(2):
                if not sock.poll(5000):
                    return
                ident, _hello = sock.recv_multipart()
                sock.send_multipart([ident, b"\xff\xffnot-msgpack"])
        finally:
            sock.close(linger=0)

    t = threading.Thread(target=_corrupt_server, daemon=True)
    t.start()
    worker = ZmqWorkerCoordinator(
        _MockRunner(), rank=1, world_size=2, leader_addr="127.0.0.1", control_port=port,
        auth_token=AUTH, config_fingerprint=FINGERPRINT, connect_timeout_s=5.0,
    )
    client = remote_engine._ClientChannel(
        leader_addr="127.0.0.1", control_port=port, auth_token=AUTH, client_id="c",
        connect_timeout_s=5.0, on_dead=lambda _r: None,
    )
    try:
        with pytest.raises(StepCoordinationError, match="undecodable handshake"):
            worker.start()
        assert worker._sock is None
        with pytest.raises(RequestPlaneError, match="undecodable handshake"):
            client.start()
        assert client._sock is None
    finally:
        t.join(timeout=5.0)

    # Worker receiving payload-less Step must NACK and raise StepCoordinationError (not IndexError).
    port2 = _free_port()
    nacks: list[wire.StepAck] = []

    def _missing_payload_leader():
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{port2}")
        try:
            ident, _hello = sock.recv_multipart()
            sock.send_multipart([ident, wire.encode_message(wire.HelloOk(rank=1))])
            # Wait for Ready
            while sock.poll(5000):
                _id, raw = sock.recv_multipart()
                msg = wire.decode_message(raw)
                if isinstance(msg, wire.Ready):
                    sock.send_multipart([ident, wire.encode_message(wire.Step(step_id=1, mode=wire.STEP_MODE_SYNC))])
                    break
            while sock.poll(5000):
                _id, raw = sock.recv_multipart()
                msg = wire.decode_message(raw)
                if isinstance(msg, wire.StepAck):
                    nacks.append(msg)
                    return
        finally:
            sock.close(linger=0)

    t2 = threading.Thread(target=_missing_payload_leader, daemon=True)
    t2.start()
    w2 = ZmqWorkerCoordinator(
        _MockRunner(), rank=1, world_size=2, leader_addr="127.0.0.1", control_port=port2,
        auth_token=AUTH, config_fingerprint=FINGERPRINT, connect_timeout_s=5.0,
    )
    try:
        w2.start()
        with pytest.raises(StepCoordinationError, match="missing payload frame"):
            w2.run_worker_loop(threading.Event())
    finally:
        w2.shutdown()
        t2.join(timeout=5.0)
    assert len(nacks) == 1 and nacks[0].ok is False and "missing payload frame" in (nacks[0].error or "")


# ---------------------------------------------------------------------------
# S2-L3: serving/keys.py OSError wrapping, non-hardlink fallback, st_uid check
# ---------------------------------------------------------------------------


def test_s2_l3_keys_oserror_wrapping_hardlink_fallback_and_uid_check(tmp_path, monkeypatch):
    # 1. Creation OSError wrapped in ApiKeyFileError.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with pytest.raises(ApiKeyFileError, match="cannot create API key file"):
        keys.load_or_create_api_key(blocker / "key")

    # 2. Hardlink unsupported (e.g. GCS-fuse) falls back to O_CREAT | O_EXCL with mode 0600.
    real_link = os.link

    def no_hardlink(src, dst):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    monkeypatch.setattr(os, "link", no_hardlink)
    gcs_key_path = tmp_path / "gcs" / "api_key"
    key, created = keys.load_or_create_api_key(gcs_key_path)
    assert created is True and key.startswith("sk-") and len(key) >= keys.MIN_KEY_LENGTH
    assert (gcs_key_path.stat().st_mode & 0o777) == 0o600
    assert [p.name for p in gcs_key_path.parent.iterdir()] == ["api_key"]
    monkeypatch.setattr(os, "link", real_link)

    # 3. _check_private rejects files owned by a different uid.
    monkeypatch.setattr(os, "getuid", lambda: gcs_key_path.stat().st_uid + 1)
    with pytest.raises(ApiKeyFileError, match="owned by uid"):
        keys.load_or_create_api_key(gcs_key_path)


# ---------------------------------------------------------------------------
# S2-L4: serving/cluster.py IPv6 URL bracketing
# ---------------------------------------------------------------------------


def test_s2_l4_public_urls_brackets_ipv6_hosts():
    assert public_urls("::1", 8000, internal_ip=lambda: None, external_ip=lambda: None) == {
        "internal": "http://[::1]:8000/v1"
    }
    assert public_urls(
        "::",
        8000,
        internal_ip=lambda: "fd00::10",
        external_ip=lambda: "2001:db8::20",
    ) == {
        "internal": "http://[fd00::10]:8000/v1",
        "external": "http://[2001:db8::20]:8000/v1",
    }


# ---------------------------------------------------------------------------
# Audit #5: Stop-string hit & local_direct coalesced abort cleanup
# (default include_finished_set=False)
# ---------------------------------------------------------------------------


def test_leak_fix_5_remote_stop_string_retires_owner_and_origin_without_finished_set():
    with _PlaneHarness() as plane:
        # 1. Non-coalescable n=1 stop hit retires all 5 owner dicts and all 3 origin structures.
        plane.origin.submit_remote([FakeEngineRequest("dist-stop-1")])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id
        assert plane.owner.needs_tee
        assert plane.origin.is_remote("dist-stop-1")

        rest = plane.origin.forward_stop_hits({"dist-stop-1": "</s>", "local-only": "x"})
        assert rest == {"local-only": "x"}
        assert not plane.origin.is_remote("dist-stop-1")
        assert plane.origin._remote_ids == set()
        assert plane.origin._child_to_parent == {}
        assert plane.origin._live_children == {}

        _wait_until(lambda: plane.owner_engine.stop_hits == [{owner_id: "</s>"}])
        _wait_until(lambda: not plane.owner.needs_tee)
        assert plane.owner._owner_to_origin == {}
        assert plane.owner._origin_to_owner == {}
        assert plane.owner._owner_child_to_parent == {}
        assert plane.owner._children == {}
        assert plane.owner._parents == {}

        # 2. Non-coalescable n=2: stopping sample 0 leaves sample 1 live; stopping sample 1 retires all.
        n2_reqs = [
            FakeEngineRequest("dist-stop-n2-0", parent_request_id="dist-stop-n2", sample_index=0),
            FakeEngineRequest("dist-stop-n2-1", parent_request_id="dist-stop-n2", sample_index=1),
        ]
        plane.origin.submit_remote(n2_reqs)
        _wait_until(lambda: len(plane.owner_engine.submitted) == 2)
        owner_child_0 = plane.owner_engine.submitted[1][0].request_id
        owner_child_1 = plane.owner_engine.submitted[1][1].request_id

        plane.origin.forward_stop_hits({"dist-stop-n2-0": "STOP"})
        assert not plane.origin.is_remote("dist-stop-n2-0")
        assert plane.origin.is_remote("dist-stop-n2")
        assert plane.origin.is_remote("dist-stop-n2-1")
        _wait_until(lambda: len(plane.owner_engine.stop_hits) == 2)
        assert plane.owner_engine.stop_hits[1] == {owner_child_0: "STOP"}
        assert plane.owner.needs_tee

        plane.origin.forward_stop_hits({"dist-stop-n2-1": "STOP"})
        assert plane.origin._remote_ids == set()
        assert plane.origin._child_to_parent == {}
        assert plane.origin._live_children == {}
        _wait_until(lambda: len(plane.owner_engine.stop_hits) == 3)
        assert plane.owner_engine.stop_hits[2] == {owner_child_1: "STOP"}
        _wait_until(lambda: not plane.owner.needs_tee)
        assert plane.owner._owner_to_origin == {}
        assert plane.owner._origin_to_owner == {}
        assert plane.owner._owner_child_to_parent == {}
        assert plane.owner._children == {}
        assert plane.owner._parents == {}

        # 3. Coalesced stop hit (owner-first + remote attached): remote stop retires entry & log.
        plane.owner.admit_local([FakeEngineRequest(AUTO_1)])
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1)])
        plane.tee(EngineCoreOutput(request_id=AUTO_1, new_token_ids=[1, 2]))
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 1)

        plane.origin.forward_stop_hits({AUTO_1: "<stop>"})
        assert not plane.origin.is_remote(AUTO_1)
        _wait_until(lambda: len(plane.owner_engine.stop_hits) == 4)
        assert plane.owner_engine.stop_hits[3] == {AUTO_1: "<stop>"}
        _wait_until(lambda: not plane.owner.needs_tee)
        assert plane.owner._entries == {}
        assert plane.owner._by_counter == {}
        assert plane.owner._scheduled_children == {}
        assert plane.owner._subscriber_children == {}

        # 4. Coalesced stop hit initiated on owner (remote-first + owner attached):
        #    owner translate_local_stop_hits propagates finish to remote and retires entry.
        auto_2 = "req-0000000002"
        plane.origin.submit_remote([FakeEngineRequest(auto_2)])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 4)
        sched_2 = plane.owner_engine.submitted[3][0].request_id
        plane.owner.admit_local([FakeEngineRequest(auto_2)])
        plane.tee(EngineCoreOutput(request_id=sched_2, new_token_ids=[7]))
        _wait_until(lambda: any(o.request_id == auto_2 for o in plane.origin_sink.all_outputs()))

        translated = plane.owner.translate_local_stop_hits({auto_2: "DONE"})
        assert translated == {sched_2: "DONE"}
        _wait_until(lambda: auto_2 in plane.origin_sink.all_finished())
        _wait_until(lambda: not plane.origin.is_remote(auto_2))
        _wait_until(lambda: not plane.owner.needs_tee)
        assert plane.owner._entries == {}


def test_leak_fix_5_local_direct_coalesced_abort_retires_entry_and_notifies_remote():
    # 1. Unshared local_direct abort retires entry immediately without recursive _abort_request.
    with _PlaneHarness() as plane:
        plane.owner.admit_local([FakeEngineRequest(AUTO_1)])
        assert plane.owner.needs_tee
        assert len(plane.owner._entries) == 1
        plane.owner.notify_local_abort(AUTO_1)
        assert plane.owner_engine.aborted == []
        assert not plane.owner.needs_tee
        assert plane.owner._entries == {}
        assert plane.owner._by_counter == {}
        assert plane.owner._scheduled_children == {}
        assert plane.owner._subscriber_children == {}

    # 2. Shared local_direct abort (with remote subscriber attached):
    #    notifies remote with finished_requests (no hang!) and retires entry + log.
    with _PlaneHarness() as plane:
        plane.owner.admit_local([FakeEngineRequest(AUTO_1)])
        plane.tee(EngineCoreOutput(request_id=AUTO_1, new_token_ids=[5, 6]))
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1)])
        _wait_until(lambda: any(o.request_id == AUTO_1 for o in plane.origin_sink.all_outputs()))

        plane.owner.notify_local_abort(AUTO_1)
        assert plane.owner_engine.aborted == []
        _wait_until(lambda: AUTO_1 in plane.origin_sink.all_finished())
        _wait_until(lambda: not plane.origin.is_remote(AUTO_1))
        _wait_until(lambda: not plane.owner.needs_tee)
        assert plane.owner._entries == {}
        assert plane.owner._scheduled_children == {}
        assert plane.owner._subscriber_children == {}

        # 3. Partial n=2 local_direct sample abort notifies remote for sample 0 while sample 1 finishes.
        auto_2 = "req-0000000002"
        n2_reqs = [
            FakeEngineRequest(f"{auto_2}-0", parent_request_id=auto_2, sample_index=0),
            FakeEngineRequest(f"{auto_2}-1", parent_request_id=auto_2, sample_index=1),
        ]
        plane.owner.admit_local(n2_reqs)
        plane.origin.submit_remote(
            [
                FakeEngineRequest(f"{auto_2}-0", parent_request_id=auto_2, sample_index=0),
                FakeEngineRequest(f"{auto_2}-1", parent_request_id=auto_2, sample_index=1),
            ]
        )
        plane.owner.notify_local_abort(f"{auto_2}-0")
        _wait_until(lambda: f"{auto_2}-0" in plane.origin_sink.all_finished())
        assert plane.origin.is_remote(f"{auto_2}-1")
        assert plane.owner.needs_tee

        plane.tee(
            EngineCoreOutput(request_id=f"{auto_2}-1", new_token_ids=[9], finish_reason=FinishReason.STOP)
        )
        _wait_until(lambda: not plane.origin.is_remote(f"{auto_2}-1"))
        plane.owner._sweep_retention()
        assert not plane.owner.needs_tee
        assert plane.owner._entries == {}


# ---------------------------------------------------------------------------
# Audit #6 (distributed side): RemoteEngineHandle MetricsCollector cleanup
# ---------------------------------------------------------------------------


def test_leak_fix_6_remote_handle_abort_and_owner_death_complete_metrics(monkeypatch):
    from easydel.inference.esurge import metrics as metrics_mod

    from .test_remote_handle_front_load import _offline_handle

    collector = metrics_mod.MetricsCollector(enable_detailed_logging=False)
    monkeypatch.setattr(metrics_mod, "_global_metrics_collector", collector)

    handle = _offline_handle(monkeypatch)
    sent_aborts: list[str] = []

    def send_to_leader(message, payload=None):
        del payload
        if isinstance(message, wire.Admit):
            handle._channel.handler(wire.AdmitAck(request_id=message.request_id, ok=True), None)
        elif isinstance(message, wire.AbortReq):
            sent_aborts.append(message.request_id)

    monkeypatch.setattr(handle._channel, "send_to_leader", send_to_leader)
    try:
        def admit_unfinished(rid: str, n: int = 1) -> None:
            handle._admission.add_request(
                rid,
                "hello",
                SamplingParams(max_tokens=8, n=n, temperature=0.8 if n > 1 else 0.0),
            )

        admit_unfinished("rem-abort-1", n=1)
        assert "rem-abort-1" in collector.request_metrics
        handle.abort_request("rem-abort-1")
        assert sent_aborts == ["rem-abort-1"]
        assert "rem-abort-1" not in collector.request_metrics
        assert any(
            m.request_id == "rem-abort-1" and m.finish_reason == "abort"
            for m in collector.completed_requests
        )

        # n=2 partial abort keeps metrics open until the last sample aborts.
        admit_unfinished("rem-abort-n2", n=2)
        assert "rem-abort-n2" in collector.request_metrics
        handle.abort_request("rem-abort-n2-0")
        assert "rem-abort-n2" in collector.request_metrics
        handle.abort_request("rem-abort-n2-1")
        assert "rem-abort-n2" not in collector.request_metrics
        assert any(
            m.request_id == "rem-abort-n2" and m.finish_reason == "abort"
            for m in collector.completed_requests
        )

        # Owner death completes every unfinished in-flight request in MetricsCollector.
        admit_unfinished("rem-dead-1", n=1)
        admit_unfinished("rem-dead-2", n=1)
        assert {"rem-dead-1", "rem-dead-2"} <= set(collector.request_metrics)
        handle._on_owner_dead("owner vanished")
        assert "rem-dead-1" not in collector.request_metrics
        assert "rem-dead-2" not in collector.request_metrics
        assert any(
            m.request_id == "rem-dead-1" and m.finish_reason == "abort"
            for m in collector.completed_requests
        )
        assert any(
            m.request_id == "rem-dead-2" and m.finish_reason == "abort"
            for m in collector.completed_requests
        )
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# Audit #18: Silent-client eviction, multi-parent admit_local rollback, ZMQ cleanup
# ---------------------------------------------------------------------------


def test_leak_fix_18_silent_client_eviction_cleans_seq_and_aborts_inflight():
    import zmq

    port = _free_port()
    owner_engine = _OwnerEngine()
    local_sink = _Sink()
    spec = wire.EngineSpec(esurge_name="m", max_model_len=128, max_num_seqs=4, tokenizer_source="tok")
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=1,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint="",
        ready_timeout_s=DEADLINE_S,
        heartbeat_interval_s=0.05,
        heartbeat_timeout_s=0.25,
        engine_spec=spec,
    )
    owner = OwnerRequestPlane(
        coordinator=leader,
        scheduler_submit=owner_engine.scheduler_submit,
        abort_request=owner_engine.abort_request,
        apply_stop_strings=owner_engine.apply_stop_strings,
        submit_local_outputs=local_sink.submit,
        next_arrival_stamp=owner_engine.next_arrival_stamp,
        info=lambda *a, **k: None,
    )
    leader.start()
    owner.ensure_started()
    ctx = zmq.Context.instance()
    raw_client = ctx.socket(zmq.DEALER)
    raw_client.setsockopt(zmq.LINGER, 0)
    raw_ident = b"silent-front-1"
    raw_client.setsockopt(zmq.IDENTITY, raw_ident)
    try:
        raw_client.connect(f"tcp://127.0.0.1:{port}")
        raw_client.send_multipart(
            [
                wire.encode_message(
                    wire.Hello(
                        rank=-1,
                        world_size=0,
                        config_fingerprint="",
                        auth=AUTH,
                        role=wire.ROLE_CLIENT,
                        client_id="silent-front-1",
                    )
                )
            ]
        )
        assert raw_client.poll(3000)
        assert isinstance(wire.decode_message(raw_client.recv_multipart()[0]), wire.HelloOk)

        req = FakeEngineRequest("orphan-req-1")
        raw_client.send_multipart(
            [
                wire.encode_message(
                    wire.Admit(
                        request_id="orphan-req-1",
                        origin_rank=-1,
                        origin_counter=1,
                        content_hash="abcd1234",
                        coalescable=False,
                        coalesce_counter=0,
                        n_samples=1,
                    )
                ),
                wire.encode_payload([req]),
            ]
        )
        _wait_until(lambda: len(owner_engine.submitted) == 1)
        scheduled_id = owner_engine.submitted[0][0].request_id

        # Tee one delta so _peer_out_seq[raw_ident] is populated.
        owner.tee_outputs(
            {0: EngineCoreOutputs(outputs=[EngineCoreOutput(request_id=scheduled_id, new_token_ids=[3])], timestamp=1.0)}
        )
        _wait_until(lambda: raw_ident in owner._peer_out_seq)
        assert owner.needs_tee

        # Stop sending heartbeats; leader evicts client after heartbeat_timeout_s + sweep interval (~0.75s).
        _wait_until(lambda: raw_ident not in owner._peer_out_seq, timeout=5.0)
        _wait_until(lambda: scheduled_id in owner_engine.aborted, timeout=5.0)
        _wait_until(lambda: not owner.needs_tee, timeout=5.0)
        assert owner._owner_to_origin == {}
        assert owner._origin_to_owner == {}
        assert owner._parents == {}
        assert owner._children == {}
    finally:
        raw_client.close(linger=0)
        owner.shutdown()
        leader.shutdown("done")


def test_leak_fix_18_partial_multiparent_admit_local_rollback_and_zmq_socket_cleanup(monkeypatch):
    # 1. Partial multi-parent admit_local failure rolls back parent 0's newly created _CoalesceEntry.
    with _PlaneHarness() as plane:
        # Pre-populate slot 2 with a remote coalesced group carrying different tokens.
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1, prompt_token_ids=(1, 2, 3))])
        plane.origin.submit_remote([FakeEngineRequest("req-0000000002", prompt_token_ids=(4, 5, 6))])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 2)

        # Reset local counter to 1; owner admits [slot-1 matches remote (attach), slot-2 diverges]!
        with pytest.raises(RequestPlaneError, match="divergence"):
            plane.owner.admit_local(
                [
                    FakeEngineRequest(AUTO_1, prompt_token_ids=(1, 2, 3)),
                    FakeEngineRequest("req-0000000002", prompt_token_ids=(9, 9, 9)),
                ]
            )
        # Slot-1 entry must NOT retain a phantom _LOCAL subscriber after rollback.
        entry1 = plane.owner._entries[(1, next(iter(k[1] for k in plane.owner._entries if k[0] == 1)))]
        assert all(ident is not None for ident, _ in entry1.subscribers)
        assert None not in entry1.seen_identities

        # Now test parent 0 creates a brand-new local entry (slot 3) and parent 1 fails in compute_admission_key.
        from easydel.inference.esurge.engine import admission as admission_mod

        real_compute = admission_mod.compute_admission_key
        calls = {"n": 0}

        def fail_second(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("hash boom")
            return real_compute(*args, **kwargs)

        monkeypatch.setattr(admission_mod, "compute_admission_key", fail_second)
        plane.owner._local_coalesce_counter = itertools.count(3)
        with pytest.raises(RuntimeError, match="hash boom"):
            plane.owner.admit_local(
                [
                    FakeEngineRequest("req-0000000003", prompt_token_ids=(7, 8)),
                    FakeEngineRequest("req-0000000004", prompt_token_ids=(9, 10)),
                ]
            )
        assert 3 not in plane.owner._by_counter
        assert not any(k[0] == 3 for k in plane.owner._entries)
        assert "req-0000000003" not in plane.owner._scheduled_children

    # 2. _ClientChannel.start() closes sock and leaves self._sock=None if Thread.start() raises.
    port = _free_port()
    spec = wire.EngineSpec(esurge_name="m", max_model_len=64, max_num_seqs=2, tokenizer_source="tok")
    leader = ZmqLeaderCoordinator(
        _MockRunner(),
        world_size=1,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token=AUTH,
        config_fingerprint="",
        ready_timeout_s=DEADLINE_S,
        engine_spec=spec,
    )
    leader.start()
    try:
        closed_sockets: list[object] = []
        import zmq

        real_ctx_instance = zmq.Context.instance

        class _TrackSocket:
            def __init__(self, inner):
                self._inner = inner

            def close(self, linger=None):
                closed_sockets.append(self)
                return self._inner.close(linger=linger)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        class _TrackCtx:
            def __init__(self, inner):
                self._inner = inner

            def socket(self, kind):
                return _TrackSocket(self._inner.socket(kind))

        monkeypatch.setattr(zmq.Context, "instance", classmethod(lambda cls: _TrackCtx(real_ctx_instance())))

        class _FailThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("thread spawn failed")

        monkeypatch.setattr(remote_engine.threading, "Thread", _FailThread)
        ch = remote_engine._ClientChannel(
            leader_addr="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            client_id="fail-thread",
            connect_timeout_s=5.0,
            on_dead=lambda _r: None,
        )
        with pytest.raises(RuntimeError, match="thread spawn failed"):
            ch.start()
        assert ch._sock is None
        assert len(closed_sockets) == 1
    finally:
        leader.shutdown("done")

    # 3. ZmqWorkerCoordinator.run_worker_loop & _ClientChannel._io_loop close wake_pull if wake_push.connect fails.
    closed_in_loop: list[str] = []

    class _FakePull:
        def bind(self, _ep):
            pass

        def close(self, linger=None):
            closed_in_loop.append("pull")

    class _FakePushFail:
        def connect(self, _ep):
            raise RuntimeError("connect failed")

        def close(self, linger=None):
            closed_in_loop.append("push")

    class _FailWakeCtx:
        def socket(self, kind):
            import zmq

            return _FakePull() if kind == zmq.PULL else _FakePushFail()

    w = ZmqWorkerCoordinator(
        _MockRunner(), rank=1, world_size=2, leader_addr="127.0.0.1", control_port=port,
        auth_token=AUTH, config_fingerprint=FINGERPRINT,
    )
    w._sock = object()
    w._ctx = _FailWakeCtx()
    with pytest.raises(RuntimeError, match="connect failed"):
        w.run_worker_loop(threading.Event())
    assert "pull" in closed_in_loop and "push" in closed_in_loop

    closed_in_loop.clear()
    ch2 = remote_engine._ClientChannel(
        leader_addr="127.0.0.1", control_port=port, auth_token=AUTH, client_id="c2", on_dead=lambda _r: None
    )
    ch2._sock = object()
    ch2._ctx = _FailWakeCtx()
    with pytest.raises(RuntimeError, match="connect failed"):
        ch2._io_loop()
    assert "pull" in closed_in_loop and "push" in closed_in_loop


# ---------------------------------------------------------------------------
# Audit #19: ReplicaAttacher interruption/stop race, subprocess & socket cleanup
# ---------------------------------------------------------------------------


def test_leak_fix_19_replica_attacher_stop_race_and_interruption_cleanup(tmp_path, monkeypatch):
    # 1. Interruption mid-attach_round in _serve_front closes replica 0's already-attached handle.
    closed_handles: list[str] = []

    class _Handle:
        def __init__(self, name: str):
            self.name = name
            self.alive = True
            self.max_model_len = 128
            self.max_num_seqs = 2
            self.esurge_name = "m"
            self.tokenizer = SimpleNamespace()

        def close(self):
            closed_handles.append(self.name)

    def interrupt_on_second(addr: str):
        if addr == "r0":
            return _Handle("h0")
        raise KeyboardInterrupt("interrupted mid-round")

    attacher = app.ReplicaAttacher(["r0", "r1"], connect=interrupt_on_second, log=lambda _m: None)
    with pytest.raises(KeyboardInterrupt):
        attacher.attach_round()
    assert attacher.router is not None
    # Simulate _serve_front's finally cleanup:
    attacher.stop()
    attacher.router.close()
    assert closed_handles == ["h0"]

    # 2. stop() racing with an in-flight _connect() closes the returned handle and never attaches it.
    closed_handles.clear()
    entered_r1 = threading.Event()
    release_r1 = threading.Event()

    def slow_r1(addr: str):
        if addr == "r0":
            return _Handle("h0")
        entered_r1.set()
        release_r1.wait(timeout=5.0)
        return _Handle("h1-orphan")

    attacher2 = app.ReplicaAttacher(["r0", "r1"], connect=slow_r1, log=lambda _m: None)
    round_done = threading.Event()

    def run_round():
        try:
            attacher2.attach_round()
        finally:
            round_done.set()

    t = threading.Thread(target=run_round, daemon=True)
    t.start()
    assert entered_r1.wait(timeout=5.0)
    # Stop while r1's connect is still blocked.
    stop_thread = threading.Thread(target=attacher2.stop, daemon=True)
    stop_thread.start()
    time.sleep(0.05)
    release_r1.set()
    t.join(timeout=5.0)
    stop_thread.join(timeout=5.0)
    assert round_done.is_set()
    assert closed_handles == ["h1-orphan"]
    assert len(attacher2.router.members) == 1
    attacher2.router.close()
    assert closed_handles == ["h1-orphan", "h0"]

    # 3. host_replicas.start_leader_engine reaps spawned child on pickle failure.
    spawned: list[subprocess.Popen] = []
    real_popen = host_replicas.subprocess.Popen

    def track_popen(argv, **kwargs):
        # Launch a tiny sleeper instead of importing easydel in the child.
        proc = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=kwargs.get("pass_fds", ()))
        spawned.append(proc)
        return proc

    monkeypatch.setattr(host_replicas.subprocess, "Popen", track_popen)
    with pytest.raises((pickle.PicklingError, AttributeError)):
        host_replicas.start_leader_engine({"bad": lambda x: x}, plan=None, env={})
    assert len(spawned) == 1
    assert spawned[0].poll() is not None, "child process must be terminated and reaped on pickle failure"

    # 4. host_replicas.run_probe reaps spawned child on read interruption.
    spawned.clear()
    real_fdopen = host_replicas.os.fdopen

    class _InterruptReader:
        def __init__(self, f):
            self._f = f

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._f.close()

        def read(self):
            raise KeyboardInterrupt("abort probe")

    monkeypatch.setattr(host_replicas.os, "fdopen", lambda fd, mode: _InterruptReader(real_fdopen(fd, mode)))
    with pytest.raises(KeyboardInterrupt):
        host_replicas.run_probe({"coordinator": "127.0.0.1:1"})
    assert len(spawned) == 1
    assert spawned[0].poll() is not None, "probe child must be terminated and reaped on interrupt"

    # 5. reuse_port_socket & bind_unix_socket close sockets on setup error.
    bad_sock_closed = []
    real_socket_cls = socket.socket

    class _FailBindSocket(real_socket_cls):
        def bind(self, address):
            raise OSError("bind failed")

        def close(self):
            bad_sock_closed.append("tcp")
            super().close()

    monkeypatch.setattr(socket, "socket", _FailBindSocket)
    with pytest.raises(OSError, match="bind failed"):
        app.reuse_port_socket("127.0.0.1", 0)
    assert bad_sock_closed == ["tcp"]

    bad_sock_closed.clear()
    unix_path = str(tmp_path / "fail.sock")

    class _FailListenUnix(real_socket_cls):
        def listen(self, backlog=0):
            raise OSError("listen failed")

        def close(self):
            bad_sock_closed.append("unix")
            super().close()

    monkeypatch.setattr(socket, "socket", _FailListenUnix)
    with pytest.raises(OSError, match="listen failed"):
        dashboard.bind_unix_socket(unix_path)
    assert bad_sock_closed == ["unix"]
    assert not os.path.exists(unix_path)


# ---------------------------------------------------------------------------
# Audit #26: Dead obsolete slot helpers removed from RoutedEngine
# ---------------------------------------------------------------------------


def test_leak_fix_26_dead_routed_engine_helpers_removed():
    assert not hasattr(RoutedEngine, "_route_for_key")
    assert not hasattr(RoutedEngine, "_released_iterator")
