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

"""Leader-address resolution for the multi-host ZMQ step-coordination plane.

eSurge's leader is ``jax.process_index() == 0`` while the JAX coordinator runs on distributed process id 0. TPU
slices number processes by chip position, so the two can be different hosts (seen on a v5p-32: coordinator on
worker 0, process index 0 on worker 2). These tests build a real leader/worker pair over loopback through
``create_step_coordinator`` with a stand-in ``jax.distributed`` key-value store, and check that the worker replays
the leader's step even when the coordinator's host is unreachable.
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import jax
import pytest
from easydel.inference.esurge.distributed import wire
from easydel.inference.esurge.distributed.coordinator import create_step_coordinator
from jax._src import distributed as jax_distributed

pytestmark = pytest.mark.cpu_ok

AUTH = "test-token"
FINGERPRINT = "test-fingerprint"
DEADLINE_S = 20.0
# Nothing listens there (TEST-NET-1, RFC 5737): a worker that dials the coordinator's host instead of the leader
# never gets a HelloOk.
UNREACHABLE_COORDINATOR = "192.0.2.1:8476"


class _KeyValueStore:
    """In-memory stand-in for the ``jax.distributed`` key-value client; counts blocking lookups."""

    def __init__(self):
        self.values: dict[str, str] = {}
        self.lookups = 0
        self._cond = threading.Condition()

    def key_value_set(self, key, value, allow_overwrite=False):
        with self._cond:
            if key in self.values and not allow_overwrite:
                raise RuntimeError(f"ALREADY_EXISTS: {key}")
            self.values[key] = value
            self._cond.notify_all()

    def blocking_key_value_get(self, key, timeout_in_ms):
        deadline = time.monotonic() + timeout_in_ms / 1000
        with self._cond:
            self.lookups += 1
            while key not in self.values:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"DEADLINE_EXCEEDED: {key}")
                self._cond.wait(remaining)
            return self.values[key]


class _Runner:
    """Records the step stream a coordinator hands to its runner."""

    def __init__(self):
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def _record(self, entry: str) -> None:
        with self._lock:
            self.calls.append(entry)

    @staticmethod
    def _output(scheduler_output):
        return SimpleNamespace(
            req_ids=list(scheduler_output.num_scheduled_tokens),
            sampled_token_ids=[[7] for _ in scheduler_output.num_scheduled_tokens],
        )

    def execute_model(self, scheduler_output):
        self._record(f"sync:{scheduler_output.tag}")
        return self._output(scheduler_output)

    def execute_model_async(self, scheduler_output):
        self._record(f"dispatch:{scheduler_output.tag}")
        return SimpleNamespace(scheduler_output=scheduler_output)

    def wait_for_execution(self, handle):
        self._record(f"drain:{handle.scheduler_output.tag}")
        return self._output(handle.scheduler_output)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(port, *, bind_host="0.0.0.0", leader_addr=None, connect_timeout_s=DEADLINE_S):
    return SimpleNamespace(
        coordination="zmq",
        distributed_auth_token=AUTH,
        distributed_control_port=port,
        distributed_control_bind_host=bind_host,
        distributed_leader_addr=leader_addr,
        distributed_service_name=None,
        distributed_connect_timeout_s=connect_timeout_s,
        distributed_ready_timeout_s=DEADLINE_S,
        distributed_step_timeout_s=5.0,
        distributed_heartbeat_interval_s=0.2,
        distributed_heartbeat_timeout_s=DEADLINE_S,
        distributed_verify_digest_interval=1,
        distributed_max_inflight_steps=4,
    )


def _build_pair(monkeypatch, *, client, coordinator, leader_config, worker_config):
    """Create the rank-0 leader and the rank-1 worker the way two processes of one JAX world would."""
    rank = {"value": 0}
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: rank["value"])
    monkeypatch.setattr(jax_distributed.global_state, "client", client)
    monkeypatch.setattr(jax_distributed.global_state, "coordinator_address", coordinator)
    spec = wire.EngineSpec(
        esurge_name="mock-engine",
        max_model_len=128,
        max_num_seqs=2,
        reserve_tokens=2,
        eos_token_ids=[0],
        tokenizer_source="mock-tokenizer",
    )
    leader = create_step_coordinator(
        _Runner(), distributed_config=leader_config, config_fingerprint=FINGERPRINT, engine_spec=spec
    )
    rank["value"] = 1
    worker_runner = _Runner()
    worker = create_step_coordinator(worker_runner, distributed_config=worker_config, config_fingerprint=FINGERPRINT)
    return leader, worker, worker_runner


def _replay_one_step(leader, worker, worker_runner) -> list[str]:
    """Handshake, run one SYNC step on the leader, and return what the worker replayed."""
    stop = threading.Event()
    errors: list[BaseException] = []

    def _worker_body():
        try:
            worker.start()
            worker.run_worker_loop(stop)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_worker_body, daemon=True)
    thread.start()
    try:
        leader.start()
        leader.execute_sync(SimpleNamespace(tag="S1", total_num_scheduled_tokens=1, num_scheduled_tokens={"S1-r0": 1}))
        deadline = time.monotonic() + DEADLINE_S
        while not worker_runner.calls and not errors and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        leader.shutdown("test teardown")
        stop.set()
        thread.join(timeout=DEADLINE_S)
        worker.shutdown()
    assert not errors, errors
    return list(worker_runner.calls)


def test_worker_reaches_a_leader_that_is_not_on_the_coordinator_host(monkeypatch):
    port = _free_port()
    leader, worker, worker_runner = _build_pair(
        monkeypatch,
        client=_KeyValueStore(),
        coordinator=UNREACHABLE_COORDINATOR,
        leader_config=_config(port, bind_host="127.0.0.1"),
        worker_config=_config(port, connect_timeout_s=5.0),
    )
    assert _replay_one_step(leader, worker, worker_runner) == ["sync:S1"]


def _ipv4_source_towards(host: str) -> str | None:
    """Source address of this host's route to ``host`` (a UDP connect only selects the route)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect((host, 9))
        except OSError:
            return None
        return probe.getsockname()[0]


def test_wildcard_bind_advertises_the_interface_facing_the_coordinator(monkeypatch):
    # The coordinator sits on another loopback address than this host's route source towards it, so the address
    # the leader advertises (its interface towards the coordinator) differs from the coordinator's own host, which
    # is what workers dialled before the leader published its address.
    if _ipv4_source_towards("127.0.1.1") != "127.0.0.1":
        pytest.skip("needs a loopback route to 127.0.1.1 with source address 127.0.0.1")
    port, coordinator_port = _free_port(), _free_port()
    client = _KeyValueStore()
    leader, worker, worker_runner = _build_pair(
        monkeypatch,
        client=client,
        coordinator=f"127.0.1.1:{coordinator_port}",
        leader_config=_config(port, bind_host="0.0.0.0"),
        worker_config=_config(port),
    )
    assert list(client.values.values()) == ["127.0.0.1"]
    assert _replay_one_step(leader, worker, worker_runner) == ["sync:S1"]


def test_explicit_leader_addr_is_used_without_waiting_for_a_published_one(monkeypatch):
    port = _free_port()
    client = _KeyValueStore()
    leader, worker, worker_runner = _build_pair(
        monkeypatch,
        client=client,
        coordinator=UNREACHABLE_COORDINATOR,
        leader_config=_config(port, bind_host="127.0.0.1", leader_addr="127.0.0.1"),
        worker_config=_config(port, leader_addr="127.0.0.1", connect_timeout_s=5.0),
    )
    assert client.lookups == 0
    assert _replay_one_step(leader, worker, worker_runner) == ["sync:S1"]


def test_without_a_distributed_client_workers_use_the_coordinator_host(monkeypatch):
    port, coordinator_port = _free_port(), _free_port()
    leader, worker, worker_runner = _build_pair(
        monkeypatch,
        client=None,
        coordinator=f"127.0.0.1:{coordinator_port}",
        leader_config=_config(port),
        worker_config=_config(port),
    )
    assert _replay_one_step(leader, worker, worker_runner) == ["sync:S1"]
