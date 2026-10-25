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

"""Subprocess / lifecycle regression tests for the worker managers and clients (no array computation)."""

import os
import pathlib
import shutil
import subprocess
import tempfile
import threading

import pytest
import zmq
from easydel.workers.esurge.auth import AuthWorkerManager
from easydel.workers.esurge.pipeline.worker_manager import WorkerManager
from easydel.workers.esurge.pipeline.zmq_workers import DetokenizerWorkerClient
from easydel.workers.response_store.worker_manager import ResponseStoreWorkerManager

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def short_dir():
    """A short temp dir: ZMQ ipc socket paths are limited to ~107 characters."""
    path = tempfile.mkdtemp(prefix="edw_", dir="/tmp")
    yield pathlib.Path(path)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(params=[("cpu", "cpu"), ("cuda", "gpu")], ids=["distributed", "accelerator"])
def inherited_model_environment(monkeypatch, request):
    """Guard real launches before imports can initialize distributed JAX or accelerators."""
    platforms, platform_name = request.param
    inherited = {
        "JAX_PLATFORMS": platforms,
        "JAX_PLATFORM_NAME": platform_name,
        "ENABLE_DISTRIBUTED_INIT": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)

    real_popen = subprocess.Popen
    launched = []

    def guarded_popen(command, *, env, **kwargs):
        # Fail without creating a child if isolation regresses, even for the GPU case.
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_PLATFORM_NAME"] == "cpu"
        assert env["ENABLE_DISTRIBUTED_INIT"] == "0"
        assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        process = real_popen(command, env=env, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    yield
    assert all(process.poll() is not None for process in launched)
    assert {name: os.environ[name] for name in inherited} == inherited


def test_auth_worker_subprocess_starts_and_serves(short_dir, inherited_model_environment):
    tmp_path = short_dir
    manager = AuthWorkerManager(
        enable_persistence=False,
        admin_key="sk-bootstrap-admin-secret-from-stdin",
        startup_timeout=180.0,
        ipc_dir=str(tmp_path),
    )
    client = manager.start()
    try:
        assert client.validate_key("sk-bootstrap-admin-secret-from-stdin") is not None
        raw, meta = client.generate_api_key(name="subprocess-key")
        assert client.validate_key(raw).key_id == meta.key_id
    finally:
        manager.shutdown()
    assert manager.auth_client is None
    assert manager.auth_endpoint is None


def test_response_store_worker_subprocess_starts_and_serves(short_dir, inherited_model_environment):
    tmp_path = short_dir
    manager = ResponseStoreWorkerManager(
        storage_dir=str(tmp_path / "store"),
        startup_timeout=180.0,
        ipc_dir=str(tmp_path),
    )
    client = manager.start()
    try:
        client.put_response("resp_1", {"id": "resp_1"})
        assert client.get_response("resp_1") == {"id": "resp_1"}
    finally:
        manager.shutdown()
    assert manager.client is None


def test_auth_manager_does_not_keep_ownership_after_shutdown(short_dir):
    tmp_path = short_dir
    manager = AuthWorkerManager(enable_persistence=False, startup_timeout=180.0, ipc_dir=str(tmp_path))
    manager.start()
    manager.shutdown()
    # Attach to an "external" endpoint: the manager must treat it as foreign and never send ``shutdown``.
    manager.start(auth_endpoint=f"ipc://{tmp_path / 'foreign.sock'}")
    assert manager._auth_owned is False
    manager.shutdown()  # would block forever if it tried to message the (nonexistent) worker


class _FakeProcess:
    def __init__(self):
        self.terminated = False
        self.pid = 1234
        self.returncode = None

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


@pytest.mark.parametrize("failure_stage", ["spawn", "bind"])
def test_failed_detokenizer_startup_terminates_tokenizer_worker(monkeypatch, tmp_path, failure_stage):
    manager = WorkerManager("dummy-tokenizer", startup_timeout=1.0, ipc_dir=str(tmp_path))
    spawned: dict[str, _FakeProcess] = {}

    def fake_spawn(mode, endpoint, *, detokenizer_max_states):
        if mode == "detokenizer" and failure_stage == "spawn":
            raise OSError("detokenizer could not launch")
        spawned[mode] = _FakeProcess()
        return spawned[mode]

    def fake_wait(worker_name, endpoint, process):
        if worker_name == "detokenizer":
            raise TimeoutError("detokenizer never bound")

    monkeypatch.setattr(manager, "_spawn_worker", fake_spawn)
    monkeypatch.setattr(manager, "_wait_for_endpoint", fake_wait)

    with pytest.raises(OSError):
        manager.start(detokenizer_max_states=8, tokenizer_endpoint=None, detokenizer_endpoint=None)

    assert spawned["tokenizer"].terminated
    if "detokenizer" in spawned:
        assert spawned["detokenizer"].terminated
    assert manager.tokenizer_endpoint is None
    manager.shutdown()


@pytest.mark.parametrize(("setting", "expected_deadline"), [("-5", 120.0), ("42", 42.0)])
def test_startup_timeout_flag_rejects_non_positive_values(monkeypatch, tmp_path, setting, expected_deadline):
    from easydel.workers.esurge.pipeline import worker_manager

    clock = [0.0]
    process = _FakeProcess()
    monkeypatch.setenv("EASYDEL_WORKER_STARTUP_TIMEOUT", setting)
    monkeypatch.setattr(worker_manager.time, "time", lambda: clock[0])
    monkeypatch.setattr(worker_manager.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    manager = WorkerManager("dummy-tokenizer", ipc_dir=str(tmp_path))
    monkeypatch.setattr(manager, "_spawn_worker", lambda *args, **kwargs: process)
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        manager.start(detokenizer_max_states=8, tokenizer_endpoint=None, detokenizer_endpoint=None)
    assert expected_deadline <= clock[0] < expected_deadline + 0.1
    assert process.terminated
    assert manager.tokenizer_endpoint is None


def test_detokenizer_drain_forces_full_token_resync(tmp_path):
    endpoint = f"ipc://{tmp_path / 'detok.sock'}"
    ctx = zmq.Context()
    server = ctx.socket(zmq.REP)
    server.bind(endpoint)
    received: list[dict] = []

    def serve():
        while True:
            msg = server.recv_pyobj()
            received.append(msg)
            if msg["cmd"] == "decode":
                server.send_pyobj(
                    {
                        "status": "ok",
                        "result": {
                            "accumulated_text": "",
                            "delta_text": "",
                            "last_decoded_index": len(msg["tokens_delta"]),
                            "finished": msg["finished"],
                            "detoktook": 0.0,
                        },
                    }
                )
            else:
                server.send_pyobj({"status": "ok"})
                if msg["cmd"] == "shutdown":
                    return

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    client = DetokenizerWorkerClient(endpoint)
    try:
        kwargs = {"finished": False, "skip_special_tokens": True}
        client.decode("r1", [1, 2, 3], **kwargs)
        client.decode("r1", [1, 2, 3, 4], **kwargs)
        assert "tokens" not in received[-1]  # steady state: delta only
        client.drain()
        client.decode("r1", [1, 2, 3, 4, 5], **kwargs)
        assert received[-1]["tokens"] == [1, 2, 3, 4, 5]
        assert received[-1]["token_offset"] == 0
    finally:
        client.shutdown()
        thread.join(timeout=10)
        server.close(0)
        ctx.term()
