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

import gc
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref

import pytest
from easydel.workers.esurge.auth import AuthWorkerManager
from easydel.workers.esurge.pipeline.worker_manager import WorkerManager
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


def test_failed_native_detokenizer_startup_terminates_tokenizer_worker(monkeypatch, short_dir):
    from easydel.workers.esurge.pipeline import worker_manager
    from transformers import AutoTokenizer

    manager = WorkerManager("dummy-tokenizer", startup_timeout=1.0, ipc_dir=str(short_dir))
    process = _FakeProcess()
    monkeypatch.setattr(manager, "_spawn_worker", lambda endpoint: process)
    monkeypatch.setattr(manager, "_wait_for_endpoint", lambda *args: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def fail_detokenizer(*args, **kwargs):
        raise OSError("native detokenizer could not start")

    monkeypatch.setattr(worker_manager, "Detokenizer", fail_detokenizer)
    with pytest.raises(OSError, match="native detokenizer"):
        manager.start(detokenizer_max_states=8, tokenizer_endpoint=None)

    assert process.terminated
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
    monkeypatch.setattr(manager, "_spawn_worker", lambda endpoint: process)
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        manager.start(detokenizer_max_states=8, tokenizer_endpoint=None)
    assert expected_deadline <= clock[0] < expected_deadline + 0.1
    assert process.terminated
    assert manager.tokenizer_endpoint is None


def _save_tiny_fast_tokenizer(directory: pathlib.Path) -> str:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tok_obj = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok_obj, unk_token="[UNK]")
    tokenizer.save_pretrained(directory)
    return str(directory)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until(predicate, *, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def test_pipeline_worker_manager_real_shutdown_reaps_and_restarts(short_dir, tmp_path):
    tok_dir = _save_tiny_fast_tokenizer(tmp_path / "tok")
    manager = WorkerManager(
        tok_dir,
        tokenizer_kwargs={"local_files_only": True},
        startup_timeout=60.0,
        ipc_dir=str(short_dir),
    )
    try:
        tok_client, detok_client = manager.start(detokenizer_max_states=8, tokenizer_endpoint=None)
        proc = manager._tokenizer_process
        endpoint = manager.tokenizer_endpoint
        assert proc is not None and proc.poll() is None
        assert endpoint is not None and endpoint.startswith("ipc://")
        sock_path = endpoint[len("ipc://") :]
        assert os.path.exists(sock_path)
        assert manager.is_running is True
        assert tok_client.tokenize("req-1", "hello world") == [1, 2]
        assert detok_client is not None

        pid = proc.pid
        manager.shutdown()
        assert manager.is_running is False
        assert manager.tokenizer_endpoint is None
        assert manager._tokenizer_process is None
        assert proc.poll() is not None
        assert not _pid_exists(pid)
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        assert not os.path.exists(sock_path)
        # Idempotent second shutdown.
        manager.shutdown()

        # Restart after shutdown spawns a fresh worker and serves normally.
        tok_client2, _ = manager.start(detokenizer_max_states=8, tokenizer_endpoint=None)
        proc2 = manager._tokenizer_process
        endpoint2 = manager.tokenizer_endpoint
        assert proc2 is not None and proc2.poll() is None
        assert proc2.pid != pid
        assert tok_client2.tokenize("req-2", "world") == [2]
        manager.shutdown()
        assert not _pid_exists(proc2.pid)
        assert endpoint2 is not None and not os.path.exists(endpoint2[len("ipc://") :])

        # Attach mode after an owned run must NOT retain ownership or unlink the foreign socket.
        foreign_sock = short_dir / "foreign_tokenizer.sock"
        foreign_sock.write_text("")
        manager.start(detokenizer_max_states=8, tokenizer_endpoint=f"ipc://{foreign_sock}")
        assert manager._tokenizer_owned is False
        manager.shutdown()
        assert foreign_sock.exists()
    finally:
        manager.shutdown()


def test_pipeline_worker_manager_sigkill_reap_and_detokenizer_failure_isolation(short_dir, monkeypatch):
    # Stubborn child ignores SIGTERM; shutdown() must escalate to SIGKILL and reap it even if detokenizer fails.
    stubborn = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert stubborn.stdout is not None
    assert stubborn.stdout.readline().strip() == "ready"
    stubborn.stdout.close()

    sock_file = short_dir / "stubborn_tokenizer.sock"
    sock_file.write_text("")
    endpoint = f"ipc://{sock_file}"

    real_wait = stubborn.wait
    calls = {"n": 0}

    def fast_timeout_wait(timeout=None):
        calls["n"] += 1
        if calls["n"] == 1 and timeout is not None:
            raise subprocess.TimeoutExpired(cmd="stubborn", timeout=timeout)
        return real_wait(timeout=timeout)

    monkeypatch.setattr(stubborn, "wait", fast_timeout_wait)

    class _FailingDetokenizer:
        def shutdown(self):
            raise RuntimeError("detokenizer boom")

    class _ClosingClient:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            raise TimeoutError("worker wedged")

        def close(self):
            pass

    manager = WorkerManager("dummy-tokenizer", ipc_dir=str(short_dir))
    manager._tokenizer_process = stubborn
    manager._tokenizer_owned = True
    manager._tokenizer_endpoint = endpoint
    manager._detokenizer_client = _FailingDetokenizer()
    client = _ClosingClient()
    manager._tokenizer_client = client
    pid = stubborn.pid

    manager.shutdown()

    assert client.shutdown_calls == 1
    assert manager._tokenizer_process is None
    assert manager.tokenizer_endpoint is None
    assert manager.is_running is False
    assert stubborn.returncode is not None
    assert not _pid_exists(pid)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    assert not sock_file.exists()


def test_pipeline_worker_manager_finalizer_reaps_abandoned_subprocess(short_dir):
    sock_file = short_dir / "abandoned.sock"
    sock_file.write_text("")
    endpoint = f"ipc://{sock_file}"

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    manager = WorkerManager("dummy-tokenizer", ipc_dir=str(short_dir))
    manager._tokenizer_process = proc
    manager._tokenizer_owned = True
    manager._tokenizer_endpoint = endpoint
    from easydel.workers.esurge.pipeline.worker_manager import _cleanup_owned_worker

    manager._process_finalizer = weakref.finalize(manager, _cleanup_owned_worker, proc, endpoint)
    ref = weakref.ref(manager)

    del manager
    gc.collect()

    assert ref() is None
    assert proc.poll() is not None
    assert not _pid_exists(pid)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    assert not sock_file.exists()


def test_tokenizer_worker_exits_when_parent_is_sigkilled(short_dir, tmp_path):
    """Owned tokenizer subprocess must self-exit and unlink its IPC socket if its parent dies via SIGKILL."""
    from easydel.workers.esurge.pipeline import worker_main

    tok_dir = _save_tiny_fast_tokenizer(tmp_path / "tok_orphan")
    sock_path = short_dir / "orphan_parent.sock"
    endpoint = f"ipc://{sock_path}"
    launcher = (
        "import os, subprocess, sys, time\n"
        f"cmd = [sys.executable, {str(pathlib.Path(worker_main.__file__))!r}, 'tokenizer', "
        f"'--endpoint', {endpoint!r}, '--tokenizer-path', {tok_dir!r}, "
        "'--tokenizer-kwargs', '{\"local_files_only\": true}', '--parent-pid', str(os.getpid())]\n"
        "child = subprocess.Popen(cmd)\n"
        "deadline = time.monotonic() + 45.0\n"
        f"while time.monotonic() < deadline and not os.path.exists({str(sock_path)!r}):\n"
        "    assert child.poll() is None, f'worker exited early: {child.returncode}'\n"
        "    time.sleep(0.05)\n"
        f"assert os.path.exists({str(sock_path)!r}), 'worker did not bind in time'\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", launcher],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_pid: int | None = None
    try:
        assert parent.stdout is not None
        line = parent.stdout.readline().strip()
        assert line.isdigit(), (
            f"expected child pid, got {line!r}, stderr={parent.stderr.read() if parent.stderr else ''}"
        )
        worker_pid = int(line)
        assert _pid_exists(worker_pid)
        assert sock_path.exists()

        # Hard-kill the intermediate parent so neither terminate() nor atexit runs in it.
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5.0)

        assert _wait_until(lambda: not _pid_exists(worker_pid), timeout=10.0), (
            f"tokenizer worker pid={worker_pid} remained orphaned after parent SIGKILL"
        )
        assert _wait_until(lambda: not sock_path.exists(), timeout=5.0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if worker_pid is not None and _pid_exists(worker_pid):
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
