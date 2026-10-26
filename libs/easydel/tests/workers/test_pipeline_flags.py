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

"""Check registered settings through the actual CPU-only pipeline worker readers."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import zmq
from easydel.utils import flags
from easydel.workers.esurge.pipeline import worker_main
from easydel.workers.esurge.pipeline.worker_manager import WorkerManager

pytestmark = pytest.mark.cpu_ok


class _CharacterTokenizer:
    def __init__(self):
        self.decoded_tokens = []

    def __call__(self, prompt, **kwargs):
        return {"input_ids": [ord(char) for char in prompt]}

    def decode(self, tokens, **kwargs):
        self.decoded_tokens.append(list(tokens))
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def run_worker(monkeypatch):
    """Replace only tokenizer loading and transport; execute real dispatch/decoding."""

    def run(mode, tokenizer_kwargs=None):
        tokenizer = _CharacterTokenizer()
        loaded_kwargs = {}
        replies = []
        messages = iter(
            [
                {"cmd": "tokenize", "prompt": "e"}
                if mode == "tokenizer"
                else {
                    "cmd": "decode",
                    "request_id": "r",
                    "tokens": [101],
                    "prompt_context": [97, 98, 99, 100],
                    "finished": False,
                    "skip_special_tokens": True,
                },
                {"cmd": "shutdown"},
            ]
        )

        class Socket:
            def bind(self, endpoint):
                pass

            def recv_pyobj(self):
                return next(messages)

            def send_pyobj(self, reply):
                replies.append(reply)

            def close(self, linger):
                pass

        class Context:
            def socket(self, kind):
                return Socket()

            def term(self):
                pass

        def load(path, **kwargs):
            loaded_kwargs.update(kwargs)
            return tokenizer

        monkeypatch.setattr(worker_main.AutoTokenizer, "from_pretrained", load)
        monkeypatch.setattr(worker_main.zmq, "Context", Context)
        kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs.copy()
        if mode == "tokenizer":
            worker_main._tokenizer_worker("unused", "unused", kwargs)
            assert replies[0] == {"status": "ok", "tokens": [101]}
        else:
            worker_main._detokenizer_worker("unused", "unused", kwargs, 8)
            assert replies[0]["status"] == "ok"
            assert replies[0]["result"]["delta_text"] == "e"
        assert replies[-1] == {"status": "ok"}
        return loaded_kwargs, tokenizer.decoded_tokens

    return run


@pytest.mark.parametrize("mode", ["tokenizer", "detokenizer"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, True),
        ("1", True),
        ("on", True),
        ("yes", True),
        ("0", False),
        ("", False),
        ("true", False),
        ("ON", False),
        ("YES", False),
        (" on ", False),
    ],
)
def test_worker_trust_flag_preserves_legacy_values(monkeypatch, run_worker, mode, raw, expected):
    if raw is None:
        monkeypatch.delenv(flags.ESURGE_WORKER_TRUST_REMOTE_CODE, raising=False)
    else:
        monkeypatch.setenv(flags.ESURGE_WORKER_TRUST_REMOTE_CODE, raw)
    loaded, _ = run_worker(mode)
    assert loaded["trust_remote_code"] is expected


@pytest.mark.parametrize("mode", ["tokenizer", "detokenizer"])
@pytest.mark.parametrize(("raw", "override"), [("1", False), ("0", True), ("1", None)])
def test_worker_explicit_trust_override_wins(monkeypatch, run_worker, mode, raw, override):
    monkeypatch.setenv(flags.ESURGE_WORKER_TRUST_REMOTE_CODE, raw)
    loaded, _ = run_worker(mode, {"trust_remote_code": override})
    assert loaded["trust_remote_code"] is override


@pytest.mark.parametrize("mode", ["tokenizer", "detokenizer"])
def test_worker_reads_trust_from_registered_getter(monkeypatch, run_worker, mode):
    monkeypatch.setenv(flags.ESURGE_WORKER_TRUST_REMOTE_CODE, "1")
    monkeypatch.setattr(flags, "get_bool", lambda name: False)
    loaded, _ = run_worker(mode)
    assert loaded["trust_remote_code"] is False


@pytest.mark.parametrize(
    ("raw", "expected_tokens"),
    [
        (None, [97, 98, 99, 100, 101]),
        ("invalid", [97, 98, 99, 100, 101]),
        ("", [97, 98, 99, 100, 101]),
        ("2", [99, 100, 101]),
        ("0", [101]),
        ("-1", [101]),
    ],
)
def test_detokenizer_context_flag_controls_decoded_context(monkeypatch, run_worker, raw, expected_tokens):
    if raw is None:
        monkeypatch.delenv(flags.ESURGE_DETOKENIZER_CONTEXT_WINDOW, raising=False)
    else:
        monkeypatch.setenv(flags.ESURGE_DETOKENIZER_CONTEXT_WINDOW, raw)
    _, decoded_tokens = run_worker("detokenizer")
    assert decoded_tokens[0] == expected_tokens


def test_detokenizer_reads_context_from_registered_getter(monkeypatch, run_worker):
    monkeypatch.setenv(flags.ESURGE_DETOKENIZER_CONTEXT_WINDOW, "4")
    monkeypatch.setattr(flags, "get_int", lambda name: 2)
    _, decoded_tokens = run_worker("detokenizer")
    assert decoded_tokens[0] == [99, 100, 101]


@pytest.mark.parametrize("mode", ["tokenizer", "detokenizer"])
def test_spawn_overrides_model_environment_without_mutating_parent(monkeypatch, mode):
    """Intercept Popen: inherited accelerator settings must never reach a child."""
    inherited = {
        "JAX_PLATFORMS": "cuda",
        "JAX_PLATFORM_NAME": "gpu",
        "ENABLE_DISTRIBUTED_INIT": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)

    def popen(command, *, env):
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_PLATFORM_NAME"] == "cpu"
        assert env["ENABLE_DISTRIBUTED_INIT"] == "0"
        assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert command[2] == mode
        return object()

    monkeypatch.setattr(subprocess, "Popen", popen)
    manager = WorkerManager("unused")
    manager._spawn_worker(mode, "unused", detokenizer_max_states=8)
    assert {name: os.environ[name] for name in inherited} == inherited


@pytest.mark.parametrize("mode", ["tokenizer", "detokenizer"])
def test_script_worker_serves_with_inherited_distributed_initialization(tmp_path, mode):
    """Run the real CLI/registry/ZMQ loop, guarding JAX imports before dispatch."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    # A direct script launch has no parent manager to sanitize its environment.
    # Guard every JAX import *before* allowing it, so even a failing regression
    # cannot initialize a distributed runtime or an accelerator.
    bootstrap = """
import importlib.abc
import os
import runpy
import sys

class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            assert os.environ['JAX_PLATFORMS'] == 'cpu'
            assert os.environ['JAX_PLATFORM_NAME'] == 'cpu'
            assert os.environ['ENABLE_DISTRIBUTED_INIT'] == '0'
            assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false'
        return None

sys.meta_path.insert(0, Guard())
worker_path = sys.argv.pop(1)
runpy.run_path(worker_path, run_name='__main__')
"""
    env = os.environ.copy()
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        ENABLE_DISTRIBUTED_INIT="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="true",
    )
    with tempfile.TemporaryDirectory(prefix="ed_flags_", dir="/tmp") as directory:
        endpoint = f"ipc://{directory}/worker.sock"
        command = [
            sys.executable,
            "-c",
            bootstrap,
            str(Path(worker_main.__file__)),
            mode,
            "--endpoint",
            endpoint,
            "--tokenizer-path",
            str(tmp_path),
            "--tokenizer-kwargs",
            '{"local_files_only": true}',
        ]
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 45000)
        socket.connect(endpoint)
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if mode == "tokenizer":
                socket.send_pyobj({"cmd": "tokenize", "prompt": "hello"})
                assert socket.recv_pyobj() == {"status": "ok", "tokens": [1]}
            else:
                socket.send_pyobj(
                    {
                        "cmd": "decode",
                        "request_id": "r",
                        "tokens": [1],
                        "finished": True,
                        "skip_special_tokens": True,
                    }
                )
                reply = socket.recv_pyobj()
                assert reply["status"] == "ok"
                assert reply["result"]["accumulated_text"] == "hello"
            socket.send_pyobj({"cmd": "shutdown"})
            assert socket.recv_pyobj() == {"status": "ok"}
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stdout + stderr
        finally:
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=15)
            socket.close(0)
            context.term()
        assert process.returncode == 0, stdout + stderr
