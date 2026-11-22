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

"""CPU tests for sandbox hardening: grading git isolation, multi-step code judging, bridge parsing and limits."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import typing as tp

import pytest
from easydel.environments import CodeJudgeEnvironment, ContainerConfig, SWERepoEnvironment, kiri_installer
from easydel.environments._base import SandboxEditFileTool
from easydel.environments.runtime import EngineSocketClient, KiriDaemonBudget, LocalSandboxSession

pytestmark = pytest.mark.cpu_ok

_NO_JAIL = ContainerConfig(use_jail="off")
_HAS_GIT = shutil.which("git") is not None
TOKEN = "unit-test-token-0123456789"


def _bash_call(command: str) -> str:
    return f"<tool_call>{json.dumps({'name': 'bash', 'arguments': {'command': command}})}</tool_call>"


# ---------------------------------------------------------------------------------------------------------------------
# swe_repo: grading must not run agent-controlled git configuration
# ---------------------------------------------------------------------------------------------------------------------

_F2P_PATCH = (
    "diff --git a/test_f2p.py b/test_f2p.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/test_f2p.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+from calc import mul\n"
    "+assert mul(3, 4) == 12\n"
)


@pytest.mark.skipif(not _HAS_GIT, reason="git is not installed")
def test_swe_grading_ignores_agent_smudge_filter_and_fsmonitor(tmp_path):
    marker = tmp_path / "fsmonitor_ran"
    fsmonitor = tmp_path / "fsmonitor.sh"
    fsmonitor.write_text(f"#!/bin/sh\ntouch {marker}\n")
    fsmonitor.chmod(0o755)
    env = SWERepoEnvironment(
        problem_statement="Implement mul.",
        files={"calc.py": "def mul(a, b):\n    return 0\n", "pyproject.toml": "[project]\nname = 'orig'\n"},
        test_patch=_F2P_PATCH,
        fail_to_pass=[f"{sys.executable} test_f2p.py"],
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        assert env.reset().info["env_error"] is False
        # Every file git writes through the filter gets rig.py's content (an always-passing "test").
        rig = (
            "printf 'pass\\n' > rig.py; "
            "git config filter.x.smudge 'cat rig.py'; git config filter.x.clean cat; "
            "echo '* filter=x' >> .git/info/attributes; "
            f"git config core.fsmonitor {fsmonitor}; "
            "printf 'rigged\\n' > pyproject.toml"
        )
        out = env.step(_bash_call(rig))
        assert out.terminated is False
        res = env.step("done")
        assert res.info["env_error"] is False, res.info
        assert res.info["resolved"] is False, res.info
        assert res.reward == 0.0
        assert env.session.read_file("pyproject.toml") == "[project]\nname = 'orig'\n"
        assert "assert mul(3, 4) == 12" in env.session.read_file("test_f2p.py")
        assert not marker.exists(), "the agent's fsmonitor command ran during grading"
    finally:
        env.close()


@pytest.mark.skipif(not _HAS_GIT, reason="git is not installed")
def test_swe_grading_still_resolves_an_honest_fix_and_restores_symlinked_infra():
    env = SWERepoEnvironment(
        problem_statement="Implement mul.",
        files={"calc.py": "def mul(a, b):\n    return 0\n", "sub/setup.cfg": "[x]\nkeep = 1\n"},
        test_patch=_F2P_PATCH,
        fail_to_pass=[f"{sys.executable} test_f2p.py"],
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        assert env.reset().info["env_error"] is False
        env.step(_bash_call("printf 'def mul(a, b):\\n    return a * b\\n' > calc.py; rm -rf sub; ln -s /tmp sub"))
        res = env.step("done")
        assert res.info["resolved"] is True, res.info
        assert res.reward == 1.0
        assert "return a * b" in res.info["patch"]
        assert not os.path.islink(os.path.join(env.session.workdir, "sub"))
        assert env.session.read_file("sub/setup.cfg") == "[x]\nkeep = 1\n"
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------------------------
# code_judge: later steps are graded in a fresh session
# ---------------------------------------------------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def test_code_judge_multistep_test_code_does_not_inherit_earlier_steps():
    env = CodeJudgeEnvironment(
        prompt="Implement add(a, b).",
        test_code="assert add(1, 1) == 2",
        max_steps=3,
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        env.reset()
        step1 = env.step(
            "```python\nimport subprocess\n"
            "open('planted.txt', 'w').write('x')\n"
            "_p = subprocess.Popen(['sleep', '60'])\n"
            "open('pid.txt', 'w').write(str(_p.pid))\n"
            "def add(a, b):\n    return 0\n```"
        )
        assert step1.reward == 0.0 and step1.terminated is False
        pid = int(env.session.read_file("pid.txt"))
        assert _pid_alive(pid)
        step2 = env.step(
            "```python\nimport os\ndef add(a, b):\n    return 0 if os.path.exists('planted.txt') else a + b\n```"
        )
        assert step2.reward == 1.0, step2.observation
        assert step2.info["all_passed"] is True
        deadline = time.monotonic() + 5.0
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(pid), "a process left by step 1 survived into step 2's evaluation"
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------------------------
# Bridge: head parsing, legacy root HostConfig keys, budget clamp
# ---------------------------------------------------------------------------------------------------------------------


class _RecordingEngine:
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="edhard-")  # system temp dir keeps the AF_UNIX path short
        self.sock_path = os.path.join(self.dir, "e.sock")
        self.requests: list[tuple[str, bytes]] = []
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        self._srv.listen(16)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            reader = conn.makefile("rb")
            request_line = reader.readline().decode("latin-1").strip()
            length = 0
            while True:
                line = reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                k, v = line.decode("latin-1").split(":", 1)
                if k.strip().lower() == "content-length":
                    length = int(v.strip())
            body = reader.read(length)
            self.requests.append((request_line, body))
            payload = b'{"Id": "cid-1"}'
            conn.sendall(f"HTTP/1.1 201 X\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload)
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self._srv.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _raw(port: int, head: bytes, body: bytes = b"") -> int:
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as s:
        s.sendall(head + body)
        data = b""
        while b"\r\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    return int(data.split(b" ", 2)[1])


@pytest.fixture
def bridge() -> tp.Iterator[tuple[_RecordingEngine, int]]:
    engine = _RecordingEngine()
    port = _free_port()
    kiri_installer._start_unix_tcp_bridge(engine.sock_path, "127.0.0.1", port, token=TOKEN)
    try:
        yield engine, port
    finally:
        kiri_installer.shutdown_kiri_daemons()
        kiri_installer._ENDPOINT_BUDGETS.pop(f"unix://{engine.sock_path}", None)
        engine.close()


def test_bridge_refuses_malformed_heads(bridge):
    engine, port = bridge
    auth = f"X-EasyDeL-Sandbox-Token: {TOKEN}\r\n".encode()
    bad_heads = [
        b"POST /v1.41/containers/abc/start HTTP/1.1\r\n" + auth + b"X-A: v\nTransfer-Encoding: chunked\r\n"
        b"Content-Length: 0\r\n\r\n",
        b"GET /_ping HTTP/1.1\r\n" + auth + b"X-A: v\rX-B: w\r\n\r\n",
        b"GET /_ping HTTP/1.1\r\n" + auth + b"Bad Name: v\r\n\r\n",
        b"GET /_ping HTTP/1.1\r\n" + auth + b"X(A): v\r\n\r\n",
        b"GET /_ping HTTP/1.1\r\n" + auth + b"X-A: v\x01w\r\n\r\n",
        b"GET /_ping HTTP/1.1\r\n" + auth + b"X-A: v\x7fw\r\n\r\n",
        b"GET /_ping?a=b\x01c HTTP/1.1\r\n" + auth + b"\r\n",
        b"GET /_ping?a=b\nX-Y:z HTTP/1.1\r\n" + auth + b"\r\n",
    ]
    for head in bad_heads:
        assert _raw(port, head) == 400, head
    assert engine.requests == []
    assert _raw(port, b"GET /_ping?a=b HTTP/1.1\r\n" + auth + b"X-A: v\tw\r\n\r\n") == 201
    assert len(engine.requests) == 1


def test_bridge_refuses_hostconfig_fields_at_the_body_root(bridge):
    engine, port = bridge
    client = EngineSocketClient(f"http://127.0.0.1:{port}", token=TOKEN)
    for spec in (
        {"Image": "x", "Privileged": True},
        {"Image": "x", "Binds": ["/:/host"]},
        {"Image": "x", "privileged": True, "HostConfig": {}},
        {"Image": "x", "CapAdd": ["SYS_ADMIN"]},
        {"Image": "x", "PidMode": "host"},
        {"Image": "x", "Devices": [{"PathOnHost": "/dev/sda"}]},
    ):
        assert client._request("POST", "/v1.41/containers/create", body=spec, timeout=5.0).status == 403, spec
    assert engine.requests == []


def test_bridge_clamps_forwarded_container_creates_to_the_daemon_budget(bridge):
    engine, port = bridge
    kiri_installer._ENDPOINT_BUDGETS[f"unix://{engine.sock_path}"] = KiriDaemonBudget(
        cpus=2.0, memory_mb=1024, pids_limit=64, tmpfs_mb=128
    )
    # A remote client has no budget of its own: it asks for unlimited resources.
    client = EngineSocketClient(f"http://127.0.0.1:{port}", token=TOKEN)
    assert client.budget is None
    unlimited = ContainerConfig(image="i", pull_if_missing=False, cpus=0, memory_mb=0, pids_limit=0, tmpfs_mb=4096)
    assert client.create_container(unlimited) == "cid-1"
    raw = {"Image": "i", "hostconfig": {"memory": -1, "PIDSLIMIT": 100000}, "NanoCpus": 99_000_000_000}
    assert client._request("POST", "/v1.41/containers/create", body=raw, timeout=5.0).status == 201
    assert len(engine.requests) == 2
    first = json.loads(engine.requests[0][1])["HostConfig"]
    assert first["Memory"] == 1024 * 1024 * 1024
    assert first["NanoCpus"] == 2_000_000_000
    assert first["PidsLimit"] == 64
    assert first["Tmpfs"]["/tmp"].endswith("size=128m")
    second = json.loads(engine.requests[1][1])
    assert "NanoCpus" not in second
    hc = second["hostconfig"]
    assert hc["Memory"] == 1024 * 1024 * 1024 and hc["PidsLimit"] == 64 and hc["NanoCpus"] == 2_000_000_000
    assert {k.casefold() for k in hc} == {"memory", "pidslimit", "nanocpus"}


# ---------------------------------------------------------------------------------------------------------------------
# Local reads: FIFOs and huge files
# ---------------------------------------------------------------------------------------------------------------------


def test_local_read_file_refuses_fifo_without_blocking():
    with LocalSandboxSession(_NO_JAIL) as sess:
        os.mkfifo(os.path.join(sess.workdir, "f"))
        outcome: list[BaseException | str] = []

        def _read() -> None:
            try:
                outcome.append(sess.read_file("f", max_chars=10))
            except BaseException as exc:
                outcome.append(exc)

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(3.0)
        assert not t.is_alive(), "read_file blocked on a FIFO"
        assert isinstance(outcome[0], OSError)
        tool = SandboxEditFileTool(lambda: sess)
        assert tool.call("f", "a", "b").startswith("Error reading 'f'")
        sess.write_files({"ok.txt": "hello"})
        assert sess.read_file("ok.txt") == "hello"


def test_edit_file_refuses_files_over_the_size_cap():
    with LocalSandboxSession(_NO_JAIL) as sess:
        tool = SandboxEditFileTool(lambda: sess)
        tool.max_file_chars = 1000
        sess.write_files({"big.txt": "x" * 1001 + "needle", "small.txt": "a needle b"})
        assert "larger than 1000" in tool.call("big.txt", "needle", "pin")
        assert sess.read_file("big.txt").endswith("needle")
        assert tool.call("small.txt", "needle", "pin") == "Updated small.txt."
        assert sess.read_file("small.txt") == "a pin b"


# ---------------------------------------------------------------------------------------------------------------------
# Kiri install: only pinned tags count as pinned
# ---------------------------------------------------------------------------------------------------------------------


class _Resp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _release(tag: str, version: str, digest: str):
    meta = {
        "tag_name": tag,
        "assets": [
            {
                "name": f"Kiri-{version}-x86_64.tar.gz",
                "browser_download_url": "https://example.invalid/k.tar.gz",
                "digest": f"sha256:{digest}",
            }
        ],
    }
    return lambda req, timeout=20.0: _Resp(json.dumps(meta).encode())


def test_unpinned_kiri_tag_requires_an_explicit_sha256(monkeypatch):
    api_digest = "a" * 64
    monkeypatch.setattr(kiri_installer.urllib.request, "urlopen", _release("v9.9.9", "9.9.9", api_digest))
    kwargs = dict(version="v9.9.9", machine="x86_64", system="darwin")
    with pytest.raises(RuntimeError, match="no pinned SHA256"):
        kiri_installer.resolve_kiri_asset_url(**kwargs)
    mine = "b" * 64
    assert kiri_installer.resolve_kiri_asset_url(**kwargs, expected_sha256=mine.upper())[3] == mine
    with pytest.raises(RuntimeError, match="64 hexadecimal"):
        kiri_installer.resolve_kiri_asset_url(**kwargs, expected_sha256="abc")


def test_pinned_kiri_tag_uses_the_pinned_digest_not_the_api_digest(monkeypatch):
    pinned = kiri_installer.PINNED_KIRI_SHA256["v0.2.1"]["Kiri-0.2.1-x86_64.tar.gz"]
    monkeypatch.setattr(kiri_installer.urllib.request, "urlopen", _release("v0.2.1", "0.2.1", "c" * 64))
    kwargs = dict(version="v0.2.1", machine="x86_64", system="darwin")
    assert kiri_installer.resolve_kiri_asset_url(**kwargs)[3] == pinned
    assert kiri_installer.resolve_kiri_asset_url(**kwargs, expected_sha256=pinned)[3] == pinned
    with pytest.raises(RuntimeError, match="contradicts"):
        kiri_installer.resolve_kiri_asset_url(**kwargs, expected_sha256="d" * 64)
