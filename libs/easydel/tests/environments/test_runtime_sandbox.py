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

"""CPU tests for sandbox runtime safety: bridge auth/policy, daemon ownership, deadlines and bounded reads."""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import typing as tp

import pytest
from easydel.environments import kiri_installer, runtime
from easydel.environments.runtime import (
    ContainerConfig,
    EngineSocketClient,
    KiriDaemonBudget,
    KiriRuntime,
    LocalSandboxSession,
    SandboxInfraError,
)

pytestmark = pytest.mark.cpu_ok

TOKEN = "unit-test-token-0123456789"


class _FakeEngine:
    """Unix-socket HTTP server; ``handler(method, path, body, conn)`` returns ``(status, body)`` or ``None``."""

    def __init__(self, handler: tp.Callable[[str, str, bytes, socket.socket], tuple[int, bytes] | None]):
        self.dir = tempfile.mkdtemp(prefix="edfake-", dir="/dev/shm")
        self.sock_path = os.path.join(self.dir, "e.sock")
        self.handler = handler
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        self._srv.listen(16)
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def endpoint(self) -> str:
        return f"unix://{self.sock_path}"

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            reader = conn.makefile("rb")
            method, target, _ = reader.readline().decode("latin-1").strip().split(" ", 2)
            headers: dict[str, str] = {}
            while True:
                line = reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                k, v = line.decode("latin-1").rstrip("\r\n").split(":", 1)
                headers[k.strip().lower()] = v.strip()
            body = reader.read(int(headers.get("content-length", "0") or 0))
            self.requests.append((method, target, headers, body))
            out = self.handler(method, target.split("?", 1)[0], body, conn)
            if out is not None:
                status, payload = out
                conn.sendall(
                    f"HTTP/1.1 {status} X\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                    + payload
                )
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        self._srv.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _basic_handler(method: str, path: str, body: bytes, conn: socket.socket) -> tuple[int, bytes]:
    if path == "/_ping":
        return 200, b"OK"
    if path.endswith("/containers/create"):
        return 201, b'{"Id": "cid-1"}'
    return 404, b'{"message": "nope"}'


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


def test_bridge_requires_token_and_refuses_escaping_container_creates():
    engine = _FakeEngine(_basic_handler)
    endpoint = kiri_installer._start_unix_tcp_bridge(engine.sock_path, "127.0.0.1", _free_port(), token=TOKEN)
    try:
        assert EngineSocketClient(endpoint).ping() is False  # no token -> 401
        assert EngineSocketClient(endpoint, token="wrong-token-0123456789").ping() is False
        authed = EngineSocketClient(endpoint, token=TOKEN)
        assert authed.ping() is True

        safe = ContainerConfig(image="img", pull_if_missing=False)
        assert authed.create_container(safe) == "cid-1"
        forwarded = [r for r in engine.requests if r[1].endswith("/containers/create")]
        assert len(forwarded) == 1
        # The secret is stripped before reaching the engine.
        assert runtime.SANDBOX_TOKEN_HEADER.lower() not in forwarded[0][2]

        with pytest.raises(SandboxInfraError, match="Binds"):
            authed.create_container(ContainerConfig(image="img", pull_if_missing=False, binds=["/:/host"]))
        raw_bodies = [
            {"Image": "img", "HostConfig": {"Privileged": True}},
            {"Image": "img", "HostConfig": {"NetworkMode": "host"}},
            {"Image": "img", "HostConfig": {"PidMode": "host"}},
            {"Image": "img", "HostConfig": {"Devices": [{"PathOnHost": "/dev/sda"}]}},
        ]
        for spec in raw_bodies:
            resp = authed._request("POST", "/v1.41/containers/create", body=spec, timeout=5.0)
            assert resp.status == 403, spec
        # Refused requests never reached the engine.
        assert len([r for r in engine.requests if r[1].endswith("/containers/create")]) == 1
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


def _bridge_and_engine():
    engine = _FakeEngine(_basic_handler)
    endpoint = kiri_installer._start_unix_tcp_bridge(engine.sock_path, "127.0.0.1", _free_port(), token=TOKEN)
    return engine, EngineSocketClient(endpoint, token=TOKEN)


def _forwarded(engine: _FakeEngine, suffix: str) -> list[tuple[str, str, dict[str, str], bytes]]:
    return [r for r in engine.requests if r[1].split("?", 1)[0].endswith(suffix)]


def test_bridge_refuses_host_mounts_through_volume_driver_and_volume_create_opts():
    engine, authed = _bridge_and_engine()
    try:
        host_bind_volume = {
            "Type": "volume",
            "Target": "/host",
            "VolumeOptions": {
                "DriverConfig": {"Name": "local", "Options": {"type": "none", "o": "bind", "device": "/"}}
            },
        }
        refused_creates = [
            {"Image": "img", "HostConfig": {"Mounts": [host_bind_volume]}},
            {"Image": "img", "HostConfig": {"Mounts": [{"Type": "bind", "Source": "/", "Target": "/h"}]}},
            {"Image": "img", "HostConfig": {"Mounts": [{"Type": "image", "Source": "x", "Target": "/h"}]}},
            {"Image": "img", "HostConfig": {"Mounts": [{"Target": "/h"}]}},
            {"Image": "img", "HostConfig": {"VolumeDriver": "rexray"}},
        ]
        for spec in refused_creates:
            assert authed._request("POST", "/v1.41/containers/create", body=spec, timeout=5.0).status == 403, spec
        assert _forwarded(engine, "/containers/create") == []

        ok = {"Image": "img", "HostConfig": {"Mounts": [{"Type": "tmpfs", "Target": "/scratch"}]}}
        assert authed._request("POST", "/v1.41/containers/create", body=ok, timeout=5.0).status == 201
        assert len(_forwarded(engine, "/containers/create")) == 1

        refused_volumes = [
            {"Name": "v", "DriverOpts": {"type": "none", "o": "bind", "device": "/"}},
            {"Name": "v", "Driver": "some-plugin"},
        ]
        for spec in refused_volumes:
            assert authed._request("POST", "/v1.41/volumes/create", body=spec, timeout=5.0).status == 403, spec
        assert _forwarded(engine, "/volumes/create") == []
        authed._request("POST", "/v1.41/volumes/create", body={"Name": "v"}, timeout=5.0)
        assert len(_forwarded(engine, "/volumes/create")) == 1
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


def test_bridge_matches_json_keys_case_insensitively_and_forwards_checked_body():
    engine, authed = _bridge_and_engine()
    try:
        refused = [
            b'{"Image": "img", "hostconfig": {"privileged": true}}',
            b'{"Image": "img", "HostConfig": {"PRIVILEGED": true}}',
            b'{"Image": "img", "HOSTCONFIG": {"binds": ["/:/h"]}}',
            b'{"Image": "img", "HostConfig": {"mounts": [{"type": "BIND", "source": "/", "target": "/h"}]}}',
            # Keys colliding case-insensitively (the engine may honour either one).
            b'{"Image": "img", "HostConfig": {}, "hostconfig": {"Privileged": true}}',
            b'{"Image": "img", "HostConfig": {"Privileged": false, "Privileged": true}}',
        ]
        for raw in refused:
            assert authed._request("POST", "/v1.41/containers/create", body=raw, timeout=5.0).status == 403, raw
        assert _forwarded(engine, "/containers/create") == []
        for raw in (b'{"Cmd": ["id"], "privileged": true}', b'{"Cmd": ["id"], "PRIVILEGED": true}'):
            assert authed._request("POST", "/v1.41/containers/cid/exec", body=raw, timeout=5.0).status == 403, raw
        assert _forwarded(engine, "/exec") == []

        # Approved bodies are forwarded re-serialised from the document the policy checked.
        authed._request("POST", "/v1.41/containers/create", body=b'{"Image":  "img" ,"Tty":false}', timeout=5.0)
        (fwd,) = _forwarded(engine, "/containers/create")
        assert json.loads(fwd[3]) == {"Image": "img", "Tty": False}
        assert int(fwd[2]["content-length"]) == len(fwd[3])
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


def test_bridge_normalises_request_targets_before_policy():
    engine, authed = _bridge_and_engine()
    privileged = {"Image": "img", "HostConfig": {"Privileged": True}}
    try:
        for target in (
            "//containers/create",
            "/containers//create",
            "/v1.43/containers/create",
            "/v1.43//containers/create?name=x",
            "/containers/%63reate",
            "/containers%2Fcreate",
            "/containers/../containers/create",
            "/x/./containers/create",
            "http://kiri/containers/create",
            "containers/create",
        ):
            assert authed._request("POST", target, body=privileged, timeout=5.0).status == 403, target
        assert engine.requests == []

        authed._request("POST", "/v1.43//containers///create?name=a", body={"Image": "img"}, timeout=5.0)
        (fwd,) = _forwarded(engine, "/containers/create")
        assert fwd[1] == "/v1.43/containers/create?name=a"
        # Encoded separators inside a segment stay encoded when forwarded.
        authed._request("GET", "/v1.41/images/reg%2Fimg/json", timeout=5.0)
        assert engine.requests[-1][1] == "/v1.41/images/reg%2Fimg/json"
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


def test_bridge_routable_bind_requires_token_and_allow_list():
    with pytest.raises(ValueError, match="token"):
        kiri_installer._start_unix_tcp_bridge("/nonexistent.sock", "10.1.2.3", 2375)
    with pytest.raises(ValueError, match="allowed_peers"):
        kiri_installer._start_unix_tcp_bridge("/nonexistent.sock", "10.1.2.3", 2375, token=TOKEN)


def test_bridge_rejects_peer_outside_allow_list():
    engine = _FakeEngine(_basic_handler)
    endpoint = kiri_installer._start_unix_tcp_bridge(
        engine.sock_path, "127.0.0.1", _free_port(), token=TOKEN, allowed_peers=["10.9.9.9"]
    )
    try:
        assert EngineSocketClient(endpoint, token=TOKEN).ping() is False
        assert engine.requests == []
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


def test_bridge_bind_failure_raises():
    occupant = socket.socket()
    occupant.bind(("127.0.0.1", 0))
    occupant.listen(1)
    port = occupant.getsockname()[1]
    try:
        with pytest.raises(SandboxInfraError, match="bind"):
            kiri_installer._start_unix_tcp_bridge("/nonexistent.sock", "127.0.0.1", port, token=TOKEN)
    finally:
        occupant.close()


# ---------------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------------

_FAKE_DAEMON_PY = r"""
import os, socket, subprocess
subprocess.Popen(["sleep", "300"])
path = os.path.join(os.environ["KIRI_DATA_DIR"], "run", "docker.sock")
try:
    os.unlink(path)
except FileNotFoundError:
    pass
srv = socket.socket(socket.AF_UNIX)
srv.bind(path)
srv.listen(8)
while True:
    c, _ = srv.accept()
    c.recv(65536)
    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
    c.close()
"""


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pgid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return True
        time.sleep(0.05)
    return False


def test_daemon_registry_stops_only_owned_process_groups():
    root = tempfile.mkdtemp(prefix="edkiri-", dir="/dev/shm")
    script = os.path.join(root, "kiri")
    with open(script, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/sh\nexec {sys.executable} -c '{_FAKE_DAEMON_PY}'\n")
    os.chmod(script, 0o755)
    stranger = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        pool = runtime.SandboxPool.from_local_shm(
            2,
            data_dir=os.path.join(root, "pool"),
            config=ContainerConfig(image="img"),
            budget=KiriDaemonBudget(tmpfs_mb=1),
            kiri_bin=script,
        )
        owned = kiri_installer.owned_kiri_daemons()
        assert pool.size == 2 and len(owned) == 2
        pgids = [kiri_installer._OWNED_DAEMONS[u].pgid for u in owned]
        assert all(_group_alive(g) for g in pgids)
        assert kiri_installer.kiri_daemon_budget(owned[0]) == KiriDaemonBudget(tmpfs_mb=1)

        # Reusing an already-running daemon does not take ownership of it.
        assert kiri_installer.ensure_kiri_daemon(data_dir=os.path.join(root, "pool", "vm-0")) in owned
        assert sorted(kiri_installer.owned_kiri_daemons()) == sorted(owned)

        pool.close()
        assert all(_wait_dead(g) for g in pgids)
        assert kiri_installer.owned_kiri_daemons() == []
        assert stranger.poll() is None and _group_alive(stranger.pid)
    finally:
        kiri_installer.shutdown_kiri_daemons()
        stranger.kill()
        stranger.wait()
        shutil.rmtree(root, ignore_errors=True)


def test_missing_kiri_binary_is_not_auto_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(kiri_installer.shutil, "which", lambda _n: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(kiri_installer, "install_kiri", lambda **kw: pytest.fail("network install attempted"))
    with pytest.raises(SandboxInfraError, match="easydel setup kiri"):
        kiri_installer.ensure_kiri_daemon(data_dir=str(tmp_path / "d"))


def test_budget_clamps_container_create_limits():
    seen: list[dict] = []

    def handler(method, path, body, conn):
        if path.endswith("/containers/create"):
            seen.append(json.loads(body))
            return 201, b'{"Id": "c"}'
        return 404, b"{}"

    engine = _FakeEngine(handler)
    try:
        client = EngineSocketClient(engine.endpoint, budget=KiriDaemonBudget(cpus=2.0, memory_mb=1024, pids_limit=64))
        client.create_container(ContainerConfig(image="i", pull_if_missing=False, cpus=8, memory_mb=4096, pids_limit=0))
        hc = seen[0]["HostConfig"]
        assert hc["NanoCpus"] == 2_000_000_000
        assert hc["Memory"] == 1024 * 1024 * 1024
        assert hc["PidsLimit"] == 64
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Exec deadlines and bounded reads
# ---------------------------------------------------------------------------


def test_local_exec_kills_child_that_never_reads_stdin_within_timeout():
    with LocalSandboxSession(ContainerConfig(use_jail="off")) as sess:
        start = time.monotonic()
        res = sess.exec(["/bin/sh", "-c", "sleep 30"], stdin="x" * (1024 * 1024), timeout=1.0)
        elapsed = time.monotonic() - start
    assert res.timed_out is True and res.ok is False
    assert elapsed < 5.0


def test_local_exec_rejects_oversized_stdin():
    with LocalSandboxSession(ContainerConfig(use_jail="off", max_stdin_bytes=10)) as sess:
        res = sess.exec(["cat"], stdin="0123456789ABC")
        assert res.ok is False and "max_stdin_bytes" in res.stderr
        assert sess.exec(["cat"], stdin="hello").stdout == "hello"


def test_container_exec_overall_deadline_against_trickling_engine():
    execs: dict[str, list[str]] = {}

    def handler(method, path, body, conn):
        if path.endswith("/exec") and method == "POST":
            eid = f"e{len(execs)}"
            execs[eid] = json.loads(body)["Cmd"]
            return 201, json.dumps({"Id": eid}).encode()
        if path.endswith("/start") and json.loads(body or b"{}").get("Detach") is False:
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/vnd.docker.raw-stream\r\n\r\n")
            try:
                for _ in range(600):  # one byte every 50 ms: never idle long enough for a per-recv timeout
                    conn.sendall(b"\x01")
                    time.sleep(0.05)
            except OSError:
                pass
            return None
        return 200, b"{}"

    engine = _FakeEngine(handler)
    try:
        client = EngineSocketClient(engine.endpoint)
        start = time.monotonic()
        res = client.exec_run("cid", ["yes"], timeout=1.0)
        elapsed = time.monotonic() - start
        assert res.timed_out is True and res.ok is False
        assert elapsed < 4.0
        # The command was wrapped so its pid is recorded, and a kill exec targeting that pid file followed.
        cmds = list(execs.values())
        pid_file = cmds[0][4]
        assert any(c[-1] == pid_file and "kill -KILL" in " ".join(c) for c in cmds[1:])
    finally:
        engine.close()


def test_container_exec_output_cap_stops_reading_and_kills():
    def handler(method, path, body, conn):
        if path.endswith("/exec") and method == "POST":
            return 201, b'{"Id": "e"}'
        if path.endswith("/start") and json.loads(body or b"{}").get("Detach") is False:
            conn.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
            frame = b"\x01\x00\x00\x00\x00\x01\x00\x00" + b"a" * 65536
            try:
                while True:
                    conn.sendall(frame)
            except OSError:
                pass
            return None
        return 200, b"{}"

    engine = _FakeEngine(handler)
    try:
        res = EngineSocketClient(engine.endpoint).exec_run("cid", ["yes"], timeout=10.0, max_output_chars=1000)
        assert res.ok is False and res.timed_out is False
        assert len(res.stdout) == 1000
        assert "output limit" in res.stderr
    finally:
        engine.close()


def test_read_file_is_bounded_locally_and_in_container():
    big = "abcdefghij" * 500_000  # 5 MB
    with LocalSandboxSession(ContainerConfig(use_jail="off")) as sess:
        sess.write_files({"big.txt": big})
        assert sess.read_file("big.txt", max_chars=100) == big[:101]
        assert sess.read_file("big.txt") == big

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("big.txt")
        info.size = len(big)
        tf.addfile(info, io.BytesIO(big.encode()))
    archive = buf.getvalue()
    sent: list[int] = []
    done = threading.Event()

    def handler(method, path, body, conn):
        if path.endswith("/archive"):
            conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Length: {len(archive)}\r\n\r\n".encode())
            try:
                for i in range(0, len(archive), 65536):
                    conn.sendall(archive[i : i + 65536])
                    sent.append(len(archive[i : i + 65536]))
            except OSError:
                pass
            finally:
                done.set()
            return None
        return 404, b"{}"

    engine = _FakeEngine(handler)
    try:
        data = EngineSocketClient(engine.endpoint).get_file("cid", "/workspace/big.txt", max_bytes=1000)
        assert data == big[:1000].encode()
        # The client closed the connection early: the engine could only push what fit in socket buffers.
        assert done.wait(10.0)
        assert sum(sent) < 2 * 1024 * 1024 < len(archive)
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Container exec through an engine that really runs the received Cmd
# ---------------------------------------------------------------------------


class _ShellEngine:
    """Fake engine executing each exec's ``Cmd`` with the local shell; ``/tmp/.ed_*`` paths map to a scratch dir."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="edexec-", dir="/dev/shm")
        self.cmds: dict[str, list[str]] = {}
        self.exit_codes: dict[str, int] = {}
        self.attached_done = threading.Event()
        self.engine = _FakeEngine(self._handle)

    def _remap(self, arg: str) -> str:
        return os.path.join(self.root, arg[len("/tmp/") :]) if arg.startswith("/tmp/.ed_") else arg

    def _handle(self, method, path, body, conn):
        if method == "PUT" and path.endswith("/archive"):
            with tarfile.open(fileobj=io.BytesIO(body)) as tf:
                for member in tf.getmembers():
                    with open(os.path.join(self.root, os.path.basename(member.name)), "wb") as f:
                        f.write(tf.extractfile(member).read())
            return 200, b""
        if method == "POST" and path.endswith("/exec"):
            eid = f"e{len(self.cmds)}"
            self.cmds[eid] = [self._remap(a) for a in json.loads(body)["Cmd"]]
            return 201, json.dumps({"Id": eid}).encode()
        if method == "POST" and path.endswith("/start"):
            eid = path.split("/")[-2]
            if json.loads(body or b"{}").get("Detach"):
                subprocess.Popen(self.cmds[eid], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                return 200, b""
            proc = subprocess.Popen(self.cmds[eid], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
            try:
                out, _ = proc.communicate(timeout=30)
            finally:
                self.attached_done.set()
            self.exit_codes[eid] = proc.returncode
            frame = b"\x01\x00\x00\x00" + len(out).to_bytes(4, "big") + out
            return 200, frame
        if method == "GET" and path.endswith("/json"):
            eid = path.split("/")[-2]
            return 200, json.dumps({"Running": False, "ExitCode": self.exit_codes.get(eid)}).encode()
        return 404, b"{}"

    def close(self):
        self.engine.close()
        shutil.rmtree(self.root, ignore_errors=True)


_PROC_FD_SCANNER = r"""
import os, re, stat, sys
pattern = re.compile(rb"EDSTDIN[0-9a-f]{32}")
hits = []
me = os.getpid()
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        continue
    for fd in fds:
        p = f"/proc/{pid}/fd/{fd}"
        try:
            if not stat.S_ISREG(os.stat(p).st_mode):
                continue
            with open(p, "rb") as f:
                data = f.read(1 << 20)
        except OSError:
            continue
        hits += [(pid, fd, m) for m in pattern.findall(data)]
marker = pattern.search(sys.stdin.buffer.read()).group(0)
print("STDIN_OK")
for pid, fd, m in hits:
    if m == marker:
        print(f"LEAK {pid} {fd}")
"""


def test_container_exec_stdin_file_is_held_only_by_the_command():
    shell = _ShellEngine()
    try:
        client = EngineSocketClient(shell.engine.endpoint)
        marker = "EDSTDIN" + os.urandom(16).hex()
        res = client.exec_run("cid", [sys.executable, "-I", "-c", _PROC_FD_SCANNER], stdin=marker, timeout=60.0)
        assert res.ok is True, res
        assert "STDIN_OK" in res.stdout
        assert "LEAK" not in res.stdout, res.stdout
        # The stdin file was unlinked before the command ran.
        assert [n for n in os.listdir(shell.root) if n.startswith(".ed_stdin_")] == []

        piped = client.exec_run("cid", ["/bin/sh", "-c", "cat; exit 3"], stdin="hello", timeout=30.0)
        assert piped.stdout == "hello" and piped.returncode == 3
        plain = client.exec_run("cid", ["echo", "hi"], timeout=30.0)
        assert plain.ok is True and plain.stdout == "hi\n"
    finally:
        shell.close()


def test_container_exec_deadline_kills_the_real_command():
    shell = _ShellEngine()
    try:
        client = EngineSocketClient(shell.engine.endpoint)
        start = time.monotonic()
        res = client.exec_run("cid", ["/bin/sh", "-c", "sleep 30 & sleep 30"], stdin="x", timeout=1.0)
        assert res.timed_out is True and res.ok is False
        # The kill exec terminated the command's whole process group, so the attached exec ended early.
        assert shell.attached_done.wait(10.0)
        assert time.monotonic() - start < 10.0
    finally:
        shell.close()


def _pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_local_kill_sandbox_processes_kills_session_groups_only():
    stranger = subprocess.Popen(["sleep", "300"], start_new_session=True)
    sess = LocalSandboxSession(ContainerConfig(use_jail="off", timeout=5.0))
    try:
        res = sess.run_bash("sleep 300 >/dev/null 2>&1 & echo $!")
        assert res.ok is True
        bg = int(res.stdout.strip())
        assert _pid_alive(bg)
        sess.kill_sandbox_processes()
        deadline = time.monotonic() + 5.0
        while _pid_alive(bg) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(bg)
        assert stranger.poll() is None
        # The session stays usable after the kill.
        assert sess.run_bash("echo ok").stdout.strip() == "ok"
    finally:
        sess.close()
        stranger.kill()
        stranger.wait()


# ---------------------------------------------------------------------------
# Jail policy
# ---------------------------------------------------------------------------


def test_no_unjailed_fallback_without_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _n: None)
    for policy in ("auto", "required"):
        with pytest.raises(SandboxInfraError, match="use_jail='off'"):
            LocalSandboxSession(ContainerConfig(use_jail=policy))

    monkeypatch.setattr(runtime, "_discover_engine_endpoint", lambda backend, auto_install=False: (None, "local", False))
    with pytest.raises(SandboxInfraError, match="bwrap"):
        KiriRuntime(backend="auto").create_session()

    with pytest.raises(SandboxInfraError, match="network=False"):
        LocalSandboxSession(ContainerConfig(use_jail="off", network=False))
    with pytest.raises(SandboxInfraError, match="read_only_rootfs"):
        LocalSandboxSession(ContainerConfig(use_jail="off", read_only_rootfs=True))

    sess = KiriRuntime(backend="auto", config=ContainerConfig(use_jail="off")).create_session()
    try:
        assert sess.jailed is False
        assert sess.run_bash("echo hi").stdout.strip() == "hi"
    finally:
        sess.close()


def test_local_exec_applies_file_size_limit():
    with LocalSandboxSession(ContainerConfig(use_jail="off", max_file_size_mb=1)) as sess:
        res = sess.run_bash("head -c 3000000 /dev/zero > out.bin; echo $?")
        assert os.path.getsize(os.path.join(sess.workdir, "out.bin")) <= 1024 * 1024
        assert res.stdout.strip() != "0" or res.returncode == -signal.SIGXFSZ
