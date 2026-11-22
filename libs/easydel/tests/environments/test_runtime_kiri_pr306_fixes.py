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

"""Targeted regression tests for PR #306 runtime, Kiri installer, base tools, and CLI fixes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import typing as tp

import pytest
from easydel.environments import SWERepoEnvironment, kiri_installer, runtime
from easydel.environments._base import SandboxEditFileTool, SandboxReadFileTool, SandboxWriteFileTool
from easydel.environments.runtime import (
    ContainerConfig,
    ContainerSandboxSession,
    EngineSocketClient,
    ExecResult,
    KiriDaemonBudget,
    KiriRuntime,
    LocalSandboxSession,
    SandboxInfraError,
    SandboxPool,
    _first_regular_tar_member,
)
from easydel.scripts import cli

pytestmark = pytest.mark.cpu_ok

TOKEN = "pr306-test-token-0123456789"
TOKEN_ALT = "pr306-other-token-9876543210"
_NO_JAIL = ContainerConfig(use_jail="off")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _UnixHTTPServer:
    """Minimal Unix-socket HTTP server for deterministic engine/bridge regression tests."""

    def __init__(self, handler: tp.Callable[[str, str, dict[str, str], bytes, socket.socket], tuple[int, bytes] | None]):
        self.dir = tempfile.mkdtemp(prefix="edpr306-", dir="/dev/shm")
        self.sock_path = os.path.join(self.dir, "e.sock")
        self.handler = handler
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        self._srv.listen(32)
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
            line = reader.readline().decode("latin-1").strip()
            if not line:
                return
            method, target, _ = line.split(" ", 2)
            headers: dict[str, str] = {}
            while True:
                hline = reader.readline()
                if not hline or hline in (b"\r\n", b"\n"):
                    break
                k, v = hline.decode("latin-1").rstrip("\r\n").split(":", 1)
                headers[k.strip().lower()] = v.strip()
            body = reader.read(int(headers.get("content-length", "0") or 0))
            self.requests.append((method, target, headers, body))
            out = self.handler(method, target, headers, body, conn)
            if out is not None:
                status, payload = out
                conn.sendall(
                    f"HTTP/1.1 {status} OK\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
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


# ---------------------------------------------------------------------------
# Finding #3: LocalSandboxSession.write_files FIFO deadlock, symlinks, dirs, permissions
# ---------------------------------------------------------------------------


def test_local_write_files_refuses_fifo_without_deadlock():
    with LocalSandboxSession(_NO_JAIL) as sess:
        fifo_path = os.path.join(sess.workdir, ".ed_eval_tests.patch")
        os.mkfifo(fifo_path)

        outcome: list[BaseException | str] = []

        def _write() -> None:
            try:
                sess.write_files({".ed_eval_tests.patch": "patch content"})
                outcome.append("wrote")
            except BaseException as exc:
                outcome.append(exc)

        t = threading.Thread(target=_write, daemon=True)
        t.start()
        t.join(3.0)
        assert not t.is_alive(), "write_files deadlocked on a FIFO with no reader"
        assert len(outcome) == 1 and isinstance(outcome[0], OSError)
        assert not isinstance(outcome[0], SandboxInfraError)


def test_local_write_files_refuses_fifo_even_with_active_reader():
    with LocalSandboxSession(_NO_JAIL) as sess:
        fifo_path = os.path.join(sess.workdir, "reader_fifo")
        os.mkfifo(fifo_path)
        rfd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            with pytest.raises(OSError, match="regular file"):
                sess.write_files({"reader_fifo": "secret-payload"})
            # Confirm zero bytes were written into the FIFO (writer closed without writing -> EOF).
            assert os.read(rfd, 64) == b""
        finally:
            os.close(rfd)


def test_local_write_files_rejects_symlinks_dirs_permissions_as_oserror(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("original-outside", encoding="utf-8")

    with LocalSandboxSession(_NO_JAIL) as sess:
        sess.write_files({"real.txt": "original-inside"})
        os.symlink(str(outside), os.path.join(sess.workdir, "escape_link"))
        os.symlink(os.path.join(sess.workdir, "real.txt"), os.path.join(sess.workdir, "internal_link"))
        os.symlink(str(tmp_path), os.path.join(sess.workdir, "dir_link"))
        os.mkdir(os.path.join(sess.workdir, "existing_dir"))
        locked_dir = os.path.join(sess.workdir, "locked")
        os.mkdir(locked_dir, 0o000)

        with pytest.raises(OSError):
            sess.write_files({"escape_link": "tampered"})
        assert outside.read_text(encoding="utf-8") == "original-outside"

        with pytest.raises(OSError):
            sess.write_files({"internal_link": "tampered"})
        assert sess.read_file("real.txt") == "original-inside"

        with pytest.raises(OSError):
            sess.write_files({"dir_link/new_outside.txt": "tampered"})
        assert not (tmp_path / "new_outside.txt").exists()

        with pytest.raises(IsADirectoryError):
            sess.write_files({"existing_dir": "nope"})
        with pytest.raises(IsADirectoryError):
            sess.write_files({".": "nope"})

        with pytest.raises(PermissionError):
            sess.write_files({"locked/file.txt": "nope"})

        # Lexical escape still raises ValueError.
        with pytest.raises(ValueError, match="escapes sandbox"):
            sess.write_files({"../../outside.txt": "nope"})
        with pytest.raises(ValueError, match="escapes sandbox"):
            sess.write_files({f"{sess.workdir}/../outside.txt": "nope"})

        root = sess.workdir
    # Closed session raises SandboxInfraError.
    with pytest.raises(SandboxInfraError, match="closed"):
        sess.write_files({"after_close.txt": "x"})
    # close() must also clean up chmod-000 directories cleanly (#17).
    assert not os.path.exists(root)


# ---------------------------------------------------------------------------
# Finding #5: Bridge policy bypasses & KiriDaemonBudget.clamp_host_config
# ---------------------------------------------------------------------------


def test_bridge_blocks_container_update_and_unlisted_routes():
    def handler(method, target, headers, body, conn):
        if target.split("?", 1)[0].endswith("/containers/create"):
            return 201, b'{"Id": "cid-1"}'
        return 200, b"{}"

    engine = _UnixHTTPServer(handler)
    port = _free_port()
    ep = kiri_installer._start_unix_tcp_bridge(engine.sock_path, "127.0.0.1", port, token=TOKEN)
    try:
        client = EngineSocketClient(ep, token=TOKEN)
        update_body = {"Memory": 0, "NanoCpus": 0, "PidsLimit": -1}
        assert client._request("POST", "/v1.41/containers/cid-1/update", body=update_body, timeout=5.0).status == 403
        assert client._request("POST", "/containers/cid-1/update", body=update_body, timeout=5.0).status == 403
        assert client._request("POST", "/v1.41/containers/cid-1/commit", body={}, timeout=5.0).status == 403
        assert client._request("POST", "/v1.41/networks/create", body={"Name": "n"}, timeout=5.0).status == 403
        assert engine.requests == []
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


def test_bridge_clamps_mounts_tmpfs_and_tmpfs_options():
    def handler(method, target, headers, body, conn):
        if target.split("?", 1)[0].endswith("/containers/create"):
            return 201, b'{"Id": "cid-1"}'
        return 200, b"{}"

    engine = _UnixHTTPServer(handler)
    port = _free_port()
    kiri_installer._ENDPOINT_BUDGETS[f"unix://{engine.sock_path}"] = KiriDaemonBudget(
        cpus=2.0, memory_mb=512, pids_limit=64, tmpfs_mb=64
    )
    ep = kiri_installer._start_unix_tcp_bridge(engine.sock_path, "127.0.0.1", port, token=TOKEN)
    try:
        client = EngineSocketClient(ep, token=TOKEN)
        spec = {
            "Image": "img",
            "ShmSize": 100 * 1024**3,
            "HostConfig": {
                "MemorySwap": -1,
                "shmsize": 100 * 1024**3,
                "Tmpfs": {
                    "/a": "rw,SIZE=10g,nr_blocks=999999999",
                    "/b": "rw,size=50%",
                    "/c": "rw,size=16m",
                },
                "mounts": [
                    {"Type": "tmpfs", "Target": "/m1"},
                    {"Type": "tmpfs", "Target": "/m2", "TmpfsOptions": {"SizeBytes": 0}},
                    {
                        "Type": "tmpfs",
                        "Target": "/m3",
                        "tmpfsoptions": {
                            "sizebytes": 10 * 1024**3,
                            "Options": [
                                ["size", "10g"],
                                ["size=10g"],
                                ["nr_blocks=999999999"],
                                ["rw,size=10g"],
                                "rw,size=10g",
                                ["noexec"],
                            ],
                        },
                    },
                    {"Type": "tmpfs", "Target": "/m4", "TmpfsOptions": {"SizeBytes": 4 * 1024 * 1024}},
                ],
            },
        }
        resp = client._request("POST", "/v1.41/containers/create", body=spec, timeout=5.0)
        assert resp.status == 201
        forwarded_doc = json.loads(engine.requests[-1][3])
        assert "ShmSize" not in forwarded_doc
        forwarded = forwarded_doc["HostConfig"]
        assert forwarded["Memory"] == 512 * 1024 * 1024
        assert forwarded["MemorySwap"] == 512 * 1024 * 1024
        assert forwarded["ShmSize"] == 64 * 1024 * 1024
        assert forwarded["Tmpfs"]["/a"] == "rw,size=64m"
        assert forwarded["Tmpfs"]["/b"] == "rw,size=64m"
        assert forwarded["Tmpfs"]["/c"] == "rw,size=16m"
        mounts = forwarded["Mounts"]
        assert mounts[0]["TmpfsOptions"]["SizeBytes"] == 64 * 1024 * 1024
        assert mounts[1]["TmpfsOptions"]["SizeBytes"] == 64 * 1024 * 1024
        assert mounts[2]["TmpfsOptions"]["SizeBytes"] == 64 * 1024 * 1024
        assert mounts[2]["TmpfsOptions"]["Options"] == [["noexec"]]
        assert mounts[3]["TmpfsOptions"]["SizeBytes"] == 4 * 1024 * 1024

        # Boolean resource limit, malformed TmpfsOptions.Options, and non-standard JSON Infinity/NaN must be 403.
        bad_bool = {"Image": "img", "HostConfig": {"Memory": True}}
        assert client._request("POST", "/v1.41/containers/create", body=bad_bool, timeout=5.0).status == 403
        bad_opts = {
            "Image": "img",
            "HostConfig": {"Mounts": [{"Type": "tmpfs", "Target": "/m", "TmpfsOptions": {"Options": [123]}}]},
        }
        assert client._request("POST", "/v1.41/containers/create", body=bad_opts, timeout=5.0).status == 403
        for raw_bad in (
            b'{"Image": "img", "HostConfig": {"Memory": Infinity}}',
            b'{"Image": "img", "HostConfig": {"Memory": NaN}}',
            b'{"Image": "img", "HostConfig": {"Mounts": [{"Type": "tmpfs", "Target": "/m", "TmpfsOptions": {"SizeBytes": Infinity}}]}}',
        ):
            assert client._request("POST", "/v1.41/containers/create", body=raw_bad, timeout=5.0).status == 403, raw_bad
    finally:
        kiri_installer.shutdown_kiri_daemons()
        kiri_installer._ENDPOINT_BUDGETS.pop(f"unix://{engine.sock_path}", None)
        engine.close()


def test_bridge_blocks_security_opt_masked_paths_readonly_paths_and_non_iterable():
    def handler(method, target, headers, body, conn):
        if target.split("?", 1)[0].endswith("/containers/create"):
            return 201, b'{"Id": "cid-1"}'
        return 200, b"{}"

    engine = _UnixHTTPServer(handler)
    port = _free_port()
    ep = kiri_installer._start_unix_tcp_bridge(engine.sock_path, "127.0.0.1", port, token=TOKEN)
    try:
        client = EngineSocketClient(ep, token=TOKEN)
        refused_specs = [
            {"Image": "img", "HostConfig": {"SecurityOpt": ["label=disable"]}},
            {"Image": "img", "HostConfig": {"SecurityOpt": ['seccomp={"defaultAction":"SCMP_ACT_ALLOW"}']}},
            {"Image": "img", "HostConfig": {"SecurityOpt": ["no-new-privileges=false"]}},
            {"Image": "img", "HostConfig": {"SecurityOpt": 123}},
            {"Image": "img", "HostConfig": {"SecurityOpt": "no-new-privileges"}},
            {"Image": "img", "HostConfig": {"MaskedPaths": []}},
            {"Image": "img", "HostConfig": {"MaskedPaths": ["/tmp"]}},
            {"Image": "img", "HostConfig": {"ReadonlyPaths": []}},
            {"Image": "img", "HostConfig": {"ReadonlyPaths": ["/proc"]}},
            {"Image": "img", "maskedpaths": []},
            {"Image": "img", "readonlypaths": []},
            {"Image": "img", "HostConfig": {"CgroupParent": "/"}},
            {"Image": "img", "HostConfig": {"Sysctls": {"net.ipv4.ip_forward": "1"}}},
            {"Image": "img", "HostConfig": {"GroupAdd": ["0"]}},
            {"Image": "img", "HostConfig": {"Tmpfs": "not-a-dict"}},
            {"Image": "img", "HostConfig": {"Mounts": [{"Type": "tmpfs", "Target": "/x", "BindOptions": {"Propagation": "rprivate"}}]}},
        ]
        for spec in refused_specs:
            resp = client._request("POST", "/v1.41/containers/create", body=spec, timeout=5.0)
            assert resp.status == 403, spec
        assert engine.requests == []

        # Safe SecurityOpt forms are still accepted.
        for ok_opt in (["no-new-privileges"], ["no-new-privileges:true"], ["no-new-privileges=true"], []):
            resp = client._request(
                "POST",
                "/v1.41/containers/create",
                body={"Image": "img", "HostConfig": {"SecurityOpt": ok_opt}},
                timeout=5.0,
            )
            assert resp.status == 201, ok_opt
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


# ---------------------------------------------------------------------------
# Finding #7: Asset URL / version validation & bounded download size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_version", ["../other", "..", ".", "v1.0?x=1", "v1.0#frag", "v1.0/sub", "v1.0\n", "v 1"])
def test_invalid_kiri_version_rejected_before_network(bad_version, monkeypatch):
    monkeypatch.setattr(kiri_installer.shutil, "which", lambda *_: pytest.fail("which called before validation"))
    monkeypatch.setattr(kiri_installer.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("urlopen called"))
    with pytest.raises(ValueError, match="version"):
        kiri_installer.resolve_kiri_asset_url(version=bad_version)
    with pytest.raises(ValueError, match="version"):
        kiri_installer.install_kiri(version=bad_version)
    with pytest.raises(ValueError, match="version"):
        cli.main(["setup", "cpu-vm", "10.0.0.1", "--kiri-version", bad_version, "--dry-run"])


@pytest.mark.parametrize(
    "bad_url",
    [
        "file:///etc/passwd",
        "http://github.com/libracoder/kiri/releases/download/v0.2.1/kiri.deb",
        "ftp://github.com/kiri.deb",
        "https://127.0.0.1/kiri.deb",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::1]/kiri.deb",
        "https://localhost/kiri.deb",
        "https://metadata.google.internal/kiri.deb",
        "https://metadata.goog/computeMetadata/v1/",
        "https://sub.metadata.goog/kiri.deb",
        "https://metadata/kiri.deb",
        "https://instance-data/kiri.deb",
        "https://github.com:8443/kiri.deb",
        "https://user:pass@github.com/kiri.deb",
        "https://0177.0.0.1/kiri.deb",
        "https://intranet/kiri.deb",
        "https://github.com/kiri.deb\n",
    ],
)
def test_unsafe_kiri_download_urls_rejected(bad_url, monkeypatch):
    with pytest.raises(ValueError):
        kiri_installer.validate_kiri_download_url(bad_url)

    meta = {
        "tag_name": "v0.2.1",
        "assets": [{"name": "kiri-lite_0.2.1_amd64.deb", "browser_download_url": bad_url}],
    }

    class _StreamResp:
        def __init__(self, data: bytes):
            self._buf = io.BytesIO(data)

        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(kiri_installer.urllib.request, "urlopen", lambda req, timeout=20.0: _StreamResp(json.dumps(meta).encode()))
    with pytest.raises(ValueError):
        kiri_installer.resolve_kiri_asset_url(version="v0.2.1", machine="x86_64", system="linux")

    with pytest.raises(ValueError):
        cli._build_remote_kiri_script(
            deb_url_amd64=bad_url,
            deb_sha_amd64="a" * 64,
            deb_url_arm64="https://github.com/libracoder/kiri/releases/download/v0.2.1/kiri-lite_0.2.1_arm64.deb",
            deb_sha_arm64="b" * 64,
            num_daemons=1,
            data_dir=None,
        )


def test_install_kiri_enforces_download_size_cap(monkeypatch, tmp_path):
    bytes_read = 0

    class _HugeStream:
        def read(self, n: int = -1) -> bytes:
            nonlocal bytes_read
            chunk = b"x" * (65536 if n < 0 else n)
            bytes_read += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(kiri_installer, "_MAX_KIRI_ARCHIVE_BYTES", 4096)
    monkeypatch.setattr(kiri_installer.shutil, "which", lambda *_: None)
    monkeypatch.setattr(
        kiri_installer,
        "resolve_kiri_asset_url",
        lambda **kw: ("v0.2.1", "Kiri-0.2.1-x86_64.tar.gz", "https://github.com/libracoder/kiri/releases/download/v0.2.1/Kiri-0.2.1-x86_64.tar.gz", "a" * 64),
    )
    monkeypatch.setattr(kiri_installer.urllib.request, "urlopen", lambda req, timeout=60.0: _HugeStream())

    with pytest.raises(RuntimeError, match="exceeds maximum allowed size"):
        kiri_installer.install_kiri(install_dir=tmp_path, force=True)
    # Must stop immediately after exceeding the cap rather than reading unbounded data.
    assert bytes_read <= 4096 + 65536


# ---------------------------------------------------------------------------
# Finding #8: Reused-daemon TCP bridge leaks & auth mismatch on bridge reuse
# ---------------------------------------------------------------------------


def test_pod_shm_pool_closes_bridges_for_reused_daemons_and_rolls_back_on_error(monkeypatch):
    def handler(method, target, headers, body, conn):
        if target == "/_ping":
            return 200, b"OK"
        return 404, b"{}"

    e1 = _UnixHTTPServer(handler)
    e2 = _UnixHTTPServer(handler)
    base_port = _free_port()
    # Find two adjacent free ports; if base_port+1 happens to be busy, retry briefly.
    for _ in range(10):
        try:
            with socket.socket() as s1, socket.socket() as s2:
                s1.bind(("127.0.0.1", base_port))
                s2.bind(("127.0.0.1", base_port + 1))
            break
        except OSError:
            base_port = _free_port()

    try:
        # Simulate pre-existing (reused) daemons: started == []
        monkeypatch.setattr(
            kiri_installer,
            "_ensure_shm_pool",
            lambda n, **kw: ([e1.endpoint, e2.endpoint][:n], []),
        )
        monkeypatch.setattr(kiri_installer, "_detect_private_ip", lambda: "127.0.0.1")

        pool = SandboxPool.from_pod_shm(
            2,
            base_port=base_port,
            host_ips=["127.0.0.1"],
            expose_bridge=True,
            bridge_token=TOKEN,
            allowed_peers=["127.0.0.1"],
        )
        assert EngineSocketClient(f"http://127.0.0.1:{base_port}", token=TOKEN).ping() is True
        assert EngineSocketClient(f"http://127.0.0.1:{base_port + 1}", token=TOKEN).ping() is True
        assert ("127.0.0.1", base_port) in kiri_installer._TCP_BRIDGES
        assert ("127.0.0.1", base_port + 1) in kiri_installer._TCP_BRIDGES

        pool.close()
        assert ("127.0.0.1", base_port) not in kiri_installer._TCP_BRIDGES
        assert ("127.0.0.1", base_port + 1) not in kiri_installer._TCP_BRIDGES
        assert EngineSocketClient(f"http://127.0.0.1:{base_port}", token=TOKEN).ping() is False
        # Reused daemons themselves are untouched and still answering.
        assert EngineSocketClient(e1.endpoint).ping() is True
        assert EngineSocketClient(e2.endpoint).ping() is True

        # Partial failure rollback: occupy base_port + 1 so second bridge fails; first bridge must be closed!
        occupant = socket.socket()
        occupant.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupant.bind(("127.0.0.1", base_port + 1))
        occupant.listen(1)
        try:
            with pytest.raises(SandboxInfraError, match="Failed to bind"):
                SandboxPool.from_pod_shm(
                    2,
                    base_port=base_port,
                    host_ips=["127.0.0.1"],
                    expose_bridge=True,
                    bridge_token=TOKEN,
                    allowed_peers=["127.0.0.1"],
                )
            assert ("127.0.0.1", base_port) not in kiri_installer._TCP_BRIDGES
            assert EngineSocketClient(f"http://127.0.0.1:{base_port}", token=TOKEN).ping() is False
        finally:
            occupant.close()
    finally:
        kiri_installer.shutdown_kiri_daemons()
        e1.close()
        e2.close()


def test_start_unix_tcp_bridge_rejects_mismatched_auth_on_reuse():
    engine = _UnixHTTPServer(lambda m, t, h, b, c: (200, b"OK"))
    port = _free_port()
    try:
        ep = kiri_installer._start_unix_tcp_bridge(
            engine.sock_path, "127.0.0.1", port, token=TOKEN, allowed_peers=["127.0.0.1"]
        )
        # Matching policy reuses cleanly.
        assert (
            kiri_installer._start_unix_tcp_bridge(
                engine.sock_path, "127.0.0.1", port, token=TOKEN, allowed_peers=["127.0.0.1"]
            )
            == ep
        )
        # Mismatched token (or missing token) refused.
        with pytest.raises(ValueError, match="different token"):
            kiri_installer._start_unix_tcp_bridge(
                engine.sock_path, "127.0.0.1", port, token=TOKEN_ALT, allowed_peers=["127.0.0.1"]
            )
        with pytest.raises(ValueError, match="different token"):
            kiri_installer._start_unix_tcp_bridge(
                engine.sock_path, "127.0.0.1", port, token=None, allowed_peers=["127.0.0.1"]
            )
        # Mismatched allowed_peers refused.
        with pytest.raises(ValueError, match="allowed_peers"):
            kiri_installer._start_unix_tcp_bridge(
                engine.sock_path, "127.0.0.1", port, token=TOKEN, allowed_peers=["127.0.0.1", "10.0.0.2"]
            )
        with pytest.raises(ValueError, match="allowed_peers"):
            kiri_installer._start_unix_tcp_bridge(engine.sock_path, "127.0.0.1", port, token=TOKEN, allowed_peers=None)
    finally:
        kiri_installer.shutdown_kiri_daemons()
        engine.close()


# ---------------------------------------------------------------------------
# Finding #12: EngineSocketClient.exec_run stdin upload & kill_all_processes timeout
# ---------------------------------------------------------------------------


def test_exec_run_stdin_upload_error_and_timeout_return_exec_result():
    def err_handler(method, target, headers, body, conn):
        if method == "PUT" and "/archive" in target:
            return 500, b'{"message": "disk full"}'
        return 200, b"{}"

    engine_err = _UnixHTTPServer(err_handler)
    try:
        res = EngineSocketClient(engine_err.endpoint).exec_run("cid", ["cat"], stdin="hello", timeout=5.0)
        assert res.ok is False and res.timed_out is False
        assert res.infra_error is not None and "disk full" in res.infra_error
    finally:
        engine_err.close()

    def hang_handler(method, target, headers, body, conn):
        if method == "PUT" and "/archive" in target:
            time.sleep(5.0)
            return None
        return 200, b"{}"

    engine_hang = _UnixHTTPServer(hang_handler)
    try:
        start = time.monotonic()
        res = EngineSocketClient(engine_hang.endpoint).exec_run("cid", ["cat"], stdin="hello", timeout=0.4)
        elapsed = time.monotonic() - start
        assert res.ok is False and res.timed_out is True
        assert elapsed < 3.0
    finally:
        engine_hang.close()


def test_kill_all_processes_raises_when_timeout_expires():
    def handler(method, target, headers, body, conn):
        path = target.split("?", 1)[0]
        if method == "POST" and path.endswith("/exec"):
            return 201, b'{"Id": "kill-1"}'
        if method == "POST" and path.endswith("/start"):
            return 200, b""
        if method == "GET" and path.endswith("/json"):
            return 200, b'{"Running": true}'
        return 404, b"{}"

    engine = _UnixHTTPServer(handler)
    try:
        client = EngineSocketClient(engine.endpoint)
        with pytest.raises(SandboxInfraError, match="Timed out"):
            client.kill_all_processes("cid-1", wait=0.15)
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Finding #13: ContainerSandboxSession.write_files path traversal & absolute paths
# ---------------------------------------------------------------------------


def test_container_write_files_validates_before_mkdir_and_handles_abs_paths():
    recorded_execs: list[list[str]] = []
    uploaded: dict[str, bytes] = {}
    mkdir_exit_code = 0

    def handler(method, target, headers, body, conn):
        path = target.split("?", 1)[0]
        if method == "POST" and path.endswith("/containers/create"):
            return 201, b'{"Id": "c1"}'
        if method == "POST" and path.endswith("/start") and "/containers/" in path:
            return 204, b""
        if method == "DELETE" and "/containers/" in path:
            return 204, b""
        if method == "POST" and path.endswith("/exec"):
            cmd = json.loads(body)["Cmd"]
            if len(cmd) >= 7 and cmd[:2] == ["/bin/sh", "-c"] and cmd[3] == "sh":
                cmd = cmd[7:]
            recorded_execs.append(cmd)
            return 201, json.dumps({"Id": f"e{len(recorded_execs)}"}).encode()
        if method == "POST" and path.endswith("/start"):
            return 200, b""
        if method == "GET" and path.endswith("/json"):
            return 200, json.dumps({"Running": False, "ExitCode": mkdir_exit_code}).encode()
        if method == "PUT" and path.endswith("/archive"):
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as tf:
                for m in tf.getmembers():
                    f = tf.extractfile(m)
                    if f is not None:
                        uploaded[m.name] = f.read()
            return 200, b""
        return 404, b"{}"

    engine = _UnixHTTPServer(handler)
    try:
        client = EngineSocketClient(engine.endpoint)
        cfg = ContainerConfig(image="img", workdir="/workspace", pull_if_missing=False)
        with ContainerSandboxSession(client, cfg) as sess:
            recorded_execs.clear()
            # Traversal must fail BEFORE any mkdir exec is sent.
            for bad_path in ("../../etc/pwn/a.txt", "/etc/pwn/a.txt", "sub/../../outside/b.txt"):
                with pytest.raises(ValueError, match="escapes base_dir"):
                    sess.write_files({bad_path: "nope"})
            assert recorded_execs == []

            # Session-rooted absolute path must create /workspace/sub (NOT /workspace/workspace/sub).
            sess.write_files({"/workspace/sub/a.txt": "hello"})
            mkdir_cmds = [c for c in recorded_execs if c[:2] == ["mkdir", "-p"]]
            assert mkdir_cmds == [["mkdir", "-p", "--", "/workspace/sub"]]
            assert uploaded["sub/a.txt"] == b"hello"

            # Non-zero mkdir exit (agent workspace error) must raise OSError, NOT SandboxInfraError.
            mkdir_exit_code = 1
            with pytest.raises(OSError, match="Failed to create directories") as exc_info:
                sess.write_files({"fail_dir/a.txt": "x"})
            assert not isinstance(exc_info.value, SandboxInfraError)
            mkdir_exit_code = 0

            # Daemon/infra error during mkdir MUST raise SandboxInfraError.
            orig_exec = client.exec_run
            client.exec_run = lambda *a, **kw: ExecResult(  # type: ignore[method-assign]
                ok=False, stdout="", stderr="", returncode=None, infra_error="daemon down"
            )
            try:
                with pytest.raises(SandboxInfraError, match="daemon down"):
                    sess.write_files({"infra_dir/a.txt": "x"})
            finally:
                client.exec_run = orig_exec  # type: ignore[method-assign]
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Finding #17: LocalSandboxSession bwrap venv/cwd/tmpfs, pgid pruning, BaseException cleanup
# ---------------------------------------------------------------------------


def test_bwrap_argv_mounts_venv_preserves_relative_cwd_and_bounds_tmpfs(monkeypatch, tmp_path):
    fake_venv = tmp_path / "my-venv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    fake_py = fake_bin / "python"
    os.symlink(sys.executable, fake_py)

    monkeypatch.setattr(runtime.shutil, "which", lambda name, path=None: "/usr/bin/bwrap" if name == "bwrap" else None)
    monkeypatch.setattr(runtime.sys, "prefix", str(fake_venv))
    monkeypatch.setattr(runtime.sys, "executable", str(fake_py))

    sess = LocalSandboxSession(ContainerConfig(use_jail="required", tmpfs_mb=64))
    try:
        # 1) sys.prefix is ro-bound, --size bounds --tmpfs /tmp, and venv python path is NOT dereferenced.
        bwrap_py = sess._build_bwrap_argv([str(fake_py), "-c", "pass"], sess.workdir)
        ro_pairs = [(bwrap_py[i + 1], bwrap_py[i + 2]) for i, tok in enumerate(bwrap_py) if tok == "--ro-bind"]
        assert (str(fake_venv), str(fake_venv)) in ro_pairs
        tmpfs_idx = bwrap_py.index("--tmpfs")
        assert bwrap_py[tmpfs_idx - 2 : tmpfs_idx + 2] == ["--size", str(64 * 1024 * 1024), "--tmpfs", "/tmp"]
        sep_idx = bwrap_py.index("--")
        assert bwrap_py[sep_idx + 1] == str(fake_py)

        # 2) Relative argv[0] is NOT resolved against host cwd even if a matching file exists there.
        host_dir = tmp_path / "host_cwd"
        host_dir.mkdir()
        (host_dir / "run.sh").write_text("#!/bin/sh\necho host\n", encoding="utf-8")
        (host_dir / "run.sh").chmod(0o755)
        monkeypatch.chdir(host_dir)
        bwrap_rel = sess._build_bwrap_argv(["./run.sh", "--flag"], sess.workdir)
        assert bwrap_rel[bwrap_rel.index("--") + 1 :] == ["./run.sh", "--flag"]
    finally:
        sess.close()

    # 3) tmpfs_mb=0 does not mount unbounded --tmpfs /tmp.
    sess_no_tmpfs = LocalSandboxSession(ContainerConfig(use_jail="required", tmpfs_mb=0))
    try:
        bwrap_zero = sess_no_tmpfs._build_bwrap_argv(["/bin/sh", "-c", "true"], sess_no_tmpfs.workdir)
        assert "--tmpfs" not in bwrap_zero
        dir_idx = bwrap_zero.index("--dir")
        assert bwrap_zero[dir_idx + 1] == "/tmp"
    finally:
        sess_no_tmpfs.close()


def test_local_session_prunes_dead_pgids_and_cleans_up_on_keyboard_interrupt(monkeypatch):
    with LocalSandboxSession(_NO_JAIL) as sess:
        res = sess.run_bash("echo done")
        assert res.ok is True
        # Dead process group was pruned immediately after proc.wait() (no PID-reuse hazard).
        assert sess._spawned_pgids == set()

        # Simulate KeyboardInterrupt during proc.wait(); spawned process group must be killed and pruned.
        spawned_pids: list[int] = []
        interrupted = False
        orig_popen = runtime.subprocess.Popen

        def _interrupting_popen(*args, **kwargs):
            proc = orig_popen(*args, **kwargs)
            spawned_pids.append(proc.pid)
            orig_wait = proc.wait

            def _wait(timeout=None):
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt("simulated interrupt")
                return orig_wait(timeout=timeout)

            proc.wait = _wait
            return proc

        monkeypatch.setattr(runtime.subprocess, "Popen", _interrupting_popen)
        with pytest.raises(KeyboardInterrupt):
            sess.run_bash("sleep 60")

        assert len(spawned_pids) == 1
        assert sess._spawned_pgids == set()
        # Process group must be dead.
        deadline = time.monotonic() + 3.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.killpg(spawned_pids[0], 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.02)
        assert not alive


# ---------------------------------------------------------------------------
# Finding #18: _first_regular_tar_member directory bug, BaseException container cleanup, max_stdin_bytes
# ---------------------------------------------------------------------------


def test_first_regular_tar_member_rejects_directory_but_skips_pax_headers():
    # Archive starting with directory + child regular file -> must return None (NOT child file!).
    dir_buf = io.BytesIO()
    with tarfile.open(fileobj=dir_buf, mode="w") as tf:
        d = tarfile.TarInfo(name="mydir")
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
        child_data = b"secret-child-content"
        c = tarfile.TarInfo(name="mydir/child.txt")
        c.size = len(child_data)
        tf.addfile(c, io.BytesIO(child_data))
    assert _first_regular_tar_member(dir_buf.getvalue(), max_bytes=1024) is None

    # Archive with PAX extended header before a regular file -> must return regular file bytes.
    pax_buf = io.BytesIO()
    with tarfile.open(fileobj=pax_buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        payload = b"hello-pax-file"
        f = tarfile.TarInfo(name="very_long_" + "a" * 120 + ".txt")
        f.size = len(payload)
        tf.addfile(f, io.BytesIO(payload))
    assert _first_regular_tar_member(pax_buf.getvalue(), max_bytes=1024) == b"hello-pax-file"


def test_container_session_init_cleans_up_on_base_exception_and_forwards_max_stdin():
    removed: list[str] = []
    start_should_interrupt = False
    mkdir_exit = 0
    seen_max_stdin: list[int] = []

    def handler(method, target, headers, body, conn):
        path = target.split("?", 1)[0]
        if method == "POST" and path.endswith("/containers/create"):
            return 201, b'{"Id": "cid-cleanup"}'
        if method == "POST" and path.endswith("/start") and "/containers/" in path:
            return 204, b""
        if method == "DELETE" and "/containers/" in path:
            removed.append(path.split("/")[3])
            return 204, b""
        if method == "POST" and path.endswith("/exec"):
            return 201, b'{"Id": "e1"}'
        if method == "POST" and path.endswith("/start"):
            return 200, b""
        if method == "GET" and path.endswith("/json"):
            return 200, json.dumps({"Running": False, "ExitCode": mkdir_exit}).encode()
        return 404, b"{}"

    engine = _UnixHTTPServer(handler)
    try:
        client = EngineSocketClient(engine.endpoint)
        cfg = ContainerConfig(image="img", pull_if_missing=False, max_stdin_bytes=20 * 1024 * 1024)

        # 1) BaseException (KeyboardInterrupt) during init must remove created container.
        orig_start = client.start_container

        def _interrupt_start(cid: str) -> None:
            if start_should_interrupt:
                raise KeyboardInterrupt("boom")
            orig_start(cid)

        client.start_container = _interrupt_start  # type: ignore[method-assign]
        start_should_interrupt = True
        with pytest.raises(KeyboardInterrupt):
            ContainerSandboxSession(client, cfg)
        assert removed == ["cid-cleanup"]

        # 2) Failed mkdir during init must raise SandboxInfraError and remove created container.
        start_should_interrupt = False
        removed.clear()
        mkdir_exit = 1
        with pytest.raises(SandboxInfraError, match="Failed to prepare container workdir"):
            ContainerSandboxSession(client, cfg)
        assert removed == ["cid-cleanup"]

        # 3) ContainerSandboxSession.exec forwards config.max_stdin_bytes.
        mkdir_exit = 0
        removed.clear()
        with ContainerSandboxSession(client, cfg) as sess:
            orig_exec = client.exec_run

            def _capture_exec(*args, **kwargs):
                seen_max_stdin.append(kwargs["max_stdin_bytes"])
                return ExecResult(ok=True, stdout="", stderr="", returncode=0)

            client.exec_run = _capture_exec  # type: ignore[method-assign]
            try:
                sess.exec(["echo", "hi"])
            finally:
                client.exec_run = orig_exec  # type: ignore[method-assign]
        assert seen_max_stdin == [20 * 1024 * 1024]
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Finding #20: kiri_installer ar parsing, temp-file cleanup, daemon fast-fail & interrupt cleanup
# ---------------------------------------------------------------------------


def test_extract_ar_data_tar_rejects_negative_and_truncated_sizes():
    # Negative size previously caused an infinite loop.
    neg_header = b"control.tar.gz  0           0     0     100644  -60       `\n"
    with pytest.raises(RuntimeError, match="Invalid or truncated"):
        kiri_installer._extract_ar_data_tar(b"!<arch>\n" + neg_header)

    # Truncated member claiming more bytes than remain in archive.
    trunc_header = b"data.tar.gz     0           0     0     100644  9999      `\n"
    with pytest.raises(RuntimeError, match="Invalid or truncated"):
        kiri_installer._extract_ar_data_tar(b"!<arch>\n" + trunc_header + b"short")

    # Bad trailer.
    bad_trailer = b"data.tar.gz     0           0     0     100644  4         XX"
    with pytest.raises(RuntimeError, match="trailer"):
        kiri_installer._extract_ar_data_tar(b"!<arch>\n" + bad_trailer + b"1234")


def test_extract_kiri_binary_cleans_up_temp_file_on_error(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"#!/bin/sh\necho ok\n"
        info = tarfile.TarInfo(name="kiri")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    def _boom(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(kiri_installer.os, "replace", _boom)
    target = str(tmp_path / "kiri")
    with pytest.raises(OSError, match="simulated replace failure"):
        kiri_installer._extract_kiri_binary_from_tar(buf.getvalue(), target)
    assert list(tmp_path.iterdir()) == []


def test_ensure_daemon_fails_fast_on_early_exit_and_cleans_up_on_interrupt(monkeypatch, tmp_path):
    bad_bin = tmp_path / "kiri-bad"
    bad_bin.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    bad_bin.chmod(0o755)

    start = time.monotonic()
    with pytest.raises(SandboxInfraError, match="exited prematurely"):
        kiri_installer._ensure_daemon(kiri_bin=str(bad_bin), data_dir=tmp_path / "d1", wait_seconds=15.0)
    assert time.monotonic() - start < 3.0

    slow_bin = tmp_path / "kiri-slow"
    slow_bin.write_text("#!/bin/sh\nexec sleep 60\n", encoding="utf-8")
    slow_bin.chmod(0o755)

    spawned: list[int] = []
    orig_popen = kiri_installer.subprocess.Popen

    def _recording_popen(*args, **kwargs):
        p = orig_popen(*args, **kwargs)
        spawned.append(p.pid)
        return p

    sleep_calls = 0
    orig_sleep = time.sleep

    def _interrupting_sleep(sec: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            raise KeyboardInterrupt("stop boot")
        orig_sleep(min(sec, 0.01))

    monkeypatch.setattr(kiri_installer.subprocess, "Popen", _recording_popen)

    with monkeypatch.context() as mp:
        mp.setattr(kiri_installer.time, "sleep", _interrupting_sleep)
        with pytest.raises(KeyboardInterrupt):
            kiri_installer._ensure_daemon(kiri_bin=str(slow_bin), data_dir=tmp_path / "d2", wait_seconds=15.0)
    assert len(spawned) == 1
    # Verify spawned daemon group was terminated.
    deadline = time.monotonic() + 3.0
    alive = True
    while time.monotonic() < deadline:
        try:
            os.killpg(spawned[0], 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.02)
    assert not alive


# ---------------------------------------------------------------------------
# Finding #24: File tools propagate SandboxInfraError cleanly
# ---------------------------------------------------------------------------


def test_file_tools_propagate_sandbox_infra_error_and_swe_repo_detects_it():
    class _DeadSession:
        def read_file(self, path: str, *, max_chars: int | None = None) -> str:
            raise SandboxInfraError("daemon crashed")

        def write_files(self, files) -> None:
            raise SandboxInfraError("daemon crashed")

    class _WriteFailSession:
        def read_file(self, path: str, *, max_chars: int | None = None) -> str:
            return "hello world"

        def write_files(self, files) -> None:
            raise SandboxInfraError("archive upload failed")

    dead = _DeadSession()
    assert SandboxReadFileTool(lambda: dead).call("a.py").startswith("Sandbox infrastructure error:")
    assert SandboxWriteFileTool(lambda: dead).call("a.py", "x").startswith("Sandbox infrastructure error:")
    assert SandboxEditFileTool(lambda: dead).call("a.py", "a", "b").startswith("Sandbox infrastructure error:")
    assert SandboxEditFileTool(lambda: _WriteFailSession()).call("a.py", "hello", "hi").startswith(
        "Sandbox infrastructure error:"
    )

    # End-to-end check with SWERepoEnvironment.step():
    # 1) Agent filesystem errors (writing to an existing directory or FIFO) must return a normal tool error
    #    (NOT env_error=True / episode discard!).
    # 2) A closed/dead session MUST return terminated=True, truncated=True, env_error=True.
    env = SWERepoEnvironment(
        problem_statement="Fix bug",
        files={"src/calc.py": "def f():\n    return 1\n"},
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        env.reset()
        os.mkfifo(os.path.join(env.session.workdir, "pipe"))
        for bad_target in ("src", ".", "pipe"):
            step_res = env.step(
                f'<tool_call>{json.dumps({"name": "write_file", "arguments": {"path": bad_target, "content": "x"}})}</tool_call>'
            )
            assert step_res.terminated is False
            assert step_res.info["env_error"] is False
            assert "Error writing" in step_res.observation

        env.session.close()
        res = env.step('<tool_call>{"name": "read_file", "arguments": {"path": "src/calc.py"}}</tool_call>')
        assert res.terminated is True and res.truncated is True
        assert res.info["env_error"] is True
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Finding #26 (#48): scripts/cli.py hardening, tunnel cleanup, validation, stale socket
# ---------------------------------------------------------------------------


def test_cli_remote_script_has_trap_atomic_mv_https_curl_and_liveness_check():
    script = cli._build_remote_kiri_script(
        deb_url_amd64="https://github.com/libracoder/kiri/releases/download/v0.2.1/kiri-lite_0.2.1_amd64.deb",
        deb_sha_amd64="a" * 64,
        deb_url_arm64="https://github.com/libracoder/kiri/releases/download/v0.2.1/kiri-lite_0.2.1_arm64.deb",
        deb_sha_arm64="b" * 64,
        num_daemons=2,
        data_dir="/dev/shm/kiri",
    )
    assert "trap cleanup EXIT" in script
    assert "--proto '=https' --proto-redir '=https' --tlsv1.2 --max-filesize" in script
    assert 'mv -f "$TMP_BIN" "$HOME/.local/bin/kiri"' in script
    assert "--unix-socket" in script
    assert 'rm -f "/dev/shm/kiri/vm-0/.kiri/run/docker.sock"' in script
    assert "[ ! -S " not in script
    # Confirm generated script passes `bash -n` syntax check.
    proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_cli_remote_script_restarts_daemon_on_stale_socket():
    """Run the generated remote script against a fake HOME with a stale Unix socket."""
    short_root = tempfile.mkdtemp(prefix="edcli-", dir="/tmp")
    try:
        fake_home = os.path.join(short_root, "h")
        bin_dir = os.path.join(fake_home, ".local", "bin")
        run_dir = os.path.join(fake_home, ".kiri", "run")
        os.makedirs(bin_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)

        # Create a stale Unix socket (bound then immediately closed without unlinking).
        sock_path = os.path.join(run_dir, "docker.sock")
        stale_srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale_srv.bind(sock_path)
        stale_srv.close()
        assert os.path.exists(sock_path)

        # Fake `kiri` binary that serves `/_ping` on `$KIRI_DATA_DIR/run/docker.sock` and records boot.
        booted_marker = os.path.join(short_root, "booted")
        fake_kiri = os.path.join(bin_dir, "kiri")
        with open(fake_kiri, "w", encoding="utf-8") as f:
            f.write(
                f"#!/bin/sh\n"
                f"touch {booted_marker}\n"
                f"exec {sys.executable} -c '"
                f'import os, socket, time\n'
                f'p = os.path.join(os.environ["KIRI_DATA_DIR"], "run", "docker.sock")\n'
                f"s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
                f"s.bind(p); s.listen(8); s.settimeout(5.0)\n"
                f"deadline = time.monotonic() + 5.0\n"
                f"while time.monotonic() < deadline:\n"
                f"    try:\n"
                f"        c, _ = s.accept()\n"
                f"    except OSError:\n"
                f"        break\n"
                f"    c.recv(4096)\n"
                f'    c.sendall(b"HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\nConnection: close\\r\\n\\r\\nOK")\n'
                f"    c.close()\n"
                f"'\n"
            )
        os.chmod(fake_kiri, 0o755)

        script = cli._build_remote_kiri_script(
            deb_url_amd64="https://github.com/libracoder/kiri/releases/download/v0.2.1/kiri-lite_0.2.1_amd64.deb",
            deb_sha_amd64="a" * 64,
            deb_url_arm64="https://github.com/libracoder/kiri/releases/download/v0.2.1/kiri-lite_0.2.1_arm64.deb",
            deb_sha_arm64="b" * 64,
            num_daemons=1,
            data_dir=None,
        )
        env = dict(os.environ, HOME=fake_home)
        proc = subprocess.run(["bash"], input=script, text=True, capture_output=True, env=env, timeout=15.0, check=False)
        assert proc.returncode == 0, proc.stderr
        assert "KIRI_CPU_VM_READY" in proc.stdout
        assert os.path.exists(booted_marker), "stale socket caused daemon boot to be skipped"
    finally:
        shutil.rmtree(short_root, ignore_errors=True)


def test_cli_setup_cpu_vm_cleans_up_earlier_tunnels_on_partial_failure(monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(cli.subprocess, "check_output", lambda cmd, text=True: "/home/u\n")
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check=True: calls.append(list(cmd)))
    monkeypatch.setattr(
        cli,
        "resolve_kiri_asset_url",
        lambda **kw: ("v0.2.1", "kiri.deb", "https://github.com/libracoder/kiri/releases/download/v0.2.1/kiri.deb", "a" * 64),
    )

    ping_calls = {"n": 0}

    def _flaky_ping(self) -> bool:
        ping_calls["n"] += 1
        # First daemon answers /_ping; second daemon fails all retries.
        return ping_calls["n"] == 1

    monkeypatch.setattr(cli.EngineSocketClient, "ping", _flaky_ping)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="did not answer /_ping"):
        cli.main(["setup", "cpu-vm", "10.0.0.1", "10.0.0.2", "--port", "2500"])

    # Both opened tunnels (host 1 and host 2 whose ping failed) must be torn down via -O exit.
    exit_calls = [c for c in calls if "-O" in c and "exit" in c]
    assert len(exit_calls) == 2
    assert exit_calls[0][-2:] == ["--", "10.0.0.1"]
    assert exit_calls[1][-2:] == ["--", "10.0.0.2"]


def test_cli_validation_num_daemons_ssh_injection_and_colon_data_dir():
    for bad_n in ("0", "-3"):
        with pytest.raises(ValueError, match="--num-daemons"):
            cli.main(["setup", "kiri", "--num-daemons", bad_n, "--start"])

    # Reject leading '-' in SSH target / user (option injection) and require '--' before target.
    with pytest.raises(ValueError, match="SSH target"):
        cli._ssh_base_cmd(target="-oProxyCommand=id", ssh_key=None)
    with pytest.raises(ValueError, match="SSH target"):
        cli._ssh_base_cmd(target="u@-evil", ssh_key=None)
    with pytest.raises(ValueError, match="--user"):
        cli.main(["setup", "cpu-vm", "10.0.0.1", "--user=-oProxyCommand=id", "--dry-run"])
    with pytest.raises(ValueError, match="SSH target"):
        cli.main(["setup", "cpu-vm", "--dry-run", "--", "-oProxyCommand=id"])
    with pytest.raises(ValueError, match="SSH target"):
        cli.cmd_setup_cpu_vm(
            argparse.Namespace(
                hosts=["-oProxyCommand=id"],
                user=None,
                ssh_key=None,
                port=2375,
                num_daemons=1,
                data_dir=None,
                repo="libracoder/kiri",
                kiri_version="v0.2.1",
                sha256_amd64=None,
                sha256_arm64=None,
                dry_run=True,
            )
        )
    ssh_cmd = cli._ssh_base_cmd(target="u@10.0.0.1", ssh_key=None)
    assert ssh_cmd[-2:] == ["--", "u@10.0.0.1"]

    # Reject ':' in remote --data-dir (would break OpenSSH -L port:sock_path).
    for bad_dir in ("/tmp/a:b", "~/foo:bar"):
        with pytest.raises(ValueError, match="--data-dir"):
            cli.main(["setup", "cpu-vm", "10.0.0.1", "--data-dir", bad_dir, "--dry-run"])


# ---------------------------------------------------------------------------
# Finding #28: KiriRuntime._client reuses EngineSocketClient & _present_images
# ---------------------------------------------------------------------------


def test_kiri_runtime_reuses_client_and_image_cache_across_sessions():
    image_inspects = 0
    counter = 0

    def handler(method, target, headers, body, conn):
        nonlocal image_inspects, counter
        path = target.split("?", 1)[0]
        if path == "/_ping":
            return 200, b"OK"
        if path == "/version":
            return 200, b'{"KiriVersion": "0.2.1"}'
        if method == "GET" and path.startswith("/v1.41/images/") and path.endswith("/json"):
            image_inspects += 1
            return 200, b'{"Id": "sha256:abc"}'
        if method == "POST" and path.endswith("/containers/create"):
            counter += 1
            return 201, json.dumps({"Id": f"cid-{counter}"}).encode()
        if method == "POST" and path.endswith("/start") and "/containers/" in path:
            return 204, b""
        if method == "DELETE" and "/containers/" in path:
            return 204, b""
        if method == "POST" and path.endswith("/exec"):
            return 201, b'{"Id": "e1"}'
        if method == "POST" and path.endswith("/start"):
            return 200, b""
        if method == "GET" and path.endswith("/json"):
            return 200, b'{"Running": false, "ExitCode": 0}'
        return 404, b"{}"

    engine = _UnixHTTPServer(handler)
    try:
        rt = KiriRuntime(
            backend="kiri",
            endpoint=engine.endpoint,
            config=ContainerConfig(image="python:3.11-slim", pull_if_missing=True),
        )
        with rt.create_session() as s1:
            assert isinstance(s1, ContainerSandboxSession)
            c1 = s1._client
        with rt.create_session() as s2:
            assert isinstance(s2, ContainerSandboxSession)
            c2 = s2._client
        assert c1 is c2
        assert image_inspects == 1
        rt.close()
        assert rt._clients == {}
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Finding #31: EngineSocketClient._read_response invalid Content-Length & chunk size -> SandboxInfraError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_response",
    [
        b"HTTP/1.1 200 OK\r\nContent-Length: -5\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Length: notanumber\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Length: +10\r\n\r\n0123456789",
        b"HTTP/1.1 200 OK\r\nContent-Length: 1_0\r\n\r\n0123456789",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-1\r\n",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0x4\r\nabcd\r\n0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1_0\r\n",
    ],
)
def test_read_response_invalid_length_or_chunk_size_raises_sandbox_infra_error(raw_response):
    a, b = socket.socketpair()
    try:
        a.sendall(raw_response)
        a.shutdown(socket.SHUT_WR)
        with pytest.raises(SandboxInfraError):
            EngineSocketClient._read_response(b, method="GET", deadline=time.monotonic() + 2.0)
    finally:
        a.close()
        b.close()


def test_install_kiri_sha256_still_verified_with_bounded_streaming(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        payload = b"#!/bin/sh\necho ok\n"
        info = tarfile.TarInfo(name="Kiri/kiri")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    tar_bytes = buf.getvalue()
    digest = hashlib.sha256(tar_bytes).hexdigest()

    class _StreamResp:
        def __init__(self, data: bytes):
            self._io = io.BytesIO(data)

        def read(self, n: int = -1) -> bytes:
            return self._io.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(kiri_installer.shutil, "which", lambda *_: None)
    monkeypatch.setattr(
        kiri_installer,
        "resolve_kiri_asset_url",
        lambda **kw: ("v0.2.1", "Kiri-0.2.1-x86_64.tar.gz", "https://github.com/libracoder/kiri/releases/download/v0.2.1/Kiri-0.2.1-x86_64.tar.gz", digest),
    )
    monkeypatch.setattr(kiri_installer.urllib.request, "urlopen", lambda req, timeout=60.0: _StreamResp(tar_bytes))
    installed = kiri_installer.install_kiri(install_dir=tmp_path, force=True)
    assert installed == str(tmp_path / "kiri")
    assert os.access(installed, os.X_OK)
