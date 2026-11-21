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

"""Stateful sandbox sessions backed by Kiri containers (default) or local child processes.

The default ``backend="kiri"`` connects directly to Kiri's ~2.5 ms Rust
container engine over its Unix socket via the Docker Engine v1.41 HTTP API,
booting an already-installed daemon when none is running (installing Kiri is an
explicit ``easydel setup kiri`` / ``easydel-install-kiri`` step). Daemons booted
by this process are tracked and stopped by :meth:`KiriRuntime.close`,
:meth:`SandboxPool.close`, ``kiri_installer.shutdown_kiri_daemons`` or at
interpreter exit. ``backend="docker"`` targets any Docker-compatible daemon,
``backend="auto"`` tries Kiri, then Docker, then falls back to local execution,
and ``backend="local"`` runs each episode as child processes inside a
persistent per-episode temporary workspace jailed with ``bwrap``; running local
commands unjailed requires an explicit ``ContainerConfig(use_jail="off")``.
"""

from __future__ import annotations

import abc
import asyncio
import collections.abc
import dataclasses
import hashlib
import io
import json
import math
import os
import posixpath
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import typing as tp
import urllib.parse
import uuid
from pathlib import Path

from easydel.trainers._shared.sandbox import SandboxResult

_POSIX = os.name == "posix"
_CPU_BUDGET_RETURNCODE: int | None = -signal.SIGXCPU if _POSIX else None
_SAFE_ENV_KEYS = ("LANG", "LC_ALL", "TZ")
_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_IDLE_CONTAINER_CMD = ["/bin/sh", "-c", "trap 'exit 0' TERM INT; while :; do sleep 3600; done"]
_TERM_GRACE_S = 0.2
_JAIL_POLICIES = ("auto", "required", "off")

SANDBOX_TOKEN_HEADER = "X-EasyDeL-Sandbox-Token"
"""HTTP header carrying the shared secret that authenticates clients to a cross-host sandbox bridge."""

# Wraps every container exec. Arguments: pid file, backstop seconds, stdin file ("" for none), argv.
# The command runs in a background subshell that alone opens the stdin file, unlinks it and execs the
# command (through ``setsid`` when available, so it leads its own process group): the parent shell
# and the backstop killer never hold the stdin file, so no other process exposes it via /proc/<pid>/fd.
# The command pid is recorded for the client's kill path, and the SIGKILL backstop is a sibling
# (never an ancestor) whose output is detached so it cannot hold the exec stream open.
_EXEC_WATCHDOG_SCRIPT = (
    "f=$1; t=$2; s=$3; shift 3; g=; if command -v setsid >/dev/null 2>&1; then g=setsid; fi; "
    '( if [ -n "$s" ]; then exec <"$s" || exit 126; rm -f "$s" </dev/null; fi; exec $g "$@" ) & p=$!; '
    'echo "$p" > "$f" 2>/dev/null; '
    '( trap \'kill "$w" 2>/dev/null; exit 0\' TERM; sleep "$t" & w=$!; '
    'wait "$w" && { kill -KILL -"$p"; kill -KILL "$p"; } ) </dev/null >/dev/null 2>&1 & k=$!; '
    'wait "$p"; r=$?; kill "$k" 2>/dev/null; exit "$r"'
)
_KILL_ALL_SCRIPT = "kill -9 -1 2>/dev/null; exit 0"
_KILL_ALL_WAIT_S = 2.0
_EXEC_KILL_SCRIPT = (
    'p=$(cat "$1" 2>/dev/null) || exit 0; kill -KILL -"$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; rm -f "$1"'
)
_TAR_READ_SLACK_BYTES = 16 * 1024


class SandboxInfraError(RuntimeError):
    """Raised when the underlying container or sandbox infrastructure fails."""


def _signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to process group ``pgid``; return ``False`` when the group is gone or not ours."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _terminate_process_groups(pgids: collections.abc.Iterable[int], *, grace: float = _TERM_GRACE_S) -> None:
    """SIGTERM every group in ``pgids``, then SIGKILL whatever survives ``grace`` seconds.

    The caller's own process group is never signalled.
    """
    if not _POSIX:
        return
    own = os.getpgrp()
    live = [g for g in pgids if g > 0 and g != own and _signal_group(g, signal.SIGTERM)]
    deadline = time.monotonic() + grace
    while live and time.monotonic() < deadline:
        time.sleep(0.01)
        live = [g for g in live if _signal_group(g, 0)]
    for g in live:
        _signal_group(g, signal.SIGKILL)


@dataclasses.dataclass(frozen=True)
class ExecResult:
    """Outcome of executing one command inside a stateful sandbox session.

    Attributes:
        ok: ``True`` when the command exited with code 0 without timing out
            or encountering an infrastructure error.
        stdout: Captured standard output (truncated to ``max_output_chars``).
        stderr: Captured standard error (truncated to ``max_output_chars``).
        returncode: Process exit status, or ``None`` on timeout/infra failure.
        timed_out: ``True`` when execution exceeded its time budget.
        infra_error: Non-empty message when the container/daemon failed rather
            than the user command itself.
    """

    ok: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False
    infra_error: str | None = None

    def to_sandbox_result(self) -> SandboxResult:
        """Convert to :class:`easydel.trainers._shared.sandbox.SandboxResult`."""
        err = f"{self.stderr}\n{self.infra_error}".strip() if self.infra_error else self.stderr
        return SandboxResult(
            ok=self.ok and self.infra_error is None,
            stdout=self.stdout,
            stderr=err,
            returncode=self.returncode,
            timed_out=self.timed_out,
        )

    def combined_output(self, max_chars: int | None = None) -> str:
        """Format stdout, stderr, and exit metadata for tool observations."""
        if self.infra_error:
            text = f"Sandbox infrastructure error: {self.infra_error}"
        elif self.timed_out:
            text = "Error: command timed out."
            if self.stdout:
                text = f"{self.stdout}\n{text}"
        else:
            parts: list[str] = []
            if self.stdout:
                parts.append(self.stdout)
            if self.stderr:
                parts.append(f"[stderr]\n{self.stderr}")
            if self.returncode not in (0, None):
                parts.append(f"[exit code: {self.returncode}]")
            text = "\n".join(parts) if parts else "(no output)"
        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + "\n... (output truncated)"
        return text


@dataclasses.dataclass
class ContainerConfig:
    """Resource, filesystem, and security configuration for one sandbox episode.

    Attributes:
        image: OCI image name used when running on Kiri or Docker backends.
        workdir: Default working directory inside a container session.
        memory_mb: Hard memory limit in MiB (0 disables cgroup/rlimit cap).
        cpus: Fractional CPU quota per container (e.g. ``1.0`` = one core).
        pids_limit: Maximum concurrent processes inside a container.
        network: Allow outbound network access. ``None`` (default) disables the
            network wherever the backend can enforce it; an explicit ``False``
            demands isolation and makes an unjailed local session raise.
        read_only_rootfs: Mount the container root filesystem read-only. The
            local backend enforces it only inside the ``bwrap`` jail and raises
            when it is requested with ``use_jail="off"``.
        tmpfs_mb: Size in MiB for ephemeral ``/tmp`` tmpfs mount (0 disables).
        workspace_tmpfs_mb: Optional tmpfs size in MiB for ``workdir`` so dense
            file writes stay in RAM and never hit host disks.
        timeout: Default per-command execution timeout in seconds.
        max_output_chars: Maximum characters retained per output stream.
        env: Extra environment variables injected into every command.
        binds: Volume bind specs (``host:container:ro``) for container backends.
        pull_if_missing: Pull ``image`` automatically when absent locally.
        use_jail: Jail policy for the local backend. ``"auto"`` and
            ``"required"`` both jail commands with ``bwrap`` and raise
            :class:`SandboxInfraError` when it is unavailable; only an explicit
            ``"off"`` runs model commands unjailed.
        max_stdin_bytes: Largest stdin payload (UTF-8 bytes) accepted per command.
        max_file_size_mb: ``RLIMIT_FSIZE`` cap in MiB for files written by local
            commands (0 disables the cap).
    """

    image: str = "python:3.11-slim"
    workdir: str = "/workspace"
    memory_mb: int = 2048
    cpus: float = 1.0
    pids_limit: int = 256
    network: bool | None = None
    read_only_rootfs: bool = False
    tmpfs_mb: int = 256
    workspace_tmpfs_mb: int = 0
    timeout: float = 30.0
    max_output_chars: int = 16384
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    binds: list[str] = dataclasses.field(default_factory=list)
    pull_if_missing: bool = True
    use_jail: str = "auto"
    max_stdin_bytes: int = 16 * 1024 * 1024
    max_file_size_mb: int = 1024


@dataclasses.dataclass(frozen=True)
class KiriDaemonBudget:
    """Per-daemon resource budget for RAM-backed (``/dev/shm``) Kiri pools.

    Containers are clamped to the budget in two places: by
    :class:`EngineSocketClient` in the process that booted the daemon, and by
    that process's TCP bridges for every container-create request they forward
    (so remote hosts of a pod pool are clamped too). A client that talks to the
    daemon's Unix socket directly from another process is not clamped. The
    defaults are deliberately conservative; raise them explicitly for
    heavyweight tasks.

    Attributes:
        cpus: Maximum CPU quota (``NanoCpus``) of any one container.
        memory_mb: Maximum memory limit in MiB of any one container.
        pids_limit: Maximum ``PidsLimit`` of any one container.
        tmpfs_mb: RAM-backed storage in MiB the daemon may claim under its data
            directory (free space is checked before the daemon boots) and the
            largest tmpfs mount any one container may request.
    """

    cpus: float = 4.0
    memory_mb: int = 8192
    pids_limit: int = 1024
    tmpfs_mb: int = 4096

    def clamp_host_config(self, host_config: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Return ``host_config`` with CPU, memory, pids and tmpfs limits capped by this budget.

        A missing, zero or negative limit (unlimited for the engine) is replaced by the cap.

        Raises:
            ValueError: When a limit is not an integer.
        """
        out = dict(host_config)

        def _capped(key: str, cap: int) -> int:
            value = int(out.get(key) or 0)
            return cap if value <= 0 else min(value, cap)

        if self.memory_mb > 0:
            out["Memory"] = _capped("Memory", int(self.memory_mb) * 1024 * 1024)
        if self.cpus > 0:
            out["NanoCpus"] = _capped("NanoCpus", int(float(self.cpus) * 1_000_000_000))
        if self.pids_limit > 0:
            out["PidsLimit"] = _capped("PidsLimit", int(self.pids_limit))
        tmpfs = out.get("Tmpfs")
        if isinstance(tmpfs, dict) and self.tmpfs_mb > 0:
            capped: dict[str, str] = {}
            for mount, opts in tmpfs.items():
                parts = [p for p in str(opts).split(",") if p and not p.startswith("size=")]
                size_mb = self.tmpfs_mb
                for p in str(opts).split(","):
                    if p.startswith("size=") and p.endswith("m") and p[5:-1].isdigit():
                        size_mb = min(int(p[5:-1]), self.tmpfs_mb)
                capped[mount] = ",".join([*parts, f"size={size_mb}m"])
            out["Tmpfs"] = capped
        return out


@dataclasses.dataclass(frozen=True)
class _HTTPResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    truncated: bool = False


class _DeadlineReader:
    """Buffered socket reader that enforces one monotonic deadline across every ``recv``."""

    def __init__(self, sock: socket.socket, deadline: float):
        self._sock = sock
        self._deadline = deadline
        self._buf = bytearray()
        self._eof = False

    def _fill(self) -> bool:
        if self._eof:
            return False
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("container engine response exceeded its deadline")
        self._sock.settimeout(remaining)
        chunk = self._sock.recv(65536)
        if not chunk:
            self._eof = True
            return False
        self._buf += chunk
        return True

    def readline(self, limit: int = 65536) -> bytes:
        while True:
            idx = self._buf.find(b"\n", 0, limit)
            if idx >= 0:
                return self._take(idx + 1)
            if len(self._buf) >= limit or not self._fill():
                return self._take(min(limit, len(self._buf)))

    def read(self, n: int) -> bytes:
        """Return ``n`` bytes, or fewer only at end of stream."""
        while len(self._buf) < n and self._fill():
            pass
        return self._take(min(n, len(self._buf)))

    def _take(self, n: int) -> bytes:
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


def _parse_endpoint(endpoint: str) -> tuple[str, str | tuple[str, int]]:
    if endpoint.startswith("unix://"):
        return "unix", endpoint[len("unix://") :]
    if endpoint.startswith(("http://", "tcp://")):
        parsed = urllib.parse.urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 2375
        return "tcp", (host, port)
    return "unix", endpoint


def _sh_quote(arg: str) -> str:
    return "'" + arg.replace("'", "'\"'\"'") + "'"


def _demux_docker_stream(payload: bytes, max_chars: int) -> tuple[str, str]:
    """Decode Docker/Kiri 8-byte multiplexed stdout/stderr frames with raw fallback."""
    if not payload:
        return "", ""
    mv = memoryview(payload)
    pos = 0
    total = len(mv)
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    out_bytes = 0
    err_bytes = 0
    byte_cap = max(max_chars * 4, 4096)

    while pos + 8 <= total:
        stream_type = mv[pos]
        if stream_type not in (0, 1, 2) or mv[pos + 1] != 0 or mv[pos + 2] != 0 or mv[pos + 3] != 0:
            text = bytes(mv).decode("utf-8", errors="replace")
            return text[:max_chars], ""
        frame_len = struct.unpack(">I", mv[pos + 4 : pos + 8])[0]
        pos += 8
        if pos + frame_len > total:
            frame = bytes(mv[pos:total])
            pos = total
        else:
            frame = bytes(mv[pos : pos + frame_len])
            pos += frame_len
        if stream_type == 1 and out_bytes < byte_cap:
            keep = frame[: byte_cap - out_bytes]
            out_chunks.append(keep)
            out_bytes += len(keep)
        elif stream_type == 2 and err_bytes < byte_cap:
            keep = frame[: byte_cap - err_bytes]
            err_chunks.append(keep)
            err_bytes += len(keep)

    if pos != total and not out_chunks and not err_chunks:
        text = payload.decode("utf-8", errors="replace")
        return text[:max_chars], ""

    stdout = b"".join(out_chunks).decode("utf-8", errors="replace")[:max_chars]
    stderr = b"".join(err_chunks).decode("utf-8", errors="replace")[:max_chars]
    return stdout, stderr


def _first_regular_tar_member(archive: bytes, max_bytes: int | None) -> bytes | None:
    """Return the data of the first regular-file member of a (possibly truncated) ustar/pax archive.

    Parses tar headers directly so an archive cut short by a body cap still yields
    the leading ``max_bytes`` bytes of the file, which :mod:`tarfile` would reject.
    """
    pos = 0
    total = len(archive)
    while pos + 512 <= total:
        header = archive[pos : pos + 512]
        if header == b"\0" * 512:
            return None
        size_field = header[124:136].split(b"\0", 1)[0].strip()
        try:
            size = int(size_field or b"0", 8)
        except ValueError:
            return None
        typeflag = header[156:157]
        data_start = pos + 512
        if typeflag in (b"0", b"\0", b"7"):
            keep = size if max_bytes is None else min(size, max(0, int(max_bytes)))
            return bytes(archive[data_start : data_start + keep])
        # Skip pax/GNU extension headers, directories, links and other non-file members.
        pos = data_start + ((size + 511) // 512) * 512
    return None


class EngineSocketClient:
    """Minimal HTTP/1.1 client for Kiri and Docker Engine v1.41 sockets."""

    api_prefix: str = "/v1.41"

    def __init__(
        self,
        endpoint: str,
        *,
        connect_timeout: float = 10.0,
        token: str | None = None,
        budget: KiriDaemonBudget | None = None,
    ):
        """Create a client for ``endpoint``.

        Args:
            endpoint: ``unix://``, ``http://`` or ``tcp://`` engine address.
            connect_timeout: Socket connect timeout in seconds.
            token: Shared secret sent as :data:`SANDBOX_TOKEN_HEADER` on every
                request (required by authenticated cross-host bridges).
            budget: Optional per-daemon resource budget; container creates are
                clamped to it.
        """
        self.endpoint = endpoint
        self._kind, self._addr = _parse_endpoint(endpoint)
        self._connect_timeout = float(connect_timeout)
        self._token = token
        self.budget = budget
        self._present_images: set[str] = set()
        self._images_lock = threading.Lock()

    def _connect(self, timeout: float) -> socket.socket:
        if self._kind == "unix":
            if not hasattr(socket, "AF_UNIX"):
                raise SandboxInfraError("Unix domain sockets are not supported on this platform.")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(tp.cast(str, self._addr))
            return sock
        addr = tp.cast(tuple[str, int], self._addr)
        sock = socket.create_connection(addr, timeout=timeout)
        sock.settimeout(timeout)
        return sock

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | dict[str, tp.Any] | None = None,
        content_type: str = "application/json",
        extra_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_body_bytes: int | None = None,
    ) -> _HTTPResponse:
        if isinstance(body, dict):
            payload = json.dumps(body).encode("utf-8")
        elif body is None:
            payload = b""
        else:
            payload = body

        headers = [
            f"{method} {path} HTTP/1.1",
            "Host: kiri",
            "Connection: close",
            f"Content-Length: {len(payload)}",
        ]
        if payload or method in ("POST", "PUT"):
            headers.append(f"Content-Type: {content_type}")
        if self._token:
            headers.append(f"{SANDBOX_TOKEN_HEADER}: {self._token}")
        if extra_headers:
            for k, v in extra_headers.items():
                headers.append(f"{k}: {v}")
        raw_head = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii")

        deadline = time.monotonic() + max(0.0, float(timeout))
        try:
            sock = self._connect(max(0.01, min(self._connect_timeout, timeout)))
        except OSError as exc:
            raise SandboxInfraError(f"Failed to connect to container engine at {self.endpoint}: {exc}") from exc

        try:
            sock.settimeout(max(0.01, deadline - time.monotonic()))
            sock.sendall(raw_head)
            if payload:
                sock.sendall(payload)
            return self._read_response(sock, method=method, max_body_bytes=max_body_bytes, deadline=deadline)
        except TimeoutError:
            raise
        except OSError as exc:
            raise SandboxInfraError(f"Container engine I/O error on {method} {path}: {exc}") from exc
        finally:
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _read_response(
        sock: socket.socket,
        *,
        method: str,
        max_body_bytes: int | None = None,
        deadline: float | None = None,
    ) -> _HTTPResponse:
        """Read one HTTP response, enforcing ``deadline`` across all reads and capping the body.

        Reading stops as soon as ``max_body_bytes`` body bytes are collected (the
        response is then flagged ``truncated``); the remainder is never pulled
        off the socket, so a peer streaming forever cannot hold the caller.
        """
        reader = _DeadlineReader(sock, deadline if deadline is not None else time.monotonic() + 30.0)
        status_line = reader.readline(65536)
        if not status_line:
            raise SandboxInfraError("Empty HTTP status line from container engine.")
        parts = status_line.decode("latin-1", errors="replace").strip().split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise SandboxInfraError(f"Malformed HTTP status line: {status_line!r}")
        status = int(parts[1])

        resp_headers: dict[str, str] = {}
        while True:
            line = reader.readline(65536)
            if not line or line in (b"\r\n", b"\n"):
                break
            decoded = line.decode("latin-1", errors="replace").rstrip("\r\n")
            if ":" in decoded:
                k, v = decoded.split(":", 1)
                resp_headers[k.strip().lower()] = v.strip()

        # Kiri keeps 204/304 connections open without a body; return immediately.
        if status in (204, 304) or method == "HEAD":
            return _HTTPResponse(status=status, headers=resp_headers, body=b"")

        cap = max_body_bytes
        transfer_encoding = resp_headers.get("transfer-encoding", "").lower()
        if "chunked" in transfer_encoding:
            chunks: list[bytes] = []
            collected = 0
            while True:
                size_line = reader.readline(65536)
                if not size_line:
                    break
                token = size_line.strip().split(b";", 1)[0]
                if not token:
                    continue
                chunk_len = int(token, 16)
                if chunk_len == 0:
                    reader.readline(65536)
                    break
                if cap is not None and collected + chunk_len > cap:
                    keep = reader.read(cap - collected)
                    chunks.append(keep)
                    return _HTTPResponse(status=status, headers=resp_headers, body=b"".join(chunks), truncated=True)
                data = reader.read(chunk_len)
                reader.read(2)
                chunks.append(data)
                collected += len(data)
            return _HTTPResponse(status=status, headers=resp_headers, body=b"".join(chunks))

        if "content-length" in resp_headers:
            length = int(resp_headers["content-length"])
            if cap is not None and length > cap:
                return _HTTPResponse(status=status, headers=resp_headers, body=reader.read(cap), truncated=True)
            return _HTTPResponse(status=status, headers=resp_headers, body=reader.read(length))

        # Read-until-close body: stop at the cap rather than draining a stream that may never end.
        chunks = []
        collected = 0
        while cap is None or collected < cap:
            want = 65536 if cap is None else min(65536, cap - collected)
            piece = reader.read(want)
            if not piece:
                return _HTTPResponse(status=status, headers=resp_headers, body=b"".join(chunks))
            chunks.append(piece)
            collected += len(piece)
        return _HTTPResponse(status=status, headers=resp_headers, body=b"".join(chunks), truncated=True)

    @staticmethod
    def _error_message(resp: _HTTPResponse) -> str:
        if resp.body:
            try:
                parsed = json.loads(resp.body.decode("utf-8", errors="replace"))
                if isinstance(parsed, dict) and "message" in parsed:
                    return str(parsed["message"])
            except json.JSONDecodeError:
                pass
            return resp.body.decode("utf-8", errors="replace").strip()
        return f"HTTP {resp.status}"

    def ping(self, timeout: float = 2.0) -> bool:
        """Return ``True`` when the engine responds to ``/_ping``."""
        try:
            resp = self._request("GET", "/_ping", extra_headers={"X-Kiri-Background": "1"}, timeout=timeout)
            return 200 <= resp.status < 300
        except Exception:
            return False

    def version(self, timeout: float = 5.0) -> dict[str, tp.Any]:
        """Return the ``/version`` payload from Kiri or Docker."""
        resp = self._request("GET", "/version", extra_headers={"X-Kiri-Background": "1"}, timeout=timeout)
        if not (200 <= resp.status < 300):
            raise SandboxInfraError(f"Engine /version failed: {self._error_message(resp)}")
        return json.loads(resp.body.decode("utf-8", errors="replace"))

    def engine_flavor(self) -> str:
        """Identify whether the connected daemon is ``'kiri'`` or ``'docker'``."""
        info = self.version()
        if "KiriVersion" in info or str(info.get("Platform", {}).get("Name", "")).lower().startswith("kiri"):
            return "kiri"
        return "docker"

    def ensure_image(self, image: str, *, pull: bool = True, timeout: float = 300.0) -> None:
        """Verify ``image`` exists locally, pulling it when ``pull=True``.

        Images confirmed present are cached per client, so repeated episode
        creation skips the inspect round trip.
        """
        with self._images_lock:
            if image in self._present_images:
                return
        encoded = urllib.parse.quote(image, safe="")
        resp = self._request("GET", f"{self.api_prefix}/images/{encoded}/json", timeout=10.0)
        if not (200 <= resp.status < 300):
            if resp.status != 404 or not pull:
                raise SandboxInfraError(f"Image '{image}' unavailable: {self._error_message(resp)}")
            pull_resp = self._request("POST", f"{self.api_prefix}/images/create?fromImage={encoded}", timeout=timeout)
            if not (200 <= pull_resp.status < 300):
                raise SandboxInfraError(f"Failed to pull image '{image}': {self._error_message(pull_resp)}")
        with self._images_lock:
            self._present_images.add(image)

    def container_gateway_ip(self, container_id: str) -> str | None:
        """Return the network gateway IP of ``container_id`` (the host bridge address), if any."""
        resp = self._request("GET", f"{self.api_prefix}/containers/{container_id}/json", timeout=10.0)
        if not (200 <= resp.status < 300):
            return None
        try:
            meta = json.loads(resp.body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return None
        settings = meta.get("NetworkSettings") if isinstance(meta, dict) else None
        if not isinstance(settings, dict):
            return None
        if settings.get("Gateway"):
            return str(settings["Gateway"])
        networks = settings.get("Networks")
        if isinstance(networks, dict):
            for net in networks.values():
                if isinstance(net, dict) and net.get("Gateway"):
                    return str(net["Gateway"])
        return None

    def create_container(self, config: ContainerConfig, *, name: str | None = None) -> str:
        """Create an isolated idle container and return its ID."""
        if config.pull_if_missing:
            self.ensure_image(config.image, pull=True)

        tmpfs: dict[str, str] = {}
        if config.tmpfs_mb > 0:
            tmpfs["/tmp"] = f"rw,exec,nosuid,size={config.tmpfs_mb}m"
        if config.workspace_tmpfs_mb > 0:
            tmpfs[config.workdir] = f"rw,exec,nosuid,size={config.workspace_tmpfs_mb}m"

        host_config: dict[str, tp.Any] = {
            "NetworkMode": "bridge" if config.network else "none",
            "ReadonlyRootfs": bool(config.read_only_rootfs),
            "SecurityOpt": ["no-new-privileges"],
        }
        if config.memory_mb > 0:
            host_config["Memory"] = int(config.memory_mb) * 1024 * 1024
        if config.cpus > 0:
            host_config["NanoCpus"] = int(float(config.cpus) * 1_000_000_000)
        if config.pids_limit > 0:
            host_config["PidsLimit"] = int(config.pids_limit)
        if tmpfs:
            host_config["Tmpfs"] = tmpfs
        if config.binds:
            host_config["Binds"] = list(config.binds)
        if self.budget is not None:
            host_config = self.budget.clamp_host_config(host_config)

        payload: dict[str, tp.Any] = {
            "Image": config.image,
            "Cmd": list(_IDLE_CONTAINER_CMD),
            "WorkingDir": config.workdir,
            "Env": [f"{k}={v}" for k, v in config.env.items()],
            "Tty": False,
            "HostConfig": host_config,
        }
        query = f"?name={urllib.parse.quote(name)}" if name else ""
        resp = self._request("POST", f"{self.api_prefix}/containers/create{query}", body=payload, timeout=30.0)
        if resp.status != 201:
            raise SandboxInfraError(f"Container create failed: {self._error_message(resp)}")
        cid = json.loads(resp.body.decode("utf-8")).get("Id")
        if not cid:
            raise SandboxInfraError("Container create returned no Id.")
        return str(cid)

    def start_container(self, container_id: str) -> None:
        """Start a created container."""
        resp = self._request("POST", f"{self.api_prefix}/containers/{container_id}/start", timeout=30.0)
        if resp.status not in (204, 304):
            raise SandboxInfraError(f"Container start failed: {self._error_message(resp)}")

    def remove_container(self, container_id: str, *, force: bool = True) -> None:
        """Stop and remove a container, ignoring 404 if already gone."""
        flag = "true" if force else "false"
        resp = self._request("DELETE", f"{self.api_prefix}/containers/{container_id}?force={flag}&v=true", timeout=15.0)
        if resp.status not in (200, 204, 404):
            raise SandboxInfraError(f"Container remove failed: {self._error_message(resp)}")

    def put_files(
        self,
        container_id: str,
        files: collections.abc.Mapping[str, bytes | str],
        *,
        base_dir: str = "/workspace",
    ) -> None:
        """Upload in-memory files into ``container_id`` via tar archive."""
        if not files:
            return
        buf = io.BytesIO()
        now = int(time.time())
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for rel_path, content in files.items():
                raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
                clean = rel_path.lstrip("/") if not posixpath.isabs(rel_path) else posixpath.relpath(rel_path, base_dir)
                clean = posixpath.normpath(clean)
                if clean == ".." or clean.startswith("../") or posixpath.isabs(clean):
                    raise ValueError(f"File path '{rel_path}' escapes base_dir '{base_dir}'.")
                info = tarfile.TarInfo(name=clean)
                info.size = len(raw)
                info.mtime = now
                info.mode = 0o644
                tf.addfile(info, io.BytesIO(raw))
        encoded_path = urllib.parse.quote(base_dir, safe="/")
        resp = self._request(
            "PUT",
            f"{self.api_prefix}/containers/{container_id}/archive?path={encoded_path}",
            body=buf.getvalue(),
            content_type="application/x-tar",
            timeout=30.0,
        )
        if not (200 <= resp.status < 300):
            raise SandboxInfraError(f"Archive upload failed: {self._error_message(resp)}")

    def get_file(self, container_id: str, path: str, *, max_bytes: int | None = None) -> bytes:
        """Download a single file from ``container_id`` via tar archive.

        Args:
            container_id: Target container.
            path: Absolute path of the file inside the container.
            max_bytes: When set, at most this many bytes of the file are
                returned and the archive download itself stops after
                ``max_bytes`` plus a small tar-header margin.

        Returns:
            The (possibly truncated) file contents.

        Raises:
            FileNotFoundError: When the path is missing or is not a regular file.
        """
        encoded_path = urllib.parse.quote(path, safe="/")
        body_cap = None if max_bytes is None else max(0, int(max_bytes)) + _TAR_READ_SLACK_BYTES
        resp = self._request(
            "GET",
            f"{self.api_prefix}/containers/{container_id}/archive?path={encoded_path}",
            timeout=30.0,
            max_body_bytes=body_cap,
        )
        if not (200 <= resp.status < 300):
            raise FileNotFoundError(f"Cannot read '{path}' from container: {self._error_message(resp)}")
        data = _first_regular_tar_member(resp.body, max_bytes)
        if data is None:
            raise FileNotFoundError(f"Archive for '{path}' contained no regular file.")
        return data

    def exec_run(
        self,
        container_id: str,
        cmd: list[str],
        *,
        workdir: str | None = None,
        env: collections.abc.Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout: float = 30.0,
        max_output_chars: int = 16384,
        max_stdin_bytes: int = 16 * 1024 * 1024,
    ) -> ExecResult:
        """Execute ``cmd`` inside a running container and return captured output.

        ``timeout`` is one monotonic deadline for the whole call (not a per-read
        idle timeout). When it expires the command's process group is SIGKILLed
        inside the container and ``timed_out=True`` is reported. Output is read
        only up to a byte cap derived from ``max_output_chars``; a command that
        exceeds it is killed and reported as failed with partial output.

        Args:
            container_id: Running container to execute in.
            cmd: Argv of the command.
            workdir: Working directory inside the container.
            env: Extra environment variables.
            stdin: Optional text fed to the command's standard input.
            timeout: Overall wall-clock budget in seconds.
            max_output_chars: Characters kept per output stream.
            max_stdin_bytes: Largest accepted UTF-8 ``stdin`` payload.

        Returns:
            The :class:`ExecResult` of the command.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        if stdin is not None and len(stdin.encode("utf-8", errors="replace")) > max_stdin_bytes:
            return ExecResult(
                ok=False,
                stdout="",
                stderr=f"stdin payload exceeds max_stdin_bytes={max_stdin_bytes}.",
                returncode=None,
            )
        stdin_path: str | None = None
        killed = False
        pid_path = f"/tmp/.ed_pid_{uuid.uuid4().hex}"
        if stdin is not None:
            stdin_name = f".ed_stdin_{uuid.uuid4().hex}"
            stdin_path = f"/tmp/{stdin_name}"
            self.put_files(container_id, {stdin_name: stdin}, base_dir="/tmp")
        # The in-container SIGKILL backstop fires a little after the client deadline,
        # so the client normally observes the timeout first and kills via the pid file.
        backstop_s = max(1, math.ceil(float(timeout))) + 2
        exec_cmd = [
            "/bin/sh",
            "-c",
            _EXEC_WATCHDOG_SCRIPT,
            "sh",
            pid_path,
            str(backstop_s),
            stdin_path or "",
            *cmd,
        ]

        create_payload: dict[str, tp.Any] = {
            "AttachStdout": True,
            "AttachStderr": True,
            "AttachStdin": False,
            "Tty": False,
            "Cmd": exec_cmd,
        }
        if workdir:
            create_payload["WorkingDir"] = workdir
        if env:
            create_payload["Env"] = [f"{k}={v}" for k, v in env.items()]

        try:
            created = self._request(
                "POST",
                f"{self.api_prefix}/containers/{container_id}/exec",
                body=create_payload,
                timeout=15.0,
            )
            if not (200 <= created.status < 300):
                return ExecResult(
                    ok=False,
                    stdout="",
                    stderr="",
                    returncode=None,
                    infra_error=f"exec create failed: {self._error_message(created)}",
                )
            exec_id = json.loads(created.body.decode("utf-8")).get("Id")
            if not exec_id:
                return ExecResult(
                    ok=False,
                    stdout="",
                    stderr="",
                    returncode=None,
                    infra_error="exec create returned no Id.",
                )

            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError("exec deadline expired before start")
                started = self._request(
                    "POST",
                    f"{self.api_prefix}/exec/{exec_id}/start",
                    body={"Detach": False, "Tty": False},
                    timeout=remaining,
                    max_body_bytes=max(max_output_chars * 8, 65536),
                )
            except TimeoutError:
                killed = True
                self._kill_exec(container_id, pid_path)
                return ExecResult(ok=False, stdout="", stderr="", returncode=None, timed_out=True)

            if not (200 <= started.status < 300):
                return ExecResult(
                    ok=False,
                    stdout="",
                    stderr="",
                    returncode=None,
                    infra_error=f"exec start failed: {self._error_message(started)}",
                )

            stdout, stderr = _demux_docker_stream(started.body, max_output_chars)
            if getattr(started, "truncated", False):
                killed = True
                self._kill_exec(container_id, pid_path)
                note = "[output limit exceeded; command killed]"
                return ExecResult(
                    ok=False,
                    stdout=stdout,
                    stderr=f"{stderr}\n{note}" if stderr else note,
                    returncode=None,
                )
            inspected = self._request("GET", f"{self.api_prefix}/exec/{exec_id}/json", timeout=10.0)
            if not (200 <= inspected.status < 300):
                return ExecResult(
                    ok=False,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=None,
                    infra_error=f"exec inspect failed: {self._error_message(inspected)}",
                )
            try:
                meta = json.loads(inspected.body.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                return ExecResult(
                    ok=False,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=None,
                    infra_error=f"exec inspect returned invalid JSON: {exc}",
                )
            if not isinstance(meta, dict) or bool(meta.get("Running", False)) or meta.get("ExitCode") is None:
                return ExecResult(
                    ok=False,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=None,
                    infra_error="exec inspect returned null ExitCode or still-running process.",
                )
            exit_code = int(meta["ExitCode"])
            return ExecResult(ok=(exit_code == 0), stdout=stdout, stderr=stderr, returncode=exit_code)
        except (SandboxInfraError, TimeoutError) as exc:
            return ExecResult(ok=False, stdout="", stderr="", returncode=None, infra_error=str(exc))
        finally:
            # After a kill the kill script removes the pid file itself; removing it here
            # concurrently could race the kill and leave the command running.
            leftovers = [p for p in (stdin_path, None if killed else pid_path) if p]
            if leftovers:
                try:
                    self._cleanup_temp(container_id, *leftovers)
                except Exception:
                    pass

    def _kill_exec(self, container_id: str, pid_path: str) -> None:
        """SIGKILL the process group recorded in ``pid_path`` by the exec watchdog (best effort)."""
        try:
            self._detached_exec(container_id, ["/bin/sh", "-c", _EXEC_KILL_SCRIPT, "sh", pid_path])
        except Exception:
            pass

    def _cleanup_temp(self, container_id: str, *paths: str) -> None:
        self._detached_exec(container_id, ["rm", "-f", *paths])

    def _detached_exec(self, container_id: str, cmd: list[str], *, user: str | None = None) -> str | None:
        """Start ``cmd`` detached in ``container_id`` and return the exec id (``None`` if not created)."""
        payload: dict[str, tp.Any] = {"AttachStdout": False, "AttachStderr": False, "Tty": False, "Cmd": cmd}
        if user is not None:
            payload["User"] = user
        created = self._request(
            "POST",
            f"{self.api_prefix}/containers/{container_id}/exec",
            body=payload,
            timeout=5.0,
        )
        if not (200 <= created.status < 300):
            return None
        exec_id = json.loads(created.body.decode("utf-8")).get("Id")
        if not exec_id:
            return None
        started = self._request(
            "POST",
            f"{self.api_prefix}/exec/{exec_id}/start",
            body={"Detach": True, "Tty": False},
            timeout=5.0,
        )
        return str(exec_id) if 200 <= started.status < 300 else None

    def kill_all_processes(self, container_id: str, *, wait: float = _KILL_ALL_WAIT_S) -> None:
        """SIGKILL every process in ``container_id`` except its pid 1, as root.

        Runs POSIX ``kill -9 -1`` in a detached root exec (the signalling shell
        and pid 1 are spared), then waits up to ``wait`` seconds for the exec to
        finish. Only processes in the container's pid namespace are reachable.

        Raises:
            SandboxInfraError: When the kill exec cannot be created or started.
        """
        exec_id = self._detached_exec(container_id, ["/bin/sh", "-c", _KILL_ALL_SCRIPT], user="root")
        if exec_id is None:
            raise SandboxInfraError(f"Could not start the process-kill exec in container {container_id}.")
        deadline = time.monotonic() + max(0.0, float(wait))
        while time.monotonic() < deadline:
            try:
                resp = self._request("GET", f"{self.api_prefix}/exec/{exec_id}/json", timeout=2.0)
                meta = json.loads(resp.body.decode("utf-8", errors="replace")) if 200 <= resp.status < 300 else {}
            except (SandboxInfraError, TimeoutError, json.JSONDecodeError):
                meta = {}
            if isinstance(meta, dict) and meta.get("Running") is False:
                return
            time.sleep(0.05)


_PRLIMIT_BIN: str | None = shutil.which("prlimit") if _POSIX else None
_POST_EXEC_RLIMIT_CODE = (
    "import os, resource, signal, sys\n"
    "cpu = int(sys.argv[1])\n"
    "mem = int(sys.argv[2])\n"
    "fsize = int(sys.argv[3])\n"
    "try:\n"
    "    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGXCPU})\n"
    "    signal.signal(signal.SIGXCPU, signal.SIG_DFL)\n"
    "except (ValueError, OSError):\n"
    "    pass\n"
    "resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))\n"
    "if mem > 0:\n"
    "    try:\n"
    "        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))\n"
    "    except (ValueError, OSError):\n"
    "        pass\n"
    "if fsize > 0:\n"
    "    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))\n"
    "os.execvp(sys.argv[4], sys.argv[4:])\n"
)


class SandboxSession(abc.ABC):
    """Stateful per-episode execution environment (local workspace or container)."""

    @property
    @abc.abstractmethod
    def workdir(self) -> str:
        """Default working directory inside the session."""

    @property
    @abc.abstractmethod
    def jailed(self) -> bool:
        """Whether execution is isolated inside a container or namespace jail."""

    @abc.abstractmethod
    def exec(
        self,
        cmd: list[str] | str,
        *,
        workdir: str | None = None,
        env: collections.abc.Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute ``cmd`` (argv list or shell string) inside the session."""

    async def exec_async(
        self,
        cmd: list[str] | str,
        *,
        workdir: str | None = None,
        env: collections.abc.Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Non-blocking wrapper around :meth:`exec` for async RL loops."""
        return await asyncio.to_thread(
            self.exec,
            cmd,
            workdir=workdir,
            env=env,
            stdin=stdin,
            timeout=timeout,
        )

    def run_python(
        self,
        code: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute an isolated Python snippet inside the session."""
        py_bin = sys.executable if isinstance(self, LocalSandboxSession) else "python3"
        return self.exec([py_bin, "-I", "-c", code], stdin=stdin, timeout=timeout)

    def run_bash(
        self,
        command: str,
        *,
        shell: str = "/bin/sh",
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute a shell command inside the session."""
        return self.exec([shell, "-c", command], timeout=timeout)

    @abc.abstractmethod
    def write_files(self, files: collections.abc.Mapping[str, bytes | str]) -> None:
        """Write relative or session-rooted files into the workspace."""

    @abc.abstractmethod
    def read_file(self, path: str, *, max_chars: int | None = None) -> str:
        """Read a UTF-8 file from the session workspace.

        Args:
            path: Workspace-relative or session-rooted path.
            max_chars: When set, only enough of the file is read to return at
                most ``max_chars + 1`` characters (the extra character lets
                callers detect truncation); the rest is never loaded.

        Returns:
            The decoded (possibly truncated) file text.
        """

    def kill_sandbox_processes(self) -> None:
        """SIGKILL every process running inside the sandbox, leaving the session usable.

        Used before grading so background processes the policy started cannot
        tamper with verification. Implementations must never signal processes
        outside the sandbox.

        Raises:
            NotImplementedError: When the session type cannot enumerate its processes.
            SandboxInfraError: When the kill could not be carried out.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement kill_sandbox_processes().")

    @abc.abstractmethod
    def close(self) -> None:
        """Destroy the container or temporary workspace."""

    def __enter__(self) -> SandboxSession:
        return self

    def __exit__(self, exc_type: tp.Any, exc: tp.Any, tb: tp.Any) -> None:
        self.close()


class ContainerSandboxSession(SandboxSession):
    """Stateful session backed by a Kiri or Docker container."""

    def __init__(self, client: EngineSocketClient, config: ContainerConfig):
        self._client = client
        self._config = config
        self._container_id = client.create_container(config)
        self._closed = False
        try:
            client.start_container(self._container_id)
            client.exec_run(
                self._container_id,
                ["mkdir", "-p", config.workdir],
                workdir="/",
                timeout=10.0,
            )
        except Exception:
            client.remove_container(self._container_id, force=True)
            raise

    @property
    def container_id(self) -> str:
        return self._container_id

    @property
    def workdir(self) -> str:
        return self._config.workdir

    @property
    def jailed(self) -> bool:
        return True

    def exec(
        self,
        cmd: list[str] | str,
        *,
        workdir: str | None = None,
        env: collections.abc.Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        if self._closed:
            return ExecResult(ok=False, stdout="", stderr="", returncode=None, infra_error="Session is closed.")
        if stdin is not None and len(stdin.encode("utf-8", errors="replace")) > self._config.max_stdin_bytes:
            return ExecResult(
                ok=False,
                stdout="",
                stderr=f"stdin payload exceeds max_stdin_bytes={self._config.max_stdin_bytes}.",
                returncode=None,
            )
        argv = ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        merged_env = dict(self._config.env)
        if env:
            merged_env.update(env)
        return self._client.exec_run(
            self._container_id,
            argv,
            workdir=workdir or self._config.workdir,
            env=merged_env or None,
            stdin=stdin,
            timeout=float(timeout if timeout is not None else self._config.timeout),
            max_output_chars=self._config.max_output_chars,
        )

    def write_files(self, files: collections.abc.Mapping[str, bytes | str]) -> None:
        if self._closed:
            raise SandboxInfraError("Session is closed.")
        dirs = sorted(
            {posixpath.dirname(p.lstrip("/")) for p in files if posixpath.dirname(p.lstrip("/")) not in ("", ".")}
        )
        if dirs:
            mkdir_targets = [posixpath.join(self._config.workdir, d) for d in dirs]
            self._client.exec_run(self._container_id, ["mkdir", "-p", *mkdir_targets], workdir="/", timeout=10.0)
        self._client.put_files(self._container_id, files, base_dir=self._config.workdir)

    def read_file(self, path: str, *, max_chars: int | None = None) -> str:
        if self._closed:
            raise SandboxInfraError("Session is closed.")
        full = path if posixpath.isabs(path) else posixpath.join(self._config.workdir, path)
        if max_chars is None:
            return self._client.get_file(self._container_id, full).decode("utf-8", errors="replace")
        limit = max(0, int(max_chars))
        raw = self._client.get_file(self._container_id, full, max_bytes=limit * 4 + 4)
        return raw.decode("utf-8", errors="replace")[: limit + 1]

    def kill_sandbox_processes(self) -> None:
        """SIGKILL every process in the container except pid 1 (runs ``kill -9 -1`` as root inside it)."""
        if self._closed:
            raise SandboxInfraError("Session is closed.")
        self._client.kill_all_processes(self._container_id)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._client.remove_container(self._container_id, force=True)
            except Exception:
                pass


class LocalSandboxSession(SandboxSession):
    """Default child-process session with a persistent per-episode workspace.

    When running unjailed (without ``bwrap``), :meth:`close` and
    :meth:`kill_sandbox_processes` only kill the process groups spawned and
    tracked by this session; any child process that calls ``setsid()`` to
    create a new session/process group escapes cleanup.
    """

    def __init__(self, config: ContainerConfig):
        """Create a session for ``config``.

        Raises:
            ValueError: For an unknown ``use_jail`` policy.
            SandboxInfraError: When the jail is required (``"auto"`` or
                ``"required"``) but ``bwrap`` is unavailable, or when running
                with ``use_jail="off"`` while ``network=False`` or
                ``read_only_rootfs=True`` are requested (they cannot be enforced
                without the jail).
        """
        policy = config.use_jail
        if policy not in _JAIL_POLICIES:
            raise ValueError(f"use_jail must be one of {_JAIL_POLICIES}; got {policy!r}.")
        bwrap = shutil.which("bwrap") if (policy != "off" and _POSIX) else None
        if policy != "off" and bwrap is None:
            raise SandboxInfraError(
                f"use_jail={policy!r} requires `bwrap` (bubblewrap) on PATH to isolate local sandbox commands. "
                "Install bubblewrap, use a container backend, or pass ContainerConfig(use_jail='off') to run "
                "model commands UNJAILED explicitly."
            )
        if bwrap is None:
            if config.network is False:
                raise SandboxInfraError(
                    "network=False was requested but cannot be enforced with use_jail='off'; "
                    "leave network unset (None) or enable the bwrap jail."
                )
            if config.read_only_rootfs:
                raise SandboxInfraError(
                    "read_only_rootfs=True cannot be enforced with use_jail='off'; enable the bwrap jail."
                )
        self._config = config
        self._bwrap = bwrap
        self._root = os.path.realpath(tempfile.mkdtemp(prefix="edenv-"))
        self._closed = False
        self._pgid_lock = threading.Lock()
        self._spawned_pgids: set[int] = set()

    @property
    def workdir(self) -> str:
        return self._root

    @property
    def jailed(self) -> bool:
        return self._bwrap is not None

    def _resolve_local_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            real = os.path.realpath(rel_or_abs)
            if real == self._root or real.startswith(self._root + os.sep):
                return real
            cfg_wd = self._config.workdir.rstrip("/")
            if cfg_wd and rel_or_abs == cfg_wd:
                return self._root
            if cfg_wd and rel_or_abs.startswith(cfg_wd + "/"):
                stripped = rel_or_abs[len(cfg_wd) + 1 :]
            else:
                stripped = rel_or_abs.lstrip("/")
            target = os.path.realpath(os.path.join(self._root, stripped))
        else:
            target = os.path.realpath(os.path.join(self._root, rel_or_abs))
        if target != self._root and not target.startswith(self._root + os.sep):
            raise ValueError(f"Path '{rel_or_abs}' escapes sandbox workspace.")
        return target

    def _build_bwrap_argv(self, argv: list[str], cwd: str) -> list[str]:
        assert self._bwrap is not None
        cmd = [
            self._bwrap,
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--cap-drop",
            "ALL",
            "--die-with-parent",
        ]
        runtime_dirs = ["/usr", "/bin", "/sbin", "/lib", "/lib64"]
        base_prefix = os.path.realpath(sys.base_prefix)
        if not any(base_prefix == d or base_prefix.startswith(d + "/") for d in runtime_dirs):
            runtime_dirs.append(base_prefix)
        for d in runtime_dirs:
            if os.path.exists(d):
                cmd += ["--ro-bind", d, d]
        cmd += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
        if not self._config.network:
            cmd.append("--unshare-net")
        else:
            for p in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
                if os.path.exists(p):
                    cmd += ["--ro-bind", p, p]
        resolved_exe = shutil.which(argv[0]) or argv[0]
        if os.path.exists(resolved_exe):
            resolved_exe = os.path.realpath(resolved_exe)
        return [*cmd, "--bind", self._root, self._root, "--chdir", cwd, "--", resolved_exe, *argv[1:]]

    def exec(
        self,
        cmd: list[str] | str,
        *,
        workdir: str | None = None,
        env: collections.abc.Mapping[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        if self._closed:
            return ExecResult(ok=False, stdout="", stderr="", returncode=None, infra_error="Session is closed.")
        cwd = self._resolve_local_path(workdir) if workdir else self._root
        os.makedirs(cwd, exist_ok=True)
        budget = float(timeout if timeout is not None else self._config.timeout)
        stdin_bytes: bytes | None = None
        if stdin is not None:
            stdin_bytes = stdin.encode("utf-8", errors="replace")
            if len(stdin_bytes) > self._config.max_stdin_bytes:
                return ExecResult(
                    ok=False,
                    stdout="",
                    stderr=f"stdin payload exceeds max_stdin_bytes={self._config.max_stdin_bytes}.",
                    returncode=None,
                )
        cpu_cap = math.ceil(budget) + 1
        mem_bytes = int(self._config.memory_mb) * 1024 * 1024 if self._config.memory_mb > 0 else 0
        fsize_bytes = int(self._config.max_file_size_mb) * 1024 * 1024 if self._config.max_file_size_mb > 0 else 0

        argv = ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        full_argv = self._build_bwrap_argv(argv, cwd) if self._bwrap is not None else argv
        if _POSIX:
            if _PRLIMIT_BIN is not None:
                rlimit_prefix = [_PRLIMIT_BIN, f"--cpu={cpu_cap}:{cpu_cap + 1}"]
                if mem_bytes > 0:
                    rlimit_prefix.append(f"--as={mem_bytes}:{mem_bytes}")
                if fsize_bytes > 0:
                    rlimit_prefix.append(f"--fsize={fsize_bytes}:{fsize_bytes}")
                spawn_argv = [*rlimit_prefix, "--", *full_argv]
            else:
                spawn_argv = [
                    sys.executable,
                    "-c",
                    _POST_EXEC_RLIMIT_CODE,
                    str(cpu_cap),
                    str(mem_bytes),
                    str(fsize_bytes),
                    *full_argv,
                ]
        else:
            spawn_argv = full_argv

        proc_env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
        proc_env.update(
            PATH=f"{os.path.dirname(os.path.realpath(sys.executable))}:{_DEFAULT_PATH}",
            HOME=self._root,
            TMPDIR=self._root,
            PYTHONDONTWRITEBYTECODE="1",
        )
        proc_env.update(self._config.env)
        if env:
            proc_env.update(env)

        try:
            proc = subprocess.Popen(
                spawn_argv,
                stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=proc_env,
                preexec_fn=None,
                start_new_session=_POSIX,
            )
            if _POSIX:
                with self._pgid_lock:
                    self._spawned_pgids.add(proc.pid)
        except OSError as exc:
            return ExecResult(ok=False, stdout="", stderr="", returncode=None, infra_error=str(exc))

        out_buf: list[bytes] = []
        err_buf: list[bytes] = []
        cap = self._config.max_output_chars
        byte_cap = max(cap * 4, 4096)

        def _drain(stream: tp.IO[bytes] | None, sink: list[bytes]) -> None:
            if stream is None:
                return
            kept = 0
            try:
                while True:
                    chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
                    if not chunk:
                        break
                    if kept < byte_cap:
                        piece = chunk[: byte_cap - kept]
                        sink.append(piece)
                        kept += len(piece)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        def _feed(stream: tp.IO[bytes], payload: bytes) -> None:
            # Runs off the caller's thread so a child that never reads stdin cannot block
            # the deadline/kill path; the write fails with EPIPE once the child is killed.
            try:
                stream.write(payload)
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                try:
                    stream.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass

        t_out = threading.Thread(target=_drain, args=(proc.stdout, out_buf), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, err_buf), daemon=True)
        t_out.start()
        t_err.start()
        t_in: threading.Thread | None = None
        if stdin_bytes is not None and proc.stdin is not None:
            t_in = threading.Thread(target=_feed, args=(proc.stdin, stdin_bytes), daemon=True)
            t_in.start()

        timed_out = False
        try:
            proc.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
            if _POSIX:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            else:
                proc.kill()
            proc.wait()

        if t_in is not None:
            t_in.join(timeout=1.0)
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)
        timed_out |= _CPU_BUDGET_RETURNCODE is not None and proc.returncode == _CPU_BUDGET_RETURNCODE
        return ExecResult(
            ok=(proc.returncode == 0 and not timed_out),
            stdout=b"".join(out_buf).decode("utf-8", errors="replace")[:cap],
            stderr=b"".join(err_buf).decode("utf-8", errors="replace")[:cap],
            returncode=proc.returncode,
            timed_out=timed_out,
        )

    def write_files(self, files: collections.abc.Mapping[str, bytes | str]) -> None:
        if self._closed:
            raise SandboxInfraError("Session is closed.")
        for rel_path, content in files.items():
            target = self._resolve_local_path(rel_path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if isinstance(content, str):
                with open(target, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(target, "wb") as f:
                    f.write(content)

    def read_file(self, path: str, *, max_chars: int | None = None) -> str:
        if self._closed:
            raise SandboxInfraError("Session is closed.")
        target = self._resolve_local_path(path)
        # O_NONBLOCK keeps an agent-created FIFO from blocking the open; O_NOFOLLOW refuses a symlink swapped in
        # after the path was resolved. Anything that is not a regular file (FIFO, socket, device) is refused.
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(f"'{path}' is not a regular file.")
            if hasattr(os, "set_blocking"):
                os.set_blocking(fd, True)
            f = os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise
        with f:
            # UTF-8 uses at most 4 bytes per character; read only what max_chars can need.
            raw = f.read() if max_chars is None else f.read(max(0, int(max_chars)) * 4 + 4)
        text = raw.decode("utf-8", errors="replace")
        return text if max_chars is None else text[: max(0, int(max_chars)) + 1]

    def kill_sandbox_processes(self) -> None:
        """SIGKILL every process group this session started and wait briefly for them to exit.

        Every command runs as the leader of a new process group, so its
        background children (``&``, ``nohup``) die with the group. When jailed,
        each command's ``bwrap`` owns a private pid namespace whose init dies
        with it (``--die-with-parent``), taking every process in the namespace,
        including ``setsid``'d ones. Unjailed local mode (explicit
        ``use_jail='off'``) cannot guarantee that a process which called
        ``setsid()`` is reaped. Processes this session did not start are never
        signalled, and the caller's own process group is skipped.
        """
        if not _POSIX:
            raise SandboxInfraError("kill_sandbox_processes requires a POSIX host.")
        own = os.getpgrp()
        with self._pgid_lock:
            pgids = [g for g in self._spawned_pgids if g > 0 and g != own]
        live = [g for g in pgids if _signal_group(g, signal.SIGKILL)]
        deadline = time.monotonic() + _KILL_ALL_WAIT_S
        while live and time.monotonic() < deadline:
            time.sleep(0.01)
            live = [g for g in live if _signal_group(g, 0)]
        with self._pgid_lock:
            self._spawned_pgids.difference_update(g for g in pgids if g not in live)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if _POSIX:
                with self._pgid_lock:
                    pgids = list(self._spawned_pgids)
                    self._spawned_pgids.clear()
                for pgid in pgids:
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                if pgids:
                    time.sleep(0.05)
                for pgid in pgids:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
            shutil.rmtree(self._root, ignore_errors=True)


def discover_engine_endpoint(backend: str = "kiri", *, auto_install: bool = False) -> tuple[str | None, str]:
    """Resolve an execution backend endpoint.

    Explicit sockets are passed to :class:`KiriRuntime` / :class:`SandboxPool`
    as ``endpoint=``; only the standard ``DOCKER_HOST`` variable is consulted.

    Args:
        backend: ``"kiri"`` (default; boots an installed Kiri daemon when none
            answers), ``"docker"`` (require Docker-compatible daemon), ``"auto"``
            (try Kiri, then Docker, then the bwrap-jailed local backend), or
            ``"local"`` (child-process sandbox).
        auto_install: Download Kiri when no binary is installed (off by default;
            run ``easydel setup kiri`` instead).

    Returns:
        ``(endpoint, resolved_backend)`` where ``endpoint`` is ``None`` for ``"local"``.

    Raises:
        SandboxInfraError: When the requested backend is unavailable.
    """
    endpoint, kind, _started = _discover_engine_endpoint(backend, auto_install=auto_install)
    return endpoint, kind


def _discover_engine_endpoint(backend: str, *, auto_install: bool = False) -> tuple[str | None, str, bool]:
    """Like :func:`discover_engine_endpoint`, also reporting whether this call booted an owned daemon."""
    mode = backend.lower()
    if mode not in {"local", "auto", "kiri", "docker"}:
        raise ValueError(f"backend must be 'local', 'auto', 'kiri', or 'docker'; got {backend!r}.")
    if mode == "local":
        return None, "local", False

    home = Path.home()
    kiri_candidates = [
        "unix:///dev/shm/kiri/.kiri/run/docker.sock",
        "unix:///dev/shm/kiri/vm-0/.kiri/run/docker.sock",
        "unix:///run/kiri/docker.sock",
        f"unix://{home / '.kiri' / 'run' / 'docker.sock'}",
        f"unix://{home / 'Library' / 'Application Support' / 'Kiri' / 'run' / 'docker.sock'}",
    ]

    if mode in {"auto", "kiri"}:
        for cand in kiri_candidates:
            sock_path = cand.removeprefix("unix://")
            if os.path.exists(sock_path) and EngineSocketClient(cand).ping():
                return cand, "kiri", False
        if mode == "kiri":
            from .kiri_installer import _ensure_daemon

            booted, started = _ensure_daemon(auto_install=auto_install)
            return booted, "kiri", started

    docker_candidates: list[str] = []
    docker_env = os.environ.get("DOCKER_HOST")
    if docker_env:
        docker_candidates.append(docker_env)
    docker_candidates.extend(
        [
            "unix:///var/run/docker.sock",
            f"unix://{home / '.orbstack' / 'run' / 'docker.sock'}",
            f"unix://{home / '.docker' / 'run' / 'docker.sock'}",
        ]
    )
    for cand in docker_candidates:
        if cand.startswith("unix://") and not os.path.exists(cand.removeprefix("unix://")):
            continue
        client = EngineSocketClient(cand)
        if client.ping():
            try:
                flavor = client.engine_flavor()
            except Exception:
                flavor = "docker"
            return cand, flavor, False

    if mode == "docker":
        raise SandboxInfraError("No reachable Docker Engine socket found.")
    return None, "local", False


class KiriRuntime:
    """Factory that provisions :class:`SandboxSession` instances.

    Defaults to ``backend="kiri"``: an existing Kiri daemon is reused, or an
    installed ``kiri`` binary is booted (installation itself is an explicit
    ``easydel setup kiri`` step unless ``auto_install=True``). Daemons booted by
    this runtime are stopped by :meth:`close`. Pass ``backend="local"`` or
    ``backend="auto"`` (or an explicit ``endpoint``) to run locally or target
    another Engine socket. When ``backend="auto"`` falls back to local
    execution the session always requires the ``bwrap`` jail.
    """

    def __init__(
        self,
        *,
        backend: str = "kiri",
        endpoint: str | None = None,
        config: ContainerConfig | None = None,
        auto_install: bool = False,
        token: str | None = None,
    ):
        """Create a runtime.

        Args:
            backend: ``"kiri"``, ``"docker"``, ``"auto"`` or ``"local"``.
            endpoint: Explicit engine endpoint (skips discovery).
            config: Default :class:`ContainerConfig` for sessions.
            auto_install: Download Kiri when no binary is installed.
            token: Shared secret sent to authenticated sandbox bridges.
        """
        self._requested_backend = backend.lower()
        self._explicit_endpoint = endpoint
        self._default_config = config or ContainerConfig()
        self._auto_install = bool(auto_install)
        self._token = token
        self._owned_lock = threading.Lock()
        self._owned_daemons: list[str] = []

    def _client(self, endpoint: str) -> EngineSocketClient:
        from .kiri_installer import kiri_daemon_budget

        return EngineSocketClient(endpoint, token=self._token, budget=kiri_daemon_budget(endpoint))

    def resolve(self) -> tuple[str | None, str]:
        """Return ``(endpoint, resolved_backend)`` for this runtime."""
        if self._explicit_endpoint is not None:
            client = self._client(self._explicit_endpoint)
            if not client.ping():
                raise SandboxInfraError(f"Container engine at '{self._explicit_endpoint}' did not answer /_ping.")
            try:
                flavor = client.engine_flavor()
            except Exception:
                flavor = "kiri" if self._requested_backend == "kiri" else "docker"
            return self._explicit_endpoint, flavor
        endpoint, kind, started = _discover_engine_endpoint(self._requested_backend, auto_install=self._auto_install)
        if started and endpoint is not None:
            with self._owned_lock:
                self._owned_daemons.append(endpoint)
        return endpoint, kind

    def create_session(self, config: ContainerConfig | None = None) -> SandboxSession:
        """Start and return one isolated :class:`SandboxSession`."""
        cfg = config or self._default_config
        endpoint, backend_kind = self.resolve()
        if endpoint is None or backend_kind == "local":
            if self._requested_backend == "auto" and cfg.use_jail == "auto":
                # Reached by fallback: never silently downgrade to unjailed execution.
                cfg = dataclasses.replace(cfg, use_jail="required")
            return LocalSandboxSession(cfg)
        return ContainerSandboxSession(self._client(endpoint), cfg)

    async def create_session_async(self, config: ContainerConfig | None = None) -> SandboxSession:
        """Non-blocking :meth:`create_session` for async RL rollout workers."""
        return await asyncio.to_thread(self.create_session, config)

    def close(self) -> None:
        """Stop the Kiri daemons this runtime booted (daemons it merely reused are left running)."""
        with self._owned_lock:
            owned = list(self._owned_daemons)
            self._owned_daemons.clear()
        if owned:
            from .kiri_installer import shutdown_kiri_daemons

            shutdown_kiri_daemons(owned)

    def __enter__(self) -> KiriRuntime:
        return self

    def __exit__(self, exc_type: tp.Any, exc: tp.Any, tb: tp.Any) -> None:
        self.close()


class SandboxPool:
    """Consistent-hashed pool of sandbox runtimes for synchronous and async RL.

    Routes ``acquire(affinity_key=...)`` deterministically by ``affinity_key``
    (e.g. ``rollout_id``) so multi-turn steps of an asynchronous trajectory
    land on the same sandbox node while balancing episodes across CPU hosts.
    Daemons booted while building the pool are stopped by :meth:`close`.
    """

    def __init__(
        self,
        endpoints: collections.abc.Sequence[str] | str | None = None,
        *,
        backend: str = "kiri",
        config: ContainerConfig | None = None,
        token: str | None = None,
        auto_install: bool = False,
    ):
        """Create a pool.

        Args:
            endpoints: Engine endpoints, or ``"shm:N"`` / ``"pod-shm:N"`` specs that
                boot local RAM-backed daemons (pod specs stay host-local; use
                :meth:`from_pod_shm` with ``expose_bridge=True`` for cross-host pools).
            backend: Backend used when no endpoints are given.
            config: Default :class:`ContainerConfig`.
            token: Shared secret sent to authenticated sandbox bridges.
            auto_install: Download Kiri when booting daemons and none is installed.
        """
        self._config = config or ContainerConfig()
        self._token = token
        self._auto_install = bool(auto_install)
        self._owned_daemons: list[str] = []
        resolved_endpoints: list[str] | None = None
        if isinstance(endpoints, str):
            resolved_endpoints = self._expand_endpoint_specs([endpoints])
        elif endpoints:
            resolved_endpoints = self._expand_endpoint_specs(list(endpoints))

        if resolved_endpoints:
            self._runtimes = [
                KiriRuntime(
                    backend="kiri" if backend == "local" else backend,
                    endpoint=ep,
                    config=self._config,
                    token=token,
                )
                for ep in resolved_endpoints
            ]
        else:
            self._runtimes = [
                KiriRuntime(backend=backend, config=self._config, token=token, auto_install=self._auto_install)
            ]
        self._rr_lock = threading.Lock()
        self._rr_counter = 0

    def _expand_endpoint_specs(self, specs: list[str]) -> list[str]:
        from .kiri_installer import _ensure_shm_pool, parse_shm_endpoint_spec

        expanded: list[str] = []
        try:
            for spec in specs:
                parsed = parse_shm_endpoint_spec(spec)
                if parsed is None:
                    expanded.append(spec)
                    continue
                _scope, count = parsed
                uris, started = _ensure_shm_pool(num_daemons=count, auto_install=self._auto_install)
                self._owned_daemons.extend(started)
                expanded.extend(uris)
        except BaseException:
            self.close()
            raise
        return expanded

    @classmethod
    def from_pod_shm(
        cls,
        num_daemons_per_host: int = 8,
        *,
        data_dir: str = "/dev/shm/kiri",
        base_port: int = 2375,
        host_ips: collections.abc.Sequence[str] | None = None,
        auto_install: bool = False,
        config: ContainerConfig | None = None,
        budget: KiriDaemonBudget | None = None,
        expose_bridge: bool = False,
        bridge_token: str | None = None,
        allowed_peers: collections.abc.Sequence[str] | None = None,
    ) -> SandboxPool:
        """Create a RAM-backed Kiri pool, optionally spanning every TPU host (e.g. 4x8 = 32 daemons).

        Cross-host pools require ``expose_bridge=True`` and a ``bridge_token``
        shared by every host; see ``kiri_installer.ensure_kiri_pod_shm_pool``.
        """
        from .kiri_installer import _ensure_pod_shm_pool

        endpoints, started = _ensure_pod_shm_pool(
            num_daemons_per_host,
            data_dir=data_dir,
            base_port=base_port,
            host_ips=list(host_ips) if host_ips is not None else None,
            auto_install=auto_install,
            budget=budget,
            expose_bridge=expose_bridge,
            bridge_token=bridge_token,
            allowed_peers=list(allowed_peers) if allowed_peers is not None else None,
        )
        pool = cls(endpoints=endpoints, backend="kiri", config=config, token=bridge_token)
        pool._owned_daemons.extend(started)
        return pool

    @classmethod
    def from_local_shm(
        cls,
        num_daemons: int = 4,
        *,
        data_dir: str = "/dev/shm/kiri",
        auto_install: bool = False,
        config: ContainerConfig | None = None,
        budget: KiriDaemonBudget | None = None,
        kiri_bin: str | None = None,
    ) -> SandboxPool:
        """Create a local RAM-backed Kiri pool in ``/dev/shm`` with zero external CPU VMs."""
        from .kiri_installer import _ensure_shm_pool

        endpoints, started = _ensure_shm_pool(
            num_daemons,
            data_dir=data_dir,
            kiri_bin=kiri_bin,
            auto_install=auto_install,
            budget=budget,
        )
        pool = cls(endpoints=endpoints, backend="kiri", config=config)
        pool._owned_daemons.extend(started)
        return pool

    @property
    def size(self) -> int:
        return len(self._runtimes)

    def select_index(self, affinity_key: str | int | None = None) -> int:
        """Select a deterministic worker index for ``affinity_key``."""
        if len(self._runtimes) == 1:
            return 0
        if affinity_key is None:
            with self._rr_lock:
                idx = self._rr_counter % len(self._runtimes)
                self._rr_counter += 1
                return idx
        digest = hashlib.sha256(str(affinity_key).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self._runtimes)

    def acquire(
        self,
        *,
        affinity_key: str | int | None = None,
        config: ContainerConfig | None = None,
    ) -> SandboxSession:
        """Provision a session pinned by ``affinity_key``."""
        idx = self.select_index(affinity_key)
        return self._runtimes[idx].create_session(config=config or self._config)

    async def acquire_async(
        self,
        *,
        affinity_key: str | int | None = None,
        config: ContainerConfig | None = None,
    ) -> SandboxSession:
        """Non-blocking :meth:`acquire` for async RL rollout workers."""
        idx = self.select_index(affinity_key)
        return await self._runtimes[idx].create_session_async(config=config or self._config)

    def close(self) -> None:
        """Stop the daemons this pool (or its runtimes) booted; reused daemons keep running."""
        for rt in getattr(self, "_runtimes", []):
            rt.close()
        owned = list(self._owned_daemons)
        self._owned_daemons.clear()
        if owned:
            from .kiri_installer import shutdown_kiri_daemons

            shutdown_kiri_daemons(owned)

    def __enter__(self) -> SandboxPool:
        return self

    def __exit__(self, exc_type: tp.Any, exc: tp.Any, tb: tp.Any) -> None:
        self.close()
