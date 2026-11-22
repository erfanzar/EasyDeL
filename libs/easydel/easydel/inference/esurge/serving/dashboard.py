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

"""The dashboard next to a serving deployment, and one ``/metrics`` over every dp front.

:class:`DashboardSupervisor` keeps ``esurge-dashboard`` running beside the endpoint (``server.dashboard``, on by
default). With dp replicas the endpoint is served by several front processes sharing one port, so a plain
``/metrics`` call would describe whichever front the kernel picked. :class:`FrontMetricsMiddleware` makes every
front answer ``/metrics`` for the whole deployment: the front that receives the call checks the caller's key once
(with the caller's address, so per-key IP rules and rate limits apply as usual), then reads every other front's own
metrics and request rows over a private unix socket and merges them. The sockets live in a directory only this user
can open, created by the process that starts the fronts, so a peer call needs no key.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import typing as tp
from pathlib import Path

DASHBOARD_BINARY = "esurge-dashboard"
DASHBOARD_STABLE_S = 300.0
"""A dashboard that ran this long before exiting is restarted without backoff growth."""
DASHBOARD_MAX_BACKOFF_S = 60.0
"""Longest wait between restarts of a dashboard that keeps exiting (and between checks of a busy port)."""
DASHBOARD_MISSING_RETRY_S = 300.0
"""How often the supervisor looks for the dashboard binary again when it found none."""
DASHBOARD_BUILD_TIMEOUT_S = 1800.0
"""Longest a one-off ``cargo build`` of the dashboard may take."""
_PEER_TIMEOUT_S = 5.0
_MAX_UNIX_PATH = 100
"""AF_UNIX paths are limited to ~108 bytes; keep socket paths below this."""


def dashboard_source_dir() -> Path:
    """``libs/esurge-dashboard`` of the checkout this package runs from."""
    return Path(__file__).resolve().parents[5] / DASHBOARD_BINARY


def find_dashboard_binary(
    *, build: bool = True, log: tp.Callable[[str], None] = print, stop: threading.Event | None = None
) -> str | None:
    """The ``esurge-dashboard`` executable: on ``PATH``, built in the checkout, or (``build``) built now with cargo.

    The build runs as a child process that ``stop`` (when set) or :data:`DASHBOARD_BUILD_TIMEOUT_S` kills.

    Returns:
        Its path, or ``None`` when there is none and it cannot be built here.
    """
    found = shutil.which(DASHBOARD_BINARY)
    if found:
        return found
    source = dashboard_source_dir()
    built = source / "target" / "release" / DASHBOARD_BINARY
    if built.is_file() and os.access(built, os.X_OK):
        return str(built)
    if not build or not (source / "Cargo.toml").is_file():
        return None
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not os.access(cargo, os.X_OK):
        return None
    log(f"building {DASHBOARD_BINARY} in {source}")
    with tempfile.TemporaryFile() as errors:
        try:
            # Own session so a kill takes rustc down too; an explicit target dir so CARGO_TARGET_DIR cannot move
            # the binary away from where this function looks for it.
            process = subprocess.Popen(
                [cargo, "build", "--release", "--locked", "--target-dir", str(source / "target")],
                cwd=source,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                start_new_session=True,
            )
        except OSError as exc:
            log(f"building {DASHBOARD_BINARY} failed: {exc}")
            return None
        deadline = time.monotonic() + DASHBOARD_BUILD_TIMEOUT_S
        while process.poll() is None:
            stopped = stop.wait(1.0) if stop is not None else bool(time.sleep(1.0))
            if stopped or time.monotonic() > deadline:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                log(f"building {DASHBOARD_BINARY} {'stopped' if stopped else 'timed out'}")
                return None
        if process.returncode != 0:
            size = errors.seek(0, os.SEEK_END)
            errors.seek(max(0, size - 500))
            log(f"building {DASHBOARD_BINARY} failed: {errors.read().decode(errors='replace')}")
            return None
    return str(built) if built.is_file() else None


def _host_port(host: str, port: int) -> str:
    bare = host.strip("[]")
    return f"[{bare}]:{int(port)}" if ":" in bare else f"{bare}:{int(port)}"


def _port_free(host: str, port: int) -> bool:
    """Whether ``host:port`` can be bound right now (nothing else listens there)."""
    bare = host.strip("[]") or "0.0.0.0"
    probe = socket.socket(socket.AF_INET6 if ":" in bare else socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((bare, int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _tied_to_parent(argv: list[str]) -> list[str]:
    """``argv`` run so the dashboard gets SIGTERM when its supervising thread's process dies (Linux ``setpriv``).

    ``setpriv --pdeathsig`` sets the death signal and execs, so no Python runs between fork and exec: a ``preexec_fn``
    would make ``subprocess`` fork the (large, multithreaded) serving process instead of spawning. Without ``setpriv``
    the dashboard still dies with the launch's process group on a normal stop.
    """
    setpriv = shutil.which("setpriv")
    return [setpriv, "--pdeathsig", "TERM", "--", *argv] if setpriv else argv


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class DashboardSupervisor:
    """Keep one ``esurge-dashboard`` process running beside the endpoint, restarting it with backoff.

    It builds the binary at most once per lifetime, waits (logging once) while the dashboard port is taken by
    something else, and stops the dashboard, or a build in progress, on :meth:`stop`.

    Args:
        host: Address the endpoint binds; the dashboard listens on the same address.
        port: Dashboard port.
        endpoint_port: The endpoint's port; the dashboard reads ``/metrics`` there.
        log: Line logger.
        find_binary: Locates (or builds) the dashboard executable; called with ``build``, ``log`` and ``stop``.
        max_backoff_s: Longest wait between restarts, and between checks of a busy port.
        missing_retry_s: Wait before looking for a missing binary again.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        endpoint_port: int,
        log: tp.Callable[[str], None],
        find_binary: tp.Callable[..., str | None] = find_dashboard_binary,
        max_backoff_s: float = DASHBOARD_MAX_BACKOFF_S,
        missing_retry_s: float = DASHBOARD_MISSING_RETRY_S,
    ) -> None:
        bare = str(host).strip().strip("[]")
        listen_host = bare or "0.0.0.0"
        if bare == "::":
            upstream_host = "::1"
        elif bare in {"0.0.0.0", ""}:
            upstream_host = "127.0.0.1"
        else:
            upstream_host = bare
        self.host, self.port = listen_host, int(port)
        self.listen = _host_port(listen_host, port)
        self.upstream = f"http://{_host_port(upstream_host, endpoint_port)}"
        self._log = log
        self._find_binary = find_binary
        self._max_backoff_s = float(max_backoff_s)
        self._missing_retry_s = float(missing_retry_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self.starts = 0

    @property
    def pid(self) -> int | None:
        """Process id of the dashboard currently (or last) started, ``None`` before the first start."""
        process = self._process
        return None if process is None else process.pid

    def start(self) -> DashboardSupervisor:
        self._thread = threading.Thread(target=self._run, name="esurge-dashboard-supervisor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        _terminate(self._process)
        if self._thread is not None:
            self._thread.join(timeout=15)

    def _run(self) -> None:
        restarts = 0
        build = True
        port_busy_logged = False
        while not self._stop.is_set():
            binary = self._find_binary(build=build, log=self._log, stop=self._stop)
            build = False
            if self._stop.is_set():
                return
            if binary is None:
                self._log(
                    f"{DASHBOARD_BINARY} not found; looking again in {self._missing_retry_s:.0f}s "
                    "(set server.dashboard: false to turn the dashboard off)"
                )
                self._stop.wait(self._missing_retry_s)
                continue
            if not _port_free(self.host, self.port):
                if not port_busy_logged:
                    self._log(f"dashboard port {self.listen} is in use; the dashboard starts once it is free")
                    port_busy_logged = True
                self._stop.wait(self._max_backoff_s)
                continue
            port_busy_logged = False
            argv = [binary, "--upstream", self.upstream, "--listen", self.listen]
            started = time.monotonic()
            try:
                process = subprocess.Popen(_tied_to_parent(argv), stdin=subprocess.DEVNULL)
            except OSError as exc:
                self._log(f"starting {DASHBOARD_BINARY} failed: {exc}")
                self._stop.wait(self._max_backoff_s)
                continue
            self._process = process
            self.starts += 1
            # stop() may have run before the process was published: it is this thread's to end then.
            if self._stop.is_set():
                _terminate(process)
                return
            self._log(f"dashboard (pid {process.pid}) on http://{self.listen} reading {self.upstream}/metrics")
            while process.poll() is None and not self._stop.wait(0.5):
                pass
            if self._stop.is_set():
                _terminate(process)
                return
            lived = time.monotonic() - started
            restarts = 0 if lived > DASHBOARD_STABLE_S else restarts + 1
            wait = min(self._max_backoff_s, 2.0**restarts)
            self._log(
                f"dashboard (pid {process.pid}) exited with code {process.returncode} after {lived:.0f}s; "
                f"restarting it in {wait:.0f}s"
            )
            self._stop.wait(wait)


def create_front_socket_dir() -> str:
    """A fresh owner-only directory for the fronts' private sockets (unpredictable name, mode 0700)."""
    directory = tempfile.mkdtemp(prefix="esurge-fronts-")
    if len(front_socket_path(directory, 99)) > _MAX_UNIX_PATH:
        os.rmdir(directory)
        directory = tempfile.mkdtemp(prefix="esurge-fronts-", dir="/tmp")
    return directory


def front_socket_path(directory: str, index: int) -> str:
    """Private unix socket of front ``index`` in ``directory``."""
    return os.path.join(directory, f"front-{int(index)}.sock")


def bind_unix_socket(path: str) -> socket.socket:
    """A listening unix stream socket at ``path`` (a stale file left by a dead front is replaced)."""
    if os.path.lexists(path):
        os.unlink(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        os.chmod(path, 0o600)
        sock.listen(64)
        return sock
    except BaseException:
        sock.close()
        try:
            if os.path.lexists(path):
                os.unlink(path)
        except OSError:
            pass
        raise


def merge_front_metrics(
    payloads: list[dict], *, window_seconds: float, recent_limit: int = 20, expected: int | None = None
) -> dict:
    """One ``/metrics`` body for the deployment from every front's own body (each with ``request_rows``).

    Counters add up; queue depths come from any front (each routes over every replica, so each already reports the
    whole deployment); the windowed per-request aggregate is recomputed over the union of the fronts' request rows, so
    percentiles and rates are exact rather than averaged. ``request_rows`` never appears in the result.
    """
    from ..metrics import RequestMetrics, summarize_requests

    row_fields = {field.name for field in dataclasses.fields(RequestMetrics)}
    merged: dict[str, tp.Any] = {
        "uptime_seconds": max((float(p.get("uptime_seconds") or 0.0) for p in payloads), default=0.0),
        "status": next(
            (p["status"] for p in payloads if p.get("status") == "ready"),
            payloads[0].get("status") if payloads else None,
        ),
        "models_loaded": max((int(p.get("models_loaded") or 0) for p in payloads), default=0),
        "fronts": len(payloads),
        "fronts_expected": len(payloads) if expected is None else int(expected),
    }
    for field in (
        "total_requests",
        "successful_requests",
        "failed_requests",
        "total_tokens_generated",
        "active_requests",
    ):
        merged[field] = sum(int(p.get(field) or 0) for p in payloads)
    # A per-request rate (each front's moving average over its own requests): average it, never add it up.
    rates = [float(p["average_tokens_per_second"]) for p in payloads if p.get("average_tokens_per_second")]
    merged["average_tokens_per_second"] = round(sum(rates) / len(rates), 2) if rates else 0.0
    auth = [p["auth_stats"] for p in payloads if isinstance(p.get("auth_stats"), dict)]
    if auth:
        merged["auth_stats"] = {
            key: (max if key in {"total_keys", "active_keys"} else sum)(
                a.get(key) for a in auth if isinstance(a.get(key), (int, float))
            )
            for key in auth[0]
            if isinstance(auth[0][key], (int, float)) and not isinstance(auth[0][key], bool)
        }
    engines: dict[str, dict] = {}
    for name in dict.fromkeys(name for p in payloads for name in (p.get("engine_metrics") or {})):
        views = [
            p["engine_metrics"][name] for p in payloads if isinstance((p.get("engine_metrics") or {}).get(name), dict)
        ]
        if not views:
            continue
        rows: dict[str, RequestMetrics] = {}
        for view in views:
            for row in view.get("request_rows") or ():
                try:
                    rows[str(row.get("request_id"))] = RequestMetrics(
                        **{key: value for key, value in row.items() if key in row_fields}
                    )
                except (TypeError, AttributeError):
                    continue
        summary = summarize_requests(list(rows.values()), window_seconds=window_seconds, recent_limit=recent_limit)
        first = {k: v for k, v in views[0].items() if k != "request_rows"}
        engines[name] = {
            **first,
            "running_requests": max(int(v.get("running_requests") or 0) for v in views),
            "waiting_requests": max(int(v.get("waiting_requests") or 0) for v in views),
            **summary,
        }
    merged["engine_metrics"] = engines
    return merged


async def _fetch_peer(path: str) -> dict | None:
    """A peer front's own metrics (with request rows) over its private socket; ``None`` if it gave none."""
    reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), _PEER_TIMEOUT_S)
    try:
        writer.write(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        await asyncio.wait_for(writer.drain(), _PEER_TIMEOUT_S)
        raw = await asyncio.wait_for(reader.read(), _PEER_TIMEOUT_S)
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), _PEER_TIMEOUT_S)
        except Exception:
            pass
    head, _, body = raw.partition(b"\r\n\r\n")
    if head.split(b" ", 2)[1:2] != [b"200"]:
        return None
    payload = json.loads(body)
    return payload if isinstance(payload, dict) else None


def _dumps(payload: tp.Any) -> bytes:
    return json.dumps(payload, default=str).encode()


class FrontMetricsMiddleware:
    """ASGI middleware on each dp front: ``GET /metrics`` answers for every front.

    A call on the endpoint port runs the front's own ``/metrics`` (key, IP rules and rate limit checked once,
    any refusal returned unchanged), then adds every peer's metrics. A call on this front's own private socket is a
    peer asking: it gets this front's metrics and request rows without a key, since only this user can reach it.

    Args:
        app: The front's ASGI app.
        own_socket: This front's private unix socket.
        sockets: Every front's private unix socket (this front's included).
        local_metrics: This front's ``/metrics`` body, without an authorization check.
        request_rows: ``request_rows(served_name)``: this front's finished-request rows for that model.
        window_seconds: The ``/metrics`` aggregation window.
    """

    def __init__(
        self,
        app: tp.Any,
        *,
        own_socket: str,
        sockets: list[str],
        local_metrics: tp.Callable[[], dict],
        request_rows: tp.Callable[[str], list[dict]],
        window_seconds: float,
    ) -> None:
        self.app = app
        self.own_socket = own_socket
        self.sockets = list(sockets)
        self.local_metrics = local_metrics
        self.request_rows = request_rows
        self.window_seconds = float(window_seconds)

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/metrics" or scope.get("method") != "GET":
            await self.app(scope, receive, send)
            return
        if self._from_peer(scope):
            await _respond(send, 200, _dumps(self._with_rows(self.local_metrics())))
            return
        await self._merged(scope, receive, send)

    def _from_peer(self, scope) -> bool:
        """Whether the request arrived on this front's private unix socket (never true for a TCP connection)."""
        server = scope.get("server")
        return scope.get("client") is None and bool(server) and server[0] == self.own_socket

    def _with_rows(self, payload: dict) -> dict:
        for name, view in (payload.get("engine_metrics") or {}).items():
            if isinstance(view, dict):
                view["request_rows"] = self.request_rows(name)
        return payload

    async def _merged(self, scope, receive, send) -> None:
        start: dict = {}
        chunks: list[bytes] = []

        async def capture(message) -> None:
            if message["type"] == "http.response.start":
                start.update(message)
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await self.app(scope, receive, capture)
        body = b"".join(chunks)
        own = None
        if start.get("status") == 200:
            try:
                own = json.loads(body)
            except ValueError:
                own = None
        if not isinstance(own, dict):
            # Refused (401/403/429, ...) or not a metrics body: the caller gets exactly what this front said.
            await send({**start, "type": "http.response.start"} if start else _start(500))
            await send({"type": "http.response.body", "body": body})
            return
        peers = [path for path in self.sockets if path != self.own_socket]
        results = await asyncio.gather(*(_fetch_peer(path) for path in peers), return_exceptions=True)
        include_auth_stats = "auth_stats" in own
        own_engines = own.get("engine_metrics") if isinstance(own.get("engine_metrics"), dict) else None
        payloads = [self._with_rows(own)] + [r for r in results if isinstance(r, dict)]
        expected = max(len(self.sockets), len(payloads))
        merged_dict = merge_front_metrics(payloads, window_seconds=self.window_seconds, expected=expected)
        if not include_auth_stats:
            merged_dict.pop("auth_stats", None)
        if own_engines is not None and isinstance(merged_dict.get("engine_metrics"), dict):
            filtered_engines: dict[str, tp.Any] = {}
            for name, own_view in own_engines.items():
                merged_view = merged_dict["engine_metrics"].get(name)
                if not isinstance(merged_view, dict):
                    continue
                if isinstance(own_view, dict) and "recent_requests" not in own_view:
                    merged_view = {k: v for k, v in merged_view.items() if k != "recent_requests"}
                filtered_engines[name] = merged_view
            merged_dict["engine_metrics"] = filtered_engines
            if "models_loaded" in own:
                merged_dict["models_loaded"] = len(filtered_engines)
        merged = _dumps(merged_dict)
        # Keep the app's own response headers (CORS, ...); only the length changes.
        headers = [(k, v) for k, v in start.get("headers", ()) if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(merged)).encode()))
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": merged})


def _start(status: int, length: int = 0) -> dict:
    return {
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(length).encode())],
    }


async def _respond(send, status: int, body: bytes) -> None:
    await send(_start(status, len(body)))
    await send({"type": "http.response.body", "body": body})
