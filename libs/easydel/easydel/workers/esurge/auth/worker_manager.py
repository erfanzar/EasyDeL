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

"""Lifecycle manager for the auth ZeroMQ worker subprocess.

Wraps the spawn / wait-for-bind / shutdown dance for
``easydel.workers.esurge.auth.worker_main`` so the API server can keep its
auth state in an isolated CPU-only Python process. The manager either
spawns the worker itself (auto-generating a unique ``ipc://`` endpoint
under the system tmp directory) or attaches to an externally-managed
endpoint passed in by the caller.

Module exports:
    - :class:`AuthWorkerManager`: spawn / connect / shutdown wrapper that
      returns a connected :class:`AuthWorkerClient` once the worker is
      ready to serve requests.

Note:
    This module is for internal use only and is not part of EasyDeL's
    public API.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from easydel.workers.loggers import get_logger

from .zmq_client import AuthWorkerClient

logger = get_logger(__name__)


class AuthWorkerManager:
    """Spawn and supervise the auth ZMQ worker subprocess.

    Owns the lifecycle of the ``worker_main`` Python subprocess that
    hosts :class:`EnhancedApiKeyManager` behind a ZMQ ``REP`` socket.
    Two operating modes:

    * **Owned**: ``start()`` is called without ``auth_endpoint``; the
      manager generates a unique ``ipc://`` socket under
      ``ipc_dir``, launches ``worker_main`` with the constructor
      arguments forwarded as CLI flags, polls until the IPC file
      appears, and connects an :class:`AuthWorkerClient` to it.
      :meth:`shutdown` sends the ``shutdown`` command, terminates the
      subprocess, and unlinks the IPC file.
    * **Attached**: ``start(auth_endpoint=...)`` is called; the
      manager only opens a client to the existing endpoint and never
      tries to terminate the upstream process.

    Attributes:
        auth_endpoint (str | None): Bound endpoint of the running
            worker, set by :meth:`start` and cleared by
            :meth:`shutdown`.
        auth_client (AuthWorkerClient | None): Connected client; same
            lifecycle as ``auth_endpoint``.
    """

    def __init__(
        self,
        *,
        require_api_key: bool = False,
        admin_key: str | None = None,
        enable_audit_logging: bool = True,
        max_audit_entries: int = 10000,
        storage_dir: str | None = None,
        enable_persistence: bool = True,
        auto_save_interval: float = 60.0,
        startup_timeout: float = 30.0,
        ipc_dir: str | None = None,
    ) -> None:
        """Capture configuration; nothing is spawned until :meth:`start` runs.

        Args:
            require_api_key: When ``True`` the spawned worker rejects
                unauthenticated callers; forwarded as ``--require-api-key``.
            admin_key: Optional bootstrap admin secret; forwarded as
                ``--admin-key``.
            enable_audit_logging: Forwarded as ``--enable-audit-logging``
                / ``--disable-audit-logging``.
            max_audit_entries: Forwarded as ``--max-audit-entries``;
                caps the in-memory audit ring in the worker.
            storage_dir: Forwarded as ``--storage-dir``; ``None`` lets
                the worker default to ``~/.cache/esurge-auth``.
            enable_persistence: Forwarded as ``--enable-persistence``
                / ``--disable-persistence``.
            auto_save_interval: Forwarded as ``--auto-save-interval``
                in seconds.
            startup_timeout: Maximum seconds :meth:`_wait_for_endpoint`
                will poll for the IPC socket file to appear before
                raising :class:`TimeoutError`.
            ipc_dir: Directory where the ``ipc://`` socket file is
                placed; defaults to the system tmp directory.
        """
        self._require_api_key = require_api_key
        self._admin_key = admin_key
        self._enable_audit_logging = enable_audit_logging
        self._max_audit_entries = max_audit_entries
        self._storage_dir = storage_dir
        self._enable_persistence = enable_persistence
        self._auto_save_interval = auto_save_interval
        self._startup_timeout = startup_timeout
        self._ipc_dir = ipc_dir or tempfile.gettempdir()

        self._auth_client: AuthWorkerClient | None = None
        self._auth_process: subprocess.Popen | None = None
        self._auth_owned = False
        self._auth_endpoint: str | None = None

    @property
    def auth_endpoint(self) -> str | None:
        """Return the ZMQ endpoint of the running auth worker.

        Returns:
            str | None: The bound endpoint URI once :meth:`start` succeeds;
            ``None`` before start or after :meth:`shutdown`.
        """
        return self._auth_endpoint

    @property
    def auth_client(self) -> AuthWorkerClient | None:
        """Return the connected auth client.

        Returns:
            AuthWorkerClient | None: The client returned by :meth:`start`;
            ``None`` before start or after :meth:`shutdown`.
        """
        return self._auth_client

    def start(self, *, auth_endpoint: str | None = None) -> AuthWorkerClient:
        """Spawn (or attach to) the worker and return a connected client.

        When ``auth_endpoint`` is ``None`` the manager allocates a fresh
        ``ipc://`` endpoint under ``ipc_dir``, launches the worker
        subprocess, and blocks until the IPC socket file appears (up to
        ``startup_timeout`` seconds). When an endpoint is supplied, no
        process is spawned and ownership stays with the caller.

        Args:
            auth_endpoint: Pre-existing worker endpoint to attach to.
                ``None`` triggers the owned-subprocess path.

        Returns:
            AuthWorkerClient: A client connected to the running worker;
            also exposed via :attr:`auth_client`.

        Raises:
            RuntimeError: If the manager already has an active client,
                or if the spawned worker exits before binding.
            TimeoutError: If the worker fails to bind within
                ``startup_timeout`` seconds.
        """
        if self._auth_client:
            raise RuntimeError("Auth worker has already started.")

        needs_auth_spawn = auth_endpoint is None

        if needs_auth_spawn:
            auth_endpoint = self._make_ipc_endpoint("auth")
            self._auth_process = self._spawn_auth_worker(auth_endpoint)
            self._auth_owned = True
            try:
                self._wait_for_endpoint(auth_endpoint, self._auth_process)
            except Exception:
                self._terminate_process()
                self._cleanup_ipc_file(auth_endpoint)
                raise

        self._auth_endpoint = auth_endpoint
        self._auth_client = AuthWorkerClient(self._auth_endpoint)
        return self._auth_client

    def shutdown(self) -> None:
        """Shut down the worker (if owned) and release all resources.

        Sends ``shutdown`` to the worker when the manager owns the
        subprocess, otherwise only closes the local client. Always
        terminates the tracked subprocess and unlinks the IPC socket
        file when ownership was retained. Idempotent: subsequent calls
        are no-ops.
        """
        if self._auth_client is None:
            return

        try:
            if self._auth_owned:
                self._auth_client.shutdown()
            else:
                self._auth_client.close()
        except Exception:
            pass
        finally:
            self._auth_client = None

        self._terminate_process()
        if self._auth_owned:
            self._cleanup_ipc_file(self._auth_endpoint)
        self._auth_endpoint = None

    # Internal helpers -----------------------------------------------------

    def _spawn_auth_worker(self, endpoint: str) -> subprocess.Popen:
        """Launch ``worker_main.py`` under a CPU-only Python interpreter.

        Builds the CLI argument list from the manager's configuration
        and seeds the child environment with ``JAX_PLATFORMS=cpu``,
        ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` and
        ``ENABLE_DISTRIBUTED_INIT=0`` so the auth worker never tries to
        grab accelerators.

        Args:
            endpoint: ZMQ endpoint to bind on the child side. Always
                an ``ipc://`` URL when called from :meth:`start`.

        Returns:
            subprocess.Popen: Handle to the spawned worker; the manager
            keeps a reference so :meth:`_terminate_process` can clean
            it up on shutdown.
        """
        worker_main_path = Path(__file__).with_name("worker_main.py")
        cmd = [
            sys.executable,
            str(worker_main_path),
            "--endpoint",
            endpoint,
        ]

        # Add auth config parameters
        if self._require_api_key:
            cmd.append("--require-api-key")
        if self._admin_key:
            cmd.extend(["--admin-key", self._admin_key])
        if self._enable_audit_logging:
            cmd.append("--enable-audit-logging")
        else:
            cmd.append("--disable-audit-logging")
        if self._max_audit_entries:
            cmd.extend(["--max-audit-entries", str(self._max_audit_entries)])
        if self._storage_dir:
            cmd.extend(["--storage-dir", self._storage_dir])
        if self._enable_persistence:
            cmd.append("--enable-persistence")
        else:
            cmd.append("--disable-persistence")
        if self._auto_save_interval:
            cmd.extend(["--auto-save-interval", str(self._auto_save_interval)])

        env = os.environ.copy()
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")

        logger.info(f"Spawning auth worker: {' '.join(cmd)}")
        return subprocess.Popen(cmd, env=env)

    def _wait_for_endpoint(self, endpoint: str, process: subprocess.Popen | None) -> None:
        """Block until the worker binds to ``endpoint`` or fails fast.

        For ``ipc://`` endpoints, readiness is detected by the socket
        file appearing on disk; for other transports the method only
        verifies the subprocess is alive until the timeout elapses.

        Args:
            endpoint: Endpoint string the worker is expected to bind.
            process: The subprocess being supervised. ``None`` skips
                liveness checks (used when attaching).

        Raises:
            RuntimeError: When ``process`` exits before binding.
            TimeoutError: When ``startup_timeout`` seconds pass without
                the IPC socket file appearing.
        """
        deadline = time.time() + self._startup_timeout
        path = None
        if endpoint.startswith("ipc://"):
            path = endpoint[len("ipc://") :]

        logger.info(f"Waiting for auth worker to bind to {endpoint}")
        while time.time() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(f"Auth worker process exited with code {process.returncode}")
            if path and os.path.exists(path):
                logger.info(f"Auth worker bound to {endpoint}")
                return
            time.sleep(0.05)

        raise TimeoutError(f"Timed out waiting for auth worker to bind to {endpoint}")

    def _make_ipc_endpoint(self, prefix: str) -> str:
        """Allocate a fresh ``ipc://`` endpoint under :attr:`_ipc_dir`.

        Uses ``uuid.uuid4().hex`` to guarantee uniqueness so re-running
        the manager (or running multiple instances) never collides on
        socket paths. Creates ``_ipc_dir`` if it does not exist.

        Args:
            prefix: Short slug embedded in the filename for
                disambiguation (e.g. ``"auth"``).

        Returns:
            str: A ZMQ endpoint of the form
            ``ipc:///<ipc_dir>/easydel_<prefix>_<hex>.sock``.
        """
        os.makedirs(self._ipc_dir, exist_ok=True)
        file_path = os.path.join(self._ipc_dir, f"easydel_{prefix}_{uuid.uuid4().hex}.sock")
        return f"ipc://{file_path}"

    def _terminate_process(self) -> None:
        """Best-effort termination of the auth worker subprocess.

        Sends ``SIGTERM``, waits up to 5 seconds for a clean exit, then
        escalates to ``SIGKILL`` if the process is still running.
        No-op when the process has already exited or was never spawned.
        """
        if not self._auth_process:
            return
        if self._auth_process.poll() is None:
            logger.info("Terminating auth worker process")
            self._auth_process.terminate()
            try:
                self._auth_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Auth worker didn't terminate, killing")
                self._auth_process.kill()
        self._auth_process = None

    def _cleanup_ipc_file(self, endpoint: str | None) -> None:
        """Unlink the IPC socket file backing ``endpoint``, if any.

        Skipped for non-IPC endpoints or when the file is already gone.

        Args:
            endpoint: Endpoint URL whose socket file should be removed.
                ``None`` is silently ignored.
        """
        if not endpoint or not endpoint.startswith("ipc://"):
            return
        path = endpoint[len("ipc://") :]
        try:
            os.unlink(path)
            logger.debug(f"Cleaned up IPC file: {path}")
        except FileNotFoundError:
            pass
