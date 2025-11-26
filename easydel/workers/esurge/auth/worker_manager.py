# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Lifecycle manager for auth ZeroMQ worker.

This module provides an AuthWorkerManager class that handles the spawning, lifecycle
management, and cleanup of the auth worker process that communicates via ZeroMQ.

Note:
    This module is for internal use only and is not part of EasyDeL's public API.
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
    """Spawns and manages auth worker process and client.

    When an explicit endpoint is supplied we simply connect to it. Otherwise we
    automatically launch the bundled ZeroMQ worker under an isolated Python
    interpreter and expose its endpoint.
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
        """Initialize the auth worker manager.

        Args:
            require_api_key: If True, all requests must provide a valid API key.
            admin_key: Optional admin key for initial setup.
            enable_audit_logging: Enable audit logging.
            max_audit_entries: Maximum audit log entries to keep.
            storage_dir: Directory for persistent storage.
            enable_persistence: Enable persistent storage.
            auto_save_interval: Auto-save interval in seconds.
            startup_timeout: Timeout for worker startup.
            ipc_dir: Directory for IPC socket files.
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
        """Get the auth worker endpoint."""
        return self._auth_endpoint

    @property
    def auth_client(self) -> AuthWorkerClient | None:
        """Get the auth worker client."""
        return self._auth_client

    def start(self, *, auth_endpoint: str | None = None) -> AuthWorkerClient:
        """Start auth worker process.

        Args:
            auth_endpoint: Optional auth worker endpoint. If None, spawns a worker.

        Returns:
            AuthWorkerClient instance.

        Raises:
            RuntimeError: If worker has already started or fails to start.
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
        """Shutdown the auth worker and clean up resources."""
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
        """Spawn auth worker process.

        Args:
            endpoint: ZeroMQ endpoint for the auth worker.

        Returns:
            subprocess.Popen instance.
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
        if self._max_audit_entries:
            cmd.extend(["--max-audit-entries", str(self._max_audit_entries)])
        if self._storage_dir:
            cmd.extend(["--storage-dir", self._storage_dir])
        if self._enable_persistence:
            cmd.append("--enable-persistence")
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
        """Wait for the worker to bind to the endpoint.

        Args:
            endpoint: ZeroMQ endpoint to wait for.
            process: Worker process to monitor.

        Raises:
            RuntimeError: If worker process exits unexpectedly.
            TimeoutError: If worker doesn't bind within timeout.
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
        """Create a unique IPC endpoint.

        Args:
            prefix: Prefix for the socket filename.

        Returns:
            IPC endpoint URL.
        """
        os.makedirs(self._ipc_dir, exist_ok=True)
        file_path = os.path.join(self._ipc_dir, f"easydel_{prefix}_{uuid.uuid4().hex}.sock")
        return f"ipc://{file_path}"

    def _terminate_process(self) -> None:
        """Terminate the auth worker process."""
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
        """Clean up IPC socket file.

        Args:
            endpoint: Endpoint URL to clean up.
        """
        if not endpoint or not endpoint.startswith("ipc://"):
            return
        path = endpoint[len("ipc://") :]
        try:
            os.unlink(path)
            logger.debug(f"Cleaned up IPC file: {path}")
        except FileNotFoundError:
            pass
