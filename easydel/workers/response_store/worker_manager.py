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

"""Lifecycle manager for the Responses store ZeroMQ worker."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from easydel.workers.loggers import get_logger

from .zmq_client import ResponseStoreWorkerClient

logger = get_logger(__name__)


class ResponseStoreWorkerManager:
    """Spawns and manages a response store worker process and ZMQ client.

    When an explicit endpoint is supplied to ``start()``, the manager
    connects to an existing worker. Otherwise it spawns a new worker
    subprocess and manages its lifecycle (startup, health check,
    shutdown, IPC cleanup).

    Attributes:
        endpoint: The ZMQ endpoint of the running worker, or ``None``.
        client: The connected ``ResponseStoreWorkerClient``, or ``None``.
    """

    def __init__(
        self,
        *,
        storage_dir: str | None = None,
        max_stored_responses: int = 10_000,
        max_stored_conversations: int = 1_000,
        compression_level: int = 3,
        startup_timeout: float = 30.0,
        ipc_dir: str | None = None,
    ) -> None:
        """Initialize the worker manager.

        Args:
            storage_dir: Directory for persistent file storage.
            max_stored_responses: Maximum response records to retain.
            max_stored_conversations: Maximum conversation records to retain.
            compression_level: zlib compression level (0-9).
            startup_timeout: Seconds to wait for the worker to bind.
            ipc_dir: Directory for IPC socket files.
        """
        self._storage_dir = storage_dir
        self._max_stored_responses = max_stored_responses
        self._max_stored_conversations = max_stored_conversations
        self._compression_level = compression_level
        self._startup_timeout = startup_timeout
        self._ipc_dir = ipc_dir or tempfile.gettempdir()

        self._client: ResponseStoreWorkerClient | None = None
        self._process: subprocess.Popen | None = None
        self._owned = False
        self._endpoint: str | None = None

    @property
    def endpoint(self) -> str | None:
        return self._endpoint

    @property
    def client(self) -> ResponseStoreWorkerClient | None:
        return self._client

    def start(self, *, endpoint: str | None = None) -> ResponseStoreWorkerClient:
        """Start or connect to a response store worker.

        Args:
            endpoint: Optional existing ZMQ endpoint to connect to.
                When ``None``, a new worker subprocess is spawned.

        Returns:
            A connected ``ResponseStoreWorkerClient``.

        Raises:
            RuntimeError: If the worker has already been started.
            TimeoutError: If the spawned worker does not bind within the timeout.
        """
        if self._client:
            raise RuntimeError("Response store worker has already started.")

        needs_spawn = endpoint is None
        if needs_spawn:
            endpoint = self._make_ipc_endpoint("responses")
            self._process = self._spawn_worker(endpoint)
            self._owned = True
            try:
                self._wait_for_endpoint(endpoint, self._process)
            except Exception:
                self._terminate_process()
                self._cleanup_ipc_file(endpoint)
                raise

        self._endpoint = endpoint
        self._client = ResponseStoreWorkerClient(endpoint)
        return self._client

    def shutdown(self) -> None:
        """Shut down the worker and release all resources.

        If the worker was spawned by this manager, it sends a shutdown
        command and terminates the process. IPC socket files are cleaned up.
        """
        if self._client is None:
            return

        try:
            if self._owned:
                self._client.shutdown()
            else:
                self._client.close()
        except Exception:
            pass
        finally:
            self._client = None

        self._terminate_process()
        if self._owned:
            self._cleanup_ipc_file(self._endpoint)
        self._endpoint = None
        self._owned = False

    def _spawn_worker(self, endpoint: str) -> subprocess.Popen:
        worker_main_path = Path(__file__).with_name("worker_main.py")
        cmd = [
            sys.executable,
            str(worker_main_path),
            "--endpoint",
            endpoint,
            "--max-stored-responses",
            str(int(self._max_stored_responses)),
            "--max-stored-conversations",
            str(int(self._max_stored_conversations)),
            "--compression-level",
            str(int(self._compression_level)),
        ]
        if self._storage_dir:
            cmd.extend(["--storage-dir", self._storage_dir])

        env = os.environ.copy()
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")

        logger.info(f"Spawning response store worker: {' '.join(cmd)}")
        return subprocess.Popen(cmd, env=env)

    def _wait_for_endpoint(self, endpoint: str, process: subprocess.Popen | None) -> None:
        deadline = time.time() + self._startup_timeout
        path = None
        if endpoint.startswith("ipc://"):
            path = endpoint[len("ipc://") :]

        logger.info(f"Waiting for response store worker to bind to {endpoint}")
        while time.time() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(f"Response store worker exited with code {process.returncode}")
            if path and os.path.exists(path):
                logger.info(f"Response store worker bound to {endpoint}")
                return
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for response store worker to bind to {endpoint}")

    def _make_ipc_endpoint(self, prefix: str) -> str:
        os.makedirs(self._ipc_dir, exist_ok=True)
        file_path = os.path.join(self._ipc_dir, f"easydel_{prefix}_{uuid.uuid4().hex}.sock")
        return f"ipc://{file_path}"

    def _terminate_process(self) -> None:
        if not self._process:
            return
        if self._process.poll() is None:
            logger.info("Terminating response store worker process")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Response store worker didn't terminate, killing")
                self._process.kill()
        self._process = None

    def _cleanup_ipc_file(self, endpoint: str | None) -> None:
        if not endpoint or not endpoint.startswith("ipc://"):
            return
        path = endpoint[len("ipc://") :]
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
