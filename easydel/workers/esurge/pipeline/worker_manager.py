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

"""Lifecycle manager for tokenizer and detokenizer ZeroMQ workers.

This module provides a WorkerManager class that handles the spawning, lifecycle
management, and cleanup of tokenizer and detokenizer worker processes that communicate
via ZeroMQ endpoints.

Note:
    This module is for internal use only and is not part of EasyDeL's public API.
    It is only accessible to EasyDeL modules that require external worker processes
    to handle specific tasks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from easydel.workers.loggers import get_logger

from .zmq_workers import DetokenizerWorkerClient, TokenizerWorkerClient

logger = get_logger(__name__)


class WorkerManager:
    """Spawns and manages tokenizer/detokenizer worker processes and clients.

    When explicit endpoints are supplied we simply connect to them. Otherwise we
    automatically launch the bundled ZeroMQ workers under isolated Python
    interpreters (using ``uv run`` when available) and expose their endpoints.
    """

    def __init__(
        self,
        tokenizer_source: str | None,
        *,
        tokenizer_kwargs: dict[str, Any] | None = None,
        startup_timeout: float = 30.0,
        ipc_dir: str | None = None,
    ) -> None:
        self._tokenizer_source = tokenizer_source
        self._tokenizer_kwargs = tokenizer_kwargs or {}
        self._startup_timeout = startup_timeout
        self._ipc_dir = ipc_dir or tempfile.gettempdir()

        self._tokenizer_client: TokenizerWorkerClient | None = None
        self._detokenizer_client: DetokenizerWorkerClient | None = None

        self._tokenizer_process: subprocess.Popen | None = None
        self._detokenizer_process: subprocess.Popen | None = None

        self._tokenizer_owned = False
        self._detokenizer_owned = False

        self._tokenizer_endpoint: str | None = None
        self._detokenizer_endpoint: str | None = None

    @property
    def tokenizer_endpoint(self) -> str | None:
        return self._tokenizer_endpoint

    @property
    def detokenizer_endpoint(self) -> str | None:
        return self._detokenizer_endpoint

    def start(
        self,
        *,
        detokenizer_max_states: int,
        tokenizer_endpoint: str | None,
        detokenizer_endpoint: str | None,
    ) -> tuple[TokenizerWorkerClient, DetokenizerWorkerClient]:
        if self._tokenizer_client or self._detokenizer_client:
            raise RuntimeError("WorkerManager has already started clients.")

        needs_tokenizer_spawn = tokenizer_endpoint is None
        needs_detokenizer_spawn = detokenizer_endpoint is None

        if needs_tokenizer_spawn or needs_detokenizer_spawn:
            if not self._tokenizer_source:
                raise ValueError("Tokenizer identifier must be provided when worker endpoints are not supplied.")

        if needs_tokenizer_spawn:
            tokenizer_endpoint = self._make_ipc_endpoint("tokenizer")
            self._tokenizer_process = self._spawn_worker(
                "tokenizer",
                tokenizer_endpoint,
                detokenizer_max_states=detokenizer_max_states,
            )
            self._tokenizer_owned = True
            try:
                self._wait_for_endpoint(tokenizer_endpoint, self._tokenizer_process)
            except Exception:
                self._terminate_process("_tokenizer_process")
                self._cleanup_ipc_file(tokenizer_endpoint)
                raise
        self._tokenizer_endpoint = tokenizer_endpoint

        if needs_detokenizer_spawn:
            detokenizer_endpoint = self._make_ipc_endpoint("detokenizer")
            self._detokenizer_process = self._spawn_worker(
                "detokenizer",
                detokenizer_endpoint,
                detokenizer_max_states=detokenizer_max_states,
            )
            self._detokenizer_owned = True
            try:
                self._wait_for_endpoint(detokenizer_endpoint, self._detokenizer_process)
            except Exception:
                self._terminate_process("_detokenizer_process")
                self._cleanup_ipc_file(detokenizer_endpoint)
                raise
        self._detokenizer_endpoint = detokenizer_endpoint

        self._tokenizer_client = TokenizerWorkerClient(self._tokenizer_endpoint)
        self._detokenizer_client = DetokenizerWorkerClient(self._detokenizer_endpoint)
        return self._tokenizer_client, self._detokenizer_client

    def shutdown(self) -> None:
        self._shutdown_client("_tokenizer_client", "_tokenizer_owned", "_tokenizer_process", self._tokenizer_endpoint)
        self._shutdown_client(
            "_detokenizer_client", "_detokenizer_owned", "_detokenizer_process", self._detokenizer_endpoint
        )
        self._tokenizer_endpoint = None
        self._detokenizer_endpoint = None

    def drain_workers(self) -> None:
        """Flush in-flight tokenizer/detokenizer state."""
        for name, client in (
            ("tokenizer", self._tokenizer_client),
            ("detokenizer", self._detokenizer_client),
        ):
            if not client:
                continue
            try:
                client.drain()
            except Exception as exc:
                logger.warning("Failed to drain %s worker: %s", name, exc)

    # Internal helpers -----------------------------------------------------

    def _spawn_worker(
        self,
        mode: str,
        endpoint: str,
        *,
        detokenizer_max_states: int,
    ) -> subprocess.Popen:
        worker_main_path = Path(__file__).with_name("worker_main.py")
        cmd = [
            sys.executable,
            str(worker_main_path),
            mode,
            "--endpoint",
            endpoint,
            "--tokenizer-path",
            self._tokenizer_source,
            "--tokenizer-kwargs",
            json.dumps(self._tokenizer_kwargs),
        ]
        if mode == "detokenizer":
            cmd.extend(["--max-states", str(detokenizer_max_states)])

        env = os.environ.copy()
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")

        return subprocess.Popen(cmd, env=env)

    def _wait_for_endpoint(self, endpoint: str, process: subprocess.Popen | None) -> None:
        deadline = time.time() + self._startup_timeout
        path = None
        if endpoint.startswith("ipc://"):
            path = endpoint[len("ipc://") :]

        while time.time() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(f"Worker process for {endpoint} exited with code {process.returncode}")
            if path and os.path.exists(path):
                return
            time.sleep(0.05)

        raise TimeoutError(f"Timed out waiting for worker to bind to {endpoint}")

    def _make_ipc_endpoint(self, prefix: str) -> str:
        os.makedirs(self._ipc_dir, exist_ok=True)
        file_path = os.path.join(self._ipc_dir, f"easydel_{prefix}_{uuid.uuid4().hex}.sock")
        return f"ipc://{file_path}"

    def _shutdown_client(
        self,
        client_attr: str,
        owned_attr: str,
        process_attr: str,
        endpoint: str | None,
    ) -> None:
        client = getattr(self, client_attr)
        if client is None:
            return
        owned = getattr(self, owned_attr)
        try:
            if owned:
                client.shutdown()
            else:
                client.close()
        except Exception:
            pass
        setattr(self, client_attr, None)

        self._terminate_process(process_attr)
        if owned:
            self._cleanup_ipc_file(endpoint)

    def _terminate_process(self, process_attr: str) -> None:
        process: subprocess.Popen | None = getattr(self, process_attr)
        if not process:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        setattr(self, process_attr, None)

    def _cleanup_ipc_file(self, endpoint: str | None) -> None:
        if not endpoint or not endpoint.startswith("ipc://"):
            return
        path = endpoint[len("ipc://") :]
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
