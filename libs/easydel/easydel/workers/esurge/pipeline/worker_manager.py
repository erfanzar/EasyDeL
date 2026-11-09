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

"""Lifecycle for the tokenizer subprocess and native in-process detokenizer."""

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

from surgecore import Detokenizer

from easydel.utils import flags
from easydel.workers.loggers import get_logger

from .zmq_workers import TokenizerWorkerClient

logger = get_logger(__name__)

DEFAULT_WORKER_STARTUP_TIMEOUT = float(
    next(spec.default for spec in flags.registered_flags() if spec.name == flags.EASYDEL_WORKER_STARTUP_TIMEOUT)
)


class WorkerManager:
    """Own one tokenizer ZMQ worker and one native batched detokenizer.

    The tokenizer may be attached to an external endpoint. Detokenization always runs in this process through
    :class:`surgecore.Detokenizer`; there is no detokenizer endpoint, subprocess or alternate implementation.
    """

    def __init__(
        self,
        tokenizer_source: str | None,
        *,
        tokenizer_kwargs: dict[str, Any] | None = None,
        startup_timeout: float | None = None,
        ipc_dir: str | None = None,
    ) -> None:
        """Store tokenizer loading and subprocess lifecycle configuration."""
        self._tokenizer_source = tokenizer_source
        self._tokenizer_kwargs = tokenizer_kwargs or {}
        self._startup_timeout = self._resolve_startup_timeout(startup_timeout)
        self._ipc_dir = ipc_dir or tempfile.gettempdir()
        self._tokenizer_client: TokenizerWorkerClient | None = None
        self._detokenizer_client: Detokenizer | None = None
        self._tokenizer_process: subprocess.Popen | None = None
        self._tokenizer_owned = False
        self._tokenizer_endpoint: str | None = None

    @property
    def tokenizer_endpoint(self) -> str | None:
        """The owned or attached tokenizer endpoint, or ``None`` before startup."""
        return self._tokenizer_endpoint

    def start(
        self,
        *,
        detokenizer_max_states: int,
        tokenizer_endpoint: str | None,
    ) -> tuple[TokenizerWorkerClient, Detokenizer]:
        """Start the tokenizer client and construct the native detokenizer."""
        if self._tokenizer_client or self._detokenizer_client:
            raise RuntimeError("WorkerManager has already started clients.")
        if not self._tokenizer_source:
            raise ValueError("Tokenizer identifier must be provided.")

        if tokenizer_endpoint is None:
            tokenizer_endpoint = self._make_ipc_endpoint("tokenizer")
            self._tokenizer_process = self._spawn_worker(tokenizer_endpoint)
            self._tokenizer_owned = True
            try:
                self._wait_for_endpoint("tokenizer", tokenizer_endpoint, self._tokenizer_process)
            except Exception:
                self._terminate_tokenizer_process()
                self._cleanup_ipc_file(tokenizer_endpoint)
                self._tokenizer_owned = False
                self._tokenizer_endpoint = None
                raise
        self._tokenizer_endpoint = tokenizer_endpoint

        try:
            self._tokenizer_client = TokenizerWorkerClient(tokenizer_endpoint)
            tokenizer_kwargs = dict(self._tokenizer_kwargs)
            if "trust_remote_code" not in tokenizer_kwargs:
                tokenizer_kwargs["trust_remote_code"] = flags.get_bool(flags.ESURGE_WORKER_TRUST_REMOTE_CODE)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_source, **tokenizer_kwargs)
            self._detokenizer_client = Detokenizer(
                tokenizer,
                context_window=max(0, flags.get_int(flags.ESURGE_DETOKENIZER_CONTEXT_WINDOW)),
                max_states=detokenizer_max_states,
            )
        except Exception:
            if self._tokenizer_client is not None:
                self._tokenizer_client.close()
                self._tokenizer_client = None
            self._terminate_tokenizer_process()
            if self._tokenizer_owned:
                self._cleanup_ipc_file(tokenizer_endpoint)
            self._tokenizer_owned = False
            self._tokenizer_endpoint = None
            raise
        return self._tokenizer_client, self._detokenizer_client

    def shutdown(self) -> None:
        """Release native decode state and stop an owned tokenizer worker."""
        if self._detokenizer_client is not None:
            self._detokenizer_client.shutdown()
            self._detokenizer_client = None
        if self._tokenizer_client is not None:
            try:
                if self._tokenizer_owned:
                    self._tokenizer_client.shutdown()
                else:
                    self._tokenizer_client.close()
            except Exception:
                pass
            self._tokenizer_client = None
        self._terminate_tokenizer_process()
        if self._tokenizer_owned:
            self._cleanup_ipc_file(self._tokenizer_endpoint)
        self._tokenizer_owned = False
        self._tokenizer_endpoint = None

    def drain_workers(self) -> None:
        """Flush native detokenization state and acknowledge the stateless tokenizer worker."""
        for name, client in (
            ("tokenizer", self._tokenizer_client),
            ("detokenizer", self._detokenizer_client),
        ):
            if client is None:
                continue
            try:
                client.drain()
            except Exception as exc:
                logger.warning("Failed to drain %s worker: %s", name, exc)

    def _resolve_startup_timeout(self, startup_timeout: float | None) -> float:
        """Resolve and validate the tokenizer subprocess startup timeout."""
        if startup_timeout is not None:
            timeout = float(startup_timeout)
            if timeout <= 0:
                raise ValueError("startup_timeout must be greater than 0.")
            return timeout
        timeout = flags.get_float(flags.EASYDEL_WORKER_STARTUP_TIMEOUT)
        if timeout <= 0:
            logger.warning(
                "Ignoring invalid %s=%r; expected a positive timeout in seconds.",
                flags.EASYDEL_WORKER_STARTUP_TIMEOUT,
                timeout,
            )
            return DEFAULT_WORKER_STARTUP_TIMEOUT
        return timeout

    def _spawn_worker(self, endpoint: str) -> subprocess.Popen:
        """Launch the CPU-isolated tokenizer subprocess."""
        worker_main_path = Path(__file__).with_name("worker_main.py")
        cmd = [
            sys.executable,
            str(worker_main_path),
            "tokenizer",
            "--endpoint",
            endpoint,
            "--tokenizer-path",
            self._tokenizer_source,
            "--tokenizer-kwargs",
            json.dumps(self._tokenizer_kwargs),
        ]
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["ENABLE_DISTRIBUTED_INIT"] = "0"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env.setdefault("PYTHONUNBUFFERED", "1")
        return subprocess.Popen(cmd, env=env)

    def _wait_for_endpoint(self, worker_name: str, endpoint: str, process: subprocess.Popen | None) -> None:
        """Block until the tokenizer binds or the configured deadline expires."""
        deadline = time.time() + self._startup_timeout
        path = endpoint[len("ipc://") :] if endpoint.startswith("ipc://") else None
        while time.time() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(
                    f"{worker_name.capitalize()} worker process for {endpoint} exited with code {process.returncode}"
                )
            if path and os.path.exists(path):
                return
            time.sleep(0.05)
        pid_text = f" (pid={process.pid})" if process is not None and process.poll() is None else ""
        raise TimeoutError(
            f"Timed out waiting {self._startup_timeout:.1f}s for {worker_name} worker to bind to {endpoint}{pid_text}. "
            "Increase `worker_startup_timeout` or set `EASYDEL_WORKER_STARTUP_TIMEOUT`."
        )

    def _make_ipc_endpoint(self, prefix: str) -> str:
        """Build a unique IPC endpoint in the configured temporary directory."""
        os.makedirs(self._ipc_dir, exist_ok=True)
        file_path = os.path.join(self._ipc_dir, f"easydel_{prefix}_{uuid.uuid4().hex}.sock")
        return f"ipc://{file_path}"

    def _terminate_tokenizer_process(self) -> None:
        """Terminate an owned tokenizer subprocess, escalating after five seconds."""
        process = self._tokenizer_process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self._tokenizer_process = None

    @staticmethod
    def _cleanup_ipc_file(endpoint: str | None) -> None:
        """Remove an owned IPC socket file if it still exists."""
        if not endpoint or not endpoint.startswith("ipc://"):
            return
        try:
            os.unlink(endpoint[len("ipc://") :])
        except FileNotFoundError:
            pass
