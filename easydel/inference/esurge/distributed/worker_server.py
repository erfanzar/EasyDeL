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

"""Worker-side ZeroMQ control-plane server for distributed eSurge serving.

Each non-leader host in a distributed eSurge cluster runs a
:class:`WorkerControlServer` that listens for commands from the leader's
:class:`~.leader_client.WorkerRpcClient`.  The server handles the full
command vocabulary defined in :mod:`~.protocol` (``hello``, ``health``,
``step``, ``shutdown``) and executes inference steps in lockstep with the
leader.

The server runs in a dedicated daemon thread and communicates over a ZeroMQ
REP socket bound to a configurable ``host:port``.  All requests must include
a valid ``auth_token`` or they are rejected.

Classes:
    WorkerControlServer: Thread-based ZeroMQ REP server for worker control.
"""

from __future__ import annotations

import threading
import time
import traceback
import typing as tp

import zmq

from .protocol import CMD_HEALTH, CMD_HELLO, CMD_SHUTDOWN, CMD_STEP, STATUS_ERROR, STATUS_OK, compute_sampled_digest


class WorkerControlServer:
    """ZeroMQ REP server running on a worker rank for lockstep step execution.

    The server listens on a ``tcp://<bind_host>:<port>`` endpoint and
    processes the following commands from the leader:

    * **hello** — Returns rank, world size, and config fingerprint for the
      leader to verify cluster consistency.
    * **health** — Returns liveness information including step counts and the
      last error (if any).
    * **step** — Executes a single inference step via the *execute_step*
      callback, computes a sampled digest for lockstep verification, and
      returns timing / result metadata.
    * **shutdown** — Gracefully terminates the server loop.

    Args:
        bind_host: Interface address to bind the ZeroMQ socket on.
        port: TCP port number.
        auth_token: Shared secret used to authenticate leader requests.
        rank: This worker's rank within the cluster.
        world_size: Total number of hosts in the cluster.
        config_fingerprint: SHA-256 digest of the engine config (see
            :func:`~.protocol.make_config_fingerprint`).
        execute_step: Callback invoked for each ``step`` command.  Receives
            the ``scheduler_output`` payload and must return a model output
            object with ``req_ids`` and ``sampled_token_ids`` attributes.
    """

    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        auth_token: str,
        rank: int,
        world_size: int,
        config_fingerprint: str,
        execute_step: tp.Callable[[tp.Any], tp.Any],
    ) -> None:
        self._bind_host = str(bind_host)
        self._port = int(port)
        self._auth_token = str(auth_token)
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._config_fingerprint = str(config_fingerprint)
        self._execute_step = execute_step

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._steps_executed = 0
        self._last_step_id = -1
        self._last_error: str | None = None

    @property
    def endpoint(self) -> str:
        """The ``tcp://<host>:<port>`` address this server is bound to."""
        return f"tcp://{self._bind_host}:{self._port}"

    @property
    def is_running(self) -> bool:
        """Whether the server thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the server thread and block until the socket is ready.

        Raises:
            RuntimeError: If the server fails to become ready within 10 seconds.
        """
        if self.is_running:
            return

        self._stop_event.clear()
        self._ready_event.clear()

        self._thread = threading.Thread(target=self._serve_loop, name=f"esurge-worker-rpc-{self._rank}", daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=10.0):
            raise RuntimeError(f"Worker control server failed to start on {self.endpoint}")

    def stop(self) -> None:
        """Gracefully stop the server by sending a shutdown command to itself.

        Waits up to 5 seconds for the server thread to join.  If the thread
        does not terminate in time it is abandoned (it is a daemon thread).
        """
        thread = self._thread
        if thread is None:
            return

        self._stop_event.set()

        # Try graceful shutdown via control command to break recv loop.
        ctx = zmq.Context.instance()
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.SNDTIMEO, 1_000)
        req.setsockopt(zmq.RCVTIMEO, 1_000)
        try:
            req.connect(self.endpoint)
            req.send_pyobj({"cmd": CMD_SHUTDOWN, "auth_token": self._auth_token})
            req.recv_pyobj()
        except Exception:
            pass
        finally:
            req.close(0)

        thread.join(timeout=5.0)
        self._thread = None

    def _is_authorized(self, message: dict[str, tp.Any]) -> bool:
        """Check whether *message* carries the expected auth token."""
        return str(message.get("auth_token", "")) == self._auth_token

    @staticmethod
    def _extract_runner_output(model_output: tp.Any) -> tuple[list[str], list[list[int]]]:
        """Extract ``(req_ids, sampled_token_ids)`` from a model output.

        Supports both dict-style and attribute-style model outputs so the
        worker is agnostic to the concrete output type returned by the
        runner callback.

        Args:
            model_output: The return value of the *execute_step* callback.

        Returns:
            A tuple of ``(req_ids, sampled_token_ids)`` with normalised types.
        """
        if isinstance(model_output, dict):
            req_ids = list(model_output.get("req_ids", []))
            sampled_token_ids = list(model_output.get("sampled_token_ids", []))
            return [str(rid) for rid in req_ids], [[int(tok) for tok in row] for row in sampled_token_ids]

        req_ids = list(getattr(model_output, "req_ids", []))
        sampled_token_ids = list(getattr(model_output, "sampled_token_ids", []))
        return [str(rid) for rid in req_ids], [[int(tok) for tok in row] for row in sampled_token_ids]

    def _serve_loop(self) -> None:
        """Main server loop executed in the daemon thread.

        Binds a ZeroMQ REP socket and polls for incoming commands until
        :meth:`stop` sets the stop event or a ``shutdown`` command is received.
        """
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1_000)
        sock.bind(self.endpoint)
        self._ready_event.set()

        try:
            while not self._stop_event.is_set():
                try:
                    message = sock.recv_pyobj()
                except zmq.error.Again:
                    continue

                if not isinstance(message, dict):
                    sock.send_pyobj({"status": STATUS_ERROR, "error": "Invalid message payload."})
                    continue

                if not self._is_authorized(message):
                    sock.send_pyobj({"status": STATUS_ERROR, "error": "Unauthorized control-plane request."})
                    continue

                cmd = message.get("cmd")

                if cmd == CMD_HELLO:
                    sock.send_pyobj(
                        {
                            "status": STATUS_OK,
                            "rank": self._rank,
                            "world_size": self._world_size,
                            "config_fingerprint": self._config_fingerprint,
                        }
                    )
                    continue

                if cmd == CMD_HEALTH:
                    sock.send_pyobj(
                        {
                            "status": STATUS_OK,
                            "rank": self._rank,
                            "world_size": self._world_size,
                            "steps_executed": self._steps_executed,
                            "last_step_id": self._last_step_id,
                            "last_error": self._last_error,
                        }
                    )
                    continue

                if cmd == CMD_STEP:
                    step_id = int(message.get("step_id", -1))
                    scheduler_output = message.get("scheduler_output")
                    started = time.perf_counter()
                    try:
                        model_output = self._execute_step(scheduler_output)
                        req_ids, sampled_token_ids = self._extract_runner_output(model_output)
                        sampled_digest = compute_sampled_digest(req_ids, sampled_token_ids)
                        self._steps_executed += 1
                        self._last_step_id = step_id
                        self._last_error = None
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        sock.send_pyobj(
                            {
                                "status": STATUS_OK,
                                "step_id": step_id,
                                "sampled_digest": sampled_digest,
                                "num_reqs": len(req_ids),
                                "timing_ms": elapsed_ms,
                            }
                        )
                    except Exception as exc:
                        self._last_step_id = step_id
                        self._last_error = str(exc)
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        sock.send_pyobj(
                            {
                                "status": STATUS_ERROR,
                                "step_id": step_id,
                                "timing_ms": elapsed_ms,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                    continue

                if cmd == CMD_SHUTDOWN:
                    sock.send_pyobj({"status": STATUS_OK})
                    break

                sock.send_pyobj({"status": STATUS_ERROR, "error": f"Unknown command: {cmd}"})
        finally:
            sock.close(0)
            ctx.term()
