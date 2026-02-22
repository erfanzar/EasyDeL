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

"""Leader-side ZeroMQ RPC client for communicating with a single eSurge worker.

The leader host creates one :class:`WorkerRpcClient` per remote worker rank.
Each client wraps a ZeroMQ REQ socket and provides typed methods for the
commands defined in :mod:`~.protocol`.  Step execution is split into a
non-blocking :meth:`~WorkerRpcClient.begin_step` (send) and a blocking
:meth:`~WorkerRpcClient.finish_step` (recv) so that the leader can dispatch
work to all workers concurrently before waiting for results.

Classes:
    WorkerRpcClient: ZeroMQ REQ client for one worker endpoint.
"""

from __future__ import annotations

import typing as tp

import zmq

from .protocol import CMD_HEALTH, CMD_HELLO, CMD_SHUTDOWN, CMD_STEP


class WorkerRpcClient:
    """Leader-side ZeroMQ REQ client for a single worker control-plane endpoint.

    Manages a persistent connection to one worker's
    :class:`~.worker_server.WorkerControlServer` and exposes the full
    control-plane command set.

    Step execution follows a two-phase pattern:

    1. :meth:`begin_step` sends the scheduler output to the worker
       (non-blocking on the leader side).
    2. :meth:`finish_step` blocks until the worker replies with its step
       result (sampled digest, timing, etc.).

    This split allows the leader to fan-out ``begin_step`` to all workers
    before collecting results, minimising idle time.

    Args:
        endpoint: ``tcp://<host>:<port>`` address of the worker's REP socket.
        auth_token: Shared secret for authenticating requests.
        connect_timeout_s: Timeout in seconds for the initial ZeroMQ connection
            and for non-step request/response round-trips.
        step_timeout_s: Timeout in seconds for waiting on step completion
            responses (typically longer than *connect_timeout_s*).
    """

    def __init__(
        self,
        *,
        endpoint: str,
        auth_token: str,
        connect_timeout_s: float,
        step_timeout_s: float,
    ) -> None:
        self.endpoint = str(endpoint)
        self._auth_token = str(auth_token)
        self._step_timeout_ms = max(1, int(float(step_timeout_s) * 1000.0))
        self._connect_timeout_ms = max(1, int(float(connect_timeout_s) * 1000.0))

        self._context = zmq.Context.instance()
        self._socket = self._create_socket()

        self._inflight_step_id: int | None = None

    def _create_socket(self) -> zmq.Socket:
        """Create and connect a configured REQ socket for this worker."""
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self._connect_timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self._step_timeout_ms)
        socket.connect(self.endpoint)
        return socket

    @property
    def has_inflight_step(self) -> bool:
        """Whether a step request has been sent and not yet collected."""
        return self._inflight_step_id is not None

    def _request(self, payload: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Send *payload* and block until a dict response is received.

        Raises:
            RuntimeError: If the worker returns a non-dict response.
            zmq.error.Again: If the send or receive times out.
        """
        self._socket.send_pyobj(payload)
        response = self._socket.recv_pyobj()
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid worker response type from {self.endpoint}: {type(response)!r}")
        return response

    def hello(self) -> dict[str, tp.Any]:
        """Perform the initial handshake with the worker.

        Returns:
            Worker metadata including ``rank``, ``world_size``, and
            ``config_fingerprint``.
        """
        return self._request({"cmd": CMD_HELLO, "auth_token": self._auth_token})

    def health(self) -> dict[str, tp.Any]:
        """Query the worker's health status.

        Returns:
            Health information including ``steps_executed``, ``last_step_id``,
            and ``last_error``.
        """
        return self._request({"cmd": CMD_HEALTH, "auth_token": self._auth_token})

    def begin_step(self, *, step_id: int, scheduler_output: tp.Any) -> None:
        """Send a step command to the worker without waiting for the response.

        The caller must subsequently invoke :meth:`finish_step` to collect the
        worker's result.  Only one step may be in-flight at a time.

        Args:
            step_id: Monotonically increasing identifier for this step.
            scheduler_output: Serializable scheduler output to forward to the
                worker's ``execute_step`` callback.

        Raises:
            RuntimeError: If a previous step has not been collected yet.
        """
        if self._inflight_step_id is not None:
            raise RuntimeError(
                f"Worker {self.endpoint} already has in-flight step {self._inflight_step_id}; "
                f"cannot send step {step_id}."
            )
        self._socket.send_pyobj(
            {
                "cmd": CMD_STEP,
                "auth_token": self._auth_token,
                "step_id": int(step_id),
                "scheduler_output": scheduler_output,
            }
        )
        self._inflight_step_id = int(step_id)

    def finish_step(self) -> dict[str, tp.Any]:
        """Block until the worker completes the in-flight step and return its result.

        Returns:
            Worker response dict containing ``status``, ``step_id``,
            ``sampled_digest``, ``num_reqs``, and ``timing_ms``.

        Raises:
            RuntimeError: If no step is currently in-flight or the worker
                returns an invalid response type.
        """
        if self._inflight_step_id is None:
            raise RuntimeError(f"No in-flight step for worker {self.endpoint}")
        try:
            response = self._socket.recv_pyobj()
        finally:
            self._inflight_step_id = None

        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid worker step response type from {self.endpoint}: {type(response)!r}")
        return response

    def shutdown(self) -> None:
        """Send a graceful shutdown command to the worker.

        Exceptions are silently ignored since the worker may already be down.
        """
        try:
            self._request({"cmd": CMD_SHUTDOWN, "auth_token": self._auth_token})
        except Exception:
            pass

    def reset_connection(self) -> None:
        """Reset the underlying socket and clear in-flight state.

        Used after transport/protocol errors that can leave a REQ socket in an
        unrecoverable send/recv state.
        """
        try:
            self._socket.close(0)
        finally:
            self._socket = self._create_socket()
            self._inflight_step_id = None

    def close(self) -> None:
        """Close the underlying ZeroMQ socket immediately."""
        self._socket.close(0)
