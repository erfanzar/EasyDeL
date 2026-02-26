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

"""High-level distributed controller for multi-host eSurge inference.

This module ties together DNS discovery, the leader-side RPC clients, and the
worker-side control server into a single :class:`DistributedController` that
the eSurge engine uses to coordinate lockstep inference across hosts.

Lifecycle:
    1. The engine creates a :class:`DistributedController` with the cluster
       parameters from :class:`~easydel.inference.esurge.config.Config`.
    2. :meth:`DistributedController.start` resolves the cluster via DNS,
       starts the worker server (on non-leader ranks), and performs the
       leader handshake with every worker.
    3. On each engine step the leader calls :meth:`~DistributedController.dispatch_step`
       to fan-out the scheduler output, runs the model locally, then calls
       :meth:`~DistributedController.verify_step` to collect and validate
       worker results.
    4. :meth:`DistributedController.shutdown` tears down all connections.

Classes:
    StepDispatch: Lightweight token representing a dispatched step.
    DistributedController: Main coordinator for leader/worker lockstep execution.

Functions:
    resolve_distributed_role: Validates and resolves the ``"auto"`` / ``"leader"``
        / ``"worker"`` role string for a given rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..logger import logger
from .discovery import DiscoveryResult, resolve_service_hosts
from .leader_client import WorkerRpcClient
from .protocol import STATUS_OK, compute_sampled_digest
from .worker_server import WorkerControlServer


@dataclass(frozen=True)
class StepDispatch:
    """Lightweight token representing one distributed step dispatched to workers.

    Returned by :meth:`DistributedController.dispatch_step` and later passed
    to :meth:`DistributedController.verify_step` so the controller can match
    worker responses to the correct step.

    Attributes:
        step_id: Monotonically increasing step identifier.
    """

    step_id: int


def resolve_distributed_role(role: str, rank: int) -> str:
    """Resolve and validate the distributed role for a given rank.

    Args:
        role: One of ``"auto"``, ``"leader"``, or ``"worker"``.  When
            ``"auto"`` is specified, rank 0 becomes the leader and all other
            ranks become workers.
        rank: The host's rank within the cluster.

    Returns:
        The resolved role string (``"leader"`` or ``"worker"``).

    Raises:
        ValueError: If *role* is not one of the accepted values, or if an
            explicit role conflicts with the given rank (e.g. ``"leader"``
            with ``rank != 0``).
    """

    normalized = str(role).strip().lower()
    if normalized not in {"auto", "leader", "worker"}:
        raise ValueError(f"Invalid `distributed_role`: {role!r}")
    if normalized == "auto":
        return "leader" if int(rank) == 0 else "worker"
    if normalized == "leader" and int(rank) != 0:
        raise ValueError("`distributed_role='leader'` requires rank 0.")
    if normalized == "worker" and int(rank) == 0:
        raise ValueError("`distributed_role='worker'` cannot be used with rank 0.")
    return normalized


class DistributedController:
    """Coordinates leader/worker lockstep execution for distributed eSurge serving.

    On the **leader** rank the controller:

    * Resolves the cluster via DNS (see :mod:`~.discovery`).
    * Opens a :class:`~.leader_client.WorkerRpcClient` to each worker and
      validates config fingerprints, ranks, and world sizes during the
      handshake.
    * Dispatches scheduler outputs to all workers at the start of each step,
      then collects and verifies their results.

    On **worker** ranks the controller:

    * Starts a :class:`~.worker_server.WorkerControlServer` that receives
      commands from the leader and invokes the local model runner.

    Args:
        enabled: Whether distributed mode is active.
        role: ``"leader"`` or ``"worker"`` (already resolved by
            :func:`resolve_distributed_role`).
        rank: This host's rank within the cluster.
        world_size: Total number of hosts expected.
        service_name: DNS name to resolve for cluster discovery.
        control_port: TCP port for the ZeroMQ control-plane sockets.
        control_bind_host: Interface address for the worker server to bind on.
        advertise_addr: Address advertised to the leader; auto-detected from
            DNS if ``None``.
        auth_token: Shared secret for control-plane authentication.
        step_timeout_s: Timeout in seconds for step completion on workers.
        connect_timeout_s: Timeout in seconds for initial connections.
        verify_sampling_digest: If ``True``, the leader compares sampled-token
            digests across all hosts after each step.
        config_fingerprint: SHA-256 config digest for compatibility checks.
        execute_step: Callback for workers to execute a model step; ``None``
            on the leader rank.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        role: str,
        rank: int,
        world_size: int,
        service_name: str | None,
        control_port: int,
        control_bind_host: str,
        advertise_addr: str | None,
        auth_token: str,
        step_timeout_s: float,
        connect_timeout_s: float,
        verify_sampling_digest: bool,
        config_fingerprint: str,
        execute_step: Any | None,
    ) -> None:
        self.enabled = bool(enabled)
        self.role = str(role)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.service_name = service_name
        self.control_port = int(control_port)
        self.control_bind_host = str(control_bind_host)
        self.advertise_addr = advertise_addr
        self.auth_token = str(auth_token)
        self.step_timeout_s = float(step_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.verify_sampling_digest = bool(verify_sampling_digest)
        self.config_fingerprint = str(config_fingerprint)
        self._execute_step = execute_step

        self._started = False
        self._step_counter = 0
        self._discovery: DiscoveryResult | None = None
        self._worker_server: WorkerControlServer | None = None
        self._worker_clients: dict[int, WorkerRpcClient] = {}

    @property
    def is_leader(self) -> bool:
        """Whether this controller is the cluster leader."""
        return self.enabled and self.role == "leader"

    @property
    def is_worker(self) -> bool:
        """Whether this controller is a cluster worker."""
        return self.enabled and self.role == "worker"

    @property
    def has_remote_workers(self) -> bool:
        """Whether the leader has established connections to remote workers."""
        return bool(self._worker_clients)

    def start(self) -> None:
        """Initialize the distributed cluster.

        For **workers**: starts the :class:`~.worker_server.WorkerControlServer`.

        For the **leader**: resolves the cluster topology via DNS, connects to
        every worker, and validates config fingerprints, ranks, and world sizes
        via the ``hello`` handshake.

        Raises:
            ValueError: If service name is missing, rank is out of range,
                a worker handshake fails, or config/rank/world-size mismatches
                are detected.
        """
        if not self.enabled or self._started:
            return

        if self.service_name is None:
            raise ValueError("`distributed_service_name` must be provided when distributed_mode=True")

        self._discovery = resolve_service_hosts(self.service_name, world_size=self.world_size)
        hosts = self._discovery.hosts

        if self.rank < 0 or self.rank >= len(hosts):
            raise ValueError(f"Invalid distributed rank {self.rank} for hosts={hosts}")

        if self.advertise_addr is None:
            self.advertise_addr = hosts[self.rank]

        if self.is_worker:
            if self._execute_step is None:
                raise ValueError("Worker distributed role requires an execute_step callback")

            self._worker_server = WorkerControlServer(
                bind_host=self.control_bind_host,
                port=self.control_port,
                auth_token=self.auth_token,
                rank=self.rank,
                world_size=self.world_size,
                config_fingerprint=self.config_fingerprint,
                execute_step=self._execute_step,
            )
            self._worker_server.start()
            logger.info(
                "Started distributed worker control server rank=%s endpoint=%s",
                self.rank,
                self._worker_server.endpoint,
            )

        if self.is_leader:
            for worker_rank, host in enumerate(hosts):
                if worker_rank == self.rank:
                    continue
                endpoint = f"tcp://{host}:{self.control_port}"
                client = WorkerRpcClient(
                    endpoint=endpoint,
                    auth_token=self.auth_token,
                    connect_timeout_s=self.connect_timeout_s,
                    step_timeout_s=self.step_timeout_s,
                )
                hello = client.hello()

                if hello.get("status") != STATUS_OK:
                    client.close()
                    raise ValueError(
                        f"Distributed worker handshake failed: rank={worker_rank} endpoint={endpoint} response={hello}"
                    )

                worker_fp = str(hello.get("config_fingerprint", ""))
                if worker_fp != self.config_fingerprint:
                    client.close()
                    raise ValueError(
                        "Distributed worker config mismatch: "
                        f"rank={worker_rank} endpoint={endpoint} "
                        f"worker_fp={worker_fp} leader_fp={self.config_fingerprint}"
                    )

                returned_rank = int(hello.get("rank", -1))
                if returned_rank != worker_rank:
                    client.close()
                    raise ValueError(
                        "Distributed worker rank mismatch: "
                        f"expected={worker_rank} got={returned_rank} endpoint={endpoint}"
                    )

                returned_world = int(hello.get("world_size", -1))
                if returned_world != self.world_size:
                    client.close()
                    raise ValueError(
                        "Distributed worker world-size mismatch: "
                        f"expected={self.world_size} got={returned_world} endpoint={endpoint}"
                    )

                self._worker_clients[worker_rank] = client

            logger.info(
                "Connected distributed leader rank=%s to %s workers (world_size=%s)",
                self.rank,
                len(self._worker_clients),
                self.world_size,
            )

        self._started = True

    def dispatch_step(self, scheduler_output: Any) -> StepDispatch | None:
        """Fan-out a scheduler output to all connected workers.

        Sends a ``begin_step`` command to every worker with the given
        *scheduler_output*.  The caller should then execute the model step
        locally before calling :meth:`verify_step` to collect worker results.

        Args:
            scheduler_output: The scheduler output payload to broadcast.

        Returns:
            A :class:`StepDispatch` token if workers were dispatched, or
            ``None`` if there are no remote workers (single-host mode).

        Raises:
            ValueError: If dispatching to any worker fails.
        """
        if not self.is_leader or not self._worker_clients:
            return None

        self._step_counter += 1
        step_id = self._step_counter
        rank = -1
        dispatched_clients: list[tuple[int, WorkerRpcClient]] = []
        current_client: WorkerRpcClient | None = None
        try:
            for rank_loop, client in self._worker_clients.items():
                rank = rank_loop
                current_client = client
                client.begin_step(step_id=step_id, scheduler_output=scheduler_output)
                dispatched_clients.append((rank_loop, client))
        except Exception as exc:
            # Best-effort recovery: drain already-dispatched workers, then
            # reset sockets so REQ/REP state does not wedge future steps.
            for worker_rank, dispatched_client in dispatched_clients:
                try:
                    dispatched_client.finish_step()
                except Exception:
                    pass
                try:
                    dispatched_client.reset_connection()
                except Exception:
                    logger.debug(
                        "Failed to reset worker RPC connection after dispatch failure rank=%s",
                        worker_rank,
                        exc_info=True,
                    )

            if current_client is not None and all(current_client is not c for _, c in dispatched_clients):
                try:
                    current_client.reset_connection()
                except Exception:
                    logger.debug(
                        "Failed to reset failed worker RPC connection after dispatch failure rank=%s",
                        rank,
                        exc_info=True,
                    )
            raise ValueError(
                f"Distributed step synchronization failure: failed to dispatch step_id={step_id} to worker_rank={rank}"
            ) from exc

        return StepDispatch(step_id=step_id)

    def verify_step(self, dispatch: StepDispatch | None, model_output: Any) -> None:
        """Collect and validate worker results for a previously dispatched step.

        Blocks until every worker has replied.  Verifies that each worker's
        response matches the expected step ID and request count.  If
        ``verify_sampling_digest`` is enabled, also checks that the sampled-token
        digest matches the leader's local result.

        Args:
            dispatch: The token returned by :meth:`dispatch_step`, or ``None``
                to no-op (single-host mode).
            model_output: The leader's own model output, used as the reference
                for digest and request-count comparisons.

        Raises:
            ValueError: If any worker fails, returns a mismatched step ID or
                request count, or produces a divergent sampling digest.
        """
        if dispatch is None or not self._worker_clients:
            return

        req_ids = [str(rid) for rid in list(getattr(model_output, "req_ids", []))]
        sampled_token_ids = [[int(tok) for tok in row] for row in list(getattr(model_output, "sampled_token_ids", []))]
        expected_digest = compute_sampled_digest(req_ids, sampled_token_ids)
        expected_num_reqs = len(req_ids)

        for worker_rank, client in self._worker_clients.items():
            try:
                response = client.finish_step()
            except Exception as exc:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} step_id={dispatch.step_id} did not respond"
                ) from exc

            if response.get("status") != STATUS_OK:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} step_id={dispatch.step_id} error={response.get('error')}"
                )

            worker_step_id = int(response.get("step_id", -1))
            if worker_step_id != dispatch.step_id:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} expected_step_id={dispatch.step_id} got={worker_step_id}"
                )

            worker_num_reqs = int(response.get("num_reqs", -1))
            if worker_num_reqs != expected_num_reqs:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} step_id={dispatch.step_id} "
                    f"expected_num_reqs={expected_num_reqs} got={worker_num_reqs}"
                )

            if self.verify_sampling_digest:
                worker_digest = str(response.get("sampled_digest", ""))
                if worker_digest != expected_digest:
                    raise ValueError(
                        "Distributed step synchronization failure: "
                        f"worker_rank={worker_rank} step_id={dispatch.step_id} digest mismatch"
                    )

    def shutdown(self) -> None:
        """Tear down the distributed control-plane.

        Stops the local worker server (if running), sends shutdown commands
        to all connected workers, closes RPC clients, and resets internal state.
        Errors during cleanup are logged at debug level but do not propagate.
        """
        if self._worker_server is not None:
            try:
                self._worker_server.stop()
            except Exception:
                logger.debug("Failed to stop worker control server cleanly", exc_info=True)
            self._worker_server = None

        if self._worker_clients:
            for client in self._worker_clients.values():
                try:
                    client.shutdown()
                except Exception:
                    logger.debug("Failed to shutdown worker client cleanly", exc_info=True)
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass
            self._worker_clients.clear()

        self._started = False
