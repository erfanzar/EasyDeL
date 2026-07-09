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

"""Step coordination between the engine loop and the runner.

The engine loop never calls the runner directly; every step goes through a
:class:`StepCoordinator`. On a single host the :class:`LocalCoordinator` is
an identity pass-through with zero overhead. On a multi-host pod the
coordinator's job is to replicate the leader's runner-call stream — every
``execute_sync``/``execute_async``/``drain`` in the same order — to every
worker process, because the jitted step is a global-mesh collective that
deadlocks unless all processes enter it together.

Correctness rests on determinism: given an identical ordered stream of
runner invocations with identical ``SchedulerOutput`` payloads, every
host's runner state is bit-identical by construction (device outputs use
replicated sharding; host-side repairs are deterministic functions of
them). Acks, digests, and heartbeats exist for failure *detection*, never
for correctness.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import threading

    from ..outputs import ModelRunnerOutput
    from ..scheduler.output import SchedulerOutput


class StepCoordinationError(RuntimeError):
    """Non-recoverable lockstep failure.

    Raised when a peer host NACKs a step, misses an ack deadline, stops
    heartbeating, or reports a sampled-token digest mismatch. A JAX
    collective cannot survive a lost participant, so the engine loop must
    abort in-flight requests and shut down — never retry the step.
    """


@dataclass
class StepHandle:
    """An in-flight asynchronously-dispatched step.

    Attributes:
        step_id: Monotonic step index assigned by the coordinator.
        runner_handle: The runner's async execution handle.
        scheduler_output: The step plan that produced this dispatch.
    """

    step_id: int
    runner_handle: typing.Any
    scheduler_output: SchedulerOutput


class StepCoordinator(typing.Protocol):
    """The engine loop's only gateway to step execution.

    Implementations must preserve the invocation order they receive: the
    sequence of ``execute_sync``/``execute_async``/``drain`` calls *is* the
    replicated program under multi-host operation.
    """

    is_leader: bool
    rank: int
    world_size: int
    supports_overlap: bool

    def start(self) -> None:
        """Establish the control plane (single host: no-op)."""
        ...

    def execute_sync(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute one step synchronously on every host."""
        ...

    def execute_async(self, scheduler_output: SchedulerOutput) -> StepHandle:
        """Dispatch one step asynchronously on every host."""
        ...

    def drain(self, handle: StepHandle) -> ModelRunnerOutput:
        """Materialize a previously dispatched step's output."""
        ...

    def check_health(self) -> None:
        """Raise :class:`StepCoordinationError` if lockstep is broken."""
        ...

    def run_worker_loop(self, stop_event: threading.Event) -> None:
        """Block replaying the leader's step stream (worker ranks only)."""
        ...

    def shutdown(self, reason: str = "") -> None:
        """Tear the control plane down (single host: no-op)."""
        ...


class LocalCoordinator:
    """Single-host coordinator: identity pass-through to the runner.

    Args:
        runner: The engine's :class:`eSurgeRunner`.
    """

    is_leader = True
    rank = 0
    world_size = 1
    supports_overlap = True

    def __init__(self, runner) -> None:
        self._runner = runner
        self._next_step_id = 0

    def start(self) -> None:
        """No control plane on a single host."""

    def execute_sync(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Run the step synchronously on the local runner."""
        self._next_step_id += 1
        return self._runner.execute_model(scheduler_output)

    def execute_async(self, scheduler_output: SchedulerOutput) -> StepHandle:
        """Dispatch the step asynchronously on the local runner."""
        self._next_step_id += 1
        return StepHandle(
            step_id=self._next_step_id,
            runner_handle=self._runner.execute_model_async(scheduler_output),
            scheduler_output=scheduler_output,
        )

    def drain(self, handle: StepHandle) -> ModelRunnerOutput:
        """Wait for the dispatched step's host-visible output."""
        return self._runner.wait_for_execution(handle.runner_handle)

    def check_health(self) -> None:
        """Single host: always healthy."""

    def run_worker_loop(self, stop_event) -> None:
        """Single host has no worker ranks."""
        raise StepCoordinationError("LocalCoordinator has no worker loop (world_size == 1).")

    def shutdown(self, reason: str = "") -> None:
        """No control plane to tear down."""


def _default_leader_addr() -> str | None:
    """Best-effort leader address from the JAX distributed runtime.

    In standard pod launches process 0 hosts the JAX coordinator, so its
    address is a correct default for the eSurge control-plane leader.
    """
    try:
        from jax._src.distributed import global_state

        addr = getattr(global_state, "coordinator_address", None)
        if addr:
            return str(addr).rsplit(":", 1)[0]
    except Exception:
        return None
    return None


def create_step_coordinator(
    runner,
    *,
    distributed_config,
    config_fingerprint: str | None,
):
    """Build the right coordinator for this process.

    Selection order:

    1. ``coordination="zmq"`` with ``jax.process_count() > 1`` builds the
       ZeroMQ leader (rank 0) or worker plane.
    2. Everything else — single host, or the multi-host *replicated* pattern
       where an outer driver (e.g. a trainer's rollout loop) calls the
       engine identically on every host — gets the pass-through
       :class:`LocalCoordinator`.

    Args:
        runner: The engine's runner.
        distributed_config: The engine's distributed config section.
        config_fingerprint: Engine-config fingerprint for the handshake
            (required for the ZMQ plane).

    Returns:
        A :class:`StepCoordinator` implementation.

    Raises:
        StepCoordinationError: If ``coordination="zmq"`` is requested but
            the auth token or leader address cannot be resolved.
    """
    import jax

    world_size = int(jax.process_count())
    coordination = str(getattr(distributed_config, "coordination", "replicated") or "replicated")
    if world_size <= 1 or coordination != "zmq":
        return LocalCoordinator(runner)

    from .zmq_coordinator import ZmqLeaderCoordinator, ZmqWorkerCoordinator

    auth_token = distributed_config.distributed_auth_token
    if not auth_token:
        raise StepCoordinationError(
            "coordination='zmq' requires `distributed_auth_token` (the same value on every host)."
        )
    fingerprint = config_fingerprint or ""
    rank = int(jax.process_index())
    control_port = int(distributed_config.distributed_control_port)
    common = dict(
        auth_token=str(auth_token),
        config_fingerprint=fingerprint,
        heartbeat_interval_s=float(distributed_config.distributed_heartbeat_interval_s),
        heartbeat_timeout_s=float(distributed_config.distributed_heartbeat_timeout_s),
    )
    if rank == 0:
        return ZmqLeaderCoordinator(
            runner,
            world_size=world_size,
            bind_host=str(distributed_config.distributed_control_bind_host),
            control_port=control_port,
            ready_timeout_s=float(distributed_config.distributed_ready_timeout_s),
            step_timeout_s=float(distributed_config.distributed_step_timeout_s),
            verify_digest_interval=int(distributed_config.distributed_verify_digest_interval),
            max_inflight_steps=int(distributed_config.distributed_max_inflight_steps),
            **common,
        )

    leader_addr = distributed_config.distributed_leader_addr or _default_leader_addr()
    if not leader_addr and distributed_config.distributed_service_name:
        from .discovery import resolve_service_hosts

        leader_addr = resolve_service_hosts(
            distributed_config.distributed_service_name, world_size=world_size
        ).hosts[0]
    if not leader_addr:
        raise StepCoordinationError(
            "coordination='zmq' worker cannot resolve the leader address: set "
            "`distributed_leader_addr`, `distributed_service_name`, or initialize jax.distributed."
        )
    return ZmqWorkerCoordinator(
        runner,
        rank=rank,
        world_size=world_size,
        leader_addr=str(leader_addr),
        control_port=control_port,
        connect_timeout_s=float(distributed_config.distributed_connect_timeout_s),
        **common,
    )
