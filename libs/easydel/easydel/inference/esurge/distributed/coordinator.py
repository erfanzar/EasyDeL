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

import socket
import typing
from dataclasses import dataclass

from ..logger import logger

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


_LEADER_ADDR_KEY = "easydel/esurge/zmq_leader_addr/{port}"
_WILDCARD_BIND_HOSTS = frozenset({"", "*", "0.0.0.0", "::", "[::]"})


def _jax_distributed_state():
    """Return JAX's distributed ``global_state``, or ``None`` if JAX's internals are unavailable."""
    try:
        from jax._src.distributed import global_state
    except Exception:
        return None
    return global_state


def _split_host_port(address: str) -> tuple[str, int | None]:
    """Split ``host:port`` (an IPv6 host may be bracketed); the port is ``None`` when absent or not numeric."""
    host, sep, port = str(address).rpartition(":")
    if not sep or not port.isdigit():
        return str(address).strip("[]"), None
    return host.strip("[]"), int(port)


def _default_leader_addr() -> str | None:
    """Fallback leader address: the host of the JAX distributed coordinator.

    Correct only when the coordinator runs on the process that becomes the eSurge leader
    (``jax.process_index() == 0``). That does not hold on every TPU slice: the TPU runtime numbers processes by
    chip position, while the coordinator is distributed process id 0 (the TPU worker id), so the two can be
    different hosts. Workers therefore read the address the leader publishes (:func:`_published_leader_addr`)
    first and use this only when nothing was published.
    """
    state = _jax_distributed_state()
    addr = getattr(state, "coordinator_address", None) if state is not None else None
    return _split_host_port(addr)[0] if addr else None


def _local_ip_towards(host: str, port: int | None, family: int = socket.AF_UNSPEC) -> str | None:
    """Source IP this host uses to reach ``host`` over ``family`` (a UDP connect only selects the route)."""
    try:
        infos = socket.getaddrinfo(host, port or 9, family=family, type=socket.SOCK_DGRAM)
    except OSError:
        return None
    for family, socktype, proto, _canonname, sockaddr in infos:
        try:
            with socket.socket(family, socktype, proto) as probe:
                probe.connect(sockaddr)
                return str(probe.getsockname()[0])
        except OSError:
            continue
    return None


def _advertised_leader_addr(bind_host: str, configured: str | None) -> str | None:
    """Address workers should dial to reach this process's control plane.

    Preference: an explicitly configured leader address, a concrete bind host, the interface this host uses to
    reach the JAX coordinator (reachable by every process of the same JAX world), then the hostname's address.
    """
    if configured:
        return str(configured)
    if str(bind_host) not in _WILDCARD_BIND_HOSTS:
        return str(bind_host)
    state = _jax_distributed_state()
    coordinator = getattr(state, "coordinator_address", None) if state is not None else None
    if coordinator:
        # Advertise an address of the family the control socket listens on: the default wildcard bind is IPv4, and
        # a hostname coordinator on a dual-stack host can also resolve to IPv6.
        family = socket.AF_INET6 if str(bind_host) in {"::", "[::]"} else socket.AF_INET
        addr = _local_ip_towards(*_split_host_port(coordinator), family=family)
        if addr:
            return addr
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return None


def _publish_leader_addr(control_port: int, bind_host: str, configured: str | None) -> str | None:
    """Publish the leader's control-plane address in the ``jax.distributed`` key-value store.

    Args:
        control_port: Control-plane port (part of the key, so engines on different ports do not collide).
        bind_host: Interface the leader binds on.
        configured: The configured ``distributed_leader_addr``, if any.

    Returns:
        The published address, or ``None`` without a distributed client or when no address could be derived.
    """
    state = _jax_distributed_state()
    client = getattr(state, "client", None) if state is not None else None
    if client is None:
        return None
    addr = _advertised_leader_addr(bind_host, configured)
    if not addr:
        logger.warning("eSurge coordinator: could not derive an address to advertise for the leader control plane.")
        return None
    client.key_value_set(_LEADER_ADDR_KEY.format(port=int(control_port)), addr, allow_overwrite=True)
    return addr


def _published_leader_addr(control_port: int, timeout_s: float) -> str | None:
    """Read the address the leader published in the ``jax.distributed`` key-value store.

    Args:
        control_port: Control-plane port the leader published under.
        timeout_s: Seconds to wait for the leader to publish.

    Returns:
        The address, or ``None`` without a distributed client or when nothing was published in time.
    """
    state = _jax_distributed_state()
    client = getattr(state, "client", None) if state is not None else None
    if client is None:
        return None
    key = _LEADER_ADDR_KEY.format(port=int(control_port))
    try:
        return str(client.blocking_key_value_get(key, max(1, int(float(timeout_s) * 1000))))
    except Exception as exc:  # deadline or a dead coordination service: fall back to the older defaults
        logger.warning("eSurge coordinator: no leader address was published within %.0fs (%s).", timeout_s, exc)
        return None


def create_step_coordinator(
    runner,
    *,
    distributed_config,
    config_fingerprint: str | None,
    engine_spec=None,
):
    """Build the right coordinator for this process.

    Selection order:

    1. ``coordination="zmq"`` with ``jax.process_count() > 1`` builds the
       ZeroMQ leader (rank 0) or worker plane.
    2. ``coordination="zmq"`` on a single process builds a workerless
       leader: no step replication, but the control socket serves
       request-plane clients — the DP-replica pattern where each replica
       engine is its own JAX world fronted by a router.
    3. Everything else — the default single host, or the multi-host
       *replicated* pattern where an outer driver (e.g. a trainer's rollout
       loop) calls the engine identically on every host — gets the
       pass-through :class:`LocalCoordinator`.

    Args:
        runner: The engine's runner.
        distributed_config: The engine's distributed config section.
        config_fingerprint: Engine-config fingerprint for the handshake
            (required for the ZMQ plane).
        engine_spec: Optional ``wire.EngineSpec`` the leader hands to
            request-plane clients in ``HelloOk``.

    Returns:
        A :class:`StepCoordinator` implementation.

    Raises:
        StepCoordinationError: If ``coordination="zmq"`` is requested but
            the auth token or leader address cannot be resolved.
    """
    import jax

    world_size = int(jax.process_count())
    coordination = str(getattr(distributed_config, "coordination", "replicated") or "replicated")
    if coordination != "zmq":
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
        leader = ZmqLeaderCoordinator(
            runner,
            world_size=world_size,
            bind_host=str(distributed_config.distributed_control_bind_host),
            control_port=control_port,
            ready_timeout_s=float(distributed_config.distributed_ready_timeout_s),
            step_timeout_s=float(distributed_config.distributed_step_timeout_s),
            verify_digest_interval=int(distributed_config.distributed_verify_digest_interval),
            max_inflight_steps=int(distributed_config.distributed_max_inflight_steps),
            engine_spec=engine_spec,
            **common,
        )
        if world_size > 1:
            advertised = _publish_leader_addr(
                control_port,
                str(distributed_config.distributed_control_bind_host),
                distributed_config.distributed_leader_addr,
            )
            if advertised:
                logger.info("eSurge coordinator: leader control plane advertised at %s:%d.", advertised, control_port)
        return leader

    # Explicit address, then the one the leader published, then the coordinator's host, then DNS discovery.
    leader_addr = distributed_config.distributed_leader_addr or _published_leader_addr(
        control_port, float(distributed_config.distributed_connect_timeout_s)
    )
    leader_addr = leader_addr or _default_leader_addr()
    if not leader_addr and distributed_config.distributed_service_name:
        from .discovery import resolve_service_hosts

        discovered = resolve_service_hosts(distributed_config.distributed_service_name, world_size=world_size)
        leader_addr = discovered.rank_to_host[0]
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
