# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backend-neutral execution-driver selection."""

from __future__ import annotations

import hashlib
import logging
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from urllib.parse import urlparse

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan._internal.network import format_http_ip_endpoint
from eyvan.capacity.drivers.base import (
    CellAccelerator,
    CellObservation,
    ExecutionDriver,
    MutationContext,
    UnsupportedCapabilityError,
    canonical_execution_backend,
    read_cell_accelerator,
)
from eyvan.identity.mtls import MutualTlsConfigurationError, MutualTlsFiles, client_ssl_context
from eyvan.worker.distributed_environment import (
    DEFAULT_GPU_RANK_LAYOUT,
    GpuRankLayout,
    gpu_rank_placement_error,
)
from eyvan.worker.runtime._multihost_site import DEFAULT_GANG_INITIALIZATION, GangInitializationPolicy
from eyvan.worker.runtime.log_records import LogPage

ExecutorFactory = tp.Callable[..., ExecutionDriver]
logger = logging.getLogger(__name__)

#: Agent-side switch for what a GPU rank is. Absent means the layout every
#: dispatch used before per-host grouping existed, so an agent upgraded without
#: being reconfigured keeps sending exactly what it sends today.
GPU_RANK_LAYOUT_ENV = "EYVAN_GPU_RANK_LAYOUT"


def gpu_rank_layout_from_environment(environment: tp.Mapping[str, str] | None = None) -> GpuRankLayout:
    """Read the GPU rank layout an agent was configured with.

    Refused rather than defaulted when the value is not a layout, for the reason
    every other spelling-sensitive switch in this repository is refused at
    startup: a typo that silently reads as "off" turns a deliberate operator
    decision into a behaviour nobody chose, and the fleet reports nothing.

    Args:
        environment: The process environment to read, or ``None`` for
            :data:`os.environ`.

    Returns:
        The configured layout, or the shipped default when the variable is
        absent or empty.

    Raises:
        ValueError: If the variable names something that is not a layout.
    """

    source = os.environ if environment is None else environment
    raw = source.get(GPU_RANK_LAYOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_GPU_RANK_LAYOUT
    try:
        return GpuRankLayout(raw)
    except ValueError:
        choices = ", ".join(layout.value for layout in GpuRankLayout)
        raise ValueError(f"{GPU_RANK_LAYOUT_ENV} must be one of: {choices}; got {raw!r}") from None


def _uses_gce_tpu_cluster_detection(observation: CellObservation) -> bool:
    """Return whether JAX should derive this cell's cluster from GCE metadata.

    True for *any* GCE TPU, not only a multislice. The question that matters is
    "can JAX discover this topology itself", and for a Cloud TPU it always can --
    a single ``v5p-16`` is two hosts and is discovered exactly like a multislice.

    Both single-slice multi-host and multislice TPUs need GCE discovery. The
    ``EXPLICIT`` path reuses the worker's management endpoint, which may be an
    external IP. Hosts must instead coordinate over internal addresses; using a
    public IP can send rendezvous traffic to a port the firewall does not permit.

    The management endpoint (control plane to worker) and coordination address
    (host to host) serve different purposes. GCE metadata gives JAX the internal
    coordination address directly.

    ``EXPLICIT`` remains correct for multi-host backends that are not GCE TPUs,
    where Eyvan genuinely does have to supply the addresses.
    """

    raw_config = observation.metadata.get("tpu_config")
    if raw_config is None:
        return False
    if not is_json_object(raw_config):
        raise ValueError("eyvan-native execution observation has invalid TPU provider configuration")
    # Validated but deliberately not branched on. A malformed record must still
    # be refused, while the *value* no longer selects the detection mode: JAX can
    # discover a Cloud TPU from metadata whether it is one slice or several, and
    # branching on it is precisely what put two-host single slices on the
    # explicit path with an external coordinator address.
    node_count = raw_config.get("node_count", 1)
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
        raise ValueError("eyvan-native execution observation has an invalid TPU node count")
    return True


@tp.runtime_checkable
class ClosableExecutionDriver(tp.Protocol):
    """Optional cleanup capability implemented by stateful drivers."""

    def close(self) -> None: ...


@tp.runtime_checkable
class HealthCheckedExecutionDriver(tp.Protocol):
    """Optional readiness capability implemented by remote drivers."""

    def health(self) -> bool: ...


@tp.runtime_checkable
class IntrospectableExecutionDriver(tp.Protocol):
    """Optional live-process introspection.

    A protocol rather than a ``getattr(executor, "process_status", None)`` probe:
    this module already establishes ``ClosableExecutionDriver`` and
    ``HealthCheckedExecutionDriver`` as the idiomatic seam, and a third ad-hoc
    probe would make the pattern look optional.
    """

    def process_status(self, remote_id: str) -> JsonObject: ...


@tp.runtime_checkable
class StructuredLogExecutionDriver(tp.Protocol):
    """Optional structured, resumable log reads.

    ``logs(remote_id) -> str`` is the live contract, and a string cannot carry
    what an operator needs from a multi-host job: which rank a line came from,
    whether it was stdout or stderr, where to resume, or whether a rank was
    unreachable as opposed to silent. Multi-host fan-out today is string
    concatenation with ``===== rank N =====`` banners.

    Optional, and a protocol rather than a ``getattr`` probe, for the same
    reason as the three above: a backend that cannot produce cursors should say
    so by not satisfying the protocol rather than by returning something that
    looks like a page.
    """

    def log_page(self, remote_id: str, *, after: str = "", limit: int = 200) -> LogPage: ...


@tp.runtime_checkable
class ExecCapableExecutionDriver(tp.Protocol):
    """Optional diagnostic exec beside a running workload.

    The *implementation* gate. ``AgentCapabilities.supports_exec`` is the
    operator-facing *admission* gate, and the two answer different questions:
    "may anyone do this here" versus "can this backend do it at all".
    Disagreement between them is a configuration error the agent reports at
    startup, with a message distinct from "backend cannot".
    """

    def exec_in_workload(self, remote_id: str, command: tp.Sequence[str], *, timeout_s: float) -> JsonObject: ...


@tp.runtime_checkable
class RankedExecCapableExecutionDriver(tp.Protocol):
    """Optional exact inner-host diagnostic selection, distinct from fan-out."""

    def exec_in_workload_rank(
        self, remote_id: str, command: tp.Sequence[str], *, timeout_s: float, runtime_rank: int
    ) -> JsonObject: ...


@dataclass(frozen=True, slots=True)
class NativeGangCell:
    """One cell selected for a native multi-cell workload group."""

    cell_id: str
    observation: CellObservation
    context: MutationContext

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("native gang cell ID must be non-empty and trimmed")
        if self.observation.cell_id != self.cell_id:
            raise ValueError("native gang observation does not belong to its declared cell")


def _worker_endpoints(
    value: JsonValue,
    *,
    field_name: str,
    require_trimmed: bool,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Validate a provider metadata value as unique native worker endpoints."""

    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} has invalid worker endpoints")
    endpoints: list[str] = []
    for endpoint in value:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(f"{field_name} has invalid worker endpoints")
        if require_trimmed and endpoint != endpoint.strip():
            raise ValueError(f"{field_name} has invalid worker endpoints")
        endpoints.append(endpoint.strip())
    if not endpoints:
        if allow_empty:
            return ()
        raise ValueError(f"{field_name} has invalid worker endpoints")
    if len(endpoints) != len(set(endpoints)):
        raise ValueError(f"{field_name} has invalid worker endpoints")
    return tuple(endpoints)


def _normalize_name(name: str) -> str:
    return canonical_execution_backend(name)


def _native_local_factory(
    *,
    endpoint: str | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    token: str | None = None,
) -> ExecutionDriver:
    del token
    if endpoint not in {None, "", "local"}:
        raise ValueError("native-local execution does not accept a remote endpoint")
    from eyvan.worker.runtime.local_process import LocalProcessExecutor

    return LocalProcessExecutor(state_dir=state_dir)


def _eyvan_native_factory(
    *,
    endpoint: str | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    token: str | None = None,
    mtls: MutualTlsFiles | None = None,
) -> ExecutionDriver:
    del state_dir
    if endpoint is None or not endpoint.strip():
        raise ValueError("eyvan-native execution requires a native worker endpoint")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("eyvan-native execution requires a native worker bearer secret")
    from eyvan.worker.runtime.native_remote import NativeRemoteExecutor

    kwargs: dict[str, tp.Any] = {"secret": token}
    if mtls is not None:
        kwargs["mtls"] = mtls
    return NativeRemoteExecutor(endpoint, **kwargs)


def _retired_ray_jobs_factory(
    *,
    endpoint: str | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    token: str | None = None,
) -> ExecutionDriver:
    """Refuse the retired Ray Jobs backend with an actionable message.

    Eyvan replaces Ray rather than dispatching to it, so the adapter is gone. The
    name is still *registered* on purpose: a workload already persisted with this
    backend must fail with a sentence an operator can act on, not with an unknown
    backend error that reads like a typo or a broken install.
    """

    del endpoint, state_dir, token
    raise RuntimeError(
        "the ray-jobs execution backend has been removed; Eyvan runs workloads natively -- "
        "resubmit with required_executor='eyvan-native' (or 'native-local' for a single host)"
    )


class ExecutionRuntimeRegistry:
    """Registry of execution-driver factories keyed by stable backend name."""

    def __init__(self) -> None:
        self._factories: dict[str, ExecutorFactory] = {}
        self._lock = RLock()

    def register(self, name: str, factory: ExecutorFactory, *, replace: bool = False) -> None:
        """Register one execution backend factory."""

        normalized = _normalize_name(name)
        if not normalized:
            raise ValueError("execution backend name must be non-empty")
        with self._lock:
            if normalized in self._factories and not replace:
                raise ValueError(f"execution backend {normalized!r} is already registered")
            self._factories[normalized] = factory

    def create(
        self,
        name: str,
        *,
        endpoint: str | None = None,
        state_dir: str | os.PathLike[str] | None = None,
        token: str | None = None,
        mtls: MutualTlsFiles | None = None,
    ) -> ExecutionDriver:
        """Construct the selected execution driver without importing others."""

        normalized = _normalize_name(name)
        with self._lock:
            factory = self._factories.get(normalized)
        if factory is None:
            choices = ", ".join(self.names())
            raise ValueError(f"unknown execution backend {name!r}; available backends: {choices}")
        kwargs: dict[str, tp.Any] = {"endpoint": endpoint, "state_dir": state_dir}
        if token is not None:
            kwargs["token"] = token
        # The public factory ABI is deliberately unchanged for registered
        # extensions. Only the shipped native factory receives the optional
        # transport configuration.
        if factory is _eyvan_native_factory and mtls is not None:
            kwargs["mtls"] = mtls
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered canonical backend names."""

        with self._lock:
            return tuple(sorted(self._factories))


DEFAULT_EXECUTION_RUNTIMES = ExecutionRuntimeRegistry()
DEFAULT_EXECUTION_RUNTIMES.register("native-local", _native_local_factory)
DEFAULT_EXECUTION_RUNTIMES.register("eyvan-native", _eyvan_native_factory)
DEFAULT_EXECUTION_RUNTIMES.register("ray-jobs", _retired_ray_jobs_factory)


class ExecutionDriverResolver:
    """Resolve and cache drivers for one cell-agent process."""

    def __init__(
        self,
        backend: str = "eyvan-native",
        *,
        registry: ExecutionRuntimeRegistry | None = None,
        state_dir: str | os.PathLike[str] | None = None,
        native_worker_port: int = 8100,
        native_worker_token: str | None = None,
        gang_initialization: GangInitializationPolicy = DEFAULT_GANG_INITIALIZATION,
        gpu_rank_layout: GpuRankLayout | None = None,
        mtls: MutualTlsFiles | None = None,
    ) -> None:
        if native_worker_port <= 0 or native_worker_port > 65535:
            raise ValueError("native worker port must be between 1 and 65535")
        if not isinstance(gang_initialization, GangInitializationPolicy):
            raise TypeError("gang initialization must be a GangInitializationPolicy")
        if gpu_rank_layout is None:
            gpu_rank_layout = gpu_rank_layout_from_environment()
        if not isinstance(gpu_rank_layout, GpuRankLayout):
            raise TypeError("gpu rank layout must be a GpuRankLayout")
        self.backend = _normalize_name(backend)
        # Transport configuration belongs only to the shipped native backend.
        # Custom registries retain their historical factory ABI and must not
        # load certificate files merely because the agent has mTLS enabled.
        if mtls is not None and self.backend == "eyvan-native":
            if not mtls.has_trust_anchor:
                raise MutualTlsConfigurationError("a mutual TLS client requires a trust bundle to verify its server")
            if not mtls.has_identity:
                raise MutualTlsConfigurationError("a mutual TLS client requires a certificate and private key")
            client_ssl_context(mtls)
        if self.backend == "eyvan-native" and (
            not isinstance(native_worker_token, str) or not native_worker_token.strip()
        ):
            raise ValueError("eyvan-native execution requires EYVAN_NATIVE_WORKER_ROOT_SECRET")
        self.registry = registry or DEFAULT_EXECUTION_RUNTIMES
        if self.backend not in self.registry.names():
            choices = ", ".join(self.registry.names())
            raise ValueError(f"unknown execution backend {backend!r}; available backends: {choices}")
        self.state_dir = Path(state_dir).expanduser() if state_dir is not None else None
        self.native_worker_port = native_worker_port
        self.native_worker_token = native_worker_token
        self.mtls = mtls
        self.gang_initialization = gang_initialization
        self.gpu_rank_layout = gpu_rank_layout
        self._drivers: dict[tuple[str, ...], ExecutionDriver] = {}
        self._lock = RLock()

    @staticmethod
    def _close_driver(driver: ExecutionDriver) -> None:
        if not isinstance(driver, ClosableExecutionDriver):
            return
        try:
            driver.close()
        except Exception:
            logger.warning("failed to close an Eyvan execution driver", exc_info=True)

    def close_cell(self, cell_id: str) -> None:
        """Dispose every cached driver for one deleted execution cell."""

        with self._lock:
            stale_keys = [key for key in self._drivers if key[1] == cell_id]
            stale_drivers = [self._drivers.pop(key) for key in stale_keys]
        for driver in stale_drivers:
            self._close_driver(driver)

    def close(self) -> None:
        """Dispose all cached execution drivers owned by this agent."""

        with self._lock:
            drivers, self._drivers = list(self._drivers.values()), {}
        for driver in drivers:
            self._close_driver(driver)

    def resolve(self, cell_id: str, observation: CellObservation) -> ExecutionDriver:
        """Return the configured driver for one observed execution cell."""

        if not cell_id or cell_id != cell_id.strip():
            raise ValueError("cell ID must be non-empty and trimmed")
        if observation.cell_id != cell_id:
            raise ValueError(
                f"execution observation belongs to cell {observation.cell_id!r}, not requested cell {cell_id!r}"
            )
        endpoint = observation.execution_endpoint
        if self.backend == "eyvan-native" and not endpoint:
            head_ip = observation.metadata.get("head_ip")
            if isinstance(head_ip, str) and head_ip.strip():
                try:
                    endpoint = format_http_ip_endpoint(head_ip, self.native_worker_port)
                except ValueError as exc:
                    raise ValueError("eyvan-native execution observation has an invalid head IP address") from exc
        token = None
        native_endpoints: tuple[str, ...] = ()
        if self.backend == "eyvan-native":
            token = self._native_worker_token(observation)
            raw_endpoints = observation.metadata.get("native_worker_endpoints")
            if raw_endpoints is not None:
                native_endpoints = _worker_endpoints(
                    raw_endpoints,
                    field_name="eyvan-native execution observation",
                    require_trimmed=False,
                    allow_empty=True,
                )
        token_fingerprint = hashlib.sha256(token.encode()).hexdigest()[:24] if token is not None else ""
        gce_tpu_cluster = self.backend == "eyvan-native" and _uses_gce_tpu_cluster_detection(observation)
        endpoint_fingerprint = "\x1f".join(native_endpoints) if len(native_endpoints) > 1 else str(endpoint or "")
        endpoint_fingerprint = f"{endpoint_fingerprint}\x1e{int(gce_tpu_cluster)}"
        key = (self.backend, cell_id, endpoint_fingerprint, token_fingerprint)
        with self._lock:
            driver = self._drivers.get(key)
            if driver is None:
                state_dir = self.state_dir
                if state_dir is not None and self.backend == "native-local":
                    state_dir = state_dir / "executions" / hashlib.sha256(cell_id.encode()).hexdigest()[:24]
                if self.backend == "eyvan-native" and len(native_endpoints) > 1:
                    from eyvan.worker.runtime.native_slice import JaxClusterDetectionMode, NativeSliceExecutor

                    assert token is not None
                    slice_kwargs: dict[str, tp.Any] = {}
                    if self.mtls is not None:
                        slice_kwargs["mtls"] = self.mtls
                    driver = NativeSliceExecutor(
                        native_endpoints,
                        secret=token,
                        jax_cluster_detection=(
                            JaxClusterDetectionMode.GCE_TPU if gce_tpu_cluster else JaxClusterDetectionMode.EXPLICIT
                        ),
                        initialization=self.gang_initialization,
                        **slice_kwargs,
                    )
                else:
                    create_kwargs: dict[str, tp.Any] = {
                        "endpoint": endpoint,
                        "state_dir": state_dir,
                        "token": token,
                    }
                    if self.mtls is not None:
                        create_kwargs["mtls"] = self.mtls
                    driver = self.registry.create(self.backend, **create_kwargs)
                self._drivers[key] = driver
            stale_keys = [driver_key for driver_key in self._drivers if driver_key[1] == cell_id and driver_key != key]
            stale_drivers = [self._drivers.pop(driver_key) for driver_key in stale_keys]
        for stale_driver in stale_drivers:
            self._close_driver(stale_driver)
        return driver

    def resolve_gang(self, cells: tp.Sequence[NativeGangCell]):
        """Resolve one native executor spanning independently fenced execution cells.

        The gang becomes GPU-coordinated only when every cell's observation
        carries a catalog GPU shape (see :meth:`_gang_gpu_accelerators`). A cell
        whose agent predates the accelerator metadata, or whose machine type has
        no catalog row, reports nothing, and the whole gang then takes the path
        it has always taken: no GPU variable written, none stripped.

        Nothing here is hardware-qualified. No GPU has executed an Eyvan
        workload; what this method decides is which bytes a rank is dispatched.
        """

        if self.backend != "eyvan-native":
            raise UnsupportedCapabilityError(
                f"execution backend {self.backend!r} does not support native multi-cell gangs"
            )
        ordered = tuple(sorted(cells, key=lambda cell: cell.cell_id))
        if len(ordered) < 2:
            raise ValueError("native gang execution requires at least two cells")
        if len({cell.cell_id for cell in ordered}) != len(ordered):
            raise ValueError("native gang execution cell IDs must be unique")

        from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
        from eyvan.worker.runtime.native_slice import JaxClusterDetectionMode

        cell_endpoints: dict[str, tuple[str, ...]] = {}
        for cell in ordered:
            raw_endpoints = cell.observation.metadata.get("native_worker_endpoints")
            if raw_endpoints is None:
                endpoint = cell.observation.execution_endpoint
                if not isinstance(endpoint, str) or not endpoint.strip():
                    raise ValueError(f"native gang cell {cell.cell_id!r} has no execution endpoint")
                cell_endpoints[cell.cell_id] = (endpoint.strip(),)
            else:
                cell_endpoints[cell.cell_id] = _worker_endpoints(
                    raw_endpoints,
                    field_name=f"native gang cell {cell.cell_id!r}",
                    require_trimmed=True,
                    allow_empty=False,
                )
        accelerators = self._gang_gpu_accelerators(ordered, cell_endpoints, layout=self.gpu_rank_layout)
        members = [
            NativeGangMember(
                cell_id=cell.cell_id,
                endpoints=cell_endpoints[cell.cell_id],
                secret=self._native_worker_token(cell.observation),
                context=cell.context,
                gpu_devices_per_host=(
                    accelerators[cell.cell_id].gpu_devices_per_host if cell.cell_id in accelerators else None
                ),
                gpu_model=accelerators[cell.cell_id].gpu_model if cell.cell_id in accelerators else None,
                jax_cluster_detection=(
                    JaxClusterDetectionMode.GCE_TPU
                    if _uses_gce_tpu_cluster_detection(cell.observation)
                    else JaxClusterDetectionMode.EXPLICIT
                ),
            )
            for cell in ordered
        ]
        gang_kwargs: dict[str, tp.Any] = {
            "initialization": self.gang_initialization,
            "gpu_rank_layout": self.gpu_rank_layout,
        }
        if self.mtls is not None:
            gang_kwargs["mtls"] = self.mtls
        return NativeGangExecutor(members, **gang_kwargs)

    @staticmethod
    def _gang_gpu_accelerators(
        cells: tp.Sequence[NativeGangCell],
        cell_endpoints: tp.Mapping[str, tuple[str, ...]],
        *,
        layout: GpuRankLayout = DEFAULT_GPU_RANK_LAYOUT,
    ) -> dict[str, CellAccelerator]:
        """Return the GPU description of every cell, or nothing for the whole gang.

        This is the only place a dispatched gang becomes GPU-coordinated, and it
        is all-or-nothing on purpose. A gang where some members carry a known GPU
        shape and others do not is exactly the state
        :class:`~eyvan.worker.runtime.native_gang.NativeGangExecutor` refuses to
        construct -- the known ranks would be pinned to exact devices while the
        rest kept whatever the submission said, and the gang would look
        coordinated. Here that state is *downgraded* rather than refused: these
        are observations, not a hand-built gang, and partial knowledge is the
        expected state of a fleet whose agents are being upgraded one at a time
        or whose machine type has no catalog row. Refusing would turn a rolling
        upgrade into a dispatch failure for a gang that runs today.

        Hosts are required to be *nameable* for the same reason the counts are.
        Ranks are placed by host, and an endpoint whose host cannot be parsed is
        not evidence that it is alone on a machine -- so a gang with one is left
        uncoordinated rather than placed against a hostname nobody read.

        What "nameable" then buys depends on ``layout``. Under
        :attr:`~eyvan.worker.distributed_environment.GpuRankLayout.HOST_LAUNCHER`,
        the shipped default, a rank is a host-local launcher that owns its host's
        whole device set, so two endpoints on one host would each be told they
        own all of it and the gang is left uncoordinated -- exactly as it was
        before per-host grouping existed. Under
        :attr:`~eyvan.worker.distributed_environment.GpuRankLayout.RANK_PROCESS`
        several ranks may share a host and split its devices, so the check
        becomes whether that split exists: the ranks on one host must agree on
        what that host carries, and the count must divide among them.

        Args:
            cells: The gang's cells, in dispatch order.
            cell_endpoints: Each cell's validated native-worker endpoints.
            layout: What a rank is in this deployment.

        Returns:
            One :class:`CellAccelerator` per cell when every cell is a known GPU
            shape and the gang's ranks can be placed, otherwise an empty
            mapping. Empty means "not GPU-coordinated", which is today's path,
            and never "no GPU".
        """

        accelerators: dict[str, CellAccelerator] = {}
        devices_by_cell: dict[str, int] = {}
        for cell in cells:
            accelerator = read_cell_accelerator(cell.observation.metadata)
            devices = accelerator.gpu_devices_per_host if accelerator is not None else None
            if accelerator is not None and devices is not None:
                accelerators[cell.cell_id] = accelerator
                devices_by_cell[cell.cell_id] = devices
        if not accelerators:
            return {}
        if len(accelerators) != len(cells):
            logger.warning(
                "native gang is not GPU-coordinated: %d of %d cells report a catalog GPU shape; cells without one: %s",
                len(accelerators),
                len(cells),
                ", ".join(sorted(cell.cell_id for cell in cells if cell.cell_id not in accelerators)),
            )
            return {}
        # One entry per rank -- the gang dispatches one process per endpoint --
        # carrying the host it lands on and what the catalog says that host
        # holds. A member with four hosts contributes its device count four
        # times, exactly as the executor flattens it.
        hosts: list[str] = []
        devices_per_rank: list[int] = []
        for cell in cells:
            for endpoint in cell_endpoints[cell.cell_id]:
                hosts.append(urlparse(endpoint).hostname or "")
                devices_per_rank.append(devices_by_cell[cell.cell_id])
        if not all(hosts):
            logger.warning(
                "native gang is not GPU-coordinated: %d of its %d worker endpoints do not name a host, "
                "so their ranks cannot be placed on one",
                sum(1 for host in hosts if not host),
                len(hosts),
            )
            return {}
        if layout is GpuRankLayout.HOST_LAUNCHER:
            if len(set(hosts)) != len(hosts):
                logger.warning(
                    "native gang is not GPU-coordinated: its %d worker endpoints name %d distinct hosts, "
                    "and the %r GPU rank layout would give two ranks on one host the same devices",
                    len(hosts),
                    len(set(hosts)),
                    GpuRankLayout.HOST_LAUNCHER.value,
                )
                return {}
            return accelerators
        reason = gpu_rank_placement_error(devices_per_rank, hosts)
        if reason is not None:
            logger.warning("native gang is not GPU-coordinated: %s", reason)
            return {}
        return accelerators

    def _native_worker_token(self, observation: CellObservation) -> str:
        """Derive the per-cell worker token from the agent's shared secret."""

        if self.native_worker_token is None:
            raise ValueError("eyvan-native execution requires EYVAN_NATIVE_WORKER_ROOT_SECRET")
        management_uid = observation.metadata.get("management_uid")
        if not isinstance(management_uid, str) or not management_uid.strip():
            raise ValueError("eyvan-native execution observation is missing its management UID")
        provider_generation = observation.metadata.get("provider_generation", 1)
        if isinstance(provider_generation, bool) or not isinstance(provider_generation, int) or provider_generation < 1:
            raise ValueError("eyvan-native execution observation has an invalid provider generation")
        from eyvan.worker.runtime.native_bootstrap import derive_native_worker_token

        token = derive_native_worker_token(
            self.native_worker_token,
            management_uid,
            provider_generation=provider_generation,
        )
        if token is None:
            raise ValueError("eyvan-native execution could not derive a native worker token")
        return token

    def readiness(self, cell_id: str, observation: CellObservation) -> tuple[bool, str]:
        """Probe the selected runtime before provider capacity becomes schedulable."""

        try:
            driver = self.resolve(cell_id, observation)
            if not isinstance(driver, HealthCheckedExecutionDriver):
                return True, "execution runtime is ready"
            if driver.health():
                return True, "execution runtime is ready"
            return False, f"{self.backend} execution runtime is not ready"
        except Exception as exc:
            return False, f"{self.backend} execution runtime is not ready: {exc}"


ConfiguredExecutionResolver = ExecutionDriverResolver


def create_execution_driver(
    name: str,
    *,
    endpoint: str | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    token: str | None = None,
) -> ExecutionDriver:
    """Construct one driver using the built-in runtime registry."""

    return DEFAULT_EXECUTION_RUNTIMES.create(name, endpoint=endpoint, state_dir=state_dir, token=token)


def available_execution_drivers() -> tuple[str, ...]:
    """Return canonical built-in execution backend names."""

    return DEFAULT_EXECUTION_RUNTIMES.names()


__all__ = (
    "DEFAULT_EXECUTION_RUNTIMES",
    "GPU_RANK_LAYOUT_ENV",
    "ConfiguredExecutionResolver",
    "ExecutionDriverResolver",
    "ExecutionRuntimeRegistry",
    "NativeGangCell",
    "StructuredLogExecutionDriver",
    "available_execution_drivers",
    "create_execution_driver",
    "gpu_rank_layout_from_environment",
)
