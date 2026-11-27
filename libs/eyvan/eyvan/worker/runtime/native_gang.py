# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinated native execution across several independently managed cells.

A gang is heterogeneous by construction: each member is an independently
allocated cell with its own fence, and nothing requires two members to hold the
same accelerator. What was missing was the environment a GPU member's rank needs
-- which devices it owns, and where the collective rendezvous is -- so a gang
containing a GPU cell formed a JAX cluster and then left every rank to guess its
devices.

Ranks are placed by host. Which is to say: a gang may now run more than one
process on one machine, each holding a disjoint slice of that machine's devices,
where before both would have been told they owned all of them and the gang was
declined instead. Whether that is even expressible depends on what a rank *is*,
which is a deployment's choice and defaults to the one every release so far has
made -- see
:class:`~eyvan.worker.distributed_environment.GpuRankLayout`.

**Nothing in the GPU path is hardware-qualified.** No GPU has executed an Eyvan
workload. What is claimed here, and tested, is that the per-rank device
assignment is exact and non-overlapping and that a submission cannot spoof it.
Whether a real multi-GPU host then trains is unknown and is not asserted
anywhere in this file. Nor is device isolation between two *workloads* packed
onto one cell: this file places the ranks of one gang, and Eyvan records no
per-lease device range.
"""

from __future__ import annotations

import copy
import logging
import typing as tp
from dataclasses import dataclass, replace
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.drivers.base import (
    ExecutionWorkloadObservation,
    ExecutionWorkloadResult,
    ExecutionWorkloadStatus,
    MutationContext,
    rollback_context,
)
from eyvan.identity.mtls import MutualTlsFiles, client_ssl_context
from eyvan.worker.distributed_environment import (
    DEFAULT_GPU_RANK_LAYOUT,
    DEFAULT_NATIVE_COORDINATOR_PORT,
    GpuDevicePlan,
    GpuRankLayout,
    plan_gpu_devices,
    without_distributed_topology_overrides,
    without_gpu_topology_overrides,
)
from eyvan.worker.runtime._multihost_site import (
    CROSS_REGION_TRANSFER_BANNED_EXIT_CODE,
    CROSS_REGION_TRANSFER_BANNED_MARKER,
    CROSS_REGION_TRANSFER_BANNED_REASON,
    DEFAULT_GANG_INITIALIZATION,
    GangInitializationPolicy,
    cross_region_ban_failure_reason,
    extract_cross_region_ban_marker,
    gang_formation_failure_reason,
    gang_initialization_outcome,
)
from eyvan.worker.runtime.log_records import (
    CURSOR_KIND_WORKLOAD_LOG,
    CURSOR_MODE_LIVE,
    CursorContext,
    CursorFresh,
    CursorResume,
    LogPage,
    decode_log_cursor,
    join_log_cursor,
    merge_pages,
)
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_slice import _RESUME_CHECKPOINT_ENV, JaxClusterDetectionMode, _exec_fanout
from eyvan.worker.runtime.systemd_transient import _rank_checkpoint_uri

_MULTIHOST_SITE_PATH = "/opt/eyvan-native/multihost"
#: The one definition now lives beside the GPU rendezvous port, because the two
#: numbers are one operator-visible fact -- the ports a firewall must open
#: between gang hosts -- and because gang dispatch has to name both when it
#: refuses to hand one endpoint to two gangs. Same value, same wire bytes.
_COORDINATOR_PORT = DEFAULT_NATIVE_COORDINATOR_PORT
_RETIRE_SUMMARY_LIMIT = 4
_FAILED_STATUSES = frozenset({ExecutionWorkloadStatus.FAILED})
_LOST_STATUSES = frozenset({ExecutionWorkloadStatus.LOST, ExecutionWorkloadStatus.NOT_FOUND})
_RUNNING_STATUSES = frozenset(
    {
        ExecutionWorkloadStatus.PENDING,
        ExecutionWorkloadStatus.SUBMITTED,
        ExecutionWorkloadStatus.RUNNING,
        ExecutionWorkloadStatus.STOPPING,
        ExecutionWorkloadStatus.CHECKPOINTING,
    }
)
_MutationAction: tp.TypeAlias = tp.Literal["stop", "retire"]

logger = logging.getLogger(__name__)


class NativeGangWorkerResult(tp.TypedDict):
    """Wire-compatible observation for one worker in a native cell gang."""

    rank: int
    cell_id: str
    endpoint: str
    remote_id: str
    status: ExecutionWorkloadStatus
    message: str
    checkpoint_uri: str | None


class NativeGangWorkloadResult(ExecutionWorkloadResult):
    """Aggregate gang status plus the typed observation from every rank."""

    workers: list[NativeGangWorkerResult]


@dataclass(frozen=True, slots=True)
class NativeGangMember:
    """Native-worker endpoints and fence owned by one execution cell.

    ``gpu_devices_per_host`` is how many GPUs each of this member's hosts
    carries, taken from the hardware catalog entry for the cell's accelerator
    type (:attr:`AcceleratorShape.devices_per_host`). Leaving it ``None`` is the
    shipped behaviour and means "this member is not GPU-coordinated": no GPU
    variable is written and no GPU variable is stripped, so a gang built the way
    every existing caller builds one behaves exactly as it does today.

    ``None`` is deliberately not "zero devices". An unknown device count is
    unknown; a zero would be a claim that the host has no GPU, and it would be
    delivered as an empty ``CUDA_VISIBLE_DEVICES``, which is how a rank ends up
    on the CPU while its peers wait for it at a collective.

    Attributes:
        cell_id: The execution cell that owns these endpoints.
        endpoints: One native-worker endpoint per host in the cell.
        secret: The cell-scoped native-worker bearer token.
        context: The cell-scoped mutation fence.
        gpu_devices_per_host: GPUs on each of this member's hosts, or ``None``
            when the member is not GPU-coordinated.
        gpu_model: The attached accelerator's provider name, when recorded.
        jax_cluster_detection: Validated discovery mode for this cell's elastic
            JAX world. Atomic cross-cell worlds remain explicitly coordinated.
    """

    cell_id: str
    endpoints: tuple[str, ...]
    secret: str
    context: MutationContext
    gpu_devices_per_host: int | None = None
    gpu_model: str | None = None
    jax_cluster_detection: JaxClusterDetectionMode = JaxClusterDetectionMode.EXPLICIT

    def __post_init__(self) -> None:
        if not isinstance(self.jax_cluster_detection, JaxClusterDetectionMode):
            raise TypeError("native gang member JAX cluster detection must be a JaxClusterDetectionMode")
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("native gang member cell ID must be non-empty and trimmed")
        if not self.endpoints or any(
            not isinstance(endpoint, str) or not endpoint.strip() for endpoint in self.endpoints
        ):
            raise ValueError("native gang member requires non-empty worker endpoints")
        if len(self.endpoints) != len(set(self.endpoints)):
            raise ValueError("native gang member worker endpoints must be unique")
        if not isinstance(self.secret, str) or not self.secret.strip():
            raise ValueError("native gang member requires a non-empty worker bearer secret")
        if self.gpu_devices_per_host is not None and (
            isinstance(self.gpu_devices_per_host, bool)
            or not isinstance(self.gpu_devices_per_host, int)
            or self.gpu_devices_per_host <= 0
        ):
            raise ValueError("native gang member GPU device count must be a positive integer or absent")
        if self.gpu_model is not None and (
            not isinstance(self.gpu_model, str) or not self.gpu_model.strip() or self.gpu_model != self.gpu_model.strip()
        ):
            raise ValueError("native gang member GPU model must be non-empty and trimmed or absent")
        if self.gpu_model is not None and self.gpu_devices_per_host is None:
            raise ValueError("native gang member naming a GPU model must also state its device count")


@dataclass(frozen=True, slots=True)
class _GangWorker:
    """One rank-addressable native worker with its cell-scoped fence."""

    cell_id: str
    cell_index: int
    endpoint: str
    context: MutationContext
    executor: NativeRemoteExecutor
    gpu_devices: int | None = None
    gpu_model: str | None = None
    jax_cluster_detection: JaxClusterDetectionMode = JaxClusterDetectionMode.EXPLICIT


@dataclass(frozen=True, slots=True)
class _GangWorkerObservation:
    """One typed worker observation together with stable gang rank metadata."""

    rank: int
    worker: _GangWorker
    observation: ExecutionWorkloadObservation
    checkpoint_uri: str | None = None

    def as_result(self) -> NativeGangWorkerResult:
        """Return the worker detail in the established mapping wire shape."""

        return {
            "rank": self.rank,
            "cell_id": self.worker.cell_id,
            "endpoint": self.worker.endpoint,
            "remote_id": self.observation.remote_id,
            "status": self.observation.status,
            "message": self.observation.message,
            "checkpoint_uri": self.checkpoint_uri,
        }


def _indeterminate_observation(remote_id: str, reason: str) -> ExecutionWorkloadObservation:
    """Represent an explicit failure to obtain a definitive worker observation."""

    return ExecutionWorkloadObservation(
        remote_id=remote_id,
        status=ExecutionWorkloadStatus.INDETERMINATE,
        message=reason,
    )


def _invalid_observation_reason(
    error: Exception,
    *,
    endpoint: str,
) -> str:
    """Keep a validated worker message when Pydantic rejects its observation."""

    if isinstance(error, ValidationError):
        for detail in error.errors():
            if "input" not in detail:
                continue
            payload = detail["input"]
            if not isinstance(payload, dict) or "message" not in payload:
                continue
            message = payload["message"]
            if isinstance(message, str) and message.strip():
                return f"native worker {endpoint} returned an invalid observation: {message}"
    return f"native worker {endpoint} returned an invalid observation: {type(error).__name__}: {error}"


def _normalize_worker_observation(
    observation: ExecutionWorkloadObservation,
    *,
    expected_remote_id: str,
    endpoint: str,
) -> ExecutionWorkloadObservation:
    """Require one typed observation to describe the requested remote workload."""

    if observation.remote_id != expected_remote_id:
        return _indeterminate_observation(
            expected_remote_id,
            f"native worker {endpoint} returned a mismatched remote_id",
        )
    if observation.status is ExecutionWorkloadStatus.INDETERMINATE and not observation.message:
        return _indeterminate_observation(
            expected_remote_id,
            f"native worker {endpoint} reported an indeterminate status without a reason",
        )
    return observation


def _aggregate_status(workers: tp.Sequence[_GangWorkerObservation]) -> ExecutionWorkloadStatus:
    """Reduce typed rank observations without manufacturing a default status."""

    states = tuple(worker.observation.status for worker in workers)
    if not states:
        raise RuntimeError("native gang has no workers to observe")
    if all(state is ExecutionWorkloadStatus.SUCCEEDED for state in states):
        return ExecutionWorkloadStatus.SUCCEEDED
    if all(state is ExecutionWorkloadStatus.NOT_FOUND for state in states):
        return ExecutionWorkloadStatus.NOT_FOUND
    if all(state is ExecutionWorkloadStatus.STOPPED for state in states):
        return ExecutionWorkloadStatus.STOPPED
    if any(state in _FAILED_STATUSES for state in states):
        return ExecutionWorkloadStatus.FAILED
    if any(state is ExecutionWorkloadStatus.INDETERMINATE for state in states):
        return ExecutionWorkloadStatus.INDETERMINATE
    if any(state in _RUNNING_STATUSES for state in states) and all(
        state in _RUNNING_STATUSES | {ExecutionWorkloadStatus.SUCCEEDED} for state in states
    ):
        return ExecutionWorkloadStatus.RUNNING
    if any(
        state in _LOST_STATUSES | {ExecutionWorkloadStatus.STOPPED, ExecutionWorkloadStatus.SUCCEEDED}
        for state in states
    ):
        return ExecutionWorkloadStatus.LOST
    if any(state in _RUNNING_STATUSES for state in states):
        return ExecutionWorkloadStatus.RUNNING
    raise RuntimeError(f"native gang has no aggregate rule for worker states {states!r}")


def _aggregate_message(workers: tp.Sequence[_GangWorkerObservation]) -> str:
    """Render each explicit rank observation, preserving failure reasons."""

    return "; ".join(
        f"rank {worker.rank}={worker.observation.status.value}"
        + (f": {worker.observation.message}" if worker.observation.message else "")
        for worker in workers
    )


def _bounded_retire_summary(failures: tp.Sequence[str]) -> str:
    """Render a bounded per-rank retire failure summary."""

    summary = "; ".join(failures[:_RETIRE_SUMMARY_LIMIT])
    remaining = len(failures) - _RETIRE_SUMMARY_LIMIT
    if remaining > 0:
        summary += f"; +{remaining} more"
    return summary


def _augment_error_detail(error: BaseException, detail: str) -> None:
    """Append cleanup detail to an error without changing its type."""

    if error.args and isinstance(error.args[0], str):
        error.args = (f"{error.args[0]}; {detail}", *error.args[1:])
    else:
        error.args = (*error.args, detail)


def _refuse_unplaceable_gpu_ranks(
    endpoints: tp.Sequence[str],
    rank_hosts: tp.Sequence[str],
    *,
    layout: GpuRankLayout,
) -> None:
    """Refuse a GPU gang whose ranks cannot be placed on the hosts it names.

    Both refusals are about a state that is *proven* wrong rather than unknown,
    and both are raised at construction so a plan that would hand two ranks the
    same cards cannot exist. Before per-host grouping, two endpoints on one host
    produced exactly that plan, and only the resolver's own check kept it off
    the wire.

    Args:
        endpoints: The gang's worker endpoints, in rank order.
        rank_hosts: The host each endpoint names, empty where none was parsed.
        layout: What a rank is in this deployment.

    Raises:
        ValueError: If a host carries several ranks under a layout with no name
            for the second, or if a rank's host cannot be named under a layout
            that places ranks by host.
    """

    if layout is GpuRankLayout.HOST_LAUNCHER:
        shared = sorted({host for host in rank_hosts if host and rank_hosts.count(host) > 1})
        if shared:
            raise ValueError(
                f"native gang worker endpoints place several ranks on host(s) {', '.join(shared)}; "
                f"the {GpuRankLayout.HOST_LAUNCHER.value!r} GPU rank layout supports one rank per host, "
                f"and {GpuRankLayout.RANK_PROCESS.value!r} supports several"
            )
        return
    unnamed = sorted(endpoint for endpoint, host in zip(endpoints, rank_hosts, strict=True) if not host)
    if unnamed:
        raise ValueError(
            f"the {GpuRankLayout.RANK_PROCESS.value!r} GPU rank layout groups ranks by host, so every worker "
            f"endpoint must name one; these do not: {', '.join(unnamed)}"
        )


def _reported_exit_status(detail: JsonObject) -> int | None:
    """Return one host's reported process exit status, or ``None`` if unknown.

    ``None`` covers three different things -- the host was not observed, the
    unit is gone, the payload predates the field -- and all three mean the same
    here: this rank offers no evidence either way, which is not the same as
    evidence that it exited cleanly.
    """

    if detail.get("observed") is not True:
        return None
    status = detail.get("exec_main_status")
    return status if isinstance(status, int) and not isinstance(status, bool) else None


class NativeGangExecutor:
    """Fan one fenced JAX workload across independently allocated Eyvan cells.

    Every worker uses the cell-scoped native-worker bearer token and mutation
    context that created its TPU slice. This differs from
    :class:`NativeSliceExecutor`, whose endpoints all belong to one cell and
    therefore share one token and fence.

    ``gpu_rank_layout`` decides what a GPU rank is, and therefore whether two
    worker endpoints may name the same machine. It defaults to the layout every
    dispatch used before per-host grouping existed, under which a host carries
    one rank and a second one on the same machine is refused rather than handed
    the same devices. See :class:`~eyvan.worker.distributed_environment.GpuRankLayout`.
    """

    def __init__(
        self,
        members: tp.Sequence[NativeGangMember],
        *,
        clients: tp.Sequence[httpx.Client | None] | None = None,
        initialization: GangInitializationPolicy = DEFAULT_GANG_INITIALIZATION,
        gpu_rank_layout: GpuRankLayout = DEFAULT_GPU_RANK_LAYOUT,
        mtls: MutualTlsFiles | None = None,
    ) -> None:
        if not isinstance(initialization, GangInitializationPolicy):
            raise TypeError("native gang initialization must be a GangInitializationPolicy")
        if not isinstance(gpu_rank_layout, GpuRankLayout):
            raise TypeError("native gang GPU rank layout must be a GpuRankLayout")
        ordered_members = tuple(sorted(members, key=lambda member: member.cell_id))
        if len(ordered_members) < 2:
            raise ValueError("native gang execution requires at least two execution cells")
        if len({member.cell_id for member in ordered_members}) != len(ordered_members):
            raise ValueError("native gang execution cell IDs must be unique")

        flattened = [
            (member, cell_index, endpoint.rstrip("/"))
            for cell_index, member in enumerate(ordered_members)
            for endpoint in member.endpoints
        ]
        endpoints = tuple(endpoint for _, _, endpoint in flattened)
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("native gang execution worker endpoints must be unique across cells")
        if clients is not None and len(clients) != len(flattened):
            raise ValueError("native gang clients must match the number of worker endpoints")
        coordinator = urlparse(endpoints[0])
        if coordinator.scheme not in {"http", "https"} or not coordinator.hostname:
            raise ValueError("native gang coordinator endpoint must be an HTTP URL with a host")
        effective_mtls = mtls
        owned_endpoints = (
            endpoints
            if clients is None
            else tuple(endpoint for endpoint, client in zip(endpoints, clients, strict=True) if client is None)
        )
        if effective_mtls is not None and owned_endpoints:
            if any(urlparse(endpoint).scheme != "https" for endpoint in owned_endpoints):
                raise ValueError("mutual TLS requires HTTPS native-worker endpoints")
            client_ssl_context(effective_mtls)

        declared_gpu = [member.gpu_devices_per_host is not None for member in ordered_members]
        if any(declared_gpu) and not all(declared_gpu):
            # Partial knowledge is the dangerous state: the declared members
            # would be pinned to exact devices while the undeclared ones kept
            # whatever the submission said, and the gang would look coordinated.
            # An unknown device count stays unknown for the whole gang.
            undeclared = ", ".join(member.cell_id for member in ordered_members if member.gpu_devices_per_host is None)
            raise ValueError(
                "native gang GPU coordination requires every member to declare its device count; "
                f"missing for cell(s) {undeclared}"
            )

        rank_hosts = tuple(urlparse(endpoint).hostname or "" for endpoint in endpoints)
        if all(declared_gpu):
            # Only a GPU-coordinated gang is placed by host. A gang that is not
            # -- a TPU slice pair, or any member whose device count is unknown --
            # is dispatched exactly as it was before per-host grouping existed,
            # including when two of its endpoints share a machine.
            _refuse_unplaceable_gpu_ranks(endpoints, rank_hosts, layout=gpu_rank_layout)

        self.members = ordered_members
        self.endpoints = endpoints
        self.initialization = initialization
        self.gpu_rank_layout = gpu_rank_layout
        self._coordinator_host = coordinator.hostname
        self._coordinator_address = f"{coordinator.hostname}:{_COORDINATOR_PORT}"
        supplied_clients = clients or (None,) * len(flattened)
        self._workers = tuple(
            _GangWorker(
                cell_id=member.cell_id,
                cell_index=cell_index,
                endpoint=endpoint,
                context=member.context,
                executor=NativeRemoteExecutor(
                    endpoint,
                    secret=member.secret,
                    client=client,
                    **({"mtls": effective_mtls} if effective_mtls is not None else {}),
                ),
                gpu_devices=member.gpu_devices_per_host,
                gpu_model=member.gpu_model,
                jax_cluster_detection=member.jax_cluster_detection,
            )
            for (member, cell_index, endpoint), client in zip(flattened, supplied_clients, strict=True)
        )
        gpu_devices_per_rank = [
            member.gpu_devices_per_host for member, _, _ in flattened if member.gpu_devices_per_host is not None
        ]
        # Every member declared, so one entry per flattened host: a member with
        # four hosts contributes its device count four times, once per rank.
        # The hosts are passed only when every endpoint names one -- a rank whose
        # host cannot be named is not evidence that it is alone on a machine, so
        # the plan is told nothing rather than told something convenient.
        self.gpu_plan: GpuDevicePlan | None = (
            plan_gpu_devices(gpu_devices_per_rank, rank_hosts=rank_hosts if all(rank_hosts) else None)
            if all(declared_gpu)
            else None
        )
        # Checkpoint destinations this executor has launched, by workload remote
        # ID. ``status`` reports each rank's derived destination beside its
        # state; an executor resolved fresh for one request knows of no launch
        # and reports no destination.
        self._checkpoint_output_uris: dict[str, str] = {}

    def health(self) -> bool:
        """Return whether every native worker accepts its cell-scoped credential."""

        return all(worker.executor.health() for worker in self._workers)

    def _cell_world(self, worker: _GangWorker, rank: int) -> tuple[str, str, int, int]:
        """Return one worker's per-cell JAX world: host, address, count, local id.

        An elastic pack is one provider allocation per cell -- islands with no
        interconnect between them, so twelve processes told to rendezvous as one
        world deadline-exceed on every host. Elastic placement therefore runs
        one JAX world per cell: coordinator, process count, and process id
        local to the cell, while the ``EYVAN_*`` variables keep describing the
        whole gang.
        """

        first_cell_rank = next(
            index for index, candidate in enumerate(self._workers) if candidate.cell_id == worker.cell_id
        )
        cell_width = sum(1 for candidate in self._workers if candidate.cell_id == worker.cell_id)
        coordinator = urlparse(self._workers[first_cell_rank].endpoint)
        if coordinator.scheme not in {"http", "https"} or not coordinator.hostname:
            raise ValueError("native gang cell coordinator endpoint must be an HTTP URL with a host")
        return (
            coordinator.hostname,
            f"{coordinator.hostname}:{_COORDINATOR_PORT}",
            cell_width,
            rank - first_cell_rank,
        )

    def _ranked_spec(self, spec: tp.Mapping[str, JsonValue], rank: int, worker: _GangWorker) -> JsonObject:
        """Copy one JSON workload spec and add its rank-local environment."""

        payload: JsonObject = copy.deepcopy(dict(spec))
        raw_environment = payload.get("environment")
        if raw_environment is None:
            source_environment: JsonObject = {}
        elif isinstance(raw_environment, dict):
            source_environment = raw_environment
        else:
            raise ValueError("native gang workload environment must be a mapping")
        environment: dict[str, str] = {}
        for key, value in source_environment.items():
            if not isinstance(value, str):
                raise ValueError("native gang workload environment must be a string mapping")
            environment[key] = value
        cross_region_environment = {
            key: value
            for key, value in environment.items()
            if key == "EYVAN_HOST_REGION" or key.startswith("EYVAN_CROSS_REGION_")
        }
        environment = without_distributed_topology_overrides(environment)
        environment.update(cross_region_environment)
        if self.gpu_plan is not None:
            # Stripped here, before anything is written back, for the same
            # reason the JAX topology is: a value the submission chose must not
            # survive into a rank whose devices Eyvan has decided.
            environment = without_gpu_topology_overrides(environment)
        # The checkpoint hook writes to ``{base}/rank-{n}``, so a resume handed
        # the bare base URI reads an object that was never written: the workload
        # sees "no checkpoint" and silently restarts from scratch. The slice
        # dispatch derives the per-rank address for exactly this reason, and a
        # gang rank is checkpointed by the same hook, so both paths have to
        # agree; the name is imported rather than restated so they cannot drift.
        resume_uri = environment.get(_RESUME_CHECKPOINT_ENV, "")
        if resume_uri:
            environment[_RESUME_CHECKPOINT_ENV] = _rank_checkpoint_uri(resume_uri, rank)
        python_path = environment.get("PYTHONPATH", "")
        one_world_per_cell = payload.get("placement") == "elastic"
        if one_world_per_cell:
            coordinator_host, coordinator_address, process_count, process_id = self._cell_world(worker, rank)
        else:
            coordinator_address = self._coordinator_address
            coordinator_host = self._coordinator_address.rsplit(":", 1)[0]
            process_count = len(self._workers)
            process_id = rank
        environment.update(
            {
                "EYVAN_DISTRIBUTED_PROCESS_COUNT": str(len(self._workers)),
                "EYVAN_DISTRIBUTED_PROCESS_ID": str(rank),
                "EYVAN_GANG_CELL_ID": worker.cell_id,
                "EYVAN_GANG_CELL_INDEX": str(worker.cell_index),
                "EYVAN_JAX_MULTIHOST": "1",
                "EYVAN_REPLICA_COUNT": str(len(self._workers)),
                "EYVAN_REPLICA_INDEX": str(rank),
                "PYTHONPATH": f"{_MULTIHOST_SITE_PATH}:{python_path}" if python_path else _MULTIHOST_SITE_PATH,
            }
        )
        if one_world_per_cell and worker.jax_cluster_detection is JaxClusterDetectionMode.GCE_TPU:
            # A management endpoint may be public; GCE metadata identifies the
            # internal coordinator and process topology of this cell instead.
            environment["EYVAN_JAX_CLUSTER_DETECTION"] = JaxClusterDetectionMode.GCE_TPU.value
        else:
            environment.update(
                {
                    "JAX_COORDINATOR_ADDRESS": coordinator_address,
                    "JAX_COORDINATOR_PORT": str(_COORDINATOR_PORT),
                    "JAX_PARTITION_INDEX": str(worker.cell_index),
                    "JAX_PROCESS_COUNT": str(process_count),
                    "JAX_PROCESS_ID": str(process_id),
                    "JAX_PROCESS_INDEX": str(process_id),
                }
            )
        # Last, and after the topology scrub above, so the deadline a cold gang
        # gets is the one this control plane configured. A caller who submits
        # its own copy has already had it stripped -- the whole ``EYVAN_JAX_``
        # prefix is Eyvan's -- and could otherwise hand every rank a one-second
        # barrier and blame the fleet.
        environment.update(self.initialization.environment())
        if self.gpu_plan is not None:
            environment.update(
                self.gpu_plan.rank_environment(
                    rank,
                    rendezvous_host=coordinator_host,
                    gpu_model=worker.gpu_model,
                    layout=self.gpu_rank_layout,
                )
            )
        payload["environment"] = environment
        return payload

    def submit(self, spec: tp.Mapping[str, JsonValue]) -> str:
        """Submit one logical workload to every gang rank or retire a partial launch.

        A retire failure during partial-launch cleanup leaves a rank running,
        so it is logged at warning level and appended to the raised error to
        keep the orphaned rank observable.
        """

        raw_submission_id = spec.get("submission_id")
        if not isinstance(raw_submission_id, str) or not raw_submission_id.strip():
            raise ValueError("native gang submission requires a non-empty submission ID")
        expected_remote_id = raw_submission_id.strip()
        attempted: list[tuple[int, _GangWorker, set[str]]] = []
        try:
            for rank, worker in enumerate(self._workers):
                remote_ids = {expected_remote_id}
                attempted.append((rank, worker, remote_ids))
                remote_id = worker.executor.submit(self._ranked_spec(spec, rank, worker), worker.context)
                remote_ids.add(remote_id)
                if remote_id != expected_remote_id:
                    raise RuntimeError(
                        f"native gang worker returned unexpected remote ID {remote_id!r}; "
                        f"expected {expected_remote_id!r}"
                    )
            return expected_remote_id
        except BaseException as launch_error:
            retire_failures: list[str] = []
            for rank, worker, remote_ids in reversed(attempted):
                for attempted_remote_id in sorted(remote_ids):
                    try:
                        worker.executor.retire(attempted_remote_id, rollback_context(worker.context))
                    except Exception as exc:
                        retire_failures.append(f"rank {rank} ({worker.endpoint}): {type(exc).__name__}")
            if retire_failures:
                detail = f"partial-launch cleanup could not retire {_bounded_retire_summary(retire_failures)}"
                logger.warning("native gang submission failed and %s", detail)
                _augment_error_detail(launch_error, detail)
            raise

    def status(self, remote_id: str) -> NativeGangWorkloadResult:
        """Aggregate typed rank observations into one durable gang state."""

        workers: list[_GangWorkerObservation] = []
        # Derived exactly as the checkpoint hook derives its write destination,
        # so a row that names a checkpoint names an object that exists. Unknown
        # while this executor launched no checkpoint, and never invented.
        checkpoint_base = self._checkpoint_output_uris.get(remote_id)
        for rank, worker in enumerate(self._workers):
            try:
                observation = _normalize_worker_observation(
                    worker.executor.status(remote_id),
                    expected_remote_id=remote_id,
                    endpoint=worker.endpoint,
                )
            except httpx.TransportError as exc:
                # An unreachable worker endpoint is a fact about the observer's
                # connection, not about the rank: a closed keep-alive or a
                # restarting daemon reads identically to a dead process, and
                # treating it as LOST killed four healthy v6e-8 hosts whose
                # ranks had all exited 0 -- the fastest rank's vanishing
                # endpoint was read as its death, and the failure settled the
                # workload and tore down its three still-running peers.
                # INDETERMINATE keeps the plane polling: a blip resolves on the
                # next read, and a host that stays unreadable escalates through
                # the indeterminate grace path, which drains the cell rather
                # than freeing the lease while the workload may still run.
                observation = _indeterminate_observation(
                    remote_id,
                    f"native worker {worker.endpoint} is unreachable: {type(exc).__name__}: {exc}",
                )
            except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                observation = _indeterminate_observation(
                    remote_id,
                    _invalid_observation_reason(exc, endpoint=worker.endpoint),
                )
            workers.append(
                _GangWorkerObservation(
                    rank=rank,
                    worker=worker,
                    observation=observation,
                    checkpoint_uri=None if checkpoint_base is None else _rank_checkpoint_uri(checkpoint_base, rank),
                )
            )
        status = _aggregate_status(workers)
        message = _aggregate_message(workers)
        if status is ExecutionWorkloadStatus.FAILED:
            terminal_reason = self._terminal_failure_reason(remote_id, workers)
            if terminal_reason is not None:
                message = f"{terminal_reason} {message}"
        aggregate = ExecutionWorkloadObservation(
            remote_id=remote_id,
            status=status,
            message=message,
        )
        return {
            "remote_id": aggregate.remote_id,
            "status": aggregate.status,
            "message": aggregate.message,
            "workers": [worker.as_result() for worker in workers],
        }

    def _cross_region_marker_detail(
        self,
        remote_id: str,
        ranks: tp.Sequence[int],
        workers: tp.Sequence[_GangWorkerObservation],
    ) -> str | None:
        """Extract the cross-region ban stderr marker from worker observations or logs."""

        by_rank = {worker.rank: worker for worker in workers}
        for rank in sorted(set(ranks)):
            obs = by_rank.get(rank)
            if obs is not None:
                detail = extract_cross_region_ban_marker(obs.observation.message)
                if detail:
                    return detail
        for rank in sorted(set(ranks)):
            if not 0 <= rank < len(self._workers):
                continue
            try:
                logs = self._workers[rank].executor.logs(remote_id)
            except (httpx.HTTPError, RuntimeError, ValueError):
                continue
            detail = extract_cross_region_ban_marker(logs)
            if detail:
                return detail
        return None

    def _terminal_failure_reason(
        self,
        remote_id: str,
        workers: tp.Sequence[_GangWorkerObservation],
    ) -> str | None:
        """Return a structured terminal failure prefix for exit code 95 or gang-init failure."""

        marker_ranks = [
            worker.rank
            for worker in workers
            if CROSS_REGION_TRANSFER_BANNED_MARKER in worker.observation.message
            or CROSS_REGION_TRANSFER_BANNED_REASON in worker.observation.message.lower()
        ]
        if not self.initialization.fatal:
            if not marker_ranks:
                return None
            return cross_region_ban_failure_reason(
                banned_ranks=marker_ranks,
                marker_detail=self._cross_region_marker_detail(remote_id, marker_ranks, workers),
            )

        banned_ranks: list[int] = list(marker_ranks)
        deadline_ranks: list[int] = []
        error_ranks: list[int] = []
        for rank, worker in enumerate(self._workers):
            try:
                detail = worker.executor.process_status(remote_id)
            except (httpx.HTTPError, RuntimeError, ValueError):
                continue
            exit_status = _reported_exit_status(detail)
            if exit_status == CROSS_REGION_TRANSFER_BANNED_EXIT_CODE:
                banned_ranks.append(rank)
                continue
            outcome = gang_initialization_outcome(exit_status)
            if outcome == "deadline":
                deadline_ranks.append(rank)
            elif outcome == "error":
                error_ranks.append(rank)
        if banned_ranks:
            return cross_region_ban_failure_reason(
                banned_ranks=banned_ranks,
                marker_detail=self._cross_region_marker_detail(remote_id, banned_ranks, workers),
            )
        if not deadline_ranks and not error_ranks:
            return None
        return gang_formation_failure_reason(
            deadline_s=self.initialization.initialization_timeout_s,
            deadline_ranks=deadline_ranks,
            error_ranks=error_ranks,
        )

    def stop(self, remote_id: str) -> None:
        """Stop every gang rank using its original cell-scoped fence."""

        self._fanout_mutation("stop", remote_id)

    def retire(self, remote_id: str) -> None:
        """Retire every gang rank using its original cell-scoped fence."""

        self._fanout_mutation("retire", remote_id)

    def _fanout_mutation(self, action: _MutationAction, remote_id: str) -> None:
        """Apply one closed mutation action to every worker."""

        first_error: BaseException | None = None
        for worker in self._workers:
            try:
                if action == "stop":
                    worker.executor.stop(remote_id, worker.context)
                elif action == "retire":
                    worker.executor.retire(remote_id, worker.context)
                else:
                    raise ValueError(f"unsupported native gang mutation action {action!r}")
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str:
        """Launch one checkpoint hook across every cell under a single shared ID.

        Two things differ from a single-cell slice and both are load-bearing.

        A gang holds one lease per cell, so every worker must be addressed with
        *its own* fence -- the shared ``context`` would be rejected by every
        worker but one. And each cell's context carries a different idempotency
        key, so passing them through unchanged would mint N different checkpoint
        IDs and leave nothing for the caller to poll: ``status`` asks each rank
        about one logical checkpoint. Each worker is therefore addressed with
        its own fence carrying the *shared* checkpoint ID, which is exactly the
        asymmetry ``submit`` already has between per-cell fences and one logical
        submission ID.

        Launch is all-or-nothing, mirroring ``submit``: a partial launch is
        retired -- each worker with its own context -- before the original error
        is raised, because a surviving hook would hold accelerators and could
        still write to the checkpoint destination. Completion is not asserted
        here; callers poll ``status``, which reports success only once every
        rank has succeeded.
        """

        if not isinstance(hook, str) or not hook.strip():
            raise ValueError("native gang checkpoint hook must be non-empty")
        checkpoint_id = context.idempotency_key
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("native gang checkpoint requires a shared idempotency key")
        attempted: list[tuple[int, _GangWorker, set[str]]] = []
        try:
            for rank, worker in enumerate(self._workers):
                remote_ids = {checkpoint_id}
                attempted.append((rank, worker, remote_ids))
                worker_context = replace(worker.context, idempotency_key=checkpoint_id)
                launched = worker.executor.checkpoint(
                    remote_id,
                    hook=hook,
                    output_uri=output_uri,
                    context=worker_context,
                )
                remote_ids.add(launched)
                if launched != checkpoint_id:
                    raise RuntimeError(
                        f"native gang worker returned unexpected checkpoint ID {launched!r}; expected {checkpoint_id!r}"
                    )
            # Recorded only once every rank launched, so a status can never name
            # a destination for a checkpoint this executor failed to start.
            if output_uri:
                self._checkpoint_output_uris[remote_id] = output_uri
            return checkpoint_id
        except BaseException as launch_error:
            retire_failures: list[str] = []
            for rank, worker, remote_ids in reversed(attempted):
                for attempted_remote_id in sorted(remote_ids):
                    try:
                        # Each worker is retired under its own cell fence; the
                        # shared checkpoint ID names *what* to retire, never who
                        # is allowed to retire it.
                        worker.executor.retire(
                            attempted_remote_id,
                            rollback_context(replace(worker.context, idempotency_key=checkpoint_id)),
                        )
                    except Exception as exc:
                        retire_failures.append(
                            f"rank {rank} cell {worker.cell_id} ({worker.endpoint}): {type(exc).__name__}"
                        )
            if retire_failures:
                detail = f"partial checkpoint cleanup could not retire {_bounded_retire_summary(retire_failures)}"
                logger.warning("native gang checkpoint failed and %s", detail)
                _augment_error_detail(launch_error, detail)
            raise

    def logs(self, remote_id: str) -> str:
        """Return rank-prefixed logs from every gang worker when available."""

        sections: list[str] = []
        for rank, worker in enumerate(self._workers):
            try:
                logs = worker.executor.logs(remote_id)
            except httpx.TransportError as exc:
                logs = f"native worker is unreachable: {type(exc).__name__}: {exc}"
            except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                logs = f"native worker log retrieval failed: {type(exc).__name__}: {exc}"
            sections.append(f"===== rank {rank} cell {worker.cell_id} {worker.endpoint} =====\n{logs}")
        return "\n".join(sections)

    def log_page(self, remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        """Return one structured page across every globally ranked gang host.

        Members are sorted by cell ID and flattened in endpoint order during
        construction, the same order used to assign JAX process IDs. Log ranks
        therefore describe the process that emitted a line rather than merely
        the cell that happened to contain it.

        A cursor this gang cannot resolve is answered as an unavailable page
        naming the codec's refusal rather than as a silent restart, for the same
        reason the slice executor does: a restart replays every host's journal
        while reporting a clean read.
        """

        expected = CursorContext(
            kind=CURSOR_KIND_WORKLOAD_LOG,
            mode=CURSOR_MODE_LIVE,
            ranks=tuple(range(len(self._workers))),
        )
        outcome = decode_log_cursor(after, expected=expected)
        if not isinstance(outcome, (CursorFresh, CursorResume)):
            return LogPage(
                unavailable=True,
                detail=f"{outcome.reason}: {outcome.detail}",
                truncated_ranks=tuple(range(len(self._workers))),
            )
        _attempt = outcome.attempt if isinstance(outcome, CursorResume) else None
        cursors = outcome.cursors if isinstance(outcome, CursorResume) else {}
        pages: dict[int, LogPage] = {}
        next_cursors: dict[int, str] = {}
        for rank, worker in enumerate(self._workers):
            try:
                page = worker.executor.log_page(remote_id, after=cursors.get(rank, ""), limit=limit)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                page = LogPage(unavailable=True, detail=f"{type(exc).__name__}: {exc}")
            pages[rank] = LogPage(
                records=tuple(replace(record, rank=rank) for record in page.records),
                next_cursor=page.next_cursor,
                unavailable=page.unavailable,
                detail=page.detail,
                truncated_ranks=(rank,) if page.truncated_ranks else (),
            )
            advanced = cursors.get(rank, "") if page.unavailable else (page.next_cursor or cursors.get(rank, ""))
            if advanced:
                next_cursors[rank] = advanced
        merged = merge_pages(pages)
        return LogPage(
            records=merged.records,
            next_cursor=join_log_cursor(_attempt or 0, next_cursors),
            unavailable=merged.unavailable,
            detail=merged.detail,
            truncated_ranks=merged.truncated_ranks,
        )

    def process_status(self, remote_id: str) -> JsonObject:
        """Return per-cell process detail across every gang member."""

        ranks: list[JsonValue] = []
        for rank, worker in enumerate(self._workers):
            try:
                detail: JsonValue = worker.executor.process_status(remote_id)
            except httpx.TransportError as exc:
                detail = {"observed": False, "detail": f"native worker is unreachable: {type(exc).__name__}: {exc}"}
            except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                detail = {"observed": False, "detail": f"process status failed: {type(exc).__name__}: {exc}"}
            ranks.append({"rank": rank, "cell_id": worker.cell_id, "endpoint": worker.endpoint, "status": detail})
        return {"remote_id": remote_id, "ranks": ranks}

    def exec_in_workload(
        self,
        remote_id: str,
        command: tp.Sequence[str],
        *,
        timeout_s: float = 30.0,
    ) -> JsonObject:
        """Run one diagnostic command beside this workload on every rank.

        The multi-cell counterpart of the slice fan-out, and unreachable ranks
        are reported the same way: named and numbered rather than omitted, so a
        gang missing a member reads as missing rather than as complete.
        """

        workers = [(rank, worker.endpoint, worker.executor) for rank, worker in enumerate(self._workers)]
        return {"remote_id": remote_id, "ranks": _exec_fanout(workers, remote_id, command, timeout_s=timeout_s)}

    def close(self) -> None:
        """Close internally owned native-worker clients."""

        for worker in self._workers:
            worker.executor.close()


__all__ = ("NativeGangExecutor", "NativeGangMember")
