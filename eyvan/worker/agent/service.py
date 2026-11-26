# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cell-agent service composed from independent capacity, state, and execution backends.

**Deployment order for managed staging: native workers first, then agents, then
the controller.** This module serves one wire method the control plane did not
previously call -- ``stage_asset``, reached at
``POST /api/v1/cells/{cell_id}/stage-actions``. An agent that predates it
answers 404, which the client reports as *this agent has no staging executor*:
absence of a capability, never a failed action. So a control plane upgraded
first stages nothing and records nothing, exactly as it does today.

For ``eyvan-native`` the same request travels one hop further, because the
workload's disk belongs to a native worker on another machine rather than to
this process. This agent fences the request against its own durable record and
forwards it to that worker's identical route; a worker that predates the route
answers 404 and one that declares no source prefixes answers 501, and both come
back here as *no executor* and are reported on unchanged. Every prefix of the
deployment order above is therefore inert rather than broken.
"""

from __future__ import annotations

import typing as tp
from dataclasses import replace
from pathlib import Path

from fastapi import HTTPException

from eyvan._internal.correlation import traced_operation
from eyvan._internal.json_types import JsonObject, JsonValue, model_json
from eyvan.capacity import DEFINITIVE_CAPACITY_LOSS_STATUSES, READY_CAPACITY_STATUSES, CapacityStatus
from eyvan.capacity.drivers.base import (
    CapacityDriver,
    CellObservation,
    ExecutionDriver,
    ExecutionDriverResolver,
    ExecutionWorkloadObservation,
    ExecutionWorkloadReport,
    ExecutionWorkloadStatus,
    MutationContext,
    StaleMutationError,
    UnsupportedCapabilityError,
    canonical_execution_backend,
    rollback_context,
)
from eyvan.capacity.drivers.registry import DEFAULT_CAPACITY_BACKENDS
from eyvan.capacity.fleet_state import FleetStateStore, StaleFenceError
from eyvan.config import EyvanSettings
from eyvan.identity.mtls import MutualTlsConfigurationError, MutualTlsFiles
from eyvan.worker.agent.models import (
    AgentCapabilities,
    AgentObservation,
    CapacityRequest,
    CellStageAction,
    CellStageActionRequest,
    CellStageActionResult,
    CellStageVerdict,
    CheckpointRequest,
    GangWorkloadMember,
    GangWorkloadMutationRequest,
    GangWorkloadRequest,
    MutationModel,
    WorkloadMutationRequest,
    WorkloadObservation,
    WorkloadRequest,
)
from eyvan.worker.runtime.native_slice import MultiHostStageCapableDriver
from eyvan.worker.runtime.registry import (
    ExecCapableExecutionDriver,
    IntrospectableExecutionDriver,
    NativeGangCell,
    RankedExecCapableExecutionDriver,
    StructuredLogExecutionDriver,
)
from eyvan.worker.runtime.stage_executor import AssetStageDriver, AssetStageVerdict, LocalAssetStageExecutor

if tp.TYPE_CHECKING:
    from eyvan.worker.runtime.native_gang import NativeGangWorkloadResult

#: Execution backends whose workloads run on the disk this agent process owns.
#:
#: The gate that keeps a stage row honest. A stage record says *cell X holds
#: asset A*, and placement is steered by it, so an agent that wrote the bytes to
#: a disk the workload cannot read would be manufacturing exactly the confident
#: false hit the whole design refuses. ``native-local`` runs the workload as a
#: child of this process, so this agent's disk is the workload's disk and the
#: executor is built here.
_HOST_LOCAL_EXECUTION_BACKENDS = frozenset({"native-local"})

#: Execution backends whose workloads run on a disk some other process owns.
#:
#: The same rule from the other side: the bytes have to land where the workload
#: will read them, so this agent forwards to the execution driver it already
#: resolves for that cell rather than writing anything itself. Whether that
#: driver can stage is a question asked of the driver -- a resolved executor
#: that does not satisfy
#: :class:`~eyvan.worker.runtime.stage_executor.AssetStageDriver` reads as no
#: executor.
#:
#: A cell with several hosts answers it differently from one with a single host,
#: which is why there are two questions in ``_remote_stager`` rather than one. A
#: slice has no single disk to name, so it does not satisfy that protocol
#: directly; it satisfies
#: :class:`~eyvan.worker.runtime.native_slice.MultiHostStageCapableDriver`
#: instead and hands back a view bound to the generation this agent just fenced
#: against, which then fans every action to every one of its machines.
_REMOTE_STAGING_EXECUTION_BACKENDS = frozenset({"eyvan-native"})

#: What one staging verdict is called on the wire.
_STAGE_VERDICTS: dict[AssetStageVerdict, CellStageVerdict] = {
    AssetStageVerdict.OK: CellStageVerdict.OK,
    AssetStageVerdict.REFUSED: CellStageVerdict.REFUSED,
    AssetStageVerdict.INCONCLUSIVE: CellStageVerdict.INCONCLUSIVE,
}


#: The host-side half of managed staging, as this agent uses it.
#:
#: Defined beside the executor rather than here, because both things that
#: satisfy it live there: the executor that runs on a disk, and the native
#: worker driver that forwards to the process which does.
CellStageDriver: tp.TypeAlias = AssetStageDriver


MutationRequest: tp.TypeAlias = CapacityRequest | CheckpointRequest | WorkloadMutationRequest | WorkloadRequest
GangMemberKey: tp.TypeAlias = tuple[str, str, int, str] | tuple[str, str, int, str, str]


class InvalidWorkloadRequestError(HTTPException, ValueError):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A workload request that failed validation, mapped to HTTP 422 by the agent API.

    Also subclasses :class:`ValueError` so direct in-process callers keep the
    established exception contract.

    The suppression is for the hazard this class does not have: the rule warns
    that a base initializer may go uncalled, and ``__init__`` below calls both
    explicitly. ``test_native_gang_agent`` asserts both halves of the contract --
    ``isinstance(error, ValueError)`` and ``status_code == 422`` -- on the same
    raised object, so a regression in either fails a test rather than a linter.
    """

    def __init__(self, detail: str) -> None:
        HTTPException.__init__(self, status_code=422, detail=detail)
        ValueError.__init__(self, detail)


class GangExecutionDriver(tp.Protocol):
    """Native multi-cell execution operations used by the agent."""

    def submit(self, spec: tp.Mapping[str, JsonValue]) -> str: ...

    def status(self, remote_id: str) -> NativeGangWorkloadResult: ...

    def stop(self, remote_id: str) -> None: ...

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str: ...

    def retire(self, remote_id: str) -> None: ...

    def logs(self, remote_id: str) -> str: ...

    def close(self) -> None: ...


@tp.runtime_checkable
class ReadinessResolver(tp.Protocol):
    """Resolver capability for probing a cell runtime."""

    def readiness(self, cell_id: str, observation: CellObservation) -> tuple[bool, str]: ...


@tp.runtime_checkable
class ClosableResolver(tp.Protocol):
    """Resolver capability for releasing every cached driver."""

    def close(self) -> None: ...


@tp.runtime_checkable
class CellClosableResolver(tp.Protocol):
    """Resolver capability for releasing one cell's cached drivers."""

    def close_cell(self, cell_id: str) -> None: ...


@tp.runtime_checkable
class GangExecutionResolver(tp.Protocol):
    """Resolver capability for constructing a native multi-cell driver."""

    def resolve_gang(self, cells: tp.Sequence[NativeGangCell]) -> GangExecutionDriver: ...


def _context(request: MutationRequest) -> MutationContext:
    mutation = request.mutation
    return MutationContext(
        idempotency_key=mutation.idempotency_key,
        generation=mutation.generation,
        fencing_token=mutation.fencing_token,
        correlation_id=mutation.correlation_id,
    )


def _workload_observation(report: ExecutionWorkloadReport) -> WorkloadObservation:
    """Project a typed executor report onto the stable agent wire model."""

    if isinstance(report, ExecutionWorkloadObservation):
        return WorkloadObservation(
            remote_id=report.remote_id,
            status=report.status,
            message=report.message,
        )
    return WorkloadObservation(
        remote_id=report["remote_id"],
        status=report["status"],
        message=report["message"],
    )


#: Statuses that prove an execution reached a terminal state.
_TERMINAL_RETIRE_STATUSES = frozenset(
    {
        ExecutionWorkloadStatus.SUCCEEDED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.LOST,
    }
)

#: Retirement must positively establish quiescence before a successor launches.
#: An absent execution is also safe; INDETERMINATE proves neither running nor stopped.
_QUIESCENT_RETIRE_STATUSES = _TERMINAL_RETIRE_STATUSES | {ExecutionWorkloadStatus.NOT_FOUND}


def _retire_observation(report: ExecutionWorkloadReport) -> WorkloadObservation:
    """Project the post-retire read-back without inventing a terminal state.

    A retire call returning is not proof that a rank stopped: the historical
    systemd replay path returns early with the durable record still
    non-terminal, and a remote worker's verdict was previously discarded on
    the way back. Report what the record now says. Only an absent record
    collapses to STOPPED -- nothing left behind can run.
    """

    observation = _workload_observation(report)
    # Distributed workload viability is not retirement evidence: FAILED or
    # LOST can describe a slice with another rank still running or unobserved.
    # Check supplied rank evidence before projecting it away for the wire.
    if not isinstance(report, ExecutionWorkloadObservation) and any(
        worker["status"] not in _QUIESCENT_RETIRE_STATUSES for worker in report.get("workers", [])
    ):
        return observation.model_copy(
            update={
                "status": ExecutionWorkloadStatus.INDETERMINATE,
                "message": f"retirement not confirmed for every rank; {observation.message}",
            }
        )
    if observation.status is ExecutionWorkloadStatus.NOT_FOUND:
        return observation.model_copy(
            update={"status": ExecutionWorkloadStatus.STOPPED, "message": "retired; execution record absent"}
        )
    return observation


class CellAgentService:
    """Reconcile capacity and delegate workloads to a selected execution backend."""

    def __init__(
        self,
        capacity: CapacityDriver | None = None,
        *,
        state: FleetStateStore | None = None,
        executors: ExecutionDriverResolver | None = None,
        settings: EyvanSettings | None = None,
        remote_retire: tp.Callable[[str, str, WorkloadMutationRequest], None] | None = None,
        #: Puts named assets on the workload's disk and takes them off again.
        #: Built here for a host-local execution backend when omitted; for a
        #: backend whose workloads run elsewhere it is left unset and resolved
        #: per cell from that cell's execution driver -- see
        #: ``_HOST_LOCAL_EXECUTION_BACKENDS`` and
        #: ``_REMOTE_STAGING_EXECUTION_BACKENDS``.
        stager: CellStageDriver | None = None,
    ):
        settings = settings or EyvanSettings.from_env()
        self.settings = settings
        self.state = state or FleetStateStore.from_uri(settings.fleet_state_uri)
        if capacity is None:
            capacity = DEFAULT_CAPACITY_BACKENDS.create(
                settings.capacity_backend,
                state=self.state,
                settings=settings,
            )
        execution_backend = canonical_execution_backend(capacity.capabilities().executor)
        if executors is None:
            from eyvan.worker.runtime.registry import ExecutionDriverResolver as ConfiguredExecutionResolver

            resolver_kwargs: dict[str, tp.Any] = {
                "state_dir": settings.agent_state_dir,
                "native_worker_port": settings.native_worker_port,
                "native_worker_token": settings.native_worker_root_secret,
                "gang_initialization": settings.gang_initialization_policy(),
            }
            if settings.mutual_tls_client_enabled and execution_backend == "eyvan-native":
                resolver_kwargs["mtls"] = MutualTlsFiles(
                    trust_bundle=settings.mutual_tls_trust_bundle,
                    certificate=settings.mutual_tls_certificate,
                    private_key=settings.mutual_tls_private_key,
                )
            executors = ConfiguredExecutionResolver(capacity.capabilities().executor, **resolver_kwargs)
        self.capacity = capacity
        self.execution_backend = execution_backend
        self.executors = executors
        self._remote_retire = remote_retire or self._retire_on_source_agent
        if stager is None and self.execution_backend in _HOST_LOCAL_EXECUTION_BACKENDS:
            candidate = LocalAssetStageExecutor.from_environment(Path(settings.agent_state_dir).expanduser() / "stages")
            # An empty source allowlist is the off switch, and it must read as
            # *no executor* rather than as a refusal of every asset. A refusal is
            # terminal -- it blocks the stage permanently, and nothing unblocks
            # it short of replacing the cell -- so an operator who sized a cache
            # before declaring what may be fetched would poison the ledger for
            # every asset in the fleet and have no way back.
            stager = candidate if candidate.allowed_prefixes else None
        self.stager = stager

    @staticmethod
    def _observation(value: CellObservation) -> AgentObservation:
        return AgentObservation(
            cell_id=value.cell_id,
            state=value.state,
            generation=value.generation,
            execution_endpoint=value.execution_endpoint,
            detail=value.detail,
            metadata=value.metadata,
            observed_tpu_vms=value.observed_tpu_vms,
            runtime_process_count=value.runtime_process_count,
            failure_code=value.failure_code,
        )

    def _with_execution_readiness(self, value: CellObservation) -> CellObservation:
        if value.state not in READY_CAPACITY_STATUSES:
            return value
        if not isinstance(self.executors, ReadinessResolver):
            return value
        try:
            ready, detail = self.executors.readiness(value.cell_id, value)
        except Exception as exc:
            # A failing probe must not take down observation endpoints: report
            # the cell as still connecting so the controller keeps fencing
            # scheduling instead of seeing an opaque HTTP 500.
            return replace(
                value,
                state=CapacityStatus.EXECUTION_CONNECTING,
                detail=f"execution readiness probe failed: {exc}",
            )
        if ready:
            return value
        # Provider capacity is still alive. Keep this distinct from a provider
        # provisioning state so the controller fences scheduling without
        # recovering a workload that can still be running on the VM.
        return replace(value, state=CapacityStatus.EXECUTION_CONNECTING, detail=detail)

    def capabilities(self) -> AgentCapabilities:
        capabilities = self.capacity.capabilities()
        return AgentCapabilities(
            provider=capabilities.provider,
            executor=capabilities.executor,
            accelerator_types=capabilities.accelerator_types,
            capacity_classes=capabilities.capacity_classes,
            # Forwarded rather than dropped: capability-based routing reads
            # this to decide which agents can serve a requested capacity
            # backend, and an agent whose declaration was stripped here
            # looked like an older agent that declares nothing.
            capacity_backends=capabilities.capacity_backends,
            supports_checkpoint=capabilities.supports_checkpoint,
            supports_callables=capabilities.supports_callables,
            supports_exec=capabilities.supports_exec,
            max_gang_replicas=capabilities.max_gang_replicas,
            resource_enforcement=capabilities.resource_enforcement,
        )

    def inventory(self) -> list[dict[str, str | int | None]]:
        """Return every cell this agent holds durable state for.

        Other routes are keyed by ``cell_id``. This enumerates the inventory
        without requiring direct access to the agent's state document.

        That matters because the document outlives the control plane that wrote
        it. A control plane whose database is replaced leaves its cells behind
        with ``desired_state: up``, and the agent faithfully keeps re-ensuring
        them -- capacity nobody asked for, recreated indefinitely, invisible to
        the control plane that would otherwise own it.

        Read-only and deliberately narrow: identity, lifecycle and intent, so an
        operator can diff it against a control plane. Nothing here can mutate.
        """

        return [
            {
                "name": record.name,
                "state": str(record.state),
                "desired_state": record.desired_state,
                "generation": record.generation,
                "project": record.project,
                "zone": record.zone,
                "accelerator_type": record.accelerator_type,
                "qr_id": record.qr_id,
            }
            # Sorted here rather than relying on the storage layer: the local
            # backend happens to write with sorted keys, but that is its choice
            # and not a contract this method should inherit.
            for record in sorted(self.state.load().values(), key=lambda item: item.name)
        ]

    def observe(self, cell_id: str) -> AgentObservation:
        return self._observation(self._with_execution_readiness(self.capacity.observe(cell_id)))

    def reconcile(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        context = _context(request)
        with traced_operation("provider.reconcile", context.correlation_id):
            value = self.capacity.reconcile(cell_id, context)
            return self._observation(self._with_execution_readiness(value))

    def apply_capacity(self, cell_id: str, request: CapacityRequest) -> AgentObservation:
        context = _context(request)
        with traced_operation("provider.apply_capacity", context.correlation_id):
            value = self.capacity.apply_capacity(cell_id, request.desired, context)
            return self._observation(self._with_execution_readiness(value))

    def drain(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        context = _context(request)
        with traced_operation("provider.drain", context.correlation_id):
            return self._observation(self.capacity.drain(cell_id, context))

    def release(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        context = _context(request)
        with traced_operation("provider.release", context.correlation_id):
            value = self.capacity.release(cell_id, context)
            self._close_execution_cell(cell_id)
            return self._observation(value)

    def worker_builds(self, cell_id: str) -> list[dict[str, object]] | None:
        """Return what code each native worker of one cell is running.

        Three answers, because there are three situations and conflating any two
        of them reports agreement nobody established.

        A list is the drivers that ran a probe. ``[]`` means this backend truly
        has no separate worker -- a local driver runs the workload in the agent
        process, whose build the agent already reports. ``None`` means this
        driver cannot be asked at all.

        The last case was returning ``[]``, and a live GCE cell running an
        active native worker therefore reported no ranks while the sweep called
        the fleet in agreement. That is the failure this whole check exists to
        prevent, reached from inside the check.
        """

        driver = self.executors.resolve(cell_id, self.capacity.observe(cell_id))
        collect = getattr(driver, "worker_builds", None)
        if collect is None:
            return None
        return list(collect())

    def close(self) -> None:
        """Release execution-driver resources when the agent process stops."""

        if isinstance(self.executors, ClosableResolver):
            self.executors.close()

    def submit(self, cell_id: str, request: WorkloadRequest) -> WorkloadObservation:
        metadata = request.spec.get("metadata")
        workload_id = metadata.get("eyvan_workload_id") if isinstance(metadata, dict) else None
        if not isinstance(workload_id, str) or not workload_id.strip():
            raise InvalidWorkloadRequestError("workload metadata must include a non-empty eyvan_workload_id")
        context = _context(request)
        expected_remote_id = request.spec.get("submission_id")
        if expected_remote_id is None:
            remote_id = context.idempotency_key
        elif isinstance(expected_remote_id, str) and expected_remote_id.strip():
            remote_id = expected_remote_id.strip()
        else:
            raise InvalidWorkloadRequestError("workload submission_id must be a non-empty string")
        submitted_executor = None
        launched_remote_id: str | None = None
        try:
            with self.state.workload_submission(
                workload_id.strip(),
                cell_id,
                remote_id,
                context.generation,
                context.fencing_token,
            ) as predecessors:
                for predecessor in predecessors:
                    self._retire_predecessor(workload_id.strip(), cell_id, context, predecessor)
                    self.state.require_lease()
                self.state.clear_workload_predecessors(
                    workload_id.strip(),
                    context.generation,
                    context.fencing_token,
                )
                self.state.require_lease()
                submitted_executor = self._executor(cell_id)
                remote_id = submitted_executor.submit(request.spec, context)
                launched_remote_id = remote_id
                self.state.require_lease()
        except StaleFenceError as exc:
            if submitted_executor is not None and launched_remote_id is not None:
                try:
                    submitted_executor.retire(launched_remote_id, rollback_context(context))
                except BaseException as cleanup_exc:
                    raise StaleMutationError(
                        f"{exc}; launched execution {launched_remote_id!r} could not be retired after lease loss"
                    ) from cleanup_exc
            raise StaleMutationError(str(exc)) from exc
        except BaseException as exc:
            # The launch already happened; every later failure in this window
            # must roll the launch back, not only fence losses -- otherwise a
            # response lost after the remote accepted the job leaves an orphan
            # holding the fence. The original error stays the one the caller
            # sees unless the cleanup itself fails too.
            if submitted_executor is not None and launched_remote_id is not None:
                try:
                    submitted_executor.retire(launched_remote_id, rollback_context(context))
                except BaseException as cleanup_exc:
                    raise RuntimeError(
                        f"{exc}; launched execution {launched_remote_id!r} could not be retired "
                        f"after {type(exc).__name__}"
                    ) from cleanup_exc
            raise
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.SUBMITTED)

    @staticmethod
    def _gang_member_key(member: GangWorkloadMember, *, include_idempotency_key: bool) -> GangMemberKey:
        mutation = member.mutation
        key = (
            member.cell_id,
            member.agent_endpoint,
            mutation.generation,
            mutation.fencing_token,
        )
        return (*key, mutation.idempotency_key) if include_idempotency_key else key

    @classmethod
    def _ordered_gang_members(cls, members: tp.Sequence[GangWorkloadMember]) -> tuple[GangWorkloadMember, ...]:
        ordered = tuple(sorted(members, key=lambda member: member.cell_id))
        if len(ordered) < 2:
            raise ValueError("native workload gang requires at least two member cells")
        if len({member.cell_id for member in ordered}) != len(ordered):
            raise ValueError("native workload gang member cells must be unique")
        if len({member.agent_endpoint for member in ordered}) != 1:
            raise ValueError("native workload gang members must use the same agent_endpoint")
        generations = {member.mutation.generation for member in ordered}
        if len(generations) != 1:
            raise ValueError("native workload gang member generations must match")
        return ordered

    @staticmethod
    def _gang_workload_id(spec: tp.Mapping[str, JsonValue]) -> str:
        metadata = spec.get("metadata")
        workload_id = metadata.get("eyvan_workload_id") if isinstance(metadata, dict) else None
        if not isinstance(workload_id, str) or not workload_id.strip():
            raise InvalidWorkloadRequestError("workload metadata must include a non-empty eyvan_workload_id")
        return workload_id.strip()

    @classmethod
    def _gang_member_payload(cls, members: tp.Sequence[GangWorkloadMember]) -> tuple[JsonObject, ...]:
        return tuple(model_json(member) for member in cls._ordered_gang_members(members))

    def _gang_cells(self, members: tp.Sequence[GangWorkloadMember]) -> tuple[NativeGangCell, ...]:
        """Capture the endpoints and fences used to launch or retire one gang."""

        if not isinstance(self.executors, GangExecutionResolver):
            raise ValueError(f"execution backend {self.execution_backend!r} does not support native workload gangs")

        cells = []
        for member in self._ordered_gang_members(members):
            observation = self.capacity.observe(member.cell_id)
            if observation.cell_id != member.cell_id:
                raise StaleMutationError(
                    f"native gang observation belongs to {observation.cell_id!r}, not member {member.cell_id!r}"
                )
            actual_endpoint = observation.metadata.get("agent_endpoint")
            if isinstance(actual_endpoint, str) and actual_endpoint.strip() and actual_endpoint != member.agent_endpoint:
                raise StaleMutationError(f"native gang member {member.cell_id!r} changed agent endpoint after placement")
            cells.append(
                NativeGangCell(
                    cell_id=member.cell_id,
                    observation=observation,
                    context=_context(WorkloadMutationRequest(mutation=member.mutation)),
                )
            )
        return tuple(cells)

    def _gang_executor(self, members: tp.Sequence[GangWorkloadMember]) -> GangExecutionDriver:
        """Resolve direct native-worker clients for every cell in one gang."""

        cells = self._gang_cells(members)
        return tp.cast(GangExecutionResolver, self.executors).resolve_gang(cells)

    def _rollback_gang(self, cells: tuple[NativeGangCell, ...], remote_id: str) -> None:
        """Retire a launch without rereading failed storage or replaying submission keys."""

        executor = tp.cast(GangExecutionResolver, self.executors).resolve_gang(
            tuple(replace(cell, context=rollback_context(cell.context)) for cell in cells)
        )
        try:
            executor.retire(remote_id)
        finally:
            self._close_gang_executor(executor)

    @staticmethod
    def _close_gang_executor(executor: GangExecutionDriver) -> None:
        executor.close()

    def _stored_gang_members(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
    ) -> tuple[GangWorkloadMember, ...]:
        fence = self.state.workload_fence(workload_id)
        if not isinstance(fence, dict) or fence.get("kind") != "gang":
            raise StaleMutationError(f"workload {workload_id!r} is not owned by a native workload gang")
        if fence.get("coordinator_cell_id") != coordinator_cell_id or fence.get("remote_id") != remote_id:
            raise StaleMutationError(f"native workload gang no longer owns {workload_id!r}")
        raw_members = fence.get("members")
        if not isinstance(raw_members, list):
            raise StaleMutationError(f"native workload gang {workload_id!r} has invalid member state")
        try:
            members = tuple(GangWorkloadMember.model_validate(member) for member in raw_members)
            return self._ordered_gang_members(members)
        except Exception as exc:
            raise StaleMutationError(f"native workload gang {workload_id!r} has invalid member state") from exc

    @classmethod
    def _validate_gang_mutation_members(
        cls,
        stored: tp.Sequence[GangWorkloadMember],
        supplied: tp.Sequence[GangWorkloadMember],
    ) -> tuple[GangWorkloadMember, ...]:
        ordered = cls._ordered_gang_members(supplied)
        expected = tuple(cls._gang_member_key(member, include_idempotency_key=False) for member in stored)
        actual = tuple(cls._gang_member_key(member, include_idempotency_key=False) for member in ordered)
        if actual != expected:
            raise StaleMutationError("native workload gang mutation does not own every member fence")
        return ordered

    def _retire_gang_predecessor(
        self,
        workload_id: str,
        coordinator_cell_id: str,
        generation: int,
        predecessor: JsonObject,
        correlation_id: str | None = None,
    ) -> None:
        """Fence a predecessor gang before launching a successor generation."""

        if predecessor.get("kind") != "gang":
            mutation = MutationContext(
                idempotency_key=f"supersede:{workload_id}:{generation}",
                generation=generation,
                fencing_token=f"gang:{coordinator_cell_id}:{generation}",
                correlation_id=correlation_id,
            )
            self._retire_predecessor(workload_id, coordinator_cell_id, mutation, predecessor)
            return
        remote_id = predecessor.get("remote_id")
        raw_members = predecessor.get("members")
        if not isinstance(remote_id, str) or not remote_id or not isinstance(raw_members, list):
            raise StaleMutationError(f"workload {workload_id!r} has an invalid predecessor gang fence")
        try:
            prior_members = tuple(GangWorkloadMember.model_validate(member) for member in raw_members)
            prior_members = self._ordered_gang_members(prior_members)
        except Exception as exc:
            raise StaleMutationError(f"workload {workload_id!r} has an invalid predecessor gang fence") from exc
        retire_members = tuple(
            member.model_copy(
                update={
                    "mutation": member.mutation.model_copy(
                        update={
                            "idempotency_key": f"supersede:{workload_id}:{generation}:{member.cell_id}",
                            "correlation_id": correlation_id,
                        }
                    )
                }
            )
            for member in prior_members
        )
        executor = self._gang_executor(retire_members)
        try:
            executor.retire(remote_id)
            status = _retire_observation(executor.status(remote_id)).status
            if status not in _QUIESCENT_RETIRE_STATUSES:
                raise RuntimeError(f"predecessor gang {remote_id!r} still reports {status} after retirement")
        finally:
            self._close_gang_executor(executor)

    def submit_gang(self, coordinator_cell_id: str, request: GangWorkloadRequest) -> WorkloadObservation:
        """Submit a fenced workload across every selected native execution cell."""

        members = self._ordered_gang_members(request.members)
        if coordinator_cell_id != members[0].cell_id:
            raise InvalidWorkloadRequestError(
                "native workload gang coordinator must be the lexicographically first member cell"
            )
        workload_id = self._gang_workload_id(request.spec)
        submission_id = request.spec.get("submission_id")
        if not isinstance(submission_id, str) or not submission_id.strip():
            raise InvalidWorkloadRequestError("native workload gang requires a non-empty submission_id")
        expected_remote_id = submission_id.strip()
        member_payload = self._gang_member_payload(members)
        submitted_executor = None
        launched_remote_id: str | None = None
        try:
            with self.state.gang_workload_submission(
                workload_id,
                remote_id=expected_remote_id,
                coordinator_cell_id=coordinator_cell_id,
                members=member_payload,
            ) as predecessors:
                generation = members[0].mutation.generation
                for predecessor in predecessors:
                    self._retire_gang_predecessor(
                        workload_id,
                        coordinator_cell_id,
                        generation,
                        predecessor,
                        correlation_id=members[0].mutation.correlation_id,
                    )
                    self.state.require_lease()
                self.state.clear_gang_workload_predecessors(workload_id, members=member_payload)
                self.state.require_lease()
                launch_cells = self._gang_cells(members)
                submitted_executor = tp.cast(GangExecutionResolver, self.executors).resolve_gang(launch_cells)
                remote_id = submitted_executor.submit(request.spec)
                launched_remote_id = remote_id
                if remote_id != expected_remote_id:
                    # The gang driver contract is to return the fenced remote ID
                    # or raise after retiring its partial launch, so this branch
                    # is defensive. Retire the fully launched gang before
                    # surfacing the mismatch so no rank is left running.
                    # This branch already owns cleanup; do not retire twice in
                    # the general post-launch exception handler below.
                    launched_remote_id = None
                    try:
                        submitted_executor.retire(remote_id)
                    except BaseException as cleanup_exc:
                        raise RuntimeError(
                            f"native workload gang returned unexpected remote ID {remote_id!r}; "
                            f"expected {expected_remote_id!r}; the launched gang could not be retired"
                        ) from cleanup_exc
                    raise RuntimeError(
                        f"native workload gang returned unexpected remote ID {remote_id!r}; "
                        f"expected {expected_remote_id!r}; the launched gang was retired"
                    )
                self.state.require_lease()
        except StaleFenceError as exc:
            if submitted_executor is not None and launched_remote_id is not None:
                try:
                    submitted_executor.retire(launched_remote_id)
                except BaseException as cleanup_exc:
                    raise StaleMutationError(
                        f"{exc}; launched gang execution {launched_remote_id!r} could not be retired after lease loss"
                    ) from cleanup_exc
            raise StaleMutationError(str(exc)) from exc
        except BaseException as exc:
            if submitted_executor is not None and launched_remote_id is not None:
                try:
                    self._rollback_gang(launch_cells, launched_remote_id)
                except BaseException as cleanup_exc:
                    raise RuntimeError(
                        f"{exc}; launched gang execution {launched_remote_id!r} could not be retired "
                        f"after {type(exc).__name__}"
                    ) from cleanup_exc
            raise
        finally:
            if submitted_executor is not None:
                self._close_gang_executor(submitted_executor)
        return WorkloadObservation(remote_id=expected_remote_id, status=ExecutionWorkloadStatus.SUBMITTED)

    def gang_workload(self, coordinator_cell_id: str, workload_id: str, remote_id: str) -> WorkloadObservation:
        """Return aggregate status for every rank in a native workload gang."""

        members = self._stored_gang_members(coordinator_cell_id, workload_id, remote_id)
        executor = self._gang_executor(members)
        try:
            return _workload_observation(executor.status(remote_id))
        finally:
            self._close_gang_executor(executor)

    def stop_gang(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
        request: GangWorkloadMutationRequest,
    ) -> WorkloadObservation:
        """Stop every native rank using the member fences supplied by the controller."""

        stored = self._stored_gang_members(coordinator_cell_id, workload_id, remote_id)
        members = self._validate_gang_mutation_members(stored, request.members)
        executor = self._gang_executor(members)
        try:
            executor.stop(remote_id)
        finally:
            self._close_gang_executor(executor)
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    def retire_gang(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
        request: GangWorkloadMutationRequest,
    ) -> WorkloadObservation:
        """Retire every native rank using the member fences supplied by the controller."""

        stored = self._stored_gang_members(coordinator_cell_id, workload_id, remote_id)
        members = self._validate_gang_mutation_members(stored, request.members)
        executor = self._gang_executor(members)
        try:
            executor.retire(remote_id)
            report = executor.status(remote_id)
        finally:
            self._close_gang_executor(executor)
        return _retire_observation(report)

    def checkpoint_gang(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
        members: tp.Sequence[GangWorkloadMember],
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> WorkloadObservation:
        """Launch one checkpoint hook on every rank of a native workload gang.

        The member fences are validated exactly as ``stop_gang`` and
        ``retire_gang`` validate theirs: a gang holds one lease per cell, so a
        request that does not own every member fence checkpoints none of them.
        The hook itself fans out through the gang driver, which addresses each
        cell with its own fence under the one shared checkpoint ID carried by
        ``context``, so ``gang_workload`` can poll one logical checkpoint.
        """

        stored = self._stored_gang_members(coordinator_cell_id, workload_id, remote_id)
        supplied = self._validate_gang_mutation_members(stored, members)
        executor = self._gang_executor(supplied)
        try:
            checkpoint_id = executor.checkpoint(
                remote_id,
                hook=hook,
                output_uri=output_uri,
                context=context,
            )
        finally:
            self._close_gang_executor(executor)
        return WorkloadObservation(
            remote_id=checkpoint_id,
            status=ExecutionWorkloadStatus.CHECKPOINTING,
            checkpoint_uri=output_uri,
        )

    def gang_logs(self, coordinator_cell_id: str, workload_id: str, remote_id: str) -> str:
        """Return rank-prefixed logs from every member of a native workload gang."""

        members = self._stored_gang_members(coordinator_cell_id, workload_id, remote_id)
        executor = self._gang_executor(members)
        try:
            return executor.logs(remote_id)
        finally:
            self._close_gang_executor(executor)

    def stage_asset(self, cell_id: str, request: CellStageActionRequest) -> CellStageActionResult:
        """Measure, materialize or remove one named asset on this cell's disk.

        The generation is the fence, and it is read from this agent's own
        durable record rather than taken from the request. Equality, not
        "at least": a request planned against an older generation names a
        machine that has been replaced, and one planned against a newer
        generation names a machine this agent has not built yet. Staging on
        either writes bytes nobody asked for and confirms a stage on a host that
        does not hold it -- so a stale generation can neither stage nor evict.

        Nothing here advances the fence. Staging is a cache operation, not a
        capacity mutation, and stamping the cell's generation from it would let
        a warm-up round trip silently take ownership away from the lifecycle
        command that actually owns the host.

        Fenced before the hop and before confirming a successful result. For
        ``eyvan-native`` the executor is on another machine, so the first check
        must happen before sending the request; the second cannot undo a remote
        write, but prevents claiming a confirmed placement on a replaced host.
        A multi-host view also checks between hosts so it can stop fan-out
        after a generation change.

        Raises:
            UnsupportedCapabilityError: There is no executor for this cell's
                disk -- this agent binds none, the worker holding it predates
                the route, or that worker declares no source prefixes. The
                control plane reads all three as "no executor" rather than as a
                failed action.
            StaleMutationError: The request's generation does not own the cell.
        """

        if self.stager is None and self.execution_backend not in _REMOTE_STAGING_EXECUTION_BACKENDS:
            raise UnsupportedCapabilityError(
                f"execution backend {self.execution_backend!r} does not stage assets on this agent's host"
            )
        observation = self.capacity.observe(cell_id)
        if observation.state is CapacityStatus.ABSENT:
            raise StaleMutationError(f"cell {cell_id!r} does not exist on this agent")
        if observation.generation != request.mutation.generation:
            raise StaleMutationError(
                f"staging generation {request.mutation.generation} does not own cell {cell_id!r} "
                f"generation {observation.generation}"
            )
        stager = self.stager if self.stager is not None else self._remote_stager(cell_id, observation)
        if request.action is CellStageAction.MEASURE:
            result = stager.measure(cell_id, request.asset, max_bytes=request.max_bytes)
        elif request.action is CellStageAction.STAGE:
            result = stager.stage(cell_id, request.asset, max_bytes=request.max_bytes)
        else:
            result = stager.evict(cell_id, request.asset)
        # A measurement is advisory; another stage request fences again before
        # writing. Successful writes/evictions must not claim an old host.
        if request.action is not CellStageAction.MEASURE and result.verdict is AssetStageVerdict.OK:
            current_generation = self._staging_fence(cell_id)
            if current_generation != request.mutation.generation:
                raise StaleMutationError(
                    f"staging generation {request.mutation.generation} no longer owns cell {cell_id!r} "
                    f"generation {current_generation}"
                )
        return CellStageActionResult(
            asset=request.asset,
            action=request.action,
            verdict=_STAGE_VERDICTS[result.verdict],
            # An eviction frees bytes rather than placing them, and reporting
            # what it removed as bytes now present would charge the budget for a
            # disk that was just emptied.
            size_bytes=0 if request.action is CellStageAction.EVICT else result.size_bytes,
            digest=None if request.action is CellStageAction.EVICT else result.digest,
            detail=result.detail[:2048],
            path=result.path,
        )

    def _remote_stager(self, cell_id: str, observation: CellObservation) -> CellStageDriver:
        """Return the route to the disk (or disks) the workload on this cell will read.

        The already-fenced observation is passed in rather than re-read. It is
        the same object the generation check just ran against, so the driver
        this resolves is the one that observation names -- and a second
        ``observe`` would be a second provider round trip per staging action,
        against a machine that had no reason to have moved in between.

        A cell with several hosts is resolved first and separately. It has no
        single disk to forward to, so it does not satisfy ``AssetStageDriver``
        directly; it hands back a view bound to the generation checked above,
        which fans each action to every one of its machines and reduces the
        answers to one. Binding it here rather than caching it is the point:
        the slice executor itself is cached across generations, and a staging
        surface hanging off it would carry no fence.

        Raises:
            UnsupportedCapabilityError: The resolved execution driver does not
                stage at all -- a backend with no staging surface of either
                shape. Read as absence of a capability, never as a failure.
        """

        driver = self.executors.resolve(cell_id, observation)
        if isinstance(driver, MultiHostStageCapableDriver):
            return driver.asset_stage_driver(
                generation=observation.generation,
                fence=lambda: self._staging_fence(cell_id),
            )
        if not isinstance(driver, AssetStageDriver):
            raise UnsupportedCapabilityError(
                f"the {self.execution_backend!r} execution driver for cell {cell_id!r} does not stage assets"
            )
        return driver

    def _staging_fence(self, cell_id: str) -> int:
        """Return the generation this agent currently believes owns one cell.

        The re-check a multi-host fan-out makes between hosts, read from the
        same place the first check was: an agent that fenced against its
        capacity record and then re-checked against something else could
        contradict itself halfway through a fetch.

        Raises:
            StaleMutationError: The cell is gone from this agent, which is not a
                generation and must not be reported as one.
        """

        observation = self.capacity.observe(cell_id)
        if observation.state is CapacityStatus.ABSENT:
            raise StaleMutationError(f"cell {cell_id!r} no longer exists on this agent")
        return observation.generation

    def _executor(self, cell_id: str) -> ExecutionDriver:
        return self.executors.resolve(cell_id, self.capacity.observe(cell_id))

    def _close_execution_cell(self, cell_id: str) -> None:
        if isinstance(self.executors, CellClosableResolver):
            self.executors.close_cell(cell_id)

    def _retire_predecessor(
        self,
        workload_id: str,
        target_cell_id: str,
        context: MutationContext,
        predecessor: JsonObject,
    ) -> None:
        cell_id = predecessor.get("cell_id")
        remote_id = predecessor.get("remote_id")
        generation = predecessor.get("generation")
        fencing_token = predecessor.get("fencing_token")
        if (
            not isinstance(cell_id, str)
            or not isinstance(remote_id, str)
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or not isinstance(fencing_token, str)
        ):
            raise StaleMutationError(f"workload {workload_id!r} has an invalid predecessor fence")
        predecessor_context = MutationContext(
            idempotency_key=f"supersede:{workload_id}:{context.generation}",
            generation=generation,
            fencing_token=fencing_token,
            correlation_id=context.correlation_id,
        )
        if cell_id != target_cell_id:
            self._remote_retire(
                cell_id,
                remote_id,
                WorkloadMutationRequest(
                    mutation=MutationModel(
                        idempotency_key=predecessor_context.idempotency_key,
                        generation=predecessor_context.generation,
                        fencing_token=predecessor_context.fencing_token,
                        correlation_id=predecessor_context.correlation_id,
                    )
                ),
            )
            return
        executor = self._executor(cell_id)
        executor.retire(remote_id, predecessor_context)
        status = _retire_observation(executor.status(remote_id)).status
        if status not in _QUIESCENT_RETIRE_STATUSES:
            raise RuntimeError(f"predecessor {remote_id!r} on cell {cell_id!r} still reports {status} after retirement")

    def _retire_on_source_agent(
        self,
        cell_id: str,
        remote_id: str,
        request: WorkloadMutationRequest,
    ) -> None:
        observation = self.capacity.observe(cell_id)
        endpoint = observation.metadata.get("agent_endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            # Retirement fences the predecessor so two generations of one
            # workload cannot run at once. A host whose capacity is definitively
            # lost cannot be running anything, so the fence is already satisfied
            # and there is no agent left to ask -- which is the ordinary way a
            # reclaimed Spot cell looks, not a fault.
            #
            # Raising here instead stopped the *successor* from being placed: the
            # error escapes the submit request for the healthy replacement cell,
            # so a preempted workload retried forever against a cell that was
            # ready for it. Only a host that still exists but cannot be reached
            # is a genuine failure, because that one may still be running.
            if observation.state in DEFINITIVE_CAPACITY_LOSS_STATUSES:
                return
            raise RuntimeError(f"cell {cell_id!r} has no agent endpoint for remote retirement")
        from eyvan.worker.agent.client import CellAgentClient

        if self.settings.mutual_tls_client_enabled:
            mtls = self.settings.mutual_tls_files()
            if not mtls.has_trust_anchor or not mtls.has_identity:
                raise MutualTlsConfigurationError(
                    "an enabled mutual TLS client requires a trust bundle, certificate and private key"
                )
            client = CellAgentClient(endpoint, token=self.settings.agent_shared_secret, mtls=mtls)
        else:
            client = CellAgentClient(endpoint, token=self.settings.agent_shared_secret)
        try:
            report = client.retire(cell_id, remote_id, request)
        finally:
            client.close()
        if report.status not in _QUIESCENT_RETIRE_STATUSES:
            raise RuntimeError(
                f"cell {cell_id!r} still reports workload {remote_id!r} as {report.status} after retirement"
            )

    def retire(self, cell_id: str, remote_id: str, request: WorkloadMutationRequest) -> WorkloadObservation:
        executor = self._executor(cell_id)
        executor.retire(remote_id, _context(request))
        return _retire_observation(executor.status(remote_id))

    def workload(self, cell_id: str, remote_id: str) -> WorkloadObservation:
        return _workload_observation(self._executor(cell_id).status(remote_id))

    def stop(self, cell_id: str, remote_id: str, request: WorkloadMutationRequest) -> WorkloadObservation:
        executor = self._executor(cell_id)
        executor.stop(remote_id, _context(request))
        return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.STOPPING)

    def checkpoint(self, cell_id: str, remote_id: str, request: CheckpointRequest) -> WorkloadObservation:
        checkpoint_job_id = self._executor(cell_id).checkpoint(
            remote_id,
            hook=request.hook,
            output_uri=request.output_uri,
            context=_context(request),
        )
        return WorkloadObservation(
            remote_id=checkpoint_job_id,
            status=ExecutionWorkloadStatus.CHECKPOINTING,
            checkpoint_uri=request.output_uri,
        )

    def logs(self, cell_id: str, remote_id: str) -> str:
        return self._executor(cell_id).logs(remote_id)

    def exec_in_workload(
        self,
        cell_id: str,
        remote_id: str,
        command: tp.Sequence[str],
        *,
        timeout_s: float = 30.0,
        runtime_rank: int | None = None,
    ) -> JsonObject:
        """Run one diagnostic command beside a live workload.

        Two gates, answering different questions. ``supports_exec`` on the
        *capacity* driver is the operator's admission decision -- may anyone do
        this on this fleet. ``ExecCapableExecutionDriver`` is whether this
        execution backend can do it at all. A backend that can but is not
        permitted, and one that is permitted but cannot, are different
        misconfigurations and get different messages.
        """

        if not self.capacity.capabilities().supports_exec:
            raise UnsupportedCapabilityError(
                "this fleet does not permit exec; set supports_exec on the capacity backend to allow it"
            )
        executor = self._executor(cell_id)
        if not isinstance(executor, ExecCapableExecutionDriver):
            raise UnsupportedCapabilityError(
                f"execution backend {self.execution_backend!r} cannot exec beside a workload"
            )
        if not command:
            raise InvalidWorkloadRequestError("exec requires a command")
        if runtime_rank is not None:
            if isinstance(executor, RankedExecCapableExecutionDriver):
                return executor.exec_in_workload_rank(
                    remote_id, list(command), timeout_s=timeout_s, runtime_rank=runtime_rank
                )
            # Backends without an exact-rank capability must not turn a selected
            # host into a fan-out, including rank zero on an unknown topology.
            raise UnsupportedCapabilityError("execution backend cannot select an exact runtime rank")
        return executor.exec_in_workload(remote_id, list(command), timeout_s=timeout_s)

    def log_page(self, cell_id: str, remote_id: str, *, after: str = "", limit: int = 200) -> JsonObject:
        """Return one resumable page of structured log records.

        A backend that cannot produce cursors does not implement the protocol,
        and that is reported as 501 rather than as an empty page: "this backend
        has no structured logs" and "this workload produced no output" are
        different answers, and a follower that cannot tell them apart will poll
        forever against a backend that will never answer.
        """

        executor = self._executor(cell_id)
        if not isinstance(executor, StructuredLogExecutionDriver):
            raise UnsupportedCapabilityError(
                f"execution backend {self.execution_backend!r} does not expose structured logs"
            )
        page = executor.log_page(remote_id, after=after, limit=limit)
        document: JsonObject = {
            "records": [dict(record.as_json()) for record in page.records],
            "next_cursor": page.next_cursor,
            "unavailable": page.unavailable,
            "detail": page.detail,
            "truncated_ranks": list(page.truncated_ranks),
        }
        return document

    def process_status(self, cell_id: str, remote_id: str) -> JsonObject:
        """Return live process detail for one workload on one cell.

        Not every execution backend can introspect processes -- ``native-local``
        and the Ray adapter do not -- so an unsupported backend is reported as
        such rather than faked.
        """

        executor = self._executor(cell_id)
        if not isinstance(executor, IntrospectableExecutionDriver):
            raise UnsupportedCapabilityError(
                f"execution backend {self.execution_backend!r} does not expose process introspection"
            )
        return executor.process_status(remote_id)
