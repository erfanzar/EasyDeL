# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared constants, errors, and value types for the controller mixins."""

from __future__ import annotations

import asyncio
import logging
import typing as tp
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from eyvan._internal.compat import StrEnum
from eyvan.capacity import (
    DEFINITIVE_CAPACITY_LOSS_STATUSES,
    PENDING_CAPACITY_STATUSES,
    QUEUED_CAPACITY_STATUSES,
    CapacityStatus,
)
from eyvan.capacity.audit import CapacityReplacementReason
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.capacity.planning import (
    cell_requires_replacement,
)
from eyvan.records import (
    ACQUISITION_EXHAUSTED_SCHEDULING_REASON,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    LeaseState,
    OutboxState,
    Principal,
    ResourceHours,
    Role,
    RoleGrant,
    WorkloadState,
)

logger = logging.getLogger(__name__)

_SHARED_TPU_QUOTA_SCHEDULING_REASON = "shared Spot TPU quota cannot fit this job's requested complete worker capacity"
#: Said when every pool a workload accepts has failed to acquire capacity
#: ``MAX_CONSECUTIVE_ACQUISITION_FAILURES`` times running. Lives in records so
#: the progress route can say it too.
_ACQUISITION_EXHAUSTED_SCHEDULING_REASON = ACQUISITION_EXHAUSTED_SCHEDULING_REASON

_PREEMPTION_TERMINAL_WORKLOAD_STATUSES = frozenset(
    {
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.SUCCEEDED,
        ExecutionWorkloadStatus.LOST,
        ExecutionWorkloadStatus.NOT_FOUND,
    }
)
_RECOVERABLE_WORKLOAD_STATUSES = frozenset({ExecutionWorkloadStatus.PENDING, ExecutionWorkloadStatus.RUNNING})
_PREEMPTION_OWNED_TERMINAL_WORKLOAD_STATUSES = frozenset(
    {
        ExecutionWorkloadStatus.SUCCEEDED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.LOST,
        ExecutionWorkloadStatus.NOT_FOUND,
    }
)
_RETRIABLE_OUTBOX_STATES = frozenset({OutboxState.PENDING, OutboxState.PROCESSING, OutboxState.FAILED})
_ACTIVE_CAPACITY_APPLY_STATES = _RETRIABLE_OUTBOX_STATES
_CAPACITY_APPLY_PROVIDER_RPC_STARTED_KEY = "provider_rpc_started"


class _FleetCapacityGate:
    """Allow concurrent cell work while a fleet mutation briefly closes admission."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._writer_lock = asyncio.Lock()
        self._closing = False
        self._active_mutations = 0

    @asynccontextmanager
    async def operation(self) -> tp.AsyncIterator[None]:
        """Admit one normal cell operation unless a fleet mutation is closing."""

        async with self._condition:
            await self._condition.wait_for(lambda: not self._closing)
        yield

    @asynccontextmanager
    async def mutation(self) -> tp.AsyncIterator[None]:
        """Serialize a short autoscaler state write with fleet configuration changes.

        Provider RPCs use :meth:`operation` so fleet shutdown never waits for
        a long cloud request. This narrower scope covers only transactional
        autoscaler writes that must not race a pool-template replacement.
        """

        async with self._condition:
            await self._condition.wait_for(lambda: not self._closing)
            self._active_mutations += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active_mutations -= 1
                if self._active_mutations == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def closing(self) -> tp.AsyncIterator[None]:
        """Block new cell work while a fleet mutation is persisted."""

        async with self._writer_lock:
            is_closing = False
            try:
                async with self._condition:
                    self._closing = True
                    is_closing = True
                    await self._condition.wait_for(lambda: self._active_mutations == 0)
                yield
            finally:
                if is_closing:
                    async with self._condition:
                        self._closing = False
                        self._condition.notify_all()


class StaleOutboxCommand(RuntimeError):
    """An at-least-once command no longer owns the current aggregate generation."""


class CapacityApplySuperseded(StaleOutboxCommand):
    """A capacity apply no longer has a live desired cell to provision."""

    def __init__(self, message: str, *, provider_diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.provider_diagnostic = provider_diagnostic


class PreemptionCheckpointError(RuntimeError):
    """A preemption checkpoint phase failed before any victim was stopped."""


class _PreemptionVictimOutcome(StrEnum):
    """Durable settlement result for one preemption victim."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PREEMPTED = "preempted"


class _ProviderPhaseGroup(StrEnum):
    """Groups provider phases whose timeout budget must be continuous."""

    REQUESTING = "REQUESTING"
    QUEUED = "QUEUED"
    PROVISIONING = "PROVISIONING"


class _PreemptionVictim(BaseModel):
    """Strict durable identity and rollback state for one preemption victim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workload_id: str = Field(min_length=1, max_length=96)
    lease_id: str = Field(min_length=1, max_length=96)
    cell_id: str = Field(min_length=1, max_length=96)
    generation: int = Field(ge=0, strict=True)
    checkpoint: bool = Field(strict=True)
    grace_period_seconds: int = Field(ge=0, strict=True)
    hook: str | None
    output_uri: str | None
    workload_state: WorkloadState
    lease_state: LeaseState

    @model_validator(mode="after")
    def validate_restore_states(self) -> _PreemptionVictim:
        if self.workload_state not in {
            WorkloadState.LEASED,
            WorkloadState.DISPATCHING,
            WorkloadState.RUNNING,
        }:
            raise ValueError("preemption victim has an invalid workload restore state")
        if self.lease_state not in {LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.QUARANTINED}:
            raise ValueError("preemption victim has an invalid lease restore state")
        return self


class _DispatchLeaseMembership(BaseModel):
    """Validated lease membership that a workload-dispatch command may submit."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    lease_id: str = Field(min_length=1, max_length=96)
    lease_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_membership(self) -> _DispatchLeaseMembership:
        if not self.lease_ids:
            return self
        if len(self.lease_ids) < 2:
            raise ValueError("dispatch lease_ids must contain a complete group")
        if len(self.lease_ids) != len(set(self.lease_ids)):
            raise ValueError("dispatch lease_ids must not contain duplicates")
        if self.lease_ids[0] != self.lease_id:
            raise ValueError("dispatch leader lease must be the first group member")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        """Return the actual single-cell or group dispatch membership."""

        return self.lease_ids or (self.lease_id,)


class _PreemptionCheckpointUris(RootModel[dict[str, str | None]]):
    """Validated durable checkpoint outputs keyed by victim workload ID."""


class _PreemptionVictimOutcomes(RootModel[dict[str, _PreemptionVictimOutcome]]):
    """Validated durable terminal outcomes keyed by victim lease ID."""


class _PreemptionStoppedVictims(RootModel[list[str]]):
    """Validated durable ordered set of stopped victim lease IDs."""


#: Workload states that hold, or are about to hold, real capacity. Shared so the
#: observation loop and the orphan sweep cannot drift apart -- a state one considers
#: live and the other does not is a workload nothing ever settles.
_LIVE_WORKLOAD_STATES: tp.Final[frozenset[WorkloadState]] = frozenset(
    {
        WorkloadState.LEASED,
        WorkloadState.DISPATCHING,
        WorkloadState.RUNNING,
        WorkloadState.CHECKPOINTING,
        WorkloadState.CANCELLING,
    }
)


@dataclass(frozen=True, slots=True)
class LeaderPhase:
    """One step of the leader cycle, with the policy it needs to run.

    Interval semantics differ per phase and are therefore declared per phase rather
    than inferred: a zero ``cell_reconcile_interval_s`` has always meant "every
    cycle", while a zero ``retention_interval_s`` is documented as "disabled".
    """

    name: str
    run: tp.Callable[[], tp.Awaitable[object]]
    #: None runs the phase every cycle; a number gives it its own slower cadence.
    interval_s: float | None = None
    zero_interval_disables: bool = False
    #: Phases this one reads the results of. If any of them failed in this cycle,
    #: this phase is skipped with that reason rather than run against a view it
    #: was told not to trust.
    #:
    #: Every phase's failure is contained either way. The previous design let a
    #: phase propagate instead, which stopped the whole rest of the cycle -- and
    #: since the phases that release capacity and settle billing run in the
    #: middle of that list, one failure upstream of them meant an accelerator
    #: nobody stopped paying for. That happened to a live v5p-16 multislice.
    #:
    #: The safety the old behaviour bought is kept and made explicit: a phase
    #: that *commits* work against the plane's view names the observations it
    #: depends on, and is skipped when they are stale. A phase that gives
    #: capacity back names nothing, because reclaiming is safe on any view and
    #: not reclaiming is the expensive mistake.
    requires: tuple[str, ...] = ()
    #: Run outside the cycle instead of inside it. The cycle starts the phase and
    #: moves on; a pass still running when the phase is next due is not started
    #: again. Only for phases no other phase ``requires`` and whose writes are
    #: safe against the phases that run meanwhile -- the cycle cannot see their
    #: failures in the pass that caused them.
    background: bool = False


@dataclass(frozen=True, slots=True)
class _LaunchedCheckpointHook:
    """A checkpoint hook unit this attempt started and therefore owns.

    In-memory only, and deliberately so: it is scoped to a single execution of
    the checkpointing phase, which is the only window in which an orphaned hook
    can exist. A crash mid-phase leaves the plan in ``checkpointing`` for the
    next leader to re-drive, and the hook's own fence rejects a stale writer.
    """

    cell: CellRecord
    remote_id: str
    lease_id: str
    generation: int
    fencing_token: str


class LeaderElector(tp.Protocol):
    """Exclusive ownership for global scheduling decisions."""

    async def acquire(self) -> bool: ...

    async def release(self) -> None: ...


class _UnreachableAgentBackoff:
    """Space out retries to cell agents that cannot be reached at all.

    An agent that refuses connections answered nothing, so the observation path
    already writes nothing for it -- but it still asked every
    ``cell_reconcile_interval_s`` forever and logged twice per ask. Each
    consecutive failure doubles the wait, up to ``max_s``; one answer clears it.

    Held in memory and keyed by cell *and* generation: a restart or a recreated
    cell starts from a prompt retry, which can only make the plane notice a
    recovery sooner. Skipping an observation changes no record, so a dead
    agent's cell still retires on its provider timeouts exactly as before.
    """

    def __init__(self, *, base_s: float, max_s: float) -> None:
        self._base_s = base_s
        self._max_s = max_s
        self._failures: dict[tuple[str, int], tuple[int, datetime]] = {}

    @property
    def enabled(self) -> bool:
        return self._base_s > 0 and self._max_s > 0

    def ready(self, cell: CellRecord, now: datetime) -> bool:
        """Return whether this cell's agent may be asked again at ``now``."""

        entry = self._failures.get((cell.id, cell.generation))
        return entry is None or not self.enabled or now >= entry[1]

    def backing_off(self, cell: CellRecord) -> bool:
        """Return whether the most recent ask of this cell's agent failed to connect."""

        return (cell.id, cell.generation) in self._failures

    def failed(self, cell: CellRecord, now: datetime) -> bool:
        """Record one unreachable ask; return True only for the first in a streak."""

        key = (cell.id, cell.generation)
        count = self._failures[key][0] + 1 if key in self._failures else 1
        # One refused connection is a blip -- an agent restarting -- so the next
        # ordinary pass still asks. Waiting starts with the second in a row.
        delay_s = 0.0
        if self.enabled and count > 1:
            delay_s = min(self._max_s, self._base_s * 2 ** min(count - 2, 30))
        self._failures[key] = (count, now + timedelta(seconds=delay_s))
        return count == 1

    def succeeded(self, cell: CellRecord) -> bool:
        """Clear a streak; return True when the agent had been unreachable."""

        return self._failures.pop((cell.id, cell.generation), None) is not None

    def retain(self, keys: tp.Collection[tuple[str, int]]) -> None:
        """Forget cells that are no longer observed, so the map cannot grow with churn."""

        for key in [key for key in self._failures if key not in keys]:
            del self._failures[key]


class SingleProcessLeader:
    """Leader implementation for the in-memory development repository."""

    async def acquire(self) -> bool:
        return True

    async def release(self) -> None:
        return None


def _system_principal() -> Principal:
    return Principal(subject="eyvan-controller", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _api_resources(resources) -> ResourceHours:
    values = {name: float(value) for name, value in resources.items()}
    known = {name: values.pop(name, 0) for name in ("cpu", "memory_gib", "gpu", "tpu")}
    return ResourceHours(**known, custom=values)


def _provider_phase_group(
    phase: CapacityStatus | None,
) -> CapacityStatus | _ProviderPhaseGroup | None:
    if phase is CapacityStatus.REQUESTING:
        return _ProviderPhaseGroup.REQUESTING
    if phase in QUEUED_CAPACITY_STATUSES:
        return _ProviderPhaseGroup.QUEUED
    if phase in PENDING_CAPACITY_STATUSES:
        return _ProviderPhaseGroup.PROVISIONING
    return phase


def _provider_phase_start(cell: CellRecord, remote_state: CapacityStatus, observed_at: datetime) -> datetime:
    if cell.provider_phase_started_at is not None and _provider_phase_group(
        cell.provider_phase
    ) == _provider_phase_group(remote_state):
        return cell.provider_phase_started_at
    return observed_at


def _replacement_reason(cell: CellRecord, pool: CapacityPoolRecord) -> CapacityReplacementReason:
    """Return why a non-supplying cell must be retired before replacement."""

    if cell.provider_phase in DEFINITIVE_CAPACITY_LOSS_STATUSES:
        return CapacityReplacementReason.PROVIDER_LOST
    if cell_requires_replacement(cell, pool):
        return CapacityReplacementReason.TEMPLATE_CHANGED
    if cell.state is CellState.REGISTERING:
        return CapacityReplacementReason.PROVISIONING_TIMEOUT
    return CapacityReplacementReason.UNHEALTHY
