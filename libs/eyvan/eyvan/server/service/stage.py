# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed staging: put the asset on the host, account for it, reclaim it.

Locality started as a hint. ``WorkloadSpec.named_assets`` steered placement
toward cells that had *probably* seen an asset, inferred from lease history and,
where an agent volunteered one, from a free-form report. Nothing put the asset
on a host, nothing knew how many bytes it took, and nothing ever removed it. A
preference pointing at data that may not be there is a preference that is
sometimes exactly wrong, and a cache with no eviction is a disk that fills.

This module owns the decisions and none of the I/O. Everything here is a pure
function over durable records, so the parts that are easy to get wrong -- what
is provably present, what the budget allows, what may be deleted -- are
decidable without a fleet. The controller phase does the round trips.

**Unproven is never a hit, and always a charge.** The two asymmetries are the
whole design. A stage this control plane cannot confirm must not steer placement
-- that would be inventing evidence -- but it still occupies the byte budget,
because a disk does not care what we can prove. Erring the other way in either
direction is what makes a cache lie: score an unknown as a hit and jobs land on
cold cells; forget to charge it and the budget is a number that bounds nothing.

**Eviction refuses before it guesses.** An asset a running workload declared is
never a victim, even when that leaves the cell over budget. The plan then
reports a shortfall rather than freeing the bytes it was asked for, because
deleting a dataset out from under a job that is reading it converts a disk-space
problem into a failed multi-hour run.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from datetime import datetime

from eyvan.capacity.stage_policy import CellStagePolicy, StageEvictionPolicy
from eyvan.records import (
    CELL_STAGE_UNPROVEN_STATES,
    CellRecord,
    CellStageRecord,
    CellStageState,
    LeaseRecord,
    LeaseState,
    WorkloadRecord,
    WorkloadState,
)

#: Lease states in which a workload is holding the cell right now. Both count:
#: a lease that has been granted but whose payload has not started still names a
#: workload that is about to read the asset, and evicting it during dispatch is
#: the same failed run as evicting it mid-epoch. A quarantined lease is still
#: holding too: its TTL lapsed, but the work was not killed and observation can
#: recover the lease to active, so its assets stay protected in the window.
_HOLDING_LEASE_STATES = frozenset({LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.REVOKING, LeaseState.QUARANTINED})

#: Workload states in which a declared asset is protected from eviction. A
#: terminal workload's lease may not have been settled yet, and its assets are
#: ordinary cache contents the moment it stops reading them.
_RUNNING_WORKLOAD_STATES = frozenset(
    {
        WorkloadState.LEASED,
        WorkloadState.DISPATCHING,
        WorkloadState.RUNNING,
        WorkloadState.CHECKPOINTING,
        WorkloadState.CANCELLING,
    }
)


class CellStageError(RuntimeError):
    """Base error raised by a cell stage executor."""


class CellStageInconclusiveError(CellStageError):
    """The executor could not determine what happened.

    A timeout, an unreachable host, a truncated transfer. Nothing was learned,
    so nothing may be recorded as fact: the stage becomes ``UNKNOWN``, which is
    never scored as a hit and never uncharged. Transient by construction and
    retried while the attempt budget lasts.
    """


class CellStageRefusedError(CellStageError):
    """The executor will never satisfy this request as written.

    An asset larger than the bound it was admitted under, an address scheme the
    host cannot fetch, a path that does not exist. Retrying an unchanged request
    cannot start working, so the stage becomes ``BLOCKED`` and no further
    automatic attempt is made -- the failure mode this exists to prevent is a
    permanently impossible fetch consuming a leader phase forever.
    """


class CellStageStaleError(CellStageError):
    """The executor rejected this request's fence.

    The cell moved on beneath the action. Not a failure of the asset and not a
    statement about it: the plan is abandoned and the row is left exactly as it
    was for whichever generation now owns the host.
    """


@dataclass(frozen=True, slots=True)
class CellStageRequest:
    """One fenced, byte-bounded instruction to an executor.

    ``max_bytes`` is a term of the request, not a hint. It is the smaller of the
    per-action ceiling and the headroom actually left in this cell's budget, so
    an executor that honours it cannot overrun the budget and one that does not
    is caught by comparing what it reports against what it was allowed.
    """

    cell_id: str
    asset: str
    cell_generation: int
    #: ``cell:<id>:<generation>``, the same shape every other cell action fences
    #: with, so an executor that already understands one understands this.
    fencing_token: str
    max_bytes: int


@dataclass(frozen=True, slots=True)
class CellStageMeasurement:
    """What an executor observed about one asset on one host."""

    size_bytes: int
    #: ``sha256:<hex>``, or ``None`` when the executor does not compute one.
    #: ``None`` is unknown content, never "no content": the bytes are still
    #: proven present, they are simply not verified.
    digest: str | None = None
    detail: str = ""


class CellStageExecutor(tp.Protocol):
    """Materialize, measure and remove named assets on one execution host.

    Deliberately not the cell-agent client. Staging is the one cell operation
    whose implementation is a property of the *asset store* rather than of the
    capacity provider, and binding it here would have made the contract
    unimplementable for anything but today's agent. A deployment that configures
    no executor stages nothing and reports every stage as unknown, which is the
    honest description of a control plane that cannot reach a disk.

    Every method is synchronous and blocking; the controller runs it on a worker
    thread, outside every transaction.
    """

    def measure(self, request: CellStageRequest) -> CellStageMeasurement:
        """Return what this asset would occupy on this host, without fetching it.

        Raises:
            CellStageRefusedError: The asset can never be staged as requested.
            CellStageInconclusiveError: The size could not be determined.
            CellStageStaleError: The request's fence is no longer current.
        """
        ...

    def stage(self, request: CellStageRequest) -> CellStageMeasurement:
        """Materialize the asset and return what was actually written.

        Raises:
            CellStageRefusedError: The asset can never be staged as requested.
            CellStageInconclusiveError: The outcome could not be determined.
            CellStageStaleError: The request's fence is no longer current.
        """
        ...

    def evict(self, request: CellStageRequest) -> None:
        """Remove the asset, returning only once the bytes are gone.

        Raises:
            CellStageInconclusiveError: Removal could not be confirmed.
            CellStageStaleError: The request's fence is no longer current.
        """
        ...


@dataclass(frozen=True, slots=True)
class StageEviction:
    """One asset an eviction plan proposes to delete, and what it frees."""

    asset: str
    reserved_bytes: int
    state: CellStageState
    last_used_at: datetime | None


@dataclass(frozen=True, slots=True)
class CellStageEvictionPlan:
    """What eviction would do to one cell, before anything is deleted.

    Returned identically by a dry run and by a live one -- a dry run is the same
    computation with the deletions withheld, not a different code path that could
    disagree with the one that deletes.
    """

    cell_id: str
    cell_generation: int
    policy: StageEvictionPolicy
    charged_bytes: int
    budget_bytes: int
    #: Bytes the caller asked to make room for, on top of staying under budget.
    wanted_bytes: int
    victims: tuple[StageEviction, ...] = ()
    protected_assets: tuple[str, ...] = ()
    #: Bytes the plan could not free because every remaining candidate is
    #: protected. Non-zero is a real answer, not a failure: it says the cell is
    #: over budget *and* every byte over it belongs to something running.
    shortfall_bytes: int = 0
    #: Rows describing a generation this cell no longer runs. Deleted rather than
    #: evicted: the host they measured has been replaced, so there is nothing on
    #: any disk to remove and an executor call would be addressed at a machine
    #: that does not exist.
    superseded_assets: tuple[str, ...] = ()

    @property
    def freed_bytes(self) -> int:
        """Return the bytes this plan's victims would release."""

        return sum(victim.reserved_bytes for victim in self.victims)

    @property
    def deletes_anything(self) -> bool:
        """Return whether executing this plan would remove bytes from a host."""

        return bool(self.victims)


def charged_bytes(
    stages: tp.Iterable[CellStageRecord],
    *,
    cell: CellRecord,
    excluding: str | None = None,
) -> int:
    """Return the bytes one cell's stage rows charge against its budget.

    Every state charges, proven or not -- see :data:`CELL_STAGE_UNPROVEN_STATES`
    for why. Only rows describing an older generation do not, because they
    describe a machine that has been replaced. This is the number the budget is
    compared against and the number eviction reduces.

    ``excluding`` drops one asset's own charge, which is what re-staging means:
    the row already holds bytes for that asset, and counting them against its own
    replacement would make every retry look larger than the first try.
    """

    return sum(stage.reserved_bytes for stage in stages if _describes(stage, cell) and stage.asset != excluding)


def confirmed_stage_index(
    stages: tp.Iterable[CellStageRecord],
    cells: tp.Sequence[CellRecord],
) -> dict[str, tuple[str, ...]]:
    """Map each asset to the cells that *provably* hold it, in cell-id order.

    The load-bearing word is provably. A row is included only when it is
    ``STAGED`` and names the generation its cell is running now; every other row
    -- staging, unknown, evicting, blocked, or confirmed against a host that has
    since been replaced -- is absent, so it can never be scored as a cache hit.
    """

    by_id = {cell.id: cell for cell in cells}
    confirmed: dict[str, list[str]] = {}
    for stage in stages:
        cell = by_id.get(stage.cell_id)
        if cell is None or not stage.is_confirmed_on(cell):
            continue
        confirmed.setdefault(stage.asset, []).append(cell.id)
    return {asset: tuple(sorted(cell_ids)) for asset, cell_ids in confirmed.items()}


def declared_assets_on_cell(
    cell_id: str,
    workloads: tp.Sequence[WorkloadRecord],
    leases: tp.Sequence[LeaseRecord],
) -> frozenset[str]:
    """Return the assets a workload currently holding this cell declared.

    These are the assets eviction may never take. Derived from live leases rather
    than from a flag on the stage row, so protection cannot be left switched on
    by a workload that has since finished, and cannot be missing for one that
    started after the row was written.
    """

    holders = {
        lease.workload_id for lease in leases if lease.cell_id == cell_id and lease.state in _HOLDING_LEASE_STATES
    }
    return frozenset(
        asset
        for workload in workloads
        if workload.id in holders and workload.state in _RUNNING_WORKLOAD_STATES
        for asset in workload.spec.named_assets
    )


def plan_eviction(
    stages: tp.Sequence[CellStageRecord],
    *,
    cell: CellRecord,
    policy: CellStagePolicy,
    protected: tp.AbstractSet[str],
    wanted_bytes: int = 0,
) -> CellStageEvictionPlan:
    """Decide what one cell would give up to fit its budget, deleting nothing.

    ``wanted_bytes`` is headroom the caller needs on top of staying under budget
    -- the size of an asset it is about to stage. Zero plans only the reduction
    the budget itself demands.

    Protection wins over the budget every time. When the only remaining
    candidates are assets a running workload declared, the plan stops and reports
    the bytes it could not free, because a cache that deletes a dataset out from
    under a reader has converted a disk-space problem into a failed run.
    """

    if wanted_bytes < 0:
        raise ValueError("cell stage eviction wanted_bytes must not be negative")
    live = [stage for stage in stages if _describes(stage, cell)]
    superseded = tuple(
        sorted(stage.asset for stage in stages if stage.cell_id == cell.id and not _describes(stage, cell))
    )
    charged = charged_bytes(stages, cell=cell)
    must_free = charged + wanted_bytes - policy.budget_bytes
    plan_protected = tuple(sorted(protected & {stage.asset for stage in live}))
    if must_free <= 0:
        return CellStageEvictionPlan(
            cell_id=cell.id,
            cell_generation=cell.generation,
            policy=policy.eviction,
            charged_bytes=charged,
            budget_bytes=policy.budget_bytes,
            wanted_bytes=wanted_bytes,
            protected_assets=plan_protected,
            superseded_assets=superseded,
        )
    # Blocked rows are not candidates. Blocked means this control plane has
    # stopped taking automatic action on the asset in either direction -- it is
    # already the terminal state of an executor that would not answer about it,
    # and handing the same asset back to the same executor on every cycle is the
    # unbounded retry loop this state exists to end. The bytes stay charged and
    # the plan reports the shortfall, which is the honest description of a cell
    # that is over budget and cannot reclaim itself.
    candidates = [stage for stage in live if stage.asset not in protected and stage.state is not CellStageState.BLOCKED]
    victims: list[StageEviction] = []
    freed = 0
    for stage in sorted(candidates, key=_ordering(policy.eviction)):
        if freed >= must_free:
            break
        victims.append(
            StageEviction(
                asset=stage.asset,
                reserved_bytes=stage.reserved_bytes,
                state=stage.state,
                last_used_at=stage.last_used_at,
            )
        )
        freed += stage.reserved_bytes
    return CellStageEvictionPlan(
        cell_id=cell.id,
        cell_generation=cell.generation,
        policy=policy.eviction,
        charged_bytes=charged,
        budget_bytes=policy.budget_bytes,
        wanted_bytes=wanted_bytes,
        victims=tuple(victims),
        protected_assets=plan_protected,
        shortfall_bytes=max(0, must_free - freed),
        superseded_assets=superseded,
    )


def stage_headroom_bytes(
    stages: tp.Iterable[CellStageRecord],
    *,
    cell: CellRecord,
    policy: CellStagePolicy,
    excluding: str | None = None,
) -> int:
    """Return the largest asset this cell may admit right now.

    The exact number a request is bounded by: whichever is smaller of the
    per-action ceiling and what is left of the budget. Never negative -- a cell
    already over budget admits nothing rather than admitting a negative amount.

    ``excluding`` has the same meaning it has in :func:`charged_bytes`, and this
    calls that rather than repeating the sum: two copies of the budget arithmetic
    is exactly how a headroom check and a budget check come to disagree.
    """

    charged = charged_bytes(stages, cell=cell, excluding=excluding)
    return max(0, min(policy.max_stage_bytes, policy.budget_bytes - charged))


def _describes(stage: CellStageRecord, cell: CellRecord) -> bool:
    """Return whether one row describes the machine this cell is running now."""

    return stage.cell_id == cell.id and stage.cell_generation == cell.generation


def _ordering(policy: StageEvictionPolicy) -> tp.Callable[[CellStageRecord], tuple[object, ...]]:
    """Return the victim order one policy implies.

    Every ordering ends in the asset name so two equally good candidates are
    resolved by identity rather than by row order -- otherwise a dry run and the
    live pass that follows it could name different victims from the same state.
    """

    if policy is StageEvictionPolicy.LARGEST_FIRST:
        return lambda stage: (-stage.reserved_bytes, stage.asset)
    return lambda stage: (_lru_key(stage), stage.asset)


def _lru_key(stage: CellStageRecord) -> tuple[int, float]:
    """Return an LRU sort key that puts never-used stages first.

    A stage nothing has ever used is the cheapest thing to give up, and it has no
    timestamp to compare -- so it sorts ahead of every dated one rather than
    being handed an invented date.

    Compared as an instant rather than as text: two rows written in different
    UTC offsets name the same moment and must order by it, which string
    comparison of their timestamps would get wrong.
    """

    if stage.last_used_at is None:
        return (0, 0.0)
    return (1, stage.last_used_at.timestamp())


#: Executor round trips one pass may make, across the whole fleet.
#:
#: A staging fetch is minutes of network, and this phase shares a leader cycle
#: with placement. Without a ceiling, the first cycle after a large fleet starts
#: declaring assets would run for hours and every phase behind it would wait --
#: so a pass takes a bounded bite and the next cycle takes the next one. Deferred
#: work is reported as ``capped`` rather than dropped.
MAX_STAGE_ACTIONS_PER_PASS = 32


@dataclass(slots=True)
class StageOutcome:
    """What one staging or eviction pass did, counted by outcome.

    Counters rather than a boolean because every one of these is a different
    operational story: ``inconclusive`` is a fleet that is not answering,
    ``blocked`` is an asset that will never stage, ``superseded`` is capacity
    churning underneath the cache, and ``capped`` is a pass that ran out of its
    round-trip budget with work still to do.
    """

    staged: int = 0
    evicted: int = 0
    inconclusive: int = 0
    blocked: int = 0
    refused_by_budget: int = 0
    superseded: int = 0
    capped: int = 0
    #: External round trips spent, against :data:`MAX_STAGE_ACTIONS_PER_PASS`.
    actions: int = 0

    def exhausted(self, *, limit: int = MAX_STAGE_ACTIONS_PER_PASS) -> bool:
        """Return whether this pass has spent its round-trip budget."""

        return self.actions >= limit

    def as_counts(self) -> dict[str, int]:
        """Return the countable outcomes, for logs and metrics."""

        return {
            "staged": self.staged,
            "evicted": self.evicted,
            "inconclusive": self.inconclusive,
            "blocked": self.blocked,
            "refused_by_budget": self.refused_by_budget,
            "superseded": self.superseded,
            "capped": self.capped,
        }


__all__ = (
    "CELL_STAGE_UNPROVEN_STATES",
    "MAX_STAGE_ACTIONS_PER_PASS",
    "CellStageError",
    "CellStageEvictionPlan",
    "CellStageExecutor",
    "CellStageInconclusiveError",
    "CellStageMeasurement",
    "CellStagePolicy",
    "CellStageRefusedError",
    "CellStageRequest",
    "CellStageStaleError",
    "StageEviction",
    "StageEvictionPolicy",
    "StageOutcome",
    "charged_bytes",
    "confirmed_stage_index",
    "declared_assets_on_cell",
    "plan_eviction",
    "stage_headroom_bytes",
)
