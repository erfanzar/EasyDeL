# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure worker-unit planning for declarative capacity pools."""

from __future__ import annotations

import math
import re
import typing as tp
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

from eyvan._internal.identifiers import stable_id
from eyvan.capacity import (
    DEFINITIVE_CAPACITY_LOSS_STATUSES,
    EXECUTION_PENDING_CAPACITY_STATUSES,
    QUEUED_CAPACITY_STATUSES,
)
from eyvan.capacity.acquisition_backoff import PoolAcquisitionBackoff, pool_acquisition_backoffs
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, backend_capabilities
from eyvan.domain import Scheduler, SchedulingResult
from eyvan.domain.models import PlacementMode, PriorityBand
from eyvan.records import (
    POOL_ALTERNATE_TEMPLATES_LABEL as POOL_ALTERNATE_TEMPLATES_LABEL,
)
from eyvan.records import (
    POOL_ALTERNATES_LABEL as POOL_ALTERNATES_LABEL,
)
from eyvan.records import (
    POOL_LABEL as POOL_LABEL,
)
from eyvan.records import (
    POOL_TEMPLATE_LABEL as POOL_TEMPLATE_LABEL,
)
from eyvan.records import (
    CapacityPoolRecord,
    CellRecord,
    CellState,
    LeaseRecord,
    PolicyRecord,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.records import (
    candidate_pool_templates as candidate_pool_templates,
)
from eyvan.server.service.scheduling import to_core_cell, to_core_workload
from eyvan.server.service.state import ACTIVE_LEASE_STATES

#: How long a shape must have gone unplaced before one pool may reclaim another
#: pool's capacity for it.
#:
#: Contention and fragmentation look identical for the first few passes: both are
#: "the group's quota is spent and this shape does not fit". They separate over
#: time, because contention clears when the current holders finish and
#: fragmentation does not. Reclaiming immediately would evict work to serve a
#: demand that was about to be satisfied anyway, and pay a whole provisioning
#: cycle to do it.
_DEFRAG_MIN_PENDING_SECONDS = 180.0

#: ``CellRecord.id`` is ``max_length=96`` and a cell id becomes a provider
#: resource name, so the minted form must fit whatever the widest counter is.
_CELL_ID_LIMIT = 96
_CELL_EPOCH_CHARS = 6


# Re-exported: the constants and ``candidate_pool_templates`` live in
# ``eyvan.records`` so the service scheduler can read them without importing
# this module (which imports the service layer).


def _virtual_pool_cell(pool: CapacityPoolRecord, cell_id: str, worker_count: int) -> CellRecord | None:
    """Treat a backend/factory shape refusal as infeasibility, not a fleet-wide failure.

    Capability declarations describe an upper bound, not a promise that this
    pool's factory can compile every shape. In particular, existing non-GCP
    factories still refuse multi-worker cells. Keep those durable requests
    pending without manufacturing a cell the real launch path cannot create.
    """

    if worker_count > min(pool.max_cells, backend_capabilities(pool.capacity_backend).max_workers_per_cell):
        return None
    if worker_count > 1 and not pool.hosts_multi_worker_cells:
        return None
    try:
        cell = pool.make_cell(cell_id, worker_count=worker_count)
    except ValueError:
        return None
    return cell.model_copy(update={"state": CellState.READY})


def _pool_workload_eligible(
    workload: WorkloadRecord,
    pool: CapacityPoolRecord,
    width: int,
    policy: PolicyRecord | None,
) -> bool:
    """Apply the same current shape and placement constraints to every acquisition."""

    if not pool.enabled:
        return False
    candidate = _virtual_pool_cell(pool, stable_id("policy_candidate", pool.id, workload.id, width), width)
    if candidate is None:
        return False
    core = to_core_workload(workload, policy)
    elastic = workload.spec.placement is PlacementMode.ELASTIC
    requested = core.total_resources.scale(width) if elastic else core.allocation_resources
    return (
        Scheduler._static_cell_failure(
            core, requested, to_core_cell(candidate), require_worker_count_equality=not elastic
        )
        is None
    )


def _candidate_pool_ids(workload: WorkloadRecord) -> tuple[str, ...]:
    """Return the pools this workload accepts, best-known choice first.

    Cross-zone alternates let a workload use pools with different Spot
    availability rather than wait only on its first choice.
    """

    primary = workload.spec.required_labels.get("eyvan_pool") or ""
    alternates = workload.spec.required_labels.get(POOL_ALTERNATES_LABEL) or ""
    ordered = [primary, *(name.strip() for name in alternates.split(","))]
    seen: set[str] = set()
    candidates: list[str] = []
    for pool_id in ordered:
        if pool_id and pool_id not in seen:
            seen.add(pool_id)
            candidates.append(pool_id)
    return tuple(candidates)


def candidate_zone_report(
    workload: WorkloadRecord,
    pools: tp.Sequence[CapacityPoolRecord],
    cells: tp.Sequence[CellRecord],
    leases: tp.Sequence[LeaseRecord],
    *,
    now: datetime,
) -> tuple[dict[str, object], ...]:
    """Describe every pool this workload accepts, best choice first.

    The same signals :func:`_preferred_candidate` ranks on, computed the same
    way and returned instead of consumed. Sharing the ranking rather than the
    numbers would be a second implementation of the decision; sharing the
    numbers and re-ranking here would be a second implementation of the order.
    So this calls the ranking function and reports the inputs beside its answer,
    counted the way :meth:`CapacitySimulation.build` counts them: enabled pools
    only, and only cells that still supply their pool -- a disabled pool or a
    drained cell is not capacity this decision could pick.

    Empty for a workload that accepts one pool: there is nothing to choose
    between, and a one-entry list reads as a decision that was never made.

    Args:
        workload: The workload whose candidates are being described.
        pools: Every pool in the fleet.
        cells: Every cell, for the free-supply and belief counts.
        now: When cooldowns are judged against.

    Returns:
        One mapping per candidate pool, ordered best first.
    """

    candidate_ids = _candidate_pool_ids(workload)
    if len(candidate_ids) < 2:
        return ()
    pools_by_id = {pool.id: pool for pool in pools}
    # The decision never asks a disabled pool, so neither may the report.
    candidates = [
        pool for pool in (pools_by_id.get(pool_id) for pool_id in candidate_ids) if pool is not None and pool.enabled
    ]
    if len(candidates) < 2:
        return ()

    free_owned: Counter[tuple[str, int, str]] = Counter()
    free_shared: Counter[tuple[str, int]] = Counter()
    cooling: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    # A cell is free when nothing holds an active lease on it. The same rule the
    # planner uses; a state-based test would count a busy cell as available the
    # moment its state stopped being the one this code happened to check.
    leased = {lease.cell_id for lease in leases if lease.state in ACTIVE_LEASE_STATES}
    for cell in cells:
        pool = pools_by_id.get(cell.pool_id or "")
        if pool is None or not pool.enabled or not cell_provides_pool_supply(cell, pool, now):
            continue
        belief = cell.availability
        if belief.cooldown_until is not None and belief.cooldown_until > now:
            cooling[pool.id] += 1
        failures[pool.id] = max(failures[pool.id], int(belief.consecutive_failures))
        if cell.id in leased:
            continue
        if cell.autoscale_request_workload_id is None:
            free_shared[(pool.id, cell.worker_count)] += 1
        else:
            free_owned[(pool.id, cell.worker_count, cell.autoscale_request_workload_id)] += 1

    preferred = _preferred_candidate(
        candidates,
        workload=workload,
        free_owned_exact_supply=free_owned,
        free_shared_exact_supply=free_shared,
        pool_cooling=cooling,
        pool_consecutive_failures=failures,
    )
    width = workload.spec.worker_count
    ordered = sorted(
        candidates,
        key=lambda pool: (
            pool.id != preferred.id,
            -free_owned[(pool.id, width, workload.id)],
            -free_shared[(pool.id, width)],
            cooling[pool.id],
            failures[pool.id],
            candidate_ids.index(pool.id),
        ),
    )
    return tuple(
        {
            "pool_id": pool.id,
            "zone": pool.zone,
            "preferred": pool.id == preferred.id,
            "free_owned_cells": free_owned[(pool.id, width, workload.id)],
            "free_shared_cells": free_shared[(pool.id, width)],
            "cooling_cells": cooling[pool.id],
            "consecutive_failures": failures[pool.id],
        }
        for pool in ordered
    )


#: Consecutive failed capacity acquisitions after which a pool stops being asked
#: for *new* capacity on behalf of a workload. Belief already ranks a cold pool
#: last, which decides *where* to ask but never whether to stop: a workload whose
#: only candidate pool keeps failing asks it again every time the cooldown
#: expires, and ``max_backoff_s`` caps that at an hour. One probe produced twenty
#: cells in a single zone over nineteen hours that way, each granted and released
#: without ever becoming READY.
#:
#: Five because a spot stockout is ordinary and a handful of retries is how spot
#: capacity is normally won; five failures in a row is the zone telling you
#: something a sixth attempt will not change.
MAX_CONSECUTIVE_ACQUISITION_FAILURES = 5


#: How far back a *deleted* cell's failure still counts toward its pool's
#: acquisition bound. An hour, the same horizon ``max_backoff_s`` caps a single
#: cell's cooldown at: long enough to see a replacement loop, short enough that
#: a pool which failed during yesterday's outage is asked again today.
ACQUISITION_FAILURE_MEMORY_S = 3600.0


def _recent_failed_acquisition_streak(cells: list[CellRecord], now: datetime) -> int:
    """Count a pool's newest incarnations that failed without ever delivering.

    A failing cell is replaced by a *new* cell, whose belief starts clean, and
    the failed one is deleted. Excluding all deleted cells would prevent the
    bound from ever tripping on a replacement loop; counting them forever would
    let old tombstones park a pool indefinitely.

    So the streak is read across outcomes, newest first, and it ends at the
    first cell that delivered capacity (one success ends a run of failures).
    Deleted cells whose outcome predates ``ACQUISITION_FAILURE_MEMORY_S`` are
    excluded: a long queue wait must not age out a newly observed failure.
    """

    streak = 0
    horizon = now.timestamp() - ACQUISITION_FAILURE_MEMORY_S

    # Tombstones are kept forever; only the recent ones can matter, so they are
    # dropped before sorting rather than walked past on every pass.
    def outcome_at(cell: CellRecord) -> datetime:
        # Ownership cleanup can refresh updated_at even on tombstones. Without
        # an observed outcome, conservatively age legacy evidence by cell birth.
        moment = cell.availability.last_observed_at or cell.created_at
        # Eyvan writes aware UTC; a naive persisted value is read as UTC too, so
        # it neither shifts by the host's zone nor breaks the sort below.
        return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)

    recent = [cell for cell in cells if cell.state is not CellState.DELETED or outcome_at(cell).timestamp() >= horizon]
    for cell in sorted(recent, key=lambda candidate: (outcome_at(candidate), candidate.id), reverse=True):
        if cell.availability.provision_latencies_s:
            break
        if cell.availability.consecutive_failures > 0:
            streak += 1
    return streak


def _acquisition_exhausted(pool: CapacityPoolRecord, pool_consecutive_failures: Counter[str]) -> bool:
    """Return whether this pool has failed to acquire capacity too often to ask again."""

    return pool_consecutive_failures[pool.id] >= MAX_CONSECUTIVE_ACQUISITION_FAILURES


def _preferred_candidate(
    candidates: list[CapacityPoolRecord],
    *,
    workload: WorkloadRecord,
    free_owned_exact_supply: Counter[tuple[str, int, str]],
    free_shared_exact_supply: Counter[tuple[str, int]],
    pool_cooling: Counter[str],
    pool_consecutive_failures: Counter[str],
) -> CapacityPoolRecord:
    """Choose which acceptable pool to place a workload in.

    Capacity that already exists wins outright: a cell this workload owns, then
    any free cell of the right width. Only when nothing is free does the choice
    become a bet on which zone will grant, and then the least-cold pool wins --
    otherwise a workload would keep asking the zone that has been reclaiming it
    while a healthy zone sits idle beside it.

    The first candidate wins ties, so a single-pool workload takes exactly the
    path it always did.
    """

    def rank(pool: CapacityPoolRecord) -> tuple[int, int, int, int]:
        owned = free_owned_exact_supply[(pool.id, workload.spec.worker_count, workload.id)]
        shared = free_shared_exact_supply[(pool.id, workload.spec.worker_count)]
        return (
            0 if owned else 1,
            0 if shared else 1,
            pool_cooling[pool.id],
            pool_consecutive_failures[pool.id],
        )

    return min(candidates, key=lambda pool: (rank(pool), candidates.index(pool)))


def pool_cell_prefix(pool: CapacityPoolRecord) -> str:
    """Return the provider-safe prefix for physical cells in ``pool``."""

    raw = re.sub(r"[^a-z0-9-]+", "-", f"{pool.fleet_name}-{pool.node_type}".lower()).strip("-")
    return (raw or "eyvan-cell")[:84].rstrip("-")


def pool_cell_epoch(pool: CapacityPoolRecord) -> str:
    """Return a short token naming this pool record's cell-id namespace.

    **Why cell ids cannot be a plain counter.** The control plane mints ids from
    ``len(managed) + 1`` and de-duplicates only against cells *it* currently
    holds. The agent, meanwhile, keeps a durable **tombstone** per released cell
    and refuses any later mutation at a generation not newer than the tombstone's
    -- permanently, by design, because that is what stops a resurrected cell from
    being adopted.

    Those two facts collide across time. A control plane whose database is empty
    re-creates the pool, begins at ``0001``, and walks straight into the
    tombstones a previous control plane left on the same agent. Every apply is
    rejected with "generation 1 is not newer than fenced cell ... generation 2",
    the cell is marked lost, the autoscaler mints the next id, and the fleet
    churns forever without ever provisioning.

    Derived from ``created_at`` specifically, not ``generation`` or
    ``initialized_at``. It is the only field that is *stable for the life of the
    pool record* and *fresh when the record is new*: ``upsert_capacity_pool``
    carries ``existing.created_at`` forward on every re-apply, so ``eyvan up``
    against a live fleet keeps minting into the same namespace, while a control
    plane starting on an empty database stamps a new one. ``generation`` moves on
    every re-apply (churning ids for no reason) and ``initialized_at`` is cleared
    to ``None`` by ``apply_fleet`` and set later by the controller, so it would
    flip the namespace midway through a pool's life.
    """

    return stable_id("cellns", pool.id, pool.created_at.isoformat()).split("_", 1)[1][:_CELL_EPOCH_CHARS]


def pool_cell_id(pool: CapacityPoolRecord, index: int) -> str:
    """Return the ``index``-th physical cell id for ``pool``.

    Owns the whole format so the length budget is enforced in one place: the
    counter is what gets the remaining room, since truncating *it* would alias
    two distinct cells onto one id.
    """

    epoch = pool_cell_epoch(pool)
    suffix = f"-{epoch}-{index:04d}"
    return f"{pool_cell_prefix(pool)[: _CELL_ID_LIMIT - len(suffix)].rstrip('-')}{suffix}"


def next_pool_cell_index(pool: CapacityPoolRecord, cells: tp.Iterable[CellRecord]) -> int:
    """Return an index this pool has never used, given every cell it has ever had.

    Deliberately monotonic rather than ``len(cells) + 1``. Counting live cells
    walks the counter *backwards* whenever one is deleted, so a pool that has
    churned hands the same id out twice -- and the second holder inherits
    whatever the first left behind at the provider under that name.

    Callers must pass *every* cell of the pool including deleted ones: a deleted
    record is precisely the evidence that its index is spent.
    """

    highest = 0
    prefix = f"{pool_cell_prefix(pool)}-{pool_cell_epoch(pool)}-"
    for cell in cells:
        if not cell.id.startswith(prefix):
            continue
        suffix = cell.id[len(prefix) :]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def cell_matches_pool(cell: CellRecord, pool: CapacityPoolRecord) -> bool:
    """Return whether a cell matches its pool template and physical shape.

    A cell its pool can no longer mint does not match it, which is an answer
    rather than an error -- and the same reasoning as
    :func:`cell_requires_replacement` below, guarded here for the same reason:
    both run inside leader phases where a raise stops the cycle.
    """

    try:
        desired = pool.make_cell(cell.id, worker_count=cell.worker_count)
    except ValueError:
        return False
    return cell_matches_template(cell, desired)


def cell_matches_template(cell: CellRecord, desired: CellRecord) -> bool:
    """Return whether a cell has the requested provisionable template."""

    return (
        cell.provider == desired.provider
        and cell.capacity_backend == desired.capacity_backend
        and cell.provider_project == desired.provider_project
        and cell.region == desired.region
        and cell.zone == desired.zone
        and cell.capacity == desired.capacity
        and cell.allocatable == desired.allocatable
        and cell.unit_prices == desired.unit_prices
        and cell.accelerator_type == desired.accelerator_type
        and cell.capacity_class == desired.capacity_class
        and cell.executor == desired.executor
        and cell.endpoint == desired.endpoint
        and cell.pool_id == desired.pool_id
        and cell.provider_options == desired.provider_options
        and cell.runtime == desired.runtime
        and cell.labels == desired.labels
        and cell.topology == desired.topology
        and cell.max_replica_resources == desired.max_replica_resources
        and cell.max_gang_replicas == desired.max_gang_replicas
        and cell.worker_count == desired.worker_count
    )


def cell_requires_replacement(cell: CellRecord, pool: CapacityPoolRecord) -> bool:
    """Return whether mutable metadata cannot reconcile a cell to its pool.

    A pool its own cell can no longer mint answers this question by itself: the
    cell cannot be reconciled, so it requires replacement. Letting ``make_cell``
    raise here instead made an ordinary pool edit fatal -- narrowing a pool so an
    existing multi-worker cell became unmintable raised inside the
    ``capacity-pools`` leader phase, which at the time stopped every later phase
    of every cycle. A phase's failure is contained now, but answering the
    question is still better than raising it: a pool edit is not an error, and a
    cell that cannot be minted has a correct answer here.
    """

    try:
        desired = pool.make_cell(cell.id, worker_count=cell.worker_count)
    except ValueError:
        return True
    return (
        cell.provider != desired.provider
        or cell.capacity_backend != desired.capacity_backend
        or cell.provider_project != desired.provider_project
        or cell.region != desired.region
        or cell.zone != desired.zone
        or cell.capacity != desired.capacity
        or cell.accelerator_type != desired.accelerator_type
        or cell.capacity_class != desired.capacity_class
        or cell.executor != desired.executor
        or cell.endpoint != desired.endpoint
        or cell.pool_id != desired.pool_id
        or cell.provider_options != desired.provider_options
        or cell.runtime != desired.runtime
        or cell.worker_count != desired.worker_count
    )


def cell_provides_pool_supply(cell: CellRecord, pool: CapacityPoolRecord, now: datetime) -> bool:
    """Return whether a real cell supplies or is still acquiring its worker units."""

    # Only these states can ever supply. Tested before the template match, which
    # mints a whole CellRecord, so the tombstones a long-lived plane keeps cost
    # nothing here; the answer for them was already False.
    if cell.state not in {CellState.READY, CellState.REGISTERING, CellState.UNAVAILABLE}:
        return False
    if not cell_matches_pool(cell, pool):
        return False
    phase = cell.provider_phase
    if phase in DEFINITIVE_CAPACITY_LOSS_STATUSES:
        return False
    if cell.state is CellState.READY:
        return True
    if cell.state not in {CellState.REGISTERING, CellState.UNAVAILABLE}:
        return False
    if phase is None:
        # Nothing has been asked of the provider yet: ``capacity.apply`` is
        # durable in the outbox and has not run, so no request exists that could
        # be late. Timing this out charges the cell for a provisioning it never
        # started, and the cost is not one wasted cell but an unbounded loop --
        # the reaped cell's queued apply is discarded as stale, a replacement is
        # launched, its apply goes to the back of the same queue, and the next
        # timeout repeats it for as long as the outbox stays behind.
        #
        # A slow outbox is a reason to wait for the apply, never a reason to
        # destroy the cell it belongs to. Provisioning is timed from when
        # provisioning actually begins, which is what ``provider_phase_started_at``
        # records.
        return True
    phase_started_at = cell.provider_phase_started_at or cell.created_at
    elapsed_s = max(0.0, (now - phase_started_at).total_seconds())
    if phase in QUEUED_CAPACITY_STATUSES:
        return pool.queue_timeout_s is None or elapsed_s < pool.queue_timeout_s
    return elapsed_s < pool.provisioning_timeout_s


def cell_execution_runtime_is_pending(cell: CellRecord) -> bool:
    """Return whether capacity exists but its native execution worker is unready."""

    return cell.provider_phase in EXECUTION_PENDING_CAPACITY_STATUSES


def _workload_band(record: WorkloadRecord | None) -> PriorityBand | None:
    """Return one workload's scheduler band, or ``None`` when it is not known.

    The record layer spells bands as strings and the scheduler orders them as an
    ``IntEnum``; this is the same ``PriorityBand[name]`` conversion the core
    scheduling service performs, kept here so the capacity layer does not import
    from the server.
    """

    if record is None:
        return None
    try:
        return PriorityBand[record.spec.priority_band.name]
    except KeyError:  # pragma: no cover - a band the scheduler does not order
        return None


def _demand_band(demand: PoolDemand, index: int) -> PriorityBand | None:
    """Return the band of one virtual shape, or ``None`` when it is not known."""

    if index >= len(demand.virtual_priorities):
        return None
    return demand.virtual_priorities[index]


def _demand_pending_seconds(demand: PoolDemand, index: int) -> float:
    """Return how long one virtual shape has waited, or 0 when that is unknown."""

    if index >= len(demand.virtual_pending_seconds):
        return 0.0
    return demand.virtual_pending_seconds[index]


def _pending_seconds(record: WorkloadRecord | None, now: datetime | None) -> float:
    """Return how long a workload has been waiting, or 0 when that is unknown.

    Zero is the safe unknown: the defragmentation gate is a *minimum* wait, so an
    unknown wait never authorises reclaiming another pool's capacity.
    """

    if record is None or now is None:
        return 0.0
    return max((now - record.created_at).total_seconds(), 0.0)


@dataclass(frozen=True, slots=True)
class PoolDemand:
    """Scheduler-selected real capacity and exact virtual cell shapes."""

    real_cell_ids: frozenset[str]
    virtual_worker_counts: tuple[int, ...]
    virtual_workload_ids: tuple[str, ...]
    #: Priority band of each virtual shape's requesting workload, and how long
    #: that workload has been waiting. Cross-pool defragmentation needs both: the
    #: band decides whether taking capacity from another pool is allowed at all,
    #: and the wait separates ordinary contention from fragmentation. Empty on a
    #: demand built without workloads, which reads as "not eligible" rather than
    #: as batch priority -- absence must never authorise an eviction.
    virtual_priorities: tuple[PriorityBand, ...] = ()
    virtual_pending_seconds: tuple[float, ...] = ()
    #: Highest band any active lease on that cell is serving. A cell absent from
    #: this mapping carries no active lease and is the cheapest thing to reclaim.
    real_cell_priorities: tp.Mapping[str, PriorityBand] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep each virtual capacity shape attached to its requesting workload."""

        if len(self.virtual_worker_counts) != len(self.virtual_workload_ids):
            raise ValueError("virtual worker counts must align with virtual workload ids")
        for name, values in (
            ("priorities", self.virtual_priorities),
            ("pending seconds", self.virtual_pending_seconds),
        ):
            if values and len(values) != len(self.virtual_worker_counts):
                raise ValueError(f"virtual {name} must align with virtual worker counts")


@dataclass(frozen=True, slots=True)
class CapacitySimulation:
    """Scheduler inventory plus ownership metadata for virtual capacity."""

    cells: tuple[CellRecord, ...]
    pool_by_cell_id: dict[str, str]
    virtual_worker_count_by_cell_id: dict[str, int]
    #: Workloads whose every acceptable pool has stopped granting capacity. They
    #: get no virtual candidate, so nothing is launched for them; carried out of
    #: the simulation so the caller can say so on the workload instead of leaving
    #: it pending with no reason.
    acquisition_exhausted_workload_ids: frozenset[str] = frozenset()
    #: Pools that refused queue admission for a workload, by workload id. A race
    #: must not spend one of its slots re-asking a zone that already said this
    #: project may not ask there.
    queue_denied_pool_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    #: Pools past the acquisition bound. Candidates already exclude them; a race
    #: must too, or it re-asks the pool the bound exists to stop asking.
    exhausted_pool_ids: frozenset[str] = frozenset()
    #: Real cells still acquiring (queued or provisioning, not READY) and the
    #: pools where each workload already owns such a cell. A placement on an
    #: acquiring cell has not won anything yet, so it keeps racing -- but never
    #: into a pool where the workload already has a request outstanding.
    acquiring_cell_ids: dict[str, int] = field(default_factory=dict)
    owned_pool_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    #: Pools where a workload's request failed outright and has not been retired
    #: yet (UNAVAILABLE with a provider failure). Not asked again by a race, and
    #: not counted as racing either: nothing is outstanding there.
    stalled_pool_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    #: The same policy snapshot used to admit virtual candidates must also gate
    #: race expansion, including races seeded by an already-acquiring cell.
    #: ``build`` always supplies it, even when empty. Hand-assembled demand-only
    #: snapshots retain their compatibility path without a simulation context.
    policies: tuple[PolicyRecord, ...] | None = None

    @classmethod
    def build(
        cls,
        pools: tuple[CapacityPoolRecord, ...],
        cells: tuple[CellRecord, ...],
        workloads: tuple[WorkloadRecord, ...],
        leases: tuple[LeaseRecord, ...],
        now: datetime,
        *,
        policies: tuple[PolicyRecord, ...] = (),
    ) -> CapacitySimulation:
        """Build real supply and shape-exact virtual candidates for pending work."""

        pools_by_id = {pool.id: pool for pool in pools}
        policy_by_scope = {(policy.org_id, policy.project_id): policy for policy in policies}

        def eligible(workload: WorkloadRecord, pool: CapacityPoolRecord, width: int) -> bool:
            return _pool_workload_eligible(
                workload, pool, width, policy_by_scope.get((workload.org_id, workload.project_id))
            )

        scheduler_cells: list[CellRecord] = []
        pool_by_cell_id: dict[str, str] = {}
        virtual_worker_count_by_cell_id: dict[str, int] = {}
        active_lease_cells = {lease.cell_id for lease in leases if lease.state in ACTIVE_LEASE_STATES}
        free_shared_exact_supply: Counter[tuple[str, int]] = Counter()
        free_owned_exact_supply: Counter[tuple[str, int, str]] = Counter()
        pool_cooling: Counter[str] = Counter()
        pool_consecutive_failures: Counter[str] = Counter()
        acquisition_exhausted_workload_ids: set[str] = set()
        # Counted separately from ``pool_consecutive_failures``, which is only
        # populated for cells that still supply the pool -- a cell reclaimed
        # after failing to provision supplies nothing, so the very belief this
        # bound needs is the belief that loop skips. Ranking wants "how cold is
        # the capacity I could still use"; this wants "how often has asking here
        # failed", and they are different questions about the same field.
        pool_acquisition_failures: Counter[str] = Counter()
        incarnations: dict[str, list[CellRecord]] = {}
        for cell in cells:
            pool_id = cell.pool_id or ""
            if pool_id not in pools_by_id:
                continue
            incarnations.setdefault(pool_id, []).append(cell)
            if cell.state is CellState.DELETED:
                # A tombstone is retained forever as id evidence, and its belief
                # is frozen: only the cell's own SUCCESS resets it, and a
                # deleted cell has none. Letting it vote here made one cell
                # that crossed the bound during an outage park every later
                # workload for that pool permanently. Recent tombstones speak
                # through the bounded streak below instead.
                continue
            pool_acquisition_failures[pool_id] = max(
                pool_acquisition_failures[pool_id], int(cell.availability.consecutive_failures)
            )
        for pool_id, pool_cells in incarnations.items():
            pool_acquisition_failures[pool_id] = max(
                pool_acquisition_failures[pool_id], _recent_failed_acquisition_streak(pool_cells, now)
            )
        # A queue-admission denial is the provider saying this project may not
        # ask for this shape in this zone. It is not a stockout to wait out: a
        # denied cell must not count as supply until the provisioning timeout or
        # prevent a cross-zone workload from reaching its other zones. Each
        # workload stops considering the pools that denied *it*.
        queue_denied_pools: dict[str, set[str]] = {}
        owned_pools: dict[str, set[str]] = {}
        stalled_pools: dict[str, set[str]] = {}
        acquiring_cell_ids: dict[str, int] = {}
        for cell in cells:
            if (
                cell.provider_failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
                and cell.autoscale_request_workload_id is not None
                and cell.pool_id is not None
            ):
                queue_denied_pools.setdefault(cell.autoscale_request_workload_id, set()).add(cell.pool_id)

        for cell in cells:
            pool = pools_by_id.get(cell.pool_id or "")
            if pool is None:
                scheduler_cells.append(cell)
                continue
            pool_by_cell_id[cell.id] = pool.id
            supplies = (
                pool.enabled
                and cell.provider_failure_code is not CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
                and cell_provides_pool_supply(cell, pool, now)
            )
            if supplies and cell.autoscale_request_workload_id is not None:
                # A request the provider has already refused (a Compute stockout
                # answers at once) still counts as supply until its timeout, so
                # the pool does not launch a duplicate; but it is not racing.
                # Counting failed requests as racing would occupy race slots
                # until timeout and prevent trying other zones.
                failed_request = cell.state is CellState.UNAVAILABLE and cell.provider_failure_code is not None
                target = stalled_pools if failed_request else owned_pools
                target.setdefault(cell.autoscale_request_workload_id, set()).add(pool.id)
            if supplies and cell.state is CellState.READY:
                scheduler_cells.append(cell)
            elif supplies and cell.state in {CellState.REGISTERING, CellState.UNAVAILABLE}:
                acquiring_cell_ids[cell.id] = cell.worker_count
                scheduler_cells.append(cell.model_copy(update={"state": CellState.READY}))
            elif cell.id in active_lease_cells:
                # Not supply, and not omissible either. The scheduler is handed
                # these cells *and* the live leases in one call, and it rejects a
                # snapshot whose lease names a cell the list does not hold --
                # correctly, since it cannot check that lease against any
                # capacity. Dropping a leased cell therefore does not skip one
                # cell, it raises out of a leader phase that has no failure
                # isolation, so nothing later in the cycle runs: no observation,
                # no settlement, no release. Disabling a pool must not prevent
                # its remaining leases from being observed and settled.
                #
                # DRAINING rather than the cell's own state because that is what
                # it now is: it carries work and must attract none. ``to_core_cell``
                # reads exactly that as ``enabled=False``.
                scheduler_cells.append(cell.model_copy(update={"state": CellState.DRAINING}))
            if not supplies:
                continue
            if cell.id not in active_lease_cells:
                if cell.autoscale_request_workload_id is None:
                    free_shared_exact_supply[(pool.id, cell.worker_count)] += 1
                else:
                    free_owned_exact_supply[(pool.id, cell.worker_count, cell.autoscale_request_workload_id)] += 1
            # How badly this pool is currently failing, for choosing between
            # pools of the same shape in different zones. Belief is kept per
            # cell, so a pool is as cold as its coldest cell.
            belief = cell.availability
            if belief.cooldown_until is not None and belief.cooldown_until > now:
                pool_cooling[pool.id] += 1
            pool_consecutive_failures[pool.id] = max(
                pool_consecutive_failures[pool.id], int(belief.consecutive_failures)
            )

        generic_workloads: list[WorkloadRecord] = []
        for workload in workloads:
            if workload.state is not WorkloadState.PENDING:
                continue
            named = [
                candidate
                for candidate in (pools_by_id.get(pool_id) for pool_id in _candidate_pool_ids(workload))
                if candidate is not None and candidate.enabled
            ]
            # A pool that has failed this many times running is not asked for new
            # capacity again. Existing free capacity is still usable -- this bounds
            # acquisition, not placement -- so a pool that suddenly grants is picked
            # up by the free-supply branch below without waiting for belief to decay.
            denied_here = queue_denied_pools.get(workload.id, set())
            eligible_named = [
                pool
                for pool in named
                if eligible(
                    workload, pool, 1 if workload.spec.placement is PlacementMode.ELASTIC else workload.spec.worker_count
                )
            ]
            if named and not eligible_named:
                continue
            candidates = [
                pool
                for pool in eligible_named
                if not _acquisition_exhausted(pool, pool_acquisition_failures) and pool.id not in denied_here
            ]
            if named and not candidates:
                # Every zone it accepts has stopped granting. Say so once, durably,
                # instead of asking again on the next tick forever.
                acquisition_exhausted_workload_ids.add(workload.id)
                continue
            if not candidates:
                generic_workloads.append(workload)
                continue
            exact_pool = _preferred_candidate(
                candidates,
                workload=workload,
                free_owned_exact_supply=free_owned_exact_supply,
                free_shared_exact_supply=free_shared_exact_supply,
                pool_cooling=pool_cooling,
                pool_consecutive_failures=pool_consecutive_failures,
            )
            if workload.spec.placement is PlacementMode.ELASTIC:
                # Pack warm free cells of any worker count first -- the same
                # biggest-first order ``Scheduler._schedule_elastic`` packs with,
                # so what the simulation counts as covered is what the scheduler
                # can actually lease -- and ask for only the deficit as one new
                # cell. The resize case this answers: a pods-4 workload with two
                # warm slices provisioned two, not four.
                packed = cls._consume_elastic_warm_supply(
                    workload,
                    candidates,
                    free_owned_exact_supply=free_owned_exact_supply,
                    free_shared_exact_supply=free_shared_exact_supply,
                    eligible_width=lambda pool_id, width, owner=workload: eligible(owner, pools_by_id[pool_id], width),
                )
                deficit = workload.spec.worker_count - packed
                if deficit == 0:
                    continue
                cell_id = stable_id("virtual_cell", exact_pool.id, workload.id, "elastic")
                virtual_cell = _virtual_pool_cell(exact_pool, cell_id, deficit)
                if virtual_cell is None:
                    continue
                pool_by_cell_id[cell_id] = exact_pool.id
                virtual_worker_count_by_cell_id[cell_id] = deficit
                scheduler_cells.append(virtual_cell)
                continue
            if workload.spec.worker_count > 1 and not exact_pool.hosts_multi_worker_cells:
                # The same guard the generic branch below has always had, and for a
                # stronger reason here: this branch calls ``make_cell`` on a pool the
                # *workload* named, so a spec that is already durable decides whether
                # this planner raises. ``make_cell`` refuses a multi-worker cell on
                # any non-TPU backend by raising, and ``capacity-pools`` is a leader
                # phase whose failure, at the time, aborted every
                # later phase of the cycle, including scheduling, settlement and
                # billing, on every tick, for the whole control plane. A workload that
                # can never be placed is skipped and left PENDING, which is what an
                # unplaceable workload is; it must not be able to stop the plane.
                continue
            for member in range(workload.spec.cell_group_count):
                supply_key = (exact_pool.id, workload.spec.worker_count)
                owned_supply_key = (*supply_key, workload.id)
                if free_owned_exact_supply[owned_supply_key]:
                    free_owned_exact_supply[owned_supply_key] -= 1
                    continue
                if free_shared_exact_supply[supply_key]:
                    free_shared_exact_supply[supply_key] -= 1
                    continue
                cell_id = stable_id("virtual_cell", exact_pool.id, workload.id, member)
                virtual_cell = _virtual_pool_cell(exact_pool, cell_id, workload.spec.worker_count)
                if virtual_cell is None:
                    continue
                pool_by_cell_id[cell_id] = exact_pool.id
                virtual_worker_count_by_cell_id[cell_id] = workload.spec.worker_count
                scheduler_cells.append(virtual_cell)

        for pool in pools:
            if not pool.enabled:
                continue
            exhausted = _acquisition_exhausted(pool, pool_acquisition_failures)
            for workload in generic_workloads:
                if exhausted or pool.id in queue_denied_pools.get(workload.id, set()):
                    # Apply the named branch's bounds here too, so generic
                    # submissions do not repeatedly request exhausted or denied pools.
                    continue
                worker_count = workload.spec.worker_count
                if worker_count > 1 and not pool.hosts_multi_worker_cells:
                    # Multi-worker cells only exist on the GCP TPU backend with the
                    # native executor; other pools can never host this shape. Asked
                    # through the pool record so this test and the refusal inside
                    # ``make_cell`` cannot drift apart.
                    continue
                for member in range(workload.spec.cell_group_count):
                    cell_id = stable_id("virtual_cell", pool.id, "generic", workload.id, member)
                    virtual_cell = _virtual_pool_cell(pool, cell_id, worker_count)
                    if virtual_cell is None or not eligible(workload, pool, worker_count):
                        continue
                    pool_by_cell_id[cell_id] = pool.id
                    virtual_worker_count_by_cell_id[cell_id] = worker_count
                    scheduler_cells.append(virtual_cell)

        return cls(
            cells=tuple(scheduler_cells),
            pool_by_cell_id=pool_by_cell_id,
            virtual_worker_count_by_cell_id=virtual_worker_count_by_cell_id,
            acquisition_exhausted_workload_ids=frozenset(acquisition_exhausted_workload_ids),
            queue_denied_pool_ids={workload_id: frozenset(pools) for workload_id, pools in queue_denied_pools.items()},
            exhausted_pool_ids=frozenset(
                pool.id for pool in pools if _acquisition_exhausted(pool, pool_acquisition_failures)
            ),
            acquiring_cell_ids=acquiring_cell_ids,
            owned_pool_ids={workload_id: frozenset(pool_ids) for workload_id, pool_ids in owned_pools.items()},
            stalled_pool_ids={workload_id: frozenset(pool_ids) for workload_id, pool_ids in stalled_pools.items()},
            policies=policies,
        )

    @staticmethod
    def _consume_elastic_warm_supply(
        workload: WorkloadRecord,
        candidate_pools: list[CapacityPoolRecord],
        *,
        free_owned_exact_supply: Counter[tuple[str, int, str]],
        free_shared_exact_supply: Counter[tuple[str, int]],
        eligible_width: tp.Callable[[str, int], bool] | None = None,
    ) -> int:
        """Reserve warm free cells toward the workload's worker count.

        Own-owned supply is preferred over shared at the same size: a cell this
        workload's earlier attempt already provisioned is the cheapest warm
        capacity it will ever see. Mirrors ``Scheduler._schedule_elastic``'s
        biggest-first order so the two never disagree about what is covered.
        """

        need = workload.spec.worker_count
        pool_ids = {pool.id for pool in candidate_pools}
        entries: list[tuple[int, int, str, int]] = []
        for (pool_id, worker_count), count in free_shared_exact_supply.items():
            if pool_id in pool_ids:
                entries.extend((-worker_count, 1, pool_id, worker_count) for _ in range(count))
        for (pool_id, worker_count, owner), count in free_owned_exact_supply.items():
            if pool_id in pool_ids and owner == workload.id:
                entries.extend((-worker_count, 0, pool_id, worker_count) for _ in range(count))
        entries.sort()
        packed = 0
        for _negated, ownership, pool_id, worker_count in entries:
            if eligible_width is not None and not eligible_width(pool_id, worker_count):
                continue
            if packed >= need or need - packed < worker_count:
                continue
            if ownership == 0:
                key = (pool_id, worker_count, workload.id)
                if not free_owned_exact_supply[key]:
                    continue
                free_owned_exact_supply[key] -= 1
            else:
                key = (pool_id, worker_count)
                if not free_shared_exact_supply[key]:
                    continue
                free_shared_exact_supply[key] -= 1
            packed += worker_count
        return packed

    def demand(
        self,
        pools: tuple[CapacityPoolRecord, ...],
        leases: tuple[LeaseRecord, ...],
        result: SchedulingResult,
        workloads: tuple[WorkloadRecord, ...] = (),
        now: datetime | None = None,
        race_width: int = 1,
    ) -> dict[str, PoolDemand]:
        """Resolve scheduler placements to physical cell shapes per pool.

        ``workloads`` and ``now`` are what let the planner decide whether one
        pool may reclaim another's capacity. Both are optional so a caller that
        does not defragment is unchanged; omitting them yields a demand that is
        never eligible to take capacity from a peer.

        ``race_width`` is how many of a workload's candidate pools may be asked
        for capacity at once. One -- the default -- is the sequential behaviour:
        ask the best zone, and try the next one on the following tick if it is
        cold. Higher spends money in more than one zone to find out which will
        actually grant, which is how spot capacity is won and is a decision an
        operator makes rather than a default anyone inherits.
        """

        real_ids: dict[str, set[str]] = {pool.id: set() for pool in pools}
        virtual_counts: dict[str, list[int]] = {pool.id: [] for pool in pools}
        virtual_workload_ids: dict[str, list[str]] = {pool.id: [] for pool in pools}
        by_workload_id = {record.id: record for record in workloads}
        cell_priorities: dict[str, dict[str, PriorityBand]] = {pool.id: {} for pool in pools}
        for lease in leases:
            if lease.state not in ACTIVE_LEASE_STATES:
                continue
            pool_id = self.pool_by_cell_id.get(lease.cell_id)
            if pool_id in real_ids:
                real_ids[pool_id].add(lease.cell_id)
                band = _workload_band(by_workload_id.get(lease.workload_id))
                if band is not None:
                    served = cell_priorities[pool_id]
                    current = served.get(lease.cell_id)
                    served[lease.cell_id] = band if current is None else max(current, band)
        race_seeds: list[tuple[str, str, int]] = []
        for placement in result.placements:
            for lease in placement.all_leases:
                pool_id = self.pool_by_cell_id.get(lease.cell_id)
                if pool_id not in real_ids:
                    continue
                worker_count = self.virtual_worker_count_by_cell_id.get(lease.cell_id)
                if worker_count is None:
                    real_ids[pool_id].add(lease.cell_id)
                    if lease.cell_id in self.acquiring_cell_ids:
                        # Placed on a request the provider has not granted yet:
                        # still a race. Accepting a *request* does not grant
                        # capacity; it may remain in WAITING_FOR_RESOURCES.
                        race_seeds.append((pool_id, lease.workload_id, self.acquiring_cell_ids[lease.cell_id]))
                else:
                    virtual_counts[pool_id].append(worker_count)
                    virtual_workload_ids[pool_id].append(lease.workload_id)
        if race_width > 1:
            pools_by_id = {pool.id: pool for pool in pools}
            policy_by_scope = {(policy.org_id, policy.project_id): policy for policy in self.policies or ()}
            self._race_alternates(
                virtual_counts,
                virtual_workload_ids,
                by_workload_id=by_workload_id,
                race_width=race_width,
                denied_pool_ids=self.queue_denied_pool_ids,
                exhausted_pool_ids=self.exhausted_pool_ids,
                owned_pool_ids=self.owned_pool_ids,
                stalled_pool_ids=self.stalled_pool_ids,
                eligible_pool=(
                    lambda workload, pool_id, width: _pool_workload_eligible(
                        workload,
                        pools_by_id[pool_id],
                        width,
                        policy_by_scope.get((workload.org_id, workload.project_id)),
                    )
                )
                if self.policies is not None
                else None,
                seeds=tuple(race_seeds),
            )
        return {
            pool.id: PoolDemand(
                real_cell_ids=frozenset(real_ids[pool.id]),
                virtual_worker_counts=tuple(virtual_counts[pool.id]),
                virtual_workload_ids=tuple(virtual_workload_ids[pool.id]),
                virtual_priorities=tuple(
                    _workload_band(by_workload_id.get(workload_id)) or PriorityBand.BATCH
                    for workload_id in virtual_workload_ids[pool.id]
                )
                if by_workload_id
                else (),
                virtual_pending_seconds=tuple(
                    _pending_seconds(by_workload_id.get(workload_id), now)
                    for workload_id in virtual_workload_ids[pool.id]
                )
                if by_workload_id and now is not None
                else (),
                real_cell_priorities=dict(cell_priorities[pool.id]),
            )
            for pool in pools
        }

    @staticmethod
    def _race_alternates(
        virtual_counts: dict[str, list[int]],
        virtual_workload_ids: dict[str, list[str]],
        *,
        by_workload_id: dict[str, WorkloadRecord],
        race_width: int,
        denied_pool_ids: tp.Mapping[str, frozenset[str]] | None = None,
        exhausted_pool_ids: frozenset[str] = frozenset(),
        owned_pool_ids: tp.Mapping[str, frozenset[str]] | None = None,
        stalled_pool_ids: tp.Mapping[str, frozenset[str]] | None = None,
        eligible_pool: tp.Callable[[WorkloadRecord, str, int], bool] | None = None,
        seeds: tuple[tuple[str, str, int], ...] = (),
    ) -> None:
        """Ask a workload's other candidate zones for capacity too.

        The scheduler places a workload into exactly one pool, so demand built
        from placements alone asks one zone and waits. That is the wrong shape
        for spot: a zone that will not grant costs a whole reconcile interval to
        discover, and the next one costs another. Asking several at once turns
        a serial search into a race, and the first zone to actually produce a
        machine wins.

        Every extra request is real money, so it is bounded twice: by
        ``race_width``, and by the workload's own declared alternates -- pools
        of the same shape that the submission path already proved
        interchangeable. The demand path also rechecks current policy and shape
        eligibility before taking a slot: declared interchangeability is not
        authority to provision under a policy that has since changed.

        The losers are not this function's problem, and that is deliberate: a
        cell requested here carries its workload's reservation, the reservation
        survives exactly as long as the workload is still waiting, and the
        recovery sweep returns it to shared supply the moment the workload is
        leased somewhere. Racing without that ordering strands paid capacity,
        which is why this is the last piece added rather than the first.
        """

        entries = [
            (pool_id, workload_id, virtual_counts[pool_id][index])
            for pool_id in list(virtual_counts)
            for index, workload_id in enumerate(list(virtual_workload_ids[pool_id]))
        ]
        entries.extend(seeds)
        for pool_id, workload_id, worker_count in entries:
            workload = by_workload_id.get(workload_id)
            if workload is None:
                continue
            denied = (denied_pool_ids or {}).get(workload_id, frozenset())
            # A pool where this workload already has a request outstanding (or
            # capacity) is already in the race; asking it again would mint a
            # second cell for the same attempt on every pass.
            owned = (owned_pool_ids or {}).get(workload_id, frozenset())
            stalled = (stalled_pool_ids or {}).get(workload_id, frozenset())
            alternates = [
                candidate
                for candidate in _candidate_pool_ids(workload)
                if candidate != pool_id
                and candidate in virtual_counts
                and candidate not in denied
                and candidate not in exhausted_pool_ids
                and candidate not in owned
                and candidate not in stalled
                and (eligible_pool is None or eligible_pool(workload, candidate, worker_count))
            ]
            # The width counts zones being asked, the ones already asked
            # included, so an ongoing race tops up rather than grows.
            slots = max(race_width - 1 - len(owned - {pool_id}), 0)
            for alternate in alternates[:slots]:
                already = sum(1 for existing in virtual_workload_ids[alternate] if existing == workload_id)
                # One request per member per pool. Without this a workload
                # already asking an alternate -- because a previous tick
                # raced it, or because it is a multislice group -- would
                # accumulate another request on every reconcile.
                if already >= workload.spec.cell_group_count:
                    continue
                virtual_counts[alternate].append(worker_count)
                virtual_workload_ids[alternate].append(workload_id)


@dataclass(frozen=True, slots=True)
class PoolScaleIntent:
    """One pool's desired worker-unit changes before shared quota allocation."""

    pool: CapacityPoolRecord
    live: tuple[CellRecord, ...]
    matching_supply: tuple[CellRecord, ...]
    demand: PoolDemand
    target_worker_units: int
    requested_worker_counts: tuple[int, ...]
    requested_exact_count: int
    demanded_virtual_worker_counts: tuple[int, ...]
    #: The shrink-clock movement this intent decided; see ``PoolScalePlan``.
    shrink_since_update: datetime | None = None
    shrink_since_changed: bool = False
    #: Whether the hysteresis held today's shape on this pass; see ``PoolScalePlan``.
    shrink_held: bool = False
    #: The pool's failed-acquisition streak while it is backing off, else None.
    #: A paused intent requests nothing new; see ``PoolScalePlan``.
    acquisition_backoff: PoolAcquisitionBackoff | None = None
    #: Launches the pool wanted on this pass and did not make because of it.
    withheld_worker_counts: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PoolScalePlan:
    """Atomic launch and retention plan after max-worker and quota limits."""

    pool: CapacityPoolRecord
    live: tuple[CellRecord, ...]
    matching_supply: tuple[CellRecord, ...]
    target_worker_units: int
    launch_worker_counts: tuple[int, ...]
    launch_workload_ids: tuple[str | None, ...]
    protected_cell_ids: frozenset[str]
    forced_drain_cell_ids: frozenset[str]
    #: New value for ``CapacityPoolRecord.demand_below_supply_since`` when this
    #: plan moved the shrink clock, with ``shrink_since_changed`` saying so.
    #: The planner reads the clock; the controller is what persists it.
    shrink_since_update: datetime | None = None
    shrink_since_changed: bool = False
    #: True when this plan holds today's shape because the shrink window has
    #: not elapsed. The controller leaves protected cells' idle clocks alone on
    #: such a pass: they are the clocks the post-window fall drains against.
    shrink_held: bool = False
    #: Set while the pool's recent acquisitions keep failing and its backoff
    #: has not expired. Nothing new is launched -- for demand or for a warm
    #: floor -- and nothing is drained to make room for a launch that is not
    #: happening. Existing cells are untouched.
    acquisition_backoff: PoolAcquisitionBackoff | None = None
    #: What this pass would have launched without the backoff, so the
    #: controller can say a launch was withheld rather than that none was wanted.
    withheld_worker_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Keep every planned physical cell associated with an optional source workload."""

        if len(self.launch_worker_counts) != len(self.launch_workload_ids):
            raise ValueError("planned worker counts must align with planned workload ids")

    @property
    def supply_worker_units_after_launch(self) -> int:
        return sum(cell.worker_count for cell in self.matching_supply) + sum(self.launch_worker_counts)


@dataclass(frozen=True, slots=True)
class QuotaAllocation:
    """Shared-quota launches and the units that must be reclaimed to allow them."""

    launch_worker_counts: dict[str, tuple[int, ...]]
    reclaim_worker_units: dict[str, int]
    blocked_workload_ids: frozenset[str]
    #: Worker units each pool must give up so a *different* pool in its quota
    #: group can launch a shape it cannot otherwise fit. Keyed by the pool that
    #: owns the cells, so the drain lands in that pool's own plan and the
    #: executor stays unchanged.
    donated_worker_units: dict[str, int] = field(default_factory=dict)


class CapacityPlanner:
    """Plan atomic physical cells while pool bounds remain worker-unit bounds."""

    def __init__(
        self,
        pools: tuple[CapacityPoolRecord, ...],
        cells_by_pool: dict[str, list[CellRecord]],
        now: datetime,
        defragmentation_min_pending_seconds: float = _DEFRAG_MIN_PENDING_SECONDS,
    ):
        self._pools = pools
        self._cells_by_pool = cells_by_pool
        self._now = now
        self._defrag_min_pending_seconds = defragmentation_min_pending_seconds
        self._quota_blocked_workload_ids: frozenset[str] = frozenset()
        # Read across every incarnation, tombstones included: a failed cell is
        # replaced, so the replacement loop is only visible in the deleted ones.
        self._acquisition_backoffs = pool_acquisition_backoffs(
            pools,
            (cell for pool_cells in cells_by_pool.values() for cell in pool_cells),
            now,
        )

    @property
    def acquisition_backoffs(self) -> dict[str, PoolAcquisitionBackoff]:
        """Return each pool's current failed-acquisition streak, active or not."""

        return dict(self._acquisition_backoffs)

    @property
    def quota_blocked_workload_ids(self) -> frozenset[str]:
        """Return exact pending workloads rejected by a finite shared provider quota."""

        return self._quota_blocked_workload_ids

    def plan(self, demand_by_pool: dict[str, PoolDemand]) -> dict[str, PoolScalePlan]:
        """Return quota-safe atomic changes for every pool."""

        intents = {pool.id: self._intent(pool, demand_by_pool[pool.id]) for pool in self._pools}
        quota = self._allocate_shared_quota(intents)
        self._quota_blocked_workload_ids = quota.blocked_workload_ids
        return {
            pool.id: self._finalize(
                intents[pool.id],
                quota.launch_worker_counts[pool.id],
                quota.reclaim_worker_units[pool.id],
                quota.donated_worker_units.get(pool.id, 0),
            )
            for pool in self._pools
        }

    def _intent(self, pool: CapacityPoolRecord, demand: PoolDemand) -> PoolScaleIntent:
        # A shape refused by the pool/backend cannot become feasible by draining
        # inventory. Demand can also arrive through race expansion, bypassing
        # virtual candidate creation. Check both boundaries and preserve all
        # parallel workload metadata when discarding an impossible shape.
        feasible = tuple(
            index
            for index, width in enumerate(demand.virtual_worker_counts)
            if _virtual_pool_cell(pool, stable_id("virtual_cell", pool.id, "intent", index), width) is not None
        )
        if len(feasible) != len(demand.virtual_worker_counts):
            demand = PoolDemand(
                real_cell_ids=demand.real_cell_ids,
                virtual_worker_counts=tuple(demand.virtual_worker_counts[index] for index in feasible),
                virtual_workload_ids=tuple(demand.virtual_workload_ids[index] for index in feasible),
                virtual_priorities=tuple(demand.virtual_priorities[index] for index in feasible)
                if demand.virtual_priorities
                else (),
                virtual_pending_seconds=tuple(demand.virtual_pending_seconds[index] for index in feasible)
                if demand.virtual_pending_seconds
                else (),
                real_cell_priorities=demand.real_cell_priorities,
            )
        managed = tuple(sorted(self._cells_by_pool[pool.id], key=lambda cell: cell.id))
        live = tuple(cell for cell in managed if cell.state is not CellState.DELETED)
        matching = tuple(cell for cell in live if cell_provides_pool_supply(cell, pool, self._now))
        floor = pool.initial_target if pool.enabled and pool.initialized_at is None else pool.warm_floor(self._now)
        real_by_id = {cell.id: cell for cell in matching}
        required_real_units = sum(
            real_by_id[cell_id].worker_count for cell_id in demand.real_cell_ids if cell_id in real_by_id
        )
        demanded_virtual_units = sum(demand.virtual_worker_counts)
        target = (
            0
            if not pool.enabled
            else min(
                pool.max_cells,
                max(floor, required_real_units + demanded_virtual_units),
            )
        )
        # Shrink hysteresis: demand that dips below supply for one planning
        # pass does not get to tear capacity down. The shortfall must hold for
        # ``shrink_stabilization_s`` before the target falls; until then the
        # shape is held exactly as it is, and the controller stamps the clock.
        # Forced drains are untouched by this -- they answer real pending work,
        # not a quiet minute, and run before held cells gain protection.
        matching_units_now = sum(cell.worker_count for cell in matching)
        shrink_since_update: datetime | None = None
        shrink_since_changed = False
        shrink_held = False
        if pool.enabled and pool.shrink_stabilization_s > 0 and target < matching_units_now:
            since = pool.demand_below_supply_since
            if since is None or (self._now - since).total_seconds() < pool.shrink_stabilization_s:
                # First sight of the shortfall, or it has not held long enough:
                # keep today's shape and start (or continue) the clock.
                target = matching_units_now
                shrink_held = True
                if since is None:
                    shrink_since_update = self._now
                    shrink_since_changed = True
            # Once the window has elapsed the target simply falls, and the clock
            # keeps running. Clearing it here would re-arm the hold on the next
            # pass and raise the target straight back, so a pool under a
            # persistent shortfall would fall for exactly one pass and then
            # hold forever. Only demand that meets supply again clears it, in
            # the branch below.
        elif pool.demand_below_supply_since is not None:
            shrink_since_update = None
            shrink_since_changed = True
        occupied_units = sum(cell.worker_count for cell in live)
        available_units = max(pool.max_cells - occupied_units, 0)
        matching_units = sum(cell.worker_count for cell in matching)
        rate_limit = (
            pool.launch_batch
            if not matching
            else min(pool.launch_batch, max(1, math.ceil(len(matching) * pool.upscaling_speed)))
        )
        requested: list[int] = []
        for worker_count in demand.virtual_worker_counts:
            if len(requested) >= rate_limit or worker_count > available_units:
                break
            requested.append(worker_count)
            available_units -= worker_count
        requested_exact_count = len(requested)
        # Only an explicit warm/initial floor permits unowned singleton cells.
        # Pending exact demand must wait for its shape, not fill a clamped target
        # with placeholders that the next pass immediately reclaims.
        floor_deficit = max(floor - matching_units - sum(requested), 0)
        while floor_deficit and available_units and len(requested) < rate_limit:
            requested.append(1)
            floor_deficit -= 1
            available_units -= 1
        # The one gate on *new* capacity for a pool whose acquisitions keep
        # failing. Per-cell belief backs off observation of a failed cell, but a
        # failed cell is replaced and its replacement starts clean. Demand and
        # warm floors must both respect the pool-level gate to bound replacement loops.
        backoff = self._acquisition_backoffs.get(pool.id)
        withheld: tuple[int, ...] = ()
        if backoff is not None and backoff.active(self._now):
            withheld = tuple(requested)
            requested = []
            requested_exact_count = 0
        else:
            backoff = None
        return PoolScaleIntent(
            pool=pool,
            live=live,
            matching_supply=matching,
            demand=demand,
            target_worker_units=target,
            requested_worker_counts=tuple(requested),
            requested_exact_count=requested_exact_count,
            demanded_virtual_worker_counts=demand.virtual_worker_counts,
            shrink_since_update=shrink_since_update,
            shrink_since_changed=shrink_since_changed,
            shrink_held=shrink_held,
            acquisition_backoff=backoff,
            withheld_worker_counts=withheld,
        )

    def _allocate_shared_quota(self, intents: dict[str, PoolScaleIntent]) -> QuotaAllocation:
        allowed = {pool_id: intent.requested_worker_counts for pool_id, intent in intents.items()}
        reclaim_worker_units = {pool_id: 0 for pool_id in intents}
        donated_worker_units: dict[str, int] = {}
        blocked_workload_ids: set[str] = set()
        grouped: dict[str, list[PoolScaleIntent]] = {}
        for intent in intents.values():
            if intent.pool.quota_group is not None:
                grouped.setdefault(intent.pool.quota_group, []).append(intent)

        for members in grouped.values():
            group_limit = min(intent.pool.quota_limit for intent in members)
            if group_limit is None:
                raise ValueError("quota group pools must configure quota_limit")
            live_units = sum(
                cell.worker_count * intent.pool.quota_units_per_cell
                for intent in members
                for cell in intent.live
                if intent.pool.quota_units_per_cell is not None
            )
            remaining_units = max(group_limit - live_units, 0)
            accepted: dict[str, list[int]] = {intent.pool.id: [] for intent in members}
            while remaining_units:
                candidates: list[tuple[PoolScaleIntent, int, int]] = []
                for intent in members:
                    accepted_count = len(accepted[intent.pool.id])
                    if accepted_count >= len(intent.requested_worker_counts):
                        continue
                    units_per_worker = intent.pool.quota_units_per_cell
                    if units_per_worker is None:
                        raise ValueError("quota group pools must configure quota_units_per_cell")
                    worker_count = intent.requested_worker_counts[accepted_count]
                    quota_cost = worker_count * units_per_worker
                    if quota_cost <= remaining_units:
                        candidates.append((intent, worker_count, quota_cost))
                if not candidates:
                    break
                intent, worker_count, quota_cost = min(
                    candidates,
                    key=lambda candidate: (
                        len(accepted[candidate[0].pool.id]) / len(candidate[0].requested_worker_counts),
                        candidate[2],
                        candidate[0].pool.id,
                    ),
                )
                accepted[intent.pool.id].append(worker_count)
                remaining_units -= quota_cost
            for intent in members:
                allowed[intent.pool.id] = tuple(accepted[intent.pool.id])
                accepted_exact_count = min(len(accepted[intent.pool.id]), intent.requested_exact_count)
                blocked_workload_ids.update(
                    intent.demand.virtual_workload_ids[accepted_exact_count : intent.requested_exact_count]
                )
            denied_exact = [intent for intent in members if len(accepted[intent.pool.id]) < intent.requested_exact_count]
            if denied_exact:
                intent = min(
                    denied_exact,
                    key=lambda candidate: (
                        len(accepted[candidate.pool.id]) / len(candidate.requested_worker_counts),
                        candidate.requested_worker_counts[len(accepted[candidate.pool.id])],
                        candidate.pool.id,
                    ),
                )
                accepted_count = len(accepted[intent.pool.id])
                worker_count = intent.requested_worker_counts[accepted_count]
                units_per_worker = intent.pool.quota_units_per_cell
                if units_per_worker is None:
                    raise ValueError("quota group pools must configure quota_units_per_cell")
                quota_shortage = worker_count * units_per_worker - remaining_units
                reclaim_worker_units[intent.pool.id] = math.ceil(quota_shortage / units_per_worker)
                self._donate_across_pools(
                    blocked=intent,
                    accepted_index=accepted_count,
                    quota_shortage=quota_shortage,
                    members=members,
                    denied_pool_ids={candidate.pool.id for candidate in denied_exact},
                    donated=donated_worker_units,
                )
        return QuotaAllocation(
            launch_worker_counts=allowed,
            reclaim_worker_units=reclaim_worker_units,
            blocked_workload_ids=frozenset(blocked_workload_ids),
            donated_worker_units=donated_worker_units,
        )

    def _donate_across_pools(
        self,
        *,
        blocked: PoolScaleIntent,
        accepted_index: int,
        quota_shortage: int,
        members: list[PoolScaleIntent],
        denied_pool_ids: set[str],
        donated: dict[str, int],
    ) -> None:
        """Take quota from peer pools so ``blocked`` can launch the shape it needs.

        Reclaiming a pool's own wrong-shaped cells is already handled by
        ``_finalize``. This is the case that leaves a fleet stuck: the group's
        quota is spent on *other* pools' cells, so nothing the blocked pool owns
        can be given up to fit a larger shape, and the demand waits forever while
        the chips it needs sit in slices too small to use.

        Deliberately narrow:

        * **idle capacity only.** ``_finalize`` protects every cell serving
          demand, so a busy cell could be promised here and never actually
          drained -- the shortage would read as covered while nothing was freed.
          Matching its rule keeps the two honest, and has the better property
          besides: defragmentation never stops running work. Fragmentation is
          idle cells holding quota, which is exactly what this can take.
        * never from a pool that is itself blocked, which would trade one stuck
          demand for another;
        * never below a peer's warm floor, which an operator is paying to hold;
        * only once the demand has waited past the fragmentation threshold;
        * all or nothing -- a partial donation drains capacity and still does
          not fit the shape.
        """

        band = _demand_band(blocked.demand, accepted_index)
        waited = _demand_pending_seconds(blocked.demand, accepted_index)
        if band is None or waited < self._defrag_min_pending_seconds:
            return

        candidates: list[tuple[int, str, str, int]] = []
        for peer in members:
            if peer.pool.id == blocked.pool.id or peer.pool.id in denied_pool_ids:
                continue
            units_per_worker = peer.pool.quota_units_per_cell
            if units_per_worker is None:
                continue
            idle = [
                cell
                for cell in peer.live
                if cell.id not in peer.demand.real_cell_ids
                and cell.autoscale_request_workload_id is None
                and cell.state not in {CellState.DRAINING, CellState.DELETING}
            ]
            spare_units = sum(cell.worker_count for cell in idle) - peer.pool.warm_floor(self._now)
            for cell in sorted(idle, key=lambda candidate: (candidate.worker_count, candidate.id)):
                if spare_units < cell.worker_count:
                    break
                spare_units -= cell.worker_count
                candidates.append((cell.worker_count, cell.id, peer.pool.id, units_per_worker))
        candidates.sort()

        freed_quota_units = 0
        taken: Counter[str] = Counter()
        for worker_count, _cell_id, pool_id, units_per_worker in candidates:
            if freed_quota_units >= quota_shortage:
                break
            taken[pool_id] += worker_count
            freed_quota_units += worker_count * units_per_worker
        if freed_quota_units < quota_shortage:
            return
        for pool_id, worker_units in taken.items():
            donated[pool_id] = max(donated.get(pool_id, 0), worker_units)

    def _finalize(
        self,
        intent: PoolScaleIntent,
        launch_worker_counts: tuple[int, ...],
        quota_reclaim_worker_units: int,
        donated_worker_units: int = 0,
    ) -> PoolScalePlan:
        if launch_worker_counts != intent.requested_worker_counts[: len(launch_worker_counts)]:
            raise ValueError("quota allocation must preserve the requested launch order")
        matching_by_id = {cell.id: cell for cell in intent.matching_supply}
        protected = {cell_id for cell_id in intent.demand.real_cell_ids if cell_id in matching_by_id}
        protected.update(cell.id for cell in intent.matching_supply if cell.autoscale_request_workload_id is not None)
        occupied_units = sum(cell.worker_count for cell in intent.live)
        available_after_launch = max(
            intent.pool.max_cells - occupied_units - sum(launch_worker_counts),
            0,
        )
        reclaiming_units = sum(
            cell.worker_count for cell in intent.live if cell.state in {CellState.DRAINING, CellState.DELETING}
        )
        available_after_reclaim = available_after_launch + reclaiming_units
        launched_counts = Counter(launch_worker_counts)
        unlaunched_exact: list[int] = []
        for worker_count in intent.demanded_virtual_worker_counts:
            if launched_counts[worker_count]:
                launched_counts[worker_count] -= 1
            else:
                unlaunched_exact.append(worker_count)

        blocked_worker_count = (
            unlaunched_exact[0]
            if unlaunched_exact
            else 1
            if intent.target_worker_units
            > sum(cell.worker_count for cell in intent.matching_supply) + sum(launch_worker_counts)
            else None
        )
        forced_drain: set[str] = set()
        capacity_units_to_free = (
            blocked_worker_count - available_after_reclaim
            if blocked_worker_count is not None and blocked_worker_count > available_after_reclaim
            else 0
        )
        quota_units_to_free = max(quota_reclaim_worker_units - reclaiming_units, 0) if unlaunched_exact else 0
        if intent.acquisition_backoff is not None:
            # Room is made for a launch; a backing-off pool is not launching,
            # and evicting idle capacity for it would turn its failures into
            # lost capacity elsewhere.
            capacity_units_to_free = quota_units_to_free = 0
        # A donation is not gated on this pool having unlaunched demand -- the
        # whole point is that the capacity leaves for a peer. Combined with
        # ``max`` rather than a sum: a pool that is both blocked and a donor
        # frees the larger requirement, and the next pass asks again if that was
        # not enough. Summing would over-drain on the strength of two estimates
        # of the same shortage.
        donation_units_to_free = max(donated_worker_units - reclaiming_units, 0)
        units_to_free = max(capacity_units_to_free, quota_units_to_free, donation_units_to_free)
        if units_to_free:
            drainable = sorted(
                (
                    cell
                    for cell in intent.live
                    if cell.id not in protected and cell.state not in {CellState.DRAINING, CellState.DELETING}
                ),
                key=lambda cell: (cell.worker_count, cell.id),
            )
            freed = 0
            for cell in drainable:
                forced_drain.add(cell.id)
                freed += cell.worker_count
                if freed >= units_to_free:
                    break

        protected_units = sum(matching_by_id[cell_id].worker_count for cell_id in protected)
        required_existing_units = max(intent.target_worker_units - sum(launch_worker_counts), 0)
        for cell in sorted(intent.matching_supply, key=lambda candidate: (candidate.worker_count, candidate.id)):
            if protected_units >= required_existing_units:
                break
            if cell.id in protected or cell.id in forced_drain:
                continue
            protected.add(cell.id)
            protected_units += cell.worker_count

        launched_exact_count = min(len(launch_worker_counts), intent.requested_exact_count)
        launch_workload_ids = intent.demand.virtual_workload_ids[:launched_exact_count] + (None,) * (
            len(launch_worker_counts) - launched_exact_count
        )

        return PoolScalePlan(
            pool=intent.pool,
            live=intent.live,
            matching_supply=intent.matching_supply,
            target_worker_units=intent.target_worker_units,
            launch_worker_counts=launch_worker_counts,
            launch_workload_ids=launch_workload_ids,
            protected_cell_ids=frozenset(protected),
            forced_drain_cell_ids=frozenset(forced_drain),
            shrink_since_update=intent.shrink_since_update,
            shrink_since_changed=intent.shrink_since_changed,
            shrink_held=intent.shrink_held,
            acquisition_backoff=intent.acquisition_backoff,
            withheld_worker_counts=intent.withheld_worker_counts,
        )
