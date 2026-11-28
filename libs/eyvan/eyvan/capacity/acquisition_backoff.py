# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The pool-level view of the acquisition backoff each cell's belief already keeps.

Availability belief is stored per cell. ``CapacityBelief.observe`` doubles a
cell's cooldown on every consecutive failure and caps it at the pool's
``max_backoff_s`` -- but that cooldown only gates *observing* the cell. A cell
whose capacity request failed is retired and *replaced*, the replacement starts
from a clean belief. Without a pool-level gate, a repeated configuration error
could therefore create an unbounded replacement loop on the scheduling loop
shared by every fleet.

This module reads the streak across incarnations: the pool's cells that failed
without ever delivering capacity, newer than the last one that did. The curve
``CapacityBelief.observe`` applies to one cell's consecutive failures is applied
to the streak's length, with the same policy (the cell's copy of its pool's
``AvailabilityPolicy``) and the same cap. The first failure is not waited on --
one refusal is how Spot capacity is normally won -- and every failed incarnation
after it doubles the wait, never beyond ``max_backoff_s``. One delivered cell
ends the streak, which is the reset. Nothing here stops a pool for good: once
``next_attempt_at`` passes, the pool is asked again.

Pure and provider-neutral: it reads pool, cell and workload records only.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from eyvan.capacity.drivers.base import CapacityProviderFailureCode, backend_capabilities, provision_outcome
from eyvan.records import CapacityPoolRecord, CellRecord, WorkloadRecord, candidate_pool_templates
from eyvan.secrets import redact_secret_material

#: Every workload ``scheduling_reason`` the controller writes for a backing-off
#: pool starts with this, so the scheduler's generic "no cell satisfies" pass
#: leaves it alone, the capacity pass clears exactly what it wrote, and the
#: progress route can tell a repeated failure from a first request.
ACQUISITION_BACKOFF_SCHEDULING_REASON_PREFIX = "capacity acquisition is backing off"

#: How long a failure stays part of a streak beyond the pool's backoff ceiling,
#: and how long an expired streak still describes the pool for reporting. The
#: same hour the planner's acquisition bound remembers deleted failures for.
ACQUISITION_BACKOFF_MEMORY_S = 3600.0

#: Ceiling for pools of hardware that already exists (attached TPUs, manual
#: hosts). Re-reading an existing machine costs nothing and buys nothing by
#: waiting, while the usual causes -- a credential or firewall fixed on the
#: host -- should be picked up within minutes, not after an hour.
EXISTING_HARDWARE_MAX_BACKOFF_S = 300.0

#: Provider prose shown to an operator, after redaction. Long enough for a
#: gcloud remedy line; short enough for a log line and a status row.
_ERROR_LIMIT = 500


@dataclass(frozen=True, slots=True)
class PoolAcquisitionBackoff:
    """A pool's current run of failed capacity acquisitions.

    Attributes:
        pool_id: The pool the streak belongs to.
        failed_attempts: Incarnations that observed a failure without ever
            delivering capacity, since the last one that did.
        last_failure_at: When the newest failure in the streak was observed.
        next_attempt_at: When the pool may be asked for new capacity again.
            Equal to ``last_failure_at`` after a single failure, which is not
            waited on.
        backoff_s: The pool-level wait the streak's length earned, after the cap.
        last_error: The newest provider diagnostic in the streak, redacted.
        failure_code: The provider's classification of that diagnostic.
        last_failed_cell_id: The newest failed incarnation.
    """

    pool_id: str
    failed_attempts: int
    last_failure_at: datetime
    next_attempt_at: datetime
    backoff_s: float
    last_error: str | None
    failure_code: CapacityProviderFailureCode | None
    last_failed_cell_id: str

    def active(self, now: datetime) -> bool:
        """Return whether the pool must not be asked for new capacity at ``now``."""

        return self.next_attempt_at > now

    def recent(self, now: datetime, *, memory_s: float = ACQUISITION_BACKOFF_MEMORY_S) -> bool:
        """Return whether this streak still describes the pool, rather than its history."""

        return self.active(now) or (now - self.last_failure_at).total_seconds() <= memory_s


def _aware(moment: datetime) -> datetime:
    # Eyvan writes aware UTC; a naive persisted value is read as UTC too.
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _outcome_at(cell: CellRecord) -> datetime:
    return _aware(cell.availability.last_observed_at or cell.created_at)


def _horizon(cell: CellRecord) -> timedelta:
    return timedelta(seconds=cell.availability_policy.max_backoff_s + ACQUISITION_BACKOFF_MEMORY_S)


def _streak_backoff_s(newest: CellRecord, failed_attempts: int) -> float:
    """Apply ``CapacityBelief.observe``'s curve to a streak, waiting from its second failure."""

    if failed_attempts < 2:
        return 0.0
    policy = newest.availability_policy
    outcome = provision_outcome(newest.provider_failure_code, newest.provider_phase)
    base = policy.backoff_base_s.get(outcome.value, policy.default_backoff_s)
    ceiling = float(policy.max_backoff_s)
    if backend_capabilities(newest.capacity_backend).existing_hardware:
        ceiling = min(ceiling, EXISTING_HARDWARE_MAX_BACKOFF_S)
    return float(min(ceiling, base * (2 ** min(failed_attempts - 2, 30))))


def pool_acquisition_backoffs(
    pools: tp.Iterable[CapacityPoolRecord],
    cells: tp.Iterable[CellRecord],
    now: datetime,
) -> dict[str, PoolAcquisitionBackoff]:
    """Return every pool's current failed-acquisition streak, keyed by pool id.

    Tombstones are read on purpose: a failed cell is deleted and replaced, so the
    evidence of a replacement loop is exactly the deleted cells. A pool with no
    streak -- never failed, or delivered since -- is absent.

    A cell counts as failed only once a failure was *observed* on it
    (``last_observed_at`` set and ``consecutive_failures`` positive); a cell not
    yet answered is in flight, not failed, and unknown stays unknown.

    Deliberately not reset by a template change: a fleet re-applied with a new
    template on every pass would otherwise re-arm the very loop this bounds, and
    external configuration failures such as IAM grants are not in the template.
    The cap bounds how long a fixed pool waits.

    A streak is a chain: it ends at a gap longer than the pool's ceiling plus
    ``ACQUISITION_BACKOFF_MEMORY_S`` between two failures, and a pool whose
    newest failure is older than that has no streak at all. So a pool that
    failed during last week's outage starts from the bottom of the curve, while
    a pool failing once an hour at the cap stays at the cap.
    """

    pool_ids = {pool.id for pool in pools}
    delivered_at: dict[str, datetime] = {}
    failing: dict[str, list[CellRecord]] = {}
    for cell in cells:
        pool_id = cell.pool_id
        if pool_id is None or pool_id not in pool_ids:
            continue
        belief = cell.availability
        moment = _outcome_at(cell)
        if belief.provision_latencies_s:
            # Appended only on the SUCCESS transition into READY: the durable
            # record that this incarnation once delivered capacity.
            if pool_id not in delivered_at or moment > delivered_at[pool_id]:
                delivered_at[pool_id] = moment
        elif belief.consecutive_failures > 0 and belief.last_observed_at is not None:
            failing.setdefault(pool_id, []).append(cell)

    backoffs: dict[str, PoolAcquisitionBackoff] = {}
    for pool_id, candidates in failing.items():
        boundary = delivered_at.get(pool_id)
        ordered = sorted(
            (cell for cell in candidates if boundary is None or _outcome_at(cell) > boundary),
            key=lambda cell: (_outcome_at(cell), cell.id),
            reverse=True,
        )
        if not ordered:
            continue
        newest = ordered[0]
        if now - _outcome_at(newest) > _horizon(newest):
            # Last week's outage, not today's pool.
            continue
        streak = [newest]
        for cell in ordered[1:]:
            # A long quiet gap ends a streak just as a success does.
            if _outcome_at(streak[-1]) - _outcome_at(cell) > _horizon(newest):
                break
            streak.append(cell)
        backoff_s = _streak_backoff_s(newest, len(streak))
        source = next((cell for cell in streak if cell.unavailable_detail), None)
        error = (
            redact_secret_material(source.unavailable_detail)[:_ERROR_LIMIT]
            if source is not None and source.unavailable_detail
            else None
        )
        backoffs[pool_id] = PoolAcquisitionBackoff(
            pool_id=pool_id,
            failed_attempts=len(streak),
            last_failure_at=_outcome_at(newest),
            next_attempt_at=_outcome_at(newest) + timedelta(seconds=backoff_s),
            backoff_s=backoff_s,
            last_error=error or None,
            failure_code=None if source is None else source.provider_failure_code,
            last_failed_cell_id=newest.id,
        )
    return backoffs


def workload_acquisition_backoff(
    workload: WorkloadRecord,
    backoffs: tp.Mapping[str, PoolAcquisitionBackoff],
    now: datetime,
) -> PoolAcquisitionBackoff | None:
    """Return the streak that explains why this workload's capacity keeps failing.

    Only when *every* pool the workload accepts has a recent streak: one healthy
    alternate is still being asked, and naming another zone's failure then would
    describe a problem the workload does not have. The streak that ends soonest
    is the one reported, because that is when something will next happen.
    """

    pool_ids = tuple(candidate_pool_templates(workload.spec.required_labels))
    if not pool_ids:
        return None
    streaks = [backoffs.get(pool_id) for pool_id in pool_ids]
    # One refusal is ordinary Spot behaviour and is retried at once; it is a
    # repeated refusal that the reader has to be told about.
    present = [streak for streak in streaks if streak is not None and streak.failed_attempts >= 2 and streak.recent(now)]
    if len(present) != len(streaks):
        return None
    return min(present, key=lambda streak: (streak.next_attempt_at, streak.pool_id))


def acquisition_backoff_reason(backoff: PoolAcquisitionBackoff, now: datetime) -> str:
    """Return the operator-facing sentence for a pool whose acquisitions keep failing."""

    attempts = f"{backoff.failed_attempts} failed capacity request" + ("" if backoff.failed_attempts == 1 else "s")
    if backoff.active(now):
        when = f"next attempt after {backoff.next_attempt_at.astimezone(UTC):%Y-%m-%d %H:%M:%S} UTC"
    else:
        when = "retrying now"
    cause = f": {backoff.last_error}" if backoff.last_error else ""
    return (
        f"{ACQUISITION_BACKOFF_SCHEDULING_REASON_PREFIX} in pool {backoff.pool_id} "
        f"after {attempts} in a row; {when}{cause}"
    )


def is_acquisition_backoff_reason(reason: str | None) -> bool:
    """Return whether a ``scheduling_reason`` was written by the acquisition backoff."""

    return reason is not None and reason.startswith(ACQUISITION_BACKOFF_SCHEDULING_REASON_PREFIX)


__all__ = (
    "ACQUISITION_BACKOFF_MEMORY_S",
    "ACQUISITION_BACKOFF_SCHEDULING_REASON_PREFIX",
    "EXISTING_HARDWARE_MAX_BACKOFF_S",
    "PoolAcquisitionBackoff",
    "acquisition_backoff_reason",
    "is_acquisition_backoff_reason",
    "pool_acquisition_backoffs",
    "workload_acquisition_backoff",
)
