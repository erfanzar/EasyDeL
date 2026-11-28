# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure team-governance decisions: fleet rules, approval holds and borrow/reclaim.

Nothing here reads storage or talks to a provider. The service layer collects
the facts (who submitted, which pools a spec names, who is running what) and
asks these functions for a verdict, so every rule is deterministic, auditable
and testable without a plane. In particular, *who may take capacity from whom*
is decided here by arithmetic, never by a person's or a model's judgement.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction

from eyvan.records import (
    POOL_ALTERNATES_LABEL,
    POOL_LABEL,
    PriorityBand,
    Role,
    RoleGrantRecord,
    TeamTier,
    UserRecord,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)

#: The request-detail key linking a system-filed approval request to the
#: workload it holds. Only admission sets it; user-filed requests may not.
HELD_WORKLOAD_KEY: tp.Final = "held_workload_id"

#: Admission verdicts.
ADMIT: tp.Final = "admit"
HOLD: tp.Final = "hold"
REFUSE: tp.Final = "refuse"

#: Submission operations that create one top-level job a person can approve as
#: a unit. Anything else (batch tasks, DAG nodes, group members, in-job
#: children, coordinators) is refused rather than held over the threshold:
#: those are released by their own machinery, which would bypass a hold.
HOLDABLE_OPERATIONS = frozenset({"workload.submit", "workload.submit_pool", "workload.submit_pool_callable"})

_BAND_RANK: Mapping[str, int] = {band.value: index for index, band in enumerate(PriorityBand)}


@dataclass(frozen=True, slots=True)
class QuotaUnits:
    """Reserved compute dimensions. CPU sums use exact wire-decimal quantities."""

    cpu: Fraction = Fraction(0)
    gpu: int = 0
    tpu: int = 0

    def __add__(self, other: QuotaUnits) -> QuotaUnits:
        return QuotaUnits(self.cpu + other.cpu, self.gpu + other.gpu, self.tpu + other.tpu)


def requested_units(spec: WorkloadDefinition) -> QuotaUnits:
    """Scale the actual reservation, not a kind inferred from an accelerator name."""

    count = spec.gang.replicas * spec.cell_group_count
    if spec.worker_resource_scope == "per_worker":
        count *= spec.worker_count
    resources = spec.gang.per_replica
    return QuotaUnits(Fraction(str(resources.cpu)) * count, resources.gpu * count, resources.tpu * count)


def requested_tpu_units(spec: WorkloadDefinition) -> int:
    return requested_units(spec).tpu


def has_approval_gate(workload: WorkloadRecord) -> bool:
    """Only a top-level non-coordinator uses the schedulable gate for approval."""

    return (
        not workload.schedulable
        and workload.parent_workload_id is None
        and not workload.depends_on
        and workload.coordination_kind is None
    )


def targeted_pool_ids(spec: WorkloadDefinition) -> tuple[str, ...]:
    """Return the pool ids a spec is pinned to (primary first), or ``()`` for a generic spec."""

    primary = spec.required_labels.get(POOL_LABEL) or ""
    alternates = spec.required_labels.get(POOL_ALTERNATES_LABEL) or ""
    ordered = [primary, *(value.strip() for value in alternates.split(","))]
    return tuple(dict.fromkeys(value for value in ordered if value))


class RuleUnchanged(Enum):
    """Marker for a rule a partial update leaves exactly as it is stored."""

    UNCHANGED = "unchanged"


#: Passed for a rule the caller did not mention; ``None`` still clears a rule.
RULE_UNCHANGED: tp.Final = RuleUnchanged.UNCHANGED


def enabled_fleet_admins(users: Iterable[UserRecord], grants: Iterable[RoleGrantRecord], moment: datetime) -> set[str]:
    """Return the enabled accounts holding an active ``fleet_admin`` grant."""

    enabled = {user.subject for user in users if not user.disabled}
    return {
        record.subject
        for record in grants
        if record.subject in enabled and record.grant.role is Role.FLEET_ADMIN and record.active_at(moment)
    }


#: Workload states that no longer hold or wait for capacity.
_FINISHED_STATES = frozenset({WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.CANCELLED})

#: Refusal shown to a plain ``user`` who submits a callable before a cap exists.
CALLABLE_NEEDS_CAP = "ask a fleet admin to set a positive quota for each resource your callable reserves"


def team_tier(trusted: bool, grants: Iterable[RoleGrantRecord], moment: datetime) -> TeamTier:
    """Derive an account's tier: an active ``fleet_admin`` grant, else its ``trusted`` flag, else ``user``."""

    if any(record.grant.role is Role.FLEET_ADMIN and record.active_at(moment) for record in grants):
        return TeamTier.ADMIN
    return TeamTier.TRUSTED if trusted else TeamTier.USER


def units_in_use(workloads: Iterable[WorkloadRecord], subject: str, *, exclude_id: str | None = None) -> QuotaUnits:
    """Return the units ``subject`` has in use or queued, across every org and project.

    Every unfinished workload counts -- pending (held or not), admitted, leased,
    dispatching, running, checkpointing and cancelling -- except coordinators
    (batch, DAG, group and service parents), which hold no capacity themselves:
    their children are counted instead, so a batch is not counted twice.
    """

    live = (
        requested_units(workload.spec)
        for workload in workloads
        if workload.created_by == subject
        and workload.state not in _FINISHED_STATES
        and workload.coordination_kind is None
        and workload.id != exclude_id
    )
    return sum(live, start=QuotaUnits())


def tpu_units_in_use(workloads: Iterable[WorkloadRecord], subject: str, *, exclude_id: str | None = None) -> int:
    return units_in_use(workloads, subject, exclude_id=exclude_id).tpu


def tpu_cap_refusal(max_tpu_units: int | None, in_use: int, requested: int) -> str | None:
    """Return why a request breaks a hard TPU cap, or ``None`` when it fits (or there is no cap).

    A request for no TPU units never breaks it: it cannot raise TPU usage, even
    for someone already over a cap an admin has since lowered.
    """

    return UserRules(max_tpu_units=max_tpu_units).cap_refusal(QuotaUnits(tpu=in_use), QuotaUnits(tpu=requested))


@dataclass(frozen=True, slots=True)
class UserRules:
    """The governance rules one account carries; ``None`` means no rule."""

    allowed_fleets: tuple[str, ...] | None = None
    approval_tpu_threshold: int | None = None
    guaranteed_tpu_units: int | None = None
    max_tpu_units: int | None = None
    max_gpu_units: int | None = None
    max_cpu_units: float | None = None
    can_borrow: bool | None = None

    @property
    def caps(self) -> Mapping[str, int | Fraction | None]:
        return {
            "cpu": Fraction(str(self.max_cpu_units)) if self.max_cpu_units is not None else None,
            "gpu": self.max_gpu_units,
            "tpu": self.max_tpu_units,
        }

    @property
    def empty(self) -> bool:
        return (
            self.allowed_fleets is None
            and self.approval_tpu_threshold is None
            and self.guaranteed_tpu_units is None
            and self.can_borrow is None
            and all(cap is None for cap in self.caps.values())
        )

    def approval_reason(self, tpu_units: int) -> str | None:
        threshold = self.approval_tpu_threshold
        if threshold is None or tpu_units <= threshold:
            return None
        return f"asks for {tpu_units} TPU units, above your approval threshold of {threshold}"

    def fleet_refusal(self, fleets: Iterable[str], *, pinned: bool) -> str | None:
        if self.allowed_fleets is None:
            return None
        allowed = set(self.allowed_fleets)
        permitted = ", ".join(sorted(allowed)) or "(none)"
        if not pinned:
            return f"your account may only submit to fleet(s) {permitted}; name one"
        targets = set(fleets)
        outside = sorted(targets - allowed)
        if outside or not targets:
            return f"your account may not submit to fleet {', '.join(outside or ['(unknown)'])}; allowed: {permitted}"
        return None

    def callable_refusal(self, requested: QuotaUnits | None) -> str | None:
        """An upload (no spec yet) needs some allowance; execution needs all it reserves."""

        caps = self.caps
        if requested is None:
            return None if any(cap is not None and cap > 0 for cap in caps.values()) else CALLABLE_NEEDS_CAP
        needed = [name for name in caps if getattr(requested, name) > 0]
        if not needed:
            return "callable has no resolved CPU, GPU or TPU reservation"
        for name in needed:
            cap = caps[name]
            if cap is None or cap <= 0:
                return f"{name.upper()} quota required: {CALLABLE_NEEDS_CAP}"
        return None

    def cap_refusal(self, in_use: QuotaUnits, requested: QuotaUnits) -> str | None:
        for name, cap in self.caps.items():
            used, extra = getattr(in_use, name), getattr(requested, name)
            if cap is not None and extra > 0 and used + extra > cap:
                unit = name.upper()
                return f"{unit} quota: you have {used} of {cap} {unit} units in use or queued; this job asks for {extra}"
        return None


def effective_can_borrow(rules: UserRules, tier: TeamTier) -> bool:
    """Return whether an account may borrow idle TPU capacity above its guarantee.

    * Explicit ``rules.can_borrow`` (``True`` or ``False``) always wins.
    * When ``None``: ``admin`` and ``trusted`` tiers may borrow idle capacity.
    * A plain ``user`` with an explicit ``guaranteed_tpu_units`` (including ``0``)
      stays within their own guarantee unless granted ``can_borrow=True`` or an
      explicit ``max_tpu_units > guaranteed_tpu_units``.
    * A plain ``user`` with no ``guaranteed_tpu_units`` (``None``) keeps the
      existing unrestricted borrowing behaviour (bounded only by ``max_tpu_units``).
    """

    if rules.can_borrow is not None:
        return rules.can_borrow
    if tier in {TeamTier.ADMIN, TeamTier.TRUSTED}:
        return True
    if rules.guaranteed_tpu_units is not None and rules.guaranteed_tpu_units >= 0:
        return rules.max_tpu_units is not None and rules.max_tpu_units > rules.guaranteed_tpu_units
    return True


def borrow_cap_refusal(
    rules: UserRules,
    tier: TeamTier,
    in_use: QuotaUnits,
    requested: QuotaUnits,
) -> str | None:
    """Return why a TPU request exceeds non-borrowing guaranteed capacity, or ``None``."""

    if requested.tpu <= 0 or effective_can_borrow(rules, tier):
        return None
    guarantee = rules.guaranteed_tpu_units or 0
    if in_use.tpu + requested.tpu > guarantee:
        return (
            f"TPU borrowing disabled: you have {in_use.tpu} of {guarantee} guaranteed TPU units "
            f"in use or queued; this job asks for {requested.tpu}"
        )
    return None


@dataclass(frozen=True, slots=True)
class AdmissionVerdict:
    """What admission does with one submission under one person's rules."""

    action: tp.Literal["admit", "hold", "refuse"]
    reason: str = ""
    tpu_units: int = 0
    refusal: tp.Literal["fleet", "approval", ""] = ""


def admission_verdict(
    rules: UserRules,
    *,
    tpu_units: int,
    fleets: Iterable[str],
    pinned: bool,
    operation: str,
    top_level: bool,
    exempt_from_approval: bool = False,
) -> AdmissionVerdict:
    """Decide admit / hold / refuse for one submission.

    Hard caps need live owner-wide usage and are decided by ``UserRules.cap_refusal``
    inside the submitting transaction.

    Args:
        rules: the submitter's rules.
        tpu_units: what the submission holds while running (``requested_tpu_units``).
        fleets: fleet names of every pool the spec may land on.
        pinned: whether the spec names pools at all; a generic spec can land anywhere.
        operation: the submission operation name.
        top_level: whether this is a schedulable job with no parent.
        exempt_from_approval: the ``admin`` and ``trusted`` tiers are never held
            for approval, whatever their threshold says.
    """

    fleet_refusal = rules.fleet_refusal(fleets, pinned=pinned)
    if fleet_refusal is not None:
        return AdmissionVerdict(REFUSE, fleet_refusal, tpu_units, "fleet")
    reason = rules.approval_reason(tpu_units)
    if exempt_from_approval or reason is None:
        return AdmissionVerdict(ADMIT, "", tpu_units)
    if top_level and operation in HOLDABLE_OPERATIONS:
        return AdmissionVerdict(HOLD, f"{reason}; waiting for a fleet admin's approval", tpu_units)
    return AdmissionVerdict(
        REFUSE,
        f"this submission {reason}; submit it as a single job to request approval, "
        "or ask a fleet admin to raise your threshold",
        tpu_units,
        "approval",
    )


# ---------------------------------------------------------------- borrow/reclaim


@dataclass(frozen=True, slots=True)
class RunningJob:
    """One running workload as the reclaim rule sees it."""

    workload_id: str
    subject: str
    tpu_units: int
    priority_band: str
    started_order: int
    """Higher means started later; ties broken by ``workload_id``."""
    reclaimable: bool = True
    """Counts toward usage either way; only a running job can actually be taken."""
    pool_ids: tuple[str, ...] | None = None
    """Capacity pool IDs currently backing this running job, if known."""
    accelerator_type: str | None = None
    """Accelerator shape currently backing this running job, if known."""


@dataclass(frozen=True, slots=True)
class ReclaimPlan:
    """Which running jobs a reclaim takes, and why it may or may not."""

    allowed: bool
    reason: str
    victims: tuple[str, ...] = ()
    freed_tpu_units: int = 0
    requester_usage: int = 0
    requester_guarantee: int | None = None
    borrowed_by: Mapping[str, int] | None = None


def usage_by_subject(jobs: Iterable[RunningJob]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for job in jobs:
        usage[job.subject] = usage.get(job.subject, 0) + job.tpu_units
    return usage


def _job_matches_target(
    job: RunningJob,
    *,
    compatible_pools: frozenset[str] | None,
    required_accel: str | None,
) -> bool:
    """Return whether reclaiming ``job`` frees capacity compatible with the requester's shape."""

    if compatible_pools is not None:
        job_pools = job.pool_ids or ()
        return any(pool_id in compatible_pools for pool_id in job_pools)
    if required_accel is not None:
        job_accel = (job.accelerator_type or "").strip().lower()
        return job_accel == required_accel
    return True


def plan_reclaim(
    requester: str,
    needed_tpu_units: int,
    jobs: Sequence[RunningJob],
    guarantees: Mapping[str, int],
    *,
    compatible_pool_ids: Iterable[str] | None = None,
    required_accelerator_type: str | None = None,
) -> ReclaimPlan:
    """Return the deterministic reclaim plan for ``requester`` asking for ``needed_tpu_units``.

    The rule:

    * A person's *guarantee* is their ``guaranteed_tpu_units``; capacity they
      use above it is **borrowed**. Someone with no guarantee is guaranteed 0,
      so everything they run is borrowed.
    * A requester may reclaim only while their usage plus the request stays
      within their own guarantee. Reclaiming never lets anyone exceed theirs.
    * Only borrowed capacity is taken, and never from the requester. A
      borrower is never pushed below their own guarantee.
    * When ``compatible_pool_ids`` or ``required_accelerator_type`` is supplied,
      only borrowed victims running on compatible pools/accelerators are
      selected so reclaiming capacity can actually place the requester's shape.
    * Victims come from the most-over-guarantee borrower first; within a
      borrower, the lowest priority band, then the most recently started job,
      then the workload id. Same inputs, same plan.
    * The plan must free at least ``needed_tpu_units``; otherwise it is refused
      and takes nothing, so a partial reclaim never strands half a request.
    """

    if needed_tpu_units <= 0:
        return ReclaimPlan(False, "ask for a positive number of TPU units")
    usage = usage_by_subject(jobs)
    own_usage = usage.get(requester, 0)
    guarantee = guarantees.get(requester)
    if guarantee is None or guarantee <= 0:
        return ReclaimPlan(
            False,
            "you have no guaranteed TPU units, so there is nothing to reclaim for you",
            (),
            0,
            own_usage,
            guarantee,
        )
    if own_usage + needed_tpu_units > guarantee:
        return ReclaimPlan(
            False,
            f"you use {own_usage} of your {guarantee} guaranteed TPU units; {needed_tpu_units} more would exceed it",
            (),
            0,
            own_usage,
            guarantee,
        )
    compat_pools: frozenset[str] | None = None
    if compatible_pool_ids is not None:
        compat_pools = frozenset(pool_id.strip() for pool_id in compatible_pool_ids if pool_id and pool_id.strip())
    req_accel = (
        required_accelerator_type.strip().lower()
        if isinstance(required_accelerator_type, str) and required_accelerator_type.strip()
        else None
    )
    borrowed = {
        subject: used - guarantees.get(subject, 0)
        for subject, used in usage.items()
        if subject != requester and used > guarantees.get(subject, 0)
    }
    order = sorted(borrowed, key=lambda subject: (-borrowed[subject], subject))
    remaining = dict(borrowed)
    victims: list[str] = []
    freed = 0
    for subject in order:
        candidates = sorted(
            (
                job
                for job in jobs
                if job.subject == subject
                and job.tpu_units > 0
                and job.reclaimable
                and _job_matches_target(job, compatible_pools=compat_pools, required_accel=req_accel)
            ),
            key=lambda job: (_BAND_RANK.get(job.priority_band, 0), -job.started_order, job.workload_id),
        )
        for job in candidates:
            if freed >= needed_tpu_units:
                break
            if job.tpu_units > remaining[subject]:
                # Taking it would push the borrower below their own guarantee.
                continue
            victims.append(job.workload_id)
            remaining[subject] -= job.tpu_units
            freed += job.tpu_units
        if freed >= needed_tpu_units:
            break
    if freed < needed_tpu_units:
        scope_note = " on compatible pools" if compat_pools is not None or req_accel is not None else ""
        return ReclaimPlan(
            False,
            f"only {freed} borrowed TPU units{scope_note} can be reclaimed without pushing anyone below "
            f"their guarantee; {needed_tpu_units} were asked for",
            (),
            0,
            own_usage,
            guarantee,
            borrowed,
        )
    return ReclaimPlan(True, "", tuple(victims), freed, own_usage, guarantee, borrowed)


__all__ = (
    "ADMIT",
    "CALLABLE_NEEDS_CAP",
    "HELD_WORKLOAD_KEY",
    "HOLD",
    "HOLDABLE_OPERATIONS",
    "REFUSE",
    "RULE_UNCHANGED",
    "AdmissionVerdict",
    "QuotaUnits",
    "ReclaimPlan",
    "RuleUnchanged",
    "RunningJob",
    "UserRules",
    "admission_verdict",
    "borrow_cap_refusal",
    "effective_can_borrow",
    "enabled_fleet_admins",
    "has_approval_gate",
    "plan_reclaim",
    "requested_tpu_units",
    "requested_units",
    "targeted_pool_ids",
    "team_tier",
    "tpu_cap_refusal",
    "tpu_units_in_use",
    "units_in_use",
    "usage_by_subject",
)
