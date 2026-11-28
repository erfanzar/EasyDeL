# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Federation: many control planes, one tenant, one budget.

Running one Eyvan per region already works by construction, and for scheduling
that is the right answer -- placement is regional, so a plane that only knows its
own cells is not missing anything it could act on.

**Budgets are where it breaks, and the break is silent.** ``BudgetRecord.spent``
is a single durable counter in one database. A tenant with projects on three
planes therefore has three independent budgets, each enforcing its own limit and
each reporting full compliance while the tenant spends three times what its
operator set. Nothing surfaces that today: every plane's answer is locally true.

**Grants, not consensus.** The fix is not a distributed transaction on every
admission -- that would put a cross-plane round trip on the scheduler's hot path
and make one plane's outage everyone's outage. Instead the tenant's *home* plane
owns the total limit and issues grants: slices of that limit, each addressed to
one peer, each with an expiry. A peer admits work only up to its outstanding
grants. Since the home plane refuses to issue grants summing above the limit,
total spend across every plane is bounded by construction and no plane has to ask
anyone anything at admission time.

**The failure mode is under-use, and that is the correct direction to err.** A
peer holding an unused grant leaves headroom idle; a peer whose grant expires
stops admitting. Both are visible and recoverable. The alternative error --
over-spend discovered at the invoice -- is neither.
"""

from __future__ import annotations

import math
import typing as tp
from datetime import datetime
from decimal import Decimal, localcontext

from pydantic import Field, model_validator

from eyvan._internal.compat import StrEnum
from eyvan.records import CanonicalResourceLimits, EyvanModel, ResourceLimits, resource_quantities, utcnow


class GrantState(StrEnum):
    """Lifecycle of one budget or quota grant."""

    ACTIVE = "active"
    #: Withdrawn by the issuing plane. Spend already committed against it is not
    #: clawed back -- the money is gone -- so revocation stops *new* admission and
    #: says nothing about what already ran.
    REVOKED = "revoked"
    #: Past its expiry. Distinct from REVOKED because it needs no issuer action,
    #: which is what makes an unreachable home plane fail closed rather than
    #: leaving a peer spending forever.
    EXPIRED = "expired"


class PlanePeerRecord(EyvanModel):
    """Another control plane this one knows about.

    Deliberately not a cluster membership protocol. A peer is a URL an operator
    wrote down, because the alternative -- planes discovering each other -- means
    a joining plane can claim a tenant's budget by asserting it is the home, and
    there is no authority in this design that could refuse.
    """

    plane_id: str = Field(min_length=1, max_length=128)
    endpoint: str = Field(min_length=1, max_length=512)
    #: Free-text, for an operator reading a list of peers.
    description: str = Field(default="", max_length=512)
    enabled: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_peer(self) -> PlanePeerRecord:
        if not self.plane_id.strip():
            raise ValueError("plane id must not be blank")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("plane endpoint must be an http(s) URL")
        return self


class BudgetGrantRecord(EyvanModel):
    """One slice of a tenant's budget, issued by its home plane to a peer.

    The grant is the peer's *entire* authority to spend for this tenant. It has an
    expiry with no default of "never", because a grant that cannot lapse turns a
    home plane that goes away permanently into an unbounded spend on every peer
    that ever held one.
    """

    id: str = Field(min_length=1, max_length=96)
    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    #: The plane that owns the tenant's total limit and issued this slice.
    home_plane_id: str = Field(min_length=1, max_length=128)
    #: The plane authorized to spend it.
    grantee_plane_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    state: GrantState = GrantState.ACTIVE
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    #: What the grantee reported spending against it, as of its last report. A
    #: *report*, never an authority: the grantee's own budget is the enforcement
    #: point, and this is how the home plane sees the fleet-wide total.
    reported_spend: float = Field(default=0.0, ge=0)
    reported_at: datetime | None = None

    @model_validator(mode="after")
    def validate_grant(self) -> BudgetGrantRecord:
        if self.currency != "USD":
            raise ValueError("Eyvan v0 budgets support only USD")
        if self.home_plane_id == self.grantee_plane_id:
            raise ValueError("a plane does not grant budget to itself; it holds the limit directly")
        if self.expires_at <= self.issued_at:
            raise ValueError("a budget grant must expire after it is issued")
        return self

    def active_at(self, moment: datetime) -> bool:
        """Return whether this grant authorizes spend at ``moment``."""

        return self.state is GrantState.ACTIVE and moment < self.expires_at


class QuotaGrantRecord(EyvanModel):
    """One slice of a tenant's *quota*, issued by its home plane to a peer.

    The same hole as budgets, in a different currency. ``QuotaRecord.limits`` is
    one durable vector in one database, so a tenant with projects on three planes
    could run three times its TPU ceiling with every plane locally compliant.
    Money and chips are separate limits and a fleet can hold one while breaching
    the other -- a tenant granted a small budget on cheap Spot capacity can still
    exhaust an accelerator pool everyone else is queued for.

    Deliberately a mirror of :class:`BudgetGrantRecord` rather than a
    generalization of it. The two differ in exactly one place -- a scalar becomes
    a vector, and headroom therefore has to hold in *every* dimension at once --
    and collapsing them into one generic grant would put that difference behind a
    type parameter where nobody reviewing an admission decision would see it.
    """

    id: str = Field(min_length=1, max_length=96)
    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    home_plane_id: str = Field(min_length=1, max_length=128)
    grantee_plane_id: str = Field(min_length=1, max_length=128)
    #: The slice itself. Omitted dimensions are zero, not unlimited: a grant that
    #: silently authorized every resource it forgot to mention would be the
    #: opposite of a ceiling.
    limits: CanonicalResourceLimits
    state: GrantState = GrantState.ACTIVE
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime

    @model_validator(mode="after")
    def validate_grant(self) -> QuotaGrantRecord:
        if self.home_plane_id == self.grantee_plane_id:
            raise ValueError("a plane does not grant quota to itself; it holds the limit directly")
        if self.expires_at <= self.issued_at:
            raise ValueError("a quota grant must expire after it is issued")
        if not any(quantity > 0 for quantity in resource_quantities(self.limits).values()):
            raise ValueError("a quota grant must carry a positive quantity in at least one dimension")
        return self

    def active_at(self, moment: datetime) -> bool:
        """Return whether this grant authorizes capacity at ``moment``."""

        return self.state is GrantState.ACTIVE and moment < self.expires_at


# Retain the full exponent span of finite wire quantities, including canonical
# worker/replica products, rather than erasing tiny rights at Decimal's default
# 28-digit precision. This matches the retained monetary authority precision.
_QUOTA_PRECISION = 2048


def _outstanding_quota_exact(grants: tp.Sequence[QuotaGrantRecord], *, now: datetime) -> dict[str, Decimal]:
    """Return exact retained peer capacity rights, including retired grants."""

    totals: dict[str, Decimal] = {}
    with localcontext(prec=_QUOTA_PRECISION):
        for grant in grants:
            for name, quantity in resource_quantities(grant.limits).items():
                totals[name] = totals.get(name, Decimal(0)) + Decimal(str(quantity))
    return totals


def outstanding_quota(grants: tp.Sequence[QuotaGrantRecord], *, now: datetime) -> dict[str, float]:
    """Return retained peer capacity rights, not merely new-admission rights.

    Expiry/revocation cannot prove that a peer stopped its existing work. Without
    a reliable release protocol, retired slices remain reserved on the home.
    ``now`` remains accepted for compatibility with existing callers. Project
    totals upward when float conversion would hide a retained positive right;
    authoritative accounting uses the exact private helper.
    """

    projected: dict[str, float] = {}
    for name, exact in _outstanding_quota_exact(grants, now=now).items():
        quantity = float(exact)
        if Decimal(str(quantity)) < exact:
            quantity = math.nextafter(quantity, math.inf)
        projected[name] = quantity
    return projected


def _quota_headroom_exact(
    limits: ResourceLimits,
    grants: tp.Sequence[QuotaGrantRecord],
    *,
    now: datetime,
    local_usage: tp.Mapping[str, Decimal] | None = None,
    replacing: str | None = None,
) -> dict[str, Decimal]:
    """Partition exact capacity between retained peer rights and local usage.

    Retirement/replacement never refunds rights without reliable release proof.
    Callers supply canonical active local usage under the admission lock.
    """

    committed = _outstanding_quota_exact(grants, now=now)
    usage = local_usage or {}
    with localcontext(prec=_QUOTA_PRECISION):
        return {
            name: max(Decimal(0), Decimal(str(quantity)) - committed.get(name, Decimal(0)) - usage.get(name, Decimal(0)))
            for name, quantity in resource_quantities(limits).items()
        }


def quota_headroom(
    limits: ResourceLimits,
    grants: tp.Sequence[QuotaGrantRecord],
    *,
    now: datetime,
    replacing: str | None = None,
) -> dict[str, float]:
    """Return how much of each dimension the home plane may still grant away.

    Callers must additionally deduct local active usage under the admission
    lock. Replacement cannot exclude old rights: retirement is not evidence that
    a peer released its capacity. ``replacing`` is accepted for compatibility,
    but deliberately does not refund the old slice.
    """

    projected: dict[str, float] = {}
    for name, exact in _quota_headroom_exact(limits, grants, now=now, replacing=replacing).items():
        quantity = float(exact)
        if Decimal(str(quantity)) > exact:
            quantity = math.nextafter(quantity, 0.0)
        projected[name] = quantity
    return projected


def exceeds_quota_headroom(
    requested: ResourceLimits,
    headroom: tp.Mapping[str, float],
) -> tuple[str, ...]:
    """Return the dimensions in which ``requested`` overruns ``headroom``.

    Every dimension, not the first: an operator correcting one number at a time
    against a four-dimensional refusal is four round trips, and the fourth is the
    one where they give up and grant themselves fleet admin.

    A dimension the tenant's own quota does not constrain has no headroom at all,
    so requesting it is refused rather than treated as unlimited.
    """

    return tuple(
        sorted(name for name, quantity in resource_quantities(requested).items() if quantity > headroom.get(name, 0.0))
    )


# Wire amounts are finite IEEE-754 floats: their decimal exponent span fits
# comfortably here, including cost products. The default 28-digit context can
# silently erase a small but positive issued right next to a larger limit.
_MONEY_PRECISION = 2048


def _sum_money(values: tp.Iterable[float | Decimal]) -> Decimal:
    with localcontext(prec=_MONEY_PRECISION):
        return sum((Decimal(str(value)) for value in values), Decimal(0))


def _retained_budget_authority(grants: tp.Sequence[BudgetGrantRecord]) -> Decimal:
    # A report is not final-consumption proof and cannot refund unused rights.
    # Reports within the amount are already covered; excess reports increase the
    # charge rather than being hidden by an issued amount that proved too small.
    return _sum_money(max(Decimal(str(grant.amount)), Decimal(str(grant.reported_spend))) for grant in grants)


def outstanding_total(grants: tp.Sequence[BudgetGrantRecord], *, now: datetime) -> float:
    """Return full retained monetary authority, including retired grants.

    No final-consumption protocol exists. Revocation, expiry, replacement and a
    period roll therefore do not refund any issued authority. ``now`` remains
    accepted for compatibility; it does not determine monetary finality.
    """

    return float(_retained_budget_authority(grants))


def _issuable_amount_exact(
    limit: float,
    grants: tp.Sequence[BudgetGrantRecord],
    *,
    now: datetime,
    local_spent: float = 0.0,
    local_reserved: Decimal = Decimal(0),
    replacing: str | None = None,
) -> Decimal:
    """Return exact headroom shared by home admission and peer issuance.

    Callers supply current-period local spend and active reservations under the
    same admission lock. All issued peer rights remain reserved until reliably
    proven final, including when retired or replaced. Reports are covered by the
    issued rights, not subtracted twice. ``replacing`` remains accepted for
    compatibility but never excludes an outstanding right.
    """

    with localcontext(prec=_MONEY_PRECISION):
        return max(
            Decimal(0),
            Decimal(str(limit)) - _retained_budget_authority(grants) - Decimal(str(local_spent)) - local_reserved,
        )


def issuable_amount(
    limit: float,
    grants: tp.Sequence[BudgetGrantRecord],
    *,
    now: datetime,
    local_spent: float = 0.0,
    local_reserved: Decimal = Decimal(0),
    replacing: str | None = None,
) -> float:
    """Return conservative headroom with the existing exported float interface.

    Admission and issuance use the exact private helper. Round this public
    projection downward if conversion would erase a retained positive right.
    """

    exact = _issuable_amount_exact(
        limit, grants, now=now, local_spent=local_spent, local_reserved=local_reserved, replacing=replacing
    )
    projected = float(exact)
    if Decimal(str(projected)) > exact:
        projected = math.nextafter(projected, 0.0)
    return projected


def federated_spend(
    local_spent: float,
    grants: tp.Sequence[BudgetGrantRecord],
) -> float:
    """Return the tenant's total reported spend across every plane.

    Sums the *reported* figures, which are as fresh as each peer's last report and
    no fresher. That staleness is why this number informs and alerts but never
    admits: admission is enforced locally against a grant that was already bounded
    when it was issued.
    """

    return float(local_spent + sum(grant.reported_spend for grant in grants))


__all__ = (
    "BudgetGrantRecord",
    "GrantState",
    "PlanePeerRecord",
    "QuotaGrantRecord",
    "exceeds_quota_headroom",
    "federated_spend",
    "issuable_amount",
    "outstanding_quota",
    "outstanding_total",
    "quota_headroom",
)
