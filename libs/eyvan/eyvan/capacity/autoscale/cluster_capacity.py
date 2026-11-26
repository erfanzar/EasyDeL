# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observed cluster free capacity as an autoscale plan input -- or as nothing.

Eyvan's autoscaler sizes a pool from *demand*: pending workloads, a warm floor,
a per-pool ceiling, and a shared provider quota. On a Kubernetes cluster it does
not own, none of those know whether the cluster has room. A pool can therefore
launch a cell into a cluster that is full of non-Eyvan work, and the cell sits
WAITING until ``queue_timeout_s`` reclaims it -- repeatedly.

This module turns a Kueue reading (:mod:`eyvan.capacity.providers.kubernetes.kueue`)
into a decision the autoscaler can act on, under one rule:

    **A signal Eyvan cannot establish is not a plan input.**

Which produces three deliberate refusals, each one a way this could otherwise
silently go wrong:

* **UNKNOWN changes nothing.** Not "launch nothing" (which strands a pool
  whenever RBAC is wrong) and not "launch everything as if the cluster were
  empty" (which is today's behaviour, stated honestly).
* **A stale reading is UNKNOWN.** A cluster observed ten minutes ago is a
  cluster that has since admitted other people's work. The observation carries
  its time and the policy carries the age it tolerates; nothing else is trusted
  to remember to refresh.
* **A lower bound never proves a shortage.** Free capacity inside a Kueue cohort
  is a lower bound because borrowing depends on queues Eyvan cannot see. It can
  prove a cell fits; it can never prove one does not, so it never caps.

**Default-off.** :class:`ClusterCapacityPolicy` starts disabled, so a control
plane that upgrades into this code plans exactly as it did before until an
operator turns it on. A capacity signal that changed fleet sizing the moment it
shipped would be a behaviour change disguised as an observation.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from datetime import datetime, timedelta

from eyvan._internal.compat import UTC
from eyvan.capacity.providers.kubernetes.kueue import (
    ClusterCapacityExplanation,
    ClusterCapacityReading,
    ClusterCapacityState,
    explain_cluster_capacity,
)

#: How old a cluster reading may be before it stops being evidence. Sixty
#: seconds is roughly one control-plane reconciliation: long enough that a
#: reading survives the pass that took it, short enough that it cannot outlive
#: the admission decisions made against the same cluster.
DEFAULT_MAX_OBSERVATION_AGE_S = 60.0

#: How far into the future a reading's timestamp may sit before it is refused.
#: A reading stamped ahead of the planner's clock is not fresh evidence, it is
#: an unsynchronized clock -- and it would otherwise never expire.
MAX_OBSERVATION_SKEW_S = 5.0


@dataclass(frozen=True, slots=True)
class ClusterCapacityPolicy:
    """Operator control over whether observed cluster capacity plans at all.

    Attributes:
        enabled: Whether an OBSERVED reading may reduce a launch count. ``False``
            reproduces the behaviour of every Eyvan release before this one:
            capacity is requested from the cluster and the cluster decides.
        max_observation_age_s: Oldest reading still treated as evidence.
    """

    enabled: bool = False
    max_observation_age_s: float = DEFAULT_MAX_OBSERVATION_AGE_S

    def __post_init__(self) -> None:
        """Refuse a policy whose freshness bound could never expire anything."""

        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if (
            isinstance(self.max_observation_age_s, bool)
            or not isinstance(self.max_observation_age_s, (int, float))
            or not 0 < float(self.max_observation_age_s) <= 3600
        ):
            raise ValueError("max_observation_age_s must be greater than 0 and at most 3600")
        object.__setattr__(self, "max_observation_age_s", float(self.max_observation_age_s))


@dataclass(frozen=True, slots=True)
class ClusterCapacityAdvice:
    """What an observed cluster allows, and whether it was allowed to say so.

    Attributes:
        requested_cells: What the autoscaler wanted to launch.
        allowed_cells: What it may launch. Equal to ``requested_cells`` whenever
            this advice is not applied, which is the whole safety property.
        state: OBSERVED only when a fresh, exact reading actually decided.
        applied: Whether ``allowed_cells`` was derived from cluster capacity at
            all. ``False`` means the reading changed nothing.
        reason: Why the advice did or did not apply, in operator terms.
        explanation: The underlying cluster explanation, when there was one.
    """

    requested_cells: int
    allowed_cells: int
    state: ClusterCapacityState
    applied: bool
    reason: str
    explanation: ClusterCapacityExplanation | None = None

    @property
    def withheld_cells(self) -> int:
        """Return how many requested cells the observed cluster cannot take."""

        return max(self.requested_cells - self.allowed_cells, 0)


def advise_launch(
    requested_cells: int,
    reading: ClusterCapacityReading | None,
    request: tp.Mapping[str, str | int | float] | None = None,
    *,
    policy: ClusterCapacityPolicy | None = None,
    now: datetime | None = None,
) -> ClusterCapacityAdvice:
    """Return how many cells the observed cluster is proven to admit.

    Args:
        requested_cells: Cells the autoscaler wants to launch this pass.
        reading: A cluster capacity reading, or ``None`` when none was taken.
        request: One cell's resource requests, as a Job container states them.
        policy: Operator policy. Defaults to disabled, which changes nothing.
        now: Planning time, used to age the reading. Defaults to current UTC.

    Returns:
        A :class:`ClusterCapacityAdvice` whose ``allowed_cells`` equals
        ``requested_cells`` unless a fresh, exact, OBSERVED reading proved the
        cluster cannot take them.

    Raises:
        ValueError: If ``requested_cells`` is negative, or ``now`` is naive.
    """

    if isinstance(requested_cells, bool) or not isinstance(requested_cells, int) or requested_cells < 0:
        raise ValueError("requested_cells must be a non-negative integer")
    active = policy or ClusterCapacityPolicy()
    if not active.enabled:
        return _unchanged(requested_cells, "cluster capacity is not enabled as a plan input")
    if reading is None:
        return _unchanged(requested_cells, "no cluster capacity reading was taken")
    moment = now or datetime.now(UTC)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    age = (moment.astimezone(UTC) - reading.observed_at).total_seconds()
    if age > active.max_observation_age_s:
        return _unchanged(
            requested_cells,
            f"the cluster capacity reading is {age:.0f}s old, past the {active.max_observation_age_s:.0f}s bound",
        )
    if age < -MAX_OBSERVATION_SKEW_S:
        return _unchanged(requested_cells, "the cluster capacity reading is stamped in the future")
    explanation = explain_cluster_capacity(reading, dict(request or {}))
    fit = explanation.fit
    if fit.state is not ClusterCapacityState.OBSERVED or fit.admits is None:
        return _unchanged(
            requested_cells,
            f"cluster free capacity is unknown: {fit.reason or 'nothing was established'}",
            explanation=explanation,
        )
    if not fit.exact:
        return _unchanged(
            requested_cells,
            f"cluster free capacity for {fit.limiting_resource or 'the request'} is a lower bound, "
            "which cannot prove a shortage",
            explanation=explanation,
        )
    allowed = min(requested_cells, fit.admits)
    return ClusterCapacityAdvice(
        requested_cells=requested_cells,
        allowed_cells=allowed,
        state=ClusterCapacityState.OBSERVED,
        applied=True,
        reason=(
            f"ClusterQueue {reading.queue.cluster_queue!r} admits {fit.admits} cell(s) of this shape"
            if allowed < requested_cells
            else f"ClusterQueue {reading.queue.cluster_queue!r} has room for all {requested_cells} cell(s)"
        ),
        explanation=explanation,
    )


def observation_age(reading: ClusterCapacityReading, now: datetime | None = None) -> timedelta:
    """Return how old one cluster capacity reading is at ``now``."""

    moment = now or datetime.now(UTC)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    return moment.astimezone(UTC) - reading.observed_at


def _unchanged(
    requested_cells: int,
    reason: str,
    *,
    explanation: ClusterCapacityExplanation | None = None,
) -> ClusterCapacityAdvice:
    """Return advice that leaves the autoscaler's own decision untouched."""

    return ClusterCapacityAdvice(
        requested_cells=requested_cells,
        allowed_cells=requested_cells,
        state=ClusterCapacityState.UNKNOWN,
        applied=False,
        reason=reason,
        explanation=explanation,
    )


__all__ = (
    "DEFAULT_MAX_OBSERVATION_AGE_S",
    "MAX_OBSERVATION_SKEW_S",
    "ClusterCapacityAdvice",
    "ClusterCapacityPolicy",
    "advise_launch",
    "observation_age",
)
