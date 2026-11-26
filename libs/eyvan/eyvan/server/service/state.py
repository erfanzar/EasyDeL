# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public state machines for workload and capacity-lease lifecycles."""

from __future__ import annotations

import typing as tp

from eyvan.records import LeaseState, WorkloadState


class InvalidTransition(ValueError):
    """Raised when a state-machine transition violates the lifecycle contract."""


WORKLOAD_TRANSITIONS: tp.Final[dict[WorkloadState, frozenset[WorkloadState]]] = {
    WorkloadState.PENDING: frozenset({WorkloadState.ADMITTED, WorkloadState.CANCELLED, WorkloadState.FAILED}),
    WorkloadState.ADMITTED: frozenset({WorkloadState.LEASED, WorkloadState.CANCELLING, WorkloadState.FAILED}),
    WorkloadState.LEASED: frozenset({WorkloadState.DISPATCHING, WorkloadState.CANCELLING, WorkloadState.FAILED}),
    WorkloadState.DISPATCHING: frozenset({WorkloadState.RUNNING, WorkloadState.CANCELLING, WorkloadState.FAILED}),
    WorkloadState.RUNNING: frozenset(
        {WorkloadState.CHECKPOINTING, WorkloadState.SUCCEEDED, WorkloadState.CANCELLING, WorkloadState.FAILED}
    ),
    WorkloadState.CHECKPOINTING: frozenset(
        {WorkloadState.RUNNING, WorkloadState.SUCCEEDED, WorkloadState.CANCELLING, WorkloadState.FAILED}
    ),
    WorkloadState.CANCELLING: frozenset({WorkloadState.SUCCEEDED, WorkloadState.CANCELLED, WorkloadState.FAILED}),
    WorkloadState.SUCCEEDED: frozenset(),
    WorkloadState.FAILED: frozenset({WorkloadState.PENDING}),
    WorkloadState.CANCELLED: frozenset({WorkloadState.PENDING}),
}

LEASE_TRANSITIONS: tp.Final[dict[LeaseState, frozenset[LeaseState]]] = {
    LeaseState.PENDING: frozenset({LeaseState.ACTIVE, LeaseState.RELEASED, LeaseState.EXPIRED, LeaseState.QUARANTINED}),
    LeaseState.ACTIVE: frozenset({LeaseState.REVOKING, LeaseState.RELEASED, LeaseState.EXPIRED, LeaseState.QUARANTINED}),
    LeaseState.REVOKING: frozenset({LeaseState.RELEASED, LeaseState.EXPIRED, LeaseState.QUARANTINED}),
    LeaseState.QUARANTINED: frozenset({LeaseState.ACTIVE, LeaseState.RELEASED, LeaseState.EXPIRED}),
    LeaseState.RELEASED: frozenset(),
    LeaseState.EXPIRED: frozenset(),
}

TERMINAL_WORKLOAD_STATES: tp.Final[frozenset[WorkloadState]] = frozenset(
    {WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.CANCELLED}
)
ACTIVE_LEASE_STATES: tp.Final[frozenset[LeaseState]] = frozenset(
    {LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.REVOKING, LeaseState.QUARANTINED}
)


def ensure_workload_transition(current: WorkloadState, target: WorkloadState) -> None:
    """Validate a workload state transition."""

    if target == current:
        return
    if target not in WORKLOAD_TRANSITIONS[current]:
        raise InvalidTransition(f"workload cannot transition from {current.value} to {target.value}")


def ensure_lease_transition(current: LeaseState, target: LeaseState) -> None:
    """Validate a lease state transition."""

    if target == current:
        return
    if target not in LEASE_TRANSITIONS[current]:
        raise InvalidTransition(f"lease cannot transition from {current.value} to {target.value}")
