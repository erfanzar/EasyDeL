# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Normalized capacity lifecycle contracts shared across Eyvan layers."""

from __future__ import annotations

from eyvan._internal.compat import StrEnum


class CapacityStatus(StrEnum):
    """Provider-neutral lifecycle state for one capacity cell.

    Provider adapters map their raw cloud states into this closed set before
    observations reach drivers, persistence, the agent wire, or the control
    plane. ``REQUESTING`` is the sole control-plane-owned state: it records
    that Eyvan is about to issue a provider request, not that a provider has
    begun provisioning capacity.
    """

    UNOBSERVED = "UNOBSERVED"
    REQUESTING = "REQUESTING"
    ABSENT = "ABSENT"
    WAITING = "WAITING"
    PROVISIONING = "PROVISIONING"
    ACTIVE = "ACTIVE"
    EXECUTION_CONNECTING = "EXECUTION_CONNECTING"
    HEALTHY = "HEALTHY"
    DRAINING = "DRAINING"
    DELETING = "DELETING"
    RECREATING = "RECREATING"
    RELEASED = "RELEASED"
    DOWN = "DOWN"
    NEEDS_RECREATE = "NEEDS_RECREATE"
    UNAVAILABLE = "UNAVAILABLE"
    HALTED_QUOTA = "HALTED_QUOTA"
    HALTED_BUDGET = "HALTED_BUDGET"
    PROVIDER_ERROR = "PROVIDER_ERROR"


READY_CAPACITY_STATUSES = frozenset({CapacityStatus.HEALTHY})
QUEUED_CAPACITY_STATUSES = frozenset({CapacityStatus.WAITING})
EXECUTION_PENDING_CAPACITY_STATUSES = frozenset({CapacityStatus.EXECUTION_CONNECTING})
PENDING_CAPACITY_STATUSES = QUEUED_CAPACITY_STATUSES | {
    CapacityStatus.ACTIVE,
    CapacityStatus.PROVISIONING,
    CapacityStatus.REQUESTING,
    *EXECUTION_PENDING_CAPACITY_STATUSES,
}
DEFINITIVE_CAPACITY_LOSS_STATUSES = frozenset(
    {
        CapacityStatus.ABSENT,
        CapacityStatus.HALTED_BUDGET,
        CapacityStatus.HALTED_QUOTA,
        CapacityStatus.NEEDS_RECREATE,
        CapacityStatus.RELEASED,
    }
)


__all__ = (
    "DEFINITIVE_CAPACITY_LOSS_STATUSES",
    "EXECUTION_PENDING_CAPACITY_STATUSES",
    "PENDING_CAPACITY_STATUSES",
    "QUEUED_CAPACITY_STATUSES",
    "READY_CAPACITY_STATUSES",
    "CapacityStatus",
)
