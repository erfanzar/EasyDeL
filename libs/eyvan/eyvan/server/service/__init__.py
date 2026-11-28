# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Eyvan application services and public lifecycle contracts."""

from __future__ import annotations

import typing as tp

from eyvan.records import (
    AuditEvent,
    BudgetRecord,
    CallableExecution,
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    NativeContainerConfig,
    NativeContainerMount,
    NativeRuntimeArtifact,
    NativeRuntimeConfig,
    Permission,
    PolicyRecord,
    Principal,
    PriorityBand,
    QuotaRecord,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.service.authz import Authorizer
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.scheduling import CoreSchedulingService, SchedulingService

if tp.TYPE_CHECKING:
    from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.state import (
    LEASE_TRANSITIONS,
    TERMINAL_WORKLOAD_STATES,
    WORKLOAD_TRANSITIONS,
    InvalidTransition,
    ensure_lease_transition,
    ensure_workload_transition,
)

__all__ = (
    "LEASE_TRANSITIONS",
    "TERMINAL_WORKLOAD_STATES",
    "WORKLOAD_TRANSITIONS",
    "AuditEvent",
    "Authorizer",
    "BudgetRecord",
    "CallableExecution",
    "CallableProjectTrustPolicy",
    "CapacityClass",
    "CellRecord",
    "CellState",
    "CommandExecution",
    "ControlPlaneService",
    "CoreSchedulingService",
    "InvalidTransition",
    "LeaseRecord",
    "LeaseState",
    "NativeContainerConfig",
    "NativeContainerMount",
    "NativeRuntimeArtifact",
    "NativeRuntimeConfig",
    "Permission",
    "PolicyRecord",
    "Principal",
    "PriorityBand",
    "QuotaRecord",
    "ResourceGang",
    "ResourceVector",
    "Role",
    "RoleGrant",
    "SchedulingService",
    "WorkloadDefinition",
    "WorkloadRecord",
    "WorkloadState",
    "ensure_lease_transition",
    "ensure_workload_transition",
)


def __getattr__(name: str) -> type[ControlPlaneService]:
    """Load the persistence-dependent application service without import cycles."""

    if name == "ControlPlaneService":
        from eyvan.server.service.control_plane import ControlPlaneService

        return ControlPlaneService
    raise AttributeError(name)
