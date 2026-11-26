# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capacity and execution driver contracts."""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from eyvan.capacity.drivers.base import (
        CapacityDriver,
        CapacityStatus,
        CellCapabilities,
        CellObservation,
        ExecutionDriver,
        ExecutionWorkloadObservation,
        ExecutionWorkloadReport,
        ExecutionWorkloadResult,
        ExecutionWorkloadStatus,
        MutationContext,
    )
    from eyvan.capacity.drivers.gcp import GcpCapacityDriver, GcpComputeCapacityDriver, GcpTpuCapacityDriver
    from eyvan.capacity.drivers.local import LocalCapacityDriver

__all__ = (
    "CapacityDriver",
    "CapacityStatus",
    "CellCapabilities",
    "CellObservation",
    "ExecutionDriver",
    "ExecutionWorkloadObservation",
    "ExecutionWorkloadReport",
    "ExecutionWorkloadResult",
    "ExecutionWorkloadStatus",
    "GcpCapacityDriver",
    "GcpComputeCapacityDriver",
    "GcpTpuCapacityDriver",
    "LocalCapacityDriver",
    "MutationContext",
)

_Export = tp.TypeVar("_Export")


def _cache_export(name: str, value: _Export) -> _Export:
    """Cache one explicitly imported lazy driver export."""

    globals()[name] = value
    return value


def __getattr__(name: str):
    """Load an optional provider implementation only when requested."""

    if name in {
        "CapacityDriver",
        "CapacityStatus",
        "CellCapabilities",
        "CellObservation",
        "ExecutionDriver",
        "ExecutionWorkloadObservation",
        "ExecutionWorkloadReport",
        "ExecutionWorkloadResult",
        "ExecutionWorkloadStatus",
        "MutationContext",
    }:
        import eyvan.capacity.drivers.base as base

        if name == "CapacityDriver":
            return _cache_export(name, base.CapacityDriver)
        if name == "CapacityStatus":
            return _cache_export(name, base.CapacityStatus)
        if name == "CellCapabilities":
            return _cache_export(name, base.CellCapabilities)
        if name == "CellObservation":
            return _cache_export(name, base.CellObservation)
        if name == "ExecutionDriver":
            return _cache_export(name, base.ExecutionDriver)
        if name == "ExecutionWorkloadObservation":
            return _cache_export(name, base.ExecutionWorkloadObservation)
        if name == "ExecutionWorkloadReport":
            return _cache_export(name, base.ExecutionWorkloadReport)
        if name == "ExecutionWorkloadResult":
            return _cache_export(name, base.ExecutionWorkloadResult)
        if name == "ExecutionWorkloadStatus":
            return _cache_export(name, base.ExecutionWorkloadStatus)
        return _cache_export(name, base.MutationContext)
    if name in {"GcpCapacityDriver", "GcpComputeCapacityDriver", "GcpTpuCapacityDriver"}:
        import eyvan.capacity.drivers.gcp as gcp

        if name == "GcpCapacityDriver":
            return _cache_export(name, gcp.GcpCapacityDriver)
        if name == "GcpComputeCapacityDriver":
            return _cache_export(name, gcp.GcpComputeCapacityDriver)
        return _cache_export(name, gcp.GcpTpuCapacityDriver)
    if name == "LocalCapacityDriver":
        from eyvan.capacity.drivers.local import LocalCapacityDriver

        return _cache_export(name, LocalCapacityDriver)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy driver exports to interactive tooling."""

    return sorted({*globals(), *__all__})
