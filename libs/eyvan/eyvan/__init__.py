# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Eyvan global fleet control plane."""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from eyvan.capacity.fleet_config import EyvanFleetConfig
    from eyvan.config import EyvanSettings
    from eyvan.records import CapacityClass
    from eyvan.sdk import AsyncEyvanClient, ClientIdentity, EyvanCallableSubmitter, EyvanClient
    from eyvan.sdk.client.easydel_setup import EasyDeLSetup
    from eyvan.sdk.client.jobs import CpuConfig, GpuConfig, SetupScript
    from eyvan.sdk.execution import (
        EyvanJob,
        EyvanJobError,
        EyvanPlacementError,
        EyvanUnavailableError,
        TpuConfig,
        execute,
        init,
    )
    from eyvan.storage import (
        StorageHandle,
        configure_storage,
        get_monitoring_storage,
        get_storage,
        monitoring_path,
        storage_path,
    )
    from eyvan.worker.runtime.in_job import in_job_context, submit_child
    from eyvan.worker.runtime.preemption import (
        PreemptionNotice,
        acknowledge_checkpoint,
        install_preemption_handler,
        preemption_notice,
        reclaim_requested,
        resume_checkpoint_uri,
    )
    from eyvan.worker.runtime.progress import report_progress

__all__ = (
    "AsyncEyvanClient",
    "CapacityClass",
    "ClientIdentity",
    "CpuConfig",
    "EasyDeLSetup",
    "EyvanCallableSubmitter",
    "EyvanClient",
    "EyvanFleetConfig",
    "EyvanJob",
    "EyvanJobError",
    "EyvanPlacementError",
    "EyvanSettings",
    "EyvanUnavailableError",
    "GpuConfig",
    "PreemptionNotice",
    "SetupScript",
    "StorageHandle",
    "TpuConfig",
    "acknowledge_checkpoint",
    "configure_storage",
    "execute",
    "get_monitoring_storage",
    "get_storage",
    "in_job_context",
    "init",
    "install_preemption_handler",
    "monitoring_path",
    "preemption_notice",
    "reclaim_requested",
    "report_progress",
    "resume_checkpoint_uri",
    "storage_path",
    "submit_child",
)
from eyvan._version import __version__

_Export = tp.TypeVar("_Export")


def _cache_export(name: str, value: _Export) -> _Export:
    """Cache one explicitly imported lazy package export."""

    globals()[name] = value
    return value


def __getattr__(name: str):
    """Load public SDK exports only when callers request them.

    Native worker subprocesses import ``eyvan.worker.runtime.local_worker`` directly.
    Keeping the package root light prevents that critical path from importing
    the client, API, and their optional observability dependencies first.
    """

    if name == "AsyncEyvanClient":
        from eyvan.sdk import AsyncEyvanClient

        return _cache_export(name, AsyncEyvanClient)
    if name == "CapacityClass":
        # The value TpuConfig(capacity_class=...) takes. records is light: it
        # pulls in pydantic models, not the client or its transport.
        from eyvan.records import CapacityClass

        return _cache_export(name, CapacityClass)
    if name == "ClientIdentity":
        from eyvan.sdk import ClientIdentity

        return _cache_export(name, ClientIdentity)
    if name == "EyvanCallableSubmitter":
        from eyvan.sdk import EyvanCallableSubmitter

        return _cache_export(name, EyvanCallableSubmitter)
    if name == "EyvanClient":
        from eyvan.sdk import EyvanClient

        return _cache_export(name, EyvanClient)
    if name == "EyvanFleetConfig":
        from eyvan.capacity.fleet_config import EyvanFleetConfig

        return _cache_export(name, EyvanFleetConfig)
    if name == "EyvanJob":
        from eyvan.sdk.execution import EyvanJob

        return _cache_export(name, EyvanJob)
    if name == "EyvanJobError":
        from eyvan.sdk.execution import EyvanJobError

        return _cache_export(name, EyvanJobError)
    if name == "EyvanPlacementError":
        from eyvan.sdk.execution import EyvanPlacementError

        return _cache_export(name, EyvanPlacementError)
    if name == "EyvanUnavailableError":
        from eyvan.sdk.execution import EyvanUnavailableError

        return _cache_export(name, EyvanUnavailableError)
    if name == "EyvanSettings":
        from eyvan.config import EyvanSettings

        return _cache_export(name, EyvanSettings)
    if name == "TpuConfig":
        from eyvan.sdk.execution import TpuConfig

        return _cache_export(name, TpuConfig)
    if name == "GpuConfig":
        from eyvan.sdk.client.jobs import GpuConfig

        return _cache_export(name, GpuConfig)
    if name == "CpuConfig":
        from eyvan.sdk.client.jobs import CpuConfig

        return _cache_export(name, CpuConfig)
    if name == "SetupScript":
        from eyvan.sdk.client.jobs import SetupScript

        return _cache_export(name, SetupScript)
    if name == "EasyDeLSetup":
        from eyvan.sdk.client.easydel_setup import EasyDeLSetup

        return _cache_export(name, EasyDeLSetup)
    if name == "execute":
        from eyvan.sdk.execution import execute

        return _cache_export(name, execute)
    if name == "init":
        from eyvan.sdk.execution import init

        return _cache_export(name, init)
    if name in {
        "StorageHandle",
        "configure_storage",
        "get_monitoring_storage",
        "get_storage",
        "monitoring_path",
        "storage_path",
    }:
        # Standard-library-only storage helpers: safe to import inside training
        # jobs without pulling in the SDK client or server dependencies.
        from eyvan import storage as _storage

        return _cache_export(name, getattr(_storage, name))
    if name in {"in_job_context", "submit_child"}:
        # Same reasoning as report_progress: this runs *inside* the job, so it
        # must not drag the client in behind it.
        from eyvan.worker.runtime import in_job

        return _cache_export(name, getattr(in_job, name))
    if name in {
        "PreemptionNotice",
        "acknowledge_checkpoint",
        "install_preemption_handler",
        "preemption_notice",
        "reclaim_requested",
        "resume_checkpoint_uri",
    }:
        # Standard-library-only in-job preemption/reclaim notice helpers.
        from eyvan.worker.runtime import preemption as _preemption

        return _cache_export(name, getattr(_preemption, name))
    if name == "report_progress":
        # Not from the SDK: this one runs *inside* the job, which is the path the
        # rest of this table exists to keep free of the client and its optional
        # dependencies. It imports nothing but the standard library.
        from eyvan.worker.runtime.progress import report_progress

        return _cache_export(name, report_progress)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazily loaded SDK names to interactive tooling."""

    return sorted({*globals(), *__all__})
