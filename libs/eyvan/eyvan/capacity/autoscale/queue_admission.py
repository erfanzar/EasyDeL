# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared local recording for provider-proven GCP TPU queue denials."""

from __future__ import annotations

from eyvan.capacity.autoscale.cache import AutoscaleConfigCache
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.records import (
    AutoscaleBlockReason,
    CapacityClass,
    CellRecord,
    GcpQueueAdmissionDenialTarget,
    WorkloadRecord,
)


class QueueAdmissionDenialRecorder:
    """Persist exact Spot denials without changing workload or CLI outcomes."""

    def __init__(self, cache: AutoscaleConfigCache) -> None:
        self._cache = cache

    def record_workload(self, workload: WorkloadRecord) -> None:
        """Record the immutable target carried by a terminal denied workload."""

        if workload.autoscale_block_reason is not AutoscaleBlockReason.QUEUE_ADMISSION_DENIED:
            return
        self._record_target(workload.queue_admission_denial)

    def record_cell(self, cell: CellRecord) -> None:
        """Backfill an exact denial from a cell reported by fleet status."""

        if cell.provider_failure_code is not CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED:
            return
        if cell.provider != "gcp":
            return
        project = cell.provider_project
        zone = cell.zone
        accelerator_type = cell.accelerator_type
        if project is None or zone is None or accelerator_type is None:
            return
        self._record_target(
            GcpQueueAdmissionDenialTarget(
                provider_project=project,
                zone=zone,
                accelerator_type=accelerator_type,
                capacity_class=cell.capacity_class,
                occurred_at=(
                    cell.provider_failure_occurred_at if cell.provider_failure_generation == cell.generation else None
                ),
                event_id=cell.provider_failure_event_id,
            )
        )

    def _record_target(self, target: GcpQueueAdmissionDenialTarget | None) -> None:
        """Write one cache-compatible Spot target while treating the cache as optional."""

        if target is None or target.capacity_class is not CapacityClass.SPOT or target.occurred_at is None:
            # Legacy status has no trustworthy event clock. It remains a durable
            # denial, but a local reread cannot turn it into a new occurrence.
            return
        try:
            self._cache.record_queue_admission_denial(
                project=target.provider_project,
                zone=target.zone,
                accelerator_type=target.accelerator_type,
                capacity_class=target.capacity_class,
                observed_at=target.occurred_at,
            )
        except (OSError, ValueError):
            # The provider's durable denial must remain visible when a local cache is unavailable or malformed.
            return


__all__ = ("QueueAdmissionDenialRecorder",)
