# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Historical status reads must not become fresh provider-denial events."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import eyvan.capacity.autoscale.cache as cache_module
from eyvan._internal.compat import UTC
from eyvan.capacity.autoscale.cache import AutoscaleConfigCache, QueueAdmissionDenialCache
from eyvan.capacity.autoscale.queue_admission import QueueAdmissionDenialRecorder
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, CapacityStatus
from eyvan.records import (
    AutoscaleBlockReason,
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    GcpQueueAdmissionDenialTarget,
    ResourceLimits,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)


class _Clock(datetime):
    current = datetime(2026, 9, 1, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz is None else cls.current.astimezone(tz)


def _source(kind: str, occurred_at: datetime) -> CellRecord | WorkloadRecord:
    if kind == "cell":
        return CellRecord(
            id=f"cell-{occurred_at.isoformat()}",
            provider="gcp",
            provider_project="project-a",
            region="us-east5",
            zone="us-east5-a",
            accelerator_type="v5p-16",
            capacity_class=CapacityClass.SPOT,
            capacity=ResourceLimits(tpu=8),
            allocatable=ResourceLimits(tpu=8),
            state=CellState.UNAVAILABLE,
            provider_phase=CapacityStatus.PROVIDER_ERROR,
            provider_phase_started_at=occurred_at,
            provider_failure_code=CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
            provider_failure_occurred_at=occurred_at,
            provider_failure_event_id=f"denial-{occurred_at.isoformat()}",
            provider_failure_generation=1,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    return WorkloadRecord(
        id=f"wl-{occurred_at.isoformat()}",
        org_id="org-a",
        project_id="project-a",
        created_by="alice",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="true")),
        state=WorkloadState.FAILED,
        autoscale_block_reason=AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
        queue_admission_denial=GcpQueueAdmissionDenialTarget(
            provider_project="project-a",
            zone="us-east5-a",
            accelerator_type="v5p-16",
            capacity_class=CapacityClass.SPOT,
            occurred_at=occurred_at,
            event_id=f"denial-{occurred_at.isoformat()}",
        ),
        created_at=occurred_at,
        updated_at=occurred_at,
    )


def _record(cache: AutoscaleConfigCache, source: CellRecord | WorkloadRecord) -> None:
    # Reconstructing the recorder matches separate CLI status/list invocations.
    recorder = QueueAdmissionDenialRecorder(cache)
    if isinstance(source, CellRecord):
        recorder.record_cell(source)
    else:
        recorder.record_workload(source)


def _blocks(cache: AutoscaleConfigCache) -> bool:
    return cache.blocks_queue_admission_denial(
        project="project-a",
        zone="us-east5-a",
        accelerator_type="v5p-16",
        capacity_class=CapacityClass.SPOT,
    )


@pytest.mark.parametrize("kind", ["cell", "workload"])
@pytest.mark.parametrize("evict_first", [False, True])
def test_historical_denial_reread_stays_expired(monkeypatch, tmp_path: Path, kind: str, evict_first: bool) -> None:
    occurred_at = datetime(2026, 9, 1, tzinfo=UTC)
    monkeypatch.setattr(cache_module, "datetime", _Clock)
    monkeypatch.setattr(_Clock, "current", occurred_at)
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    source = _source(kind, occurred_at)
    _record(cache, source)
    assert _blocks(cache)

    _Clock.current = occurred_at + timedelta(seconds=QueueAdmissionDenialCache.DEFAULT_TTL_S + 60)
    assert cache.list_queue_admission_denials() == ()
    if evict_first:
        assert not _blocks(cache)
    _record(cache, source)
    assert not _blocks(cache), "unchanged historical status must not resurrect an expired denial"
    assert cache.list_queue_admission_denials() == ()


@pytest.mark.parametrize("kind", ["cell", "workload"])
def test_new_provider_denial_refreshes_but_old_status_cannot_replace_it(monkeypatch, tmp_path: Path, kind: str) -> None:
    occurred_at = datetime(2026, 9, 1, tzinfo=UTC)
    monkeypatch.setattr(cache_module, "datetime", _Clock)
    monkeypatch.setattr(_Clock, "current", occurred_at)
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    old = _source(kind, occurred_at)
    _record(cache, old)
    _Clock.current = occurred_at + timedelta(hours=25)
    assert not _blocks(cache)

    fresh = _source(kind, _Clock.current)
    _record(cache, fresh)
    assert _blocks(cache)
    assert cache.list_queue_admission_denials()[0].observed_at == fresh.updated_at
    _record(cache, old)
    assert _blocks(cache), "reading an older event must not erase a newer provider denial"
    assert cache.list_queue_admission_denials()[0].observed_at == fresh.updated_at


def test_cell_heartbeat_timestamp_is_not_a_new_denial(monkeypatch, tmp_path: Path) -> None:
    occurred_at = datetime(2026, 9, 1, tzinfo=UTC)
    monkeypatch.setattr(cache_module, "datetime", _Clock)
    monkeypatch.setattr(_Clock, "current", occurred_at + timedelta(hours=25))
    cache = AutoscaleConfigCache(tmp_path / "autoscale")
    old = _source("cell", occurred_at)
    assert isinstance(old, CellRecord)
    observed_again = old.model_copy(update={"updated_at": _Clock.current, "last_heartbeat_at": _Clock.current})
    _record(cache, observed_again)
    assert not _blocks(cache), "a refreshed cell observation is not a newly denied provider attempt"
