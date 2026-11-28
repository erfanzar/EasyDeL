# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Show acquisition backoff instead of presenting repeated refusals as a first request."""

from __future__ import annotations

import pytest
from test_callable_progress import _plain_output, _progress
from test_workload_progress import _operator, _pool, _seed, _workload

from eyvan.capacity import CapacityStatus
from eyvan.records import CellState, PlacementMode, WorkloadProgressPhase
from eyvan.sdk.client.callables import _CallableProgressRenderer
from eyvan.sdk.job_receipt import JobLifecycleState
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.workload_progress import WorkloadProgressResolver

_BACKOFF_REASON = (
    "capacity acquisition is backing off in pool research-us-central1-c-v6e-16 after 4 failed capacity "
    "requests in a row; next attempt after 2026-10-25 12:30:00 UTC: the active gcloud account cannot "
    "bind native-worker secrets in project 'example-project-1'"
)


async def _requesting_progress(scheduling_reason: str | None):
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(update={"scheduling_reason": scheduling_reason})
    requesting = pool.make_cell(
        "research-us-central1-c-v6e-16-0005",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"state": CellState.REGISTERING, "provider_phase": CapacityStatus.REQUESTING})
    await _seed(repository, pool, workload, cells=(requesting,))
    return await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)


@pytest.mark.asyncio
async def test_a_request_after_repeated_refusals_is_reported_as_a_provider_failure() -> None:
    progress = await _requesting_progress(_BACKOFF_REASON)

    assert progress.capacity.requesting_workers == 2
    assert progress.phase is WorkloadProgressPhase.PROVIDER_FAILURE
    assert progress.scheduling_reason == _BACKOFF_REASON


@pytest.mark.asyncio
async def test_a_genuine_first_request_is_still_requesting_capacity() -> None:
    progress = await _requesting_progress(None)

    assert progress.phase is WorkloadProgressPhase.REQUESTING_CAPACITY


@pytest.mark.asyncio
async def test_a_backing_off_pool_with_nothing_in_flight_is_a_provider_failure() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(update={"scheduling_reason": _BACKOFF_REASON})
    await _seed(repository, pool, workload)

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.PROVIDER_FAILURE


@pytest.mark.asyncio
async def test_capacity_that_is_actually_provisioning_wins_over_an_old_backoff_reason() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    workload = _workload(pool).model_copy(update={"scheduling_reason": _BACKOFF_REASON})
    provisioning = pool.make_cell(
        "research-us-central1-c-v6e-16-0006",
        worker_count=2,
        autoscale_request_workload_id=workload.id,
    ).model_copy(update={"state": CellState.REGISTERING, "provider_phase": CapacityStatus.PROVISIONING})
    await _seed(repository, pool, workload, cells=(provisioning,))

    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)

    assert progress.phase is WorkloadProgressPhase.PROVISIONING


def test_the_sdk_names_the_repeated_failure_and_the_next_attempt() -> None:
    progress = _progress(phase=WorkloadProgressPhase.PROVIDER_FAILURE, requesting_workers=2).model_copy(
        update={"scheduling_reason": _BACKOFF_REASON}
    )
    renderer = _CallableProgressRenderer(None, _plain_output(), None)  # type: ignore[arg-type]

    update = renderer._progress_update(progress, placement=PlacementMode.ELASTIC)

    assert update.state is JobLifecycleState.BLOCKED
    assert "requesting" not in update.summary
    assert update.summary == "capacity requests keep failing; the plane is backing off before asking again"
    assert update.diagnostic == _BACKOFF_REASON
    assert "next attempt after" in (update.diagnostic or "")


def test_the_sdk_keeps_the_plain_wording_for_a_first_request() -> None:
    progress = _progress(phase=WorkloadProgressPhase.WAITING_FOR_AUTOSCALER)
    renderer = _CallableProgressRenderer(None, _plain_output(), None)  # type: ignore[arg-type]

    update = renderer._progress_update(progress, placement=PlacementMode.ELASTIC)

    assert update.summary == "0/2 workers ready | requesting one elastic cell"
