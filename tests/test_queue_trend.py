# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Queue pressure over time, derived from the ledger rather than a new store.

``OverviewData.queueTrend`` was typed and hard-set to ``[]``, so the dashboard's
queue-pressure chart rendered an empty state forever. The data it needed already
existed: the audit ledger records ``workload.submitted`` and the admission
transition with timestamps and a monotonic sequence.

Two choices are asserted because both are what keep this cheap. Buckets are wall
clock, not event count -- the question is "is the queue growing", and count-based
buckets compress exactly the quiet periods that answer it. And the scan is
bounded, because this backs a panel that polls: an unbounded read of the
fastest-growing table would make an idle browser tab the heaviest query the
control plane serves.
"""

from __future__ import annotations

import math
import typing as tp
from datetime import timedelta
from itertools import pairwise

import pytest
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import AuditEvent, Principal, Role, RoleGrant, WorkloadState, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import AuditEventRow, SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.control_plane import (
    _QUEUE_TREND_MAX_EVENTS,
    QUEUE_TREND_MAX_BUCKETS,
    QUEUE_TREND_MAX_WINDOW_S,
    QUEUE_TREND_MIN_WINDOW_S,
)
from eyvan.server.service.errors import InputError


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


async def _emit(service: ControlPlaneService, *events: AuditEvent) -> None:
    async with service.repository.transaction() as tx:
        for event in events:
            await tx.append_event(event)


def _submitted(identifier: str, when) -> AuditEvent:
    return AuditEvent(
        id=identifier,
        actor="alice",
        org_id="acme",
        project_id="research",
        entity_type="workload",
        entity_id=identifier,
        action="workload.submitted",
        timestamp=when,
    )


def _admitted(identifier: str, when) -> AuditEvent:
    return AuditEvent(
        id=identifier,
        actor="scheduler",
        org_id="acme",
        project_id="research",
        entity_type="workload",
        entity_id=identifier,
        action="workload.admitted",
        previous_state=WorkloadState.PENDING.value,
        new_state=WorkloadState.ADMITTED.value,
        timestamp=when,
    )


@pytest.mark.asyncio
async def test_submissions_and_admissions_land_in_their_own_buckets() -> None:
    service = await _service()
    now = utcnow()
    await _emit(
        service,
        _submitted("evt-1", now - timedelta(hours=3)),
        _submitted("evt-2", now - timedelta(hours=3)),
        _admitted("evt-3", now - timedelta(hours=1)),
    )

    trend = await service.queue_trend(
        _principal(), org_id="acme", project_id="research", window_s=14_400, buckets=4, now=now
    )

    assert len(trend.points) == 4
    assert trend.bucket_seconds == 3_600
    assert [point.submitted for point in trend.points] == [0, 2, 0, 0]
    assert [point.admitted for point in trend.points] == [0, 0, 0, 1]


@pytest.mark.asyncio
async def test_events_older_than_the_window_are_excluded() -> None:
    service = await _service()
    now = utcnow()
    await _emit(service, _submitted("evt-1", now - timedelta(days=3)))

    trend = await service.queue_trend(
        _principal(), org_id="acme", project_id="research", window_s=3_600, buckets=2, now=now
    )

    assert sum(point.submitted for point in trend.points) == 0


@pytest.mark.asyncio
async def test_a_quiet_period_is_a_zero_bucket_not_a_missing_one() -> None:
    """Count-based buckets would compress exactly the gap that answers the question."""

    service = await _service()
    now = utcnow()
    await _emit(service, _submitted("evt-1", now - timedelta(minutes=5)))

    trend = await service.queue_trend(
        _principal(), org_id="acme", project_id="research", window_s=3_600, buckets=6, now=now
    )

    assert len(trend.points) == 6
    assert [point.submitted for point in trend.points] == [0, 0, 0, 0, 0, 1]


@pytest.mark.asyncio
async def test_buckets_are_returned_oldest_first_and_evenly_spaced() -> None:
    service = await _service()
    now = utcnow()

    trend = await service.queue_trend(
        _principal(), org_id="acme", project_id="research", window_s=3_600, buckets=4, now=now
    )

    starts = [point.started_at for point in trend.points]
    assert starts == sorted(starts)
    gaps = {(later - earlier).total_seconds() for earlier, later in pairwise(starts)}
    assert gaps == {900.0}


@pytest.mark.asyncio
async def test_non_workload_events_are_ignored() -> None:
    """A budget or cell event is not queue pressure and must not read as one."""

    service = await _service()
    now = utcnow()
    await _emit(
        service,
        AuditEvent(
            id="evt-budget",
            actor="root",
            org_id="acme",
            project_id="research",
            entity_type="budget",
            entity_id="acme/research",
            action="workload.submitted",
            timestamp=now - timedelta(minutes=1),
        ),
    )

    trend = await service.queue_trend(
        _principal(), org_id="acme", project_id="research", window_s=3_600, buckets=2, now=now
    )

    assert sum(point.submitted for point in trend.points) == 0


@pytest.mark.asyncio
async def test_another_projects_events_do_not_appear() -> None:
    service = await _service()
    now = utcnow()
    async with service.repository.transaction() as tx:
        await tx.append_event(
            AuditEvent(
                id="evt-other",
                actor="bob",
                org_id="acme",
                project_id="other",
                entity_type="workload",
                entity_id="wl-other",
                action="workload.submitted",
                timestamp=now - timedelta(minutes=1),
            )
        )

    trend = await service.queue_trend(
        _principal(), org_id="acme", project_id="research", window_s=3_600, buckets=2, now=now
    )

    assert sum(point.submitted for point in trend.points) == 0


@pytest.mark.asyncio
async def test_a_nonpositive_window_or_bucket_count_is_refused() -> None:
    service = await _service()

    for window, buckets in (
        (0.0, 4),
        (QUEUE_TREND_MIN_WINDOW_S - 1, 4),
        (QUEUE_TREND_MAX_WINDOW_S + 1, 4),
        (math.inf, 4),
        (math.nan, 4),
        (3_600.0, 0),
        (3_600.0, QUEUE_TREND_MAX_BUCKETS + 1),
    ):
        with pytest.raises(InputError):
            await service.queue_trend(
                _principal(), org_id="acme", project_id="research", window_s=window, buckets=buckets
            )


def test_the_ledger_scan_is_bounded() -> None:
    """This backs a polling panel; an unbounded scan makes an idle tab expensive."""

    assert _QUEUE_TREND_MAX_EVENTS > 0
    assert _QUEUE_TREND_MAX_EVENTS <= 100_000


@pytest.fixture(params=("memory", "sqlite", "postgres"))
async def mature_trend_service(request, tmp_path) -> tp.AsyncIterator[tuple[ControlPlaneService, str]]:
    """Exercise recent-window semantics on the model, local SQL, and production SQL."""

    backend = request.param
    if backend == "memory":
        repository = InMemoryRepository()
        await repository.open()
        yield ControlPlaneService(repository=repository), backend
        return
    if backend == "postgres":
        engine = create_async_engine(request.getfixturevalue("postgres_url"))
        async with engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'queue-trend.db'}")
    await create_schema(engine)
    try:
        yield ControlPlaneService(repository=SqlAlchemyRepository(engine)), backend
    finally:
        await engine.dispose()


async def _seed_many(service: ControlPlaneService, events: list[AuditEvent], *, first_sequence: int) -> None:
    """Bulk seed a mature ledger without making a 10k-row test flush 10k times."""

    repository = service.repository
    if isinstance(repository, InMemoryRepository):
        await _emit(service, *events)
        return
    assert isinstance(repository, SqlAlchemyRepository)
    rows = []
    for offset, event in enumerate(events):
        stored = event.model_copy(update={"sequence": first_sequence + offset})
        rows.append(
            {
                "sequence": stored.sequence,
                "id": stored.id,
                "org_id": stored.org_id,
                "project_id": stored.project_id,
                "entity_type": stored.entity_type,
                "entity_id": stored.entity_id,
                "occurred_at": stored.timestamp,
                "data": stored.model_dump(mode="json"),
            }
        )
    async with repository.engine.begin() as connection:
        # Executemany keeps this adversarial storage test cheap enough for the
        # SQLite and PostgreSQL gates while still exercising the real query.
        await connection.execute(insert(AuditEventRow), rows)


@pytest.mark.asyncio
async def test_mature_ledgers_keep_current_events_and_refuse_incomplete_windows(mature_trend_service) -> None:
    service, _backend = mature_trend_service
    now = utcnow()
    old = [_submitted(f"old-{index:05d}", now - timedelta(days=3)) for index in range(_QUEUE_TREND_MAX_EVENTS + 1)]
    current = _submitted("current-visible", now - timedelta(seconds=1))
    await _seed_many(service, [*old, current], first_sequence=1)

    trend = await service.queue_trend(
        _principal(), org_id="acme", project_id="research", window_s=3_600, buckets=2, now=now
    )

    # The old implementation returned the oldest 5,000 lifetime rows and lost
    # this event forever as soon as the ledger matured.
    assert sum(point.submitted for point in trend.points) == 1

    overflow = [
        _submitted(f"current-{index:05d}", now - timedelta(seconds=2)) for index in range(_QUEUE_TREND_MAX_EVENTS + 1)
    ]
    await _seed_many(service, overflow, first_sequence=len(old) + 2)

    with pytest.raises(InputError, match=r"incomplete.*shorter window"):
        await service.queue_trend(_principal(), org_id="acme", project_id="research", window_s=3_600, buckets=2, now=now)


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
