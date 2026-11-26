# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A pool whose capacity requests keep failing backs off, says so, and resets on success.

Backoff must span replacement cells: a fresh cell's clean availability belief
must not reset the pool's repeated-failure history and allow unbounded churn.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest
from test_controller import FakeAgent

import eyvan.domain.availability as availability_module
import eyvan.records as records_module
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    AvailabilityPolicy,
    CapacityPoolRecord,
    CellState,
    Principal,
    Role,
    RoleGrant,
    WorkloadProgressPhase,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.workload_progress import WorkloadProgressResolver
from eyvan.worker.agent.client import CellAgentCapacityProviderError

_T0 = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
_TICK_S = 15
_MAX_BACKOFF_S = 600
_PROVISIONING_TIMEOUT_S = 60
_DENIAL = (
    "the active gcloud account cannot bind native-worker secrets in project 'example-research-project' "
    "(missing: secretmanager.secrets.setIamPolicy)"
)
_FLEET = "research-us-east5-a"


class _Clock:
    now = _T0


class _ClockDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return _Clock.now


@pytest.fixture(autouse=True)
def _simulated_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    _Clock.now = _T0
    # Every controller, persistence and service clock reads records.utcnow.
    monkeypatch.setattr(records_module, "datetime", _ClockDatetime)
    monkeypatch.setattr(availability_module, "datetime", _ClockDatetime)


class _RefusingAgent(FakeAgent):
    """A cell agent whose provider refuses every request with one configuration error."""

    def __init__(self) -> None:
        super().__init__()
        self.refusing = True
        self.apply_times: list[datetime] = []
        self.capacity_state = CapacityStatus.HEALTHY

    def apply_capacity(self, cell_id, request):
        self.apply_times.append(_Clock.now)
        if self.refusing:
            raise CellAgentCapacityProviderError(_DENIAL, code=CapacityProviderFailureCode.CONFIGURATION)
        return super().apply_capacity(cell_id, request)

    def reconcile(self, cell_id, request):
        if self.refusing:
            raise CellAgentCapacityProviderError(_DENIAL, code=CapacityProviderFailureCode.CONFIGURATION)
        return super().reconcile(cell_id, request)


def _admin() -> Principal:
    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _operator() -> Principal:
    return Principal(
        subject="operator",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _pool(*, min_cells: int) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=f"{_FLEET}-v5p-16",
        fleet_name=_FLEET,
        node_type="v5p-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="example-research-project",
        region="us-east5",
        zone="us-east5-a",
        capacity={"cpu": 32, "tpu": 8},
        allocatable={"cpu": 32, "tpu": 8},
        accelerator_type="v5p-16",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=min_cells,
        max_cells=2,
        provisioning_timeout_s=_PROVISIONING_TIMEOUT_S,
        availability_policy=AvailabilityPolicy(max_backoff_s=_MAX_BACKOFF_S),
    )


async def _plane(*, min_cells: int) -> tuple[InMemoryRepository, ControlPlaneService, EyvanController, _RefusingAgent]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = _RefusingAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
    )
    await service.apply_fleet(_admin(), _FLEET, (_pool(min_cells=min_cells),), idempotency_key="fleet-v1")
    return repository, service, controller, agent


async def _run(controller: EyvanController, seconds: float, *, schedule: bool = True) -> None:
    """Advance the simulated clock in scheduler ticks, running the capacity phases each tick."""

    end = _Clock.now + timedelta(seconds=seconds)
    while _Clock.now < end:
        _Clock.now += timedelta(seconds=_TICK_S)
        await controller.reconcile_capacity_pools_once()
        await controller.dispatch_outbox_once()
        await controller.observe_cells_once()
        if schedule:
            await controller.reconcile_once()


@pytest.mark.asyncio
async def test_a_pool_whose_requests_always_fail_backs_off_on_a_capped_doubling_curve() -> None:
    """A warm-floor pool: nothing is pending, so only the planner's own floor recreates cells."""

    _repository, _service, controller, agent = await _plane(min_cells=1)

    await _run(controller, 3 * 3600)

    gaps = [(later - earlier).total_seconds() for earlier, later in pairwise(agent.apply_times)]
    # Before the fix: a new cell roughly every provisioning timeout, ~180 in three hours.
    assert 5 <= len(agent.apply_times) <= 30, f"{len(agent.apply_times)} create attempts in 3 h: {gaps}"
    # One refusal is how Spot capacity is normally won: the first retry is not delayed.
    assert gaps[0] <= _PROVISIONING_TIMEOUT_S + 6 * _TICK_S
    # The wait grows, never shrinks by more than scheduling slack, and is capped.
    for earlier, later in pairwise(gaps):
        assert later >= earlier - 2 * _TICK_S, gaps
    assert gaps[-1] >= _MAX_BACKOFF_S, gaps
    assert max(gaps) <= _MAX_BACKOFF_S + _PROVISIONING_TIMEOUT_S + 8 * _TICK_S, gaps


@pytest.mark.asyncio
async def test_the_backoff_is_visible_on_the_workload_and_in_the_audit_ledger_and_resets_on_success() -> None:
    repository, service, controller, agent = await _plane(min_cells=0)
    workload, _ = await service.submit_pool_workload(
        _operator(),
        fleet_name=_FLEET,
        node_type="v5p-16",
        org_id="acme",
        project_id="research",
        command="python train.py",
        idempotency_key="workload-1",
    )

    while len(agent.apply_times) < 3:
        await _run(controller, _TICK_S)
        assert _Clock.now < _T0 + timedelta(hours=1), "three attempts should happen well within an hour"

    async with repository.transaction() as transaction:
        pending = await transaction.get_workload(workload.id)
        events = await transaction.list_events(entity_type="capacity_pool", limit=1000)
    assert pending is not None and pending.state is WorkloadState.PENDING
    reason = pending.scheduling_reason or ""
    assert reason.startswith("capacity acquisition is backing off"), reason
    assert "failed capacity requests in a row" in reason
    assert "cannot bind native-worker secrets" in reason, "the last error is named"
    backoff_events = [event for event in events if event.action == "capacity_pool.acquisition_backoff"]
    assert backoff_events, [event.action for event in events]
    latest = backoff_events[-1].payload
    assert latest["failed_attempts"] >= 2
    assert latest["next_attempt_at"]
    assert "cannot bind native-worker secrets" in str(latest["last_error"])

    # Problem 2: what the SDK reads must not say "requesting one elastic cell".
    progress = await WorkloadProgressResolver(service, repository).resolve(_operator(), workload.id)
    assert progress.phase is WorkloadProgressPhase.PROVIDER_FAILURE
    assert progress.scheduling_reason == pending.scheduling_reason

    # The grant is fixed: the next attempt succeeds and the streak ends.
    agent.refusing = False
    attempts_before = len(agent.apply_times)
    deadline = _Clock.now + timedelta(seconds=2 * _MAX_BACKOFF_S)
    ready = False
    while not ready:
        await _run(controller, _TICK_S, schedule=False)
        assert _Clock.now < deadline, "the paused pool must be asked again once its backoff passes"
        async with repository.transaction() as transaction:
            ready = any(cell.state is CellState.READY for cell in await transaction.list_cells())
    # Either the still-unretired failed cell recovers on its next observation,
    # or one fresh create succeeds -- never a burst of creates.
    assert len(agent.apply_times) <= attempts_before + 1
    await _run(controller, _TICK_S, schedule=False)

    async with repository.transaction() as transaction:
        recovered = await transaction.get_workload(workload.id)
        events = await transaction.list_events(entity_type="capacity_pool", limit=1000)
    assert recovered is not None
    assert not (recovered.scheduling_reason or "").startswith("capacity acquisition is backing off")
    assert events[-1].action == "capacity_pool.acquisition_backoff_cleared"


def _failed_cell(pool: CapacityPoolRecord, index: int, observed_at: datetime, *, delivered: bool = False):
    cell = pool.make_cell(f"{pool.id}-{index:04d}")
    belief = cell.availability.model_copy(
        update={
            "consecutive_failures": 0 if delivered else 1,
            "last_observed_at": observed_at,
            "provision_latencies_s": (30.0,) if delivered else (),
        }
    )
    return cell.model_copy(
        update={
            "state": CellState.DELETED,
            "provider_phase": CapacityStatus.PROVIDER_ERROR,
            "provider_failure_code": None if delivered else CapacityProviderFailureCode.CONFIGURATION,
            "unavailable_detail": None if delivered else _DENIAL,
            "availability": belief,
        }
    )


def test_streak_curve_waits_from_the_second_failure_doubles_and_caps() -> None:
    from eyvan.capacity.acquisition_backoff import pool_acquisition_backoffs

    pool = _pool(min_cells=1)
    observed: list[float] = []
    for count in range(1, 9):
        cells = [_failed_cell(pool, index, _T0 + timedelta(seconds=index)) for index in range(count)]
        streak = pool_acquisition_backoffs((pool,), cells, _T0 + timedelta(seconds=count))[pool.id]
        assert streak.failed_attempts == count
        observed.append(streak.backoff_s)
    assert observed == [0.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 600.0]


def test_one_delivered_cell_ends_the_streak_and_old_failures_are_history() -> None:
    from eyvan.capacity.acquisition_backoff import (
        ACQUISITION_BACKOFF_MEMORY_S,
        pool_acquisition_backoffs,
    )

    pool = _pool(min_cells=1)
    failures = [_failed_cell(pool, index, _T0 + timedelta(seconds=index)) for index in range(6)]
    delivered = _failed_cell(pool, 99, _T0 + timedelta(seconds=10), delivered=True)
    assert pool_acquisition_backoffs((pool,), [*failures, delivered], _T0 + timedelta(seconds=11)) == {}

    # One failure after the success starts a fresh streak at the bottom of the curve.
    later = _failed_cell(pool, 100, _T0 + timedelta(seconds=20))
    fresh = pool_acquisition_backoffs((pool,), [*failures, delivered, later], _T0 + timedelta(seconds=21))[pool.id]
    assert fresh.failed_attempts == 1 and fresh.backoff_s == 0.0

    long_after = _T0 + timedelta(seconds=_MAX_BACKOFF_S + ACQUISITION_BACKOFF_MEMORY_S + 60)
    assert pool_acquisition_backoffs((pool,), failures, long_after) == {}, "last week's outage is not a streak"


def test_an_unanswered_replacement_is_in_flight_not_failed() -> None:
    from eyvan.capacity.acquisition_backoff import pool_acquisition_backoffs

    pool = _pool(min_cells=1)
    in_flight = pool.make_cell(f"{pool.id}-0001")
    assert pool_acquisition_backoffs((pool,), [in_flight], _T0) == {}
