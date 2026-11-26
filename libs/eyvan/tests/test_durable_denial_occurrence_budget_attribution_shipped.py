# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable denial occurrence and budget attribution on shipped repositories."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from test_controller import FakeAgent
from test_historical_status_reads_become_fresh_provider import _blocks, _record, _source
from test_period_aware_fleet_budget_reconciliation import _ADMIN, _START, _lease, _settle

from eyvan.capacity.autoscale.cache import AutoscaleConfigCache
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, CapacityStatus
from eyvan.records import (
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    OutboxRecord,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.invariants import Invariant, check_fleet
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import rolled_budget


@pytest.fixture(params=["memory", "local", "postgres"])
async def repository(request, tmp_path):
    engine = admin = None
    schema = "attribution_" + uuid4().hex
    if request.param == "memory":
        repo = InMemoryRepository()
    elif request.param == "local":
        repo = LocalJsonRepository(tmp_path / "state.json")
    else:
        admin = create_async_engine(request.getfixturevalue("postgres_url"))
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(
            request.getfixturevalue("postgres_url"), connect_args={"server_settings": {"search_path": schema}}
        )
        await create_schema(engine)
        repo = SqlAlchemyRepository(engine)
    await repo.open()
    try:
        yield repo
    finally:
        await repo.close()
        if engine is not None:
            await engine.dispose()
        if admin is not None:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await admin.dispose()


async def _cell(repository, cell_id="cell"):
    async with repository.transaction() as tx:
        cell = await tx.get_cell(cell_id)
    assert cell is not None
    return cell


async def test_denial_observation_stamps_once_and_preserves_heartbeat(repository):
    agent = FakeAgent()
    agent.capacity_state = CapacityStatus.PROVIDER_ERROR
    agent.capacity_failure_code = CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, agent_factory=lambda _: agent)
    cell = CellRecord(
        id="cell",
        provider="gcp",
        region="us-east5",
        provider_project="provider-project",
        zone="us-east5-a",
        capacity={"tpu": 8},
        allocatable={"tpu": 8},
        accelerator_type="v5p-16",
        capacity_class="spot",
        state=CellState.REGISTERING,
        provider_phase=CapacityStatus.PROVIDER_ERROR,
        provider_phase_started_at=_START - timedelta(days=1),
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
    assert await controller._observe_cell_once_locked(cell, _START)
    first = await _cell(repository)
    assert first.provider_failure_occurred_at == _START
    assert first.provider_failure_event_id
    assert first.provider_failure_generation == 1
    assert await controller._observe_cell_once_locked(first, _START + timedelta(hours=25))
    reread = await _cell(repository)
    assert reread.provider_failure_occurred_at == _START
    assert reread.provider_failure_event_id == first.provider_failure_event_id
    assert reread.last_heartbeat_at == _START + timedelta(hours=25)
    # A new generation is an actual new provider attempt, even if a recreate
    # retained the preceding failure code in its durable snapshot.
    successor = reread.model_copy(update={"generation": 2, "state": CellState.REGISTERING})
    async with repository.transaction() as tx:
        await tx.put_cell(successor)
    new_at = _START + timedelta(days=2)
    assert await controller._observe_cell_once_locked(successor, new_at)
    renewed = await _cell(repository)
    assert renewed.provider_failure_occurred_at == new_at
    assert renewed.provider_failure_generation == 2
    assert renewed.provider_failure_event_id != first.provider_failure_event_id


async def test_failure_settlement_propagates_denial_source(repository, monkeypatch):
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service)
    cell = CellRecord(
        id="cell",
        provider="gcp",
        region="us-east5",
        provider_project="provider-project",
        zone="us-east5-a",
        capacity={"tpu": 8},
        allocatable={"tpu": 8},
        accelerator_type="v5p-16",
        capacity_class="spot",
        state=CellState.REGISTERING,
        pool_id="pool",
        autoscale_request_workload_id="wl",
    )
    workload = WorkloadRecord(
        id="wl",
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="true")),
    )
    command = OutboxRecord(
        id="apply-denied",
        topic="capacity.apply",
        aggregate_type="cell",
        aggregate_id="cell",
        generation=1,
        idempotency_key="apply-denied",
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
        await tx.put_workload(workload)
    monkeypatch.setattr("eyvan.server.controller.settlement.utcnow", lambda: _START)
    await controller._record_capacity_failure(
        command, "queue denied", failure_code=CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
    )
    denied = await _cell(repository)
    async with repository.transaction() as tx:
        terminal = await tx.get_workload("wl")
    assert terminal is not None and terminal.state is WorkloadState.FAILED
    assert terminal.queue_admission_denial.occurred_at == _START
    assert terminal.queue_admission_denial.event_id == command.id == denied.provider_failure_event_id
    assert denied.provider_failure_occurred_at == _START
    monkeypatch.setattr("eyvan.server.controller.settlement.utcnow", lambda: _START + timedelta(days=2))
    await controller._record_capacity_failure(
        command, "queue denied", failure_code=CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
    )
    assert await _cell(repository) == denied


@pytest.mark.parametrize("kind", ["cell", "workload"])
def test_legacy_denial_does_not_invent_read_time(tmp_path, kind):
    source = _source(kind, datetime.now(UTC))
    if isinstance(source, CellRecord):
        source = source.model_copy(
            update={
                "provider_failure_occurred_at": None,
                "provider_failure_event_id": None,
                "provider_failure_generation": None,
            }
        )
    else:
        source = source.model_copy(
            update={
                "queue_admission_denial": source.queue_admission_denial.model_copy(
                    update={"occurred_at": None, "event_id": None}
                )
            }
        )
    cache = AutoscaleConfigCache(tmp_path / "cache")
    _record(cache, source)
    assert not _blocks(cache)
    assert cache.list_queue_admission_denials() == ()


@pytest.mark.parametrize("error", [-1, 0, 1])
async def test_budget_opening_balance_and_period_attribution(repository, monkeypatch, error):
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(
                org_id="acme", project_id="research", limit=100, spent=10, period="daily", updated_by="controller"
            )
        )
    opened = await service.roll_budget_period(_ADMIN, org_id="acme", project_id="research", now=_START)
    assert opened.spent == opened.period_opening_spent == 10
    assert opened.settlement_period_id
    await _settle(
        repository, service, monkeypatch, _lease("new", created_at=_START, hourly_cost=5), at=_START + timedelta(hours=1)
    )
    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
        lease = await tx.get_lease("new")
        assert budget.spent == 15
        assert lease.settlement_budget_period_id == budget.settlement_period_id == opened.settlement_period_id
        if error:
            await tx.put_budget(budget.model_copy(update={"spent": budget.spent + error}))
    violations = await check_fleet(repository, now=_START + timedelta(hours=1))
    assert [v.invariant for v in violations] == ([Invariant.BUDGET_MATCHES_SETTLED_COST] if error else [])
    rolled = await service.roll_budget_period(
        _ADMIN, org_id="acme", project_id="research", now=_START + timedelta(days=3)
    )
    assert rolled.spent == rolled.period_opening_spent == 0
    assert rolled.settlement_period_id != opened.settlement_period_id
    assert await check_fleet(repository, now=_START + timedelta(days=3)) == ()


async def test_unknown_legacy_budget_is_not_falsely_corrupt(repository):
    async with repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(
                org_id="acme",
                project_id="research",
                limit=100,
                spent=10,
                period="daily",
                period_started_at=_START,
                updated_by="controller",
            )
        )
    assert await check_fleet(repository, now=_START) == ()


def test_pure_roll_matches_durable_attribution():
    budget = BudgetRecord(
        org_id="acme", project_id="research", limit=100, spent=10, period="daily", updated_by="controller"
    )
    opened = rolled_budget(budget, _START)
    assert opened.period_opening_spent == 10
    assert opened.settlement_period_id
    preserved = rolled_budget(opened, _START + timedelta(hours=1))
    assert preserved == opened
    rolled = rolled_budget(preserved, _START + timedelta(days=2))
    assert rolled.period_opening_spent == rolled.spent == 0
    assert rolled.settlement_period_id != opened.settlement_period_id


@pytest.mark.parametrize("period", ["daily", "lifetime", "unknown"])
async def test_settlement_window_and_replay_preserve_charging(repository, monkeypatch, period):
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(
                org_id="acme",
                project_id="research",
                limit=100,
                period=period,
                period_started_at=_START,
                updated_by="controller",
            )
        )
    await _settle(
        repository,
        service,
        monkeypatch,
        _lease("old", created_at=_START, hourly_cost=10),
        at=_START + timedelta(hours=1),
    )
    boundary = _START + timedelta(days=1)
    await _settle(
        repository,
        service,
        monkeypatch,
        _lease("spanning", created_at=boundary - timedelta(hours=1), hourly_cost=5),
        at=boundary,
    )
    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
        leases = await tx.list_leases()
    assert budget.spent == (5 if period == "daily" else 15)
    assert sum(lease.settled_cost for lease in leases) == 15
    assert await check_fleet(repository, now=boundary) == ()
    assert not await service.settle_orphaned_lease(_ADMIN, "spanning", reason="orphaned", expected_generation=1)
    async with repository.transaction() as tx:
        assert await tx.get_budget("acme", "research") == budget
        assert await tx.list_leases() == leases


async def test_budget_configuration_cannot_forge_or_reset_attribution(repository):
    service = ControlPlaneService(repository)
    request = BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=100,
        spent=10,
        period="lifetime",
        settlement_period_id="forged",
        period_opening_spent=0,
        updated_by="controller",
    )
    first, replayed = await service.put_budget(_ADMIN, request, idempotency_key="budget")
    assert not replayed
    assert first.period_opening_spent == 10
    assert first.settlement_period_id != "forged"
    replay, replayed = await service.put_budget(
        _ADMIN,
        request.model_copy(update={"settlement_period_id": "other", "period_opening_spent": 99}),
        idempotency_key="budget",
    )
    assert replayed and replay == first
    edited, replayed = await service.put_budget(
        _ADMIN, request.model_copy(update={"limit": 200}), idempotency_key="edit"
    )
    assert not replayed
    assert edited.period_opening_spent == first.period_opening_spent
    assert edited.settlement_period_id == first.settlement_period_id
    assert await check_fleet(repository) == ()


def test_unknown_optional_attribution_preserves_legacy_wire():
    budget = BudgetRecord(org_id="acme", project_id="research", limit=100, updated_by="controller")
    lease = _lease("legacy", created_at=_START, hourly_cost=10)
    cell = CellRecord(id="cell", provider="local", region="local", capacity={"cpu": 1}, allocatable={"cpu": 1})
    for record, names in (
        (budget, ("period_opening_spent", "settlement_period_id")),
        (lease, ("settlement_budget_period_id",)),
        (cell, ("provider_failure_occurred_at", "provider_failure_event_id", "provider_failure_generation")),
    ):
        document = record.model_dump(mode="json")
        assert all(name not in document for name in names)
        assert type(record).model_validate(document) == record
