# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eyvan.records import (
    BudgetRecord,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceHours,
    ResourceVector,
    Role,
    RoleGrant,
)
from eyvan.server.api.schemas import BudgetPutRequest
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service._control_plane import lease_settlement

NOW = datetime(2026, 9, 30, 12, tzinfo=UTC)
START = NOW - timedelta(hours=23)


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)
    await service.put_budget(
        _admin(),
        BudgetRecord(
            org_id="acme",
            project_id="research",
            limit=100,
            spent=75,
            accelerator_hours_spent={"tpu": 4},
            period="daily",
            period_started_at=START,
            alert_thresholds=(0.5, 0.8),
            alerted_thresholds=(0.5,),
            updated_by="root",
        ),
        idempotency_key="initial-budget",
    )
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [100, 200])
async def test_api_configuration_edit_keeps_the_original_spend_window(limit: float) -> None:
    service = await _service()
    request = BudgetPutRequest(limit=limit, period="daily", alert_thresholds=(0.5, 0.8))

    updated, replay = await service.put_budget(
        _admin(), request.to_record("acme", "research"), idempotency_key="configuration-edit"
    )
    status = await service.budget_status(_admin(), org_id="acme", project_id="research", now=NOW + timedelta(hours=2))
    rolled = await service.roll_budget_period(
        _admin(), org_id="acme", project_id="research", now=NOW + timedelta(hours=2)
    )

    assert updated.period_started_at == START
    assert updated.alerted_thresholds == (0.5,)
    assert updated.spent == 75
    assert updated.accelerator_hours_spent == {"tpu": 4}
    assert replay is False
    assert status.spent == 0
    assert status.period_started_at == START + timedelta(days=1)
    assert status.alerted_thresholds == ()
    assert rolled.spent == 0
    assert rolled.accelerator_hours_spent == {}
    assert rolled.period_started_at == START + timedelta(days=1)


@pytest.mark.asyncio
async def test_settlement_after_an_api_edit_does_not_repeat_an_announced_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = await _service()
    request = BudgetPutRequest(limit=100, period="daily", alert_thresholds=(0.5, 0.8))
    await service.put_budget(_admin(), request.to_record("acme", "research"), idempotency_key="configuration-edit")
    monkeypatch.setattr(lease_settlement, "utcnow", lambda: NOW)
    lease = LeaseRecord(
        id="orphaned-lease",
        workload_id="missing-workload",
        cell_id="cell-a",
        org_id="acme",
        project_id="research",
        state=LeaseState.ACTIVE,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(tpu=1)),
        hourly_cost=10,
        reserved_cost=10,
        resource_hours_reserved=ResourceHours(tpu=1),
        created_at=NOW - timedelta(hours=1),
        run_started_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(hours=1),
        fencing_token="fence",
    )
    async with service.repository.transaction() as tx:
        await tx.put_lease(lease)

    assert await service.settle_orphaned_lease(
        _admin(), lease.id, reason="orphaned", expected_generation=lease.generation
    )
    async with service.repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
        settled = await tx.get_lease(lease.id)
        events = await tx.list_events(org_id="acme", project_id="research", limit=100)

    assert budget is not None
    assert budget.spent == 85
    assert budget.accelerator_hours_spent == {"tpu": 5}
    assert settled is not None and settled.settled_cost == 10
    thresholds = [event.payload["threshold"] for event in events if event.action == "budget.threshold_crossed"]
    assert thresholds == [0.8], "only the newly reached threshold should emit an event"
    assert budget.period_started_at == START
    assert budget.alerted_thresholds == (0.5, 0.8)


@pytest.mark.asyncio
@pytest.mark.parametrize("thresholds, fired", [((0.8,), ()), ((0.5,), (0.5,))])
async def test_removing_a_threshold_drops_only_its_fired_ledger_entry(
    thresholds: tuple[float, ...], fired: tuple[float, ...]
) -> None:
    service = await _service()
    request = BudgetPutRequest(limit=100, period="daily", alert_thresholds=thresholds)

    updated, _ = await service.put_budget(
        _admin(), request.to_record("acme", "research"), idempotency_key="remove-threshold"
    )
    stored = await service.get_budget(_admin(), org_id="acme", project_id="research")

    assert stored == updated
    assert BudgetRecord.model_validate(stored.model_dump(mode="json")) == stored
    assert stored.period_started_at == START
    assert stored.alerted_thresholds == fired
    assert stored.alert_thresholds == thresholds
    assert stored.spent == 75
    assert stored.accelerator_hours_spent == {"tpu": 4}
