# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A definitive refusal may settle only the delivery that still owns dispatch."""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta

import pytest
from test_retained_secret_dispatch_authority import repository as _repository_fixture
from test_secret_dispatch_transactions import _plane, _seed_dispatch, _snapshot

from eyvan.records import LeaseState, OutboxState, WorkloadState, utcnow
from eyvan.worker.agent.client import CellAgentInvalidConfigurationError

repository = _repository_fixture


@pytest.mark.parametrize("gang", [False, True], ids=["single", "gang"])
@pytest.mark.parametrize("window", ["before_release", "before_settlement"])
async def test_late_refusal_cannot_settle_successor_dispatch(repository, tmp_path, monkeypatch, gang, window):
    service, _, agent, controller = await _plane(repository, tmp_path)
    _, _, _, successor = await _plane(repository, tmp_path)
    # Both deliveries reach the same worker, with its real idempotent submit.
    successor._agent_factory = lambda _cell: agent
    assert successor.worker_id != controller.worker_id
    seeded, _, command = await _seed_dispatch(repository, gang=gang)
    first_entered = threading.Event()
    first_resume = threading.Event()
    successor_entered = threading.Event()
    successor_resume = threading.Event()
    settlement_entered = asyncio.Event()
    settlement_resume = asyncio.Event()
    submit = agent.submit_gang if gang else agent.submit
    calls = 0

    def controlled_submit(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            assert first_resume.wait(timeout=10), "first refusal was not resumed"
            raise CellAgentInvalidConfigurationError("launch rejected before admission")
        successor_entered.set()
        assert successor_resume.wait(timeout=10), "successor submit was not resumed"
        return submit(*args)

    monkeypatch.setattr(agent, "submit_gang" if gang else "submit", controlled_submit)
    settle = service.settle_terminal_workload

    async def paused_settlement(*args, **kwargs):
        # Pause after the controller's snapshot, immediately before the service
        # transaction: a controller-only precheck cannot close this window.
        settlement_entered.set()
        await asyncio.wait_for(settlement_resume.wait(), timeout=10)
        return await settle(*args, **kwargs)

    if window == "before_settlement":
        monkeypatch.setattr(service, "settle_terminal_workload", paused_settlement)

    first_delivery = asyncio.create_task(controller.dispatch_outbox_once())
    successor_delivery = None
    try:
        assert await asyncio.to_thread(first_entered.wait, 5), "initial submit did not start"
        original, _, _, _ = await _snapshot(repository, seeded.id)
        assert original.dispatch_reservation is not None
        if window == "before_settlement":
            first_resume.set()
            await asyncio.wait_for(settlement_entered.wait(), timeout=5)
            released, pending, commands, events = await _snapshot(repository, seeded.id)
            assert released.dispatch_reservation is None
            assert all(lease.state is LeaseState.PENDING for lease in pending)
            assert "dispatch_reservation" not in next(item for item in commands if item.id == command.id).payload
            assert sum(event.action == "workload.dispatch_reservation_failed" for event in events) == 1

        # Expire the real durable claim, then let the repository grant a new one.
        async with repository.transaction() as tx:
            current = await tx.get_outbox(command.id)
            assert current is not None and current.claimed_by == controller.worker_id
            await tx.put_outbox(current.model_copy(update={"claimed_until": utcnow() - timedelta(seconds=1)}))
        reclaimed = await repository.claim_outbox_id(
            worker_id=successor.worker_id, command_id=command.id, lease_seconds=60
        )
        assert reclaimed is not None and reclaimed.attempts == current.attempts + 1
        if window == "before_settlement":
            successor_delivery = asyncio.create_task(successor._dispatch_claimed_outbox(reclaimed))
            assert await asyncio.to_thread(successor_entered.wait, 5), "successor submit did not start"

        owned, leases, commands, events = await _snapshot(repository, seeded.id)
        owned_command = next(item for item in commands if item.id == command.id)
        assert owned.state is WorkloadState.LEASED and owned.remote_id is None
        assert owned.dispatch_reservation is not None
        assert owned.dispatch_reservation.lease_ids == tuple(lease.id for lease in leases)
        assert all(lease.state is LeaseState.PENDING and lease.settled_at is None for lease in leases)
        assert owned_command.state is OutboxState.PROCESSING
        assert owned_command.claimed_by == successor.worker_id
        assert owned_command.payload["dispatch_reservation"] == owned.dispatch_reservation.model_dump(mode="json")
        if window == "before_release":
            assert owned.dispatch_reservation == original.dispatch_reservation
        else:
            assert owned.dispatch_reservation != original.dispatch_reservation

        first_resume.set()
        settlement_resume.set()
        assert await asyncio.wait_for(first_delivery, timeout=5) == 1
        after, after_leases, after_commands, after_events = await _snapshot(repository, seeded.id)
        # Compare complete durable rows, not just mock calls or a final status.
        assert after == owned, "stale refusal changed the successor-owned workload"
        assert after_leases == leases, "stale refusal released or settled successor leases"
        assert next(item for item in after_commands if item.id == command.id) == owned_command
        assert after_events == events, "stale refusal emitted terminal/settlement/dispatch-failed audit"

        if successor_delivery is None:
            successor_delivery = asyncio.create_task(successor._dispatch_claimed_outbox(reclaimed))
            assert await asyncio.to_thread(successor_entered.wait, 5)
        successor_resume.set()
        await asyncio.wait_for(successor_delivery, timeout=5)
        recovered, active, commands, events = await _snapshot(repository, seeded.id)
        assert recovered.state is WorkloadState.DISPATCHING
        assert recovered.remote_id == f"{seeded.id}-a1-g1"
        # Recovery preserves the admission fence while advancing only the
        # delivery-attempt counter to the successor's real claim.
        assert recovered.dispatch_reservation == owned.dispatch_reservation.model_copy(
            update={"outbox_attempt": reclaimed.attempts}
        )
        assert all(lease.state is LeaseState.ACTIVE and lease.settled_at is None for lease in active)
        assert next(item for item in commands if item.id == command.id).state is OutboxState.SUCCEEDED
        assert not any(event.action in {"workload.failed", "lease.settled", "lease.released"} for event in events)
        assert len(agent.executions) == 1
    finally:
        first_resume.set()
        successor_resume.set()
        settlement_resume.set()
        await asyncio.gather(
            first_delivery, *([successor_delivery] if successor_delivery else []), return_exceptions=True
        )
