# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cancellation must not steal a descendant's executing preemption ownership."""

import pytest
from test_group_lease_lifecycle import _preemption, _workload
from test_running_retry_properties import OPERATOR, _snapshot

from eyvan.records import LeaseState, WorkloadState
from eyvan.server.controller._support import _PreemptionVictimOutcome
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.service.errors import ConflictError


@pytest.fixture(params=["memory", "local"])
async def repository(request, tmp_path):
    repo = InMemoryRepository() if request.param == "memory" else LocalJsonRepository(tmp_path / "state.json")
    await repo.open()
    try:
        yield repo
    finally:
        await repo.close()


async def _executing_child(repository, depth=1):
    controller, command = await _preemption(repository)
    prepared = await controller._prepare_preemption_plan(command)
    assert prepared is not None
    stopping = await controller._advance_preemption_to_stopping(prepared, {}, {})
    async with repository.transaction() as tx:
        victim = await tx.get_workload("victim")
        assert victim is not None
        root = _workload("parent", victim.spec)
        await tx.put_workload(root)
        parent_id = root.id
        if depth == 2:
            intermediate = _workload("intermediate", victim.spec, parent_workload_id=root.id)
            await tx.put_workload(intermediate)
            parent_id = intermediate.id
        await tx.put_workload(victim.model_copy(update={"parent_workload_id": parent_id}))
    return controller, stopping


async def _finish_and_replay(controller, stopping):
    for victim in controller._preemption_victims(stopping):
        stopping = await controller._record_preemption_victim_stopped(
            stopping.id, victim.lease_id, _PreemptionVictimOutcome.PREEMPTED
        )
    await controller._finalize_preemption_plan(stopping, {})
    settled = await _snapshot(controller.repository, "victim")
    target = (await _snapshot(controller.repository, "priority"))["workload"]
    assert settled["workload"].state is WorkloadState.PENDING
    assert settled["workload"].generation == 2
    assert settled["workload"].preemption_id is None
    assert settled["workload"].lease_id is None and settled["workload"].lease_ids == ()
    assert all(lease.state is LeaseState.RELEASED and lease.settled_at for lease in settled["leases"].values())
    assert sum(event.action == "lease.settled" for event in settled["events"]) == 2
    assert target.state is WorkloadState.PENDING and target.preemption_id is None
    assert settled["outbox"][stopping.id].payload["execution_completed"] is True
    await controller._finalize_preemption_plan(stopping, {})
    assert await _snapshot(controller.repository, "victim") == settled
    assert (await _snapshot(controller.repository, "priority"))["workload"] == target


@pytest.mark.parametrize("depth", [1, 2])
async def test_cascade_rejects_executing_descendant_atomically_and_preemption_recovers(repository, depth):
    controller, stopping = await _executing_child(repository, depth)
    before = await _snapshot(repository, "victim")
    async with repository.transaction() as tx:
        parents_before = {item.id: item for item in await tx.list_workloads() if item.id != "victim"}
    with pytest.raises(ConflictError, match="preemption is already executing"):
        await controller.service.cancel_workload(OPERATOR, "victim", idempotency_key="direct-child")
    with pytest.raises(ConflictError, match="preemption is already executing"):
        await controller.service.cancel_workload(OPERATOR, "parent", idempotency_key="cancel-tree")
    assert await _snapshot(repository, "victim") == before
    async with repository.transaction() as tx:
        assert {item.id: item for item in await tx.list_workloads() if item.id != "victim"} == parents_before
    await _finish_and_replay(controller, stopping)
    # Refusal must not bind the cancellation key: retry the same operation once
    # its preemption owner has finished and cancellation can own the subtree.
    parent, replayed = await controller.service.cancel_workload(OPERATOR, "parent", idempotency_key="cancel-tree")
    assert parent.state is WorkloadState.CANCELLED and not replayed
    assert (await _snapshot(repository, "victim"))["workload"].state is WorkloadState.CANCELLED


async def test_non_cascading_parent_cancel_preserves_executing_child(repository):
    controller, stopping = await _executing_child(repository)
    before = await _snapshot(repository, "victim")
    parent, replayed = await controller.service.cancel_workload(
        OPERATOR, "parent", idempotency_key="parent-only", cascade=False
    )
    assert parent.state is WorkloadState.CANCELLED and not replayed
    after = await _snapshot(repository, "victim")
    for key in ("workload", "leases", "outbox", "cells", "budget"):
        assert after[key] == before[key]
    await _finish_and_replay(controller, stopping)


async def test_terminal_child_does_not_block_parent_cancellation(repository):
    controller, stopping = await _executing_child(repository)
    for victim in controller._preemption_victims(stopping):
        stopping = await controller._record_preemption_victim_stopped(
            stopping.id, victim.lease_id, _PreemptionVictimOutcome.SUCCEEDED
        )
    await controller._finalize_preemption_plan(stopping, {})
    before = await _snapshot(repository, "victim")
    assert before["workload"].state is WorkloadState.SUCCEEDED
    parent, replayed = await controller.service.cancel_workload(OPERATOR, "parent", idempotency_key="terminal-tree")
    assert parent.state is WorkloadState.CANCELLED and not replayed
    after = await _snapshot(repository, "victim")
    for key in ("workload", "leases", "outbox", "cells", "budget"):
        assert after[key] == before[key]
