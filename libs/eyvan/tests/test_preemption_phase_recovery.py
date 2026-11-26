# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable phase replay after injected controller interruption, not OS-crash proof."""

from datetime import timedelta

import pytest
from test_group_lease_lifecycle import _preemption
from test_priority_preemption_properties import PerCellAgent, _controller, _requeue
from test_priority_preemption_properties import repository as _repository_fixture
from test_running_retry_properties import _snapshot

from eyvan.records import LeaseState, OutboxState, WorkloadState, utcnow

repository = _repository_fixture


class InjectedInterruption(BaseException):
    """Bypass ordinary RPC retry/abort handlers at a controlled preemption boundary."""


@pytest.mark.parametrize("boundary", ["stopping", "unrecorded-stop", "first-victim", "finalized"])
async def test_fresh_controller_recovers_durable_preemption_phase(repository, monkeypatch, boundary):
    first, command = await _preemption(repository)
    agents = {f"cell-{suffix}": PerCellAgent("RUNNING") for suffix in "ab"}
    first._agent_factory = lambda cell: agents[cell.id]
    async with repository.transaction() as tx:
        victim = await tx.get_workload("victim")
        await tx.put_workload(victim.model_copy(update={"remote_id": "remote-victim"}))
    before = await _snapshot(repository, "victim")
    target_before = (await _snapshot(repository, "priority"))["workload"]
    method_name = {
        "stopping": "_advance_preemption_to_stopping",
        "unrecorded-stop": "_record_preemption_victim_stopped",
        "first-victim": "_record_preemption_victim_stopped",
        "finalized": "_finalize_preemption_plan",
    }[boundary]
    original = getattr(first, method_name)
    interrupted = []

    async def interrupt_at_boundary(*args, **kwargs):
        # A stop may have completed remotely before its outcome is committed.
        # Other cases interrupt only after the original durable commit returns.
        if boundary != "unrecorded-stop":
            await original(*args, **kwargs)
        interrupted.append(boundary)
        raise InjectedInterruption(boundary)

    monkeypatch.setattr(first, method_name, interrupt_at_boundary)
    with pytest.raises(InjectedInterruption, match=boundary):
        await first._dispatch_preemption(command)
    assert interrupted == [boundary]
    partial = await _snapshot(repository, "victim")
    target_partial = (await _snapshot(repository, "priority"))["workload"]
    assert target_partial.state is WorkloadState.PENDING
    assert target_partial.preemption_id == (None if boundary == "finalized" else command.id)
    durable = partial["outbox"][command.id]
    assert durable.payload["phase"] == "stopping"
    assert (
        len(durable.payload.get("victim_outcomes", {}))
        == {
            "stopping": 0,
            "unrecorded-stop": 0,
            "first-victim": 1,
            "finalized": 2,
        }[boundary]
    )
    if boundary == "unrecorded-stop":
        assert len(agents["cell-a"].stops) == 1
        assert agents["cell-b"].stops == []
        assert agents["cell-a"].remote_states["remote-victim"].name == "STOPPED"
    if boundary != "finalized":
        assert not durable.payload.get("execution_completed", False)
        assert all(lease.settled_at is None for lease in partial["leases"].values())
    else:
        assert durable.payload["execution_completed"] is True
    # Preserve the durable phase and outcomes; only expire the interrupted
    # owner's claim. The replacement must reclaim through the real outbox API.
    async with repository.transaction() as tx:
        current = await tx.get_outbox(command.id)
        await tx.put_outbox(current.model_copy(update={"claimed_until": utcnow() - timedelta(seconds=1)}))
    completed_observations = {
        partial["leases"][lease_id].cell_id: list(agents[partial["leases"][lease_id].cell_id].observations)
        for lease_id in durable.payload.get("victim_outcomes", {})
    }
    replacement = _controller(repository, agents)
    assert replacement.worker_id != first.worker_id
    assert await replacement.dispatch_outbox_once() == 1
    for cell_id, observed in completed_observations.items():
        assert agents[cell_id].observations == observed
    recovered = await _snapshot(repository, "victim")
    victim = recovered["workload"]
    assert victim.state is WorkloadState.PENDING
    assert victim.attempt == victim.generation == 2
    assert victim.preemption_id is None and victim.lease_ids == ()
    assert all(lease.state is LeaseState.RELEASED and lease.settled_at for lease in recovered["leases"].values())
    assert recovered["cells"] == before["cells"]
    assert recovered["budget"] == before["budget"]  # Fixture is unpriced.
    assert sum(event.action == "lease.settled" for event in recovered["events"]) == 2
    assert sum(event.action == "workload.preempted" for event in recovered["events"]) == 1
    assert recovered["outbox"][command.id].state is OutboxState.SUCCEEDED
    for suffix, agent in zip("ab", agents.values(), strict=True):
        assert len(agent.stops) == 1
        cell_id, remote_id, request = agent.stops[0]
        assert (cell_id, remote_id) == (f"cell-{suffix}", "remote-victim")
        assert request.mutation.fencing_token == f"fence-{suffix}"
        assert request.mutation.generation == 1
    target_recovered = (await _snapshot(repository, "priority"))["workload"]
    assert target_recovered.state is WorkloadState.PENDING
    assert target_recovered.preemption_id is None
    for field in ("generation", "attempt", "lease_id", "lease_ids", "remote_id", "spec"):
        assert getattr(target_recovered, field) == getattr(target_before, field)
    observations = {cell: list(agent.observations) for cell, agent in agents.items()}
    await _requeue(repository, command.id)
    assert await _controller(repository, agents).dispatch_outbox_once() == 1
    replayed = await _snapshot(repository, "victim")
    for key in ("workload", "leases", "events", "cells", "budget"):
        assert replayed[key] == recovered[key]
    assert (await _snapshot(repository, "priority"))["workload"] == target_recovered
    assert {cell: agent.observations for cell, agent in agents.items()} == observations
    assert all(len(agent.stops) == 1 for agent in agents.values())


async def test_a_terminal_gang_sibling_cannot_erase_the_checkpoint_uri(repository):
    """Order must not decide whether a saved checkpoint survives.

    Both gang members share one workload id, so the terminal branch used to
    overwrite a rank's real checkpoint URI with None whenever it was processed
    after the rank that had checkpointed -- the resumed attempt then restarted
    from scratch with the checkpoint still sitting on the cell.
    """

    first, command = await _preemption(repository, checkpointable=True, remote_id="remote-victim")
    agents = {
        "cell-a": PerCellAgent("RUNNING"),
        "cell-b": PerCellAgent("SUCCEEDED"),
    }
    first._agent_factory = lambda cell: agents[cell.id]
    await first._dispatch_preemption(command)
    durable = (await _snapshot(repository, "victim"))["outbox"][command.id]
    assert durable.payload["phase"] == "stopping"
    assert durable.payload.get("checkpoint_uris", {}).get("victim") == "gs://checkpoints/gang"
