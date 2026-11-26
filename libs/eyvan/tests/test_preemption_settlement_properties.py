# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded service-state loss histories; no controller or provider dispatch."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_group_lease_lifecycle import repository as _repository_fixture
from test_running_retry_properties import ADMIN, _run, _seed, _snapshot

from eyvan.records import LeaseState, WorkloadState
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError

repository = _repository_fixture


async def _history(repository, losses: list[tuple[str, bool]], rate: int, held_hours: int) -> None:
    key = f"loss-{len(losses)}-{rate}-{held_hours}"
    workload_id, cell = await _seed(repository, key, max_failure_retries=0)
    initial = await _snapshot(repository, workload_id)
    initial_budget = initial["budget"]
    released = {}
    counters = {"provider": 0, "capacity": 0, "recycle": 0}
    for index, (kind, checkpointing) in enumerate(losses):
        lease = await _run(repository, workload_id, cell, key, rate, held_hours)
        state = WorkloadState.RUNNING
        checkpoint = None
        if checkpointing:
            state = WorkloadState.CHECKPOINTING
            await ControlPlaneService(repository).transition_workload(ADMIN, workload_id, state)
            checkpoint = f"gs://test-checkpoints/{key}/{index}"
        before = await _snapshot(repository, workload_id)
        workload = before["workload"]
        assert workload.state is state
        assert before["budget"].revision == initial_budget.revision + index

        # These histories do not establish priority-preemption ownership.
        with pytest.raises(ConflictError, match="preemption no longer owns"):
            await ControlPlaneService(repository).settle_preempted_workload(
                ADMIN, workload_id, require_preemption_owner=True
            )
        assert await _snapshot(repository, workload_id) == before

        resumed, settled, budget = await ControlPlaneService(repository).settle_preempted_workload(
            ADMIN,
            workload_id,
            checkpoint_uri=checkpoint,
            provider_preempted=kind == "provider",
            recycle=kind == "recycle",
            expected_generation=workload.generation,
            expected_lease_id=lease.id,
            expected_state=state,
        )
        counters[kind] += 1
        assert resumed.state is WorkloadState.PENDING
        assert resumed.generation == workload.generation + 1
        assert resumed.attempt == workload.attempt + 1
        assert resumed.preemption_count == counters["provider"]
        assert resumed.capacity_loss_count == counters["capacity"]
        assert resumed.failure_count == 0
        assert resumed.lease_id is None and resumed.lease_ids == () and resumed.remote_id is None
        assert resumed.checkpoint_uri == checkpoint
        assert settled.id == lease.id and settled.state is LeaseState.RELEASED
        assert settled.settled_at is not None and settled.settlement_reason == "preempted"
        assert rate * held_hours <= settled.settled_cost <= rate * (held_hours + 0.01)
        assert 2 * held_hours <= settled.accelerator_hours_settled["tpu"] <= 2 * (held_hours + 0.01)
        released[lease.id] = settled
        after = await _snapshot(repository, workload_id)
        assert after["workload"] == resumed and after["budget"] == budget
        assert after["leases"] == released
        assert budget.revision == initial_budget.revision + index + 1
        assert budget.spent == pytest.approx(initial_budget.spent + sum(row.settled_cost for row in released.values()))
        assert budget.accelerator_hours_spent["tpu"] == pytest.approx(
            initial_budget.accelerator_hours_spent["tpu"]
            + sum(row.accelerator_hours_settled["tpu"] for row in released.values())
        )
        replayed, replayed_lease, replayed_budget = await ControlPlaneService(repository).settle_preempted_workload(
            ADMIN,
            workload_id,
            checkpoint_uri=checkpoint,
            provider_preempted=kind == "provider",
            recycle=kind == "recycle",
        )
        assert replayed == resumed and replayed_lease is None and replayed_budget == budget
        assert await _snapshot(repository, workload_id) == after
        with pytest.raises(ConflictError):
            await ControlPlaneService(repository).settle_preempted_workload(
                ADMIN,
                workload_id,
                expected_generation=workload.generation,
                expected_lease_id=lease.id,
                expected_state=state,
            )
        assert await _snapshot(repository, workload_id) == after
        actions = [event.action for event in after["events"]]
        assert actions.count("budget.lease_settled") == index + 1
        assert actions.count("lease.settled") == index + 1
        assert actions.count("workload.preempted") == index + 1


@settings(max_examples=25, deadline=None)
@given(
    losses=st.lists(
        st.tuples(st.sampled_from(["provider", "capacity", "recycle"]), st.booleans()), min_size=1, max_size=4
    ),
    rate=st.integers(min_value=0, max_value=30),
    held_hours=st.integers(min_value=1, max_value=3),
)
async def test_generated_preemption_settlements_conserve_budget(losses, rate, held_hours) -> None:
    await _history(InMemoryRepository(), losses, rate, held_hours)


@pytest.mark.parametrize(
    "losses,rate",
    [
        ([("provider", False)], 0),
        ([("capacity", True)], 13),
        ([("recycle", True)], 7),
        ([("provider", True), ("capacity", False), ("recycle", True), ("provider", False)], 17),
    ],
)
async def test_preemption_conservation_across_repositories(repository, losses, rate) -> None:
    await _history(repository, losses, rate, held_hours=2)
