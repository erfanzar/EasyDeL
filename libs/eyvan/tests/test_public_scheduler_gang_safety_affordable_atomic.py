# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public scheduler regressions for gang safety and affordable atomic groups."""

from dataclasses import replace
from itertools import combinations, permutations

import pytest
from test_gang_preemption_is_atomic import _member
from test_scheduler import _cell, _workload

from eyvan.domain import (
    Budget,
    CheckpointMode,
    CheckpointPolicy,
    IdentityScope,
    InfeasibilityCode,
    PriorityBand,
    Quota,
    ResourceVector,
    Scheduler,
)
from eyvan.records import (
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    ResourceLimits,
    WorkloadDefinition,
    WorkloadRecord,
)
from eyvan.records import (
    ResourceGang as RecordGang,
)
from eyvan.records import (
    ResourceVector as RecordResources,
)
from eyvan.server.service.scheduling import CoreSchedulingService


def _gang_result(sibling):
    workload = _workload(
        "urgent",
        resources=ResourceVector(cpu=8),
        priority=PriorityBand.INTERACTIVE,
        required_labels=(("target", "yes"),),
    )
    return Scheduler().schedule(
        (workload,),
        (_cell("cell-a", labels=(("target", "yes"),)), _cell("cell-b")),
        leases=(_member("lease-a", "cell-a"), sibling),
    )


def test_unsafe_revocable_sibling_protects_the_whole_attempt() -> None:
    sibling = replace(_member("lease-b", "cell-b"), checkpoint_policy=CheckpointPolicy())
    for leases in permutations((_member("lease-a", "cell-a"), sibling)):
        result = Scheduler().schedule(
            (_workload("urgent", resources=ResourceVector(cpu=8)),),
            (_cell("cell-a"), _cell("cell-b")),
            leases=leases,
        )
        assert not result.preemptions
        assert not result.placements


def test_higher_priority_sibling_protects_the_whole_attempt() -> None:
    sibling = replace(_member("lease-b", "cell-b"), priority=PriorityBand.PRODUCTION)
    assert not _gang_result(sibling).preemptions


@pytest.mark.parametrize("priority", [PriorityBand.BATCH, PriorityBand.PRODUCTION])
def test_a_different_attempt_is_not_part_of_the_victim_gang(priority) -> None:
    sibling = replace(_member("lease-b", "cell-b"), priority=priority, attempt=2)
    result = _gang_result(sibling)
    assert len(result.preemptions) == 1
    assert tuple(victim.lease_id for victim in result.preemptions[0].victims) == ("lease-a",)


@pytest.mark.parametrize(
    "policy",
    [CheckpointPolicy(allow_forced_termination=True), CheckpointPolicy(mode=CheckpointMode.ON_PREEMPTION)],
)
def test_every_eligible_same_attempt_sibling_remains_a_victim(policy) -> None:
    sibling = replace(_member("lease-b", "cell-b"), checkpoint_policy=policy)
    result = _gang_result(sibling)
    assert len(result.preemptions) == 1
    assert {victim.lease_id for victim in result.preemptions[0].victims} == {"lease-a", "lease-b"}


def _group_workload():
    return _workload(
        "group",
        resources=ResourceVector(tpu=1),
        cell_group_count=2,
        preferred_cells=("expensive",),
        estimated_duration_seconds=3600,
    )


def _group_cells():
    return tuple(
        _cell(name, capacity=ResourceVector(tpu=1), unit_prices=ResourceVector(tpu=price))
        for name, price in (("expensive", 100), ("cheap-a", 1), ("cheap-b", 1))
    )


def test_atomic_group_finds_an_affordable_pair_in_every_input_order() -> None:
    budget = Budget(IdentityScope("tenant", "project"), limit=2)
    for cells in permutations(_group_cells()):
        result = Scheduler().schedule((_group_workload(),), cells, budgets=(budget,))
        assert not result.infeasibilities
        assert len(result.placements) == 1
        leases = result.placements[0].all_leases
        assert tuple(lease.cell_id for lease in leases) == ("cheap-a", "cheap-b")
        assert sum(lease.reserved_cost for lease in leases) == 2
        assert ResourceVector.total(lease.resource_hours_reserved for lease in leases) == ResourceVector(tpu=2)


def test_atomic_group_keeps_the_best_soft_preference_that_has_an_affordable_completion() -> None:
    workload = replace(_group_workload(), preferred_cells=("expensive", "preferred", "second"))
    cells = (
        *_group_cells(),
        _cell("preferred", capacity=ResourceVector(tpu=1), unit_prices=ResourceVector(tpu=2)),
        _cell("second", capacity=ResourceVector(tpu=1), unit_prices=ResourceVector(tpu=4)),
    )
    for order in (cells, tuple(reversed(cells))):
        result = Scheduler().schedule((workload,), order, budgets=(Budget(IdentityScope("tenant"), limit=3),))
        assert not result.infeasibilities
        assert tuple(lease.cell_id for lease in result.placements[0].all_leases) == ("preferred", "cheap-a")


@pytest.mark.parametrize("budget_limit", [101, None])
def test_atomic_group_preserves_affordable_preferred_selection(budget_limit) -> None:
    budgets = () if budget_limit is None else (Budget(IdentityScope("tenant"), limit=budget_limit),)
    result = Scheduler().schedule((_group_workload(),), _group_cells(), budgets=budgets)
    assert tuple(lease.cell_id for lease in result.placements[0].all_leases) == ("expensive", "cheap-a")


@pytest.mark.parametrize(
    "budget",
    [
        Budget(IdentityScope("tenant"), limit=1),
        Budget(IdentityScope("tenant"), limit=102, accelerator_hour_limits=ResourceVector(tpu=1)),
        Budget(IdentityScope("tenant"), limit=3, spent=2),
    ],
)
def test_atomic_group_still_rejects_an_unaffordable_complete_group(budget) -> None:
    result = Scheduler().schedule((_group_workload(),), _group_cells(), budgets=(budget,))
    assert not result.placements
    assert not result.preemptions
    assert result.infeasibilities[0].code is InfeasibilityCode.BUDGET_EXCEEDED


def test_atomic_group_fallback_respects_active_reservations_and_ancestor_limits() -> None:
    running_cell = _cell("running", capacity=ResourceVector(cpu=8))
    running = replace(
        _member("running-lease", "running", workload_id="running-job"),
        reserved_cost=1,
        resource_hours_reserved=ResourceVector(tpu=1),
    )
    budgets = (
        Budget(IdentityScope("tenant"), limit=3, accelerator_hour_limits=ResourceVector(tpu=3)),
        Budget(IdentityScope("tenant", "project"), limit=102),
    )
    result = Scheduler().schedule(
        (_group_workload(),), (*_group_cells(), running_cell), leases=(running,), budgets=budgets
    )
    assert tuple(lease.cell_id for lease in result.placements[0].all_leases) == ("cheap-a", "cheap-b")
    rejected = Scheduler().schedule(
        (_group_workload(),),
        (*_group_cells(), running_cell),
        leases=(running,),
        budgets=(replace(budgets[0], accelerator_hour_limits=ResourceVector(tpu=2)), budgets[1]),
    )
    assert not rejected.placements
    assert rejected.infeasibilities[0].code is InfeasibilityCode.BUDGET_EXCEEDED


def test_atomic_group_fallback_does_not_bypass_quota() -> None:
    result = Scheduler().schedule(
        (_group_workload(),),
        _group_cells(),
        quotas=(Quota(IdentityScope("tenant"), ResourceVector(tpu=1)),),
        budgets=(Budget(IdentityScope("tenant"), limit=2),),
    )
    assert not result.placements
    assert result.infeasibilities[0].code is InfeasibilityCode.QUOTA_EXCEEDED


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("prices", [(9, 4, 3, 2, 1), (0, 2, 0, 4, 1), (3, 3, 3, 3, 3)])
def test_atomic_affordable_preference_matches_bounded_exhaustive_groups(count, prices) -> None:
    cells = tuple(
        _cell(f"cell-{index}", capacity=ResourceVector(tpu=1), unit_prices=ResourceVector(tpu=price))
        for index, price in enumerate(prices)
    )
    # Unique preference ranks make the expected score sequence explicit without
    # calling the implementation's private scoring or affordability helpers.
    workload = replace(_group_workload(), cell_group_count=count, preferred_cells=tuple(cell.cell_id for cell in cells))
    for limit in range(sum(prices) + 1):
        affordable = [
            group for group in combinations(range(len(prices)), count) if sum(prices[index] for index in group) <= limit
        ]
        result = Scheduler().schedule(
            (workload,), tuple(reversed(cells)), budgets=(Budget(IdentityScope("tenant"), limit=limit),)
        )
        assert bool(result.placements) == bool(affordable)
        if affordable:
            assert not result.infeasibilities
            assert tuple(lease.cell_id for lease in result.placements[0].all_leases) == tuple(
                f"cell-{index}" for index in min(affordable)
            )
        else:
            assert result.infeasibilities[0].code is InfeasibilityCode.BUDGET_EXCEEDED


def test_validated_record_adapter_finds_the_affordable_atomic_group() -> None:
    workload = WorkloadRecord(
        id="group",
        org_id="tenant",
        project_id="project",
        created_by="user",
        spec=WorkloadDefinition(
            name="group",
            execution=CommandExecution(command="true"),
            gang=RecordGang(per_replica=RecordResources(cpu=0, memory_gib=0, tpu=1)),
            cell_group_count=2,
            preferred_cells=("expensive",),
            estimated_duration_seconds=3600,
        ),
    )
    cells = tuple(
        CellRecord(
            id=cell.cell_id,
            provider="local",
            region="test",
            state=CellState.READY,
            capacity=ResourceLimits(tpu=1),
            allocatable=ResourceLimits(tpu=1),
            unit_prices={"tpu": float(cell.unit_prices["tpu"])},
        )
        for cell in _group_cells()
    )
    budgets = (BudgetRecord(org_id="tenant", project_id="project", limit=2, updated_by="operator"),)
    result = CoreSchedulingService().schedule((workload,), cells, budgets=budgets)
    assert not result.infeasibilities
    assert tuple(lease.cell_id for lease in result.placements[0].all_leases) == ("cheap-a", "cheap-b")
