# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cross-check bounded elastic budget decisions against exhaustive subsets."""

from itertools import combinations

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from test_scheduler import _cell, _workload

from eyvan.domain import Budget, IdentityScope, PlacementMode, ResourceVector, Scheduler, WorkerResourceScope

pytestmark = pytest.mark.slow


@settings(max_examples=200, deadline=None)
@given(
    supply=st.lists(st.tuples(st.integers(1, 5), st.integers(1, 10)), min_size=1, max_size=6),
    workers=st.integers(1, 12),
    limit=st.integers(1, 500),
)
# The whole three-worker cell is too expensive. Losing the cheaper two-worker
# frontier entry makes the affordable two-plus-one pack disappear.
@example(supply=[(3, 3), (2, 1), (2, 2), (1, 4)], workers=3, limit=48)
def test_budget_feasibility_matches_exhaustive_whole_cell_subsets(supply, workers, limit):
    cells = tuple(
        _cell(
            f"cell-{index}",
            capacity=ResourceVector(tpu=8 * width),
            worker_count=width,
            unit_prices=ResourceVector(tpu=price),
        )
        for index, (width, price) in enumerate(supply)
    )
    feasible = [
        subset
        for count in range(1, len(supply) + 1)
        for subset in combinations(range(len(supply)), count)
        if sum(supply[index][0] for index in subset) == workers
        and sum(8 * supply[index][0] * supply[index][1] for index in subset) <= limit
    ]
    workload = _workload(
        "pack",
        resources=ResourceVector(tpu=8),
        worker_count=workers,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
        estimated_duration_seconds=3600,
    )
    result = Scheduler().schedule((workload,), cells, budgets=(Budget(IdentityScope("tenant"), limit=limit),))
    assert bool(result.placements) == bool(feasible)
    if feasible:
        leases = result.placements[0].all_leases
        indices = [int(lease.cell_id.split("-")[1]) for lease in leases]
        assert len(set(indices)) == len(indices)
        assert sum(supply[index][0] for index in indices) == workers
        assert sum(8 * supply[index][0] * supply[index][1] for index in indices) <= limit
        assert not result.infeasibilities
