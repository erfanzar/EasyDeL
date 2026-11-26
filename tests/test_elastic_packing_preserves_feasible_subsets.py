# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from itertools import permutations

import pytest
from test_scheduler import _cell, _workload

from eyvan.domain import PlacementMode, ResourceVector, Scheduler, WorkerResourceScope


@pytest.mark.parametrize(("need", "widths", "expected"), [(4, (3, 2, 2), (2, 2)), (10, (6, 5, 5, 2, 2), (5, 5))])
def test_a_feasible_pack_is_found_without_sacrificing_the_fewest_worlds(need, widths, expected) -> None:
    workload = _workload(
        "pack",
        resources=ResourceVector(tpu=8),
        worker_count=need,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    cells = tuple(
        _cell(f"cell-{i}", capacity=ResourceVector(tpu=8 * width), worker_count=width) for i, width in enumerate(widths)
    )
    placements = set()
    for order in permutations(cells):
        result = Scheduler().schedule((workload,), order)
        assert not result.infeasibilities
        placement = result.placements[0]
        placements.add(tuple(lease.cell_id for lease in placement.all_leases))
        assert sorted(int(lease.resources["tpu"] / 8) for lease in placement.all_leases) == list(expected)
    assert len(placements) == 1


def test_an_affordable_elastic_pack_is_not_hidden_by_a_more_expensive_single_cell() -> None:
    from eyvan.domain import Budget, IdentityScope

    workload = _workload(
        "pack",
        resources=ResourceVector(tpu=8),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
        estimated_duration_seconds=3600,
    )
    cells = (
        _cell("expensive", capacity=ResourceVector(tpu=32), worker_count=4, unit_prices=ResourceVector(tpu=10)),
        _cell("cheap-a", capacity=ResourceVector(tpu=16), worker_count=2, unit_prices=ResourceVector(tpu=1)),
        _cell("cheap-b", capacity=ResourceVector(tpu=16), worker_count=2, unit_prices=ResourceVector(tpu=1)),
    )
    result = Scheduler().schedule((workload,), cells, budgets=(Budget(IdentityScope("tenant"), limit=40),))
    assert not result.infeasibilities
    assert {lease.cell_id for lease in result.placements[0].all_leases} == {"cheap-a", "cheap-b"}
