# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Property-based invariants for the scheduler over generated snapshots.

Unit tests pin scenarios; these properties pin the invariants every scenario
must satisfy no matter what the generator produces: whole cells only, exact
worker totals, no per-cell overcommit under contention, and input-order
independence.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import integers, lists
from test_scheduler import _cell, _workload

from eyvan.domain import PlacementMode, ResourceVector, Scheduler, WorkerResourceScope

_WORKERS_PER_CELL = integers(min_value=1, max_value=4)
_WORKER_COUNT = integers(min_value=1, max_value=10)

_SNAPSHOT = lists(
    _WORKERS_PER_CELL.map(lambda workers: (workers,)),
    min_size=1,
    max_size=5,
)


def _cells(sizes: list[tuple[int]]) -> tuple:
    return tuple(
        _cell(
            f"cell-{index}-{workers}w",
            capacity=ResourceVector(tpu=8 * workers),
            worker_count=workers,
        )
        for index, (workers,) in enumerate(sizes)
    )


def _workloads(requests: list[int]) -> tuple:
    return tuple(
        _workload(
            f"workload-{index}-{workers}w",
            resources=ResourceVector(tpu=8),
            worker_count=workers,
            worker_resource_scope=WorkerResourceScope.PER_WORKER,
            placement=PlacementMode.ELASTIC,
        )
        for index, workers in enumerate(requests)
    )


@settings(max_examples=120, deadline=None, suppress_health_check=list(HealthCheck))
@given(sizes=_SNAPSHOT, requests=lists(_WORKER_COUNT, min_size=1, max_size=4))
def test_elastic_placements_are_exact_and_never_overcommit_a_cell(sizes, requests) -> None:
    cells = _cells(sizes)
    workloads = _workloads(requests)
    cell_workers = {cell.cell_id: cell.worker_count for cell in cells}

    result = Scheduler().schedule(workloads, cells)

    placed_ids = {placement.workload_id for placement in result.placements}
    failed_ids = {failure.workload_id for failure in result.infeasibilities}
    assert placed_ids.isdisjoint(failed_ids)
    assert placed_ids | failed_ids == {workload.workload_id for workload in workloads}

    lease_uses_per_cell: dict[str, int] = {cell.cell_id: 0 for cell in cells}
    for placement in result.placements:
        workload = next(item for item in workloads if item.workload_id == placement.workload_id)
        # Elastic packs whole cells: the leased workers always total exactly
        # the request -- never a partial pack, never more than was asked.
        assert sum(cell_workers[lease.cell_id] for lease in placement.all_leases) == workload.worker_count
        for lease in placement.all_leases:
            assert lease.cell_id in lease_uses_per_cell
            lease_uses_per_cell[lease.cell_id] += 1

    # A whole cell is the unit of an elastic lease, so each cell appears at
    # most once across every decision in the cycle.
    assert all(uses <= 1 for uses in lease_uses_per_cell.values())
    total_supply = sum(workers for (workers,) in sizes)
    assert (
        sum(cell_workers[lease.cell_id] for placement in result.placements for lease in placement.all_leases)
        <= total_supply
    )


@settings(max_examples=120, deadline=None, suppress_health_check=list(HealthCheck))
@given(sizes=_SNAPSHOT, requests=lists(_WORKER_COUNT, min_size=1, max_size=4))
def test_elastic_schedule_is_independent_of_cell_and_workload_order(sizes, requests) -> None:
    cells = _cells(sizes)
    workloads = _workloads(requests)

    def decisions(workload_order, cell_order):
        result = Scheduler().schedule(workload_order, cell_order)
        return {
            placement.workload_id: tuple(sorted(lease.cell_id for lease in placement.all_leases))
            for placement in result.placements
        } | {failure.workload_id: ("infeasible",) for failure in result.infeasibilities}

    assert decisions(workloads, cells) == decisions(tuple(reversed(workloads)), tuple(reversed(cells)))


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
