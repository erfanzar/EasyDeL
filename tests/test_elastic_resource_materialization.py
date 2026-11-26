# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Elastic candidate accounting reuses its validated immutable resource vector."""

from itertools import permutations

import pytest
from test_scheduler import _cell, _workload

from eyvan.domain import PlacementMode, ResourceVector, Scheduler, WorkerResourceScope


class ResourceTrackingScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.validated = {}
        self.reused = []

    def _static_cell_failure(self, workload, requested, cell, **kwargs):
        self.validated[cell.cell_id] = requested
        return super()._static_cell_failure(workload, requested, cell, **kwargs)

    def _cost(self, workload, requested, cell):
        self.reused.append(requested is self.validated[cell.cell_id])
        return super()._cost(workload, requested, cell)


@pytest.mark.parametrize("order", list(permutations((0, 1, 2))))
def test_elastic_costing_reuses_the_validated_resources_without_changing_the_pack(order):
    workload = _workload(
        "reuse",
        resources=ResourceVector(tpu=8),
        worker_count=4,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    cells = tuple(
        _cell(f"cell-{index}", capacity=ResourceVector(tpu=8 * width), worker_count=width)
        for index, width in enumerate((3, 2, 2))
    )
    scheduler = ResourceTrackingScheduler()
    result = scheduler.schedule((workload,), tuple(cells[index] for index in order))
    assert not result.infeasibilities
    leases = result.placements[0].all_leases
    assert {lease.cell_id for lease in leases} == {"cell-1", "cell-2"}
    assert all(lease.resources == ResourceVector(tpu=16) for lease in leases)
    assert scheduler.reused == [True, True, True]
