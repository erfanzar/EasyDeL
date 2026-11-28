# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ordinary placement preserves its decision without materializing losing leases."""

from decimal import Decimal
from itertools import permutations

import pytest

from eyvan.domain import (
    CapacityCell,
    CapacityClass,
    CapacityLease,
    ResourceGang,
    ResourceVector,
    Scheduler,
    WorkloadIdentity,
    WorkloadSpec,
)


class CountingScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.materialized_cells = []

    def _make_lease(
        self, workload, requested, cell, hourly_cost, reserved_cost, resource_hours, **kwargs
    ) -> CapacityLease:
        self.materialized_cells.append(cell.cell_id)
        return super()._make_lease(workload, requested, cell, hourly_cost, reserved_cost, resource_hours, **kwargs)


@pytest.mark.parametrize("order", list(permutations(("a", "b", "c"))))
@pytest.mark.parametrize("preferred", [(), ("b",)])
def test_only_the_selected_ordinary_cell_materializes_a_lease(order, preferred) -> None:
    cells = {
        name: CapacityCell(name, ResourceVector(cpu=2), CapacityClass.SPOT, unit_prices=ResourceVector(cpu=2))
        for name in order
    }
    workload = WorkloadSpec(
        "materialize-once",
        WorkloadIdentity("tenant", "project", "user"),
        (ResourceGang("main", 1, ResourceVector(cpu=1)),),
        preferred_cells=preferred,
        estimated_duration_seconds=1800,
    )
    selected = "b" if preferred else "a"
    reference = Scheduler().schedule([workload], [cells[selected]])
    scheduler = CountingScheduler()
    result = scheduler.schedule([workload], cells.values())
    assert result == reference
    lease = result.placements[0].lease
    assert lease.cell_id == selected
    assert lease.hourly_cost == 2 and lease.reserved_cost == 1
    assert lease.resource_hours_reserved == ResourceVector(cpu=Decimal("0.5"))
    assert lease.resources == ResourceVector(cpu=1)
    assert scheduler.materialized_cells == [selected]
