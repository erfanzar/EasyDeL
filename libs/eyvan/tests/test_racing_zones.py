# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Asking several zones for capacity at once.

Parallel candidate requests avoid waiting through each unavailable zone before
trying one that can grant usable Spot capacity.

Racing costs money in every zone until one wins, so it is off by default and
bounded twice: by the operator's declared width, and by the workload's own
alternates -- pools the submission path already proved interchangeable. These
tests are mostly about those bounds, and about the loser never being asked for
more than it was already asked for.
"""

from __future__ import annotations

import pytest

from eyvan.capacity.planning import POOL_ALTERNATES_LABEL, CapacitySimulation
from eyvan.domain import SchedulingResult
from eyvan.records import (
    CapacityPoolRecord,
    CommandExecution,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)

ZONES = ("us-east5-a", "us-east5-b", "us-central2-b")


def _pool(zone: str) -> CapacityPoolRecord:
    resources = ResourceVector(cpu=0, memory_gib=0, tpu=8)
    return CapacityPoolRecord(
        id=f"pool-{zone}",
        fleet_name="research-fleet",
        node_type=f"v6e-8-{zone}",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region=zone.rsplit("-", 1)[0],
        zone=zone,
        capacity=resources,
        allocatable=resources,
        accelerator_type="v6e-8",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        min_cells=0,
        initial_cells=0,
        max_cells=4,
        enabled=True,
    )


def _workload(*, alternates: tuple[str, ...], cell_group_count: int = 1) -> WorkloadRecord:
    labels = {"eyvan_pool": f"pool-{ZONES[0]}"}
    if alternates:
        labels[POOL_ALTERNATES_LABEL] = ",".join(alternates)
    return WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="racer",
            execution=CommandExecution(command="python train.py"),
            required_labels=labels,
            cell_group_count=cell_group_count,
        ),
        state=WorkloadState.PENDING,
        created_by="alice",
    )


def _demand(*, race_width: int, alternates: tuple[str, ...], cell_group_count: int = 1) -> dict[str, tuple[str, ...]]:
    """Return which workloads each pool is asked to provide capacity for."""

    pools = tuple(_pool(zone) for zone in ZONES)
    workload = _workload(alternates=alternates, cell_group_count=cell_group_count)
    # One virtual cell per slice, which is what the scheduler produces: a
    # two-slice group is placed as two leases in one pool.
    members = [f"virtual-{index}" for index in range(cell_group_count)]
    simulation = CapacitySimulation(
        cells=(),
        pool_by_cell_id={member: f"pool-{ZONES[0]}" for member in members},
        virtual_worker_count_by_cell_id=dict.fromkeys(members, 1),
    )

    class _Lease:
        def __init__(self, cell_id: str) -> None:
            self.cell_id = cell_id
            self.workload_id = "wl-1"

    class _Placement:
        all_leases = tuple(_Lease(member) for member in members)

    result = SchedulingResult.__new__(SchedulingResult)
    object.__setattr__(result, "placements", (_Placement(),))
    demand = simulation.demand(pools, (), result, (workload,), None, race_width=race_width)
    return {pool_id: entry.virtual_workload_ids for pool_id, entry in demand.items()}


# --------------------------------------------------------------------------- off by default


def test_one_zone_is_asked_when_racing_is_off() -> None:
    """The default, and every fleet that has ever run. Racing spends money."""

    asked = _demand(race_width=1, alternates=(f"pool-{ZONES[1]}", f"pool-{ZONES[2]}"))

    assert asked[f"pool-{ZONES[0]}"] == ("wl-1",)
    assert asked[f"pool-{ZONES[1]}"] == ()
    assert asked[f"pool-{ZONES[2]}"] == ()


# --------------------------------------------------------------------------- racing


def test_a_race_asks_the_alternates_too() -> None:
    """The point: find out which zone will actually grant, in parallel."""

    asked = _demand(race_width=3, alternates=(f"pool-{ZONES[1]}", f"pool-{ZONES[2]}"))

    assert asked[f"pool-{ZONES[0]}"] == ("wl-1",)
    assert asked[f"pool-{ZONES[1]}"] == ("wl-1",)
    assert asked[f"pool-{ZONES[2]}"] == ("wl-1",)


def test_the_width_bounds_how_many_zones_are_asked() -> None:
    """Two of three is a real choice: race a bit, without paying for the fleet."""

    asked = _demand(race_width=2, alternates=(f"pool-{ZONES[1]}", f"pool-{ZONES[2]}"))

    raced = [pool_id for pool_id, workloads in asked.items() if workloads]
    assert len(raced) == 2


def test_a_workload_with_no_alternates_is_never_raced() -> None:
    """Nothing here invents a candidate; the fleet declares interchangeability."""

    asked = _demand(race_width=3, alternates=())

    assert [pool_id for pool_id, workloads in asked.items() if workloads] == [f"pool-{ZONES[0]}"]


def test_a_multislice_workload_races_its_whole_shape() -> None:
    """Half a gang in another zone is not a candidate placement; it is waste."""

    asked = _demand(race_width=2, alternates=(f"pool-{ZONES[1]}",), cell_group_count=2)

    assert asked[f"pool-{ZONES[1]}"] == ("wl-1", "wl-1")


def test_a_race_does_not_accumulate_requests_across_ticks() -> None:
    """Reconcile runs every few seconds; an unbounded ask would fill the fleet.

    The second call stands in for the second tick: the alternate already carries
    this workload's request, and asking again would launch another machine for a
    job that needs one.
    """

    pools = tuple(_pool(zone) for zone in ZONES)
    workload = _workload(alternates=(f"pool-{ZONES[1]}",))
    counts: dict[str, list[int]] = {pool.id: [] for pool in pools}
    ids: dict[str, list[str]] = {pool.id: [] for pool in pools}
    counts[f"pool-{ZONES[0]}"].append(1)
    ids[f"pool-{ZONES[0]}"].append("wl-1")

    for _tick in range(3):
        CapacitySimulation._race_alternates(
            counts,
            ids,
            by_workload_id={"wl-1": workload},
            race_width=2,
        )

    assert ids[f"pool-{ZONES[1]}"] == ["wl-1"], "one request per member, however many ticks run"


@pytest.mark.parametrize("width", [1, 2, 3, 10])
def test_racing_never_asks_a_pool_that_is_not_a_candidate(width: int) -> None:
    """A width larger than the alternates must not reach outside them."""

    asked = _demand(race_width=width, alternates=(f"pool-{ZONES[1]}",))

    assert asked[f"pool-{ZONES[2]}"] == ()


def test_a_race_never_re_asks_a_zone_that_denied_the_workload() -> None:
    """Use race slots for unasked alternates rather than re-requesting a zone that denied admission."""

    pools = tuple(_pool(zone) for zone in ZONES)
    workload = _workload(alternates=(f"pool-{ZONES[0]}", f"pool-{ZONES[2]}"))
    workload = workload.model_copy(
        update={
            "spec": workload.spec.model_copy(
                update={
                    "required_labels": {
                        "eyvan_pool": f"pool-{ZONES[1]}",
                        POOL_ALTERNATES_LABEL: f"pool-{ZONES[0]},pool-{ZONES[2]}",
                    }
                }
            )
        }
    )
    simulation = CapacitySimulation(
        cells=(),
        pool_by_cell_id={"virtual-0": f"pool-{ZONES[1]}"},
        virtual_worker_count_by_cell_id={"virtual-0": 1},
        queue_denied_pool_ids={"wl-1": frozenset({f"pool-{ZONES[0]}"})},
    )

    class _Lease:
        cell_id = "virtual-0"
        workload_id = "wl-1"

    class _Placement:
        all_leases = (_Lease(),)

    result = SchedulingResult.__new__(SchedulingResult)
    object.__setattr__(result, "placements", (_Placement(),))
    demand = simulation.demand(pools, (), result, (workload,), None, race_width=2)
    asked = {pool_id: entry.virtual_workload_ids for pool_id, entry in demand.items()}

    assert asked[f"pool-{ZONES[0]}"] == (), "the denying zone is not asked again"
    assert asked[f"pool-{ZONES[2]}"] == ("wl-1",), "its race slot goes to a zone that has not refused"


def test_a_race_never_asks_a_pool_past_the_acquisition_bound() -> None:
    pools = tuple(_pool(zone) for zone in ZONES)
    workload = _workload(alternates=(f"pool-{ZONES[1]}", f"pool-{ZONES[2]}"))
    simulation = CapacitySimulation(
        cells=(),
        pool_by_cell_id={"virtual-0": f"pool-{ZONES[0]}"},
        virtual_worker_count_by_cell_id={"virtual-0": 1},
        exhausted_pool_ids=frozenset({f"pool-{ZONES[1]}"}),
    )

    class _Lease:
        cell_id = "virtual-0"
        workload_id = "wl-1"

    class _Placement:
        all_leases = (_Lease(),)

    result = SchedulingResult.__new__(SchedulingResult)
    object.__setattr__(result, "placements", (_Placement(),))
    demand = simulation.demand(pools, (), result, (workload,), None, race_width=3)
    asked = {pool_id: entry.virtual_workload_ids for pool_id, entry in demand.items()}

    assert asked[f"pool-{ZONES[1]}"] == ()
    assert asked[f"pool-{ZONES[2]}"] == ("wl-1",)


def test_a_workload_naming_no_pool_is_not_offered_a_zone_that_denied_it() -> None:
    """Exclude denied zones from generic pool selection even when the workload names no pool."""

    from eyvan.capacity.drivers.base import CapacityProviderFailureCode
    from eyvan.records import CellState, utcnow

    pools = tuple(_pool(zone) for zone in ZONES)
    generic = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="batch-0",
            execution=CommandExecution(command="true"),
            capacity_class="spot",
            gang={"replicas": 1, "per_replica": {"cpu": 0, "memory_gib": 0, "tpu": 8}},
        ),
        state=WorkloadState.PENDING,
        created_by="alice",
    )
    denied = (
        pools[0]
        .make_cell("pool-us-east5-a-0001", worker_count=1, autoscale_request_workload_id="wl-1")
        .model_copy(
            update={
                "state": CellState.UNAVAILABLE,
                "provider_failure_code": CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
            }
        )
    )

    simulation = CapacitySimulation.build(pools, (denied,), (generic,), (), utcnow())

    offered = {
        pool_id
        for cell_id, pool_id in simulation.pool_by_cell_id.items()
        if cell_id in simulation.virtual_worker_count_by_cell_id
    }
    assert f"pool-{ZONES[0]}" not in offered
    assert offered == {f"pool-{ZONES[1]}", f"pool-{ZONES[2]}"}


def test_a_request_the_provider_refused_holds_no_race_slot_and_is_not_asked_again() -> None:
    """Free race slots after provider refusal without re-requesting the stalled pools."""

    from eyvan.capacity.drivers.base import CapacityProviderFailureCode
    from eyvan.records import CellState, utcnow

    pools = tuple(_pool(zone) for zone in ZONES)
    workload = _workload(alternates=(f"pool-{ZONES[1]}", f"pool-{ZONES[2]}"))
    stocked_out = (
        pools[1]
        .make_cell("pool-us-east5-b-0001", worker_count=1, autoscale_request_workload_id="wl-1")
        .model_copy(
            update={"state": CellState.UNAVAILABLE, "provider_failure_code": CapacityProviderFailureCode.STOCKOUT}
        )
    )

    built = CapacitySimulation.build(pools, (stocked_out,), (workload,), (), utcnow())
    assert built.stalled_pool_ids == {"wl-1": frozenset({f"pool-{ZONES[1]}"})}
    assert built.owned_pool_ids.get("wl-1", frozenset()) == frozenset()

    simulation = CapacitySimulation(
        cells=(),
        pool_by_cell_id={"virtual-0": f"pool-{ZONES[0]}"},
        virtual_worker_count_by_cell_id={"virtual-0": 1},
        stalled_pool_ids=built.stalled_pool_ids,
    )

    class _Lease:
        cell_id = "virtual-0"
        workload_id = "wl-1"

    class _Placement:
        all_leases = (_Lease(),)

    result = SchedulingResult.__new__(SchedulingResult)
    object.__setattr__(result, "placements", (_Placement(),))
    demand = simulation.demand(pools, (), result, (workload,), None, race_width=2)
    asked = {pool_id: entry.virtual_workload_ids for pool_id, entry in demand.items()}

    assert asked[f"pool-{ZONES[1]}"] == (), "the stocked-out zone is not asked again"
    assert asked[f"pool-{ZONES[2]}"] == ("wl-1",), "and its failure did not use up the race slot"
