# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Why this zone and not that one.

The belief state -- cooldowns, consecutive provider failures -- has chosen
which zone a cross-zone workload asks next since alternates existed, and said
so nowhere. "It is waiting on us-east5-b, which reclaimed it twice, while
us-central2-b sits idle" was answerable only by reading cell records by hand,
and on a spot-first fleet it is the question asked most often.

The property that matters is not the wording: it is that the report and the
decision cannot disagree, because a rationale that contradicts the choice is
worse than none.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.capacity.planning import POOL_ALTERNATES_LABEL, candidate_zone_report
from eyvan.records import (
    CapacityBeliefState,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)

NOW = utcnow()
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
    )


def _cell(
    cell_id: str,
    zone: str,
    *,
    reserved_for: str | None = None,
    cooling: bool = False,
    failures: int = 0,
    state: CellState = CellState.READY,
) -> CellRecord:
    # Minted from the pool template, the way real cells are, so the report's
    # supply gate -- the same one the decision uses -- recognises them.
    cell = _pool(zone).make_cell(cell_id, worker_count=1)
    return cell.model_copy(
        update={
            "state": state,
            "autoscale_request_workload_id": reserved_for,
            "availability": CapacityBeliefState(
                cooldown_until=NOW + timedelta(minutes=10) if cooling else None,
                consecutive_failures=failures,
            ),
        }
    )


def _workload(*, alternates: tuple[str, ...] = ZONES[1:]) -> WorkloadRecord:
    labels = {"eyvan_pool": f"pool-{ZONES[0]}"}
    if alternates:
        labels[POOL_ALTERNATES_LABEL] = ",".join(f"pool-{zone}" for zone in alternates)
    return WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="racer",
            execution=CommandExecution(command="python train.py"),
            required_labels=labels,
        ),
        state=WorkloadState.PENDING,
        created_by="alice",
    )


def _report(cells: list[CellRecord], *, leases: list[LeaseRecord] | None = None, workload=None):
    return candidate_zone_report(
        workload or _workload(),
        [_pool(zone) for zone in ZONES],
        cells,
        leases or [],
        now=NOW,
    )


# --------------------------------------------------------------------------- when it applies


def test_a_workload_naming_one_pool_has_no_zones_to_explain() -> None:
    """A one-entry list reads as a decision that was never made."""

    assert _report([], workload=_workload(alternates=())) == ()


def test_every_candidate_is_reported_even_with_nothing_to_say() -> None:
    """Absence of evidence is itself the answer when a fleet is cold everywhere."""

    report = _report([])

    assert [zone["pool_id"] for zone in report] == [f"pool-{zone}" for zone in ZONES]
    assert sum(bool(zone["preferred"]) for zone in report) == 1


# --------------------------------------------------------------------------- what it says


def test_capacity_this_workload_already_owns_wins_outright() -> None:
    """A cell reserved for this job is the strongest reason to prefer a zone."""

    report = _report([_cell("cell-b", ZONES[1], reserved_for="wl-1")])

    assert report[0]["pool_id"] == f"pool-{ZONES[1]}"
    assert report[0]["preferred"] is True
    assert report[0]["free_owned_cells"] == 1


def test_a_free_shared_cell_beats_an_empty_zone() -> None:
    """Existing capacity beats a bet on which provider will grant."""

    report = _report([_cell("cell-c", ZONES[2])])

    assert report[0]["pool_id"] == f"pool-{ZONES[2]}"
    assert report[0]["free_shared_cells"] == 1


def test_a_cell_someone_is_using_is_not_free() -> None:
    """A busy cell counted as available is how a zone looks better than it is."""

    cell = _cell("cell-b", ZONES[1])
    lease = LeaseRecord(
        id="lease-1",
        workload_id="wl-other",
        cell_id="cell-b",
        org_id="acme",
        project_id="research",
        state=LeaseState.ACTIVE,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        generation=1,
        fencing_token="token-1",
        expires_at=NOW + timedelta(hours=1),
    )

    report = _report([cell], leases=[lease])

    zone = next(entry for entry in report if entry["pool_id"] == f"pool-{ZONES[1]}")
    assert zone["free_shared_cells"] == 0


def test_a_cooling_zone_ranks_below_a_healthy_one() -> None:
    """The afternoon this exists for: one zone reclaiming, another holding."""

    report = _report(
        [
            _cell("cell-a", ZONES[0], cooling=True, failures=2),
            _cell("cell-c", ZONES[2]),
        ]
    )

    assert report[0]["pool_id"] == f"pool-{ZONES[2]}"
    cold = next(zone for zone in report if zone["pool_id"] == f"pool-{ZONES[0]}")
    assert cold["cooling_cells"] == 1
    assert cold["consecutive_failures"] == 2
    assert cold["preferred"] is False


def test_a_pool_is_as_cold_as_its_coldest_cell() -> None:
    """One reclaiming cell is enough; averaging would hide the zone that fails."""

    report = _report(
        [
            _cell("cell-a1", ZONES[0], failures=5),
            _cell("cell-a2", ZONES[0], failures=0),
        ]
    )

    zone = next(entry for entry in report if entry["pool_id"] == f"pool-{ZONES[0]}")
    assert zone["consecutive_failures"] == 5


# --------------------------------------------------------------------------- counted like the decision


def test_a_disabled_pool_is_not_offered_as_a_candidate_at_all() -> None:
    """The decision filters a disabled pool out before ranking; so must the report.

    Its cells would otherwise read as free supply and outrank zones the
    decision could actually pick -- a rationale that contradicts the choice.
    """

    pools = [_pool(zone) for zone in ZONES]
    pools[1] = pools[1].model_copy(update={"enabled": False})

    report = candidate_zone_report(_workload(), pools, [_cell("cell-b", ZONES[1])], [], now=NOW)

    assert [entry["pool_id"] for entry in report] == [f"pool-{ZONES[0]}", f"pool-{ZONES[2]}"]
    assert report[0]["pool_id"] == f"pool-{ZONES[0]}"


def test_cells_that_no_longer_supply_do_not_speak_for_their_zone() -> None:
    """Only cells the decision could still use may colour the ranking.

    A tombstoned cell is retained forever, and counting its frozen belief made
    a drained zone read as cold rather than as empty -- which argued a workload
    out of the very zone the decision counts as simply untried.
    """

    report = _report([_cell("cell-a1", ZONES[0], state=CellState.DELETED, failures=9, cooling=True)])

    primary = next(entry for entry in report if entry["pool_id"] == f"pool-{ZONES[0]}")
    assert primary["consecutive_failures"] == 0
    assert primary["cooling_cells"] == 0
    assert report[0]["pool_id"] == f"pool-{ZONES[0]}", "the tombstone must not hand the choice to another zone"


# --------------------------------------------------------------------------- the invariant


@pytest.mark.parametrize(
    "cells",
    [
        [],
        [_cell("c1", ZONES[1], reserved_for="wl-1")],
        [_cell("c1", ZONES[2])],
        [_cell("c1", ZONES[0], cooling=True), _cell("c2", ZONES[1])],
        [_cell("c1", ZONES[0], failures=9), _cell("c2", ZONES[2], failures=1)],
        [_cell("c1", ZONES[0]), _cell("c2", ZONES[1]), _cell("c3", ZONES[2])],
    ],
)
def test_the_rationale_never_contradicts_the_choice(cells: list[CellRecord]) -> None:
    """Exactly one preferred zone, and it is the one listed first.

    A rationale that disagrees with the decision is worse than none: it teaches
    an operator a model of the scheduler that is wrong, and they act on it.
    """

    report = _report(cells)

    assert sum(bool(zone["preferred"]) for zone in report) == 1
    assert report[0]["preferred"] is True
