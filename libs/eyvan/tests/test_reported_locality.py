# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An agent that can see the disk, beating an inference about it.

Locality was derived entirely from lease history: a cell that ran a job
declaring asset X twenty minutes ago probably still has X. That inference is
free and often right, and it is still a guess about a disk the control plane
cannot see.

An agent can see it. So a reported asset now ranks ahead of every inferred one,
and inference still covers every cell whose agent reports nothing -- which is
what makes this deployable before every agent is upgraded rather than after.

**Still only a preference.** ``preferred_cells`` is the fourth key of a
lexicographic score, so a wrong report costs one cache miss. It cannot make a
workload unschedulable, cannot strand it in a zone, and cannot reach across
tenants: the asset names come from the submitter's own spec, so a match happens
only where they already knew the name.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
)
from eyvan.server.controller.observation import _reported_assets
from eyvan.server.service.locality import build_index, preferred_cells_for

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _cell(cell_id: str, *, assets: tuple[str, ...] = ()) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        endpoint="http://agent:8090",
        materialized_assets=assets,
    )


def _workload(workload_id: str, *, assets: tuple[str, ...]) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
            named_assets=assets,
        ),
        created_by="alice",
    )


def _lease(lease_id: str, *, workload_id: str, cell_id: str, minutes_ago: int) -> LeaseRecord:
    return LeaseRecord(
        id=lease_id,
        workload_id=workload_id,
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        state=LeaseState.RELEASED,
        fencing_token=f"{cell_id}:1",
        expires_at=NOW + timedelta(hours=1),
        updated_at=NOW - timedelta(minutes=minutes_ago),
    )


# --------------------------------------------------------------------------- ranking


def test_a_reported_cell_outranks_an_inferred_one() -> None:
    """The whole point: an agent that can see the disk beats a guess about it."""

    index = build_index(
        [_workload("wl-1", assets=("dataset",))],
        [_lease("lease-1", workload_id="wl-1", cell_id="inferred", minutes_ago=5)],
        now=NOW,
        cells=[_cell("reported", assets=("dataset",))],
    )

    assert index["dataset"][0] == "reported"
    assert "inferred" in index["dataset"]


def test_inference_still_covers_cells_that_report_nothing() -> None:
    """Additive, not a replacement.

    An agent that never learns to report leaves the fleet exactly where it was,
    which is what makes this safe to deploy before every agent is upgraded.
    """

    index = build_index(
        [_workload("wl-1", assets=("dataset",))],
        [_lease("lease-1", workload_id="wl-1", cell_id="inferred", minutes_ago=5)],
        now=NOW,
        cells=[_cell("silent")],
    )

    assert index["dataset"] == ("inferred",)


def test_a_cell_in_both_signals_appears_once() -> None:
    """A duplicate would count twice in the ranking.

    ``preferred_cells_for`` scores a cell by how many of the workload's assets
    it holds, so one cell listed twice for one asset would silently outweigh a
    cell that genuinely holds two.
    """

    index = build_index(
        [_workload("wl-1", assets=("dataset",))],
        [_lease("lease-1", workload_id="wl-1", cell_id="cell-a", minutes_ago=5)],
        now=NOW,
        cells=[_cell("cell-a", assets=("dataset",))],
    )

    assert index["dataset"] == ("cell-a",)


def test_a_cell_holding_more_assets_still_wins() -> None:
    """Reporting must not defeat the property the ranking exists for.

    A cell with three of four assets should beat one with a single reported
    asset; otherwise a partial report drags a job away from the better cell.
    """

    index = build_index(
        [_workload("wl-1", assets=("a", "b"))],
        [_lease("lease-1", workload_id="wl-1", cell_id="both", minutes_ago=5)],
        now=NOW,
        cells=[_cell("one", assets=("a",))],
    )

    assert preferred_cells_for(("a", "b"), index)[0] == "both"


def test_reported_order_does_not_depend_on_cell_listing_order() -> None:
    """Placement that depended on iteration order would be unreproducible."""

    cells = [_cell("cell-b", assets=("dataset",)), _cell("cell-a", assets=("dataset",))]

    forward = build_index([], [], now=NOW, cells=cells)
    reverse = build_index([], [], now=NOW, cells=list(reversed(cells)))

    assert forward == reverse
    assert forward["dataset"] == ("cell-a", "cell-b")


def test_a_fleet_with_no_reports_is_unchanged() -> None:
    """The existing behaviour, asserted so the addition cannot have moved it."""

    inferred_only = build_index(
        [_workload("wl-1", assets=("dataset",))],
        [_lease("lease-1", workload_id="wl-1", cell_id="cell-a", minutes_ago=5)],
        now=NOW,
    )

    assert inferred_only["dataset"] == ("cell-a",)


# --------------------------------------------------------------------------- the report


def test_an_agent_report_is_read_from_the_observation_metadata() -> None:
    """Carried in the existing free-form metadata rather than a new wire field.

    An agent that never learns to report keeps validating, and a control plane
    that predates the convention keeps parsing -- a required field would have
    been a deployment ordering problem in exchange for nothing.
    """

    assert _reported_assets({"materialized_assets": ["dataset", "checkpoint"]}) == ("checkpoint", "dataset")


def test_a_malformed_report_is_ignored_rather_than_rejected() -> None:
    """This runs on the observation hot path.

    Refusing an otherwise good observation over a bad hint would trade one cache
    miss for a cell the control plane cannot see at all.
    """

    assert _reported_assets({"materialized_assets": "dataset"}) == ()
    assert _reported_assets({"materialized_assets": [1, None, ""]}) == ()
    assert _reported_assets({}) == ()
    assert _reported_assets(None) == ()


def test_a_report_is_deduplicated_and_ordered() -> None:
    """Two agents reporting the same set must produce the same index.

    Otherwise placement depends on report ordering, which nothing controls.
    """

    assert _reported_assets({"materialized_assets": ["b", "a", "b", " a "]}) == ("a", "b")


async def test_an_observation_records_what_the_agent_reported() -> None:
    """The seam. A parser nothing calls reports on nothing.

    This is the hop between an agent's metadata and the record the placement
    index reads, and it is the one nothing else here covers.
    """

    from test_controller import FakeAgent, principal

    from eyvan.config import EyvanSettings
    from eyvan.records import Role
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = FakeAgent()
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _cell: agent)
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        _cell("cell-a").model_copy(update={"state": CellState.REGISTERING}),
        idempotency_key="cell",
    )
    assert await controller.dispatch_outbox_once() == 1
    agent.metadata = {"materialized_assets": ["dataset"]}
    await controller.observe_cells_once()

    async with repository.transaction() as tx:
        stored = await tx.get_cell("cell-a")
    assert stored is not None
    assert stored.materialized_assets == ("dataset",)


async def test_the_service_consults_reported_assets_when_resolving_a_spec() -> None:
    """The seam between the index and the code that uses it.

    ``build_index`` accepting cells proves nothing about whether the control
    plane passes any -- a mutation that dropped the argument left every test
    above green, which is this package's recurring defect at one more seam.
    """

    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        await tx.put_cell(_cell("reported", assets=("dataset",)))

    spec = WorkloadDefinition(
        name="train",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        named_assets=("dataset",),
    )
    resolved = await service.resolve_locality_preferences(spec, now=NOW)

    assert resolved.preferred_cells == ("reported",)


async def test_an_explicit_pin_still_wins_over_a_report() -> None:
    """A pin is a statement the submitter made about placement.

    A cache hint has no standing to argue with it, and an agent's report has no
    more standing than the inference it replaces.
    """

    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        await tx.put_cell(_cell("reported", assets=("dataset",)))

    spec = WorkloadDefinition(
        name="train",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
        named_assets=("dataset",),
        preferred_cells=("operator-chose-this",),
    )
    resolved = await service.resolve_locality_preferences(spec, now=NOW)

    assert resolved.preferred_cells == ("operator-chose-this",)
