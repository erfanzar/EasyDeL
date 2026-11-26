# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expand one pool declaration across zones into interchangeable submission alternates.

The shorthand must preserve the shape and template compatibility required for
cross-zone placement without repeated per-zone configuration.
"""

from __future__ import annotations

import pytest
import yaml

from eyvan.capacity.fleet_config import EyvanFleetConfig

BASE = """
name: research-fleet
provider:
  type: gcp
  project: my-project
  region: us-east5
  zone: us-east5-a
agent_endpoint: http://agent:8090
pools:
  v6e-8:
    kind: tpu
    capacity_class: spot
    max_workers: 4
    provider_options:
      bootstrap_egress: cloud-nat
"""


def _config(pool_lines: str) -> EyvanFleetConfig:
    document = yaml.safe_load(BASE)
    document["pools"]["v6e-8"].update(yaml.safe_load(pool_lines))
    return EyvanFleetConfig.model_validate(document)


def _invalid(pool_lines: str) -> str:
    with pytest.raises(ValueError) as raised:
        _config(pool_lines)
    return str(raised.value)


# --------------------------------------------------------------------------- expansion


def test_one_declaration_becomes_one_pool_per_zone() -> None:
    """The whole feature: say it once, get capacity in three places."""

    config = _config("zones: [us-east5-a, us-east5-b, us-central2-b]")

    assert set(config.pools) == {"v6e-8-us-east5-a", "v6e-8-us-east5-b", "v6e-8-us-central2-b"}
    assert [pool.zone for pool in config.pools.values()] == ["us-east5-a", "us-east5-b", "us-central2-b"]


def test_every_sibling_names_the_same_shape() -> None:
    """The shape is what a user asks for; the zone is where it is found.

    Without this each sibling would take its own name as its accelerator type,
    and `--tpu v6e-8` would match none of them.
    """

    config = _config("zones: [us-east5-a, us-east5-b]")

    assert {pool.accelerator_type for pool in config.pools.values()} == {"v6e-8"}


def test_siblings_are_interchangeable_so_they_can_be_alternates() -> None:
    """The condition the submission path checks before offering an alternate.

    Equal admission width and equal topology. If expansion ever changed either
    per zone, the pools would be refused as ambiguous instead of raced -- with
    an error about a misconfiguration that is really a bug here.
    """

    config = _config("zones: [us-east5-a, us-east5-b, us-central2-b]")
    pools = list(config.pools.values())

    assert len({pool.resources.model_dump_json() for pool in pools}) == 1
    assert len({repr(sorted((pool.topology or {}).items())) for pool in pools}) == 1
    assert len({pool.capacity_class for pool in pools}) == 1


def test_everything_else_declared_once_is_inherited_by_each_zone() -> None:
    """Say the interesting things once; only the location differs."""

    config = _config("zones: [us-east5-a, us-east5-b]\nmax_workers: 7")

    assert {pool.max_cells for pool in config.pools.values()} == {7}


def test_a_pool_naming_no_zones_is_untouched() -> None:
    """Additive: every fleet written before this keeps its exact shape."""

    config = _config("zone: us-east5-a")

    assert set(config.pools) == {"v6e-8"}
    assert config.pools["v6e-8"].zone == "us-east5-a"


# --------------------------------------------------------------------------- refusals


def test_zone_and_zones_together_are_refused() -> None:
    """Two ways of saying where a pool runs is one way too many."""

    assert "zone and zones" in _invalid("zone: us-east5-a\nzones: [us-east5-b]")


def test_a_repeated_zone_is_refused() -> None:
    """It would collide on the expanded name and silently drop one."""

    assert "twice" in _invalid("zones: [us-east5-a, us-east5-a]")


@pytest.mark.parametrize("zones", ["zones: []", "zones: 'us-east5-a'", "zones: [null]", "zones: ['  ']"])
def test_a_malformed_zone_list_is_refused(zones: str) -> None:
    """A pool that will not say where it runs is not a pool."""

    assert _invalid(zones)


def test_two_pools_expanding_onto_one_name_are_refused() -> None:
    """Silently keeping the last would lose declared capacity."""

    document = yaml.safe_load(BASE)
    document["pools"]["v6e-8"]["zones"] = ["us-east5-a"]
    document["pools"]["v6e-8-us-east5-a"] = dict(document["pools"]["v6e-8"])
    document["pools"]["v6e-8-us-east5-a"].pop("zones")
    document["pools"]["v6e-8-us-east5-a"]["zone"] = "us-east5-a"

    with pytest.raises(ValueError) as raised:
        EyvanFleetConfig.model_validate(document)

    assert "declared twice" in str(raised.value)


# --------------------------------------------------------------------------- end to end


@pytest.mark.asyncio
async def test_a_multi_zone_declaration_gives_one_job_capacity_in_every_zone() -> None:
    """The acceptance property, through the real submission path.

    Declaring zones is only worth anything if the pools it produces are offered
    to a workload as alternates for each other. Everything between here and
    there -- equal admission width, equal topology, one shared accelerator type
    -- is what makes that happen, and each of them is easy to break by accident.
    """

    from eyvan.capacity.planning import POOL_ALTERNATES_LABEL
    from eyvan.records import CapacityPoolRecord, Principal, Role, RoleGrant
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    config = _config("zones: [us-east5-a, us-east5-b, us-central2-b]")
    service = ControlPlaneService(InMemoryRepository())
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    pools = []
    for payload in config.pool_payloads():
        document = dict(payload)
        document["agent_endpoint"] = document["agent_endpoint"] or "http://agent:8090"
        pools.append(CapacityPoolRecord.model_validate(document))
    await service.apply_fleet(fleet_admin, "research-fleet", pools, idempotency_key="fleet-1")

    operator = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    workload, _replayed = await service.submit_pool_workload(
        operator,
        fleet_name="research-fleet",
        node_type="v6e-8",
        org_id="acme",
        project_id="research",
        command="python train.py",
        idempotency_key="run-1",
    )

    alternates = workload.spec.metadata.get(POOL_ALTERNATES_LABEL, "")
    assert workload.spec.metadata["eyvan_pool"] not in alternates
    assert len(alternates.split(",")) == 2, "the other two zones must be raceable"


@pytest.mark.asyncio
async def test_a_zone_can_still_be_pinned_out_of_a_multi_zone_pool() -> None:
    """Declaring three zones must not take away the ability to choose one."""

    from eyvan.records import CapacityPoolRecord, Principal, Role, RoleGrant
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    config = _config("zones: [us-east5-a, us-east5-b]")
    service = ControlPlaneService(InMemoryRepository())
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    pools = []
    for payload in config.pool_payloads():
        document = dict(payload)
        document["agent_endpoint"] = document["agent_endpoint"] or "http://agent:8090"
        pools.append(CapacityPoolRecord.model_validate(document))
    await service.apply_fleet(fleet_admin, "research-fleet", pools, idempotency_key="fleet-1")

    operator = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    workload, _replayed = await service.submit_pool_workload(
        operator,
        fleet_name="research-fleet",
        node_type="v6e-8",
        org_id="acme",
        project_id="research",
        command="python train.py",
        zone="us-east5-b",
        idempotency_key="run-1",
    )

    pool_id = workload.spec.metadata["eyvan_pool"]
    assert next(pool for pool in pools if pool.id == pool_id).zone == "us-east5-b"


# --------------------------------------------------------------------------- scheduled, not just recorded


async def _submitted_cross_zone_workload(zones: str):
    from eyvan.records import CapacityPoolRecord, Principal, Role, RoleGrant
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    config = _config(f"zones: {zones}")
    service = ControlPlaneService(InMemoryRepository())
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    pools = []
    for payload in config.pool_payloads():
        document = dict(payload)
        document["agent_endpoint"] = document["agent_endpoint"] or "http://agent:8090"
        pools.append(CapacityPoolRecord.model_validate(document))
    await service.apply_fleet(fleet_admin, "research-fleet", pools, idempotency_key="fleet-1")
    operator = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    workload, _replayed = await service.submit_pool_workload(
        operator,
        fleet_name="research-fleet",
        node_type="v6e-8",
        org_id="acme",
        project_id="research",
        command="python train.py",
        idempotency_key="run-1",
    )
    return service, tuple(pools), workload


@pytest.mark.asyncio
async def test_cross_zone_pool_template_pairs_are_not_a_cartesian_product() -> None:
    service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b]")
    for index, pool in enumerate(pools):
        cell = pool.make_cell(f"paired-{index}", worker_count=workload.spec.worker_count)
        assert _placed(service, workload, cell)
        crossed = cell.model_copy(
            update={"labels": {**cell.labels, "eyvan_pool_template": pools[1 - index].template_fingerprint}}
        )
        assert not _placed(service, workload, crossed)


@pytest.mark.asyncio
async def test_cross_zone_pairs_are_revalidated_before_lease_admission() -> None:
    from eyvan.records import CellState
    from eyvan.server.controller import EyvanController
    from eyvan.server.service.errors import ConflictError
    from eyvan.server.service.scheduling import CoreSchedulingService

    service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b]")
    service.scheduler = CoreSchedulingService()
    cell = (
        pools[0]
        .make_cell("candidate", worker_count=workload.spec.worker_count)
        .model_copy(update={"state": CellState.READY})
    )
    decision = service.scheduler.schedule([workload], [cell]).placements[0]
    crossed = cell.model_copy(update={"labels": {**cell.labels, "eyvan_pool_template": pools[1].template_fingerprint}})
    async with service.repository.transaction() as tx:
        await tx.put_cell(crossed)
    controller = EyvanController(service.repository, service)
    with pytest.raises(ConflictError):
        await controller._commit_placement(decision)
    async with service.repository.transaction() as tx:
        assert await tx.list_leases() == []
        await tx.put_cell(cell)
    await controller._commit_placement(decision)
    async with service.repository.transaction() as tx:
        leases = await tx.list_leases()
        assert len(leases) == 1
        assert leases[0].cell_id == cell.id
        assert leases[0].workload_id == workload.id


def _placed(service, workload, cell) -> bool:
    from eyvan.records import CellState
    from eyvan.server.service.scheduling import CoreSchedulingService

    ready = cell.model_copy(update={"state": CellState.READY})
    result = CoreSchedulingService().schedule([workload], [ready])
    return any(placement.workload_id == workload.id for placement in result.placements)


@pytest.mark.asyncio
async def test_a_cross_zone_workload_can_be_placed_in_every_zone_it_names() -> None:
    """Allow placement in every named alternate without requiring first-choice-only labels on each cell."""

    service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b, us-central2-b]")

    for index, pool in enumerate(pools):
        cell = pool.make_cell(f"cell-{index}", worker_count=workload.spec.worker_count)
        assert _placed(service, workload, cell), f"no placement in {pool.zone}"


@pytest.mark.asyncio
async def test_the_template_pin_still_refuses_a_cell_from_a_template_it_was_not_submitted_against() -> None:
    service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b]")
    alternate = pools[1]
    drifted = alternate.make_cell("drifted", worker_count=workload.spec.worker_count)
    drifted = drifted.model_copy(update={"labels": {**drifted.labels, "eyvan_pool_template": "0" * 64}})
    stranger = alternate.model_copy(update={"id": "someone-elses-pool"}).make_cell(
        "stranger", worker_count=workload.spec.worker_count
    )

    assert not _placed(service, workload, drifted)
    assert not _placed(service, workload, stranger)


@pytest.mark.asyncio
async def test_a_workload_stored_before_alternate_templates_keeps_its_first_choice() -> None:
    """Durable specs written before the templates label existed are not loosened."""

    from eyvan.capacity.planning import POOL_ALTERNATE_TEMPLATES_LABEL

    service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b]")
    labels = dict(workload.spec.required_labels)
    labels.pop(POOL_ALTERNATE_TEMPLATES_LABEL)
    legacy = workload.model_copy(update={"spec": workload.spec.model_copy(update={"required_labels": labels})})
    primary = next(pool for pool in pools if pool.id == labels["eyvan_pool"])
    other = next(pool for pool in pools if pool.id != primary.id)

    assert _placed(service, legacy, primary.make_cell("primary", worker_count=legacy.spec.worker_count))
    assert not _placed(service, legacy, other.make_cell("other", worker_count=legacy.spec.worker_count))


@pytest.mark.asyncio
async def test_a_raced_cross_zone_workload_creates_demand_in_every_raced_zone() -> None:
    """The planner end to end: real submission, real simulation, real scheduler."""

    from datetime import UTC, datetime

    from eyvan.capacity.planning import CapacitySimulation

    _service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b, us-central2-b]")
    now = datetime.now(UTC)
    simulation = CapacitySimulation.build(pools, (), (workload,), (), now)
    from eyvan.server.service.scheduling import CoreSchedulingService

    scheduled = CoreSchedulingService().schedule([workload], simulation.cells)
    demand = simulation.demand(pools, (), scheduled, (workload,), now, race_width=3)

    asked = sorted(pool_id for pool_id, entry in demand.items() if workload.id in entry.virtual_workload_ids)
    assert asked == sorted(pool.id for pool in pools)


@pytest.mark.asyncio
async def test_a_cross_zone_workload_can_race_into_another_region() -> None:
    """Do not pin an alternate-region workload to its first-choice pool's region label."""

    from eyvan.records import CapacityPoolRecord

    service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b]")
    elsewhere = pools[1].model_copy(update={"region": "europe-west4", "zone": "europe-west4-a"})
    moved = [pools[0], CapacityPoolRecord.model_validate(elsewhere.model_dump())]
    labels = dict(workload.spec.required_labels)
    labels["eyvan_pool_alternate_templates"] = moved[1].template_fingerprint
    workload = workload.model_copy(update={"spec": workload.spec.model_copy(update={"required_labels": labels})})

    cell = moved[1].make_cell("far-away", worker_count=workload.spec.worker_count)

    assert _placed(service, workload, cell)


@pytest.mark.asyncio
async def test_racing_into_another_region_still_honours_the_projects_allowed_regions() -> None:
    """Dropping the first choice's region pin must not also drop the project policy it stood in for."""

    from eyvan.records import CapacityPoolRecord, CellState, PolicyRecord
    from eyvan.server.service.scheduling import CoreSchedulingService

    _service, pools, workload = await _submitted_cross_zone_workload("[us-east5-a, us-east5-b]")
    elsewhere = CapacityPoolRecord.model_validate(
        pools[1].model_copy(update={"region": "europe-west4", "zone": "europe-west4-a"}).model_dump()
    )
    labels = dict(workload.spec.required_labels)
    labels["eyvan_pool_alternate_templates"] = elsewhere.template_fingerprint
    workload = workload.model_copy(update={"spec": workload.spec.model_copy(update={"required_labels": labels})})
    policy = PolicyRecord(
        org_id=workload.org_id, project_id=workload.project_id, allowed_regions=(pools[0].region,), updated_by="alice"
    )
    far = elsewhere.make_cell("far-away", worker_count=workload.spec.worker_count).model_copy(
        update={"state": CellState.READY}
    )
    near = (
        pools[0].make_cell("near", worker_count=workload.spec.worker_count).model_copy(update={"state": CellState.READY})
    )

    refused = CoreSchedulingService().schedule([workload], [far], policies=[policy])
    placed = CoreSchedulingService().schedule([workload], [near], policies=[policy])

    assert not refused.placements, "a region the project's policy excludes is never used"
    assert placed.placements
