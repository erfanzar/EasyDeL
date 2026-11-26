# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What a run cost, and what it would have cost on demand.

The sentence a spot-first scheduler exists to say. It is also the easiest place
in the whole system to lie: an unpriced pool settles to zero, and a plausible
on-demand comparison could be produced from any discount constant. Both would
read as a fact and end up in somebody's budget, so most of these tests are
about saying *unknown* out loud.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk import cli
from eyvan.sdk.spend import render, spend_of
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError


@pytest.fixture(autouse=True)
def _isolated_cli_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI cost rendering must not read or write the developer's saved context."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "artifacts"))


SPOT_POOL = {
    "id": "pool-spot",
    "accelerator_type": "v5p-8",
    "capacity_class": "spot",
    "unit_prices": {"tpu": 1.2},
}
ON_DEMAND_POOL = {
    "id": "pool-od",
    "accelerator_type": "v5p-8",
    "capacity_class": "on-demand",
    "unit_prices": {"tpu": 4.0},
}


def _lease(
    cell_id: str,
    *,
    settled: float = 0.0,
    hours: float = 0.0,
    reserved: float = 0.0,
    settled_at: str | None = None,
) -> dict:
    return {
        "id": f"lease-{cell_id}",
        "cell_id": cell_id,
        "settled_cost": settled,
        "reserved_cost": reserved,
        "accelerator_hours_settled": {"tpu": hours} if hours else {},
        "settled_at": settled_at,
    }


# --------------------------------------------------------------------------- the arithmetic


def test_hours_and_cost_are_summed_across_every_lease() -> None:
    """A multislice run holds one lease per slice, and the bill is all of them."""

    spend = spend_of(
        [_lease("cell-a", settled=4.8, hours=4.0), _lease("cell-b", settled=4.8, hours=4.0)],
        priced_cells={"cell-a", "cell-b"},
    )

    assert spend.hours == 8.0
    assert spend.settled == 9.6
    assert spend.leases == 2
    assert spend.priced


def test_a_lease_from_an_unpriced_pool_makes_the_total_untrustworthy() -> None:
    """Half a bill presented as the whole one is worse than no bill."""

    spend = spend_of([_lease("cell-a", settled=4.8, hours=4.0), _lease("cell-b", hours=4.0)], priced_cells={"cell-a"})

    assert spend.hours == 8.0
    assert not spend.priced


def test_measurement_survives_when_pricing_does_not() -> None:
    """Accelerator-hours are measured, so they are known even when money is not."""

    spend = spend_of([_lease("cell-a", hours=16.0)], priced_cells=set())

    assert spend.hours == 16.0
    assert not spend.priced


def test_a_workload_that_never_leased_says_so() -> None:
    lines = render(
        "wl-1", "failed", spend_of([]), shape=None, capacity_class=None, on_demand_hourly=None, on_demand_source="n/a"
    )

    assert any("never held a lease" in line for line in lines)


def test_a_settled_lease_stops_counting_as_reserved_money() -> None:
    """Settlement turned the reservation into the bill, so only it is money held.

    Settlement keeps the original ``reserved_cost`` standing on the record next
    to the ``settled_cost`` it became; presenting both on a finished run would
    report an estimate as money the tenant still holds.
    """

    spend = spend_of(
        [
            _lease("cell-a", settled=4.8, hours=4.0, reserved=24.0, settled_at="2026-01-01T00:00:00+00:00"),
            _lease("cell-b", reserved=12.0),
        ],
        priced_cells={"cell-a", "cell-b"},
    )

    assert spend.reserved == 12.0
    assert spend.settled == 4.8
    assert spend.leases == 2


@pytest.mark.parametrize("field", ["settled", "reserved", "hours"])
@pytest.mark.parametrize(("value", "expected"), [(0.1, 0.3), (0.00000000004, 0.00000000012)])
def test_fractional_lease_totals_do_not_reintroduce_settlement_drift(field: str, value: float, expected: float) -> None:
    leases = [
        _lease(
            f"cell-{index}",
            **{field: value},
            settled_at=None if field == "reserved" else "2026-01-01T00:00:00+00:00",
        )
        for index in range(3)
    ]

    spend = spend_of(leases, priced_cells={f"cell-{index}" for index in range(3)})

    assert getattr(spend, field) == expected
    assert isinstance(getattr(spend, field), float)
    assert spend.priced


@pytest.mark.parametrize("value", [True, False, "0.1", None, {}, []])
def test_non_numeric_amounts_do_not_pollute_valid_lease_totals(value: object) -> None:
    spend = spend_of(
        [
            {
                "cell_id": "bad",
                "settled_cost": value,
                "reserved_cost": value,
                "accelerator_hours_settled": {"tpu": value},
            },
            _lease("paid", settled=2.5, hours=2.0, reserved=99.0, settled_at="2026-01-01T00:00:00+00:00"),
            _lease("running", reserved=1.0),
        ],
        priced_cells={"bad", "paid", "running"},
    )

    assert spend.settled == 2.5
    assert spend.reserved == 1.0
    assert spend.accelerator_hours == {"tpu": 2.0}
    assert spend.priced


# --------------------------------------------------------------------------- what it prints


def test_an_unpriced_run_reports_unknown_rather_than_zero() -> None:
    """$0.00 from an unpriced pool is the absence of a price, not of a bill."""

    lines = render(
        "wl-1",
        "succeeded",
        spend_of([_lease("cell-a", hours=16.0)], priced_cells=set()),
        shape="v5p-8",
        capacity_class="spot",
        on_demand_hourly=None,
        on_demand_source="no on-demand pool declares a price",
    )

    assert any("cost        unknown" in line for line in lines)
    assert not any("$0.00" in line for line in lines)


def test_a_priced_run_shows_what_it_cost_and_what_it_saved() -> None:
    """The whole point, when the fleet actually declares both prices."""

    lines = render(
        "wl-1",
        "succeeded",
        spend_of([_lease("cell-a", settled=9.6, hours=8.0)], priced_cells={"cell-a"}),
        shape="v5p-8",
        capacity_class="spot",
        on_demand_hourly=4.0,
        on_demand_source="declared by pool pool-od",
    )
    rendered = "\n".join(lines)

    assert "cost        $9.60" in rendered
    assert "on demand   $32.00" in rendered
    assert "saved       $22.40" in rendered
    assert "70% less" in rendered


def test_no_saving_is_claimed_when_the_run_itself_is_unpriced() -> None:
    """Subtracting a real price from an unknown one produces a fictional saving."""

    lines = render(
        "wl-1",
        "succeeded",
        spend_of([_lease("cell-a", hours=8.0)], priced_cells=set()),
        shape="v5p-8",
        capacity_class="spot",
        on_demand_hourly=4.0,
        on_demand_source="declared by pool pool-od",
    )
    rendered = "\n".join(lines)

    assert "on demand   $32.00" in rendered
    assert "saved" not in rendered


# --------------------------------------------------------------------------- who may ask, and what it costs


def _principal(role: Role) -> Principal:
    return Principal(subject="alice", grants=(RoleGrant(role=role, org_id="acme", project_id="research"),))


def _pool(pool_id: str, *, capacity_class: CapacityClass, prices: dict[str, float]) -> CapacityPoolRecord:
    resources = ResourceVector(cpu=8, memory_gib=32, tpu=4)
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="v5p-8",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="a-project",
        region="us-east5",
        zone="us-east5-a",
        capacity=resources,
        allocatable=resources,
        accelerator_type="v5p-8",
        capacity_class=capacity_class,
        unit_prices=prices,
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        max_cells=4,
    )


async def _plane_with(
    *pools: CapacityPoolRecord,
    settled: float = 9.6,
    hours: float = 8.0,
    pool_id: str = "pool-spot",
    hourly_cost: float | None = None,
) -> ControlPlaneService:
    """A plane holding one finished run that used ``pool_id``."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    spec = WorkloadDefinition(
        name="run",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=8, memory_gib=32, tpu=4)),
        required_labels={"eyvan_pool": pool_id},
        metadata={"eyvan_pool": pool_id},
    )
    workload = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=spec,
        state=WorkloadState.SUCCEEDED,
        created_by="alice",
    )
    pool = next((pool for pool in pools if pool.id == pool_id), None)
    cell = (
        CellRecord(
            id="cell-a",
            provider=pool.provider,
            capacity_backend=pool.capacity_backend,
            provider_project=pool.provider_project,
            region=pool.region,
            zone=pool.zone,
            state=CellState.READY,
            capacity=pool.capacity,
            allocatable=pool.allocatable,
            accelerator_type=pool.accelerator_type,
            capacity_class=pool.capacity_class,
            pool_id=pool.id,
            unit_prices=pool.unit_prices,
        )
        if pool is not None
        else None
    )
    frozen_hourly = cell.hourly_cost() if cell is not None else 0.0
    lease = LeaseRecord(
        id="lease-a",
        workload_id="wl-1",
        cell_id="cell-a",
        cell_generation=cell.generation if cell is not None else 1,
        hourly_cost=frozen_hourly if hourly_cost is None else hourly_cost,
        org_id="acme",
        project_id="research",
        state=LeaseState.RELEASED,
        resources=spec.gang,
        fencing_token="fence",
        settled_cost=settled,
        accelerator_hours_settled={"tpu": hours},
        # Required of any settled lease by the record itself: a bill without the
        # clock that produced it is not auditable.
        settled_at=utcnow(),
        settlement_reason="released",
        expires_at=utcnow() + timedelta(minutes=5),
    )
    async with repository.transaction() as transaction:
        await transaction.put_workload(workload)
        await transaction.put_lease(lease)
        if cell is not None:
            await transaction.put_cell(cell)
        for pool in pools:
            await transaction.put_capacity_pool(pool)
    return service


@pytest.mark.asyncio
async def test_a_project_operator_may_ask_what_their_own_run_cost() -> None:
    """The whole reason this moved into the plane.

    The client used to total this itself, which meant listing capacity pools for
    their unit prices -- a fleet read. So the one command that answers "what did
    this cost me" was refused to everybody who could not administer the fleet,
    and its help never said so.
    """

    service = await _plane_with(_pool("pool-spot", capacity_class=CapacityClass.SPOT, prices={"tpu": 1.2}))

    spend = await service.workload_cost(_principal(Role.PROJECT_OPERATOR), "wl-1")

    assert spend.priced
    assert spend.settled == 9.6
    assert spend.accelerator_hours == {"tpu": 8.0}


@pytest.mark.asyncio
async def test_the_comparison_finds_a_declared_on_demand_pool() -> None:
    """The comparison this feature exists for, and it could never fire.

    The client compared the capacity class against ``"on-demand"`` while the
    enum's value is ``on_demand``, so no pool ever matched and every run
    reported the alternative as unknown. It survived because the test fixture
    was a hand-written dict carrying the same wrong string -- a pool record
    cannot hold it, and nothing in that test was a pool record.
    """

    service = await _plane_with(
        _pool("pool-spot", capacity_class=CapacityClass.SPOT, prices={"tpu": 1.2}),
        _pool("pool-od", capacity_class=CapacityClass.ON_DEMAND, prices={"tpu": 4.0}),
    )

    spend = await service.workload_cost(_principal(Role.PROJECT_OPERATOR), "wl-1")

    assert spend.on_demand_hourly == 4.0
    assert "pool-od" in spend.on_demand_source


@pytest.mark.asyncio
async def test_an_unpriced_pool_reports_unknown_rather_than_free() -> None:
    """An unpriced pool settles to zero, which is not the same as costing nothing."""

    service = await _plane_with(_pool("pool-spot", capacity_class=CapacityClass.SPOT, prices={}), settled=0.0)

    spend = await service.workload_cost(_principal(Role.PROJECT_OPERATOR), "wl-1")

    assert not spend.priced
    # Measured, so still known: the hours are real even when the money is not.
    assert spend.accelerator_hours == {"tpu": 8.0}


@pytest.mark.asyncio
async def test_a_current_pool_price_cannot_price_a_legacy_lease() -> None:
    service = await _plane_with(
        _pool("pool-spot", capacity_class=CapacityClass.SPOT, prices={"tpu": 1.2}),
        hourly_cost=0.0,
    )

    spend = await service.workload_cost(_principal(Role.PROJECT_OPERATOR), "wl-1")

    assert not spend.priced
    assert spend.shape == "v5p-8"
    assert spend.accelerator_hours == {"tpu": 8.0}


@pytest.mark.asyncio
async def test_a_run_placed_by_the_advanced_api_is_unknowable_rather_than_unpriced() -> None:
    """No pool label means nothing can be said about prices, not that there are none."""

    service = await _plane_with(pool_id="not-a-pool")

    spend = await service.workload_cost(_principal(Role.PROJECT_OPERATOR), "wl-1")

    assert not spend.priced
    assert spend.shape is None


@pytest.mark.asyncio
async def test_a_supplied_price_overrides_the_fleet_and_says_so() -> None:
    service = await _plane_with(
        _pool("pool-spot", capacity_class=CapacityClass.SPOT, prices={"tpu": 1.2}),
        _pool("pool-od", capacity_class=CapacityClass.ON_DEMAND, prices={"tpu": 4.0}),
    )

    spend = await service.workload_cost(_principal(Role.PROJECT_OPERATOR), "wl-1", on_demand_hourly=7.5)

    assert spend.on_demand_hourly == 7.5
    assert "--on-demand-price" in spend.on_demand_source


@pytest.mark.asyncio
async def test_another_tenant_cannot_read_what_a_run_cost() -> None:
    """Authorized as a read of the workload, which is the permission it implies."""

    service = await _plane_with(_pool("pool-spot", capacity_class=CapacityClass.SPOT, prices={"tpu": 1.2}))
    outsider = Principal(
        subject="mallory",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="rival", project_id="research"),),
    )

    with pytest.raises(AuthorizationError):
        await service.workload_cost(outsider, "wl-1")


# --------------------------------------------------------------------------- the command renders what it is told


class _Plane:
    """A control plane that answers the cost route and nothing else."""

    def __init__(self, document: dict) -> None:
        self.document = document
        self.params: dict | None = None

    def request(self, method, path, *, params=None, json=None, headers=None):
        del method, json, headers
        assert path == "/api/v1/workloads/wl-1/cost", f"the command read {path}"
        self.params = params
        return self.document

    def request_text(self, method, path, *, params=None, headers=None):
        del method, path, params, headers
        return ""

    def replace_identity_headers(self, headers) -> None:
        del headers


def _document(**overrides) -> dict:
    return {
        "workload_id": "wl-1",
        "state": "succeeded",
        "accelerator_hours": {"tpu": 8.0},
        "settled": 9.6,
        "priced": True,
        "reserved": 0.0,
        "leases": 1,
        "shape": "v5p-8",
        "capacity_class": "spot",
        "on_demand_hourly": 4.0,
        "on_demand_source": "declared by pool pool-od",
        **overrides,
    }


def _invoke(monkeypatch, plane: _Plane, *args: str):
    monkeypatch.setattr(cli, "EyvanClient", lambda *_a, **_k: plane)
    return CliRunner().invoke(
        cli.main, ["--user", "alice", "--org", "acme", "--project", "research", "workload", "cost", "wl-1", *args]
    )


def test_the_command_prints_the_planes_answer(monkeypatch) -> None:
    plane = _Plane(_document())

    result = _invoke(monkeypatch, plane)

    assert result.exit_code == 0, result.output
    assert "v5p-8" in result.output
    assert "cost        $9.60" in result.output
    assert "on demand   $32.00" in result.output
    assert "saved       $22.40" in result.output


def test_the_command_passes_a_supplied_price_to_the_plane(monkeypatch) -> None:
    """Resolved where the pools are, so one implementation decides the price."""

    plane = _Plane(_document(on_demand_hourly=7.5, on_demand_source="from --on-demand-price"))

    result = _invoke(monkeypatch, plane, "--on-demand-price", "7.5")

    assert result.exit_code == 0, result.output
    assert plane.params == {"on_demand_hourly": 7.5}
    assert "--on-demand-price" in result.output


def test_an_unpriced_run_prints_unknown_rather_than_zero(monkeypatch) -> None:
    plane = _Plane(_document(priced=False, settled=0.0, on_demand_hourly=None, on_demand_source="no pool declares one"))

    result = _invoke(monkeypatch, plane)

    assert "cost        unknown" in result.output
    assert "$0.00" not in result.output
