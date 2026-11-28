# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Price resolution from durable fleet state, and the drift it makes visible.

``eyvan/pricing.py`` held the package's only specificity-ranked price lookup --
global < project < region < zone, with a deterministic tiebreak and provenance on
every quote -- and nothing imported it, while the prices that actually govern
admission sat in flat ``unit_prices`` dicts resolved by nothing.

Two things are asserted here. **Parity**: a catalog quote and the cell's own
``hourly_cost`` describe the same money, which is what makes it safe to use the
catalog for pre-submission estimates. **Drift**: two records that should price
the same shape and do not are reported -- the failure nothing else in the package
can see, because the scheduler is handed prices and a budget sums what it is
told, so a low-lying stale price wins every cost-ranked comparison silently.
"""

from __future__ import annotations

from decimal import Decimal

from eyvan.domain.pricing import PriceKey
from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    ResourceLimits,
)
from eyvan.server.service.pricing_resolver import (
    accelerator_unit_price,
    audit_price_drift,
    catalog_from_pools,
)


def _pool(
    pool_id: str = "research/v5p-16",
    *,
    prices: dict[str, float] | None = None,
    zone: str | None = "us-east5-a",
    region: str = "us-east5",
    accelerator_type: str | None = "v5p-16",
    capacity_class: CapacityClass = CapacityClass.ON_DEMAND,
) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type="v5p-16",
        provider="manual",
        capacity_backend="manual",
        region=region,
        zone=zone,
        agent_endpoint="http://agent:8080",
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        unit_prices=prices if prices is not None else {"tpu": 3.0},
        accelerator_type=accelerator_type,
        capacity_class=capacity_class,
    )


def _cell(
    cell_id: str = "cell-a",
    *,
    prices: dict[str, float] | None = None,
    zone: str | None = "us-east5-a",
    region: str = "us-east5",
    accelerator_type: str | None = "v5p-16",
    capacity_class: CapacityClass = CapacityClass.ON_DEMAND,
    tpu: int = 8,
) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="manual",
        capacity_backend="manual",
        region=region,
        zone=zone,
        state=CellState.READY,
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=tpu),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=tpu),
        unit_prices=prices if prices is not None else {"tpu": 3.0},
        accelerator_type=accelerator_type,
        capacity_class=capacity_class,
    )


def test_an_unpriced_record_is_distinguishable_from_a_zero_priced_one() -> None:
    """Only one of the two is a defect, so they must not collapse to one answer."""

    assert accelerator_unit_price({}) is None
    assert accelerator_unit_price({"cpu": 0.05}) is None, "cpu is not an accelerator dimension"
    assert accelerator_unit_price({"tpu": 0.0}) == ("tpu", Decimal("0"))
    assert accelerator_unit_price({"gpu": 2.5}) == ("gpu", Decimal("2.5"))


def test_a_catalog_quote_carries_the_pool_it_came_from() -> None:
    """Provenance is why this module exists: a price without a source is unauditable."""

    catalog = catalog_from_pools([_pool()])

    quote = catalog.quote(
        PriceKey(
            provider="manual",
            accelerator_type="v5p-16",
            capacity_class="on_demand",
            region="us-east5",
            zone="us-east5-a",
        )
    )

    assert quote.hourly_unit_usd == Decimal("3.0")
    assert quote.source == "pool:research/v5p-16"


def test_a_zone_price_wins_over_the_region_price_for_that_zone() -> None:
    """Specificity ranking is the property core/ does not have."""

    catalog = catalog_from_pools(
        [
            _pool("research/regional", zone=None, prices={"tpu": 3.0}),
            _pool("research/zonal", zone="us-east5-b", prices={"tpu": 5.0}),
        ]
    )

    zonal = catalog.quote(
        PriceKey(
            provider="manual",
            accelerator_type="v5p-16",
            capacity_class="on_demand",
            region="us-east5",
            zone="us-east5-b",
        )
    )
    elsewhere = catalog.quote(
        PriceKey(
            provider="manual",
            accelerator_type="v5p-16",
            capacity_class="on_demand",
            region="us-east5",
            zone="us-east5-c",
        )
    )

    assert zonal.hourly_unit_usd == Decimal("5.0")
    assert elsewhere.hourly_unit_usd == Decimal("3.0"), "an unlisted zone falls back to the region"


def test_spot_and_on_demand_are_priced_separately() -> None:
    catalog = catalog_from_pools(
        [
            _pool("research/od", prices={"tpu": 3.0}, capacity_class=CapacityClass.ON_DEMAND),
            _pool("research/spot", prices={"tpu": 1.0}, capacity_class=CapacityClass.SPOT),
        ]
    )

    spot = catalog.quote(
        PriceKey(
            provider="manual",
            accelerator_type="v5p-16",
            capacity_class="spot",
            region="us-east5",
            zone="us-east5-a",
        )
    )

    assert spot.hourly_unit_usd == Decimal("1.0")


def test_a_catalog_quote_and_the_cells_own_hourly_cost_describe_the_same_money() -> None:
    """The parity that makes the catalog usable as a pre-submission estimator."""

    pool = _pool(prices={"tpu": 3.0})
    cell = _cell(prices={"tpu": 3.0}, tpu=8)
    catalog = catalog_from_pools([pool])

    quote = catalog.quote(
        PriceKey(
            provider="manual",
            accelerator_type="v5p-16",
            capacity_class="on_demand",
            region="us-east5",
            zone="us-east5-a",
        )
    )
    projected = quote.hourly_unit_usd * cell.capacity.tpu

    assert float(projected) == cell.hourly_cost() == 24.0


def test_a_cell_that_disagrees_with_its_pool_is_reported() -> None:
    findings = audit_price_drift(pools=[_pool(prices={"tpu": 3.0})], cells=[_cell(prices={"tpu": 9.0})])

    assert len(findings) == 1
    finding = findings[0]
    assert finding.record_id == "cell-a"
    assert finding.observed_hourly_unit == Decimal("9.0")
    assert finding.expected_hourly_unit == Decimal("3.0")
    assert finding.expected_source == "pool:research/v5p-16"
    assert finding.relative_delta == Decimal("2")


def test_a_cell_that_agrees_with_its_pool_is_not_reported() -> None:
    assert audit_price_drift(pools=[_pool(prices={"tpu": 3.0})], cells=[_cell(prices={"tpu": 3.0})]) == ()


def test_a_deliberate_zone_override_is_not_drift() -> None:
    """Otherwise every legitimate per-zone rate would read as a mistake."""

    pools = [
        _pool("research/regional", zone=None, prices={"tpu": 3.0}),
        _pool("research/zonal", zone="us-east5-b", prices={"tpu": 5.0}),
    ]
    cells = [
        _cell("cell-a", zone="us-east5-a", prices={"tpu": 3.0}),
        _cell("cell-b", zone="us-east5-b", prices={"tpu": 5.0}),
    ]

    assert audit_price_drift(pools=pools, cells=cells) == ()


def test_an_unpriced_cell_is_not_drift_because_it_is_a_different_defect() -> None:
    """Conflating the two would bury D7, which admits work for free."""

    assert audit_price_drift(pools=[_pool(prices={"tpu": 3.0})], cells=[_cell(prices={})]) == ()


def test_a_shape_no_pool_prices_is_a_coverage_gap_not_a_disagreement() -> None:
    """Reporting it here would name an expectation that does not exist."""

    findings = audit_price_drift(
        pools=[_pool(accelerator_type="v5p-16")],
        cells=[_cell("cell-x", accelerator_type="v6e-16", prices={"tpu": 9.0})],
    )

    assert findings == ()


def test_findings_are_ordered_deterministically() -> None:
    """A doctor report that reshuffles between runs cannot be diffed."""

    pools = [_pool(prices={"tpu": 3.0})]
    cells = [_cell(f"cell-{name}", prices={"tpu": 9.0}) for name in ("c", "a", "b")]

    findings = audit_price_drift(pools=pools, cells=cells)

    assert [finding.record_id for finding in findings] == ["cell-a", "cell-b", "cell-c"]


def test_relative_delta_is_zero_rather_than_undefined_against_a_zero_price() -> None:
    """A division-by-zero here would crash the report and hide every other finding."""

    findings = audit_price_drift(pools=[_pool(prices={"tpu": 0.0})], cells=[_cell(prices={"tpu": 4.0})])

    assert len(findings) == 1
    assert findings[0].relative_delta == Decimal(0)
