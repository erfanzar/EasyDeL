# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve accelerator prices from durable fleet state, and audit their drift.

``eyvan/pricing.py`` has always held the only specificity-ranked price lookup in
the package -- global < project < region < zone, with a deterministic tiebreak
and price *provenance* on every quote -- and nothing imported it. Meanwhile the
prices that actually govern admission live as a flat ``unit_prices`` dict on each
pool and each cell, resolved by nothing at all: whatever the fleet YAML said.

This module joins the two. It builds a catalog from durable records, so a quote
is derived from the fleet an operator declared rather than from a second source
that can disagree with it, and it reports where those records disagree with each
other.

**Why the audit is the valuable half.** Cost-ranked placement compares
``hourly_cost`` across candidate cells. If two cells of the same shape in the
same zone carry different unit prices -- a stale pool, a hand-edited record, a
half-applied fleet change -- the scheduler prefers the one that lies lowest, and
does so silently and forever. Nothing in the package detects that today; a
budget cannot, because it sums whatever it is told.

The audit only reports. It does not rewrite prices: an operator's declared price
may be a negotiated rate this package has no standing to overrule, and silently
correcting one would replace a visible inconsistency with an invisible one.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from decimal import Decimal

from eyvan.domain.pricing import PriceKey, PriceSample, PricingCatalog
from eyvan.records import CapacityPoolRecord, CellRecord, utcnow

#: Resource dimension priced per accelerator. A TPU pool prices ``tpu`` and a GPU
#: pool prices ``gpu``; CPU-only pools have no accelerator and are priced by their
#: scalar dimensions alone, which this resolver deliberately does not model as an
#: accelerator quote.
ACCELERATOR_DIMENSIONS: tuple[str, ...] = ("tpu", "gpu")


@dataclass(frozen=True, slots=True)
class PriceDrift:
    """One disagreement between two records that should price the same thing."""

    accelerator_type: str
    capacity_class: str
    region: str
    zone: str | None
    #: The quote the catalog resolves for this shape, and where it came from.
    expected_hourly_unit: Decimal
    expected_source: str
    #: What this record actually declares.
    observed_hourly_unit: Decimal
    record_kind: tp.Literal["cell", "pool"]
    record_id: str

    @property
    def relative_delta(self) -> Decimal:
        """Return the signed fractional gap, or zero when the expectation is zero.

        Zero rather than an exception for a zero expectation: an unpriced shape is
        defect D7's condition and is reported by its own check, so raising here
        would turn one finding into a crash that hides the rest.
        """

        if self.expected_hourly_unit == 0:
            return Decimal(0)
        return (self.observed_hourly_unit - self.expected_hourly_unit) / self.expected_hourly_unit


def accelerator_unit_price(unit_prices: tp.Mapping[str, float]) -> tuple[str, Decimal] | None:
    """Return the priced accelerator dimension and its unit price, if there is one.

    ``None`` for a CPU-only or unpriced record. Unpriced is not zero-priced: a
    caller must be able to tell "this shape costs nothing per accelerator" from
    "nobody said what this shape costs", because only the second is a defect.
    """

    for dimension in ACCELERATOR_DIMENSIONS:
        price = unit_prices.get(dimension)
        if price is not None:
            return dimension, Decimal(str(price))
    return None


def catalog_from_pools(pools: tp.Sequence[CapacityPoolRecord]) -> PricingCatalog:
    """Build a price catalog from the pools an operator declared.

    Pools rather than cells because a pool is the thing an operator writes and a
    cell is the thing the autoscaler creates from it. Sourcing quotes from cells
    would let a drifted cell define the expectation it is being audited against.

    Note the catalog's matching rule, which is easy to get backwards: a sample is
    a candidate only when each of its scope dimensions is either unset or equal to
    the query's. So a zone-scoped price answers a zone-scoped question and *not* a
    region-wide one -- asking "what does this shape cost in us-east5" when only
    ``us-east5-a`` is priced correctly raises rather than inventing a regional
    average. Callers auditing a cell therefore quote with that cell's full scope.
    """

    catalog = PricingCatalog()
    observed_at = utcnow()
    for pool in pools:
        priced = accelerator_unit_price(pool.unit_prices)
        if priced is None or not pool.accelerator_type:
            continue
        _dimension, hourly = priced
        catalog.put(
            PriceSample(
                key=PriceKey(
                    provider=pool.provider,
                    accelerator_type=pool.accelerator_type,
                    capacity_class=str(pool.capacity_class),
                    region=pool.region,
                    zone=pool.zone,
                ),
                hourly_unit_usd=hourly,
                source=f"pool:{pool.id}",
                observed_at=observed_at,
            )
        )
    return catalog


def audit_price_drift(
    *,
    pools: tp.Sequence[CapacityPoolRecord],
    cells: tp.Sequence[CellRecord],
) -> tuple[PriceDrift, ...]:
    """Report every cell whose declared price disagrees with its pool's.

    A cell is audited against the catalog's most *specific* matching quote, which
    is what makes a zone-level override legitimate rather than a finding: an
    operator who priced ``us-east5-a`` differently from ``us-east5`` gets the
    narrower number and no drift, while a cell that matches a quote exactly and
    still disagrees with it gets reported.

    Unpriced cells are not drift. They are D7's shape and have their own check --
    conflating "priced wrong" with "not priced" would bury the second, which is
    the more dangerous of the two because it admits work for free.
    """

    catalog = catalog_from_pools(pools)
    findings: list[PriceDrift] = []
    for cell in cells:
        priced = accelerator_unit_price(cell.unit_prices)
        if priced is None or not cell.accelerator_type:
            continue
        _dimension, observed = priced
        key = PriceKey(
            provider=cell.provider,
            accelerator_type=cell.accelerator_type,
            capacity_class=str(cell.capacity_class),
            region=cell.region,
            zone=cell.zone,
        )
        try:
            quote = catalog.quote(key)
        except LookupError:
            # No pool prices this shape at all. That is a coverage gap, not a
            # disagreement, and reporting it here would name a phantom expectation.
            continue
        if quote.hourly_unit_usd == observed:
            continue
        findings.append(
            PriceDrift(
                accelerator_type=cell.accelerator_type,
                capacity_class=str(cell.capacity_class),
                region=cell.region,
                zone=cell.zone,
                expected_hourly_unit=quote.hourly_unit_usd,
                expected_source=quote.source,
                observed_hourly_unit=observed,
                record_kind="cell",
                record_id=cell.id,
            )
        )
    findings.sort(key=lambda finding: (finding.accelerator_type, finding.record_id))
    return tuple(findings)


__all__ = (
    "ACCELERATOR_DIMENSIONS",
    "PriceDrift",
    "accelerator_unit_price",
    "audit_price_drift",
    "catalog_from_pools",
)
