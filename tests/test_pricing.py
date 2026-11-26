# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from eyvan._internal.compat import UTC
from eyvan.domain.pricing import PriceKey, PriceSample, PricingCatalog


def sample(key: PriceKey, price: str, *, age: int = 0) -> PriceSample:
    return PriceSample(key, Decimal(price), "test", datetime.now(UTC) - timedelta(seconds=age))


def test_catalog_prefers_most_specific_matching_price() -> None:
    generic = sample(PriceKey("gcp", "v5p-16", "spot"), "4")
    project = sample(PriceKey("gcp", "v5p-16", "spot", project="research"), "3")
    catalog = PricingCatalog((generic, project))

    assert catalog.quote(PriceKey("gcp", "v5p-16", "spot", project="research")).hourly_unit_usd == Decimal("3")
    assert catalog.quote(PriceKey("gcp", "v5p-16", "spot", project="other")).hourly_unit_usd == Decimal("4")


def test_projection_reserves_cost_and_accelerator_hours() -> None:
    key = PriceKey("gcp", "v6e-8", "on-demand")
    catalog = PricingCatalog((sample(key, "2.5"),))

    projection = catalog.project(key, accelerator_units=8, duration_seconds=1800)

    assert projection.hourly_cost_usd == Decimal("20.0")
    assert projection.accelerator_hours == Decimal("4")
    assert projection.reserved_cost_usd == Decimal("10.0")


def test_catalog_rejects_unknown_price() -> None:
    with pytest.raises(LookupError, match="no price"):
        PricingCatalog().quote(PriceKey("gcp", "v5p-16", "spot"))


def test_quote_ties_resolve_by_scope_tuple_not_insertion_order() -> None:
    now = datetime.now(UTC)
    by_project = PriceSample(PriceKey("gcp", "v5p-16", "spot", project="research"), Decimal("3"), "test", now)
    by_region = PriceSample(PriceKey("gcp", "v5p-16", "spot", region="us-central2"), Decimal("4"), "test", now)
    key = PriceKey("gcp", "v5p-16", "spot", project="research", region="us-central2")

    quotes = {PricingCatalog(order).quote(key) for order in ((by_project, by_region), (by_region, by_project))}

    assert len(quotes) == 1
    # The lexicographically greatest scope tuple wins: ("research", ...) > ("", ...).
    assert quotes.pop().hourly_unit_usd == Decimal("3")
