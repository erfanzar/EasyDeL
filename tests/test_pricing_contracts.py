# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public pricing boundaries, freshness, scope selection and reservation provenance."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import permutations

import pytest

from eyvan.domain.pricing import PriceKey, PriceSample, PricingCatalog

NOW = datetime(2026, 9, 12, tzinfo=UTC)
KEY = PriceKey("provider", "accelerator", "spot", project="123456", region="1", zone="2")


def _sample(key=KEY, price=Decimal("1.25"), *, age=0, source="configured"):
    return PriceSample(key, price, source, NOW - timedelta(seconds=age))


@pytest.mark.parametrize("price", [0, 0.125, "0.125", Decimal("0.125")])
def test_sample_normalizes_finite_nonnegative_money(price):
    sample = _sample(price=price)
    assert isinstance(sample.hourly_unit_usd, Decimal)
    assert sample.hourly_unit_usd == Decimal(str(price))


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf"), float("-inf"), Decimal("sNaN"), "not-a-price", ""])
def test_sample_refuses_invalid_money_with_price_context(price):
    with pytest.raises(ValueError) as error:
        _sample(price=price)
    assert "price" in str(error.value).lower()


def test_sample_refuses_naive_observation_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        PriceSample(KEY, Decimal("1"), "configured", NOW.replace(tzinfo=None))


def test_put_retains_newest_sample_in_either_insertion_order():
    old = _sample(price=Decimal("1"), age=60)
    new = _sample(price=Decimal("2"))
    for ordered in permutations((old, new)):
        catalog = PricingCatalog()
        for sample in ordered:
            catalog.put(sample)
        assert catalog.quote(KEY) == new


def test_specificity_precedes_recency():
    old_specific = _sample(replace(KEY, zone=None), age=60)
    new_generic = _sample(replace(KEY, project=None, region=None, zone=None))
    for ordered in permutations((old_specific, new_generic)):
        assert PricingCatalog(ordered).quote(KEY) == old_specific


def test_recency_precedes_scope_tie_breaker_for_equal_specificity():
    old_project = _sample(replace(KEY, region=None, zone=None), age=60)
    new_region = _sample(replace(KEY, project=None, zone=None))
    for ordered in permutations((old_project, new_region)):
        assert PricingCatalog(ordered).quote(KEY) == new_region


def test_equal_rank_scope_selection_is_independent_of_insertion_order():
    # Scope identifiers are opaque: numeric project/region names remain valid.
    project = _sample(replace(KEY, region=None, zone=None))
    region = _sample(replace(KEY, project=None, zone=None))
    zone = _sample(replace(KEY, project=None, region=None))
    for ordered in permutations((project, region, zone)):
        assert PricingCatalog(ordered).quote(KEY) == project
    for ordered in permutations((region, zone)):
        assert PricingCatalog(ordered).quote(KEY) == region


@pytest.mark.parametrize("dimension", ["provider", "accelerator_type", "capacity_class", "project", "region", "zone"])
def test_mismatched_scope_cannot_supply_a_quote(dimension):
    catalog = PricingCatalog((_sample(replace(KEY, **{dimension: "other"})),))
    with pytest.raises(LookupError):
        catalog.quote(KEY)


@pytest.mark.parametrize("dimension", ["project", "region", "zone"])
def test_scoped_sample_cannot_answer_an_unscoped_query(dimension):
    catalog = PricingCatalog((_sample(),))
    with pytest.raises(LookupError):
        catalog.quote(replace(KEY, **{dimension: None}))


@pytest.mark.parametrize(("units", "seconds"), [(0, 1), (-1, 1), (1, 0), (1, -1), (0, 0)])
def test_projection_refuses_nonpositive_units_or_duration(units, seconds):
    with pytest.raises(ValueError) as error:
        PricingCatalog((_sample(),)).project(KEY, accelerator_units=units, duration_seconds=seconds)
    assert "positive" in str(error.value).lower()


@pytest.mark.parametrize(("units", "seconds"), [(1, 1), (1, 3600), (3, 900), (7, 450)])
def test_projection_preserves_amounts_and_selected_provenance(units, seconds):
    sample = _sample(source="provider-observation")
    result = PricingCatalog((sample,)).project(KEY, accelerator_units=units, duration_seconds=seconds)
    expected_hours = Decimal(units * seconds) / Decimal(3600)
    assert result.accelerator_hours == expected_hours
    assert result.hourly_cost_usd == Decimal("1.25") * units
    assert result.reserved_cost_usd == Decimal("1.25") * expected_hours
    assert all(
        isinstance(value, Decimal)
        for value in (result.accelerator_hours, result.hourly_cost_usd, result.reserved_cost_usd)
    )
    assert result.price_observed_at == NOW
    assert result.price_source == "provider-observation"
