# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public vector contracts for methods excluded by the dataclass mutation pass."""

from decimal import Decimal

import pytest

from eyvan.domain import ResourceVector


@pytest.mark.parametrize("values", [[("cpu", 1), ("cpu", 1)], [("cpu", 1), ("cpu", 2)]])
def test_duplicate_resource_pairs_are_rejected_even_when_equal(values):
    with pytest.raises(ValueError, match=r"cpu.*more than once"):
        ResourceVector(values)


def test_keyword_resource_cannot_overwrite_mapping_entry():
    with pytest.raises(ValueError, match=r"cpu.*more than once"):
        ResourceVector({"cpu": 1}, cpu=2)


def test_returned_dictionary_cannot_mutate_vector():
    vector = ResourceVector(cpu=2)
    copied = vector.as_dict()
    copied["cpu"] = Decimal(99)
    assert vector["cpu"] == Decimal(2)


def test_zero_resource_preserves_mapping_presence_without_positive_capacity():
    vector = ResourceVector(cpu=0)
    assert not vector
    assert len(vector) == 1
    assert tuple(vector) == ("cpu",)
    assert vector["cpu"] == 0
    with pytest.raises(KeyError):
        _ = vector["gpu"]
    assert vector.get("gpu", Decimal("0.25")) == Decimal("0.25")


def test_subtraction_retains_union_and_exact_decimal_difference():
    left = ResourceVector(cpu=Decimal("0.3"), gpu=2)
    right = ResourceVector(cpu=Decimal("0.1"), unused=0)
    assert (left - right).as_dict() == {"cpu": Decimal("0.2"), "gpu": Decimal(2), "unused": Decimal(0)}
    assert left["cpu"] == Decimal("0.3")


@pytest.mark.parametrize("right", [ResourceVector(cpu=2), ResourceVector(gpu=1)])
def test_subtraction_refuses_underflow_in_present_or_missing_resource(right):
    with pytest.raises(ValueError, match="negative"):
        _ = ResourceVector(cpu=1) - right


@pytest.mark.parametrize("factor", [-1, Decimal("NaN"), float("inf"), True])
def test_scaling_refuses_invalid_multiplier_even_for_empty_vector(factor):
    with pytest.raises((ValueError, TypeError)):
        ResourceVector().scale(factor)


def test_missing_capacity_is_zero_but_missing_limit_is_unconstrained():
    request = ResourceVector(cpu=1)
    assert not request.fits_within(ResourceVector())
    assert request.within_limits(ResourceVector())
    assert request.fits_within(ResourceVector(cpu=1))
    assert request.within_limits(ResourceVector(cpu=1))
    assert not request.within_limits(ResourceVector(cpu=Decimal("0.5")))


@pytest.mark.parametrize(
    ("vector", "capacity", "expected"),
    [
        (ResourceVector(), ResourceVector(), Decimal(0)),
        (ResourceVector(cpu=0), ResourceVector(), Decimal(0)),
        (ResourceVector(cpu=1), ResourceVector(), Decimal("Infinity")),
        (ResourceVector(cpu=1), ResourceVector(cpu=0), Decimal("Infinity")),
        (ResourceVector(cpu=1, gpu=3), ResourceVector(cpu=4, gpu=4), Decimal("0.75")),
    ],
)
def test_dominant_share_respects_zero_missing_and_largest_dimension(vector, capacity, expected):
    assert vector.dominant_share(capacity) == expected
