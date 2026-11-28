# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public constructor diagnostics, canonical topology and capacity-class refusal."""

from decimal import Decimal

import pytest

from eyvan.domain.models import CapacityClass, ResourceGang, ResourceVector


@pytest.mark.parametrize("value", [True, False])
def test_boolean_resource_refusal_names_the_resource(value):
    with pytest.raises(TypeError) as error:
        ResourceVector(cpu=value)
    assert "cpu" in str(error.value)
    assert "bool" in str(error.value).lower()


@pytest.mark.parametrize("value", ["invalid-number", Decimal("NaN"), float("inf"), float("-inf")])
def test_invalid_resource_quantity_refusal_preserves_field_context(value):
    with pytest.raises(ValueError) as error:
        ResourceVector(cpu=value)
    assert "cpu" in str(error.value)
    assert "finite" in str(error.value).lower()


@pytest.mark.parametrize("name", ["", " cpu", "cpu ", "\t", 1, None])
def test_resource_names_require_nonempty_trimmed_strings(name):
    with pytest.raises(ValueError) as error:
        ResourceVector([(name, 1)])
    assert "resource name" in str(error.value)


def test_identical_topology_duplicates_are_canonicalized_without_conflict():
    gang = ResourceGang("gang", 1, ResourceVector(cpu=1), (("zone", "east"), ("rack", "one"), ("zone", "east")))
    assert gang.required_topology == (("rack", "one"), ("zone", "east"))


def test_conflicting_topology_values_are_refused_with_field_context():
    with pytest.raises(ValueError) as error:
        ResourceGang("gang", 1, ResourceVector(cpu=1), (("zone", "east"), ("zone", "west")))
    assert "required_topology" in str(error.value)
    assert "zone" in str(error.value)


@pytest.mark.parametrize("pair", [("", "east"), ("zone", ""), (1, "east"), ("zone", 1), ("zone", " east")])
def test_invalid_topology_pairs_name_the_owning_field(pair):
    with pytest.raises(ValueError) as error:
        ResourceGang("gang", 1, ResourceVector(cpu=1), (pair,))
    assert "required_topology" in str(error.value)


@pytest.mark.parametrize("value", [None, 1, True, []])
def test_nonstring_capacity_class_refusal_is_actionable(value):
    with pytest.raises(TypeError) as error:
        CapacityClass.parse(value)
    assert "capacity class" in str(error.value).lower()


@pytest.mark.parametrize("value", ["", "not-a-capacity-class"])
def test_unknown_capacity_class_refusal_lists_supported_choices(value):
    with pytest.raises(ValueError) as error:
        CapacityClass.parse(value)
    message = str(error.value)
    for expected in ("reserved", "on_demand", "spot", "flex_start"):
        assert expected in message
