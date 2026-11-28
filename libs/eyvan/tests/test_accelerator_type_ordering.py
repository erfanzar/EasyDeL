# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider-neutral presentation keeps opaque accelerator names usable."""

from itertools import permutations

import pytest

from eyvan.domain.accelerators import sort_accelerator_types
from eyvan.server.service.errors import NoMatchingCapacityPoolError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((), ()),
        (("v5p-128", "v5p-8", "v5p-16", "v5p-8"), ("v5p-8", "v5p-16", "v5p-128")),
        (("a-b-10", "a-b-2", "a-9"), ("a-9", "a-b-2", "a-b-10")),
        (("gpu-02", "gpu-2", "gpu-0"), ("gpu-0", "gpu-02", "gpu-2")),
        (("gpu-1", "gpu-0", "gpu"), ("gpu", "gpu-0", "gpu-1")),
        (("-2", "-10", "gpu-8"), ("-10", "-2", "gpu-8")),
        (("gpu-", "gpu", "-2", "gpu-large"), ("-2", "gpu", "gpu-", "gpu-large")),
        (("gpu-٢", "gpu-10", "gpu-2"), ("gpu-2", "gpu-٢", "gpu-10")),
        (("gpu-²", "gpu-10", "gpu-2"), ("gpu-2", "gpu-10", "gpu-²")),
        (("gpu-①", "gpu-8"), ("gpu-8", "gpu-①")),
    ],
)
def test_accelerator_names_have_stable_numeric_or_opaque_order(values, expected):
    for order in permutations(values):
        assert sort_accelerator_types(iter(order)) == expected


def test_unknown_shape_error_preserves_opaque_provider_names():
    error = NoMatchingCapacityPoolError(
        fleet="research",
        requested_accelerator_type="gpu-4",
        available_enabled_accelerator_types=("gpu-²", "gpu-8", "gpu-2"),
    )
    assert error.available_enabled_accelerator_types == ("gpu-2", "gpu-8", "gpu-²")
    assert error.requested_accelerator_type == "gpu-4"
    assert "no enabled capacity pool" in str(error)
