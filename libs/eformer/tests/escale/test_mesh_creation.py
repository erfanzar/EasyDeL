# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mesh creation utilities."""

import jax
import pytest
from jax.sharding import AxisType

from eformer.escale import create_mesh


def _has_multiple_slices() -> bool:
    devices = jax.devices()
    if devices and hasattr(devices[0], "slice_index"):
        try:
            return len({d.slice_index for d in devices}) > 1
        except Exception:
            return False
    return False


@pytest.mark.parametrize("use_jax", [True, False])
@pytest.mark.parametrize(
    ("axis_type", "expected"),
    [
        ("auto", AxisType.Auto),
        ("explicit", AxisType.Explicit),
        ("manual", AxisType.Manual),
    ],
)
def test_create_mesh_axis_types_strings(use_jax, axis_type, expected):
    if use_jax and _has_multiple_slices():
        pytest.skip("jax.make_mesh does not support multi-slice meshes")
    axis_names = ("x", "y")
    mesh = create_mesh(
        axis_dims=(1, -1),
        axis_names=axis_names,
        use_jax=use_jax,
        axis_types=axis_type,
    )
    assert mesh.axis_names == axis_names
    assert mesh.axis_types == (expected, expected)
