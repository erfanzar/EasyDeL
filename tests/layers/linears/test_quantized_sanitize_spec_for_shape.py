# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from types import SimpleNamespace

from jax.sharding import PartitionSpec

from easydel.infra.sharding import mesh_partition_product, sanitize_partition_spec_for_shape


def _fake_mesh(shape: dict):
    return SimpleNamespace(shape=shape)


def test_mesh_partition_product_none():
    assert mesh_partition_product(_fake_mesh({"tp": 4}), None) == 1


def test_mesh_partition_product_single():
    assert mesh_partition_product(_fake_mesh({"tp": 4}), "tp") == 4


def test_mesh_partition_product_tuple():
    assert mesh_partition_product(_fake_mesh({"tp": 4, "dp": 2}), ("tp", "dp")) == 8


def test_mesh_partition_product_missing_axis():
    assert mesh_partition_product(_fake_mesh({"tp": 4}), "fsdp") == 1


def test_sanitize_noop_when_divisible():
    spec = PartitionSpec(None, "tp")
    result = sanitize_partition_spec_for_shape(spec, (128, 96), _fake_mesh({"tp": 4}))
    # 96 % 4 == 0 -> no change
    assert result == PartitionSpec(None, "tp")


def test_sanitize_drops_axis_when_not_divisible():
    spec = PartitionSpec(None, "tp")
    result = sanitize_partition_spec_for_shape(spec, (128, 97), _fake_mesh({"tp": 4}))
    # 97 % 4 != 0 -> drop "tp"
    assert result == PartitionSpec(None, None)


def test_sanitize_preserves_other_axes():
    spec = PartitionSpec("dp", "tp")
    result = sanitize_partition_spec_for_shape(spec, (8, 97), _fake_mesh({"dp": 2, "tp": 4}))
    # dim0: 8 % 2 == 0 -> keep; dim1: 97 % 4 != 0 -> drop
    assert result == PartitionSpec("dp", None)


def test_sanitize_unsharded_spec_unchanged():
    spec = PartitionSpec(None, None)
    result = sanitize_partition_spec_for_shape(spec, (7, 3), _fake_mesh({"tp": 4}))
    assert result == PartitionSpec(None, None)
