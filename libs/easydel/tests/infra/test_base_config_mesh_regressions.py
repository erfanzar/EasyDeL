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

from spectrax import PartitionAxis, PartitionManager

from easydel.infra.base_config import _mesh_shape_ep


def test_mesh_shape_ep_matches_folded_expert_layout_without_aliases():
    mesh = SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 2, "tp": 2, "sp": 1})
    pm = PartitionManager(PartitionAxis())

    shape, names = _mesh_shape_ep(mesh, pm, fsdp_is_ep_bound=True, sp_is_ep_bound=True)

    assert shape == (1, 2, 2)
    assert names == ("dp", "ep", "tp")


def test_mesh_shape_ep_does_not_double_count_aliased_axes():
    mesh = SimpleNamespace(shape={"ep": 2, "tp": 2, "sp": 1})
    pm = PartitionManager(
        PartitionAxis(
            data_parallel_axis="ep",
            fully_sharded_data_parallel_axis="ep",
            expert_parallel_axis="ep",
            tensor_parallel_axis="tp",
            sequence_parallel_axis="sp",
        )
    )

    shape, names = _mesh_shape_ep(mesh, pm, fsdp_is_ep_bound=True, sp_is_ep_bound=True)

    assert shape == (1, 2, 2)
    assert names == ("ep", "ep", "tp")
