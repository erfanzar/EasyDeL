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

"""Host-assembled checkpoint leaf placement (issue #290).

Multi-process loads build each global parameter from host memory instead of ``device_put``-ing a single-device
tensor onto a global sharding. The host path is forced here so it is exercised on a single host.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from easydel.modules.auto.auto_configuration import place_leaf_on_sharding
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices())
    return Mesh(devices.reshape(-1), ("fsdp",))


def test_host_assembly_matches_values_and_target_sharding():
    mesh = _mesh()
    n = mesh.devices.size
    target = NamedSharding(mesh, PartitionSpec(None, "fsdp"))
    leaf = jax.device_put(
        jnp.arange(8 * 16 * n, dtype=jnp.bfloat16).reshape(8, 16 * n),
        jax.devices()[0],
    )

    placed = place_leaf_on_sharding(leaf, target, from_host=True)

    assert placed.sharding == target
    assert placed.dtype == leaf.dtype
    np.testing.assert_array_equal(np.asarray(placed), np.asarray(leaf))
    for shard in placed.addressable_shards:
        assert shard.data.shape == (8, 16)


def test_host_and_device_put_paths_agree():
    mesh = _mesh()
    target = NamedSharding(mesh, PartitionSpec("fsdp"))
    leaf = jnp.linspace(-1.0, 1.0, 4 * mesh.devices.size, dtype=jnp.float32)

    via_host = place_leaf_on_sharding(leaf, target, from_host=True)
    via_device = place_leaf_on_sharding(leaf, target, from_host=False)

    assert via_host.sharding == via_device.sharding == target
    np.testing.assert_array_equal(np.asarray(via_host), np.asarray(via_device))
