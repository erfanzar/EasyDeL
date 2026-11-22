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

"""StatePacker: pack/unpack roundtrip, jit-boundary equality, budget/grouping rules."""

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.state_packing import StatePacker
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@pytest.fixture()
def mesh():
    return Mesh(np.array(jax.devices()).reshape(-1), axis_names=("d",))


def _make_leaves(mesh):
    """Emulate a flat model state: two packable per-layer kinds + odd singletons."""
    n_dev = mesh.devices.size
    replicated = NamedSharding(mesh, PartitionSpec())
    sharded = NamedSharding(mesh, PartitionSpec("d", None))
    rng = np.random.default_rng(0)
    leaves = []
    # kind A: 6 sharded [n_dev*2, 16] f32 (packable, group of 6)
    for _ in range(6):
        leaves.append(jax.device_put(rng.normal(size=(n_dev * 2, 16)).astype(np.float32), sharded))
    # kind B: 4 replicated [8] f32 norms (packable, group of 4)
    for _ in range(4):
        leaves.append(jax.device_put(rng.normal(size=(8,)).astype(np.float32), replicated))
    # singleton embedding-like [32, 16] + one int32 buffer (unpackable kinds)
    leaves.append(jax.device_put(rng.normal(size=(32, 16)).astype(np.float32), replicated))
    leaves.append(jax.device_put(np.arange(8, dtype=np.int32), replicated))
    # a pair (below _MIN_GROUP_LEAVES): must pass through
    for _ in range(2):
        leaves.append(jax.device_put(rng.normal(size=(4, 4)).astype(np.float32), replicated))
    return tuple(leaves)


def test_pack_unpack_roundtrip_through_jit(mesh):
    leaves = _make_leaves(mesh)
    packer = StatePacker(leaves)
    assert packer.is_effective
    # 6+4 grouped into 2 args; 4 passthrough -> 6 args total vs 14 leaves
    assert len(packer.groups) == 2
    assert packer.packed_arg_count == 6

    packed = packer.pack(leaves)
    shardings = packer.packed_shardings([leaf.sharding for leaf in leaves])
    assert len(packed) == len(shardings)
    for arr, s in zip(packed, shardings, strict=True):
        assert arr.sharding.is_equivalent_to(s, arr.ndim)

    @jax.jit
    def restore(packed_args):
        return packer.unpack(packed_args)

    restored = restore(packed)
    assert len(restored) == len(leaves)
    for orig, back in zip(leaves, restored, strict=True):
        assert orig.dtype == back.dtype
        assert orig.shape == back.shape
        np.testing.assert_array_equal(np.asarray(orig), np.asarray(back))


def test_repack_after_weight_swap_matches_plan(mesh):
    leaves = _make_leaves(mesh)
    packer = StatePacker(leaves)
    packed_a = packer.pack(leaves)
    # new weights, same structure (a hot-swap)
    swapped = tuple(
        jax.device_put(jnp.asarray((np.asarray(leaf) * 2).astype(leaf.dtype)), leaf.sharding) for leaf in leaves
    )
    packed_b = packer.pack(swapped)
    assert len(packed_a) == len(packed_b)
    for a, b in zip(packed_a, packed_b, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
    restored = packer.unpack(packed_b)
    for orig, back in zip(swapped, restored, strict=True):
        np.testing.assert_array_equal(np.asarray(orig), np.asarray(back))


def test_budget_excludes_large_groups(mesh):
    leaves = _make_leaves(mesh)
    # budget below the [n_dev*2, 16] group's bytes but above the tiny norm group
    norm_group_bytes = 4 * 8 * 4
    packer = StatePacker(leaves, max_packed_bytes=norm_group_bytes)
    assert len(packer.groups) == 1
    assert packer.groups[0].shape == (8,)
    assert packer.packed_bytes <= norm_group_bytes
    restored = packer.unpack(packer.pack(leaves))
    for orig, back in zip(leaves, restored, strict=True):
        np.testing.assert_array_equal(np.asarray(orig), np.asarray(back))


def test_small_groups_pass_through(mesh):
    replicated = NamedSharding(mesh, PartitionSpec())
    leaves = tuple(
        jax.device_put(np.full((4,), i, np.float32), replicated) for i in range(3)
    )  # 3 identical leaves < _MIN_GROUP_LEAVES
    packer = StatePacker(leaves)
    assert not packer.groups
    assert not packer.is_effective
