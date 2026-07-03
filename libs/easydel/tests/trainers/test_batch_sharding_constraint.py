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

"""Rank-aware batch sharding: leaves that don't fit the (batch, seq) spec must
still get their batch dimension sharded instead of silently replicating.

Regression context: a `(3, batch, seq)` multimodal position-id tensor and the
rank-1 packed-embed row tables were landing fully replicated on every device
because the trainer applied one `P(("dp","fsdp"), "sp")` spec tree-wide.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.trainers.training_utils import constrain_batch_sharding

pytestmark = pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (fake) devices")


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:8]).reshape(1, 2, 2, 1, 2, 1)
    return Mesh(devices, axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"))


def _apply(batch, spec, mesh):
    with mesh:
        return jax.jit(lambda b: constrain_batch_sharding(b, spec, mesh=mesh))(batch)


def test_batch_leaves_get_rank_aware_specs():
    mesh = _mesh()
    spec = PartitionSpec(("dp", "fsdp"), "sp")
    batch = {
        "input_ids": np.zeros((8, 16), dtype=np.int32),
        "position_ids": np.zeros((3, 8, 16), dtype=np.int32),  # (K, batch, seq)
        "embed_rows": np.zeros((32,), dtype=np.int32),  # rank-1, divisible by dp*fsdp=4
        "odd_meta": np.zeros((5,), dtype=np.int32),  # fits nothing -> replicated
    }
    out = _apply(batch, spec, mesh)

    def assert_sharded_as(name, expected_spec):
        expected = NamedSharding(mesh, expected_spec)
        actual = out[name].sharding
        assert actual.is_equivalent_to(expected, out[name].ndim), (name, actual, expected)

    assert_sharded_as("input_ids", PartitionSpec(("dp", "fsdp"), "sp"))
    # dim0 (size 3) is not divisible by dp*fsdp; the batch axes must move to dim 1.
    assert_sharded_as("position_ids", PartitionSpec(None, ("dp", "fsdp"), "sp"))
    assert_sharded_as("embed_rows", PartitionSpec(("dp", "fsdp")))
    assert_sharded_as("odd_meta", PartitionSpec())

    # Sharded values must round-trip unchanged.
    np.testing.assert_array_equal(np.asarray(out["position_ids"]), batch["position_ids"])


def test_none_spec_is_noop():
    mesh = _mesh()
    batch = {"input_ids": np.zeros((8, 16), dtype=np.int32)}
    assert constrain_batch_sharding(batch, None, mesh=mesh) is batch
