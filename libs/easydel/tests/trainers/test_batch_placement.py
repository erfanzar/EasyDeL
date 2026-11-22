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

"""Batch leaves reach the compiled step batch-sharded instead of replicated on every device."""

from __future__ import annotations

import jax
import ml_dtypes
import numpy as np
import pytest
from easydel.trainers.batch_placement import BatchPlacedStep, batch_leaf_sharding, place_batch
from jax.sharding import Mesh, NamedSharding, PartitionSpec

SPEC = PartitionSpec(("dp", "fsdp"), "sp")


@pytest.fixture(scope="module")
def mesh():
    devices = np.array(jax.devices()).reshape(1, -1, 1, 1, 1)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def _batch(n: int) -> dict:
    rows = 2 * n
    rng = np.random.default_rng(0)
    return {
        "input_ids": rng.integers(0, 100, size=(rows, 8), dtype=np.int32),
        "position_ids": rng.integers(0, 8, size=(3, rows, 8), dtype=np.int32),
        "embeds": rng.standard_normal((rows, 8, 4)).astype(ml_dtypes.bfloat16),
        "table": np.arange(n * 2 + 1, dtype=np.int32),
        "names": np.array(["a", "b"]),
        "flag": None,
    }


def test_leaves_are_batch_sharded_and_values_preserved(mesh):
    n = mesh.devices.size
    batch = _batch(n)
    placed = place_batch(batch, SPEC, mesh)

    assert placed["input_ids"].sharding.spec == PartitionSpec(("dp", "fsdp"), "sp")
    assert placed["position_ids"].sharding.spec == PartitionSpec(None, ("dp", "fsdp"), "sp")
    assert placed["embeds"].sharding.is_equivalent_to(NamedSharding(mesh, PartitionSpec(("dp", "fsdp"), "sp")), 3)
    assert placed["embeds"].dtype == ml_dtypes.bfloat16
    if n > 1:
        # 2n+1 rows do not divide the batch axes, so the table stays replicated.
        assert placed["table"].sharding.spec == PartitionSpec()
        assert placed["input_ids"].addressable_shards[0].data.shape == (2, 8)
    for key in ("input_ids", "position_ids", "embeds", "table"):
        np.testing.assert_array_equal(np.asarray(placed[key]), batch[key])
    assert placed["names"] is batch["names"]
    assert placed["flag"] is None


def test_already_placed_leaf_is_reused_and_wrong_placement_is_resharded(mesh):
    batch = _batch(mesh.devices.size)
    placed = place_batch(batch, SPEC, mesh)
    again = place_batch(placed, SPEC, mesh)
    assert again["input_ids"] is placed["input_ids"]

    replicated = jax.device_put(batch["input_ids"], NamedSharding(mesh, PartitionSpec()))
    moved = place_batch({"input_ids": replicated}, SPEC, mesh)["input_ids"]
    assert moved.sharding.is_equivalent_to(batch_leaf_sharding(batch["input_ids"].shape, SPEC, mesh), 2)
    np.testing.assert_array_equal(np.asarray(moved), batch["input_ids"])


def test_spec_axes_missing_from_mesh_are_dropped():
    devices = np.array(jax.devices()).reshape(-1)
    small = Mesh(devices, ("fsdp",))
    sharding = batch_leaf_sharding((2 * devices.size, 8), SPEC, small)
    assert sharding.spec == PartitionSpec("fsdp", None)


def test_wrapped_step_receives_sharded_batch_and_matches_unsharded_result(mesh):
    n = mesh.devices.size
    batch = _batch(n)

    def step(scale, data, k):
        return (data["input_ids"].astype(np.float32) * scale).sum() + k + data["position_ids"].sum()

    compiled = jax.jit(step, in_shardings=(None, None), static_argnums=(2,))
    wrapped = BatchPlacedStep(compiled, batch_argnum=1, partition_spec=SPEC, mesh=mesh)
    wrapped.static_argnums_ = (2,)
    assert compiled.static_argnums_ == (2,)

    numeric = {k: batch[k] for k in ("input_ids", "position_ids")}
    out = wrapped(np.float32(2.0), numeric, 3)
    expected = numeric["input_ids"].astype(np.float64).sum() * 2.0 + 3 + numeric["position_ids"].sum()
    assert float(out) == pytest.approx(expected)

    input_shardings = wrapped.lower(np.float32(2.0), numeric, 3).compile().input_shardings[0]
    expected_ids = NamedSharding(mesh, PartitionSpec(("dp", "fsdp"), "sp"))
    expected_pos = NamedSharding(mesh, PartitionSpec(None, ("dp", "fsdp"), "sp"))
    assert input_shardings[1]["input_ids"].is_equivalent_to(expected_ids, 2)
    assert input_shardings[1]["position_ids"].is_equivalent_to(expected_pos, 3)
