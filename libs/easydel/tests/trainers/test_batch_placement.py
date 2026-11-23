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
from easydel.trainers.batch_placement import (
    BatchPlacedStep,
    _local_slice_for_global_index,
    batch_leaf_sharding,
    batch_shard_group_rows,
    place_batch,
)
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


@pytest.mark.cpu_ok
def test_batch_shard_group_rows_partitions_multi_host_torus():
    import types

    from easydel.trainers import data_lanes
    from jax.experimental import mesh_utils

    # Simulated 64-chip (16-host) 3D torus: dp=2, fsdp=8, tp=4 -> 8 2-host groups, non-contiguous dp/fsdp rows.
    class _Chip:
        def __init__(self, idx, coords, proc):
            self.id = idx
            self.coords = coords
            self.core_on_chip = 0
            self.process_index = proc
            self.slice_index = 0
            self.platform = "tpu"
            self.device_kind = "TPU v5"

    chips = []
    for z in range(4):
        for yb in range(2):
            for xb in range(2):
                proc = xb + 2 * yb + 4 * z
                for y in range(2):
                    for x in range(2):
                        chips.append(_Chip(len(chips), (2 * xb + x, 2 * yb + y, z), proc))
    names = ("pp", "dp", "fsdp", "ep", "tp", "sp")
    devs = mesh_utils.create_device_mesh((1, 2, 8, 1, 4, 1), chips, allow_split_physical_axes=True)
    fake_mesh = types.SimpleNamespace(devices=devs, axis_names=names)
    groups = data_lanes.batch_shard_process_groups(devs, names, ("dp", "fsdp"))
    assert len(groups) == 8

    for global_bs in (16, 32):
        seen: list[int] = []
        for grp in groups:
            rows_per_proc = {batch_shard_group_rows(global_bs, SPEC, fake_mesh, process_index=p) for p in grp}
            # Every host in the same batch-shard group must get the exact same sorted global row tuple.
            assert len(rows_per_proc) == 1
            rows = next(iter(rows_per_proc))
            assert rows == tuple(sorted(set(rows)))
            seen.extend(rows)
        # Across all groups, rows partition range(global_bs) with zero overlap and zero gaps.
        assert sorted(seen) == list(range(global_bs))

    # Replicated fallback when global_batch_size is not divisible by batch_ways (16).
    assert batch_shard_group_rows(12, SPEC, fake_mesh, process_index=0) == tuple(range(12))
    # Replicated fallback when no batch axis is on the mesh.
    no_batch_mesh = types.SimpleNamespace(devices=devs.reshape(64), axis_names=("tp",))
    assert batch_shard_group_rows(16, SPEC, no_batch_mesh, process_index=0) == tuple(range(16))


def test_batch_shard_group_rows_matches_live_jax_mesh(mesh):
    import types

    g_live = 2 * mesh.devices.size
    duck_mesh = types.SimpleNamespace(devices=mesh.devices, axis_names=mesh.axis_names)
    assert batch_shard_group_rows(g_live, SPEC, mesh) == tuple(range(g_live))
    assert batch_shard_group_rows(g_live, SPEC, mesh) == batch_shard_group_rows(g_live, SPEC, duck_mesh)


@pytest.mark.cpu_ok
def test_local_slice_for_global_index_handles_contiguous_noncontiguous_and_mrope():
    # Simulated group owning non-contiguous global rows (4, 36) out of global_batch_size=64.
    group_rows = (4, 36)
    g2l = {g: i for i, g in enumerate(group_rows)}
    host_2d = np.stack([np.full(8, 100, dtype=np.int32), np.full(8, 200, dtype=np.int32)], axis=0)
    # Device owning global row 4 -> local row 0 (zero-copy view).
    s0 = _local_slice_for_global_index(host_2d, (slice(4, 5), slice(0, 8)), 0, 64, g2l)
    np.testing.assert_array_equal(s0, host_2d[0:1])
    assert np.shares_memory(s0, host_2d)
    # Device owning global row 36 with sequence-parallel slice 4:8 -> local row 1, cols 4:8.
    s1 = _local_slice_for_global_index(host_2d, (slice(36, 37), slice(4, 8)), 0, 64, g2l)
    np.testing.assert_array_equal(s1, host_2d[1:2, 4:8])
    assert np.shares_memory(s1, host_2d)

    # mRoPE (3, local_B, S) lifted along batch_axis=1.
    host_mrope = np.arange(3 * 2 * 6, dtype=np.int32).reshape(3, 2, 6)
    m1 = _local_slice_for_global_index(host_mrope, (slice(0, 3), slice(36, 37), slice(2, 6)), 1, 64, g2l)
    np.testing.assert_array_equal(m1, host_mrope[:, 1:2, 2:6])

    # Non-contiguous stride fallback (e.g. if a slice hits rows in non-adjacent local slots).
    non_contig = _local_slice_for_global_index(
        np.arange(3 * 4, dtype=np.int32).reshape(3, 4),
        (slice(0, 4, 2), slice(0, 4)),
        0,
        4,
        {0: 0, 1: 1, 2: 2},
    )
    np.testing.assert_array_equal(non_contig, np.array([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=np.int32))


def test_place_batch_local_lifting_validation_and_global_compat(mesh, monkeypatch):
    import easydel.trainers.batch_placement as bp

    n = mesh.devices.size
    global_bs = 4 * max(n, 2)
    local_bs = global_bs // 2
    # Simulate a process group owning the second half of the global rows when we capture callbacks.
    fake_rows = tuple(range(local_bs, global_bs))
    monkeypatch.setattr(bp, "batch_shard_group_rows", lambda g, s, m, **kw: fake_rows)

    captured: dict[str, tuple[tuple[int, ...], NamedSharding, np.ndarray]] = {}

    def _fake_make_array(shape, sharding, cb):
        # Invoke the callback on a global slice inside fake_rows to verify local remapping.
        if shape[0] == global_bs:
            sample = cb((slice(local_bs, local_bs + 1), *(slice(0, d) for d in shape[1:])))
            captured[f"axis0_{shape}"] = (shape, sharding, sample)
        elif len(shape) == 3 and shape[0] == 3 and shape[1] == global_bs:
            sample = cb((slice(0, 3), slice(local_bs, local_bs + 1), slice(0, shape[2])))
            captured[f"mrope_{shape}"] = (shape, sharding, sample)
        return np.zeros(shape, dtype=np.int32)

    monkeypatch.setattr(jax, "make_array_from_callback", _fake_make_array)

    local_ids = np.arange(local_bs * 8, dtype=np.int32).reshape(local_bs, 8) + 10
    local_pos = np.arange(3 * local_bs * 8, dtype=np.int32).reshape(3, local_bs, 8) + 100
    aux = np.arange(17, dtype=np.int32)
    placed = place_batch(
        {"input_ids": local_ids, "position_ids": local_pos, "table": aux, "flag": None},
        SPEC,
        mesh,
        global_batch_size=global_bs,
    )
    assert placed["input_ids"].shape == (global_bs, 8)
    assert placed["position_ids"].shape == (3, global_bs, 8)
    assert placed["table"].shape == (17,)
    assert placed["flag"] is None
    np.testing.assert_array_equal(captured[f"axis0_{(global_bs, 8)}"][2], local_ids[0:1])
    np.testing.assert_array_equal(captured[f"mrope_{(3, global_bs, 8)}"][2], local_pos[:, 0:1, :])

    # Global-sized batch still accepted cleanly when global_batch_size is passed.
    global_ids = np.arange(global_bs * 8, dtype=np.int32).reshape(global_bs, 8)
    placed_g = place_batch({"input_ids": global_ids}, SPEC, mesh, global_batch_size=global_bs)
    assert placed_g["input_ids"].shape == (global_bs, 8)

    # Mismatched batch size raises loudly.
    with pytest.raises(ValueError, match="expected host batch size"):
        place_batch(
            {"input_ids": np.zeros((local_bs + 1, 8), dtype=np.int32)},
            SPEC,
            mesh,
            global_batch_size=global_bs,
        )

    # Flat packed image embeds in a local-lifted batch raise loudly.
    with pytest.raises(ValueError, match="flat packed embed key"):
        place_batch(
            {"input_ids": local_ids, "image_embeds": np.zeros((4, 16), dtype=np.float32)},
            SPEC,
            mesh,
            global_batch_size=global_bs,
        )


def test_batch_placed_step_with_global_batch_size_preserves_attrs(mesh):
    n = mesh.devices.size
    batch = _batch(n)
    global_bs = 2 * n

    def step(scale, data, k):
        return (data["input_ids"].astype(np.float32) * scale).sum() + k

    compiled = jax.jit(step, in_shardings=(None, None), static_argnums=(2,))
    wrapped = BatchPlacedStep(compiled, batch_argnum=1, partition_spec=SPEC, mesh=mesh)
    wrapped.static_argnums_ = (2,)
    lifted = wrapped.with_global_batch_size(global_bs)
    assert lifted.static_argnums_ == (2,)
    assert lifted._global_batch_size == global_bs
    out = lifted(np.float32(2.0), {"input_ids": batch["input_ids"]}, 5)
    assert float(out) == pytest.approx(batch["input_ids"].astype(np.float64).sum() * 2.0 + 5)


@pytest.mark.cpu_ok
def test_execution_accumulation_passes_global_microbatch_size_to_place_batch(monkeypatch):
    from easydel.trainers.execution_accumulation import ExecutionAccumulation

    calls: list[tuple[tuple[int, ...], int | None]] = []

    def _mock_place(part, spec, mesh, *, global_batch_size=None):
        del spec, mesh
        calls.append((tuple(part["input_ids"].shape), global_batch_size))
        return part

    monkeypatch.setattr("easydel.trainers.execution_accumulation.place_batch", _mock_place)

    ea = object.__new__(ExecutionAccumulation)
    ea.num_steps = 2
    ea._batch_spec = SPEC
    ea._mesh = object()
    ea._global_microbatch_size = 64

    local_batch = {"input_ids": np.arange(4 * 8, dtype=np.int32).reshape(4, 8)}
    micros = list(ea.microbatches(local_batch))
    assert len(micros) == 2
    assert calls == [((2, 8), 64), ((2, 8), 64)]

    # Backward compat: when _global_microbatch_size is None (or unset), 3-arg place_batch mocks still work.
    three_arg_calls: list[int] = []
    monkeypatch.setattr(
        "easydel.trainers.execution_accumulation.place_batch",
        lambda part, spec, mesh: (three_arg_calls.append(int(part["input_ids"].shape[0])) or part),
    )
    del ea._global_microbatch_size
    list(ea.microbatches(local_batch))
    assert three_arg_calls == [2, 2]


@pytest.mark.cpu_ok
def test_multi_host_torus_local_batches_reconstruct_exact_global_batch_and_tp_replicas_agree():
    import types

    from easydel.trainers import data_lanes
    from jax.experimental import mesh_utils

    class _Chip:
        def __init__(self, idx, coords, proc):
            self.id = idx
            self.coords = coords
            self.core_on_chip = 0
            self.process_index = proc
            self.slice_index = 0
            self.platform = "tpu"
            self.device_kind = "TPU v5"

    chips = []
    for z in range(4):
        for yb in range(2):
            for xb in range(2):
                proc = xb + 2 * yb + 4 * z
                for y in range(2):
                    for x in range(2):
                        chips.append(_Chip(len(chips), (2 * xb + x, 2 * yb + y, z), proc))
    names = ("pp", "dp", "fsdp", "ep", "tp", "sp")
    devs = mesh_utils.create_device_mesh((1, 2, 8, 1, 4, 1), chips, allow_split_physical_axes=True)
    fake_mesh = types.SimpleNamespace(devices=devs, axis_names=names)
    groups = data_lanes.batch_shard_process_groups(devs, names, ("dp", "fsdp"))

    global_bs, seq_len = 32, 6
    global_data = np.arange(global_bs * seq_len, dtype=np.int32).reshape(global_bs, seq_len)

    # Every group loads ONLY its own local rows; hosts in the same group share identical local arrays.
    per_proc_local: dict[int, tuple[np.ndarray, dict[int, int]]] = {}
    for grp in groups:
        rows = batch_shard_group_rows(global_bs, SPEC, fake_mesh, process_index=grp[0])
        local_arr = global_data[list(rows)].copy()
        g2l = {g: i for i, g in enumerate(rows)}
        for p in grp:
            per_proc_local[p] = (local_arr, g2l)

    # Every device in the 64-chip torus gets its batch coordinate's slice from its own host's local array.
    reconstructed = np.full_like(global_data, -1)
    by_batch_coord: dict[tuple[int, int], list[np.ndarray]] = {}
    shard_size = global_bs // (2 * 8)
    for pos in np.ndindex(devs.shape):
        dev = devs[pos]
        dp_i, fsdp_i = pos[1], pos[2]
        rank = dp_i * 8 + fsdp_i
        g_slice = slice(rank * shard_size, (rank + 1) * shard_size)
        local_arr, g2l = per_proc_local[dev.process_index]
        dev_shard = _local_slice_for_global_index(local_arr, (g_slice, slice(0, seq_len)), 0, global_bs, g2l)
        by_batch_coord.setdefault((dp_i, fsdp_i), []).append(dev_shard)
        reconstructed[g_slice] = dev_shard

    # All 4 TP replicas of every (dp, fsdp) shard agree bit-for-bit, and global tensor is exact.
    for shards in by_batch_coord.values():
        assert len(shards) == 4
        for s in shards[1:]:
            np.testing.assert_array_equal(s, shards[0])
    np.testing.assert_array_equal(reconstructed, global_data)
