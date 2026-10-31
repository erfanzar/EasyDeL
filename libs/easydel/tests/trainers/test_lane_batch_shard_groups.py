# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Which hosts the grain data lanes treat as one row shard (fake devices, no array computation).

The trainer hands each host's batch to the step as a replicated host value and every device keeps the rows of its own
batch shard, so hosts whose devices share a batch shard must read identical rows.
"""

from __future__ import annotations

import types

import jax
import numpy as np
import pytest
from easydel.trainers import data_lanes
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec

pytestmark = pytest.mark.cpu_ok

NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
BATCH = ("dp", "fsdp")


class FakeChip:
    def __init__(self, idx, coords, process_index):
        self.id = idx
        self.coords = coords
        self.core_on_chip = 0
        self.process_index = process_index
        self.slice_index = 0
        self.platform = "tpu"
        self.device_kind = "TPU v5"


def v5p_128():
    """64 chips on a 4x4x4 torus, 16 hosts of 2x2x1 chips, process-major."""
    chips = []
    for z in range(4):
        for yb in range(2):
            for xb in range(2):
                proc = xb + 2 * yb + 4 * z
                for y in range(2):
                    for x in range(2):
                        chips.append(FakeChip(len(chips), (2 * xb + x, 2 * yb + y, z), proc))
    return chips


def per_host_mesh():
    """Each host holds one fsdp row (its 4 chips are the tp group), as the per-host hybrid mesh lays it out."""
    chips = sorted(v5p_128(), key=lambda d: (d.process_index, d.id))
    return np.asarray(chips, dtype=object).reshape(1, 1, 16, 1, 4, 1)


def torus_mesh():
    return mesh_utils.create_device_mesh((1, 1, 16, 1, 4, 1), v5p_128(), allow_split_physical_axes=True)


def test_per_host_layout_keeps_one_shard_per_process():
    groups = data_lanes.batch_shard_process_groups(per_host_mesh(), NAMES, BATCH)
    assert groups == tuple((p,) for p in range(16))


def test_hosts_sharing_a_batch_shard_form_one_group():
    devs = torus_mesh()
    groups = data_lanes.batch_shard_process_groups(devs, NAMES, BATCH)
    assert len(groups) == 8 and all(len(g) == 2 for g in groups)
    assert sorted(p for g in groups for p in g) == list(range(16))
    # Every batch shard (fsdp index) lives inside exactly one group.
    member = {p: i for i, g in enumerate(groups) for p in g}
    for f in range(16):
        assert len({member[d.process_index] for d in devs[0, 0, f].flat}) == 1


def test_non_batch_axes_merge_and_absent_axes_are_ignored():
    devs = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            devs[i, j] = types.SimpleNamespace(process_index=2 * i + j)
    # Batch over "fsdp" (rows): hosts of one row share it.
    assert data_lanes.batch_shard_process_groups(devs, ("fsdp", "sp"), ("dp", "fsdp")) == ((0, 1), (2, 3))
    # No batch axis on the mesh: one batch shard, every host reads the same rows.
    assert data_lanes.batch_shard_process_groups(devs, ("tp", "sp"), ("dp", "fsdp")) == ((0, 1, 2, 3),)


def _trainer(devs, **arguments):
    defaults = dict(grain_shard_index=None, grain_shard_count=None, step_partition_spec=PartitionSpec(BATCH, "sp"))
    defaults.update(arguments)
    mesh = types.SimpleNamespace(devices=devs, axis_names=NAMES)
    return types.SimpleNamespace(arguments=types.SimpleNamespace(**defaults), mesh=mesh)


def test_host_shard_uses_the_batch_shard_group(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 16)
    devs = torus_mesh()
    groups = data_lanes.batch_shard_process_groups(devs, NAMES, BATCH)
    for proc in (0, 5, 15):
        monkeypatch.setattr(jax, "process_index", lambda proc=proc: proc)
        index, count = data_lanes._host_shard(_trainer(devs))
        assert count == 8 and proc in groups[index]
    monkeypatch.setattr(jax, "process_index", lambda: 5)
    assert data_lanes._host_shard(_trainer(per_host_mesh())) == (5, 16)


def test_explicit_grain_shard_wins(monkeypatch):
    monkeypatch.setattr(jax, "process_index", lambda: 5)
    monkeypatch.setattr(jax, "process_count", lambda: 16)
    assert data_lanes._host_shard(_trainer(torus_mesh(), grain_shard_index=3, grain_shard_count=7)) == (3, 7)
