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

"""dp replicas of one serving launch: replica worlds planned from the global mesh, per-replica limits."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os

import numpy as np
import pytest
from easydel.inference.esurge.serving import ServingConfig
from easydel.inference.esurge.serving.app import per_replica_sections, replica_dims
from easydel.inference.esurge.serving.replicas import (
    ReplicaPlan,
    box_layout,
    plan_replicas,
    split_dp_slices,
)

pytestmark = pytest.mark.cpu_ok

AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


@dataclasses.dataclass(frozen=True)
class _Chip:
    id: int
    coords: tuple[int, int, int]
    process_index: int


def _v5p_128() -> list[_Chip]:
    """64 chips of a 4x4x4 torus, 2x2x1 chips per host, numbered like the real slice."""
    chips = []
    for z, y, x in itertools.product(range(4), range(4), range(4)):
        chips.append(_Chip(len(chips), (x, y, z), x // 2 + 2 * (y // 2) + 4 * z))
    return chips


def _dp2_mesh(chips: list[_Chip]) -> np.ndarray:
    """A (1,2,1,4,8,1) mesh whose dp slices are the y<2 and y>=2 halves (as create_device_mesh lays them out)."""
    halves = [sorted((c for c in chips if (c.coords[1] >= 2) == bool(r)), key=lambda c: c.id) for r in (0, 1)]
    return np.asarray(halves, dtype=object).reshape(1, 2, 1, 4, 8, 1)


class _KV:
    def __init__(self):
        self.store: dict[str, str] = {}

    def key_value_set(self, key, value, allow_overwrite=False):
        assert allow_overwrite or key not in self.store
        self.store[key] = value

    def blocking_key_value_get(self, key, timeout_in_ms):
        return self.store[key]


def _plan_all(chips, mesh):
    kv = _KV()
    procs = sorted({c.process_index for c in chips})
    # Every process publishes before any reads: run the publishing half first by planning twice.
    for p in procs:
        local = [c for c in chips if c.process_index == p]
        kv.key_value_set(
            f"easydel/esurge/serving/replica_host/19667/{p}",
            json.dumps({"address": f"10.0.0.{p}", "position": [0, 0, 0]}),
            allow_overwrite=True,
        )
    out = {}
    for p in procs:
        local = [c for c in chips if c.process_index == p]
        out[p] = plan_replicas(
            mesh,
            AXES,
            process_index=p,
            all_devices=chips,
            local_devices=local,
            address=f"10.0.0.{p}",
            client=kv,
            port=19667,
            timeout_s=1,
        )
    # Second pass: every process now sees every other's real host position.
    return {
        p: plan_replicas(
            mesh,
            AXES,
            process_index=p,
            all_devices=chips,
            local_devices=[c for c in chips if c.process_index == p],
            address=f"10.0.0.{p}",
            client=kv,
            port=19667,
            timeout_s=1,
        )
        for p in procs
    }


def test_each_dp_half_of_a_v5p_128_becomes_its_own_slice():
    chips = _v5p_128()
    plans = _plan_all(chips, _dp2_mesh(chips))
    groups: dict[int, list[tuple[int, int]]] = {}
    for p, (plan, env) in plans.items():
        groups.setdefault(plan.index, []).append((plan.rank, p))
        assert plan.num_processes == 8 and plan.count == 2
        assert env["TPU_TOPOLOGY"] == "4x2x4"
        assert env["TPU_TOPOLOGY_WRAP"] == "true,false,true"  # y spans 2 of 4 rows: no wraparound there
        assert env["TPU_HOST_BOUNDS"] == "2,1,4" and env["TPU_CHIPS_PER_HOST_BOUNDS"] == "2,2,1"
        assert env["TPU_ACCELERATOR_TYPE"] == "v5p-64" and env["TPU_WORKER_ID"] == str(plan.rank)
    assert sorted(p for _, p in groups[0]) == [0, 1, 4, 5, 8, 9, 12, 13]
    assert sorted(p for _, p in groups[1]) == [2, 3, 6, 7, 10, 11, 14, 15]
    for members in groups.values():
        assert sorted(r for r, _ in members) == list(range(8))
        # Worker order is libtpu's host order (z, y, x) inside the box, and rank 0 hosts the coordinator.
        ordered = [p for _, p in sorted(members)]
        hostnames = plans[ordered[0]][1]["TPU_WORKER_HOSTNAMES"].split(",")
        assert hostnames == [f"10.0.0.{p}" for p in ordered]
        assert plans[ordered[0]][0].coordinator == f"10.0.0.{ordered[0]}:19667"
        assert all(plans[p][0].leaders == plans[ordered[0]][0].leaders for _, p in members)
    front = [p for p, (plan, _) in plans.items() if plan.is_front]
    assert front == [0]


def test_dp_slices_sharing_a_host_keep_one_engine():
    chips = _v5p_128()
    # dp split along x inside every host: each process owns chips of both slices.
    halves = [sorted((c for c in chips if (c.coords[0] % 2) == r), key=lambda c: c.id) for r in (0, 1)]
    mesh = np.asarray(halves, dtype=object).reshape(1, 2, 1, 4, 8, 1)
    assert split_dp_slices(mesh, AXES) is None
    assert split_dp_slices(np.asarray(chips, dtype=object).reshape(1, 1, 1, 8, 8, 1), AXES) is None


def test_a_slice_that_is_not_a_whole_host_box_is_refused():
    chips = _v5p_128()
    scattered = [c for c in chips if c.process_index in (0, 3, 5, 6, 9, 10, 12, 15)]
    assert box_layout(scattered, chips, [c for c in chips if c.process_index == 0]) is None


def test_plan_round_trips_and_hands_the_token_through_a_pipe():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"tok-123")
    os.close(write_fd)
    plan = ReplicaPlan(index=1, rank=3, num_processes=8, coordinator="10.0.0.2:19667", leaders=("a", "b"))
    back = ReplicaPlan.from_json(dataclasses.replace(plan, token_fd=read_fd).to_json())
    assert back.leaders == ("a", "b") and not back.is_front
    assert back.read_token() == "tok-123"
    assert "tok-123" not in back.to_json()


def test_launch_wide_limits_split_over_replicas():
    sections = {
        "runtime": {"max_num_seqs": 256, "max_num_batched_tokens": 32768, "max_num_seq_buckets": [32, 64, 256]},
        "cache": {"max_cache_tokens": 8118528, "page_size": 128},
        "distributed": {"coordination": "zmq"},
    }
    out = per_replica_sections(sections, 2)
    assert out["runtime"] == {"max_num_seqs": 128, "max_num_batched_tokens": 16384, "max_num_seq_buckets": [16, 32, 128]}
    assert out["cache"] == {"max_cache_tokens": 4059264, "page_size": 128}
    assert out["distributed"] == {"coordination": "zmq"}
    assert sections["runtime"]["max_num_seqs"] == 256, "input must not be modified"
    config = ServingConfig.from_mapping({"model": {"checkpoint": "gs://b/c", "sharding_axis_dims": [1, 2, 1, 4, 8, 1]}})
    assert replica_dims(config) == (1, 1, 1, 4, 8, 1)
