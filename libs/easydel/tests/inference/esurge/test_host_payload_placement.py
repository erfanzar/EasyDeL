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

"""Per-step host-payload placement under the ZMQ step-replication plane.

On a multi-process run ``jax.device_put`` of a host array onto a sharding that spans every process first
all-gathers the value to check that all processes passed the same one, and blocks on the result. Under the ZMQ
plane the payloads are identical by construction, so the runner places them with ``put_host_payload`` from the
local copy instead. These tests pin that the result is indistinguishable from ``jax.device_put`` (values,
canonical dtype, shape, sharding, weak type) for replicated and sharded targets, tuples, lists, scalars and
already-placed arrays, and which policies take that path. The skipped collective only exists on a multi-host
slice, which these single-process tests cannot reach.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners import host_sync
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@pytest.fixture
def shardings():
    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    return NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec("x"))


@pytest.fixture
def multi_process(monkeypatch):
    # The local-placement branch exists only on multi-process runs; act as one of two processes.
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: 2)


def _host_values(num_devices: int) -> list[np.ndarray]:
    rows = 2 * num_devices
    return [
        np.arange(rows * 3, dtype=np.int64).reshape(rows, 3),  # canonicalized to int32 without x64
        np.linspace(-1.0, 1.0, rows, dtype=np.float64),  # canonicalized to float32 without x64
        np.arange(rows) % 3 == 0,
        np.full((rows,), -1, dtype=np.int32),
    ]


def _assert_identical(placed, reference) -> None:
    assert isinstance(placed, jax.Array)
    assert placed.shape == reference.shape
    assert placed.dtype == reference.dtype
    assert placed.weak_type == reference.weak_type
    assert placed.sharding == reference.sharding
    np.testing.assert_array_equal(np.asarray(placed), np.asarray(reference))


def test_replicated_plane_places_host_arrays_like_device_put(multi_process, shardings):
    for value in _host_values(len(jax.devices())):
        for sharding in shardings:
            placed = host_sync.put_host_payload(value, sharding, host_payloads_replicated=True)
            _assert_identical(placed, jax.device_put(value, sharding))


def test_tuples_lists_and_numpy_scalars_keep_their_structure(multi_process, shardings):
    replicated, sharded = shardings
    ids, weights, mask, _ = _host_values(len(jax.devices()))
    values = (ids, weights, np.int32(7))
    targets = (sharded, replicated, replicated)
    placed = host_sync.put_host_payload(values, targets, host_payloads_replicated=True)
    assert isinstance(placed, tuple)
    for got, want in zip(placed, jax.device_put(values, targets), strict=True):
        _assert_identical(got, want)

    placed_list = host_sync.put_host_payload([ids, mask], replicated, host_payloads_replicated=True)
    assert isinstance(placed_list, list)
    for got, value in zip(placed_list, [ids, mask], strict=True):
        _assert_identical(got, jax.device_put(value, replicated))


def test_device_arrays_scalars_and_default_placement_match_device_put(multi_process, shardings):
    replicated, sharded = shardings
    on_device = jax.device_put(np.arange(2 * len(jax.devices()), dtype=np.int32), replicated)
    _assert_identical(
        host_sync.put_host_payload(on_device, sharded, host_payloads_replicated=True),
        jax.device_put(on_device, sharded),
    )
    _assert_identical(
        host_sync.put_host_payload(3, replicated, host_payloads_replicated=True),
        jax.device_put(3, replicated),
    )
    # A target that is not a Sharding (None = default placement) keeps plain device_put semantics.
    host = np.arange(4, dtype=np.int64)
    _assert_identical(
        host_sync.put_host_payload(host, None, host_payloads_replicated=True),
        jax.device_put(host, None),
    )


@pytest.mark.parametrize(
    ("process_count", "replicated", "skips_check"),
    [(1, True, False), (1, False, False), (2, False, False), (2, True, True)],
)
def test_only_replicated_multi_process_runs_skip_the_equality_check(
    monkeypatch, shardings, process_count, replicated, skips_check
):
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: process_count)
    assert host_sync.device_put_equality_check_redundant(replicated) is skips_check
    value = _host_values(len(jax.devices()))[0]
    for sharding in shardings:
        _assert_identical(
            host_sync.put_host_payload(value, sharding, host_payloads_replicated=replicated),
            jax.device_put(value, sharding),
        )
