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
"""Accelerator tests: weight transport round-trips sharded pytrees bit-exactly."""

import socket

import jax
import jax.numpy as jnp
import numpy as np
from easydel.trainers._online_rl.weight_transport import (
    JaxTransferClient,
    JaxTransferServerBackend,
    RemoteSubscriberBridge,
    RemoteWeightSubscriber,
    SocketMessageChannel,
    WeightPublisher,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices())
    n = devices.size
    rows = 2 if n % 2 == 0 and n > 1 else 1
    return Mesh(devices.reshape(rows, n // rows), ("dp", "tp"))


def _sharded_tree(mesh: Mesh) -> dict:
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    rows, cols = mesh.devices.shape
    return {
        "layer": {
            "kernel": jax.device_put(
                jax.random.normal(k1, (8 * rows, 16 * cols), dtype=jnp.float32),
                NamedSharding(mesh, PartitionSpec("dp", "tp")),
            ),
            "bias": jax.device_put(
                jax.random.normal(k2, (16 * cols,), dtype=jnp.bfloat16),
                NamedSharding(mesh, PartitionSpec("tp")),
            ),
        },
        "embed": [
            jax.device_put(
                jax.random.randint(k3, (4, 8), 0, 1000, dtype=jnp.int32),
                NamedSharding(mesh, PartitionSpec()),
            )
        ],
    }


def _assert_bit_exact_with_shardings(received, sent) -> None:
    sent_leaves, sent_def = jax.tree.flatten(sent)
    received_leaves, received_def = jax.tree.flatten(received)
    assert received_def == sent_def
    for got, want in zip(received_leaves, sent_leaves, strict=True):
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        assert got.sharding.is_equivalent_to(want.sharding, want.ndim)
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


def test_in_process_transport_round_trips_sharded_pytree():
    mesh = _mesh()
    tree = _sharded_tree(mesh)
    released = []
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("sampler")
        sub.request()
        publisher.publish(1, tree, release=released.append)
        offer = sub.receive(timeout=5.0)
        received = sub.pull(offer)
        _assert_bit_exact_with_shardings(received, tree)
        sub.ack(offer)
        assert sub.installed_version == 1
    assert len(released) == 1 and released[0] is tree


def test_jax_transfer_backend_round_trips_between_two_endpoints():
    mesh = _mesh()
    tree = _sharded_tree(mesh)
    backend = JaxTransferServerBackend()
    client = JaxTransferClient(mesh=mesh)
    left, right = socket.socketpair()
    with WeightPublisher(backend, pull_timeout=120.0) as publisher:
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "sampler")
        remote = RemoteWeightSubscriber(SocketMessageChannel(right), client, pull_timeout=120.0)
        for version, scale in ((1, 1.0), (3, 2.0)):
            payload = jax.tree.map(lambda x, s=scale: (x * s).astype(x.dtype), tree)
            payload = jax.tree.map(lambda x, ref: jax.device_put(x, ref.sharding), payload, tree)
            remote.request()
            publisher.publish(version, payload)
            offer = remote.receive(timeout=30.0)
            assert offer.version == version
            assert offer.descriptor["kind"] == "jax_transfer"
            received = remote.pull(offer)
            _assert_bit_exact_with_shardings(received, payload)
            remote.ack(offer)
            assert remote.installed_version == version
        assert publisher.installed_version("sampler") == 3
        remote.close()
        bridge.join(timeout=10.0)
