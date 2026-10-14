# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Module-level tests for the standalone collective ops (mesh mode, manual mode, configs)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import (
    AllGatherConfig,
    AllReduceConfig,
    ReduceScatterConfig,
    all_gather,
    all_reduce,
    reduce_scatter,
)
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec

if len(jax.devices()) < 2:
    pytest.skip("collective tests need at least 2 devices", allow_module_level=True)

_TP = min(4, len(jax.devices()))


def _mesh() -> Mesh:
    return Mesh(np.array(jax.devices()[:_TP]), axis_names=("tp",))


def _stacked_sum(x: jax.Array) -> jax.Array:
    return x.reshape(_TP, x.shape[0] // _TP, *x.shape[1:]).sum(0)


def test_all_reduce_mesh_mode_sums_partials():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(0), (8 * _TP, 128), dtype=jnp.float32)
    out = all_reduce(x, "tp", mesh=mesh)
    assert out.shape == (8, 128)
    assert jnp.allclose(out, _stacked_sum(x), rtol=1e-5, atol=1e-5)


def test_all_reduce_manual_mode_and_grad():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(1), (8 * _TP, 128), dtype=jnp.float32)
    fn = shard_map(
        lambda v: all_reduce(v, "tp"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    assert jnp.allclose(fn(x), _stacked_sum(x), rtol=1e-5, atol=1e-5)
    g = jax.grad(lambda v: jnp.sum(fn(v) ** 2))(x)
    g_ref = jax.grad(lambda v: jnp.sum(_stacked_sum(v) ** 2))(x)
    assert jnp.allclose(g, g_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ["auto", "one_shot", "ring"])
def test_all_reduce_mode_override_is_accepted_on_xla(mode: str):
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(2), (8 * _TP, 128), dtype=jnp.float32)
    out = all_reduce(x, "tp", mode=mode, platform="xla", mesh=mesh)
    assert jnp.allclose(out, _stacked_sum(x), rtol=1e-5, atol=1e-5)


def test_all_reduce_cfg_override():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(3), (8 * _TP, 128), dtype=jnp.float32)
    cfg = AllReduceConfig(mode="ring", platform="xla", backend="any")
    out = all_reduce(x, "tp", cfg=cfg, mesh=mesh)
    assert jnp.allclose(out, _stacked_sum(x), rtol=1e-5, atol=1e-5)


def test_all_gather_mesh_mode_reconstructs_global():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(4), (8 * _TP, 64), dtype=jnp.float32)
    out = all_gather(x, "tp", mesh=mesh)
    assert out.shape == x.shape
    assert jnp.allclose(out, x)


def test_all_gather_axis1_mesh_mode():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(5), (16, 32 * _TP), dtype=jnp.float32)
    out = all_gather(x, "tp", gather_axis=1, mesh=mesh)
    assert out.shape == x.shape
    assert jnp.allclose(out, x)


def test_all_gather_grad_flows():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(6), (8 * _TP, 64), dtype=jnp.float32)
    fn = shard_map(
        lambda v: all_gather(v, "tp"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    g = jax.grad(lambda v: jnp.sum(fn(v) ** 2))(x)
    assert jnp.allclose(g, 2.0 * x, rtol=1e-5, atol=1e-5)


def test_reduce_scatter_mesh_mode_sums_and_shards():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(7), (8 * _TP, 128), dtype=jnp.float32)
    out = reduce_scatter(x, "tp", mesh=mesh)
    assert out.shape == (8, 128)
    assert jnp.allclose(out, _stacked_sum(x), rtol=1e-5, atol=1e-5)


def test_reduce_scatter_grad_flows():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(8), (8 * _TP, 64), dtype=jnp.float32)
    fn = shard_map(
        lambda v: reduce_scatter(v, "tp"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec("tp"),
        check_vma=False,
    )
    g = jax.grad(lambda v: jnp.sum(fn(v) ** 2))(x)
    g_ref = jax.grad(lambda v: jnp.sum(_stacked_sum(v) ** 2))(x)
    assert jnp.allclose(g, g_ref, rtol=1e-5, atol=1e-5)


def test_configs_are_hashable_and_defaulted():
    assert AllReduceConfig().mode == "auto"
    assert AllGatherConfig().mode == "auto"
    assert ReduceScatterConfig().mode == "auto"
    assert hash(AllReduceConfig()) == hash(AllReduceConfig())
    assert AllGatherConfig(mode="one_shot") != AllGatherConfig(mode="ring")
    assert AllGatherConfig(mode="one_shot") == AllGatherConfig(mode="one_shot")
