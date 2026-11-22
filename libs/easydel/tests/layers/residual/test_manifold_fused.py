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

"""Opt-in fused (Pallas TPU) mHC path of ``ManifoldHyperConnection`` against its default XLA path."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from easydel.layers.residual import (
    HyperStreamSharding,
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    manifold_residual_write,
)
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="the fused path runs Pallas TPU kernels")

HC = 4


def _source(topology):
    if jax.device_count() < 4:
        pytest.skip("needs four devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(topology), ("data", "tokens", "features"))
    axis = spx.PartitionAxis(
        batch_axis="data",
        query_sequence_axis="tokens",
        hidden_state_axis="features",
        decode_batch_axis="data",
        decode_hidden_state_axis="features",
    )
    return SimpleNamespace(mesh=mesh, runtime_sharding_resolver=coerce_runtime_sharding_resolver(axis, mesh=mesh))


def _pair(dtype, hidden, source=None):
    """Same parameters in a default module and a ``fused_kernel="pallas"`` module."""
    cfg = ManifoldHyperConnectionConfig(hidden_size=hidden, hc_mult=HC, iters=8)
    kw = dict(dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, mesh_source=source)
    off = ManifoldHyperConnection(cfg, rngs=spx.Rngs(3), **kw)
    on = ManifoldHyperConnection(cfg, rngs=spx.Rngs(3), fused_kernel="pallas", fused_tile_rows=128, **kw)
    rng = np.random.default_rng(11)
    graph_off, state = spx.export(off)
    leaves = {
        k: jnp.asarray(rng.normal(0.05, 0.3, v.shape) * (0.1 if k.endswith("fn") else 1.0), dtype)
        for k, v in state.flatten().items()
    }
    state = spx.State.from_flat(leaves)
    graph_on, _ = spx.export(on)
    return graph_off, graph_on, state


def _rel(a, b):
    a = np.asarray(jnp.asarray(a, jnp.float32), np.float64)
    b = np.asarray(jnp.asarray(b, jnp.float32), np.float64)
    return np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)


def _loss(outputs, write):
    post, comb, collapsed = outputs
    w = jnp.linspace(-0.7, 1.3, collapsed.size).reshape(collapsed.shape)
    return (
        jnp.sum(post * jnp.linspace(-1.0, 1.0, post.size).reshape(post.shape))
        + jnp.sum(comb * jnp.linspace(0.3, -0.9, comb.size).reshape(comb.shape))
        + jnp.sum(collapsed.astype(jnp.float32) * w)
        + jnp.sum(write.astype(jnp.float32) * jnp.linspace(-0.5, 0.8, write.size).reshape(write.shape))
    )


def _site(graph, x, state):
    module = spx.bind(graph, state)
    post, comb, collapsed = module(x)
    return (post, comb, collapsed), module.residual_write(x, jnp.tanh(collapsed), post, comb)


def _check(graph_off, graph_on, state, x, fwd_tol, grad_tol):
    off = jax.jit(lambda x, s: _site(graph_off, x, s))(x, state)
    on = jax.jit(lambda x, s: _site(graph_on, x, s))(x, state)
    for name, a, b in zip(("post", "comb", "collapsed", "write"), (*off[0], off[1]), (*on[0], on[1]), strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert _rel(a, b) < fwd_tol, f"{name}: {_rel(a, b):.3e}"
    g_off = jax.jit(jax.grad(lambda x, s: _loss(*_site(graph_off, x, s)), argnums=(0, 1)))(x, state)
    g_on = jax.jit(jax.grad(lambda x, s: _loss(*_site(graph_on, x, s)), argnums=(0, 1)))(x, state)
    names = ["dx", *sorted(state.flatten())]
    flat_off = [g_off[0], *(g_off[1].flatten()[k] for k in names[1:])]
    flat_on = [g_on[0], *(g_on[1].flatten()[k] for k in names[1:])]
    for name, a, b in zip(names, flat_off, flat_on, strict=True):
        assert a.dtype == b.dtype, name
        assert np.isfinite(np.asarray(b, np.float32)).all(), name
        assert _rel(a, b) < grad_tol, f"grad {name}: {_rel(a, b):.3e}"


def test_default_is_unfused_and_residual_write_is_the_free_function():
    graph_off, _, state = _pair(jnp.bfloat16, 256)
    module = spx.bind(graph_off, state)
    assert module.fused_kernel == "none"
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(2, 8, HC, 256)), jnp.bfloat16)
    y = jnp.asarray(rng.normal(size=(2, 8, 256)), jnp.bfloat16)
    post, comb, _ = module(x)
    np.testing.assert_array_equal(
        np.asarray(module.residual_write(x, y, post, comb), np.float32),
        np.asarray(manifold_residual_write(x, y, post, comb, precision=jax.lax.Precision.HIGHEST), np.float32),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_fused_matches_default_on_one_device(dtype):
    graph_off, graph_on, state = _pair(dtype, 256)
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 136, HC, 256)), dtype)  # 272 rows: padded tile
    if dtype == jnp.float32:
        _check(graph_off, graph_on, state, x, fwd_tol=2e-5, grad_tol=2e-4)
    else:
        _check(graph_off, graph_on, state, x, fwd_tol=1.6e-2, grad_tol=4e-2)


@pytest.mark.parametrize("topology", [(2, 1, 2), (1, 2, 2)])
def test_fused_matches_default_under_a_sharded_mesh_including_parameter_grads(topology):
    source = _source(topology)
    graph_off, graph_on, state = _pair(jnp.float32, 256, source)
    x = jnp.asarray(np.random.default_rng(2).normal(size=(4, 128, HC, 256)), jnp.float32)
    spec = source.runtime_sharding_resolver.resolve(dynamic_axes=HyperStreamSharding, shape=x.shape)
    assert spec[3] == "features"
    x = jax.device_put(x, NamedSharding(source.mesh, spec))
    _check(graph_off, graph_on, state, x, fwd_tol=2e-5, grad_tol=2e-4)


def test_unknown_fused_kernel_is_refused():
    with pytest.raises(ValueError, match="fused_kernel"):
        ManifoldHyperConnection(ManifoldHyperConnectionConfig(hidden_size=128), rngs=spx.Rngs(0), fused_kernel="auto")
