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

"""Feature-local mHC math, gradients, layouts and checkpoint leaves."""

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
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P


def _source(topology):
    if jax.device_count() < 4:
        pytest.skip("Requires four devices for nontrivial feature and token partitions")
    # Noncanonical names ensure the implementation uses the resolver, not 'tp'.
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(topology), ("data", "tokens", "features"))
    axis = spx.PartitionAxis(
        batch_axis="data",
        query_sequence_axis="tokens",
        hidden_state_axis="features",
        decode_batch_axis="data",
        decode_hidden_state_axis="features",
    )
    return SimpleNamespace(mesh=mesh, runtime_sharding_resolver=coerce_runtime_sharding_resolver(axis, mesh=mesh))


def _reference(x, fn, base, scale, eps=1e-6, iters=5):
    """Independent dense equations; stream reduction uses an explicit loop."""
    h = x.shape[2]
    xf = x.astype(jnp.float32)
    flat = xf.reshape(*x.shape[:2], -1)
    norm = flat / jnp.sqrt(jnp.mean(flat * flat, axis=-1, keepdims=True) + eps)
    logits = jnp.matmul(norm, fn.astype(jnp.float32).T, precision=jax.lax.Precision.HIGHEST)
    base, scale = base.astype(jnp.float32), scale.astype(jnp.float32)
    pre = jax.nn.sigmoid(logits[..., :h] * scale[0] + base[:h]) + eps
    post = 2 * jax.nn.sigmoid(logits[..., h : 2 * h] * scale[1] + base[h : 2 * h])
    comb = jax.nn.softmax((logits[..., 2 * h :] * scale[2] + base[2 * h :]).reshape(*x.shape[:2], h, h), -1) + eps
    for axis in [-2] + [-1, -2] * (iters - 1):
        comb = comb / (jnp.sum(comb, axis=axis, keepdims=True) + eps)
    collapsed = sum(pre[..., i, None] * xf[..., i, :] for i in range(h)).astype(x.dtype)
    return post, comb, collapsed


def _loss(outputs):
    return sum(jnp.mean(y.astype(jnp.float32) * jnp.linspace(-0.7, 1.3, y.size).reshape(y.shape)) for y in outputs)


@pytest.mark.parametrize("topology", [(1, 1, 4), (2, 1, 2), (1, 2, 2), (2, 2, 1)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("seq", [1, 8])
@pytest.mark.parametrize(
    "hc,fused",
    # Two streams deliberately do not divide the four-way feature mesh; four
    # streams reach the packed TPU coefficients kernel when fused.
    [(2, False), (2, True), (4, False), (4, True)],
)
def test_sharded_connection_outputs_gradients_and_layout(topology, dtype, seq, hc, fused):
    source = _source(topology)
    cfg = ManifoldHyperConnectionConfig(hidden_size=32, hc_mult=hc, iters=5, use_fused_coefficients=fused)
    module = ManifoldHyperConnection(
        cfg, dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(12), mesh_source=source
    )
    graphdef, state = spx.export(module)
    rng = np.random.default_rng(43)
    leaves = state.flatten()
    for key, value in leaves.items():
        leaves[key] = jnp.asarray(rng.normal(0.1, 0.2, value.shape), dtype)
    state = spx.State.from_flat(leaves)
    x = jnp.asarray(rng.normal(size=(4, seq, hc, 32)), dtype)
    spec = source.runtime_sharding_resolver.resolve(dynamic_axes=HyperStreamSharding, shape=x.shape)
    assert spec[-1] == "features"
    assert spec[2] is None
    x = jax.device_put(x, NamedSharding(source.mesh, spec))

    def actual(x, state):
        return spx.bind(graphdef, state)(x)

    def reference(x, state):
        p = state.flatten()
        return _reference(x, p["parameters/fn"], p["parameters/base"], p["parameters/scale"])

    coeff = P(spec[0], spec[1], None)
    outputs = jax.jit(actual)(x, state)
    expected = jax.jit(reference)(x, state)
    for got, want in zip(outputs, expected, strict=True):
        tol = 8e-3 if dtype == jnp.bfloat16 else 3e-6
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tol, atol=tol)
    assert outputs[1].sharding.is_equivalent_to(NamedSharding(source.mesh, P(*coeff, None)), 4)

    got = jax.jit(jax.grad(lambda x, s: _loss(actual(x, s)), argnums=(0, 1)))(x, state)
    # Keep the reference parameter-gradient accumulation in fp32. On this TPU
    # stack, joining three scalar cotangents then converting to bf16 can produce
    # corrupt values; duplicating that lowering would invalidate the reference.
    reference_state = jax.tree.map(lambda value: value.astype(jnp.float32), state)
    want = jax.jit(jax.grad(lambda x, s: _loss(reference(x, s)), argnums=(0, 1)))(x, reference_state)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        assert np.isfinite(np.asarray(a, np.float32)).all()
        assert np.isfinite(np.asarray(b, np.float32)).all()
        np.testing.assert_allclose(
            np.asarray(a, np.float32),
            np.asarray(b, np.float32),
            rtol=3e-2 if dtype == jnp.bfloat16 else 2e-5,
            atol=2e-3 if dtype == jnp.bfloat16 else 3e-6,
        )
    post, comb, collapsed = outputs
    updated = jax.jit(
        lambda x, y, p, c: manifold_residual_write(x, y, p, c, precision=jax.lax.Precision.HIGHEST),
        out_shardings=NamedSharding(source.mesh, spec),
    )(x, jax.nn.silu(collapsed), post, comb)
    independent = post.astype(dtype)[..., None] * jax.nn.silu(collapsed)[..., None, :] + jnp.matmul(
        jnp.swapaxes(comb.astype(dtype), -1, -2), x, precision=jax.lax.Precision.HIGHEST
    )
    np.testing.assert_allclose(
        np.asarray(updated, np.float32), np.asarray(independent, np.float32), rtol=8e-3, atol=8e-3
    )
    assert updated.sharding.is_equivalent_to(NamedSharding(source.mesh, spec), 4)
    mix = (2 + hc) * hc
    assert {k: v.shape for k, v in state.flatten().items()} == {
        "parameters/fn": (mix, hc * 32),
        "parameters/base": (mix,),
        "parameters/scale": (3,),
    }


@pytest.mark.parametrize("topology", [(1, 1, 4), (2, 1, 2), (1, 2, 2)])
@pytest.mark.parametrize("fused", [False, True])
def test_bf16_wide_connection_matches_unpartitioned_reference(topology, fused):
    """Check wide connection outputs against independent single-device math."""
    source = _source(topology)
    module = ManifoldHyperConnection(
        ManifoldHyperConnectionConfig(hidden_size=4096, use_fused_coefficients=fused),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(7),
        mesh_source=source,
    )
    graphdef, state = spx.export(module)
    x = jnp.asarray(np.random.default_rng(31).normal(size=(2, 128, 4, 4096)), jnp.bfloat16)
    spec = source.runtime_sharding_resolver.resolve(dynamic_axes=HyperStreamSharding, shape=x.shape)

    def actual(x, state):
        return spx.bind(graphdef, state)(x)

    def reference(x, state):
        p = state.flatten()
        return _reference(x, p["parameters/fn"], p["parameters/base"], p["parameters/scale"], iters=20)

    single = SingleDeviceSharding(jax.devices()[0])
    want = jax.jit(reference)(jax.device_put(x, single), jax.tree.map(lambda v: jax.device_put(v, single), state))
    output_specs = (P(spec[0], spec[1], None), P(spec[0], spec[1], None, None), P(spec[0], spec[1], spec[3]))
    got = jax.jit(actual, out_shardings=tuple(NamedSharding(source.mesh, s) for s in output_specs))(
        jax.device_put(x, NamedSharding(source.mesh, spec)), state
    )
    for observed, expected in zip(got, want, strict=True):
        tol = 1e-2 if observed.dtype == jnp.bfloat16 else 3e-6
        np.testing.assert_allclose(
            np.asarray(observed, np.float32), np.asarray(expected, np.float32), rtol=tol, atol=tol, equal_nan=False
        )
