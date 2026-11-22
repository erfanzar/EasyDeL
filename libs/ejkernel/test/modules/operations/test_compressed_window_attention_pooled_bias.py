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

"""Pooled-bias tests for the compressed_window_attention public operation.

``expand_pooled_bias`` is checked against an independent NumPy expansion (values) and the pool-sum contract
(gradients); the pooled call is checked against the same call with the per-key bias (value and every cotangent),
on the XLA path and, when present, the Pallas TPU path and a head-sharded ``shard_map``.
"""

import jax
import numpy as np
import pytest
from ejkernel.modules import compressed_window_attention
from ejkernel.modules.operations._pooled_bias import expand_pooled_bias
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps

_NEG = float(np.finfo(np.float32).min)


def _pooled_inputs(seed=0, s=256, kpool=4, tails=3, heads=8, dim=128):
    rng = np.random.default_rng(seed)
    n_pools = -(-s // kpool)
    pool = np.where(rng.random((1, s, n_pools)) < 0.4, rng.standard_normal((1, s, n_pools)) * 0.1, _NEG)
    pool = pool.astype(np.float32)
    tail = rng.integers(-1, s, size=(1, s, tails)).astype(np.int32)
    tail[:, :, 0] = np.arange(s)[None]  # the diagonal is always a tail key
    q = jnp.asarray(rng.standard_normal((1, heads, s, dim)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((1, s, dim)), jnp.float32)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    return q, kv, pool, tail, sinks


def _numpy_expand(pool, tail, kpool, kv_len):
    bias = np.repeat(pool, kpool, axis=-1)[..., :kv_len].copy()
    for b in range(tail.shape[0]):
        for s in range(tail.shape[1]):
            for k in tail[b, s]:
                if k >= 0 and bias[b, s, k] == _NEG:
                    bias[b, s, k] = 0.0
    return bias


def test_expand_pooled_bias_value_and_gradient():
    """Values equal the NumPy expansion; the pool gradient is the pool-sum over keys the pool kept open."""
    _, _, pool, tail, _ = _pooled_inputs(seed=1, s=130, kpool=4)
    got = expand_pooled_bias(jnp.asarray(pool), 4, 130, jnp.asarray(tail))
    np.testing.assert_array_equal(np.asarray(got), _numpy_expand(pool, tail, 4, 130))
    g = jnp.asarray(np.random.default_rng(2).standard_normal((1, 130, 130)), jnp.float32)
    dpool = jax.grad(lambda p: jnp.sum(expand_pooled_bias(p, 4, 130, jnp.asarray(tail)) * g))(jnp.asarray(pool))
    opened_by_tail = _numpy_expand(pool, tail, 4, 130) != np.repeat(pool, 4, axis=-1)[..., :130]
    keep = np.where(opened_by_tail, 0.0, np.asarray(g))
    keep = np.pad(keep, ((0, 0), (0, 0), (0, pool.shape[-1] * 4 - 130)))
    expected = keep.reshape(1, 130, pool.shape[-1], 4).sum(-1)
    np.testing.assert_allclose(np.asarray(dpool), expected, rtol=1e-5, atol=1e-5)


def test_expand_pooled_bias_rejects_bad_kpool():
    """Non-positive kpool or pools that do not cover the keys are rejected."""
    with pytest.raises(ValueError, match="kpool"):
        expand_pooled_bias(jnp.zeros((1, 4, 2)), 0, 8)
    with pytest.raises(ValueError, match="cover"):
        expand_pooled_bias(jnp.zeros((1, 4, 2)), 2, 8)


@pytest.mark.parametrize("platform", ["xla", "pallas"])
def test_pooled_call_matches_per_key_call(platform):
    """``bias_pool`` + ``bias_tails`` equal the per-key bias call in value and in q / kv / sink / pool cotangents."""
    if platform == "pallas" and jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU path requires a TPU backend")
    q, kv, pool, tail, sinks = _pooled_inputs(seed=3)
    pool, tail = jnp.asarray(pool), jnp.asarray(tail)
    cot = jnp.asarray(np.random.default_rng(4).standard_normal(q.shape), jnp.float32)

    def pooled(a, b, c, p):
        out = compressed_window_attention(
            a, b, None, c, platform=platform, causal=True, bias_pool=p, bias_tails=tail, bias_kpool=4
        )
        return jnp.sum(out * cot)

    def per_key(a, b, c, p):
        out = compressed_window_attention(
            a, b, expand_pooled_bias(p, 4, kv.shape[1], tail), c, platform=platform, causal=True
        )
        return jnp.sum(out * cot)

    vp, gp = jax.value_and_grad(pooled, argnums=(0, 1, 2, 3))(q, kv, sinks, pool)
    vk, gk = jax.value_and_grad(per_key, argnums=(0, 1, 2, 3))(q, kv, sinks, pool)
    np.testing.assert_allclose(float(vp), float(vk), rtol=1e-6)
    for a, b in zip(gp, gk, strict=True):
        assert a.shape == b.shape
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_pooled_call_head_sharded_shard_map():
    """Head-sharded shard_map with a pooled bias: same output and cotangents, pooled ``[B, S, P]`` bias cotangent."""
    if jax.default_backend() != "tpu" or len(jax.devices()) < 2:
        pytest.skip("needs a multi-device TPU mesh")
    n = 4 if len(jax.devices()) >= 4 else 2
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:n]), ("tp",))
    q, kv, pool, tail, sinks = _pooled_inputs(seed=5)
    pool, tail = jnp.asarray(pool), jnp.asarray(tail)
    cot = jnp.asarray(np.random.default_rng(6).standard_normal(q.shape), jnp.float32)
    specs = (Ps(None, "tp", None, None), Ps(), Ps(), Ps("tp"), Ps(), Ps())

    def sharded(a, b, c, p):
        out = compressed_window_attention(
            a, b, None, c, platform="pallas", bias_pool=p, bias_tails=tail, bias_kpool=4,
            mesh=mesh, in_specs=specs, out_specs=Ps(None, "tp", None, None),
        )  # fmt: skip
        return jnp.sum(out * cot)

    def single(a, b, c, p):
        return jnp.sum(compressed_window_attention(a, b, expand_pooled_bias(p, 4, kv.shape[1], tail), c) * cot)

    gs = jax.jit(jax.grad(sharded, argnums=(0, 1, 2, 3)))(q, kv, sinks, pool)
    g1 = jax.jit(jax.grad(single, argnums=(0, 1, 2, 3)))(q, kv, sinks, pool)
    assert gs[3].shape == pool.shape
    for a, b in zip(gs, g1, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-4, atol=2e-4)


def test_pooled_argument_validation():
    """Both biases, a missing kpool, or no bias at all are rejected."""
    q, kv, pool, _, sinks = _pooled_inputs(seed=7, s=16, heads=2, dim=8)
    with pytest.raises(ValueError, match="either"):
        compressed_window_attention(q, kv, jnp.zeros((1, 16, 16)), sinks, bias_pool=jnp.asarray(pool), bias_kpool=4)
    with pytest.raises(ValueError, match="bias_kpool"):
        compressed_window_attention(q, kv, None, sinks, bias_pool=jnp.asarray(pool))
    with pytest.raises(ValueError, match="bias_pool"):
        compressed_window_attention(q, kv, None, sinks)
