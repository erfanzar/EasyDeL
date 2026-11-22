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

"""Correctness and gradient tests for ragged_gather / ragged_gather_reduce."""

import jax
import jax.numpy as jnp
import pytest
from ejkernel.kernels._xla.ragged_gather import ragged_gather
from ejkernel.kernels._xla.ragged_gather_reduce import ragged_gather_reduce


def test_ragged_gather_matches_take():
    x = jax.random.normal(jax.random.PRNGKey(0), (32, 64), jnp.float32)
    idx = jax.random.randint(jax.random.PRNGKey(1), (48,), 0, 32).astype(jnp.int32)
    out = ragged_gather(x, idx)
    assert out.shape == (48, 64)
    assert jnp.array_equal(out, x[idx])


def test_ragged_gather_invalid_indices_are_zero_rows():
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 16), jnp.float32)
    idx = jnp.array([0, -1, 3, -5], jnp.int32)
    out = ragged_gather(x, idx)
    assert jnp.array_equal(out[0], x[0])
    assert jnp.array_equal(out[2], x[3])
    assert jnp.all(out[1] == 0)
    assert jnp.all(out[3] == 0)


def test_ragged_gather_grad_is_scatter_add():
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 16), jnp.float32)
    idx = jnp.array([2, 2, 5, -1], jnp.int32)

    g = jax.grad(lambda v: jnp.sum(ragged_gather(v, idx) ** 2))(x)
    g_ref = jax.grad(lambda v: jnp.sum(jnp.where((idx >= 0)[:, None], v[jnp.maximum(idx, 0)], 0.0) ** 2))(x)
    assert jnp.allclose(g, g_ref, rtol=1e-6, atol=1e-6)
    # duplicated index 2 must accumulate both cotangents
    assert jnp.allclose(g[2], 4.0 * x[2], rtol=1e-5, atol=1e-5)
    # invalid slot contributes nothing anywhere
    assert jnp.allclose(g[5], 2.0 * x[5], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_ragged_gather_reduce_matches_reference(dtype):
    m, k, n, d = 16, 4, 24, 32
    x = jax.random.normal(jax.random.PRNGKey(0), (n, d), dtype)
    idx = jax.random.randint(jax.random.PRNGKey(1), (m * k,), 0, n).astype(jnp.int32)
    w = jax.random.uniform(jax.random.PRNGKey(2), (m * k,), jnp.float32).astype(dtype)

    out = ragged_gather_reduce(x, idx, w, reduce_group_size=k)
    ref = (x[idx].astype(jnp.float32) * w.astype(jnp.float32)[:, None]).reshape(m, k, d).sum(1).astype(dtype)
    assert out.shape == (m, d)
    assert jnp.allclose(out.astype(jnp.float32), ref.astype(jnp.float32), rtol=1e-5, atol=1e-5)


def test_ragged_gather_reduce_invalid_slots_contribute_zero():
    n, d = 8, 16
    x = jax.random.normal(jax.random.PRNGKey(0), (n, d), jnp.float32)
    idx = jnp.array([1, -1, 4, -1], jnp.int32)
    w = jnp.array([0.5, 100.0, 2.0, 100.0], jnp.float32)
    out = ragged_gather_reduce(x, idx, w, reduce_group_size=2)
    assert jnp.allclose(out[0], 0.5 * x[1], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(out[1], 2.0 * x[4], rtol=1e-6, atol=1e-6)


def test_ragged_gather_reduce_grads_flow_to_x_and_weights():
    m, k, n, d = 4, 2, 8, 16
    x = jax.random.normal(jax.random.PRNGKey(0), (n, d), jnp.float32)
    idx = jax.random.randint(jax.random.PRNGKey(1), (m * k,), 0, n).astype(jnp.int32)
    w = jax.random.uniform(jax.random.PRNGKey(2), (m * k,), jnp.float32)

    def ref(x, w):
        rows = x[idx] * w[:, None]
        return jnp.sum(rows.reshape(m, k, d).sum(1) ** 2)

    def op(x, w):
        return jnp.sum(ragged_gather_reduce(x, idx, w, reduce_group_size=k) ** 2)

    gx, gw = jax.grad(op, argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(ref, argnums=(0, 1))(x, w)
    assert jnp.allclose(gx, gx_ref, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(gw, gw_ref, rtol=1e-5, atol=1e-5)


def test_ragged_gather_reduce_validation():
    x = jnp.ones((8, 16), jnp.float32)
    idx = jnp.zeros((6,), jnp.int32)
    w = jnp.ones((6,), jnp.float32)
    with pytest.raises(ValueError, match="reduce_group_size"):
        ragged_gather_reduce(x, idx, w, reduce_group_size=4)
    # mismatched indices/weights lengths are rejected by the shared "mk" shape annotation
    with pytest.raises(Exception, match="mk"):
        ragged_gather_reduce(x, idx, jnp.ones((5,), jnp.float32), reduce_group_size=3)
