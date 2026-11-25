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


import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange
from ejkernel.kernels._triton import recurrent_gla
from ejkernel.kernels._xla.gla import recurrent_gla as xla_recurrent_gla
from ejkernel.utils import numeric_gen

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def run_recurrent_gla_test(
    batch_size=4,
    seq_len=2048,
    num_heads=4,
    key_dim=512,
    value_dim=512,
    atol=1e-3,
    rtol=1e-3,
):
    """
    Run a recurrent GLA test in both batched and flattened (cu_seqlens) mode.
    """
    print(f"\nRunning test: B={batch_size}, T={seq_len}, H={num_heads}, K={key_dim}, V={value_dim}")

    query = numeric_gen(batch_size, seq_len, num_heads, key_dim)
    key = numeric_gen(batch_size, seq_len, num_heads, key_dim)
    value = numeric_gen(batch_size, seq_len, num_heads, value_dim)
    gate = jax.nn.log_sigmoid(numeric_gen(batch_size, seq_len, num_heads, key_dim))
    init_state = numeric_gen(batch_size, num_heads, key_dim, value_dim)
    output_batched, state_batched = recurrent_gla(query, key, value, gate, initial_state=init_state)
    query_flat, key_flat, value_flat, gate_flat = map(
        lambda x: rearrange(x, "b t h d -> 1 (b t) h d"),
        (query, key, value, gate),
    )
    cu_seqlens = jnp.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32)
    output_var, state_var = recurrent_gla(
        query_flat,
        key_flat,
        value_flat,
        gate_flat,
        initial_state=init_state,
        cu_seqlens=cu_seqlens,
    )

    assert jnp.allclose(output_batched.ravel(), output_var.ravel(), rtol=rtol, atol=atol), "Output tensors differ"
    assert jnp.allclose(state_batched.ravel(), state_var.ravel(), rtol=rtol, atol=atol), "Final states differ"

    print("✅ Passed")


@pytest.mark.parametrize(
    "scenario",
    [
        dict(batch_size=4, seq_len=2048, num_heads=4, key_dim=512, value_dim=512),
        dict(batch_size=2, seq_len=1024, num_heads=8, key_dim=256, value_dim=256),
        dict(batch_size=1, seq_len=512, num_heads=2, key_dim=128, value_dim=64),
        dict(batch_size=8, seq_len=128, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=2, seq_len=4096, num_heads=1, key_dim=32, value_dim=32),
    ],
)
def test_recurrent_gla_variants(scenario):
    run_recurrent_gla_test(**scenario)


def test_recurrent_gla_matches_xla_smoke():
    key = jax.random.PRNGKey(0)
    kq, kk, kv, ks = jax.random.split(key, 4)

    batch_size, seq_len, num_heads, key_dim, value_dim = 2, 128, 4, 64, 64
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    key = jax.random.normal(kk, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    value = jax.random.normal(kv, (batch_size, seq_len, num_heads, value_dim), dtype=jnp.float32)
    init_state = jax.random.normal(ks, (batch_size, num_heads, key_dim, value_dim), dtype=jnp.float32)

    out_tri, state_tri = recurrent_gla(query, key, value, initial_state=init_state)
    out_xla, state_xla = xla_recurrent_gla(query, key, value, initial_state=init_state)

    out_tri = jax.block_until_ready(out_tri)
    out_xla = jax.block_until_ready(out_xla)
    state_tri = jax.block_until_ready(state_tri)
    state_xla = jax.block_until_ready(state_xla)

    np.testing.assert_allclose(out_tri, out_xla, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state_tri, state_xla, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reverse", [False, True])
def test_recurrent_gla_matches_xla_with_4d_gate_and_gamma(reverse: bool):
    rng = jax.random.PRNGKey(10 + int(reverse))
    kq, kk, kv, kg, kgg, ks = jax.random.split(rng, 6)

    batch_size, seq_len, num_heads, key_dim, value_dim = 2, 32, 4, 64, 64
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(kk, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(kv, (batch_size, seq_len, num_heads, value_dim), dtype=jnp.float32) * 0.5
    gate = jax.nn.log_sigmoid(jax.random.normal(kg, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32))
    g_gamma = -jnp.abs(jax.random.normal(kgg, (num_heads,), dtype=jnp.float32)) * 0.1
    init_state = jax.random.normal(ks, (batch_size, num_heads, key_dim, value_dim), dtype=jnp.float32) * 0.2

    out_tri, state_tri = recurrent_gla(
        query,
        key,
        value,
        g=gate,
        g_gamma=g_gamma,
        initial_state=init_state,
        reverse=reverse,
    )
    out_xla, state_xla = xla_recurrent_gla(
        query,
        key,
        value,
        g=gate,
        g_gamma=g_gamma,
        initial_state=init_state,
        reverse=reverse,
    )

    out_tri, state_tri = jax.block_until_ready((out_tri, state_tri))
    out_xla, state_xla = jax.block_until_ready((out_xla, state_xla))

    np.testing.assert_allclose(out_tri, out_xla, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(state_tri, state_xla, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("with_initial_state", [True, False])
@pytest.mark.parametrize("reverse", [False, True])
def test_recurrent_gla_grad_matches_xla(with_initial_state: bool, reverse: bool):
    rng = jax.random.PRNGKey(20 + 2 * int(with_initial_state) + int(reverse))
    kq, kk, kv, kg, ks = jax.random.split(rng, 5)

    batch_size, seq_len, num_heads, key_dim, value_dim = 2, 16, 2, 32, 32
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(kk, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(kv, (batch_size, seq_len, num_heads, value_dim), dtype=jnp.float32) * 0.5
    gate = jax.nn.log_sigmoid(jax.random.normal(kg, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32))
    init_state = (
        jax.random.normal(ks, (batch_size, num_heads, key_dim, value_dim), dtype=jnp.float32) * 0.2
        if with_initial_state
        else None
    )

    def _loss(fn, q_, k_, v_, g_, h0_):
        out, st = fn(q_, k_, v_, g=g_, initial_state=h0_, reverse=reverse)
        return jnp.mean(out) + 0.25 * jnp.mean(st)

    argnums = (0, 1, 2, 3, 4) if with_initial_state else (0, 1, 2, 3)
    grads_tri = jax.grad(lambda q_, k_, v_, g_, h0_: _loss(recurrent_gla, q_, k_, v_, g_, h0_), argnums=argnums)(
        query, key, value, gate, init_state
    )
    grads_xla = jax.grad(lambda q_, k_, v_, g_, h0_: _loss(xla_recurrent_gla, q_, k_, v_, g_, h0_), argnums=argnums)(
        query, key, value, gate, init_state
    )

    grads_tri = jax.tree_util.tree_map(jax.block_until_ready, grads_tri)
    grads_xla = jax.tree_util.tree_map(jax.block_until_ready, grads_xla)
    for g_tri, g_xla in zip(grads_tri, grads_xla, strict=True):
        np.testing.assert_allclose(g_tri, g_xla, rtol=2e-3, atol=2e-3)


def test_recurrent_gla_jit_forward_and_grad():
    rng = jax.random.PRNGKey(42)
    kq, kk, kv, kg, kgg = jax.random.split(rng, 5)

    batch_size, seq_len, num_heads, key_dim, value_dim = 2, 16, 2, 32, 32
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(kk, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(kv, (batch_size, seq_len, num_heads, value_dim), dtype=jnp.float32) * 0.5
    gate = jax.nn.log_sigmoid(jax.random.normal(kg, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32))
    g_gamma = -jnp.abs(jax.random.normal(kgg, (num_heads,), dtype=jnp.float32)) * 0.1

    @jax.jit
    def _fwd_jit(q_, k_, v_, g_, gg_):
        return recurrent_gla(q_, k_, v_, g=g_, g_gamma=gg_)

    out_tri, state_tri = jax.block_until_ready(_fwd_jit(query, key, value, gate, g_gamma))
    out_xla, state_xla = jax.block_until_ready(xla_recurrent_gla(query, key, value, g=gate, g_gamma=g_gamma))
    np.testing.assert_allclose(out_tri, out_xla, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(state_tri, state_xla, rtol=1e-3, atol=1e-3)

    def _loss_tri(q_, k_, v_, g_, gg_):
        out, st = recurrent_gla(q_, k_, v_, g=g_, g_gamma=gg_)
        return jnp.mean(out) + 0.2 * jnp.mean(st)

    def _loss_xla(q_, k_, v_, g_, gg_):
        out, st = xla_recurrent_gla(q_, k_, v_, g=g_, g_gamma=gg_)
        return jnp.mean(out) + 0.2 * jnp.mean(st)

    grads_tri = jax.jit(jax.grad(_loss_tri, argnums=(0, 1, 2, 3)))(query, key, value, gate, g_gamma)
    grads_xla = jax.grad(_loss_xla, argnums=(0, 1, 2, 3))(query, key, value, gate, g_gamma)
    grads_tri = jax.tree_util.tree_map(jax.block_until_ready, grads_tri)
    grads_xla = jax.tree_util.tree_map(jax.block_until_ready, grads_xla)
    for g_tri, g_xla in zip(grads_tri, grads_xla, strict=True):
        np.testing.assert_allclose(g_tri, g_xla, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    pytest.main([__file__])
