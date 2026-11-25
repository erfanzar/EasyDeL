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
from ejkernel.kernels._triton import recurrent
from ejkernel.kernels._xla.recurrent import recurrent as xla_recurrent
from ejkernel.utils import numeric_gen

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def run_recurrent_test(
    batch_size=4,
    seq_len=2048,
    num_heads=4,
    key_dim=512,
    value_dim=512,
    atol=1e-3,
    rtol=1e-3,
):
    """
    Run a recurrent test in both batched and flattened (cu_seqlens) mode.
    """
    print(f"\nRunning test: B={batch_size}, T={seq_len}, H={num_heads}, K={key_dim}, V={value_dim}")

    query = numeric_gen(batch_size, seq_len, num_heads, key_dim)
    key = numeric_gen(batch_size, seq_len, num_heads, key_dim)
    value = numeric_gen(batch_size, seq_len, num_heads, value_dim)
    init_state = numeric_gen(batch_size, num_heads, key_dim, value_dim)
    output_batched, state_batched = recurrent(query, key, value, initial_state=init_state)
    query_flat, key_flat, value_flat = map(lambda x: rearrange(x, "b t h d -> 1 (b t) h d"), (query, key, value))
    cu_seqlens = jnp.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32)
    output_var, state_var = recurrent(query_flat, key_flat, value_flat, initial_state=init_state, cu_seqlens=cu_seqlens)
    assert jnp.allclose(output_batched.ravel(), output_var.ravel(), rtol=rtol, atol=atol), "Output tensors differ"
    assert jnp.allclose(state_batched.ravel(), state_var.ravel(), rtol=rtol, atol=atol), "Final states differ"

    print("✅ Passed")


@pytest.mark.parametrize(
    "scenario",
    [
        dict(batch_size=8, seq_len=128, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=1024, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=2048, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=4096, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=8192, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=4, seq_len=2048, num_heads=4, key_dim=512, value_dim=512),
        dict(batch_size=2, seq_len=1024, num_heads=8, key_dim=256, value_dim=256),
        dict(batch_size=1, seq_len=512, num_heads=2, key_dim=128, value_dim=64),
        dict(batch_size=8, seq_len=128, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=2, seq_len=4096, num_heads=1, key_dim=32, value_dim=32),
    ],
)
def test_recurrent_variants(scenario):
    run_recurrent_test(**scenario)


def test_recurrent_matches_xla_smoke():
    key = jax.random.PRNGKey(0)
    kq, kk, kv, ks = jax.random.split(key, 4)

    batch_size, seq_len, num_heads, key_dim, value_dim = 2, 64, 4, 64, 64
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    key = jax.random.normal(kk, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    value = jax.random.normal(kv, (batch_size, seq_len, num_heads, value_dim), dtype=jnp.float32)
    init_state = jax.random.normal(ks, (batch_size, num_heads, key_dim, value_dim), dtype=jnp.float32)

    out_tri, state_tri = recurrent(query, key, value, initial_state=init_state)
    out_xla, state_xla = xla_recurrent(query, key, value, initial_state=init_state)

    out_tri = jax.block_until_ready(out_tri)
    out_xla = jax.block_until_ready(out_xla)
    state_tri = jax.block_until_ready(state_tri)
    state_xla = jax.block_until_ready(state_xla)

    np.testing.assert_allclose(out_tri, out_xla, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state_tri, state_xla, rtol=1e-4, atol=1e-4)


def _reference_recurrent_scan(q, k, v, *, softmax_scale=None, initial_state=None):
    batch, _seq_len, num_heads, key_dim = q.shape
    num_kv_heads = k.shape[2]
    value_dim = v.shape[-1]
    group_size = num_heads // num_kv_heads
    if group_size > 1:
        k = jnp.repeat(k, group_size, axis=2)
        v = jnp.repeat(v, group_size, axis=2)
    scale = (key_dim**-0.5) if softmax_scale is None else softmax_scale
    h0 = (
        jnp.zeros((batch, num_heads, key_dim, value_dim), dtype=jnp.float32)
        if initial_state is None
        else initial_state.astype(jnp.float32)
    )

    def _step(h, xs):
        qt, kt, vt = xs
        h = h + kt[:, :, :, None] * vt[:, :, None, :]
        ot = jnp.sum(h * (qt * scale)[:, :, :, None], axis=2)
        return h, ot

    h_final, out_t = jax.lax.scan(
        _step,
        h0,
        (
            jnp.swapaxes(q.astype(jnp.float32), 0, 1),
            jnp.swapaxes(k.astype(jnp.float32), 0, 1),
            jnp.swapaxes(v.astype(jnp.float32), 0, 1),
        ),
    )
    return jnp.swapaxes(out_t, 0, 1).astype(q.dtype), h_final


def test_recurrent_grad_without_gates_or_initial_state():
    rng = jax.random.PRNGKey(101)
    kq, kk, kv = jax.random.split(rng, 3)

    batch_size, seq_len, num_heads, head_dim = 2, 16, 2, 32
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.5

    def _loss_tri(q_, k_, v_):
        out, st = recurrent(q_, k_, v_, initial_state=None)
        return jnp.mean(out) + 0.1 * jnp.mean(st)

    def _loss_xla(q_, k_, v_):
        out, st = xla_recurrent(q_, k_, v_, initial_state=None)
        return jnp.mean(out) + 0.1 * jnp.mean(st)

    grads_tri = jax.jit(jax.grad(_loss_tri, argnums=(0, 1, 2)))(query, key, value)
    grads_xla = jax.grad(_loss_xla, argnums=(0, 1, 2))(query, key, value)
    grads_tri = jax.tree_util.tree_map(jax.block_until_ready, grads_tri)
    grads_xla = jax.tree_util.tree_map(jax.block_until_ready, grads_xla)
    for g_tri, g_xla in zip(grads_tri, grads_xla, strict=True):
        np.testing.assert_allclose(g_tri, g_xla, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("g_gamma_rank", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_recurrent_all_gates_fwd_and_bwd_match_xla(g_gamma_rank: int, reverse: bool):
    rng = jax.random.PRNGKey(200 + 2 * g_gamma_rank + int(reverse))
    kq, kk, kv, kg, kgg, kgk, kgv, ks = jax.random.split(rng, 8)

    batch_size, seq_len, num_heads, head_dim = 2, 16, 2, 32
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.4
    key = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.4
    value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.4
    g = jax.nn.log_sigmoid(jax.random.normal(kg, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32))
    g_gamma_shape = (num_heads,) if g_gamma_rank == 1 else (batch_size, num_heads)
    g_gamma = -jnp.abs(jax.random.normal(kgg, g_gamma_shape, dtype=jnp.float32)) * 0.1
    gk = jax.nn.log_sigmoid(jax.random.normal(kgk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32))
    gv = jax.nn.log_sigmoid(jax.random.normal(kgv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32))
    init_state = jax.random.normal(ks, (batch_size, num_heads, head_dim, head_dim), dtype=jnp.float32) * 0.2

    out_tri, st_tri = jax.block_until_ready(
        recurrent(query, key, value, g=g, g_gamma=g_gamma, gk=gk, gv=gv, initial_state=init_state, reverse=reverse)
    )
    out_xla, st_xla = jax.block_until_ready(
        xla_recurrent(query, key, value, g=g, g_gamma=g_gamma, gk=gk, gv=gv, initial_state=init_state, reverse=reverse)
    )
    np.testing.assert_allclose(out_tri, out_xla, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(st_tri, st_xla, rtol=2e-3, atol=2e-3)

    def _loss(fn, q_, k_, v_, g_, gk_, gv_, h0_):
        out, st = fn(q_, k_, v_, g=g_, g_gamma=g_gamma, gk=gk_, gv=gv_, initial_state=h0_, reverse=reverse)
        return jnp.mean(out) + 0.25 * jnp.mean(st)

    grads_tri = jax.grad(
        lambda q_, k_, v_, g_, gk_, gv_, h0_: _loss(recurrent, q_, k_, v_, g_, gk_, gv_, h0_),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(query, key, value, g, gk, gv, init_state)
    grads_xla = jax.grad(
        lambda q_, k_, v_, g_, gk_, gv_, h0_: _loss(xla_recurrent, q_, k_, v_, g_, gk_, gv_, h0_),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(query, key, value, g, gk, gv, init_state)
    grads_tri = jax.tree_util.tree_map(jax.block_until_ready, grads_tri)
    grads_xla = jax.tree_util.tree_map(jax.block_until_ready, grads_xla)
    for g_tri, g_xla in zip(grads_tri, grads_xla, strict=True):
        np.testing.assert_allclose(g_tri, g_xla, rtol=3e-3, atol=3e-3)


def test_recurrent_gqa_and_non_pow2_dims_match_reference():
    rng = jax.random.PRNGKey(303)
    kq, kk, kv, ks = jax.random.split(rng, 4)

    batch_size, seq_len, num_heads, num_kv_heads, key_dim, value_dim = 2, 12, 4, 2, 48, 40
    query = jax.random.normal(kq, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * 0.4
    key = jax.random.normal(kk, (batch_size, seq_len, num_kv_heads, key_dim), dtype=jnp.float32) * 0.4
    value = jax.random.normal(kv, (batch_size, seq_len, num_kv_heads, value_dim), dtype=jnp.float32) * 0.4
    init_state = jax.random.normal(ks, (batch_size, num_heads, key_dim, value_dim), dtype=jnp.float32) * 0.2

    out_tri, st_tri = jax.block_until_ready(
        recurrent(query, key, value, initial_state=init_state, block_k=64, block_v=64)
    )
    out_ref, st_ref = jax.block_until_ready(_reference_recurrent_scan(query, key, value, initial_state=init_state))
    np.testing.assert_allclose(out_tri, out_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(st_tri, st_ref, rtol=2e-3, atol=2e-3)

    def _loss_tri(q_, k_, v_, h0_):
        out, st = recurrent(q_, k_, v_, initial_state=h0_, block_k=64, block_v=64)
        return jnp.mean(out) + 0.2 * jnp.mean(st)

    def _loss_ref(q_, k_, v_, h0_):
        out, st = _reference_recurrent_scan(q_, k_, v_, initial_state=h0_)
        return jnp.mean(out) + 0.2 * jnp.mean(st)

    grads_tri = jax.jit(jax.grad(_loss_tri, argnums=(0, 1, 2, 3)))(query, key, value, init_state)
    grads_ref = jax.grad(_loss_ref, argnums=(0, 1, 2, 3))(query, key, value, init_state)
    grads_tri = jax.tree_util.tree_map(jax.block_until_ready, grads_tri)
    grads_ref = jax.tree_util.tree_map(jax.block_until_ready, grads_ref)
    for g_tri, g_ref in zip(grads_tri, grads_ref, strict=True):
        np.testing.assert_allclose(g_tri, g_ref, rtol=3e-3, atol=3e-3)


def test_recurrent_jit_grad_with_traced_g_gamma_and_cu_seqlens():
    rng = jax.random.PRNGKey(404)
    kq, kk, kv, kg, kgg, ks = jax.random.split(rng, 6)

    total_tokens, num_heads, head_dim = 18, 2, 32
    cu_seqlens = jnp.array([0, 5, 11, 18], dtype=jnp.int32)
    num_seqs = cu_seqlens.shape[0] - 1

    query = jax.random.normal(kq, (1, total_tokens, num_heads, head_dim), dtype=jnp.float32) * 0.4
    key = jax.random.normal(kk, (1, total_tokens, num_heads, head_dim), dtype=jnp.float32) * 0.4
    value = jax.random.normal(kv, (1, total_tokens, num_heads, head_dim), dtype=jnp.float32) * 0.4
    g = jax.nn.log_sigmoid(jax.random.normal(kg, (1, total_tokens, num_heads, head_dim), dtype=jnp.float32))
    g_gamma = -jnp.abs(jax.random.normal(kgg, (num_seqs, num_heads), dtype=jnp.float32)) * 0.1
    init_state = jax.random.normal(ks, (num_seqs, num_heads, head_dim, head_dim), dtype=jnp.float32) * 0.2

    def _loss_tri(q_, k_, v_, g_, h0_, gg_, cu_):
        out, st = recurrent(q_, k_, v_, g=g_, g_gamma=gg_, initial_state=h0_, cu_seqlens=cu_)
        return jnp.mean(out) + 0.25 * jnp.mean(st)

    def _loss_xla(q_, k_, v_, g_, h0_, gg_, cu_):
        out, st = xla_recurrent(q_, k_, v_, g=g_, g_gamma=gg_, initial_state=h0_, cu_seqlens=cu_)
        return jnp.mean(out) + 0.25 * jnp.mean(st)

    grads_tri = jax.jit(jax.grad(_loss_tri, argnums=(0, 1, 2, 3, 4)))(
        query, key, value, g, init_state, g_gamma, cu_seqlens
    )
    grads_xla = jax.grad(_loss_xla, argnums=(0, 1, 2, 3, 4))(query, key, value, g, init_state, g_gamma, cu_seqlens)
    grads_tri = jax.tree_util.tree_map(jax.block_until_ready, grads_tri)
    grads_xla = jax.tree_util.tree_map(jax.block_until_ready, grads_xla)
    for g_tri, g_xla in zip(grads_tri, grads_xla, strict=True):
        np.testing.assert_allclose(g_tri, g_xla, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    pytest.main([__file__])
