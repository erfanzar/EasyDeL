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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from ejkernel.kernels._triton import rwkv4 as triton_rwkv4
from ejkernel.kernels._triton import rwkv6 as triton_rwkv6
from ejkernel.kernels._triton import rwkv7 as triton_rwkv7
from ejkernel.kernels._triton import rwkv7_mul as triton_rwkv7_mul
from ejkernel.kernels._xla.rwkv4 import rwkv4 as xla_rwkv4
from ejkernel.kernels._xla.rwkv6 import rwkv6 as xla_rwkv6
from ejkernel.kernels._xla.rwkv7 import rwkv7 as xla_rwkv7
from ejkernel.kernels._xla.rwkv7 import rwkv7_mul as xla_rwkv7_mul

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_rwkv4_matches_xla_and_grad():
    B, T, C = 2, 16, 64
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (C,), dtype=jnp.float16)
    u = jax.random.normal(jax.random.PRNGKey(1), (C,), dtype=jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(2), (B, T, C), dtype=jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(3), (B, T, C), dtype=jnp.float16)
    state0 = jax.random.normal(jax.random.PRNGKey(4), (B, 3, C), dtype=jnp.float32)

    out_tri, st_tri = triton_rwkv4(w, u, k, v, state=None)
    out_xla, st_xla = xla_rwkv4(w, u, k, v, state=None)

    out_tri, st_tri = jax.block_until_ready(out_tri), jax.block_until_ready(st_tri)
    out_xla, st_xla = jax.block_until_ready(out_xla), jax.block_until_ready(st_xla)

    np.testing.assert_allclose(np.asarray(out_tri, np.float32), np.asarray(out_xla, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(st_tri, np.float32), np.asarray(st_xla, np.float32), rtol=2e-2, atol=2e-2)

    def loss_tri(w_, u_, k_, v_, state_):
        o, st = triton_rwkv4(w_, u_, k_, v_, state=state_)
        return jnp.sum(o) + jnp.sum(st)

    def loss_xla(w_, u_, k_, v_, state_):
        o, st = xla_rwkv4(w_, u_, k_, v_, state=state_)
        return jnp.sum(o) + jnp.sum(st)

    grads_tri = jax.grad(loss_tri, argnums=(0, 1, 2, 3, 4))(w, u, k, v, state0)
    grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2, 3, 4))(w, u, k, v, state0)
    for g_tri, g_xla in zip(grads_tri, grads_xla, strict=True):
        np.testing.assert_allclose(
            np.asarray(g_tri, np.float32),
            np.asarray(g_xla, np.float32),
            rtol=2e-2,
            atol=2e-2,
        )


def test_rwkv6_matches_xla_and_varlen():
    B, T, H, K, V = 2, 16, 2, 32, 32
    key = jax.random.PRNGKey(0)
    r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, V), dtype=jnp.float16)
    w = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, K), dtype=jnp.float16) * -0.01
    u = jax.random.normal(jax.random.PRNGKey(4), (H, K), dtype=jnp.float16)
    h0 = jax.random.normal(jax.random.PRNGKey(5), (B, H, K, V), dtype=jnp.float32)

    out_tri, st_tri = triton_rwkv6(r, k, v, w, u, initial_state=h0)
    out_xla, st_xla = xla_rwkv6(r, k, v, w, u, initial_state=h0)

    out_tri, st_tri = jax.block_until_ready(out_tri), jax.block_until_ready(st_tri)
    out_xla, st_xla = jax.block_until_ready(out_xla), jax.block_until_ready(st_xla)

    np.testing.assert_allclose(np.asarray(out_tri, np.float32), np.asarray(out_xla, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(st_tri, np.float32), np.asarray(st_xla, np.float32), rtol=2e-2, atol=2e-2)

    r_p, k_p, v_p, w_p = map(lambda x: rearrange(x, "b t h d -> 1 (b t) h d"), (r, k, v, w))
    cu = jnp.arange(0, (B + 1) * T, T, dtype=jnp.int32)
    out_var, st_var = triton_rwkv6(r_p, k_p, v_p, w_p, u, initial_state=h0, cu_seqlens=cu)
    out_var, st_var = jax.block_until_ready(out_var), jax.block_until_ready(st_var)
    np.testing.assert_allclose(
        np.asarray(out_var.reshape(B, T, H, V), np.float32),
        np.asarray(out_tri, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(np.asarray(st_var, np.float32), np.asarray(st_tri, np.float32), rtol=2e-2, atol=2e-2)


def test_rwkv7_matches_xla_and_mul_wrapper():
    B, T, H, K, V = 2, 16, 2, 32, 32
    key = jax.random.PRNGKey(0)
    r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float16)
    w = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float16) * -0.01
    k = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, K), dtype=jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, V), dtype=jnp.float16)
    a = jax.random.normal(jax.random.PRNGKey(4), (B, T, H, K), dtype=jnp.float16) * 0.01
    b = jax.random.normal(jax.random.PRNGKey(5), (B, T, H, K), dtype=jnp.float16) * 0.01
    kk = jax.random.normal(jax.random.PRNGKey(6), (B, T, H, K), dtype=jnp.float16) * 0.01
    h0 = jax.random.normal(jax.random.PRNGKey(7), (B, H, K, V), dtype=jnp.float32)

    out_tri, st_tri = triton_rwkv7(r, w, k, v, a, b, initial_state=h0)
    out_xla, st_xla = xla_rwkv7(r, w, k, v, a, b, initial_state=h0)

    out_tri, st_tri = jax.block_until_ready(out_tri), jax.block_until_ready(st_tri)
    out_xla, st_xla = jax.block_until_ready(out_xla), jax.block_until_ready(st_xla)
    np.testing.assert_allclose(np.asarray(out_tri, np.float32), np.asarray(out_xla, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(st_tri, np.float32), np.asarray(st_xla, np.float32), rtol=2e-2, atol=2e-2)

    out_mul_tri, st_mul_tri = triton_rwkv7_mul(r, w, k, v, kk, a, initial_state=h0)
    out_mul_xla, st_mul_xla = xla_rwkv7_mul(r, w, k, v, kk, a, initial_state=h0)
    out_mul_tri, st_mul_tri = jax.block_until_ready(out_mul_tri), jax.block_until_ready(st_mul_tri)
    out_mul_xla, st_mul_xla = jax.block_until_ready(out_mul_xla), jax.block_until_ready(st_mul_xla)
    np.testing.assert_allclose(
        np.asarray(out_mul_tri, np.float32), np.asarray(out_mul_xla, np.float32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        np.asarray(st_mul_tri, np.float32), np.asarray(st_mul_xla, np.float32), rtol=2e-2, atol=2e-2
    )


def test_rwkv6_grad_matches_xla_and_vjp():
    B, T, H, K, V = 1, 8, 2, 16, 16
    key = jax.random.PRNGKey(11)
    kr, kk, kv, kw, ku, kh = jax.random.split(key, 6)

    r = jax.random.normal(kr, (B, T, H, K), dtype=jnp.float16)
    k = jax.random.normal(kk, (B, T, H, K), dtype=jnp.float16)
    v = jax.random.normal(kv, (B, T, H, V), dtype=jnp.float16)
    w = jax.random.normal(kw, (B, T, H, K), dtype=jnp.float16) * -0.01
    u = jax.random.normal(ku, (H, K), dtype=jnp.float16)
    h0 = jax.random.normal(kh, (B, H, K, V), dtype=jnp.float32)

    def loss_tri(r_, k_, v_, w_, u_, h0_):
        o, st = triton_rwkv6(r_, k_, v_, w_, u_, initial_state=h0_)
        return jnp.mean(o) + 0.1 * jnp.mean(st)

    def loss_xla(r_, k_, v_, w_, u_, h0_):
        o, st = xla_rwkv6(r_, k_, v_, w_, u_, initial_state=h0_)
        return jnp.mean(o) + 0.1 * jnp.mean(st)

    grads_tri = jax.jit(jax.grad(loss_tri, argnums=(0, 1, 2, 3, 4, 5)))(r, k, v, w, u, h0)
    grads_xla = jax.jit(jax.grad(loss_xla, argnums=(0, 1, 2, 3, 4, 5)))(r, k, v, w, u, h0)
    grads_tri = jax.tree_util.tree_map(jax.block_until_ready, grads_tri)
    grads_xla = jax.tree_util.tree_map(jax.block_until_ready, grads_xla)

    for g_tri, g_xla in zip(grads_tri, grads_xla, strict=True):
        np.testing.assert_allclose(
            np.asarray(g_tri, np.float32),
            np.asarray(g_xla, np.float32),
            rtol=3e-2,
            atol=3e-2,
        )

    def fwd_tri(r_, k_, v_, w_, u_, h0_):
        return triton_rwkv6(r_, k_, v_, w_, u_, initial_state=h0_)

    @jax.jit
    def vjp_eval(r_, k_, v_, w_, u_, h0_):
        (out, st), vjp_fn = jax.vjp(fwd_tri, r_, k_, v_, w_, u_, h0_)
        cot = (jnp.ones_like(out), jnp.ones_like(st))
        return vjp_fn(cot)

    vjp_grads = vjp_eval(r, k, v, w, u, h0)
    for grad in vjp_grads:
        assert jnp.all(jnp.isfinite(grad))


def test_rwkv7_and_rwkv7_mul_grad_match_xla_and_vjp():
    B, T, H, K, V = 1, 8, 2, 16, 16
    key = jax.random.PRNGKey(17)
    kr, kw, kk, kv, ka, kb, kkk, kh = jax.random.split(key, 8)

    r = jax.random.normal(kr, (B, T, H, K), dtype=jnp.float16)
    w = jax.random.normal(kw, (B, T, H, K), dtype=jnp.float16) * -0.01
    k = jax.random.normal(kk, (B, T, H, K), dtype=jnp.float16)
    v = jax.random.normal(kv, (B, T, H, V), dtype=jnp.float16)
    a = jax.random.normal(ka, (B, T, H, K), dtype=jnp.float16) * 0.01
    b = jax.random.normal(kb, (B, T, H, K), dtype=jnp.float16) * 0.01
    kk_mul = jax.random.normal(kkk, (B, T, H, K), dtype=jnp.float16) * 0.01
    h0 = jax.random.normal(kh, (B, H, K, V), dtype=jnp.float32)

    def loss_tri_ab(r_, w_, k_, v_, a_, b_, h0_):
        o, st = triton_rwkv7(r_, w_, k_, v_, a_, b_, initial_state=h0_)
        return jnp.mean(o) + 0.1 * jnp.mean(st)

    def loss_xla_ab(r_, w_, k_, v_, a_, b_, h0_):
        o, st = xla_rwkv7(r_, w_, k_, v_, a_, b_, initial_state=h0_)
        return jnp.mean(o) + 0.1 * jnp.mean(st)

    grads_tri_ab = jax.jit(jax.grad(loss_tri_ab, argnums=(0, 1, 2, 3, 4, 5, 6)))(r, w, k, v, a, b, h0)
    grads_xla_ab = jax.jit(jax.grad(loss_xla_ab, argnums=(0, 1, 2, 3, 4, 5, 6)))(r, w, k, v, a, b, h0)
    grads_tri_ab = jax.tree_util.tree_map(jax.block_until_ready, grads_tri_ab)
    grads_xla_ab = jax.tree_util.tree_map(jax.block_until_ready, grads_xla_ab)

    for g_tri, g_xla in zip(grads_tri_ab, grads_xla_ab, strict=True):
        np.testing.assert_allclose(
            np.asarray(g_tri, np.float32),
            np.asarray(g_xla, np.float32),
            rtol=3e-2,
            atol=3e-2,
        )

    def loss_tri_mul(r_, w_, k_, v_, kk_, a_, h0_):
        o, st = triton_rwkv7_mul(r_, w_, k_, v_, kk_, a_, initial_state=h0_)
        return jnp.mean(o) + 0.1 * jnp.mean(st)

    def loss_xla_mul(r_, w_, k_, v_, kk_, a_, h0_):
        o, st = xla_rwkv7_mul(r_, w_, k_, v_, kk_, a_, initial_state=h0_)
        return jnp.mean(o) + 0.1 * jnp.mean(st)

    grads_tri_mul = jax.jit(jax.grad(loss_tri_mul, argnums=(0, 1, 2, 3, 4, 5, 6)))(r, w, k, v, kk_mul, a, h0)
    grads_xla_mul = jax.jit(jax.grad(loss_xla_mul, argnums=(0, 1, 2, 3, 4, 5, 6)))(r, w, k, v, kk_mul, a, h0)
    grads_tri_mul = jax.tree_util.tree_map(jax.block_until_ready, grads_tri_mul)
    grads_xla_mul = jax.tree_util.tree_map(jax.block_until_ready, grads_xla_mul)

    for g_tri, g_xla in zip(grads_tri_mul, grads_xla_mul, strict=True):
        np.testing.assert_allclose(
            np.asarray(g_tri, np.float32),
            np.asarray(g_xla, np.float32),
            rtol=3e-2,
            atol=3e-2,
        )

    def fwd_tri_ab(r_, w_, k_, v_, a_, b_, h0_):
        return triton_rwkv7(r_, w_, k_, v_, a_, b_, initial_state=h0_)

    @jax.jit
    def vjp_eval_ab(r_, w_, k_, v_, a_, b_, h0_):
        (out_ab, st_ab), vjp_fn_ab = jax.vjp(fwd_tri_ab, r_, w_, k_, v_, a_, b_, h0_)
        cot_ab = (jnp.ones_like(out_ab), jnp.ones_like(st_ab))
        return vjp_fn_ab(cot_ab)

    vjp_grads_ab = vjp_eval_ab(r, w, k, v, a, b, h0)
    for grad in vjp_grads_ab:
        assert jnp.all(jnp.isfinite(grad))

    def fwd_tri_mul(r_, w_, k_, v_, kk_, a_, h0_):
        return triton_rwkv7_mul(r_, w_, k_, v_, kk_, a_, initial_state=h0_)

    @jax.jit
    def vjp_eval_mul(r_, w_, k_, v_, kk_, a_, h0_):
        (out_mul, st_mul), vjp_fn_mul = jax.vjp(fwd_tri_mul, r_, w_, k_, v_, kk_, a_, h0_)
        cot_mul = (jnp.ones_like(out_mul), jnp.ones_like(st_mul))
        return vjp_fn_mul(cot_mul)

    vjp_grads_mul = vjp_eval_mul(r, w, k, v, kk_mul, a, h0)
    for grad in vjp_grads_mul:
        assert jnp.all(jnp.isfinite(grad))
