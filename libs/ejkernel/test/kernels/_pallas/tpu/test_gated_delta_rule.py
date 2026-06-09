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

"""TPU Pallas tests for gated_delta_rule."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.gated_delta_rule import gated_delta_rule as gated_delta_rule_pallas
from ejkernel.kernels._xla.gated_delta_rule import gated_delta_rule as gated_delta_rule_xla


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def _make_inputs(batch=1, seq_len=8, heads=2, qk_dim=8, v_dim=8, dtype=jnp.float32, seed=0):
    key = jax.random.PRNGKey(seed)
    kq, kk, kv, kb, kd = jax.random.split(key, 5)
    q = jax.random.normal(kq, (batch, seq_len, heads, qk_dim), dtype=dtype)
    k = jax.random.normal(kk, (batch, seq_len, heads, qk_dim), dtype=dtype)
    v = jax.random.normal(kv, (batch, seq_len, heads, v_dim), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(kb, (batch, seq_len, heads), dtype=dtype))
    decay = jax.random.normal(kd, (batch, seq_len, heads), dtype=dtype) * -0.01
    return q, k, v, beta, decay


def test_pallas_interface_returns_shapes_and_finite():
    q, k, v, beta, decay = _make_inputs(dtype=jnp.float32)
    out, state = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    assert out.shape == v.shape
    assert state.shape == (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(state))


def test_pallas_chunked_matches_xla_recurrent():
    q, k, v, beta, decay = _make_inputs(seq_len=16, dtype=jnp.float32, seed=1)
    out_pallas, state_pallas = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    out_xla, state_xla = gated_delta_rule_xla(q, k, v, beta, decay, chunk_size=8, use_chunked=False)

    assert jnp.allclose(out_pallas, out_xla, atol=2e-2, rtol=0)
    assert jnp.allclose(state_pallas, state_xla, atol=2e-2, rtol=0)


def test_pallas_no_decay():
    q, k, v, beta, _ = _make_inputs(dtype=jnp.bfloat16, seed=2)
    out, state = gated_delta_rule_pallas(q, k, v, beta, None, chunk_size=8, use_chunked=True)
    assert out.shape == v.shape
    assert state.shape == (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(state))


def test_pallas_single_step_continuation():
    q, k, v, beta, decay = _make_inputs(seq_len=9, dtype=jnp.float32, seed=3)
    out_full, _ = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    _, state = gated_delta_rule_pallas(q[:, :8], k[:, :8], v[:, :8], beta[:, :8], decay[:, :8], chunk_size=8)
    out_last, state_last = gated_delta_rule_pallas(
        q[:, 8:],
        k[:, 8:],
        v[:, 8:],
        beta[:, 8:],
        decay[:, 8:],
        initial_state=state,
    )
    assert out_last.shape == (q.shape[0], 1, q.shape[2], v.shape[3])
    assert state_last.shape == state.shape
    assert jnp.allclose(out_last, out_full[:, 8:], atol=2e-2, rtol=0)


@pytest.mark.parametrize(("dtype", "seed"), ((jnp.float32, 13), (jnp.bfloat16, 33)))
def test_pallas_single_step_matches_xla(dtype, seed):
    q, k, v, beta, decay = _make_inputs(seq_len=1, dtype=dtype, seed=seed)
    init = jax.random.normal(
        jax.random.PRNGKey(seed + 1),
        (q.shape[0], q.shape[2], q.shape[3], v.shape[3]),
        dtype=dtype,
    )
    out_pallas, state_pallas = gated_delta_rule_pallas(q, k, v, beta, decay, initial_state=init)
    out_xla, state_xla = gated_delta_rule_xla(q, k, v, beta, decay, initial_state=init, use_chunked=False)
    assert out_pallas.dtype == dtype
    assert state_pallas.dtype == dtype
    atol = 0.02 if dtype == jnp.bfloat16 else 1e-5
    assert jnp.allclose(out_pallas, out_xla, atol=atol, rtol=0)
    assert jnp.allclose(state_pallas, state_xla, atol=atol, rtol=0)


def test_pallas_bfloat16_inputs():
    q, k, v, beta, decay = _make_inputs(dtype=jnp.bfloat16, seed=4)
    out, state = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=True)
    assert out.dtype in (jnp.bfloat16, jnp.float32)
    assert state.dtype in (jnp.bfloat16, jnp.float32)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(state))


def test_pallas_backward_matches_xla_recurrent():
    q, k, v, beta, decay = _make_inputs(dtype=jnp.float32, seed=5)

    def loss_pallas(q_in, k_in, v_in, beta_in, decay_in):
        out, state = gated_delta_rule_pallas(q_in, k_in, v_in, beta_in, decay_in, chunk_size=8, use_chunked=True)
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    def loss_xla(q_in, k_in, v_in, beta_in, decay_in):
        out, state = gated_delta_rule_xla(q_in, k_in, v_in, beta_in, decay_in, chunk_size=8, use_chunked=False)
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    grads_pallas = jax.grad(loss_pallas, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)

    for grad_pallas, grad_xla, reference in zip(grads_pallas, grads_xla, (q, k, v, beta, decay), strict=True):
        assert grad_pallas.shape == reference.shape
        assert jnp.all(jnp.isfinite(grad_pallas))
        assert jnp.allclose(grad_pallas, grad_xla, atol=2e-2, rtol=0)


def test_pallas_single_step_backward_matches_xla():
    q, k, v, beta, decay = _make_inputs(seq_len=1, dtype=jnp.float32, seed=50)
    init = (
        jax.random.normal(
            jax.random.PRNGKey(51),
            (q.shape[0], q.shape[2], q.shape[3], v.shape[3]),
            dtype=jnp.float32,
        )
        * 0.01
    )

    def loss_pallas(q_in, k_in, v_in, beta_in, decay_in, state_in):
        out, state = gated_delta_rule_pallas(q_in, k_in, v_in, beta_in, decay_in, initial_state=state_in)
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    def loss_xla(q_in, k_in, v_in, beta_in, decay_in, state_in):
        out, state = gated_delta_rule_xla(
            q_in,
            k_in,
            v_in,
            beta_in,
            decay_in,
            initial_state=state_in,
            use_chunked=False,
        )
        return jnp.sum(out) + 0.1 * jnp.sum(state)

    grads_pallas = jax.grad(loss_pallas, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, beta, decay, init)
    grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, beta, decay, init)
    for grad_pallas, grad_xla in zip(grads_pallas, grads_xla, strict=True):
        assert jnp.all(jnp.isfinite(grad_pallas))
        assert jnp.allclose(grad_pallas, grad_xla, atol=1e-5, rtol=0)


def test_pallas_use_chunked_false_falls_back_to_xla_recurrent():
    q, k, v, beta, decay = _make_inputs(dtype=jnp.float32, seed=6)
    out_pallas, state_pallas = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=8, use_chunked=False)
    out_xla, state_xla = gated_delta_rule_xla(q, k, v, beta, decay, chunk_size=8, use_chunked=False)
    assert jnp.array_equal(out_pallas, out_xla)
    assert jnp.array_equal(state_pallas, state_xla)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_pallas_realistic_dims_fwd_bwd(dtype):
    """Test with dimensions matching real models (K=128, V=128, C=64, many chunks)."""
    batch, seq_len, heads, qk_dim, v_dim = 1, 512, 2, 128, 128
    q, k, v, beta, decay = _make_inputs(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=dtype,
        seed=7,
    )

    out, state = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=64, use_chunked=True)
    assert out.shape == v.shape
    assert jnp.all(jnp.isfinite(out)), "Forward output contains NaN/Inf"
    assert jnp.all(jnp.isfinite(state)), "Forward state contains NaN/Inf"

    def loss_fn(q_in, k_in, v_in, beta_in, decay_in):
        o, s = gated_delta_rule_pallas(q_in, k_in, v_in, beta_in, decay_in, chunk_size=64, use_chunked=True)
        return jnp.sum(o) + 0.1 * jnp.sum(s)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    for i, g in enumerate(grads):
        assert jnp.all(jnp.isfinite(g)), f"Gradient {i} contains NaN/Inf"


def test_pallas_realistic_dims_matches_xla():
    """Pallas matches XLA at realistic dims (f32 for tight tolerance)."""
    q, k, v, beta, decay = _make_inputs(
        batch=1,
        seq_len=256,
        heads=2,
        qk_dim=128,
        v_dim=128,
        dtype=jnp.float32,
        seed=8,
    )
    out_p, st_p = gated_delta_rule_pallas(q, k, v, beta, decay, chunk_size=64, use_chunked=True)
    out_x, st_x = gated_delta_rule_xla(q, k, v, beta, decay, chunk_size=64, use_chunked=False)
    assert jnp.allclose(out_p, out_x, atol=2e-2, rtol=0), f"Output max diff: {jnp.max(jnp.abs(out_p - out_x))}"
    assert jnp.allclose(st_p, st_x, atol=2e-2, rtol=0), f"State max diff: {jnp.max(jnp.abs(st_p - st_x))}"
