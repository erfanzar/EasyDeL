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
from ejkernel.kernels._triton.ring_attention._interface import ring_attention as triton_ring_attention
from ejkernel.kernels._xla.ring_attention._interface import ring_attention as xla_ring_attention
from ejkernel.ops import BwdParams, FwdParams

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_ring_attention_matches_xla_single_device():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch, seq_len, num_heads, head_dim = 1, 64, 4, 32
    # Keep K/V heads equal to Q heads for parity with the current XLA implementation.
    num_kv_heads = num_heads

    q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

    fwd_params = FwdParams(q_blocksize=64, kv_blocksize=64, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=64, kv_blocksize=64, num_warps=4, num_stages=2)

    out_triton = triton_ring_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
        axis_name=None,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    out_xla = xla_ring_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
        axis_name=None,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("logits_soft_cap", [None, 20.0])
def test_ring_attention_jit_fwd_and_bwd_matches_xla(causal, logits_soft_cap):
    """Verify jax.jit forward and backward (dq, dk, dv) parity vs XLA, including traced array auxiliary args."""
    kq, kk, kv, kw = jax.random.split(jax.random.PRNGKey(17), 4)
    batch, seq_len, num_heads, head_dim = 2, 64, 4, 32
    q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16) * 0.5
    k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16) * 0.5
    v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16) * 0.5
    w = jax.random.normal(kw, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)

    q_seg = jnp.concatenate([jnp.zeros((batch, 32), dtype=jnp.int32), jnp.ones((batch, 32), dtype=jnp.int32)], axis=1)
    q_pos = jnp.broadcast_to(jnp.concatenate([jnp.arange(32), jnp.arange(32)], axis=0)[None, :], (batch, seq_len))
    fwd_params = FwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)
    scale = head_dim**-0.5

    @jax.jit
    def fwd_triton(q_, k_, v_, seg_, pos_):
        return triton_ring_attention(
            q_,
            k_,
            v_,
            q_segment_ids=seg_,
            kv_segment_ids=seg_,
            q_position_ids=pos_,
            kv_position_ids=pos_,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=scale,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )

    out_triton = fwd_triton(q, k, v, q_seg, q_pos)
    out_xla = xla_ring_attention(
        q,
        k,
        v,
        q_segment_ids=q_seg,
        kv_segment_ids=q_seg,
        q_position_ids=q_pos,
        kv_position_ids=q_pos,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    def loss_triton(q_, k_, v_, seg_, pos_):
        out = triton_ring_attention(
            q_,
            k_,
            v_,
            q_segment_ids=seg_,
            kv_segment_ids=seg_,
            q_position_ids=pos_,
            kv_position_ids=pos_,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=scale,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )
        return jnp.sum(out.astype(jnp.float32) * w)

    def loss_xla(q_, k_, v_, seg_, pos_):
        out = xla_ring_attention(
            q_,
            k_,
            v_,
            q_segment_ids=seg_,
            kv_segment_ids=seg_,
            q_position_ids=pos_,
            kv_position_ids=pos_,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=scale,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )
        return jnp.sum(out.astype(jnp.float32) * w)

    grads_triton = jax.jit(jax.grad(loss_triton, argnums=(0, 1, 2)))(q, k, v, q_seg, q_pos)
    grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2))(q, k, v, q_seg, q_pos)
    for name, got, want in zip(("dq", "dk", "dv"), grads_triton, grads_xla, strict=True):
        got_f32 = np.asarray(got, dtype=np.float32)
        want_f32 = np.asarray(want, dtype=np.float32)
        assert np.all(np.isfinite(got_f32))
        assert np.linalg.norm(want_f32) > 1e-2
        np.testing.assert_allclose(got_f32, want_f32, rtol=5e-2, atol=5e-2, err_msg=f"ring {name} mismatch")


def test_ring_attention_jit_grad_with_softmax_aux_matches_xla():
    """Verify jax.jit(jax.grad(...)) with traced softmax_aux produces accurate q/k/v and sink gradients."""
    kq, kk, kv, kw = jax.random.split(jax.random.PRNGKey(43), 4)
    batch, seq_len, num_heads, head_dim = 1, 64, 4, 32
    q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16) * 0.5
    k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16) * 0.5
    v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16) * 0.5
    w = jax.random.normal(kw, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
    sinks = jnp.array([0.4, -0.2, 0.8], dtype=jnp.float32)
    fwd_params = FwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)

    def loss_triton(q_, k_, v_, aux_):
        out = triton_ring_attention(
            q_,
            k_,
            v_,
            softmax_aux=aux_,
            causal=True,
            softmax_scale=head_dim**-0.5,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )
        return jnp.sum(out.astype(jnp.float32) * w)

    def loss_xla(q_, k_, v_, aux_):
        out = xla_ring_attention(
            q_,
            k_,
            v_,
            softmax_aux=aux_,
            causal=True,
            softmax_scale=head_dim**-0.5,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )
        return jnp.sum(out.astype(jnp.float32) * w)

    grads_triton = jax.jit(jax.grad(loss_triton, argnums=(0, 1, 2, 3)))(q, k, v, sinks)
    grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2, 3))(q, k, v, sinks)
    for name, got, want in zip(("dq", "dk", "dv", "d_aux"), grads_triton, grads_xla, strict=True):
        got_f32 = np.asarray(got, dtype=np.float32)
        want_f32 = np.asarray(want, dtype=np.float32)
        assert np.all(np.isfinite(got_f32))
        assert np.linalg.norm(want_f32) > 1e-2
        np.testing.assert_allclose(got_f32, want_f32, rtol=5e-2, atol=5e-2, err_msg=f"ring sink {name} mismatch")
