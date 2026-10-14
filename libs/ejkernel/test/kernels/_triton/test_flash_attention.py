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


"""Tests for advanced flash attention features (Triton backend)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._triton.flash_attention import flash_attention
from ejkernel.kernels._xla.flash_attention import flash_attention as xla_flash_attention

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


class TestLogitsSoftCap:
    """Test logits soft cap feature."""

    def test_soft_cap_basic(self):
        """Test that soft cap is applied correctly."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 64
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        out_capped = flash_attention(q, k, v, logits_soft_cap=20.0)

        out_uncapped = flash_attention(q, k, v)

        assert out_capped.shape == out_uncapped.shape
        assert not jnp.allclose(out_capped, out_uncapped, rtol=1e-2)

    def test_soft_cap_gradient(self):
        """Test that gradients work with soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 16, 2, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        def loss_fn(q, k, v):
            out = flash_attention(q, k, v, logits_soft_cap=20.0)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.all(jnp.isfinite(grads[0]))
        assert jnp.all(jnp.isfinite(grads[1]))
        assert jnp.all(jnp.isfinite(grads[2]))

    def test_soft_cap_values(self):
        """Test different soft cap values."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)

        out_10 = flash_attention(q, k, v, logits_soft_cap=10.0)
        out_20 = flash_attention(q, k, v, logits_soft_cap=20.0)
        out_30 = flash_attention(q, k, v, logits_soft_cap=30.0)

        assert not jnp.allclose(out_10, out_20, rtol=1e-2)
        assert not jnp.allclose(out_20, out_30, rtol=1e-2)


class TestSoftmaxAux:
    """Test softmax_aux (attention sinks) feature."""

    @pytest.mark.parametrize("sinks", [[0.0, 0.7, -0.5, 1.2], [0.3, -0.5]])
    def test_gqa_per_head_sinks_match_xla_forward_and_gradient(self, sinks):
        """One sink per query head or one per KV head preserve GQA semantics."""
        kq, kk = jax.random.split(jax.random.PRNGKey(91))
        q = jax.random.normal(kq, (1, 4, 4, 16), dtype=jnp.float16) * 0.25
        k = jax.random.normal(kk, (1, 33, 2, 16), dtype=jnp.float16) * 0.25
        v = jnp.broadcast_to(jnp.arange(1, 34, dtype=jnp.float16)[None, :, None, None], (1, 33, 2, 16))
        aux = jnp.asarray(sinks, dtype=jnp.float16)
        actual = flash_attention(q, k, v, softmax_aux=aux, softmax_scale=1.0)
        expected = xla_flash_attention(q, k, v, softmax_aux=aux, softmax_scale=1.0)
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=0.1
        )

        def loss(fn, q_, k_, v_):
            return jnp.sum(fn(q_, k_, v_, softmax_aux=aux, softmax_scale=1.0).astype(jnp.float32))

        gradient = jax.grad(loss, argnums=(1, 2, 3))
        gradients = gradient(flash_attention, q, k, v)
        reference_gradients = gradient(xla_flash_attention, q, k, v)
        for got, want in zip(gradients, reference_gradients, strict=True):
            got_f32 = np.asarray(got, dtype=np.float32)
            want_f32 = np.asarray(want, dtype=np.float32)
            # A zero backward must not pass a parity check on an insensitive input.
            assert np.linalg.norm(want_f32) > 1e-2
            np.testing.assert_allclose(got_f32, want_f32, rtol=0.1, atol=0.1)
            assert np.linalg.norm(got_f32 - want_f32) < 0.1 * np.linalg.norm(want_f32)

    def test_attention_sinks_basic(self):
        """Test basic attention sinks functionality."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 64
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        sinks = jax.random.normal(key, (num_sinks,), dtype=jnp.float16) * 0.1

        out_with_sinks = flash_attention(q, k, v, softmax_aux=sinks)
        out_no_sinks = flash_attention(q, k, v)

        assert out_with_sinks.shape == out_no_sinks.shape
        assert not jnp.allclose(out_with_sinks, out_no_sinks, rtol=1e-2)

    def test_attention_sinks_gradient(self):
        """Test gradients with attention sinks."""
        batch, seq_len, num_heads, head_dim = 2, 16, 2, 32
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        sinks = jax.random.normal(key, (num_sinks,), dtype=jnp.float16)

        def loss_fn(q, k, v):
            out = flash_attention(q, k, v, softmax_aux=sinks)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        for grad in grads:
            assert jnp.all(jnp.isfinite(grad))


class TestCombinedFeatures:
    """Test combinations of advanced features."""

    def test_soft_cap_and_sinks(self):
        """Test using both soft cap and sinks together."""
        batch, seq_len, num_heads, head_dim = 2, 16, 4, 32
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        sinks = jax.random.normal(key, (num_sinks,), dtype=jnp.float16)

        out = flash_attention(q, k, v, logits_soft_cap=20.0, softmax_aux=sinks)

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.all(jnp.isfinite(out))

    def test_causal_with_features(self):
        """Test causal attention with advanced features."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        num_sinks = 2
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        sinks = jax.random.normal(key, (num_sinks,), dtype=jnp.float16)

        out = flash_attention(q, k, v, causal=True, logits_soft_cap=20.0, softmax_aux=sinks)

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.all(jnp.isfinite(out))


def test_flash_attention_matches_xla_smoke():
    key = jax.random.PRNGKey(0)
    kq, kk, kv, ka = jax.random.split(key, 4)

    batch, seq_len, num_heads, head_dim = 2, 64, 4, 64
    q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
    sinks = jax.random.normal(ka, (4,), dtype=jnp.float32) * 0.1

    out_triton = flash_attention(q, k, v, causal=True, logits_soft_cap=20.0, softmax_aux=sinks)
    out_xla = xla_flash_attention(q, k, v, causal=True, logits_soft_cap=20.0, softmax_aux=sinks)

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_flash_attention_gqa_head_bias_and_gradient_match_xla(dtype):
    key = jax.random.PRNGKey(91)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, 48, 4, 64), dtype=dtype)
    k = jax.random.normal(kk, (1, 96, 2, 64), dtype=dtype)
    v = jax.random.normal(kv, (1, 96, 2, 64), dtype=dtype)
    # Distinct per-query-head biases; the two heads sharing a KV head must not share bias.
    bias = jnp.arange(4, dtype=jnp.float32)[None, :, None, None] * jnp.linspace(-1, 1, 96)[None, None, None, :]
    bias = jnp.broadcast_to(bias, (1, 4, 48, 96))

    def loss(fn, q, k, v):
        return jnp.sum(fn(q, k, v, bias=bias).astype(jnp.float32) ** 2)

    actual = flash_attention(q, k, v, bias=bias)
    expected = xla_flash_attention(q, k, v, bias=bias)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=3e-2, rtol=3e-2
    )
    grads = jax.grad(loss, argnums=(1, 2, 3))
    actual_grads = grads(flash_attention, q, k, v)
    expected_grads = grads(xla_flash_attention, q, k, v)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32), atol=6e-2, rtol=6e-2
        )


def test_flash_attention_large_q_block_lse_and_backward():
    from ejkernel.ops import FwdParams

    key = jax.random.PRNGKey(17)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (2, 260, 2, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (2, 64, 2, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (2, 64, 2, 64), dtype=jnp.bfloat16)
    config = FwdParams(q_blocksize=256)
    got = flash_attention(q, k, v, fwd_params=config)
    want = xla_flash_attention(q, k, v)
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32), atol=3e-2, rtol=3e-2
    )
    dq = jax.grad(lambda x: jnp.sum(flash_attention(x, k, v, fwd_params=config).astype(jnp.float32)))(q)
    dq_ref = jax.grad(lambda x: jnp.sum(xla_flash_attention(x, k, v).astype(jnp.float32)))(q)
    np.testing.assert_allclose(
        np.asarray(dq, dtype=np.float32), np.asarray(dq_ref, dtype=np.float32), atol=5e-2, rtol=5e-2
    )


def test_flash_attention_sinks_counted_once_across_kv_tiles():
    key = jax.random.PRNGKey(23)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, 16, 2, 32), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (1, 256, 2, 32), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (1, 256, 2, 32), dtype=jnp.bfloat16)
    sinks = jnp.full((4,), 2.0, dtype=jnp.float32)  # shared: length matches neither head count
    actual = flash_attention(q, k, v, softmax_aux=sinks)
    expected = xla_flash_attention(q, k, v, softmax_aux=sinks)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=3e-2, rtol=3e-2
    )


def test_flash_attention_rejects_more_than_16_sinks():
    q = jnp.ones((1, 16, 2, 32), jnp.bfloat16)
    with pytest.raises(ValueError, match="at most 16 attention sinks"):
        flash_attention(q, q, q, softmax_aux=jnp.zeros((17,), jnp.float32))


def _dense_sink_attention(q, k, v, sinks, scale, causal, soft_cap=None):
    """Independent float32 dense attention where each head has extra sink logits."""
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    reps = q.shape[2] // k.shape[2]
    k = jnp.repeat(k, reps, axis=2)
    v = jnp.repeat(v, reps, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    sink = sinks.astype(jnp.float32)
    if soft_cap is not None:
        logits = soft_cap * jnp.tanh(logits / soft_cap)
        sink = soft_cap * jnp.tanh(sink / soft_cap)
    if causal:
        qs, ks = logits.shape[-2:]
        keep = jnp.arange(qs)[:, None] >= (jnp.arange(ks)[None, :] - ks + qs)
        logits = jnp.where(keep, logits, -jnp.inf)
    sink_cols = jnp.broadcast_to(sink[None, :, None, :], (*logits.shape[:3], sink.shape[-1]))
    probs = jax.nn.softmax(jnp.concatenate([logits, sink_cols], axis=-1), axis=-1)[..., : logits.shape[-1]]
    return jnp.einsum("bhqk,bkhd->bqhd", probs, v)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 20.0])
def test_flash_attention_sink_gradient_matches_dense_reference(causal, soft_cap):
    """The Triton backward returns d(loss)/d(sink), unscaled by softmax_scale, like dense autodiff."""
    kq, kk, kv, ks = jax.random.split(jax.random.PRNGKey(41), 4)
    q = (jax.random.normal(kq, (2, 48, 4, 32)) * 0.5).astype(jnp.float16)
    k = (jax.random.normal(kk, (2, 48, 2, 32)) * 0.5).astype(jnp.float16)
    v = jax.random.normal(kv, (2, 48, 2, 32)).astype(jnp.float16)
    sinks = jax.random.normal(ks, (4, 3), dtype=jnp.float32)  # per query head, 3 sinks
    scale = 0.5  # deliberately not 1.0: sink logits must not be scaled by it

    def loss_triton(s):
        out = flash_attention(q, k, v, causal=causal, softmax_scale=scale, logits_soft_cap=soft_cap, softmax_aux=s)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    def loss_dense(s):
        return jnp.sum(_dense_sink_attention(q, k, v, s, scale, causal, soft_cap) ** 2)

    got = np.asarray(jax.grad(loss_triton)(sinks), dtype=np.float32)
    want = np.asarray(jax.grad(loss_dense)(sinks), dtype=np.float32)
    assert got.shape == sinks.shape
    assert np.linalg.norm(want) > 1e-2  # a zero (dropped) gradient must not pass
    np.testing.assert_allclose(got, want, rtol=5e-2, atol=5e-2)


def test_flash_attention_backward_block_larger_than_forward_block():
    """Backward tiles wider than the forward's must not read past the lse/Delta row stride."""
    from ejkernel.ops import BwdParams, FwdParams

    kq, kk, kv = jax.random.split(jax.random.PRNGKey(53), 3)
    q = jax.random.normal(kq, (2, 16, 2, 32), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (2, 16, 2, 32), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (2, 16, 2, 32), dtype=jnp.bfloat16)
    fwd = FwdParams(q_blocksize=16, kv_blocksize=16)
    bwd = BwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=0)

    def loss(fn, x, **kw):
        return jnp.sum(fn(x, k, v, causal=True, **kw).astype(jnp.float32))

    got = jax.grad(lambda x: loss(flash_attention, x, fwd_params=fwd, bwd_params=bwd))(q)
    want = jax.grad(lambda x: loss(xla_flash_attention, x))(q)
    got_f32 = np.asarray(got, dtype=np.float32)
    assert np.all(np.isfinite(got_f32))
    np.testing.assert_allclose(got_f32, np.asarray(want, dtype=np.float32), atol=5e-2, rtol=5e-2)


def test_flash_attention_causal_q_longer_than_kv_zeroes_fully_masked_rows():
    """Query blocks with no visible key (q_len > kv_len) must be written as zeros, not left uninitialised."""
    from ejkernel.ops import FwdParams

    kq, kk, kv = jax.random.split(jax.random.PRNGKey(67), 3)
    q_len, kv_len = 64, 16
    q = jax.random.normal(kq, (1, q_len, 2, 32), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (1, kv_len, 2, 32), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (1, kv_len, 2, 32), dtype=jnp.bfloat16)
    fwd = FwdParams(q_blocksize=16, kv_blocksize=16)

    out = np.asarray(flash_attention(q, k, v, causal=True, fwd_params=fwd), dtype=np.float32)

    # Independent dense reference: bottom-right aligned causal mask, rows without keys -> 0.
    scale = 1.0 / np.sqrt(32.0)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    keep = jnp.arange(q_len)[:, None] >= (jnp.arange(kv_len)[None, :] - kv_len + q_len)
    probs = jax.nn.softmax(jnp.where(keep, logits, -jnp.inf), axis=-1)
    probs = jnp.where(jnp.any(keep, axis=-1)[None, None, :, None], probs, 0.0)
    want = np.asarray(jnp.einsum("bhqk,bkhd->bqhd", probs, v.astype(jnp.float32)))

    fully_masked = q_len - kv_len
    assert np.all(out[:, :fully_masked] == 0.0)
    np.testing.assert_allclose(out, want, atol=3e-2, rtol=3e-2)

    dq = jax.grad(lambda x: jnp.sum(flash_attention(x, k, v, causal=True, fwd_params=fwd).astype(jnp.float32)))(q)
    assert np.all(np.isfinite(np.asarray(dq, dtype=np.float32)))


def test_flash_attention_soft_cap_large_finite_logits_stay_finite():
    """Soft-cap tanh must not overflow for large positive logits (r12 S05-1).

    Q = K = 10 gives dot products of 1600; the old exp(2x) form evaluated inf/inf. With V = 1 every
    softmax row averages ones, so the exact output is all ones and the gradient is finite.
    """
    q = jnp.full((1, 16, 1, 16), 10.0, dtype=jnp.float16)
    k = jnp.full((1, 16, 1, 16), 10.0, dtype=jnp.float16)
    v = jnp.ones((1, 16, 1, 16), dtype=jnp.float16)
    out = np.asarray(flash_attention(q, k, v, softmax_scale=1.0, logits_soft_cap=20.0), dtype=np.float32)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, np.ones_like(out), atol=1e-2)

    def loss(x):
        return jnp.sum(flash_attention(x, k, v, softmax_scale=1.0, logits_soft_cap=20.0).astype(jnp.float32))

    dq = np.asarray(jax.grad(loss)(q), dtype=np.float32)
    assert np.all(np.isfinite(dq))


def test_flash_attention_large_sink_gradient_finite_on_fully_masked_causal_rows():
    """Fully masked causal rows save lse=0; a large sink must not overflow to inf * 0 = NaN (r13 S05-1)."""
    from ejkernel.ops import FwdParams

    q = jnp.zeros((1, 32, 1, 16), dtype=jnp.float16)
    k = jnp.zeros((1, 16, 1, 16), dtype=jnp.float16)
    v = jnp.ones((1, 16, 1, 16), dtype=jnp.float16)
    fwd = FwdParams(q_blocksize=16, kv_blocksize=16)

    def loss(aux):
        out = flash_attention(q, k, v, causal=True, softmax_aux=aux, fwd_params=fwd)
        return jnp.sum(out.astype(jnp.float32))

    g = np.asarray(jax.grad(loss)(jnp.asarray([100.0], jnp.float32)), dtype=np.float32)
    assert np.all(np.isfinite(g)), g


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
