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


"""Unified tests for flash_attention (XLA backend)."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla.attention._interface import attention
from ejkernel.kernels._xla.flash_attention import flash_attention
from ejkernel.ops import FwdParams


def naive_attention(q, k, v, softmax_scale=None, mask=None):
    """Reference implementation of attention for testing."""
    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(q.shape[-1])

    B, _Tq, H, D = q.shape
    _, Tk, Hk, d = v.shape

    if Hk == 1 and H > 1:
        k = jnp.broadcast_to(k, (B, Tk, H, D))
        v = jnp.broadcast_to(v, (B, Tk, H, d))

    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * softmax_scale

    if mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

    weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return output


class TestBasicFunctionality:
    """Test basic flash attention functionality."""

    def test_basic_attention(self):
        """Test basic attention computation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 128, 8, 64
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        output_flash = flash_attention(q, k, v, fwd_params=FwdParams(q_blocksize=64, kv_blocksize=64))

        output_naive = naive_attention(q, k, v)

        max_diff = jnp.max(jnp.abs(output_flash - output_naive))
        assert max_diff < 1e-2, f"Outputs differ too much: {max_diff}"

    def test_causal_attention(self):
        """Test causal (sliding window) attention."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        causal_mask = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))

        output_flash = flash_attention(
            q, k, v, sliding_window=(T - 1, 0), fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32)
        )

        output_naive = naive_attention(q, k, v, mask=causal_mask)

        max_diff = jnp.max(jnp.abs(output_flash - output_naive))
        assert max_diff < 1e-2, f"Causal outputs differ: {max_diff}"

    def test_mqa_attention(self):
        """Test multi-query attention (MQA)."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 8, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, 1, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, 1, D), dtype=jnp.float32)

        output_flash = flash_attention(q, k, v, fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32))
        output_naive = naive_attention(q, k, v)

        max_diff = jnp.max(jnp.abs(output_flash - output_naive))
        assert max_diff < 1e-2, f"MQA outputs differ: {max_diff}"

    def test_different_dtypes(self):
        """Test with different data types."""
        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32

        for dtype in [jnp.float32, jnp.bfloat16]:
            q = jax.random.normal(keys[0], (B, T, H, D), dtype=dtype)
            k = jax.random.normal(keys[1], (B, T, H, D), dtype=dtype)
            v = jax.random.normal(keys[2], (B, T, H, D), dtype=dtype)

            output = flash_attention(q, k, v, fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32))
            assert output.dtype == dtype
            assert jnp.all(jnp.isfinite(output))


class TestSlidingWindow:
    """Test sliding window attention."""

    def test_sliding_window(self):
        """Test sliding window attention."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 256, 4, 64
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        output_full = flash_attention(q, k, v, fwd_params=FwdParams(q_blocksize=64, kv_blocksize=64))

        output_windowed = flash_attention(
            q, k, v, sliding_window=(64, 64), fwd_params=FwdParams(q_blocksize=64, kv_blocksize=64)
        )

        assert output_windowed.shape == q.shape
        diff = jnp.mean(jnp.abs(output_full - output_windowed))
        assert diff > 1e-6, "Sliding window should affect output"

    def test_asymmetric_window(self):
        """Test asymmetric sliding window."""
        key = jax.random.PRNGKey(111)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 128, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        output_sym = flash_attention(
            q, k, v, sliding_window=(32, 32), fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32)
        )

        output_asym = flash_attention(
            q, k, v, sliding_window=(64, 16), fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32)
        )

        diff = jnp.mean(jnp.abs(output_sym - output_asym))
        assert diff > 1e-6, "Asymmetric window should differ from symmetric"


class TestLogitsSoftCap:
    """Test logits soft cap feature."""

    def test_soft_cap_basic(self):
        """Test that soft cap is applied correctly."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 64
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        out_capped = flash_attention(q, k, v, logits_soft_cap=20.0)

        out_uncapped = flash_attention(q, k, v)

        assert out_capped.shape == out_uncapped.shape
        assert not jnp.allclose(out_capped, out_uncapped, atol=1e-4)

    def test_soft_cap_gradient(self):
        """Test that gradients work with soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 16, 2, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            out = flash_attention(q, k, v, logits_soft_cap=20.0)
            return jnp.sum(out**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.all(jnp.isfinite(grads[0]))
        assert jnp.all(jnp.isfinite(grads[1]))
        assert jnp.all(jnp.isfinite(grads[2]))

    def test_soft_cap_values(self):
        """Test different soft cap values."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        out_10 = flash_attention(q, k, v, logits_soft_cap=10.0)
        out_20 = flash_attention(q, k, v, logits_soft_cap=20.0)
        out_30 = flash_attention(q, k, v, logits_soft_cap=30.0)

        assert not jnp.allclose(out_10, out_20, atol=1e-4)
        assert not jnp.allclose(out_20, out_30, atol=1e-4)

    def test_soft_cap_numerical_stability(self):
        """Test soft cap with large values."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32

        q = jax.random.normal(keys[0], (B, T, H, D)) * 10.0
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        output = flash_attention(q, k, v, logits_soft_cap=30.0)
        assert jnp.all(jnp.isfinite(output)), "Soft cap should prevent overflow"


class TestSoftmaxAux:
    """Test softmax_aux (attention sinks) feature."""

    def test_attention_sinks_basic(self):
        """Test basic attention sinks functionality."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 64
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        sinks = jax.random.normal(key, (num_sinks,)) * 0.1

        out_with_sinks = flash_attention(q, k, v, softmax_aux=sinks)
        out_without_sinks = flash_attention(q, k, v)

        assert out_with_sinks.shape == q.shape
        assert not jnp.allclose(out_with_sinks, out_without_sinks, atol=1e-4)

    def test_attention_sinks_gradient(self):
        """Test gradients with attention sinks."""
        batch, seq_len, num_heads, head_dim = 2, 16, 2, 32
        num_sinks = 2
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        sinks = jax.random.normal(key, (num_sinks,))

        def loss_fn(q, k, v):
            out = flash_attention(q, k, v, softmax_aux=sinks)
            return jnp.sum(out**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)

    def test_different_sink_sizes(self):
        """Test different numbers of attention sinks."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        outputs = {}
        for num_sinks in [2, 4, 8]:
            sinks = jax.random.normal(key, (num_sinks,)) * 0.1
            outputs[num_sinks] = flash_attention(q, k, v, softmax_aux=sinks)

        assert not jnp.allclose(outputs[2], outputs[4], atol=1e-4)
        assert not jnp.allclose(outputs[4], outputs[8], atol=1e-4)


class TestCombinedFeatures:
    """Test combinations of features."""

    def test_soft_cap_and_window(self):
        """Test soft cap with sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 128, 4, 64
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        out = flash_attention(
            q,
            k,
            v,
            sliding_window=(64, 64),
            logits_soft_cap=30.0,
        )

        assert out.shape == q.shape
        assert jnp.all(jnp.isfinite(out))

    def test_soft_cap_and_sinks(self):
        """Test soft cap with attention sinks."""
        batch, seq_len, num_heads, head_dim = 2, 64, 4, 64
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        sinks = jax.random.normal(key, (num_sinks,))

        out = flash_attention(
            q,
            k,
            v,
            logits_soft_cap=30.0,
            softmax_aux=sinks,
        )

        assert out.shape == q.shape
        assert jnp.all(jnp.isfinite(out))

    def test_all_features(self):
        """Test all features combined."""
        batch, seq_len, num_heads, head_dim = 2, 128, 4, 64
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        sinks = jax.random.normal(key, (num_sinks,))

        out = flash_attention(
            q,
            k,
            v,
            sliding_window=(64, 64),
            logits_soft_cap=30.0,
            softmax_aux=sinks,
            fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32),
        )

        assert out.shape == q.shape
        assert jnp.all(jnp.isfinite(out))


class TestGradients:
    """Test gradient computation."""

    def test_basic_gradients(self):
        """Test basic gradient computation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        def loss_fn(q, k, v):
            output = flash_attention(q, k, v, fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32))
            return jnp.sum(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert all(jnp.all(jnp.isfinite(g)) for g in grads)
        assert all(g.shape == inp.shape for g, inp in zip(grads, [q, k, v], strict=False))

    def test_gradients_with_all_features(self):
        """Test gradients with all features enabled."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        num_sinks = 2
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))
        sinks = jax.random.normal(keys[2], (num_sinks,))

        def loss_fn(q, k, v):
            output = flash_attention(
                q,
                k,
                v,
                sliding_window=(32, 32),
                logits_soft_cap=30.0,
                softmax_aux=sinks,
                fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32),
            )
            return jnp.sum(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)


class TestVanillaComparison:
    """Compare flash_attention with vanilla attention."""

    def test_basic_vs_vanilla(self):
        """Compare basic flash attention with vanilla."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        flash_out = flash_attention(q, k, v, fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32))
        vanilla_out, _ = attention(q, k, v)

        max_diff = float(jnp.max(jnp.abs(flash_out - vanilla_out)))
        print(f"\nBasic flash vs vanilla: max_diff={max_diff:.6e}")
        assert jnp.allclose(flash_out, vanilla_out, atol=2e-2)

    def test_causal_vs_vanilla(self):
        """Compare causal flash attention with vanilla."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = jnp.broadcast_to(causal_mask, (B, 1, T, T))

        flash_out = flash_attention(q, k, v, causal=True, fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32))
        vanilla_out, _ = attention(q, k, v, attention_mask=causal_mask)

        max_diff = float(jnp.max(jnp.abs(flash_out - vanilla_out)))
        print(f"\nCausal flash vs vanilla: max_diff={max_diff:.6e}")
        assert jnp.allclose(flash_out, vanilla_out, atol=2e-2)

    def test_bias_vs_vanilla(self):
        """Compare flash attention with bias against vanilla."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 4)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))
        bias = jax.random.normal(keys[3], (B, H, T, T)) * 0.1

        flash_out = flash_attention(q, k, v, bias=bias, fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32))
        vanilla_out, _ = attention(q, k, v, bias=bias)

        max_diff = float(jnp.max(jnp.abs(flash_out - vanilla_out)))
        print(f"\nBias flash vs vanilla: max_diff={max_diff:.6e}")
        assert jnp.allclose(flash_out, vanilla_out, atol=2e-2)

    def test_sliding_window_vs_vanilla(self):
        """Compare flash attention with sliding window against vanilla."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        window = 16
        flash_out = flash_attention(
            q, k, v, sliding_window=(window, window), fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32)
        )
        vanilla_out, _ = attention(q, k, v, sliding_window=window)

        max_diff = float(jnp.max(jnp.abs(flash_out - vanilla_out)))
        print(f"\nSliding window flash vs vanilla: max_diff={max_diff:.6e}")
        assert jnp.allclose(flash_out, vanilla_out, atol=2e-2)

    def test_softmax_scale_vs_vanilla(self):
        """Compare flash attention with custom softmax softmax_scale against vanilla."""
        key = jax.random.PRNGKey(111)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        softmax_scale = 0.5
        flash_out = flash_attention(
            q, k, v, softmax_scale=softmax_scale, fwd_params=FwdParams(q_blocksize=32, kv_blocksize=32)
        )
        vanilla_out, _ = attention(q, k, v, softmax_scale=softmax_scale)

        max_diff = float(jnp.max(jnp.abs(flash_out - vanilla_out)))
        print(f"\nSoftmax softmax_scale flash vs vanilla: max_diff={max_diff:.6e}")
        assert jnp.allclose(flash_out, vanilla_out, atol=5e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
