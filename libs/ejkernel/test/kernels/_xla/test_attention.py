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


"""Tests for XLA attention kernel with sliding window support."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla.attention._interface import attention


def naive_attention_with_window(q, k, v, softmax_scale=None, sliding_window=None):
    """Reference implementation with sliding window for testing."""
    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(q.shape[-1])

    B, qs, H, _D = q.shape
    _, ks, _, _ = k.shape

    kv_heads = k.shape[2]
    if kv_heads != H:
        num_reps = H // kv_heads
        k = jnp.repeat(k, num_reps, axis=2)
        v = jnp.repeat(v, num_reps, axis=2)

    logits = jnp.einsum("bqhd,bkhd->bhqk", q * softmax_scale, k)

    if sliding_window is not None:
        if isinstance(sliding_window, int):
            left_window = sliding_window
            right_window = sliding_window
        else:
            left_window, right_window = sliding_window

        q_pos = jnp.arange(qs)[:, None]
        k_pos = jnp.arange(ks)[None, :]
        window_mask = (k_pos >= q_pos - left_window) & (k_pos <= q_pos + right_window)
        window_mask = jnp.broadcast_to(window_mask[None, None, :, :], (B, H, qs, ks))
        logits = jnp.where(window_mask, logits, jnp.finfo(logits.dtype).min)

    weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return output


class TestSlidingWindow:
    """Test sliding window attention functionality."""

    def test_symmetric_window(self):
        """Test symmetric sliding window."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        window_size = 16
        output_xla, _ = attention(q, k, v, sliding_window=window_size, dtype=jnp.float32)
        output_naive = naive_attention_with_window(q, k, v, sliding_window=window_size)

        max_diff = jnp.max(jnp.abs(output_xla - output_naive))
        assert max_diff < 2e-3, f"Symmetric window outputs differ: {max_diff}"

    def test_asymmetric_window(self):
        """Test asymmetric sliding window."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        window = (32, 8)
        output_xla, _ = attention(q, k, v, sliding_window=window, dtype=jnp.float32)
        output_naive = naive_attention_with_window(q, k, v, sliding_window=window)

        max_diff = jnp.max(jnp.abs(output_xla - output_naive))
        assert max_diff < 2e-3, f"Asymmetric window outputs differ: {max_diff}"

    def test_causal_window(self):
        """Test causal attention as a special case of sliding window."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 32, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        window = (T - 1, 0)
        output_xla, _ = attention(q, k, v, sliding_window=window, dtype=jnp.float32)
        output_naive = naive_attention_with_window(q, k, v, sliding_window=window)

        max_diff = jnp.max(jnp.abs(output_xla - output_naive))
        assert max_diff < 2e-3, f"Causal window outputs differ: {max_diff}"

    def test_no_window_vs_full_attention(self):
        """Test that no window gives same result as full attention."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 32, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        output_no_window, _ = attention(q, k, v, sliding_window=None, dtype=jnp.float32)
        output_large_window, _ = attention(q, k, v, sliding_window=(T, T), dtype=jnp.float32)

        max_diff = jnp.max(jnp.abs(output_no_window - output_large_window))
        assert max_diff < 1e-5, f"No window vs large window differ: {max_diff}"

    def test_window_with_gqa(self):
        """Test sliding window with grouped query attention."""
        key = jax.random.PRNGKey(111)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 64, 8, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, 2, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, 2, D), dtype=jnp.float32)

        window_size = 16
        output_xla, _ = attention(q, k, v, sliding_window=window_size, dtype=jnp.float32)
        output_naive = naive_attention_with_window(q, k, v, sliding_window=window_size)

        max_diff = jnp.max(jnp.abs(output_xla - output_naive))
        assert max_diff < 2e-3, f"GQA with window outputs differ: {max_diff}"

    def test_window_effect_on_output(self):
        """Test that sliding window actually affects the output."""
        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 128, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        output_full, _ = attention(q, k, v, sliding_window=None, dtype=jnp.float32)
        output_windowed, _ = attention(q, k, v, sliding_window=16, dtype=jnp.float32)

        diff = jnp.mean(jnp.abs(output_full - output_windowed))
        assert diff > 1e-3, f"Sliding window should affect output, but diff={diff}"


class TestWindowGradients:
    """Test gradients with sliding window."""

    def test_symmetric_window_gradients(self):
        """Test gradients work with symmetric window."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 32, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        def loss_fn(q, k, v):
            output, _ = attention(q, k, v, sliding_window=16)
            return jnp.sum(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert all(jnp.all(jnp.isfinite(g)) for g in grads)
        assert all(g.shape == inp.shape for g, inp in zip(grads, [q, k, v], strict=False))

    def test_asymmetric_window_gradients(self):
        """Test gradients work with asymmetric window."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 32, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D))
        k = jax.random.normal(keys[1], (B, T, H, D))
        v = jax.random.normal(keys[2], (B, T, H, D))

        def loss_fn(q, k, v):
            output, _ = attention(q, k, v, sliding_window=(24, 8))
            return jnp.sum(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert all(jnp.all(jnp.isfinite(g)) for g in grads)


class TestWindowWithOtherFeatures:
    """Test sliding window combined with other features."""

    def test_window_with_mask(self):
        """Test sliding window with additional mask."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 32, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        mask = jnp.tril(jnp.ones((B, 1, T, T), dtype=bool))

        output, _ = attention(q, k, v, attention_mask=mask, sliding_window=16, dtype=jnp.float32)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_window_with_bias(self):
        """Test sliding window with attention bias."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 4)

        B, T, H, D = 2, 32, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)
        bias = jax.random.normal(keys[3], (B, H, T, T), dtype=jnp.float32) * 0.1

        output, _ = attention(q, k, v, bias=bias, sliding_window=16, dtype=jnp.float32)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_window_with_dropout(self):
        """Test sliding window with dropout."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 4)

        B, T, H, D = 2, 32, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        output, _ = attention(
            q,
            k,
            v,
            sliding_window=16,
            deterministic=False,
            dropout_prob=0.1,
            dropout_rng=keys[3],
            dtype=jnp.float32,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))


class TestEdgeCases:
    """Test edge cases for sliding window."""

    def test_window_size_zero(self):
        """Test window size of 0 (each position only attends to itself)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        B, T, H, D = 2, 16, 4, 32
        q = jax.random.normal(keys[0], (B, T, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float32)

        output, _ = attention(q, k, v, sliding_window=0, dtype=jnp.float32)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_different_sequence_lengths(self):
        """Test sliding window with different query and key sequence lengths."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        B, Tq, Tk, H, D = 2, 32, 64, 4, 32
        q = jax.random.normal(keys[0], (B, Tq, H, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, Tk, H, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, Tk, H, D), dtype=jnp.float32)

        output, _ = attention(q, k, v, sliding_window=16, dtype=jnp.float32)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
