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


"""Tests for XLA ragged decode attention implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._xla.ragged_decode_attention import ragged_decode_attention


def ragged_decode_ref(
    query,
    key,
    value,
    sequence_start,
    sequence_end,
    softmax_scale=1.0,
    sliding_window=None,
    logits_soft_cap=None,
    softmax_aux=None,
):
    """Dense reference for ragged decode.

    Matches the intended semantics of `softmax_aux`: it contributes to the
    softmax normalization mass but does not contribute to the output values.
    """
    B, HQ, D = query.shape
    S, HKV = key.shape[1], key.shape[2]
    assert HQ % HKV == 0
    group_size = HQ // HKV

    out = jnp.zeros((B, HQ, D), dtype=query.dtype)
    if softmax_aux is not None and getattr(softmax_aux, "ndim", None) != 1:
        raise ValueError(f"softmax_aux must be 1D (num_sinks,), got shape {softmax_aux.shape}")

    for b in range(B):
        s_b = int(sequence_start[b])
        e_b = int(sequence_end[b])
        if e_b <= s_b:
            continue
        qpos = e_b - 1

        left = sliding_window[0] if sliding_window is not None else None
        right = sliding_window[1] if sliding_window is not None else None

        t_idx = jnp.arange(S)
        mask = (t_idx >= s_b) & (t_idx < e_b)

        if left is not None:
            mask = mask & (t_idx >= (qpos - left))
        if right is not None:
            mask = mask & (t_idx <= (qpos + right))

        for hq in range(HQ):
            kvh = hq // group_size
            q_vec = query[b, hq]

            k_b = key[b, :, kvh, :]
            v_b = value[b, :, kvh, :]

            logits = jnp.einsum("d,sd->s", q_vec, k_b) * float(softmax_scale)
            if logits_soft_cap is not None and logits_soft_cap > 0:
                cap = float(logits_soft_cap)
                logits = cap * jnp.tanh(logits / cap)

            logits_seq = jnp.where(mask, logits, -jnp.inf)

            if softmax_aux is None:
                m = jnp.max(logits_seq)
                p_seq = jnp.exp(logits_seq - m)
                denom = jnp.sum(p_seq)
            else:
                logits_sink = softmax_aux.astype(jnp.float32)
                m = jnp.maximum(jnp.max(logits_seq), jnp.max(logits_sink))
                p_seq = jnp.exp(logits_seq - m)
                p_sink = jnp.exp(logits_sink - m)
                denom = jnp.sum(p_seq) + jnp.sum(p_sink)

            denom = jnp.where(denom == 0.0, 1.0, denom)
            w_seq = p_seq / denom
            o_vec = jnp.einsum("s,sd->d", w_seq, v_b)
            out = out.at[b, hq].set(o_vec.astype(query.dtype))

    return out


class TestBasicFunctionality:
    """Test basic ragged decode attention functionality."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 512, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([384, 512], dtype=jnp.int32)

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_gqa_support(self):
        """Test Grouped Query Attention (GQA) with different ratios."""
        batch, seq_len, head_dim = 2, 256, 64

        test_cases = [
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 8),
        ]

        for num_heads, num_kv_heads in test_cases:
            query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
            key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
            value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

            sequence_start = jnp.array([0, 0], dtype=jnp.int32)
            sequence_end = jnp.array([128, 256], dtype=jnp.int32)

            output = ragged_decode_attention(
                query=query,
                key=key,
                value=value,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )

            assert output.shape == (batch, num_heads, head_dim)
            assert not jnp.any(jnp.isnan(output))

    def test_different_sequence_lengths(self):
        """Test with varying sequence lengths per batch."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 4, 512, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([128, 256, 384, 512], dtype=jnp.int32)

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))


class TestSlidingWindow:
    """Test sliding window attention."""

    def test_symmetric_window(self):
        """Test symmetric sliding window."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 512, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([512, 512], dtype=jnp.int32)

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            sliding_window=(128, 128),
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_asymmetric_window(self):
        """Test asymmetric sliding window (different left/right)."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 512, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([512, 512], dtype=jnp.int32)

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            sliding_window=(256, 64),
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_window_with_ragged_sequences(self):
        """Test sliding window combined with ragged sequences."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 3, 512, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 384, 512], dtype=jnp.int32)

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            sliding_window=(128, 128),
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))


class TestLogitSoftCap:
    """Test logit soft capping."""

    def test_soft_cap_basic(self):
        """Test that soft capping works without errors."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 256], dtype=jnp.int32)

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            logits_soft_cap=30.0,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_soft_cap_different_values(self):
        """Test different soft cap values."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 256], dtype=jnp.int32)

        for cap_value in [10.0, 20.0, 30.0, 50.0]:
            output = ragged_decode_attention(
                query=query,
                key=key,
                value=value,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                logits_soft_cap=cap_value,
            )

            assert output.shape == (batch, num_heads, head_dim)
            assert not jnp.any(jnp.isnan(output))


class TestSoftmaxAux:
    """Test attention sinks (softmax aux)."""

    def test_attention_sinks_shared(self):
        """Test attention sinks shared across all heads."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 8, 2, 64
        num_sinks = 4

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 256], dtype=jnp.int32)

        softmax_aux = jnp.ones(num_sinks) * 5.0

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            softmax_aux=softmax_aux,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_attention_sinks_per_kv_head(self):
        """Per-KV-head sink logits are supported."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 8, 2, 64
        num_sinks = 4

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 256], dtype=jnp.int32)

        softmax_aux = jnp.ones((num_kv_heads, num_sinks), dtype=jnp.float32) * 5.0
        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            softmax_aux=softmax_aux,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_different_sink_strengths(self):
        """Test varying sink strength values."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 256], dtype=jnp.int32)

        softmax_aux = jnp.array([10.0, 8.0, 6.0, 4.0])

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            softmax_aux=softmax_aux,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))


class TestCombinedFeatures:
    """Test combinations of features."""

    def test_all_features_combined(self):
        """Test all features together."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 512, 8, 2, 64
        num_sinks = 4

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([384, 512], dtype=jnp.int32)

        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * 5.0

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            softmax_scale=float(1.0 / jnp.sqrt(head_dim)),
            sliding_window=(256, 256),
            logits_soft_cap=30.0,
            softmax_aux=softmax_aux,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_window_with_soft_cap(self):
        """Test sliding window combined with soft capping."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 512, 8, 2, 64

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([512, 512], dtype=jnp.int32)

        output = ragged_decode_attention(
            query=query,
            key=key,
            value=value,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            sliding_window=(128, 128),
            logits_soft_cap=30.0,
        )

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_matches_dense_reference_with_sinks(self):
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 16, 2, 64

        key = jax.random.PRNGKey(0)
        kq, kk, kv, ka = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch, num_heads, head_dim), dtype=jnp.float32)
        k = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)

        sequence_start = jnp.array([0, 32], dtype=jnp.int32)
        sequence_end = jnp.array([200, 180], dtype=jnp.int32)
        softmax_aux = jax.random.normal(ka, (4,), dtype=jnp.float32) * 0.1

        out_xla = ragged_decode_attention(
            query=query,
            key=k,
            value=v,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            softmax_scale=1.0,
            sliding_window=(128, 0),
            logits_soft_cap=20.0,
            softmax_aux=softmax_aux,
        )
        out_ref = ragged_decode_ref(
            query,
            k,
            v,
            sequence_start,
            sequence_end,
            softmax_scale=1.0,
            sliding_window=(128, 0),
            logits_soft_cap=20.0,
            softmax_aux=softmax_aux,
        )

        out_xla = jax.block_until_ready(out_xla)
        out_ref = jax.block_until_ready(out_ref)
        np.testing.assert_allclose(out_xla, out_ref, rtol=2e-4, atol=2e-4)


class TestJITCompilation:
    """Test JIT compilation."""

    def test_jit_basic(self):
        """Test basic JIT compilation."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 8, 2, 64

        @jax.jit
        def run_attention(q, k, v, start, end):
            return ragged_decode_attention(
                query=q,
                key=k,
                value=v,
                sequence_start=start,
                sequence_end=end,
            )

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 256], dtype=jnp.int32)

        output = run_attention(query, key, value, sequence_start, sequence_end)

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_jit_with_all_features(self):
        """Test JIT compilation with all features."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 256, 8, 2, 64
        num_sinks = 4

        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * 5.0

        @jax.jit
        def run_attention(q, k, v, start, end):
            return ragged_decode_attention(
                query=q,
                key=k,
                value=v,
                sequence_start=start,
                sequence_end=end,
                sliding_window=(128, 128),
                logits_soft_cap=30.0,
                softmax_aux=softmax_aux,
            )

        query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_kv_heads, head_dim))

        sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        sequence_end = jnp.array([256, 256], dtype=jnp.int32)

        output = run_attention(query, key, value, sequence_start, sequence_end)

        assert output.shape == (batch, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
