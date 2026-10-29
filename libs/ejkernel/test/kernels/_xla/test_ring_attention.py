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


"""Unified tests for ring_attention (XLA implementation)."""

import jax
import jax.numpy as jnp
import pytest
from ejkernel.kernels._xla.attention._interface import attention
from ejkernel.kernels._xla.ring_attention._interface import ring_attention
from ejkernel.ops import FwdParams


class TestBasicFunctionality:
    """Test basic ring attention functionality."""

    def test_basic_ring_attention(self):
        """Test basic ring attention without special features."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(q, k, v)

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_causal_masking(self):
        """Test with causal masking."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(q, k, v)

        output_causal = ring_attention(
            q,
            k,
            v,
            causal=True,
        )

        assert output_causal.shape == q.shape
        diff = float(jnp.mean(jnp.abs(output - output_causal).astype(jnp.bfloat16)))
        assert diff > 1e-6, "Causal masking should affect output"

    def test_different_block_sizes(self):
        """Test that different block sizes produce similar outputs."""
        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        outputs = []
        for blocksize in [32, 64, 128]:
            output = ring_attention(
                q,
                k,
                v,
                fwd_params=FwdParams(q_blocksize=blocksize, kv_blocksize=blocksize),
            )
            outputs.append(output)

        for i in range(len(outputs) - 1):
            diff = float(jnp.mean(jnp.abs(outputs[i] - outputs[i + 1]).astype(jnp.bfloat16)))
            assert diff < 1e-3, f"Block size configuration affects output too much: {diff}"


class TestSlidingWindow:
    """Test sliding window attention."""

    def test_symmetric_sliding_window(self):
        """Test symmetric sliding window (int)."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output_full = ring_attention(
            q,
            k,
            v,
        )

        output_windowed = ring_attention(
            q,
            k,
            v,
            sliding_window=64,
        )

        assert output_windowed.shape == q.shape
        assert jnp.all(jnp.isfinite(output_windowed))

        diff = float(jnp.mean(jnp.abs(output_full - output_windowed).astype(jnp.bfloat16)))
        assert diff > 1e-6, "Sliding window should affect output"

    def test_asymmetric_sliding_window(self):
        """Test asymmetric sliding window (tuple)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output_sym = ring_attention(
            q,
            k,
            v,
            sliding_window=64,
        )

        output_asym = ring_attention(
            q,
            k,
            v,
            sliding_window=(32, 96),
        )

        assert output_asym.shape == q.shape
        assert jnp.all(jnp.isfinite(output_asym))

        diff = float(jnp.mean(jnp.abs(output_sym - output_asym).astype(jnp.bfloat16)))
        assert diff > 1e-6, "Asymmetric should differ from symmetric"

    def test_different_window_sizes(self):
        """Test different window sizes."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        outputs = {}
        for window_size in [32, 64, 128]:
            outputs[window_size] = ring_attention(
                q,
                k,
                v,
                sliding_window=window_size,
            )

        diff_32_64 = float(jnp.mean(jnp.abs(outputs[32] - outputs[64]).astype(jnp.bfloat16)))
        diff_64_128 = float(jnp.mean(jnp.abs(outputs[64] - outputs[128]).astype(jnp.bfloat16)))

        assert diff_32_64 > 1e-6
        assert diff_64_128 > 1e-6


class TestLogitSoftCap:
    """Test logit soft cap feature."""

    def test_soft_cap_affects_output(self):
        """Test that soft cap actually changes the output."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output_no_cap = ring_attention(
            q,
            k,
            v,
        )

        output_with_cap = ring_attention(
            q,
            k,
            v,
            logits_soft_cap=30.0,
        )

        assert output_with_cap.shape == q.shape
        assert jnp.all(jnp.isfinite(output_with_cap))

        diff = float(jnp.mean(jnp.abs(output_no_cap - output_with_cap).astype(jnp.bfloat16)))
        assert diff > 1e-6, "Soft cap should affect output"

    def test_soft_cap_numerical_stability(self):
        """Test that soft cap prevents overflow with large values."""
        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim)) * 10.0
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(
            q,
            k,
            v,
            logits_soft_cap=30.0,
        )

        assert jnp.all(jnp.isfinite(output)), "Soft cap should prevent overflow"


class TestAttentionSink:
    """Test softmax_aux (attention sinks) feature."""

    def test_attention_sink_affects_output(self):
        """Test that softmax_aux changes the output."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output_no_sink = ring_attention(
            q,
            k,
            v,
            sliding_window=64,
        )

        softmax_aux = jnp.full((4,), -2.0, dtype=jnp.float32)
        output_with_sink = ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            softmax_aux=softmax_aux,
        )

        assert output_with_sink.shape == q.shape
        assert jnp.all(jnp.isfinite(output_with_sink))

        diff = float(jnp.mean(jnp.abs(output_no_sink - output_with_sink).astype(jnp.bfloat16)))
        assert diff > 1e-6, "Attention sink should affect output"

    def test_different_sink_sizes(self):
        """Test different num_sinks values."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        outputs = {}
        for sink_size in [2, 4, 8]:
            outputs[sink_size] = ring_attention(
                q,
                k,
                v,
                sliding_window=64,
                softmax_aux=jnp.full((sink_size,), -2.0, dtype=jnp.float32),
            )

        diff_2_4 = float(jnp.mean(jnp.abs(outputs[2] - outputs[4]).astype(jnp.bfloat16)))
        diff_4_8 = float(jnp.mean(jnp.abs(outputs[4] - outputs[8]).astype(jnp.bfloat16)))

        assert diff_2_4 > 1e-6
        assert diff_4_8 > 1e-6


def _dense_sink_reference(q, k, v, sinks_per_q_head):
    """Dense f32 GQA attention; raw sink logits appended ONCE per row to the softmax denominator."""
    hp = jax.lax.Precision.HIGHEST
    num_q_heads, num_kv_heads = q.shape[2], k.shape[2]
    reps = num_q_heads // num_kv_heads
    kf = jnp.repeat(k.astype(jnp.float32), reps, axis=2)
    vf = jnp.repeat(v.astype(jnp.float32), reps, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32) * q.shape[-1] ** -0.5, kf, precision=hp)
    batch, _, q_len, _ = logits.shape
    aux = jnp.broadcast_to(sinks_per_q_head[None, :, None, :], (batch, num_q_heads, q_len, sinks_per_q_head.shape[-1]))
    weights = jax.nn.softmax(jnp.concatenate([logits, aux], axis=-1), axis=-1)[..., : k.shape[1]]
    return jnp.einsum("bhqk,bkhd->bqhd", weights, vf, precision=hp)


def _rel(a, b):
    a = jnp.asarray(a, jnp.float32)
    b = jnp.asarray(b, jnp.float32)
    return float(jnp.max(jnp.abs(a - b)) / jnp.maximum(jnp.max(jnp.abs(b)), 1e-30))


class TestAttentionSinkParity:
    """Sink layouts vs an independent dense reference; sinks must be counted once, not once per KV chunk."""

    B, S, HQ, HKV, D, NS = 2, 256, 8, 2, 64, 3
    PARITY_REL = 3e-2
    EFFECT_REL = 0.2

    def _inputs(self):
        kq, kk, kv = jax.random.split(jax.random.PRNGKey(5), 3)
        q = jax.random.normal(kq, (self.B, self.S, self.HQ, self.D), dtype=jnp.float32)
        k = jax.random.normal(kk, (self.B, self.S, self.HKV, self.D), dtype=jnp.float32)
        v = jax.random.normal(kv, (self.B, self.S, self.HKV, self.D), dtype=jnp.float32)
        return q, k, v

    def _layouts(self):
        hq, hkv, ns = self.HQ, self.HKV, self.NS
        # Sinks in [3, 8] take a large share next to ~256 keys (logsumexp ~6), so ignoring or
        # double-counting them is far outside the tolerance.
        per_q = jnp.linspace(3.0, 8.0, hq * ns, dtype=jnp.float32).reshape(hq, ns)
        per_kv = jnp.linspace(3.0, 8.0, hkv * ns, dtype=jnp.float32).reshape(hkv, ns)
        shared = jnp.linspace(3.0, 8.0, ns, dtype=jnp.float32)
        return {
            "per_q_head_2d": (per_q, per_q),
            "per_kv_head_2d": (per_kv, jnp.repeat(per_kv, hq // hkv, axis=0)),
            "shared_2d": (shared[None, :], jnp.broadcast_to(shared[None, :], (hq, ns))),
            "shared_1d": (shared, jnp.broadcast_to(shared[None, :], (hq, ns))),
        }

    @pytest.mark.parametrize("layout", ["per_q_head_2d", "per_kv_head_2d", "shared_2d", "shared_1d"])
    @pytest.mark.parametrize("block", [64, 256])
    def test_sink_layout_matches_dense_reference(self, layout, block):
        q, k, v = self._inputs()
        sinks, expanded = self._layouts()[layout]
        out = ring_attention(
            q, k, v, causal=False, softmax_aux=sinks, fwd_params=FwdParams(q_blocksize=block, kv_blocksize=block)
        )
        ref = _dense_sink_reference(q, k, v, expanded)
        nosink = _dense_sink_reference(q, k, v, jnp.full((self.HQ, 1), -1e30, jnp.float32))
        err, effect = _rel(out, ref), _rel(nosink, ref)
        print(f"{layout} block={block}: rel err {err:.3e}; sink effect {effect:.3e}")
        assert effect > self.EFFECT_REL, effect
        assert err < self.PARITY_REL, err


class TestCombinedFeatures:
    """Test combinations of features."""

    def test_window_and_soft_cap(self):
        """Test sliding window with soft cap."""
        key = jax.random.PRNGKey(111)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            logits_soft_cap=30.0,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_asymmetric_window_and_sink(self):
        """Test asymmetric window with softmax_aux."""
        key = jax.random.PRNGKey(222)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        softmax_aux = jnp.full((8,), -2.0, dtype=jnp.float32)
        output = ring_attention(
            q,
            k,
            v,
            sliding_window=(32, 96),
            softmax_aux=softmax_aux,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_all_features_combined(self):
        """Test all features together."""
        key = jax.random.PRNGKey(333)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        softmax_aux = jnp.full((4,), -2.0, dtype=jnp.float32)
        output = ring_attention(
            q,
            k,
            v,
            causal=True,
            sliding_window=(32, 96),
            softmax_aux=softmax_aux,
            logits_soft_cap=30.0,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))


class TestMaskingComparison:
    """Test masking by comparing with vanilla attention."""

    @staticmethod
    def mask_to_segment_ids(attn_mask):
        """Convert attention mask to q_segment_ids and kv_segment_ids.

        Args:
            attn_mask: Boolean mask [batch, 1, seq_len, kv_len] or [batch, num_heads, seq_len, kv_len]
                      True means attend, False means mask out.

        Returns:
            (q_segment_ids, kv_segment_ids) both [batch, seq_len] as int32
        """
        if attn_mask.ndim == 4:
            attn_mask = attn_mask[:, -1, :, :]
        q_valid = jnp.any(attn_mask, axis=-1)
        kv_valid = jnp.any(attn_mask, axis=-2)
        return (
            jnp.where(q_valid, 0, -1).astype(jnp.int32),
            jnp.where(kv_valid, 0, -1).astype(jnp.int32),
        )

    def test_causal_mask_vs_vanilla(self):
        """Compare causal masking with vanilla attention."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))

        vanilla_output, _ = attention(q, k, v, attention_mask=causal_mask)

        ring_output = ring_attention(
            q,
            k,
            v,
            causal=True,
        )

        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output).astype(jnp.bfloat16)))

        print("\nCausal mask comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_output, ring_output, atol=2e-2), (
            f"Ring attention differs from vanilla with causal mask! Max diff: {max_diff}"
        )

    def test_padding_mask_vs_vanilla(self):
        """Compare padding masking with vanilla attention."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        padding_mask = jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool)
        padding_mask = padding_mask.at[0, 0, :, 48:].set(False)
        padding_mask = padding_mask.at[0, 0, 48:, :].set(False)
        padding_mask = padding_mask.at[1, 0, :, 32:].set(False)
        padding_mask = padding_mask.at[1, 0, 32:, :].set(False)

        vanilla_output, _ = attention(q, k, v, attention_mask=padding_mask)

        q_seg_ids, kv_seg_ids = self.mask_to_segment_ids(padding_mask)

        ring_output = ring_attention(
            q,
            k,
            v,
            q_segment_ids=q_seg_ids,
            kv_segment_ids=kv_seg_ids,
        )

        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output).astype(jnp.bfloat16)))

        print("\nPadding mask comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_output, ring_output, atol=0.3), (
            f"Ring attention with segment IDs differs from vanilla with padding mask! Max diff: {max_diff}"
        )

    def test_combined_causal_padding_mask_vs_vanilla(self):
        """Compare combined causal + padding masking with vanilla attention."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        padding_mask = jnp.ones((batch_size, seq_len, seq_len), dtype=bool)
        padding_mask = padding_mask.at[0, :, 48:].set(False)
        padding_mask = padding_mask.at[0, 48:, :].set(False)
        padding_mask = padding_mask.at[1, :, 32:].set(False)
        padding_mask = padding_mask.at[1, 32:, :].set(False)

        combined_mask = causal_mask[None, :, :] & padding_mask
        combined_mask = combined_mask[:, None, :, :]

        vanilla_output, _ = attention(q, k, v, attention_mask=combined_mask)

        q_seg_ids, kv_seg_ids = self.mask_to_segment_ids(combined_mask)

        ring_output = ring_attention(
            q,
            k,
            v,
            q_segment_ids=q_seg_ids,
            kv_segment_ids=kv_seg_ids,
            causal=True,
        )

        valid_mask = q_seg_ids > 0
        valid_mask = valid_mask[:, :, None, None]

        vanilla_valid = jnp.where(valid_mask, vanilla_output, 0.0)
        ring_valid = jnp.where(valid_mask, ring_output, 0.0)
        ring_valid = jnp.where(jnp.isnan(ring_valid), 0.0, ring_valid)

        max_diff = float(jnp.max(jnp.abs(vanilla_valid - ring_valid)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_valid - ring_valid).astype(jnp.bfloat16)))

        print("\nCombined causal + padding mask comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_valid, ring_valid, atol=5e-2), (
            f"Ring attention with segment IDs + causal differs from vanilla with combined mask! Max diff: {max_diff}"
        )

    def test_sliding_window_vs_vanilla(self):
        """Compare sliding window with vanilla attention."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        window_size = 16

        vanilla_output, _ = attention(q, k, v, sliding_window=window_size)

        ring_output = ring_attention(
            q,
            k,
            v,
            sliding_window=window_size,
        )

        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output).astype(jnp.bfloat16)))

        print("\nSliding window comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_output, ring_output, atol=2e-2), (
            f"Ring attention with sliding_window differs from vanilla! Max diff: {max_diff}"
        )

    def test_asymmetric_sliding_window_vs_vanilla(self):
        """Compare asymmetric sliding window with vanilla attention."""
        key = jax.random.PRNGKey(111)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        window_size = (8, 24)

        vanilla_output, _ = attention(q, k, v, sliding_window=window_size)

        ring_output = ring_attention(
            q,
            k,
            v,
            sliding_window=window_size,
        )

        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output).astype(jnp.bfloat16)))

        print("\nAsymmetric sliding window comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_output, ring_output, atol=2e-2), (
            f"Ring attention with asymmetric sliding_window differs from vanilla! Max diff: {max_diff}"
        )

    def test_bias_vs_vanilla(self):
        """Compare bias handling with vanilla attention."""
        key = jax.random.PRNGKey(222)
        keys = jax.random.split(key, 4)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))
        bias = jax.random.normal(keys[3], (batch_size, num_heads, seq_len, seq_len)) * 0.1

        vanilla_output, _ = attention(q, k, v, bias=bias)

        ring_output = ring_attention(
            q,
            k,
            v,
            bias=bias,
        )

        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output).astype(jnp.bfloat16)))

        print("\nBias comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_output, ring_output, atol=2e-2), (
            f"Ring attention with bias differs from vanilla! Max diff: {max_diff}"
        )

    def test_softmax_scale_vs_vanilla(self):
        """Compare custom softmax_scale with vanilla attention."""
        key = jax.random.PRNGKey(333)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        custom_scale = 0.5

        vanilla_output, _ = attention(q, k, v, softmax_scale=custom_scale)

        ring_output = ring_attention(
            q,
            k,
            v,
            softmax_scale=custom_scale,
        )

        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output).astype(jnp.bfloat16)))

        print("\nSoftmax softmax_scale comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_output, ring_output, atol=5e-2), (
            f"Ring attention with softmax_scale differs from vanilla! Max diff: {max_diff}"
        )

    def test_bias_and_sliding_window_vs_vanilla(self):
        """Compare bias + sliding window combination with vanilla attention."""
        key = jax.random.PRNGKey(444)
        keys = jax.random.split(key, 4)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))
        bias = jax.random.normal(keys[3], (batch_size, num_heads, seq_len, seq_len)) * 0.1

        window_size = 16

        vanilla_output, _ = attention(q, k, v, bias=bias, sliding_window=window_size)

        ring_output = ring_attention(
            q,
            k,
            v,
            bias=bias,
            sliding_window=window_size,
        )

        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output).astype(jnp.bfloat16)))

        print("\nBias + sliding window comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(vanilla_output, ring_output, atol=1e-1), (
            f"Ring attention with bias + sliding_window differs from vanilla! Max diff: {max_diff}"
        )

    def test_all_features_combined_vs_vanilla(self):
        """Compare all features combined with vanilla attention."""
        key = jax.random.PRNGKey(555)
        keys = jax.random.split(key, 4)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))
        bias = jax.random.normal(keys[3], (batch_size, num_heads, seq_len, seq_len)) * 0.1

        padding_mask = jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool)
        padding_mask = padding_mask.at[0, 0, :, 48:].set(False)
        padding_mask = padding_mask.at[0, 0, 48:, :].set(False)
        padding_mask = padding_mask.at[1, 0, :, 32:].set(False)
        padding_mask = padding_mask.at[1, 0, 32:, :].set(False)

        window_size = (8, 16)
        custom_scale = 0.5

        vanilla_output, _ = attention(
            q,
            k,
            v,
            attention_mask=padding_mask,
            bias=bias,
            sliding_window=window_size,
            softmax_scale=custom_scale,
        )

        q_seg_ids, kv_seg_ids = self.mask_to_segment_ids(padding_mask)

        ring_output = ring_attention(
            q,
            k,
            v,
            q_segment_ids=q_seg_ids,
            kv_segment_ids=kv_seg_ids,
            bias=bias,
            sliding_window=window_size,
            softmax_scale=custom_scale,
        )

        valid_mask = q_seg_ids > 0
        valid_mask = valid_mask[:, :, None, None]

        vanilla_valid = jnp.where(valid_mask, vanilla_output, 0.0)
        ring_valid = jnp.where(valid_mask, ring_output, 0.0)

        ring_valid = jnp.where(jnp.isnan(ring_valid), 0.0, ring_valid)

        max_diff = float(jnp.max(jnp.abs(vanilla_valid - ring_valid)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_valid - ring_valid).astype(jnp.bfloat16)))

        print("\nAll features combined comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert mean_diff < 0.1, (
            f"Ring attention with all features has high mean difference from vanilla! Mean diff: {mean_diff}"
        )


class TestGradients:
    """Test gradient computation."""

    def test_basic_gradients(self):
        """Test that gradients work for basic case."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            output = ring_attention(
                q,
                k,
                v,
            )
            return jnp.mean(output**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)

    def test_gradients_with_features(self):
        """Test gradients with advanced features."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            softmax_aux = jnp.full((2,), -2.0, dtype=jnp.float32)
            output = ring_attention(
                q,
                k,
                v,
                sliding_window=32,
                logits_soft_cap=30.0,
                softmax_aux=softmax_aux,
            )
            return jnp.mean(output**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)


@pytest.mark.parametrize("layout", ["shared_1d", "one_by_s", "hq_by_s", "hkv_by_s"])
def test_sink_gradient_matches_dense_reference(layout):
    """The XLA ring backward must return the sink gradient, not zero (r11 S03-1/S04-2)."""
    batch, seq, hq, hkv, dim, num_sinks = 2, 32, 4, 2, 16, 3
    kq, kk, kv, kw, ks = jax.random.split(jax.random.PRNGKey(7), 5)
    q = jax.random.normal(kq, (batch, seq, hq, dim), jnp.float32) * 0.5
    k = jax.random.normal(kk, (batch, seq, hkv, dim), jnp.float32) * 0.5
    v = jax.random.normal(kv, (batch, seq, hkv, dim), jnp.float32)
    w = jax.random.normal(kw, (batch, seq, hq, dim), jnp.float32)
    shape = {
        "shared_1d": (num_sinks,),
        "one_by_s": (1, num_sinks),
        "hq_by_s": (hq, num_sinks),
        "hkv_by_s": (hkv, num_sinks),
    }
    # Sinks near the typical logsumexp of a causal row so they carry a large share of the mass.
    sinks = jax.random.normal(ks, shape[layout], jnp.float32) + 1.0
    scale = dim**-0.5

    def per_head_rows(aux):
        if aux.ndim == 1:
            return jnp.broadcast_to(aux[None, :], (hq, aux.shape[0]))
        if aux.shape[0] == 1:
            return jnp.broadcast_to(aux, (hq, aux.shape[1]))
        if aux.shape[0] == hkv:
            return jnp.repeat(aux, hq // hkv, axis=0)
        return aux

    def dense(aux):
        kf = jnp.repeat(k, hq // hkv, axis=2)
        vf = jnp.repeat(v, hq // hkv, axis=2)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, kf, precision="highest") * scale
        causal = jnp.tril(jnp.ones((seq, seq), bool))
        logits = jnp.where(causal[None, None], logits, -jnp.inf)
        rows = per_head_rows(aux)
        sink_cols = jnp.broadcast_to(rows[None, :, None, :], (batch, hq, seq, rows.shape[1]))
        probs = jax.nn.softmax(jnp.concatenate([logits, sink_cols], -1), axis=-1)[..., :seq]
        return jnp.einsum("bhqk,bkhd->bqhd", probs, vf, precision="highest")

    def ring(aux):
        return ring_attention(q, k, v, softmax_aux=aux, causal=True, chunk_size=8)

    ref_grad = jax.grad(lambda a: jnp.sum(dense(a) * w))(sinks)
    got_grad = jax.grad(lambda a: jnp.sum(ring(a) * w))(sinks)
    assert got_grad.shape == sinks.shape
    assert jnp.all(jnp.isfinite(got_grad))
    # Negative control: the true sink gradient is far from zero, so a dropped cotangent fails.
    assert float(jnp.max(jnp.abs(ref_grad))) > 0.5, ref_grad
    err = float(jnp.max(jnp.abs(got_grad - ref_grad)))
    # Default (bf16-pass) matmul precision inside the ring kernel: tolerance relative to magnitude.
    assert err < 3e-2 * float(jnp.max(jnp.abs(ref_grad))) + 1e-2, (err, got_grad, ref_grad)
    # Forward parity as a sanity check on the same layout.
    assert float(jnp.max(jnp.abs(ring(sinks) - dense(sinks)))) < 2e-2


def test_differentiated_causal_ring_attention_stays_causal():
    """Under jax.grad the custom_vjp rules get raw args; causal must still mask (r11 S03-1 follow-up)."""
    batch, seq, heads, dim = 2, 32, 2, 16
    kq, kk, kv, kw = jax.random.split(jax.random.PRNGKey(11), 4)
    q = jax.random.normal(kq, (batch, seq, heads, dim), jnp.float32)
    k = jax.random.normal(kk, (batch, seq, heads, dim), jnp.float32)
    v = jax.random.normal(kv, (batch, seq, heads, dim), jnp.float32)
    w = jax.random.normal(kw, (batch, seq, heads, dim), jnp.float32)
    scale = dim**-0.5

    def dense(q, causal):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision="highest") * scale
        if causal:
            logits = jnp.where(jnp.tril(jnp.ones((seq, seq), bool))[None, None], logits, -jnp.inf)
        return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), v, precision="highest")

    def ring_loss(q):
        return jnp.sum(ring_attention(q, k, v, causal=True, chunk_size=8) * w)

    loss, dq = jax.value_and_grad(ring_loss)(q)
    ref_loss, ref_dq = jax.value_and_grad(lambda q: jnp.sum(dense(q, True) * w))(q)
    _, noncausal_dq = jax.value_and_grad(lambda q: jnp.sum(dense(q, False) * w))(q)
    assert float(jnp.max(jnp.abs(ref_dq - noncausal_dq))) > 0.5  # negative control
    tol = 3e-2 * float(jnp.max(jnp.abs(ref_dq))) + 1e-2
    assert abs(float(loss - ref_loss)) < 3e-2 * abs(float(ref_loss)) + 1e-1
    assert float(jnp.max(jnp.abs(dq - ref_dq))) < tol


def test_disabled_sink_rows_have_zero_not_nan_gradient():
    """All--inf (disabled) sink rows must give a finite, zero sink gradient (r12 S04-1)."""
    batch, seq, heads, dim = 1, 16, 2, 8
    kq, kk, kv, kw = jax.random.split(jax.random.PRNGKey(13), 4)
    q = jax.random.normal(kq, (batch, seq, heads, dim), jnp.float32)
    k = jax.random.normal(kk, (batch, seq, heads, dim), jnp.float32)
    v = jax.random.normal(kv, (batch, seq, heads, dim), jnp.float32)
    w = jax.random.normal(kw, (batch, seq, heads, dim), jnp.float32)

    def grad_for(aux):
        return jax.grad(lambda a: jnp.sum(ring_attention(q, k, v, softmax_aux=a, causal=True, chunk_size=4) * w))(aux)

    # Fully disabled: finite zero gradient, and the output equals the no-sink output.
    disabled = jnp.full((1, 2), -jnp.inf, jnp.float32)
    g = grad_for(disabled)
    assert jnp.all(jnp.isfinite(g)) and float(jnp.max(jnp.abs(g))) == 0.0, g
    out_disabled = ring_attention(q, k, v, softmax_aux=disabled, causal=True, chunk_size=4)
    out_none = ring_attention(q, k, v, causal=True, chunk_size=4)
    assert float(jnp.max(jnp.abs(out_disabled - out_none))) < 1e-5

    # Mixed per-head layout: the disabled head row gets zero gradient, the live row a nonzero one,
    # and the live -inf entry of a mixed row gets zero.
    mixed = jnp.asarray([[-jnp.inf, -jnp.inf], [0.5, -jnp.inf]], jnp.float32)
    g = grad_for(mixed)
    assert jnp.all(jnp.isfinite(g)), g
    assert float(jnp.max(jnp.abs(g[0]))) == 0.0 and float(g[1, 1]) == 0.0, g
    assert abs(float(g[1, 0])) > 1e-3, g


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
