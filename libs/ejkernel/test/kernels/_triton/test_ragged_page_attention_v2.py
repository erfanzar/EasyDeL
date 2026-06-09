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


"""Unified tests for ragged_page_attention_v2 (Triton implementation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._triton.ragged_page_attention_v2 import ragged_page_attention_v2
from ejkernel.kernels._xla.ragged_page_attention_v2 import ragged_page_attention_v2 as xla_ragged_page_attention_v2

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def create_test_data(
    num_seqs=2,
    max_seq_len=128,
    num_q_heads=8,
    num_kv_heads=4,
    head_size=64,
    page_size=16,
    seed=42,
):
    """Helper function to create test data for ragged page attention."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)

    seq_lens = jax.random.randint(keys[0], (num_seqs,), 32, max_seq_len + 1)
    total_query_tokens = int(jnp.sum(seq_lens))

    queries = jax.random.normal(keys[1], (total_query_tokens, num_q_heads, head_size))

    max_pages = (int(jnp.max(seq_lens)) + page_size - 1) // page_size
    num_pages = num_seqs * max_pages

    kv_pages = jax.random.normal(keys[2], (num_pages, page_size, 2 * num_kv_heads, head_size))

    context_lens = seq_lens

    block_tables = jnp.arange(num_seqs * max_pages).reshape(num_seqs, max_pages)

    query_start_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(seq_lens)])

    num_seqs_val = num_seqs

    return {
        "queries": queries,
        "kv_pages": kv_pages,
        "context_lens": context_lens,
        "block_tables": block_tables,
        "query_start_loc": query_start_loc,
        "num_seqs": num_seqs_val,
        "seq_lens": seq_lens,
        "page_size": page_size,
    }


class TestBasicFunctionality:
    """Test basic ragged page attention functionality."""

    def test_basic_forward_pass(self):
        """Test that basic forward pass works without any features."""
        data = create_test_data(num_seqs=3, max_seq_len=64)
        output = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
        )
        assert output.shape == data["queries"].shape
        assert jnp.all(jnp.isfinite(output))

    def test_output_dtype_matches_input(self):
        """Test that output dtype matches input queries dtype."""
        data = create_test_data(num_seqs=2, max_seq_len=64)
        for dtype in [jnp.float32, jnp.bfloat16]:
            queries_typed = data["queries"].astype(dtype)
            kv_pages_typed = data["kv_pages"].astype(dtype)
            output = ragged_page_attention_v2(
                queries=queries_typed,
                kv_pages=kv_pages_typed,
                context_lens=data["context_lens"],
                block_tables=data["block_tables"],
                query_start_loc=data["query_start_loc"],
                num_seqs=data["num_seqs"],
            )
            assert output.dtype == dtype

    def test_different_head_configurations(self):
        """Test different head configurations (GQA)."""
        for num_q_heads, num_kv_heads in [(8, 8), (8, 4), (16, 4)]:
            data = create_test_data(
                num_seqs=2,
                max_seq_len=64,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
            )
            output = ragged_page_attention_v2(
                queries=data["queries"],
                kv_pages=data["kv_pages"],
                context_lens=data["context_lens"],
                block_tables=data["block_tables"],
                query_start_loc=data["query_start_loc"],
                num_seqs=data["num_seqs"],
            )
            expected_shape = (data["queries"].shape[0], num_q_heads, data["queries"].shape[2])
            assert output.shape == expected_shape

    def test_variable_sequence_lengths(self):
        """Test with highly variable sequence lengths."""
        num_seqs = 4
        seq_lens = jnp.array([16, 64, 128, 256], dtype=jnp.int32)
        total_tokens = int(jnp.sum(seq_lens))
        num_q_heads, num_kv_heads, head_size, page_size = 8, 4, 64, 16

        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 5)

        queries = jax.random.normal(keys[0], (total_tokens, num_q_heads, head_size))
        max_pages = (int(jnp.max(seq_lens)) + page_size - 1) // page_size
        num_pages = num_seqs * max_pages
        kv_pages = jax.random.normal(keys[1], (num_pages, page_size, 2 * num_kv_heads, head_size))
        context_lens = seq_lens
        block_tables = jnp.arange(num_pages).reshape(num_seqs, max_pages)
        query_start_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(seq_lens)])
        num_seqs_val = num_seqs

        output = ragged_page_attention_v2(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs_val,
        )

        for i in range(num_seqs):
            start = int(query_start_loc[i])
            end = int(query_start_loc[i + 1])
            seq_output = output[start:end]
            assert jnp.all(jnp.isfinite(seq_output))


class TestSoftCap:
    """Test soft cap feature."""

    def test_soft_cap_affects_output(self):
        """Test that soft cap actually changes the output."""
        data = create_test_data(num_seqs=2, max_seq_len=128, seed=42)

        output_no_cap = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
        )

        output_with_cap = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=30.0,
        )

        diff = float(jnp.mean(jnp.abs(output_no_cap - output_with_cap)))
        assert diff > 1e-6, f"Soft cap should affect output, got diff={diff}"
        assert jnp.all(jnp.isfinite(output_with_cap))

    def test_soft_cap_values(self):
        """Test different soft cap values produce different results."""
        data = create_test_data(num_seqs=2, max_seq_len=64, seed=123)

        output_cap_10 = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=10.0,
        )

        output_cap_50 = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=50.0,
        )

        diff = float(jnp.mean(jnp.abs(output_cap_10 - output_cap_50)))
        assert diff > 1e-6, "Different soft cap values should produce different outputs"

    def test_soft_cap_numerical_stability(self):
        """Test that soft cap doesn't cause numerical instabilities."""
        data = create_test_data(num_seqs=2, max_seq_len=128, seed=999)

        large_queries = data["queries"] * 10.0

        output = ragged_page_attention_v2(
            queries=large_queries,
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=30.0,
        )

        assert jnp.all(jnp.isfinite(output)), "Soft cap should prevent overflow/underflow"


class TestSlidingWindow:
    """Test sliding window feature."""

    def test_sliding_window_affects_output(self):
        """Test that sliding window changes the output."""
        data = create_test_data(num_seqs=2, max_seq_len=256, seed=42)

        output_full = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
        )

        output_windowed = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            sliding_window=64,
        )

        diff = float(jnp.mean(jnp.abs(output_full - output_windowed)))
        assert diff > 1e-6, f"Sliding window should affect output, got diff={diff}"
        assert jnp.all(jnp.isfinite(output_windowed))

    def test_sliding_window_sizes(self):
        """Test different sliding window sizes."""
        data = create_test_data(num_seqs=2, max_seq_len=256, seed=456)

        outputs = {}
        for window_size in [32, 64, 128]:
            outputs[window_size] = ragged_page_attention_v2(
                queries=data["queries"],
                kv_pages=data["kv_pages"],
                context_lens=data["context_lens"],
                block_tables=data["block_tables"],
                query_start_loc=data["query_start_loc"],
                num_seqs=data["num_seqs"],
                sliding_window=window_size,
            )

        diff_32_64 = float(jnp.mean(jnp.abs(outputs[32] - outputs[64])))
        diff_64_128 = float(jnp.mean(jnp.abs(outputs[64] - outputs[128])))

        assert diff_32_64 > 1e-6, "Window size 32 vs 64 should differ"
        assert diff_64_128 > 1e-6, "Window size 64 vs 128 should differ"


class TestMaskValue:
    """Test custom mask value feature."""

    def test_custom_mask_value(self):
        """Test that custom mask value works."""
        data = create_test_data(num_seqs=2, max_seq_len=128, seed=42)

        output_default = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
        )

        output_custom = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            mask_value=-1e10,
        )

        assert jnp.all(jnp.isfinite(output_default))
        assert jnp.all(jnp.isfinite(output_custom))


class TestCombinedFeatures:
    """Test combinations of features."""

    def test_soft_cap_and_sliding_window(self):
        """Test soft cap and sliding window together."""
        data = create_test_data(num_seqs=2, max_seq_len=256, seed=42)

        output = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=30.0,
            sliding_window=128,
        )

        assert output.shape == data["queries"].shape
        assert jnp.all(jnp.isfinite(output))

    def test_all_features_combined(self):
        """Test all features together."""
        data = create_test_data(num_seqs=3, max_seq_len=256, seed=123)

        output = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=30.0,
            sliding_window=128,
            mask_value=-1e10,
            softmax_scale=0.125,
        )

        assert output.shape == data["queries"].shape
        assert jnp.all(jnp.isfinite(output))

    def test_feature_combinations_produce_different_results(self):
        """Test that different feature combinations produce different results."""
        data = create_test_data(num_seqs=2, max_seq_len=128, seed=456)

        output_baseline = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
        )

        output_cap = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=30.0,
        )

        output_window = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            sliding_window=64,
        )

        output_both = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=30.0,
            sliding_window=64,
        )

        diff_base_cap = float(jnp.mean(jnp.abs(output_baseline - output_cap)))
        diff_base_window = float(jnp.mean(jnp.abs(output_baseline - output_window)))
        diff_cap_both = float(jnp.mean(jnp.abs(output_cap - output_both)))
        diff_window_both = float(jnp.mean(jnp.abs(output_window - output_both)))

        assert diff_base_cap > 1e-6
        assert diff_base_window > 1e-6
        assert diff_cap_both > 1e-6
        assert diff_window_both > 1e-6


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sequence(self):
        """Test with a single sequence."""
        data = create_test_data(num_seqs=1, max_seq_len=128, seed=42)

        output = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=30.0,
            sliding_window=64,
        )

        assert output.shape == data["queries"].shape
        assert jnp.all(jnp.isfinite(output))

    def test_very_large_soft_cap(self):
        """Test with very large soft cap value (should behave like no cap)."""
        data = create_test_data(num_seqs=2, max_seq_len=64, seed=456)

        output_no_cap = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
        )

        output_large_cap = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            logits_soft_cap=1e6,
        )

        diff = float(jnp.max(jnp.abs(output_no_cap - output_large_cap)))
        assert diff < 0.1, "Very large soft cap should behave similar to no cap"

    def test_window_larger_than_sequence(self):
        """Test sliding window larger than sequence length."""
        data = create_test_data(num_seqs=2, max_seq_len=64, seed=789)

        output_no_window = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
        )

        output_large_window = ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            sliding_window=1000,
        )

        diff = float(jnp.max(jnp.abs(output_no_window - output_large_window)))
        assert diff < 1e-5, "Window larger than sequence should match full attention"


def test_ragged_page_attention_v2_matches_xla():
    data = create_test_data(
        num_seqs=2,
        max_seq_len=64,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=32,
        page_size=16,
        seed=0,
    )

    softmax_scale = float(data["queries"].shape[-1]) ** -0.5

    out_triton = ragged_page_attention_v2(
        queries=data["queries"],
        kv_pages=data["kv_pages"],
        context_lens=data["context_lens"],
        block_tables=data["block_tables"],
        query_start_loc=data["query_start_loc"],
        num_seqs=data["num_seqs"],
        softmax_scale=softmax_scale,
        logits_soft_cap=None,
        sliding_window=None,
        optimized=False,
    )
    out_xla = xla_ragged_page_attention_v2(
        queries=data["queries"],
        kv_pages=data["kv_pages"],
        context_lens=data["context_lens"],
        block_tables=data["block_tables"],
        query_start_loc=data["query_start_loc"],
        num_seqs=data["num_seqs"],
        softmax_scale=softmax_scale,
        logits_soft_cap=None,
        sliding_window=None,
        softmax_aux=None,
        compute_dtype=jnp.float32,
        optimized=False,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
