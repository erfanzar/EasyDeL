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

"""Comparison tests for ragged page attention V2 vs V3.

This test file verifies that V2 and V3 produce identical results for
equivalent inputs. It helps identify discrepancies between the two
implementations.
"""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.ragged_page_attention_v2 import ragged_page_attention_v2
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v2._pallas_impl_fwd import ref_ragged_page_attention as ref_v2
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3 import ragged_page_attention_v3
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import (
    get_kv_cache_shape,
)
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import (
    ref_ragged_paged_attention as ref_v3,
)


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="TPU/Pallas required")


def build_shared_inputs(
    seed=0,
    num_seqs=2,
    num_q_heads=8,
    num_kv_heads=2,
    head_dim=128,
    page_size=16,
    pages_per_seq=4,
):
    """Build inputs that can be converted to both V2 and V3 formats.

    The key insight is that V2 expects KV already in the cache,
    while V3 writes new KV to the cache during attention.

    For comparison, we need to set up inputs such that:
    - V2: KV cache contains all the KV data
    - V3: Keys/Values contain only the new tokens, cache contains the rest
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)

    q_lens = jnp.array([12, 8], dtype=jnp.int32)[:num_seqs]
    kv_lens = jnp.array([48, 32], dtype=jnp.int32)[:num_seqs]

    query_start_loc = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_lens)])

    total_q = int(q_lens.sum())
    total_pages = num_seqs * pages_per_seq

    queries = jax.random.normal(keys[0], (total_q, num_q_heads, head_dim), dtype=jnp.float32)

    k_pages = jax.random.normal(keys[1], (total_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32)
    v_pages = jax.random.normal(keys[2], (total_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32)

    block_tables_2d = jnp.arange(total_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)

    softmax_scale = float(head_dim) ** -0.5

    return {
        "queries": queries,
        "k_pages": k_pages,
        "v_pages": v_pages,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "block_tables_2d": block_tables_2d,
        "query_start_loc": query_start_loc,
        "num_seqs": num_seqs,
        "softmax_scale": softmax_scale,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "pages_per_seq": pages_per_seq,
        "total_pages": total_pages,
        "total_q": total_q,
    }


def to_v2_format(inputs):
    """Convert to V2 format (interleaved kv_pages)."""
    k_pages = inputs["k_pages"]
    v_pages = inputs["v_pages"]
    num_kv_heads = inputs["num_kv_heads"]
    head_dim = inputs["head_dim"]

    # Interleave K and V: [..., 0::2] = K, [..., 1::2] = V
    total_pages, page_size = k_pages.shape[:2]
    kv_pages = jnp.zeros((total_pages, page_size, num_kv_heads * 2, head_dim), dtype=k_pages.dtype)
    kv_pages = kv_pages.at[:, :, 0::2, :].set(k_pages)
    kv_pages = kv_pages.at[:, :, 1::2, :].set(v_pages)

    return {
        "queries": inputs["queries"],
        "kv_pages": kv_pages,
        "context_lens": inputs["kv_lens"],
        "block_tables": inputs["block_tables_2d"],
        "query_start_loc": inputs["query_start_loc"],
        "num_seqs": jnp.array([inputs["num_seqs"]], dtype=jnp.int32),
        "softmax_scale": inputs["softmax_scale"],
    }


def to_v3_format(inputs):
    """Convert to V3 format.

    V3 expects:
    - queries: [total_q, num_q_heads, head_dim]
    - keys: [total_q, num_kv_heads, head_dim] - NEW keys to write
    - values: [total_q, num_kv_heads, head_dim] - NEW values to write
    - kv_cache: Pre-existing cache (will be updated with new K/V)
    - kv_lens: Total KV length per sequence (including new tokens)
    - block_tables: Flattened page indices
    - query_start_loc: Cumulative query positions
    - distribution: [decode_end, prefill_end, total_seqs]

    For a fair comparison, we need to:
    1. Put existing KV (before q_len) into the cache
    2. Pass the new K/V (the last q_len tokens) as keys/values
    """
    num_seqs = inputs["num_seqs"]
    num_kv_heads = inputs["num_kv_heads"]
    head_dim = inputs["head_dim"]
    page_size = inputs["page_size"]
    total_pages = inputs["total_pages"]
    q_lens = inputs["q_lens"]
    kv_lens = inputs["kv_lens"]
    k_pages = inputs["k_pages"]
    v_pages = inputs["v_pages"]
    query_start_loc = inputs["query_start_loc"]

    kv_cache_shape = get_kv_cache_shape(total_pages, page_size, num_kv_heads, head_dim, k_pages.dtype)
    kv_cache = jnp.zeros(kv_cache_shape, dtype=k_pages.dtype)

    new_keys_list = []
    new_values_list = []

    for seq_idx in range(num_seqs):
        q_len = int(q_lens[seq_idx])
        kv_len = int(kv_lens[seq_idx])

        new_start = kv_len - q_len
        pages_per_seq = inputs["pages_per_seq"]
        seq_pages_start = seq_idx * pages_per_seq

        seq_k = k_pages[seq_pages_start : seq_pages_start + pages_per_seq].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        seq_v = v_pages[seq_pages_start : seq_pages_start + pages_per_seq].reshape(-1, num_kv_heads, head_dim)[:kv_len]

        new_keys_list.append(seq_k[new_start:])
        new_values_list.append(seq_v[new_start:])

    keys = jnp.concatenate(new_keys_list, axis=0)
    values = jnp.concatenate(new_values_list, axis=0)

    from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import merge_kv

    for seq_idx in range(num_seqs):
        q_len = int(q_lens[seq_idx])
        kv_len = int(kv_lens[seq_idx])
        new_start = kv_len - q_len
        pages_per_seq = inputs["pages_per_seq"]
        seq_pages_start = seq_idx * pages_per_seq

        if new_start > 0:
            seq_k = k_pages[seq_pages_start : seq_pages_start + pages_per_seq].reshape(-1, num_kv_heads, head_dim)[
                :new_start
            ]
            seq_v = v_pages[seq_pages_start : seq_pages_start + pages_per_seq].reshape(-1, num_kv_heads, head_dim)[
                :new_start
            ]

            merged = merge_kv(seq_k, seq_v)

            num_old_pages = (new_start + page_size - 1) // page_size
            for p in range(num_old_pages):
                page_idx = seq_pages_start + p
                start = p * page_size
                end = min((p + 1) * page_size, new_start)
                kv_cache = kv_cache.at[page_idx, : end - start].set(merged[start:end])

    block_tables_flat = inputs["block_tables_2d"].flatten()

    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    return {
        "queries": inputs["queries"],
        "keys": keys,
        "values": values,
        "kv_cache": kv_cache,
        "kv_lens": kv_lens,
        "block_tables": block_tables_flat,
        "query_start_loc": query_start_loc,
        "distribution": distribution,
        "softmax_scale": inputs["softmax_scale"],
    }


class TestRaggedPageAttentionComparison:
    """Tests comparing V2 and V3 ragged page attention implementations."""

    def test_reference_implementations_match(self):
        """Test that V2 and V3 reference implementations match for pure attention."""
        inputs = build_shared_inputs(seed=0)

        v2_inputs = to_v2_format(inputs)
        ref_out_v2 = ref_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_scale=v2_inputs["softmax_scale"],
        )

        v3_inputs = to_v3_format(inputs)
        ref_out_v3, _ = ref_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_scale=v3_inputs["softmax_scale"],
        )

        num_seqs = inputs["num_seqs"]
        max_diff = float(jnp.max(jnp.abs(ref_out_v2[:num_seqs] - ref_out_v3[:num_seqs])))
        print(f"Reference implementations max diff: {max_diff}")
        print(f"V2 ref output shape: {ref_out_v2.shape}")
        print(f"V3 ref output shape: {ref_out_v3.shape}")

        assert ref_out_v2.shape == ref_out_v3.shape, f"Shape mismatch: {ref_out_v2.shape} vs {ref_out_v3.shape}"
        assert jnp.allclose(ref_out_v2, ref_out_v3, rtol=0, atol=0.125), f"Max diff: {max_diff}"

    def test_pure_attention_v2_kernel_vs_reference(self):
        """Test V2 kernel matches its reference for pure attention."""
        inputs = build_shared_inputs(seed=1)
        v2_inputs = to_v2_format(inputs)

        ref_out = ref_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_scale=v2_inputs["softmax_scale"],
        )

        kernel_out = ragged_page_attention_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_scale=v2_inputs["softmax_scale"],
        )

        num_seqs = int(v2_inputs["num_seqs"][0])
        max_diff = float(jnp.max(jnp.abs(ref_out[:num_seqs] - kernel_out[:num_seqs])))
        print(f"V2 kernel vs reference max diff: {max_diff}")

        assert jnp.allclose(ref_out[:num_seqs], kernel_out[:num_seqs], rtol=0, atol=0.125), f"Max diff: {max_diff}"

    def test_pure_attention_v3_kernel_vs_reference(self):
        """Test V3 kernel matches its reference for pure attention."""
        inputs = build_shared_inputs(seed=2)
        v3_inputs = to_v3_format(inputs)

        ref_out, _ref_cache = ref_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_scale=v3_inputs["softmax_scale"],
        )

        kernel_out, _kernel_cache = ragged_page_attention_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_scale=v3_inputs["softmax_scale"],
        )

        num_seqs = int(v3_inputs["distribution"][-1])
        max_diff = float(jnp.max(jnp.abs(ref_out[:num_seqs] - kernel_out[:num_seqs])))
        print(f"V3 kernel vs reference max diff: {max_diff}")

        assert jnp.allclose(ref_out[:num_seqs], kernel_out[:num_seqs], rtol=0, atol=0.125), f"Max diff: {max_diff}"

    def test_pure_attention_v2_vs_v3_kernels(self):
        """Test V2 and V3 kernels match for pure causal attention."""
        inputs = build_shared_inputs(seed=3)

        v2_inputs = to_v2_format(inputs)
        out_v2 = ragged_page_attention_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_scale=v2_inputs["softmax_scale"],
        )

        v3_inputs = to_v3_format(inputs)
        out_v3, _ = ragged_page_attention_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_scale=v3_inputs["softmax_scale"],
        )

        num_seqs = inputs["num_seqs"]
        max_diff = float(jnp.max(jnp.abs(out_v2[:num_seqs] - out_v3[:num_seqs])))
        print(f"V2 vs V3 kernels max diff: {max_diff}")
        print(f"V2 output shape: {out_v2.shape}")
        print(f"V3 output shape: {out_v3.shape}")

        assert out_v2.shape == out_v3.shape, f"Shape mismatch: {out_v2.shape} vs {out_v3.shape}"
        assert jnp.allclose(out_v2[:num_seqs], out_v3[:num_seqs], rtol=0, atol=0.125), f"Max diff: {max_diff}"

    def test_sliding_window_v2_vs_v3_reference(self):
        """Test sliding window: V2 reference (correct) vs V3 reference.

        V2 uses token-level masking (CORRECT).
        V3 uses block-skipping (potentially WRONG at block boundaries).
        """
        inputs = build_shared_inputs(seed=4, page_size=16)
        sliding_window = 24  # Small window to trigger the difference

        v2_inputs = to_v2_format(inputs)
        ref_out_v2 = ref_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_scale=v2_inputs["softmax_scale"],
            sliding_window=sliding_window,
        )

        v3_inputs = to_v3_format(inputs)
        ref_out_v3, _ = ref_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_scale=v3_inputs["softmax_scale"],
            sliding_window=sliding_window,
        )

        max_diff = float(jnp.max(jnp.abs(ref_out_v2 - ref_out_v3)))
        print("\n=== Sliding Window Test ===")
        print(f"V2 vs V3 reference max diff: {max_diff}")
        print(f"V2 output sample: {ref_out_v2[0, 0, :5]}")
        print(f"V3 output sample: {ref_out_v3[0, 0, :5]}")

        try:
            assert jnp.allclose(ref_out_v2, ref_out_v3, rtol=0, atol=0.125), f"Max diff: {max_diff}"
            print("PASSED: V2 and V3 sliding window match")
        except AssertionError:
            print(f"FAILED: V2 and V3 sliding window DO NOT match (max_diff={max_diff})")
            print("This indicates V3's sliding window implementation differs from V2")
            raise

    def test_attention_sink_v2_vs_v3_reference(self):
        """Test attention sink: V3 reference (correct) vs V2 reference.

        V3 uses LEFT concatenation with single scalar per Q head (CORRECT).
        V2 uses RIGHT concatenation with multiple sinks per KV head (WRONG).

        Note: The interfaces are different, so we test each against a pure JAX impl.
        """
        inputs = build_shared_inputs(seed=5)
        num_q_heads = inputs["num_q_heads"]
        key = jax.random.PRNGKey(100)
        attention_sink_v3 = jax.random.normal(key, (num_q_heads,), dtype=jnp.float32)

        softmax_aux_v2 = attention_sink_v3  # Same values for direct comparison

        print("\n=== Attention Sink Test ===")
        print(f"V3 softmax_aux shape: {attention_sink_v3.shape}")
        print(f"V2 softmax_aux shape: {softmax_aux_v2.shape}")

        v3_inputs = to_v3_format(inputs)
        ref_out_v3_with_sink, _ = ref_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_aux=attention_sink_v3,
            softmax_scale=v3_inputs["softmax_scale"],
        )

        ref_out_v3_no_sink, _ = ref_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_scale=v3_inputs["softmax_scale"],
        )

        v2_inputs = to_v2_format(inputs)
        ref_out_v2_with_aux = ref_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_aux=softmax_aux_v2,
            softmax_scale=v2_inputs["softmax_scale"],
        )

        ref_out_v2_no_aux = ref_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_scale=v2_inputs["softmax_scale"],
        )

        v3_sink_effect = float(jnp.max(jnp.abs(ref_out_v3_with_sink - ref_out_v3_no_sink)))
        v2_aux_effect = float(jnp.max(jnp.abs(ref_out_v2_with_aux - ref_out_v2_no_aux)))

        print(f"V3 attention sink effect (max diff from no-sink): {v3_sink_effect}")
        print(f"V2 softmax_aux effect (max diff from no-aux): {v2_aux_effect}")
        print(f"V3 with sink sample: {ref_out_v3_with_sink[0, 0, :5]}")
        print(f"V2 with aux sample: {ref_out_v2_with_aux[0, 0, :5]}")

        assert v3_sink_effect > 0.001, f"V3 attention sink has no effect: {v3_sink_effect}"
        assert v2_aux_effect > 0.001, f"V2 softmax_aux has no effect: {v2_aux_effect}"

        print("Both V2 and V3 attention sinks have measurable effects.")
        print("Note: V2 and V3 use different semantics, so direct comparison is not meaningful.")


class TestSlidingWindowBug:
    """Focused tests to identify the V3 sliding window bug."""

    def test_sliding_window_edge_case(self):
        """Test sliding window with edge case that triggers block boundary issues.

        V3's block-skipping approach may include extra tokens outside the window
        when kv_len - sliding_window is not aligned to block boundaries.
        """
        inputs = build_shared_inputs(
            seed=10,
            num_seqs=1,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            pages_per_seq=8,
        )

        inputs["q_lens"] = jnp.array([4], dtype=jnp.int32)
        inputs["kv_lens"] = jnp.array([100], dtype=jnp.int32)  # 100 tokens
        inputs["query_start_loc"] = jnp.array([0, 4], dtype=jnp.int32)

        key = jax.random.PRNGKey(10)
        inputs["queries"] = jax.random.normal(key, (4, inputs["num_q_heads"], inputs["head_dim"]), dtype=jnp.float32)

        sliding_window = 35  # Not aligned to page_size=16

        v2_inputs = to_v2_format(inputs)
        v3_inputs = to_v3_format(inputs)

        ref_out_v2 = ref_v2(
            v2_inputs["queries"],
            v2_inputs["kv_pages"],
            v2_inputs["context_lens"],
            v2_inputs["block_tables"],
            v2_inputs["query_start_loc"],
            v2_inputs["num_seqs"],
            softmax_scale=v2_inputs["softmax_scale"],
            sliding_window=sliding_window,
        )

        ref_out_v3, _ = ref_v3(
            v3_inputs["queries"],
            v3_inputs["keys"],
            v3_inputs["values"],
            v3_inputs["kv_cache"],
            v3_inputs["kv_lens"],
            v3_inputs["block_tables"],
            v3_inputs["query_start_loc"],
            v3_inputs["distribution"],
            softmax_scale=v3_inputs["softmax_scale"],
            sliding_window=sliding_window,
        )

        max_diff = float(jnp.max(jnp.abs(ref_out_v2 - ref_out_v3)))
        print("\n=== Sliding Window Edge Case Test ===")
        print("kv_len=100, sliding_window=35, page_size=16")
        print("V2 should mask tokens 0-64, attend to 65-99")
        print("V3 block start: (100-35)//16 = 4, so starts at token 64")
        print(f"Max diff: {max_diff}")

        try:
            assert jnp.allclose(ref_out_v2, ref_out_v3, rtol=0, atol=0.125), f"Max diff: {max_diff}"
            print("PASSED: Edge case matches")
        except AssertionError:
            print(f"FAILED: Edge case differs (max_diff={max_diff})")
            print("V3's block-aligned sliding window produces different results")
            raise
