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


import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._triton import page_attention
from ejkernel.kernels._xla.page_attention import page_attention as xla_page_attention

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_page_attention_single_pass():
    """Test paged attention with single-pass (no partitioning)."""
    print("=" * 60)
    print("PAGED ATTENTION - SINGLE PASS TEST")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    num_seqs = 2
    num_kv_heads = 4
    query_group_size = 2
    head_size = 64
    kv_blocksize = 16
    num_blocks = 8
    max_context_len = 64

    query = jax.random.normal(keys[0], (num_seqs, num_kv_heads * query_group_size, head_size))
    key_cache = jax.random.normal(keys[1], (num_blocks, num_kv_heads, kv_blocksize, head_size))
    value_cache = jax.random.normal(keys[2], (num_blocks, num_kv_heads, kv_blocksize, head_size))

    context_lens = jnp.array([48, 32], dtype=jnp.int32)

    block_tables = jnp.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=jnp.int32,
    )

    attn_scale = 1.0 / jnp.sqrt(head_size).astype(jnp.float32)

    print("Input shapes:")
    print(f"  query       : {query.shape}")
    print(f"  key_cache   : {key_cache.shape}")
    print(f"  value_cache : {value_cache.shape}")
    print(f"  context_lens: {context_lens.shape}")
    print(f"  block_tables: {block_tables.shape}")
    print(f"  attn_scale  : {attn_scale}")

    print("\nRunning paged attention (single-pass)...")
    output = page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=float(attn_scale),
        max_context_len=max_context_len,
        num_splits=1,
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output mean : {jnp.mean(output):.6f}")
    print(f"Output std  : {jnp.std(output):.6f}")
    print(f"Output range: [{jnp.min(output):.6f}, {jnp.max(output):.6f}]")

    expected = xla_page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=float(attn_scale),
    )
    np.testing.assert_allclose(np.asarray(output), np.asarray(expected), rtol=2e-2, atol=2e-2)
    print("\n✓ Single-pass test passed!")


def test_page_attention_multi_pass():
    """Test paged attention with multi-pass (partitioning)."""
    print("\n" + "=" * 60)
    print("PAGED ATTENTION - MULTI-PASS TEST")
    print("=" * 60)

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 4)

    num_seqs = 2
    num_kv_heads = 4
    query_group_size = 2
    head_size = 128
    kv_blocksize = 16
    num_blocks = 32
    max_num_blocks_per_seq = 16
    max_context_len = 256
    num_splits = 4

    query = jax.random.normal(keys[0], (num_seqs, num_kv_heads * query_group_size, head_size))
    key_cache = jax.random.normal(keys[1], (num_blocks, num_kv_heads, kv_blocksize, head_size))
    value_cache = jax.random.normal(keys[2], (num_blocks, num_kv_heads, kv_blocksize, head_size))

    context_lens = jnp.array([240, 192], dtype=jnp.int32)

    block_tables = jnp.arange(num_seqs * max_num_blocks_per_seq, dtype=jnp.int32).reshape(
        num_seqs, max_num_blocks_per_seq
    )

    attn_scale = 1.0 / jnp.sqrt(head_size).astype(jnp.float32)

    print("Input shapes:")
    print(f"  query       : {query.shape}")
    print(f"  key_cache   : {key_cache.shape}")
    print(f"  value_cache : {value_cache.shape}")
    print(f"  context_lens: {context_lens.shape}")
    print(f"  block_tables: {block_tables.shape}")
    print(f"  num_splits  : {num_splits}")

    print("\nRunning paged attention (multi-pass with partitioning)...")
    output = page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=float(attn_scale),
        max_context_len=max_context_len,
        num_splits=num_splits,
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output mean : {jnp.mean(output):.6f}")
    print(f"Output std  : {jnp.std(output):.6f}")
    print(f"Output range: [{jnp.min(output):.6f}, {jnp.max(output):.6f}]")

    expected = xla_page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=float(attn_scale),
    )
    np.testing.assert_allclose(np.asarray(output), np.asarray(expected), rtol=2e-2, atol=2e-2)
    print("\n✓ Multi-pass test passed!")


def test_page_attention_auto_split():
    """Test paged attention with automatic split determination."""
    print("\n" + "=" * 60)
    print("PAGED ATTENTION - AUTO SPLIT TEST")
    print("=" * 60)

    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 4)

    num_seqs = 4
    num_kv_heads = 8
    query_group_size = 1
    head_size = 64
    kv_blocksize = 32
    num_blocks = 128
    max_num_blocks_per_seq = 32
    max_context_len = 1024

    query = jax.random.normal(keys[0], (num_seqs, num_kv_heads * query_group_size, head_size))
    key_cache = jax.random.normal(keys[1], (num_blocks, num_kv_heads, kv_blocksize, head_size))
    value_cache = jax.random.normal(keys[2], (num_blocks, num_kv_heads, kv_blocksize, head_size))

    context_lens = jnp.array([1024, 768, 512, 256], dtype=jnp.int32)

    block_tables = jnp.arange(num_seqs * max_num_blocks_per_seq, dtype=jnp.int32).reshape(
        num_seqs, max_num_blocks_per_seq
    )

    attn_scale = 1.0 / jnp.sqrt(head_size).astype(jnp.float32)

    print("Input shapes:")
    print(f"  query       : {query.shape}")
    print(f"  key_cache   : {key_cache.shape}")
    print(f"  value_cache : {value_cache.shape}")
    print(f"  context_lens: {context_lens}")
    print(f"  max_context : {max_context_len}")

    print("\nRunning paged attention (auto-split)...")
    output = page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=float(attn_scale),
        max_context_len=max_context_len,
        num_splits=0,
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output mean : {jnp.mean(output):.6f}")
    print(f"Output std  : {jnp.std(output):.6f}")

    expected = xla_page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=float(attn_scale),
    )
    np.testing.assert_allclose(np.asarray(output), np.asarray(expected), rtol=2e-2, atol=2e-2)
    print("\n✓ Auto-split test passed!")


def test_page_attention_matches_xla_basic():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    num_seqs = 2
    num_kv_heads = 2
    q_group = 2
    num_heads = num_kv_heads * q_group
    head_dim = 64
    block_size = 16
    num_blocks = 8

    query = jax.random.normal(kq, (num_seqs, num_heads, head_dim), dtype=jnp.bfloat16)
    key_cache = jax.random.normal(kk, (num_blocks, num_kv_heads, block_size, head_dim), dtype=jnp.bfloat16)
    value_cache = jax.random.normal(kv, (num_blocks, num_kv_heads, block_size, head_dim), dtype=jnp.bfloat16)

    context_lens = jnp.array([24, 16], dtype=jnp.int32)
    block_tables = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

    out_triton = page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=None,
        max_context_len=None,
        num_splits=0,
    )
    out_xla = xla_page_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=None,
        max_context_len=None,
        num_splits=0,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_page_attention_partitioned_matches_single_pass():
    """Single- and multi-partition decoding match XLA on ragged context lengths."""
    key = jax.random.PRNGKey(7)
    kq, kk, kv = jax.random.split(key, 3)

    num_seqs = 2
    num_kv_heads = 2
    q_group = 2
    num_heads = num_kv_heads * q_group
    head_dim = 64
    block_size = 16
    # 16 blocks per sequence (256 tokens) so the 200-token context fits inside its own block-table row.
    num_blocks = 32

    query = jax.random.normal(kq, (num_seqs, num_heads, head_dim), dtype=jnp.float32)
    key_cache = jax.random.normal(kk, (num_blocks, num_kv_heads, block_size, head_dim), dtype=jnp.float32)
    value_cache = jax.random.normal(kv, (num_blocks, num_kv_heads, block_size, head_dim), dtype=jnp.float32)

    # 200 tokens: with num_splits=4 the partition size is next_pow2(50) = 64,
    # so the final partition holds 200 - 3*64 = 8 tokens (a ragged tail).
    context_lens = jnp.array([200, 96], dtype=jnp.int32)
    block_tables = jnp.arange(num_blocks, dtype=jnp.int32).reshape(num_seqs, num_blocks // num_seqs)

    kwargs = dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=None,
        max_context_len=None,
    )

    out_single = jax.block_until_ready(page_attention(**kwargs, num_splits=1))
    out_multi = jax.block_until_ready(page_attention(**kwargs, num_splits=4))
    out_xla = jax.block_until_ready(xla_page_attention(**kwargs))

    for output in (out_single, out_multi):
        np.testing.assert_allclose(
            np.asarray(output, dtype=np.float32),
            np.asarray(out_xla, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
    np.testing.assert_allclose(
        np.asarray(out_single, dtype=np.float32),
        np.asarray(out_multi, dtype=np.float32),
        rtol=1e-4,
        atol=1e-5,
    )


if __name__ == "__main__":
    test_page_attention_single_pass()
    test_page_attention_multi_pass()
    test_page_attention_auto_split()
    test_page_attention_partitioned_matches_single_pass()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
