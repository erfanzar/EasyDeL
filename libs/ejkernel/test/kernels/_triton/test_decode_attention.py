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

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._triton import decode_attention
from ejkernel.kernels._xla.decode_attention import decode_attention as xla_decode_attention

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_decode_attention_matches_xla_smoke():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch = 4
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    page_size = 8
    max_pages = 4
    num_kv_splits = 4

    total_pages = batch * max_pages
    key_buffer = jax.random.normal(kk, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_buffer = jax.random.normal(kv, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    query = jax.random.normal(kq, (batch, num_q_heads, head_dim), dtype=jnp.bfloat16)

    req_to_tokens = jnp.arange(total_pages, dtype=jnp.int32).reshape(batch, max_pages)
    seq_lens = jnp.array([3, 17, 23, 29], dtype=jnp.int32)

    scale = 1.0 / math.sqrt(head_dim)

    out_triton, lse_triton = decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        logits_soft_cap=20.0,
    )
    out_xla, lse_xla = xla_decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        logits_soft_cap=20.0,
    )

    out_triton, lse_triton = jax.block_until_ready(out_triton), jax.block_until_ready(lse_triton)
    out_xla, lse_xla = jax.block_until_ready(out_xla), jax.block_until_ready(lse_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(lse_triton, dtype=np.float32),
        np.asarray(lse_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("logits_soft_cap", [None, 25.0])
@pytest.mark.parametrize("num_kv_splits,block_n", [(1, 32), (3, 64), (8, 128)])
def test_decode_attention_variants_match_xla(page_size, logits_soft_cap, num_kv_splits, block_n):
    """Test page_size > 1, logits_soft_cap, configurable block_n, GQA=8, and non-pow2 splits."""
    key = jax.random.PRNGKey(10 + page_size + num_kv_splits)
    kq, kk, kv = jax.random.split(key, 3)

    batch = 3
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 64
    max_pages = max(4, 160 // page_size)
    total_pages = batch * max_pages

    key_buffer = jax.random.normal(kk, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.float32)
    value_buffer = jax.random.normal(kv, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.float32)
    query = jax.random.normal(kq, (batch, num_q_heads, head_dim), dtype=jnp.float32)
    if logits_soft_cap is not None:
        query = query * 40.0  # exercise large |logits / cap| stability

    req_to_tokens = jnp.arange(total_pages, dtype=jnp.int32).reshape(batch, max_pages)
    max_len = max_pages * page_size
    seq_lens = jnp.array([min(1, max_len), min(37, max_len), min(max_len - 1, 129)], dtype=jnp.int32)
    scale = 1.0 / math.sqrt(head_dim)

    out_tri, lse_tri = decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        logits_soft_cap=logits_soft_cap,
        block_n=block_n,
    )
    out_xla, lse_xla = xla_decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        logits_soft_cap=logits_soft_cap,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(out_xla), dtype=np.float32),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(lse_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(lse_xla), dtype=np.float32),
        rtol=2e-3,
        atol=2e-3,
    )


def test_decode_attention_jit_matches_xla():
    """Verify jax.jit tracing and parity vs XLA with page_size > 1 and logits_soft_cap."""
    key = jax.random.PRNGKey(99)
    kq, kk, kv = jax.random.split(key, 3)

    batch, num_q_heads, num_kv_heads, head_dim = 4, 8, 2, 128
    page_size, max_pages = 16, 6
    total_pages = batch * max_pages

    key_buffer = jax.random.normal(kk, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_buffer = jax.random.normal(kv, (total_pages * page_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    query = jax.random.normal(kq, (batch, num_q_heads, head_dim), dtype=jnp.bfloat16)
    req_to_tokens = jnp.arange(total_pages, dtype=jnp.int32).reshape(batch, max_pages)
    seq_lens = jnp.array([15, 32, 77, 96], dtype=jnp.int32)
    scale = 1.0 / math.sqrt(head_dim)

    @jax.jit
    def run_tri(q, kb, vb, rt, sl):
        return decode_attention(
            q,
            kb,
            vb,
            rt,
            sl,
            softmax_scale=scale,
            num_kv_splits=4,
            page_size=page_size,
            logits_soft_cap=30.0,
            block_n=64,
        )

    out_tri, lse_tri = run_tri(query, key_buffer, value_buffer, req_to_tokens, seq_lens)
    out_xla, lse_xla = xla_decode_attention(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=4,
        page_size=page_size,
        logits_soft_cap=30.0,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(out_xla), dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(lse_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(lse_xla), dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
