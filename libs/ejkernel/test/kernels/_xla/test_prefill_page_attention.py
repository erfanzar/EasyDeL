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

"""Tests for XLA prefill paged attention reference implementation."""

import os

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.prefill_page_attention._pallas_impl_fwd import ref_prefill_page_attention
from ejkernel.kernels._xla.prefill_page_attention import prefill_page_attention


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


_RUN = os.getenv("EJKERNEL_RUN_PREFILL_PAGE_ATTENTION") == "1"
pytestmark = pytest.mark.skipif(
    (not _has_tpu()) or (not _RUN),
    reason="Prefill page attention tests are slow to compile on TPU; "
    "set EJKERNEL_RUN_PREFILL_PAGE_ATTENTION=1 to enable.",
)


def _build_inputs(seed: int):
    key = jax.random.PRNGKey(seed)
    key, kq, kk, kv = jax.random.split(key, 4)

    chunk_size = 8
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    total_pages = 4

    query = jax.random.normal(kq, (chunk_size, num_q_heads, head_dim), dtype=jnp.float32)
    key_cache = jax.random.normal(kk, (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32)
    value_cache = jax.random.normal(kv, (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32)

    page_indices = jnp.array([1, 3], dtype=jnp.int32)
    context_len = jnp.array([12], dtype=jnp.int32)  # <= 2 * page_size

    return query, key_cache, value_cache, context_len, page_indices


@pytest.mark.parametrize("use_jit", [False, True])
def test_matches_reference(use_jit):
    query, key_cache, value_cache, context_len, page_indices = _build_inputs(seed=0)

    fn = jax.jit(prefill_page_attention) if use_jit else prefill_page_attention
    out = fn(query, key_cache, value_cache, context_len, page_indices)

    ref = ref_prefill_page_attention(query, key_cache, value_cache, context_len, page_indices)

    assert out.shape == ref.shape
    assert jnp.isfinite(out).all()
    assert jnp.allclose(out, ref, rtol=2e-4, atol=2e-4)


def test_sliding_window_and_soft_cap():
    query, key_cache, value_cache, context_len, page_indices = _build_inputs(seed=1)

    out = prefill_page_attention(
        query,
        key_cache,
        value_cache,
        context_len,
        page_indices,
        sliding_window=6,
        attn_logits_soft_cap=20.0,
    )
    ref = ref_prefill_page_attention(
        query,
        key_cache,
        value_cache,
        context_len,
        page_indices,
        sliding_window=6,
        attn_logits_soft_cap=20.0,
    )

    assert jnp.allclose(out, ref, rtol=3e-4, atol=3e-4)
