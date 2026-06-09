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

"""Tests for Pallas TPU prefill paged attention."""

import os

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.prefill_page_attention import prefill_page_attention as prefill_tpu
from ejkernel.kernels._xla.prefill_page_attention import prefill_page_attention as prefill_xla


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


_RUN = os.getenv("EJKERNEL_RUN_PREFILL_PAGE_ATTENTION") == "1"
pytestmark = pytest.mark.skipif(
    (not _has_tpu()) or (not _RUN),
    reason="Prefill page attention TPU tests are slow to compile; set EJKERNEL_RUN_PREFILL_PAGE_ATTENTION=1 to enable.",
)


def _build_inputs(seed: int):
    key = jax.random.PRNGKey(seed)
    key, kq, kk, kv = jax.random.split(key, 4)

    chunk_size = 16
    page_size = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128

    total_pages = 4
    page_indices = jnp.array([0, 1], dtype=jnp.int32)
    context_len = jnp.array([24], dtype=jnp.int32)

    query = jax.random.normal(kq, (chunk_size, num_q_heads, head_dim), dtype=jnp.float32)
    key_cache = jax.random.normal(kk, (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32)
    value_cache = jax.random.normal(kv, (num_kv_heads, total_pages, page_size, head_dim), dtype=jnp.float32)

    return query, key_cache, value_cache, context_len, page_indices


def test_numerical_correctness_vs_xla():
    query, key_cache, value_cache, context_len, page_indices = _build_inputs(seed=0)

    out_tpu = prefill_tpu(query, key_cache, value_cache, context_len, page_indices)
    out_xla = prefill_xla(query, key_cache, value_cache, context_len, page_indices)

    assert out_tpu.shape == out_xla.shape
    assert jnp.isfinite(out_tpu).all()
    assert jnp.allclose(out_tpu, out_xla, rtol=0.0, atol=0.15)


def test_sliding_window_vs_xla():
    query, key_cache, value_cache, context_len, page_indices = _build_inputs(seed=1)

    out_tpu = prefill_tpu(query, key_cache, value_cache, context_len, page_indices, sliding_window=64)
    out_xla = prefill_xla(query, key_cache, value_cache, context_len, page_indices, sliding_window=64)

    assert jnp.allclose(out_tpu, out_xla, rtol=0.0, atol=0.15)
