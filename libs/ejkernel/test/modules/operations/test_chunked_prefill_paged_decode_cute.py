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

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from ejkernel.callib._cute_ffi import has_cute_ffi_support
from ejkernel.modules.operations import chunked_prefill_paged_decode

from ._utils import assert_allclose, device_platform, has_cutlass

pytestmark = [
    pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only CUTE validation"),
    pytest.mark.skipif(not has_cutlass(), reason="CUTLASS CuTe DSL not installed"),
    pytest.mark.skipif(not has_cute_ffi_support(), reason="CuTe primitive support unavailable"),
]


def test_chunked_prefill_paged_decode_module_cute_matches_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    num_seqs = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    block_size = 8
    q_lens = [4, 2]

    kv_lens = jnp.array(q_lens, dtype=jnp.int32)
    query_start_loc = jnp.array([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = int(kv_lens.max())
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.float16)
    keys = jax.random.normal(kk, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)
    values = jax.random.normal(kv, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float16)
    key_cache = jnp.zeros((num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float16)
    value_cache = jnp.zeros_like(key_cache)

    out_cute, new_kc_cute, new_vc_cute = chunked_prefill_paged_decode(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        sliding_window=16,
        logits_soft_cap=20.0,
        platform="cute",
    )
    out_xla, new_kc_xla, new_vc_xla = chunked_prefill_paged_decode(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        sliding_window=16,
        logits_soft_cap=20.0,
        platform="xla",
    )

    out_cute, new_kc_cute, new_vc_cute = (
        jax.block_until_ready(out_cute),
        jax.block_until_ready(new_kc_cute),
        jax.block_until_ready(new_vc_cute),
    )
    out_xla, new_kc_xla, new_vc_xla = (
        jax.block_until_ready(out_xla),
        jax.block_until_ready(new_kc_xla),
        jax.block_until_ready(new_vc_xla),
    )

    assert_allclose(out_cute, out_xla, atol=2e-2, rtol=2e-2)
    assert_allclose(new_kc_cute, new_kc_xla, atol=0.0, rtol=0.0)
    assert_allclose(new_vc_cute, new_vc_xla, atol=0.0, rtol=0.0)
