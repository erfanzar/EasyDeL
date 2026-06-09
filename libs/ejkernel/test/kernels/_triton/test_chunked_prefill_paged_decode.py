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

from ejkernel.kernels._triton.chunked_prefill_paged_decode import chunked_prefill_paged_decode
from ejkernel.kernels._xla.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as xla_chunked_prefill_paged_decode,
)

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_chunked_prefill_paged_decode_matches_xla_smoke():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    num_seqs = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 32
    block_size = 8

    q_lens = [4, 1]
    kv_lens = jnp.array(q_lens, dtype=jnp.int32)  # ctx_len=0
    query_start_loc = jnp.array([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = int(kv_lens.max())
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    keys = jax.random.normal(kk, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    values = jax.random.normal(kv, (total_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    key_cache = jnp.zeros((num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value_cache = jnp.zeros_like(key_cache)

    scale = 1.0 / math.sqrt(head_dim)

    out_triton, kc_triton, vc_triton = chunked_prefill_paged_decode(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_scale=scale,
        sliding_window=16,
        logits_soft_cap=20.0,
        seq_threshold_3d=0,
        num_par_softmax_segments=16,
    )
    out_xla, kc_xla, vc_xla = xla_chunked_prefill_paged_decode(
        queries,
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        softmax_scale=scale,
        sliding_window=16,
        logits_soft_cap=20.0,
    )

    out_triton, kc_triton, vc_triton = (
        jax.block_until_ready(out_triton),
        jax.block_until_ready(kc_triton),
        jax.block_until_ready(vc_triton),
    )
    out_xla, kc_xla, vc_xla = (
        jax.block_until_ready(out_xla),
        jax.block_until_ready(kc_xla),
        jax.block_until_ready(vc_xla),
    )

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32), np.asarray(out_xla, dtype=np.float32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        np.asarray(kc_triton, dtype=np.float32), np.asarray(kc_xla, dtype=np.float32), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(vc_triton, dtype=np.float32), np.asarray(vc_xla, dtype=np.float32), rtol=0, atol=0
    )
