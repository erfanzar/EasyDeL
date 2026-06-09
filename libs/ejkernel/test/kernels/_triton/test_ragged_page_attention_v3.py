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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._triton.ragged_page_attention_v3._interface import (
    ragged_page_attention_v3 as triton_ragged_page_attention_v3,
)
from ejkernel.kernels._xla.ragged_page_attention_v3._interface import (
    ragged_page_attention_v3 as xla_ragged_page_attention_v3,
)

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def _kv_cache_shape(
    *, num_pages: int, page_size: int, num_kv_heads: int, head_dim_padded: int, dtype: jnp.dtype
) -> tuple[int, ...]:
    pack = 2 if jnp.dtype(dtype).itemsize == 2 else 1
    if pack == 1:
        return (num_pages, page_size, num_kv_heads * 2, 1, head_dim_padded)
    return (num_pages, page_size, num_kv_heads, 2, head_dim_padded)


def test_ragged_page_attention_v3_matches_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv, kc = jax.random.split(key, 4)

    num_seqs = 2
    max_num_seqs = 2
    pages_per_seq = 4
    page_size = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    head_dim_padded = 128
    dtype = jnp.float16

    kv_lens = jnp.array([40, 32], dtype=jnp.int32)
    q_lens = jnp.array([8, 4], dtype=jnp.int32)

    total_q = int(jnp.sum(q_lens))
    query_start_loc = jnp.pad(jnp.cumsum(q_lens), (1, 0)).astype(jnp.int32)

    queries = jax.random.normal(kq, (total_q, num_q_heads, head_dim), dtype=dtype)
    keys = jax.random.normal(kk, (total_q, num_kv_heads, head_dim), dtype=dtype)
    values = jax.random.normal(kv, (total_q, num_kv_heads, head_dim), dtype=dtype)

    total_pages = max_num_seqs * pages_per_seq
    kv_cache = jax.random.normal(
        kc,
        _kv_cache_shape(
            num_pages=total_pages,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            head_dim_padded=head_dim_padded,
            dtype=dtype,
        ),
        dtype=dtype,
    )
    kv_cache_xla = kv_cache
    kv_cache_triton = kv_cache.copy()

    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    out_xla, cache_xla = xla_ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache_xla,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_scale=head_dim**-0.5,
    )
    out_triton, cache_triton = triton_ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache_triton,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_scale=head_dim**-0.5,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)
    cache_triton = jax.block_until_ready(cache_triton)
    cache_xla = jax.block_until_ready(cache_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(cache_triton),
        np.asarray(cache_xla),
        rtol=0.0,
        atol=0.0,
    )
