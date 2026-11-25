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


def _align_to(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _kv_cache_shape(
    *,
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> tuple[int, ...]:
    pack = 2 if jnp.dtype(dtype).itemsize == 2 else 1
    head_dim_padded = _align_to(head_dim, 128)
    combined_heads = num_kv_heads if head_dim == 64 else num_kv_heads * 2
    heads_per_pack = _align_to(combined_heads, pack) // pack
    return (num_pages, page_size, heads_per_pack, pack, head_dim_padded)


def _make_rpa_v3_batch(
    *,
    seed: int = 0,
    num_seqs: int = 2,
    max_num_seqs: int | None = None,
    pages_per_seq: int = 4,
    page_size: int = 16,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    kv_lens_list: list[int] | None = None,
    q_lens_list: list[int] | None = None,
    pad_tokens: int = 0,
    dtype: jnp.dtype = jnp.float16,
):
    if max_num_seqs is None:
        max_num_seqs = num_seqs
    if kv_lens_list is None:
        kv_lens_list = [40, 32][:num_seqs]
    if q_lens_list is None:
        q_lens_list = [8, 4][:num_seqs]

    key = jax.random.PRNGKey(seed)
    kq, kk, kv, kc = jax.random.split(key, 4)

    padded_kv_lens = list(kv_lens_list) + [0] * (max_num_seqs - num_seqs)
    padded_q_lens = list(q_lens_list) + [0] * (max_num_seqs - num_seqs)

    kv_lens = jnp.array(padded_kv_lens, dtype=jnp.int32)
    q_lens = jnp.array(padded_q_lens, dtype=jnp.int32)
    query_start_loc = jnp.pad(jnp.cumsum(q_lens), (1, 0)).astype(jnp.int32)
    total_q = int(jnp.sum(q_lens)) + pad_tokens

    queries = jax.random.normal(kq, (total_q, num_q_heads, head_dim), dtype=jnp.float32).astype(dtype)
    keys = jax.random.normal(kk, (total_q, num_kv_heads, head_dim), dtype=jnp.float32).astype(dtype)
    values = jax.random.normal(kv, (total_q, num_kv_heads, head_dim), dtype=jnp.float32).astype(dtype)

    total_pages = max_num_seqs * pages_per_seq
    kv_cache = jax.random.normal(
        kc,
        _kv_cache_shape(
            num_pages=total_pages,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ),
        dtype=jnp.float32,
    ).astype(dtype)

    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    return dict(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_ragged_page_attention_v3_matches_xla(head_dim):
    batch = _make_rpa_v3_batch(seed=0, head_dim=head_dim, dtype=jnp.float16)
    scale = head_dim**-0.5

    out_xla, cache_xla = xla_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=scale,
    )
    out_triton, cache_triton = triton_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=scale,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_triton), dtype=np.float32),
        np.asarray(jax.block_until_ready(out_xla), dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(cache_triton)),
        np.asarray(jax.block_until_ready(cache_xla)),
        rtol=0.0,
        atol=0.0,
    )


def test_ragged_page_attention_v3_softcap_sliding_sinks_matches_xla():
    """Verify overflow-safe logits_soft_cap, sliding_window, and softmax_aux parity vs XLA."""
    head_dim = 128
    batch = _make_rpa_v3_batch(
        seed=17,
        num_seqs=2,
        pages_per_seq=6,
        page_size=16,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        kv_lens_list=[80, 64],
        q_lens_list=[12, 5],
        dtype=jnp.float32,
    )
    # Amplify queries to exercise large |logits / cap| > 89 (overflow-safe tanh).
    queries = batch["queries"] * 50.0
    sinks = jax.random.normal(jax.random.PRNGKey(3), (8,), dtype=jnp.float32)
    scale = head_dim**-0.5

    out_xla, cache_xla = xla_ragged_page_attention_v3(
        queries,
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=scale,
        sliding_window=32,
        logits_soft_cap=15.0,
    )
    out_tri, cache_tri = triton_ragged_page_attention_v3(
        queries,
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=scale,
        sliding_window=32,
        logits_soft_cap=15.0,
        num_warps=4,
        num_stages=2,
    )

    out_tri = jax.block_until_ready(out_tri)
    assert jnp.all(jnp.isfinite(out_tri))
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(jax.block_until_ready(out_xla), dtype=np.float32),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(cache_tri)),
        np.asarray(jax.block_until_ready(cache_xla)),
        rtol=0.0,
        atol=0.0,
    )


def test_ragged_page_attention_v3_padded_tokens_and_nonpow2_pages_matches_xla():
    """Verify padded tokens beyond distribution[2] never corrupt cache and non-pow2 pages work."""
    head_dim = 128
    batch = _make_rpa_v3_batch(
        seed=29,
        num_seqs=2,
        max_num_seqs=4,
        pages_per_seq=3,  # non-power-of-2 pages_per_seq
        page_size=16,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        kv_lens_list=[36, 28],
        q_lens_list=[6, 4],
        pad_tokens=6,  # trailing padded tokens that must not corrupt kv_cache
        dtype=jnp.bfloat16,
    )
    scale = head_dim**-0.5

    out_xla, cache_xla = xla_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=scale,
        sliding_window=24,
        logits_soft_cap=20.0,
    )
    out_tri, cache_tri = triton_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=scale,
        sliding_window=24,
        logits_soft_cap=20.0,
        num_kv_pages_per_block=3,
        num_queries_per_block=32,
        num_warps=4,
        num_stages=2,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(out_xla), dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(cache_tri)),
        np.asarray(jax.block_until_ready(cache_xla)),
        rtol=0.0,
        atol=0.0,
    )


def test_ragged_page_attention_v3_jit_matches_xla():
    """Verify jax.jit tracing and parity vs XLA without nested ejit donation issues."""
    head_dim = 128
    batch = _make_rpa_v3_batch(
        seed=41,
        num_seqs=2,
        max_num_seqs=3,
        pages_per_seq=3,
        page_size=16,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        kv_lens_list=[30, 42],
        q_lens_list=[5, 7],
        pad_tokens=4,
        dtype=jnp.float16,
    )
    sinks = jax.random.normal(jax.random.PRNGKey(9), (8,), dtype=jnp.float32)
    scale = head_dim**-0.5

    @jax.jit
    def run_tri(q, k, v, cache, kvl, bt, qsl, dist, aux):
        return triton_ragged_page_attention_v3(
            q,
            k,
            v,
            cache,
            kvl,
            bt,
            qsl,
            dist,
            aux,
            softmax_scale=scale,
            sliding_window=24,
            logits_soft_cap=25.0,
            num_warps=4,
            num_stages=2,
        )

    out_tri, cache_tri = run_tri(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
    )
    out_xla, cache_xla = xla_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=scale,
        sliding_window=24,
        logits_soft_cap=25.0,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(out_xla), dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(cache_tri)),
        np.asarray(jax.block_until_ready(cache_xla)),
        rtol=0.0,
        atol=0.0,
    )
