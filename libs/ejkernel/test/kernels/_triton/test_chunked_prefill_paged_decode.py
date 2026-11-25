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


def _make_chunked_batch(
    *,
    seed: int,
    q_lens: list[int],
    ctx_lens: list[int],
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    block_size: int = 16,
    dtype: jnp.dtype = jnp.bfloat16,
):
    num_seqs = len(q_lens)
    assert len(ctx_lens) == num_seqs
    kv_lens_list = [c + q for c, q in zip(ctx_lens, q_lens, strict=True)]
    kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
    cu = [0]
    for q in q_lens:
        cu.append(cu[-1] + int(q))
    query_start_loc = jnp.array(cu, dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    max_kv = max(kv_lens_list)
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq
    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)

    key = jax.random.PRNGKey(seed)
    kq, kk, kv, kkc, kvc = jax.random.split(key, 5)
    queries = jax.random.normal(kq, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32).astype(dtype)
    keys = jax.random.normal(kk, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float32).astype(dtype)
    values = jax.random.normal(kv, (total_tokens, num_kv_heads, head_dim), dtype=jnp.float32).astype(dtype)
    # Pre-populate caches with random prefix data to verify untouched slots are preserved.
    key_cache = jax.random.normal(
        kkc, (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(dtype)
    value_cache = jax.random.normal(
        kvc, (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(dtype)

    return dict(
        queries=queries,
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
    )


@pytest.mark.parametrize(
    "q_lens,ctx_lens,seq_threshold_3d",
    [
        ([12, 1, 7], [20, 45, 0], 0),  # mixed chunked prefill + decode (2D path)
        ([1, 1, 1], [31, 48, 17], 8),  # pure decode (3D segmented path)
    ],
)
def test_chunked_prefill_paged_decode_softcap_sliding_sinks_matches_xla(q_lens, ctx_lens, seq_threshold_3d):
    """Verify fused Triton KV scatter + softcap/sliding/sinks/alibi parity vs XLA."""
    head_dim = 64
    batch = _make_chunked_batch(
        seed=17 + seq_threshold_3d,
        q_lens=q_lens,
        ctx_lens=ctx_lens,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        block_size=16,
        dtype=jnp.float32,
    )
    # Amplify queries to test overflow-safe softcap (|logits/cap| > 89).
    queries = batch["queries"] * 50.0
    sinks = jax.random.normal(jax.random.PRNGKey(5), (8,), dtype=jnp.float32)
    alibi = jax.random.normal(jax.random.PRNGKey(6), (8,), dtype=jnp.float32) * 0.02
    scale = 1.0 / math.sqrt(head_dim)

    out_tri, kc_tri, vc_tri = chunked_prefill_paged_decode(
        queries,
        batch["keys"],
        batch["values"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        alibi_slopes=alibi,
        softmax_aux=sinks,
        softmax_scale=scale,
        sliding_window=24,
        logits_soft_cap=15.0,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=6,
    )
    out_xla, kc_xla, vc_xla = xla_chunked_prefill_paged_decode(
        queries,
        batch["keys"],
        batch["values"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        alibi_slopes=alibi,
        softmax_aux=sinks,
        softmax_scale=scale,
        sliding_window=24,
        logits_soft_cap=15.0,
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
        np.asarray(jax.block_until_ready(kc_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(kc_xla), dtype=np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(vc_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(vc_xla), dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_chunked_prefill_paged_decode_jit_matches_xla():
    """Verify jax.jit tracing with aliased Triton KV scatter matches XLA."""
    head_dim = 64
    batch = _make_chunked_batch(
        seed=42,
        q_lens=[10, 3, 1],
        ctx_lens=[18, 29, 40],
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        block_size=16,
        dtype=jnp.bfloat16,
    )
    scale = 1.0 / math.sqrt(head_dim)

    @jax.jit
    def run_tri(q, k, v, kc, vc, kvl, bt, qsl):
        return chunked_prefill_paged_decode(
            q,
            k,
            v,
            kc,
            vc,
            kvl,
            bt,
            qsl,
            softmax_scale=scale,
            sliding_window=32,
            logits_soft_cap=20.0,
            seq_threshold_3d=0,
        )

    out_tri, kc_tri, vc_tri = run_tri(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
    )
    out_xla, kc_xla, vc_xla = xla_chunked_prefill_paged_decode(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=32,
        logits_soft_cap=20.0,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(out_xla), dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(kc_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(kc_xla), dtype=np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(vc_tri), dtype=np.float32),
        np.asarray(jax.block_until_ready(vc_xla), dtype=np.float32),
        rtol=0,
        atol=0,
    )
