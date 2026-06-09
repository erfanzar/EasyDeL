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

from ejkernel.kernels._triton import unified_attention
from ejkernel.kernels._xla.unified_attention import unified_attention as xla_unified_attention

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def _ref_unified_attention(
    queries: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    softmax_scale: float,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    alibi_slopes: jax.Array | None = None,
    qq_bias: jax.Array | None = None,
    softmax_aux: jax.Array | None = None,
) -> jax.Array:
    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_blocks, block_size, num_kv_heads, _head_dim_kv = map(int, key_cache.shape)
    num_seqs = int(kv_lens.shape[0])
    sliding_window = 0 if sliding_window is None else int(sliding_window)

    outs = []
    for s in range(num_seqs):
        q_start = int(query_start_loc[s])
        q_end = int(query_start_loc[s + 1])
        q_len = q_end - q_start
        kv_len = int(kv_lens[s])
        context_len = kv_len - q_len

        num_blocks = (kv_len + block_size - 1) // block_size
        blk_ids = block_tables[s, :num_blocks]

        k = key_cache[blk_ids].reshape(num_blocks * block_size, num_kv_heads, head_dim)[:kv_len]
        v = value_cache[blk_ids].reshape(num_blocks * block_size, num_kv_heads, head_dim)[:kv_len]

        if num_q_heads != num_kv_heads:
            repeats = num_q_heads // num_kv_heads
            k = jnp.repeat(k, repeats, axis=1)
            v = jnp.repeat(v, repeats, axis=1)

        q = queries[q_start:q_end]  # [q_len, H, D]
        logits = jnp.einsum("thd,khd->thk", q.astype(jnp.float32), k.astype(jnp.float32)) * float(softmax_scale)

        if logits_soft_cap is not None and logits_soft_cap > 0:
            logits = float(logits_soft_cap) * jnp.tanh(logits / float(logits_soft_cap))

        key_pos = jnp.arange(kv_len, dtype=jnp.int32)
        q_abs = jnp.int32(context_len) + jnp.arange(q_len, dtype=jnp.int32)  # [q_len]

        causal = key_pos[None, :] <= q_abs[:, None]
        if sliding_window > 0:
            left = q_abs - jnp.int32(sliding_window) + jnp.int32(1)
            causal = causal & (key_pos[None, :] >= left[:, None])

        if alibi_slopes is not None:
            key_rel = (key_pos - jnp.int32(context_len)).astype(jnp.float32)
            logits = logits + alibi_slopes[None, :, None] * key_rel[None, None, :]

        if qq_bias is not None:
            key_rel = key_pos - jnp.int32(context_len)
            is_qk = (key_rel >= 0) & (key_rel < qq_bias.shape[0])
            key_rel_clip = jnp.clip(key_rel, 0, qq_bias.shape[0] - 1)
            row = jnp.arange(q_len, dtype=jnp.int32)[:, None]
            qq = qq_bias[row, key_rel_clip[None, :]]
            qq = jnp.where(is_qk[None, :], qq, 0.0)
            logits = logits + qq[:, None, :]

        logits = jnp.where(causal[:, None, :], logits, -jnp.inf)

        if softmax_aux is not None:
            sink_logits = jnp.broadcast_to(softmax_aux.astype(jnp.float32)[None, :, None], (q_len, num_q_heads, 1))
            logits_aug = jnp.concatenate([logits, sink_logits], axis=-1)
            weights = jax.nn.softmax(logits_aug, axis=-1)[..., :kv_len]
        else:
            weights = jax.nn.softmax(logits, axis=-1)

        out = jnp.einsum("thk,khd->thd", weights.astype(jnp.float32), v.astype(jnp.float32)).astype(queries.dtype)
        outs.append(out)

    out = jnp.concatenate(outs, axis=0)
    assert out.shape == (total_tokens, num_q_heads, head_dim)
    return out


def _make_inputs(
    *,
    rng_seed: int = 0,
    num_seqs: int = 4,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    block_size: int = 16,
    kv_lens: list[int] | None = None,
    q_lens: list[int] | None = None,
    dtype=jnp.bfloat16,
):
    if kv_lens is None:
        kv_lens = [64] * num_seqs
    if q_lens is None:
        q_lens = [min(4, kv) for kv in kv_lens]
    assert len(kv_lens) == len(q_lens) == num_seqs
    assert all(1 <= q <= kv for q, kv in zip(q_lens, kv_lens, strict=True))

    max_kv = max(kv_lens)
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq

    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)
    kv_lens_arr = jnp.array(kv_lens, dtype=jnp.int32)

    cu = [0]
    for q in q_lens:
        cu.append(cu[-1] + int(q))
    query_start_loc = jnp.array(cu, dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    key = jax.random.PRNGKey(rng_seed)
    k1, k2, k3 = jax.random.split(key, 3)

    queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32).astype(dtype)
    key_cache = jax.random.normal(k2, (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32).astype(
        dtype
    )
    value_cache = jax.random.normal(
        k3, (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(dtype)

    return dict(
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        query_start_loc=query_start_loc,
        kv_lens=kv_lens_arr,
        block_tables=block_tables,
        max_q_len=max(q_lens),
    )


def test_unified_attention_matches_reference_basic():
    batch = _make_inputs(
        rng_seed=0,
        num_seqs=3,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        kv_lens=[64, 31, 17],
        q_lens=[16, 1, 4],
        dtype=jnp.bfloat16,
    )
    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])

    out = unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=None,
        logits_soft_cap=None,
    )
    ref = _ref_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
    )
    assert jnp.allclose(out.astype(jnp.float32), ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("seq_threshold_3d", [0, 8])
def test_unified_attention_features_softcap_sliding_sinks_bias(seq_threshold_3d: int):
    batch = _make_inputs(
        rng_seed=1,
        num_seqs=4,
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=64,
        kv_lens=[32, 33, 17, 48],
        q_lens=[1, 1, 1, 1],  # decode-only shapes
        dtype=jnp.float16,
    )
    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])

    num_q_heads = batch["queries"].shape[1]
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)
    softmax_aux = jax.random.normal(k1, (num_q_heads,), dtype=jnp.float32)
    alibi = jax.random.normal(k2, (num_q_heads,), dtype=jnp.float32) * 0.01
    qq_bias = jax.random.normal(k3, (batch["max_q_len"], batch["max_q_len"]), dtype=jnp.float32) * 0.02

    out = unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=32,
        logits_soft_cap=20.0,
        softmax_aux=softmax_aux,
        alibi_slopes=alibi,
        qq_bias=qq_bias,
        seq_threshold_3d=int(seq_threshold_3d),
        num_par_softmax_segments=16,
    )

    ref = _ref_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=32,
        logits_soft_cap=20.0,
        softmax_aux=softmax_aux,
        alibi_slopes=alibi,
        qq_bias=qq_bias,
    )
    assert jnp.allclose(out.astype(jnp.float32), ref.astype(jnp.float32), rtol=2e-2, atol=5e-2)


def test_unified_attention_matches_xla_smoke():
    batch = _make_inputs(
        rng_seed=0,
        num_seqs=2,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        kv_lens=[32, 24],
        q_lens=[4, 3],
        dtype=jnp.bfloat16,
    )
    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])

    out_triton = unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=16,
        logits_soft_cap=20.0,
    )
    out_xla = xla_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=16,
        logits_soft_cap=20.0,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
