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

import importlib.util
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("CUTLASS CuTe DSL not installed", allow_module_level=True)
if jax.devices()[0].platform != "gpu":
    pytest.skip("CUTE tests require GPU backend", allow_module_level=True)

from ejkernel.kernels._cute.unified_attention import unified_attention as cute_unified_attention
from ejkernel.kernels._xla.unified_attention import unified_attention as xla_unified_attention


def _make_inputs(
    *,
    seed: int,
    num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    kv_lens: list[int],
    q_lens: list[int],
    dtype: jnp.dtype,
) -> dict[str, jax.Array]:
    max_kv = max(kv_lens)
    max_blocks_per_seq = (max_kv + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_blocks_per_seq

    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)
    kv_lens_arr = jnp.asarray(kv_lens, dtype=jnp.int32)

    query_start = [0]
    for q in q_lens:
        query_start.append(query_start[-1] + int(q))
    query_start_loc = jnp.asarray(query_start, dtype=jnp.int32)

    total_tokens = int(query_start_loc[-1])
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32).astype(dtype)
    key_cache = jax.random.normal(k2, (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32).astype(
        dtype
    )
    value_cache = jax.random.normal(
        k3, (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(dtype)

    return {
        "queries": queries,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "kv_lens": kv_lens_arr,
        "block_tables": block_tables,
        "query_start_loc": query_start_loc,
        "max_q_len": max(q_lens),
    }


def test_unified_attention_cute_matches_xla_basic():
    batch = _make_inputs(
        seed=0,
        num_seqs=3,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        kv_lens=[64, 31, 17],
        q_lens=[16, 1, 4],
        dtype=jnp.bfloat16,
    )
    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])

    out_cute = cute_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
    )
    out_xla = xla_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        atol=2e-2,
        rtol=1e-2,
    )


def test_unified_attention_cute_matches_xla_with_optional_features():
    batch = _make_inputs(
        seed=11,
        num_seqs=4,
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=64,
        block_size=16,
        kv_lens=[32, 33, 17, 48],
        q_lens=[1, 1, 1, 1],
        dtype=jnp.float16,
    )

    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)
    softmax_aux = jax.random.normal(k1, (batch["queries"].shape[1],), dtype=jnp.float32)
    alibi = jax.random.normal(k2, (batch["queries"].shape[1],), dtype=jnp.float32) * 0.01
    qq_bias = jax.random.normal(k3, (batch["max_q_len"], batch["max_q_len"]), dtype=jnp.float32) * 0.02

    out_cute = cute_unified_attention(
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
        seq_threshold_3d=8,
        num_par_softmax_segments=16,
    )
    out_xla = xla_unified_attention(
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

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        atol=5e-2,
        rtol=2e-2,
    )
